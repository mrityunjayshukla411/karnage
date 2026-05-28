#!/usr/bin/env python3
"""
extract.py — Extract the LLVM MatcherTable from libtriton.so to JSON.

Flow:
  1. Locate libtriton.so (explicit path or auto-detect from installed triton)
  2. Extract the LLVM commit hash baked into the binary
  3. Download and build the matching LLVM source (tablegen targets only)
  4. Build the opcode→mnemonic map by reading AsmWriter tables from the binary
  5. Locate the MatcherTable section in the binary
  6. Parse morph variants from SelectionDAGISel.h and MVT map from GenVT.inc
  7. Build the OPC_SwitchType context map for accurate input_mvt resolution
  8. Scan the MatcherTable and write hierarchical JSON

Output shape:
  {
    "meta": { "llvm_commit": "...", "binary": "...", "total_patterns": N, ... },
    "instructions": {
      "<mnemonic>": [
        {
          "opcode": <int>,
          "patterns": [
            {
              "hit_num": <int>,
              "input_mvt":  { "hex": "0xNN", "type": "<name or empty>" },
              "results":    [ { "hex": "0xNN", "type": "<name>" }, ... ],
              "n_results":  <int>,
              "num_ops":    <int>,
              "op_idx":     <int>,
              "arm_len":    <int>,
              "location":   { "file_offset": "0xNNNNNNNN", "mt_offset": "0xNNNNNNNN" },
              "encoding": {
                "morph_variant": "<OPC_MorphNodeTo...>",
                "morph_byte":    "0xNN",
                "flags_byte":    "0xNN",
                "opc_lo":        "0xNN",
                "opc_hi":        "0xNN",
                "raw_bytes":     "NN NN ..."
              }
            }, ...
          ]
        }, ...         <- multiple opcode objects when variants share a mnemonic
      ]
    }
  }

Usage:
    python extract.py --library /path/to/libtriton.so --output out.json
    python extract.py --from-triton --output out.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from karnage.builder.builder import (
    _get_target_library_path,
    _extract_binary_hash,
    build_llvm,
)
from karnage.extractor.extractor import get_matchertable_bounds

from karnage.utils.models import MatcherEntry

from karnage.extractor.extractor import walk
from karnage.utils.parser import (
    build_opcode_mnemonic_map,
    parse_opcode_enum,
    parse_mvt_map,
)
from karnage.utils.targets import NVPTXBackend
from karnage.utils.logger import logger


def _build_pattern(e: MatcherEntry, morph_name_map: dict[int, str]) -> dict:
    """Serialize one MatcherEntry as the JSON 'pattern' object."""
    return {
        "hit_num":   e.hit_num,
        "input_mvt": {
            "hex":  f"0x{e.input_mvt:02x}",
            "type": e.input_mvt_type,
        },
        "results": [
            {"hex": f"0x{mvt:02x}", "type": typ}
            for mvt, typ in zip(e.result_mvts, e.result_mvt_types)
        ],
        "n_results": e.n_results,
        "num_ops":   e.num_ops,
        "op_idx":    e.op_idx,
        "arm_len":   e.arm_len,
        "location": {
            "file_offset": f"0x{e.file_offset:08x}",
            "mt_offset":   f"0x{e.mt_offset:08x}",
        },
        "encoding": {
            "morph_variant": morph_name_map.get(e.morph_byte, f"0x{e.morph_byte:02x}"),
            "morph_byte":    f"0x{e.morph_byte:02x}",
            "flags_byte":    f"0x{e.flags_byte:02x}",
            "opc_lo":        f"0x{e.opc_lo:02x}",
            "opc_hi":        f"0x{e.opc_hi:02x}",
            "raw_bytes":     e.raw_bytes.hex(" "),
        },
    }


def extract(library: Path, output: Path) -> None:
    target = NVPTXBackend()

    commit_hash = _extract_binary_hash(library)
    logger.info(f"LLVM commit: {commit_hash}")

    llvm_cache_dir = build_llvm(commit_hash, target).parent
    inc = target.inc_paths(llvm_cache_dir)

    data = library.read_bytes()

    opc_map  = build_opcode_mnemonic_map(library, target, data=data)
    mt_off, mt_size = get_matchertable_bounds(library, target)
    mvt      = parse_mvt_map(inc["genvt"])
    # Restrict to types valid for this target; removes scalable / other-target
    # types from the global MVT enum that corrupt false-positive SwitchType arms.
    mvt_filtered = target.filter_mvt_map(mvt)

    full_enum = parse_opcode_enum(inc["seldagisell_h"])

    entries = walk(data, mt_off, mt_size, full_enum, opc_map, mvt_filtered)

    # Reverse map: morph byte value → OPC_MorphNodeTo* name string
    morph_name_map: dict[int, str] = {v: k for k, v in full_enum.items() if "MorphNodeTo" in k}

    # Group: mnemonic → opcode → [MatcherEntry, ...]
    # Using two nested defaultdicts keeps insertion order (Python 3.7+).
    grouped: dict[str, dict[int, list[MatcherEntry]]] = defaultdict(lambda: defaultdict(list))
    for e in entries:
        grouped[e.mnemonic][e.opcode].append(e)

    # Build the hierarchical instructions dict sorted alphabetically by mnemonic
    instructions: dict[str, list[dict]] = {}
    for mnemonic in sorted(grouped):
        opcode_objects = []
        for opcode in sorted(grouped[mnemonic]):
            opcode_objects.append({
                "opcode":   opcode,
                "patterns": [
                    _build_pattern(e, morph_name_map)
                    for e in grouped[mnemonic][opcode]
                ],
            })
        instructions[mnemonic] = opcode_objects

    doc = {
        "meta": {
            "llvm_commit":          commit_hash,
            "binary":               str(library),
            "total_patterns":       len(entries),
            "total_mnemonics":      len(instructions),
            "total_opcode_objects": sum(len(v) for v in instructions.values()),
        },
        "instructions": instructions,
    }

    with output.open("w") as f:
        json.dump(doc, f, indent=2)

    logger.success(
        f"Wrote {len(entries):,} patterns across {len(instructions):,} mnemonics → {output}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract LLVM MatcherTable to JSON.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--library",     type=Path, help="Explicit path to libtriton.so")
    src.add_argument("--from-triton", action="store_true",
                     help="Auto-detect libtriton.so from the installed triton package")
    ap.add_argument("--output", type=Path, default=Path("matcher_table.json"),
                    help="Output JSON path (default: matcher_table.json)")
    args = ap.parse_args()

    library = args.library if args.library else _get_target_library_path("triton")
    if not library.exists():
        ap.error(f"Library not found: {library}")

    extract(library, args.output)


if __name__ == "__main__":
    main()
