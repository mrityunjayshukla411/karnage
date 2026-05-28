#!/usr/bin/env python3
"""
karnage — LLVM MatcherTable fault-injection toolkit.

Subcommands
-----------
  extract   Scan the binary and produce both matcher_table.json and
            adjacency.json in one step.
  inject    Run GDB-based bit-flip tests using those two files.

Quick start
-----------
  # Step 1 – extract MatcherTable and adjacency table from libtriton.so
  python main.py extract --from-triton

  # Step 2 – run bit-flip tests
  python main.py inject --script matmul.py --library /path/to/libtriton.so
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
from karnage.extractor.extractor import get_matchertable_bounds, run, walk
from karnage.injector import run_tester
from karnage.utils.constants import (
    DEFAULT_ADJACENCY,
    DEFAULT_MATCHER_TABLE,
    DEFAULT_OUTPUT_DIR,
)
from karnage.utils.logger import logger
from karnage.utils.models import MatcherEntry
from karnage.utils.parser import (
    build_opcode_mnemonic_map,
    parse_mvt_map,
    parse_opcode_enum,
)
from karnage.utils.targets import NVPTXBackend


# ---------------------------------------------------------------------------
# Shared serialisation helper
# ---------------------------------------------------------------------------

def _build_pattern(e: MatcherEntry, morph_name_map: dict[int, str]) -> dict:
    """Serialise one :class:`~karnage.utils.models.MatcherEntry` as a JSON pattern object.

    Args:
        e:              The entry to serialise.
        morph_name_map: Reverse mapping ``{morph_byte_value: OPC_MorphNodeTo*_name}``
                        used to resolve the human-readable morph variant name.

    Returns:
        JSON-serialisable dict matching the ``pattern`` schema in
        ``matcher_table.json``.
    """
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


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

def _cmd_extract(args: argparse.Namespace) -> None:
    """Extract the MatcherTable then build the adjacency table in one pass.

    Writes two output files:

    - ``--matcher-table`` (default: ``matcher_table.json``) — full pattern
      database produced by scanning the MatcherTable bytecode.
    - ``--adjacency`` (default: ``adjacency.json``) — pairs of opcodes that
      differ by a single bit in their encoding, derived from the above.

    Args:
        args: Parsed CLI arguments for the ``extract`` subcommand.
    """
    library: Path = args.library if args.library else _get_target_library_path("triton")
    if not library.exists():
        raise SystemExit(f"Library not found: {library}")

    target = NVPTXBackend()

    commit_hash = _extract_binary_hash(library)
    logger.info(f"LLVM commit: {commit_hash}")

    llvm_cache_dir = build_llvm(commit_hash, target).parent
    inc = target.inc_paths(llvm_cache_dir)

    data = library.read_bytes()

    opc_map         = build_opcode_mnemonic_map(library, target, data=data)
    mt_off, mt_size = get_matchertable_bounds(library, target)
    mvt             = parse_mvt_map(inc["genvt"])
    mvt_filtered    = target.filter_mvt_map(mvt)
    full_enum       = parse_opcode_enum(inc["seldagisell_h"])

    entries = walk(data, mt_off, mt_size, full_enum, opc_map, mvt_filtered)

    morph_name_map: dict[int, str] = {
        v: k for k, v in full_enum.items() if "MorphNodeTo" in k
    }

    grouped: dict[str, dict[int, list[MatcherEntry]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for e in entries:
        grouped[e.mnemonic][e.opcode].append(e)

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

    with args.matcher_table.open("w") as f:
        json.dump(doc, f, indent=2)

    logger.success(
        f"Wrote {len(entries):,} patterns across {len(instructions):,} "
        f"mnemonics → {args.matcher_table}"
    )

    # --- Adjacency pass ---
    run(args.matcher_table, args.adjacency)


# ---------------------------------------------------------------------------
# inject
# ---------------------------------------------------------------------------

def _cmd_inject(args: argparse.Namespace) -> None:
    """Validate inputs and run the GDB-based bit-flip test suite.

    Args:
        args: Parsed CLI arguments for the ``inject`` subcommand.
    """
    for path, flag in [
        (args.script,        "--script"),
        (args.matcher_table, "--matcher-table"),
        (args.adjacency,     "--adjacency"),
        (args.library,       "--library"),
    ]:
        if not path.exists():
            raise SystemExit(f"{flag}: not found: {path}")

    target_mnemonics = (
        frozenset(m.strip() for m in args.mnemonics.split(",") if m.strip())
        if args.mnemonics else None
    )

    run_tester(
        triton_script      = args.script,
        matcher_table_json = args.matcher_table,
        adjacency_json     = args.adjacency,
        libtriton_so       = args.library,
        output_dir         = args.output,
        max_flips          = args.max_flips,
        report_json        = args.report,
        cooldown_every     = args.cooldown_every,
        cooldown_secs      = args.cooldown_secs,
        filter_by_ptx      = args.filter_by_ptx,
        target_mnemonics   = target_mnemonics,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the top-level argument parser with all subcommands."""
    ap = argparse.ArgumentParser(
        prog="karnage",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # -- extract --
    p_extract = sub.add_parser(
        "extract",
        help="Extract MatcherTable and build adjacency table from a shared library.",
        description=(
            "Scan the target binary to produce matcher_table.json, then compute "
            "single-bit adjacency pairs and write adjacency.json."
        ),
    )
    src = p_extract.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--library", type=Path,
        help="Explicit path to libtriton.so",
    )
    src.add_argument(
        "--from-triton", action="store_true",
        help="Auto-detect libtriton.so from the installed triton package",
    )
    p_extract.add_argument(
        "--matcher-table", type=Path, default=Path(DEFAULT_MATCHER_TABLE),
        metavar="PATH",
        help=f"Output path for the MatcherTable JSON (default: {DEFAULT_MATCHER_TABLE})",
    )
    p_extract.add_argument(
        "--adjacency", type=Path, default=Path(DEFAULT_ADJACENCY),
        metavar="PATH",
        help=f"Output path for the adjacency JSON (default: {DEFAULT_ADJACENCY})",
    )
    p_extract.set_defaults(func=_cmd_extract)

    # -- inject --
    p_inject = sub.add_parser(
        "inject",
        help="Run GDB-based MatcherTable bit-flip tests.",
        description=(
            "For each adjacent opcode pair, flip the corresponding bit in "
            "libtriton.so's in-memory MatcherTable via GDB and compare the "
            "generated PTX and tensor outputs against a clean baseline."
        ),
    )
    p_inject.add_argument(
        "--script", type=Path, required=True,
        help="Triton application script to test (e.g. matmul.py)",
    )
    p_inject.add_argument(
        "--matcher-table", type=Path, default=Path(DEFAULT_MATCHER_TABLE),
        metavar="PATH",
        help=f"matcher_table.json from extract (default: {DEFAULT_MATCHER_TABLE})",
    )
    p_inject.add_argument(
        "--adjacency", type=Path, default=Path(DEFAULT_ADJACENCY),
        metavar="PATH",
        help=f"adjacency.json from extract (default: {DEFAULT_ADJACENCY})",
    )
    p_inject.add_argument(
        "--library", type=Path, required=True,
        help="Path to libtriton.so",
    )
    p_inject.add_argument(
        "--output", type=Path, default=Path(DEFAULT_OUTPUT_DIR),
        metavar="DIR",
        help=f"Root output directory for per-flip results (default: {DEFAULT_OUTPUT_DIR})",
    )
    p_inject.add_argument(
        "--max-flips", type=int, default=None, metavar="N",
        help="Stop after N flips (useful for a quick sanity check)",
    )
    p_inject.add_argument(
        "--report", type=Path, default=None, metavar="PATH",
        help="Write a JSON summary of all flip results to this path",
    )
    p_inject.add_argument(
        "--cooldown-every", type=int, default=0, metavar="N",
        help="Pause for --cooldown-secs after every N flips (0 = disabled)",
    )
    p_inject.add_argument(
        "--cooldown-secs", type=float, default=30.0, metavar="S",
        help="Seconds to sleep during each cooldown pause (default: 30)",
    )
    p_inject.add_argument(
        "--filter-by-ptx", action="store_true",
        help="Only test opcodes whose mnemonic appears in the baseline PTX",
    )
    p_inject.add_argument(
        "--mnemonics", type=str, default=None, metavar="m1,m2,...",
        help="Comma-separated list of mnemonics to test exclusively",
    )
    p_inject.set_defaults(func=_cmd_inject)

    return ap


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate subcommand handler."""
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
