"""
injector.py — Identify adjacent LLVM instructions by single-bit opcode distance.

Two instructions are adjacent when their encodings differ by exactly one bit in
either opc_lo OR opc_hi, but not both bytes simultaneously.  For each such pair
the output records which byte to target and which bit to flip.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from karnage.utils.exceptions import MatcherTableLoadError
from karnage.utils.logger import logger
from karnage.utils.models import (
    OpcodeInfo,
    FlipInfo,
    AdjacencyEntry
)

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_matcher_table(path: Path) -> dict:
    if not path.exists():
        raise MatcherTableLoadError(
            f"matcher_table JSON not found: {path}",
            context={"path": str(path)},
        )
    try:
        with path.open() as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise MatcherTableLoadError(
            f"Invalid JSON in {path}: {exc}",
            context={"path": str(path)},
        ) from exc

    if "instructions" not in data:
        raise MatcherTableLoadError(
            f"Expected top-level 'instructions' key in {path}",
            context={"path": str(path)},
        )
    return data


# ---------------------------------------------------------------------------
# Opcode map
# ---------------------------------------------------------------------------

def _collect_opcode_map(data: dict) -> dict[tuple[int, int], OpcodeInfo]:
    """Return a dict keyed by (opc_lo, opc_hi) → OpcodeInfo."""
    opcode_map: dict[tuple[int, int], OpcodeInfo] = {}
    for mnemonic, opc_list in data["instructions"].items():
        for opc_obj in opc_list:
            enc = opc_obj["patterns"][0]["encoding"]
            lo  = int(enc["opc_lo"], 16)
            hi  = int(enc["opc_hi"], 16)
            key = (lo, hi)
            if key in opcode_map:
                # Duplicate (lo, hi) across different mnemonics — keep first.
                logger.warning(
                    f"Duplicate (opc_lo={lo:#04x}, opc_hi={hi:#04x}): "
                    f"keeping '{opcode_map[key].mnemonic}', skipping '{mnemonic}'"
                )
                continue
            opcode_map[key] = OpcodeInfo(
                mnemonic=mnemonic,
                opcode=opc_obj["opcode"],
                opc_lo=lo,
                opc_hi=hi,
                num_patterns=len(opc_obj["patterns"]),
            )
    return opcode_map


# ---------------------------------------------------------------------------
# Adjacency search
# ---------------------------------------------------------------------------

def find_adjacent_pairs(data: dict) -> list[AdjacencyEntry]:
    """Return all pairs of instructions that differ by exactly one bit in one byte."""
    opcode_map = _collect_opcode_map(data)
    logger.info(f"Scanning {len(opcode_map):,} unique (opc_lo, opc_hi) entries for adjacency")

    seen:  set[tuple[tuple[int, int], tuple[int, int]]] = set()
    pairs: list[AdjacencyEntry] = []

    for (lo, hi), info in opcode_map.items():
        # --- single bit flip in opc_lo (opc_hi unchanged) ---
        for bit in range(8):
            neighbor_lo = lo ^ (1 << bit)
            neighbor_key = (neighbor_lo, hi)
            if neighbor_key not in opcode_map:
                continue
            canonical = tuple(sorted([(lo, hi), neighbor_key]))
            if canonical in seen:
                continue
            seen.add(canonical)

            neighbor = opcode_map[neighbor_key]
            a, b = (info, neighbor) if info.opcode < neighbor.opcode else (neighbor, info)
            pairs.append(AdjacencyEntry(
                a=a, b=b,
                flip=FlipInfo(byte="opc_lo", bit=bit, mask=(1 << bit)),
            ))

        # --- single bit flip in opc_hi (opc_lo unchanged) ---
        for bit in range(8):
            neighbor_hi = hi ^ (1 << bit)
            neighbor_key = (lo, neighbor_hi)
            if neighbor_key not in opcode_map:
                continue
            canonical = tuple(sorted([(lo, hi), neighbor_key]))
            if canonical in seen:
                continue
            seen.add(canonical)

            neighbor = opcode_map[neighbor_key]
            a, b = (info, neighbor) if info.opcode < neighbor.opcode else (neighbor, info)
            pairs.append(AdjacencyEntry(
                a=a, b=b,
                flip=FlipInfo(byte="opc_hi", bit=bit, mask=(1 << bit)),
            ))

    pairs.sort(key=lambda p: (p.a.opcode, p.b.opcode, p.flip.byte, p.flip.bit))
    logger.info(f"Found {len(pairs):,} adjacent pairs")
    return pairs


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def build_report(pairs: list[AdjacencyEntry], source_meta: dict) -> dict:
    """
    Build the adjacency report keyed by opcode integer (as a string).

    Keying by opcode rather than mnemonic avoids ambiguity when multiple
    opcode variants share the same mnemonic (e.g. 'or.b32' opcode 1712 vs
    1713).  Mnemonic is preserved as a field inside each entry.
    """
    from collections import defaultdict

    info_by_opcode:     dict[int, OpcodeInfo]  = {}
    neighbors_by_opcode: dict[int, list[dict]] = defaultdict(list)

    def _neighbor_entry(other: OpcodeInfo, flip: FlipInfo) -> dict:
        return {
            "mnemonic":     other.mnemonic,
            "opcode":       other.opcode,
            "opc_lo":       f"0x{other.opc_lo:02x}",
            "opc_hi":       f"0x{other.opc_hi:02x}",
            "num_patterns": other.num_patterns,
            "flip": {
                "byte": flip.byte,
                "bit":  flip.bit,
                "mask": f"0x{flip.mask:02x}",
            },
        }

    for entry in pairs:
        info_by_opcode[entry.a.opcode] = entry.a
        info_by_opcode[entry.b.opcode] = entry.b
        neighbors_by_opcode[entry.a.opcode].append(_neighbor_entry(entry.b, entry.flip))
        neighbors_by_opcode[entry.b.opcode].append(_neighbor_entry(entry.a, entry.flip))

    instructions = {}
    for opcode in sorted(info_by_opcode):
        oi  = info_by_opcode[opcode]
        adj = sorted(
            neighbors_by_opcode[opcode],
            key=lambda n: (n["opcode"], n["flip"]["byte"], n["flip"]["bit"]),
        )
        instructions[str(opcode)] = {
            "mnemonic":     oi.mnemonic,
            "opcode":       opcode,
            "opc_lo":       f"0x{oi.opc_lo:02x}",
            "opc_hi":       f"0x{oi.opc_hi:02x}",
            "num_patterns": oi.num_patterns,
            "adjacent":     adj,
        }

    return {
        "meta": {
            "source":               source_meta.get("source", ""),
            "llvm_commit":          source_meta.get("llvm_commit", ""),
            "total_opcodes":        source_meta.get("total_opcode_objects", 0),
            "total_adjacent_pairs": len(pairs),
        },
        "instructions": instructions,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(input_path: Path, output_path: Path) -> None:
    logger.info(f"Loading matcher table: {input_path}")
    data = load_matcher_table(input_path)

    source_meta = {**data.get("meta", {}), "source": input_path.name}

    pairs  = find_adjacent_pairs(data)
    report = build_report(pairs, source_meta)

    with output_path.open("w") as f:
        json.dump(report, f, indent=2)

    logger.success(
        f"Wrote {len(pairs):,} adjacent pairs → {output_path}"
    )
