"""
extractor/extractor.py — LLVM MatcherTable scan.

Provides:
  get_matchertable_bounds()    — locate the table in a binary
  build_switchtype_context()  — forward-scan OPC_SwitchType for input MVT context
  scan()                      — single O(mt_size) pass
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from karnage.utils.logger import logger
from karnage.utils.parser import (
    find_symbol_linker_vma,
    linker_vma_to_file_offset,
    estimate_symbol_byte_size,
)
from karnage.utils.targets import TargetBackend

from dataclasses import dataclass


@dataclass(frozen=True)
class MatcherEntry:
    """One OPC_MorphNodeTo* hit inside the LLVM MatcherTable."""

    opcode:           int
    mnemonic:         str
    hit_num:          int
    n_results:        int
    morph_byte:       int
    flags_byte:       int           # EmitNodeInfo flags; 0 for space-optimized variants
    opc_lo:           int           # TARGET_OPC low byte  (data[i+1])
    opc_hi:           int           # TARGET_OPC high byte (data[i+2]); opcode = opc_lo | opc_hi<<8
    file_offset:      int
    mt_offset:        int
    arm_len:          int
    input_mvt:        int
    input_mvt_type:   str           # "" if unknown
    result_mvts:      tuple[int, ...]
    result_mvt_types: tuple[str, ...]
    num_ops:          int
    op_idx:           int
    raw_bytes:        bytes


def get_matchertable_bounds(
    binary: Path,
    target: TargetBackend,
) -> tuple[int, int]:
    """
    Locate the MatcherTable using the target's matchertable_symbol().

    Returns (file_offset, size_in_bytes).
    Raises ParserError if the symbol is not found (stripped binary).
    """
    symbol = target.matchertable_symbol
    logger.debug(f"Locating MatcherTable symbol: {symbol!r}")

    vma  = find_symbol_linker_vma(binary, symbol)
    size = estimate_symbol_byte_size(binary, symbol)

    file_offset = linker_vma_to_file_offset(binary, ".rodata", vma)
    logger.info(
        f"MatcherTable: file_offset=0x{file_offset:08x}  size={size:,} bytes"
    )
    return file_offset, size


# ---------------------------------------------------------------------------
# VBR helper
# ---------------------------------------------------------------------------

def _read_vbr(data: bytes, pos: int, end: int) -> tuple[int, int]:
    """
    Read a LLVM VBR-encoded integer from *data* at *pos*.

    LLVM VBR encoding: bit 7 of each byte signals "more bytes follow".
    Bits 0-6 of each byte contribute 7 data bits, LSB-first.

    Returns (decoded_value, new_pos).
    """
    if pos >= end:
        return 0, pos
    val = data[pos]; pos += 1
    if not (val & 0x80):
        return val, pos           # single-byte case (value < 128)
    result = val & 0x7F
    shift  = 7
    while pos < end:
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        shift  += 7
        if not (b & 0x80):
            break
    return result, pos


# ---------------------------------------------------------------------------
# OPC_SwitchType context scanner
# ---------------------------------------------------------------------------

def build_switchtype_context(
    data:                bytes,
    mt_offset:           int,
    mt_size:             int,
    switch_type_opcode:  int,
    mvt_map:             dict[int, str],
) -> dict[int, tuple[int, int]]:
    """
    Forward-scan the MatcherTable for OPC_SwitchType arms and build a
    position→context map.

    OPC_SwitchType arm binary layout (per SelectionDAGISel.cpp executor):

        [OPC_SwitchType]
          [arm_size: VBR]  [mvt: 1 byte]  [body: arm_size bytes]
          ...
          [0x00]           <- terminator (arm_size == 0)

    Returns {absolute_data_position: (input_mvt, arm_size)} for every byte
    position that falls inside an arm body.  Lookup is O(1) so the scan
    itself can use this map for any detected MorphNodeTo position.

    The scanner is opportunistic: it checks every byte that equals
    switch_type_opcode and validates the arm structure before recording.
    False positives (where the value appears as an operand) are rejected if
    arm sizes run out of bounds or no valid arms are found.  A first-write-
    wins policy means earlier (likely real) SwitchType entries are not
    overwritten by later false positives whose bodies overlap.
    """
    mt_end   = mt_offset + mt_size
    context: dict[int, tuple[int, int]] = {}

    for i in range(mt_offset, mt_end):
        if data[i] != switch_type_opcode:
            continue

        p    = i + 1
        arms: list[tuple[int, int, int]] = []   # (body_start, mvt, arm_size)
        valid = True

        while p < mt_end:
            arm_size, p = _read_vbr(data, p, mt_end)
            if arm_size == 0:
                break                           # terminator
            if p >= mt_end or arm_size > mt_end - p:
                valid = False
                break
            mvt = data[p]; p += 1              # getSimpleVT — 1 byte for standard types
            arms.append((p, mvt, arm_size))
            p += arm_size

        if not valid or not arms:
            continue

        for body_start, mvt, arm_size in arms:
            for pos in range(body_start, body_start + arm_size):
                if pos not in context:          # first-write-wins
                    context[pos] = (mvt, arm_size)

    logger.debug(f"SwitchType context: {len(context):,} positions mapped")
    return context


# ---------------------------------------------------------------------------
# MatcherTable scanner
# ---------------------------------------------------------------------------

def scan(
    data:                bytes,
    mt_offset:           int,
    mt_size:             int,
    morph_variants:      dict[int, tuple[int, bool]],
    opcode_map:          dict[int, str],
    mvt_map:             dict[int, str],
    *,
    switchtype_context:  dict[int, tuple[int, int]] | None = None,
) -> tuple[MatcherEntry, ...]:
    """
    Single O(mt_size) pass over the MatcherTable.

    morph_variants maps each MorphNodeTo opcode byte to
    (n_results, has_explicit_flags_byte).

    The bytecode layout at position i (the morph byte) differs by variant:

      Variants WITHOUT an explicit flags byte
      (*None, *Chain, *GlueInput, *GlueOutput):
        i+0        : morph byte
        i+1..i+2   : TARGET_OPC (little-endian u16)
        i+3..i+2+n : result MVT bytes  (n = n_results)
        i+3+n      : num_ops
        i+4+n      : op_idx (first operand index)

      Variants WITH an explicit flags byte (bare OPC_MorphNodeTo0/1/2):
        i+0        : morph byte
        i+1..i+2   : TARGET_OPC
        i+3        : EmitNodeInfo flags byte
        i+4..i+3+n : result MVT bytes
        i+4+n      : num_ops
        i+5+n      : op_idx

    The 2 bytes preceding i carry the OPC_SwitchType arm header when the
    MorphNodeTo is the first instruction in its arm body.  For entries
    deeper in the arm body, switchtype_context (built by
    build_switchtype_context) provides the accurate input_mvt and arm_len.

    Returns a frozen tuple of MatcherEntry, with hit_num per-opcode.
    """
    mt_end = mt_offset + mt_size

    raw_entries: list[dict] = []
    for i in range(mt_offset, mt_end):
        b = data[i]
        if b not in morph_variants:
            continue

        n_results, has_flags = morph_variants[b]
        opc_lo = data[i + 1]
        opc_hi = data[i + 2]
        opc    = opc_lo | (opc_hi << 8)
        if opc not in opcode_map:
            continue

        # Byte positions for flags (optional), result VTs, and num_ops
        flags_byte  = 0
        if has_flags:
            if i + 3 >= mt_end:
                continue
            flags_byte = data[i + 3]

        vt_start    = i + 3 + (1 if has_flags else 0)
        num_ops_pos = vt_start + n_results
        if num_ops_pos >= mt_end:
            continue

        num_ops = data[num_ops_pos]
        if not (1 <= num_ops <= 10):
            continue

        op_idx_pos = num_ops_pos + 1
        op_idx     = data[op_idx_pos] if op_idx_pos < mt_end else 0

        result_mvts = tuple(
            data[vt_start + j] if vt_start + j < mt_end else 0
            for j in range(n_results)
        )

        # Resolve OPC_SwitchType arm context for this position.
        # Prefer the forward-scan context map; fall back to backward read only
        # when the map has no entry (map is empty or position wasn't covered).
        if switchtype_context and i in switchtype_context:
            input_mvt, arm_len = switchtype_context[i]
        else:
            # Backward read: correct only when MorphNodeTo is the first
            # instruction in the SwitchType arm body (no preceding checks).
            arm_len   = data[i - 2] if i >= 2 + mt_offset else 0
            input_mvt = data[i - 1] if i >= 1 + mt_offset else 0

        raw_end   = num_ops_pos + 2
        # raw_bytes layout:
        #   no-flags: [morph][opc_lo][opc_hi][vt*n_results][num_ops][op_idx]
        #   has-flags: [morph][opc_lo][opc_hi][flags][vt*n_results][num_ops][op_idx]
        raw_bytes = data[i:raw_end]

        raw_entries.append({
            "opcode":      opc,
            "mnemonic":    opcode_map[opc],
            "n_results":   n_results,
            "morph_byte":  b,
            "flags_byte":  flags_byte,
            "opc_lo":      opc_lo,
            "opc_hi":      opc_hi,
            "file_offset": i,
            "mt_offset":   i - mt_offset,
            "arm_len":     arm_len,
            "input_mvt":   input_mvt,
            "result_mvts": result_mvts,
            "num_ops":     num_ops,
            "op_idx":      op_idx,
            "raw_bytes":   raw_bytes,
        })

    # Sort by opcode then file offset; assign hit_num per-opcode
    raw_entries.sort(key=lambda r: (r["opcode"], r["file_offset"]))
    hit_counter: Counter = Counter()

    entries: list[MatcherEntry] = []
    for r in raw_entries:
        hit_counter[r["opcode"]] += 1
        mvts  = r["result_mvts"]
        types = tuple(mvt_map.get(m, "") for m in mvts)

        entries.append(MatcherEntry(
            opcode           = r["opcode"],
            mnemonic         = r["mnemonic"],
            hit_num          = hit_counter[r["opcode"]],
            n_results        = r["n_results"],
            morph_byte       = r["morph_byte"],
            flags_byte       = r["flags_byte"],
            opc_lo           = r["opc_lo"],
            opc_hi           = r["opc_hi"],
            file_offset      = r["file_offset"],
            mt_offset        = r["mt_offset"],
            arm_len          = r["arm_len"],
            input_mvt        = r["input_mvt"],
            input_mvt_type   = mvt_map.get(r["input_mvt"], ""),
            result_mvts      = mvts,
            result_mvt_types = types,
            num_ops          = r["num_ops"],
            op_idx           = r["op_idx"],
            raw_bytes        = r["raw_bytes"],
        ))

    logger.info(f"MatcherTable scan: {len(entries):,} entries found")
    return tuple(entries)
