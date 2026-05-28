"""
extractor/extractor.py — LLVM MatcherTable bounds locator.

Provides:
  get_matchertable_bounds()    — locate the table in a binary
"""

from __future__ import annotations

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
