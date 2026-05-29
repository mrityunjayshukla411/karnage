"""
extractor/extractor.py --- LLVM MatcherTable bounds locator.

Provides:
  get_matchertable_bounds()    --- locate the table in a binary
  walk()   --- entry point; returns tuple[MatcherEntry, ...]
"""

from __future__ import annotations

from pathlib import Path
import re
import json
from collections import Counter
from dataclasses import dataclass

from karnage.utils.logger import logger
from karnage.utils.parser import (
    find_symbol_linker_vma,
    linker_vma_to_file_offset,
    estimate_symbol_byte_size,
)

from karnage.utils.models import (
    OpcodeInfo,
    FlipInfo,
    AdjacencyEntry
)

from karnage.utils.exceptions import MatcherTableLoadError

from karnage.utils.targets import TargetBackend

from karnage.utils.models import MatcherEntry

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
# VBR helpers
# ---------------------------------------------------------------------------

def _read_vbr(data: bytes, pos: int, end: int) -> tuple[int, int]:
    """Read LLVM VBR unsigned integer.  Returns (value, new_pos)."""
    if pos >= end:
        return 0, pos
    val = data[pos]; pos += 1
    if not (val & 0x80):
        return val, pos
    result = val & 0x7F
    shift  = 7
    while pos < end:
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        shift  += 7
        if not (b & 0x80):
            break
    return result, pos


def _skip_vbr(data: bytes, pos: int, end: int) -> int:
    """Advance past a VBR value without returning it."""
    _, pos = _read_vbr(data, pos, end)
    return pos


def _skip_signed_vbr(data: bytes, pos: int, end: int) -> int:
    """Advance past a signed VBR (GetSignedVBR format)."""
    while pos < end:
        b = data[pos]; pos += 1
        if not (b & 0x80):
            break
    return pos


def _skip_get_simple_vt(data: bytes, pos: int, end: int) -> int:
    """Advance past a getSimpleVT() encoding (VBR of the MVT enum value)."""
    return _skip_vbr(data, pos, end)


def _read_get_simple_vt(data: bytes, pos: int, end: int) -> tuple[int, int]:
    """Read getSimpleVT() encoding.  Returns (mvt_value, new_pos)."""
    return _read_vbr(data, pos, end)


# ---------------------------------------------------------------------------
# Variant info for MorphNodeTo* and EmitNode* opcodes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _NodeVariant:
    n_results:    int | None  # None means "read from byte"
    has_flags:    bool        # explicit EmitNodeInfo flags byte
    hwmode_vts:   bool        # VTs via getHwModeVT (1 byte) instead of getSimpleVT (VBR)
    is_morph:     bool        # True=MorphNodeTo (terminal), False=EmitNode (continues)


def _build_node_variant_map(enum_map: dict[str, int]) -> dict[int, _NodeVariant]:
    """
    Derive {opcode_byte: _NodeVariant} for every MorphNodeTo* / EmitNode* opcode.

    Byte-layout determined from SelectionDAGISel.cpp:
      - has_flags=True  → explicit EmitNodeInfo flags byte after TargetOpc
      - n_results=None  → NumVTs read from one byte; otherwise implicit from opcode
    """
    morph_generic = re.compile(r'OPC_MorphNodeTo$')
    morph_hwmode  = re.compile(r'OPC_MorphNodeToByHwMode$')
    morph_bare    = re.compile(r'OPC_MorphNodeTo(\d+)$')
    morph_suffix  = re.compile(r'OPC_MorphNodeTo(\d+)(None|Chain|GlueInput|GlueOutput)$')
    emit_generic  = re.compile(r'OPC_EmitNode$')
    emit_hwmode   = re.compile(r'OPC_EmitNodeByHwMode$')
    emit_bare     = re.compile(r'OPC_EmitNode(\d+)$')
    emit_suffix   = re.compile(r'OPC_EmitNode(\d+)(None|Chain)$')

    result: dict[int, _NodeVariant] = {}

    for name, val in enum_map.items():
        if morph_generic.fullmatch(name):
            v = _NodeVariant(n_results=None,  has_flags=True,  hwmode_vts=False, is_morph=True)
        elif morph_hwmode.fullmatch(name):
            v = _NodeVariant(n_results=None,  has_flags=True,  hwmode_vts=True,  is_morph=True)
        elif (m := morph_bare.fullmatch(name)):
            v = _NodeVariant(n_results=int(m.group(1)), has_flags=True,  hwmode_vts=False, is_morph=True)
        elif (m := morph_suffix.fullmatch(name)):
            v = _NodeVariant(n_results=int(m.group(1)), has_flags=False, hwmode_vts=False, is_morph=True)
        elif emit_generic.fullmatch(name):
            v = _NodeVariant(n_results=None,  has_flags=True,  hwmode_vts=False, is_morph=False)
        elif emit_hwmode.fullmatch(name):
            v = _NodeVariant(n_results=None,  has_flags=True,  hwmode_vts=True,  is_morph=False)
        elif (m := emit_bare.fullmatch(name)):
            v = _NodeVariant(n_results=int(m.group(1)), has_flags=True,  hwmode_vts=False, is_morph=False)
        elif (m := emit_suffix.fullmatch(name)):
            v = _NodeVariant(n_results=int(m.group(1)), has_flags=False, hwmode_vts=False, is_morph=False)
        else:
            continue
        result[val] = v

    return result


# ---------------------------------------------------------------------------
# Per-opcode byte-skip dispatch table
# ---------------------------------------------------------------------------

def _build_skip_table(
    enum_map:     dict[str, int],
    node_variants: dict[int, _NodeVariant],
) -> dict[int, int | str]:
    """
    Build a lightweight dispatch map:  opcode_byte → skip_descriptor.

    Descriptor meanings:
        int  n   --- skip exactly n bytes
        'vbr'    --- skip one VBR (unsigned)
        'svbr'   --- skip one signed VBR (GetSignedVBR)
        'mvt'    --- skip one getSimpleVT() VBR (same encoding as 'vbr')
        'hwmode' --- skip 1 byte (HwMode index)
        special  --- None (handled by the walk loop directly)

    Opcodes in node_variants (MorphNodeTo/EmitNode) and control flow opcodes
    (Scope, SwitchType, SwitchOpcode, CompleteMatch) return None; the walk
    loop handles them explicitly.
    """
    E = enum_map

    def e(name: str) -> int | None:
        return E.get(name)

    skip: dict[int, int | str | None] = {}

    def add(names, descriptor):
        for n in names:
            v = e(n)
            if v is not None:
                skip[v] = descriptor

    # --- 0-byte opcodes ---
    add([
        'OPC_RecordNode', 'OPC_RecordMemRef', 'OPC_CaptureGlueInput',
        'OPC_CaptureDeactivationSymbol', 'OPC_MoveParent',
        'OPC_MoveChild0', 'OPC_MoveChild1', 'OPC_MoveChild2', 'OPC_MoveChild3',
        'OPC_MoveChild4', 'OPC_MoveChild5', 'OPC_MoveChild6', 'OPC_MoveChild7',
        'OPC_MoveSibling0', 'OPC_MoveSibling1', 'OPC_MoveSibling2', 'OPC_MoveSibling3',
        'OPC_MoveSibling4', 'OPC_MoveSibling5', 'OPC_MoveSibling6', 'OPC_MoveSibling7',
        'OPC_RecordChild0', 'OPC_RecordChild1', 'OPC_RecordChild2', 'OPC_RecordChild3',
        'OPC_RecordChild4', 'OPC_RecordChild5', 'OPC_RecordChild6', 'OPC_RecordChild7',
        'OPC_CheckPatternPredicate0', 'OPC_CheckPatternPredicate1',
        'OPC_CheckPatternPredicate2', 'OPC_CheckPatternPredicate3',
        'OPC_CheckPatternPredicate4', 'OPC_CheckPatternPredicate5',
        'OPC_CheckPatternPredicate6', 'OPC_CheckPatternPredicate7',
        'OPC_CheckPredicate0', 'OPC_CheckPredicate1', 'OPC_CheckPredicate2',
        'OPC_CheckPredicate3', 'OPC_CheckPredicate4', 'OPC_CheckPredicate5',
        'OPC_CheckPredicate6', 'OPC_CheckPredicate7',
        'OPC_CheckTypeI32', 'OPC_CheckTypeI64',
        'OPC_CheckChild0TypeI32', 'OPC_CheckChild1TypeI32', 'OPC_CheckChild2TypeI32',
        'OPC_CheckChild3TypeI32', 'OPC_CheckChild4TypeI32', 'OPC_CheckChild5TypeI32',
        'OPC_CheckChild6TypeI32', 'OPC_CheckChild7TypeI32',
        'OPC_CheckChild0TypeI64', 'OPC_CheckChild1TypeI64', 'OPC_CheckChild2TypeI64',
        'OPC_CheckChild3TypeI64', 'OPC_CheckChild4TypeI64', 'OPC_CheckChild5TypeI64',
        'OPC_CheckChild6TypeI64', 'OPC_CheckChild7TypeI64',
        'OPC_CheckImmAllOnesV', 'OPC_CheckImmAllZerosV',
        'OPC_CheckFoldableChainNode',
        'OPC_EmitMergeInputChains1_0', 'OPC_EmitMergeInputChains1_1',
        'OPC_EmitMergeInputChains1_2',
        'OPC_EmitConvertToTarget0', 'OPC_EmitConvertToTarget1',
        'OPC_EmitConvertToTarget2', 'OPC_EmitConvertToTarget3',
        'OPC_EmitConvertToTarget4', 'OPC_EmitConvertToTarget5',
        'OPC_EmitConvertToTarget6', 'OPC_EmitConvertToTarget7',
    ], 0)

    # --- 1-byte opcodes ---
    add([
        'OPC_MoveChild',            # ChildNo
        'OPC_MoveSibling',          # SiblingNo
        'OPC_CheckSame',            # RecNo
        'OPC_CheckChild0Same', 'OPC_CheckChild1Same',
        'OPC_CheckChild2Same', 'OPC_CheckChild3Same',
        'OPC_CheckPatternPredicate',  # PredNo
        'OPC_CheckPredicate',         # PredNo
        'OPC_CheckCondCode',          # ISD::CondCode
        'OPC_CheckChild2CondCode',    # ISD::CondCode
        'OPC_CheckTypeByHwMode',      # HwMode index
        'OPC_CheckChild0TypeByHwMode', 'OPC_CheckChild1TypeByHwMode',
        'OPC_CheckChild2TypeByHwMode', 'OPC_CheckChild3TypeByHwMode',
        'OPC_CheckChild4TypeByHwMode', 'OPC_CheckChild5TypeByHwMode',
        'OPC_CheckChild6TypeByHwMode', 'OPC_CheckChild7TypeByHwMode',
        'OPC_CheckComplexPat0', 'OPC_CheckComplexPat1', 'OPC_CheckComplexPat2',
        'OPC_CheckComplexPat3', 'OPC_CheckComplexPat4', 'OPC_CheckComplexPat5',
        'OPC_CheckComplexPat6', 'OPC_CheckComplexPat7',
        'OPC_EmitConvertToTarget',    # RecNo
        'OPC_EmitRegisterI32',        # RegNo
        'OPC_EmitRegisterI64',        # RegNo
        'OPC_EmitCopyToReg0', 'OPC_EmitCopyToReg1', 'OPC_EmitCopyToReg2',
        'OPC_EmitCopyToReg3', 'OPC_EmitCopyToReg4', 'OPC_EmitCopyToReg5',
        'OPC_EmitCopyToReg6', 'OPC_EmitCopyToReg7',
    ], 1)

    # --- 2-byte opcodes ---
    add([
        'OPC_CheckOpcode',              # opc_lo, opc_hi
        'OPC_CheckPatternPredicateTwoByte',  # PredNo_lo, PredNo_hi
        'OPC_CheckComplexPat',          # CPNum + RecNo
        'OPC_EmitCopyToReg',            # RecNo + DestPhysReg
        'OPC_EmitNodeXForm',            # XFormNo + RecNo
        'OPC_CheckTypeResByHwMode',     # Res + HwMode
    ], 2)

    # CheckChild*Opcode --- 2 bytes (opc_lo + opc_hi), not in all LLVM versions
    for ch in range(8):
        v = e(f'OPC_CheckChild{ch}Opcode')
        if v is not None:
            skip[v] = 2

    # --- 3-byte opcodes ---
    add([
        'OPC_EmitCopyToRegTwoByte',     # RecNo + DestPhysReg_lo + DestPhysReg_hi
    ], 3)

    # --- 4-byte opcodes ---
    add(['OPC_Coverage'], 4)

    # --- VBR opcodes (unsigned) ---
    add([
        'OPC_CheckAndImm', 'OPC_CheckOrImm',
    ], 'vbr')

    # --- signed VBR opcodes ---
    add([
        'OPC_CheckInteger',
        'OPC_CheckChild0Integer', 'OPC_CheckChild1Integer',
        'OPC_CheckChild2Integer', 'OPC_CheckChild3Integer',
        'OPC_CheckChild4Integer',
    ], 'svbr')

    # --- MVT VBR (getSimpleVT) ---
    add([
        'OPC_CheckType',           # MVT via getSimpleVT
        'OPC_CheckValueType',      # MVT via getSimpleVT
        'OPC_CheckChild0Type', 'OPC_CheckChild1Type', 'OPC_CheckChild2Type',
        'OPC_CheckChild3Type', 'OPC_CheckChild4Type', 'OPC_CheckChild5Type',
        'OPC_CheckChild6Type', 'OPC_CheckChild7Type',
    ], 'mvt')

    # OPC_CheckTypeRes: 1 byte Res + MVT VBR --- special
    # OPC_CheckPredicateWithOperands: OpNum + OpNum bytes + PredNo --- special
    # OPC_EmitMergeInputChains: NumChains + NumChains bytes --- special
    # OPC_EmitRegister: MVT VBR + RegNo byte --- special
    # OPC_EmitRegisterByHwMode: 1 byte HwMode + RegNo byte --- special
    # OPC_EmitRegister2: MVT VBR + 2 byte RegNo --- special
    # OPC_EmitRegisterByHwMode2: 1 byte HwMode + 2 byte RegNo --- special
    # OPC_EmitIntegerI8/I16/I32/I64: no VT byte + signed VBR value --- special
    # OPC_EmitIntegerByHwMode: 1 byte HwMode + signed VBR value --- special
    # OPC_EmitInteger: MVT VBR + signed VBR value --- special

    # Mark special opcodes explicitly (value None means "handled in walk loop")
    _specials = [
        'OPC_Scope', 'OPC_SwitchType', 'OPC_SwitchOpcode', 'OPC_CompleteMatch',
        'OPC_CheckTypeRes',
        'OPC_CheckPredicateWithOperands',
        'OPC_EmitMergeInputChains',
        'OPC_EmitRegister', 'OPC_EmitRegisterByHwMode',
        'OPC_EmitRegister2', 'OPC_EmitRegisterByHwMode2',
        'OPC_EmitInteger',
        'OPC_EmitIntegerI8', 'OPC_EmitIntegerI16',
        'OPC_EmitIntegerI32', 'OPC_EmitIntegerI64',
        'OPC_EmitIntegerByHwMode',
    ]
    for nm in _specials:
        v = e(nm)
        if v is not None:
            skip[v] = None  # walker handles these explicitly

    # Node variants (MorphNodeTo*, EmitNode*) --- also handled explicitly
    for ov in node_variants:
        skip[ov] = None

    return skip


# ---------------------------------------------------------------------------
# Node-body parser (advances pos past MorphNodeTo / EmitNode payload)
# ---------------------------------------------------------------------------

def _parse_node_body(
    data:    bytes,
    pos:     int,
    end:     int,
    variant: _NodeVariant,
    mvt_map: dict[int, str],
    mt_offset: int,
    morph_byte: int,
    ctx_mvt: int,
    ctx_arm_len: int,
    opcode_map: dict[int, str],
    entry_file_offset: int,
) -> tuple[dict | None, int]:
    """
    Parse the bytes following an OPC_MorphNodeTo* / OPC_EmitNode* opcode byte.

    ALWAYS advances pos past all bytes of this node, even when returning fields=None.
    Returning fields=None means "don't record", not "backtrack".  This is critical for
    EmitNode variants: the walker must keep the position correct so subsequent bytes are
    read as real opcodes, not as operand data from a half-parsed EmitNode body.
    """
    valid = True   # set False when this entry should not be recorded

    # --- TargetOpc (2 bytes) ---
    if pos + 1 >= end:
        return None, end
    opc_lo = data[pos]; pos += 1
    opc_hi = data[pos]; pos += 1
    opc    = opc_lo | (opc_hi << 8)

    if opc not in opcode_map:
        valid = False

    # --- Flags byte (only for has_flags variants) ---
    flags_byte = 0
    if variant.has_flags:
        if pos >= end:
            return None, end
        flags_byte = data[pos]; pos += 1

    # --- NumVTs (from byte when n_results is None) ---
    n_results = variant.n_results
    if n_results is None:
        if pos >= end:
            return None, end
        n_results = data[pos]; pos += 1

    # --- Result VT bytes (getSimpleVT or HwMode per variant) ---
    result_mvts: list[int] = []
    for _ in range(n_results):
        if variant.hwmode_vts:
            if pos >= end:
                return None, end
            mvt = data[pos]; pos += 1
        else:
            mvt, pos = _read_get_simple_vt(data, pos, end)
        result_mvts.append(mvt)

    if any(m not in mvt_map for m in result_mvts):
        valid = False

    # --- NumOps ---
    if pos >= end:
        return None, end
    num_ops = data[pos]; pos += 1

    if not (0 <= num_ops <= 64):
        valid = False
        # Still need to advance past RecNos: best-effort, assume NumOps byte
        # was garbage, treat as 0 ops so we don't consume unbounded data.
        num_ops = 0

    # --- First operand index (first byte of first RecNo, for MatcherEntry.op_idx) ---
    op_idx = data[pos] if pos < end else 0

    # --- Skip all RecNo VBR values ---
    for _ in range(num_ops):
        pos = _skip_vbr(data, pos, end)

    if not valid:
        return None, pos

    # raw_bytes spans morph_byte through op_idx (matching existing format)
    raw_bytes_end = entry_file_offset + 3 + (1 if variant.has_flags else 0) + n_results + 2
    raw_bytes = data[entry_file_offset : raw_bytes_end]

    return {
        'opcode':      opc,
        'mnemonic':    opcode_map[opc],
        'n_results':   n_results,
        'morph_byte':  morph_byte,
        'flags_byte':  flags_byte,
        'opc_lo':      opc_lo,
        'opc_hi':      opc_hi,
        'file_offset': entry_file_offset,
        'mt_offset':   entry_file_offset - mt_offset,
        'arm_len':     ctx_arm_len,
        'input_mvt':   ctx_mvt,
        'result_mvts': tuple(result_mvts),
        'num_ops':     num_ops,
        'op_idx':      op_idx,
        'raw_bytes':   raw_bytes,
    }, pos


# ---------------------------------------------------------------------------
# Main walker
# ---------------------------------------------------------------------------

def walk(
    data:        bytes,
    mt_offset:   int,
    mt_size:     int,
    enum_map:    dict[str, int],
    opcode_map:  dict[int, str],
    mvt_map:     dict[int, str],
) -> tuple[MatcherEntry, ...]:
    """
    Walk the MatcherTable bytecode from the beginning, following the decision-
    tree structure, and collect every OPC_MorphNodeTo* hit.

    Unlike the opportunistic scan(), this function correctly handles:
    - OPC_Scope (try / fall-through)
    - OPC_SwitchType (propagates input MVT context into each arm)
    - OPC_SwitchOpcode (pushes each arm body to the work queue)
    - All other opcodes (skipped by consuming their exact byte footprint)

    Returns tuple[MatcherEntry, ...] with the same schema as extractor.scan().
    """
    mt_end = mt_offset + mt_size

    opc_scope      = enum_map.get('OPC_Scope')
    opc_switchtype = enum_map.get('OPC_SwitchType')
    opc_switchopc  = enum_map.get('OPC_SwitchOpcode')
    opc_complete   = enum_map.get('OPC_CompleteMatch')

    opc_typecheck_res     = enum_map.get('OPC_CheckTypeRes')
    opc_predwithops       = enum_map.get('OPC_CheckPredicateWithOperands')
    opc_mergechains       = enum_map.get('OPC_EmitMergeInputChains')
    opc_emitreg           = enum_map.get('OPC_EmitRegister')
    opc_emitreg_hwm       = enum_map.get('OPC_EmitRegisterByHwMode')
    opc_emitreg2          = enum_map.get('OPC_EmitRegister2')
    opc_emitreg_hwm2      = enum_map.get('OPC_EmitRegisterByHwMode2')
    opc_emitint           = enum_map.get('OPC_EmitInteger')
    opc_emitint_i8        = enum_map.get('OPC_EmitIntegerI8')
    opc_emitint_i16       = enum_map.get('OPC_EmitIntegerI16')
    opc_emitint_i32       = enum_map.get('OPC_EmitIntegerI32')
    opc_emitint_i64       = enum_map.get('OPC_EmitIntegerI64')
    opc_emitint_hwm       = enum_map.get('OPC_EmitIntegerByHwMode')

    node_variants = _build_node_variant_map(enum_map)
    skip_table    = _build_skip_table(enum_map, node_variants)

    # Work queue items: (pos, limit, ctx_mvt, ctx_arm_len)
    #   pos          --- current byte position in data
    #   limit        --- one past the last byte of this sub-tree (mt_end for root)
    #   ctx_mvt      --- MVT byte from the enclosing OPC_SwitchType arm (0 = unknown)
    #   ctx_arm_len  --- arm_size from that arm (0 = unknown)
    queue: list[tuple[int, int, int, int]] = [
        (mt_offset, mt_end, 0, 0)
    ]

    raw_entries: list[dict] = []

    while queue:
        pos, limit, ctx_mvt, ctx_arm_len = queue.pop()

        while pos < limit:
            b = data[pos]; pos += 1

            # ----------------------------------------------------------------
            # Control-flow opcodes (handled inline)
            # ----------------------------------------------------------------

            if b == opc_scope:
                # Multi-arm layout:
                #   [NumToSkip1 VBR][arm1: N1 bytes][NumToSkip2 VBR][arm2: N2 bytes]...[0x00]
                #
                # FailIndex stored in MatchScope points to the *next arm's NumToSkip VBR*,
                # not to raw opcodes.  The failure handler (SelectionDAGISel.cpp ~4516)
                # reads NumToSkip from FailIndex itself before resuming execution.
                # We therefore iterate all arms here and push each body separately.
                while pos < limit:
                    skip, pos = _read_vbr(data, pos, limit)
                    if skip == 0:
                        break                   # terminator
                    arm_body_start = pos
                    arm_body_end   = pos + skip
                    if arm_body_end > limit:
                        pos = limit
                        break
                    queue.append((arm_body_start, arm_body_end, ctx_mvt, ctx_arm_len))
                    pos = arm_body_end
                break                           # all arms queued; current path ends

            if b == opc_switchtype:
                # Arms: [CaseSize VBR][MVT getSimpleVT][body: CaseSize bytes] ... [0x00]
                while pos < limit:
                    case_size, pos = _read_vbr(data, pos, limit)
                    if case_size == 0:
                        break                       # terminator
                    mvt, pos = _read_get_simple_vt(data, pos, limit)
                    arm_body_start = pos
                    arm_body_end   = pos + case_size
                    if arm_body_end > limit:
                        pos = limit
                        break
                    queue.append((arm_body_start, arm_body_end, mvt, case_size))
                    pos = arm_body_end
                break                               # SwitchType exhausts the current scope

            if b == opc_switchopc:
                # Arms: [CaseSize VBR][opc_lo][opc_hi][body: CaseSize bytes] ... [0x00]
                while pos < limit:
                    case_size, pos = _read_vbr(data, pos, limit)
                    if case_size == 0:
                        break
                    if pos + 2 > limit:
                        pos = limit
                        break
                    pos += 2                        # skip opc_lo + opc_hi
                    arm_body_start = pos
                    arm_body_end   = pos + case_size
                    if arm_body_end > limit:
                        pos = limit
                        break
                    queue.append((arm_body_start, arm_body_end, ctx_mvt, ctx_arm_len))
                    pos = arm_body_end
                break                               # SwitchOpcode exhausts current scope

            if b == opc_complete:
                # OPC_CompleteMatch: NumResults (1 byte) + NumResults VBR ResSlots (terminal)
                if pos >= limit:
                    break
                num_results = data[pos]; pos += 1
                for _ in range(num_results):
                    pos = _skip_vbr(data, pos, limit)
                break                               # terminal

            # ----------------------------------------------------------------
            # MorphNodeTo* and EmitNode* (node variants)
            # ----------------------------------------------------------------

            if b in node_variants:
                variant = node_variants[b]
                morph_file_off = pos - 1           # position of the opcode byte
                fields, pos = _parse_node_body(
                    data, pos, limit, variant, mvt_map,
                    mt_offset, b, ctx_mvt, ctx_arm_len, opcode_map,
                    morph_file_off,
                )
                if fields is not None:
                    raw_entries.append(fields)
                if variant.is_morph:
                    break                           # terminal
                continue                           # EmitNode continues

            # ----------------------------------------------------------------
            # Special multi-field opcodes (not node variants, not control flow)
            # ----------------------------------------------------------------

            if b == opc_typecheck_res:
                # OPC_CheckTypeRes: 1 byte Res + MVT VBR
                pos += 1
                pos = _skip_get_simple_vt(data, pos, limit)
                continue

            if b == opc_predwithops:
                # OPC_CheckPredicateWithOperands: OpNum + OpNum bytes + PredNo
                if pos >= limit:
                    break
                op_num = data[pos]; pos += 1
                pos += op_num
                pos += 1                            # PredNo
                continue

            if b == opc_mergechains:
                # OPC_EmitMergeInputChains: NumChains + NumChains bytes
                if pos >= limit:
                    break
                num_chains = data[pos]; pos += 1
                pos += num_chains
                continue

            if b == opc_emitreg:
                # MVT VBR + RegNo (1 byte)
                pos = _skip_get_simple_vt(data, pos, limit)
                pos += 1
                continue

            if b == opc_emitreg_hwm:
                # HwMode index (1 byte) + RegNo (1 byte)
                pos += 2
                continue

            if b == opc_emitreg2:
                # MVT VBR + RegNo (2 bytes)
                pos = _skip_get_simple_vt(data, pos, limit)
                pos += 2
                continue

            if b == opc_emitreg_hwm2:
                # HwMode index (1 byte) + RegNo (2 bytes)
                pos += 3
                continue

            if b in (opc_emitint_i8, opc_emitint_i16, opc_emitint_i32, opc_emitint_i64):
                # VT is implicit; value is signed VBR
                pos = _skip_signed_vbr(data, pos, limit)
                continue

            if b == opc_emitint:
                # OPC_EmitInteger: MVT VBR + signed VBR value
                pos = _skip_get_simple_vt(data, pos, limit)
                pos = _skip_signed_vbr(data, pos, limit)
                continue

            if b == opc_emitint_hwm:
                # HwMode index (1 byte) + signed VBR value
                pos += 1
                pos = _skip_signed_vbr(data, pos, limit)
                continue

            # ----------------------------------------------------------------
            # Simple fixed-skip or single-VBR opcodes
            # ----------------------------------------------------------------

            descriptor = skip_table.get(b)
            if descriptor is None:
                # Unknown opcode --- gap in our table or data corruption.
                # Abandon this sub-tree; guessing a 1-byte skip would cascade.
                logger.debug(f"Walker: unknown opcode 0x{b:02x} at pos 0x{pos-1:x}, abandoning branch")
                break
            if isinstance(descriptor, int):
                pos += descriptor
            elif descriptor == 'vbr':
                pos = _skip_vbr(data, pos, limit)
            elif descriptor == 'svbr':
                pos = _skip_signed_vbr(data, pos, limit)
            elif descriptor == 'mvt':
                pos = _skip_get_simple_vt(data, pos, limit)

    # Sort by opcode then file offset; assign hit_num per-opcode
    raw_entries.sort(key=lambda r: (r['opcode'], r['file_offset']))
    hit_counter: Counter = Counter()

    entries: list[MatcherEntry] = []
    for r in raw_entries:
        hit_counter[r['opcode']] += 1
        mvts  = r['result_mvts']
        types = tuple(mvt_map.get(m, '') for m in mvts)

        entries.append(MatcherEntry(
            opcode           = r['opcode'],
            mnemonic         = r['mnemonic'],
            hit_num          = hit_counter[r['opcode']],
            n_results        = r['n_results'],
            morph_byte       = r['morph_byte'],
            flags_byte       = r['flags_byte'],
            opc_lo           = r['opc_lo'],
            opc_hi           = r['opc_hi'],
            file_offset      = r['file_offset'],
            mt_offset        = r['mt_offset'],
            arm_len          = r['arm_len'],
            input_mvt        = r['input_mvt'],
            input_mvt_type   = mvt_map.get(r['input_mvt'], ''),
            result_mvts      = mvts,
            result_mvt_types = types,
            num_ops          = r['num_ops'],
            op_idx           = r['op_idx'],
            raw_bytes        = r['raw_bytes'],
        ))

    logger.info(f"MatcherTable walk: {len(entries):,} entries found")
    return tuple(entries)

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
                # Duplicate (lo, hi) across different mnemonics --- keep first.
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