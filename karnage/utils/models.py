"""Immutable data structures shared across the karnage pipeline.

All public types are frozen dataclasses so they can be used as dict keys,
stored in sets, and passed between pipeline stages without defensive copies.
``FlipResult`` is the only mutable type because it is built incrementally
during a test run.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MatcherEntry:
    """One fully-decoded OPC_MorphNodeTo* hit inside the LLVM MatcherTable.

    Produced by the extractor's walker and consumed by the CLI serialiser.
    Each instance corresponds to a single instruction-selection pattern arm
    found at a specific byte offset in the MatcherTable.

    Attributes:
        opcode:           Integer opcode value (``opc_lo | opc_hi << 8``).
        mnemonic:         Assembler mnemonic string, e.g. ``"fma.rn.f32"``.
        hit_num:          Zero-based index of this entry within all entries
                          for the same mnemonic, in walk order.
        n_results:        Number of result value types produced by this node.
        morph_byte:       Raw OPC_MorphNodeTo* opcode byte from the MatcherTable.
        flags_byte:       EmitNodeInfo flags byte; zero for space-optimised variants.
        opc_lo:           Low byte of the target opcode (``data[morph_offset + 1]``).
        opc_hi:           High byte of the target opcode (``data[morph_offset + 2]``).
        file_offset:      Absolute byte offset of the morph opcode within the
                          shared-object file on disk.
        mt_offset:        Byte offset of the morph opcode relative to the start
                          of the MatcherTable symbol.
        arm_len:          Byte length of the full pattern arm.
        input_mvt:        Integer MVT value of the primary input operand.
        input_mvt_type:   Human-readable MVT name; empty string if unknown.
        result_mvts:      Tuple of integer MVT values for each result.
        result_mvt_types: Tuple of human-readable MVT names, parallel to
                          ``result_mvts``.
        num_ops:          Total operand count for the emitted node.
        op_idx:           Index of the first operand within the operand list.
        raw_bytes:        Raw bytes of the pattern arm as captured from the binary.
    """

    opcode: int
    mnemonic: str
    hit_num: int
    n_results: int
    morph_byte: int
    flags_byte: int
    opc_lo: int
    opc_hi: int
    file_offset: int
    mt_offset: int
    arm_len: int
    input_mvt: int
    input_mvt_type: str
    result_mvts: tuple[int, ...]
    result_mvt_types: tuple[str, ...]
    num_ops: int
    op_idx: int
    raw_bytes: bytes


@dataclass(frozen=True)
class FlipInfo:
    """Description of a single bit flip applied to an opcode byte.

    Attributes:
        byte: Which opcode byte is flipped --- either ``"opc_lo"`` or ``"opc_hi"``.
        bit:  Bit position within that byte (0 = LSB, 7 = MSB).
        mask: Bitmask for the flip, equal to ``1 << bit``.
    """

    byte: str
    bit: int
    mask: int


@dataclass(frozen=True)
class OpcodeInfo:
    """Lightweight summary of one opcode entry used in adjacency analysis.

    Attributes:
        mnemonic:     Assembler mnemonic string.
        opcode:       Integer opcode value.
        opc_lo:       Low byte of the target opcode encoding.
        opc_hi:       High byte of the target opcode encoding.
        num_patterns: Number of MatcherTable pattern arms for this opcode.
    """

    mnemonic: str
    opcode: int
    opc_lo: int
    opc_hi: int
    num_patterns: int


@dataclass(frozen=True)
class AdjacencyEntry:
    """A pair of opcodes that differ by exactly one bit in their encoding.

    By convention ``b.opcode > a.opcode`` so each pair is stored only once.

    Attributes:
        a:    The lower-opcode instruction.
        b:    The higher-opcode instruction.
        flip: Which byte and bit position separate the two encodings.
    """

    a: OpcodeInfo
    b: OpcodeInfo
    flip: FlipInfo


@dataclass(frozen=True)
class PatchSpec:
    """Specification for one bit-flip experiment derived from adjacency.json.

    Attributes:
        flip_id:    Sequential identifier assigned by the iterator in runner.py.
        opcode_a:   Integer opcode of the instruction being flipped.
        mnemonic_a: Mnemonic of the source instruction.
        opcode_b:   Integer opcode of the target instruction after the flip.
        mnemonic_b: Mnemonic of the target instruction.
        flip_byte:  Which encoding byte is flipped (``"opc_lo"`` or ``"opc_hi"``).
        flip_bit:   Bit position within that byte (0 = LSB).
        flip_mask:  Bitmask for the flip, equal to ``1 << flip_bit``.
        patch_vmas: Linker VMAs of every MatcherTable byte that encodes
                    ``opcode_a``; one entry per pattern occurrence.
    """

    flip_id: int
    opcode_a: int
    mnemonic_a: str
    opcode_b: int
    mnemonic_b: str
    flip_byte: str
    flip_bit: int
    flip_mask: int
    patch_vmas: tuple[int, ...]


@dataclass
class FlipResult:
    """Outcome of running a single bit-flip experiment.

    Mutable so it can be populated incrementally during a test run.

    Attributes:
        spec:          The :class:`PatchSpec` that produced this result.
        crashed:       ``True`` if GDB or the inferior exited non-zero.
        script_ran:    ``True`` if ``_wrapper.py`` reached its ``_done`` sentinel
                       (i.e. the user script completed without raising).
        ptx_changed:   ``True`` if the generated PTX differs from the baseline.
        tensor_names:  Names of ``torch.Tensor`` globals saved by the wrapper.
        tensors_match: Per-tensor ``torch.allclose`` result keyed by name.
        max_abs_diffs: Per-tensor maximum absolute difference from the baseline.
    """

    spec: PatchSpec
    crashed: bool
    script_ran: bool
    ptx_changed: bool
    tensor_names: list[str]
    tensors_match: dict[str, bool]
    max_abs_diffs: dict[str, float]
