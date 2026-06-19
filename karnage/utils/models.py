"""Immutable data structures shared across the karnage pipeline.

All public types are frozen dataclasses so they can be used as dict keys,
stored in sets, and passed between pipeline stages without defensive copies.
``FlipResult`` is the only mutable type because it is built incrementally
during a test run.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PatchSpec:
    """Specification for one bit-flip experiment derived from a flip site.

    Attributes:
        flip_id:       Sequential identifier assigned by the iterator in runner.py.
        func_name:     Fully demangled C++ name of the containing function.
        site_vma:      Linker VMA of the byte to patch (opcode or 2nd byte).
        instr_type:    ``"short_jcc"``, ``"long_jcc"``, or ``"cmov"``.
        opcode_before: Human-readable mnemonic before the flip (e.g. ``"je"``).
        opcode_after:  Human-readable mnemonic after the flip (e.g. ``"jne"``).
        flip_mask:     Bitmask XOR'd onto the target byte (always ``0x01``).
    """

    flip_id: int
    func_name: str
    site_vma: int
    instr_type: str
    opcode_before: str
    opcode_after: str
    flip_mask: int


@dataclass
class FlipResult:
    """Outcome of running a single bit-flip experiment.

    Mutable so it can be populated incrementally during a test run.

    Attributes:
        spec:           The :class:`PatchSpec` that produced this result.
        crashed:        ``True`` if GDB or the inferior exited non-zero.
        timed_out:      ``True`` if the flip run exceeded the per-flip timeout.
        script_ran:     ``True`` if ``_wrapper.py`` reached its ``_done`` sentinel.
        ptx_changed:    ``True`` if the generated PTX differs from the baseline.
        stdout_changed: ``True`` if the application's stdout differs from the baseline.
    """

    spec: PatchSpec
    crashed: bool
    timed_out: bool
    script_ran: bool
    ptx_changed: bool
    stdout_changed: bool
