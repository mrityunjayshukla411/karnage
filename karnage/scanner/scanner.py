"""Target-independent Triton/LLVM function discovery and flip-site detection.

Two-step pipeline:

1. **Discover** --- run ``nm --demangle`` on the binary, keep T/t symbols whose
   demangled name matches at least one target-independent class prefix and no
   target-specific architecture prefix.

2. **Detect** --- for each discovered function, run ``objdump -d`` over a small
   window and collect three flip-candidate instruction classes:

   * Short Jcc (rel8)  --- opcodes 0x70–0x7F, 2 bytes, flip bit 0 of opcode byte
   * Long  Jcc (rel32) --- opcodes 0F 84–0F 8F, 6 bytes, flip bit 0 of 2nd byte
   * CMOV             --- opcodes 0F 40–0F 4F, 3+ bytes, flip bit 0 of 2nd byte

**Tier taxonomy** (reflects the Triton compilation pipeline)::

    Python @triton.jit
        ↓
    TTIR  (mlir::triton::*)           Tier 0 — all Triton targets
        ↓
    TTGIR (mlir::triton::gpu::*)      Tier 0 — all GPU targets
        ↓
    LLVM IR passes                    Tier 0 — InstCombiner, GVN, SROA, …
        ↓  [backend fork]
    NVPTX: SelectionDAG               Tier 1/2 — NVPTX-only (AMDGPU uses GlobalISel)
    AMD:   GlobalISel                 (not in libtriton.so)

Tier 0 functions are the only ones with an *architectural* cross-backend guarantee.
Tier 1–3 functions in libtriton.so are NVPTX-specific.

Results are returned as a :class:`ScanResult` and can be serialised to JSON
with :func:`scan_result_to_dict` / loaded back with :func:`scan_result_from_dict`.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from karnage.utils.exceptions import ParserError, ScannerError
from karnage.utils.logger import console, logger
from karnage.utils.parser import BinaryCache, _run_nm

# ---------------------------------------------------------------------------
# Target-independent include / exclude patterns
# ---------------------------------------------------------------------------

# Class/namespace prefixes that identify candidate functions.
# Any T/t nm symbol whose demangled name contains at least one of these is
# included (subject to the exclude filter below).
_INCLUDE_CLASSES: tuple[str, ...] = (
    # --- Tier 0: TTIR (mlir::triton::*) ---
    # Triton IR transforms run for ALL Triton targets before the backend fork.
    # Vendor sub-namespaces (nvidia_gpu, amd) are filtered by _TARGET_PREFIXES.
    "mlir::triton::",
    # --- Tier 0: TTGIR (mlir::triton::gpu::*) ---
    # TritonGPU IR transforms — shared by all GPU targets (NVIDIA, AMD, future).
    "mlir::triton::gpu::",
    # --- Tier 0: LLVM IR passes ---
    # Operate on llvm::Instruction / BasicBlock — target-agnostic by LLVM design.
    "InstCombinerImpl::",
    "InstCombiner::",
    "GVN::",
    "SROA::",
    "LoopVectorize::",
    "SLPVectorize::",
    "SimplifyCFGOpt::",
    "MemCpyOptimizer::",
    # --- Tier 1/2: SelectionDAG (NVPTX-only — AMDGPU uses GlobalISel) ---
    "DAGCombiner::",
    "SelectionDAGBuilder::",
    "SelectionDAGISel::",
    "SelectionDAGLegalize::",
    "LegalizeDAG::",
    "DAGTypeLegalizer::",
    "ScheduleDAGSDNodes::",
    "ScheduleDAGRRList::",
    # --- Tier 2/3: Target lowering base and register allocation (NVPTX-only) ---
    "TargetLowering::",
    "TargetLoweringBase::",
    "RegisterCoalescer::",
    "LiveRangeCalc::",
    "MachineScheduler::",
    "SchedBoundary::",
)

# Split _INCLUDE_CLASSES into two groups with different matching semantics.
# MLIR namespace prefixes need strict matching (see _include_match below);
# LLVM class names are unique enough for a plain substring check.
_MLIR_NS_INCLUDES: tuple[str, ...] = tuple(
    c for c in _INCLUDE_CLASSES if c.startswith("mlir::")
)
_CLASS_INCLUDES: tuple[str, ...] = tuple(
    c for c in _INCLUDE_CLASSES if not c.startswith("mlir::")
)


def _include_match(name: str) -> bool:
    """Return True if *name* genuinely belongs to one of our target namespaces.

    LLVM class patterns (DAGCombiner::, InstCombinerImpl::, …) use a plain
    substring match — they are unique enough that false positives are rare.

    MLIR namespace patterns (mlir::triton::, mlir::triton::gpu::) use a
    stricter check: the namespace must appear *before* the first ``<`` in the
    demangled name.  This prevents template instantiations such as
    ``llvm::SmallVectorTemplateBase<mlir::triton::X*>::growAndEmplaceBack``
    from being misidentified as Triton functions.
    """
    if any(c in name for c in _CLASS_INCLUDES):
        return True
    # For MLIR namespaces, only search the portion of the name before the
    # first '<' so that template argument types cannot trigger a match.
    search = name[: name.index("<")] if "<" in name else name
    return any(ns in search for ns in _MLIR_NS_INCLUDES)


# Any symbol that contains one of these strings is excluded even if it also
# matches an include pattern.  Covers LLVM target namespaces and Triton
# vendor-specific MLIR sub-namespaces.
_TARGET_PREFIXES: tuple[str, ...] = (
    # Triton MLIR vendor-specific sub-namespaces — must be checked before
    # the parent namespace matches in _INCLUDE_CLASSES.
    "nvidia_gpu",       # mlir::triton::nvidia_gpu::* — NVPTX-specific Triton ops
    "triton::amd",      # mlir::triton::amd::* — AMD-specific Triton ops
    "triton::gpu::amd", # mlir::triton::gpu::amd::* — AMD GPU-specific ops
    # LLVM target-specific namespaces
    "NVPTX",
    "AMDGPU",
    "X86",
    "AArch64",
    "ARM",
    "Thumb",
    "MIPS",
    "RISCV",
    "PPC",
    "PowerPC",
    "SystemZ",
    "Hexagon",
    "WebAssembly",
    "Wasm",
    "LoongArch",
    "M68k",
    "MSP430",
    "AVR",
    "BPF",
    "SPARC",
    "Sparc",
    "Lanai",
)

# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------

# Tier 0 — TTIR, TTGIR, and LLVM IR passes.
# These run before the backend fork → architectural cross-backend guarantee.
_TIER0_CLASSES: tuple[str, ...] = (
    "mlir::triton::",      # TTIR — all Triton backends
    "mlir::triton::gpu::", # TTGIR — all GPU backends
    "InstCombinerImpl::",
    "InstCombiner::",
    "GVN::",
    "SROA::",
    "LoopVectorize::",
    "SLPVectorize::",
    "SimplifyCFGOpt::",
    "MemCpyOptimizer::",
)

# Tier 1 — SelectionDAG visitors for individual operations (NVPTX-only).
# A single flip rarely crashes; it tends to misapply one optimisation.
# NOTE: these do NOT run for AMDGPU, which uses GlobalISel instead.
_TIER1_PATTERNS: tuple[str, ...] = (
    "DAGCombiner::visit",
)

# Tier 2 — SelectionDAG infrastructure (NVPTX-only).
# Higher crash probability than visitors.
_TIER2_CLASSES: tuple[str, ...] = (
    "DAGCombiner::",
    "LegalizeDAG::",
    "DAGTypeLegalizer::",
    "SelectionDAGBuilder::",
    "ScheduleDAGSDNodes::",
    "ScheduleDAGRRList::",
    "RegisterCoalescer::",
    "LiveRangeCalc::",
    "MachineScheduler::",
    "SchedBoundary::",
)

# Tier 3 — TargetLowering base, SelectionDAGISel drivers (NVPTX-only).
# Early branches are mostly null/type guards; flipping crashes almost always.
# Everything else defaults to tier 3.


def _tier_of(demangled: str) -> int:
    if any(c in demangled for c in _TIER0_CLASSES):
        return 0
    for pat in _TIER1_PATTERNS:
        if pat in demangled:
            return 1
    if any(c in demangled for c in _TIER2_CLASSES):
        return 2
    return 3


def _class_of(demangled: str) -> str:
    """Extract a meaningful class label from a demangled C++ symbol."""
    name = re.sub(r"^\(anonymous namespace\)::", "", demangled)
    # MLIR/Triton namespace hierarchy — use the namespace as the label.
    if "mlir::triton::gpu::" in name:
        return "mlir::triton::gpu"
    if "mlir::triton::" in name:
        return "mlir::triton"
    if "mlir::" in name:
        return "mlir"
    # LLVM C++ classes — take the outermost component.
    parts = name.split("::")
    return parts[0] if parts else demangled


# ---------------------------------------------------------------------------
# Instruction detection helpers
# ---------------------------------------------------------------------------

# objdump -d output line (Intel syntax):
#   "   <hex_addr>:   <raw_bytes>   <mnemonic> ..."
_OBJDUMP_LINE_RE = re.compile(
    r"^\s+([0-9a-f]+):\s+"  # address (hex, no 0x prefix in objdump output)
    r"((?:[0-9a-f]{2} ?)+)\s+"  # raw bytes
    r"(\S+)"  # mnemonic
)

# Condition-code inversion table (bit 0 flip of the condition nibble)
_JCC_NAMES: dict[int, str] = {
    0x70: "jo",   0x71: "jno",
    0x72: "jb",   0x73: "jae",
    0x74: "je",   0x75: "jne",
    0x76: "jbe",  0x77: "ja",
    0x78: "js",   0x79: "jns",
    0x7A: "jp",   0x7B: "jnp",
    0x7C: "jl",   0x7D: "jge",
    0x7E: "jle",  0x7F: "jg",
    # Long Jcc second bytes (0F 8x)
    0x84: "je",   0x85: "jne",
    0x82: "jb",   0x83: "jae",
    0x86: "jbe",  0x87: "ja",
    0x88: "js",   0x89: "jns",
    0x80: "jo",   0x81: "jno",
    0x8A: "jp",   0x8B: "jnp",
    0x8C: "jl",   0x8D: "jge",
    0x8E: "jle",  0x8F: "jg",
}

_CMOV_NAMES: dict[int, str] = {
    0x40: "cmovo",  0x41: "cmovno",
    0x42: "cmovb",  0x43: "cmovae",
    0x44: "cmove",  0x45: "cmovne",
    0x46: "cmovbe", 0x47: "cmova",
    0x48: "cmovs",  0x49: "cmovns",
    0x4A: "cmovp",  0x4B: "cmovnp",
    0x4C: "cmovl",  0x4D: "cmovge",
    0x4E: "cmovle", 0x4F: "cmovg",
}

_SHORT_JCC_LO, _SHORT_JCC_HI = 0x70, 0x7F
_LONG_JCC_PREFIX = 0x0F
_LONG_JCC_LO, _LONG_JCC_HI = 0x84, 0x8F
_CMOV_LO, _CMOV_HI = 0x40, 0x4F
_FLIP_MASK = 0x01


def _scan_flip_sites(
    binary: Path,
    func_vma: int,
    window: int,
) -> list[dict]:
    """
    Disassemble *window* bytes starting at *func_vma* and return all
    flip-candidate instructions.

    Uses ``objdump -d -M intel`` for accurate decode — avoids treating embedded
    bytes that happen to be in the 0x70-0x7F range as instructions.

    Returns a list of dicts (one per site) matching the :class:`FlipSite`
    keyword arguments.
    """
    result = subprocess.run(
        [
            "objdump",
            "-d",
            "-M",
            "intel",
            f"--start-address=0x{func_vma:x}",
            f"--stop-address=0x{func_vma + window:x}",
            str(binary),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.debug(f"objdump failed for VMA 0x{func_vma:x}: {result.stderr[:120]}")
        return []

    sites: list[dict] = []

    for line in result.stdout.splitlines():
        m = _OBJDUMP_LINE_RE.match(line)
        if not m:
            continue
        try:
            addr = int(m.group(1), 16)
            raw = [int(b, 16) for b in m.group(2).split()]
        except ValueError:
            continue

        if not raw:
            continue

        b0 = raw[0]

        # --- Short Jcc (rel8) ---
        if len(raw) == 2 and _SHORT_JCC_LO <= b0 <= _SHORT_JCC_HI:
            flipped = b0 ^ _FLIP_MASK
            sites.append(
                {
                    "site_vma": addr,  # patch the opcode byte itself
                    "instr_type": "short_jcc",
                    "opcode_before": _JCC_NAMES.get(b0, f"0x{b0:02x}"),
                    "opcode_after": _JCC_NAMES.get(flipped, f"0x{flipped:02x}"),
                    "flip_mask": _FLIP_MASK,
                    "offset_from_func": addr - func_vma,
                }
            )
            continue

        # --- Long Jcc (rel32) and CMOV ---
        if len(raw) >= 2 and b0 == _LONG_JCC_PREFIX:
            b1 = raw[1]

            if _LONG_JCC_LO <= b1 <= _LONG_JCC_HI and len(raw) == 6:
                flipped = b1 ^ _FLIP_MASK
                sites.append(
                    {
                        "site_vma": addr + 1,  # patch the second byte
                        "instr_type": "long_jcc",
                        "opcode_before": _JCC_NAMES.get(b1, f"0x{b1:02x}"),
                        "opcode_after": _JCC_NAMES.get(flipped, f"0x{flipped:02x}"),
                        "flip_mask": _FLIP_MASK,
                        "offset_from_func": addr + 1 - func_vma,
                    }
                )
                continue

            if _CMOV_LO <= b1 <= _CMOV_HI and len(raw) >= 3:
                flipped = b1 ^ _FLIP_MASK
                sites.append(
                    {
                        "site_vma": addr + 1,  # patch the second byte
                        "instr_type": "cmov",
                        "opcode_before": _CMOV_NAMES.get(b1, f"0x{b1:02x}"),
                        "opcode_after": _CMOV_NAMES.get(flipped, f"0x{flipped:02x}"),
                        "flip_mask": _FLIP_MASK,
                        "offset_from_func": addr + 1 - func_vma,
                    }
                )

    return sites


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlipSite:
    """One flippable instruction inside a target-independent LLVM function.

    Attributes:
        site_vma:        Linker VMA of the byte to patch (opcode or 2nd byte).
        instr_type:      ``"short_jcc"``, ``"long_jcc"``, or ``"cmov"``.
        opcode_before:   Human-readable mnemonic before the flip (e.g. ``"je"``).
        opcode_after:    Human-readable mnemonic after the flip (e.g. ``"jne"``).
        flip_mask:       Bitmask XOR'd onto the target byte (always ``0x01``).
        offset_from_func: ``site_vma - func_vma`` for display purposes.
    """

    site_vma: int
    instr_type: str
    opcode_before: str
    opcode_after: str
    flip_mask: int
    offset_from_func: int


@dataclass(frozen=True)
class FunctionSites:
    """A candidate function together with all its flip-candidate sites.

    Attributes:
        name:       Fully demangled C++ symbol name.
        vma:        Linker VMA of the function's first byte.
        tier:       Pipeline tier: 0 = cross-backend (TTIR/TTGIR/LLVM IR);
                    1 = NVPTX SelectionDAG visitors; 2 = NVPTX SelectionDAG
                    infrastructure; 3 = likely crash / target-lowering drivers.
        class_name: Namespace or class label (e.g. ``"DAGCombiner"`` or
                    ``"mlir::triton::gpu"``).
        sites:      All flip sites found within the function's disassembly window.
    """

    name: str
    vma: int
    tier: int
    class_name: str
    sites: tuple[FlipSite, ...]


@dataclass(frozen=True)
class ScanResult:
    """Full scan output for one binary.

    Attributes:
        binary:    Path to the scanned shared object.
        functions: All discovered target-independent functions, sorted by VMA.
    """

    binary: Path
    functions: tuple[FunctionSites, ...]


# ---------------------------------------------------------------------------
# Core entry point
# ---------------------------------------------------------------------------


def scan_binary(
    binary: Path,
    *,
    window: int = 0x2000,
    function_pattern: str | None = None,
    cache: BinaryCache | None = None,
) -> ScanResult:
    """Discover target-independent LLVM functions and collect flip sites.

    Args:
        binary:           Path to the shared object (e.g. ``libtriton.so``).
        window:           Byte window passed to ``objdump`` per function.
                          Larger values find more sites but take longer.
        function_pattern: Optional regex; only functions whose demangled name
                          matches are included.  ``None`` applies no extra filter.
        cache:            :class:`~karnage.utils.parser.BinaryCache` instance.
                          Shared with the caller to avoid re-running ``nm``.

    Returns:
        :class:`ScanResult` containing all discovered functions and their sites.

    Raises:
        ScannerError: If ``nm`` fails or the binary cannot be read.
    """
    _cache = cache or BinaryCache()

    logger.info(f"Scanning binary: {binary}")
    try:
        symbols = _cache.nm_symbols(binary)
    except ParserError as exc:
        raise ScannerError(
            f"nm failed on {binary}: {exc}",
            context={"binary": str(binary)},
        ) from exc

    pat = re.compile(function_pattern) if function_pattern else None

    # --- Filter symbols ---
    candidates: list[tuple[int, str]] = []  # (vma, demangled_name)
    for vma, _, sym_type, name in symbols:
        if sym_type not in ("T", "t"):
            continue
        if not _include_match(name):
            continue
        if any(tgt in name for tgt in _TARGET_PREFIXES):
            continue
        if pat is not None and not pat.search(name):
            continue
        candidates.append((vma, name))

    logger.info(f"Found {len(candidates):,} target-independent candidate functions")

    # --- Collect flip sites (progress bar) ---
    function_sites: list[FunctionSites] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Scanning functions", total=len(candidates))
        for func_vma, func_name in candidates:
            short = func_name.split("::")[-1][:42]
            progress.update(task, description=f"[cyan]{short}[/cyan]")
            raw_sites = _scan_flip_sites(binary, func_vma, window)
            flip_sites = tuple(FlipSite(**s) for s in raw_sites)
            function_sites.append(
                FunctionSites(
                    name=func_name,
                    vma=func_vma,
                    tier=_tier_of(func_name),
                    class_name=_class_of(func_name),
                    sites=flip_sites,
                )
            )
            progress.advance(task)

    function_sites.sort(key=lambda f: f.vma)
    total_sites = sum(len(f.sites) for f in function_sites)
    logger.info(
        f"Scan complete: {len(function_sites):,} functions, {total_sites:,} flip sites"
    )

    return ScanResult(binary=binary, functions=tuple(function_sites))


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def scan_result_to_dict(result: ScanResult) -> dict:
    """Serialise a :class:`ScanResult` to a JSON-safe dict.

    Args:
        result: The scan result to serialise.

    Returns:
        Dict with ``"meta"`` and ``"functions"`` top-level keys.
    """
    from collections import Counter

    type_counter: Counter = Counter()
    for fs in result.functions:
        for s in fs.sites:
            type_counter[s.instr_type] += 1

    functions_dict: dict[str, dict] = {}
    for fs in result.functions:
        functions_dict[fs.name] = {
            "vma": f"0x{fs.vma:016x}",
            "tier": fs.tier,
            "class": fs.class_name,
            "sites": [
                {
                    "site_vma": f"0x{s.site_vma:016x}",
                    "instr_type": s.instr_type,
                    "opcode_before": s.opcode_before,
                    "opcode_after": s.opcode_after,
                    "flip_mask": f"0x{s.flip_mask:02x}",
                    "offset_from_func": f"0x{s.offset_from_func:x}",
                }
                for s in fs.sites
            ],
        }

    return {
        "meta": {
            "binary": str(result.binary),
            "total_functions": len(result.functions),
            "total_sites": sum(len(f.sites) for f in result.functions),
            "site_counts": dict(type_counter),
        },
        "functions": functions_dict,
    }


def scan_result_from_dict(d: dict, binary: Path | None = None) -> ScanResult:
    """Deserialise a :class:`ScanResult` from a dict (loaded from JSON).

    Args:
        d:      Dict produced by :func:`scan_result_to_dict`.
        binary: Override the binary path; falls back to ``d["meta"]["binary"]``.

    Returns:
        Reconstructed :class:`ScanResult`.
    """
    resolved_binary = binary or Path(d["meta"]["binary"])
    functions: list[FunctionSites] = []
    for name, fd in d.get("functions", {}).items():
        sites = tuple(
            FlipSite(
                site_vma=int(s["site_vma"], 16),
                instr_type=s["instr_type"],
                opcode_before=s["opcode_before"],
                opcode_after=s["opcode_after"],
                flip_mask=int(s["flip_mask"], 16),
                offset_from_func=int(s["offset_from_func"], 16),
            )
            for s in fd.get("sites", [])
        )
        functions.append(
            FunctionSites(
                name=name,
                vma=int(fd["vma"], 16),
                tier=fd["tier"],
                class_name=fd["class"],
                sites=sites,
            )
        )
    functions.sort(key=lambda f: f.vma)
    return ScanResult(binary=resolved_binary, functions=tuple(functions))
