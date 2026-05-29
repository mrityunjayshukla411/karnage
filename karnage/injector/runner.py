"""GDB-based MatcherTable bit-flip test orchestration.

For each adjacent instruction pair produced by the extractor, this module:

1. Computes the ELF VMAs of every ``opc_lo`` / ``opc_hi`` byte occurrence
   for the source opcode by combining ``matcher_table.json`` offsets with
   the nm-derived MatcherTable symbol VMA.
2. Writes a ``patch_spec.json`` consumed by ``_gdb_script.py``.
3. Spawns GDB with ``_wrapper.py`` as the inferior.
4. Compares generated PTX and saved tensors against the baseline run.
5. Logs a one-line summary per flip and optionally writes a JSON report.

VMA arithmetic::

    mt_vma  = find_symbol_linker_vma(libtriton_so, matchertable_symbol)
    vma     = mt_vma + pattern.mt_offset + byte_index
              where byte_index = 1 (opc_lo) or 2 (opc_hi)
    runtime = load_base + vma   (load_base from /proc/{pid}/maps at run time)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from karnage.utils.constants import (
    ENV_ALWAYS_COMPILE,
    ENV_OUTPUT_DIR,
    ENV_PATCH_SPEC,
    ENV_TRITON_CACHE,
)
from karnage.utils.exceptions import MatcherTableLoadError
from karnage.utils.logger import logger
from karnage.utils.models import FlipResult, PatchSpec
from karnage.utils.parser import find_symbol_linker_vma
from karnage.utils.targets import NVPTXBackend

_THIS_DIR   = Path(__file__).parent
_GDB_SCRIPT = _THIS_DIR / "_gdb_script.py"
_WRAPPER    = _THIS_DIR / "_wrapper.py"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    """Load a JSON file, raising a domain error if it is missing.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content as a dict.

    Raises:
        MatcherTableLoadError: If *path* does not exist.
    """
    if not path.exists():
        raise MatcherTableLoadError(
            f"File not found: {path}",
            context={"path": str(path)},
        )
    with path.open() as f:
        return json.load(f)


def _build_patch_map(mt_data: dict) -> dict[int, list[int]]:
    """Build a ``{opcode: [mt_offset, ...]}`` lookup from matcher_table.json.

    Each ``mt_offset`` is the byte offset of the morph opcode byte from the
    start of the MatcherTable symbol.  ``opc_lo`` lives at ``mt_offset + 1``
    and ``opc_hi`` at ``mt_offset + 2``.

    Args:
        mt_data: Parsed ``matcher_table.json`` dict with an ``"instructions"``
                 top-level key.

    Returns:
        Dict mapping each integer opcode to the list of MatcherTable byte
        offsets at which that opcode's morph byte appears.
    """
    result: dict[int, list[int]] = {}
    for _, opc_list in mt_data["instructions"].items():
        for opc_obj in opc_list:
            opcode     = opc_obj["opcode"]
            mt_offsets = [
                int(pat["location"]["mt_offset"], 16)
                for pat in opc_obj["patterns"]
            ]
            result[opcode] = mt_offsets
    return result


def _iter_patch_specs(
    adj_data:  dict,
    patch_map: dict[int, list[int]],
    mt_vma:    int,
) -> Iterator[PatchSpec]:
    """Yield one :class:`~karnage.utils.models.PatchSpec` per adjacent pair.

    Skips opcodes that appear in ``adjacency.json`` but are absent from
    ``matcher_table.json`` (e.g. synthetic opcodes that have no MatcherTable
    pattern).

    Args:
        adj_data:  Parsed ``adjacency.json`` dict.
        patch_map: Opcode → list-of-mt-offsets mapping from
                   :func:`_build_patch_map`.
        mt_vma:    Linker VMA of the MatcherTable symbol in ``libtriton.so``.

    Yields:
        :class:`~karnage.utils.models.PatchSpec` objects in adjacency order.
    """
    flip_id = 0
    for opcode_str, instr in adj_data["instructions"].items():
        opcode_a   = int(opcode_str)
        mt_offsets = patch_map.get(opcode_a)
        if not mt_offsets:
            logger.warning(
                f"Opcode {opcode_a} ({instr['mnemonic']!r}) "
                f"not in matcher_table.json --- skipping"
            )
            continue

        for neighbor in instr["adjacent"]:
            flip       = neighbor["flip"]
            byte_index = 1 if flip["byte"] == "opc_lo" else 2
            mask       = int(flip["mask"], 16)
            patch_vmas = tuple(mt_vma + off + byte_index for off in mt_offsets)

            yield PatchSpec(
                flip_id    = flip_id,
                opcode_a   = opcode_a,
                mnemonic_a = instr["mnemonic"],
                opcode_b   = neighbor["opcode"],
                mnemonic_b = neighbor["mnemonic"],
                flip_byte  = flip["byte"],
                flip_bit   = flip["bit"],
                flip_mask  = mask,
                patch_vmas = patch_vmas,
            )
            flip_id += 1


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _run_inferior(
    triton_script:   Path,
    output_dir:      Path,
    *,
    patch_spec_path: Path | None = None,
) -> bool:
    """Run the Triton script, optionally under GDB with a patch applied.

    **Baseline mode** (``patch_spec_path=None``): runs ``_wrapper.py``
    directly under the system Python interpreter.

    **Flip mode**: runs ``gdb --batch`` with ``_gdb_script.py`` as the GDB
    command script; ``KARNAGE_PATCH_SPEC`` points GDB to the patch JSON.

    Output files written to *output_dir*:
    - ``stdout.txt`` / ``stderr.txt`` --- captured process output.
    - ``returncode.txt`` --- integer exit code as a string.

    Args:
        triton_script:   Path to the user Triton script.
        output_dir:      Directory to write output files into (created if absent).
        patch_spec_path: Path to ``patch_spec.json``; ``None`` for baseline runs.

    Returns:
        ``True`` if the process exited with code 0, ``False`` otherwise.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ}
    env[ENV_OUTPUT_DIR]     = str(output_dir / "tensors")
    env[ENV_TRITON_CACHE]   = str(output_dir / "triton_cache")
    env[ENV_ALWAYS_COMPILE] = "1"

    if patch_spec_path is None:
        cmd = [sys.executable, str(_WRAPPER), str(triton_script)]
    else:
        env[ENV_PATCH_SPEC] = str(patch_spec_path)
        cmd = [
            "gdb", "--batch", "-q",
            "-x", str(_GDB_SCRIPT),
            "--args", sys.executable, str(_WRAPPER), str(triton_script),
        ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    except FileNotFoundError as exc:
        (output_dir / "stderr.txt").write_text(f"Command not found: {exc}\n")
        (output_dir / "returncode.txt").write_text("-1")
        return False

    (output_dir / "stdout.txt").write_text(proc.stdout)
    (output_dir / "stderr.txt").write_text(proc.stderr)
    (output_dir / "returncode.txt").write_text(str(proc.returncode))
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# PTX collection
# ---------------------------------------------------------------------------

def _collect_ptx(output_dir: Path) -> list[str]:
    """Return the contents of all ``.ptx`` files under ``triton_cache/``.

    Files are sorted by path so comparisons between baseline and flip runs
    are deterministic.  With ``TRITON_ALWAYS_COMPILE=1``, fresh PTX is
    written on every run, so stale cache entries do not interfere.

    Args:
        output_dir: Run output directory containing ``triton_cache/``.

    Returns:
        List of PTX file contents as strings; empty list if no cache exists.
    """
    cache_dir = output_dir / "triton_cache"
    if not cache_dir.exists():
        return []
    return [f.read_text(errors="replace") for f in sorted(cache_dir.rglob("*.ptx"))]


# ---------------------------------------------------------------------------
# Spec filtering
# ---------------------------------------------------------------------------

# Captures the instruction mnemonic from a PTX line, e.g.:
#   "  fma.rn.f32 %f1, %f2, %f3, %f4;" → "fma.rn.f32"
#   "  @%p0 bra $L__BB0_2;"             → "bra"
_PTX_INSTR_RE = re.compile(
    r"^\s*"
    r"(?:@[!%\w]+\s+)?"                           # optional predicate
    r"([a-z][a-z0-9]*(?:\.[a-z][a-z0-9]*)*)\b"   # mnemonic (dot-separated tokens)
)
_PTX_SKIP_CHARS = frozenset({".", "/", "$", "{", "}", "@"})


def extract_ptx_mnemonics(baseline_dir: Path) -> frozenset[str]:
    """Parse all baseline PTX files and return the set of instruction mnemonics.

    Used by the relevance filter to skip opcodes that the kernel never emits
    (e.g. ``abs.bf16`` in a matrix-multiply kernel).

    Args:
        baseline_dir: The baseline run output directory produced by
                      :func:`_run_inferior` with no patch applied.

    Returns:
        Frozen set of mnemonic strings (e.g. ``{"fma.rn.f32", "ld.global.u32"}``).
    """
    mnemonics: set[str] = set()
    for ptx_text in _collect_ptx(baseline_dir):
        for line in ptx_text.splitlines():
            stripped = line.strip()
            if not stripped or stripped[0] in _PTX_SKIP_CHARS or stripped.endswith(":"):
                continue
            m = _PTX_INSTR_RE.match(line)
            if m:
                mnemonics.add(m.group(1))
    return frozenset(mnemonics)


def _apply_filters(
    specs:            list[PatchSpec],
    *,
    ptx_mnemonics:    frozenset[str] | None,
    target_mnemonics: frozenset[str] | None,
) -> list[PatchSpec]:
    """Return a filtered copy of *specs* with irrelevant flips removed.

    Two independent filters can be applied, and both are used when provided:

    - **Relevance filter** (``ptx_mnemonics``): keeps only specs whose
      ``mnemonic_a`` appears in the baseline PTX.  Skips instructions the
      kernel never emits.
    - **Targeted filter** (``target_mnemonics``): keeps only specs in an
      explicitly requested mnemonic set.

    Args:
        specs:            Full list of :class:`~karnage.utils.models.PatchSpec`.
        ptx_mnemonics:    Mnemonics seen in the baseline PTX; ``None`` disables
                          the relevance filter.
        target_mnemonics: Explicit mnemonic allowlist; ``None`` disables the
                          targeted filter.

    Returns:
        Filtered list of :class:`~karnage.utils.models.PatchSpec` objects.
    """
    out = specs
    if target_mnemonics is not None:
        before = len(out)
        out = [s for s in out if s.mnemonic_a in target_mnemonics]
        logger.info(
            f"Targeted filter: {before:,} → {len(out):,} specs "
            f"({before - len(out):,} dropped, "
            f"kept mnemonics: {', '.join(sorted(target_mnemonics))})"
        )
    if ptx_mnemonics is not None:
        before = len(out)
        out = [s for s in out if s.mnemonic_a in ptx_mnemonics]
        logger.info(
            f"Relevance filter: {before:,} → {len(out):,} specs "
            f"({before - len(out):,} dropped, "
            f"{len(ptx_mnemonics):,} mnemonics in baseline PTX)"
        )
    return out


# ---------------------------------------------------------------------------
# Result comparison
# ---------------------------------------------------------------------------

def _compare(
    spec:         PatchSpec,
    baseline_dir: Path,
    flip_dir:     Path,
) -> FlipResult:
    """Compare a flip run against the baseline and return a :class:`~karnage.utils.models.FlipResult`.

    Crash detection is based on two sentinels written by ``_wrapper.py``
    rather than on the GDB return code alone (GDB ``--batch`` exits 0 even
    when the inferior crashes).

    PTX comparison is skipped when either run produced no PTX, to avoid
    reporting a ``PTX_DIFF`` when the flip simply prevented compilation.

    Tensor comparison uses :func:`torch.allclose` with ``atol=rtol=1e-5``.

    Args:
        spec:         The :class:`~karnage.utils.models.PatchSpec` that was applied.
        baseline_dir: Root output directory of the clean baseline run.
        flip_dir:     Root output directory of this flip run.

    Returns:
        Populated :class:`~karnage.utils.models.FlipResult`.
    """
    import torch

    rc_file        = flip_dir / "returncode.txt"
    error_sentinel = flip_dir / "tensors" / "_error.txt"
    done_sentinel  = flip_dir / "tensors" / "_done"
    gdb_failed     = (
        not rc_file.exists()
        or int(rc_file.read_text().strip() or "1") != 0
    )
    script_ran = done_sentinel.exists() and not error_sentinel.exists()
    crashed    = gdb_failed or not script_ran

    baseline_ptx = _collect_ptx(baseline_dir)
    flip_ptx     = _collect_ptx(flip_dir)
    ptx_changed  = bool(baseline_ptx) and bool(flip_ptx) and baseline_ptx != flip_ptx

    tensors_match: dict[str, bool]  = {}
    max_abs_diffs: dict[str, float] = {}
    tensor_names:  list[str]        = []

    if not crashed:
        b_dir = baseline_dir / "tensors"
        f_dir = flip_dir / "tensors"
        if b_dir.exists() and f_dir.exists():
            for pt_file in sorted(b_dir.glob("*.pt")):
                name      = pt_file.stem
                flip_file = f_dir / pt_file.name
                if not flip_file.exists():
                    continue
                tensor_names.append(name)
                try:
                    a = torch.load(pt_file,  map_location="cpu")
                    b = torch.load(flip_file, map_location="cpu")
                    a, b = a.float(), b.float()
                    tensors_match[name] = bool(torch.allclose(a, b, atol=1e-5, rtol=1e-5))
                    max_abs_diffs[name] = float((a - b).abs().max())
                except Exception as exc:
                    # torch.load can raise many unrelated exception types
                    # (pickle errors, version mismatches, CUDA errors) that
                    # have no single catchable base class --- broad catch is
                    # intentional here.
                    logger.warning(f"  tensor compare failed for {name!r}: {exc}")

    return FlipResult(
        spec          = spec,
        crashed       = crashed,
        script_ran    = script_ran,
        ptx_changed   = ptx_changed,
        tensor_names  = tensor_names,
        tensors_match = tensors_match,
        max_abs_diffs = max_abs_diffs,
    )


# ---------------------------------------------------------------------------
# Report serialisation
# ---------------------------------------------------------------------------

def _serialise_result(r: FlipResult) -> dict:
    """Serialise a :class:`~karnage.utils.models.FlipResult` to a JSON-safe dict.

    Args:
        r: Flip result to serialise.

    Returns:
        Dict suitable for ``json.dump``.
    """
    return {
        "flip_id":       r.spec.flip_id,
        "opcode_a":      r.spec.opcode_a,
        "mnemonic_a":    r.spec.mnemonic_a,
        "opcode_b":      r.spec.opcode_b,
        "mnemonic_b":    r.spec.mnemonic_b,
        "flip_byte":     r.spec.flip_byte,
        "flip_bit":      r.spec.flip_bit,
        "crashed":       r.crashed,
        "script_ran":    r.script_ran,
        "ptx_changed":   r.ptx_changed,
        "tensor_names":  r.tensor_names,
        "tensors_match": r.tensors_match,
        "max_abs_diffs": r.max_abs_diffs,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_flipper(
    triton_script:      Path,
    matcher_table_json: Path,
    adjacency_json:     Path,
    libtriton_so:       Path,
    output_dir:         Path,
    *,
    max_flips:          int | None = None,
    report_json:        Path | None = None,
    cooldown_every:     int = 0,
    cooldown_secs:      float = 30.0,
    filter_by_ptx:      bool = False,
    target_mnemonics:   frozenset[str] | None = None,
) -> list[FlipResult]:
    """Run the full bit-flip test suite and return all results.

    Steps:

    1. Resolve the MatcherTable symbol VMA via ``nm``.
    2. Build an ``opcode → [mt_offset, ...]`` map from ``matcher_table.json``.
    3. Run a clean baseline (no GDB, no patch) to capture reference PTX and
       tensors.
    4. Apply relevance and/or targeted mnemonic filters to the spec list.
    5. For each remaining :class:`~karnage.utils.models.PatchSpec`: write
       ``patch_spec.json``, spawn GDB, compare results against the baseline.
    6. Optionally write a machine-readable JSON report.

    Args:
        triton_script:      Path to the Triton application script.
        matcher_table_json: Path to ``matcher_table.json`` from the extract step.
        adjacency_json:     Path to ``adjacency.json`` from the inject step.
        libtriton_so:       Path to ``libtriton.so``.
        output_dir:         Root output directory; per-flip results go into
                            ``flip_NNNNNN/`` subdirectories.
        max_flips:          Stop after this many flips; ``None`` runs all.
        report_json:        If given, write a JSON array of serialised results
                            to this path at the end.
        cooldown_every:     Pause after every *N* flips (0 = disabled).
        cooldown_secs:      Duration of each cooldown pause in seconds.
        filter_by_ptx:      When ``True``, only test opcodes whose mnemonic
                            appears in the baseline PTX.
        target_mnemonics:   Explicit set of mnemonics to test; ``None`` means
                            no targeted filter.

    Returns:
        List of :class:`~karnage.utils.models.FlipResult` objects, one per
        flip that was actually executed.
    """
    output_dir = output_dir.resolve()

    mt_data  = _load_json(matcher_table_json)
    adj_data = _load_json(adjacency_json)

    target = NVPTXBackend()
    mt_vma = find_symbol_linker_vma(libtriton_so, target.matchertable_symbol)
    logger.info(f"MatcherTable VMA: 0x{mt_vma:016x}")

    patch_map = _build_patch_map(mt_data)
    specs     = list(_iter_patch_specs(adj_data, patch_map, mt_vma))
    logger.info(f"Total specs before filtering: {len(specs):,}")

    # --- Baseline ---
    baseline_dir = output_dir / "baseline"
    logger.info("Running baseline (no patch)...")
    if not _run_inferior(triton_script, baseline_dir):
        logger.warning("Baseline run failed --- see baseline/stderr.txt")

    # --- Filter specs ---
    ptx_mnemonics: frozenset[str] | None = None
    if filter_by_ptx:
        ptx_mnemonics = extract_ptx_mnemonics(baseline_dir)
        if not ptx_mnemonics:
            logger.warning(
                "Relevance filter requested but no PTX files found in baseline --- "
                "filter has no effect. Check that TRITON_CACHE_DIR is being written."
            )

    specs = _apply_filters(
        specs,
        ptx_mnemonics    = ptx_mnemonics,
        target_mnemonics = target_mnemonics,
    )

    if max_flips is not None:
        specs = specs[:max_flips]
    logger.info(f"Flips to run after filtering: {len(specs):,}")

    # --- Flip runs ---
    results: list[FlipResult] = []
    for spec in specs:
        logger.info(
            f"[{spec.flip_id:>6}] {spec.mnemonic_a} (opc {spec.opcode_a})"
            f"  --{spec.flip_byte}[{spec.flip_bit}]-->  "
            f"{spec.mnemonic_b} (opc {spec.opcode_b})"
            f"  [{len(spec.patch_vmas)} site(s)]"
        )

        flip_dir = output_dir / f"flip_{spec.flip_id:06d}"
        flip_dir.mkdir(parents=True, exist_ok=True)

        patch_spec_file = flip_dir / "patch_spec.json"
        patch_spec_file.write_text(json.dumps({
            "patch_vmas": list(spec.patch_vmas),
            "mask":       spec.flip_mask,
        }, indent=2))

        _run_inferior(triton_script, flip_dir, patch_spec_path=patch_spec_file)

        result = _compare(spec, baseline_dir, flip_dir)
        results.append(result)

        flags = []
        if result.crashed:
            flags.append("CRASH" if not result.script_ran else "SCRIPT_FAILED")
        if result.ptx_changed:
            flags.append("PTX_DIFF")
        tensor_diffs = [n for n, ok in result.tensors_match.items() if not ok]
        if tensor_diffs:
            diffs_str = ", ".join(
                f"{n}={result.max_abs_diffs[n]:.3e}" for n in tensor_diffs
            )
            flags.append(f"TENSOR_DIFF({diffs_str})")
        logger.info(f"         -> {', '.join(flags) if flags else 'no change'}")

        flips_done = len(results)
        if cooldown_every > 0 and flips_done % cooldown_every == 0:
            logger.info(
                f"[cooldown] {flips_done} flips done --- "
                f"sleeping {cooldown_secs:.0f}s to let GPU cool..."
            )
            time.sleep(cooldown_secs)

    if report_json:
        with report_json.open("w") as f:
            json.dump([_serialise_result(r) for r in results], f, indent=2)
        logger.success(f"Report written → {report_json}")

    return results
