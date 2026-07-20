"""GDB-based bit-flip test orchestration for target-independent LLVM code.

For each :class:`~karnage.utils.models.PatchSpec` derived from a
``flip_sites.json`` scan result this module:

1. Writes ``patch_spec.json`` and ``spec.json`` into the flip directory.
2. Spawns ``gdb --batch`` with ``_wrapper.py`` as the inferior.
3. Diffs the application's stdout against the baseline.
4. Optionally diffs generated PTX against the baseline.
5. Logs a one-line summary per flip and optionally writes a JSON report.

Stdout / stderr separation::

    gdb.write(..., gdb.STDERR)  →  proc.stderr  →  gdb_stderr.txt
    inferior print() / sys.stdout  →  proc.stdout  →  app_stdout.txt

    The runner diffs app_stdout.txt between baseline and each flip run.
    No tensor files are written; storage per flip is a few small text files.

VMA arithmetic::

    runtime_addr = load_base + site_vma
    (load_base resolved from /proc/{pid}/maps at run time by _gdb_script.py)
"""

from __future__ import annotations

import functools
import json
import os
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from karnage.utils.constants import (
    ENV_ALWAYS_COMPILE,
    ENV_COMPILE_ONLY,
    ENV_OUTPUT_DIR,
    ENV_PATCH_SPEC,
    ENV_SIGNATURE_IN,
    ENV_SIGNATURE_OUT,
    ENV_TRITON_CACHE,
)
from karnage.utils.exceptions import FlipperError, ScannerError
from karnage.utils.logger import console, logger
from karnage.utils.models import FlipResult, PatchSpec

_THIS_DIR = Path(__file__).parent
_GDB_SCRIPT = _THIS_DIR / "_gdb_script.py"
_WRAPPER = _THIS_DIR / "_wrapper.py"
_WRAPPER_COMPILE_REPLAY = _THIS_DIR / "_wrapper_compile_replay.py"


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise ScannerError(
            f"File not found: {path}",
            context={"path": str(path)},
        )
    with path.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# PatchSpec iterator
# ---------------------------------------------------------------------------


def iter_patch_specs(
    sites_data: dict,
    *,
    tier_filter: int | None = None,
    type_filter: str | None = None,
    function_pattern: str | None = None,
    function_names: frozenset[str] | None = None,
    flip_ids: frozenset[int] | None = None,
) -> Iterator[PatchSpec]:
    """Yield one :class:`~karnage.utils.models.PatchSpec` per flip site.

    Args:
        sites_data:       Parsed ``flip_sites.json`` dict.
        tier_filter:      Only yield specs from functions of this tier.
        type_filter:      Only yield sites of this instruction type.
        function_pattern: Regex matched against the function's demangled name.
        function_names:   Exact demangled names to include (e.g. loaded from a
                          ``--function-list`` file).  When provided, only
                          functions whose full name is in this set are yielded.
                          Composed with *function_pattern* if both are given.
        flip_ids:         Exact ``flip_id`` values to include (e.g. loaded from
                          a ``--flip-ids-file``, typically the pruned candidate
                          list from a ``--compile-only`` prescreen run). Unlike
                          the other filters, this one doesn't change which
                          sites consume a ``flip_id`` --- it only restricts
                          which already-numbered sites get yielded --- so IDs
                          from a prior call line up correctly here as long as
                          *tier_filter* / *type_filter* / *function_pattern* /
                          *function_names* are identical between the two
                          calls (those still shift the numbering, same as
                          before this filter existed).

    Yields:
        :class:`~karnage.utils.models.PatchSpec` in function-name / site order.
    """
    pat = re.compile(function_pattern) if function_pattern else None
    flip_id = 0

    for func_name, fd in sites_data.get("functions", {}).items():
        tier = fd.get("tier", 3)
        if tier_filter is not None and tier != tier_filter:
            continue
        if function_names is not None and func_name not in function_names:
            continue
        if pat is not None and not pat.search(func_name):
            continue

        for site in fd.get("sites", []):
            if type_filter is not None and site["instr_type"] != type_filter:
                continue
            this_flip_id = flip_id
            flip_id += 1
            if flip_ids is not None and this_flip_id not in flip_ids:
                continue
            yield PatchSpec(
                flip_id=this_flip_id,
                func_name=func_name,
                site_vma=int(site["site_vma"], 16),
                instr_type=site["instr_type"],
                opcode_before=site["opcode_before"],
                opcode_after=site["opcode_after"],
                flip_mask=int(site["flip_mask"], 16),
            )


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _run_inferior(
    triton_script: Path,
    output_dir: Path,
    *,
    patch_spec_path: Path | None = None,
    timeout: float | None = None,
    compile_only: bool = False,
    signature_out: Path | None = None,
    signature_in: Path | None = None,
    compile_replay: bool = False,
) -> bool:
    """Run the Triton script, optionally under GDB with a patch applied.

    Files written to *output_dir*:

    - ``app_stdout.txt`` --- written by ``_wrapper.py`` via redirected
      ``sys.stdout``; contains only the user script's ``print()`` output.
      Not written at all in ``compile_replay`` mode (there is no app).
    - ``gdb_stdout.txt`` --- raw GDB process stdout (thread events, vfork
      messages, debuginfod prompts); kept for debugging, not used for diffing.
    - ``gdb_stderr.txt`` --- GDB Python diagnostics (``gdb.write(..., gdb.STDERR)``).
    - ``returncode.txt`` --- integer exit code as a string.

    Sentinels ``_done`` and ``_error.txt`` are written by ``_wrapper.py`` (or
    ``_wrapper_compile_replay.py`` in ``compile_replay`` mode) inside
    *output_dir* (via ``KARNAGE_OUTPUT_DIR``).

    Args:
        triton_script:   Path to the user Triton script --- or, when
                         *compile_replay* is set, path to a capture blob
                         written by :func:`karnage.compile_capture.capture_all_compiles`
                         (the parameter is reused rather than duplicated,
                         since both cases mean "the thing that gets run per
                         script-equivalent unit").
        output_dir:      Directory to write output files into.
        patch_spec_path: Path to ``patch_spec.json``; ``None`` for baseline.
        timeout:         SIGTERM the GDB process after this many seconds.
        compile_only:    Set ``KARNAGE_COMPILE_ONLY=1`` so ``_wrapper.py``
                         patches Triton to compile every kernel without
                         launching it on the GPU --- see ``_wrapper.py``'s
                         module docstring for the full mechanism.
        signature_out:   Write a kernel-call signature manifest here (the
                         baseline run only) --- mutually exclusive with
                         *signature_in*. See ``_wrapper.py``'s "Signature
                         capture / replay" docs.
        signature_in:    Replay recorded kernel calls from this manifest
                         instead of running *triton_script* at all ---
                         mutually exclusive with *signature_out*.
        compile_replay:  Use ``_wrapper_compile_replay.py`` instead of
                         ``_wrapper.py`` --- replays every specialization in
                         the *triton_script* capture blob directly through
                         ``triton.compile()``, no torch/CUDA access at all.
                         Mutually exclusive with *compile_only* /
                         *signature_out* / *signature_in* (a fundamentally
                         different execution path, validated separately in
                         ``karnage/tests/test_compile_equivalence.py``).

    Returns:
        ``True`` if the process exited with code 0.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper = _WRAPPER_COMPILE_REPLAY if compile_replay else _WRAPPER

    env = {**os.environ}
    env[ENV_OUTPUT_DIR] = str(output_dir)          # sentinels land here directly
    env[ENV_TRITON_CACHE] = str(output_dir / "triton_cache")
    env[ENV_ALWAYS_COMPILE] = "1"
    if compile_only:
        env[ENV_COMPILE_ONLY] = "1"
    if signature_out is not None:
        env[ENV_SIGNATURE_OUT] = str(signature_out)
    if signature_in is not None:
        env[ENV_SIGNATURE_IN] = str(signature_in)

    if patch_spec_path is None:
        cmd = [sys.executable, str(wrapper), str(triton_script)]
    else:
        env[ENV_PATCH_SPEC] = str(patch_spec_path)
        cmd = [
            "gdb",
            "--batch",
            "-q",
            # Suppress the debuginfod auto-download prompt before our script loads.
            "-iex", "set debuginfod enabled off",
            "-x", str(_GDB_SCRIPT),
            "--args", sys.executable, str(wrapper), str(triton_script),
        ]

    try:
        # start_new_session=True puts the child (gdb, or the bare wrapper for
        # a baseline run) in its own process group. On an ordinary exit this
        # changes nothing, but it lets the timeout path below reap the whole
        # tree instead of leaking it --- see this function's docstring update:
        # subprocess.run(timeout=...) only SIGKILLs the direct child, so a
        # ptrace'd GDB inferior (grandchild, survives its tracer dying) or a
        # ptxas subprocess spawned by Triton (grandchild of a killed baseline
        # wrapper) would otherwise be orphaned and keep running --- holding a
        # GPU context or a Triton cache lock file that makes a *later*, fresh
        # run hang for reasons that have nothing to do with that run itself.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass  # exited between the timeout firing and the kill
            stdout, stderr = proc.communicate()
            rc = -1
            (output_dir / "_timeout").touch()
        else:
            rc = proc.returncode
    except FileNotFoundError as exc:
        (output_dir / "gdb_stderr.txt").write_text(f"Command not found: {exc}\n")
        (output_dir / "returncode.txt").write_text("-1")
        return False

    # app_stdout.txt is written by _wrapper.py via redirected sys.stdout.
    # proc.stdout contains only GDB's own noise; save it for debugging.
    (output_dir / "gdb_stdout.txt").write_text(stdout)
    (output_dir / "gdb_stderr.txt").write_text(stderr)
    (output_dir / "returncode.txt").write_text(str(rc))
    return rc == 0


# ---------------------------------------------------------------------------
# Codegen output collection — backend-agnostic
# ---------------------------------------------------------------------------

# Text-based codegen artefacts produced by the Triton cache.
# NVIDIA: .ptx                AMD: .amdgcn
# Both:   .llir (LLVM IR), .ttgir (TritonGPU IR), .ttir (Triton IR)
# Excluded: .hsaco / .cubin (binary), .json (metadata)
_CODEGEN_SUFFIXES: tuple[str, ...] = (
    "*.ptx",    # NVIDIA PTX assembly
    "*.amdgcn", # AMD GCN assembly
    "*.llir",   # LLVM IR  (common to both)
    "*.ttgir",  # TritonGPU IR (common to both)
    "*.ttir",   # Triton IR    (common to both)
)


def _collect_codegen(output_dir: Path) -> list[str]:
    """Collect all text-based codegen artefacts from *output_dir*/triton_cache.

    Returns one string per file, sorted by path, so lists from different runs
    can be compared element-wise only when the same set of files was produced.
    Uses a content-keyed dict to handle filename differences between runs.
    """
    cache_dir = output_dir / "triton_cache"
    if not cache_dir.exists():
        return []
    files: list[Path] = []
    for pattern in _CODEGEN_SUFFIXES:
        files.extend(cache_dir.rglob(pattern))
    return [f.read_text(errors="replace") for f in sorted(files)]


# ---------------------------------------------------------------------------
# PTX mnemonic extraction (for --filter-by-ptx, NVIDIA only)
# ---------------------------------------------------------------------------

_PTX_INSTR_RE = re.compile(
    r"^\s*"
    r"(?:@[!%\w]+\s+)?"
    r"([a-z][a-z0-9]*(?:\.[a-z][a-z0-9]*)*)\b"
)
_PTX_SKIP_CHARS = frozenset({".", "/", "$", "{", "}", "@"})


def extract_ptx_mnemonics(baseline_dir: Path) -> frozenset[str]:
    """Extract PTX instruction mnemonics from the baseline cache (NVIDIA only)."""
    cache_dir = baseline_dir / "triton_cache"
    if not cache_dir.exists():
        return frozenset()
    mnemonics: set[str] = set()
    for ptx_file in sorted(cache_dir.rglob("*.ptx")):
        for line in ptx_file.read_text(errors="replace").splitlines():
            stripped = line.strip()
            if not stripped or stripped[0] in _PTX_SKIP_CHARS or stripped.endswith(":"):
                continue
            m = _PTX_INSTR_RE.match(line)
            if m:
                mnemonics.add(m.group(1))
    return frozenset(mnemonics)


# ---------------------------------------------------------------------------
# Result comparison
# ---------------------------------------------------------------------------


def _compare(
    spec: PatchSpec,
    baseline_dir: Path,
    flip_dir: Path,
    *,
    compile_only: bool = False,
) -> FlipResult:
    """Compare a flip run against the baseline using stdout and codegen diffs.

    ``codegen_changed`` covers all text-based Triton cache artefacts: PTX
    (NVIDIA), AMDGCN (AMD), and the shared LLVM IR / TTGIR / TTIR files.

    Crash detection uses the ``_done`` / ``_error.txt`` sentinels written by
    ``_wrapper.py`` and the saved ``returncode.txt``, not the GDB exit code
    alone (GDB ``--batch`` exits 0 even when the inferior crashes).

    Args:
        spec:         The :class:`~karnage.utils.models.PatchSpec` applied.
        baseline_dir: Output directory of the clean baseline run.
        flip_dir:     Output directory of this flip run.
        compile_only: Skip the stdout diff --- in compile-only mode the
                     kernel never launches, so its output (and anything the
                     script prints from it) is never meaningfully computed;
                     ``stdout_changed`` is always ``False`` for these runs.

    Returns:
        Populated :class:`~karnage.utils.models.FlipResult`.
    """
    rc_file = flip_dir / "returncode.txt"
    done_sentinel = flip_dir / "_done"
    error_sentinel = flip_dir / "_error.txt"
    timeout_sentinel = flip_dir / "_timeout"

    timed_out = timeout_sentinel.exists()
    gdb_failed = not rc_file.exists() or int(rc_file.read_text().strip() or "1") != 0
    script_ran = done_sentinel.exists() and not error_sentinel.exists()
    crashed = gdb_failed or not script_ran

    # Codegen diff — backend-agnostic: PTX, AMDGCN, LLIR, TTGIR, TTIR
    baseline_codegen = _collect_codegen(baseline_dir)
    flip_codegen = _collect_codegen(flip_dir)
    codegen_changed = (
        bool(baseline_codegen) and bool(flip_codegen)
        and baseline_codegen != flip_codegen
    )

    # Stdout diff — only meaningful when the script actually ran and its
    # kernel actually launched (never true in compile-only mode)
    stdout_changed = False
    if not compile_only and not crashed and not timed_out:
        b_out = baseline_dir / "app_stdout.txt"
        f_out = flip_dir / "app_stdout.txt"
        if b_out.exists() and f_out.exists():
            stdout_changed = b_out.read_text(errors="replace") != f_out.read_text(
                errors="replace"
            )

    return FlipResult(
        spec=spec,
        crashed=crashed,
        timed_out=timed_out,
        script_ran=script_ran,
        codegen_changed=codegen_changed,
        stdout_changed=stdout_changed,
    )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _serialise_result(r: FlipResult) -> dict:
    return {
        "flip_id": r.spec.flip_id,
        "func_name": r.spec.func_name,
        "site_vma": f"0x{r.spec.site_vma:016x}",
        "instr_type": r.spec.instr_type,
        "opcode_before": r.spec.opcode_before,
        "opcode_after": r.spec.opcode_after,
        "crashed": r.crashed,
        "timed_out": r.timed_out,
        "script_ran": r.script_ran,
        "codegen_changed": r.codegen_changed,
        "stdout_changed": r.stdout_changed,
    }


def _spec_to_dict(spec: PatchSpec) -> dict:
    return {
        "flip_id": spec.flip_id,
        "func_name": spec.func_name,
        "site_vma": f"0x{spec.site_vma:016x}",
        "instr_type": spec.instr_type,
        "opcode_before": spec.opcode_before,
        "opcode_after": spec.opcode_after,
        "flip_mask": f"0x{spec.flip_mask:02x}",
    }


def _spec_from_dict(d: dict) -> PatchSpec:
    return PatchSpec(
        flip_id=d["flip_id"],
        func_name=d["func_name"],
        site_vma=int(d["site_vma"], 16),
        instr_type=d["instr_type"],
        opcode_before=d["opcode_before"],
        opcode_after=d["opcode_after"],
        flip_mask=int(d["flip_mask"], 16),
    )


# ---------------------------------------------------------------------------
# Post-hoc report reconstruction
# ---------------------------------------------------------------------------


def reconstruct_report(output_dir: Path) -> dict[str, list[dict]]:
    """Reconstruct per-script JSON reports from an existing flip output directory.

    Scans all ``flip_NNNNNN/`` subdirectories, reads ``spec.json`` from each,
    and re-runs the comparison logic against the baseline directory.  Requires
    that the run was performed with a version of karnage that writes
    ``spec.json`` (i.e. this version or later).

    Script names are auto-detected from subdirectories of ``baseline/`` that
    contain an ``app_stdout.txt`` file.  Legacy single-script output dirs
    (``app_stdout.txt`` directly inside ``baseline/``) are returned under the
    key ``"baseline"``.

    Args:
        output_dir: Root output directory passed to ``karnage flip --output``.

    Returns:
        Dict mapping each script stem to its list of serialised result dicts.
    """
    baseline_dir = output_dir / "baseline"
    if not baseline_dir.exists():
        raise FileNotFoundError(f"baseline/ not found in {output_dir}")

    script_names: list[str] = sorted(
        d.name
        for d in baseline_dir.iterdir()
        if d.is_dir() and (d / "app_stdout.txt").exists()
    )
    legacy = not script_names and (baseline_dir / "app_stdout.txt").exists()

    flip_dirs = sorted(output_dir.glob("flip_*/"))
    results: dict[str, list[dict]] = (
        {"baseline": []} if legacy else {name: [] for name in script_names}
    )

    for flip_dir in flip_dirs:
        spec_file = flip_dir / "spec.json"
        if not spec_file.exists():
            logger.warning(f"No spec.json in {flip_dir.name} — skipping")
            continue
        spec = _spec_from_dict(json.loads(spec_file.read_text()))
        if legacy:
            results["baseline"].append(
                _serialise_result(_compare(spec, baseline_dir, flip_dir))
            )
        else:
            for name in script_names:
                results[name].append(
                    _serialise_result(
                        _compare(spec, baseline_dir / name, flip_dir / name)
                    )
                )

    return results


# ---------------------------------------------------------------------------
# Per-flip worker (thread-safe)
# ---------------------------------------------------------------------------


def _flip_one(
    spec: PatchSpec,
    *,
    triton_scripts: list[Path],
    output_dir: Path,
    flip_timeout: float | None,
    baseline_dir: Path,
    compile_only: bool = False,
    replay_signatures: bool = False,
    compile_replay: bool = False,
) -> dict[str, FlipResult]:
    """Run one bit-flip experiment across all scripts and return per-script results.

    Safe to call from multiple threads simultaneously; every flip gets its own
    subdirectory so there is no shared mutable state between concurrent calls.
    Each script runs in its own ``flip_NNNNNN/{script.stem}/`` subdir so
    Triton caches and sentinel files never collide.

    Args:
        compile_only:      Forwarded to :func:`_run_inferior` (skip the GPU
                           launch) and :func:`_compare` (skip the now-
                           meaningless stdout diff).
        replay_signatures: Replay from ``<baseline_dir>/<script.stem>/
                           signatures.json`` (written by the baseline run;
                           see :func:`run_flipper`) instead of running
                           *script* at all --- no torch/CUDA touch beyond
                           what ``triton.compile()`` itself needs.
        compile_replay:    Forwarded to :func:`_run_inferior` --- *script* is
                           actually a capture blob path in this mode (see
                           :func:`run_flipper`), replayed via
                           ``karnage.compile_replay`` with zero GPU/CUDA
                           access. Stdout diffing is skipped, same as
                           *compile_only* (there is no app to print anything).

    Returns:
        Dict mapping ``script.stem`` to its :class:`~karnage.utils.models.FlipResult`.
    """
    flip_dir = output_dir / f"flip_{spec.flip_id:06d}"
    flip_dir.mkdir(parents=True, exist_ok=True)
    patch_spec_path = flip_dir / "patch_spec.json"
    patch_spec_path.write_text(
        json.dumps({"patch_vmas": [spec.site_vma], "mask": spec.flip_mask}, indent=2)
    )
    (flip_dir / "spec.json").write_text(json.dumps(_spec_to_dict(spec), indent=2))
    results: dict[str, FlipResult] = {}
    for script in triton_scripts:
        signature_in = (
            baseline_dir / script.stem / "signatures.json"
            if replay_signatures
            else None
        )
        _run_inferior(
            script,
            flip_dir / script.stem,
            patch_spec_path=patch_spec_path,
            timeout=flip_timeout,
            compile_only=compile_only,
            signature_in=signature_in,
            compile_replay=compile_replay,
        )
        results[script.stem] = _compare(
            spec,
            baseline_dir / script.stem,
            flip_dir / script.stem,
            compile_only=compile_only or compile_replay,
        )
    return results


def _log_result(per_script: dict[str, FlipResult]) -> None:
    """Emit the one-line per-flip summary to the logger.

    Shows a compact per-script outcome so interesting scripts stand out:
    ``softmax[STDOUT_DIFF,CODEGEN_DIFF] matmul[ok] layernorm[CRASH]``
    """
    spec = next(iter(per_script.values())).spec
    short_fn = spec.func_name.split("::")[-1][:30]
    parts: list[str] = []
    for stem, result in per_script.items():
        flags: list[str] = []
        if result.timed_out:
            flags.append("TIMEOUT")
        elif result.crashed:
            flags.append("CRASH" if not result.script_ran else "SCRIPT_FAILED")
        if result.codegen_changed:
            flags.append("CODEGEN_DIFF")
        if result.stdout_changed:
            flags.append("STDOUT_DIFF")
        parts.append(f"{stem}[{','.join(flags) if flags else 'ok'}]")
    logger.info(
        f"[{spec.flip_id:>6}] {short_fn}  "
        f"0x{spec.site_vma:x}  {spec.instr_type}  "
        f"{spec.opcode_before}→{spec.opcode_after}  → {' '.join(parts)}"
    )


def _iter_batches(lst: list, size: int) -> Iterator[list]:
    """Yield successive *size*-sized sublists. ``size <= 0`` yields one batch."""
    if size <= 0:
        yield lst
    else:
        for i in range(0, len(lst), size):
            yield lst[i : i + size]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_flipper(
    triton_scripts: list[Path],
    flip_sites_json: Path,
    output_dir: Path,
    *,
    max_flips: int | None = None,
    report_dir: Path | None = None,
    cooldown_every: int = 0,
    cooldown_secs: float = 30.0,
    filter_by_ptx: bool = False,
    tier_filter: int | None = None,
    type_filter: str | None = None,
    function_pattern: str | None = None,
    function_names: frozenset[str] | None = None,
    flip_ids: frozenset[int] | None = None,
    flip_timeout: float | None = None,
    workers: int = 1,
    compile_only: bool = False,
    replay_signatures: bool = False,
    compile_replay: bool = False,
) -> list[FlipResult]:
    """Run the full bit-flip test suite and return all results.

    Steps:

    1. Load ``flip_sites.json`` and build :class:`~karnage.utils.models.PatchSpec` list.
    2. Run a clean baseline (no GDB, no patch).
    3. Apply optional filters (tier, type, function name, flip ID, PTX relevance).
    4. For each spec: write ``patch_spec.json`` + ``spec.json``, spawn GDB,
       diff stdout and PTX against the baseline.
    5. Optionally write a JSON report.

    ``spec.json`` persists the full :class:`~karnage.utils.models.PatchSpec` in every
    flip directory so :func:`reconstruct_report` can rebuild the report later
    even if ``--report`` was not passed.

    Args:
        triton_scripts:   One or more Triton application scripts --- or, when
                          *compile_replay* is set, one or more capture blob
                          paths written by
                          :func:`karnage.compile_capture.capture_all_compiles`
                          (e.g. one per workload under
                          ``compile_specializations/``). Every entry is run
                          for *each* flip; results are aggregated with
                          ``any()`` so a flip is flagged if it affects at least
                          one. Each runs in its own subdir
                          (``flip_NNNNNN/{script.stem}/``) so Triton caches
                          never collide.
        flip_sites_json:  Path to ``flip_sites.json`` from the scan step.
        output_dir:       Root output directory for per-flip subdirectories.
        max_flips:        Stop after this many flips; ``None`` runs all.
        report_dir:       Directory to write one ``{stem}.json`` report per
                          script.  Created if it does not exist.
        cooldown_every:   Pause after every *N* flips (0 = disabled).
        cooldown_secs:    Duration of each cooldown pause in seconds.
        filter_by_ptx:    Keep only specs whose mnemonic root appears in
                          the baseline PTX.
        tier_filter:      Keep only specs from functions of this tier.
        type_filter:      Keep only sites of this instruction type.
        function_pattern: Regex filter on function name.
        function_names:   Exact demangled function names to target (loaded from
                          a ``--function-list`` file).  Composed with
                          *function_pattern* if both are given.
        flip_ids:         Exact ``flip_id`` values to target (loaded from a
                          ``--flip-ids-file``), typically the pruned candidate
                          list from a prior ``compile_only=True`` run. Requires
                          *tier_filter* / *type_filter* / *function_pattern* /
                          *function_names* to match the run that produced
                          those IDs --- see :func:`iter_patch_specs`.
        flip_timeout:     Per-flip GDB process timeout in seconds.
        workers:          Number of concurrent GDB flip processes.  Each flip
                          gets its own output directory and Triton cache so
                          there is no shared state between workers.  Cooldown
                          is applied between batches of ``cooldown_every``
                          completed flips.  Results are sorted by ``flip_id``
                          before being returned regardless of completion order.
                          In ``compile_only`` mode, GPU memory/execution
                          contention no longer bounds concurrency (no kernel
                          ever launches), so this can safely go much higher ---
                          the practical ceiling becomes CPU cores / host RAM
                          and the smaller per-worker GPU memory still needed
                          for CUDA context + input tensor allocation.
        compile_only:     Patch Triton (via ``KARNAGE_COMPILE_ONLY``, see
                          ``_wrapper.py``) so every kernel compiles but never
                          launches on the GPU. ``codegen_changed`` is still
                          fully meaningful; ``stdout_changed`` is always
                          ``False`` (the kernel's output is never computed).
                          GDB still applies the bit-flip exactly as normal ---
                          only the wrapped script's behavior changes.
        replay_signatures: Requires ``compile_only=True``. The baseline run
                          additionally records a kernel-call signature
                          manifest (``<baseline_dir>/<script.stem>/
                          signatures.json``); every flip then replays those
                          recorded calls directly instead of running
                          *triton_scripts* at all --- no torch/CUDA touch
                          beyond what ``triton.compile()`` itself needs.
                          Requires each script to gate its kernel-launching
                          code behind ``if __name__ == "__main__":`` (checked
                          by ``_wrapper.py`` during the baseline run; fails
                          fast with a clear message otherwise). See
                          ``_wrapper.py``'s "Signature capture / replay" docs.
        compile_replay:   Mutually exclusive with *compile_only* /
                          *replay_signatures* --- a different execution path,
                          not a further refinement of them. *triton_scripts*
                          are capture blob paths, not real scripts; each is
                          replayed via ``karnage.compile_replay`` (direct
                          ``triton.compile()`` calls) with zero GPU/CUDA
                          access, so GDB's bit-flip patch is exercised purely
                          by the compiler itself. Since no GPU/torch
                          contention exists at all here, ``workers`` can go
                          higher still than in ``compile_only`` mode. See
                          ``karnage/compile_capture.py`` and
                          ``karnage/compile_replay.py`` module docstrings.

    Returns:
        Dict mapping each script stem to its list of
        :class:`~karnage.utils.models.FlipResult` objects.
    """
    if replay_signatures and not compile_only:
        raise FlipperError(
            "replay_signatures requires compile_only=True (--replay-signatures "
            "requires --compile-only)"
        )
    if compile_replay and (compile_only or replay_signatures):
        raise FlipperError(
            "compile_replay is mutually exclusive with compile_only / "
            "replay_signatures --- it's a different execution path (direct "
            "triton.compile() replay from a capture blob), not a further "
            "refinement of them"
        )

    output_dir = output_dir.resolve()
    sites_data = _load_json(flip_sites_json)

    specs = list(
        iter_patch_specs(
            sites_data,
            tier_filter=tier_filter,
            type_filter=type_filter,
            function_pattern=function_pattern,
            function_names=function_names,
            flip_ids=flip_ids,
        )
    )
    if function_names is not None:
        matched = len({s.func_name for s in specs})
        missed = function_names - {s.func_name for s in specs}
        logger.info(
            f"Function list: {len(function_names)} requested, "
            f"{matched} found in flip_sites.json, {len(missed)} not found"
        )
        if missed:
            for name in sorted(missed):
                logger.warning(f"  not in flip_sites.json: {name}")
    if flip_ids is not None:
        found = {s.flip_id for s in specs}
        missed_ids = flip_ids - found
        logger.info(
            f"Flip ID list: {len(flip_ids)} requested, "
            f"{len(found)} found, {len(missed_ids)} not found"
        )
        if missed_ids:
            logger.warning(
                f"  not found (check --tier/--type/--function match the "
                f"run that produced these IDs): {sorted(missed_ids)[:20]}"
                + (" ..." if len(missed_ids) > 20 else "")
            )
    logger.info(f"Total specs before filtering: {len(specs):,}")

    if compile_only:
        logger.info("Compile-only mode: kernels compile but never launch on the GPU")
    if replay_signatures:
        logger.info(
            "Replay-signatures mode: baseline captures kernel-call signatures once; "
            "every flip replays them directly, with no app/torch/CUDA touch"
        )
    if compile_replay:
        logger.info(
            "Compile-replay mode: replaying captured triton.compile() calls "
            "directly, zero GPU/CUDA access --- GDB's patch is exercised "
            "purely by the compiler"
        )

    # --- Baseline ---
    baseline_dir = output_dir / "baseline"
    logger.info(
        f"Running baseline ({len(triton_scripts)} script(s), no patch)..."
    )
    for script in triton_scripts:
        script_baseline = baseline_dir / script.stem
        signature_out = (
            script_baseline / "signatures.json" if replay_signatures else None
        )
        if not _run_inferior(
            script,
            script_baseline,
            timeout=flip_timeout,
            compile_only=compile_only,
            signature_out=signature_out,
            compile_replay=compile_replay,
        ):
            timeout_note = (
                " (timed out)"
                if (script_baseline / "_timeout").exists()
                else ""
            )
            logger.warning(
                f"Baseline run failed for {script.name}{timeout_note} "
                f"--- see {script_baseline}/gdb_stderr.txt"
            )

    # --- PTX relevance filter ---
    if filter_by_ptx:
        ptx_mnemonics: frozenset[str] = frozenset()
        for script in triton_scripts:
            ptx_mnemonics |= extract_ptx_mnemonics(baseline_dir / script.stem)
        if not ptx_mnemonics:
            logger.warning(
                "Relevance filter requested but no PTX files found in baseline --- "
                "filter has no effect."
            )
        else:
            before = len(specs)
            specs = [
                s
                for s in specs
                if any(
                    m == s.opcode_before.split(".")[0] or m.startswith(s.opcode_before)
                    for m in ptx_mnemonics
                )
            ]
            logger.info(f"PTX relevance filter: {before:,} → {len(specs):,} specs")

    if max_flips is not None:
        specs = specs[:max_flips]
    logger.info(f"Flips to run: {len(specs):,}")

    # --- Flip runs ---
    results: dict[str, list[FlipResult]] = {s.stem: [] for s in triton_scripts}
    flip_fn = functools.partial(
        _flip_one,
        triton_scripts=triton_scripts,
        output_dir=output_dir,
        flip_timeout=flip_timeout,
        baseline_dir=baseline_dir,
        compile_only=compile_only,
        replay_signatures=replay_signatures,
        compile_replay=compile_replay,
    )
    desc_suffix = f" ({workers} workers)" if workers > 1 else ""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"Running flips{desc_suffix}", total=len(specs))

        first_batch = True
        for batch in _iter_batches(specs, cooldown_every):
            if not first_batch and cooldown_every > 0:
                logger.info(
                    f"[cooldown] {len(results)} flips done --- "
                    f"sleeping {cooldown_secs:.0f}s..."
                )
                time.sleep(cooldown_secs)
            first_batch = False

            if workers == 1:
                for spec in batch:
                    short_fn = spec.func_name.split("::")[-1][:30]
                    progress.update(
                        task,
                        description=(
                            f"[cyan]{short_fn}[/cyan] "
                            f"[dim]{spec.opcode_before}→{spec.opcode_after}[/dim]"
                        ),
                    )
                    try:
                        per_script = flip_fn(spec)
                    except Exception as exc:
                        # A crashed/timed-out flip is already reported as data
                        # by _run_inferior/_compare (crashed=True); reaching
                        # here means something else went wrong (e.g. disk
                        # I/O). One bad site must never abort the whole sweep.
                        logger.warning(
                            f"[{spec.flip_id}] flip_fn raised {type(exc).__name__}: "
                            f"{exc} --- skipping, not recorded in results"
                        )
                        progress.advance(task)
                        continue
                    _log_result(per_script)
                    for stem, r in per_script.items():
                        results[stem].append(r)
                    progress.advance(task)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(flip_fn, spec): spec for spec in batch}
                    for future in as_completed(futures):
                        spec = futures[future]
                        try:
                            per_script = future.result()
                        except Exception as exc:
                            logger.warning(
                                f"[{spec.flip_id}] flip_fn raised "
                                f"{type(exc).__name__}: {exc} --- skipping, "
                                f"not recorded in results"
                            )
                            progress.advance(task)
                            continue
                        _log_result(per_script)
                        for stem, r in per_script.items():
                            results[stem].append(r)
                        progress.advance(task)

    # Parallel completion order is nondeterministic; sort each script's list.
    for stem in results:
        results[stem].sort(key=lambda r: r.spec.flip_id)

    if report_dir:
        report_dir.mkdir(parents=True, exist_ok=True)
        for stem, script_results in results.items():
            report_path = report_dir / f"{stem}.json"
            with report_path.open("w") as f:
                json.dump([_serialise_result(r) for r in script_results], f, indent=2)
            crashed = sum(1 for r in script_results if r.crashed)
            codegen = sum(1 for r in script_results if r.codegen_changed)
            stdout = sum(1 for r in script_results if r.stdout_changed)
            logger.success(
                f"{stem}: {len(script_results)} flips — "
                f"{crashed} crashed, {codegen} codegen diff, {stdout} stdout diff "
                f"→ {report_path}"
            )

    return results
