"""ncu-based performance measurement for codegen-changed, non-crashed flip sites.

Given a report produced by ``karnage flip`` (including a fast ``--compile-only``
prescreen, whose ``stdout_changed`` is never meaningful --- see
:func:`_load_candidates`), finds flips that changed codegen (PTX / AMDGCN /
LLVM IR) without crashing, and measures each one's effect using ``ncu``
(Nsight Compute). Because this profiling run is itself a real execution
(unlike a compile-only prescreen), it also determines the *real*
``stdout_changed`` as a side effect of the same run used for timing --- no
separate full-execution pass is needed first. A flip that turns out to be
functionally invisible (no crash, no stdout change) *and* meaningfully slower
is the most dangerous kind of miscompile this whole pipeline is hunting for.

Each condition (baseline, and every candidate flip) is profiled with ncu's
``basic`` metric set (Duration, Compute/Memory Throughput, Achieved Occupancy,
Registers Per Thread, etc.) --- not just duration --- so a flagged regression
comes with the diagnostic context to explain *why* (e.g. occupancy dropped,
more registers spilled, throughput shifted). The ``Duration`` metric is the
one used for the regression decision by default; see ``primary_metric``.
Every condition is profiled ``repeats`` times (whole-process re-invocations,
not just kernel launches within one run) and medians are taken across all
launches from all repeats, since a single sample is thin evidence for a
performance claim.

Why this does *not* reuse the GDB-based flip mechanism
--------------------------------------------------------
:mod:`karnage.flipper` applies each bit-flip live in memory via ``gdb --batch``
ptrace (see ``_gdb_script.py``).  Empirically, ``ncu`` cannot profile a process
that is already ptraced by another debugger --- ``ncu -- gdb --batch --args
prog`` silently reports "No kernels were profiled", while the identical
invocation through any other intermediary (or no intermediary at all) works
fine.  This is Linux's ptrace exclusivity (one tracer per tracee), not a flag
or configuration issue, so nesting ``ncu`` under the GDB patcher is a dead end.

Instead, this module patches the target byte **on disk**, directly in the
real ``--library`` file (translating the flip site's linker VMA to a file
offset via :func:`karnage.utils.parser.linker_vma_to_file_offset`), runs the
script under plain ``ncu`` with no GDB involved, then restores the original
byte immediately afterward.  This only works because ``libtriton.so`` is
itself the Python C extension module Triton imports directly (loaded by
absolute path through Python's import machinery, not looked up by soname on
``LD_LIBRARY_PATH``), so writing the patched bytes into the real file is
sufficient to make the next ``import`` pick them up --- no need to redirect
the loader.

Because every measurement mutates one real, shared file in place, this
module is strictly single-threaded: one flip's patch -> profile -> restore
cycle always completes before the next begins.  There is no ``--workers``
option here.

Safety net
----------
The per-flip patch/restore is a try/finally (:func:`_patched_byte`), so the
original byte is restored on ordinary exceptions and on ``KeyboardInterrupt``.
It is *not* restored on ``SIGKILL`` or a host/container crash, since those
don't let Python's ``finally`` blocks run.  For that case, :func:`run_perf`
copies the library to ``<output>/<name>.golden`` before touching it at all,
and does a final MD5 comparison against that golden copy when the run
finishes --- if they don't match, the exact restore command is logged.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
from contextlib import contextmanager
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

from karnage.flipper.runner import iter_patch_specs
from karnage.utils.constants import ENV_ALWAYS_COMPILE, ENV_TRITON_CACHE
from karnage.utils.exceptions import PerfError
from karnage.utils.logger import console, logger
from karnage.utils.models import PatchSpec
from karnage.utils.parser import linker_vma_to_file_offset

_DEFAULT_PRIMARY_METRIC = "Duration"
_DEFAULT_REPEATS = 5


# ---------------------------------------------------------------------------
# Report filtering
# ---------------------------------------------------------------------------


def _load_candidates(report_path: Path) -> list[dict]:
    """Load a ``karnage flip`` report and keep only codegen-changed, non-crashed flips.

    Deliberately does **not** filter on the input report's ``stdout_changed``
    --- reports from a ``--compile-only`` prescreen always report it as
    ``False`` regardless of reality (the kernel never ran, so there's no real
    stdout to diff), which would make it meaningless as a filter here. Instead
    ``run_perf`` determines the *real* ``stdout_changed`` itself, for free,
    from the real execution it already does for timing --- see its module
    docstring. ``crashed``, by contrast, stays a reliable filter even from a
    compile-only report: it reflects a genuine compile-time crash either way.

    Older reports name the codegen-diff field ``ptx_changed``; newer ones
    call it ``codegen_changed`` --- both are accepted.

    Args:
        report_path: Path to a per-script report JSON written by ``flip`` or
                     reconstructed by ``report``.

    Returns:
        List of raw result dicts (as written by the flip runner) with
        ``codegen_changed`` (or ``ptx_changed``) true and ``crashed`` false,
        in report order.
    """
    results = json.loads(report_path.read_text())
    candidates = []
    for r in results:
        codegen_changed = r.get("codegen_changed", r.get("ptx_changed", False))
        if codegen_changed and not r["crashed"]:
            candidates.append(r)
    return candidates


# ---------------------------------------------------------------------------
# On-disk byte patch
# ---------------------------------------------------------------------------


@contextmanager
def _patched_byte(library: Path, offset: int, mask: int) -> Iterator[None]:
    """Flip one byte of *library* on disk for the duration of the ``with`` block.

    Reads the original byte at *offset*, XORs *mask* onto it, writes it back,
    yields, then restores the original byte in a ``finally`` --- so restoration
    happens on ordinary exceptions and ``KeyboardInterrupt``, though not on
    ``SIGKILL`` (see the module docstring for the golden-backup fallback).

    Args:
        library: Path to the shared object to patch in place.
        offset:  Byte offset within the file (already translated from a VMA).
        mask:    XOR mask applied to the byte (always ``0x01`` in practice).
    """
    with library.open("r+b") as f:
        f.seek(offset)
        original = f.read(1)
        if not original:
            raise PerfError(
                f"file offset 0x{offset:x} is past the end of {library}",
                context={"library": str(library), "offset": offset},
            )
        f.seek(offset)
        f.write(bytes([original[0] ^ mask]))
        f.flush()
        os.fsync(f.fileno())
    try:
        yield
    finally:
        with library.open("r+b") as f:
            f.seek(offset)
            f.write(original)
            f.flush()
            os.fsync(f.fileno())


def _md5(path: Path) -> str:
    """Return the hex MD5 digest of *path*, read in chunks."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# ncu invocation
# ---------------------------------------------------------------------------


def _parse_ncu_csv(log_text: str) -> list[dict[str, float]]:
    """Group ``ncu --csv --set basic`` rows into one metrics dict per kernel launch.

    Called on the ``--log-file`` contents, not ``proc.stdout``: ``--log-file``
    captures *all* ncu tool output, including the ``--csv`` table itself, so
    the ``==PROF==`` diagnostic lines share the file with the CSV table. This
    locates the CSV header line (starting with ``"ID"``) and parses from
    there on, ignoring everything before it.

    With ``--set basic``, ncu emits one CSV row per (kernel launch, metric)
    pair --- e.g. ~45 rows sharing the same "ID" for one launch --- plus
    blank-"Metric Name" rows that are just section-header separators (e.g.
    "LaunchStats", "Occupancy"); those are skipped. Rows sharing an "ID" are
    grouped back into a single ``{metric_name: value}`` dict per launch.
    Metric values are locale-grouped (e.g. ``"60,68,960"``); commas are
    stripped unconditionally before parsing.

    Args:
        log_text: Contents of the ``ncu --log-file`` output.

    Returns:
        One ``{metric_name: value}`` dict per kernel launch that matched the
        ``-k`` filter, in launch order.
    """
    lines = log_text.splitlines()
    header_idx = next(
        (i for i, line in enumerate(lines) if line.startswith('"ID"')), None
    )
    if header_idx is None:
        return []

    by_launch: dict[str, dict[str, float]] = {}
    order: list[str] = []
    for row in csv.DictReader(lines[header_idx:]):
        name = row.get("Metric Name") or ""
        if not name:
            continue  # section-header separator row, not a metric
        raw = (row.get("Metric Value") or "").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        launch_id = row.get("ID", "")
        if launch_id not in by_launch:
            by_launch[launch_id] = {}
            order.append(launch_id)
        by_launch[launch_id][name] = value

    return [by_launch[launch_id] for launch_id in order]


def _run_ncu(
    triton_script: Path,
    kernel_name: str,
    output_dir: Path,
    *,
    launch_skip: int = 0,
    timeout: float | None = None,
    ncu_path: str = "ncu",
) -> list[dict[str, float]]:
    """Profile *triton_script* once under ``ncu`` (no GDB) with the ``basic`` metric set.

    Forces a fresh, uncached Triton compilation (same env-var contract
    :func:`karnage.flipper.runner._run_inferior` uses) so the currently
    on-disk state of the library --- patched or not --- is actually what gets
    compiled against, never a stale cached kernel.

    Args:
        triton_script: Triton application script to run.
        kernel_name:   ``ncu -k`` filter (exact name or ``regex:...``).
        output_dir:    Directory for this run's Triton cache and ncu log.
        launch_skip:   Kernel launches to skip before profiling starts
                       (``ncu --launch-skip``); use to skip warmup iterations.
        timeout:       Kill the ``ncu`` process after this many seconds.
        ncu_path:      ``ncu`` executable to invoke.

    Returns:
        One ``{metric_name: value}`` dict per matched kernel launch (see
        :func:`_parse_ncu_csv`).

    Raises:
        PerfError: If ``ncu`` exits non-zero, times out, or produces no rows
                   for *kernel_name*.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "ncu_log.txt"

    env = {**os.environ}
    env[ENV_TRITON_CACHE] = str(output_dir / "triton_cache")
    env[ENV_ALWAYS_COMPILE] = "1"

    cmd = [
        ncu_path,
        "--target-processes", "all",
        "--set", "basic",
        "--csv",
        "-k", kernel_name,
        "--log-file", str(log_file),
    ]
    if launch_skip:
        cmd += ["--launch-skip", str(launch_skip)]
    cmd += ["--", sys.executable, str(triton_script)]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise PerfError(
            f"ncu timed out after {timeout}s profiling {triton_script.name}",
            context={"script": str(triton_script)},
        ) from exc
    except FileNotFoundError as exc:
        raise PerfError(f"ncu executable not found: {exc}") from exc

    # --log-file captures *all* ncu tool output --- including the --csv table
    # itself, not just the "==PROF==" diagnostic noise --- so proc.stdout ends
    # up containing only the profiled application's own prints.
    (output_dir / "app_stdout.txt").write_text(proc.stdout)

    if proc.returncode != 0:
        raise PerfError(
            f"ncu exited {proc.returncode} for {triton_script.name} --- see {log_file}",
            context={"returncode": proc.returncode, "log_file": str(log_file)},
        )

    try:
        log_text = log_file.read_text()
    except FileNotFoundError as exc:
        raise PerfError(
            f"ncu exited 0 but did not write {log_file} for {triton_script.name}",
            context={"log_file": str(log_file)},
        ) from exc

    launches = _parse_ncu_csv(log_text)
    if not launches:
        raise PerfError(
            f"ncu produced no rows matching kernel filter {kernel_name!r} "
            f"for {triton_script.name} --- see {log_file}",
            context={"kernel_name": kernel_name, "script": str(triton_script)},
        )
    return launches


def _profile_repeated(
    triton_script: Path,
    kernel_name: str,
    output_dir: Path,
    *,
    repeats: int,
    launch_skip: int = 0,
    timeout: float | None = None,
    ncu_path: str = "ncu",
) -> tuple[list[dict[str, float]], str]:
    """Run :func:`_run_ncu` *repeats* times and concatenate all launch samples.

    Each repeat is a full process re-invocation (fresh ``ncu`` + fresh
    ``triton_script`` process), not just multiple kernel launches within one
    run --- this is what actually averages out run-to-run noise (process
    startup variance, GPU clock state, etc.), which a single invocation
    cannot. Each repeat gets its own subdirectory so Triton caches and ncu
    logs never collide.

    Also captures the application's stdout from the first repeat --- this is
    a real execution (unlike a ``--compile-only`` prescreen), so its stdout is
    a genuine, comparable signal; one repeat is enough since the same patched
    library and inputs should produce deterministic output run to run, and
    ``repeats`` exists for timing noise, not output variance.

    Args:
        triton_script: Triton application script to run.
        kernel_name:   ``ncu -k`` filter (exact name or ``regex:...``).
        output_dir:    Parent directory; each repeat writes into
                       ``output_dir/rep_NN/``.
        repeats:       Number of full process re-invocations.
        launch_skip:   Forwarded to each repeat's :func:`_run_ncu` call.
        timeout:       Forwarded to each repeat's :func:`_run_ncu` call.
        ncu_path:      ``ncu`` executable to invoke.

    Returns:
        Tuple of (all per-launch metric dicts from every repeat, concatenated;
        the first repeat's captured application stdout).
    """
    samples: list[dict[str, float]] = []
    stdout_text = ""
    for i in range(repeats):
        rep_dir = output_dir / f"rep_{i:02d}"
        samples.extend(
            _run_ncu(
                triton_script,
                kernel_name,
                rep_dir,
                launch_skip=launch_skip,
                timeout=timeout,
                ncu_path=ncu_path,
            )
        )
        if i == 0:
            stdout_path = rep_dir / "app_stdout.txt"
            if stdout_path.exists():
                stdout_text = stdout_path.read_text(errors="replace")
    return samples, stdout_text


def _medians(launches: list[dict[str, float]]) -> dict[str, float]:
    """Return the median of every metric across all sampled kernel launches.

    Args:
        launches: Per-launch ``{metric_name: value}`` dicts, as returned by
                   :func:`_profile_repeated` (already concatenated across
                   repeats and, within each repeat, across kernel launches).

    Returns:
        ``{metric_name: median_value}`` for every metric name present in at
        least one launch.
    """
    names = {name for launch in launches for name in launch}
    return {
        name: statistics.median(
            launch[name] for launch in launches if name in launch
        )
        for name in names
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_perf(
    triton_script: Path,
    report_json: Path,
    flip_sites_json: Path,
    library: Path,
    output_dir: Path,
    *,
    kernel_name: str,
    primary_metric: str = _DEFAULT_PRIMARY_METRIC,
    repeats: int = _DEFAULT_REPEATS,
    launch_skip: int = 0,
    threshold_pct: float = 5.0,
    max_sites: int | None = None,
    run_timeout: float | None = None,
) -> list[dict]:
    """Measure the performance impact of every codegen-changed, non-crashed flip.

    Steps:

    1. Filter *report_json* down to codegen-changed, non-crashed flips (see
       :func:`_load_candidates` --- deliberately does not filter on the input
       report's ``stdout_changed``, since it's never meaningful from a
       ``--compile-only`` prescreen).
    2. Rebuild each candidate's :class:`~karnage.utils.models.PatchSpec` from
       *flip_sites_json* by ``flip_id`` (via
       :func:`karnage.flipper.runner.iter_patch_specs`, unfiltered --- must
       match the exact iteration that produced the report).
    3. Back up *library* to ``<output_dir>/<library.name>.golden`` and record
       its MD5.  Aborts before touching *library* if the backup fails.
    4. Profile *triton_script* *repeats* times with *library* untouched (the
       baseline), collecting the full ``basic`` ncu metric set each time, plus
       its stdout (see :func:`_profile_repeated`).
    5. For each candidate, serially: translate its VMA to a file offset,
       patch that byte on disk, profile *repeats* times, restore the byte,
       and diff its stdout against the baseline's --- a real execution, so
       this is the authoritative ``stdout_changed`` for this flip, not the
       (possibly meaningless) value from the input report.
    6. Write a JSON report to ``<output_dir>/perf_report.json``.
    7. Re-verify *library*'s MD5 against the golden copy's as a final
       integrity check.

    Args:
        triton_script:   Triton script to profile (must launch the kernel
                          named by *kernel_name*).
        report_json:      Report produced by ``karnage flip`` / ``report``.
        flip_sites_json:  ``flip_sites.json`` from the ``scan`` step.
        library:          Path to the real, loaded shared library (e.g.
                          ``.../triton/_C/libtriton.so``) --- patched on disk
                          in place, one byte at a time.
        output_dir:       Root directory for the golden backup, per-run
                          Triton caches, ncu logs, and the final report.
        kernel_name:      ``ncu -k`` filter identifying the kernel to time.
        primary_metric:   Which metric from the ``basic`` set (as ncu's
                          display name, e.g. ``"Duration"``,
                          ``"Compute (SM) Throughput"``) drives the
                          ``pct_change`` / ``regressed`` decision. All other
                          collected metrics are still recorded in the report
                          for diagnostic context.
        repeats:          Full process re-invocations per condition (baseline
                          and each flip); medians are taken across all
                          kernel launches from all repeats.
        launch_skip:      Kernel launches to skip before profiling (warmup),
                          applied within each repeat.
        threshold_pct:    Percent slowdown of *primary_metric* (vs. baseline
                          median) at or above which a flip is flagged
                          ``regressed``.
        max_sites:        Profile at most this many candidates (debug cap).
        run_timeout:      Per-``ncu``-invocation timeout in seconds.

    Returns:
        List of per-flip result dicts, one per successfully profiled
        candidate, each with ``baseline_metrics``, ``flip_metrics`` (medians
        for every collected metric), ``pct_change`` and ``regressed`` (both
        computed from *primary_metric*), and ``stdout_changed`` (the real,
        measured value from this run --- not copied from *report_json*).
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = _load_candidates(report_json)
    logger.info(
        f"{len(candidates)} candidate flips (codegen changed, no crash) --- "
        f"stdout_changed will be measured for real during profiling"
    )
    if max_sites is not None:
        candidates = candidates[:max_sites]
    if not candidates:
        logger.warning("No candidate flips to profile --- nothing to do")
        return []

    sites_data = json.loads(flip_sites_json.read_text())
    specs_by_id: dict[int, PatchSpec] = {
        s.flip_id: s for s in iter_patch_specs(sites_data)
    }

    # --- golden backup ---
    golden = output_dir / f"{library.name}.golden"
    try:
        shutil.copy2(library, golden)
    except OSError as exc:
        raise PerfError(
            f"failed to back up {library} to {golden} before patching --- aborting",
            context={"library": str(library), "golden": str(golden)},
        ) from exc
    golden_md5 = _md5(golden)
    logger.info(f"Golden backup written to {golden} (md5={golden_md5})")

    # --- baseline ---
    logger.info(
        f"Running baseline ncu profile ({repeats}x, library untouched)..."
    )
    baseline_launches, baseline_stdout = _profile_repeated(
        triton_script,
        kernel_name,
        output_dir / "baseline",
        repeats=repeats,
        launch_skip=launch_skip,
        timeout=run_timeout,
    )
    baseline_metrics = _medians(baseline_launches)
    if primary_metric not in baseline_metrics:
        raise PerfError(
            f"primary_metric {primary_metric!r} not found in ncu output --- "
            f"available metrics: {sorted(baseline_metrics)}",
            context={"primary_metric": primary_metric},
        )
    baseline_primary = baseline_metrics[primary_metric]
    logger.success(
        f"baseline {primary_metric}: {baseline_primary:,.2f} "
        f"(n={len(baseline_launches)} launches across {repeats} runs)"
    )

    # --- per-flip profiling (strictly serial: mutates the real library) ---
    results: list[dict] = []
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
        task = progress.add_task("Profiling silent flips", total=len(candidates))
        for r in candidates:
            flip_id = r["flip_id"]
            spec = specs_by_id.get(flip_id)
            if spec is None:
                logger.warning(
                    f"flip_id {flip_id} not found in {flip_sites_json} --- skipping"
                )
                progress.advance(task)
                continue

            short_fn = spec.func_name.split("::")[-1][:30]
            progress.update(task, description=f"[cyan]{short_fn}[/cyan]")

            offset = linker_vma_to_file_offset(library, ".text", spec.site_vma)
            flip_dir = output_dir / f"flip_{flip_id:06d}"

            try:
                with _patched_byte(library, offset, spec.flip_mask):
                    flip_launches, flip_stdout = _profile_repeated(
                        triton_script,
                        kernel_name,
                        flip_dir,
                        repeats=repeats,
                        launch_skip=launch_skip,
                        timeout=run_timeout,
                    )
            except PerfError as exc:
                logger.warning(f"[{flip_id}] ncu failed: {exc}")
                progress.advance(task)
                continue

            flip_metrics = _medians(flip_launches)
            if primary_metric not in flip_metrics:
                logger.warning(
                    f"[{flip_id}] ncu produced no {primary_metric!r} sample --- skipping"
                )
                progress.advance(task)
                continue

            flip_primary = flip_metrics[primary_metric]
            pct_change = (flip_primary - baseline_primary) / baseline_primary * 100.0
            regressed = pct_change >= threshold_pct
            stdout_changed = flip_stdout != baseline_stdout

            results.append(
                {
                    "flip_id": flip_id,
                    "func_name": spec.func_name,
                    "site_vma": f"0x{spec.site_vma:016x}",
                    "instr_type": spec.instr_type,
                    "opcode_before": spec.opcode_before,
                    "opcode_after": spec.opcode_after,
                    "primary_metric": primary_metric,
                    "baseline_metrics": baseline_metrics,
                    "flip_metrics": flip_metrics,
                    "n_baseline_launches": len(baseline_launches),
                    "n_flip_launches": len(flip_launches),
                    "pct_change": pct_change,
                    "regressed": regressed,
                    "stdout_changed": stdout_changed,
                }
            )
            flag = "REGRESSED" if regressed else "ok"
            stdout_flag = " STDOUT_DIFF" if stdout_changed else ""
            logger.info(
                f"[{flip_id:>6}] {short_fn}  {primary_metric}: "
                f"{baseline_primary:,.2f} -> {flip_primary:,.2f}  "
                f"({pct_change:+.1f}%)  {flag}{stdout_flag}"
            )
            progress.advance(task)

    # --- final integrity check ---
    final_md5 = _md5(library)
    if final_md5 != golden_md5:
        logger.failure(
            f"library MD5 mismatch after perf run! expected {golden_md5}, "
            f"got {final_md5} --- restore with: cp {golden} {library}"
        )
    else:
        logger.success(f"library MD5 verified unchanged: {final_md5}")

    report_path = output_dir / "perf_report.json"
    with report_path.open("w") as f:
        json.dump(results, f, indent=2)

    regressed_count = sum(1 for r in results if r["regressed"])
    stdout_changed_count = sum(1 for r in results if r["stdout_changed"])
    silent_regressions = [
        r for r in results if r["regressed"] and not r["stdout_changed"]
    ]
    logger.success(
        f"{len(results)} flips profiled, {regressed_count} regressed "
        f">= {threshold_pct}%, {stdout_changed_count} changed stdout --- {report_path}"
    )
    if silent_regressions:
        logger.failure(
            f"{len(silent_regressions)} SILENT regressions --- codegen changed, "
            f"no crash, stdout identical, but measurably slower:"
        )
        for r in sorted(silent_regressions, key=lambda r: -r["pct_change"])[:5]:
            logger.info(
                f"  flip_{r['flip_id']:06d}  "
                f"{r['func_name'].split('::')[-1][:40]}  {r['pct_change']:+.1f}%"
            )
    elif regressed_count:
        worst = sorted(
            (r for r in results if r["regressed"]), key=lambda r: -r["pct_change"]
        )[:5]
        for r in worst:
            logger.info(
                f"  flip_{r['flip_id']:06d}  "
                f"{r['func_name'].split('::')[-1][:40]}  {r['pct_change']:+.1f}%"
            )

    return results
