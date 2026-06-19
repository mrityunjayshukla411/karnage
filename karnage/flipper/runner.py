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

import json
import os
import re
import subprocess
import sys
import time
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
    ENV_OUTPUT_DIR,
    ENV_PATCH_SPEC,
    ENV_TRITON_CACHE,
)
from karnage.utils.exceptions import ScannerError
from karnage.utils.logger import console, logger
from karnage.utils.models import FlipResult, PatchSpec

_THIS_DIR = Path(__file__).parent
_GDB_SCRIPT = _THIS_DIR / "_gdb_script.py"
_WRAPPER = _THIS_DIR / "_wrapper.py"


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


def _iter_patch_specs(
    sites_data: dict,
    *,
    tier_filter: int | None = None,
    type_filter: str | None = None,
    function_pattern: str | None = None,
) -> Iterator[PatchSpec]:
    """Yield one :class:`~karnage.utils.models.PatchSpec` per flip site.

    Args:
        sites_data:       Parsed ``flip_sites.json`` dict.
        tier_filter:      Only yield specs from functions of this tier.
        type_filter:      Only yield sites of this instruction type.
        function_pattern: Regex matched against the function's demangled name.

    Yields:
        :class:`~karnage.utils.models.PatchSpec` in function-name / site order.
    """
    pat = re.compile(function_pattern) if function_pattern else None
    flip_id = 0

    for func_name, fd in sites_data.get("functions", {}).items():
        tier = fd.get("tier", 3)
        if tier_filter is not None and tier != tier_filter:
            continue
        if pat is not None and not pat.search(func_name):
            continue

        for site in fd.get("sites", []):
            if type_filter is not None and site["instr_type"] != type_filter:
                continue
            yield PatchSpec(
                flip_id=flip_id,
                func_name=func_name,
                site_vma=int(site["site_vma"], 16),
                instr_type=site["instr_type"],
                opcode_before=site["opcode_before"],
                opcode_after=site["opcode_after"],
                flip_mask=int(site["flip_mask"], 16),
            )
            flip_id += 1


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _run_inferior(
    triton_script: Path,
    output_dir: Path,
    *,
    patch_spec_path: Path | None = None,
    timeout: float | None = None,
) -> bool:
    """Run the Triton script, optionally under GDB with a patch applied.

    Files written to *output_dir*:

    - ``app_stdout.txt`` --- written by ``_wrapper.py`` via redirected
      ``sys.stdout``; contains only the user script's ``print()`` output.
    - ``gdb_stdout.txt`` --- raw GDB process stdout (thread events, vfork
      messages, debuginfod prompts); kept for debugging, not used for diffing.
    - ``gdb_stderr.txt`` --- GDB Python diagnostics (``gdb.write(..., gdb.STDERR)``).
    - ``returncode.txt`` --- integer exit code as a string.

    Sentinels ``_done`` and ``_error.txt`` are written by ``_wrapper.py``
    inside *output_dir* (via ``KARNAGE_OUTPUT_DIR``).

    Args:
        triton_script:   Path to the user Triton script.
        output_dir:      Directory to write output files into.
        patch_spec_path: Path to ``patch_spec.json``; ``None`` for baseline.
        timeout:         SIGTERM the GDB process after this many seconds.

    Returns:
        ``True`` if the process exited with code 0.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ}
    env[ENV_OUTPUT_DIR] = str(output_dir)          # sentinels land here directly
    env[ENV_TRITON_CACHE] = str(output_dir / "triton_cache")
    env[ENV_ALWAYS_COMPILE] = "1"

    if patch_spec_path is None:
        cmd = [sys.executable, str(_WRAPPER), str(triton_script)]
    else:
        env[ENV_PATCH_SPEC] = str(patch_spec_path)
        cmd = [
            "gdb",
            "--batch",
            "-q",
            # Suppress the debuginfod auto-download prompt before our script loads.
            "-iex", "set debuginfod enabled off",
            "-x", str(_GDB_SCRIPT),
            "--args", sys.executable, str(_WRAPPER), str(triton_script),
        ]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=timeout
        )
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        rc = -1
        (output_dir / "_timeout").touch()
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
# PTX collection (kept for --filter-by-ptx and PTX_DIFF reporting)
# ---------------------------------------------------------------------------


def _collect_ptx(output_dir: Path) -> list[str]:
    cache_dir = output_dir / "triton_cache"
    if not cache_dir.exists():
        return []
    return [f.read_text(errors="replace") for f in sorted(cache_dir.rglob("*.ptx"))]


# ---------------------------------------------------------------------------
# PTX mnemonic extraction (for --filter-by-ptx)
# ---------------------------------------------------------------------------

_PTX_INSTR_RE = re.compile(
    r"^\s*"
    r"(?:@[!%\w]+\s+)?"
    r"([a-z][a-z0-9]*(?:\.[a-z][a-z0-9]*)*)\b"
)
_PTX_SKIP_CHARS = frozenset({".", "/", "$", "{", "}", "@"})


def extract_ptx_mnemonics(baseline_dir: Path) -> frozenset[str]:
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


# ---------------------------------------------------------------------------
# Result comparison
# ---------------------------------------------------------------------------


def _compare(
    spec: PatchSpec,
    baseline_dir: Path,
    flip_dir: Path,
) -> FlipResult:
    """Compare a flip run against the baseline using stdout diff and PTX diff.

    Crash detection uses the ``_done`` / ``_error.txt`` sentinels written by
    ``_wrapper.py`` and the saved ``returncode.txt``, not the GDB exit code
    alone (GDB ``--batch`` exits 0 even when the inferior crashes).

    Args:
        spec:         The :class:`~karnage.utils.models.PatchSpec` applied.
        baseline_dir: Output directory of the clean baseline run.
        flip_dir:     Output directory of this flip run.

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

    # PTX diff
    baseline_ptx = _collect_ptx(baseline_dir)
    flip_ptx = _collect_ptx(flip_dir)
    ptx_changed = bool(baseline_ptx) and bool(flip_ptx) and baseline_ptx != flip_ptx

    # Stdout diff — only meaningful when the script actually ran
    stdout_changed = False
    if not crashed and not timed_out:
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
        ptx_changed=ptx_changed,
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
        "ptx_changed": r.ptx_changed,
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


def reconstruct_report(output_dir: Path) -> list[dict]:
    """Reconstruct a JSON report from an existing flip output directory.

    Scans all ``flip_NNNNNN/`` subdirectories, reads ``spec.json`` from each,
    and re-runs the comparison logic against the baseline directory.  Requires
    that the run was performed with a version of karnage that writes
    ``spec.json`` (i.e. this version or later).

    Args:
        output_dir: Root output directory passed to ``karnage flip --output``.

    Returns:
        List of serialised result dicts, the same format as ``--report`` JSON.
    """
    baseline_dir = output_dir / "baseline"
    if not baseline_dir.exists():
        raise FileNotFoundError(f"baseline/ not found in {output_dir}")

    flip_dirs = sorted(output_dir.glob("flip_*/"))
    results: list[dict] = []

    for flip_dir in flip_dirs:
        spec_file = flip_dir / "spec.json"
        if not spec_file.exists():
            logger.warning(f"No spec.json in {flip_dir.name} — skipping")
            continue
        spec = _spec_from_dict(json.loads(spec_file.read_text()))
        result = _compare(spec, baseline_dir, flip_dir)
        results.append(_serialise_result(result))

    return results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_flipper(
    triton_script: Path,
    flip_sites_json: Path,
    output_dir: Path,
    *,
    max_flips: int | None = None,
    report_json: Path | None = None,
    cooldown_every: int = 0,
    cooldown_secs: float = 30.0,
    filter_by_ptx: bool = False,
    tier_filter: int | None = None,
    type_filter: str | None = None,
    function_pattern: str | None = None,
    flip_timeout: float | None = None,
) -> list[FlipResult]:
    """Run the full bit-flip test suite and return all results.

    Steps:

    1. Load ``flip_sites.json`` and build :class:`~karnage.utils.models.PatchSpec` list.
    2. Run a clean baseline (no GDB, no patch).
    3. Apply optional filters (tier, type, function name, PTX relevance).
    4. For each spec: write ``patch_spec.json`` + ``spec.json``, spawn GDB,
       diff stdout and PTX against the baseline.
    5. Optionally write a JSON report.

    ``spec.json`` persists the full :class:`~karnage.utils.models.PatchSpec` in every
    flip directory so :func:`reconstruct_report` can rebuild the report later
    even if ``--report`` was not passed.

    Args:
        triton_script:    Path to the Triton application script.
        flip_sites_json:  Path to ``flip_sites.json`` from the scan step.
        output_dir:       Root output directory for per-flip subdirectories.
        max_flips:        Stop after this many flips; ``None`` runs all.
        report_json:      Write a JSON array of serialised results here.
        cooldown_every:   Pause after every *N* flips (0 = disabled).
        cooldown_secs:    Duration of each cooldown pause in seconds.
        filter_by_ptx:    Keep only specs whose mnemonic root appears in
                          the baseline PTX.
        tier_filter:      Keep only specs from functions of this tier.
        type_filter:      Keep only sites of this instruction type.
        function_pattern: Regex filter on function name.
        flip_timeout:     Per-flip GDB process timeout in seconds.

    Returns:
        List of :class:`~karnage.utils.models.FlipResult` objects.
    """
    output_dir = output_dir.resolve()
    sites_data = _load_json(flip_sites_json)

    specs = list(
        _iter_patch_specs(
            sites_data,
            tier_filter=tier_filter,
            type_filter=type_filter,
            function_pattern=function_pattern,
        )
    )
    logger.info(f"Total specs before filtering: {len(specs):,}")

    # --- Baseline ---
    baseline_dir = output_dir / "baseline"
    logger.info("Running baseline (no patch)...")
    if not _run_inferior(triton_script, baseline_dir):
        logger.warning("Baseline run failed --- see baseline/gdb_stderr.txt")

    # --- PTX relevance filter ---
    if filter_by_ptx:
        ptx_mnemonics = extract_ptx_mnemonics(baseline_dir)
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
    results: list[FlipResult] = []
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
        task = progress.add_task("Running flips", total=len(specs))
        for spec in specs:
            short_fn = spec.func_name.split("::")[-1][:30]
            progress.update(
                task,
                description=(
                    f"[cyan]{short_fn}[/cyan] "
                    f"[dim]{spec.opcode_before}→{spec.opcode_after}[/dim]"
                ),
            )

            flip_dir = output_dir / f"flip_{spec.flip_id:06d}"
            flip_dir.mkdir(parents=True, exist_ok=True)

            # patch_spec.json --- consumed by _gdb_script.py
            (flip_dir / "patch_spec.json").write_text(
                json.dumps(
                    {"patch_vmas": [spec.site_vma], "mask": spec.flip_mask},
                    indent=2,
                )
            )
            # spec.json --- full PatchSpec for retroactive report reconstruction
            (flip_dir / "spec.json").write_text(
                json.dumps(_spec_to_dict(spec), indent=2)
            )

            _run_inferior(
                triton_script,
                flip_dir,
                patch_spec_path=flip_dir / "patch_spec.json",
                timeout=flip_timeout,
            )
            result = _compare(spec, baseline_dir, flip_dir)
            results.append(result)

            flags = []
            if result.timed_out:
                flags.append("TIMEOUT")
            elif result.crashed:
                flags.append("CRASH" if not result.script_ran else "SCRIPT_FAILED")
            if result.ptx_changed:
                flags.append("PTX_DIFF")
            if result.stdout_changed:
                flags.append("STDOUT_DIFF")

            outcome = ", ".join(flags) if flags else "no change"
            logger.info(
                f"[{spec.flip_id:>6}] {short_fn}  "
                f"0x{spec.site_vma:x}  {spec.instr_type}  "
                f"{spec.opcode_before}→{spec.opcode_after}  → {outcome}"
            )
            progress.advance(task)

            flips_done = len(results)
            if cooldown_every > 0 and flips_done % cooldown_every == 0:
                logger.info(
                    f"[cooldown] {flips_done} flips done --- "
                    f"sleeping {cooldown_secs:.0f}s..."
                )
                time.sleep(cooldown_secs)

    if report_json:
        with report_json.open("w") as f:
            json.dump([_serialise_result(r) for r in results], f, indent=2)
        logger.success(f"Report written → {report_json}")

    return results
