#!/usr/bin/env python3
"""
karnage --- Target-independent LLVM bit-flip explorer.

Subcommands
-----------
  scan    Discover target-independent LLVM functions in a shared library and
          collect flip-candidate instructions (short Jcc, long Jcc, CMOV).
          Outputs flip_sites.json.

  flip    Run GDB-based bit-flip tests using flip_sites.json and compare
          generated PTX and application stdout against a clean baseline.
          --compile-only runs a fast GPU-launch-free prescreen (kernels
          compile but never run) to prune the candidate list by codegen
          change alone before the expensive full run.

  report  Reconstruct a JSON report from an existing output directory.
          Use this when you forgot --report during a flip run.

  perf    Measure the performance impact of codegen-changed, non-crashed
          flips from a flip report using ncu. Since profiling is itself a
          real execution, it also measures the real stdout_changed as a
          side effect (the input report's stdout_changed is ignored, since
          it's never meaningful from a --compile-only prescreen).

  capture Run a workload for real (one real GPU launch) and record every
          triton.compile() call it makes, for later GPU-free replay via
          'flip --compile-replay' --- true zero-CUDA-touch fault injection,
          reusing GDB + --workers with no per-worker library copies needed.

Quick start
-----------
  # Step 1 - discover flip sites
  python main.py scan --library /path/to/libtriton.so

  # Step 1a - preview discovered functions without writing JSON
  python main.py scan --library /path/to/libtriton.so --list

  # Step 2 - run bit-flip tests across all kernels
  python main.py flip \\
      --script triton_kernels/*.py \\
      --library /path/to/libtriton.so \\
      --function-list common_high_level_functions_amd_nvidia.txt \\
      --workers 8 \\
      --flip-timeout 120 \\
      --cooldown-every 100 --cooldown-secs 300 \\
      --output test_results/ \\
      --report reports/run_01/

  # Step 2a (optional) - fast GPU-launch-free prescreen to prune candidates
  # by codegen change alone, then restrict the real Step 2 run to survivors
  python main.py flip --compile-only --workers 64 \\
      --script triton_kernels/vector_add.py \\
      --library /path/to/libtriton.so \\
      --output prescreen/ --report prescreen_reports/
  # (filter prescreen_reports/*.json for codegen_changed=True flip_ids,
  #  write them one per line to pruned_ids.txt)
  python main.py flip --flip-ids-file pruned_ids.txt --workers 8 \\
      --script triton_kernels/vector_add.py \\
      --library /path/to/libtriton.so \\
      --output test_results/ --report reports/run_01/

  # Step 3 - rebuild report from an existing output dir (if --report was omitted)
  python main.py report --output test_results/ --report results.json

  # Step 4 - measure performance impact of codegen-changed, non-crashed
  # flips using ncu (also measures the real stdout_changed for free)
  python main.py perf \\
      --report reports/run_01/vector_add.json \\
      --script triton_kernels/vector_add.py \\
      --library /path/to/libtriton.so \\
      --sites flip_sites.json \\
      --kernel-name vector_add_kernel \\
      --output perf_results/

  # Step 5 (optional) - true zero-GPU fault injection: capture once (real
  # GPU launch), then replay every flip's compile with no CUDA access at all
  python main.py capture \\
      --script triton_kernels/attention.py \\
      --output compile_specializations/attention.pkl
  python main.py flip --compile-replay --workers 64 \\
      --script compile_specializations/attention.pkl \\
      --library /path/to/libtriton.so \\
      --output compile_replay_results/ --report compile_replay_reports/
"""

import argparse
import json
from pathlib import Path

from karnage.flipper import run_flipper
from karnage.flipper.runner import reconstruct_report
from karnage.perf import run_perf
from karnage.scanner import scan_binary
from karnage.scanner.scanner import scan_result_to_dict
from karnage.utils.constants import DEFAULT_FLIP_SITES, DEFAULT_OUTPUT_DIR
from karnage.utils.logger import logger


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


def _cmd_scan(args: argparse.Namespace) -> None:
    """Discover target-independent LLVM functions and write flip_sites.json.

    Args:
        args: Parsed CLI arguments for the ``scan`` subcommand.
    """
    if not args.library.exists():
        raise SystemExit(f"--library: not found: {args.library}")

    result = scan_binary(
        args.library,
        window=args.window,
        function_pattern=args.function,
    )

    # Apply tier filter for listing / output
    functions = result.functions
    if args.tier is not None:
        functions = tuple(f for f in functions if f.tier == args.tier)

    if args.list:
        # Pretty-print table of discovered functions + site counts
        total_sites = sum(len(f.sites) for f in functions)
        print(
            f"\n{'Tier':>4}  {'Class':30}  {'Sites':>6}  Function\n"
            + "-" * 100
        )
        for fs in functions:
            print(
                f"  {fs.tier}   {fs.class_name:30}  {len(fs.sites):6}  {fs.name}"
            )
        by_type: dict[str, int] = {}
        for fs in functions:
            for s in fs.sites:
                by_type[s.instr_type] = by_type.get(s.instr_type, 0) + 1
        print(
            f"\n{len(functions)} functions, {total_sites} total sites  "
            + "  ".join(f"{k}={v}" for k, v in sorted(by_type.items()))
        )
        return

    # Filter the result dict before writing
    raw = scan_result_to_dict(result)
    if args.tier is not None:
        raw["functions"] = {
            k: v for k, v in raw["functions"].items() if v["tier"] == args.tier
        }
        # Recompute meta counts
        total = sum(len(v["sites"]) for v in raw["functions"].values())
        by_type: dict[str, int] = {}
        for v in raw["functions"].values():
            for s in v["sites"]:
                by_type[s["instr_type"]] = by_type.get(s["instr_type"], 0) + 1
        raw["meta"]["total_functions"] = len(raw["functions"])
        raw["meta"]["total_sites"] = total
        raw["meta"]["site_counts"] = by_type

    with args.output.open("w") as f:
        json.dump(raw, f, indent=2)

    logger.success(
        f"Wrote {raw['meta']['total_functions']:,} functions, "
        f"{raw['meta']['total_sites']:,} sites → {args.output}"
    )


# ---------------------------------------------------------------------------
# flip
# ---------------------------------------------------------------------------


def _cmd_flip(args: argparse.Namespace) -> None:
    """Validate inputs and run the GDB-based bit-flip test suite.

    Args:
        args: Parsed CLI arguments for the ``flip`` subcommand.
    """
    for path in args.script:
        if not path.exists():
            raise SystemExit(f"--script: not found: {path}")
    for path, flag in [
        (args.sites, "--sites"),
        (args.library, "--library"),
    ]:
        if not path.exists():
            raise SystemExit(f"{flag}: not found: {path}")

    function_names: frozenset[str] | None = None
    if args.function_list is not None:
        if not args.function_list.exists():
            raise SystemExit(f"--function-list: not found: {args.function_list}")
        lines = args.function_list.read_text().splitlines()
        function_names = frozenset(l.strip() for l in lines if l.strip())

    flip_ids: frozenset[int] | None = None
    if args.flip_ids_file is not None:
        if not args.flip_ids_file.exists():
            raise SystemExit(f"--flip-ids-file: not found: {args.flip_ids_file}")
        lines = args.flip_ids_file.read_text().splitlines()
        try:
            flip_ids = frozenset(int(line.strip()) for line in lines if line.strip())
        except ValueError as exc:
            raise SystemExit(
                f"--flip-ids-file: expected one integer flip_id per line: {exc}"
            ) from exc

    if args.replay_signatures and not args.compile_only:
        raise SystemExit("--replay-signatures requires --compile-only")
    if args.compile_replay and (args.compile_only or args.replay_signatures):
        raise SystemExit(
            "--compile-replay is mutually exclusive with --compile-only / "
            "--replay-signatures --- it's a different execution path (direct "
            "triton.compile() replay from a capture blob), not a further "
            "refinement of them"
        )

    run_flipper(
        triton_scripts=args.script,
        flip_sites_json=args.sites,
        output_dir=args.output,
        max_flips=args.max_flips,
        report_dir=args.report,
        cooldown_every=args.cooldown_every,
        cooldown_secs=args.cooldown_secs,
        filter_by_ptx=args.filter_by_ptx,
        tier_filter=args.tier,
        type_filter=args.type,
        function_pattern=args.function,
        function_names=function_names,
        flip_ids=flip_ids,
        flip_timeout=args.flip_timeout,
        workers=args.workers,
        compile_only=args.compile_only,
        replay_signatures=args.replay_signatures,
        compile_replay=args.compile_replay,
    )


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _cmd_report(args: argparse.Namespace) -> None:
    """Reconstruct per-script JSON reports from an existing flip output directory.

    Reads ``spec.json`` from every ``flip_NNNNNN/`` subdirectory, re-runs the
    stdout and codegen comparison against the baseline, and writes one
    ``{script_stem}.json`` file per script into ``--report DIR``.

    Args:
        args: Parsed CLI arguments for the ``report`` subcommand.
    """
    if not args.output.is_dir():
        raise SystemExit(f"--output: directory not found: {args.output}")
    if not (args.output / "baseline").is_dir():
        raise SystemExit(
            f"--output: no baseline/ directory found in {args.output}\n"
            "This directory was not produced by 'karnage flip'."
        )

    try:
        results = reconstruct_report(args.output)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    args.report.mkdir(parents=True, exist_ok=True)
    for stem, script_results in results.items():
        report_path = args.report / f"{stem}.json"
        with report_path.open("w") as f:
            json.dump(script_results, f, indent=2)
        crashed = sum(1 for r in script_results if r["crashed"])
        codegen_changed = sum(1 for r in script_results if r["codegen_changed"])
        stdout_changed = sum(1 for r in script_results if r["stdout_changed"])
        logger.success(
            f"{stem}: {len(script_results)} flips — "
            f"{crashed} crashed, {codegen_changed} codegen diff, {stdout_changed} stdout diff "
            f"→ {report_path}"
        )


# ---------------------------------------------------------------------------
# perf
# ---------------------------------------------------------------------------


def _cmd_perf(args: argparse.Namespace) -> None:
    """Measure the performance impact of codegen-changed, non-crashed flips using ncu.

    Args:
        args: Parsed CLI arguments for the ``perf`` subcommand.
    """
    for path, flag in [
        (args.report, "--report"),
        (args.script, "--script"),
        (args.library, "--library"),
        (args.sites, "--sites"),
    ]:
        if not path.exists():
            raise SystemExit(f"{flag}: not found: {path}")

    run_perf(
        triton_script=args.script,
        report_json=args.report,
        flip_sites_json=args.sites,
        library=args.library,
        output_dir=args.output,
        kernel_name=args.kernel_name,
        primary_metric=args.primary_metric,
        repeats=args.repeats,
        launch_skip=args.launch_skip,
        threshold_pct=args.threshold,
        max_sites=args.max_sites,
        run_timeout=args.run_timeout,
        ncu_metrics=args.ncu_metrics,
    )


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------


def _cmd_capture(args: argparse.Namespace) -> None:
    """Run a workload for real and capture every triton.compile() call it makes.

    Args:
        args: Parsed CLI arguments for the ``capture`` subcommand.
    """
    if not args.script.exists():
        raise SystemExit(f"--script: not found: {args.script}")

    import runpy

    from karnage.compile_capture import capture_all_compiles

    with capture_all_compiles(args.output) as store:
        runpy.run_path(str(args.script), run_name="__main__")

    n_kernels = len(store)
    n_specs = sum(len(v) for v in store.values())
    if n_specs == 0:
        raise SystemExit(
            f"capture produced no entries --- did {args.script.name} call any "
            f"@triton.jit kernel?"
        )
    logger.success(
        f"Captured {n_specs} specialization(s) across {n_kernels} kernel(s) "
        f"from {args.script.name} → {args.output}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the top-level argument parser with all subcommands."""
    ap = argparse.ArgumentParser(
        prog="karnage",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # -- scan --
    p_scan = sub.add_parser(
        "scan",
        help="Discover target-independent LLVM functions and collect flip sites.",
        description=(
            "Run nm on the target binary to find target-independent LLVM functions "
            "(DAGCombiner, InstCombiner, LegalizeDAG, etc.), then use objdump to "
            "locate short Jcc, long Jcc, and CMOV flip candidates within each. "
            "Writes flip_sites.json."
        ),
    )
    p_scan.add_argument(
        "--library",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to the shared library to scan (e.g. libtriton.so)",
    )
    p_scan.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_FLIP_SITES),
        metavar="PATH",
        help=f"Output path for the flip-sites JSON (default: {DEFAULT_FLIP_SITES})",
    )
    p_scan.add_argument(
        "--list",
        action="store_true",
        help="Print a summary table and exit without writing JSON",
    )
    p_scan.add_argument(
        "--function",
        metavar="PATTERN",
        default=None,
        help="Regex filter applied to function demangled names",
    )
    p_scan.add_argument(
        "--tier",
        type=int,
        choices=[0, 1, 2, 3],
        default=None,
        metavar="{0,1,2,3}",
        help=(
            "Only include functions of this tier. "
            "0=cross-backend TTIR/TTGIR/LLVM IR (architectural guarantee); "
            "1=NVPTX SelectionDAG visitors; 2=NVPTX SelectionDAG infra; "
            "3=likely crash"
        ),
    )
    p_scan.add_argument(
        "--window",
        type=lambda x: int(x, 0),
        default=0x2000,
        metavar="BYTES",
        help="objdump disassembly window per function in bytes (default: 0x2000)",
    )
    p_scan.set_defaults(func=_cmd_scan)

    # -- flip --
    p_flip = sub.add_parser(
        "flip",
        help="Run GDB-based bit-flip tests from flip_sites.json.",
        description=(
            "For each flip site in flip_sites.json, XOR one bit of the corresponding "
            "instruction byte in the loaded shared library via GDB, then compare the "
            "generated PTX and tensor outputs against a clean baseline."
        ),
    )
    p_flip.add_argument(
        "--script",
        type=Path,
        required=True,
        nargs="+",
        metavar="PATH",
        help=(
            "Triton application script(s) to test.  Pass multiple paths to run "
            "all scripts for every flip; results are aggregated (a flip is flagged "
            "if it affects any script).  Example: --script a.py b.py c.py"
        ),
    )
    p_flip.add_argument(
        "--library",
        type=Path,
        required=True,
        help="Path to the shared library (needed to verify it exists; GDB finds it at runtime)",
    )
    p_flip.add_argument(
        "--sites",
        type=Path,
        default=Path(DEFAULT_FLIP_SITES),
        metavar="PATH",
        help=f"flip_sites.json from scan (default: {DEFAULT_FLIP_SITES})",
    )
    p_flip.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        metavar="DIR",
        help=f"Root output directory for per-flip results (default: {DEFAULT_OUTPUT_DIR})",
    )
    p_flip.add_argument(
        "--max-flips",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N flips (useful for a quick sanity check)",
    )
    p_flip.add_argument(
        "--report",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory to write per-script JSON reports into.  "
            "One file per script is created: ``{DIR}/{script_stem}.json``.  "
            "Created if it does not exist."
        ),
    )
    p_flip.add_argument(
        "--tier",
        type=int,
        choices=[0, 1, 2, 3],
        default=None,
        metavar="{0,1,2,3}",
        help=(
            "Only test sites in functions of this tier. "
            "0=cross-backend (TTIR/TTGIR/LLVM IR); 1=NVPTX DAG visitors; "
            "2=NVPTX DAG infra; 3=likely crash"
        ),
    )
    p_flip.add_argument(
        "--type",
        choices=["short_jcc", "long_jcc", "cmov"],
        default=None,
        metavar="{short_jcc,long_jcc,cmov}",
        help="Only test sites of this instruction type",
    )
    p_flip.add_argument(
        "--function",
        metavar="PATTERN",
        default=None,
        help="Regex filter applied to function demangled names",
    )
    p_flip.add_argument(
        "--function-list",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Path to a text file of exact demangled function names to target "
            "(one per line). Only flip sites belonging to listed functions are run. "
            "Can be combined with --function."
        ),
    )
    p_flip.add_argument(
        "--flip-ids-file",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Path to a text file of exact flip_id values to target (one per "
            "line), typically the pruned candidate list from a --compile-only "
            "prescreen run. Requires --tier/--type/--function/--function-list "
            "to match the run that produced those IDs."
        ),
    )
    p_flip.add_argument(
        "--compile-only",
        action="store_true",
        help=(
            "Patch Triton so every kernel compiles but never launches on the "
            "GPU (codegen_changed is still meaningful; stdout_changed is "
            "always False). No GPU launch means no GPU memory/execution "
            "contention between workers, so --workers can go much higher for "
            "a fast prescreen pass; see --flip-ids-file to restrict a "
            "follow-up full run to just the codegen-changed candidates."
        ),
    )
    p_flip.add_argument(
        "--replay-signatures",
        action="store_true",
        help=(
            "Requires --compile-only. The baseline run records what each "
            "kernel was called with (a one-time real run); every flip then "
            "replays those calls directly instead of running the script at "
            "all -- no torch tensor allocation, no CUDA touch beyond what "
            "compilation itself needs. Requires each --script to gate its "
            'kernel-launching code behind `if __name__ == "__main__":` '
            "(checked during the baseline run; fails fast with a clear "
            "message otherwise)."
        ),
    )
    p_flip.add_argument(
        "--compile-replay",
        action="store_true",
        help=(
            "Mutually exclusive with --compile-only / --replay-signatures -- "
            "a different execution path, not a further refinement of them. "
            "--script arguments are capture blob paths written by "
            "karnage.compile_capture.capture_all_compiles (e.g. one per "
            "workload under compile_specializations/), not real scripts. "
            "Each flip replays every captured triton.compile() call directly "
            "(karnage.compile_replay), with zero GPU/CUDA access at all -- "
            "GDB's bit-flip patch is exercised purely by the compiler. See "
            "karnage/tests/test_compile_equivalence.py for the proof this is "
            "the same compilation a real run would have done."
        ),
    )
    p_flip.add_argument(
        "--filter-by-ptx",
        action="store_true",
        help="Only test instructions whose mnemonic root appears in the baseline PTX",
    )
    p_flip.add_argument(
        "--flip-timeout",
        type=float,
        default=None,
        metavar="SECS",
        help="Per-flip GDB process timeout in seconds (default: no limit)",
    )
    p_flip.add_argument(
        "--cooldown-every",
        type=int,
        default=0,
        metavar="N",
        help="Pause for --cooldown-secs after every N flips (0 = disabled)",
    )
    p_flip.add_argument(
        "--cooldown-secs",
        type=float,
        default=30.0,
        metavar="S",
        help="Seconds to sleep during each cooldown pause (default: 30)",
    )
    p_flip.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of concurrent GDB flip processes (default: 1). "
            "Each worker gets its own output directory and Triton cache. "
            "GPU memory is the practical limit: ~200-600 MB per worker. "
            "With --compile-only, no kernel ever launches, so this limit "
            "doesn't apply the same way and --workers can go much higher "
            "(bounded instead by CPU cores/host RAM). "
            "Cooldown is applied between batches of --cooldown-every flips."
        ),
    )
    p_flip.set_defaults(func=_cmd_flip)

    # -- report --
    p_report = sub.add_parser(
        "report",
        help="Reconstruct a JSON report from an existing flip output directory.",
        description=(
            "Read spec.json from every flip_NNNNNN/ subdirectory, re-compare "
            "app_stdout.txt and PTX against the baseline, and write a JSON report. "
            "Use this when --report was not passed to 'karnage flip'."
        ),
    )
    p_report.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        metavar="DIR",
        help=f"Root output directory produced by 'karnage flip' (default: {DEFAULT_OUTPUT_DIR})",
    )
    p_report.add_argument(
        "--report",
        type=Path,
        required=True,
        metavar="DIR",
        help=(
            "Directory to write reconstructed reports into.  "
            "One ``{script_stem}.json`` is written per script detected in the output dir."
        ),
    )
    p_report.set_defaults(func=_cmd_report)

    # -- perf --
    p_perf = sub.add_parser(
        "perf",
        help="Measure the performance impact of codegen-changed, non-crashed flips using ncu.",
        description=(
            "Filters a flip report down to codegen-changed, non-crashed flips "
            "(ignores the input report's stdout_changed, since it's never "
            "meaningful from a --compile-only prescreen), then profiles each "
            "one with ncu against a clean baseline to find flips that silently "
            "degrade kernel performance. Since profiling is itself a real "
            "execution, it also measures the real stdout_changed as a side "
            "effect and records it in the output report -- no separate full "
            "run is needed first just to get that signal. Patches the target "
            "library on disk (no GDB) since ncu cannot profile a process "
            "already ptraced by GDB; runs strictly single-threaded since it "
            "mutates the real library file in place."
        ),
    )
    p_perf.add_argument(
        "--report",
        type=Path,
        required=True,
        metavar="PATH",
        help=(
            "Per-script report JSON from 'flip' or 'report' to filter for "
            "codegen-changed, non-crashed flips"
        ),
    )
    p_perf.add_argument(
        "--script",
        type=Path,
        required=True,
        metavar="PATH",
        help="Triton application script to profile (must launch --kernel-name)",
    )
    p_perf.add_argument(
        "--library",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to the real, loaded shared library — patched on disk in place",
    )
    p_perf.add_argument(
        "--sites",
        type=Path,
        default=Path(DEFAULT_FLIP_SITES),
        metavar="PATH",
        help=f"flip_sites.json from scan (default: {DEFAULT_FLIP_SITES})",
    )
    p_perf.add_argument(
        "--kernel-name",
        required=True,
        metavar="NAME",
        help="ncu -k filter identifying the kernel to time (exact name or 'regex:...')",
    )
    p_perf.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="DIR",
        help=(
            "Root output directory: golden library backup, per-run Triton "
            "caches, ncu logs, and perf_report.json"
        ),
    )
    p_perf.add_argument(
        "--primary-metric",
        default="Duration",
        metavar="METRIC",
        help=(
            "ncu metric (display name, as it appears in the ncu CSV output) "
            "driving the regression decision, e.g. 'Duration', "
            "'Compute (SM) Throughput' (default: Duration). All collected "
            "metrics are still recorded regardless of this choice --- with "
            "the default 'basic' set (see --ncu-metrics) that's the full "
            "~200-metric set; with --ncu-metrics it's only what you asked "
            "for, so this must name one of those."
        ),
    )
    p_perf.add_argument(
        "--ncu-metrics",
        default=None,
        metavar="METRIC1,METRIC2,...",
        help=(
            "Comma-separated raw ncu metric names (ncu --metrics) to collect "
            "instead of the 'basic' named set (ncu --set basic). 'basic' is "
            "ncu's smallest predefined set but still ~200 metrics, most "
            "needing their own kernel replay pass --- empirically tens of "
            "minutes for one kernel on one run. Naming metrics explicitly "
            "collects only those, e.g. --ncu-metrics gpu__time_duration.sum "
            "for just timing (one replay pass). When set, --primary-metric "
            "must match one of these metrics' ncu CSV display name, not "
            "'Duration' by default assumption."
        ),
    )
    p_perf.add_argument(
        "--repeats",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Full process re-invocations per condition (baseline and each "
            "flip); medians are taken across all launches from all repeats "
            "(default: 5)"
        ),
    )
    p_perf.add_argument(
        "--launch-skip",
        type=int,
        default=0,
        metavar="N",
        help="Kernel launches to skip before profiling starts (skip warmup iterations)",
    )
    p_perf.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        metavar="PCT",
        help="Percent slowdown vs. baseline median to flag as regressed (default: 5.0)",
    )
    p_perf.add_argument(
        "--max-sites",
        type=int,
        default=None,
        metavar="N",
        help="Profile at most N candidate flips (useful for a quick sanity check)",
    )
    p_perf.add_argument(
        "--run-timeout",
        type=float,
        default=None,
        metavar="SECS",
        help="Per-ncu-invocation timeout in seconds (default: no limit)",
    )
    p_perf.set_defaults(func=_cmd_perf)

    # -- capture --
    p_capture = sub.add_parser(
        "capture",
        help="Run a workload for real and capture every triton.compile() call for GPU-free replay.",
        description=(
            "Runs --script for real (one real GPU launch) and records the "
            "exact (signature, constants, attrs, target, options) tuple its "
            "normal JIT path passes to triton.compile() for every kernel "
            "specialization it exercises, plus the resulting compiled "
            "artifacts (ttir/ttgir/llir/ptx/cubin). Requires --script to "
            "gate its kernel-launching code behind "
            "`if __name__ == \"__main__\":` so it can later be re-imported "
            "as a module (not run) during GPU-free replay --- see "
            "karnage/compile_replay.py. Output feeds 'flip --compile-replay'."
        ),
    )
    p_capture.add_argument(
        "--script",
        type=Path,
        required=True,
        metavar="PATH",
        help="Triton workload script to run for real and capture compiles from",
    )
    p_capture.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="PATH",
        help=(
            "Where to write the captured specialization blob (pickle), e.g. "
            "compile_specializations/attention.pkl"
        ),
    )
    p_capture.set_defaults(func=_cmd_capture)

    return ap


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate subcommand handler."""
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
