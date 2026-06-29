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

  report  Reconstruct a JSON report from an existing output directory.
          Use this when you forgot --report during a flip run.

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

  # Step 3 - rebuild report from an existing output dir (if --report was omitted)
  python main.py report --output test_results/ --report results.json
"""

import argparse
import json
from pathlib import Path

from karnage.flipper import run_flipper
from karnage.flipper.runner import reconstruct_report
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
        flip_timeout=args.flip_timeout,
        workers=args.workers,
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

    return ap


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate subcommand handler."""
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
