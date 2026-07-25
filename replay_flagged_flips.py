#!/usr/bin/env python3
"""Re-profile a real workload script under only the flip sites flagged in a perf_report.json.

Takes a ``perf_report.json`` written by ``python main.py perf``, keeps only the
flips where ``regressed`` or ``stdout_changed`` is true, and re-measures each
one with ``ncu`` (or ``rocprofv3`` / wall-clock) against a *different* target
script than the one the original perf run used --- e.g. a real vLLM workload
instead of the raw Triton kernel script. Reuses
:mod:`karnage.perf.runner`'s on-disk byte-patch mechanism (GDB/ptrace can't
share a tracer with ncu, so the flip is applied by patching the real library
file directly, same as ``karnage perf``) and its ncu/rocprof/wall profiling
machinery --- this module only adds the perf_report-based flip_id filter and
passes extra CLI args through to the profiled script.

Example
-------
  python replay_flagged_flips.py \\
      --perf-report results/perf-vllm-trtion-attn/perf_report.json \\
      --library ../.envs/vllm/lib/python3.12/site-packages/triton/_C/libtriton.so \\
      --script ../vllm-multi-model.py \\
      --kernel-name kernel_unified_attention \\
      --ncu-metrics gpu__time_duration.sum --primary-metric gpu__time_duration.sum \\
      --output results/replay-flagged-vllm-multi-model/ \\
      --model llama --max-tokens 256 --query "What is the capital of France?"
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from karnage.flipper.runner import iter_patch_specs
from karnage.perf.runner import (
    _DEFAULT_ROCPROF_PATH,
    _md5,
    _medians,
    _patched_byte,
    _profile_repeated,
)
from karnage.utils.constants import DEFAULT_FLIP_SITES
from karnage.utils.models import PatchSpec
from karnage.utils.parser import linker_vma_to_file_offset

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_SCRIPT = _THIS_DIR.parent / "vllm-multi-model.py"

_MODEL_CHOICES = ("qwen", "llama", "mistral", "gemma", "deepseek")


def _load_flagged(perf_report: Path, filter_mode: str) -> list[dict]:
    results = json.loads(perf_report.read_text())
    if filter_mode == "regressed":
        keep = lambda r: r["regressed"]
    elif filter_mode == "stdout_changed":
        keep = lambda r: r["stdout_changed"]
    else:
        keep = lambda r: r["regressed"] or r["stdout_changed"]
    return [r for r in results if keep(r)]


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--perf-report", type=Path, required=True, metavar="PATH",
                     help="perf_report.json written by 'python main.py perf'")
    ap.add_argument("--sites", type=Path, default=Path(DEFAULT_FLIP_SITES), metavar="PATH",
                     help=f"flip_sites.json from the scan step (default: {DEFAULT_FLIP_SITES})")
    ap.add_argument("--library", type=Path, required=True, metavar="PATH",
                     help="Path to the real, loaded libtriton.so --- patched on disk in place")
    ap.add_argument("--script", type=Path, default=_DEFAULT_SCRIPT, metavar="PATH",
                     help=f"Workload script to profile (default: {_DEFAULT_SCRIPT})")
    ap.add_argument("--output", type=Path, required=True, metavar="DIR",
                     help="Root output directory: golden library backup, per-run Triton caches, "
                          "profiler logs, and replay_report.json")
    ap.add_argument("--filter", choices=["either", "regressed", "stdout_changed"], default="either",
                     help="Which perf_report flag(s) select a flip for re-profiling (default: either)")
    ap.add_argument("--max-flips", type=int, default=None, metavar="N",
                     help="Re-profile at most N flagged flips (debug cap)")

    # --- profiling options, mirroring 'main.py perf' ---
    ap.add_argument("--kernel-name", default=None, metavar="NAME",
                     help="ncu/rocprof kernel filter (exact name or 'regex:...'); "
                          "required unless --profiler wall")
    ap.add_argument("--profiler", choices=["ncu", "rocprof", "wall"], default="ncu",
                     help="Profiling backend (default: ncu)")
    ap.add_argument("--primary-metric", default="Duration", metavar="METRIC",
                     help="Metric driving the regression decision (default: Duration)")
    ap.add_argument("--ncu-metrics", default=None, metavar="METRIC1,METRIC2,...",
                     help="Raw ncu metrics to collect instead of the 'basic' set; ncu only")
    ap.add_argument("--repeats", type=int, default=5, metavar="N",
                     help="Full process re-invocations per condition (default: 5)")
    ap.add_argument("--launch-skip", type=int, default=0, metavar="N",
                     help="Kernel launches to skip before profiling (warmup)")
    ap.add_argument("--threshold", type=float, default=5.0, metavar="PCT",
                     help="Percent slowdown vs. baseline to flag as regressed (default: 5.0)")
    ap.add_argument("--run-timeout", type=float, default=None, metavar="SECS",
                     help="Per-profiler-invocation timeout in seconds")

    # --- passthrough args for vllm-multi-model.py ---
    ap.add_argument("--model", choices=_MODEL_CHOICES, default="qwen")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--query", type=str, default="Explain in one paragraph why the sky appears blue.")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    ap.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    return ap


def main() -> None:
    args = _build_parser().parse_args()

    for path, flag in [
        (args.perf_report, "--perf-report"),
        (args.sites, "--sites"),
        (args.library, "--library"),
        (args.script, "--script"),
    ]:
        if not path.exists():
            raise SystemExit(f"{flag}: not found: {path}")

    if args.profiler in ("rocprof", "wall") and args.ncu_metrics is not None:
        raise SystemExit(f"--ncu-metrics is not valid with --profiler {args.profiler}")
    if args.profiler in ("ncu", "rocprof") and args.kernel_name is None:
        raise SystemExit(f"--kernel-name is required with --profiler {args.profiler}")

    flagged = _load_flagged(args.perf_report, args.filter)
    print(f"{len(flagged)} flip(s) match --filter {args.filter}")
    if args.max_flips is not None:
        flagged = flagged[: args.max_flips]
    if not flagged:
        print("Nothing to replay.")
        return

    sites_data = json.loads(args.sites.read_text())
    specs_by_id: dict[int, PatchSpec] = {s.flip_id: s for s in iter_patch_specs(sites_data)}

    script_args = [
        "--model", args.model,
        "--max-tokens", str(args.max_tokens),
        "--query", args.query,
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--dtype", args.dtype,
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
    ]

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    golden = output_dir / f"{args.library.name}.golden"
    shutil.copy2(args.library, golden)
    golden_md5 = _md5(golden)
    print(f"Golden backup written to {golden} (md5={golden_md5})")

    print(f"Running baseline {args.profiler} profile ({args.repeats}x, library untouched)...")
    baseline_launches, baseline_stdout = _profile_repeated(
        args.script,
        args.kernel_name,
        output_dir / "baseline",
        repeats=args.repeats,
        launch_skip=args.launch_skip,
        timeout=args.run_timeout,
        profiler=args.profiler,
        metrics=args.ncu_metrics,
        rocprof_path=_DEFAULT_ROCPROF_PATH,
        script_args=script_args,
    )
    baseline_metrics = _medians(baseline_launches)
    if args.primary_metric not in baseline_metrics:
        raise SystemExit(
            f"primary_metric {args.primary_metric!r} not found in {args.profiler} output "
            f"--- available metrics: {sorted(baseline_metrics)}"
        )
    baseline_primary = baseline_metrics[args.primary_metric]
    print(f"baseline {args.primary_metric}: {baseline_primary:,.2f} "
          f"(n={len(baseline_launches)} launches across {args.repeats} runs)")

    replay_results: list[dict] = []
    for r in flagged:
        flip_id = r["flip_id"]
        spec = specs_by_id.get(flip_id)
        if spec is None:
            print(f"[{flip_id}] not found in {args.sites} --- skipping")
            continue

        offset = linker_vma_to_file_offset(args.library, ".text", spec.site_vma)
        flip_dir = output_dir / f"flip_{flip_id:06d}"
        short_fn = spec.func_name.split("::")[-1][:40]

        try:
            with _patched_byte(args.library, offset, spec.flip_mask):
                flip_launches, flip_stdout = _profile_repeated(
                    args.script,
                    args.kernel_name,
                    flip_dir,
                    repeats=args.repeats,
                    launch_skip=args.launch_skip,
                    timeout=args.run_timeout,
                    profiler=args.profiler,
                    metrics=args.ncu_metrics,
                    rocprof_path=_DEFAULT_ROCPROF_PATH,
                    script_args=script_args,
                )
        except Exception as exc:
            print(f"[{flip_id}] {args.profiler} failed: {exc} --- skipping")
            continue

        flip_metrics = _medians(flip_launches)
        if args.primary_metric not in flip_metrics:
            print(f"[{flip_id}] {args.profiler} produced no {args.primary_metric!r} sample --- skipping")
            continue

        flip_primary = flip_metrics[args.primary_metric]
        pct_change = (flip_primary - baseline_primary) / baseline_primary * 100.0
        regressed = pct_change >= args.threshold
        stdout_changed = flip_stdout != baseline_stdout

        flag = "REGRESSED" if regressed else "ok"
        stdout_flag = " STDOUT_DIFF" if stdout_changed else ""
        print(
            f"[{flip_id:>6}] {short_fn}  {spec.opcode_before}->{spec.opcode_after}  "
            f"{args.primary_metric}: {baseline_primary:,.2f} -> {flip_primary:,.2f} "
            f"({pct_change:+.1f}%)  {flag}{stdout_flag}  "
            f"(orig perf: regressed={r['regressed']} stdout_changed={r['stdout_changed']})"
        )

        replay_results.append(
            {
                "flip_id": flip_id,
                "func_name": spec.func_name,
                "site_vma": f"0x{spec.site_vma:016x}",
                "instr_type": spec.instr_type,
                "opcode_before": spec.opcode_before,
                "opcode_after": spec.opcode_after,
                "orig_perf_regressed": r["regressed"],
                "orig_perf_pct_change": r["pct_change"],
                "orig_perf_stdout_changed": r["stdout_changed"],
                "primary_metric": args.primary_metric,
                "baseline_metrics": baseline_metrics,
                "flip_metrics": flip_metrics,
                "pct_change": pct_change,
                "regressed": regressed,
                "stdout_changed": stdout_changed,
            }
        )

    final_md5 = _md5(args.library)
    if final_md5 != golden_md5:
        print(f"WARNING: library MD5 mismatch after replay! expected {golden_md5}, "
              f"got {final_md5} --- restore with: cp {golden} {args.library}")
    else:
        print(f"library MD5 verified unchanged: {final_md5}")

    report_path = output_dir / "replay_report.json"
    report_path.write_text(json.dumps(replay_results, indent=2))

    regressed_count = sum(1 for r in replay_results if r["regressed"])
    stdout_changed_count = sum(1 for r in replay_results if r["stdout_changed"])
    silent = [r for r in replay_results if r["regressed"] and not r["stdout_changed"]]
    print(f"{len(replay_results)} flip(s) re-profiled, {regressed_count} regressed >= "
          f"{args.threshold}%, {stdout_changed_count} changed stdout -> {report_path}")
    if silent:
        print(f"{len(silent)} SILENT regressions (slower, but identical stdout):")
        for r in sorted(silent, key=lambda r: -r["pct_change"])[:5]:
            print(f"  flip_{r['flip_id']:06d}  {r['func_name'].split('::')[-1][:40]}  {r['pct_change']:+.1f}%")


if __name__ == "__main__":
    main()
