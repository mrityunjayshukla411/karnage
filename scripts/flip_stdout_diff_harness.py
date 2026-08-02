#!/usr/bin/env python3
"""flip_stdout_diff_harness.py --- run STDOUT_DIFF-flagged flip sites through real vLLM
inference and save raw outputs for offline comparison (e.g. BERTScore).

Companion to flip_perf_harness.py, reusing its tested machinery (on-disk byte-patch
flip mechanism with read-back verification, vllm serve / vllm bench serve lifecycle,
ShareGPT dataset + engine flags) rather than duplicating it -- see that module's
docstring for why the flip is applied on disk (not GDB) and why ShareGPT (not the
`random` dataset).

Unlike flip_perf_harness.py, this script does NOT compute a correctness gate, noise
floor, or pass/fail judgment -- output-quality comparison (e.g. BERTScore) is done
offline, afterward, by the caller, directly against the raw generated_texts saved in
every rep's bench_result.json. This script's only job is: pick the right flips, run
enough repetitions of each condition, verify the flip actually applied, and get out of
the way.

Flip selection:
  --flip-report-json PATH   Select every flip_id where "stdout_changed" is true in a
                             karnage perf-report-shaped JSON (list of dicts with at
                             least "flip_id" and "stdout_changed" keys -- e.g.
                             results/perf-vllm-trtion-attn/perf_report.json). A flip
                             that is ALSO "regressed": true is not excluded -- both
                             flags can be true on the same flip_id, and that's fine.
  --flip-id / --flip-ids    Direct override: run these exact flip_id(s), bypassing the
                             report file entirely. Composes (union) with
                             --flip-report-json if both are given.

Golden baseline: run once per (model, num_prompts) -- NOT once per flip_id, since
baseline output doesn't depend on which flip is about to be tested -- before any
flipped reps for that model, and reused across every flip_id tested against that
model. --golden-reps repetitions (default 10).

Flipped reps: --flipped-reps repetitions (default 10) per (model, flip_id, num_prompts).

Output layout:
  <output-dir>/golden_baseline/<model>/n<N>/rep_NN/bench_result.json
  <output-dir>/runs/<model>/n<N>/<flip_id>/flipped/rep_NN/bench_result.json
  <output-dir>/raw_results.csv   -- one row per rep: perf metrics + flip_verified/
                                     server_ready/valid + rep_dir (for locating the
                                     full generated_texts). No correctness columns.

Resumable: raw_results.csv is the source of truth, keyed on (model, flip_id or empty
for baseline, condition, num_prompts, rep_idx) -- a rerun skips completed reps.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_THIS_DIR))

import flip_perf_harness as fph  # noqa: E402 -- reuse tested server/patch machinery
from karnage.flipper.runner import iter_patch_specs  # noqa: E402
from karnage.utils.models import PatchSpec  # noqa: E402
from karnage.utils.parser import linker_vma_to_file_offset  # noqa: E402

CSV_FIELDS = [
    "model", "flip_id", "condition", "num_prompts", "rep_idx",
    "throughput_req_s", "throughput_tok_s", "ttft_ms", "tpot_ms", "itl_ms",
    "flip_verified", "server_ready", "valid",
    "error", "rep_dir", "rep_start_ts", "duration_s",
]


def _handle_sigterm(signum, frame):
    # Same rationale as flip_perf_harness.py: let an active patched_byte_verified()
    # context manager's finally still run and restore the library.
    raise SystemExit(128 + signum)


# ---------------------------------------------------------------------------
# Flip selection
# ---------------------------------------------------------------------------


def load_stdout_diff_flip_ids(report_path: Path) -> set[int]:
    report = json.loads(report_path.read_text())
    ids = {int(r["flip_id"]) for r in report if r.get("stdout_changed")}
    if not ids:
        sys.exit(f"--flip-report-json: no entries with stdout_changed=true in {report_path}")
    return ids


def resolve_target_flip_ids(args: argparse.Namespace) -> set[int]:
    ids: set[int] = set()
    if args.flip_id is not None:
        ids.add(args.flip_id)
    if args.flip_ids:
        ids |= {int(x) for x in args.flip_ids.split(",") if x.strip()}
    if args.flip_report_json:
        ids |= load_stdout_diff_flip_ids(args.flip_report_json)
    if not ids:
        sys.exit("Must pass --flip-report-json and/or --flip-id/--flip-ids")
    return ids


# ---------------------------------------------------------------------------
# CSV resumability (schema differs from flip_perf_harness.py's -- no flip_site
# int requirement, since baseline rows have no flip_id)
# ---------------------------------------------------------------------------


def load_completed_keys(csv_path: Path) -> set[tuple[str, str, str, int, int]]:
    if not csv_path.exists():
        return set()
    completed = set()
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            completed.add((
                row["model"], row["flip_id"], row["condition"],
                int(row["num_prompts"]), int(row["rep_idx"]),
            ))
    return completed


def append_csv_row(csv_path: Path, row: dict) -> None:
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# One rep --- shared between baseline and flipped
# ---------------------------------------------------------------------------


def run_one_rep(
    model_name: str, model_id: str, flip_id_label: str, condition: str,
    num_prompts: int, rep_idx: int, rep_dir: Path,
    args: argparse.Namespace, flip_verified: bool,
) -> dict:
    """Launch vllm serve, run vllm bench serve, tear down. No correctness gate --
    just records perf metrics and whether the mechanism (server, flip) worked."""
    rep_dir.mkdir(parents=True, exist_ok=True)
    rep_start = time.time()
    row = {
        "model": model_name, "flip_id": flip_id_label, "condition": condition,
        "num_prompts": num_prompts, "rep_idx": rep_idx,
        "throughput_req_s": "", "throughput_tok_s": "", "ttft_ms": "", "tpot_ms": "", "itl_ms": "",
        "flip_verified": flip_verified, "server_ready": False, "valid": False,
        "error": "", "rep_dir": str(rep_dir), "rep_start_ts": fph.now_iso(), "duration_s": "",
    }

    port = fph.find_free_port()
    server = fph.launch_vllm_serve(model_id, port, rep_dir / "triton_cache", rep_dir / "server.log", args)
    try:
        rc, result = fph.run_bench_serve(model_id, port, rep_dir, args, seed=args.seed, num_prompts=num_prompts)
    finally:
        fph.stop_process(server)
        fph.wait_gpu_idle(args.cooldown_secs)

    row["server_ready"] = rc == 0 and result is not None
    if not row["server_ready"]:
        row["error"] = f"bench_serve_failed(rc={rc})"
    else:
        row["throughput_req_s"] = result.get("request_throughput")
        row["throughput_tok_s"] = result.get("output_throughput")
        row["ttft_ms"] = result.get("mean_ttft_ms")
        row["tpot_ms"] = result.get("mean_tpot_ms")
        row["itl_ms"] = result.get("mean_itl_ms")

    row["duration_s"] = round(time.time() - rep_start, 1)
    row["valid"] = bool(row["server_ready"] and row["flip_verified"] is not False and not row["error"])
    return row


# ---------------------------------------------------------------------------
# Baseline reps --- once per (model, num_prompts), shared across all flip_ids
# ---------------------------------------------------------------------------


def run_baseline_reps(
    model_name: str, model_id: str, num_prompts: int, golden_reps: int,
    args: argparse.Namespace, output_dir: Path, library: Path,
    golden_md5: str, golden_backup: Path,
    csv_path: Path, completed: set,
) -> None:
    base_dir = output_dir / "golden_baseline" / model_name / f"n{num_prompts}"
    for rep_idx in range(golden_reps):
        key = (model_name, "", "baseline", num_prompts, rep_idx)
        if key in completed:
            continue
        fph.assert_library_clean(library, golden_md5, golden_backup)
        rep_dir = base_dir / f"rep_{rep_idx:02d}"
        print(f"[baseline] {model_name} n={num_prompts} rep={rep_idx}")
        row = run_one_rep(
            model_name, model_id, "", "baseline", num_prompts, rep_idx, rep_dir,
            args, flip_verified=True,
        )
        append_csv_row(csv_path, row)
        completed.add(key)
        status = "OK" if row["valid"] else f"FLAGGED({row['error']})"
        print(f"          -> {status}")


# ---------------------------------------------------------------------------
# Flipped reps --- per (model, flip_id, num_prompts)
# ---------------------------------------------------------------------------


def run_flipped_reps(
    model_name: str, model_id: str, flip_id: int, spec: PatchSpec, offset: int,
    num_prompts: int, flipped_reps: int,
    args: argparse.Namespace, output_dir: Path, library: Path,
    golden_md5: str, golden_backup: Path,
    csv_path: Path, completed: set,
) -> None:
    flip_dir = output_dir / "runs" / model_name / f"n{num_prompts}" / str(flip_id) / "flipped"
    flip_id_label = str(flip_id)
    for rep_idx in range(flipped_reps):
        key = (model_name, flip_id_label, "flipped", num_prompts, rep_idx)
        if key in completed:
            continue
        fph.assert_library_clean(library, golden_md5, golden_backup)
        rep_dir = flip_dir / f"rep_{rep_idx:02d}"
        print(f"[flipped] {model_name} flip={flip_id} n={num_prompts} rep={rep_idx}")

        row = None
        with fph.patched_byte_verified(library, offset, spec.flip_mask, golden_backup) as verified:
            if not verified:
                rep_dir.mkdir(parents=True, exist_ok=True)
                row = {
                    "model": model_name, "flip_id": flip_id_label, "condition": "flipped",
                    "num_prompts": num_prompts, "rep_idx": rep_idx,
                    "throughput_req_s": "", "throughput_tok_s": "", "ttft_ms": "", "tpot_ms": "", "itl_ms": "",
                    "flip_verified": False, "server_ready": False, "valid": False,
                    "error": "flip_write_verification_failed", "rep_dir": str(rep_dir),
                    "rep_start_ts": fph.now_iso(), "duration_s": "",
                }
            else:
                row = run_one_rep(
                    model_name, model_id, flip_id_label, "flipped", num_prompts, rep_idx, rep_dir,
                    args, flip_verified=True,
                )
        fph.assert_library_clean(library, golden_md5, golden_backup)

        append_csv_row(csv_path, row)
        completed.add(key)
        status = "OK" if row["valid"] else f"FLAGGED({row['error']})"
        print(f"         -> {status}")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flip-report-json", type=Path, default=None, metavar="PATH",
                     help="karnage perf-report-shaped JSON; selects every flip_id with stdout_changed=true")
    ap.add_argument("--flip-id", type=int, default=None, metavar="ID",
                     help="Run this single flip_id directly, bypassing --flip-report-json")
    ap.add_argument("--flip-ids", type=str, default=None, metavar="ID1,ID2,...",
                     help="Run these flip_ids directly, bypassing --flip-report-json")
    ap.add_argument("--flip-sites-json", type=Path, default=fph._DEFAULT_FLIP_SITES, metavar="PATH",
                     help=f"flip_sites.json from the scan step (default: {fph._DEFAULT_FLIP_SITES})")
    ap.add_argument("--library", type=Path, default=fph._DEFAULT_LIBRARY, metavar="PATH",
                     help="Path to the real, loaded libtriton.so -- patched on disk in place")
    ap.add_argument("--models", type=str, default=None, metavar="name1,name2,...",
                     help=f"Subset of {{{','.join(fph.MODELS)}}} to run (default: all)")
    ap.add_argument("--num-prompts", type=int, required=True, metavar="N",
                     help="Batch size: prompts per vllm bench serve invocation")
    ap.add_argument("--golden-reps", type=int, default=10, metavar="N",
                     help="Baseline repetitions per (model, num_prompts), run once and reused "
                          "across every flip_id tested against that model (default: 10)")
    ap.add_argument("--flipped-reps", type=int, default=10, metavar="N",
                     help="Repetitions per (model, flip_id, num_prompts) (default: 10)")
    ap.add_argument("--max-model-len", type=int, default=4000, metavar="N",
                     help="vllm serve --max-model-len (default: 4000; lower it per-model if it "
                          "doesn't fit -- see flip_perf_harness.py's COMMON_ENGINE_FLAGS comment)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.8, metavar="FRAC",
                     help="vllm serve --gpu-memory-utilization (default: 0.8)")
    ap.add_argument("--max-concurrency", type=int, default=None, metavar="N",
                     help="Forwarded to vllm bench serve --max-concurrency (default: None, i.e. "
                          "all num_prompts fire at once under the server's normal batching)")
    ap.add_argument("--hf-output-len", type=int, default=None, metavar="N",
                     help="Forwarded to vllm bench serve --hf-output-len -- fixes output length "
                          "instead of ConversationDataset's dynamic per-conversation length "
                          "(workaround for a vLLM zero-token-completion crash on some tokenizers; "
                          "see flip_perf_harness.py's run_bench_serve comment)")
    ap.add_argument("--seed", type=int, default=42,
                     help="Fixed seed for dataset sampling and engine seed -- reused verbatim "
                          "across baseline and every flipped rep (default: 42)")
    ap.add_argument("--cooldown-secs", type=float, default=20.0,
                     help="Sleep between reps after the server is torn down (default: 20s)")
    ap.add_argument("--model-cooldown-secs", type=float, default=120.0, metavar="SECS",
                     help="Additional sleep after ALL reps (baseline + every flip_id) for one "
                          "model finish, before moving to the next model -- on top of the "
                          "per-rep --cooldown-secs, to let the GPU cool down across the bigger "
                          "gap of switching model weights entirely (default: 120s)")
    ap.add_argument("--server-startup-timeout", type=float, default=600.0,
                     help="Max seconds to wait for vllm serve to become ready (default: 600s)")
    ap.add_argument("--bench-timeout", type=float, default=1800.0,
                     help="Max seconds for one vllm bench serve invocation (default: 1800s)")
    ap.add_argument("--output-dir", type=Path, required=True, metavar="DIR",
                     help="Root output directory: golden_baseline/, runs/, raw_results.csv")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    signal.signal(signal.SIGTERM, _handle_sigterm)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    library = args.library.resolve()
    csv_path = output_dir / "raw_results.csv"

    if not library.exists():
        sys.exit(f"--library not found: {library}")
    if not args.flip_sites_json.exists():
        sys.exit(f"--flip-sites-json not found: {args.flip_sites_json}")

    golden_backup = output_dir / f"{library.name}.golden"
    if not golden_backup.exists():
        shutil.copy2(library, golden_backup)
        print(f"[init] golden backup written: {golden_backup}")
    golden_md5 = fph.md5sum(golden_backup)
    fph.assert_library_clean(library, golden_md5, golden_backup)

    target_flip_ids = resolve_target_flip_ids(args)
    sites_data = json.loads(args.flip_sites_json.read_text())
    all_specs = {s.flip_id: s for s in iter_patch_specs(sites_data)}
    missing = target_flip_ids - set(all_specs)
    if missing:
        sys.exit(f"flip_ids not found in {args.flip_sites_json}: {sorted(missing)}")
    print(f"[init] {len(target_flip_ids)} target flip_id(s): {sorted(target_flip_ids)}")

    model_names = args.models.split(",") if args.models else list(fph.MODELS)
    unknown = set(model_names) - set(fph.MODELS)
    if unknown:
        sys.exit(f"unknown --models entries: {sorted(unknown)} -- choices: {sorted(fph.MODELS)}")

    completed = load_completed_keys(csv_path)
    print(f"[resume] {len(completed)} reps already completed in {csv_path}")

    for model_idx, model_name in enumerate(model_names):
        model_id = fph.MODELS[model_name]

        run_baseline_reps(
            model_name, model_id, args.num_prompts, args.golden_reps,
            args, output_dir, library, golden_md5, golden_backup, csv_path, completed,
        )

        for flip_id in sorted(target_flip_ids):
            spec = all_specs[flip_id]
            offset = linker_vma_to_file_offset(library, ".text", spec.site_vma)
            run_flipped_reps(
                model_name, model_id, flip_id, spec, offset, args.num_prompts, args.flipped_reps,
                args, output_dir, library, golden_md5, golden_backup, csv_path, completed,
            )

        if model_idx < len(model_names) - 1 and args.model_cooldown_secs > 0:
            print(f"[cooldown] {model_name} done -- sleeping {args.model_cooldown_secs:.0f}s before next model")
            fph.wait_gpu_idle(args.model_cooldown_secs)

    print(f"[done] raw_results.csv: {csv_path}")
    print(f"       golden baseline outputs: {output_dir / 'golden_baseline'}/<model>/n{args.num_prompts}/rep_NN/bench_result.json")
    print(f"       flipped outputs:         {output_dir / 'runs'}/<model>/n{args.num_prompts}/<flip_id>/flipped/rep_NN/bench_result.json")


if __name__ == "__main__":
    main()
