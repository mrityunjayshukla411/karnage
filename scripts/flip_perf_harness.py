#!/usr/bin/env python3
"""flip_perf_harness.py --- vLLM serving perf-degradation harness for Karnage flip sites.

Measures end-to-end vLLM serving degradation caused by known flip-sites, applied to
libtriton.so, vs. an unflipped baseline. Built on top of scripts/vllm_bench.sh's model
list / engine flags / random-dataset flags, but drives `vllm serve` + `vllm bench serve`
(not `vllm bench throughput`/`latency`) since only `bench serve` reports TTFT/TPOT/ITL.

Flip mechanism: on-disk byte patch (karnage/perf/runner.py's approach), not GDB ptrace.
GDB's new_objfile hook only fires in the process it launches directly; vLLM's engine
commonly loads/JITs libtriton.so in a spawned engine-core subprocess that GDB would not
be tracing (follow-fork-mode defaults to "parent"), and this repo's own perf module
already established that GDB ptrace doesn't compose with real-execution measurement.
Patching the library on disk before `vllm serve` starts sidesteps both problems and is
the same mechanism already validated against a real vLLM workload in
replay_flagged_flips.py.

Verification: after writing the patched byte, the byte is read back and compared to the
expected value (flip_verified). The whole library's MD5 is checked against a golden
backup before every single rep (baseline or flipped) as a global integrity guard, and
again after each flipped rep's restore. A rep whose flip failed to verify is aborted
before the server is even started, and is recorded with valid=False rather than dropped.

Correctness gate --- noise-floor test, not exact match: vLLM's continuous batching
means batch composition (which in-flight requests get decoded together at each step)
isn't perfectly reproducible across separate process launches, and GPU floating-point
reduction (attention softmax, layernorm, matmul accumulation) is not associative, so
batch-composition differences produce ULP-level logit differences between otherwise
identical runs. This is invisible on confident predictions, but the `random` dataset
feeds nonsense/out-of-distribution token IDs as prompts, so the model's next-token
logits are often near-tied at the argmax -- a ULP-level perturbation is enough to flip
the greedy-decoded token, and the divergence compounds autoregressively for the rest of
that request. Measured directly: unflipped-vs-unflipped comparisons diverge on ~38% of
requests at temperature=0 with an identical fixed seed. An exact-match gate is therefore
unusable (it fails baseline reps at the same rate as flipped ones). Instead, multiple
unflipped golden references are captured per (model, num_prompts); the natural
divergence rate among them (leave-one-out) establishes a noise floor, and a rep is
flagged incorrect only if its divergence from the golden set is both statistically
significant AND large in absolute terms (see `correctness_gate()`), which distinguishes
real corruption from ordinary batching noise.

Resumability: raw_results.csv is the source of truth. Completed (model, flip_site,
condition, num_prompts, rep_idx) rows are loaded at startup and skipped; the loop order
is deterministic (model -> scale -> flip_site -> condition-block -> rep), so a rerun
resumes in the same order it would have followed uninterrupted.

Condition order is blocked, not interleaved: all N baseline reps for a (model,
flip_site, num_prompts) cell run first, then all N flipped reps. Each rep still gets its
own fresh `vllm serve` launch, unique Triton cache, and post-rep cooldown, so nothing is
shared across reps within a block.

Multi-scale sweep: num_prompts is a swept factor (e.g. 10/100/1000 via --scale-spec),
not a single fixed value, so the summary can show whether a flip's serving-level
degradation grows with load. Reps and golden-reference counts are configurable per
scale, since a full grid at every scale with the same rep count is prohibitively
expensive (bench-serve time scales with num_prompts under whatever concurrency
vLLM's auto-derived --max-num-seqs allows) -- see --scale-spec.

--retrofit-correctness: recomputes correctness_pass/valid for an EXISTING (old-schema)
--output-dir in place, using bench_result.json files already on disk from a prior run
under the old exact-match gate. Adds golden references up to --golden-refs and rewrites
raw_results.csv under the current schema, without re-running any vllm serve/bench serve
for reps that already completed -- see run_retrofit().
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from karnage.flipper.runner import iter_patch_specs  # noqa: E402
from karnage.utils.models import PatchSpec  # noqa: E402
from karnage.utils.parser import linker_vma_to_file_offset  # noqa: E402

# ---------------------------------------------------------------------------
# Config mirrored from scripts/vllm_bench.sh
# ---------------------------------------------------------------------------

MODELS: dict[str, str] = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma": "google/gemma-2-9b-it",
    "deepseek": "deepseek-ai/deepseek-llm-7b-chat",
}

# vllm serve flags. --max-num-seqs is deliberately NOT set -- letting vLLM derive
# it from the model config and available memory never was the constraint (see
# below); every one of the 5 target models started fine with it auto-derived.
#
# --max-model-len and --gpu-memory-utilization are CLI overrides (see
# build_parser()), not baked in here, because the right value is memory-
# constrained per model rather than a single constant that fits all five:
# gemma-2-9b's bf16 weights (~18GB) leave only ~1.3GB of this GPU's ~19.3GB
# budget (24GB total * the 0.8 default utilization) for KV cache, and
# empirically (real vllm serve launches, not guessed) that's only enough for
# a max_model_len around 1500-4000 depending on run-to-run engine overhead
# variance -- one measured run needed max_model_len <=~4032 to fit, another
# only <=~1520. The default --max-model-len 4000 matches the original
# measurement's margin and works for qwen/llama/mistral/deepseek (whose
# native contexts are all much larger, so 4000 is a deliberate shared cap for
# consistency, not a memory need -- see below); gemma needs a lower
# --max-model-len override when it doesn't fit. ShareGPT samples are already
# filtered by vLLM's is_valid_sequence() defaults to prompt+output <= 2048
# tokens (see DATASET_FLAGS below), so lowering --max-model-len to 2048
# doesn't drop or truncate any request that would otherwise have been sent --
# it only shrinks the KV-cache reservation for a length nothing ever reaches.
# Going lower than 2048 would start rejecting real samples at the engine
# (context-length-exceeded), which desyncs correctness_gate()'s positional
# comparison against the golden set -- so 2048 is the practical floor.
COMMON_ENGINE_FLAGS: list[str] = [
    "--attention-backend", "TRITON_ATTN",
    "--dtype", "bfloat16",
    "--enforce-eager",
]

# ShareGPT (real user/assistant conversations, not synthetic token sequences) via
# vLLM's HF-hosted loader -- see ConversationDataset in
# vllm/benchmarks/datasets/datasets.py. Chosen over the `random` dataset because
# random's prompts are a literal arithmetic sequence of token IDs mod vocab_size
# (see that module's RandomDataset docstring), not real language -- which pushes
# the model into an out-of-distribution regime with frequent near-tied next-token
# logits, making it far more sensitive to ordinary GPU batching noise than real
# traffic would be (see this harness's earlier noise-floor investigation). A real
# "serving degradation" claim needs realistic input. Chosen over MT-Bench
# (philschmid/mt-bench) because MT-Bench has only ~80 unique questions -- at
# num_prompts=1000 each would repeat ~12x, whereas ShareGPT's corpus is large
# enough that even n=1000 stays mostly unique.
#
# --dataset-path (not --hf-name) is what actually gets passed to HF's
# load_dataset() -- ConversationDataset's routing only checks hf_name to SELECT
# the dataset class, dataset_path is what's loaded, so --hf-name alone silently
# passes dataset_path=None to load_dataset() and crashes. --hf-split train is
# required too: --hf-split defaults to None, and load_dataset(..., split=None)
# returns a DatasetDict (keyed by split name), so iterating it without a split
# yields split-name strings ("train") instead of conversation rows. Both
# confirmed empirically, not assumed -- see conversation history for the actual
# tracebacks. ConversationDataset.sample() also verified to already filter out
# any conversation whose prompt exceeds 1024 tokens or prompt+output exceeds
# 2048 (vLLM's own is_valid_sequence() defaults), well under our max-model-len
# 4000, so no request here can ever exceed the context window.
DATASET_FLAGS: list[str] = [
    "--dataset-name", "hf",
    "--dataset-path", "Aeala/ShareGPT_Vicuna_unfiltered",
    "--hf-split", "train",
]

CSV_FIELDS = [
    "model", "flip_site", "condition", "num_prompts", "rep_idx",
    "throughput_req_s", "throughput_tok_s", "ttft_ms", "tpot_ms", "itl_ms",
    "correctness_matches", "correctness_match_rate", "correctness_noise_floor_rate",
    "correctness_p_value", "correctness_effect_size", "correctness_n_golden_refs",
    "correctness_pass", "flip_verified", "server_ready", "valid",
    "error", "rep_dir", "rep_start_ts", "duration_s",
]

_DEFAULT_LIBRARY = _REPO_ROOT / "../.envs/vllm/lib/python3.12/site-packages/triton/_C/libtriton.so"
_DEFAULT_FLIP_SITES = _REPO_ROOT / "flip_sites.json"

_ABORT = False


@dataclass(frozen=True)
class Scale:
    """One (num_prompts, reps, golden_refs) cell of a multi-scale sweep."""
    num_prompts: int
    reps: int
    golden_refs: int


def _handle_sigterm(signum, frame):
    # Turns SIGTERM into a normal Python exception path so any active
    # patched_byte_verified() context manager's `finally` still runs and
    # restores the library --- mirrors perf/runner.py's documented safety
    # envelope (restored on ordinary exceptions/KeyboardInterrupt, not SIGKILL).
    raise SystemExit(128 + signum)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def wait_gpu_idle(cooldown_secs: float) -> None:
    """Sleep, then give a fresh Triton cache dir per rep does the rest.

    rocm-smi VRAM readback is best-effort diagnostics only (logged, never
    blocks) --- there is no reliable universal "GPU is idle" signal across
    ROCm versions, and this environment is AMD (rocm-smi present, no
    nvidia-smi), so no ncu/nvidia-smi-based reset is attempted here.
    """
    time.sleep(cooldown_secs)
    try:
        out = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        data = json.loads(out)
        for _, card in data.items():
            used = int(card.get("VRAM Total Used Memory (B)", 0))
            print(f"[gpu] VRAM used after cooldown: {used / 1e6:.0f} MB", file=sys.stderr)
    except Exception:
        pass  # best-effort only


# ---------------------------------------------------------------------------
# On-disk byte patch with read-back + MD5 verification
# ---------------------------------------------------------------------------


@contextmanager
def patched_byte_verified(
    library: Path, offset: int, mask: int, golden_backup: Path
) -> Iterator[bool]:
    """Patch one byte of *library* on disk, verifying the write stuck.

    Yields ``True`` if the read-back after writing matches the expected
    post-XOR value, ``False`` otherwise. Callers must not proceed to run
    anything under the patch when this yields ``False`` --- the restore in
    ``finally`` still runs either way, so the library is never left patched
    on the way out of this context manager.
    """
    with library.open("r+b") as f:
        f.seek(offset)
        raw = f.read(1)
        if not raw:
            raise RuntimeError(f"offset 0x{offset:x} is past the end of {library}")
        original = raw[0]
        expected = original ^ mask
        f.seek(offset)
        f.write(bytes([expected]))
        f.flush()
        os.fsync(f.fileno())

    with library.open("rb") as f:
        f.seek(offset)
        readback = f.read(1)[0]
    verified = readback == expected

    try:
        yield verified
    finally:
        with library.open("r+b") as f:
            f.seek(offset)
            f.write(bytes([original]))
            f.flush()
            os.fsync(f.fileno())
        with library.open("rb") as f:
            f.seek(offset)
            restored = f.read(1)[0]
        if restored != original:
            print(
                f"CRITICAL: failed to restore byte at offset 0x{offset:x} in "
                f"{library}! expected 0x{original:02x} got 0x{restored:02x}. "
                f"Restore manually: cp {golden_backup} {library}",
                file=sys.stderr,
            )


def assert_library_clean(library: Path, golden_md5: str, golden_backup: Path) -> None:
    """Abort the whole harness loudly if the library isn't in its golden state.

    Run before every single rep (baseline or flipped) --- catches a prior
    rep's restore having silently failed, before it can silently corrupt
    every subsequent measurement.
    """
    cur = md5sum(library)
    if cur != golden_md5:
        sys.exit(
            f"ABORT: {library} does not match golden backup (md5 {cur} != "
            f"{golden_md5}). Library is in an unknown/patched state --- refusing "
            f"to run further reps. Restore manually with: cp {golden_backup} {library}"
        )


# ---------------------------------------------------------------------------
# vLLM server + bench serve lifecycle
# ---------------------------------------------------------------------------


def launch_vllm_serve(
    model: str, port: int, triton_cache_dir: Path, log_path: Path, args: argparse.Namespace
) -> subprocess.Popen:
    env = {**os.environ}
    env["VLLM_LOGGING_LEVEL"] = "ERROR"
    # Forces Triton to recompile from whatever is currently on disk in
    # libtriton.so (patched or not) rather than reusing a kernel cached from
    # a previous rep's different binary state --- same env-var contract
    # karnage/flipper/runner.py and karnage/perf/runner.py already use.
    env["TRITON_CACHE_DIR"] = str(triton_cache_dir)
    env["TRITON_ALWAYS_COMPILE"] = "1"

    cmd = [
        "vllm", "serve", model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--seed", str(args.seed),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        *COMMON_ENGINE_FLAGS,
    ]
    log_f = open(log_path, "w")
    return subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, start_new_session=True,
    )


def stop_process(proc: subprocess.Popen, grace: float = 20.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def run_bench_serve(
    model: str, port: int, rep_dir: Path, args: argparse.Namespace, seed: int, num_prompts: int
) -> tuple[int, dict | None]:
    result_path = rep_dir / "bench_result.json"
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "openai",
        "--base-url", f"http://127.0.0.1:{port}",
        "--model", model,
        "--seed", str(seed),
        "--num-prompts", str(num_prompts),
        "--temperature", "0",
        "--ready-check-timeout-sec", str(int(args.server_startup_timeout)),
        "--save-result", "--save-detailed",
        "--result-dir", str(rep_dir),
        "--result-filename", result_path.name,
        *DATASET_FLAGS,
    ]
    if args.max_concurrency is not None:
        # Caps in-flight requests so num_prompts are sent (mostly) sequentially
        # instead of vllm bench serve's default of firing all of them at once
        # (request_rate=inf, max_concurrency=None) and letting the server's
        # continuous batching mix them together -- lets the same flip_id be
        # measured under single-request-at-a-time conditions vs. the harness's
        # normal heavily-batched conditions. Use a separate --output-dir per
        # --max-concurrency value: raw_results.csv's resumability key doesn't
        # include concurrency, so mixing values into one output dir would
        # either silently skip reps as "already completed" or blend rows from
        # different concurrency regimes under the same key.
        cmd += ["--max-concurrency", str(args.max_concurrency)]
    log_path = rep_dir / "bench_serve.log"
    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            timeout=args.bench_timeout, start_new_session=True,
        )
    if proc.returncode != 0 or not result_path.exists():
        return proc.returncode, None
    return 0, json.loads(result_path.read_text())


# ---------------------------------------------------------------------------
# Golden (correctness reference) generation --- multiple refs per (model, num_prompts)
# ---------------------------------------------------------------------------


def _text_similar(a: str, b: str, threshold: float) -> bool:
    """Fuzzy text match: identical, or difflib similarity ratio >= threshold.

    Lexical (character-sequence) similarity, not semantic -- no embeddings or
    LLM judging. This is deliberately the tolerant middle ground between exact
    match (rejects ordinary GPU batching noise -- see module docstring) and a
    real semantic-equivalence checker (out of scope): a small wording change or
    minor rephrasing scores high and passes, while a genuinely different
    continuation (e.g. a flip corrupting output, or a batching-noise divergence
    that cascades into a different sentence) scores low and correctly fails.
    """
    if a == b:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def compute_noise_floor(
    golden_texts_sets: list[list[str]], similarity_threshold: float
) -> tuple[int, int]:
    """Leave-one-out natural divergence rate among G unflipped golden references.

    For each ref held out in turn, a request "matches" if its text is fuzzy-similar
    (see _text_similar) to ANY of the other G-1 refs' text at that index. Returns
    (L, T) -- matches and total trials (T = G * num_prompts) -- rather than a bare
    rate, so callers can feed both into a proportions test alongside a rep's own
    (k, M) counts.
    """
    g = len(golden_texts_sets)
    m = len(golden_texts_sets[0])
    matches = 0
    for j in range(g):
        others = golden_texts_sets[:j] + golden_texts_sets[j + 1:]
        for i in range(m):
            if any(_text_similar(golden_texts_sets[j][i], ref[i], similarity_threshold) for ref in others):
                matches += 1
    return matches, g * m


def correctness_gate(
    rep_texts: list[str], golden_texts_sets: list[list[str]],
    noise_floor_matches: int, noise_floor_trials: int,
    alpha: float, min_effect: float, similarity_threshold: float,
) -> dict:
    """Noise-floor correctness test for one rep's generated_texts.

    A request matches if it is fuzzy-similar (see _text_similar) to ANY of the G
    golden references at that index -- tolerant of both ordinary GPU batching/
    FP-associativity noise (see module docstring) and small wording deviations
    that don't change the substance of the answer, without requiring a real
    semantic-equivalence checker. The rep's match rate is compared against the
    noise floor (leave-one-out divergence rate among the golden refs themselves)
    via a one-sided Fisher exact test. Flagging requires BOTH statistical
    significance AND a minimum absolute effect size, since a fixed alpha alone
    would behave very differently across num_prompts spanning two orders of
    magnitude -- at large num_prompts even a trivial rate difference becomes
    "significant" from sample size alone.
    """
    from scipy.stats import fisher_exact

    m = len(rep_texts)
    k = sum(
        1 for i in range(m)
        if any(_text_similar(rep_texts[i], ref[i], similarity_threshold) for ref in golden_texts_sets)
    )
    match_rate = k / m if m else 0.0
    noise_floor_rate = noise_floor_matches / noise_floor_trials if noise_floor_trials else 0.0

    table = [[k, m - k], [noise_floor_matches, noise_floor_trials - noise_floor_matches]]
    try:
        _, p_value = fisher_exact(table, alternative="less")
    except Exception:
        p_value = float("nan")

    effect_size = noise_floor_rate - match_rate
    correctness_pass = not (p_value < alpha and effect_size >= min_effect)
    return {
        "correctness_matches": k,
        "correctness_match_rate": match_rate,
        "correctness_noise_floor_rate": noise_floor_rate,
        "correctness_p_value": p_value,
        "correctness_effect_size": effect_size,
        "correctness_n_golden_refs": len(golden_texts_sets),
        "correctness_pass": correctness_pass,
    }


def ensure_golden_set(
    model_name: str, model_id: str, num_prompts: int, golden_refs: int,
    args: argparse.Namespace, output_dir: Path,
    library: Path, golden_md5: str, golden_backup: Path,
) -> tuple[list[list[str]], int, int]:
    """Ensure >= golden_refs unflipped reference generations exist for (model, num_prompts).

    Resumable: globs existing ref_NN/bench_result.json files and only generates
    the shortfall, numbered continuing from the highest existing index (so a
    process killed mid-generation picks up where it left off rather than
    overwriting). Recomputes and caches noise_floor.json from every ref found on
    disk (not just the requested count), so a partially-complete set never
    silently trusts a stale cached rate.
    """
    scale_dir = output_dir / "golden" / model_name / str(num_prompts)
    scale_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(scale_dir.glob("ref_*/bench_result.json"))
    texts_sets: list[list[str]] = [json.loads(p.read_text())["generated_texts"] for p in existing]
    next_idx = len(existing)

    while len(texts_sets) < golden_refs:
        ref_dir = scale_dir / f"ref_{next_idx:02d}"
        ref_dir.mkdir(parents=True, exist_ok=True)
        print(f"[golden] {model_name} n={num_prompts} generating ref {next_idx}/{golden_refs}")
        assert_library_clean(library, golden_md5, golden_backup)
        port = find_free_port()
        server = launch_vllm_serve(model_id, port, ref_dir / "triton_cache", ref_dir / "server.log", args)
        try:
            rc, result = run_bench_serve(model_id, port, ref_dir, args, seed=args.seed, num_prompts=num_prompts)
        finally:
            stop_process(server)
            wait_gpu_idle(args.cooldown_secs)
        if rc != 0 or result is None:
            sys.exit(
                f"ABORT: failed to generate golden ref {next_idx} for {model_name} "
                f"n={num_prompts} -- see {ref_dir}"
            )
        texts_sets.append(result["generated_texts"])
        next_idx += 1

    matches, trials = compute_noise_floor(texts_sets, args.correctness_similarity_threshold)
    rate = matches / trials if trials else 0.0
    (scale_dir / "noise_floor.json").write_text(json.dumps({
        "num_prompts": num_prompts, "n_golden_refs": len(texts_sets),
        "loo_trials": trials, "loo_matches": matches, "noise_floor_rate": rate,
        "similarity_threshold": args.correctness_similarity_threshold,
        "computed_at": now_iso(),
    }, indent=2))
    print(f"[golden] {model_name} n={num_prompts}: {len(texts_sets)} refs, noise_floor_rate={rate:.3f}")
    return texts_sets, matches, trials


# ---------------------------------------------------------------------------
# CSV resumability
# ---------------------------------------------------------------------------


def load_completed_keys(csv_path: Path) -> set[tuple[str, int, str, int, int]]:
    if not csv_path.exists():
        return set()
    completed = set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "num_prompts" not in reader.fieldnames:
            sys.exit(
                f"{csv_path} is an old-schema file (no num_prompts column) -- "
                f"run with --retrofit-correctness first, or use a fresh --output-dir"
            )
        for row in reader:
            completed.add((
                row["model"], int(row["flip_site"]), row["condition"],
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
# One rep
# ---------------------------------------------------------------------------


def run_rep(
    model_name: str, model_id: str, flip_id: int, spec: PatchSpec, offset: int,
    condition: str, num_prompts: int, rep_idx: int, args: argparse.Namespace, output_dir: Path,
    library: Path, golden_texts_sets: list[list[str]], noise_floor_matches: int, noise_floor_trials: int,
    golden_md5: str, golden_backup: Path,
) -> dict:
    rep_dir = (
        output_dir / "runs" / model_name / f"n{num_prompts}" / str(flip_id)
        / f"{condition}_{rep_idx:02d}"
    )
    rep_dir.mkdir(parents=True, exist_ok=True)
    rep_start = time.time()
    row = {
        "model": model_name, "flip_site": flip_id, "condition": condition,
        "num_prompts": num_prompts, "rep_idx": rep_idx,
        "throughput_req_s": "", "throughput_tok_s": "", "ttft_ms": "", "tpot_ms": "", "itl_ms": "",
        "correctness_matches": "", "correctness_match_rate": "", "correctness_noise_floor_rate": "",
        "correctness_p_value": "", "correctness_effect_size": "",
        "correctness_n_golden_refs": len(golden_texts_sets),
        "correctness_pass": "", "flip_verified": "", "server_ready": False, "valid": False,
        "error": "", "rep_dir": str(rep_dir), "rep_start_ts": now_iso(), "duration_s": "",
    }

    assert_library_clean(library, golden_md5, golden_backup)

    def _serve_and_bench(flip_verified) -> None:
        row["flip_verified"] = flip_verified
        port = find_free_port()
        server = launch_vllm_serve(model_id, port, rep_dir / "triton_cache", rep_dir / "server.log", args)
        try:
            rc, result = run_bench_serve(model_id, port, rep_dir, args, seed=args.seed, num_prompts=num_prompts)
        finally:
            stop_process(server)
            wait_gpu_idle(args.cooldown_secs)

        row["server_ready"] = rc == 0 and result is not None
        if not row["server_ready"]:
            row["error"] = f"bench_serve_failed(rc={rc})"
            return
        row["throughput_req_s"] = result.get("request_throughput")
        row["throughput_tok_s"] = result.get("output_throughput")
        row["ttft_ms"] = result.get("mean_ttft_ms")
        row["tpot_ms"] = result.get("mean_tpot_ms")
        row["itl_ms"] = result.get("mean_itl_ms")
        generated = result.get("generated_texts", [])
        gate = correctness_gate(
            generated, golden_texts_sets, noise_floor_matches, noise_floor_trials,
            args.correctness_alpha, args.correctness_min_effect, args.correctness_similarity_threshold,
        )
        row.update(gate)

    try:
        if condition == "baseline":
            _serve_and_bench(flip_verified=True)
        else:
            with patched_byte_verified(library, offset, spec.flip_mask, golden_backup) as verified:
                if not verified:
                    row["flip_verified"] = False
                    row["error"] = "flip_write_verification_failed"
                else:
                    _serve_and_bench(flip_verified=True)
            assert_library_clean(library, golden_md5, golden_backup)
    except Exception as exc:  # noqa: BLE001 -- one bad rep must not abort the sweep
        row["error"] = (row["error"] + ";" if row["error"] else "") + f"{type(exc).__name__}: {exc}"

    row["duration_s"] = round(time.time() - rep_start, 1)
    row["valid"] = bool(
        row["server_ready"]
        and row["correctness_pass"] is True
        and row["flip_verified"] is not False
        and not row["error"]
    )
    return row


# ---------------------------------------------------------------------------
# Summary: mean +/- std, % degradation, paired significance test
# ---------------------------------------------------------------------------


def compute_summary(csv_path: Path, output_dir: Path) -> None:
    import numpy as np
    import pandas as pd
    from scipy import stats

    df = pd.read_csv(csv_path)
    metrics = {
        "throughput_req_s": "higher_better",
        "throughput_tok_s": "higher_better",
        "ttft_ms": "lower_better",
        "tpot_ms": "lower_better",
        "itl_ms": "lower_better",
    }

    rows = []
    for (model, flip_site, num_prompts), group in df.groupby(["model", "flip_site", "num_prompts"]):
        valid = group[group["valid"]]
        base = valid[valid["condition"] == "baseline"].set_index("rep_idx")
        flip = valid[valid["condition"] == "flipped"].set_index("rep_idx")
        paired_idx = base.index.intersection(flip.index)

        summary_row = {
            "model": model, "flip_site": flip_site, "num_prompts": num_prompts,
            "n_baseline_valid": len(base), "n_flipped_valid": len(flip), "n_paired": len(paired_idx),
        }
        for metric, direction in metrics.items():
            b = base.loc[paired_idx, metric].dropna().astype(float)
            f = flip.loc[paired_idx, metric].dropna().astype(float)
            common = b.index.intersection(f.index)
            b, f = b.loc[common], f.loc[common]
            n = len(common)
            prefix = metric
            if n == 0:
                summary_row[f"{prefix}_baseline_mean"] = None
                summary_row[f"{prefix}_flipped_mean"] = None
                summary_row[f"{prefix}_pct_change"] = None
                summary_row[f"{prefix}_degraded"] = None
                summary_row[f"{prefix}_test"] = "no_valid_pairs"
                summary_row[f"{prefix}_p_value"] = None
                continue

            b_mean, b_std = float(b.mean()), float(b.std(ddof=1)) if n > 1 else 0.0
            f_mean, f_std = float(f.mean()), float(f.std(ddof=1)) if n > 1 else 0.0
            pct_change = (f_mean - b_mean) / b_mean * 100.0 if b_mean else None
            degraded = (
                (pct_change is not None)
                and ((pct_change < 0) if direction == "higher_better" else (pct_change > 0))
            )

            test_name, p_value = None, None
            if n >= 2:
                diffs = (f - b).to_numpy()
                use_ttest = False
                if n >= 30:
                    try:
                        _, p_norm = stats.shapiro(diffs)
                        use_ttest = p_norm > 0.05
                    except Exception:
                        use_ttest = True
                test_name = "paired_t_test" if use_ttest else "wilcoxon_signed_rank"
                try:
                    if use_ttest:
                        _, p_value = stats.ttest_rel(f.to_numpy(), b.to_numpy())
                    elif np.allclose(diffs, 0):
                        p_value = 1.0
                    else:
                        _, p_value = stats.wilcoxon(f.to_numpy(), b.to_numpy())
                except Exception as exc:
                    test_name = f"{test_name}_failed"
                    p_value = None

            summary_row[f"{prefix}_baseline_mean"] = b_mean
            summary_row[f"{prefix}_baseline_std"] = b_std
            summary_row[f"{prefix}_flipped_mean"] = f_mean
            summary_row[f"{prefix}_flipped_std"] = f_std
            summary_row[f"{prefix}_pct_change"] = pct_change
            summary_row[f"{prefix}_degraded"] = degraded
            summary_row[f"{prefix}_test"] = test_name
            summary_row[f"{prefix}_p_value"] = p_value
        rows.append(summary_row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2, default=str))
    print(
        f"[summary] wrote {output_dir / 'summary.csv'} and summary.json "
        f"({len(rows)} (model, flip_site, num_prompts) rows)"
    )


def compute_scale_trend(output_dir: Path) -> None:
    """Pivot summary.csv's per-scale rows into one row per (model, flip_site).

    Answers "does degradation grow with load" at a glance -- e.g.
    throughput_tok_s_pct_change_n10/_n100/_n1000 side by side -- without
    hand-pivoting summary.csv. No-op if summary.csv doesn't exist yet or has
    no rows (e.g. nothing valid to summarize).
    """
    import pandas as pd

    summary_path = output_dir / "summary.csv"
    if not summary_path.exists():
        return
    df = pd.read_csv(summary_path)
    if df.empty:
        return

    per_scale_cols = [
        c for c in df.columns
        if any(c.endswith(suffix) for suffix in (
            "_baseline_mean", "_flipped_mean", "_pct_change", "_degraded", "_p_value", "_test",
        ))
    ]

    rows = []
    for (model, flip_site), group in df.groupby(["model", "flip_site"]):
        row = {"model": model, "flip_site": flip_site}
        for _, r in group.iterrows():
            n = int(r["num_prompts"])
            for col in per_scale_cols:
                row[f"{col}_n{n}"] = r[col]
        rows.append(row)

    trend_df = pd.DataFrame(rows)
    trend_df.to_csv(output_dir / "summary_trend.csv", index=False)
    trend_df.to_json(output_dir / "summary_trend.json", orient="records", indent=2)
    print(
        f"[summary] wrote {output_dir / 'summary_trend.csv'} "
        f"({len(rows)} (model, flip_site) rows across scales)"
    )


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def parse_target_flip_ids(args: argparse.Namespace) -> set[int]:
    ids: set[int] = set()
    if args.target_flip_ids:
        ids |= {int(x) for x in args.target_flip_ids.split(",") if x.strip()}
    if args.target_flip_ids_file:
        ids |= {int(x) for x in json.loads(args.target_flip_ids_file.read_text())}
    if not ids:
        sys.exit("Must pass --target-flip-ids and/or --target-flip-ids-file")
    return ids


def parse_scale_spec(spec: str) -> list[Scale]:
    scales = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 3:
            sys.exit(f"--scale-spec entries must be num_prompts:reps:golden_refs, got {chunk!r}")
        try:
            num_prompts, reps, golden_refs = (int(p) for p in parts)
        except ValueError:
            sys.exit(f"--scale-spec entries must be three integers, got {chunk!r}")
        if golden_refs < 3:
            sys.exit(
                f"--scale-spec: golden_refs must be >= 3 (a leave-one-out noise floor "
                f"is degenerate below that), got {golden_refs} in {chunk!r}"
            )
        scales.append(Scale(num_prompts, reps, golden_refs))
    if not scales:
        sys.exit("--scale-spec parsed to zero scales")
    return scales


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-flip-ids", type=str, default=None, metavar="ID1,ID2,...",
                     help="Comma-separated flip_ids (from flip_sites.json's iteration order) to test")
    ap.add_argument("--target-flip-ids-file", type=Path, default=None, metavar="PATH",
                     help="JSON file containing a list of flip_ids to test")
    ap.add_argument("--flip-sites-json", type=Path, default=_DEFAULT_FLIP_SITES, metavar="PATH",
                     help=f"flip_sites.json from the scan step (default: {_DEFAULT_FLIP_SITES})")
    ap.add_argument("--library", type=Path, default=_DEFAULT_LIBRARY, metavar="PATH",
                     help="Path to the real, loaded libtriton.so -- patched on disk in place")
    ap.add_argument("--models", type=str, default=None, metavar="name1,name2,...",
                     help=f"Subset of {{{','.join(MODELS)}}} to run (default: all)")
    ap.add_argument("--max-model-len", type=int, default=4000, metavar="N",
                     help="vllm serve --max-model-len (default: 4000, fits qwen/llama/mistral/"
                          "deepseek comfortably). gemma-2-9b is tight on KV-cache memory at this "
                          "value and may need a lower override (e.g. 2048) to avoid an OOM at "
                          "engine startup -- see COMMON_ENGINE_FLAGS's comment for why 2048 costs "
                          "nothing in coverage (ShareGPT samples here never exceed 2048 tokens "
                          "anyway) while going lower risks rejecting real samples mid-run.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.8, metavar="FRAC",
                     help="vllm serve --gpu-memory-utilization (default: 0.8). Raise this (e.g. "
                          "0.85) alongside a lower --max-model-len if a model still doesn't fit.")
    ap.add_argument("--max-concurrency", type=int, default=None, metavar="N",
                     help="vllm bench serve --max-concurrency (default: None, i.e. vllm bench "
                          "serve's own default of unlimited/all-at-once -- the harness's normal "
                          "heavily-batched measurement). Set to 1 to measure single-request-at-a-"
                          "time conditions instead. Use a separate --output-dir per value used -- "
                          "see run_bench_serve()'s comment for why mixing them in one dir is unsafe.")
    ap.add_argument("--reps", type=int, default=10, metavar="N",
                     help="Repetitions per condition (baseline and flipped, run as two blocks) "
                          "per (model, flip_site, num_prompts) cell -- single-scale fallback, "
                          "ignored if --scale-spec is given (default: 10)")
    ap.add_argument("--num-prompts", type=int, default=50, metavar="N",
                     help="Prompts per vllm bench serve invocation -- single-scale fallback, "
                          "ignored if --scale-spec is given (default: 50)")
    ap.add_argument("--golden-refs", type=int, default=5, metavar="G",
                     help="Golden reference count for the noise-floor correctness gate -- "
                          "single-scale fallback, ignored if --scale-spec is given (default: 5)")
    ap.add_argument("--scale-spec", type=str, default=None, metavar="N:REPS:GREFS[,...]",
                     help="Comma-separated num_prompts:reps:golden_refs triples, e.g. "
                          "'10:10:5,100:5:5,1000:3:4' -- sweeps num_prompts as a factor so the "
                          "summary can show whether degradation grows with load. Takes precedence "
                          "over --reps/--num-prompts/--golden-refs when given.")
    ap.add_argument("--correctness-alpha", type=float, default=0.01, metavar="P",
                     help="Significance threshold for the noise-floor correctness gate's one-sided "
                          "Fisher exact test (default: 0.01)")
    ap.add_argument("--correctness-min-effect", type=float, default=0.10, metavar="FRAC",
                     help="Minimum match-rate drop below the noise floor (in addition to "
                          "significance) required to flag a rep as incorrect (default: 0.10)")
    ap.add_argument("--correctness-similarity-threshold", type=float, default=0.8, metavar="RATIO",
                     help="A request 'matches' golden if difflib.SequenceMatcher ratio >= this "
                          "(1.0 = exact match only). Lexical fuzzy match, not semantic -- tolerates "
                          "small wording deviations and minor batching-noise drift without requiring "
                          "the text to be byte-identical (default: 0.8)")
    ap.add_argument("--retrofit-correctness", action="store_true",
                     help="Retrofit an existing old-schema --output-dir in place: add golden refs "
                          "up to --golden-refs, recompute correctness_pass/valid for every existing "
                          "row from its already-saved bench_result.json (no new vllm serve/bench "
                          "serve runs for reps that already completed), then exit.")
    ap.add_argument("--seed", type=int, default=42,
                     help="Fixed seed for both dataset sampling (vllm bench serve --seed) and engine "
                          "seed (vllm serve --seed) -- reused verbatim across every condition/rep/model "
                          "so the same prompt set is never resampled (default: 42)")
    ap.add_argument("--cooldown-secs", type=float, default=20.0,
                     help="Sleep between reps after the server is torn down (default: 20s)")
    ap.add_argument("--server-startup-timeout", type=float, default=600.0,
                     help="Max seconds to wait for vllm serve to become ready (default: 600s)")
    ap.add_argument("--bench-timeout", type=float, default=1800.0,
                     help="Max seconds for one vllm bench serve invocation (default: 1800s)")
    ap.add_argument("--output-dir", type=Path, required=True, metavar="DIR",
                     help="Root output directory: golden backups, per-rep run dirs, raw_results.csv, summary")
    ap.add_argument("--summarize-only", action="store_true",
                     help="Skip the sweep; just (re)compute summary.csv/json from an existing raw_results.csv")
    return ap


# ---------------------------------------------------------------------------
# Retrofit: recompute correctness/valid for an existing old-schema output dir
# ---------------------------------------------------------------------------


def run_retrofit(args: argparse.Namespace) -> None:
    """Recompute correctness_pass/valid in place for a run made under the old exact-match gate.

    All existing rows' throughput/TTFT/TPOT numbers are already valid measurements
    -- only correctness_pass (computed via exact-match against a single golden ref)
    is wrong. This reads each row's already-saved rep_dir/bench_result.json (no new
    vllm serve/bench serve invocations for reps that already completed), builds a
    proper golden_refs-sized golden set per model (migrating the existing single
    ref as ref_00 and generating the shortfall), and rewrites raw_results.csv under
    the current schema. A .csv.pre-retrofit-bak copy of the original is kept.
    """
    output_dir = args.output_dir.resolve()
    csv_path = output_dir / "raw_results.csv"
    if not csv_path.exists():
        sys.exit(f"--retrofit-correctness: no {csv_path} found")
    library = args.library.resolve()
    golden_backup = output_dir / f"{library.name}.golden"
    if not golden_backup.exists():
        sys.exit(f"--retrofit-correctness: no golden backup {golden_backup} found -- library integrity can't be verified")
    golden_md5 = md5sum(golden_backup)
    assert_library_clean(library, golden_md5, golden_backup)

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        old_fieldnames = reader.fieldnames or []
        old_rows = list(reader)
    if "num_prompts" in old_fieldnames:
        sys.exit(f"{csv_path} already has the current schema (num_prompts column present) -- nothing to retrofit")
    print(f"[retrofit] {len(old_rows)} legacy rows loaded from {csv_path}")

    by_model: dict[str, list[dict]] = {}
    for row in old_rows:
        by_model.setdefault(row["model"], []).append(row)

    for model_name, rows in by_model.items():
        if model_name not in MODELS:
            print(f"[retrofit] WARNING: unknown model {model_name!r}, skipping its {len(rows)} rows")
            continue
        model_id = MODELS[model_name]
        old_golden_path = output_dir / "golden" / model_name / "bench_result.json"
        if not old_golden_path.exists():
            print(f"[retrofit] WARNING: no legacy golden reference for {model_name}, skipping its {len(rows)} rows")
            continue
        old_golden = json.loads(old_golden_path.read_text())
        num_prompts = old_golden["completed"]

        scale_dir = output_dir / "golden" / model_name / str(num_prompts)
        ref0_dir = scale_dir / "ref_00"
        if not (ref0_dir / "bench_result.json").exists():
            ref0_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_golden_path, ref0_dir / "bench_result.json")
            print(f"[retrofit] migrated legacy golden -> {ref0_dir / 'bench_result.json'}")

        golden_texts_sets, noise_matches, noise_trials = ensure_golden_set(
            model_name, model_id, num_prompts, args.golden_refs,
            args, output_dir, library, golden_md5, golden_backup,
        )

        for row in rows:
            row["num_prompts"] = num_prompts
            bench_result_path = Path(row["rep_dir"]) / "bench_result.json"
            if not bench_result_path.exists():
                row["error"] = (row.get("error", "") + ";" if row.get("error") else "") + "retrofit_missing_bench_result"
            else:
                generated = json.loads(bench_result_path.read_text()).get("generated_texts", [])
                gate = correctness_gate(
                    generated, golden_texts_sets, noise_matches, noise_trials,
                    args.correctness_alpha, args.correctness_min_effect, args.correctness_similarity_threshold,
                )
                row.update(gate)
            row["rep_start_ts"] = row.pop("timestamp", "")
            row["duration_s"] = ""  # not recorded by the pre-retrofit harness version
            row["valid"] = bool(
                row.get("server_ready") in (True, "True")
                and row.get("correctness_pass") is True
                and row.get("flip_verified") in (True, "True")
                and not row.get("error")
            )

    backup_path = csv_path.with_suffix(".csv.pre-retrofit-bak")
    if not backup_path.exists():
        shutil.copy2(csv_path, backup_path)
        print(f"[retrofit] pre-retrofit backup saved to {backup_path}")

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in old_rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    print(f"[retrofit] rewrote {csv_path} with {len(old_rows)} rows under the current schema")

    compute_summary(csv_path, output_dir)
    compute_scale_trend(output_dir)


def main() -> None:
    args = build_parser().parse_args()
    signal.signal(signal.SIGTERM, _handle_sigterm)

    if args.golden_refs < 3:
        # Guards --golden-refs directly, independent of --scale-spec's own per-entry
        # check: with G=1 the leave-one-out noise floor has no "others" to compare
        # against and silently degenerates to a 0.0 rate, which makes the
        # correctness gate unable to ever fail a rep (see correctness_gate() ---
        # effect_size = noise_floor_rate - match_rate can't clear a positive
        # min_effect against a floor of 0). G=2 is only marginally better. This
        # applies to both the single-scale fallback path and --retrofit-correctness,
        # which both read args.golden_refs directly rather than through a Scale.
        sys.exit(
            f"--golden-refs must be >= 3 (a leave-one-out noise floor is degenerate "
            f"below that), got {args.golden_refs}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    library = args.library.resolve()
    csv_path = output_dir / "raw_results.csv"

    if args.retrofit_correctness:
        run_retrofit(args)
        return

    if args.summarize_only:
        if not csv_path.exists():
            sys.exit(f"--summarize-only: no {csv_path} found")
        compute_summary(csv_path, output_dir)
        compute_scale_trend(output_dir)
        return

    if not library.exists():
        sys.exit(f"--library not found: {library}")
    if not args.flip_sites_json.exists():
        sys.exit(f"--flip-sites-json not found: {args.flip_sites_json}")

    golden_backup = output_dir / f"{library.name}.golden"
    if not golden_backup.exists():
        shutil.copy2(library, golden_backup)
        print(f"[init] golden backup written: {golden_backup}")
    golden_md5 = md5sum(golden_backup)
    assert_library_clean(library, golden_md5, golden_backup)

    target_flip_ids = parse_target_flip_ids(args)
    sites_data = json.loads(args.flip_sites_json.read_text())
    all_specs = {s.flip_id: s for s in iter_patch_specs(sites_data)}
    missing = target_flip_ids - set(all_specs)
    if missing:
        sys.exit(f"flip_ids not found in {args.flip_sites_json}: {sorted(missing)}")

    model_names = args.models.split(",") if args.models else list(MODELS)
    unknown = set(model_names) - set(MODELS)
    if unknown:
        sys.exit(f"unknown --models entries: {sorted(unknown)} -- choices: {sorted(MODELS)}")

    scales = parse_scale_spec(args.scale_spec) if args.scale_spec else [
        Scale(args.num_prompts, args.reps, args.golden_refs)
    ]
    print(f"[init] scales: {scales}")

    completed = load_completed_keys(csv_path)
    print(f"[resume] {len(completed)} reps already completed in {csv_path}")

    for model_name in model_names:
        model_id = MODELS[model_name]

        for scale in scales:
            golden_texts_sets, noise_matches, noise_trials = ensure_golden_set(
                model_name, model_id, scale.num_prompts, scale.golden_refs,
                args, output_dir, library, golden_md5, golden_backup,
            )

            for flip_id in sorted(target_flip_ids):
                spec = all_specs[flip_id]
                offset = linker_vma_to_file_offset(library, ".text", spec.site_vma)

                for condition in ("baseline", "flipped"):
                    for rep_idx in range(scale.reps):
                        key = (model_name, flip_id, condition, scale.num_prompts, rep_idx)
                        if key in completed:
                            continue
                        print(
                            f"[rep] {model_name} n={scale.num_prompts} flip={flip_id} "
                            f"{condition} rep={rep_idx}"
                        )
                        row = run_rep(
                            model_name, model_id, flip_id, spec, offset, condition,
                            scale.num_prompts, rep_idx, args, output_dir, library,
                            golden_texts_sets, noise_matches, noise_trials,
                            golden_md5, golden_backup,
                        )
                        append_csv_row(csv_path, row)
                        completed.add(key)
                        status = "OK" if row["valid"] else f"FLAGGED({row['error'] or 'correctness/verify fail'})"
                        print(f"       -> {status}")

    compute_summary(csv_path, output_dir)
    compute_scale_trend(output_dir)


if __name__ == "__main__":
    main()
