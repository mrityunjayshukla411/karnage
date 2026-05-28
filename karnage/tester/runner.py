"""
runner.py — Orchestrate GDB-based MatcherTable bit-flip testing.

For each adjacent pair in adjacency.json:
  1. Compute the ELF VMAs of every opc_lo / opc_hi byte occurrence for the
     source opcode (using matcher_table.json location data + nm-derived
     MatcherTable symbol VMA).
  2. Write a patch-spec JSON file consumed by _gdb_script.py.
  3. Spawn GDB with the inferior running under _wrapper.py.
  4. Compare generated PTX and saved tensors against the baseline run.
  5. Log a one-line summary and (optionally) write a full JSON report.

VMA arithmetic:
  mt_vma  = find_symbol_linker_vma(libtriton_so, matchertable_symbol)
  vma     = mt_vma + pattern.mt_offset + byte_index
              where byte_index = 1 (opc_lo) or 2 (opc_hi)
  runtime = load_base + vma        (load_base from /proc/{pid}/maps at run time)
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

from karnage.utils.exceptions import MatcherTableLoadError
from karnage.utils.logger import logger
from karnage.utils.parser import find_symbol_linker_vma
from karnage.utils.targets import NVPTXBackend

_THIS_DIR   = Path(__file__).parent
_GDB_SCRIPT = _THIS_DIR / "_gdb_script.py"
_WRAPPER    = _THIS_DIR / "_wrapper.py"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PatchSpec:
    """One bit-flip experiment derived from an adjacency.json entry."""
    flip_id:    int
    opcode_a:   int
    mnemonic_a: str
    opcode_b:   int
    mnemonic_b: str
    flip_byte:  str            # "opc_lo" or "opc_hi"
    flip_bit:   int
    flip_mask:  int
    patch_vmas: tuple[int, ...]  # one VMA per pattern occurrence of opcode_a


@dataclass
class FlipResult:
    spec:          PatchSpec
    crashed:       bool   # GDB exited non-zero
    script_ran:    bool   # _wrapper.py completed (written _done sentinel)
    ptx_changed:   bool
    tensor_names:  list[str]
    tensors_match: dict[str, bool]   # name → torch.allclose result
    max_abs_diffs: dict[str, float]  # name → max |baseline - flip|


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    if not path.exists():
        raise MatcherTableLoadError(
            f"File not found: {path}",
            context={"path": str(path)},
        )
    with path.open() as f:
        return json.load(f)


def _build_patch_map(mt_data: dict) -> dict[int, list[int]]:
    """
    Return {opcode: [mt_offset, ...]} with one entry per pattern occurrence.

    mt_offset is the byte offset of the morph opcode byte from the start of
    the MatcherTable (identical to MatcherEntry.mt_offset).  opc_lo lives at
    mt_offset+1, opc_hi at mt_offset+2.
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
    """Yield one PatchSpec per (opcode_a, adjacent_neighbor) pair."""
    flip_id = 0
    for opcode_str, instr in adj_data["instructions"].items():
        opcode_a   = int(opcode_str)
        mt_offsets = patch_map.get(opcode_a)
        if not mt_offsets:
            logger.warning(f"Opcode {opcode_a} ({instr['mnemonic']!r}) "
                           f"not in matcher_table.json — skipping")
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
    """
    Run the Triton script, optionally under GDB with a patch spec.

    Baseline (patch_spec_path=None): runs python _wrapper.py directly.
    Flip run: runs gdb --batch with _gdb_script.py as the GDB script.

    Returns True if the inferior exited with code 0.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ}
    env["KARNAGE_OUTPUT_DIR"]    = str(output_dir / "tensors")
    env["TRITON_CACHE_DIR"]      = str(output_dir / "triton_cache")
    env["TRITON_ALWAYS_COMPILE"] = "1"

    if patch_spec_path is None:
        cmd = [sys.executable, str(_WRAPPER), str(triton_script)]
    else:
        env["KARNAGE_PATCH_SPEC"] = str(patch_spec_path)
        cmd = [
            "gdb", "--batch", "-q",
            "-x", str(_GDB_SCRIPT),
            "--args", sys.executable, str(_WRAPPER), str(triton_script),
        ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )
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
    """
    Return the contents of all .ptx files found under triton_cache/, sorted
    by file path so comparison is deterministic across runs.

    Triton stores compiled PTX in {TRITON_CACHE_DIR}/{hash}/*.ptx.  With
    TRITON_ALWAYS_COMPILE=1, fresh PTX is written on every run.  Because the
    cache key is derived from kernel source (not compilation output), baseline
    and flip runs produce the same directory structure; only file content
    differs when the MatcherTable patch changes instruction selection.
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
    r'^\s*'
    r'(?:@[!%\w]+\s+)?'                          # optional predicate
    r'([a-z][a-z0-9]*(?:\.[a-z][a-z0-9]*)*)\b'  # mnemonic (dot-separated tokens)
)
_PTX_SKIP_CHARS = frozenset({'.', '/', '$', '{', '}', '@'})


def extract_ptx_mnemonics(baseline_dir: Path) -> frozenset[str]:
    """
    Parse all baseline PTX files and return the set of instruction mnemonics
    that appear (e.g. {"fma.rn.f32", "ld.global.u32", "mov.u64"}).

    These are matched against PatchSpec.mnemonic_a to skip flips for
    instructions the test script never uses.
    """
    mnemonics: set[str] = set()
    for ptx_text in _collect_ptx(baseline_dir):
        for line in ptx_text.splitlines():
            stripped = line.strip()
            if not stripped or stripped[0] in _PTX_SKIP_CHARS or stripped.endswith(':'):
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
    """
    Return a filtered copy of *specs*.

    ptx_mnemonics    — keep only specs whose mnemonic_a appears in the
                       baseline PTX (relevance filter).
    target_mnemonics — keep only specs whose mnemonic_a is in this explicit
                       set (targeted filter).

    Both filters are applied when both are provided (intersection).
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
    import torch

    # Detect crash / incomplete run.
    # GDB --batch exits with 0 even when the inferior fails, so we cannot rely
    # on returncode.txt alone.  The wrapper writes _done only on clean exit.
    rc_file       = flip_dir / "returncode.txt"
    error_sentinel = flip_dir / "tensors" / "_error.txt"
    done_sentinel  = flip_dir / "tensors" / "_done"
    gdb_failed     = (
        not rc_file.exists()
        or int(rc_file.read_text().strip() or "1") != 0
    )
    script_ran = done_sentinel.exists() and not error_sentinel.exists()
    crashed    = gdb_failed or not script_ran

    # PTX diff: only meaningful when both runs actually generated PTX.
    # An empty flip list (script did not run) must not be reported as PTX_DIFF.
    baseline_ptx = _collect_ptx(baseline_dir)
    flip_ptx     = _collect_ptx(flip_dir)
    ptx_changed  = bool(baseline_ptx) and bool(flip_ptx) and baseline_ptx != flip_ptx

    # Tensor diff
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
                    tensors_match[name]  = bool(torch.allclose(a, b, atol=1e-5, rtol=1e-5))
                    max_abs_diffs[name]  = float((a - b).abs().max())
                except Exception as exc:
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

def run_tester(
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
    """
    Run the full bit-flip test suite.

    Steps:
      1. Resolve the MatcherTable symbol VMA via nm.
      2. Build a lookup from opcode → list[mt_offset] from matcher_table.json.
      3. Run a clean baseline (no GDB, no patch) to capture reference PTX and
         tensors.
      4. Apply filters (targeted and/or relevance) to the spec list.
      5. For each remaining PatchSpec: write a patch-spec JSON, spawn GDB,
         compare results against the baseline.
      6. Optionally write a machine-readable JSON report.
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
        logger.warning("Baseline run failed — see baseline/stderr.txt")

    # --- Filter specs ---
    ptx_mnemonics: frozenset[str] | None = None
    if filter_by_ptx:
        ptx_mnemonics = extract_ptx_mnemonics(baseline_dir)
        if not ptx_mnemonics:
            logger.warning(
                "Relevance filter requested but no PTX files found in baseline — "
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

        # One-line summary
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

        # Cooldown: give the GPU time to shed heat every N completed flips.
        flips_done = len(results)
        if cooldown_every > 0 and flips_done % cooldown_every == 0:
            logger.info(
                f"[cooldown] {flips_done} flips done — "
                f"sleeping {cooldown_secs:.0f}s to let GPU cool..."
            )
            time.sleep(cooldown_secs)

    if report_json:
        with report_json.open("w") as f:
            json.dump([_serialise_result(r) for r in results], f, indent=2)
        logger.success(f"Report written → {report_json}")

    return results
