#!/usr/bin/env python3
"""
prune_flip_sites.py — annotate flip_sites.json with Intel PT branch coverage
and split into covered / unobserved subsets.

For every flip site, checks whether its VMA was observed as a branch source
(from_vma) or branch target (to_vma) across the per-kernel *_branch_addrs.json
files produced by get_triton_calls.sh / trace_all_kernels.sh.

Adds per-site fields:
  "hit_kernels"   : list of kernel stems where the VMA appeared
  "taken_counts"  : {stem: count} for jcc sites where the VMA was a branch
                    source (= taken-branch count); only present when non-empty.
                    Intel PT does not record not-taken branches, so
                    not_taken_count is unavailable from PT data.

Outputs (schema identical to flip_sites.json, new fields appended to each site):
  flip_sites_covered.json    — hit_kernels non-empty
  flip_sites_unobserved.json — hit_kernels empty

Usage:
    python3 prune_flip_sites.py \\
        --sites      flip_sites.json \\
        --branch-dir <dir containing *_branch_addrs.json> \\
        --covered    flip_sites_covered.json \\
        --unobserved flip_sites_unobserved.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_branch_file(path: Path) -> tuple[dict[int, int], dict[int, int]]:
    """Return (from_vmas, to_vmas) as {static_vma_int: hit_count} dicts."""
    with path.open() as f:
        data = json.load(f)
    from_vmas = {int(k, 16): v for k, v in data.get("from_vmas", {}).items()}
    to_vmas = {int(k, 16): v for k, v in data.get("to_vmas", {}).items()}
    return from_vmas, to_vmas


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--sites",
        type=Path,
        default=Path("flip_sites.json"),
        metavar="PATH",
        help="Input flip_sites.json from scan (default: flip_sites.json)",
    )
    ap.add_argument(
        "--branch-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory (searched recursively) for *_branch_addrs.json files",
    )
    ap.add_argument(
        "--covered",
        type=Path,
        default=Path("flip_sites_covered.json"),
        metavar="PATH",
        help="Output for sites with hit_kernels non-empty (default: flip_sites_covered.json)",
    )
    ap.add_argument(
        "--unobserved",
        type=Path,
        default=Path("flip_sites_unobserved.json"),
        metavar="PATH",
        help="Output for sites with hit_kernels empty (default: flip_sites_unobserved.json)",
    )
    args = ap.parse_args()

    if not args.sites.exists():
        raise SystemExit(f"--sites: not found: {args.sites}")
    if not args.branch_dir.is_dir():
        raise SystemExit(f"--branch-dir: not a directory: {args.branch_dir}")

    # ── load per-kernel branch data ──────────────────────────────────────────
    branch_files = sorted(args.branch_dir.glob("**/*_branch_addrs.json"))
    if not branch_files:
        raise SystemExit(f"No *_branch_addrs.json files found under {args.branch_dir}")

    # stem → (from_vmas: {int: count}, to_vmas: {int: count})
    kernel_from: dict[str, dict[int, int]] = {}
    kernel_to: dict[str, dict[int, int]] = {}

    print("Loading branch address files:")
    for bf in branch_files:
        stem = bf.name.removesuffix("_branch_addrs.json")
        fv, tv = _load_branch_file(bf)
        kernel_from[stem] = fv
        kernel_to[stem] = tv
        print(f"  {stem}: {len(fv):,} from_vmas, {len(tv):,} to_vmas  ({bf})")

    stems = sorted(kernel_from.keys())
    print(f"\nKernels ({len(stems)}): {', '.join(stems)}\n")

    # Per-kernel union set (from ∪ to) for quick membership test
    kernel_any: dict[str, frozenset[int]] = {
        stem: frozenset(kernel_from[stem]) | frozenset(kernel_to[stem])
        for stem in stems
    }

    # ── load flip_sites.json ─────────────────────────────────────────────────
    with args.sites.open() as f:
        flip = json.load(f)

    functions: dict = flip["functions"]

    # ── annotate and split ───────────────────────────────────────────────────
    total_sites = 0
    covered_count = 0

    # {itype: [covered, total]}
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    # {func_name: [covered, total]}
    by_func: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    covered_functions: dict = {}
    unobserved_functions: dict = {}

    for func_name, fd in functions.items():
        covered_sites: list = []
        unobserved_sites: list = []

        for site in fd["sites"]:
            vma_int = int(site["site_vma"], 16)
            itype = site["instr_type"]
            total_sites += 1
            by_type[itype][1] += 1
            by_func[func_name][1] += 1

            hit_kernels = [s for s in stems if vma_int in kernel_any[s]]

            annotated = dict(site)
            annotated["hit_kernels"] = hit_kernels

            # For jcc types, surface taken-branch counts per kernel.
            # (Not-taken branches are not recorded by Intel PT.)
            if itype in ("short_jcc", "long_jcc"):
                taken: dict[str, int] = {
                    s: kernel_from[s][vma_int]
                    for s in hit_kernels
                    if vma_int in kernel_from[s]
                }
                if taken:
                    annotated["taken_counts"] = taken

            if hit_kernels:
                covered_count += 1
                by_type[itype][0] += 1
                by_func[func_name][0] += 1
                covered_sites.append(annotated)
            else:
                unobserved_sites.append(annotated)

        if covered_sites:
            covered_functions[func_name] = {**fd, "sites": covered_sites}
        if unobserved_sites:
            unobserved_functions[func_name] = {**fd, "sites": unobserved_sites}

    unobserved_count = total_sites - covered_count

    # ── build output dicts (same schema as flip_sites.json) ──────────────────
    base_meta = dict(flip["meta"])

    covered_out = {
        "meta": {
            **base_meta,
            "total_functions": len(covered_functions),
            "total_sites": covered_count,
            "site_counts": {t: by_type[t][0] for t in sorted(by_type)},
            "coverage_source": "Intel PT branch trace",
            "coverage_kernels": stems,
        },
        "functions": covered_functions,
    }
    unobserved_out = {
        "meta": {
            **base_meta,
            "total_functions": len(unobserved_functions),
            "total_sites": unobserved_count,
            "site_counts": {
                t: by_type[t][1] - by_type[t][0] for t in sorted(by_type)
            },
            "coverage_source": "Intel PT branch trace",
            "coverage_kernels": stems,
        },
        "functions": unobserved_functions,
    }

    with args.covered.open("w") as f:
        json.dump(covered_out, f, indent=2)
    with args.unobserved.open("w") as f:
        json.dump(unobserved_out, f, indent=2)

    # ── summary ──────────────────────────────────────────────────────────────
    pct = 100.0 * covered_count / total_sites if total_sites else 0.0
    print("=" * 64)
    print(" Coverage pruning summary")
    print("=" * 64)
    print(f"  Total sites in   : {total_sites:>10,}")
    print(f"  Covered          : {covered_count:>10,}  ({pct:.1f}%)")
    print(f"  Unobserved       : {unobserved_count:>10,}  ({100.0 - pct:.1f}%)")

    print("\n  By instruction type:")
    for itype in sorted(by_type):
        cov, tot = by_type[itype]
        p = 100.0 * cov / tot if tot else 0.0
        print(f"    {itype:<12}  {cov:>8,} / {tot:>8,}  ({p:.1f}% covered)")

    print("\n  By function (top 20 by total site count):")
    top = sorted(by_func.items(), key=lambda kv: -kv[1][1])[:20]
    for fname, (cov, tot) in top:
        p = 100.0 * cov / tot if tot else 0.0
        label = fname if len(fname) <= 72 else fname[:69] + "..."
        print(f"    {cov:>6,}/{tot:>6,} ({p:5.1f}%)  {label}")

    print(f"\n  Output files:")
    print(f"    covered    → {args.covered}")
    print(f"    unobserved → {args.unobserved}")


if __name__ == "__main__":
    main()
