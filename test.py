#!/usr/bin/env python3
"""
test.py — GDB-based MatcherTable bit-flip tester.

For each adjacent instruction pair in adjacency.json, flips the corresponding
bit in libtriton.so's in-memory MatcherTable via GDB's ptrace interface, then
checks whether the generated PTX or the kernel's numerical output changed.

Prerequisites:
  - GDB with Python support installed (gdb --version)
  - Docker: run with --cap-add SYS_PTRACE (or --privileged)
  - adjacency.json produced by:   python inject.py
  - matcher_table.json produced by: python extract.py

Usage:
    python test.py --script matmul.py \\
                   --library /path/to/libtriton.so \\
                   --output results/

    python test.py --script attention.py \\
                   --library /path/to/libtriton.so \\
                   --max-flips 50 \\
                   --report report.json

Output layout (per flip):
    results/
      baseline/
        tensors/         <- {name}.pt files, one per top-level torch.Tensor
        triton_cache/    <- Triton-compiled PTX files
        stdout.txt
        stderr.txt
      flip_000000/
        patch_spec.json  <- {"patch_vmas": [...], "mask": N}
        tensors/
        triton_cache/
        stdout.txt / stderr.txt / returncode.txt
      ...
      report.json        <- summary (if --report given)
"""

import argparse
from pathlib import Path

from karnage.tester import run_tester


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MatcherTable bit-flip tester via GDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--script", type=Path, required=True,
        help="Triton application script to test (e.g. matmul.py)",
    )
    ap.add_argument(
        "--matcher-table", type=Path, default=Path("matcher_table.json"),
        metavar="PATH",
        help="matcher_table.json from extract.py (default: %(default)s)",
    )
    ap.add_argument(
        "--adjacency", type=Path, default=Path("adjacency.json"),
        metavar="PATH",
        help="adjacency.json from inject.py (default: %(default)s)",
    )
    ap.add_argument(
        "--library", type=Path, required=True,
        help="Path to libtriton.so",
    )
    ap.add_argument(
        "--output", type=Path, default=Path("test_results"),
        metavar="DIR",
        help="Root output directory for per-flip results (default: %(default)s)",
    )
    ap.add_argument(
        "--max-flips", type=int, default=None, metavar="N",
        help="Stop after N flips (useful for a quick sanity check)",
    )
    ap.add_argument(
        "--report", type=Path, default=None, metavar="PATH",
        help="Write a JSON summary of all flip results to this path",
    )
    ap.add_argument(
        "--cooldown-every", type=int, default=0, metavar="N",
        help="Pause for --cooldown-secs after every N flips (0 = disabled)",
    )
    ap.add_argument(
        "--cooldown-secs", type=float, default=30.0, metavar="S",
        help="Seconds to sleep during each cooldown pause (default: 30)",
    )
    ap.add_argument(
        "--filter-by-ptx", action="store_true",
        help=(
            "Only test opcodes whose mnemonic appears in the baseline PTX. "
            "Skips instructions the kernel never uses (e.g. abs.bf16 in matmul)."
        ),
    )
    ap.add_argument(
        "--mnemonics", type=str, default=None, metavar="m1,m2,...",
        help=(
            "Comma-separated list of mnemonics to test exclusively "
            "(e.g. --mnemonics fma.rn.f32,mul.rn.f32). "
            "Can be combined with --filter-by-ptx."
        ),
    )
    args = ap.parse_args()

    for path, flag in [
        (args.script,        "--script"),
        (args.matcher_table, "--matcher-table"),
        (args.adjacency,     "--adjacency"),
        (args.library,       "--library"),
    ]:
        if not path.exists():
            ap.error(f"{flag}: not found: {path}")

    target_mnemonics = (
        frozenset(m.strip() for m in args.mnemonics.split(",") if m.strip())
        if args.mnemonics else None
    )

    run_tester(
        triton_script      = args.script,
        matcher_table_json = args.matcher_table,
        adjacency_json     = args.adjacency,
        libtriton_so       = args.library,
        output_dir         = args.output,
        max_flips          = args.max_flips,
        report_json        = args.report,
        cooldown_every     = args.cooldown_every,
        cooldown_secs      = args.cooldown_secs,
        filter_by_ptx      = args.filter_by_ptx,
        target_mnemonics   = target_mnemonics,
    )


if __name__ == "__main__":
    main()
