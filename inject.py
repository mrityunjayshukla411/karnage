#!/usr/bin/env python3
"""
inject.py — Find adjacent LLVM instructions in the MatcherTable.

Two instructions are adjacent when their opcode encodings (opc_lo, opc_hi)
differ by exactly one bit in either byte (but not both simultaneously).

Usage:
    python inject.py --input matcher_table.json --output adjacency.json

Output shape:
  {
    "meta": {
      "source":               "matcher_table.json",
      "llvm_commit":          "...",
      "total_opcodes":        5370,
      "total_adjacent_pairs": 29666
    },
    "instructions": {
      "<opcode_int>": {
        "mnemonic":     "...",
        "opcode":       N,
        "opc_lo":       "0xNN",
        "opc_hi":       "0xNN",
        "num_patterns": K,
        "adjacent": [
          {
            "mnemonic":     "...",
            "opcode":       M,
            "opc_lo":       "0xNN",
            "opc_hi":       "0xNN",
            "num_patterns": K,
            "flip": { "byte": "opc_lo", "bit": 3, "mask": "0x08" }
          },
          ...
        ]
      },
      ...
    }
  }

  Keys are opcode integers (as strings) rather than mnemonics so that
  multiple instruction variants sharing one mnemonic remain distinct.
"""

import argparse
from pathlib import Path

from karnage.extractor.extractor import run


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Find adjacent LLVM instructions by single-bit opcode distance."
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("matcher_table.json"),
        help="Path to matcher_table.json (default: matcher_table.json)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("adjacency.json"),
        help="Output JSON path (default: adjacency.json)",
    )
    args = ap.parse_args()

    if not args.input.exists():
        ap.error(f"Input file not found: {args.input}")

    run(args.input, args.output)


if __name__ == "__main__":
    main()
