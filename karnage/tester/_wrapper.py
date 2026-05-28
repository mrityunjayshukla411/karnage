#!/usr/bin/env python3
"""
_wrapper.py — Inferior wrapper for GDB-based MatcherTable fault injection.

GDB runs this script as the inferior:
    python _wrapper.py <triton_script.py> [args...]

After the user script finishes, every top-level torch.Tensor in its global
namespace is saved to KARNAGE_OUTPUT_DIR/{name}.pt so the outer orchestrator
can compare baseline vs patched results without parsing stdout.

Environment variables consumed:
    KARNAGE_OUTPUT_DIR  — directory for tensor files (created if absent)
    TRITON_CACHE_DIR    — set by runner.py; Triton writes PTX here
    TRITON_ALWAYS_COMPILE — set to "1" by runner.py
"""
import os
import sys
import pathlib
import runpy

output_dir = pathlib.Path(os.environ["KARNAGE_OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)

if len(sys.argv) < 2:
    sys.exit("_wrapper: missing user script argument")

# Shift argv so the user script sees itself as sys.argv[0]
sys.argv = sys.argv[1:]
script = sys.argv[0]

try:
    ns = runpy.run_path(script, run_name="__main__")
except SystemExit:
    raise
except BaseException as exc:
    # Write a sentinel so _compare() in runner.py can detect script-level errors
    (output_dir / "_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
    sys.exit(1)

try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for name, val in ns.items():
        if isinstance(val, torch.Tensor) and not name.startswith("_"):
            torch.save(val.detach().cpu(), output_dir / f"{name}.pt")
except ImportError:
    pass

# Sentinel: runner.py checks this to know the script actually completed.
# Written last so any earlier failure leaves it absent.
(output_dir / "_done").touch()
