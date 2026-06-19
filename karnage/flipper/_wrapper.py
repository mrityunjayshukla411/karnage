"""
_wrapper.py --- Inferior wrapper for karnage bit-flip injection.

GDB runs this script as the inferior:
    gdb --batch -q -x _gdb_script.py --args python _wrapper.py <triton_script.py>

Stdout isolation:
    sys.stdout is redirected to KARNAGE_OUTPUT_DIR/app_stdout.txt before the
    user script runs.  GDB's own thread/process event messages share the GDB
    process stdout fd but bypass sys.stdout, so they are captured by the runner
    as gdb_stdout.txt and never enter app_stdout.txt.  The runner diffs
    app_stdout.txt between the baseline and each flip run.

Sentinels written to KARNAGE_OUTPUT_DIR:
    _done       --- script completed without exception
    _error.txt  --- exception type and message on failure

Environment variables consumed:
    KARNAGE_OUTPUT_DIR    --- directory for app_stdout.txt and sentinels
    TRITON_CACHE_DIR      --- set by runner.py; Triton writes PTX here
    TRITON_ALWAYS_COMPILE --- set to "1" by runner.py
"""

import os
import pathlib
import runpy
import sys

output_dir = pathlib.Path(os.environ["KARNAGE_OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)

if len(sys.argv) < 2:
    sys.exit("_wrapper: missing user script argument")

# Shift argv so the user script sees itself as sys.argv[0].
sys.argv = sys.argv[1:]
script = sys.argv[0]

# Redirect stdout to a file.  GDB's own [New Thread ...] / [Detaching ...] /
# [Inferior N exited] messages are written directly to the GDB process stdout fd
# and do not go through sys.stdout, so they will not appear in this file.
_app_out = open(output_dir / "app_stdout.txt", "w")
sys.stdout = _app_out

try:
    runpy.run_path(script, run_name="__main__")
except SystemExit:
    raise
except BaseException as exc:
    _app_out.flush()
    _app_out.close()
    sys.stdout = sys.__stdout__
    (output_dir / "_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
    sys.exit(1)

_app_out.flush()
_app_out.close()
sys.stdout = sys.__stdout__

# Written last — absence means the script crashed or was killed.
(output_dir / "_done").touch()
