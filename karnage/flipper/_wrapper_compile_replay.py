"""
_wrapper_compile_replay.py --- GDB inferior for GPU-free compile-replay fault injection.

GDB runs this script as the inferior, exactly like _wrapper.py:
    gdb --batch -q -x _gdb_script.py --args python \\
        _wrapper_compile_replay.py <capture_blob.pkl>

Unlike _wrapper.py (which runs a full Triton application, real tensors, real
GPU launches), this replays every specialization captured by
karnage.compile_capture directly through karnage.compile_replay.replay_compile()
--- no torch, no CUDA device access at all. _gdb_script.py's bit-flip patch is
applied to the loaded libtriton.so exactly as normal (that mechanism is a pure
ptrace operation and doesn't care what the inferior does afterward); the flip
is exercised purely by triton.compile() itself, with the same zero-GPU-
dependency guarantee validated in karnage/tests/test_compile_equivalence.py.

Loads karnage/compile_replay.py by direct file path rather than `import
karnage.compile_replay`, matching _wrapper.py's self-contained style (this
script must not depend on `karnage` being importable via whatever sys.path
the inferior process happens to start with).

Environment variables consumed:
    KARNAGE_OUTPUT_DIR    --- directory for sentinels (_done / _error.txt)
    TRITON_CACHE_DIR      --- set by runner.py; Triton writes ttir/ttgir/
                              llir/ptx/cubin here, same as _wrapper.py
    TRITON_ALWAYS_COMPILE --- set to "1" by runner.py; critical here ---
                              triton.compile() checks its own on-disk cache
                              before recompiling (see compile_capture.py's
                              module docstring), so without this a cache hit
                              would silently skip the patched compiler
                              entirely and the flip would never be exercised.

Sentinels written to KARNAGE_OUTPUT_DIR:
    _done       --- every captured specialization replayed without exception
    _error.txt  --- exception type and message on the first failure
"""

import importlib.util
import os
import pathlib
import pickle
import sys

output_dir = pathlib.Path(os.environ["KARNAGE_OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)

if len(sys.argv) < 2:
    sys.exit("_wrapper_compile_replay: missing capture blob path argument")

blob_path = sys.argv[1]

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_COMPILE_REPLAY_PATH = _THIS_DIR.parent / "compile_replay.py"
_spec = importlib.util.spec_from_file_location(
    "_karnage_compile_replay_module", _COMPILE_REPLAY_PATH
)
_compile_replay = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _compile_replay
_spec.loader.exec_module(_compile_replay)
replay_compile = _compile_replay.replay_compile

try:
    with open(blob_path, "rb") as f:
        store = pickle.load(f)

    for entries in store.values():
        for entry in entries:
            replay_compile(entry)
except BaseException as exc:
    (output_dir / "_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
    sys.exit(1)

# Written last — absence means the process crashed or was killed.
(output_dir / "_done").touch()
