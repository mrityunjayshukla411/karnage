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
    KARNAGE_COMPILE_ONLY  --- see "Compile-only mode" below
    KARNAGE_SIGNATURE_OUT --- see "Signature capture / replay" below
    KARNAGE_SIGNATURE_IN  --- see "Signature capture / replay" below

Compile-only mode (KARNAGE_COMPILE_ONLY=1):
    Patches triton.runtime.jit.KernelInterface.__getitem__ so every
    `kernel[grid](*args)` call in the user script compiles the kernel (writing
    the normal .ttir/.ttgir/.llir/.ptx/.cubin cache artifacts, same as always)
    but never launches it on the GPU.  Triton's own run() unconditionally
    compiles regardless of warmup=True/False; the launch is gated behind a
    single `if not warmup:` block, so passing warmup=True skips it entirely
    (see karnage/flipper/runner.py for the full rationale).

    Because the kernel never runs, script logic that depends on its output
    (result assertions, prints on the output tensor) now operates on
    uninitialized memory and will likely raise --- *after* compilation already
    succeeded and captured everything karnage cares about.  If at least one
    kernel already compiled successfully by the time that happens, the
    exception is treated as expected downstream fallout, not a crash: _done
    is still written, with a note recording what was suppressed.

Signature capture / replay (KARNAGE_SIGNATURE_OUT / KARNAGE_SIGNATURE_IN):
    Even in compile-only mode, the user script still runs in full up to each
    kernel call site --- any real torch/CUDA setup it does (tensor
    allocation, torch.randn, CUDA context init) still happens.  Signature
    capture/replay decouples compilation from the application entirely for
    repeated flips:

    - Capture (KARNAGE_SIGNATURE_OUT=<path>, combined with
      KARNAGE_COMPILE_ONLY=1): the compile-only monkeypatch additionally
      records each intercepted call's kernel identity --- resolved by
      scanning the *calling frame's* globals for a name bound to that exact
      kernel object, done at call time rather than after the script finishes,
      so it still works even when the script raises downstream of a
      successful compile --- plus its args/kwargs/grid, converted to a
      JSON-safe description.  Requires the script to gate its side-effecting
      code behind `if __name__ == "__main__":` (checked upfront, before
      running anything, so a non-conforming script fails fast with a clear
      message) --- this is what lets replay import "just the kernel" below.

    - Replay (KARNAGE_SIGNATURE_IN=<path>): instead of running the script at
      all, imports it as a real module (not run_name="__main__", so its
      `if __name__ == "__main__":` block correctly never executes), looks up
      each recorded kernel by its captured attribute name, reconstructs each
      call's args as _ReplayTensor / plain-value stand-ins (no real tensors,
      no torch allocation), and calls kernel.warmup(...) directly for every
      recorded call.  Zero GPU/CUDA touch beyond what triton.compile() itself
      needs.  Any exception here is a genuine import or compile failure ---
      there is no "downstream fallout" to suppress, since no app logic beyond
      the kernel calls themselves ever runs.
"""

import json
import os
import pathlib
import re
import runpy
import sys

output_dir = pathlib.Path(os.environ["KARNAGE_OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)

if len(sys.argv) < 2:
    sys.exit("_wrapper: missing user script argument")

# Shift argv so the user script sees itself as sys.argv[0].
sys.argv = sys.argv[1:]
script = sys.argv[0]

compile_only = os.environ.get("KARNAGE_COMPILE_ONLY") == "1"
signature_out = os.environ.get("KARNAGE_SIGNATURE_OUT")
signature_in = os.environ.get("KARNAGE_SIGNATURE_IN")
replay_mode = signature_in is not None

_compiled_ok_count = 0
_captured_calls: dict = {}

_MAIN_GUARD_RE = re.compile(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:')


def _describe_arg(arg):
    """JSON-safe description of one kernel-call argument, for signature capture."""
    if hasattr(arg, "dtype") and hasattr(arg, "shape") and hasattr(arg, "stride"):
        try:
            aligned16 = arg.data_ptr() % 16 == 0
        except Exception:
            aligned16 = True
        return {
            "kind": "tensor",
            "dtype": str(arg.dtype),
            "shape": list(arg.shape),
            "stride": list(arg.stride()),
            "aligned16": aligned16,
        }
    return {"kind": "scalar", "value": arg}


def _describe_grid(grid):
    """Best-effort JSON-safe grid description --- informational only.

    warmup()/run(warmup=True) never evaluates grid at all, so replay always
    passes a dummy grid regardless of what's recorded here.
    """
    if callable(grid):
        return None
    try:
        return list(grid)
    except TypeError:
        return None


class _ReplayTensor:
    """Lightweight tensor stand-in used during signature replay.

    Unlike Triton's own MockTensor (triton/runtime/jit.py), this preserves
    the *real* recorded stride and alignment class captured during signature
    capture, rather than assuming row-major-contiguous shape-derived stride
    and always-16-byte-aligned data --- see the module docstring's
    "Signature capture / replay" section.
    """

    def __init__(self, dtype, shape, stride, aligned16):
        self.dtype = dtype
        self.shape = shape
        self._stride = tuple(stride)
        self._aligned16 = aligned16

    def stride(self):
        return self._stride

    def data_ptr(self):
        return 0 if self._aligned16 else 1

    def ptr_range(self):
        return 0


if compile_only and not replay_mode:
    import triton.runtime.jit as _jit

    if signature_out is not None:
        _src = pathlib.Path(script).read_text(errors="replace")
        if not _MAIN_GUARD_RE.search(_src):
            sys.exit(
                f"_wrapper: --replay-signatures requires the target script to "
                f"gate its kernel-launching code behind "
                f'`if __name__ == "__main__":` --- {script} does not have this '
                f"guard. Wrap its kernel-launching code in that block to use "
                f"--replay-signatures with it."
            )

    def _compile_only_getitem(self, grid):
        def _warmup_call(*args, **kwargs):
            global _compiled_ok_count
            result = self.run(grid=grid, warmup=True, *args, **kwargs)
            _compiled_ok_count += 1
            if signature_out is not None:
                caller_globals = sys._getframe(1).f_globals
                kernel_attr = next(
                    (n for n, v in caller_globals.items() if v is self), None
                )
                if kernel_attr is not None:
                    _captured_calls.setdefault(kernel_attr, []).append(
                        {
                            "grid": _describe_grid(grid),
                            "args": [_describe_arg(a) for a in args],
                            "kwargs": dict(kwargs),
                        }
                    )
            return result

        return _warmup_call

    _jit.KernelInterface.__getitem__ = _compile_only_getitem


def _parse_torch_dtype(name: str):
    import torch

    return getattr(torch, name.split(".")[-1])


def _replay(script_path: str, manifest_path: str) -> None:
    """Replay recorded kernel calls from *manifest_path* without running *script_path*.

    Imports *script_path* as a real module (its `if __name__ == "__main__":`
    block correctly never executes), looks up each kernel by its recorded
    attribute name, and calls `.warmup(...)` with reconstructed
    (non-GPU-touching) args for every recorded call.
    """
    import importlib.util

    manifest = json.loads(pathlib.Path(manifest_path).read_text())

    spec = importlib.util.spec_from_file_location("_karnage_replay_target", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    for kernel_attr, calls in manifest.items():
        if not hasattr(module, kernel_attr):
            raise RuntimeError(
                f"replay manifest expects a kernel named {kernel_attr!r} in "
                f"{script_path}, but the module has no such attribute"
            )
        kernel_obj = getattr(module, kernel_attr)
        for call in calls:
            replay_args = []
            for desc in call["args"]:
                if desc["kind"] == "tensor":
                    replay_args.append(
                        _ReplayTensor(
                            dtype=_parse_torch_dtype(desc["dtype"]),
                            shape=desc["shape"],
                            stride=desc["stride"],
                            aligned16=desc["aligned16"],
                        )
                    )
                else:
                    replay_args.append(desc["value"])
            kernel_obj.warmup(*replay_args, grid=(1,), **call["kwargs"])


# Redirect stdout to a file.  GDB's own [New Thread ...] / [Detaching ...] /
# [Inferior N exited] messages are written directly to the GDB process stdout fd
# and do not go through sys.stdout, so they will not appear in this file.
_app_out = open(output_dir / "app_stdout.txt", "w")
sys.stdout = _app_out

try:
    if replay_mode:
        _replay(script, signature_in)
    else:
        runpy.run_path(script, run_name="__main__")
except SystemExit:
    raise
except BaseException as exc:
    if compile_only and not replay_mode and _compiled_ok_count > 0:
        # At least one kernel compiled successfully; since it never actually
        # ran, anything downstream operating on its (uninitialized) output is
        # expected to misbehave --- not a real failure.
        _app_out.flush()
        _app_out.close()
        sys.stdout = sys.__stdout__
        (output_dir / "_compile_only_note.txt").write_text(
            f"suppressed post-warmup exception: {type(exc).__name__}: {exc}\n"
        )
        (output_dir / "_done").touch()
        sys.exit(0)

    _app_out.flush()
    _app_out.close()
    sys.stdout = sys.__stdout__
    (output_dir / "_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
    sys.exit(1)

_app_out.flush()
_app_out.close()
sys.stdout = sys.__stdout__

if signature_out is not None:
    signature_path = pathlib.Path(signature_out)
    signature_path.parent.mkdir(parents=True, exist_ok=True)
    signature_path.write_text(json.dumps(_captured_calls, indent=2))

# Written last — absence means the script crashed or was killed.
(output_dir / "_done").touch()
