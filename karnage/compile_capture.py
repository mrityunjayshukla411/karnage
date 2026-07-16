"""Capture the exact ``triton.compile()`` inputs/outputs a real run produces.

For a real, on-device run of any Triton workload, records the precise
``(signature, constants, attrs, target, options)`` tuple the normal JIT path
passes to ``triton.compile()`` for every kernel specialization it exercises,
plus the resulting compiled artifacts (``ttir``/``ttgir``/``llir``/``ptx``/
``cubin``). This is the write side of a capture/replay pair with
:mod:`karnage.compile_replay`: replaying a captured entry through
``triton.compile()`` directly needs zero GPU/CUDA access, so it can run in
parallel, off-device, under GDB-based fault injection --- see
:mod:`karnage.compile_replay`'s module docstring for why that's provably the
same compilation a real run would have done, not just a similar one.

Mechanism (verified directly against the installed Triton source in this
project's own venv; also matches a previously-validated proof of concept for
a toy vector-add kernel)
-----------------------------------------------------------------------------
``JITFunction.create_binder()`` is the *only* place ``self.compile`` gets
assigned --- as an instance attribute, not a class attribute --- and it needs
a live device to resolve ``driver.active.get_current_target()``. It fires
lazily on a kernel's first real call. So capturing a kernel requires letting
its first invocation run for real first; only then does ``kernel_obj.compile``
exist to wrap.

Once wrapped, forcing a genuine second compile (to actually exercise the
wrapper, since the first specialization already got cached by the unpatched
compile) means clearing that device's ``kernel_cache``/``kernel_key_cache``
dicts **in place** (``.clear()``) --- never reassigning or deleting
``device_caches[device]`` itself, which would re-run ``create_binder()`` and
silently reset ``.compile`` back to the unpatched original. The forced-recompile
call itself uses ``warmup=True`` (triggers compilation without re-launching the
kernel, since we already validated correctness on the first, real call).

Every specialization *after* that first one is captured automatically and
cheaply: ``JITFunction.run()`` only calls ``.compile`` on a genuine cache miss,
so the wrapper naturally sees each distinct specialization exactly once with
no extra bookkeeping needed.

Patches ``triton.runtime.jit.KernelInterface.__getitem__`` at the class level
(same pattern as ``karnage/flipper/_wrapper.py``'s ``--compile-only`` mode) so
this covers every kernel a script calls transparently, without needing to know
kernel names in advance.
"""

from __future__ import annotations

import pickle
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

CaptureStore = dict[tuple[str, str], list[dict]]


def _ptxas_version() -> str:
    """Return ``ptxas --version`` output, for cross-checking captures later.

    A captured blob's compiled artifacts are only meaningful to compare
    against a replay done with the same CUDA toolkit --- ``ptxas`` runs as an
    external subprocess during the cubin stage, and a toolchain mismatch
    produces a "different cubin" result that has nothing to do with Triton or
    Karnage.
    """
    try:
        result = subprocess.run(
            ["ptxas", "--version"], capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return f"<unavailable: {type(exc).__name__}: {exc}>"


@contextmanager
def capture_all_compiles(output_path: Path) -> Iterator[CaptureStore]:
    """Capture every ``triton.compile()`` call made inside this ``with`` block.

    Intended usage: run a real Triton workload script's normal code (real
    tensors, real kernel launches) inside this context. Every kernel it calls
    is captured transparently --- no need to know kernel names up front, and
    multiple distinct kernels/specializations from one workload are all
    captured into the same store, keyed by ``(source_file, qualname)``.

    Args:
        output_path: Where to ``pickle.dump`` the full capture store on exit.
                     Parent directories are created if needed.

    Yields:
        The live capture store dict, updated in place as kernels are called
        --- ``{(source_file, qualname): [capture_dict, ...]}``. Also written
        to *output_path* when the ``with`` block exits normally or raises.

    Each ``capture_dict`` has keys: ``source_file``, ``qualname``,
    ``signature``, ``constants``, ``attrs``, ``target``, ``options``, ``asm``
    (the compiled ``{ext: content}`` dict), ``ptxas_version``.
    """
    import triton.runtime.jit as _jit
    from triton.runtime import driver

    store: CaptureStore = {}
    patched_kernel_ids: set[int] = set()
    ptxas_version = _ptxas_version()
    original_getitem = _jit.KernelInterface.__getitem__

    def _make_capturing_compile(kernel_key, source_file, qualname, real_compile):
        def _capturing_compile(src, target=None, options=None, **kwargs):
            result = real_compile(src, target=target, options=options, **kwargs)
            store.setdefault(kernel_key, []).append(
                {
                    "source_file": source_file,
                    "qualname": qualname,
                    "signature": dict(src.signature),
                    "constants": dict(src.constants),
                    "attrs": dict(src.attrs),
                    "target": target,
                    "options": dict(options) if options else {},
                    "asm": dict(result.asm),
                    "ptxas_version": ptxas_version,
                }
            )
            return result

        return _capturing_compile

    def _capture_getitem(self, grid):
        def _call(*args, **kwargs):
            # Real launch, always --- correctness and target/device resolution
            # both depend on this being a genuine, unmodified run.
            result = self.run(grid=grid, warmup=False, *args, **kwargs)

            if id(self) not in patched_kernel_ids:
                patched_kernel_ids.add(id(self))
                source_file = self.fn.__code__.co_filename
                qualname = self.fn.__qualname__
                kernel_key = (source_file, qualname)
                real_compile = self.compile
                self.compile = _make_capturing_compile(
                    kernel_key, source_file, qualname, real_compile
                )

                device = driver.active.get_current_device()
                kernel_cache, kernel_key_cache, *_rest = self.device_caches[device]
                kernel_cache.clear()
                kernel_key_cache.clear()

                # Forces a genuine recompile through the now-wrapped .compile,
                # capturing the specialization the real call above already
                # validated. warmup=True: no need to re-launch, only to
                # re-trigger compilation.
                self.run(grid=grid, warmup=True, *args, **kwargs)

            return result

        return _call

    _jit.KernelInterface.__getitem__ = _capture_getitem
    try:
        yield store
    finally:
        _jit.KernelInterface.__getitem__ = original_getitem
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(store, f)
