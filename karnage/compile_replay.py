"""Replay a captured ``triton.compile()`` call with zero GPU/CUDA access.

Given one entry captured by :mod:`karnage.compile_capture`, reconstructs the
exact ``ASTSource`` Triton's own JIT path would have built and calls
``triton.compile()`` directly --- bypassing ``JITFunction.run()``/``.compile``
entirely, so no device query (``driver.active.get_current_device()`` et al.)
ever happens. This module deliberately never imports ``torch`` and never
imports ``triton`` at module level (only lazily, inside functions) so that
whichever process calls :func:`replay_compile` controls exactly when Triton
first loads relative to setting ``CUDA_VISIBLE_DEVICES=""``.

Why this is provably the *same* compilation a real run would have done, not
merely a similar one: :func:`karnage.compile_capture.capture_all_compiles`
records the precise ``(signature, constants, attrs, target, options)`` tuple
Triton's real JIT path passed to ``triton.compile()`` for a real, on-device
run --- this module feeds that exact tuple back into the same ``triton.compile``
free function. See ``karnage/tests/test_compile_equivalence.py`` for the test
that proves byte-for-byte equality of every IR stage between the two.

Critical gotcha: source file identity matters. Triton embeds the kernel's
source file path and line number into LLVM debug info (``!DIFile``) and PTX
(``.loc`` directives). A capture's ``source_file`` must be importable at
exactly that path when replaying, or the resulting IR/PTX/cubin will differ
from the real run for reasons that have nothing to do with actual codegen ---
this is why capture records the kernel's defining file path (not the
workload script that happened to call it) and replay imports from exactly
that path, not by trying to reconstruct the kernel definition some other way.
"""

from __future__ import annotations

import importlib.util
import sys


def _import_kernel(source_file: str, qualname: str):
    """Import the kernel object a capture entry was recorded against.

    Imports *source_file* as a fresh, uniquely-named module (not
    ``run_name="__main__"``, so any ``if __name__ == "__main__":``-guarded
    code in that file does not execute --- the same import-safety convention
    ``karnage/flipper/_wrapper.py``'s ``--replay-signatures`` mode already
    requires of target scripts) and resolves *qualname* on it via chained
    ``getattr`` (handles both flat module-level names and dotted qualnames).

    Args:
        source_file: Exact file path the kernel was defined in (from
                     ``kernel_obj.fn.__code__.co_filename`` at capture time).
        qualname:    The kernel's ``__qualname__`` at capture time.

    Returns:
        The re-imported ``JITFunction`` (or ``Autotuner``, etc.) object.

    Raises:
        ImportError: If *qualname* can't be resolved on the imported module.
                     (A plain built-in, not ``karnage.utils.exceptions``'s
                     ``CompileCaptureError`` --- this module must stay
                     importable with zero ``karnage`` package dependency;
                     see the module docstring.)
    """
    spec = importlib.util.spec_from_file_location(
        "_karnage_compile_replay_target", source_file
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    obj = module
    for part in qualname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise ImportError(
                f"replay could not resolve {qualname!r} in {source_file} "
                f"(failed at {part!r})"
            ) from exc
    return obj


def replay_compile(entry: dict):
    """Replay one :func:`karnage.compile_capture.capture_all_compiles` entry.

    Calls ``triton.compile()`` directly with *entry*'s exact captured
    ``target``/``options`` --- both fully explicit, so no device is ever
    queried. Safe to call with ``CUDA_VISIBLE_DEVICES=""`` and no GPU present.

    Args:
        entry: One capture dict as produced by
               :func:`karnage.compile_capture.capture_all_compiles` (keys:
               ``source_file``, ``qualname``, ``signature``, ``constants``,
               ``attrs``, ``target``, ``options``; ``asm``/``ptxas_version``
               are ignored here --- they're the *expected* output for
               comparison, not replay input).

    Returns:
        The ``CompiledKernel`` from this replay's ``triton.compile()`` call
        --- compare its ``.asm`` dict against *entry*'s ``asm`` to check
        equivalence.
    """
    from triton.compiler import ASTSource
    from triton.compiler import compile as triton_compile

    kernel_obj = _import_kernel(entry["source_file"], entry["qualname"])
    src = ASTSource(
        fn=kernel_obj,
        signature=entry["signature"],
        constexprs=entry["constants"],
        attrs=entry["attrs"],
    )
    return triton_compile(src, target=entry["target"], options=entry["options"])
