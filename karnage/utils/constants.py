"""Centralised constants for the karnage pipeline.

Environment variable names
--------------------------
``_gdb_script.py`` and ``_wrapper.py`` run inside GDB's embedded Python
interpreter and **cannot** import from the karnage package.  They read these
variables as raw string literals.  :mod:`karnage.flipper.runner` uses the
constants below when *setting* those variables, so both sides of the contract
are defined from one place.

Parser tunables
---------------
``SYMBOL_SIZE_FALLBACK`` is used by :func:`~karnage.utils.parser.estimate_symbol_byte_size`
as a last resort when ``nm`` reports size=0 and no higher-VMA symbol exists.
"""

# ---------------------------------------------------------------------------
# Environment variable names
# ---------------------------------------------------------------------------

ENV_OUTPUT_DIR = "KARNAGE_OUTPUT_DIR"
"""Directory where ``_wrapper.py`` writes per-tensor ``.pt`` files."""

ENV_PATCH_SPEC = "KARNAGE_PATCH_SPEC"
"""Path to the ``patch_spec.json`` consumed by ``_gdb_script.py``."""

ENV_TARGET_SO = "KARNAGE_TARGET_SO"
"""Substring matched against loaded objfile names in ``_gdb_script.py``."""

ENV_TRITON_CACHE = "TRITON_CACHE_DIR"
"""Directory where Triton writes compiled PTX; set by ``runner.py``."""

ENV_ALWAYS_COMPILE = "TRITON_ALWAYS_COMPILE"
"""Forces Triton to recompile on every run; set to ``"1"`` by ``runner.py``."""

ENV_COMPILE_ONLY = "KARNAGE_COMPILE_ONLY"
"""When set to ``"1"``, ``_wrapper.py`` patches Triton so kernels compile but
never launch on the GPU --- see the module docstring in ``_wrapper.py``."""

ENV_SIGNATURE_OUT = "KARNAGE_SIGNATURE_OUT"
"""Path to write a kernel-call signature manifest during the baseline
compile-only run --- see the "Signature capture / replay" section of
``_wrapper.py``'s module docstring."""

ENV_SIGNATURE_IN = "KARNAGE_SIGNATURE_IN"
"""Path to a signature manifest to replay instead of running the target
script --- mutually exclusive with ``ENV_SIGNATURE_OUT``."""

# ---------------------------------------------------------------------------
# Default names / paths used by the CLI
# ---------------------------------------------------------------------------

DEFAULT_FLIP_SITES = "flip_sites.json"
"""Default output path for the ``scan`` subcommand."""

DEFAULT_OUTPUT_DIR = "test_results"
"""Default root output directory for the ``flip`` subcommand."""

DEFAULT_TARGET_SO = "libtriton.so"
"""Default shared-library name matched by ``_gdb_script.py``."""

# ---------------------------------------------------------------------------
# Parser tunables
# ---------------------------------------------------------------------------

SYMBOL_SIZE_FALLBACK = 250_000
"""Byte-size last resort used by ``estimate_symbol_byte_size`` when nm reports
size=0 and no higher-VMA symbol exists."""
