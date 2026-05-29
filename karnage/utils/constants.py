"""Centralised constants for the karnage pipeline.

Keeping all magic strings and tunable numbers here avoids silent
inconsistencies when the same value appears in multiple files.

Environment variable names
--------------------------
``_gdb_script.py`` and ``_wrapper.py`` run inside GDB's embedded Python
interpreter and **cannot** import from the karnage package.  They read these
variables as raw string literals.  :mod:`karnage.injector.runner` uses the
constants below when *setting* those variables, so both sides of the contract
are defined from one place.

Parser tunables
---------------
``SYMBOL_SIZE_FALLBACK`` and ``MAX_OPCODES`` were previously bare numeric
literals buried inside :mod:`karnage.utils.parser`.  Naming them here makes
the intent explicit and allows targeted overrides in tests.
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

# ---------------------------------------------------------------------------
# Default names / paths used by the CLI
# ---------------------------------------------------------------------------

DEFAULT_TARGET_SO = "libtriton.so"
"""Default shared-library name matched by ``_gdb_script.py``."""

DEFAULT_MATCHER_TABLE = "matcher_table.json"
"""Default output path for the ``extract`` subcommand."""

DEFAULT_ADJACENCY = "adjacency.json"
"""Default output path for the ``inject`` subcommand."""

DEFAULT_OUTPUT_DIR = "test_results"
"""Default root output directory for the ``test`` subcommand."""

# ---------------------------------------------------------------------------
# Parser tunables
# ---------------------------------------------------------------------------

SYMBOL_SIZE_FALLBACK = 250_000
"""Byte-size last resort used by ``estimate_symbol_byte_size`` when nm reports
size=0 and no higher-VMA symbol exists.  Large enough to cover the NVPTX
MatcherTable in recent LLVM releases while staying well below typical binary
sizes."""

MAX_OPCODES = 10_000
"""Upper bound for the opcode-index scan in ``build_opcode_mnemonic_map``.
Scanning stops earlier if the OpInfo0 table ends before this limit."""
