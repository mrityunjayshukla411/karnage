# karnage/utils/__init__.py

from .logger import logger
from .exceptions import (
    KarnageError,
    LibraryNotFoundError,
    CommitResolutionError,
    LLVMProjectDownloadError,
    LLVMProjectBuildError,
    ParserError,
)

from .targets import (
    TargetBackend,
    NVPTXBackend,
)

from .models import (
    MatcherEntry,
    FlipInfo,
    OpcodeInfo,
    AdjacencyEntry,
    PatchSpec,
    FlipResult,
)

from .constants import (
    ENV_OUTPUT_DIR,
    ENV_PATCH_SPEC,
    ENV_TARGET_SO,
    ENV_TRITON_CACHE,
    ENV_ALWAYS_COMPILE,
    DEFAULT_TARGET_SO,
    DEFAULT_MATCHER_TABLE,
    DEFAULT_ADJACENCY,
    DEFAULT_OUTPUT_DIR,
    SYMBOL_SIZE_FALLBACK,
    MAX_OPCODES,
)

from .subprocess_runner import run_subprocess

__all__ = [
    "logger",

    "KarnageError",
    "LibraryNotFoundError",
    "CommitResolutionError",
    "LLVMProjectDownloadError",
    "LLVMProjectBuildError",
    "ParserError",

    "TargetBackend",
    "NVPTXBackend",

    "MatcherEntry",
    "FlipInfo",
    "OpcodeInfo",
    "AdjacencyEntry",
    "PatchSpec",
    "FlipResult",

    "ENV_OUTPUT_DIR",
    "ENV_PATCH_SPEC",
    "ENV_TARGET_SO",
    "ENV_TRITON_CACHE",
    "ENV_ALWAYS_COMPILE",
    "DEFAULT_TARGET_SO",
    "DEFAULT_MATCHER_TABLE",
    "DEFAULT_ADJACENCY",
    "DEFAULT_OUTPUT_DIR",
    "SYMBOL_SIZE_FALLBACK",
    "MAX_OPCODES",

    "run_subprocess",
]
