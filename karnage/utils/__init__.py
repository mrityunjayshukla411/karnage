# karnage/utils/__init__.py

from .constants import (
    DEFAULT_ADJACENCY,
    DEFAULT_MATCHER_TABLE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TARGET_SO,
    ENV_ALWAYS_COMPILE,
    ENV_OUTPUT_DIR,
    ENV_PATCH_SPEC,
    ENV_TARGET_SO,
    ENV_TRITON_CACHE,
    MAX_OPCODES,
    SYMBOL_SIZE_FALLBACK,
)
from .exceptions import (
    CommitResolutionError,
    KarnageError,
    LibraryNotFoundError,
    LLVMProjectBuildError,
    LLVMProjectDownloadError,
    ParserError,
)
from .logger import logger
from .models import (
    AdjacencyEntry,
    FlipInfo,
    FlipResult,
    MatcherEntry,
    OpcodeInfo,
    PatchSpec,
)
from .subprocess_runner import run_subprocess
from .targets import (
    NVPTXBackend,
    TargetBackend,
)

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
