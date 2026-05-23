class KarnageError(Exception):
    """Base exception for all chitragupt errors."""

    def __init__(self, message: str, context: dict | None = None) -> None:
        super().__init__(message)
        self.context: dict = context or {}


class LLVMProjectDownloadError(KarnageError):
    """libtriton.so (or equivalent target library) was not found."""

class LibraryNotFoundError(KarnageError):
    """libtriton.so (or equivalent target library) was not found."""


class CommitResolutionError(KarnageError):
    """Cannot determine the LLVM commit hash embedded in the binary."""

class LLVMProjectBuildError(KarnageError):
    """CMake configure or build step for llvm-project failed."""