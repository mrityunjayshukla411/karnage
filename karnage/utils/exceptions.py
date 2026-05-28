class KarnageError(Exception):
    """Base exception for all karnage errors."""

    def __init__(self, message: str, context: dict | None = None) -> None:
        super().__init__(message)
        self.context: dict = context or {}


class LLVMProjectDownloadError(KarnageError):
    """LLVM project download failed or got interrupted while"""

class LibraryNotFoundError(KarnageError):
    """libtriton.so (or equivalent target library) was not found."""


class CommitResolutionError(KarnageError):
    """Cannot determine the LLVM commit hash embedded in the binary."""

class LLVMProjectBuildError(KarnageError):
    """CMake configure or build step for llvm-project failed."""

class ParserError(KarnageError):
    """Error occured while parsing files related to matcher table"""

class InjectorError(KarnageError):
    """Base error for the injector module."""

class MatcherTableLoadError(InjectorError):
    """matcher_table.json could not be loaded or has an unexpected schema."""