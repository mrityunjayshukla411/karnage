"""Custom exception hierarchy for the karnage pipeline.

All exceptions carry a ``context`` dict with machine-readable fields that
supplement the human-readable message.  Callers can inspect ``exc.context``
to extract structured information (binary path, symbol name, etc.) without
parsing the message string.
"""


class KarnageError(Exception):
    """Base class for all karnage exceptions.

    Args:
        message: Human-readable description of the error.
        context: Optional dict of structured fields for programmatic inspection
                 (e.g. ``{"binary": "/path/to/lib.so", "symbol": "..."}``).
    """

    def __init__(self, message: str, context: dict | None = None) -> None:
        super().__init__(message)
        self.context: dict = context or {}


class LLVMProjectDownloadError(KarnageError):
    """The LLVM source archive could not be downloaded or extracted.

    Raised when ``curl``/``wget`` returns a non-zero exit code or when
    ``unzip`` fails to extract the downloaded archive.
    """


class LibraryNotFoundError(KarnageError):
    """The target shared library (e.g. ``libtriton.so``) was not found.

    Raised when the explicit ``--library`` path does not exist or when
    auto-detection via ``pip show`` fails to locate the package.
    """


class CommitResolutionError(KarnageError):
    """The LLVM commit hash embedded in the binary could not be determined.

    Raised when ``strings`` produces no output matching the expected
    ``LLVM version X.Y.Z (<40-char hex>)`` pattern.
    """


class LLVMProjectBuildError(KarnageError):
    """A CMake configure or build step for ``llvm-project`` failed.

    Raised when either the ``cmake -S … -B …`` configure invocation or a
    ``cmake --build … --target …`` build invocation exits non-zero.
    """


class ParserError(KarnageError):
    """An error occurred while parsing binary or source files.

    Covers failures from ``nm``, ``readelf``, regex mismatches in
    ``SelectionDAGISel.h`` / ``GenVT.inc``, and empty AsmWriter tables.
    """


class InjectorError(KarnageError):
    """Base class for injector-module errors."""


class MatcherTableLoadError(InjectorError):
    """``matcher_table.json`` could not be loaded or has an unexpected schema.

    Raised when the file is missing, not valid JSON, or missing required
    top-level keys such as ``"instructions"``.
    """
