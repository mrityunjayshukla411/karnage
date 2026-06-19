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
        context: Optional dict of structured fields for programmatic inspection.
    """

    def __init__(self, message: str, context: dict | None = None) -> None:
        super().__init__(message)
        self.context: dict = context or {}


class ParserError(KarnageError):
    """An error occurred while parsing binary files.

    Covers failures from ``nm``, ``readelf``, and regex mismatches.
    """


class ScannerError(KarnageError):
    """An error occurred during target-independent function discovery.

    Raised when ``nm`` fails on the target binary or when no matching
    symbols are found.
    """


class FlipperError(KarnageError):
    """An error occurred during the GDB-based flip test run.

    Raised when ``flip_sites.json`` cannot be loaded or has an unexpected
    schema.
    """
