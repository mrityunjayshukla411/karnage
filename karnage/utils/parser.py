"""Binary introspection utilities for the karnage pipeline.

Provides:

  Binary introspection (nm / readelf)
    BinaryCache                  --- explicit-lifecycle cache for subprocess output.
    find_symbol_linker_vma       --- look up a symbol's linker-assigned VMA.
    estimate_symbol_byte_size    --- look up or estimate a symbol's byte size.
    linker_vma_to_file_offset    --- translate a VMA to an on-disk byte offset.
"""

import re
import subprocess
from pathlib import Path
from typing import TypeAlias

from karnage.utils.constants import SYMBOL_SIZE_FALLBACK
from karnage.utils.exceptions import ParserError
from karnage.utils.logger import logger
from karnage.utils.subprocess_runner import run_subprocess

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

# One row from `nm -S -Cn`: (linker_vma, elf_size, sym_type, demangled_name)
_NmSymbol: TypeAlias = tuple[int, int, str, str]


# ---------------------------------------------------------------------------
# Raw subprocess helpers  (private --- call through BinaryCache instead)
# ---------------------------------------------------------------------------


def _run_nm(binary: Path) -> list[_NmSymbol]:
    """Run ``nm -S -Cn`` on *binary* and return the parsed symbol table.

    ``-S`` adds the ELF size field so the gap-to-next-symbol heuristic in
    :func:`estimate_symbol_byte_size` is only needed for symbols whose ELF
    size is zero (common in older toolchains for static-local arrays).

    The returned list is sorted by VMA so callers can rely on sequential
    ordering when computing symbol gaps.

    Args:
        binary: Path to the shared object or executable.

    Returns:
        List of ``(vma, size, sym_type, demangled_name)`` tuples sorted
        by ``vma`` in ascending order.

    Raises:
        ParserError: If ``nm`` exits non-zero or the binary cannot be read.
    """
    try:
        result = run_subprocess(["nm", "-S", "-Cn", str(binary)], timeout=120)
    except subprocess.CalledProcessError as exc:
        raise ParserError(
            f"nm failed on {binary}",
            context={"binary": str(binary), "stderr": exc.stderr},
        ) from exc

    _hex = re.compile(r"^[0-9a-f]+$")
    symbols: list[_NmSymbol] = []
    for line in result.stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4:
            addr_str, size_str, sym_type, name = parts
            if not _hex.match(addr_str):
                continue
            size = int(size_str, 16) if _hex.match(size_str) else 0
            symbols.append((int(addr_str, 16), size, sym_type, name.strip()))
        elif len(parts) == 3:
            addr_str, sym_type, name = parts
            if not _hex.match(addr_str):
                continue
            symbols.append((int(addr_str, 16), 0, sym_type, name.strip()))

    symbols.sort()
    return symbols


def _run_readelf(binary: Path) -> str:
    """Run ``readelf -S --wide`` on *binary* and return the raw stdout.

    The ``--wide`` flag prevents section-name truncation and is required for
    section names longer than 17 characters.  Results should be stored in a
    :class:`BinaryCache` rather than fetched on every VMA translation.

    Args:
        binary: Path to the shared object or executable.

    Returns:
        Raw ``readelf`` output as a UTF-8 string.

    Raises:
        ParserError: If ``readelf`` exits non-zero.
    """
    try:
        result = run_subprocess(["readelf", "-S", "--wide", str(binary)], timeout=60)
    except subprocess.CalledProcessError as exc:
        raise ParserError(
            f"readelf failed on {binary}",
            context={"binary": str(binary), "stderr": exc.stderr},
        ) from exc
    return result.stdout


# ---------------------------------------------------------------------------
# BinaryCache
# ---------------------------------------------------------------------------


class BinaryCache:
    """Per-process cache for expensive ``nm`` and ``readelf`` subprocess output.

    Running ``nm`` or ``readelf`` on a 300 MB shared object takes ~1–2 seconds.
    When multiple symbols are looked up from the same binary it is critical to
    invoke each tool only once.  Storing results in an explicit class (rather
    than module-level dicts) gives callers control over the cache lifetime ---
    it can be cleared between test runs, replaced with a test double, or
    scoped to a single pipeline execution.

    A module-level default instance (:data:`_default_cache`) is provided so
    callers that omit the ``cache`` keyword argument benefit from caching
    transparently without managing a cache object themselves.

    Example:
        >>> cache = BinaryCache()
        >>> vma = find_symbol_linker_vma(binary, "SomeSym", cache=cache)
        >>> off = linker_vma_to_file_offset(binary, ".rodata", vma, cache=cache)
        >>> cache.clear()   # release memory when done with this binary
    """

    def __init__(self) -> None:
        self._nm: dict[str, list[_NmSymbol]] = {}
        self._readelf: dict[str, str] = {}

    def nm_symbols(self, binary: Path) -> list[_NmSymbol]:
        """Return the nm symbol table for *binary*, running ``nm`` if not cached.

        Args:
            binary: Path to the shared object.

        Returns:
            List of ``(vma, size, sym_type, demangled_name)`` tuples sorted by
            VMA in ascending order.  The list is shared --- do not mutate it.
        """
        key = str(binary)
        if key not in self._nm:
            self._nm[key] = _run_nm(binary)
        return self._nm[key]

    def readelf_output(self, binary: Path) -> str:
        """Return ``readelf -S --wide`` output for *binary*, running it if not cached.

        Args:
            binary: Path to the shared object.

        Returns:
            Raw ``readelf`` stdout as a string.
        """
        key = str(binary)
        if key not in self._readelf:
            self._readelf[key] = _run_readelf(binary)
        return self._readelf[key]

    def clear(self) -> None:
        """Discard all cached subprocess output, releasing the associated memory."""
        self._nm.clear()
        self._readelf.clear()

    def __len__(self) -> int:
        """Return the number of distinct binaries currently held in cache."""
        return len(self._nm)


# Module-level default cache.  Shared across the process lifetime; callers
# that do not pass an explicit `cache` argument use this instance silently.
_default_cache = BinaryCache()


# ---------------------------------------------------------------------------
# Binary introspection --- public API
# ---------------------------------------------------------------------------


def find_symbol_linker_vma(
    binary: Path,
    demangled_name: str,
    *,
    cache: BinaryCache | None = None,
) -> int:
    """Return the linker-assigned VMA of a symbol by its fully demangled name.

    Args:
        binary:         Path to the shared object.
        demangled_name: Fully demangled C++ name as printed by ``nm -C``.
        cache:          :class:`BinaryCache` instance; falls back to
                        :data:`_default_cache` when ``None``.

    Returns:
        Linker-assigned virtual address of the symbol.

    Raises:
        ParserError: If the symbol is not present in the nm output.
    """
    _cache = cache or _default_cache
    for vma, _size, _typ, name in _cache.nm_symbols(binary):
        if name == demangled_name:
            return vma
    raise ParserError(
        f"Symbol not found: {demangled_name!r} --- binary may have been stripped.",
        context={"binary": str(binary), "symbol": demangled_name},
    )


def estimate_symbol_byte_size(
    binary: Path,
    demangled_name: str,
    *,
    cache: BinaryCache | None = None,
) -> int:
    """Return the byte size of a named symbol, using progressively weaker heuristics.

    Resolution order:

    1. **ELF size from ``nm -S``** --- exact and preferred.
    2. **Gap to next symbol** --- fallback for symbols whose ELF size is zero.
    3. **SYMBOL_SIZE_FALLBACK** --- last resort.

    Args:
        binary:         Path to the shared object.
        demangled_name: Fully demangled C++ symbol name.
        cache:          :class:`BinaryCache` instance; falls back to
                        :data:`_default_cache` when ``None``.

    Returns:
        Best available estimate of the symbol's byte size.

    Raises:
        ParserError: If the symbol is not present in the nm output.
    """
    _cache = cache or _default_cache
    symbols = _cache.nm_symbols(binary)
    for i, (vma, size, _typ, name) in enumerate(symbols):
        if name != demangled_name:
            continue
        if size > 0:
            return size
        for next_vma, *_ in symbols[i + 1 :]:
            if next_vma > vma:
                return next_vma - vma
        return SYMBOL_SIZE_FALLBACK

    raise ParserError(
        f"Symbol not found: {demangled_name!r}",
        context={"binary": str(binary), "symbol": demangled_name},
    )


def linker_vma_to_file_offset(
    binary: Path,
    section: str,
    vma: int,
    *,
    cache: BinaryCache | None = None,
) -> int:
    """Translate a linker-assigned VMA to the byte offset in the on-disk file.

    Uses ``readelf -S --wide`` section headers to perform::

        file_offset = (vma - section.sh_addr) + section.sh_offset

    Args:
        binary:  Path to the shared object.
        section: ELF section name, e.g. ``".rodata"``.
        vma:     Linker-assigned virtual address to translate.
        cache:   :class:`BinaryCache` instance; falls back to
                 :data:`_default_cache` when ``None``.

    Returns:
        Byte offset of *vma* within the on-disk file.

    Raises:
        ParserError: If *section* is not present in the readelf output.
    """
    _cache = cache or _default_cache
    elf_text = _cache.readelf_output(binary)

    pat = re.compile(
        r"\[\s*\d+\]\s+" + re.escape(section) + r"\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)",
        re.IGNORECASE,
    )
    for line in elf_text.splitlines():
        m = pat.search(line)
        if m:
            section_vma = int(m.group(1), 16)
            section_foff = int(m.group(2), 16)
            return vma - section_vma + section_foff

    raise ParserError(
        f"Section {section!r} not found when converting VMA 0x{vma:x}",
        context={"binary": str(binary), "section": section, "vma": hex(vma)},
    )
