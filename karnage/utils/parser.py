"""
Binary introspection and source-file parsing utilities for the karnage pipeline.

Three separate concerns are provided here:

  Binary introspection (nm / readelf)
    BinaryCache                  --- explicit-lifecycle cache for subprocess output.
    find_symbol_linker_vma       --- look up a symbol's linker-assigned VMA.
    estimate_symbol_byte_size    --- look up or estimate a symbol's byte size.
    linker_vma_to_file_offset    --- translate a VMA to an on-disk byte offset.

  Source-file parsing
    parse_opcode_enum            --- parse BuiltinOpcodes from SelectionDAGISel.h.
    parse_mvt_map                --- parse MVT::SimpleValueType values from GenVT.inc.

  AsmWriter table reading
    build_opcode_mnemonic_map    --- build {opcode: mnemonic} by reading OpInfo0/AsmStrs.
"""

import re
import struct
import subprocess
from pathlib import Path
from typing import TypeAlias

from karnage.utils.constants import MAX_OPCODES, SYMBOL_SIZE_FALLBACK
from karnage.utils.exceptions import ParserError
from karnage.utils.logger import logger
from karnage.utils.subprocess_runner import run_subprocess
from karnage.utils.targets import TargetBackend

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
        self._nm:      dict[str, list[_NmSymbol]] = {}
        self._readelf: dict[str, str]             = {}

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

def _find_rodata_symbol(
    binary: Path,
    pattern: str,
    *,
    cache: BinaryCache | None = None,
) -> tuple[int, int]:
    """Find the linker VMA and ELF size of a read-only data symbol by regex.

    Only symbols with nm type ``R`` or ``r`` (read-only data) are considered.
    The nm output is fetched from *cache* so no extra subprocess is invoked
    when called repeatedly for the same binary.

    Args:
        binary:  Path to the shared object.
        pattern: Regular-expression pattern matched against the demangled name.
        cache:   :class:`BinaryCache` instance to use; falls back to
                 :data:`_default_cache` when ``None``.

    Returns:
        ``(vma, elf_size)`` for the first symbol whose demangled name matches
        *pattern*.

    Raises:
        ParserError: If no read-only symbol matches the pattern.
    """
    _cache = cache or _default_cache
    pat = re.compile(pattern)
    for vma, size, sym_type, name in _cache.nm_symbols(binary):
        if sym_type not in ("R", "r"):
            continue
        if pat.search(name):
            return vma, size

    raise ParserError(
        f"Read-only data symbol not found (pattern={pattern!r})",
        context={"binary": str(binary), "pattern": pattern},
    )


def find_symbol_linker_vma(
    binary: Path,
    demangled_name: str,
    *,
    cache: BinaryCache | None = None,
) -> int:
    """Return the linker-assigned VMA of a symbol by its fully demangled name.

    Pass the result to :func:`linker_vma_to_file_offset` to get the byte
    position of the symbol within the on-disk file.

    Args:
        binary:         Path to the shared object.  May be stripped; in that
                        case the symbol will not be found and an error is raised.
        demangled_name: Fully demangled C++ name as printed by ``nm -C``, e.g.
                        ``"llvm::NVPTXDAGToDAGISel::SelectCode(llvm::SDNode*)::MatcherTable"``.
        cache:          :class:`BinaryCache` instance; falls back to
                        :data:`_default_cache` when ``None``.

    Returns:
        Linker-assigned virtual address of the symbol.

    Raises:
        ParserError: If the symbol is not present in the nm output (binary may
                     be stripped or the name may be incorrect).
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
    2. **Gap to next symbol** --- fallback for symbols whose ELF size is zero
       (common for static-local arrays in older toolchains).
    3. **SYMBOL_SIZE_FALLBACK** --- last resort when no symbol with a higher
       VMA exists at all.

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
        # Gap heuristic: find the next symbol at a strictly higher address.
        for next_vma, *_ in symbols[i + 1:]:
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

    A column-aware regex is used so the result is insensitive to section-name
    length or minor readelf version differences across distributions.

    Args:
        binary:  Path to the shared object.
        section: ELF section name, e.g. ``".rodata"``.  Matched as an exact
                 word so ``".rodata.cst16"`` will *not* match a ``".rodata"``
                 query.
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

    # Match lines like:
    #   [  5] .rodata    PROGBITS  0000000001234567  00123456
    # Section name is matched as a whole word so .rodata.cst16 ≠ .rodata.
    pat = re.compile(
        r"\[\s*\d+\]\s+" + re.escape(section) + r"\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)",
        re.IGNORECASE,
    )
    for line in elf_text.splitlines():
        m = pat.search(line)
        if m:
            section_vma  = int(m.group(1), 16)
            section_foff = int(m.group(2), 16)
            return vma - section_vma + section_foff

    raise ParserError(
        f"Section {section!r} not found when converting VMA 0x{vma:x}",
        context={"binary": str(binary), "section": section, "vma": hex(vma)},
    )


# ---------------------------------------------------------------------------
# Source-file parsing
# ---------------------------------------------------------------------------

def parse_opcode_enum(seldagisell_h_path: Path) -> dict[str, int]:
    """Parse the ``BuiltinOpcodes`` enum from ``SelectionDAGISel.h``.

    Returns a ``{opcode_name: int_value}`` mapping for every ``OPC_*``
    identifier in declaration order, starting at zero.  Both line comments
    (``//``) and block comments (``/* */``) are stripped before extraction.

    Args:
        seldagisell_h_path: Absolute path to ``SelectionDAGISel.h`` in the
                            LLVM source tree (not the build tree).

    Returns:
        Dict mapping each ``OPC_`` name to its sequential integer value.

    Raises:
        ParserError: If the file cannot be read or the ``BuiltinOpcodes``
                     enum body is not found.
    """
    try:
        text = seldagisell_h_path.read_text()
    except OSError as exc:
        raise ParserError(
            f"Cannot read SelectionDAGISel.h: {exc}",
            context={"path": str(seldagisell_h_path)},
        ) from exc

    m = re.search(r"enum BuiltinOpcodes\s*\{(.*?)\};", text, re.DOTALL)
    if not m:
        raise ParserError(
            "BuiltinOpcodes enum not found in SelectionDAGISel.h",
            context={"path": str(seldagisell_h_path)},
        )

    body = re.sub(r"//[^\n]*", "", m.group(1))
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    names = re.findall(r"\b(OPC_\w+)\b", body)
    return {name: idx for idx, name in enumerate(names)}


def parse_mvt_map(gen_vt_path: Path) -> dict[int, str]:
    """Parse ``GET_VT_ATTR`` macros from ``GenVT.inc`` to build an MVT enum map.

    Each ``GET_VT_ATTR(Ty, sz, ...)`` macro defines one ``MVT::SimpleValueType``
    entry.  The integer enum value is the 1-based ordinal of each entry in
    file order, because ``INVALID_SIMPLE_VALUE_TYPE = 0`` is hardcoded before
    the macro list.

    Note: ``sz`` is bit-width for scalars and element-count for vectors ---
    **not** the enum integer value.

    Args:
        gen_vt_path: Absolute path to ``GenVT.inc`` in the LLVM build tree.

    Returns:
        Dict mapping each ``MVT::SimpleValueType`` integer to its type-name
        string, e.g. ``{1: "Other", 2: "i1", 3: "i8", ...}``.

    Raises:
        ParserError: If the file cannot be read or no ``GET_VT_ATTR`` entries
                     are found.
    """
    try:
        text = gen_vt_path.read_text()
    except OSError as exc:
        raise ParserError(
            f"Cannot read GenVT.inc: {exc}",
            context={"path": str(gen_vt_path)},
        ) from exc

    names = re.findall(r"GET_VT_ATTR\((\w+),", text)
    if not names:
        raise ParserError(
            "No GET_VT_ATTR entries found in GenVT.inc",
            context={"path": str(gen_vt_path)},
        )

    # MVT::SimpleValueType ordinals start at 1:
    # INVALID_SIMPLE_VALUE_TYPE = 0 precedes the GET_VT_ATTR entries.
    return {idx: name for idx, name in enumerate(names, start=1)}


# ---------------------------------------------------------------------------
# AsmWriter opcode → mnemonic map
# ---------------------------------------------------------------------------

def _detect_asmstrs_mask(data: bytes, opinfo0_foff: int, opinfo0_size: int) -> int:
    """Detect whether the OpInfo0 table uses a 16-bit or 17-bit AsmStrs index.

    LLVM ≤ 21 stores a 16-bit index (mask ``0xFFFF``).
    LLVM ≥ 22 stores a 17-bit index (mask ``0x1FFFF``).

    The entire OpInfo0 symbol is probed (not just the first few entries) so
    that targets with more than ~1 000 opcodes, whose 17-bit indices appear
    late in the table, are correctly detected.

    Args:
        data:          Full binary image loaded as a :class:`bytes` object.
        opinfo0_foff:  File offset of the first byte of the OpInfo0 symbol.
        opinfo0_size:  Total byte length of the OpInfo0 symbol.

    Returns:
        ``0x1FFFF`` if bit 16 is set in any entry, ``0xFFFF`` otherwise.
    """
    end = opinfo0_foff + opinfo0_size
    off = opinfo0_foff
    while off + 4 <= min(end, len(data)):
        if struct.unpack_from("<I", data, off)[0] & 0x10000:
            return 0x1FFFF
        off += 4
    return 0xFFFF


def build_opcode_mnemonic_map(
    binary: Path,
    target: TargetBackend,
    max_opcodes: int = MAX_OPCODES,
    *,
    data: bytes | None = None,
    cache: BinaryCache | None = None,
) -> dict[int, str]:
    """Build a complete ``{opcode_int: mnemonic_str}`` map from the binary.

    Reads ``OpInfo0`` and ``AsmStrs`` directly --- static-local arrays of
    ``getMnemonic`` that nm exposes as read-only symbols with exact VMAs.
    No disassembly is required.  The AsmStrs index width (16- or 17-bit) is
    auto-detected from the live OpInfo0 data so the same code handles both
    LLVM ≤ 21 and LLVM ≥ 22.

    Args:
        binary:      Path to the target shared object.
        target:      Backend descriptor used to locate the ``OpInfo0`` and
                     ``AsmStrs`` symbols via :attr:`~TargetBackend.opinfo_symbol_pattern`.
        max_opcodes: Maximum opcode index to scan; scanning stops early if the
                     OpInfo0 table is shorter than this limit.
        data:        Pre-loaded binary image.  Pass this to avoid a second
                     ``Path.read_bytes()`` call when the caller already holds it.
        cache:       :class:`BinaryCache` instance; falls back to
                     :data:`_default_cache` when ``None``.

    Returns:
        Dict mapping each opcode integer to its mnemonic string.  Opcodes
        with an empty or null mnemonic are omitted.

    Raises:
        ParserError: If ``OpInfo0`` or ``AsmStrs`` symbols cannot be located,
                     or if the resulting map is empty (likely a wrong target or
                     stripped binary).
    """
    if data is None:
        data = binary.read_bytes()

    _cache = cache or _default_cache

    base = target.opinfo_symbol_pattern
    opinfo0_vma, opinfo0_size = _find_rodata_symbol(binary, base + r".*::OpInfo0", cache=_cache)
    asmstrs_vma, _            = _find_rodata_symbol(binary, base + r".*::AsmStrs",  cache=_cache)
    logger.debug(
        f"OpInfo0 VMA=0x{opinfo0_vma:x}  size={opinfo0_size}  AsmStrs VMA=0x{asmstrs_vma:x}"
    )

    opinfo0_foff = linker_vma_to_file_offset(binary, ".rodata", opinfo0_vma, cache=_cache)
    asmstrs_foff = linker_vma_to_file_offset(binary, ".rodata", asmstrs_vma, cache=_cache)

    asmstrs_mask = _detect_asmstrs_mask(data, opinfo0_foff, opinfo0_size)
    logger.debug(f"AsmStrs mask=0x{asmstrs_mask:x}")

    result: dict[int, str] = {}
    for opcode in range(max_opcodes):
        entry_off = opinfo0_foff + opcode * 4
        if entry_off + 4 > len(data):
            break
        val = struct.unpack_from("<I", data, entry_off)[0]
        low = val & asmstrs_mask
        if low == 0:
            continue
        str_off = asmstrs_foff + low - 1
        if str_off >= len(data):
            continue
        try:
            end      = data.index(b"\x00", str_off)
            mnemonic = data[str_off:end].decode("ascii", errors="replace")
            mnemonic = mnemonic.replace("\t", "").strip()
            if mnemonic:
                result[opcode] = mnemonic
        except ValueError:
            continue

    if not result:
        raise ParserError(
            "No opcodes extracted from AsmWriter tables.",
            context={"binary": str(binary)},
        )

    logger.info(f"Opcode→mnemonic map: {len(result):,} entries")
    return result
