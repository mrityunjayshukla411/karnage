import re
import struct
import subprocess
from functools import lru_cache
from pathlib import Path

from karnage.utils.exceptions import ParserError
from karnage.utils.logger import logger
from karnage.utils.targets import TargetBackend

# ---------------------------------------------------------------------------
# nm cache — keyed by binary path; value is a sorted list of
# (linker_vma, elf_size, sym_type, demangled_name) tuples.
# Running nm on a 300 MB shared library is expensive; one run per binary.
# ---------------------------------------------------------------------------

_nm_cache: dict[str, list[tuple[int, int, str, str]]] = {}

# Cached readelf -S --wide output keyed by binary path.
# Section headers never change for a given binary; no need to re-run.
_readelf_cache: dict[str, str] = {}


def _nm_symbols(binary: Path) -> list[tuple[int, int, str, str]]:
    """
    Run `nm -S -Cn` on *binary* and return
    (linker_vma, elf_size, sym_type, demangled_name) sorted by vma.

    Results are cached per binary path so callers can query multiple
    symbols without re-invoking nm.  `-S` adds the ELF symbol size field
    (avoids the gap-to-next-symbol heuristic entirely for sized symbols).
    """
    key = str(binary)
    if key in _nm_cache:
        return _nm_cache[key]

    try:
        result = subprocess.run(
            ["nm", "-S", "-Cn", str(binary)],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ParserError(
            f"nm failed on {binary}",
            context={"binary": str(binary), "stderr": exc.stderr},
        ) from exc

    _hex = re.compile(r'^[0-9a-f]+$')
    symbols: list[tuple[int, int, str, str]] = []
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

    _nm_cache[key] = symbols
    return symbols


def find_symbol_linker_vma(binary: Path, demangled_name: str) -> int:
    """
    Return the linker-assigned VMA of a symbol by its fully demangled C++ name.

    Pass the result to linker_vma_to_file_offset() to get the byte position
    in the file on disk.

    Raises ParserError if the symbol is not found (binary may be stripped).
    """
    for vma, _size, _typ, name in _nm_symbols(binary):
        if name == demangled_name:
            return vma
    raise ParserError(
        f"Symbol not found: {demangled_name!r} — binary may have been stripped.",
        context={"binary": str(binary), "symbol": demangled_name},
    )


def estimate_symbol_byte_size(binary: Path, demangled_name: str) -> int:
    """
    Return the byte size of a symbol.

    Prefers the actual ELF symbol size from `nm -S`.  Falls back to the
    gap-to-next-symbol heuristic when the ELF size is 0 (e.g. for some
    static-local arrays in older toolchains), and to 250 000 as a last
    resort when no following symbol exists at a larger address.
    """
    symbols = _nm_symbols(binary)
    for i, (vma, size, _typ, name) in enumerate(symbols):
        if name != demangled_name:
            continue
        if size > 0:
            return size
        # Gap heuristic fallback
        for next_vma, *_ in symbols[i + 1:]:
            if next_vma > vma:
                return next_vma - vma
        return 250_000

    raise ParserError(
        f"Symbol not found: {demangled_name!r}",
        context={"binary": str(binary), "symbol": demangled_name},
    )


def _find_rodata_symbol(binary: Path, pattern: str) -> tuple[int, int]:
    """
    Find the linker VMA and ELF size of a read-only data symbol (nm type R/r)
    by regex pattern against the demangled name.

    Returns (vma, size).  Reuses the cached nm output — no extra subprocess.
    """
    pat = re.compile(pattern)
    for vma, size, sym_type, name in _nm_symbols(binary):
        if sym_type not in ('R', 'r'):
            continue
        if pat.search(name):
            return vma, size

    raise ParserError(
        f"Read-only data symbol not found (pattern={pattern!r})",
        context={"binary": str(binary), "pattern": pattern},
    )


def linker_vma_to_file_offset(binary: Path, section: str, vma: int) -> int:
    """
    Translate a linker-assigned VMA to the byte offset in the on-disk file.

    Uses `readelf -S --wide` with a column-aware regex so the result is
    not sensitive to section-name length or readelf version differences.
    The --wide flag prevents name truncation.

        file_offset = (vma - section.sh_addr) + section.sh_offset
    """
    key = str(binary)
    if key not in _readelf_cache:
        try:
            result = subprocess.run(
                ["readelf", "-S", "--wide", str(binary)],
                capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise ParserError(
                f"readelf failed on {binary}",
                context={"binary": str(binary), "stderr": exc.stderr},
            ) from exc
        _readelf_cache[key] = result.stdout

    # Match lines like:
    #   [  5] .rodata           PROGBITS  0000000001234567  00123456
    # The section name is an exact word bounded by whitespace, so
    # `.rodata.cst16` will NOT match a `.rodata` lookup.
    pat = re.compile(
        r'\[\s*\d+\]\s+' + re.escape(section) + r'\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)',
        re.IGNORECASE,
    )
    for line in _readelf_cache[key].splitlines():
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
# BuiltinOpcodes enum parsers
# ---------------------------------------------------------------------------

def parse_opcode_enum(seldagisell_h_path: Path) -> dict[str, int]:
    """
    Parse the BuiltinOpcodes enum from SelectionDAGISel.h.

    Returns {opcode_name: sequential_int_value} for every OPC_* entry,
    in enum order (starting at 0).
    """
    try:
        text = seldagisell_h_path.read_text()
    except OSError as exc:
        raise ParserError(
            f"Cannot read SelectionDAGISel.h: {exc}",
            context={"path": str(seldagisell_h_path)},
        ) from exc

    m = re.search(r'enum BuiltinOpcodes\s*\{(.*?)\};', text, re.DOTALL)
    if not m:
        raise ParserError(
            "BuiltinOpcodes enum not found in SelectionDAGISel.h",
            context={"path": str(seldagisell_h_path)},
        )

    body = re.sub(r'//[^\n]*', '', m.group(1))
    body = re.sub(r'/\*.*?\*/', '', body, flags=re.DOTALL)
    names = re.findall(r'\b(OPC_\w+)\b', body)
    return {name: idx for idx, name in enumerate(names)}


def parse_mvt_map(gen_vt_path: Path) -> dict[int, str]:
    """
    Parse GET_VT_ATTR macros from GenVT.inc and return {enum_value: type_name}.

    The macro signature is:
        GET_VT_ATTR(Ty, sz, Any, Int, FP, Vec, Sc, Tup, NF, NElem, EltTy)
    where `sz` is size-in-bits for scalars and element-count for vectors —
    NOT the enum value.  The MVT::SimpleValueType enum value is simply the
    0-based ordinal of each entry in file order (Other=0, i1=1, i2=2, ...).

    Raises ParserError if no entries are found.
    """
    try:
        text = gen_vt_path.read_text()
    except OSError as exc:
        raise ParserError(
            f"Cannot read GenVT.inc: {exc}",
            context={"path": str(gen_vt_path)},
        ) from exc

    names = re.findall(r'GET_VT_ATTR\((\w+),', text)
    if not names:
        raise ParserError(
            "No GET_VT_ATTR entries found in GenVT.inc",
            context={"path": str(gen_vt_path)},
        )

    # MVT::SimpleValueType enum starts at 1: INVALID_SIMPLE_VALUE_TYPE = 0
    # is hardcoded before the GET_VT_ATTR entries, so Other = 1, i1 = 2, ...
    return {idx: name for idx, name in enumerate(names, start=1)}


# ---------------------------------------------------------------------------
# AsmWriter opcode→mnemonic map
# ---------------------------------------------------------------------------

def _detect_asmstrs_mask(data: bytes, opinfo0_foff: int, opinfo0_size: int) -> int:
    """
    Detect whether OpInfo0 uses a 16-bit or 17-bit AsmStrs index.

    LLVM ≤ 21 stores a 16-bit index (mask 0xFFFF).
    LLVM ≥ 22 stores a 17-bit index (mask 0x1FFFF).

    Probes the entire OpInfo0 symbol (opinfo0_size bytes) so targets with
    more than 1000 opcodes don't miss a 17-bit index that appears late in
    the table.
    """
    end = opinfo0_foff + opinfo0_size
    off = opinfo0_foff
    while off + 4 <= min(end, len(data)):
        if struct.unpack_from('<I', data, off)[0] & 0x10000:
            return 0x1FFFF
        off += 4
    return 0xFFFF


def build_opcode_mnemonic_map(
    binary: Path,
    target: TargetBackend,
    max_opcodes: int = 10_000,
    *,
    data: bytes | None = None,
) -> dict[int, str]:
    """
    Build a complete {opcode_int: mnemonic_str} map by reading OpInfo0 and
    AsmStrs directly from the binary.

    Pass `data` to reuse an already-loaded binary image; otherwise the
    binary is read from disk once here.

    OpInfo0 and AsmStrs are static-local arrays of getMnemonic.  nm exposes
    them as read-only symbols with their exact VMAs, so no disassembly is
    needed.  The AsmStrs index mask (16- or 17-bit) is auto-detected from
    the OpInfo0 data itself.
    """
    if data is None:
        data = binary.read_bytes()

    base = target.opinfo_symbol_pattern
    opinfo0_vma, opinfo0_size = _find_rodata_symbol(binary, base + r".*::OpInfo0")
    asmstrs_vma, _            = _find_rodata_symbol(binary, base + r".*::AsmStrs")
    logger.debug(f"OpInfo0 VMA=0x{opinfo0_vma:x}  size={opinfo0_size}  AsmStrs VMA=0x{asmstrs_vma:x}")

    opinfo0_foff = linker_vma_to_file_offset(binary, ".rodata", opinfo0_vma)
    asmstrs_foff = linker_vma_to_file_offset(binary, ".rodata", asmstrs_vma)

    asmstrs_mask = _detect_asmstrs_mask(data, opinfo0_foff, opinfo0_size)
    logger.debug(f"AsmStrs mask=0x{asmstrs_mask:x}")

    result: dict[int, str] = {}
    for opcode in range(max_opcodes):
        entry_off = opinfo0_foff + opcode * 4
        if entry_off + 4 > len(data):
            break
        val = struct.unpack_from('<I', data, entry_off)[0]
        low = val & asmstrs_mask
        if low == 0:
            continue
        str_off = asmstrs_foff + low - 1
        if str_off >= len(data):
            continue
        try:
            end      = data.index(b'\x00', str_off)
            mnemonic = data[str_off:end].decode('ascii', errors='replace')
            mnemonic = mnemonic.replace('\t', '').strip()
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
