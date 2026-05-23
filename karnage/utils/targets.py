from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

class TargetBackend(ABC):
    """
    Abstract interface that every supported LLVM target must implement.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short canonical name, e.g. 'NVPTX' or 'AMDGPU'."""

    @property
    @abstractmethod
    def cmake_target_name(self) -> str:
        """Value passed to -DLLVM_TARGETS_TO_BUILD=<this>."""

    @property
    @abstractmethod
    def tablegen_targets(self) -> tuple[str, ...]:
        """cmake --build targets specific to this backend, built in order."""

    @property
    @abstractmethod
    def matchertable_symbol(self) -> str:
        """
        Fully demangled C++ symbol name of the MatcherTable static array.

        This is the symbol located by scanner/elf.py to find the MatcherTable
        bounds in the binary.
        """

    @abstractmethod
    def inc_paths(self, llvm_path: Path) -> dict[str, Path]:
        """
        Map logical name → absolute Path for every .inc file this target needs.

        Keys are stable logical names consumed by llvm/inc_parser.py:
          "dagsel"       — NVPTXGenDAGISel.inc
          "asmwriter"    — NVPTXGenAsmWriter.inc
          "instrinfo"    — NVPTXGenInstrInfo.inc
          "genvt"        — GenVT.inc
          "seldagisell_h"— SelectionDAGISel.h  (source tree, not build)
        """

    @property
    @abstractmethod
    def opinfo_symbol_pattern(self) -> str:
        """
        Regex pattern matching the getMnemonic (or equivalent) symbol in nm
        output.  Used by scanner/opcode_map.py to locate OpInfo0/AsmStrs.
        """

@dataclass(frozen=True)
class NVPTXBackend(TargetBackend):
    """
    TargetBackend for NVPTX
    """

    name: str = "NVPTX"
    cmake_target_name: str = "NVPTX"
    tablegen_targets: tuple[str, ...] = ("NVPTXCommonTableGen",)
    matchertable_symbol: str = "llvm::NVPTXDAGToDAGISel::SelectCode(llvm::SDNode*)::MatcherTable"
    opinfo_symbol_pattern: str = r"NVPTXInstPrinter.*getMnemonic"

    def inc_paths(self, llvm_path: Path) -> dict[str, Path]:
        """
        SelectionDAGISel.h lives in the source tree (sibling of build/).
        All other paths are under build/.
        """
        build_path = llvm_path / "build"
        src_path = llvm_path / "repo" / "llvm"
        return {
            "dagsel":        build_path / "lib/Target/NVPTX/NVPTXGenDAGISel.inc",
            "asmwriter":     build_path / "lib/Target/NVPTX/NVPTXGenAsmWriter.inc",
            "instrinfo":     build_path / "lib/Target/NVPTX/NVPTXGenInstrInfo.inc",
            "genvt":         build_path / "include/llvm/CodeGen/GenVT.inc",
            "seldagisell_h": src_path  / "include/llvm/CodeGen/SelectionDAGISel.h",
        }
