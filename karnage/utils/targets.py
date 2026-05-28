"""Abstract target-backend interface and built-in backend implementations.

A :class:`TargetBackend` describes everything the pipeline needs to know about
a specific LLVM code-generation target: which CMake targets to build, where
the generated ``.inc`` files live, which nm symbol pattern identifies the
``getMnemonic`` function, and how to filter the global MVT enum down to the
types actually used by the target's ISA.

Adding support for a new backend (e.g. AMDGPU) requires only a new
:class:`TargetBackend` subclass — no changes to the pipeline core.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


class TargetBackend(ABC):
    """Abstract interface that every supported LLVM target must implement.

    Concrete subclasses are frozen dataclasses so they can be passed around
    as values without defensive copies.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short canonical name used in log messages, e.g. ``"NVPTX"``."""

    @property
    @abstractmethod
    def cmake_target_name(self) -> str:
        """Value passed to ``-DLLVM_TARGETS_TO_BUILD=<this>`` during CMake configure."""

    @property
    @abstractmethod
    def tablegen_targets(self) -> tuple[str, ...]:
        """Ordered list of ``cmake --build --target`` names to build for this backend."""

    @property
    @abstractmethod
    def matchertable_symbol(self) -> str:
        """Fully demangled C++ symbol name of the MatcherTable static array.

        Used by :func:`~karnage.utils.parser.find_symbol_linker_vma` to
        locate the MatcherTable in the binary.

        Example::

            "llvm::NVPTXDAGToDAGISel::SelectCode(llvm::SDNode*)::MatcherTable"
        """

    @property
    @abstractmethod
    def opinfo_symbol_pattern(self) -> str:
        """Regex pattern matching the ``getMnemonic`` symbol in ``nm`` output.

        Used by :func:`~karnage.utils.parser.build_opcode_mnemonic_map` to
        locate the ``OpInfo0`` and ``AsmStrs`` arrays.

        Example::

            r"NVPTXInstPrinter.*getMnemonic"
        """

    @abstractmethod
    def inc_paths(self, llvm_path: Path) -> dict[str, Path]:
        """Return a ``{logical_name: absolute_path}`` map for all required ``.inc`` files.

        Keys are stable logical names consumed by the parser:

        - ``"dagsel"``        — ``*GenDAGISel.inc``
        - ``"asmwriter"``     — ``*GenAsmWriter.inc``
        - ``"instrinfo"``     — ``*GenInstrInfo.inc``
        - ``"genvt"``         — ``GenVT.inc``
        - ``"seldagisell_h"`` — ``SelectionDAGISel.h`` (source tree, not build)

        Args:
            llvm_path: Root of the per-commit cache directory, containing
                       ``repo/`` and ``build/`` subdirectories.

        Returns:
            Dict of ``{logical_name: Path}`` for each required file.
        """

    def filter_mvt_map(self, mvt_map: dict[int, str]) -> dict[int, str]:
        """Return the subset of *mvt_map* that this backend legitimately uses.

        The global ``MVT::SimpleValueType`` enum contains types from all LLVM
        targets.  Keeping only the ISA-relevant types prevents false-positive
        ``OPC_SwitchType`` arm matches during MatcherTable scanning.

        The default implementation passes everything through.  Backends
        should override to exclude scalable-vector or other-target-specific
        types.

        Args:
            mvt_map: Full ``{enum_value: type_name}`` dict from
                     :func:`~karnage.utils.parser.parse_mvt_map`.

        Returns:
            Filtered ``{enum_value: type_name}`` dict.
        """
        return mvt_map


@dataclass(frozen=True)
class NVPTXBackend(TargetBackend):
    """Target backend for the LLVM NVPTX (NVIDIA PTX) code generator.

    Targets ``libtriton.so`` as built by OpenAI Triton, which embeds an
    NVPTX backend linked from its own LLVM build.
    """

    name:                 str           = "NVPTX"
    cmake_target_name:    str           = "NVPTX"
    tablegen_targets:     tuple[str, ...] = ("NVPTXCommonTableGen",)
    matchertable_symbol:  str           = (
        "llvm::NVPTXDAGToDAGISel::SelectCode(llvm::SDNode*)::MatcherTable"
    )
    opinfo_symbol_pattern: str          = r"NVPTXInstPrinter.*getMnemonic"

    def filter_mvt_map(self, mvt_map: dict[int, str]) -> dict[int, str]:
        """Filter the MVT map down to types valid in the PTX ISA.

        PTX scalar types: ``i1``, ``i8``, ``i16``, ``i32``, ``i64``,
        ``f16``, ``bf16``, ``f32``, ``f64``, ``i128`` (for wide ops) and
        fixed-width vector variants.

        Excluded categories:

        - Scalable vectors (``nxv*``) — AArch64 SVE / RISC-V V extension.
        - Other-target scalars: ``f80`` (x87), ``ppcf128``, ``f128``
          (quad-precision), ``i2``, ``i256``, ``i512`` (no PTX equivalent).
        - Architecture-prefixed types: ``riscv``, ``aarch64``, ``arm``,
          ``mips``, ``x86``, ``ppc``.

        Args:
            mvt_map: Full global MVT dict from
                     :func:`~karnage.utils.parser.parse_mvt_map`.

        Returns:
            Filtered dict containing only PTX-compatible MVT entries.
        """
        _EXCLUDED_PREFIXES = ("nxv", "riscv", "aarch64", "arm", "mips", "x86", "ppc")
        _EXCLUDED_EXACT    = frozenset({"i2", "f80", "f128", "ppcf128", "i256", "i512"})
        return {
            k: v for k, v in mvt_map.items()
            if v not in _EXCLUDED_EXACT
            and not any(v.startswith(p) for p in _EXCLUDED_PREFIXES)
        }

    def inc_paths(self, llvm_path: Path) -> dict[str, Path]:
        """Return paths for all NVPTX tablegen output files.

        ``SelectionDAGISel.h`` is read from the *source* tree (``repo/``),
        not the build tree, because it is a hand-written header that CMake
        does not regenerate.  All other files are generated into ``build/``.

        Args:
            llvm_path: Root of the per-commit cache directory.

        Returns:
            Dict of ``{logical_name: Path}`` for each required file.
        """
        build_path = llvm_path / "build"
        src_path   = llvm_path / "repo" / "llvm"
        return {
            "dagsel":        build_path / "lib/Target/NVPTX/NVPTXGenDAGISel.inc",
            "asmwriter":     build_path / "lib/Target/NVPTX/NVPTXGenAsmWriter.inc",
            "instrinfo":     build_path / "lib/Target/NVPTX/NVPTXGenInstrInfo.inc",
            "genvt":         build_path / "include/llvm/CodeGen/GenVT.inc",
            "seldagisell_h": src_path   / "include/llvm/CodeGen/SelectionDAGISel.h",
        }
