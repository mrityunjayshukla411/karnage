"""LLVM source-tree download and build orchestration.

This module handles the three-step process of preparing the LLVM headers
that the extractor needs:

1. **Detect** --- extract the LLVM commit hash embedded in the target binary
   (``libtriton.so``) via ``strings``.
2. **Download** --- fetch the matching ``llvm-project`` source archive from
   GitHub if not already cached.
3. **Build** --- run CMake to generate only the tablegen ``.inc`` files required
   by the target backend; all other LLVM build targets are disabled.

Results are cached under ``.karnage_cache/llvm-<commit>/`` relative to the
current working directory.  Subsequent calls with the same commit hash skip
the download and build steps entirely.
"""

import re
import shutil
import subprocess
from pathlib import Path

from karnage.utils.exceptions import (
    CommitResolutionError,
    LibraryNotFoundError,
    LLVMProjectBuildError,
    LLVMProjectDownloadError,
)
from karnage.utils.logger import logger
from karnage.utils.subprocess_runner import run_subprocess
from karnage.utils.targets import TargetBackend

_LLVM_VERSION_RE = re.compile(r"LLVM version \d+\.\d+\.[^ ]+\s+\(([0-9a-fA-F]{40})\)")


def _cache_root() -> Path:
    """Return the root directory for per-commit LLVM build caches.

    Evaluated lazily so callers that import this module from a non-standard
    working directory (e.g. a test suite) get the correct absolute path.

    Returns:
        ``<cwd>/.karnage_cache`` as an absolute :class:`~pathlib.Path`.
    """
    return Path.cwd() / ".karnage_cache"


def _get_target_library_path(target_lib: str = "triton") -> Path:
    """Auto-detect the shared-object path for an installed Python package.

    Runs ``pip show <target_lib>`` to find the package install location, then
    constructs the canonical in-package path
    ``<Location>/<target_lib>/_C/lib<target_lib>.so``.

    Args:
        target_lib: Python package name, e.g. ``"triton"``.

    Returns:
        Absolute path to the shared object (the file may not yet exist if the
        package is installed but the build artefact is missing).

    Raises:
        LibraryNotFoundError: If ``pip show`` returns a non-zero exit code,
            meaning the package is not installed in the current environment.
    """
    try:
        result = run_subprocess(["pip", "show", target_lib], timeout=30)
    except subprocess.CalledProcessError as exc:
        raise LibraryNotFoundError(
            f"Target library ({target_lib!r}) not found",
            context={"package": target_lib, "stderr": exc.stderr},
        )

    location = next(
        line.split(":", 1)[1].strip()
        for line in result.stdout.splitlines()
        if line.startswith("Location:")
    )
    return Path(location) / target_lib / "_C" / f"lib{target_lib}.so"


def _extract_binary_hash(libtriton_path: Path) -> str:
    """Extract the 40-character LLVM commit hash baked into the binary.

    Searches the binary's string table (via ``strings``) for a line matching::

        LLVM version X.Y.Z (<40-hex-char commit hash>)

    This version string is embedded by the LLVM build system and is present
    in every non-stripped ``libtriton.so`` build.

    Args:
        libtriton_path: Path to the shared object to inspect.

    Returns:
        40-character lowercase hexadecimal LLVM commit hash.

    Raises:
        CommitResolutionError: If ``strings`` fails or the expected pattern
            is not found (e.g. stripped binary or mismatched LLVM version).
    """
    try:
        result = run_subprocess(["strings", str(libtriton_path)], timeout=60)
    except subprocess.CalledProcessError as exc:
        raise CommitResolutionError(
            f"strings command failed on {libtriton_path}",
            context={"path": str(libtriton_path), "stderr": exc.stderr},
        ) from exc

    for line in result.stdout.splitlines():
        match = _LLVM_VERSION_RE.search(line)
        if match:
            return match.group(1)

    raise CommitResolutionError(
        "Could not find LLVM commit hash in binary",
        context={"path": str(libtriton_path)},
    )


def _download_archive(url: str, dest_dir: Path) -> None:
    """Download a URL into *dest_dir*, trying ``curl`` first then ``wget``.

    Both tools are invoked in silent/quiet mode so progress output does not
    pollute the karnage log.  ``curl -f`` and ``wget`` both return non-zero on
    HTTP errors (4xx / 5xx), which is propagated as a
    :exc:`~karnage.utils.exceptions.LLVMProjectDownloadError`.

    Args:
        url:      Full HTTPS URL to download.
        dest_dir: Directory to write the downloaded file into; the filename is
                  taken from the last path component of *url*.

    Raises:
        LLVMProjectDownloadError: If neither ``curl`` nor ``wget`` is found on
            PATH, or if the chosen tool exits non-zero.
    """
    filename = url.rsplit("/", 1)[-1]

    if shutil.which("curl"):
        cmd = ["curl", "-fsSL", "-o", filename, url]
    elif shutil.which("wget"):
        cmd = ["wget", "-q", "-O", filename, url]
    else:
        raise LLVMProjectDownloadError(
            "Neither curl nor wget found --- cannot download LLVM source",
            context={"url": url},
        )

    try:
        run_subprocess(cmd, cwd=dest_dir, timeout=1800)
    except subprocess.CalledProcessError as exc:
        raise LLVMProjectDownloadError(
            f"Download failed: {url}",
            context={"url": url, "stderr": exc.stderr},
        ) from exc


def build_llvm(
    commit_hash: str,
    target: TargetBackend,
    force_rebuild: bool = False,
) -> Path:
    """Download and build the LLVM tablegen targets for a specific commit.

    The build is cached under ``.karnage_cache/llvm-<commit>/`` and skipped
    on subsequent calls unless *force_rebuild* is set.  Only the tablegen
    targets listed in :attr:`~TargetBackend.tablegen_targets` are compiled;
    all other LLVM build targets (tools, tests, examples, docs) are disabled
    to keep build times short.

    Directory layout::

        .karnage_cache/
          llvm-<commit>/
            repo/      ← extracted llvm-project source
            build/     ← CMake build tree (only tablegen targets built)

    Args:
        commit_hash:   40-character LLVM commit hash, as returned by
                       :func:`_extract_binary_hash`.
        target:        Backend descriptor that provides the CMake target name
                       and the list of tablegen targets to build.
        force_rebuild: When ``True``, delete any existing cache entry and
                       rebuild from scratch.

    Returns:
        Path to the CMake build directory (``…/llvm-<commit>/build``).

    Raises:
        LLVMProjectDownloadError: If the source archive cannot be fetched or
            extracted.
        LLVMProjectBuildError:    If CMake configuration or any tablegen build
            target fails.
    """
    llvm_cache_dir = _cache_root() / f"llvm-{commit_hash}"
    repo_dir = llvm_cache_dir / "repo"
    build_dir = llvm_cache_dir / "build"

    def _is_cache_valid() -> bool:
        return build_dir.exists() and all(
            p.exists() for p in target.inc_paths(llvm_cache_dir).values()
        )

    if force_rebuild and llvm_cache_dir.exists():
        logger.info(f"Deleting cache for force-rebuild: {llvm_cache_dir}")
        shutil.rmtree(llvm_cache_dir)

    if _is_cache_valid() and not force_rebuild:
        logger.info(f"Build for {commit_hash} already cached --- skipping.")
        return build_dir

    llvm_cache_dir.mkdir(parents=True, exist_ok=True)

    # --- Download ---
    archive_name = f"{commit_hash}.zip"
    archive_path = llvm_cache_dir / archive_name

    if not repo_dir.exists():
        url = f"https://github.com/llvm/llvm-project/archive/{commit_hash}.zip"
        logger.info(f"Downloading LLVM source: {url}")
        _download_archive(url, llvm_cache_dir)

        logger.info("Extracting LLVM archive")
        try:
            run_subprocess(
                ["unzip", "-q", archive_name], cwd=llvm_cache_dir, timeout=300
            )
        except subprocess.CalledProcessError as exc:
            raise LLVMProjectDownloadError(
                "Failed to extract LLVM source archive",
                context={"archive": str(archive_path), "stderr": exc.stderr},
            ) from exc

        (llvm_cache_dir / f"llvm-project-{commit_hash}").rename(repo_dir)
        archive_path.unlink(missing_ok=True)

    # --- CMake configure ---
    build_dir.mkdir(parents=True, exist_ok=True)
    cmake_cache = build_dir / "CMakeCache.txt"

    if not cmake_cache.exists():
        logger.info(f"Configuring LLVM build for {target.cmake_target_name}")
        try:
            run_subprocess(
                [
                    "cmake",
                    "-S",
                    str(repo_dir / "llvm"),
                    "-B",
                    str(build_dir),
                    "-DCMAKE_BUILD_TYPE=Release",
                    f"-DLLVM_TARGETS_TO_BUILD={target.cmake_target_name}",
                    "-DLLVM_ENABLE_PROJECTS=",
                    "-DLLVM_INCLUDE_TESTS=OFF",
                    "-DLLVM_INCLUDE_EXAMPLES=OFF",
                    "-DLLVM_INCLUDE_BENCHMARKS=OFF",
                    "-DLLVM_INCLUDE_DOCS=OFF",
                ],
                timeout=600,
            )
        except subprocess.CalledProcessError as exc:
            raise LLVMProjectBuildError(
                "CMake configuration failed",
                context={"build_dir": str(build_dir), "stderr": exc.stderr},
            ) from exc

    # --- Build tablegen targets ---
    logger.info(f"Building LLVM tablegen targets for {target.name}")
    for tgt in target.tablegen_targets:
        try:
            run_subprocess(
                ["cmake", "--build", str(build_dir), "--target", tgt, "--parallel"],
            )
        except subprocess.CalledProcessError as exc:
            raise LLVMProjectBuildError(
                f"cmake --build failed for target {tgt!r}",
                context={"target": tgt, "stderr": exc.stderr},
            ) from exc

    logger.success(f"LLVM build complete: {build_dir}")
    return build_dir
