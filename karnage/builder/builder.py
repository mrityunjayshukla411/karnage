from karnage.utils.logger import logger
from karnage.utils.exceptions import (
    LibraryNotFoundError,
    CommitResolutionError,
    LLVMProjectDownloadError,
    LLVMProjectBuildError,
)

from karnage.utils.targets import (
    TargetBackend,
    NVPTXBackend
)

import re
import shutil
import subprocess
from pathlib import Path


_LLVM_VERSION_RE = re.compile(
    r'LLVM version \d+\.\d+\.[^ ]+\s+\(([0-9a-fA-F]{40})\)'
)

# Evaluated lazily so callers that import this module from a different working
# directory (e.g. a test suite) get the correct path.
def _cache_root() -> Path:
    return Path.cwd() / ".karnage_cache"


def _get_target_library_path(target_lib: str = "triton") -> Path:
    """
    Return the absolute path of the target library's shared object using
    `pip show`.

    The returned path is constructed from the package's install Location plus
    the canonical in-package sub-path for the given library name.
    """
    try:
        result = subprocess.run(
            ["pip", "show", target_lib],
            capture_output=True,
            text=True,
            check=True,
        )
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

    # Construct the path using target_lib rather than a hardcoded "triton"
    # so that callers passing a different package name get the right result.
    return Path(location) / target_lib / "_C" / f"lib{target_lib}.so"


def _extract_binary_hash(libtriton_path: Path) -> str:
    """
    Extract the LLVM commit hash embedded in the binary via the version string.
    """
    try:
        result = subprocess.run(
            ["strings", str(libtriton_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=True,
        )
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
    """
    Download *url* into *dest_dir*, trying curl first then wget.

    curl is available on macOS by default; wget on most Linux distros.
    Using `-f` / `--fail` ensures a non-zero exit on HTTP errors.
    """
    filename = url.rsplit("/", 1)[-1]

    if shutil.which("curl"):
        cmd = ["curl", "-fsSL", "-o", filename, url]
    elif shutil.which("wget"):
        cmd = ["wget", "-q", "-O", filename, url]
    else:
        raise LLVMProjectDownloadError(
            "Neither curl nor wget found — cannot download LLVM source",
            context={"url": url},
        )

    try:
        subprocess.run(cmd, cwd=str(dest_dir), check=True)
    except subprocess.CalledProcessError as exc:
        raise LLVMProjectDownloadError(
            f"Download failed: {url}",
            context={"url": url, "stderr": exc.stderr},
        ) from exc


def build_llvm(
    commit_hash:   str,
    target:        TargetBackend,
    force_rebuild: bool = False,
) -> Path:
    """
    Clone and build the LLVM project at a specific commit, generating only
    the tablegen targets required by *target*.

    Returns the build directory path.
    """
    llvm_cache_dir = _cache_root() / f"llvm-{commit_hash}"
    repo_dir  = llvm_cache_dir / "repo"
    build_dir = llvm_cache_dir / "build"

    def _is_cache_valid() -> bool:
        return build_dir.exists() and all(
            p.exists() for p in target.inc_paths(llvm_cache_dir).values()
        )

    if force_rebuild and llvm_cache_dir.exists():
        logger.info(f"Deleting cache for force-rebuild: {llvm_cache_dir}")
        shutil.rmtree(llvm_cache_dir)

    if _is_cache_valid() and not force_rebuild:
        logger.info(f"Build for {commit_hash} already cached — skipping.")
        return build_dir

    llvm_cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Download source archive
    # ------------------------------------------------------------------ #
    archive_name = f"{commit_hash}.zip"
    archive_path = llvm_cache_dir / archive_name

    if not repo_dir.exists():
        url = f"https://github.com/llvm/llvm-project/archive/{commit_hash}.zip"
        logger.info(f"Downloading LLVM source: {url}")
        _download_archive(url, llvm_cache_dir)

        logger.info("Extracting LLVM archive")
        try:
            subprocess.run(
                ["unzip", "-q", archive_name],
                cwd=str(llvm_cache_dir),
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise LLVMProjectDownloadError(
                "Failed to extract LLVM source archive",
                context={"archive": str(archive_path), "stderr": exc.stderr},
            ) from exc

        (llvm_cache_dir / f"llvm-project-{commit_hash}").rename(repo_dir)
        archive_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # CMake configure
    # ------------------------------------------------------------------ #
    build_dir.mkdir(parents=True, exist_ok=True)
    cmake_cache = build_dir / "CMakeCache.txt"

    if not cmake_cache.exists():
        logger.info(f"Configuring LLVM build for {target.cmake_target_name}")
        try:
            subprocess.run(
                [
                    "cmake",
                    "-S", str(repo_dir / "llvm"),
                    "-B", str(build_dir),
                    "-DCMAKE_BUILD_TYPE=Release",
                    f"-DLLVM_TARGETS_TO_BUILD={target.cmake_target_name}",
                    "-DLLVM_ENABLE_PROJECTS=",
                    "-DLLVM_INCLUDE_TESTS=OFF",
                    "-DLLVM_INCLUDE_EXAMPLES=OFF",
                    "-DLLVM_INCLUDE_BENCHMARKS=OFF",
                    "-DLLVM_INCLUDE_DOCS=OFF",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise LLVMProjectBuildError(
                "CMake configuration failed",
                context={"build_dir": str(build_dir), "stderr": exc.stderr},
            ) from exc

    # ------------------------------------------------------------------ #
    # Build (tablegen targets only)
    # ------------------------------------------------------------------ #
    logger.info(f"Building LLVM tablegen targets for {target.name}")
    for tgt in target.tablegen_targets:
        try:
            subprocess.run(
                ["cmake", "--build", str(build_dir), "--target", tgt, "--parallel"],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise LLVMProjectBuildError(
                f"cmake --build failed for target {tgt!r}",
                context={"target": tgt, "stderr": exc.stderr},
            ) from exc

    logger.success(f"LLVM build complete: {build_dir}")
    return build_dir
