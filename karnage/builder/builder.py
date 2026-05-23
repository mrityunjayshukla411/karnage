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
import subprocess
from pathlib import Path
import shutil


_LLVM_VERSION_RE = re.compile(
    r'LLVM version \d+\.\d+\.[^ ]+\s+\(([0-9a-fA-F]{40})\)'
)

CACHE_ROOT = Path.cwd() / ".karnage_cache"

def _get_target_library_path(target_lib: str ="triton") -> Path:
    """
    Extracts the absolute path of the target library using pip show
    
    :param target_lib: Description
    :type target_lib: str
    :return: Description
    :rtype: Path
    """
    try:
        result = subprocess.run(
            ["pip","show",target_lib],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as exc:
        raise LibraryNotFoundError(
            f"Target library ({target_lib}) not found",
            context={"path": str(target_lib), "stderr": exc.stderr}
        )
    
    location = next(
    line.split(":", 1)[1].strip()
    for line in result.stdout.splitlines()
    if line.startswith("Location:")
    )

    return Path(location) / "triton" / "_C" / "libtriton.so"


def _extract_binary_hash(libtriton_path: Path) -> str:
    """
    Extracts the LLVM commit hash used to build a particular
    version of triton 
    
    :param libtriton_path: Description
    :type libtriton_path: Path
    :return: Description
    :rtype: str
    """

    try:
        result = subprocess.run(
            ["strings" , str(libtriton_path)],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE, 
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


def build_llvm(commit_hash: str,
               target: TargetBackend,
               force_rebuild: bool = False,
    )-> Path:
    """
    Clone and build llvm project from a specific commit hash and only for
    specific build targets
    """

    llvm_cache_dir = CACHE_ROOT / f"llvm-{commit_hash}"
    repo_dir = llvm_cache_dir / "repo"
    build_dir = llvm_cache_dir / "build"

    def _is_cache_build_valid() -> bool:
        """Returns True if build_dir and all required .inc files exist."""
        return build_dir.exists() and all(
            p.exists() for p in target.inc_paths(llvm_cache_dir).values()
        )


    if force_rebuild and llvm_cache_dir.exists():
        logger.info(f"Deleting the cache for a force rebuild {llvm_cache_dir}.")
        shutil.rmtree(llvm_cache_dir)
    
    if _is_cache_build_valid() and not force_rebuild:
        logger.info(f"Build for commit_hash={commit_hash} already exists — skipping build.")

        return build_dir
        
    llvm_cache_dir.mkdir(parents=True, exist_ok=True)

    #///=== Download source code for the specific commit hash ===============///
    
    archive_path = llvm_cache_dir / f"{commit_hash}.zip"

    if not repo_dir.exists():
        logger.info("Downloading the llvm-project")
        try:
            subprocess.run(
                ["wget", f"https://github.com/llvm/llvm-project/archive/{commit_hash}.zip"],
                cwd=str(llvm_cache_dir),
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise LLVMProjectDownloadError(
                "wget failed to download LLVM source code",
                context={"stderr": exc.stderr},
            ) from exc

        logger.info("Extracting the llvm-project archive")
        try:
            subprocess.run(
                ["unzip", "-q", str(archive_path)],
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

    #///=== CMake configure ====================================================///

    build_dir.mkdir(parents=True, exist_ok=True)
    cmake_cache = build_dir / "CMakeCache.txt"

    if not cmake_cache.exists():
        logger.info(f"Configuring LLVM build for target {target.cmake_target_name}")
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

    #///=== CMake build (tablegen targets only) ================================///

    logger.info(f"Building LLVM tablegen targets for {target.name}")
    for tgt in target.tablegen_targets:
        try:
            subprocess.run(
                ["cmake", "--build", str(build_dir), "--target", tgt, "--parallel"],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise LLVMProjectBuildError(
                f"cmake --build failed for target '{tgt}'",
                context={"target": tgt, "stderr": exc.stderr},
            ) from exc

    logger.success(f"LLVM build complete: {build_dir}")
    return build_dir

location = _get_target_library_path("triton")
if isinstance(location,Path):
    logger.success(f"Found target library path at {location}")
    commit_hash = _extract_binary_hash(location)
    if isinstance(commit_hash,str):
        logger.success(f"Hash found: {commit_hash}")
        build_dir = build_llvm(commit_hash, NVPTXBackend())

        if isinstance(build_dir,Path):
            logger.success(f"LLVM project downloaded and build successfully")
        else:
            logger.error("Failed to build llvm project")


    else:
        logger.error("NO HASH")
else:
    logger.error("Target library not found")


