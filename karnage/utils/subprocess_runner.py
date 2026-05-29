"""Thin subprocess wrapper used throughout the karnage pipeline.

All subprocess calls in the pipeline should go through :func:`run_subprocess`
rather than calling :func:`subprocess.run` directly.  This provides:

- A consistent ``capture_output=True / text=True`` call signature.
- Debug-level logging of every command before it runs.
- An optional *timeout* that prevents runaway processes from hanging
  indefinitely (e.g. a stalled ``nm`` on a network-mounted filesystem).
"""

import subprocess
from pathlib import Path


def run_subprocess(
    cmd: list[str | Path],
    *,
    cwd: Path | str | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command, capturing its output with an optional timeout.

    Always passes ``capture_output=True`` and ``text=True`` so callers
    receive ``stdout`` / ``stderr`` as strings on the returned
    :class:`subprocess.CompletedProcess` object.

    Wrapping the resulting :exc:`subprocess.CalledProcessError` into the
    appropriate :exc:`~karnage.utils.exceptions.KarnageError` subclass is
    the **caller's** responsibility --- this function deliberately stays
    domain-agnostic.

    Args:
        cmd:     Command and arguments.  :class:`~pathlib.Path` objects are
                 converted to strings automatically.
        cwd:     Working directory for the child process.  ``None`` inherits
                 the current working directory.
        env:     Full environment for the child process.  ``None`` inherits
                 the calling process's environment.
        timeout: Maximum seconds to wait for the process to exit.  ``None``
                 means no limit.  Raises :exc:`subprocess.TimeoutExpired` if
                 the process outlasts the limit.

    Returns:
        Completed process with ``stdout`` and ``stderr`` as strings.

    Raises:
        subprocess.CalledProcessError: If the process exits with a non-zero
            return code.
        subprocess.TimeoutExpired:     If *timeout* is set and the process
            does not finish in time.
        FileNotFoundError:             If the executable is not found on PATH.
    """
    from karnage.utils.logger import logger  # local import avoids circular dep at module load

    str_cmd = [str(c) for c in cmd]
    logger.debug(f"$ {' '.join(str_cmd)}")
    return subprocess.run(
        str_cmd,
        capture_output=True,
        text=True,
        check=True,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        timeout=timeout,
    )
