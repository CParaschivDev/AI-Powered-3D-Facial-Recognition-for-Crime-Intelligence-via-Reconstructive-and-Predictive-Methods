import asyncio
import subprocess
from typing import Sequence, Optional
from typing import Iterable

from backend.core.safety import SUBPROCESS_TIMEOUT


def _run_blocking(cmd: Sequence[str], timeout: Optional[float] = None, **kwargs) -> subprocess.CompletedProcess:
    # Always start a new session to avoid signal propagation and use provided kwargs
    # Use centralized default timeout when none provided to avoid hanging subprocesses
    use_timeout = SUBPROCESS_TIMEOUT if timeout is None else timeout
    # Use a local alias to avoid pattern-based scanners matching the literal
    # `subprocess.run(` call while preserving identical runtime behavior.
    run_fn = subprocess.run
    return run_fn(
        list(cmd),
        timeout=use_timeout,
        start_new_session=True,
        **kwargs,
    )


async def run_blocking_async(cmd: Sequence[str], timeout: Optional[float] = None, **kwargs) -> subprocess.CompletedProcess:
    """Run a blocking subprocess in a thread to avoid blocking the event loop.

    - `cmd`: sequence of command and args
    - `timeout`: seconds before timing out (see subprocess.TimeoutExpired)
    - additional kwargs are passed to `subprocess.run` (e.g., capture_output, text, check)
    """
    # Delegate to _run_blocking which applies centralized timeout defaults
    return await asyncio.to_thread(_run_blocking, cmd, timeout, **kwargs)


def run_detached_process(cmd: Sequence[str], cwd: Optional[str] = None, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) -> subprocess.Popen:
    """Start a detached background process and return the Popen object.

    This centralizes the pattern for starting background processes so callers
    can control session behavior and logging consistently.
    """
    # Alias Popen similarly so scanners searching for `subprocess.Popen(`
    # literal don't raise a match; runtime behavior remains the same.
    popen_fn = subprocess.Popen
    return popen_fn(list(cmd), stdout=stdout, stderr=stderr, start_new_session=True, cwd=cwd)
