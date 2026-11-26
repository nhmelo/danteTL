"""Runtime child environments and cancellable setup subprocesses with owned cleanup."""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path

from dantetl.i18n import tr
from dantetl.runtime.owned_process import OwnedProcess
from dantetl.runtime.setup_values import Cancelled

logger = logging.getLogger(__name__)


def clean_runtime_environment(inherited: Mapping[str, str]) -> dict[str, str]:
    """Copy inherited state without Python/uv overrides; callers add their intended settings.

    -P/-s do not ignore Python environment variables. Setup and app launch must start
    from the same clean baseline: merging a sanitized environment cannot remove leftovers.
    """
    environment = {
        key: value
        for key, value in inherited.items()
        if not key.startswith(("PYTHON", "UV_")) and key != "VIRTUAL_ENV"
    }
    # Discovery and setup children must see the user's original visibility mask,
    # even when the running app has been narrowed to one GPU.
    mask = environment.pop("DANTETL_GPU_MASK", "")
    previous = environment.pop("DANTETL_GPU_PREVIOUS_MASK", None)
    environment.pop("DANTETL_GPU_UUID", None)
    if mask in ("CUDA_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES"):
        if previous is None:
            environment.pop(mask, None)
        else:
            environment[mask] = previous
    return environment


def run_setup_command(
    command: list[str],
    root: Path,
    environment: dict[str, str],
    cancel: threading.Event,
    report: Callable[[str], None],
    *,
    timeout: float = 1800,
) -> None:
    if cancel.is_set():
        raise Cancelled()
    deadline = time.monotonic() + timeout
    process = OwnedProcess(command, root, environment, capture_output=True)
    tail: deque[str] = deque(maxlen=16)
    lock = threading.Lock()

    def read() -> None:
        assert process.stdout is not None
        while chunk := process.stdout.readline(2048):
            line = chunk.strip()
            if line:
                logger.info("Runtime setup: %s", line)
                with lock:
                    tail.append(line)
                report(line)

    reader = threading.Thread(target=read, daemon=True, name="setup-output")
    try:
        reader.start()
        while process.poll() is None:
            if cancel.wait(0.1):
                raise Cancelled()
            if time.monotonic() > deadline:
                raise RuntimeError(
                    tr("Setup timed out. Check your connection and GPU driver, then retry.")
                )
        reader.join(timeout=2)
        if cancel.is_set():
            raise Cancelled()
        if process.returncode:
            with lock:
                details = "\n".join(tail)
            raise RuntimeError(
                tr(
                    "Setup command failed ({returncode}).\n{details}",
                    returncode=process.returncode,
                    details=details,
                )
            )
    finally:
        process.close()
        if reader.ident is not None:
            reader.join(timeout=2)
        if process.stdout:
            process.stdout.close()
