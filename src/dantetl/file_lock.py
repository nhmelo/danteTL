"""Bounded, cross-process exclusion for worker-owned Windows state files."""

import errno
import msvcrt
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def file_lock(
    path: Path,
    *,
    check: Callable[[], None] | None = None,
    busy_message: str = "",
    timeout: float = 20,
) -> Generator[None]:
    """Hold byte 0 of ``path``. ``check`` runs before each attempt and may raise to stop waiting.

    A lock still held by another process after ``timeout`` seconds raises ``OSError``, with
    ``busy_message`` when given. A crash releases the OS lock.
    """
    # Keep the sidecar across releases: replacing or deleting it splits the lock identity.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0)  # Windows permits byte-range locks beyond EOF.
        deadline = time.monotonic() + timeout
        while True:
            if check:
                check()
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EDEADLK):
                    raise
                if time.monotonic() >= deadline:
                    if busy_message:
                        raise OSError(busy_message) from None
                    raise
                time.sleep(0.05)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
