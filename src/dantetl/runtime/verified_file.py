"""Publish one pinned asset; interrupted transfers keep their exact content-addressed prefix."""

import hashlib
import os
import threading
from collections.abc import Callable
from pathlib import Path

from dantetl.file_lock import file_lock
from dantetl.runtime.background_io import background_io
from dantetl.runtime.setup_values import Cancelled


def check_cancel(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise Cancelled()


def matches(
    path: Path,
    digest: str,
    size: int,
    cancel: threading.Event,
    *,
    progress: Callable[[int], None] | None = None,
) -> bool:
    if not path.is_file() or (size and path.stat().st_size != size):
        return False
    checksum = hashlib.sha256()
    with background_io(), path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            check_cancel(cancel)
            checksum.update(chunk)
            if progress is not None:
                progress(stream.tell())
    return checksum.hexdigest() == digest


def install_file(
    destination: Path,
    digest: str,
    size: int,
    cancel: threading.Event,
    receive: Callable[[Path], None],
    mismatch: str,
) -> bool:
    """Return whether this call published the file. The lock spans resume through rename."""
    if matches(destination, digest, size, cancel):
        return False
    # Serialize writers of different revisions destined for the same model directory, too.
    with file_lock(destination.parent / ".download.lock", check=lambda: check_cancel(cancel)):
        if matches(destination, digest, size, cancel):
            return False
        partial = destination.with_name(f".{digest}.part")
        if partial.is_symlink():
            partial.unlink()
        if not matches(partial, digest, size, cancel):
            if size and partial.exists() and partial.stat().st_size >= size:
                partial.unlink()
            try:
                receive(partial)
            finally:
                if partial.exists() and partial.stat().st_size == 0:
                    partial.unlink()
            check_cancel(cancel)
            if not matches(partial, digest, size, cancel):
                if not size or (partial.exists() and partial.stat().st_size >= size):
                    partial.unlink(missing_ok=True)
                raise RuntimeError(mismatch)
        check_cancel(cancel)
        os.replace(partial, destination)
        return True
