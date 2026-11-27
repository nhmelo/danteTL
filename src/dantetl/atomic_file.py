"""Atomic replacement of small worker-owned state files."""

import os
from pathlib import Path
from uuid import uuid4


def replace_file(path: Path, data: bytes) -> None:
    """Publish the complete contents or leave the previous file; never leave a temporary file.

    A unique sibling name keeps concurrent writers and processes from sharing a temporary file.
    The contents reach the disk before the rename, so a crash or power loss cannot publish
    an empty or partial file under the final name.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
