"""Bounded, atomically replaced JSON for non-secret cloud settings."""

import json
from dataclasses import dataclass
from pathlib import Path

from dantetl.atomic_file import replace_file
from dantetl.i18n import tr


@dataclass(frozen=True, slots=True)
class SettingsFile:
    max_bytes: int
    size_error: str

    def _checked(self, data: bytes) -> bytes:
        if len(data) > self.max_bytes:
            raise ValueError(tr(self.size_error))
        return data

    def read(self, path: Path) -> object:
        with path.open("rb") as stream:
            data = self._checked(stream.read(self.max_bytes + 1))
        return json.loads(data.decode("utf-8"))

    def write(self, path: Path, value: dict[str, object]) -> None:
        data = self._checked(json.dumps(value, indent=2).encode("ascii"))
        replace_file(path, data)
