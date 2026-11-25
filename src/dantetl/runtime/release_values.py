"""Portable release identity and assets shared with update presentation."""

from dataclasses import dataclass
from typing import Literal

REPOSITORY = "https://github.com/nhmelo/dantejp"
ISSUES_URL = "https://github.com/nhmelo/danteTL/issues"
ARCHIVE = "DanteTL-windows-x64.zip"
MANIFEST = "DanteTL-update.json"
SIGNATURE = MANIFEST + ".sig"


@dataclass(frozen=True, slots=True)
class UpdateProgress:
    stage: Literal["preparing", "downloading", "verifying", "unpacking", "preparing_restart"]
    percent: int | None = None


@dataclass(frozen=True)
class Release:
    version: str
    archive_size: int
    manifest_size: int
    signature_size: int

    @property
    def notes(self) -> str:
        return f"{REPOSITORY}/releases/tag/v{self.version}"

    def asset(self, name: str) -> str:
        return f"{REPOSITORY}/releases/download/v{self.version}/{name}"

    @property
    def signed_assets(self) -> bool:
        return (
            0 < self.archive_size <= 2**31
            and 0 < self.manifest_size <= 4096
            and self.signature_size == 64
        )
