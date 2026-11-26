"""Immutable setup progress and review data, independent of operations/installers."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from dantetl.i18n import tr
from dantetl.runtime.hardware_values import Adapter, compatibility_issue


class Cancelled(Exception):
    """The owning setup operation was cancelled before publication."""


@dataclass(frozen=True, slots=True)
class DownloadProgress:
    name: str
    received: int = 0
    total: int = 0


@dataclass(frozen=True, slots=True)
class ModelProgress:
    message: str = field(default_factory=lambda: tr("Starting"))
    received: int = 0
    total: int = 0
    done: bool = False
    cancelled: bool = False
    error: str = ""
    download: DownloadProgress | None = None
    stage: int = 0
    stages: int = 0
    parallel: tuple["ModelProgress", "ModelProgress"] | None = None


Progress = Callable[[ModelProgress], None]


class SetupLane(StrEnum):
    MODELS = "models"
    RUNTIME = "runtime"
    GUIDE = "guide"


@dataclass(frozen=True)
class SetupJob:
    token: int = 0
    action: str = ""
    key: str = ""
    busy: bool = False
    cancelling: bool = False
    progress: ModelProgress = field(default_factory=ModelProgress)
    gpu: str = ""


@dataclass(frozen=True)
class RuntimeTarget:
    path: Path
    gpu: str = ""


@dataclass(frozen=True)
class RuntimeStatus:
    adapters: tuple[Adapter, ...] = ()
    targets: tuple[tuple[str, Path | None], ...] = ()
    current: Path | None = None
    recommendation: str = "cpu"
    active_profile: str = ""
    active_gpu: str = ""
    saved: RuntimeTarget | None = None

    def target(self, key: str, gpu: str = "") -> RuntimeTarget | None:
        if compatibility_issue(key, self.adapters):
            return None
        if key != "cpu" and not gpu:
            return None
        if gpu and not any(a.gpu_id(key) == gpu for a in self.adapters):
            return None
        path = next((path for name, path in self.targets if name == key), None)
        return RuntimeTarget(path, gpu) if path else None

    def preferred_gpu(self, key: str) -> str:
        if self.saved and self.saved.path.name.split("-", 1)[0] == key:
            return self.saved.gpu
        return self.active_gpu if self.active_profile == key else ""

    def in_use(self, key: str, gpu: str) -> bool:
        target = self.target(key, gpu)
        active = target.path == self.current if target else key == self.active_profile
        return active and (not gpu or gpu == self.active_gpu)


@dataclass(frozen=True)
class SetupReview:
    token: int
    model: str
    runtime: str
    download_bytes: int
    install_runtime: bool
    model_keys: tuple[str, ...]
    gpu: str = ""
