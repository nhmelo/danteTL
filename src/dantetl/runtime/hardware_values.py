"""Hardware identity and pure advice; detection stays at the adapter boundary."""

from dataclasses import dataclass
from typing import Protocol

from dantetl.i18n import tr


@dataclass(frozen=True, slots=True)
class CudaDevice:
    driver: int
    major: int
    minor: int
    uuid: str

    @property
    def supported(self) -> bool:
        return self.driver >= 13000 and (self.major, self.minor) >= (7, 5)


@dataclass(frozen=True, slots=True)
class VulkanDevice:
    uuid: str
    index: int


@dataclass(frozen=True, slots=True)
class Adapter:
    name: str
    vendor: int
    dedicated_bytes: int = 0
    cuda: CudaDevice | None = None
    vulkan: VulkanDevice | None = None
    primary: bool = False

    def gpu_id(self, profile: str) -> str:
        if profile == "nvidia" and self.cuda and self.cuda.supported:
            return self.cuda.uuid
        if profile == "amd" and self.vulkan:
            return self.vulkan.uuid
        return ""


@dataclass(frozen=True, slots=True)
class HardwareStatus:
    adapters: tuple[Adapter, ...] = ()
    cuda_ocr: bool = False
    llama_device: str | None = None
    error: str = ""
    active_gpu: str = ""


def compatible_adapters(profile: str, adapters: tuple[Adapter, ...]) -> tuple[Adapter, ...]:
    """Offer the highest-memory compatible card first, retaining DXGI order for ties."""
    return tuple(
        sorted((a for a in adapters if a.gpu_id(profile)), key=lambda a: -a.dedicated_bytes)
    )


def recommended_profile(adapters: tuple[Adapter, ...]) -> str:
    supported = [a for a in adapters if a.vulkan or (a.cuda and a.cuda.supported)]
    if not supported:
        return "cpu"
    best = max(supported, key=lambda a: (a.dedicated_bytes, bool(a.cuda and a.cuda.supported)))
    return "nvidia" if best.cuda and best.cuda.supported else "amd"


def compatibility_issue(profile: str, adapters: tuple[Adapter, ...]) -> str:
    if profile == "nvidia":
        if any(a.cuda and a.cuda.supported for a in adapters):
            return ""
        for adapter in adapters:
            if cuda := adapter.cuda:
                return tr(
                    "CUDA requires compute capability 7.5+ and a CUDA 13+ driver. "
                    "{name}: compute capability {capability}, driver CUDA {version}.",
                    name=adapter.name,
                    capability=f"{cuda.major}.{cuda.minor}",
                    version=f"{cuda.driver // 1000}.{cuda.driver % 1000 // 10}",
                )
        return tr("No compatible CUDA device was detected.")
    if profile == "amd" and not any(a.vulkan for a in adapters):
        return tr("No GPU with Vulkan 1.2, compute queues and 16-bit storage was detected.")
    return ""


class HardwareReader(Protocol):
    def snapshot(self) -> HardwareStatus | None: ...
