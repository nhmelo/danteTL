"""Actionable hardware advice, separate from model downloads and inference readiness."""

from dataclasses import dataclass

from dantetl.i18n import mark, tr
from dantetl.runtime.hardware_values import HardwareStatus
from dantetl.runtime.setup_values import RuntimeTarget


@dataclass(frozen=True, slots=True)
class SetupAdvice:
    message: str = ""
    destination: str = "runtime"
    action: str = mark("Set up acceleration")
    restart_target: RuntimeTarget | None = None


def acceleration_advice(
    hardware: HardwareStatus | None,
    ocr_device: str,
    *,
    actual_provider: str = "",
    fallback: str = "",
) -> SetupAdvice:
    if hardware is None:
        return SetupAdvice()
    if hardware.error:
        return SetupAdvice(
            tr(
                "Graphics hardware could not be checked. CPU remains available; "
                "you can choose acceleration manually."
            )
        )
    gpu_requested = (ocr_device == "auto" and hardware.cuda_ocr) or ocr_device == "auto-directml"
    if gpu_requested and actual_provider == "CPUExecutionProvider":
        return SetupAdvice(
            tr("GPU OCR fell back to CPU. ") + (fallback or tr("Check the acceleration setup.")),
        )
    return SetupAdvice()
