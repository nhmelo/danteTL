import threading
from unittest.mock import Mock

import pytest

from dantetl.application.setup_status import acceleration_advice
from dantetl.runtime.hardware import HardwareCheck
from dantetl.runtime.hardware_values import Adapter, HardwareStatus

NVIDIA = Adapter("NVIDIA fixture", 0x10DE, 16 * 2**30)
AMD = Adapter("AMD fixture", 0x1002, 8 * 2**30)


@pytest.mark.parametrize(
    ("hardware", "device", "provider", "expected", "destination"),
    [
        (None, "cpu", "", "", "runtime"),
        (HardwareStatus(), "cpu", "", "", "runtime"),
        (HardwareStatus((NVIDIA, AMD)), "cpu", "", "", "runtime"),
        (HardwareStatus((NVIDIA,), True, "cuda"), "cpu", "", "", "runtime"),
        (HardwareStatus((NVIDIA,), False, "vulkan"), "auto", "CPUExecutionProvider", "", "runtime"),
        (HardwareStatus((NVIDIA,), True, "cuda"), "auto", "CUDAExecutionProvider", "", "runtime"),
        (
            HardwareStatus((NVIDIA,), True, "cuda"),
            "auto",
            "CPUExecutionProvider",
            "fell back",
            "runtime",
        ),
        (HardwareStatus((AMD,)), "cpu", "", "", "runtime"),
        (HardwareStatus((AMD,), False, "vulkan"), "cpu", "CPUExecutionProvider", "", "runtime"),
        (HardwareStatus(error="DXGI unavailable"), "cpu", "", "could not be checked", "runtime"),
    ],
)
def test_acceleration_advice_does_not_treat_amd_cpu_ocr_or_unknown_hardware_as_failure(
    hardware, device, provider, expected, destination
):
    advice = acceleration_advice(
        hardware, device, actual_provider=provider, fallback="probe failed"
    )
    assert expected in advice.message if expected else not advice.message
    assert advice.destination == destination
    if "fell back" in expected:
        assert "probe failed" in advice.message


def test_hardware_check_is_once_off_thread_and_failure_is_reported(monkeypatch):
    threads = []
    entered, release = threading.Event(), threading.Event()

    def detect():
        threads.append(threading.get_ident())
        entered.set()
        assert release.wait(2)
        return (NVIDIA,)

    monkeypatch.setattr("dantetl.runtime.hardware.detect_adapters", detect)
    monkeypatch.setattr("dantetl.runtime.llama.installed_device", lambda: "cuda")
    monkeypatch.setattr("dantetl.runtime.ocr.gpu_runtime_installed", lambda: True)
    check = HardwareCheck()
    check.start()
    assert entered.wait(1) and check.snapshot() is None
    release.set()
    check._thread.join(2)
    assert check.snapshot() == HardwareStatus((NVIDIA,), True, "cuda")
    assert threads == [check._thread.ident] and threads[0] != threading.get_ident()
    monkeypatch.setattr(
        "dantetl.runtime.hardware.detect_adapters", Mock(side_effect=OSError("DXGI"))
    )
    failed = HardwareCheck()
    failed.start()
    failed._thread.join(2)
    status = failed.snapshot()
    assert failed.finished and status is not None and status.error == "DXGI"


@pytest.mark.parametrize("cuda_crashes", [False, True])
def test_driver_evidence_matches_adapter_identity_and_survives_other_driver_failure(
    monkeypatch, cuda_crashes
):
    from dantetl.runtime import hardware
    from dantetl.runtime.hardware_values import CudaDevice, recommended_profile

    monkeypatch.setattr(
        hardware,
        "_windows_adapters",
        lambda: {
            "000000000000000b": AMD,
            "000000000000000a": NVIDIA,
        },
    )

    def probe(command, root, environment, cancel, receive, *, timeout):
        if command[-1] == "nvidia":
            receive("CUDA 000000000000000a 13000 6 1")
            if cuda_crashes:
                raise RuntimeError("driver exited")
        else:
            receive("Vulkan 000000000000000b")
            receive("Vulkan 000000000000000c")  # Not a detected hardware adapter.

    monkeypatch.setattr(hardware, "run_setup_command", probe)
    amd, nvidia = hardware.detect_adapters()
    assert amd.vulkan and not nvidia.vulkan
    assert nvidia.cuda == (None if cuda_crashes else CudaDevice(13000, 6, 1))
    assert recommended_profile((amd, nvidia)) == "amd"
