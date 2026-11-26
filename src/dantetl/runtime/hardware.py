"""Read Windows adapter identity without loading inference libraries or requiring WMI."""

import ctypes as ct
import logging
import os
import re
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import UUID

from dantetl.i18n import tr
from dantetl.paths import module_command, resource_root
from dantetl.runtime.hardware_values import (
    Adapter,
    CudaDevice,
    HardwareStatus,
    VulkanDevice,
    compatibility_issue,
    compatible_adapters,
)
from dantetl.runtime.setup_process import clean_runtime_environment, run_setup_command


class HardwareCheck:
    """One startup check; never load model/native inference libraries or install anything."""

    def __init__(self) -> None:
        self._done = threading.Event()
        self._result = HardwareStatus()
        self._thread = threading.Thread(target=self._run, name="hardware-check", daemon=True)

    def start(self) -> None:
        self._thread.start()

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    def snapshot(self) -> HardwareStatus | None:
        return self._result if self.finished else None

    def _run(self) -> None:
        try:
            from dantetl.runtime.llama import installed_device
            from dantetl.runtime.ocr import gpu_runtime_installed

            self._result = HardwareStatus(
                detect_adapters(),
                gpu_runtime_installed(),
                installed_device(),
                active_gpu=os.environ.get("DANTETL_GPU_UUID", ""),
            )
        except Exception as error:
            self._result = HardwareStatus(error=str(error))
        finally:
            self._done.set()


class _Description(ct.Structure):
    _fields_ = [
        ("name", ct.c_wchar * 128),
        ("vendor", ct.c_uint32),
        ("device", ct.c_uint32),
        ("subsystem", ct.c_uint32),
        ("revision", ct.c_uint32),
        ("video", ct.c_size_t),
        ("system", ct.c_size_t),
        ("shared", ct.c_size_t),
        ("luid_low", ct.c_uint32),
        ("luid_high", ct.c_int32),
        ("flags", ct.c_uint32),
    ]


def _method(pointer: ct.c_void_p, index: int, *arguments: Any) -> Any:
    table = ct.cast(pointer, ct.POINTER(ct.POINTER(ct.c_void_p))).contents
    return ct.WINFUNCTYPE(ct.c_int32, ct.c_void_p, *arguments)(table[index])


def _windows_adapters() -> dict[str, Adapter]:
    if sys.platform != "win32":
        return {}
    dxgi = ct.WinDLL("dxgi.dll")
    create = dxgi.CreateDXGIFactory1
    create.argtypes = (ct.c_void_p, ct.POINTER(ct.c_void_p))
    create.restype = ct.c_int32
    iid = (ct.c_ubyte * 16).from_buffer_copy(UUID("770aae78-f26f-4dba-a829-253c83d1b387").bytes_le)
    factory = ct.c_void_p()
    if create(iid, ct.byref(factory)) < 0:
        raise OSError(
            tr("Windows could not enumerate graphics adapters. Choose a runtime manually.")
        )
    found = {}
    try:
        enum = _method(factory, 12, ct.c_uint32, ct.POINTER(ct.c_void_p))
        for index in range(32):
            adapter = ct.c_void_p()
            result = enum(factory, index, ct.byref(adapter))
            if result & 0xFFFFFFFF == 0x887A0002:  # DXGI_ERROR_NOT_FOUND
                break
            if result < 0:
                raise OSError(tr("Windows adapter enumeration failed. Choose a runtime manually."))
            try:
                desc = _Description()
                if _method(adapter, 10, ct.POINTER(_Description))(adapter, ct.byref(desc)) < 0:
                    continue
                if not desc.flags & 2:  # Exclude the Microsoft software adapter.
                    luid = bytes(ct.string_at(ct.byref(desc, _Description.luid_low.offset), 8))
                    found[luid.hex()] = Adapter(
                        desc.name, desc.vendor, desc.video, primary=index == 0
                    )
            finally:
                _method(adapter, 2)(adapter)
    finally:
        _method(factory, 2)(factory)
    return found


def _driver_devices(profile: str, cancel: threading.Event) -> dict[str, CudaDevice | VulkanDevice]:
    environment = clean_runtime_environment(os.environ)
    environment.update(PYTHONUTF8="1", PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    result: dict[str, CudaDevice | VulkanDevice] = {}

    def receive(line: str) -> None:
        if len(result) >= 32:
            return
        if match := re.fullmatch(
            r"CUDA ([a-f0-9]{16}) ([a-f0-9]{32}) (\d{1,6}) (\d{1,3}) (\d{1,3})", line
        ):
            result[match[1]] = CudaDevice(int(match[3]), int(match[4]), int(match[5]), match[2])
        elif match := re.fullmatch(r"Vulkan ([a-f0-9]{16}) ([a-f0-9]{32}) (\d{1,2})", line):
            result[match[1]] = VulkanDevice(match[2], int(match[3]))

    try:
        run_setup_command(
            module_command(Path(sys.executable), "dantetl.runtime.driver_probe", profile),
            resource_root(),
            environment,
            cancel,
            receive,
            timeout=10,
        )
        return result
    except (OSError, RuntimeError) as error:
        logging.getLogger(__name__).warning("%s driver check failed: %s", profile, error)
        return {}


def detect_adapters(
    profile: str = "",
    cancel: threading.Event | None = None,
) -> tuple[Adapter, ...]:
    """Match independent driver queries to DXGI identities, never enumeration order."""
    adapters = _windows_adapters()
    cancel = cancel or threading.Event()
    if profile != "amd" and any(a.vendor == 0x10DE for a in adapters.values()):
        for luid, cuda in _driver_devices("nvidia", cancel).items():
            if luid in adapters and isinstance(cuda, CudaDevice):
                adapters[luid] = replace(adapters[luid], cuda=cuda)
    if profile != "nvidia" and adapters:
        for luid, vulkan in _driver_devices("amd", cancel).items():
            if luid in adapters and isinstance(vulkan, VulkanDevice):
                adapters[luid] = replace(adapters[luid], vulkan=vulkan)
    return tuple(adapters.values())


def require_profile(
    profile: str, cancel: threading.Event | None = None, gpu: str = ""
) -> dict[str, str]:
    """Resolve a stable device identity to the native selector before starting a process."""
    if profile == "cpu" and not gpu:
        return {}
    adapters = detect_adapters(profile, cancel)
    if issue := compatibility_issue(profile, adapters):
        raise RuntimeError(issue)
    candidates = compatible_adapters(profile, adapters)
    if gpu:
        matches = [a for a in candidates if a.gpu_id(profile) == gpu]
        if len(matches) != 1:
            raise RuntimeError(
                tr("The selected GPU is unavailable. Choose another GPU in Acceleration.")
            )
        adapter = matches[0]
    else:
        adapter = candidates[0]
    identity = adapter.gpu_id(profile)
    if profile == "nvidia":
        variable, value = "CUDA_VISIBLE_DEVICES", f"GPU-{UUID(hex=identity)}"
    else:
        assert adapter.vulkan is not None
        variable, value = "GGML_VK_VISIBLE_DEVICES", str(adapter.vulkan.index)
    baseline = clean_runtime_environment(os.environ)
    result = {"DANTETL_GPU_UUID": identity, "DANTETL_GPU_MASK": variable, variable: value}
    if variable in baseline:
        result["DANTETL_GPU_PREVIOUS_MASK"] = baseline[variable]
    return result
