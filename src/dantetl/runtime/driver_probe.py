"""Windows driver queries for setup. Run only in a contained, short-lived helper."""

import ctypes as ct
import os
import sys
from collections.abc import Iterator

from dantetl.runtime.owned_process import restore_child_output

_U32 = ct.c_uint32
_Handle = ct.c_void_p
_VULKAN_1_2 = (1 << 22) | (2 << 12)


def _cuda() -> Iterator[tuple[str, str, int, int, int]]:
    driver = ct.WinDLL("nvcuda.dll", winmode=0x800)  # LOAD_LIBRARY_SEARCH_SYSTEM32
    version, device, major, minor = (ct.c_int() for _ in range(4))
    luid = (ct.c_ubyte * 8)()
    mask = _U32()
    count = ct.c_int()
    uuid = (ct.c_ubyte * 16)()
    if (
        driver.cuInit(0)
        or driver.cuDriverGetVersion(ct.byref(version))
        or driver.cuDeviceGetCount(ct.byref(count))
    ):
        return
    for index in range(min(count.value, 32)):
        if (
            driver.cuDeviceGet(ct.byref(device), index) == 0
            and driver.cuDeviceGetLuid(luid, ct.byref(mask), device) == 0
            and driver.cuDeviceGetUuid(uuid, device) == 0
            and driver.cuDeviceGetAttribute(ct.byref(major), 75, device) == 0
            and driver.cuDeviceGetAttribute(ct.byref(minor), 76, device) == 0
        ):
            yield bytes(luid).hex(), bytes(uuid).hex(), version.value, major.value, minor.value


class _ApplicationInfo(ct.Structure):
    _fields_ = [
        ("sType", _U32),
        ("pNext", _Handle),
        ("appName", ct.c_char_p),
        ("appVersion", _U32),
        ("engineName", ct.c_char_p),
        ("engineVersion", _U32),
        ("apiVersion", _U32),
    ]


class _InstanceInfo(ct.Structure):
    _fields_ = [
        ("sType", _U32),
        ("pNext", _Handle),
        ("flags", _U32),
        ("application", ct.POINTER(_ApplicationInfo)),
        ("layerCount", _U32),
        ("layerNames", _Handle),
        ("extensionCount", _U32),
        ("extensionNames", _Handle),
    ]


class _DeviceID(ct.Structure):
    _fields_ = [
        ("sType", _U32),
        ("pNext", _Handle),
        ("uuid", ct.c_ubyte * 16),
        ("driverUUID", ct.c_ubyte * 16),
        ("luid", ct.c_ubyte * 8),
        ("nodeMask", _U32),
        ("luidValid", _U32),
    ]


class _Properties2(ct.Structure):
    # VkPhysicalDeviceProperties is a fixed ABI. Only its leading scalars are
    # read; reserve aligned space for the remaining name, UUID, limits and sparse properties.
    _fields_ = [
        ("sType", _U32),
        ("pNext", ct.POINTER(_DeviceID)),
        ("apiVersion", _U32),
        ("driverVersion", _U32),
        ("vendor", _U32),
        ("device", _U32),
        ("deviceType", _U32),
        ("remainingProperties", ct.c_uint64 * 128),
    ]


class _Storage16(ct.Structure):
    _fields_ = [
        ("sType", _U32),
        ("pNext", _Handle),
        ("storageBuffer16BitAccess", _U32),
        ("uniformAndStorage", _U32),
        ("pushConstant", _U32),
        ("inputOutput", _U32),
    ]


class _Features2(ct.Structure):
    _fields_ = [
        ("sType", _U32),
        ("pNext", ct.POINTER(_Storage16)),
        ("features", _U32 * 55),
    ]


class _QueueFamily(ct.Structure):
    _fields_ = [
        ("flags", _U32),
        ("count", _U32),
        ("timestampBits", _U32),
        ("granularity", _U32 * 3),
    ]


def _vulkan() -> Iterator[tuple[str, str, int]]:
    vk = ct.WinDLL("vulkan-1.dll", winmode=0x800)
    version = _U32()
    if vk.vkEnumerateInstanceVersion(ct.byref(version)) or version.value < _VULKAN_1_2:
        return
    app = _ApplicationInfo(sType=0, appName=b"DanteTL", apiVersion=_VULKAN_1_2)
    info = _InstanceInfo(sType=1, application=ct.pointer(app))
    instance = _Handle()
    if vk.vkCreateInstance(ct.byref(info), None, ct.byref(instance)) != 0:
        return
    try:
        count = _U32(32)
        devices = (_Handle * count.value)()
        if vk.vkEnumeratePhysicalDevices(instance, ct.byref(count), devices) != 0:
            return
        visible = list(range(count.value))
        # Preserve external visibility restrictions and the original physical indices.
        if (override := os.environ.get("GGML_VK_VISIBLE_DEVICES")) is not None:
            indices = override.replace(",", " ").split()
            if not indices or not all(
                i.isascii() and i.isdecimal() and len(i) <= 3 and int(i) < count.value
                for i in indices
            ):
                return
            visible = list(dict.fromkeys(map(int, indices)))
        for index in visible:
            physical = _Handle(devices[index])
            identity = _DeviceID(sType=1000071004)
            props = _Properties2(sType=1000059001, pNext=ct.pointer(identity))
            vk.vkGetPhysicalDeviceProperties2(physical, ct.byref(props))
            if not identity.luidValid or props.deviceType not in (1, 2):
                continue
            storage = _Storage16(sType=1000083000)
            features = _Features2(sType=1000059000, pNext=ct.pointer(storage))
            vk.vkGetPhysicalDeviceFeatures2(physical, ct.byref(features))
            families = (_QueueFamily * 64)()
            family_count = _U32(len(families))
            vk.vkGetPhysicalDeviceQueueFamilyProperties(physical, ct.byref(family_count), families)
            if (
                props.apiVersion >= _VULKAN_1_2
                and storage.storageBuffer16BitAccess
                and any(q.count and q.flags & 2 for q in families[: family_count.value])
            ):
                yield bytes(identity.luid).hex(), bytes(identity.uuid).hex(), index
    finally:
        vk.vkDestroyInstance(instance, None)


if __name__ == "__main__":
    restore_child_output()
    try:
        if sys.argv[1] == "nvidia":
            for luid, uuid, driver, major, minor in _cuda():
                print(f"CUDA {luid} {uuid} {driver} {major} {minor}", flush=True)
        else:
            for luid, uuid, index in _vulkan():
                print(f"Vulkan {luid} {uuid} {index}", flush=True)
    except (AttributeError, OSError):
        pass  # Driver absent, too old to expose these queries, or inaccessible.
