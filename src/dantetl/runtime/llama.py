"""Embedded llama.cpp discovery and native loading, independent of model prompts/weights."""

import ctypes
import importlib
import os
import threading
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from dantetl.i18n import tr

_dll_handles: list[Any] = []
_load_lock = threading.Lock()
_runtime: Any = None


def _modern(runtime_version: str) -> bool:
    try:
        return Version(runtime_version) >= Version("0.4.7")
    except InvalidVersion:
        return False


def supports_plamo() -> bool:
    """Read metadata only; never load a native engine during UI discovery."""
    try:
        # Upstream 0.3.37 adds the tokenizer; the older fork used a separate 0.4.x API.
        return Version(distribution("llama-cpp-python").version) >= Version("0.3.37")
    except (PackageNotFoundError, InvalidVersion):
        return False


def flash_attention_options(runtime: Any, device: str) -> dict[str, Any]:
    """Keep the existing on/off policy across the old boolean and new enum API."""
    if _modern(getattr(runtime, "__version__", "0")):
        return {"flash_attn_type": 1 if device == "cuda" else 0}
    return {"flash_attn": device == "cuda"}


def installed_device() -> str | None:
    """Inspect the installed wheel without loading a native runtime on the GUI thread."""
    try:
        names = {Path(str(p)).name for p in distribution("llama-cpp-python").files or ()}
    except PackageNotFoundError:
        return None
    if "ggml-vulkan.dll" in names:
        return "vulkan"
    if "ggml-cuda.dll" in names:
        return "cuda"
    cpu = "ggml-cpu.dll" in names or any(n.startswith("ggml-cpu-") for n in names)
    return "cpu" if {"llama.dll", "libllama.dll"} & names and cpu else None


def menu_sequence_width(device: str) -> int:
    """Choose a bounded width for main_gpu=0 from the loaded runtime's visible devices."""
    if device == "cpu":
        return 8  # Close to the 16-slot throughput, with earlier first labels.
    from ctypes import CDLL, CFUNCTYPE, c_char_p, c_void_p

    try:
        from llama_cpp import _ggml  # pyright: ignore[reportMissingImports]

        # DXGI order can differ from llama.cpp, especially with Vulkan device filtering.
        find = CFUNCTYPE(c_void_p, c_char_p)(("ggml_backend_dev_by_name", _ggml.libggml))
        base = CDLL(str(Path(_ggml.__file__).parent / "lib" / "ggml-base.dll"))
        describe = CFUNCTYPE(c_char_p, c_void_p)(("ggml_backend_dev_description", base))
        handle = find(b"CUDA0" if device == "cuda" else b"Vulkan0")
        description = describe(handle) if handle else b""
    except (ImportError, AttributeError, OSError):
        description = b""
    if description and b"RTX 5080" in description:
        return 8
    if device == "vulkan" and description == b"AMD Radeon(TM) Graphics":
        return 16
    return 4


def load_runtime(device: str) -> Any:
    """Load native libraries on a worker or in the isolated setup probe only."""
    global _runtime
    with _load_lock:
        if _runtime is not None:
            return _runtime
        _prepare_libraries(device)
        runtime = importlib.import_module("llama_cpp")
        if runtime.__file__ is None:
            raise RuntimeError(tr("Language-model runtime has no library location."))
        directory = Path(runtime.__file__).parent / "lib"
        # ggml's default plugin search uses the executable and working directories.
        # Select from the installed package before initializing any model.
        if _modern(getattr(runtime, "__version__", "0")):
            ggml = importlib.import_module("llama_cpp._ggml")
            ggml.ggml_backend_load_all_from_path(str(directory).encode("utf-8"))
        elif device == "cpu" and os.name == "nt":
            _load_cpu_backend(directory)
        _runtime = runtime
        return runtime


def _load_cpu_backend(directory: Path) -> None:
    # The upstream 0.3 binding does not expose ggml's plugin API in Python.
    library = ctypes.CDLL(str(directory / "ggml.dll"))
    cpu_device = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p)(
        ("ggml_backend_dev_by_name", library)
    )
    if cpu_device(b"CPU"):  # Older single-backend CPU wheels register at startup.
        return
    load = ctypes.CFUNCTYPE(None, ctypes.c_char_p)(("ggml_backend_load_all_from_path", library))
    load(str(directory).encode("utf-8"))
    if not cpu_device(b"CPU"):
        raise RuntimeError(
            tr("The installed language-model engine does not provide {device}.", device="cpu")
        )


def _prepare_libraries(device: str) -> None:
    if os.name == "nt" and not _dll_handles:
        paths = set()
        packages = ("llama-cpp-python",)
        if device == "cuda":
            packages += ("nvidia-cuda-runtime", "nvidia-cublas")
        for name in packages:
            package = distribution(name)
            paths.update(
                str(package.locate_file(path).parent)
                for path in package.files or ()
                if str(path).endswith(".dll")
            )
        _dll_handles.extend(os.add_dll_directory(path) for path in sorted(paths))
        os.environ["PATH"] = os.pathsep.join((*sorted(paths), os.environ.get("PATH", "")))
