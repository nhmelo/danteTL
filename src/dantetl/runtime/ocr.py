"""Select and load local OCR execution providers without owning model algorithms."""

import os
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

from dantetl.i18n import tr
from dantetl.paths import state_root

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
DML = "DmlExecutionProvider"
_dll_handles: list[Any] = []


class AcceleratorFailure(Exception):
    """Native inference failed on an automatically chosen accelerator; CPU may still work."""

    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


def run_inference[T](call: Callable[[], T], *, recoverable: bool) -> T:
    """Mark failures of one native inference call, never the Python code around it."""
    try:
        return call()
    except Exception as error:
        if not recoverable:
            raise
        raise AcceleratorFailure(error) from error


def gpu_runtime_installed(name: str = "onnxruntime-gpu") -> bool:
    try:
        distribution(name)
        return True
    except PackageNotFoundError:
        return False


def prepare_cuda(ort: Any) -> None:
    """Expose pip-installed DLLs to cuDNN's lazy loader, only inside this process."""
    cache = state_root() / "data" / "cuda-cache"
    cache = Path(os.environ.get("CUDA_CACHE_PATH", str(cache)))
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_CACHE_PATH", str(cache))
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", "536870912")
    # Blackwell's first PTX compilation can take a minute. Explicit setup warms
    # this bounded, writable cache so later launches reuse the compiled kernels.
    if os.name == "nt" and not _dll_handles:
        directories = set()
        # Our pinned 1.30 build uses CUDA 13. Keep the earlier 1.26 development/runtime
        # environment usable; selecting by ORT avoids accidentally loading Gemma's CUDA
        # 13 libraries into the legacy CUDA 12 OCR provider.
        current = tuple(int(part) for part in ort.__version__.split(".")[:2]) >= (1, 27)
        suffix = "" if current else "-cu12"
        packages = tuple(
            f"nvidia-{name}{suffix}"
            for name in ("cuda-runtime", "cublas", "cuda-nvrtc", "nvjitlink", "cufft", "curand")
        ) + ("nvidia-cudnn-cu13" if current else "nvidia-cudnn-cu12",)
        for name in packages:
            package = distribution(name)
            directories.update(
                str(package.locate_file(path).parent)
                for path in package.files or ()
                if str(path).endswith(".dll")
            )
        paths = sorted(directories)
        _dll_handles.extend(os.add_dll_directory(path) for path in paths)
        # cuDNN also loads sublibraries with LoadLibrary; add_dll_directory alone
        # did not resolve its tensor-IR engine on the tested Windows runtime.
        os.environ["PATH"] = os.pathsep.join((*paths, os.environ.get("PATH", "")))
    ort.preload_dlls(directory="")


def create_sessions(
    paths: tuple[Any, ...], threads: int, provider: str, *, cuda_arena_bytes: int = 512 * 1024**2
) -> list[Any]:
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    if provider == CUDA:
        if CUDA not in ort.get_available_providers():
            raise RuntimeError(
                tr("CUDA runtime is not installed. Open Settings → Acceleration or choose CPU.")
            )
        prepare_cuda(ort)
    if provider == DML and DML not in ort.get_available_providers():
        raise RuntimeError(tr("DirectML is unavailable. Choose CPU OCR; AMD Gemma uses Vulkan."))
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    options.log_severity_level = 3
    if provider == DML:
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    providers: list[Any] = [CPU]
    if provider == CUDA:
        providers.insert(
            0,
            (
                CUDA,
                {
                    "device_id": 0,
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "gpu_mem_limit": cuda_arena_bytes,
                    "use_tf32": 0,
                },
            ),
        )
    elif provider == DML:
        providers.insert(0, (DML, {"device_id": 0}))
    sessions = []
    for path in paths:
        session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        session.disable_fallback()  # The adapter owns visible whole-engine fallback.
        if provider not in session.get_providers():
            raise RuntimeError(
                tr("Requested OCR provider {provider} did not initialize.", provider=provider)
            )
        sessions.append(session)
    return sessions
