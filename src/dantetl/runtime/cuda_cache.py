"""One writable CUDA kernel cache shared by ONNX Runtime and llama.cpp in this process."""

import os
from pathlib import Path

from dantetl.paths import state_root

_MAX_BYTES = 512 * 1024**2


def use_kernel_cache() -> None:
    """Point CUDA's driver cache at per-user state before the first kernel compiles.

    First-use compilation on new GPUs can take a minute; a persistent, bounded cache lets later
    launches reuse it. Explicit environment settings win.
    """
    cache = Path(os.environ.get("CUDA_CACHE_PATH", str(state_root() / "data/cuda-cache")))
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_CACHE_PATH", str(cache))
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(_MAX_BYTES))
