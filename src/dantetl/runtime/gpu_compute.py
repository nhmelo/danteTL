"""Exercise the installed llama GPU backend inside the disposable setup probe."""

import ctypes as ct
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from dantetl.i18n import tr


class _ContextParams(ct.Structure):
    _fields_ = [("size", ct.c_size_t), ("buffer", ct.c_void_p), ("no_alloc", ct.c_bool)]


def verify_gpu_compute(device: str) -> None:
    from llama_cpp import _ggml  # pyright: ignore[reportMissingImports]

    base = ct.CDLL(str(Path(_ggml.__file__).parent / "lib" / "ggml-base.dll"))
    ptr = ct.c_void_p

    def function(name: str, result: Any, *arguments: Any) -> Any:
        return ct.CFUNCTYPE(result, *arguments)((name, base))

    def checked(handle: Any) -> Any:
        if not handle:
            raise RuntimeError(
                tr("GPU offload is unavailable. Update your graphics driver or use CPU.")
            )
        return handle

    # Direct dispatch to the selected backend. A CPU graph or offload flag alone
    # would accept devices on which the installed kernels cannot execute.
    name = {"cuda": b"CUDA0", "vulkan": b"Vulkan0"}[device]
    initialize = ct.CFUNCTYPE(ptr, ct.c_char_p, ct.c_char_p)(
        ("ggml_backend_init_by_name", _ggml.libggml)
    )
    with ExitStack() as resources:
        backend = checked(initialize(name, None))
        resources.callback(function("ggml_backend_free", None, ptr), backend)
        context = checked(
            function("ggml_init", ptr, _ContextParams)(_ContextParams(65536, None, True))
        )
        resources.callback(function("ggml_free", None, ptr), context)
        tensor = function("ggml_new_tensor_2d", ptr, ptr, ct.c_int, ct.c_int64, ct.c_int64)
        left, right = checked(tensor(context, 0, 32, 1)), checked(tensor(context, 0, 32, 1))
        product = checked(function("ggml_mul_mat", ptr, ptr, ptr, ptr)(context, left, right))
        graph = checked(
            function("ggml_new_graph_custom", ptr, ptr, ct.c_size_t, ct.c_bool)(context, 16, False)
        )
        function("ggml_build_forward_expand", None, ptr, ptr)(graph, product)
        buffer = checked(
            function("ggml_backend_alloc_ctx_tensors", ptr, ptr, ptr)(context, backend)
        )
        resources.callback(function("ggml_backend_buffer_free", None, ptr), buffer)
        upload = function("ggml_backend_tensor_set", None, ptr, ptr, ct.c_size_t, ct.c_size_t)
        for handle, value in ((left, 2.0), (right, 3.0)):
            data = (ct.c_float * 32)(*([value] * 32))
            upload(handle, data, 0, ct.sizeof(data))
        status = function("ggml_backend_graph_compute", ct.c_int, ptr, ptr)(backend, graph)
        answer = ct.c_float()
        if status == 0:
            function("ggml_backend_tensor_get", None, ptr, ptr, ct.c_size_t, ct.c_size_t)(
                product, ct.byref(answer), 0, ct.sizeof(answer)
            )
        if status or answer.value != 192.0:
            raise RuntimeError(
                tr("GPU offload is unavailable. Update your graphics driver or use CPU.")
            )
