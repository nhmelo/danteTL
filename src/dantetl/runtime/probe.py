"""Fresh-process library check, without model downloads or first-use model preparation."""

import argparse
import json
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

from dantetl.i18n import tr


def verify(profile: str) -> None:
    if profile not in ("cpu", "nvidia", "amd"):
        raise ValueError(tr("Unknown runtime profile."))
    installed = []
    for name in ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml"):
        try:
            version(name)
            installed.append(name)
        except PackageNotFoundError:
            pass
    expected = "onnxruntime-gpu" if profile == "nvidia" else "onnxruntime"
    if installed != [expected]:
        raise RuntimeError(
            tr(
                "Expected only {expected}; found {installed}.",
                expected=expected,
                installed=installed,
            )
        )
    # Verify the same native imports used at app startup, in this throwaway process.
    import meikiocr  # noqa: F401
    import onnxruntime as ort
    import windows_capture  # noqa: F401
    from PySide6 import QtWidgets  # noqa: F401

    if profile == "nvidia":
        import numpy as np
        from onnxruntime.datasets import get_example

        from dantetl.runtime.ocr import CUDA, prepare_cuda

        prepare_cuda(ort)
        if CUDA not in ort.get_available_providers():
            raise RuntimeError(tr("The installed OCR runtime does not provide CUDA."))
        # Run the tiny multiplication fixture included in the pinned ONNX wheel.
        # Provider enumeration succeeds even when cuBLAS rejects the GPU.
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        session = ort.InferenceSession(get_example("mul_1.onnx"), options, providers=[CUDA])
        values = np.arange(1, 7, dtype=np.float32).reshape(3, 2)
        output = session.run(None, {"X": values})[0]
        if (
            CUDA not in session.get_providers()
            or not isinstance(output, np.ndarray)
            or not np.array_equal(output, values * values)
        ):
            raise RuntimeError(tr("The installed OCR runtime does not provide CUDA."))
    from dantetl.runtime.llama import installed_device, load_runtime

    device = {"nvidia": "cuda", "amd": "vulkan", "cpu": "cpu"}[profile]
    if installed_device() != device:
        raise RuntimeError(
            tr("The installed language-model engine does not provide {device}.", device=device)
        )
    runtime = load_runtime(device)
    runtime.llama_backend_init()
    try:
        if device != "cpu" and not runtime.llama_supports_gpu_offload():
            raise RuntimeError(
                tr("GPU offload is unavailable. Update your graphics driver or use CPU.")
            )
        if device != "cpu":
            from dantetl.runtime.gpu_compute import verify_gpu_compute

            verify_gpu_compute(device)
    finally:
        runtime.llama_backend_free()
    print(tr("Library checks passed. Model preparation happens after restart."), flush=True)


def verify_ocr(image_path: Path, *, latin: bool = False) -> dict[str, object]:
    """Offline diagnostic against an explicit synthetic fixture, with no font dependency."""
    from unittest.mock import patch

    from PySide6.QtGui import QImage

    from dantetl.domain import CapturedFrame, Rect, WindowGeometry
    from dantetl.ocr.meiki import MeikiRecognizer
    from dantetl.ocr.models import model_directory, paddle_directory
    from dantetl.ocr.paddle import PaddleRecognizer

    image = QImage(str(image_path)).convertToFormat(QImage.Format.Format_RGB32)
    if image.isNull() or image.width() != 1920 or image.height() != 1080:
        raise ValueError(tr("Expected the 1920x1080 OCR verification fixture"))
    expected = "Open inventory" if latin else "新しい冒険が、ここから始まる。"
    directory = paddle_directory() if latin else model_directory()
    bounds = Rect(0, 0, image.width(), image.height())
    frame = CapturedFrame(
        1, 0.0, WindowGeometry(bounds, bounds, bounds, 96), bytes(image.constBits())
    )
    with patch("socket.socket", side_effect=RuntimeError(tr("Diagnostic must remain offline"))):
        recognizer = (PaddleRecognizer if latin else MeikiRecognizer)(directory, device="cpu")
        try:
            rows = recognizer.recognize(frame)
            provider = recognizer.provider
        finally:
            recognizer.close()
    text = "".join(row.text for row in rows)
    if expected not in text:
        raise RuntimeError(tr("OCR diagnostic did not recognize the fixture: {text}", text=text))
    return {"provider": provider, "text": text, "models": str(directory)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("cpu", "nvidia", "amd"))
    parser.add_argument("--application-only", action="store_true", help=tr("Check app import only"))
    parser.add_argument(
        "--ocr-image", type=Path, help=tr("Check CPU OCR against the synthetic fixture")
    )
    parser.add_argument("--report", type=Path, help=tr("Write results for a windowed executable"))
    parser.add_argument("--latin-ocr-image", type=Path)
    args = parser.parse_args()
    if not args.application_only:
        verify(args.profile)
    result: dict[str, object] = {"profile": args.profile}
    if not args.application_only:
        result["libraries"] = "passed"
    from dantetl.paths import resource_root, state_root

    application = distribution("dantetl")
    (entry,) = (
        e for e in application.entry_points if e.group == "gui_scripts" and e.name == "dantetl"
    )
    # Import the real startup graph before publishing an app install. Importing this
    # small probe alone would accept a wheel with missing nested application modules.
    entry.load()
    result["application"] = {
        "module": __file__,
        "version": application.version,
        "resources": str(resource_root()),
        "state": str(state_root()),
    }
    if args.ocr_image:
        result["ocr"] = verify_ocr(args.ocr_image)
    if args.latin_ocr_image:
        result["latin_ocr"] = verify_ocr(args.latin_ocr_image, latin=True)
    if args.report:
        args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
