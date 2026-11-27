import hashlib
from dataclasses import replace

import pytest

from dantetl.domain import CapturedFrame, Rect, WindowGeometry
from dantetl.ocr.meiki import MeikiRecognizer, convert_regions
from dantetl.runtime.assets import ModelAsset, validate_asset
from dantetl.runtime.ocr import CPU, CUDA, DML


def test_crop_padding_preserves_pixels_bounds_and_full_scan_behavior(tmp_path, monkeypatch):
    import numpy as np

    monkeypatch.setattr("dantetl.ocr.meiki.local_model_paths", lambda path: ())
    seen = []

    class Engine:
        def __init__(self, *args):
            self.active_provider = CPU
            self.detected = []
            self.unsure = ()

        def run_ocr(self, image, **kwargs):
            seen.append(image)
            if image.shape[1] == 195:
                self.detected = []
                return []
            self.detected = [{"bbox": [15, 20, 55, 45]}, {"bbox": [500, 300, 550, 350]}]
            return [
                {"text": "設定", "chars": [], "is_vertical": False},
                {"text": "padding", "chars": [], "is_vertical": False},
            ]

    monkeypatch.setattr("dantetl.ocr.meiki._LocalMeiki", Engine)
    recognizer = MeikiRecognizer(tmp_path)
    bounds = Rect(0, 0, 195, 210)
    frame = CapturedFrame(
        7, 12, WindowGeometry(bounds, bounds, bounds, 168), bytes((20, 35, 55, 255)) * 195 * 210
    )
    regions = recognizer.recognize(frame, full_size=(1920, 1080))
    assert [(r.text, r.bounds) for r in regions] == [("設定", Rect(15, 20, 40, 25))]
    assert seen[-1].shape == (544, 960, 3)
    assert len(seen) == 2
    assert np.all(seen[-1] == (20, 35, 55))
    recognizer.recognize(frame)
    assert seen[-1].shape == (210, 195, 3)
    # A crop that already needs downscaling must keep upstream scale/aspect behavior.
    large = Rect(0, 0, 1200, 200)
    recognizer.recognize(
        replace(
            frame, geometry=WindowGeometry(large, large, large, 168), bgra=bytes(1200 * 200 * 4)
        ),
        full_size=(1920, 1080),
    )
    assert seen[-1].shape == (200, 1200, 3)
    recognizer.close()


@pytest.mark.parametrize("text, calls", [("設定", 1), ("", 2)])
def test_crop_retry_is_bounded_and_keeps_successful_original_results(
    tmp_path, monkeypatch, text, calls
):
    monkeypatch.setattr("dantetl.ocr.meiki.local_model_paths", lambda path: ())
    seen = []

    class Engine:
        def __init__(self, *args):
            self.active_provider = CPU
            self.detected = [{"bbox": [10, 10, 60, 40]}]

        def run_ocr(self, image, **kwargs):
            seen.append(image.shape)
            return [{"text": text, "chars": [], "is_vertical": False}]

    monkeypatch.setattr("dantetl.ocr.meiki._LocalMeiki", Engine)
    bounds = Rect(0, 0, 195, 210)
    frame = CapturedFrame(1, 0, WindowGeometry(bounds, bounds, bounds, 96), bytes(195 * 210 * 4))
    recognizer = MeikiRecognizer(tmp_path)
    regions = recognizer.recognize(frame, full_size=(1920, 1080))
    assert len(seen) == calls
    assert regions[0].text == text and regions[0].bounds == Rect(10, 10, 50, 30)
    recognizer.close()


@pytest.mark.parametrize("provider", [CPU, CUDA])
@pytest.mark.parametrize("vertical", [False, True])
def test_partial_gpu_batches_keep_a_fixed_shape_and_never_emit_padded_detections(
    monkeypatch, provider, vertical
):
    import numpy as np
    from meikiocr import MeikiOCR

    from dantetl.ocr.meiki import _LocalMeiki

    seen = []

    def run(self, batch, is_vertical):
        seen.append((batch.shape, is_vertical))
        return [np.arange(len(batch)), np.ones((len(batch), 2, 4)), np.ones((len(batch), 2))]

    monkeypatch.setattr(MeikiOCR, "_run_recognition_inference", run)
    engine = object.__new__(_LocalMeiki)
    engine.active_provider, engine.max_batch_size = provider, 4
    h, w = (480, 32) if vertical else (32, 960)
    for count in (4, 1, 3, 4, 1):
        outputs = engine._run_recognition_inference(np.ones((count, 3, h, w), np.float32), vertical)
        assert all(len(output) == count for output in outputs)
        assert outputs[0].tolist() == list(range(count))
    assert [shape[0] for shape, _ in seen] == ([4] * 5 if provider == CUDA else [4, 1, 3, 4, 1])
    assert all(flag == vertical for _, flag in seen)


def test_detection_boxes_remain_in_original_frame_pixels_and_empty_detections_survive() -> None:
    regions = convert_regions(
        [{"bbox": [100, 200, 500, 240]}, {"bbox": [-10, -10, 80, 100]}],
        [
            {
                "text": "日本",
                "is_vertical": False,
                "chars": [
                    {"char": "日", "bbox": [102, 201, 130, 239], "conf": 0.8},
                    {"char": "本", "bbox": [135, 201, 165, 239], "conf": 0.6},
                ],
            },
            {"text": "", "is_vertical": True, "chars": []},
        ],
        1920,
        1080,
    )
    assert regions[0].bounds == Rect(100, 200, 400, 40)
    assert regions[0].characters[1].bounds == Rect(135, 201, 30, 38)
    assert regions[0].confidence == pytest.approx(0.7)
    assert regions[1].bounds == Rect(0, 0, 80, 100)
    assert regions[1].confidence is None
    assert regions[1].vertical


def test_model_validation_detects_missing_incomplete_and_same_size_corruption(tmp_path) -> None:
    path = tmp_path / "test.onnx"
    asset = ModelAsset("repo", "revision", path.name, hashlib.sha256(b"good").hexdigest(), 4)
    with pytest.raises(RuntimeError, match="Missing or incomplete"):
        validate_asset(path, asset)
    path.write_bytes(b"bad")
    with pytest.raises(RuntimeError, match="Missing or incomplete"):
        validate_asset(path, asset)
    path.write_bytes(b"evil")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        validate_asset(path, asset)
    path.write_bytes(b"good")
    validate_asset(path, asset)
    with pytest.raises(RuntimeError):
        validate_asset(path, replace(asset, size=5))


def test_missing_models_never_invoke_upstream_downloader(tmp_path, monkeypatch) -> None:
    def unexpected_download(*args, **kwargs):
        pytest.fail("Runtime attempted a model download")

    monkeypatch.setattr("meikiocr.ocr.hf_hub_download", unexpected_download)
    with pytest.raises(RuntimeError, match="Open Models"):
        MeikiRecognizer(tmp_path)


@pytest.mark.parametrize("fail_on_run", [False, True])
@pytest.mark.parametrize("provider, device", [(CUDA, "auto"), (DML, "auto-directml")])
def test_auto_gpu_failure_falls_back_once_and_reports_the_actual_provider(
    tmp_path, monkeypatch, fail_on_run, provider, device
) -> None:
    monkeypatch.setattr("dantetl.ocr.meiki.local_model_paths", lambda path: ())
    created = []

    class Engine:
        def __init__(self, paths, threads, provider):
            created.append(provider)
            if provider != CPU and not fail_on_run:
                raise RuntimeError("GPU unavailable")
            self.active_provider, self.detected = provider, []

        def run_ocr(self, image, **kwargs):
            if self.active_provider != CPU:
                raise RuntimeError()  # Even an exception with no message must fall back.
            return []

    monkeypatch.setattr("dantetl.ocr.meiki._LocalMeiki", Engine)
    recognizer = MeikiRecognizer(tmp_path, device=device)
    box = Rect(0, 0, 2, 2)
    frame = CapturedFrame(1, 0, WindowGeometry(box, box, box, 96), bytes(16))
    assert recognizer.recognize(frame) == ()
    assert recognizer.recognize(frame) == ()
    assert created == [provider, CPU]
    assert recognizer.provider == CPU and "using CPU" in recognizer.message


def test_auto_gpu_keeps_its_provider_when_python_post_processing_fails(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("dantetl.ocr.meiki.local_model_paths", lambda path: ())
    created = []

    class Engine:
        def __init__(self, paths, threads, provider):
            created.append(provider)
            self.active_provider, self.detected = provider, []

        def run_ocr(self, image, **kwargs):
            return []

    def conversion_bug(*args, **kwargs):
        raise ValueError("conversion bug")

    monkeypatch.setattr("dantetl.ocr.meiki._LocalMeiki", Engine)
    monkeypatch.setattr("dantetl.ocr.meiki.convert_regions", conversion_bug)
    recognizer = MeikiRecognizer(tmp_path, device="auto")
    box = Rect(0, 0, 2, 2)
    with pytest.raises(ValueError, match="conversion bug"):
        recognizer.recognize(CapturedFrame(1, 0, WindowGeometry(box, box, box, 96), bytes(16)))
    assert created == [CUDA] and recognizer.provider == CUDA and not recognizer.message


def test_explicit_cpu_never_initializes_gpu_and_strict_cuda_never_silently_falls_back(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("dantetl.ocr.meiki.local_model_paths", lambda path: ())
    created = []

    class Engine:
        def __init__(self, paths, threads, provider):
            created.append(provider)
            self.active_provider = provider
            if provider == CUDA:
                raise RuntimeError("GPU unavailable")

    monkeypatch.setattr("dantetl.ocr.meiki._LocalMeiki", Engine)
    assert MeikiRecognizer(tmp_path, device="cpu").provider == CPU
    with pytest.raises(RuntimeError, match="GPU unavailable"):
        MeikiRecognizer(tmp_path, device="cuda")
    assert created == [CPU, CUDA]


def test_directml_compatibility_probe_rejects_silent_wrong_results(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import numpy as np

    from dantetl.ocr.meiki import _verify_directml

    reference = Mock()
    reference.run.return_value = [np.array([0.9])]
    monkeypatch.setattr("dantetl.ocr.meiki.create_sessions", lambda *a: [reference])
    session = Mock()
    session.get_inputs.return_value = [SimpleNamespace(name="images"), SimpleNamespace(name="size")]
    session.run.return_value = [np.array([0.9001])]
    paths = (tmp_path / "detect", tmp_path / "horizontal", tmp_path / "vertical")
    _verify_directml([session] * 3, paths, 2)
    session.run.return_value = [np.array([0.2])]
    with pytest.raises(RuntimeError, match="accuracy check failed"):
        _verify_directml([session] * 3, paths, 2)
