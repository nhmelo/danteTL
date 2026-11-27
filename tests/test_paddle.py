import json
from importlib.resources import files
from unittest.mock import Mock

import numpy as np
import pytest

from dantetl.domain import CapturedFrame, Rect, WindowGeometry
from dantetl.ocr import paddle
from dantetl.runtime.ocr import CPU, CUDA


@pytest.mark.parametrize(
    "prefix, strength, first, following, word, expected",
    [
        ("O", 0.52, 3, 0, "", ""),
        ("A", 0.98, 3, 0, "", "A"),
        ("O", 0.97, 3, 5, "K", "OK"),
        ("*", 0.72, 2, 10, "Menu", "Menu"),
        (":", 0.42, 0, 3, "end", "end"),
        ("¿", 0.98, 0, 3, "Fin?", "¿Fin?"),
        ("*", 0.72, 2, 4, "Menu", "*Menu"),
    ],
)
def test_icon_evidence_preserves_confident_short_labels_and_punctuation(
    tmp_path, monkeypatch, prefix, strength, first, following, word, expected
):
    dictionary = files("dantetl.ocr").joinpath("paddle_latin", "dictionary.json")
    alphabet = ("", *json.loads(dictionary.read_bytes()), " ")
    bounds = Rect(0, 0, 960, 192)
    frame = CapturedFrame(1, 1, WindowGeometry(bounds, bounds, bounds, 96), bytes(960 * 192 * 4))
    probability = np.zeros((1, 1, 192, 960), np.float32)
    probability[:, :, 40:48, 100:170] = 1
    scores = np.zeros((1, 120, len(alphabet)), np.float32)
    scores[0, first, alphabet.index(prefix)] = strength
    for i, char in enumerate(word, following):
        scores[0, i, alphabet.index(char)] = 0.99
    # A nonblank emission in the padded part of the reader tensor is never text.
    scores[0, -1, alphabet.index("Z")] = 0.99
    detector, reader = Mock(), Mock()
    detector.run.return_value = [probability]
    reader.run.return_value = [scores]
    reader.get_outputs.return_value = [Mock(shape=scores.shape)]
    for session in (detector, reader):
        session.get_inputs.return_value = [Mock()]
    monkeypatch.setattr(paddle, "validate_asset", lambda *args: None)
    monkeypatch.setattr(paddle, "create_sessions", Mock(side_effect=[(detector,), (reader,)]))
    engine = paddle.PaddleRecognizer(tmp_path)
    try:
        assert [r.text for r in engine.recognize(frame)] == ([expected] if expected else [])
    finally:
        engine.close()


@pytest.mark.parametrize("x,y,expected_reads", [(172, 40, 1), (200, 40, 2), (172, 48, 2)])
def test_overlapping_same_baseline_fragments_are_read_together(
    monkeypatch, tmp_path, x, y, expected_reads
):
    detector, reader = Mock(), Mock()
    for session in (detector, reader):
        session.get_inputs.return_value = [Mock()]
    probability = np.zeros((1, 1, 192, 960), np.float32)
    probability[:, :, 40:48, 100:170] = 1
    probability[:, :, y : y + 8, x : x + 70] = 1
    detector.run.return_value = [probability]
    reader.get_outputs.return_value = [Mock(shape=[1, 120, 838])]
    reader.run.return_value = [np.zeros((1, 120, 838), np.float32)]
    monkeypatch.setattr(paddle, "validate_asset", lambda *args: None)
    monkeypatch.setattr(paddle, "create_sessions", Mock(side_effect=[(detector,), (reader,)]))
    engine = paddle.PaddleRecognizer(tmp_path)
    bounds = Rect(0, 0, 960, 192)
    try:
        engine.recognize(
            CapturedFrame(1, 1, WindowGeometry(bounds, bounds, bounds, 96), bytes(960 * 192 * 4))
        )
        assert reader.run.call_count == expected_reads
    finally:
        engine.close()


@pytest.mark.parametrize("device", ["auto", "cuda"])
def test_failed_cuda_scan_retries_whole_frame_only_in_auto(tmp_path, monkeypatch, device):
    allocations = []
    detector = Mock()
    detector.run.return_value = [np.zeros((1, 1, 32, 32), dtype=np.float32)]
    unavailable = Mock()
    unavailable.run.side_effect = RuntimeError("CUDA workspace exhausted")
    for session in (detector, unavailable):
        session.get_inputs.return_value = [Mock()]
    bounds = Rect(0, 0, 32, 32)
    frame = CapturedFrame(1, 1, WindowGeometry(bounds, bounds, bounds, 96), bytes(32 * 32 * 4))
    reader = Mock()
    reader.get_outputs.return_value = [Mock(shape=[1, 32, 838])]

    def sessions(paths, threads, provider, *, cuda_arena_bytes=512 * 1024**2):
        reader_path = paths[0].name == "rec.onnx"
        allocations.append((provider, reader_path, cuda_arena_bytes))
        return [reader if reader_path else unavailable if provider == CUDA else detector]

    monkeypatch.setattr(paddle, "validate_asset", lambda *args: None)
    monkeypatch.setattr(paddle, "create_sessions", sessions)
    engine = paddle.PaddleRecognizer(tmp_path, device=device)
    assert allocations == [(CUDA, False, 512 * 1024**2), (CUDA, True, 2 * 1024**3)]
    try:
        if device == "cuda":
            with pytest.raises(RuntimeError, match="workspace exhausted"):
                engine.recognize(frame)
            assert engine.provider == CUDA and len(allocations) == 2
        else:
            assert engine.recognize(frame) == ()
            assert engine.provider == CPU and "workspace exhausted" in engine.message
            assert [provider for provider, _, _ in allocations] == [CUDA, CUDA, CPU, CPU]
    finally:
        engine.close()


def test_auto_cuda_keeps_its_provider_when_python_decoding_fails(tmp_path, monkeypatch):
    detector, reader = Mock(), Mock()
    detector.run.return_value = [np.zeros((1, 1, 32, 32), dtype=np.float32)]
    reader.get_outputs.return_value = [Mock(shape=[1, 32, 838])]
    for session in (detector, reader):
        session.get_inputs.return_value = [Mock()]
    providers = []

    def sessions(paths, threads, provider, *, cuda_arena_bytes=512 * 1024**2):
        providers.append(provider)
        return [reader if paths[0].name == "rec.onnx" else detector]

    def decoding_bug(boxes):
        raise ValueError("decoding bug")

    monkeypatch.setattr(paddle, "validate_asset", lambda *args: None)
    monkeypatch.setattr(paddle, "create_sessions", sessions)
    monkeypatch.setattr(paddle, "_line_crops", decoding_bug)
    engine = paddle.PaddleRecognizer(tmp_path, device="auto")
    bounds = Rect(0, 0, 32, 32)
    frame = CapturedFrame(1, 1, WindowGeometry(bounds, bounds, bounds, 96), bytes(32 * 32 * 4))
    try:
        with pytest.raises(ValueError, match="decoding bug"):
            engine.recognize(frame)
        assert providers == [CUDA, CUDA] and engine.provider == CUDA and not engine.message
    finally:
        engine.close()


def test_missing_latin_assets_fail_before_trying_any_provider(tmp_path, monkeypatch):
    sessions = Mock()
    monkeypatch.setattr(paddle, "create_sessions", sessions)
    with pytest.raises(RuntimeError, match="det.onnx"):
        paddle.PaddleRecognizer(tmp_path, device="auto")
    sessions.assert_not_called()
