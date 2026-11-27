"""The single compatibility boundary for pinned MeikiOCR 0.3.5.

Its constructor downloads models and exposes neither paths nor thread settings. We
initialize its three sessions explicitly, preserving its OCR algorithms. No global
monkeypatches, hub lookups, automatic downloads, or GPU dependency are needed.
"""

import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from meikiocr import MeikiOCR
from meikiocr.ocr import INPUT_DET_HEIGHT, INPUT_DET_WIDTH

from dantetl.domain import CapturedFrame, Rect
from dantetl.i18n import tr
from dantetl.ocr.crop_recovery import recovery_preserves_lines, split_glyphs
from dantetl.ocr.models import local_model_paths
from dantetl.ocr.scan_plan import ScanCosts
from dantetl.ocr.values import TextCharacter, TextRegion
from dantetl.ocr.wide_lines import recover_wide_lines
from dantetl.runtime.ocr import (
    CPU,
    CUDA,
    DML,
    AcceleratorFailure,
    create_sessions,
    run_inference,
)


class _LocalMeiki(MeikiOCR):
    def __init__(self, paths: tuple[Path, ...], threads: int, provider: str = CPU) -> None:
        sessions = create_sessions(paths, threads, provider)
        if provider == DML:
            _verify_directml(sessions, paths, threads)
        self.det_session, self.rec_session, self.vrec_session = sessions
        self.active_provider = provider
        self.max_batch_size = 4
        self.detected: list[dict[str, Any]] = []

    def _run_recognition_inference(self, batch_tensor: Any, is_vertical: bool) -> Any:
        # CUDA/cuDNN on the verified runtime replans when the recognition batch
        # changes (about 0.5 s on 4 -> 1 -> 4). Keep its shape constant; discard
        # padded outputs before Meiki associates them with any real detections.
        if batch_tensor is None or self.active_provider != CUDA:
            return super()._run_recognition_inference(batch_tensor, is_vertical)
        count = len(batch_tensor)
        if 0 < count < self.max_batch_size:
            batch_tensor = np.pad(
                batch_tensor, ((0, self.max_batch_size - count), (0, 0), (0, 0), (0, 0))
            )
        outputs: Any = super()._run_recognition_inference(batch_tensor, is_vertical)
        return [output[:count] for output in outputs]

    def run_detection(self, image: Any, conf_threshold: float = 0.5) -> list[dict[str, Any]]:
        # Public run_ocr omits line bounds, including detections with no recognized chars.
        # Retain the detector's original boxes, in its original ordering.
        self.detected = super().run_detection(image, conf_threshold)
        return self.detected

    def read_pieces(self, image: Any, pieces: list[Rect]) -> tuple[TextRegion, ...]:
        boxes = [{"bbox": [b.x, b.y, b.x + b.width, b.y + b.height]} for b in pieces]
        results: list[dict[str, Any]] = [{} for _ in boxes]
        self._process_recognition_pipeline(
            image, boxes, list(range(len(boxes))), results, 0.1, 1, False
        )
        return convert_regions(boxes, results, image.shape[1], image.shape[0])


def _verify_directml(sessions: list[Any], paths: tuple[Path, ...], threads: int) -> None:
    """Fail closed on silent provider miscomputation, before publishing game OCR.

    DirectML 1.24.4 initialized but returned incorrect detector/recognizer values
    on our Windows fixture. Compare small deterministic inputs with CPU once per
    load; automatic mode reports fallback. Temporary reference sessions are freed.
    This is a compatibility probe, not a general OCR accuracy guarantee.
    """
    for session, path, (height, width) in zip(
        sessions, paths, ((544, 960), (32, 960), (480, 32)), strict=True
    ):
        reference = create_sessions((path,), threads, CPU)[0]
        image = np.linspace(0, 1, 3 * height * width, dtype=np.float32).reshape(1, 3, height, width)
        inputs = {
            session.get_inputs()[0].name: image,
            session.get_inputs()[1].name: np.array([[width, height]], dtype=np.int64),
        }
        expected = reference.run(None, inputs)
        actual = session.run(None, inputs)
        del reference
        if any(
            not np.allclose(a, b, rtol=0.02, atol=0.02)
            for a, b in zip(actual, expected, strict=True)
        ):
            raise RuntimeError(
                tr("DirectML model accuracy check failed; CPU OCR is required on this setup.")
            )


def clipped_box(
    values: list[int], width: int, height: int, origin: tuple[int, int] = (0, 0)
) -> Rect:
    x1, y1, x2, y2 = values
    x1, x2, y1, y2 = x1 - origin[0], x2 - origin[0], y1 - origin[1], y2 - origin[1]
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    return Rect(x1, y1, max(0, x2 - x1), max(0, y2 - y1))


def convert_regions(
    detections: list[dict[str, Any]],
    results: list[dict[str, Any]],
    width: int,
    height: int,
    *,
    origin: tuple[int, int] = (0, 0),
) -> tuple[TextRegion, ...]:
    regions = []
    for detected, result in zip(detections, results, strict=True):
        bounds = clipped_box(detected["bbox"], width, height, origin)
        if min(bounds.size) <= 0:
            continue
        characters = []
        for char in result["chars"]:
            confidence = float(char["conf"])
            box = clipped_box(char["bbox"], width, height, origin)
            if math.isfinite(confidence) and min(box.size) > 0:
                characters.append(TextCharacter(char["char"], box, min(1.0, max(0.0, confidence))))
        regions.append(
            TextRegion(result["text"], bounds, tuple(characters), bool(result["is_vertical"]))
        )
    return tuple(regions)


class MeikiRecognizer:
    def __init__(self, directory: Path, *, threads: int = 2, device: str = "cpu") -> None:
        if device not in ("cpu", "auto", "cuda", "directml", "auto-directml"):
            raise ValueError(tr("Unknown OCR device."))
        self._paths = local_model_paths(directory)  # Validate before GPU fallback.
        if not 1 <= threads <= 8:
            raise ValueError(tr("OCR CPU threads must be between 1 and 8."))
        cv2.setNumThreads(1)
        self._threads, self._device = threads, device
        self.message = ""
        try:
            provider = CPU if device == "cpu" else DML if "directml" in device else CUDA
            self._engine = _LocalMeiki(self._paths, threads, provider)
        except Exception as error:
            if not device.startswith("auto"):
                raise
            self._fallback(error)

    @property
    def provider(self) -> str:
        return self._engine.active_provider

    @property
    def scan_costs(self) -> ScanCosts | None:
        # Conservative measured priors for the pinned detector/recognizer. These
        # compare plans, not predict UI latency. Follow actual CPU fallback; the
        # experimental DirectML provider has not passed this planner's corpus.
        detector = INPUT_DET_WIDTH, INPUT_DET_HEIGHT
        return {
            CPU: ScanCosts(40, 30, detector_size=detector),
            CUDA: ScanCosts(20, 16, detector_size=detector),
        }.get(self.provider)

    def _fallback(self, error: Exception) -> None:
        logging.getLogger(__name__).warning("GPU OCR unavailable; using CPU: %s", error)
        detail = (str(error).splitlines() or [type(error).__name__])[0][:180]
        self.message = tr("GPU unavailable; using CPU. {detail}", detail=detail)
        if hasattr(self, "_engine"):
            del self._engine  # Release GPU sessions before allocating the CPU engine.
        self._engine = _LocalMeiki(self._paths, self._threads, CPU)

    def recognize(
        self, frame: CapturedFrame, *, full_size: tuple[int, int] | None = None
    ) -> tuple[TextRegion, ...]:
        bgra = np.frombuffer(frame.bgra, dtype=np.uint8).reshape(frame.height, frame.width, 4)
        bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        regions = self._recognize(bgr, frame.width, frame.height)
        if (
            full_size is not None
            and frame.width < INPUT_DET_WIDTH
            and frame.height < INPUT_DET_HEIGHT
        ):
            empty = not any(region.text.strip() for region in regions)
            fragments = () if empty else split_glyphs(bgr, regions)
            if not empty and not fragments:
                return regions
            # A tiny crop otherwise magnifies glyphs to fill the detector (e.g. 2.6x),
            # missing labels recognized in a full scan. Empty results get the usual
            # natural-scale retry. Only weak, ink-connected fragments get one more
            # centered attempt, accepted only if every other line survives. Full
            # scans and ordinary successful crops keep upstream behavior. Estimate
            # padding background from bounded edges to avoid a black boundary.
            edges = np.concatenate(
                (
                    bgr[0, :: max(1, frame.width // 128)],
                    bgr[-1, :: max(1, frame.width // 128)],
                    bgr[:: max(1, frame.height // 128), 0],
                    bgr[:: max(1, frame.height // 128), -1],
                )
            )
            background = tuple(float(value) for value in np.median(edges, axis=0))

            def padded_at(left: int, top: int) -> Any:
                return cv2.copyMakeBorder(
                    bgr,
                    top,
                    INPUT_DET_HEIGHT - frame.height - top,
                    left,
                    INPUT_DET_WIDTH - frame.width - left,
                    cv2.BORDER_CONSTANT,
                    value=background,
                )

            if empty:
                retried = self._recognize(padded_at(0, 0), frame.width, frame.height)
                if any(region.text.strip() for region in retried):
                    regions = retried
                fragments = split_glyphs(bgr, regions)
            if fragments:
                origin = (
                    (INPUT_DET_WIDTH - frame.width) // 2,
                    (INPUT_DET_HEIGHT - frame.height) // 2,
                )
                centered = self._recognize(
                    padded_at(*origin), frame.width, frame.height, origin=origin
                )
                if recovery_preserves_lines(regions, centered, fragments):
                    return centered
        return regions

    def read_line(self, bgr: np.ndarray) -> TextRegion | None:
        height, width = bgr.shape[:2]
        rows = self._engine.read_pieces(bgr, [Rect(0, 0, width, height)])
        return rows[0] if len(rows) == 1 else None

    def _recognize(
        self, bgr: Any, width: int, height: int, *, origin: tuple[int, int] = (0, 0)
    ) -> tuple[TextRegion, ...]:
        # Meiki scales detection internally and maps both box types back to this image.
        def read() -> tuple[TextRegion, ...]:
            # Only native inference may move an automatic device to CPU. Conversion bugs raise.
            recoverable = self._device.startswith("auto") and self.provider != CPU
            results = run_inference(
                lambda: self._engine.run_ocr(bgr, det_threshold=0.5, rec_threshold=0.1),
                recoverable=recoverable,
            )
            regions = convert_regions(self._engine.detected, results, width, height, origin=origin)
            # Padding belongs only to detection. Retry the original source pixels in
            # the same coordinate system as the clipped regions.
            x, y = origin
            source = bgr[y : y + height, x : x + width]
            return recover_wide_lines(
                regions,
                lambda boxes: run_inference(
                    lambda: self._engine.read_pieces(source, boxes), recoverable=recoverable
                ),
            )

        try:
            return read()
        except AcceleratorFailure as failure:
            self._fallback(failure.error)
            return read()

    def close(self) -> None:
        """Release all native ONNX sessions on their owning worker before it completes."""
        if hasattr(self, "_engine"):
            del self._engine
