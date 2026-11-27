"""Local PP-OCRv5 detection and Latin CTC recognition for horizontal game text."""

import json
import logging
import math
from importlib.resources import files
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dantetl.domain import CapturedFrame, Rect, union_bounds
from dantetl.i18n import tr
from dantetl.ocr.models import PADDLE_ASSETS
from dantetl.ocr.values import TextRegion
from dantetl.runtime.assets import validate_asset
from dantetl.runtime.ocr import CPU, CUDA, AcceleratorFailure, create_sessions, run_inference


def _line_crops(boxes: list[tuple[Rect, Rect]]) -> list[tuple[Rect, Rect]]:
    """Read adjacent fragments once when their inner boxes overlap on the same baseline."""
    ordered = sorted(boxes, key=lambda pair: pair[1].x)
    successors: dict[int, list[int]] = {}
    predecessors: dict[int, list[int]] = {}
    for i, (_, a) in enumerate(ordered):
        for j in range(i + 1, len(ordered)):
            b = ordered[j][1]
            overlap = a.x + a.width - b.x
            if overlap <= 0:
                break
            unit = min(a.height, b.height)
            if (
                overlap <= 0.5 * unit
                and min(a.width, b.width) >= 3 * unit
                and max(a.height, b.height) <= 1.25 * unit
                and abs((a.y + a.height / 2) - (b.y + b.height / 2)) <= 0.2 * unit
            ):
                successors.setdefault(i, []).append(j)
                predecessors.setdefault(j, []).append(i)

    crops = []
    used: set[int] = set()
    for i, (sample, bounds) in enumerate(ordered):
        if i in used:
            continue
        while len(neighbors := successors.get(i, [])) == 1:
            j = neighbors[0]
            if len(predecessors[j]) != 1:
                break
            next_sample, next_bounds = ordered[j]
            combined = union_bounds((sample, next_sample))
            if combined.width * 48 > combined.height * 3200:
                break  # Keep each crop within the recognizer's input budget.
            sample, bounds = combined, union_bounds((bounds, next_bounds))
            used.add(j)
            i = j
        crops.append((sample, bounds))
    return crops


class PaddleRecognizer:
    def __init__(self, directory: Path, *, device: str = "cpu") -> None:
        if device not in ("cpu", "cuda", "auto"):
            raise ValueError(tr("PaddleOCR Latin supports CPU and NVIDIA CUDA."))
        for asset in PADDLE_ASSETS:
            validate_asset(directory / asset.filename, asset)
        self._directory = directory
        self._automatic = device == "auto"
        self.message = ""
        # Verbatim PostProcess.character_dict from the pinned recognizer's inference.yml.
        dictionary = files("dantetl.ocr").joinpath("paddle_latin", "dictionary.json")
        alphabet = json.loads(dictionary.read_bytes())
        self._alphabet = ("", *alphabet, " ")
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self._detector: Any = None
        self._reader: Any = None
        cv2.setNumThreads(1)
        try:
            self._load(CPU if device == "cpu" else CUDA)
        except Exception as error:
            if not self._automatic:
                raise
            self._fallback(error)

    def _load(self, provider: str) -> None:
        det, rec = (self._directory / asset.filename for asset in PADDLE_ASSETS)
        (self._detector,) = create_sessions((det,), 2, provider)
        # cuDNN needs a roughly 944 MiB workspace for the reader on Blackwell.
        (self._reader,) = create_sessions((rec,), 2, provider, cuda_arena_bytes=2 * 1024**3)
        if self._reader.get_outputs()[0].shape[-1] != len(self._alphabet):
            raise RuntimeError(
                tr("The PaddleOCR recognizer and character dictionary do not match.")
            )
        self.provider = provider

    def _fallback(self, error: Exception) -> None:
        self.close()
        detail = (str(error).splitlines() or [type(error).__name__])[0][:180]
        self.message = tr("GPU unavailable; using CPU. {detail}", detail=detail)
        logging.getLogger(__name__).warning("%s", self.message)
        self._load(CPU)

    def close(self) -> None:
        self._detector = self._reader = None

    def recognize(
        self, frame: CapturedFrame, *, full_size: tuple[int, int] | None = None
    ) -> tuple[TextRegion, ...]:
        bgra = np.frombuffer(frame.bgra, np.uint8).reshape(frame.height, frame.width, 4)
        image = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        try:
            return self._recognize(image, full_size)
        except AcceleratorFailure as failure:
            self._fallback(failure.error)
        return self._recognize(image, full_size)

    def _recognize(
        self, image: np.ndarray, full_size: tuple[int, int] | None
    ) -> tuple[TextRegion, ...]:
        # Only native inference may move an automatic device to CPU. Decoding bugs raise.
        recoverable = self._automatic and self.provider == CUDA
        lines = []
        for sample, bounds in self._boxes(image, full_size, recoverable=recoverable):
            crop = image[sample.y : sample.y + sample.height, sample.x : sample.x + sample.width]
            region = self._read(crop, sample, bounds, recoverable=recoverable)
            if region is not None:
                lines.append(region)
        return tuple(sorted(lines, key=lambda line: (line.bounds.y, line.bounds.x)))

    def read_line(self, bgr: np.ndarray) -> TextRegion | None:
        height, width = bgr.shape[:2]
        bounds = Rect(0, 0, width, height)
        return self._read(bgr, bounds, bounds)

    def _boxes(
        self, image: np.ndarray, full_size: tuple[int, int] | None, *, recoverable: bool
    ) -> list[tuple[Rect, Rect]]:
        height, width = image.shape[:2]
        fw, fh = full_size or (width, height)
        scale = 960 / max(fw, fh)
        full_dw, full_dh = (max(32, round(n * scale / 32) * 32) for n in (fw, fh))
        # Match the full detector's pixel scale on regional rescans. Enlarging a
        # crop can turn a previously intact line into individual word detections.
        dw, dh = max(1, round(width * full_dw / fw)), max(1, round(height * full_dh / fh))
        reduced = cv2.resize(image, (dw, dh))
        reduced = cv2.copyMakeBorder(reduced, 0, -dh % 32, 0, -dw % 32, cv2.BORDER_REPLICATE)
        reduced = reduced.astype(np.float32) / 255
        tensor = np.ascontiguousarray(((reduced - self._mean) / self._std).transpose(2, 0, 1)[None])
        feed = {self._detector.get_inputs()[0].name: tensor}
        prob = run_inference(lambda: self._detector.run(None, feed), recoverable=recoverable)
        prob = prob[0][0, 0]
        contours, _ = cv2.findContours(
            (prob[:dh, :dw] > 0.3).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:128]:
            x, y, w, h = cv2.boundingRect(contour)
            if min(w, h) < 3:
                continue
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [contour - np.array([[[x, y]]], np.int32)], 1)
            if cv2.mean(prob[y : y + h, x : x + w], mask=mask)[0] < 0.6:
                continue
            pad = 1.5 * w * h / (2 * (w + h))
            left = max(0, math.floor((x - pad) * width / dw))
            right = min(width, math.ceil((x + w + pad) * width / dw))
            top = max(0, math.floor((y - pad) * height / dh))
            bottom = min(height, math.ceil((y + h + pad) * height / dh))
            sample = Rect(left, top, right - left, bottom - top)
            # The reader needs context above and below the glyphs. Publishing that
            # context as text can make adjacent rows overlap.
            # Keep any edge touched by a regional sample for the worker's full-scan retry.
            inset_x = round(pad * width / dw / 2)
            inset_y = round(pad * height / dh / 2)
            if left > 3:
                left += inset_x
            if right < width - 3:
                right -= inset_x
            if top > 3:
                top += inset_y
            if bottom < height - 3:
                bottom -= inset_y
            boxes.append((sample, Rect(left, top, right - left, bottom - top)))
        return _line_crops(boxes)

    def _read(
        self, crop: np.ndarray, sample: Rect, bounds: Rect, *, recoverable: bool = False
    ) -> TextRegion | None:
        height, width = crop.shape[:2]
        width = min(3200, max(1, math.ceil(width * 48 / height)))
        resized = cv2.resize(crop, (width, 48)).astype(np.float32) / 127.5 - 1
        feed = np.zeros((1, 3, 48, max(960, width)), np.float32)
        feed[0, :, :, :width] = resized.transpose(2, 0, 1)
        inputs = {self._reader.get_inputs()[0].name: feed}
        scores = run_inference(lambda: self._reader.run(None, inputs), recoverable=recoverable)
        scores = scores[0][0]
        # Ignore decoder timesteps that describe the input's right-hand padding.
        stride = feed.shape[3] / len(scores)
        scores = scores[: math.ceil(width / stride)]
        ids = scores.argmax(axis=1)
        keep = np.flatnonzero((ids != 0) & np.r_[True, ids[1:] != ids[:-1]])
        if not keep.size:
            return None
        step = sample.width * stride / width
        # Background in the reader's margin can emit a weak mark. A separate
        # icon inside the contour can do the same, with a large blank before the word.
        if len(keep) >= 4 and not self._alphabet[ids[keep[0]]].isalnum():
            first, second = keep[:2]
            strength = scores[first, ids[first]]
            in_margin = sample.x + (first + 0.5) * step < bounds.x
            detached = (second - first) * step >= sample.height * 0.9
            if self._alphabet[ids[second]].isalpha() and (
                (in_margin and strength < 0.5) or (detached and strength < 0.85)
            ):
                keep = keep[1:]
        text = "".join(self._alphabet[i] for i in ids[keep]).strip()
        confidence = float(scores[keep, ids[keep]].mean())
        # Confident one-letter labels survive; weak icon readings stay out of tracking.
        if not text or confidence < (0.85 if len(text) <= 2 else 0.5):
            return None
        # Emission positions can exclude a silent icon but aren't glyph edges.
        # Leave room for wide characters and preserve crop-edge retry evidence.
        left = bounds.x
        if left > sample.x:
            left = max(left, sample.x + math.floor(keep[0] * step - bounds.height / 2))
        right = bounds.x + bounds.width
        if right < sample.x + sample.width:
            right = min(right, sample.x + math.ceil(keep[-1] * step + bounds.height))
        return TextRegion(
            text,
            Rect(left, bounds.y, max(1, right - left), bounds.height),
            line_confidence=confidence,
        )
