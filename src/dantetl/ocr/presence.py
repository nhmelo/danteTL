"""Bounded exact-pixel evidence for missed text; no image history or model calls."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import TYPE_CHECKING

from dantetl.domain import CapturedFrame, Rect, WindowGeometry
from dantetl.languages.selection import DEFAULT_SOURCE
from dantetl.languages.source import SourceLanguage
from dantetl.ocr.text_content import cleaned_rows
from dantetl.ocr.values import MAX_OCR_REGIONS, OcrResult

if TYPE_CHECKING:
    from dantetl.ocr.observation import MatchingEvidence


@dataclass(frozen=True, slots=True)
class PixelEvidence:
    """Immutable exact-source digests; never retains pixels or a frame history."""

    sequence: int
    captured_at: float
    geometry: WindowGeometry
    crops: tuple[tuple[Rect, bytes], ...]

    def matches(self, frame: CapturedFrame) -> bool:
        return (self.sequence, self.captured_at, self.geometry) == (
            frame.sequence,
            frame.captured_at,
            frame.geometry,
        )


def prepare_evidence(
    result: OcrResult,
    bounds: tuple[Rect, ...] = (),
    *,
    language: SourceLanguage = DEFAULT_SOURCE,
    matching: MatchingEvidence | None = None,
) -> PixelEvidence:
    """Worker-only preparation of the bounded superset tracking may inspect.

    Track references and pending geometry occupy at most MAX_OCR_REGIONS slots; OCR
    contributes at most MAX_OCR_REGIONS. Compute each eligible crop once. The reader
    enforces the original one-million-pixel budget in *tracking's* query order, so unused
    proposals cannot consume retention evidence ahead of missed tracks.
    """
    rows = cleaned_rows(result, language, matching)
    queries = dict.fromkeys((*bounds[:MAX_OCR_REGIONS], *(row.bounds for row in rows)))
    reader = FrameEvidence(result.frame, pixel_budget=1024 * 65_536)
    crops = tuple(
        (box, digest) for box in queries if (digest := reader.fingerprint(box)) is not None
    )
    frame = result.frame
    return PixelEvidence(frame.sequence, frame.captured_at, frame.geometry, crops)


class EvidenceReader:
    """Cheap application-thread lookup with the original per-observation budget."""

    def __init__(
        self, frame: CapturedFrame, evidence: PixelEvidence | None, pixel_budget: int = 1_000_000
    ) -> None:
        self._crops = dict(evidence.crops) if evidence and evidence.matches(frame) else {}
        self._remaining = pixel_budget

    def fingerprint(self, bounds: Rect) -> bytes | None:
        digest = self._crops.get(bounds)
        area = bounds.width * bounds.height
        if digest is None or area > self._remaining:
            return None
        self._remaining -= area
        return digest


class FrameEvidence:
    """Hash small text crops with a per-observation budget, retaining only digests.

    Equal pixels support keeping previously recognized text. Changed pixels alone
    do not prove absence: the tracker must also receive negative OCR evidence.
    Unavailable/oversized crops return None and fall back to OCR confirmation.
    """

    def __init__(self, frame: CapturedFrame, pixel_budget: int = 1_000_000) -> None:
        self._frame = frame
        self._pixels = memoryview(frame.bgra)
        self._remaining = pixel_budget

    def fingerprint(self, bounds: Rect) -> bytes | None:
        frame = self._frame
        area = bounds.width * bounds.height
        if (
            min(bounds.size) <= 0
            or area > min(65_536, self._remaining)
            or len(self._pixels) != frame.width * frame.height * 4
            or not Rect(0, 0, frame.width, frame.height).contains(bounds)
        ):
            return None
        self._remaining -= area
        digest = blake2b(digest_size=16)
        for y in range(bounds.y, bounds.y + bounds.height):
            offset = (y * frame.width + bounds.x) * 4
            digest.update(self._pixels[offset : offset + bounds.width * 4])
        return digest.digest()
