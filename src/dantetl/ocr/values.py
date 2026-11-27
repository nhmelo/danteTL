"""Immutable OCR observations and worker status; physical pixel coordinates."""

from dataclasses import dataclass, replace
from enum import StrEnum

from dantetl.domain import CapturedFrame, Rect

MAX_TRACKS = 128
"""Text tracks retained and analysed per source by default."""

MAX_OCR_REGIONS = 4 * MAX_TRACKS
"""OCR rows inspected per reading. Exceeding the track slots keeps detector order from evicting
an existing track on a dense screen."""


@dataclass(frozen=True, slots=True)
class TextCharacter:
    text: str
    bounds: Rect
    confidence: float


@dataclass(frozen=True, slots=True)
class TextRegion:
    """Client-relative physical pixels; confidence is mean retained-character confidence.

    ``separated`` marks a label that layout evidence (its own icon cell) split from a merged
    line; tracking lets such labels replace a retained line that holds them.
    """

    text: str
    bounds: Rect
    characters: tuple[TextCharacter, ...] = ()
    vertical: bool = False
    line_confidence: float | None = None
    separated: bool = False

    def translated(self, x: int, y: int) -> "TextRegion":
        def moved(b: Rect) -> Rect:
            return replace(b, x=b.x + x, y=b.y + y)

        return replace(
            self,
            bounds=moved(self.bounds),
            characters=tuple(replace(c, bounds=moved(c.bounds)) for c in self.characters),
        )

    @property
    def confidence(self) -> float | None:
        if not self.characters:
            return self.line_confidence
        return sum(char.confidence for char in self.characters) / len(self.characters)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class OcrResult:
    frame: CapturedFrame
    regions: tuple[TextRegion, ...]
    duration_ms: float
    completed_at: float
    scanned: tuple[Rect, ...] | None = None

    def covers(self, bounds: Rect) -> bool:
        return self.scanned is None or any(area.contains(bounds) for area in self.scanned)


class OcrState(StrEnum):
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class OcrSnapshot:
    state: OcrState
    result: OcrResult | None = None
    message: str = ""
    processed_frames: int = 0
    skipped_frames: int = 0
    queue_ms: float = 0
    mean_duration_ms: float = 0
    processing_since: float | None = None
    provider: str = ""
