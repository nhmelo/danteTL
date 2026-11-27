"""Sequence-scoped evidence exchanged by visual analysis and tracking."""

from dataclasses import dataclass, field
from weakref import ref

from dantetl.domain import Rect
from dantetl.languages.selection import DEFAULT_SOURCE
from dantetl.languages.source import SourceLanguage
from dantetl.ocr.lettering import Lettering
from dantetl.ocr.presence import PixelEvidence
from dantetl.ocr.values import MAX_OCR_REGIONS, OcrResult, TextRegion


@dataclass(frozen=True, slots=True)
class TextTrack:
    track_id: int
    text: str
    bounds: Rect
    vertical: bool = False
    text_revision: int = 1


@dataclass(frozen=True, slots=True)
class MatchingEvidence:
    """Pure scoring output for one exact OCR source and immutable prior track state."""

    observation: ref[OcrResult] = field(compare=False, repr=False)
    sources: tuple[TextTrack, ...]
    limit: int
    content: tuple[TextRegion, ...]
    regions: tuple[tuple[int, TextRegion], ...]
    normalized: tuple[tuple[int, str], ...]
    candidates: tuple[tuple[float, int, int], ...]
    dash_edges: tuple[tuple[int, int], ...]
    lettering: tuple[tuple[int, Lettering], ...]
    language: SourceLanguage

    def matches(
        self,
        result: OcrResult,
        sources: tuple[TextTrack, ...],
        limit: int = MAX_OCR_REGIONS,
        *,
        language: SourceLanguage = DEFAULT_SOURCE,
    ) -> bool:
        return self.sources == sources and self.describes(result, language, limit)

    def describes(
        self, result: OcrResult, language: SourceLanguage, limit: int = MAX_OCR_REGIONS
    ) -> bool:
        return self.observation() is result and self.language == language and self.limit == limit


@dataclass(frozen=True, slots=True)
class CandidateReview:
    """Comparisons and prepared scores for one completed OCR source."""

    sequence: int = 0
    checked: frozenset[TextTrack] = frozenset()
    unchanged: frozenset[TextTrack] = frozenset()
    pixels: PixelEvidence | None = None
    matching: MatchingEvidence | None = None


@dataclass(frozen=True, slots=True)
class TrackPresence:
    """Anchor observations on a capture, which may be newer than the OCR source."""

    sequence: int = 0
    absent: frozenset[TextTrack] = frozenset()
    pattern_loss: frozenset[TextTrack] = frozenset()
    present: frozenset[TextTrack] = frozenset()
