"""Bounded visual evidence snapshot, independent of the analysis worker."""

from dataclasses import dataclass

from dantetl.domain import CapturedFrame, Rect
from dantetl.ocr.observation import CandidateReview, TextTrack, TrackPresence


@dataclass(frozen=True, slots=True)
class MenuSample:
    line: TextTrack
    sequence: int
    guard: Rect


@dataclass(frozen=True, slots=True)
class SettledScan:
    deadline: float
    areas: tuple[Rect, ...] | None


@dataclass(frozen=True, slots=True, kw_only=True)
class VisualState:
    frame: CapturedFrame | None = None
    request_ocr: bool = False
    hidden: frozenset[TextTrack] = frozenset()
    blocked: frozenset[int] = frozenset()
    stable_regions: tuple[int, ...] = ()
    ocr_sequence: int = 0
    duration_ms: float = 0
    avoided: int = 0
    message: str = ""
    review: CandidateReview = CandidateReview()
    presence: TrackPresence = TrackPresence()
    scan_areas: tuple[Rect, ...] | None = None
    stable_tracks: tuple[tuple[int, int, str], ...] = ()
    changed_areas: tuple[Rect, ...] = ()
    cleared: frozenset[TextTrack] = frozenset()
    rejected_sequence: int = 0
    review_message: str = ""
    complete_lines: frozenset[TextTrack] = frozenset()
    matching_menus: tuple[MenuSample, ...] = ()
    settled_scan: SettledScan | None = None
    matching_lines: frozenset[TextTrack] = frozenset()
