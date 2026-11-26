"""Immutable translation inputs and published results; no worker or cache construction."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from dantetl.languages.target import ENGLISH, TargetLanguage
from dantetl.ocr.observation import TextTrack
from dantetl.translation.grouping import TextBlock


@dataclass(frozen=True, slots=True)
class TranslationRequest:
    track_id: int
    text_revision: int
    source: str
    members: tuple[tuple[int, int, str], ...] = ()

    @classmethod
    def from_track(cls, track: TextTrack) -> "TranslationRequest":
        return cls(track.track_id, track.text_revision, track.text)

    @classmethod
    def from_block(cls, block: TextBlock) -> "TranslationRequest":
        first = block.lines[0]
        if len(block.lines) == 1:
            return cls.from_track(first)
        return cls(first.track_id, first.text_revision, block.text, block.key)

    @property
    def line_sources(self) -> tuple[tuple[int, int, str], ...]:
        return self.members or ((self.track_id, self.text_revision, self.source),)


@dataclass(frozen=True, slots=True)
class TranslatedText:
    request: TranslationRequest
    english: str
    target: TargetLanguage = ENGLISH


@dataclass(frozen=True, slots=True)
class TranslationRejection:
    request: TranslationRequest
    message: str


class TranslationState(StrEnum):
    LOADING = "loading"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TranslationLaneSnapshot:
    reference: str
    name: str
    state: TranslationState
    message: str = ""


@dataclass(frozen=True, slots=True)
class TranslationSnapshot:
    state: TranslationState
    results: tuple[TranslatedText, ...] = ()
    pending: int = 0
    last_batch_ms: float = 0
    message: str = ""
    processing_since: float | None = None
    queue_ms: float = 0
    cache_hits: int = 0
    translated_sources: int = 0
    load_ms: float = 0
    last_inference_ms: float = 0
    backend_generation: int = 0
    rejected: tuple[TranslationRejection, ...] = ()
    lanes: tuple[TranslationLaneSnapshot, ...] = ()
    cancelled_obsolete: int = 0
    # A single backend accepts every retained request. A router pins an explicit subset.
    routed_requests: frozenset[TranslationRequest] | None = None
    cached_candidates: tuple[TranslatedText, ...] = ()


class TranslationWorker(Protocol):
    def update(
        self,
        tracks: tuple[TextTrack | TextBlock, ...],
        *,
        retain: tuple[TextBlock, ...] = (),
        candidates: tuple[TextBlock, ...] = (),
        target: TargetLanguage = ENGLISH,
        game: str = "",
        reset_context: bool = False,
        corrections: tuple[tuple[TextTrack, TextTrack], ...] = (),
    ) -> None: ...
    def invalidate(self) -> None: ...
    def snapshot(self) -> TranslationSnapshot: ...
