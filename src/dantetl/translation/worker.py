"""Local translation scheduling, backend swapping, and stale-result rejection."""

import logging
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dantetl.i18n import tr
from dantetl.languages.selection import DEFAULT_SOURCE
from dantetl.languages.source import SourceLanguage
from dantetl.languages.target import ENGLISH, TargetLanguage
from dantetl.ocr.observation import TextTrack
from dantetl.translation.backend import (
    BatchResult,
    TranslationBackend,
)
from dantetl.translation.cache import TranslationCache
from dantetl.translation.cache_ownership import CacheScope
from dantetl.translation.dialogue.context import DialogueMemory, DialogueTurn
from dantetl.translation.execution import BackendExecution
from dantetl.translation.grouping import TextBlock
from dantetl.translation.schedule import Batch, Demand, TranslationSchedule
from dantetl.translation.values import (
    TranslatedText,
    TranslationRejection,
    TranslationRequest,
    TranslationSnapshot,
    TranslationState,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _FirstAnswer:
    translation: TranslatedText
    # Only a newly inferred answer needs persistence once confirmed.
    cache_key: str | None
    scope: CacheScope | None


class TranslationSession:
    """One model owner or two independent cloud calls; at most 128 current requests.

    The UI replaces the desired set, never appends a job queue. Models and SQLite
    belong to this worker. A swap drains the active call, closes the old backend,
    then loads the latest requested backend before accepting results again.
    """

    def __init__(
        self,
        factory: Callable[[], TranslationBackend],
        cache_path: Path,
        *,
        on_change: Callable[[], None] | None = None,
        language: SourceLanguage = DEFAULT_SOURCE,
    ) -> None:
        self._language = language
        self._target = ENGLISH
        self._game = ""
        self._factory, self._cache_path = factory, cache_path
        self._notify = on_change or (lambda: None)
        self._condition = threading.Condition()
        self._done = threading.Event()
        self._stop = False
        self._generation = 0
        self._backend_generation = 0
        self._desired: tuple[TranslationRequest, ...] = ()
        self._candidates: tuple[TranslationRequest, ...] = ()
        self._first_answers: dict[TranslationRequest, _FirstAnswer] = {}
        self._cache_writes: deque[_FirstAnswer] = deque(maxlen=128)
        self._retained: tuple[TranslationRequest, ...] = ()
        self._demand_changed = False
        self._cancelled_obsolete = 0
        self._results: dict[TranslationRequest, TranslatedText] = {}
        self._state = TranslationState.LOADING
        self._message = ""
        self._duration = 0.0
        self._processing_since: float | None = None
        self._queued_at: dict[TranslationRequest, float] = {}
        self._queue_ms = 0.0
        self._cache_hits = 0
        self._translated_sources = 0
        self._load_ms = 0.0
        self._last_inference_ms = 0.0
        self._dialogue = DialogueMemory()
        self._blocks: dict[TranslationRequest, TextBlock] = {}
        self._contexts: dict[TranslationRequest, tuple[DialogueTurn, ...]] = {}
        self._rejections: OrderedDict[tuple[TranslationRequest, tuple[DialogueTurn, ...]], str] = (
            OrderedDict()
        )
        self._thread = threading.Thread(target=self._run, name="translation-owner", daemon=True)

    def start(self) -> None:
        self._thread.start()

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
    ) -> None:
        desired = tuple(
            TranslationRequest.from_block(t)
            if isinstance(t, TextBlock)
            else TranslationRequest.from_track(t)
            for t in tracks
            if self._language.contains_text(t.text)
        )[:128]
        desired_set = set(desired)
        provisional = tuple(
            key
            for block in candidates[:128]
            if self._language.contains_text(block.text)
            and (key := TranslationRequest.from_block(block)) not in desired_set
        )[: 128 - len(desired)]
        demanded = desired_set.union(provisional)
        retained = tuple(
            key
            for block in retain[:128]
            if (key := TranslationRequest.from_block(block)) not in demanded
        )[: 128 - len(desired) - len(provisional)]
        with self._condition:
            if target != self._target or game != self._game:
                self._target = target
                self._game = game
                self._invalidate_locked()
            now = time.monotonic()
            blocks = tuple(t for t in tracks[:128] if isinstance(t, TextBlock))
            allowed = demanded.union(retained)
            self._blocks = {
                key: b
                for b in (*retain[:128], *candidates[:128], *blocks)
                if (key := TranslationRequest.from_block(b)) in allowed
            }
            if reset_context:
                self._dialogue.reset()
            self._dialogue.observe(retain[:128] or blocks, now, corrections=corrections)
            contexts = {
                key: () if reset_context else context
                for key, context in self._contexts.items()
                if key in allowed
            }
            first_readings = set(provisional).union(self._candidates)
            for key in (*desired, *provisional):
                if key not in contexts or key in first_readings:
                    contexts[key] = (
                        self._dialogue.context_for(self._blocks[key]) if key in self._blocks else ()
                    )
            if (
                self._stop
                or self._state == TranslationState.FAILED
                or (
                    desired == self._desired
                    and provisional == self._candidates
                    and retained == self._retained
                    and contexts == self._contexts
                )
            ):
                return
            self._desired, self._retained = desired, retained
            self._candidates = provisional
            self._first_answers = {
                key: answer
                for key, answer in self._first_answers.items()
                if key in demanded and self._contexts.get(key) == contexts.get(key)
            }
            self._contexts = contexts
            self._demand_changed = True
            self._queued_at = {key: self._queued_at.get(key, now) for key in demanded}
            self._results = {key: value for key, value in self._results.items() if key in allowed}
            for key in desired:
                if (answer := self._first_answers.pop(key, None)) is not None:
                    self._publish_answer(answer.translation, now)
                    if answer.cache_key is not None:
                        self._cache_writes.append(answer)
            self._condition.notify()

    def invalidate(self) -> None:
        with self._condition:
            self._invalidate_locked()

    def _invalidate_locked(self) -> None:
        self._generation += 1
        self._desired = ()
        self._candidates = ()
        self._first_answers.clear()
        self._retained = ()
        self._results.clear()
        self._rejections.clear()
        self._queued_at.clear()
        self._dialogue.reset()
        self._contexts.clear()
        self._blocks.clear()
        self._condition.notify()

    def set_source(
        self, language: SourceLanguage, factory: Callable[[], TranslationBackend]
    ) -> None:
        with self._condition:
            self._factory = factory
            if self._language != language:
                self._language = language
                self._invalidate_locked()

    def switch_backend(self, factory: Callable[[], TranslationBackend]) -> None:
        with self._condition:
            if self._stop or self._state == TranslationState.FAILED:
                return
            self._factory = factory
            self._backend_generation += 1
            self._first_answers.clear()
            self._generation += 1
            self._results.clear()
            self._rejections.clear()
            self._state = TranslationState.LOADING
            self._message = ""
            self._queued_at = {key: time.monotonic() for key in self._desired}
            self._duration = self._queue_ms = self._load_ms = 0
            self._last_inference_ms = 0
            self._dialogue.reset()
            self._contexts.clear()
            self._cache_hits = self._translated_sources = 0
            self._cancelled_obsolete = 0
            self._condition.notify()

    def request_stop(self) -> None:
        with self._condition:
            self._stop = True
            self._candidates = ()
            self._first_answers.clear()
            self._desired = ()
            self._retained = ()
            self._results.clear()
            self._rejections.clear()
            if self._state not in (TranslationState.FAILED, TranslationState.STOPPED):
                self._state = TranslationState.STOPPING
            self._condition.notify()

    @property
    def finished(self) -> bool:
        return self._done.is_set() and not self._thread.is_alive()

    def snapshot(self) -> TranslationSnapshot:
        with self._condition:
            rejected = tuple(
                TranslationRejection(key, message)
                for key in self._desired
                if (message := self._rejections.get((key, self._contexts.get(key, ()))))
            )
            return TranslationSnapshot(
                self._state,
                tuple(self._results.values()),
                sum(self._needs_translation(key) for key in self._desired),
                self._duration,
                " · ".join(
                    filter(
                        None,
                        (
                            self._message,
                            tr(
                                "{count} text block(s) rejected: {message}",
                                count=len(rejected),
                                message=rejected[0].message,
                            )
                            if rejected
                            else "",
                        ),
                    )
                ),
                processing_since=self._processing_since,
                queue_ms=self._queue_ms,
                cache_hits=self._cache_hits,
                translated_sources=self._translated_sources,
                load_ms=self._load_ms,
                last_inference_ms=self._last_inference_ms,
                backend_generation=self._backend_generation,
                rejected=rejected,
                cached_candidates=tuple(
                    answer.translation
                    for answer in self._first_answers.values()
                    if answer.cache_key is None
                ),
                cancelled_obsolete=self._cancelled_obsolete,
            )

    def _needs_translation(self, key: TranslationRequest) -> bool:
        return (
            key not in self._results and (key, self._contexts.get(key, ())) not in self._rejections
        )

    def _jobs(self) -> tuple[TranslationRequest, ...]:
        return tuple(
            key
            for key in (*self._desired, *self._candidates)
            if key not in self._first_answers and self._needs_translation(key)
        )

    def _publish_answer(self, translation: TranslatedText, now: float) -> None:
        key = translation.request
        self._results[key] = translation
        if key in self._blocks:
            self._dialogue.published(self._blocks[key], translation.english, now)

    def _flush_confirmed(self, cache: TranslationCache) -> None:
        with self._condition:
            writes = tuple(self._cache_writes)
            self._cache_writes.clear()
        for answer in writes:
            if answer.cache_key is not None:
                cache.put(answer.cache_key, answer.translation.english, scope=answer.scope)

    def _wake(self, _: object = None) -> None:
        with self._condition:
            self._condition.notify()

    def _interrupted(self, generation: int) -> bool:
        with self._condition:
            return self._stop or self._demand_changed or generation != self._generation

    def _run(self) -> None:
        execution: BackendExecution | None = None
        cache: TranslationCache | None = None
        active_backend = -1
        schedule = TranslationSchedule()
        active_language = None
        try:
            cache = TranslationCache(self._cache_path)
            while True:
                with self._condition:
                    stale = self._stop or schedule.stale(self._generation)
                if stale and execution is not None:
                    execution.cancel_pending()
                    with self._condition:
                        self._processing_since = None
                completed = schedule.completed()
                for batch, future in completed:
                    self._publish_batch(
                        batch,
                        future.result(),
                        cache,
                        inference=True,
                        remaining_since=schedule.processing_since,
                    )
                if completed:
                    continue
                with self._condition:
                    self._condition.wait_for(
                        lambda active=active_backend: (
                            self._stop
                            or active != self._backend_generation
                            or self._demand_changed
                            or schedule.has_completion()
                            or schedule.stale(self._generation)
                            or schedule.needs_plan(
                                self._generation,
                                ((key, self._contexts.get(key, ())) for key in self._jobs()),
                            )
                        )
                    )
                    if self._stop:
                        return
                    if schedule.has_completion() or schedule.stale(self._generation):
                        continue
                    backend_generation, factory = self._backend_generation, self._factory
                    if self._demand_changed:
                        self._demand_changed = False
                        if schedule.processing_since is not None:
                            wanted = {
                                (key.source, self._contexts.get(key, ()))
                                for key in (*self._desired, *self._retained)
                            }
                            self._cancelled_obsolete += schedule.cancel_unused(wanted)
                            self._processing_since = schedule.processing_since
                    jobs = self._jobs()
                    contexts: dict[TranslationRequest, tuple[DialogueTurn, ...]] = {
                        key: self._contexts.get(key) or () for key in jobs
                    }
                    demand = Demand(
                        self._generation,
                        jobs,
                        contexts,
                        frozenset(
                            key
                            for key in jobs
                            if key in self._blocks and self._blocks[key].dialogue
                        ),
                        target=self._target,
                        language=self._language,
                        game=self._game,
                        candidates=frozenset(self._candidates),
                    )
                self._flush_confirmed(cache)
                if (
                    active_backend == backend_generation
                    and active_language != demand.language
                    and execution is not None
                    and execution.switch_source(demand.language)
                ):
                    active_language = demand.language
                if active_backend != backend_generation or active_language != demand.language:
                    if execution is not None:
                        previous, execution = execution, None
                        previous.close()
                    started = time.perf_counter()
                    execution = BackendExecution(factory(), schedule)
                    active_backend, active_language = backend_generation, demand.language
                    schedule.reset()
                    with self._condition:
                        published = not self._stop and active_backend == self._backend_generation
                        if published:
                            self._state = TranslationState.READY
                            self._message = cache.warning
                            self._load_ms = (time.perf_counter() - started) * 1000
                    if published:
                        self._notify()
                    continue
                assert execution is not None
                batch = execution.plan(demand, cache)
                if batch is None:
                    execution.advance(lambda gen=demand.generation: self._interrupted(gen))
                    with self._condition:
                        self._processing_since = schedule.processing_since
                    continue
                with self._condition:
                    if (
                        self._stop
                        or batch.generation != self._generation
                        or (self._demand_changed and (demand.candidates or self._candidates))
                    ):
                        continue
                    if not batch.activate(
                        (*self._desired, *self._candidates),
                        self._contexts,
                        self._queued_at,
                        now=time.monotonic(),
                        measured=time.perf_counter(),
                    ):
                        continue
                    if batch.missing:
                        since = schedule.processing_since
                        self._processing_since = (
                            min(batch.started, since) if since is not None else batch.started
                        )
                outcome = execution.dispatch(batch, self._wake)
                if outcome is None:
                    with self._condition:
                        self._processing_since = schedule.processing_since
                else:
                    self._publish_batch(
                        batch,
                        outcome,
                        cache,
                        inference=bool(batch.missing),
                        remaining_since=schedule.processing_since,
                    )
        except Exception as error:
            logger.exception("Translation failed")
            with self._condition:
                self._state = TranslationState.FAILED
                self._results.clear()
                self._first_answers.clear()
                self._message = str(error)
            self._notify()
        finally:
            if execution is not None:
                try:
                    execution.close()
                except Exception:
                    logger.exception("Translation backend cleanup failed")
            if cache is not None:
                try:
                    self._flush_confirmed(cache)
                    cache.close()
                except Exception:
                    logger.exception("Translation cache cleanup failed")
            with self._condition:
                self._processing_since = None
                if self._state != TranslationState.FAILED:
                    self._state = TranslationState.STOPPED
            self._done.set()
            self._notify()

    def _publish_batch(
        self,
        batch: Batch,
        outcome: BatchResult,
        cache: TranslationCache,
        *,
        inference: bool,
        remaining_since: float | None = None,
    ) -> None:
        translated, rejected_sources = outcome
        if batch.parallel_label:
            with self._condition:
                if (
                    self._stop
                    or batch.generation != self._generation
                    or not any(
                        (key.source, self._contexts.get(key, ())) == batch.missing[0]
                        for key in (*self._desired, *self._retained)
                    )
                ):
                    self._processing_since = remaining_since
                    return
        for source, english in translated.items():
            if not batch.speculative:
                cache.put(batch.keys[source], english, scope=batch.cache_scope)
            batch.answers[source] = english
        duration = (time.perf_counter() - batch.measured) * 1000
        with self._condition:
            if inference:
                self._processing_since = remaining_since
            published = not self._stop and batch.generation == self._generation
            if published:
                confirmed = set(self._desired)
                current = confirmed.union(self._candidates)
                for key in batch.jobs:
                    if key not in current or batch.contexts[key] != self._contexts.get(key, ()):
                        continue
                    source = batch.inputs[key]
                    if source in rejected_sources:
                        failure = (key, batch.contexts[key])
                        self._rejections[failure] = rejected_sources[source][:512]
                        self._rejections.move_to_end(failure)
                        while len(self._rejections) > 256:
                            self._rejections.popitem(last=False)
                    english = batch.answers.get(source)
                    if english is not None:
                        translation = TranslatedText(key, english, batch.target)
                        answer = _FirstAnswer(
                            translation,
                            batch.keys[source] if source in translated else None,
                            batch.cache_scope,
                        )
                        if key not in confirmed:
                            self._first_answers[key] = answer
                            continue
                        self._publish_answer(translation, time.monotonic())
                        if batch.speculative:
                            self._cache_writes.append(answer)
                            self._demand_changed = True
                self._duration, self._queue_ms = duration, batch.queue_ms
                self._cache_hits = len(batch.sources) - len(batch.missing)
                self._translated_sources = len(batch.missing) - len(rejected_sources)
                if inference:
                    self._last_inference_ms = duration
                self._message = cache.warning
        if published:
            self._notify()
