"""Run a selected backend on its owner thread, including optional menu decoding."""

import logging
from collections.abc import Callable

from dantetl.i18n import tr
from dantetl.languages.source import SourceLanguage
from dantetl.languages.target import ENGLISH, TargetLanguage
from dantetl.translation.backend import (
    BatchResult,
    ContextualBackend,
    DeferredBackend,
    LabelDecoder,
    MultilingualBackend,
    ParallelLabelBackend,
    RequestRejected,
    Source,
    TranslationBackend,
)
from dantetl.translation.cache import TranslationCache
from dantetl.translation.limits import MAX_ENGLISH_CHARACTERS, MAX_SOURCE_CHARACTERS
from dantetl.translation.schedule import Batch, Demand, TranslationSchedule

logger = logging.getLogger(__name__)
_LABEL_RUNTIME_ERRORS = (AttributeError, RuntimeError, ValueError, OSError, MemoryError)


class BackendExecution:
    def __init__(self, engine: TranslationBackend, schedule: TranslationSchedule) -> None:
        self._engine = engine
        self._schedule = schedule
        self._decoder: LabelDecoder | None = None
        try:
            self._width = engine.label_width if isinstance(engine, ParallelLabelBackend) else 0
        except BaseException:
            # The session only owns this execution once construction succeeds.
            try:
                engine.close()
            except Exception:
                logger.exception("Translation backend cleanup failed")
            raise

    def switch_source(self, language: SourceLanguage) -> bool:
        if not isinstance(self._engine, MultilingualBackend):
            return False
        self._close_decoder()
        self._engine.set_source(language)
        self._width = (
            self._engine.label_width if isinstance(self._engine, ParallelLabelBackend) else 0
        )
        return True

    def cancel_pending(self) -> None:
        if isinstance(self._engine, DeferredBackend):
            self._engine.cancel_pending()
        elif self._decoder is not None:
            self._decoder.reset()
        self._schedule.reset()

    def plan(self, demand: Demand, cache: TranslationCache) -> Batch | None:
        batch = self._schedule.plan(demand, self._engine, cache, self._width)
        if batch is None or not batch.parallel_label or self._decoder is not None:
            return batch
        assert isinstance(self._engine, ParallelLabelBackend)
        try:
            self._decoder = self._engine.prepare_labels(demand.target)
        except _LABEL_RUNTIME_ERRORS:
            logger.warning("Could not prepare parallel menu translation", exc_info=True)
        self._width = self._decoder.width if self._decoder else 0
        # Allocation can outlive a demand revision. Replan before any dispatch.
        return None

    def advance(self, interrupted: Callable[[], bool]) -> None:
        while self._decoder is not None and self._schedule.processing_since is not None:
            if self._schedule.has_completion() or interrupted():
                return
            try:
                self._decoder.step()
            except _LABEL_RUNTIME_ERRORS:
                self._fall_back()

    def dispatch(self, batch: Batch, wake: Callable[[object], None]) -> BatchResult | None:
        if not batch.missing:
            return {}, {}
        if batch.parallel_label and self._decoder is not None:
            try:
                future = self._decoder.submit(batch.missing[0], batch.target)
            except _LABEL_RUNTIME_ERRORS:
                self._fall_back()
                return None
            if future is not None:
                self._schedule.submitted(batch, future)
                return None
        if isinstance(self._engine, DeferredBackend):
            future = self._engine.submit_translation(batch.missing, target=batch.target)
            self._schedule.submitted(batch, future)
            future.add_done_callback(wake)
            return None
        return _translate_batch(self._engine, batch.missing, target=batch.target)

    def _fall_back(self) -> None:
        logger.warning("Menu decoding unavailable; resuming individual calls", exc_info=True)
        self._close_decoder()
        self._width = 0
        self._schedule.reset()

    def _close_decoder(self) -> None:
        if self._decoder is not None:
            decoder, self._decoder = self._decoder, None
            decoder.close()

    def close(self) -> None:
        try:
            self._close_decoder()
        finally:
            self._engine.close()


def _translate_batch(
    engine: TranslationBackend, sources: tuple[Source, ...], *, target: TargetLanguage = ENGLISH
) -> tuple[dict[Source, str], dict[Source, str]]:
    """Isolate known rejections. Split a rejected batch once; native failures stay fatal."""
    answers: dict[Source, str] = {}
    rejected = {
        source: tr("Source text exceeds the 512-character translation limit.")
        for source in sources
        if len(source[0]) > MAX_SOURCE_CHARACTERS
    }
    valid = tuple(source for source in sources if source not in rejected)

    def accept(source: Source, english: str) -> None:
        if not english.strip() or len(english) > MAX_ENGLISH_CHARACTERS:
            rejected[source] = tr("Translation backend returned empty or oversized text.")
        else:
            answers[source] = english

    if isinstance(engine, ContextualBackend):
        for source in valid:
            try:
                accept(source, engine.translate_with_context(*source, target=target))
            except RequestRejected as error:
                rejected[source] = str(error) or tr("Model rejected this request.")
    elif valid:
        try:
            translated = engine.translate(tuple(source[0] for source in valid), target=target)
        except RequestRejected as error:
            if len(valid) == 1:
                rejected[valid[0]] = str(error) or tr("Model rejected this request.")
            else:
                for source in valid:
                    good, bad = _translate_batch(engine, (source,), target=target)
                    answers.update(good)
                    rejected.update(bad)
        else:
            for source, english in zip(valid, translated, strict=True):
                accept(source, english)
    return answers, rejected
