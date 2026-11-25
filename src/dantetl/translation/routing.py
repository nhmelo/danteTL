"""Two explicit translation lanes, sharing scene validity but never blocking each other."""

from collections.abc import Callable
from pathlib import Path

from dantetl.i18n import tr
from dantetl.languages.selection import DEFAULT_SOURCE
from dantetl.languages.source import SourceLanguage
from dantetl.languages.target import ENGLISH, TargetLanguage
from dantetl.ocr.observation import TextTrack
from dantetl.translation.backend import TranslationBackend
from dantetl.translation.grouping import TextBlock
from dantetl.translation.ollama.values import OllamaIdentity
from dantetl.translation.plan import TranslationPlan
from dantetl.translation.selection import CloudIdentity, LocalIdentity
from dantetl.translation.values import (
    TranslationLaneSnapshot,
    TranslationRequest,
    TranslationSnapshot,
    TranslationState,
)
from dantetl.translation.worker import TranslationSession

type ModelIdentity = LocalIdentity | CloudIdentity | OllamaIdentity | str


class RoutedTranslationSession:
    """At most two model workers, with unchanged request routes pinned.

    Local preparation can hand over an already-started worker. Existing model
    prompts and loading contracts remain the responsibility of TranslationSession.
    """

    def __init__(
        self,
        plan: TranslationPlan,
        factories: dict[str, Callable[[], TranslationBackend]],
        cache_path: Path,
        *,
        language: SourceLanguage = DEFAULT_SOURCE,
        prepared: tuple[str, TranslationSession] | None = None,
        on_change: Callable[[], None] | None = None,
        identities: dict[str, ModelIdentity] | None = None,
        names: dict[str, str] | None = None,
    ) -> None:
        """Adopt one prepared owner and share one worker for identical assignments."""
        if prepared and prepared[0] not in plan.active_models:
            raise ValueError(tr("Prepared model does not belong to the active translation plan."))
        self._language = language
        self._cache_path, self._notify = cache_path, on_change
        self._plan = plan
        self._routes: dict[TranslationRequest, str] = {}
        self._stopping, self._generation = False, 0
        self._workers: dict[str, TranslationSession] = {}
        self._retiring: list[TranslationSession] = []
        self._factories = {k: factories[k] for k in plan.active_models if k in factories}
        self._identities = {k: (identities or {}).get(k, k) for k in plan.active_models}
        self._names = {k: (names or {}).get(k, k) for k in plan.active_models}
        unstarted = []
        for key in plan.active_models:
            if prepared and prepared[0] == key:
                self._workers[key] = prepared[1]
            elif key in factories:
                worker = TranslationSession(
                    factories[key], cache_path, on_change=on_change, language=language
                )
                self._workers[key] = worker
                unstarted.append(worker)
        self._unstarted = tuple(unstarted)

    def start(self) -> None:
        for worker in self._unstarted:
            worker.start()
        self._unstarted = ()

    @property
    def stopping(self) -> bool:
        return self._stopping

    def reconfigure(
        self,
        plan: TranslationPlan,
        factories: dict[str, Callable[[], TranslationBackend]],
        *,
        language: SourceLanguage | None = None,
        identities: dict[str, ModelIdentity] | None = None,
        names: dict[str, str] | None = None,
    ) -> None:
        """Keep unchanged owners; coalesce pending replacements until cleanup completes.

        Retiring workers never publish. No replacement starts until retirement finishes,
        so repeated edits cannot accumulate native models or cloud clients. Existing
        workers keep processing while a changed lane drains.
        """
        if self._stopping:
            raise ValueError(tr("Cannot reconfigure a stopping session."))
        desired = {k: factories[k] for k in plan.active_models if k in factories}
        next_identities = {k: (identities or {}).get(k, k) for k in plan.active_models}
        self._names = {k: (names or {}).get(k, k) for k in plan.active_models}
        source_changed = language is not None and language != self._language
        if language is not None:
            self._language = language
        if (
            not source_changed
            and plan == self._plan
            and desired.keys() == self._factories.keys()
            and next_identities == self._identities
        ):
            self._factories = desired
            return  # Display names/recipes can change without revoking live results.
        for key, worker in tuple(self._workers.items()):
            if (
                key not in desired
                or self._identities.get(key) != next_identities[key]
                or (source_changed and worker.finished)
            ):
                del self._workers[key]
                worker.request_stop()
                self._retiring.append(worker)
            elif source_changed:
                worker.set_source(self._language, desired[key])
        self._plan, self._factories = plan, desired
        self._identities = next_identities
        self._generation += 1
        self._routes.clear()
        self.advance()

    def advance(self) -> None:
        """Adopt completed cleanup without making snapshot reads change resource state."""
        self._retiring = [worker for worker in self._retiring if not worker.finished]
        if self._stopping or self._retiring:
            return
        for key, factory in self._factories.items():
            if key not in self._workers:
                worker = TranslationSession(
                    factory, self._cache_path, on_change=self._notify, language=self._language
                )
                self._workers[key] = worker
                worker.start()

    def update(
        self,
        tracks: tuple[TextTrack | TextBlock, ...],
        *,
        retain: tuple[TextBlock, ...] = (),
        target: TargetLanguage = ENGLISH,
        game: str = "",
        reset_context: bool = False,
        corrections: tuple[tuple[TextTrack, TextTrack], ...] = (),
    ) -> None:
        self.advance()

        def key(item: TextTrack | TextBlock) -> TranslationRequest:
            return (
                TranslationRequest.from_block(item)
                if isinstance(item, TextBlock)
                else TranslationRequest.from_track(item)
            )

        current = {key(item): item for item in (*tracks[:128], *retain[:128])}
        self._routes = {
            request: self._routes.get(request, self._destination(item))
            for request, item in list(current.items())[:128]
        }
        for remote, worker in self._workers.items():
            worker.update(
                tuple(item for item in tracks[:128] if self._routes.get(key(item)) == remote),
                retain=tuple(
                    item for item in retain[:128] if self._routes.get(key(item)) == remote
                ),
                reset_context=reset_context,
                target=target,
                game=game,
                corrections=corrections,
            )

    def _destination(self, item: TextTrack | TextBlock) -> str:
        return self._plan.model_for(item)

    def invalidate(self) -> None:
        self._routes.clear()
        for worker in self._workers.values():
            worker.invalidate()

    def request_stop(self) -> None:
        self._stopping = True
        self.invalidate()
        for worker in self._workers.values():
            worker.request_stop()
        for worker in self._retiring:
            worker.request_stop()

    @property
    def finished(self) -> bool:
        return (
            self._stopping
            and all(worker.finished for worker in self._workers.values())
            and all(worker.finished for worker in self._retiring)
        )

    def snapshot(self) -> TranslationSnapshot:
        lanes = {key: worker.snapshot() for key, worker in self._workers.items()}
        snapshots = list(lanes.values())
        pending_models = tuple(k for k in self._factories if k not in self._workers)
        if self._stopping:
            state = TranslationState.STOPPED if self.finished else TranslationState.STOPPING
        elif not pending_models and all(s.state == TranslationState.FAILED for s in snapshots):
            state = TranslationState.FAILED
        elif any(s.state == TranslationState.READY for s in snapshots):
            state = TranslationState.READY
        else:
            state = TranslationState.LOADING
        processing = [s.processing_since for s in snapshots if s.processing_since is not None]
        results = []
        for remote, snapshot in lanes.items():
            results.extend(
                result for result in snapshot.results if self._routes.get(result.request) == remote
            )
        return TranslationSnapshot(
            state,
            tuple(results),
            sum(s.pending for s in snapshots),
            max((s.last_batch_ms for s in snapshots), default=0),
            " · ".join(s.message for s in snapshots if s.message),
            processing_since=min(processing) if processing else None,
            queue_ms=max((s.queue_ms for s in snapshots), default=0),
            cache_hits=sum(s.cache_hits for s in snapshots),
            translated_sources=sum(s.translated_sources for s in snapshots),
            load_ms=max((s.load_ms for s in snapshots), default=0),
            last_inference_ms=max((s.last_inference_ms for s in snapshots), default=0),
            backend_generation=self._generation,
            routed_requests=frozenset(
                request
                for request, key in self._routes.items()
                if key in self._factories or key in self._workers
            ),
            cancelled_obsolete=sum(s.cancelled_obsolete for s in snapshots),
            rejected=tuple(r for s in snapshots for r in s.rejected),
            lanes=tuple(
                TranslationLaneSnapshot(
                    key,
                    self._names.get(key, key),
                    lanes[key].state if key in lanes else TranslationState.LOADING,
                    lanes[key].message
                    if key in lanes
                    else tr("Waiting for previous model to unload…"),
                )
                for key in (*lanes, *pending_models)
            ),
        )
