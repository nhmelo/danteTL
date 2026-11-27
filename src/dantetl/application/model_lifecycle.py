"""Committed model/route transitions and worker ownership, independent of Qt presentation."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum

from dantetl.application.model_inventory import ModelInventory
from dantetl.application.translation_activity import (
    InactiveReason,
    TranslationActivity,
    TranslationMode,
)
from dantetl.i18n import tr
from dantetl.languages.selection import DEFAULT_SOURCE
from dantetl.languages.source import SourceLanguage
from dantetl.translation.backend import BackendChoice
from dantetl.translation.cloud.values import CloudProfile
from dantetl.translation.ollama.values import OllamaIdentity, OllamaModel
from dantetl.translation.plan import TranslationPlan
from dantetl.translation.routing import RoutedTranslationSession
from dantetl.translation.selection import CloudIdentity, LocalIdentity
from dantetl.translation.values import TranslationSnapshot, TranslationState, TranslationWorker
from dantetl.translation.worker import TranslationSession

Worker = TranslationSession | RoutedTranslationSession
logger = logging.getLogger(__name__)


class ModelPhase(StrEnum):
    IDLE = "idle"
    CONSENT = "consent"
    DRAINING = "draining"
    LOADING = "loading"
    PREPARING = "preparing"
    READY = "ready"
    FAILED = "failed"
    CLOSING = "closing"


@dataclass(frozen=True, slots=True)
class RouteModel:
    reference: str
    identity: LocalIdentity | CloudIdentity | OllamaIdentity
    available: bool
    choice: BackendChoice | CloudProfile | OllamaModel = field(compare=False)


@dataclass(frozen=True)
class RouteSelection:
    game: str = ""
    plan: TranslationPlan = TranslationPlan()
    models: tuple[RouteModel, ...] = ()
    language: SourceLanguage = DEFAULT_SOURCE

    def prepared_choice(self, loaded: BackendChoice) -> BackendChoice | None:
        return next(
            (
                m.choice
                for m in self.models
                if m.available
                and isinstance(m.choice, BackendChoice)
                and m.choice.matches_preparation(loaded)
            ),
            None,
        )


@dataclass(frozen=True)
class ModelSnapshot:
    phase: ModelPhase
    selected: BackendChoice | None
    proposed: BackendChoice | None
    route: RouteSelection
    standby: bool
    message: str = ""
    assignment: TranslationPlan | None = None
    preparing: bool = False
    preparation_revision: int = 0
    worker: TranslationSnapshot | None = None


@dataclass(frozen=True)
class PreparationOutcome:
    committed: bool
    error: str = ""


@dataclass(frozen=True)
class PreparedWorker:
    """A live worker and the exact model it loaded, independent of current selection."""

    choice: BackendChoice
    worker: TranslationSession


RouteInput = PreparedWorker | RoutedTranslationSession | None


@dataclass(frozen=True, slots=True)
class _Consent:
    """A staged assignment waiting for the answer to one consent token."""

    token: int
    choice: BackendChoice
    plan: TranslationPlan


@dataclass(frozen=True, slots=True)
class _Preparation:
    """An accepted model being prepared on its own worker.

    ``plan`` becomes None once the choice is revoked or the app closes, and never comes back.
    ``worker`` is created only after the previous owner has drained.
    """

    choice: BackendChoice
    plan: TranslationPlan | None
    worker: TranslationSession | None = None


_Operation = _Consent | _Preparation | None


class ModelLifecycle:
    def __init__(
        self,
        create: Callable[[BackendChoice], TranslationSession],
        inventory: ModelInventory,
    ) -> None:
        self._create, self._inventory = create, inventory
        self._session: Worker | None = None
        self._worker_choice: BackendChoice | None = None
        self._selected: BackendChoice | None = None
        self._assignment: TranslationPlan | None = None
        self._operation: _Operation = None
        self._consent_serial = 0
        # Outcome of the last completed operation; an operation or closing overrides it.
        self._settled = ModelPhase.IDLE
        self._closing = False
        self._message = ""
        self._standby = False
        self._probing = False
        self._prepared_this_run: set[str] = set()
        self._preparation_revision = 0
        self._route = RouteSelection()
        self._route_factory: Callable[[RouteInput], RoutedTranslationSession] | None = None

    @property
    def _phase(self) -> ModelPhase:
        operation = self._operation
        if self._closing:
            return ModelPhase.CLOSING
        if isinstance(operation, _Consent):
            return ModelPhase.CONSENT
        if isinstance(operation, _Preparation):
            ready = operation.worker is not None and operation.plan is not None
            return ModelPhase.PREPARING if ready else ModelPhase.DRAINING
        return self._settled

    @property
    def _preparing(self) -> BackendChoice | None:
        operation = self._operation
        return operation.choice if isinstance(operation, _Preparation) else None

    def snapshot(self) -> ModelSnapshot:
        phase, message = self._phase, self._message
        worker = self._session.snapshot() if self._session else None
        if phase == ModelPhase.READY and worker:
            phase = {
                TranslationState.LOADING: ModelPhase.LOADING,
                TranslationState.STOPPING: ModelPhase.DRAINING,
                TranslationState.FAILED: ModelPhase.FAILED,
            }.get(worker.state, phase)
            message = worker.message
        return ModelSnapshot(
            phase,
            self._selected,
            self._operation.choice if self._operation else None,
            self._route,
            self._standby,
            message,
            self._assignment,
            self._preparing is not None,
            self._preparation_revision,
            worker,
        )

    @property
    def preparation_message(self) -> str | None:
        """Current preparation status, available without sampling a model worker."""
        return self._message if self._preparing is not None else None

    @property
    def selected(self) -> BackendChoice | None:
        return self._selected

    @property
    def assignment(self) -> TranslationPlan | None:
        """Committed immutable policy; reading it never polls workers."""
        return self._assignment

    @property
    def preparation_revision(self) -> int:
        """Invalidate cached model options only when this run adds a prepared model."""
        return self._preparation_revision

    @property
    def publication(self) -> TranslationWorker | None:
        """Narrow publication/invalidation port; worker lifecycle stays owned here."""
        return self._session

    def _route_input(self) -> RouteInput:
        if isinstance(self._session, TranslationSession):
            assert self._worker_choice is not None
            return PreparedWorker(self._worker_choice, self._session)
        return self._session

    def is_prepared(self, choice: BackendChoice) -> bool:
        return not choice.unavailable_reason and (
            choice.preparation_id in self._prepared_this_run or self._inventory.prepared(choice)
        )

    def restore_selection(
        self, choice: BackendChoice | None, *, assignment: TranslationPlan | None = None
    ) -> None:
        if self._session or self._preparing:
            raise ValueError(tr("Initial selection can only be restored before loading a worker."))
        self._assignment = assignment
        self._selected = choice if choice and self.is_prepared(choice) else None
        self._settled = ModelPhase.READY if self._selected else ModelPhase.IDLE

    def refresh_choices(self, choices: tuple[BackendChoice, ...]) -> None:
        if self._selected:
            self._selected = next((b for b in choices if b.key == self._selected.key), None)

    def _allows_translation(self, *, running: bool, enabled: bool) -> bool:
        return (
            running
            and enabled
            and any(m.available for m in self._route.models)
            and not self._preparing
            and not self._probing
            and not self._closing
        )

    @property
    def assignment_busy(self) -> bool:
        return self._operation is not None

    def request_assignment(self, plan: TranslationPlan, choice: BackendChoice | None) -> int | None:
        """Stage assignments for consent without touching the committed route/worker."""
        if self.assignment_busy or self._probing or self._closing:
            raise ValueError(tr("Wait for the current model operation to finish."))
        if choice and choice.unavailable_reason:
            raise ValueError(choice.unavailable_reason)
        if choice and not self.is_prepared(choice):
            self._consent_serial += 1
            self._operation = _Consent(self._consent_serial, choice, plan)
            return self._consent_serial
        self._assignment = plan
        if choice:
            self._selected, self._standby = choice, True
            self._settled, self._message = ModelPhase.READY, ""
        elif not plan.active_models:
            self._selected, self._standby = None, False
        return None

    def consent(self, token: int, accepted: bool) -> BackendChoice | None:
        operation = self._operation
        if not isinstance(operation, _Consent) or token != operation.token or self._closing:
            return None
        choice = operation.choice
        if not accepted:
            self._operation = None
            self._settled = ModelPhase.READY if self._selected else ModelPhase.IDLE
            return choice
        self._operation = _Preparation(choice, operation.plan)
        self._standby = True
        self._message = tr("Finishing the previous model…")
        if self._session:
            self._session.request_stop()
        return choice

    def advance_preparation(self, *, closing: bool = False) -> PreparationOutcome | None:
        operation = self._operation
        if not isinstance(operation, _Preparation):
            return None
        choice = self._inventory.current_preparation(operation.choice)
        session = self._session
        committed, error = False, ""
        if closing:
            self.shutdown()
            if session and not session.finished:
                return None
        elif choice is None or operation.plan is None:
            # Revocation survives a later rediscovery while the stale owner drains.
            self._operation = replace(operation, plan=None)
            self._standby = False
            self.drain()
            if self._session is not None:
                self._message = tr("Finishing the previous model…")
                return None
            self._settled = ModelPhase.FAILED
            error = tr("Model unavailable · open Settings")
        elif operation.worker is None:
            if session and not session.finished:
                return None
            worker = self._create(choice)
            self._session, self._worker_choice = worker, choice
            self._operation = replace(operation, worker=worker)
            worker.start()
            self._message = tr("Checking model files and running a test translation…")
            return None
        else:
            # Until revocation or closing, nothing else may drain the preparation's own worker.
            result = operation.worker.snapshot()
            if result.state not in (TranslationState.READY, TranslationState.FAILED):
                return None
            if result.state == TranslationState.FAILED and not operation.worker.finished:
                return None
            if result.state == TranslationState.READY:
                committed = True
                if choice.preparation_id not in self._prepared_this_run:
                    self._prepared_this_run.add(choice.preparation_id)
                    self._preparation_revision += 1
                try:
                    self._inventory.record_prepared(choice.preparation_id)
                except OSError as error_record:
                    logger.warning("Could not save preparation receipt: %s", error_record)
                self._selected, self._standby = choice, True
                self._settled = ModelPhase.READY
                self._assignment = operation.plan
            else:
                self._session, self._standby = None, False
                error = result.message
                self._settled = ModelPhase.FAILED
        self._message = error
        self._operation = None
        return PreparationOutcome(committed, error)

    def configure_route(
        self,
        route: RouteSelection,
        factory: Callable[[RouteInput], RoutedTranslationSession],
    ) -> bool:
        previous = self._route
        changed = route != previous
        self._route, self._route_factory = route, factory
        if (
            changed
            and isinstance(self._session, TranslationSession)
            and self._worker_choice
            and not self._preparing
            and not self._session.finished
            and (choice := route.prepared_choice(self._worker_choice)) is not None
        ):
            if route.language != previous.language:
                self._session.set_source(route.language, choice.create)
            self._worker_choice = choice
            return True
        if changed and not self._preparing:
            self._standby = False
            if isinstance(self._session, RoutedTranslationSession) and not self._session.stopping:
                self._session = factory(self._route_input())
            else:
                self.drain()
        elif (
            not changed
            and isinstance(self._session, RoutedTranslationSession)
            and not self._session.stopping
            and any(
                a.choice is not b.choice for a, b in zip(previous.models, route.models, strict=True)
            )
        ):
            factory(self._session)  # Presentation-only changes preserve worker/result generations.
        return changed

    def drain(self) -> None:
        if self._session:
            self._session.request_stop()
            if self._session.finished:
                self._session = None

    def stop_capture(self, *, closing: bool) -> None:
        self._standby = self._preparing is not None and not closing
        if not self._preparing or closing:
            self.drain()

    def shutdown(self) -> None:
        self._closing = True
        operation = self._operation
        # Closing withdraws a pending consent and the plan a preparation would commit.
        if isinstance(operation, _Consent):
            self._operation = None
        elif isinstance(operation, _Preparation):
            self._operation = replace(operation, plan=None)
        self._standby = False
        self.drain()

    def finish_shutdown(self) -> bool:
        self.shutdown()
        if self._preparing:
            self.advance_preparation(closing=True)
        return self._session is None and self._preparing is None

    def suspend_for_probe(self, active: bool) -> bool:
        """Keep committed selection; grant a probe only after inference has drained."""
        self._probing = active
        if not active or self._preparing or self._closing:
            return False
        self._standby = False
        self.drain()
        return self._session is None

    @property
    def capture_drained(self) -> bool:
        return self._session is None or bool(self._preparing) or self._standby

    def resume_after_failed_restart(self) -> None:
        self._closing = False
        self._settled = ModelPhase.READY if self._selected else ModelPhase.IDLE

    def advance_activity(
        self, *, running: bool, enabled: bool, probing: bool = False
    ) -> TranslationActivity:
        """One decision owns probe, preparation, standby, teardown and route activation.

        Only ACTIVE exposes the publication port. Inactive and switching workers remain
        private until their native cleanup completes, even if their snapshot has failed.
        """
        probe_ready = self.suspend_for_probe(probing)
        if self._closing:
            self.drain()
            return TranslationActivity(TranslationMode.INACTIVE)
        if probing:
            return TranslationActivity(TranslationMode.PROBING, probe_ready=probe_ready)
        if self._preparing:
            return TranslationActivity(TranslationMode.PREPARING)
        session = self._session
        if not running and self._standby and enabled and session and not session.finished:
            return TranslationActivity(TranslationMode.STANDBY, snapshot=session.snapshot())
        if not self._allows_translation(running=running, enabled=enabled):
            self.drain()
            reason = (
                InactiveReason.UNLOADING
                if session and not session.finished
                else InactiveReason.OFF
                if running
                else InactiveReason.NO_MODEL
                if self._selected is None
                else InactiveReason.STOPPED
            )
            return TranslationActivity(
                TranslationMode.INACTIVE,
                snapshot=session.snapshot() if session else None,
                reason=reason,
            )
        worker = self._activate_route()
        return TranslationActivity(
            TranslationMode.ACTIVE if worker else TranslationMode.SWITCHING,
            worker=worker,
            model_name=self._selected.name if self._selected else tr("the translation model"),
        )

    def _activate_route(self) -> Worker | None:
        """Adopt/start the committed route only after advance_activity permits activation."""
        session = self._session
        if session and session.finished and session.snapshot().state == TranslationState.STOPPED:
            self._session = session = None
        # No new native owners until every previous lane has finished teardown.
        if session and session.snapshot().state == TranslationState.STOPPING:
            return None
        if isinstance(session, TranslationSession):
            assert self._worker_choice is not None
            if self._route.prepared_choice(self._worker_choice) is None:
                self.drain()
                if self._session:
                    return None
                session = None
        if session is None or isinstance(session, TranslationSession):
            assert self._route_factory
            self._session = session = self._route_factory(self._route_input())
        session.advance()
        return session
