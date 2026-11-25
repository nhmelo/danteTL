"""Authoritative assignment, route and classifier resolution, independent of presentation."""

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Literal

from dantetl.application.cloud_services import CloudServices
from dantetl.application.live_session import LiveSession
from dantetl.application.model_inventory import ModelInventory
from dantetl.application.model_lifecycle import (
    ModelLifecycle,
    PreparedWorker,
    RouteInput,
    RouteModel,
    RouteSelection,
)
from dantetl.application.preference_values import Preferences
from dantetl.application.translation_activity import (
    InactiveReason,
    TranslationActivity,
    TranslationMode,
)
from dantetl.application.values import ModelConsent
from dantetl.languages.selection import source_language
from dantetl.languages.source import SourceLanguage
from dantetl.translation.backend import BackendChoice, TranslationBackend
from dantetl.translation.cloud.values import CloudProfile
from dantetl.translation.ollama.backend import OllamaBackend
from dantetl.translation.ollama.values import OllamaConfig, OllamaIdentity, OllamaModel
from dantetl.translation.plan import TranslationPlan
from dantetl.translation.routing import ModelIdentity, RoutedTranslationSession
from dantetl.translation.selection import (
    AssignmentChange,
    ClassifierIdentity,
    CloudIdentity,
    ModelOption,
)
from dantetl.translation.values import TranslationState
from dantetl.translation.worker import TranslationSession


def model_reference(choice: BackendChoice | CloudProfile | None) -> str:
    if isinstance(choice, BackendChoice):
        return "local:" + choice.preference_key
    return "cloud:" + choice.key if choice else ""


class TranslationCoordinator:
    """Resolve model policy; an owned lifecycle exclusively commits and operates workers."""

    def __init__(
        self,
        inventory: ModelInventory,
        cloud: CloudServices,
        create: Callable[[BackendChoice, SourceLanguage], TranslationSession],
        route_create: Callable[..., RoutedTranslationSession],
        cache: Path,
        preferences: Preferences,
        *,
        on_change: Callable[[], None] = lambda: None,
    ) -> None:
        self.language = source_language(preferences.source_language)
        self._lifecycle = ModelLifecycle(
            lambda choice: create(choice, self.language),
            inventory,
        )
        self.inventory, self.cloud = inventory, cloud
        self._route_create, self._cache, self._notify = route_create, cache, on_change
        self._ollama_config = preferences.ollama
        self._ollama_unavailable: frozenset[OllamaModel] = frozenset()
        self._ollama_attempts: dict[str, int] = {}
        self._preferences = preferences
        self._restored = False
        self._options_key = None
        self._options: tuple[ModelOption, ...] = ()
        self.game_from_window = preferences.game_from_window
        self.manual_game = preferences.game_hint
        self.window_title = ""
        self.restore()

    @property
    def lifecycle(self) -> ModelLifecycle:
        return self._lifecycle

    @property
    def choices(self) -> tuple[BackendChoice, ...]:
        return self.inventory.snapshot().data.backends

    @property
    def plan(self) -> TranslationPlan:
        return self._lifecycle.assignment or TranslationPlan()

    def restore(self) -> None:
        if self._restored or not self.inventory.snapshot().loaded:
            return
        p = self._preferences
        local_key = p.model
        initial = next((c for c in self.choices if c.preference_key == p.model), None)
        if p.model == "auto":
            initial = next((c for c in self.choices if self._lifecycle.is_prepared(c)), None)
            local_key = initial.preference_key if initial else ""
        legacy = TranslationPlan.legacy(
            p.translation_route,
            local_key,
            p.cloud_model,
        )
        plan = TranslationPlan(
            legacy.short if p.short_model == "legacy" else p.short_model,
            legacy.long if p.long_model == "legacy" else p.long_model,
            p.translate_short,
            p.translate_long,
        )
        self._lifecycle.restore_selection(initial, assignment=plan)
        self._restored = True

    @property
    def restored(self) -> bool:
        return self._restored

    def resolve(self, reference: str) -> BackendChoice | CloudProfile | OllamaModel | None:
        if reference.startswith("ollama:"):
            return next(
                (
                    m
                    for m in self._ollama_config.models
                    if m.reference(self._ollama_config.server) == reference
                ),
                None,
            )
        return next(
            (c for c in (*self.choices, *self.cloud.profiles) if model_reference(c) == reference),
            None,
        )

    def update_ollama(self, config: OllamaConfig, unavailable: frozenset[OllamaModel]) -> None:
        self._ollama_config, self._ollama_unavailable = config, unavailable
        # Explicit rediscovery can revive a failed lane without restarting healthy siblings.
        refs = {m.reference(config.server) for m in config.models if m not in unavailable}
        self._ollama_attempts = {k: v for k, v in self._ollama_attempts.items() if k in refs}
        snapshot = self._lifecycle.snapshot().worker
        for lane in snapshot.lanes if snapshot else ():
            if lane.reference in refs and lane.state == TranslationState.FAILED:
                self._ollama_attempts[lane.reference] = (
                    self._ollama_attempts.get(lane.reference, 0) + 1
                )

    def options(self) -> tuple[ModelOption, ...]:
        cloud = self.cloud.snapshot()
        key = (
            self.inventory.snapshot().revision,
            self._ollama_config,
            self._ollama_unavailable,
            cloud.profiles,
            cloud.available,
            cloud.revision,
            self._lifecycle.preparation_revision,
        )
        if key == self._options_key:
            return self._options
        self._options_key = key
        self._options = (
            tuple(
                ModelOption(
                    model_reference(c),
                    c.name,
                    c.key,
                    prepared=self._lifecycle.is_prepared(c),
                    unavailable_reason=c.unavailable_reason,
                    performance_notice=c.performance_notice,
                    classifies=c.classifies,
                    output_codes=c.output_codes,
                )
                for c in self.choices
            )
            + tuple(
                ModelOption(
                    model_reference(p),
                    p.name,
                    p.key,
                    provider=p.provider,
                    prepared=self.cloud.available(p.key),
                    classifies=True,
                )
                for p in self.cloud.profiles
            )
            + tuple(
                ModelOption(
                    m.reference(self._ollama_config.server),
                    m.display_name,
                    m.name,
                    provider="ollama",
                    prepared=m not in self._ollama_unavailable,
                )
                for m in self._ollama_config.models
            )
        )
        return self._options

    @property
    def active_options(self) -> tuple[ModelOption, ...]:
        options = {c.reference: c for c in self.options()}
        return tuple(options[key] for key in self.plan.active_models if key in options)

    @property
    def game(self) -> str:
        return self.window_title if self.game_from_window else self.manual_game

    def set_game(self, *, automatic: bool, manual: str, title: str) -> None:
        self.game_from_window = automatic
        self.manual_game = " ".join(manual.split())[:120]
        self.window_title = " ".join(title.split())[:120]

    def assign(
        self, change: AssignmentChange
    ) -> ModelConsent | Literal["unchanged", "committed", "models", "runtime"]:
        option = next((o for o in self.options() if o.reference == change.reference), None)
        if change.plan == self.plan and (option is None or option.prepared):
            return "unchanged"
        if option and option.unavailable_reason:
            return "models" if option.key.startswith("imported-") else "runtime"
        choice = self.resolve(change.reference)
        token = self._lifecycle.request_assignment(
            change.plan, choice if isinstance(choice, BackendChoice) else None
        )
        if token is None:
            return "committed"
        assert option is not None
        return ModelConsent(token, option)

    def _route_model(self, reference: str) -> RouteModel | None:
        choice = self.resolve(reference)
        if isinstance(choice, BackendChoice):
            return RouteModel(
                reference, choice.identity, self._lifecycle.is_prepared(choice), choice
            )
        if isinstance(choice, OllamaModel):
            return RouteModel(
                reference,
                OllamaIdentity(
                    self._ollama_config.server,
                    choice,
                    self.game,
                    self._ollama_attempts.get(reference, 0),
                ),
                choice not in self._ollama_unavailable,
                choice,
            )
        if choice is None:
            return None
        identity = CloudIdentity.from_profile(
            choice, self.game, self.cloud.credential_revision(choice)
        )
        return RouteModel(reference, identity, self.cloud.available(choice.key), choice)

    def refresh_route(self) -> bool:
        self.restore()
        self._lifecycle.refresh_choices(self.choices)
        models = tuple(m for ref in self.plan.active_models if (m := self._route_model(ref)))
        route = RouteSelection(
            language=self.language,
            game=self.game
            if any(isinstance(m.choice, (CloudProfile, OllamaModel)) for m in models)
            else "",
            plan=self.plan,
            models=models,
        )
        return self._lifecycle.configure_route(route, partial(self._create_route, route))

    def _create_route(
        self,
        route: RouteSelection,
        local: RouteInput,
    ) -> RoutedTranslationSession:
        factories = {}
        identities: dict[str, ModelIdentity] = {}
        names = {}
        for model in route.models:
            if not model.available:
                continue
            choice, reference = model.choice, model.reference
            if isinstance(choice, BackendChoice):
                factories[reference] = choice.create
            elif isinstance(choice, CloudProfile):
                factories[reference] = partial(
                    self.cloud.create_backend, choice, route.game, route.language
                )
            else:
                assert isinstance(model.identity, OllamaIdentity)
                factories[reference] = partial(OllamaBackend, model.identity, route.language)
            identities[reference] = model.identity
            names[reference] = choice.name
        if isinstance(local, RoutedTranslationSession):
            local.reconfigure(
                route.plan, factories, language=route.language, identities=identities, names=names
            )
            return local
        result = self._route_create(
            route.plan,
            factories,
            self._cache,
            language=route.language,
            prepared=(model_reference(local.choice), local.worker)
            if isinstance(local, PreparedWorker)
            else None,
            on_change=self._notify,
            identities=identities,
            names=names,
        )
        result.start()
        return result

    def configure_history(self, live: LiveSession, *, enabled: bool, reference: str) -> None:
        choice = self.resolve(reference)
        factory: Callable[[], TranslationBackend] | None = None
        identity = ClassifierIdentity(reference, None)
        if (
            isinstance(choice, BackendChoice)
            and choice.classifies
            and self._lifecycle.is_prepared(choice)
        ):
            factory = choice.create
            identity = ClassifierIdentity(reference, choice.identity, available=True)
        elif isinstance(choice, CloudProfile) and self.cloud.available(choice.key):
            factory = partial(self.cloud.create_backend, choice, "", self.language)
            identity = ClassifierIdentity(
                reference,
                CloudIdentity.from_profile(choice, "", self.cloud.credential_revision(choice)),
                available=True,
            )
        live.configure_history(enabled, identity, factory, layout_only=reference == "layout")

    def advance_activity(
        self, *, running: bool, enabled: bool, probing: bool = False
    ) -> TranslationActivity:
        activity = self._lifecycle.advance_activity(
            running=running, enabled=enabled, probing=probing
        )
        if (
            activity.mode == TranslationMode.INACTIVE
            and activity.reason != InactiveReason.UNLOADING
            and self.choices
            and all(c.unavailable_reason for c in self.choices)
        ):
            return replace(activity, reason=InactiveReason.NEEDS_RUNTIME)
        return activity
