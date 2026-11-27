"""Category assignments preserve bounded work, ownership and exact-source publication."""

import threading
from collections.abc import Callable
from dataclasses import replace
from unittest.mock import Mock

import pytest
from translation_fixtures import Engine, block, until

from dantetl.application.live_session import LiveSession
from dantetl.application.model_inventory import ModelInventory
from dantetl.application.model_lifecycle import ModelLifecycle, RouteModel, RouteSelection
from dantetl.application.preference_values import Preferences
from dantetl.application.preferences import decode_preferences
from dantetl.translation.backend import BackendChoice, TranslationBackend
from dantetl.translation.grouping import TextBlock
from dantetl.translation.plan import TranslationPlan
from dantetl.translation.routing import RoutedTranslationSession
from dantetl.translation.values import TranslationState
from dantetl.translation.worker import TranslationSession


@pytest.mark.parametrize(
    "keys", [("local:a", "local:b"), ("cloud:a", "cloud:b"), ("cloud:a", "local:b")]
)
def test_two_assignments_progress_independently_and_remain_stable(tmp_path, keys):
    release = threading.Event()
    fast, slow = Engine("fast"), Engine("slow", release)
    session = RoutedTranslationSession(
        TranslationPlan(*keys), {keys[0]: lambda: fast, keys[1]: lambda: slow}, tmp_path / "cache"
    )
    session.start()
    short, long = block(1, "設定"), block(2, "一緒に行きましょう。", dialogue=True)
    try:
        session.update((long, short))
        assert slow.entered.wait(2)
        until(lambda: len(session.snapshot().results) == 1)
        assert session.snapshot().results[0].english == "fast: 設定"
        release.set()
        until(lambda: len(session.snapshot().results) == 2)
        original = session.snapshot().results
        for _ in range(40):
            session.update((short, long))
            assert session.snapshot().results == original
        assert len(fast.calls) == len(slow.calls) == 1
        # Geometry and category noise cannot reroute the same accepted content/revision.
        session.update((short, replace(long, dialogue=False)))
        assert session.snapshot().results == original
    finally:
        release.set()
        session.request_stop()
        until(lambda: session.finished)
        assert fast.closed.is_set() and slow.closed.is_set()


def test_same_model_uses_one_prepared_owner_and_disabled_text_is_never_requested(tmp_path):
    engine = Engine("shared")
    factory = Mock(return_value=engine)
    prepared = TranslationSession(factory, tmp_path / "cache")
    prepared.start()
    until(lambda: prepared.snapshot().state == TranslationState.READY)
    plan = TranslationPlan("local:a", "local:a", short_enabled=False)
    session = RoutedTranslationSession(
        plan, {"local:a": factory}, tmp_path / "cache", prepared=("local:a", prepared)
    )
    session.start()
    short, long = block(1, "設定"), block(2, "説明" * 24)
    try:
        session.update((short, long), retain=(short, long))
        until(lambda: bool(session.snapshot().results))
        assert factory.call_count == 1 and engine.calls == [(long.text,)]
        session.update((short,), retain=(short, long))
        assert session.snapshot().results[0].request.source == long.text
        assert not plan.two_local_models
    finally:
        session.request_stop()
        until(lambda: session.finished)


def test_growing_published_label_retires_only_when_its_successor_has_no_route(tmp_path):
    release = threading.Event()
    short, long = Engine("short"), Engine("long", release)
    factories: dict[str, Callable[[], TranslationBackend]] = {
        "local:a": lambda: short,
        "local:b": lambda: long,
    }
    plan = TranslationPlan("local:a", "local:b")
    live = LiveSession()
    session = RoutedTranslationSession(plan, factories, tmp_path / "cache")
    label = block(1, "設定")
    grown = TextBlock((replace(label.lines[0], text=label.text * 21, text_revision=2),))
    try:
        session.start()
        for long_enabled in (True, False):
            session.reconfigure(replace(plan, long_enabled=long_enabled), factories)
            until(lambda: bool(live.publish(session, (label,), (label,)).translations))
            scene = live.publish(session, (grown,), (grown,))
            if long_enabled:
                assert long.entered.wait(2)
                assert scene.blocks == (label,)
            else:
                assert not scene.blocks and not scene.translations
                assert scene.snapshot.pending == 0
    finally:
        release.set()
        session.request_stop()
        until(lambda: session.finished)


def test_reassign_drains_changed_owner_and_rejects_previous_inflight_reply(tmp_path):
    release = threading.Event()
    old, other, new = Engine("old", release), Engine("other"), Engine("new")
    choices = {
        k: BackendChoice(k, lambda e=e: e, k) for k, e in (("a", old), ("b", other), ("c", new))
    }
    lifecycle = ModelLifecycle(
        lambda c: TranslationSession(c.create, tmp_path / "cache"), Mock(spec=ModelInventory)
    )
    factories = {"local:" + k: c.create for k, c in choices.items()}

    def configure(plan):
        def reconfigure(existing):
            if isinstance(existing, RoutedTranslationSession):
                existing.reconfigure(plan, factories)
                return existing
            return start(plan)

        models = tuple(
            RouteModel(ref, choice.identity, True, choice)
            for ref in plan.active_models
            if (choice := choices.get(ref.removeprefix("local:")))
        )
        lifecycle.configure_route(RouteSelection(plan=plan, models=models), reconfigure)

    def start(plan):
        result = RoutedTranslationSession(plan, factories, tmp_path / "cache")
        result.start()
        return result

    configure(TranslationPlan("local:a", "local:b"))
    first = lifecycle.advance_activity(running=True, enabled=True).worker
    assert isinstance(first, RoutedTranslationSession)
    text = block(1, "設定")
    try:
        first.update((text, block(2, "説明" * 24)))
        assert old.entered.wait(2)
        assert other.entered.wait(2)
        configure(TranslationPlan("local:c", "local:b"))
        assert lifecycle.advance_activity(running=True, enabled=True).worker is first
        assert not new.entered.is_set() and first.snapshot().results == ()
        release.set()
        until(old.closed.is_set)
        current = lifecycle.advance_activity(running=True, enabled=True).worker
        assert current is first and not other.closed.is_set()
        assert current is not None
        current.update((text,))
        until(lambda: bool(current.snapshot().results))
        assert current.snapshot().results[0].english == "new: 設定"
    finally:
        release.set()
        lifecycle.shutdown()
        until(lifecycle.finish_shutdown)


def test_failed_lane_does_not_take_down_other_and_missing_model_never_falls_back(tmp_path):
    engine = Engine("working")

    def fail():
        raise RuntimeError("fixture load failure")

    session = RoutedTranslationSession(
        TranslationPlan("local:a", "cloud:b"),
        {"local:a": lambda: engine, "cloud:b": fail},
        tmp_path / "cache",
    )
    session.start()
    try:
        session.update((block(1, "設定"), block(2, "説明" * 24)))
        until(lambda: bool(session.snapshot().results) and "fixture" in session.snapshot().message)
        assert session.snapshot().state == TranslationState.READY
        assert engine.calls == [("設定",)]
    finally:
        session.request_stop()
        until(lambda: session.finished)


def test_preferences_validate_new_assignments_and_keep_manual_game_on_upgrade():
    cloud = "cloud:cloud-" + "a" * 32
    decoded = decode_preferences(
        {
            "version": 1,
            "short_model": "local:gemma-e4b",
            "long_model": cloud,
            "translate_short": False,
            "game_hint": "Honkai Star Rail",
        }
    )
    assert decoded.short_model == "local:gemma-e4b" and decoded.long_model == cloud
    assert not decoded.translate_short and not decoded.game_from_window
    assert decode_preferences({"version": 1, "short_model": "../bad"}).short_model == "legacy"
    assert Preferences().game_from_window
    assert TranslationPlan.legacy("long", "gemma", cloud[6:]) == TranslationPlan(
        "local:gemma", cloud
    )


def test_two_model_owners_initialize_and_share_persistent_cache(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from dantetl.translation.cache import TranslationCache

    start = threading.Barrier(2)
    path = tmp_path / "new-cache.sqlite3"

    def write(key):
        start.wait(2)
        cache = TranslationCache(path)
        try:
            cache.put(key, key + " English")
            assert not cache.warning
        finally:
            cache.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(write, key) for key in ("local-a", "local-b")]
        for future in futures:
            future.result(3)
    cache = TranslationCache(path)
    try:
        assert cache.get("local-a") == "local-a English"
        assert cache.get("local-b") == "local-b English"
        assert not cache.warning
    finally:
        cache.close()
