"""Selection and route ownership replays without a widget hierarchy."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
from ownership_fixtures import Replay

from dantetl.application.model_inventory import InventoryData
from dantetl.application.preference_values import Preferences
from dantetl.application.values import ModelConsent
from dantetl.translation.backend import BackendChoice
from dantetl.translation.cloud.values import CloudProfile, CloudProvider
from dantetl.translation.plan import TranslationPlan
from dantetl.translation.selection import AssignmentChange


def test_prepared_worker_adoption_uses_loaded_identity_after_other_category_changes(tmp_path):
    from dantetl.translation.values import TranslationState

    first = BackendChoice("First", Mock(), "first", "first-weights")
    second = BackendChoice("Second", Mock(), "second")
    replay = Replay(tmp_path, data=InventoryData((first, second)))
    try:
        owner = replay.translation
        prompt = owner.assign(
            AssignmentChange(TranslationPlan("local:first", "local:first"), "local:first")
        )
        assert isinstance(prompt, ModelConsent)
        owner.lifecycle.consent(prompt.token, True)
        owner.lifecycle.advance_preparation()
        prepared = replay.workers[-1]
        prepared.state = TranslationState.READY
        outcome = owner.lifecycle.advance_preparation()
        assert outcome and outcome.committed
        # The other category can change before the prepared worker joins a router.
        plan = TranslationPlan("local:first", "local:second")
        assert owner.assign(AssignmentChange(plan, "local:second")) == "committed"
        assert owner.lifecycle.snapshot().selected == second
        owner.refresh_route()
        assert owner.advance_activity(running=True, enabled=True).worker
        _, arguments = replay.routes[-1]
        assert arguments["prepared"] == ("local:first", prepared)
        assert len(replay.workers) == 1 and prepared.stops == 0
        assert owner.plan == plan
    finally:
        replay.close()


def test_prepared_worker_cannot_join_a_route_for_another_prepared_revision(tmp_path):
    from dantetl.translation.values import TranslationState

    first = BackendChoice("Candidate", Mock(), "new", "first")
    replay = Replay(tmp_path, data=InventoryData((first,)))
    try:
        owner = replay.translation
        plan = TranslationPlan("local:new", "local:new")
        prompt = owner.assign(AssignmentChange(plan, "local:new"))
        assert isinstance(prompt, ModelConsent)
        owner.lifecycle.consent(prompt.token, True)
        owner.lifecycle.advance_preparation()
        worker = replay.workers[0]
        worker.state = TranslationState.READY
        outcome = owner.lifecycle.advance_preparation()
        assert outcome and outcome.committed
        second = replace(first, preparation_id="second", create=Mock())
        replay.discover(InventoryData((second,), prepared=frozenset({"second"})))
        owner.refresh_route()
        assert owner.advance_activity(running=True, enabled=True).worker is None
        assert worker.stops and not replay.routes
        worker.ended = True
        assert owner.advance_activity(running=True, enabled=True).worker
        args, config = replay.routes[-1]
        assert args[1]["local:new"] is second.create and config["prepared"] is None
    finally:
        replay.close()


def test_successful_preparation_updates_options_even_if_receipt_write_fails(tmp_path):
    from dantetl.translation.values import TranslationState

    choice = BackendChoice("New", Mock(), "new", "new-weights")
    replay = Replay(tmp_path, data=InventoryData((choice,)))
    try:
        # A transient persistence failure must not hide readiness in the running app.
        replay.receipts.clear()
        replay.inventory.record_prepared = Mock(side_effect=OSError("read-only"))
        from ownership_fixtures import Worker

        from dantetl.application.translation_coordinator import TranslationCoordinator

        worker = Worker(choice)
        owner = TranslationCoordinator(
            replay.inventory,
            replay.cloud,
            lambda choice, language: worker,
            Mock(),
            tmp_path / "cache",
            Preferences(),
        )
        assert not owner.options()[0].prepared
        prompt = owner.assign(
            AssignmentChange(TranslationPlan("local:new", "local:new"), "local:new")
        )
        assert isinstance(prompt, ModelConsent)
        owner.lifecycle.consent(prompt.token, True)
        owner.lifecycle.advance_preparation()
        worker.state = TranslationState.READY
        outcome = owner.lifecycle.advance_preparation()
        assert outcome and outcome.committed
        assert owner.options()[0].prepared
        assert owner.lifecycle.snapshot().preparation_revision == 1
        owner.lifecycle.shutdown()
        worker.ended = True
        assert owner.lifecycle.finish_shutdown()
    finally:
        replay.close()


@pytest.mark.parametrize(
    "values,expected",
    [
        (Preferences(), TranslationPlan("local:base", "local:base")),
        (Preferences(model="none"), TranslationPlan()),
        (Preferences(model="gemma"), TranslationPlan("local:gemma", "local:gemma")),
        (
            Preferences(short_model="local:absent", long_model="", translate_short=False),
            TranslationPlan("local:absent", "", False, True),
        ),
        (
            Preferences(translation_route="long", cloud_model="remote"),
            TranslationPlan("local:base", "cloud:remote"),
        ),
    ],
)
def test_restore_legacy_explicit_none_disabled_and_unavailable_ids(tmp_path, values, expected):
    replay = Replay(
        tmp_path, preferences=values, data=InventoryData((BackendChoice("Base", Mock(), "base"),))
    )
    try:
        assert replay.translation.plan == expected
        assert not replay.workers and not replay.routes
        replay.app.configure(opacity=60)
        assert replay.translation.plan == expected
    finally:
        replay.close()


def test_route_revision_reconfigures_owner_without_factory_calls_on_render(tmp_path):
    replay = Replay(tmp_path)
    try:
        remote = CloudProfile.new(CloudProvider.DEEPSEEK)
        replay.cloud.profiles = (remote,)
        replay.cloud._available = {remote.key}
        reference = "cloud:" + remote.key
        replay.app.advance(1)
        replay.app.assign(AssignmentChange(TranslationPlan(reference, reference), reference))
        replay.app.start_capture(replay.capture.target, 1)
        replay.app.advance(1)
        worker = replay.route_workers[-1]
        assert len(replay.routes) == 1
        for _ in range(100):
            assert replay.app.controls.assignment.active_models == (reference,)
            replay.app.notices()
        assert worker.stops == 0
        replay.cloud.revision += 1
        replay.app.refresh_routes()
        assert worker.stops == 0 and not worker.changes
        assert len(replay.routes) == 1  # The router owns selective resource teardown.
        replay.cloud._available.clear()
        replay.app.refresh_routes()
        assert replay.translation.advance_activity(running=True, enabled=True).worker is None
    finally:
        replay.close()


def test_two_local_categories_keep_distinct_lazy_recipes_and_one_committed_plan(tmp_path):
    factories = (Mock(), Mock())
    choices = tuple(
        BackendChoice(key, factory, key)
        for key, factory in zip(("short", "long"), factories, strict=True)
    )
    replay = Replay(tmp_path, data=InventoryData(choices), preferences=Preferences(model="none"))
    try:
        plan = TranslationPlan("local:short", "local:long")
        replay.app.assign(AssignmentChange(plan, "local:long"))
        assert replay.translation.advance_activity(running=True, enabled=True).worker
        assert len(replay.routes) == 1
        args, _ = replay.routes[0]
        assert args[0] == plan
        assert args[1] == {"local:short": factories[0], "local:long": factories[1]}
        assert replay.app.controls.assignment == plan
        for factory in factories:
            factory.assert_not_called()
    finally:
        replay.close()


def test_history_selection_is_independent_and_disabled_by_default(tmp_path):
    factory = Mock()
    replay = Replay(
        tmp_path,
        data=InventoryData((BackendChoice("Classifier", factory, "classifier", classifies=True),)),
    )
    try:
        original = replay.translation.plan
        assert not replay.live.history_enabled
        replay.app.configure(history_enabled=True, history_model="local:classifier")
        assert replay.live.history_enabled and replay.translation.plan == original
        replay.app.configure(history_enabled=False)
        assert not replay.live.history_enabled and replay.translation.plan == original
        factory.assert_not_called()  # no capture, so no eager classifier load
    finally:
        replay.close()
