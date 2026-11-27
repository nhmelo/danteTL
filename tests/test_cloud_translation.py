import asyncio
import json
import threading
import time
from dataclasses import replace

import httpx2
import pytest

from dantetl.domain import Rect
from dantetl.languages.target import ENGLISH, TARGET_LANGUAGES
from dantetl.ocr.observation import TextTrack
from dantetl.translation.backend import RequestRejected
from dantetl.translation.cache import TranslationCache, translation_key
from dantetl.translation.cloud.backend import CloudAllowance, CloudBackend
from dantetl.translation.cloud.profiles import CloudProfiles
from dantetl.translation.cloud.protocol import parse_answer, request_body
from dantetl.translation.cloud.responses import PlanUnavailable, ResponseStream
from dantetl.translation.cloud.values import CloudProfile, CloudProvider
from dantetl.translation.dialogue.context import DialogueTurn
from dantetl.translation.grouping import TextBlock
from dantetl.translation.routing import RoutedTranslationSession
from dantetl.translation.values import TranslationState
from dantetl.translation.worker import TranslationSession


def track(text="設定", key=1, revision=1):
    return TextTrack(key, text, Rect(20, 20, 120, 30), text_revision=revision)


def until(predicate):
    deadline = time.monotonic() + 4
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def stop(session):
    session.request_stop()
    until(lambda: session.finished)


def test_transport_overlaps_two_requests_and_cancels_both_on_stop(tmp_path):
    entered = []
    cancelled = []

    async def handle(request):
        index = len(entered)
        entered.append(request)
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.append(index)
        return httpx2.Response(200, json=response())

    allowance = CloudAllowance()
    engine = CloudBackend(
        replace(CloudProfile.new(CloudProvider.OPENAI), requests_per_minute=600),
        "fixture",
        allowance=allowance,
        transport=httpx2.MockTransport(handle),
    )
    session = TranslationSession(lambda: engine, tmp_path / "cache")
    session.start()
    try:
        session.update(tuple(track(f"日本語{i}", key=i) for i in range(3)))
        until(lambda: len(entered) == 2)
        assert allowance.snapshot().requests == 2
        assert not session.snapshot().results
    finally:
        stop(session)
    assert sorted(cancelled) == [0, 1] and len(entered) == 2
    assert not engine._thread.is_alive()


def response(text="Settings", *, provider=CloudProvider.OPENAI, **changes):
    if provider == CloudProvider.GEMINI:
        return {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "role": "model",
                        "parts": [{"text": text}],
                    },
                }
            ]
        }
    if provider == CloudProvider.CLAUDE:
        return {
            "type": "message",
            "role": "assistant",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": text}],
        }
    return {
        "choices": [{"finish_reason": "stop", "message": {"content": text, **changes}}],
        "usage": {"prompt_tokens": 80, "completion_tokens": 3},
    }


def test_target_switch_cancels_old_cloud_work_and_sends_new_language(tmp_path):
    instructions = []
    cancelled = threading.Event()

    async def handle(request):
        instruction = json.loads(request.content)["messages"][0]["content"]
        instructions.append(instruction)
        if "English" in instruction:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return httpx2.Response(200, json=response("Ajustes"))

    engine = CloudBackend(
        replace(CloudProfile.new(CloudProvider.OPENAI), requests_per_minute=600),
        "fixture",
        transport=httpx2.MockTransport(handle),
    )
    session = TranslationSession(lambda: engine, tmp_path / "cache")
    session.start()
    try:
        rows = (track(),)
        session.update(rows)
        until(lambda: len(instructions) == 1)
        session.update(rows, target=TARGET_LANGUAGES["es"])
        until(lambda: bool(session.snapshot().results))
        assert cancelled.is_set() and len(instructions) == 2
        assert "Spanish" in instructions[1]
        assert [(r.target.code, r.english) for r in session.snapshot().results] == [
            ("es", "Ajustes")
        ]
    finally:
        stop(session)


def test_ocr_revision_interrupts_only_the_obsolete_cloud_request(tmp_path):
    old, corrected, menu = "城を出ます。", "城を出ますか。", "設定"
    entered, cancelled = [], []
    release_menu = threading.Event()

    async def handle(request):
        source = json.loads(json.loads(request.content)["messages"][1]["content"])["CURRENT"]
        entered.append(source)
        try:
            if source == old:
                await asyncio.Event().wait()
            while source == menu and not release_menu.is_set():
                await asyncio.sleep(0.005)
        except asyncio.CancelledError:
            cancelled.append(source)
            raise
        return httpx2.Response(
            200, json=response("Leave the castle?" if source == corrected else "Settings")
        )

    engine = CloudBackend(
        replace(CloudProfile.new(CloudProvider.OPENAI), requests_per_minute=600),
        "fixture",
        transport=httpx2.MockTransport(handle),
    )
    session = TranslationSession(lambda: engine, tmp_path / "cache")
    session.start()
    try:
        label = track(menu, key=2)
        session.update((TextBlock((track(old),), dialogue=True), label))
        until(lambda: len(entered) == 2)
        session.update((TextBlock((track(corrected, revision=2),), dialogue=True), label))
        until(lambda: bool(session.snapshot().results))
        assert set(entered[:2]) == {old, menu} and entered[2:] == [corrected]
        assert cancelled == [old]
        assert [(r.request.source, r.english) for r in session.snapshot().results] == [
            (corrected, "Leave the castle?")
        ]
        release_menu.set()
        until(lambda: len(session.snapshot().results) == 2)
    finally:
        stop(session)


def test_profile_storage_is_bounded_and_contains_no_credentials(tmp_path):
    store = CloudProfiles(tmp_path / "profiles.json")
    profile = CloudProfile.new(CloudProvider.OPENAI)
    store.write((profile,))
    assert store.read() == (profile,)
    assert "api_key" not in store.path.read_text()
    full = tuple(CloudProfile.new(CloudProvider.OPENAI) for _ in range(16))
    store.write(full)
    assert store.read() == full
    with pytest.raises(ValueError):
        store.write(full + (profile,))
    assert store.read() == full
    with pytest.raises(ValueError):
        store.write((profile, profile))
    with pytest.raises(ValueError):
        replace(profile, provider="untrusted", model="injected")


def test_transport_error_and_traceback_never_reveal_credentials(tmp_path, caplog):
    secret = "fixture-do-not-log-this-key"

    def broken(request):
        raise ValueError("Authorization: Bearer " + secret)

    def backend():
        return CloudBackend(
            CloudProfile.new(CloudProvider.OPENAI), secret, transport=httpx2.MockTransport(broken)
        )

    session = TranslationSession(backend, tmp_path / "cache")
    session.start()
    try:
        session.update((track(),))
        until(lambda: session.snapshot().state == TranslationState.FAILED)
        assert secret not in caplog.text
        assert "Cloud request failed unexpectedly" in session.snapshot().message
        assert secret not in session.snapshot().message
    finally:
        stop(session)


@pytest.mark.parametrize("provider", [CloudProvider.OPENAI, CloudProvider.DEEPSEEK])
def test_prompt_bounds_and_provider_specific_reasoning(provider):
    profile = CloudProfile.new(provider)
    context = (DialogueTurn("私の名前はリナです。", "My name is Rina."),)
    body = request_body(profile, "Honkai: Star Rail", ("会心率", context))
    messages = body["messages"]
    assert isinstance(messages, list)
    assert json.loads(messages[1]["content"])["CURRENT"] == "会心率"
    assert json.loads(messages[1]["content"])["RECENT"] == [[context[0].source, context[0].english]]
    assert (
        body.get("thinking") == {"type": "disabled"}
        if provider == CloudProvider.DEEPSEEK
        else body["reasoning_effort"] == "none"
    )
    assert body["stream"] is False and "tools" not in body
    with pytest.raises(RequestRejected):
        request_body(profile, "", ("語" * 513, ()))


def test_cache_identity_includes_provider_model_game_and_excludes_key():
    profile = CloudProfile.new(CloudProvider.OPENAI)
    engines = [
        CloudBackend(profile, "fake-one", game="A"),
        CloudBackend(profile, "fake-two", game="A"),
        CloudBackend(profile, "fake-one", game="B"),
        CloudBackend(replace(profile, model="gpt-6-luna"), "fake-one", game="A"),
    ]
    try:
        assert engines[0].cache_identity == engines[1].cache_identity
        assert len({e.cache_identity for e in engines}) == 3
        assert all("fake" not in e.cache_identity for e in engines)
    finally:
        for engine in engines:
            engine.close()


@pytest.mark.parametrize(
    "body",
    [
        {},
        response(""),
        response("Settings", refusal="No"),
        response("Settings", reasoning_content="reasoning"),
        {"choices": [{"finish_reason": "length", "message": {"content": "Partial"}}]},
        response("x" * 4097),
        response("<think>hidden</think>Answer"),
    ],
)
def test_rejects_partial_refused_malformed_or_reasoning_output(body):
    with pytest.raises(RequestRejected):
        parse_answer(body)


@pytest.mark.parametrize(
    "provider",
    [CloudProvider.OPENAI, CloudProvider.DEEPSEEK, CloudProvider.GEMINI, CloudProvider.CLAUDE],
)
def test_cached_text_publishes_while_cloud_is_blocked_and_static_text_is_stable(tmp_path, provider):
    entered, release = threading.Event(), threading.Event()
    calls = []

    async def handle(request):
        calls.append(json.loads(request.content))
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.005)
        return httpx2.Response(200, json=response("New translation", provider=provider))

    profile = replace(CloudProfile.new(provider), requests_per_minute=600)
    engine = CloudBackend(profile, "fixture", transport=httpx2.MockTransport(handle))
    path = tmp_path / "cache.db"
    cache = TranslationCache(path)
    cache.put(translation_key(engine.cache_identity, "設定"), "Settings")
    cache.close()
    session = TranslationSession(lambda: engine, path)
    session.start()
    try:
        new = track("新しい言葉", key=2)
        cached = track("設定")
        session.update((new,))
        assert entered.wait(2)
        session.update((new, cached))
        until(lambda: len(session.snapshot().results) == 1)
        assert session.snapshot().results[0].english == "Settings"
        for _ in range(30):
            session.update((new, cached))
            assert session.snapshot().results[0].english == "Settings"
        release.set()
        until(lambda: len(session.snapshot().results) == 2)
        assert len(calls) == 1
        assert "fixture" not in json.dumps(calls)
        assert engine._allowance.snapshot().requests == 1
    finally:
        release.set()
        stop(session)
    assert not engine._thread.is_alive()


@pytest.mark.parametrize(
    "provider",
    [CloudProvider.OPENAI, CloudProvider.DEEPSEEK, CloudProvider.GEMINI, CloudProvider.CLAUDE],
)
def test_cancelled_generation_never_publishes_or_leaves_network_owner_running(tmp_path, provider):
    entered = threading.Event()
    cancelled = threading.Event()

    async def handle(request):
        entered.set()
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()
        return httpx2.Response(200, json=response("Stale", provider=provider))

    engine = CloudBackend(
        CloudProfile.new(provider), "fixture", transport=httpx2.MockTransport(handle)
    )
    session = TranslationSession(lambda: engine, tmp_path / "cache")
    session.start()
    session.update((track(),))
    assert entered.wait(2)
    session.invalidate()
    until(cancelled.is_set)
    assert not session.snapshot().results
    stop(session)
    assert not engine._thread.is_alive()


@pytest.mark.parametrize(
    "provider",
    [CloudProvider.OPENAI, CloudProvider.DEEPSEEK, CloudProvider.GEMINI, CloudProvider.CLAUDE],
)
def test_account_failure_stops_network_attempts_but_keeps_existing_english(tmp_path, provider):
    calls = []

    def handle(request):
        calls.append(request)
        return (
            httpx2.Response(200, json=response(provider=provider))
            if len(calls) == 1
            else httpx2.Response(401, json={"error": {"message": "SECRET BODY"}})
        )

    engine = CloudBackend(
        CloudProfile.new(provider), "PRIVATE-KEY", transport=httpx2.MockTransport(handle)
    )
    session = TranslationSession(lambda: engine, tmp_path / "cache")
    session.start()
    try:
        session.update((track(),))
        until(lambda: len(session.snapshot().results) == 1)
        session.update(tuple(track("設定" if n == 1 else f"日本語{n}", key=n) for n in range(1, 9)))
        until(lambda: session.snapshot().pending == 0)
        assert session.snapshot().state == TranslationState.READY
        assert len(session.snapshot().results) == 1 and len(calls) == 2
        assert (
            "PRIVATE" not in session.snapshot().message
            and "SECRET" not in session.snapshot().message
        )
    finally:
        stop(session)


@pytest.mark.parametrize(
    "provider",
    [CloudProvider.OPENAI, CloudProvider.DEEPSEEK, CloudProvider.GEMINI, CloudProvider.CLAUDE],
)
def test_hybrid_groups_dialogue_and_local_work_is_not_blocked(tmp_path, provider):
    entered, release = threading.Event(), threading.Event()
    sources = []

    async def handle(request):
        body = json.loads(request.content)
        user = (
            body["contents"][0]["parts"][0]["text"]
            if provider == CloudProvider.GEMINI
            else body["messages"][-1]["content"]
        )
        sources.append(json.loads(user))
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.005)
        return httpx2.Response(200, json=response("Complete dialogue.", provider=provider))

    class Local:
        batch_size = 4
        cache_identity = "local-fixture"

        def translate(self, texts, *, target=ENGLISH):
            return tuple("Local " + text for text in texts)

        def close(self):
            pass

    local = TranslationSession(Local, tmp_path / "cache")
    local.start()
    engine = CloudBackend(
        CloudProfile.new(provider), "fixture", transport=httpx2.MockTransport(handle)
    )
    from dantetl.translation.plan import TranslationPlan

    session = RoutedTranslationSession(
        TranslationPlan("local", "cloud"),
        {"local": Local, "cloud": lambda: engine},
        tmp_path / "cache",
        prepared=("local", local),
    )
    session.start()
    dialogue = TextBlock(
        (track("一緒に駅へ", key=2), track("行きましょう。", key=3)), dialogue=True
    )
    label = TextBlock((track(),))
    try:
        session.update((dialogue, label))
        assert entered.wait(2)
        until(lambda: len(session.snapshot().results) == 1)
        assert session.snapshot().results[0].english == "Local 設定"
        release.set()
        until(lambda: len(session.snapshot().results) == 2)
        assert sources[0]["CURRENT"] == dialogue.text
        before = session.snapshot().results
        for _ in range(20):
            session.update((label, dialogue))
            assert session.snapshot().results == before
        assert len(sources) == 1
    finally:
        release.set()
        stop(session)


def event(kind, **body):
    return ("data: " + json.dumps({"type": kind, **body}) + "\n\n").encode()


def completed(text="Critical Rate"):
    return event(
        "response.completed",
        response={
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
            "usage": {"input_tokens": 40, "output_tokens": 3},
        },
    )


def test_plan_stream_requires_completion_and_handles_usage_failure_after_partial_output():
    partial = event("response.output_text.delta", delta="Critical ")
    stream = ResponseStream()
    stream.feed(partial)
    with pytest.raises(RequestRejected):
        stream.finish()
    with pytest.raises(PlanUnavailable):
        stream.feed(
            event(
                "response.failed",
                response={"error": {"code": "subscription_sharing_usage_limit_exceeded"}},
            )
        )
    stream = ResponseStream()
    for byte in partial + completed():
        stream.feed(bytes([byte]))
    assert stream.finish()[0] == "Critical Rate"


def test_chatgpt_uses_only_oauth_responses_and_no_api_billing_fallback(tmp_path):
    profile = CloudProfile(
        "cloud-" + "a" * 32,
        "My plan",
        CloudProvider.CHATGPT,
        "gpt-6-luna",
        account="account-" + "b" * 32,
    )
    calls = []

    def handle(request):
        calls.append(request)
        assert request.url == "https://api.openai.com/v1/responses"
        assert request.headers["Authorization"] == "Bearer plan-fixture"
        body = json.loads(request.content)
        assert body["stream"] is True and body["store"] is False
        assert "max_output_tokens" not in body and "temperature" not in body
        return httpx2.Response(
            200,
            content=event(
                "response.failed",
                response={"error": {"code": "subscription_sharing_usage_limit_exceeded"}},
            ),
        )

    engine = CloudBackend(
        profile,
        "plan",
        access_token=lambda work: "plan-fixture",
        transport=httpx2.MockTransport(handle),
    )
    try:
        for _ in range(2):
            with pytest.raises(RequestRejected, match="plan usage"):
                engine.translate(("会心率",))
        assert len(calls) == 1
    finally:
        engine.close()


def test_budget_is_shared_across_backend_recreation_and_cache_does_not_spend_it():
    profile = replace(CloudProfile.new(CloudProvider.DEEPSEEK), request_limit=1)
    allowance = CloudAllowance()
    engines = [
        CloudBackend(
            profile,
            "fixture",
            allowance=allowance,
            transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json=response())),
        )
        for _ in range(2)
    ]
    try:
        assert engines[0].translate(("設定",)) == ("Settings",)
        with pytest.raises(RequestRejected, match="limit"):
            engines[1].translate(("戻る",))
        assert allowance.snapshot().requests == 1
    finally:
        for engine in engines:
            engine.close()
