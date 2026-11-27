import asyncio
import json
import threading
import time
from dataclasses import replace

import httpx2
import pytest

from dantetl.application.live_session import LiveSession
from dantetl.application.preference_values import Preferences
from dantetl.application.preferences import decode_preferences
from dantetl.domain import Rect
from dantetl.ocr.observation import TextTrack
from dantetl.translation.backend import RequestRejected
from dantetl.translation.cloud.backend import CloudAllowance, CloudBackend
from dantetl.translation.cloud.values import CloudProfile, CloudProvider
from dantetl.translation.dialogue.classifier import (
    Classification,
    ClassificationInput,
    ClassifierSession,
    parse_classification,
)
from dantetl.translation.dialogue.history import DialogueHistory
from dantetl.translation.dialogue.layout import PanelContext
from dantetl.translation.grouping import TextBlock
from dantetl.translation.values import TranslatedText, TranslationRequest


def until(predicate):
    deadline = time.monotonic() + 4
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.005)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "```json\n{}\n```",
        "[]",
        "null",
        '{"kind":"dialogue"}',
        '{"kind":"dialogue","speaker":"Alice"}',
        '{"kind":"description","speaker":"no","name":"x"}',
        '{"kind":[],"speaker":"yes"}',
        "x" * 257,
    ],
)
def test_invalid_verdict_is_request_scoped(text):
    with pytest.raises(RequestRejected):
        parse_classification(text)


def test_worker_cancels_stale_generation_and_closes_on_owner():
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    owners, calls = [], []
    a, b = ClassificationInput("古い会話です。"), ClassificationInput("新しい会話です。")

    class Engine:
        def __init__(self):
            owners.append(threading.get_ident())

        def classify(self, value, cancelled):
            owners.append(threading.get_ident())
            calls.append(value)
            if value == a:
                entered.set()
                assert release.wait(3)
                assert cancelled.is_set()
            return Classification("dialogue", "no")

        def close(self):
            owners.append(threading.get_ident())
            closed.set()

    session = ClassifierSession(Engine)
    try:
        assert not owners  # lazy model loading
        session.update((a,))
        assert entered.wait(3)
        session.invalidate()
        session.update((b,))
        release.set()
        until(lambda: bool(session.snapshot().results))
        assert session.snapshot().results == ((b, Classification("dialogue", "no")),)
        for _ in range(20):
            session.update((b,))
        assert calls == [a, b]
    finally:
        release.set()
        session.request_stop()
        until(lambda: session.finished)
    assert closed.is_set() and len(set(owners)) == 1 and owners[0] != threading.get_ident()


def test_worker_bounds_candidates_and_rejections():
    calls = []

    class Engine:
        def classify(self, value, cancelled):
            calls.append(value)
            raise RequestRejected("bad response")

        def close(self):
            pass

    session = ClassifierSession(Engine)
    try:
        desired = tuple(ClassificationInput(f"会話番号{i}です。") for i in range(200))
        session.update(desired)
        until(lambda: len(calls) == 128 and session.snapshot().pending == 0)
        session.update(desired)
        assert not session.snapshot().results
        assert len(calls) == 128
    finally:
        session.request_stop()
        until(lambda: session.finished)


def test_rejected_input_is_not_retried_after_temporary_withdrawal():
    a, b = ClassificationInput("同じ文章です。"), ClassificationInput("新しい会話です。")
    calls = []

    class Engine:
        def classify(self, value, cancelled):
            calls.append(value)
            if value == a:
                raise RequestRejected("invalid verdict")
            return Classification("dialogue", "no")

        def close(self):
            pass

    session = ClassifierSession(Engine)
    try:
        session.update((a,))
        until(lambda: session.snapshot().pending == 0)
        session.update(())
        session.update((a, b))
        until(lambda: session.snapshot().pending == 0)
        assert calls == [a, b]
    finally:
        session.request_stop()
        until(lambda: session.finished)


def test_temporary_withdrawal_finishes_and_caches_one_active_check():
    a, b = ClassificationInput("同じ文章です。"), ClassificationInput("次の会話です。")
    entered, release = threading.Event(), threading.Event()
    calls, cancellations = [], []

    class Engine:
        def classify(self, value, cancelled):
            calls.append(value)
            if value == a:
                entered.set()
                assert release.wait(3)
                cancellations.append(cancelled.is_set())
            return Classification("dialogue", "no")

        def close(self):
            pass

    session = ClassifierSession(Engine)
    try:
        session.update((a,))
        assert entered.wait(3)
        for _ in range(20):
            session.update(())
            session.update((a, b))
        # Finish while a is absent; its answer may be cached but must not publish.
        session.update((b,))
        release.set()
        until(lambda: session.snapshot().pending == 0)
        assert [k for k, _ in session.snapshot().results] == [b]
        session.update((a, b))
        until(lambda: session.snapshot().pending == 0)
        assert len(session.snapshot().results) == 2
        assert calls == [a, b] and cancellations == [False]
    finally:
        release.set()
        session.request_stop()
        until(lambda: session.finished)


def test_noisy_early_candidate_cannot_jump_ahead_of_waiting_dialogue():
    a, b, c, changed = (ClassificationInput(f"文章番号{i}です。") for i in range(4))
    entered, release = threading.Event(), threading.Event()
    calls = []

    class Engine:
        def classify(self, value, cancelled):
            calls.append(value)
            if value == a:
                entered.set()
                assert release.wait(3)
            return Classification("dialogue", "no")

        def close(self):
            pass

    session = ClassifierSession(Engine)
    try:
        session.update((a, b, c))
        assert entered.wait(3)
        session.update((changed, b, c))
        release.set()
        until(lambda: session.snapshot().pending == 0)
        assert calls == [a, b, c, changed]
    finally:
        release.set()
        session.request_stop()
        until(lambda: session.finished)


def test_current_outcomes_survive_bounded_cache_churn_and_reset_allows_retry():
    accepted = ClassificationInput("待っていてください。")
    rejected = ClassificationInput("判定に失敗する文章です。")
    calls = []

    class Engine:
        def classify(self, value, cancelled):
            calls.append(value)
            if value == rejected:
                raise RequestRejected("invalid verdict")
            return Classification("dialogue", "no")

        def close(self):
            pass

    session = ClassifierSession(Engine)
    try:
        earlier = ClassificationInput("前の文章番号0です。")
        for batch in range(10):
            work = tuple(
                ClassificationInput(f"前の文章番号{i}です。")
                for i in range(batch * 32, (batch + 1) * 32)
            )
            session.update((accepted, rejected, *work))
            until(lambda: session.snapshot().pending == 0)
        assert calls.count(accepted) == calls.count(rejected) == 1
        session.update((accepted, rejected, earlier))
        until(lambda: session.snapshot().pending == 0)
        assert calls.count(earlier) == 2  # old outcomes are evicted, memory stays bounded
        assert calls.count(accepted) == calls.count(rejected) == 1
        session.invalidate()
        session.update((accepted, rejected))
        until(lambda: session.snapshot().pending == 0)
        assert calls.count(accepted) == calls.count(rejected) == 2
        session.update(())
        assert session.snapshot().message == "Waiting for dialogue."
    finally:
        session.request_stop()
        until(lambda: session.finished)


@pytest.mark.parametrize("kind", ["dialogue", "description", "ui", "uncertain"])
def test_exact_outcomes_are_cached_but_real_text_and_speaker_edits_are_new_work(kind):
    a = ClassificationInput("ここで待っていてください。", "旅人")
    changed = replace(a, text="そこで待っていてください。")
    renamed = replace(a, speaker="少女")
    calls = []

    class Engine:
        def classify(self, value, cancelled):
            calls.append(value)
            return Classification(kind, "uncertain")

        def close(self):
            pass

    session = ClassifierSession(Engine)
    try:
        session.update((a,))
        until(lambda: session.snapshot().pending == 0)
        for _ in range(20):
            session.update(())
            session.update((a,))
        session.update((a, changed, renamed))
        until(lambda: session.snapshot().pending == 0)
        assert calls == [a, changed, renamed]
        assert {v.kind for _, v in session.snapshot().results} == {kind}
    finally:
        session.request_stop()
        until(lambda: session.finished)


def test_fatal_classifier_failure_stops_and_finished_waits_for_owner_cleanup():
    cleaning, release = threading.Event(), threading.Event()
    calls = []

    class Engine:
        def classify(self, value, cancelled):
            calls.append(value)
            raise RuntimeError("private provider details")

        def close(self):
            cleaning.set()
            assert release.wait(3)

    session = ClassifierSession(Engine)
    desired = (ClassificationInput("これは新しい会話です。"),)
    try:
        session.update(desired)
        assert cleaning.wait(2)
        assert not session.finished
        assert (
            "failed" in session.snapshot().message and "private" not in session.snapshot().message
        )
        assert not session.snapshot().results
    finally:
        release.set()
        until(lambda: session.finished)
    session.update(desired)
    assert len(calls) == 1  # failed owners do not automatically loop/reload
    assert session.snapshot().pending == 0


def block(key, text, x=100, y=300, width=800, height=40):
    return TextBlock((TextTrack(key, text, Rect(x, y, width, height)),))


def translated(b):
    return TranslatedText(
        TranslationRequest.from_block(b), "Translation " + str(b.lines[0].track_id)
    )


def settle(history, blocks, now=0):
    history.observe(
        blocks,
        blocks,
        (3840, 2160),
        now,
        assisted=True,
        scores={t: 0.98 for b in blocks for t in b.lines},
    )
    history.observe(blocks, blocks, (3840, 2160), now + 1.3, assisted=True)


def test_static_scene_admits_dialogue_after_the_first_eight_boxes():
    history = DialogueHistory()
    blocks = tuple(block(i, f"これは文章番号{i}です。", y=i * 180) for i in range(12))
    settle(history, blocks)
    requests = history.classification_requests()
    assert len(requests) == 12
    history.accept_classifications(tuple((k, Classification("dialogue", "no")) for k in requests))
    history.published(tuple(map(translated, blocks)))
    assert {t.source for t in history.turns} == {b.text for b in blocks}


def test_unrelated_neighbour_order_and_noise_do_not_reclassify_body():
    history = DialogueHistory()
    body = block(3, "ここで一緒に待っていてください。", y=300)
    a, b = block(1, "近くの表示です", y=220), block(2, "もう一つの表示です", y=220)
    settle(history, (a, b, body))
    key = next(k for k in history.classification_requests() if k.text == body.text)
    assert key.speaker == ""  # ambiguous neighbours, not a speaker
    changed = block(2, "もうーつの表示です", y=220)
    settle(history, (changed, a, body), 2)
    assert next(k for k in history.classification_requests() if k.text == body.text) == key


def test_changed_speaker_must_settle_and_cannot_use_previous_verdict():
    history = DialogueHistory()
    name = block(1, "旅人", y=200, width=100)
    body = block(2, "ここで一緒に待っていてください。", y=260)
    settle(history, (name, body))
    key = next(k for k in history.classification_requests() if k.text == body.text)
    changed = block(1, "少女", y=200, width=100)
    blocks = (changed, body)
    for now in (2.0, 2.3):
        history.observe(blocks, blocks, (3840, 2160), now, assisted=True)
        assert body.text not in {k.text for k in history.classification_requests()}
    history.accept_classifications(((key, Classification("dialogue", "yes")),))
    history.published(tuple(map(translated, blocks)))
    assert not history.turns
    history.observe(blocks, blocks, (3840, 2160), 2.6, assisted=True)
    current = next(k for k in history.classification_requests() if k.text == body.text)
    assert current == ClassificationInput(body.text, changed.text)


def test_semantics_admits_multiple_unnamed_bubbles_and_rejects_biography_button():
    history = DialogueHistory()
    a = block(
        1, "も、申し訳ございません、ホロウ災害の発生により、捜査活動に影響が出ております。", x=2465
    )
    b = block(2, "皆様、こちらは現在十四分街における捜査活動を監督している長官です。")
    label = block(3, "スクロール画面の切替", y=900, width=300)
    biography = block(
        4, "戦争に巻き込まれた調香師の少女。特殊な能力を持ち、軍のセラピストを務めていた。", y=960
    )
    blocks = (a, b, label, biography)
    settle(history, blocks)
    history.published(tuple(map(translated, blocks)))
    assert not history.turns  # no heuristic admission while checking
    history.accept_classifications(
        tuple(
            (
                k,
                Classification(
                    "description"
                    if k.text == biography.text
                    else "ui"
                    if k.text == label.text
                    else "dialogue",
                    "no",
                ),
            )
            for k in history.classification_requests()
        )
    )
    history.published(tuple(map(translated, blocks)))
    assert {t.source for t in history.turns} == {a.text, b.text}
    assert all(t.speaker is None for t in history.turns)
    history.clear()
    settle(history, blocks, 2)
    history.published(tuple(map(translated, blocks)))
    assert not history.turns


def test_speaker_verdict_and_exact_ready_english_required_after_edit_and_reset():
    history = DialogueHistory()
    name = block(1, "スクロール画面の切替", y=200, width=400)
    body = block(2, "ここから先は危険です。一緒に進んでください。", y=260)
    settle(history, (name, body))
    key = next(k for k in history.classification_requests() if k.text == body.text)
    assert key.speaker == name.text
    history.accept_classifications(((key, Classification("dialogue", "no")),))
    history.published((translated(name), translated(body)))
    assert history.turns[-1].speaker is None
    changed = block(2, "この場所で待っていてください。", y=260)
    settle(history, (name, changed), 3)
    history.accept_classifications(((key, Classification("dialogue", "yes")),))
    history.published((translated(changed),))
    assert len(history.turns) == 1  # stale verdict
    current = next(k for k in history.classification_requests() if k.text == changed.text)
    history.accept_classifications(((current, Classification("dialogue", "yes")),))
    history.published((translated(body),))
    assert len(history.turns) == 1  # stale English
    history.published((translated(changed), translated(name)))
    assert history.turns[-1].speaker == name.text  # semantic verdict + exact high-confidence label
    history.reset_scene()
    history.accept_classifications(((current, Classification("dialogue", "yes")),))
    history.published((translated(changed),))
    assert len(history.turns) == 2


def test_history_defaults_off_and_preferences_validate_model_ids():
    assert not Preferences().history_enabled
    assert not LiveSession().history_enabled
    value = decode_preferences(
        {"version": 1, "history_enabled": True, "history_model": "local:gemma"}
    )
    assert value.history_enabled and value.history_model == "local:gemma"
    assert (
        decode_preferences({"version": 1, "history_model": "https://invalid"}).history_model == ""
    )


@pytest.mark.parametrize(
    "provider",
    [CloudProvider.OPENAI, CloudProvider.DEEPSEEK, CloudProvider.GEMINI, CloudProvider.CLAUDE],
)
@pytest.mark.parametrize("detached", (False, True))
def test_cloud_classifies_in_one_call_with_shared_allowance_and_no_transcript(provider, detached):
    received = []
    verdict = '{"kind":"description","speaker":"no"}'

    def handler(request):
        received.append(json.loads(request.content))
        if provider == CloudProvider.GEMINI:
            value = {
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"role": "model", "parts": [{"text": verdict}]},
                    }
                ]
            }
        elif provider == CloudProvider.CLAUDE:
            value = {
                "type": "message",
                "role": "assistant",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": verdict}],
            }
        else:
            value = {"choices": [{"finish_reason": "stop", "message": {"content": verdict}}]}
        return httpx2.Response(200, json=value)

    allowance = CloudAllowance()
    engine = CloudBackend(
        CloudProfile.new(provider),
        "fixture",
        allowance=allowance,
        transport=httpx2.MockTransport(handler),
    )
    try:
        answer = engine.classify(
            ClassificationInput(
                "少女は調香師として働いていた。",
                "画面の切替",
                PanelContext(detached_response=detached),
            ),
            threading.Event(),
        )
        assert answer == Classification("description", "no")
        assert len(received) == allowance.snapshot().requests == 1
        serialized = json.dumps(received[0], ensure_ascii=False)
        assert ("EVEN WITH NO SIBLINGS" in serialized) == detached
        assert (
            "SPEAKER" in serialized and "RECENT" not in serialized and "CURRENT" not in serialized
        )
    finally:
        engine.close()


def test_cloud_classification_cancel_drains_io():
    entered, stopped = threading.Event(), threading.Event()

    async def handler(request):
        entered.set()
        try:
            await asyncio.sleep(30)
        finally:
            stopped.set()
        return httpx2.Response(500)

    session = ClassifierSession(
        lambda: CloudBackend(
            replace(CloudProfile.new(CloudProvider.OPENAI), requests_per_minute=600),
            "fixture",
            transport=httpx2.MockTransport(handler),
        )
    )
    session.update((ClassificationInput("ここから一緒に行きましょう。"),))
    assert entered.wait(3)
    session.request_stop()
    until(lambda: session.finished)
    assert stopped.is_set() and not session.snapshot().results


def test_semantic_whole_turn_survives_temporary_grouping_split_without_fragment_calls():
    history = DialogueHistory()
    a = TextTrack(1, "ここから先は危険だから、", Rect(100, 200, 600, 40))
    b = TextTrack(2, "私と一緒に進んでください。", Rect(100, 255, 600, 40))
    whole = TextBlock((a, b), dialogue=True)
    settle(history, (whole,))
    key = history.classification_requests()[0]
    history.accept_classifications(((key, Classification("dialogue", "no")),))
    history.published((translated(whole),))
    assert len(history.turns) == 1
    # Model replacement clears semantic evidence, but not this utterance's entry token.
    history.reset_scene()
    settle(history, (whole,), 2)
    history.published((translated(whole),))
    assert len(history.turns) == 1
    history.accept_classifications(((key, Classification("dialogue", "no")),))
    history.published((translated(whole),))
    assert len(history.turns) == 1
    fragments = (TextBlock((a,)), TextBlock((b,)))
    settle(history, fragments, 4)
    assert not history.classification_requests()
    history.published(tuple(map(translated, fragments)))
    assert len(history.turns) == 1 and history.turns[0].source == whole.text
    settle(history, (whole,), 6)
    history.accept_classifications(((key, Classification("dialogue", "no")),))
    history.published((translated(whole),))
    assert len(history.turns) == 1
