import hashlib
import json
from dataclasses import replace

import httpx2
import pytest

from dantetl.languages.selection import DEFAULT_SOURCE
from dantetl.translation.backend import RequestRejected
from dantetl.translation.cloud.backend import CloudBackend
from dantetl.translation.cloud.protocol import (
    CLAUDE_VERSION,
    parse_answer,
    request_body,
    response_usage,
)
from dantetl.translation.cloud.values import ENDPOINTS, MODELS, CloudProfile
from dantetl.translation.dialogue.context import DialogueTurn
from dantetl.translation.prompt import translation_instruction


def answer(provider, text="Critical Rate", *, reason=None, parts=None):
    if provider == "gemini":
        return {
            "candidates": [
                {
                    "finishReason": reason or "STOP",
                    "content": {"role": "model", "parts": parts or [{"text": text}]},
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 50,
                "candidatesTokenCount": 3,
                "thoughtsTokenCount": 7,
            },
        }
    return {
        "type": "message",
        "role": "assistant",
        "stop_reason": reason or "end_turn",
        "content": parts or [{"type": "text", "text": text}],
        "usage": {
            "input_tokens": 10,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 10,
            "output_tokens": 10,
        },
    }


@pytest.mark.parametrize(
    "provider,model",
    [(provider, model) for provider in ("gemini", "claude") for model, _ in MODELS[provider]],
)
def test_offered_models_send_native_text_only_requests_and_count_all_tokens(provider, model):
    profile = replace(CloudProfile.new(provider), model=model)
    context = (DialogueTurn("私の名前はリナです。", "My name is Rina."),)
    calls = []

    def handle(request):
        calls.append(request)
        body = json.loads(request.content)
        assert "fixture-secret" not in str(request.url) + json.dumps(body)
        assert not request.url.query and "tools" not in body and "temperature" not in body
        if provider == "gemini":
            assert str(request.url) == ENDPOINTS[provider] + model + ":generateContent"
            assert request.headers["x-goog-api-key"] == "fixture-secret"
            assert "Authorization" not in request.headers
            assert body["systemInstruction"]["parts"][0]["text"] == translation_instruction(
                DEFAULT_SOURCE
            )
            user = json.loads(body["contents"][0]["parts"][0]["text"])
            config = body["generationConfig"]
            assert config["maxOutputTokens"] == 2048 and config["candidateCount"] == 1
            assert config["thinkingConfig"] == {
                "thinkingLevel": "MINIMAL" if "flash-lite" in model else "LOW",
                "includeThoughts": False,
            }
        else:
            assert str(request.url) == ENDPOINTS[provider]
            assert request.headers["Authorization"] == "Bearer fixture-secret"
            assert request.headers["anthropic-version"] == CLAUDE_VERSION
            assert (
                body["system"] == translation_instruction(DEFAULT_SOURCE) and body["model"] == model
            )
            assert body["stream"] is False and len(body["messages"]) == 1
            user = json.loads(body["messages"][0]["content"])
            assert body["max_tokens"] == 768 and body["output_config"] == {"effort": "low"}
            assert body["thinking"] == {
                "type": "between_tools" if "sonnet" in model else "disabled"
            }
        assert user == {
            "GAME": "Honkai: Star Rail",
            "CURRENT": "会心率",
            "RECENT": [[context[0].source, context[0].english]],
        }
        return httpx2.Response(200, json=answer(provider))

    engine = CloudBackend(
        profile,
        "fixture-secret",
        game="Honkai: Star Rail",
        transport=httpx2.MockTransport(handle),
    )
    try:
        assert engine.translate_with_context("会心率", context) == "Critical Rate"
        assert len(calls) == 1
        usage = engine._allowance.snapshot()
        assert (usage.requests, usage.input_tokens, usage.output_tokens) == (1, 50, 10)
        for text, turns in (("語" * 513, ()), ("語", context * 3)):
            with pytest.raises(RequestRejected):
                request_body(profile, "", (text, turns))
    finally:
        engine.close()
    assert not engine._thread.is_alive()


@pytest.mark.parametrize("provider", ["gemini", "claude"])
def test_native_responses_require_complete_text_without_tools_thoughts_or_refusals(provider):
    bad = [
        None,
        {},
        answer(provider, ""),
        answer(provider, "x" * 4097),
        answer(provider, "<think>Internal</think>English"),
    ]
    if provider == "gemini":
        for reason in ("MAX_TOKENS", "SAFETY", "RECITATION", "OTHER"):
            bad.append(answer(provider, reason=reason))
        for part in (
            {"text": "Reasoning", "thought": True},
            {"functionCall": {"name": "tool"}},
            {"text": "English", "inlineData": {}},
        ):
            bad.append(answer(provider, parts=[part]))
        blocked = answer(provider)
        blocked["promptFeedback"] = {"blockReason": "SAFETY"}
        bad.append(blocked)
        multiple = answer(provider)
        multiple["candidates"] *= 2
        bad.append(multiple)
        valid = answer(
            provider, parts=[{"text": "Critical ", "thoughtSignature": "opaque"}, {"text": "Rate"}]
        )
    else:
        for reason in ("max_tokens", "refusal", "tool_use", "pause_turn", "stop_sequence"):
            bad.append(answer(provider, reason=reason))
        for kind in ("thinking", "redacted_thinking", "tool_use", "refusal"):
            bad.append(answer(provider, parts=[{"type": kind, "text": "Never paint this"}]))
        valid = answer(
            provider,
            parts=[
                {"type": "text", "text": "Critical "},
                {"type": "text", "text": "Rate", "citations": []},
            ],
        )
    for value in bad:
        with pytest.raises(RequestRejected):
            parse_answer(value, provider)
    assert parse_answer(valid, provider) == "Critical Rate"
    assert (
        response_usage(
            provider, {"usageMetadata": {"promptTokenCount": -1}, "usage": {"input_tokens": True}}
        )
        is None
    )


@pytest.mark.parametrize("provider", ["gemini", "claude"])
def test_rejected_answer_is_scoped_and_does_not_poison_next_request(provider):
    replies = iter([answer(provider, reason="max_tokens"), answer(provider)])
    engine = CloudBackend(
        replace(CloudProfile.new(provider), requests_per_minute=600),
        "fixture",
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json=next(replies))),
    )
    try:
        with pytest.raises(RequestRejected, match="incomplete"):
            engine.translate(("失敗",))
        assert engine.translate(("会心率",)) == ("Critical Rate",)
    finally:
        engine.close()


@pytest.mark.parametrize(
    "provider,status,body",
    [
        ("gemini", 400, {"error": {"details": [{"reason": "API_KEY_INVALID"}]}}),
        ("claude", 401, {"error": {"type": "authentication_error"}}),
    ],
)
def test_native_account_error_blocks_further_attempts_without_exposing_provider_body(
    provider, status, body
):
    body["error"]["message"] = "secret fixture-key"
    calls = []

    def handle(request):
        calls.append(request)
        return httpx2.Response(status, json=body)

    engine = CloudBackend(
        CloudProfile.new(provider), "fixture-key", transport=httpx2.MockTransport(handle)
    )
    try:
        for _ in range(2):
            with pytest.raises(RequestRejected, match="account needs attention") as error:
                engine.translate(("設定",))
            assert "secret" not in str(error.value) and "fixture-key" not in str(error.value)
        assert len(calls) == 1
    finally:
        engine.close()


@pytest.mark.parametrize("provider,status", [("gemini", 429), ("claude", 529)])
def test_native_overload_retries_once_and_never_changes_provider(provider, status):
    calls = []

    def handle(request):
        calls.append(str(request.url))
        return httpx2.Response(status, json={"error": {"message": "busy"}})

    engine = CloudBackend(
        CloudProfile.new(provider), "fixture", transport=httpx2.MockTransport(handle)
    )
    try:
        with pytest.raises(RequestRejected, match=f"HTTP {status}"):
            engine.translate(("設定",))
        assert len(calls) == 2 and calls[0] == calls[1]
    finally:
        engine.close()


def test_existing_cache_identities_survive_protocol_extraction_and_new_models_are_distinct():
    engines = []
    try:
        for provider in ("openai", "deepseek", "gemini", "claude"):
            for model, _ in MODELS[provider]:
                profile = replace(CloudProfile.new(provider), model=model)
                engine = CloudBackend(profile, "fixture", game="Game")
                engines.append(engine)
                if provider in ("openai", "deepseek"):
                    old = [
                        ENDPOINTS[provider],
                        model,
                        translation_instruction(DEFAULT_SOURCE),
                        "Game",
                        "ja",
                        "en",
                        "provider-minimum-reasoning",
                        768,
                    ]
                    expected = hashlib.sha256(
                        json.dumps(old, ensure_ascii=False).encode()
                    ).hexdigest()
                    assert engine.cache_identity == "cloud:" + expected
        assert len({engine.cache_identity for engine in engines}) == len(engines)
    finally:
        for engine in engines:
            engine.close()
