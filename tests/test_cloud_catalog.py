import asyncio
import json
import threading
from dataclasses import replace

import httpx2
import pytest

from dantetl.translation.cloud.catalog import CatalogCache, fetch_models
from dantetl.translation.cloud.profiles import CloudProfiles
from dantetl.translation.cloud.protocol import decoding_identity, request_body
from dantetl.translation.cloud.values import CatalogModel, CloudProfile, CloudProvider


@pytest.mark.parametrize(
    "provider",
    [CloudProvider.OPENAI, CloudProvider.DEEPSEEK, CloudProvider.GEMINI, CloudProvider.CLAUDE],
)
def test_discovery_is_authenticated_metadata_only_and_accepts_new_models(provider):
    requests = []

    def handle(request):
        requests.append(request)
        row = {"id": "future-text-model"}
        if provider == CloudProvider.GEMINI:
            row = {
                "name": "models/future-text-model",
                "version": "002",
                "supportedGenerationMethods": ["generateContent"],
                "thinking": False,
            }
        return httpx2.Response(
            200, json={"models" if provider == CloudProvider.GEMINI else "data": [row]}
        )

    models = fetch_models(
        provider, "private-key", threading.Event(), transport=httpx2.MockTransport(handle)
    )
    assert len(models) == 1 and models[0].model == "future-text-model"
    profile = models[0].profile(provider)
    assert profile.decoding != "preset"
    assert len(requests) == 1 and requests[0].method == "GET" and not requests[0].content
    assert "private-key" in requests[0].headers.get(
        "authorization", requests[0].headers.get("x-goog-api-key", "")
    )
    assert "private-key" not in str(requests[0].url)
    body = request_body(profile, "Game", ("会心率", ()))
    if provider == CloudProvider.GEMINI:
        config = body["generationConfig"]
        assert isinstance(config, dict) and "thinkingConfig" not in config
        assert profile.model_revision == "002"
    elif provider == CloudProvider.CLAUDE:
        assert "thinking" not in body and "output_config" not in body
    else:
        assert "thinking" not in body and "reasoning_effort" not in body


@pytest.mark.parametrize("provider", [CloudProvider.GEMINI, CloudProvider.CLAUDE])
def test_pagination_uses_fixed_origin_and_caps_repeated_tokens(provider):
    requests = []

    def handle(request):
        requests.append(request)
        row = {"id": "model-" + str(len(requests))}
        if provider == CloudProvider.GEMINI:
            row = {"name": "models/" + row["id"], "supportedGenerationMethods": ["generateContent"]}
        value: dict[str, object] = {"models" if provider == CloudProvider.GEMINI else "data": [row]}
        if len(requests) == 1:
            value.update(
                {"nextPageToken": "https://evil.invalid/secret"}
                if provider == CloudProvider.GEMINI
                else {"has_more": True, "last_id": "https://evil.invalid/secret"}
            )
        return httpx2.Response(200, json=value)

    result = fetch_models(
        provider, "fixture-key", threading.Event(), transport=httpx2.MockTransport(handle)
    )
    assert len(result) == 2
    assert requests[0].url.host == requests[1].url.host
    assert "evil.invalid" in str(requests[1].url.params)
    count = 0

    def repeat(request):
        nonlocal count
        count += 1
        return httpx2.Response(
            200,
            json={
                "models": [],
                "data": [],
                "has_more": True,
                "last_id": "again",
                "nextPageToken": "again",
            },
        )

    with pytest.raises(ValueError, match="pagination"):
        fetch_models(provider, "key", threading.Event(), transport=httpx2.MockTransport(repeat))
    assert count == 2


def test_filters_non_text_models_and_uses_claude_capabilities():
    def fetch(provider, rows):
        return fetch_models(
            provider,
            "key",
            threading.Event(),
            transport=httpx2.MockTransport(
                lambda _: httpx2.Response(
                    200, json={"models" if provider == CloudProvider.GEMINI else "data": rows}
                )
            ),
        )

    assert [
        m.model
        for m in fetch(CloudProvider.OPENAI, [{"id": "text-embedding-3"}, {"id": "future-chat"}])
    ] == ["future-chat"]
    assert not fetch(
        CloudProvider.GEMINI,
        [{"name": "models/embedding", "supportedGenerationMethods": ["embedContent"]}],
    )
    assert not fetch(CloudProvider.DEEPSEEK, [{"id": "image", "output_modalities": ["image"]}])
    model = fetch(
        CloudProvider.CLAUDE,
        [
            {
                "id": "future-claude",
                "lifecycle": "deprecated",
                "capabilities": {
                    "thinking": {"types": {"disabled": {"supported": True}}},
                    "effort": {"low": {"supported": True}},
                },
            }
        ],
    )[0]
    assert model.decoding == "claude-disabled-low" and model.lifecycle == "deprecated"
    body = request_body(model.profile(CloudProvider.CLAUDE), "", ("設定", ()))
    assert body["thinking"] == {"type": "disabled"}
    assert body["output_config"] == {"effort": "low"}


def test_refresh_errors_cancellation_bounds_and_secret_redaction(tmp_path):
    cache = CatalogCache(tmp_path)
    key = "api-" + "a" * 32
    old = (CatalogModel("old-model", "Old"),)
    cache.write(CloudProvider.OPENAI, key, old)

    def broken(_):
        raise ValueError("Authorization: PRIVATE-SECRET")

    for handle in (
        broken,
        lambda _: httpx2.Response(401, text="PRIVATE-SECRET"),
        lambda _: httpx2.Response(302, headers={"Location": "https://evil.invalid"}),
        lambda _: httpx2.Response(200, content=b"x" * 262145),
        lambda _: httpx2.Response(200, json={"data": [{"id": f"m{i}"} for i in range(513)]}),
    ):
        with pytest.raises(ValueError) as error:
            fetch_models(
                CloudProvider.OPENAI,
                "PRIVATE-SECRET",
                threading.Event(),
                transport=httpx2.MockTransport(handle),
            )
        assert "PRIVATE-SECRET" not in str(error.value)
        assert cache.read(CloudProvider.OPENAI, key) == old
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(ValueError, match="cancelled"):
        fetch_models(
            CloudProvider.OPENAI,
            "key",
            cancelled,
            transport=httpx2.MockTransport(lambda _: pytest.fail("network after cancel")),
        )
    assert cache.read(CloudProvider.OPENAI, key) == old


def test_catalog_and_saved_profile_survive_offline_without_leaking_or_crossing_accounts(tmp_path):
    cache = CatalogCache(tmp_path / "catalogs")
    a, b = "api-" + "a" * 32, "api-" + "b" * 32
    entry = CatalogModel("future-model", "Future model", revision="002")
    cache.write(CloudProvider.GEMINI, a, (entry,))
    assert cache.read(CloudProvider.GEMINI, a) == (entry,)
    assert (
        cache.read(CloudProvider.OPENAI, a) is None and cache.read(CloudProvider.GEMINI, b) is None
    )
    cache.write(CloudProvider.GEMINI, b, ())
    assert cache.read(CloudProvider.GEMINI, b) == ()
    store = CloudProfiles(tmp_path / "profiles.json")
    profile = replace(entry.profile(CloudProvider.GEMINI), credential=a)
    store.write((profile,))
    assert store.read() == (profile,)
    assert decoding_identity(profile) != decoding_identity(replace(profile, model_revision="003"))
    assert "secret" not in json.dumps(json.loads(store.path.read_text()))
    (cache.root / (a + ".json")).write_text("broken", encoding="utf-8")
    assert cache.read(CloudProvider.GEMINI, a) is None
    with pytest.raises(ValueError):
        cache.write(CloudProvider.GEMINI, "../escape", ())
    # Existing saved presets keep their decoding contract and cache identity.
    preset = CloudProfile.new(CloudProvider.OPENAI)
    assert request_body(preset, "", ("設定", ()))["reasoning_effort"] == "none"


@pytest.mark.parametrize("waiting_for", ["headers", "body"])
def test_cancel_interrupts_catalog_request_before_another_byte_arrives(waiting_for):
    entered, cancel, closed = threading.Event(), threading.Event(), threading.Event()

    async def wait():
        entered.set()
        try:
            await asyncio.sleep(5)
        finally:
            closed.set()

    class Body(httpx2.AsyncByteStream):
        async def __aiter__(self):
            await wait()
            yield b'{"data":[]}'

    async def handle(request):
        if waiting_for == "headers":
            await wait()
        return httpx2.Response(200, stream=Body())

    errors = []

    def refresh():
        try:
            fetch_models(
                CloudProvider.OPENAI, "fixture", cancel, transport=httpx2.MockTransport(handle)
            )
        except ValueError as error:
            errors.append(str(error))

    thread = threading.Thread(target=refresh)
    thread.start()
    try:
        assert entered.wait(2)
        cancel.set()
        assert closed.wait(1)
    finally:
        cancel.set()
        thread.join(2)
    assert not thread.is_alive() and len(errors) == 1 and "cancelled" in errors[0]
