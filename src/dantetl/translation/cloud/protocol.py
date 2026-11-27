"""Provider wire formats for bounded text-only translation, without I/O or credentials storage."""

from dantetl.i18n import tr
from dantetl.languages.selection import DEFAULT_SOURCE
from dantetl.languages.source import SourceLanguage
from dantetl.languages.target import ENGLISH, TargetLanguage
from dantetl.translation.backend import RequestRejected, Source
from dantetl.translation.cloud.values import ENDPOINTS, CloudProfile, CloudProvider
from dantetl.translation.limits import MAX_ENGLISH_CHARACTERS
from dantetl.translation.prompt import translation_instruction, translation_payload

CLAUDE_VERSION = "2026-10-21"


def request_url(profile: CloudProfile) -> str:
    endpoint = ENDPOINTS[profile.provider]
    if profile.provider == CloudProvider.GEMINI:
        # Profile validation admits bounded model identifiers, never arbitrary URLs.
        return endpoint + profile.model + ":generateContent"
    return endpoint


def request_headers(profile: CloudProfile, secret: str) -> dict[str, str]:
    if profile.provider == CloudProvider.GEMINI:
        return {"x-goog-api-key": secret}
    headers = {"Authorization": "Bearer " + secret}
    if profile.provider == CloudProvider.CLAUDE:
        headers["anthropic-version"] = CLAUDE_VERSION
    return headers


def _gemini_generation(profile: CloudProfile) -> dict[str, object]:
    config: dict[str, object] = {
        "candidateCount": 1,
        # Includes internal thinking. Final visible text remains capped at 4096 characters.
        "maxOutputTokens": 2048,
        "responseMimeType": "text/plain",
        "thinkingConfig": {
            "thinkingLevel": "MINIMAL" if profile.model == "gemini-3.5-flash-lite" else "LOW",
            "includeThoughts": False,
        },
    }
    if profile.decoding != "preset":
        config.pop("thinkingConfig")
        if profile.decoding == "default":
            config["thinkingConfig"] = {"includeThoughts": False}
    return config


def _claude_decoding(profile: CloudProfile) -> dict[str, object]:
    if profile.decoding != "preset":
        config: dict[str, object] = {"max_tokens": 2048}
        if profile.decoding.endswith("-low"):
            config["output_config"] = {"effort": "low"}
        if profile.decoding.startswith("claude-"):
            kind = profile.decoding.removeprefix("claude-").removesuffix("-low")
            config["thinking"] = {"type": "between_tools" if kind == "between" else kind}
        return config
    return {
        "max_tokens": 768,
        "output_config": {"effort": "low"},
        "thinking": {
            "type": "between_tools" if profile.model == "claude-sonnet-5-5" else "disabled"
        },
    }


def decoding_identity(profile: CloudProfile) -> tuple[object, ...]:
    revision = (profile.model_revision,) if profile.model_revision else ()
    if profile.provider == CloudProvider.GEMINI:
        return ("generate-content-v1", _gemini_generation(profile), *revision)
    if profile.provider == CloudProvider.CLAUDE:
        return (CLAUDE_VERSION, _claude_decoding(profile), *revision)
    if profile.decoding != "preset":
        return ("provider-default-v1", 2048, *revision)
    # Preserve existing providers' cache identities and inference settings.
    return ("provider-minimum-reasoning", 768)


def request_body(
    profile: CloudProfile,
    game: str,
    source: Source,
    *,
    language: SourceLanguage = DEFAULT_SOURCE,
    target: TargetLanguage = ENGLISH,
) -> dict[str, object]:
    user = translation_payload(game, source)
    return prompt_body(profile, translation_instruction(language, target), user)


def prompt_body(profile: CloudProfile, instruction: str, user: str) -> dict[str, object]:
    """Build the selected provider's request for translation or classification."""
    match profile.provider:
        case CloudProvider.OPENAI:
            return _openai_body(profile, instruction, user)
        case CloudProvider.DEEPSEEK:
            return _deepseek_body(profile, instruction, user)
        case CloudProvider.CHATGPT:
            return _chatgpt_body(profile, instruction, user)
        case CloudProvider.CLAUDE:
            return {
                "model": profile.model,
                "system": instruction,
                "messages": [{"role": "user", "content": user}],
                "stream": False,
                **_claude_decoding(profile),
            }
        case CloudProvider.GEMINI:
            return {
                "systemInstruction": {"parts": [{"text": instruction}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": _gemini_generation(profile),
            }
        case _:
            raise RequestRejected(tr("Choose a supported provider and model."))


def _completion_body(model: str, instruction: str, user: str) -> dict[str, object]:
    return {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user},
        ],
    }


def _openai_body(profile: CloudProfile, instruction: str, user: str) -> dict[str, object]:
    body = _completion_body(profile.model, instruction, user)
    preset = profile.decoding == "preset"
    body.update(max_completion_tokens=768 if preset else 2048, store=False)
    if preset:
        body["reasoning_effort"] = "none"
    return body


def _deepseek_body(profile: CloudProfile, instruction: str, user: str) -> dict[str, object]:
    body = _completion_body(profile.model, instruction, user)
    preset = profile.decoding == "preset"
    body["max_tokens"] = 768 if preset else 2048
    if preset:
        body.update(thinking={"type": "disabled"}, temperature=0)
    return body


def _chatgpt_body(profile: CloudProfile, instruction: str, user: str) -> dict[str, object]:
    body: dict[str, object] = {
        "model": profile.model,
        "instructions": instruction,
        "input": [{"role": "user", "content": user}],
        "store": False,
        "stream": True,
    }
    if profile.model.startswith("gpt-5.4-mini") or profile.model == "gpt-6-luna":
        body["reasoning"] = {"effort": "none"}
    elif profile.model.startswith("gpt-6"):
        body["reasoning"] = {"effort": "low"}
    return body


def _gemini_text(value: dict) -> str:
    feedback = value.get("promptFeedback", {})
    if not isinstance(feedback, dict) or feedback.get("blockReason"):
        raise ValueError
    candidates = value["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError
    candidate = candidates[0]
    if not isinstance(candidate, dict) or candidate.get("finishReason") != "STOP":
        raise ValueError
    safety = candidate.get("safetyRatings", [])
    if not isinstance(safety, list) or any(
        not isinstance(rating, dict) or rating.get("blocked") for rating in safety
    ):
        raise ValueError
    content = candidate["content"]
    if not isinstance(content, dict) or content.get("role") != "model":
        raise ValueError
    parts = content["parts"]
    if not isinstance(parts, list) or not 1 <= len(parts) <= 32:
        raise ValueError
    texts = []
    for part in parts:
        if (
            not isinstance(part, dict)
            or not isinstance(part.get("text"), str)
            or part.get("thought")
            or set(part) - {"text", "thought", "thoughtSignature"}
        ):
            raise ValueError
        texts.append(part["text"])
    return "".join(texts)


def _claude_text(value: dict) -> str:
    if (
        value.get("type") != "message"
        or value.get("role") != "assistant"
        or value.get("stop_reason") != "end_turn"
    ):
        raise ValueError
    parts = value["content"]
    if not isinstance(parts, list) or not 1 <= len(parts) <= 32:
        raise ValueError
    texts = []
    for part in parts:
        if (
            not isinstance(part, dict)
            or part.get("type") != "text"
            or not isinstance(part.get("text"), str)
            or part.get("citations")
        ):
            raise ValueError
        texts.append(part["text"])
    return "".join(texts)


def _completion_text(value: dict) -> str:
    choices = value["choices"]
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
        raise ValueError
    message = choice["message"]
    if (
        not isinstance(message, dict)
        or message.get("refusal")
        or message.get("tool_calls")
        or message.get("reasoning_content")
    ):
        raise ValueError
    return message["content"]


def parse_answer(value: object, provider: CloudProvider = CloudProvider.OPENAI) -> str:
    try:
        if not isinstance(value, dict):
            raise ValueError
        text = (
            _gemini_text(value)
            if provider == CloudProvider.GEMINI
            else _claude_text(value)
            if provider == CloudProvider.CLAUDE
            else _completion_text(value)
        )
        if (
            not isinstance(text, str)
            or not text.strip()
            or len(text) > MAX_ENGLISH_CHARACTERS
            or "<think" in text.lower()
        ):
            raise ValueError
        return text.strip()
    except (KeyError, TypeError, ValueError, IndexError):
        raise RequestRejected(
            tr("Cloud model returned an incomplete or unsupported answer.")
        ) from None


def response_usage(provider: CloudProvider, value: object) -> object:
    if not isinstance(value, dict):
        return None
    if provider not in (CloudProvider.GEMINI, CloudProvider.CLAUDE):
        return value.get("usage")
    usage = value.get("usageMetadata" if provider == CloudProvider.GEMINI else "usage")
    if not isinstance(usage, dict):
        return None
    keys = (
        (("promptTokenCount",), ("candidatesTokenCount", "thoughtsTokenCount"))
        if provider == CloudProvider.GEMINI
        else (
            ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"),
            ("output_tokens",),
        )
    )
    counts = []
    for group in keys:
        values = [usage.get(key, 0) for key in group]
        if not all(type(n) is int and 0 <= n <= 1_000_000 for n in values):
            return None
        counts.append(sum(values))
    return {"input_tokens": counts[0], "output_tokens": counts[1]}


def account_error(value: object) -> bool:
    """Recognize fixed machine codes only; never expose provider messages or echoed secrets."""
    if not isinstance(value, dict) or not isinstance(value.get("error"), dict):
        return False
    error = value["error"]
    if error.get("code") in ("insufficient_quota", "credit_balance_exhausted", "invalid_api_key"):
        return True
    if error.get("type") in ("authentication_error", "permission_error"):
        return True
    details = error.get("details", [])
    return isinstance(details, list) and any(
        isinstance(detail, dict) and detail.get("reason") in ("API_KEY_INVALID", "API_KEY_EXPIRED")
        for detail in details
    )
