"""Non-secret provider/model/account values; no I/O or backend factories."""

import re
from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4

from dantetl.i18n import mark, tr

MAX_CLOUD_PROFILES = 16


class CloudProvider(StrEnum):
    """Supported cloud services. Values are the persisted profile and connection names."""

    CHATGPT = "chatgpt"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    CLAUDE = "claude"


PROVIDERS = {
    CloudProvider.CHATGPT: mark("ChatGPT plan"),
    CloudProvider.OPENAI: mark("OpenAI · API key"),
    CloudProvider.DEEPSEEK: mark("DeepSeek · API key"),
    CloudProvider.GEMINI: mark("Gemini · API key"),
    CloudProvider.CLAUDE: mark("Claude · API key"),
}

MODELS = {
    CloudProvider.OPENAI: (
        ("gpt-5.4-mini-2026-03-17", "GPT-5.4 mini"),
        ("gpt-6-luna", "GPT-6 Luna"),
    ),
    CloudProvider.DEEPSEEK: (("deepseek-flash", "DeepSeek Flash"),),
    CloudProvider.GEMINI: (
        ("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite"),
        ("gemini-3.8-flash", "Gemini 3.8 Flash"),
        ("gemini-4-hydrogen", "Gemini 4 Hydrogen"),
    ),
    CloudProvider.CLAUDE: (
        ("claude-haiku-5-5", "Claude Haiku 5.5"),
        ("claude-sonnet-5-5", "Claude Sonnet 5.5"),
    ),
    CloudProvider.CHATGPT: (),
}

ENDPOINTS = {
    CloudProvider.OPENAI: "https://api.openai.com/v1/chat/completions",
    CloudProvider.DEEPSEEK: "https://api.deepseek.com/chat/completions",
    CloudProvider.CHATGPT: "https://api.openai.com/v1/responses",
    CloudProvider.GEMINI: "https://generativelanguage.googleapis.com/v1beta/models/",
    CloudProvider.CLAUDE: "https://api.anthropic.com/v1/messages",
}


@dataclass(frozen=True, slots=True)
class CloudProfile:
    key: str
    name: str
    provider: CloudProvider
    model: str
    requests_per_minute: int = 120
    request_limit: int = 1000
    account: str = ""
    credential: str = ""
    decoding: str = "preset"
    model_revision: str = ""

    def __post_init__(self) -> None:
        if (
            not all(
                isinstance(value, str)
                for value in (
                    self.key,
                    self.name,
                    self.provider,
                    self.model,
                    self.account,
                    self.credential,
                    self.decoding,
                    self.model_revision,
                )
            )
            or type(self.requests_per_minute) is not int
            or type(self.request_limit) is not int
        ):
            raise ValueError(tr("Invalid cloud model settings."))
        if not re.fullmatch(r"cloud-[0-9a-f]{32}", self.key):
            raise ValueError(tr("Invalid cloud profile ID."))
        if not self.name.strip() or len(self.name) > 80:
            raise ValueError(tr("Choose a name of at most 80 characters."))
        if self.credential and not re.fullmatch(r"(?:api|cloud)-[0-9a-f]{32}", self.credential):
            raise ValueError(tr("Invalid API connection ID."))
        if self.provider == CloudProvider.CHATGPT:
            if self.credential:
                raise ValueError(tr("ChatGPT plan models cannot use API credentials."))
            if not re.fullmatch(r"account-[0-9a-f]{32}", self.account) or not re.fullmatch(
                r"[A-Za-z0-9._:-]{1,120}", self.model
            ):
                raise ValueError(tr("Sign in to ChatGPT and choose an account model first."))
        elif self.provider not in PROVIDERS or not re.fullmatch(
            r"[A-Za-z0-9._:-]{1,120}", self.model
        ):
            raise ValueError(tr("Choose a supported provider and model."))
        allowed = {
            CloudProvider.CHATGPT: ("preset",),
            CloudProvider.OPENAI: ("preset", "default"),
            CloudProvider.DEEPSEEK: ("preset", "default"),
            CloudProvider.GEMINI: ("preset", "default", "gemini-no-thinking"),
            CloudProvider.CLAUDE: (
                "preset",
                "default",
                "claude-disabled-low",
                "claude-between-low",
                "claude-adaptive-low",
                "claude-disabled",
                "claude-between",
            ),
        }
        if self.decoding not in allowed[self.provider] or len(self.model_revision) > 120:
            raise ValueError(tr("Unsupported cloud request settings."))
        if (
            self.provider != CloudProvider.CHATGPT
            and self.decoding == "preset"
            and self.model not in dict(MODELS[self.provider])
        ):
            raise ValueError(tr("Refresh models to configure this model's request settings."))
        if not 10 <= self.requests_per_minute <= 600 or not 1 <= self.request_limit <= 100_000:
            raise ValueError(tr("Cloud request limits are out of range."))
        object.__setattr__(self, "provider", CloudProvider(self.provider))

    @classmethod
    def new(cls, provider: CloudProvider) -> "CloudProfile":
        if not MODELS.get(provider):
            raise ValueError(tr("Choose an account model after signing in to ChatGPT."))
        model, label = MODELS[provider][0]
        return cls("cloud-" + uuid4().hex, label, provider, model)

    @property
    def credential_key(self) -> str:
        return self.credential or self.key


@dataclass(frozen=True, slots=True)
class ApiConnection:
    key: str
    provider: CloudProvider
    name: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.key, str)
            or not re.fullmatch(r"(?:api|cloud)-[0-9a-f]{32}", self.key)
            or self.provider not in PROVIDERS
            or self.provider == CloudProvider.CHATGPT
            or not isinstance(self.name, str)
            or not self.name.strip()
            or len(self.name) > 90
        ):
            raise ValueError(tr("Invalid API connection settings."))
        object.__setattr__(self, "provider", CloudProvider(self.provider))


@dataclass(frozen=True, slots=True)
class ChatGPTAccount:
    key: str
    label: str
    enabled: bool
    models: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class CatalogModel:
    model: str
    label: str
    decoding: str = "default"
    revision: str = ""
    lifecycle: str = "active"

    def profile(self, provider: CloudProvider) -> CloudProfile:
        return CloudProfile(
            "cloud-" + "0" * 32,
            self.label[:80],
            provider,
            self.model,
            decoding=self.decoding,
            model_revision=self.revision,
        )


def recommended(provider: CloudProvider) -> tuple[CatalogModel, ...]:
    return tuple(CatalogModel(key, label, "preset") for key, label in MODELS.get(provider, ()))


@dataclass(frozen=True, slots=True)
class CloudUsage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


USAGE_URL = "https://chatgpt.com/settings/usage"
