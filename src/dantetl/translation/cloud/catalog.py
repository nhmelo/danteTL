"""Explicit, bounded provider discovery and non-secret per-connection catalog caching."""

import asyncio
import json
import re
import threading
import time
from dataclasses import asdict
from pathlib import Path

import httpx2

from dantetl.atomic_file import replace_file
from dantetl.i18n import tr
from dantetl.translation.cloud.credentials import valid_credential
from dantetl.translation.cloud.protocol import request_headers
from dantetl.translation.cloud.values import CatalogModel, CloudProfile, CloudProvider, recommended
from dantetl.translation.http_request import ResponseTooLarge, read_response

URLS = {
    CloudProvider.OPENAI: "https://api.openai.com/v1/models",
    CloudProvider.DEEPSEEK: "https://api.deepseek.com/models",
    CloudProvider.GEMINI: "https://generativelanguage.googleapis.com/v1beta/models",
    CloudProvider.CLAUDE: "https://api.anthropic.com/v1/models",
}


def _model(provider: CloudProvider, row: object) -> CatalogModel | None:
    if not isinstance(row, dict):
        return None
    key = row.get("name" if provider == CloudProvider.GEMINI else "id", "")
    if not isinstance(key, str):
        return None
    if provider == CloudProvider.GEMINI:
        key = key.removeprefix("models/")
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,120}", key):
        return None
    # Catalogs also contain embeddings, audio and image generators. Unknown text models
    # remain testable; an ID alone never earns a compatibility badge.
    if provider == CloudProvider.GEMINI and "generateContent" not in row.get(
        "supportedGenerationMethods", []
    ):
        return None
    if provider == CloudProvider.DEEPSEEK and "text" not in row.get("output_modalities", ["text"]):
        return None
    if provider == CloudProvider.OPENAI and any(
        word in key.lower()
        for word in (
            "embedding",
            "whisper",
            "tts",
            "audio",
            "realtime",
            "dall-e",
            "image",
            "moderation",
            "sora",
        )
    ):
        return None
    known = next((m for m in recommended(provider) if m.model == key), None)
    label = row.get("displayName", row.get("display_name", row.get("name", key)))
    if not isinstance(label, str) or not label.strip() or len(label) > 160:
        label = key
    if known:
        label = known.label
    decoding = known.decoding if known else "default"
    if not known and provider == CloudProvider.GEMINI and row.get("thinking") is False:
        decoding = "gemini-no-thinking"
    if not known and provider == CloudProvider.CLAUDE:
        capabilities = row.get("capabilities")
        if isinstance(capabilities, dict):
            thinking, effort = capabilities.get("thinking", {}), capabilities.get("effort", {})
            types = thinking.get("types", {}) if isinstance(thinking, dict) else {}
            low = (
                isinstance(effort, dict)
                and isinstance(effort.get("low"), dict)
                and effort["low"].get("supported") is True
            )
            if isinstance(types, dict):
                for kind, mode in (
                    ("disabled", "disabled"),
                    ("between_tools", "between"),
                    ("adaptive", "adaptive"),
                ):
                    detail = types.get(kind, {})
                    if (
                        isinstance(detail, dict)
                        and detail.get("supported") is True
                        and (kind != "adaptive" or low)
                    ):
                        decoding = "claude-" + mode + ("-low" if low else "")
                        break
    revision = row.get("version", "") if provider == CloudProvider.GEMINI else ""
    revision = revision if isinstance(revision, str) and len(revision) <= 120 else ""
    lifecycle = row.get("lifecycle", "active")
    if lifecycle not in ("active", "deprecated", "retired"):
        lifecycle = "active"
    model = CatalogModel(key, label, decoding, revision, lifecycle)
    model.profile(provider)  # Validate persisted/request values at the boundary.
    return model


class _DiscoveryError(ValueError):
    """Sanitized discovery diagnostic, safe for display."""


def fetch_models(
    provider: CloudProvider,
    secret: str,
    cancel: threading.Event,
    *,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> tuple[CatalogModel, ...]:
    """Metadata only. Never test inference, follow provider URLs, or switch credentials."""
    if provider not in URLS or not valid_credential(secret):
        raise _DiscoveryError(
            tr("Choose an API provider and enter its key before refreshing models.")
        )
    headers = request_headers(CloudProfile.new(provider), secret)
    rows: dict[str, CatalogModel] = {}
    params: dict[str, str | int] = {}
    tokens: set[str] = set()
    deadline = time.monotonic() + 25
    try:
        for _ in range(8):
            status, content = asyncio.run(
                read_response(
                    "GET",
                    URLS[provider],
                    cancel=cancel,
                    deadline=deadline,
                    max_bytes=262_144,
                    transport=transport,
                    skip_error_body=True,
                    headers=headers,
                    params=params,
                )
            )
            if status != 200:
                raise _DiscoveryError(
                    tr(
                        "Model refresh failed (HTTP {status_code}). Check the "
                        "connection; the previous list is kept.",
                        status_code=status,
                    )
                )
            value = json.loads(content)
            page = (
                value.get("models" if provider == CloudProvider.GEMINI else "data")
                if isinstance(value, dict)
                else None
            )
            if not isinstance(page, list) or len(page) > 1024:
                raise _DiscoveryError(tr("Provider returned an unsupported model catalog."))
            for raw in page:
                model = _model(provider, raw)
                if model:
                    rows[model.model] = model
                if len(rows) > 512:
                    raise _DiscoveryError(tr("Provider model catalog exceeded the model limit."))
            token = (
                value.get("nextPageToken", "")
                if provider == CloudProvider.GEMINI
                else value.get("last_id", "")
                if value.get("has_more")
                else ""
            )
            if not token:
                if cancel.is_set():
                    raise InterruptedError
                if time.monotonic() >= deadline:
                    raise TimeoutError
                return tuple(rows.values())
            if not isinstance(token, str) or len(token) > 2048 or token in tokens:
                raise _DiscoveryError(tr("Provider returned invalid catalog pagination."))
            tokens.add(token)
            params = {"pageToken" if provider == CloudProvider.GEMINI else "after_id": token}
        raise _DiscoveryError(tr("Provider catalog exceeded the pagination limit."))
    except InterruptedError:
        raise _DiscoveryError(tr("Model refresh cancelled.")) from None
    except TimeoutError:
        raise _DiscoveryError(
            tr("Model refresh cancelled or timed out; the previous list is kept.")
        ) from None
    except ResponseTooLarge:
        raise _DiscoveryError(tr("Provider model catalog exceeded the size limit.")) from None
    except (httpx2.HTTPError, json.JSONDecodeError, UnicodeError):
        raise _DiscoveryError(
            tr(
                "Could not read the provider catalog. Check your connection; the previous "
                "list is kept."
            )
        ) from None
    except _DiscoveryError:
        raise
    except Exception:
        raise _DiscoveryError(tr("Provider discovery failed; the previous list is kept.")) from None


class CatalogCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, connection: str) -> Path:
        if not re.fullmatch(r"(?:api|cloud)-[0-9a-f]{32}", connection):
            raise ValueError(tr("Invalid catalog connection."))
        return self.root / (connection + ".json")

    def read(self, provider: CloudProvider, connection: str) -> tuple[CatalogModel, ...] | None:
        try:
            path = self._path(connection)
            if not path.exists() or path.stat().st_size > 262_144:
                return None
            value = json.loads(path.read_text("utf-8"))
            if value.get("provider") != provider or value.get("version") != 1:
                return None
            rows = value["models"]
            if not isinstance(rows, list) or len(rows) > 512:
                return None
            models = tuple(CatalogModel(**row) for row in rows)
            for model in models:
                model.profile(provider)
                if model.lifecycle not in ("active", "deprecated", "retired"):
                    return None
            return models
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return None

    def write(
        self, provider: CloudProvider, connection: str, models: tuple[CatalogModel, ...]
    ) -> None:
        path = self._path(connection)
        if len(models) > 512:
            raise ValueError(tr("Catalog exceeded the model limit."))
        payload = json.dumps(
            {"version": 1, "provider": provider, "models": [asdict(m) for m in models]}
        )
        if len(payload.encode("utf-8")) > 262_144:
            raise ValueError(tr("Catalog exceeded the size limit."))
        replace_file(path, payload.encode("utf-8"))
