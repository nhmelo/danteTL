"""The persisted ``kind:key`` model reference format and imported-model keys, owned in one place.

References stay plain strings where they are stored and compared (preferences, plans, options).
Build and inspect them only through these functions.
"""

import re
from enum import StrEnum


class ModelKind(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"
    OLLAMA = "ollama"


def format_reference(kind: ModelKind, key: str) -> str:
    return f"{kind.value}:{key}"


def reference_kind(reference: str) -> ModelKind | None:
    prefix, separator, _ = reference.partition(":")
    if not separator:
        return None
    try:
        return ModelKind(prefix)
    except ValueError:
        return None


def reference_key(reference: str) -> str:
    """The key after a known kind prefix; any other text is returned unchanged."""
    return reference.partition(":")[2] if reference_kind(reference) else reference


def is_remote(reference: str) -> bool:
    """Cloud and Ollama references are configured independently of local downloads."""
    return reference_kind(reference) in (ModelKind.CLOUD, ModelKind.OLLAMA)


_IMPORTED_PREFIX = "imported-"
_IMPORTED_KEY = re.compile(r"imported-[0-9a-f]{64}")


def imported_key(identity: str) -> str:
    return _IMPORTED_PREFIX + identity


def is_imported(key: str) -> bool:
    return key.startswith(_IMPORTED_PREFIX)


def valid_imported_key(key: str) -> bool:
    return _IMPORTED_KEY.fullmatch(key) is not None
