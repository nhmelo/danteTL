"""Bounded GGUF inspection and local registrations; never executes model code or downloads."""

import hashlib
import json
import re
import struct
import threading
from dataclasses import asdict
from pathlib import Path
from typing import BinaryIO

from dantetl.atomic_file import replace_file
from dantetl.file_lock import file_lock
from dantetl.i18n import tr
from dantetl.paths import state_root
from dantetl.runtime.setup_values import Cancelled, ModelProgress, Progress
from dantetl.translation.models.values import PROFILES, GgufInfo, ImportedModel


class _Reader:
    """Skip vocabulary arrays without retaining them; cap metadata work and string sizes."""

    def __init__(self, stream: BinaryIO, size: int) -> None:
        self.stream, self.size = stream, size

    def read(self, count: int) -> bytes:
        if count < 0 or self.stream.tell() + count > min(self.size, 64 * 2**20):
            raise ValueError(tr("GGUF metadata is truncated or exceeds the supported size."))
        data = self.stream.read(count)
        if len(data) != count:
            raise ValueError(tr("Truncated GGUF file."))
        return data

    def number(self, fmt: str) -> int:
        return struct.unpack("<" + fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self, *, keep: bool = True) -> str:
        length = self.number("Q")
        if length > 262_144:
            raise ValueError(tr("GGUF metadata string is too large."))
        data = self.read(length)
        return data.decode("utf-8") if keep else ""

    def value(self, kind: int, keep: bool = False) -> str | int | None:
        widths = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if kind == 8:
            return self.string(keep=keep)
        if kind == 9:
            subtype, length = self.number("I"), self.number("Q")
            if length > 1_000_000 or subtype not in (*widths, 8):
                raise ValueError(tr("Unsupported GGUF metadata array."))
            if subtype == 8:
                for _ in range(length):
                    self.string(keep=False)
            else:
                self.read(length * widths[subtype])
            return None
        if kind not in widths:
            raise ValueError(tr("Unsupported GGUF metadata value."))
        data = self.read(widths[kind])
        return int.from_bytes(data, "little") if keep else None


def inspect_gguf(path: Path, profile: str = "auto") -> GgufInfo:
    if profile not in PROFILES:
        raise ValueError(tr("Unknown prompt profile."))
    size = path.stat().st_size
    if not path.is_file() or path.suffix.lower() != ".gguf" or not 24 <= size <= 32 * 2**30:
        raise ValueError(tr("Choose one complete GGUF language model (up to 32 GiB)."))
    metadata: dict[str, str | int | None] = {}
    wanted = {"general.name", "general.architecture", "tokenizer.chat_template", "split.count"}
    with path.open("rb") as stream:
        reader = _Reader(stream, size)
        if reader.read(4) != b"GGUF" or reader.number("I") not in (2, 3):
            raise ValueError(tr("This is not a supported GGUF v2/v3 file."))
        tensors, count = reader.number("Q"), reader.number("Q")
        if not 1 <= tensors <= 100_000 or count > 100_000:
            raise ValueError(tr("This is not a complete language model."))
        for _ in range(count):
            key = reader.string()
            value = reader.value(reader.number("I"), keep=key in wanted)
            if key in wanted:
                if key in metadata:
                    raise ValueError(tr("Duplicate GGUF metadata field."))
                metadata[key] = value
    if metadata.get("split.count", 1) != 1:
        raise ValueError(
            tr("Split GGUF models are not supported yet. Choose a single-file version.")
        )
    architecture = metadata.get("general.architecture")
    if not isinstance(architecture, str) or not re.fullmatch(r"[a-z0-9_-]{1,64}", architecture):
        raise ValueError(tr("Missing or invalid model architecture."))
    if any(term in architecture for term in ("bert", "clip", "embedding")):
        raise ValueError(
            tr("Choose a text-generation model, not an embedding or vision projector.")
        )
    template = metadata.get("tokenizer.chat_template", "")
    if not isinstance(template, str) or len(template) > 65_536:
        raise ValueError(tr("The embedded chat template is unsupported."))
    if "source_lang_code" in template:
        raise ValueError(
            tr("TranslateGemma needs a specialized translation profile; it is not supported yet.")
        )
    if profile == "auto":
        profile = "gemma4" if architecture == "gemma4" else "embedded"
        if (
            architecture == "plamo3"
            and "<|plamo:tag|>dataset" in template
            and "translation" in template
        ):
            profile = "plamo3"
    if profile == "plamo3" and architecture != "plamo3":
        raise ValueError(tr("The PLaMo 3 profile requires a PLaMo 3 model."))
    if profile == "gemma4" and architecture != "gemma4":
        raise ValueError(tr("The Gemma 4 profile requires a Gemma 4 model."))
    if profile == "embedded" and not template:
        raise ValueError(
            tr("No embedded chat template. Select the model's documented prompt profile.")
        )
    name = metadata.get("general.name") or path.stem
    name = " ".join(str(name).split())[:100]
    return GgufInfo(name, architecture, template, size, profile)


def fingerprint(path: Path, progress: Progress, cancel: threading.Event) -> tuple[str, int]:
    before = path.stat()
    digest, received = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 2**20):
            if cancel.is_set():
                raise Cancelled()
            digest.update(chunk)
            received += len(chunk)
            progress(ModelProgress("Checking local model…", received, before.st_size))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(
            tr("The model changed while it was being checked. Retry after copying finishes.")
        )
    return digest.hexdigest(), received


def describe(
    path: Path, profile: str, progress: Progress, cancel: threading.Event
) -> ImportedModel:
    path = path.resolve(strict=True)
    info = inspect_gguf(path, profile)
    digest, size = fingerprint(path, progress, cancel)
    template_hash = hashlib.sha256(info.template.encode()).hexdigest()
    identity = hashlib.sha256(f"{digest}:{info.profile}:{template_hash}".encode()).hexdigest()
    return ImportedModel(
        "imported-" + identity,
        str(path),
        info.name,
        digest,
        size,
        info.architecture,
        info.profile,
        template_hash,
    )


_MAX_REGISTRY_BYTES = 262_144


def registry_path() -> Path:
    return state_root() / "data/imported-models.json"


def read_registry(path: Path | None = None) -> tuple[ImportedModel, ...]:
    path = path or registry_path()
    try:
        with path.open("rb") as stream:
            return _decode_registry(stream.read(_MAX_REGISTRY_BYTES + 1))
    except FileNotFoundError:
        return ()


def _decode_registry(payload: bytes) -> tuple[ImportedModel, ...]:
    if len(payload) > _MAX_REGISTRY_BYTES:
        raise ValueError(tr("Imported model list is too large."))
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError(tr("Unsupported imported model list."))
    entries = data.get("models")
    if not isinstance(entries, list) or len(entries) > 32:
        raise ValueError(tr("Imported model list must contain at most 32 models."))
    models = []
    for entry in entries:
        try:
            model = ImportedModel(**entry)
            if (
                not re.fullmatch(r"imported-[0-9a-f]{64}", model.key)
                or not re.fullmatch(r"[0-9a-f]{64}", model.sha256)
                or not re.fullmatch(r"[0-9a-f]{64}", model.template_hash)
                or model.profile not in PROFILES
                or model.profile == "auto"
                or not Path(model.path).is_absolute()
                or len(model.path) > 4096
                or not isinstance(model.name, str)
                or not 1 <= len(model.name) <= 100
                or not isinstance(model.size, int)
                or not 24 <= model.size <= 32 * 2**30
                or not re.fullmatch(r"[a-z0-9_-]{1,64}", model.architecture)
            ):
                raise ValueError(tr("Invalid imported model registration."))
        except (TypeError, KeyError) as error:
            raise ValueError(tr("Invalid imported model registration.")) from error
        if any(m.key == model.key for m in models):
            raise ValueError(tr("Duplicate imported model registration."))
        models.append(model)
    return tuple(models)


def save_registration(
    model: ImportedModel, *, remove: bool = False, path: Path | None = None
) -> None:
    """Called on the setup worker. Files remain owned by the user, including on removal."""
    path = path or registry_path()
    with file_lock(path.with_suffix(".lock")):
        models = [m for m in read_registry(path) if m.key != model.key]
        if not remove:
            models.append(model)
        if len(models) > 32:
            raise ValueError(tr("Remove an imported model before adding another (limit 32)."))
        payload = json.dumps(
            {"version": 1, "models": [asdict(m) for m in models]}, ensure_ascii=False
        ).encode("utf-8")
        _decode_registry(payload)
        replace_file(path, payload)


def available(model: ImportedModel) -> bool:
    """Cheap discovery on the inventory worker; load still verifies integrity."""
    try:
        return Path(model.path).is_file() and Path(model.path).stat().st_size == model.size
    except OSError:
        return False
