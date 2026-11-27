"""Validated user preferences and coalesced atomic writes; no window handles or model paths."""

import json
import re
import threading
from dataclasses import asdict
from pathlib import Path

from dantetl.application.hotkeys import parse_shortcut
from dantetl.application.preference_values import MAX_CAPTURE_RATE, Preferences, valid_executable_id
from dantetl.atomic_file import replace_file
from dantetl.file_lock import file_lock
from dantetl.i18n import INTERFACE_LANGUAGES, tr
from dantetl.languages.selection import SOURCE_LANGUAGES
from dantetl.languages.target import TARGET_LANGUAGES
from dantetl.translation.ollama.values import OllamaConfig
from dantetl.translation.references import valid_imported_key

_MAX_BYTES = 32_768


def decode_preferences(value: object) -> Preferences:
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError(tr("Unsupported settings format; defaults are being used."))
    defaults = asdict(Preferences())
    ranges = {"opacity": (20, 100), "text_size": (10, 60), "capture_rate": (0, MAX_CAPTURE_RATE)}
    choices = {
        "source_language": set(SOURCE_LANGUAGES),
        "output_language": set(TARGET_LANGUAGES),
        "interface_language": set(INTERFACE_LANGUAGES),
        "translation_route": {"local", "cloud", "long", "short"},
        "ocr_device": {"auto", "cpu", "auto-directml"},
        "model": {
            "auto",
            "none",
            "gemma",
            "gemma-e4b",
            "qwen35-4b",
            "plamo-2b",
            "plamo-0.8b",
        },
        "setup_model": {
            "",
            "gemma",
            "gemma-e4b",
            "qwen35-4b",
            "plamo-2b",
            "plamo-0.8b",
        },
    }
    for key, default in defaults.items():
        if key == "ollama":
            try:
                defaults[key] = OllamaConfig.decode(value[key])
            except (KeyError, ValueError, TypeError):
                defaults[key] = OllamaConfig()
            continue
        candidate = value.get(key, default)
        if key == "favorite_executables":
            if isinstance(candidate, (list, tuple)):
                defaults[key] = tuple(
                    sorted({entry for entry in candidate if valid_executable_id(entry)})
                )
            continue
        if type(candidate) is not type(default):
            continue
        if key in ranges and not ranges[key][0] <= candidate <= ranges[key][1]:
            continue
        if (
            key in choices
            and candidate not in choices[key]
            and not (key == "model" and valid_imported_key(candidate))
        ):
            continue
        if key in ("overlay_hotkey", "capture_hotkey"):
            try:
                parse_shortcut(candidate)
            except ValueError:
                continue
        if (
            key == "cloud_model"
            and candidate
            and not re.fullmatch(r"cloud-[0-9a-f]{32}", candidate)
        ):
            continue
        if key == "game_hint" and (len(candidate) > 120 or any(ord(c) < 32 for c in candidate)):
            continue
        if (
            key in ("short_model", "long_model", "history_model")
            and candidate not in ("", "legacy" if key != "history_model" else "layout")
            and not re.fullmatch(
                r"(?:local:[a-z0-9._-]{1,100}|cloud:cloud-[0-9a-f]{32}|ollama:[0-9a-f]{64})",
                candidate,
            )
        ):
            continue
        defaults[key] = candidate
    if "game_from_window" not in value and defaults["game_hint"]:
        defaults["game_from_window"] = False
    return Preferences(**defaults)


class PreferenceStore:
    """One pending value replaces older writes; shutdown can poll idle without blocking Qt."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.initial = Preferences()
        self.load_warning = ""
        self._error = ""
        self._lock = threading.Lock()
        self._pending: Preferences | None = None
        self._thread: threading.Thread | None = None
        self._last: Preferences | None = None
        self._written: Preferences | None = None
        self._read_only = False
        if path and path.exists():
            try:
                _, self.initial = self._read(path)
            except (OSError, ValueError, TypeError) as error:
                # Never silently overwrite corrupt or newer settings on startup.
                self.load_warning = str(error)
                self._read_only = True

    @property
    def idle(self) -> bool:
        with self._lock:
            return self._thread is None

    @property
    def error(self) -> str:
        with self._lock:
            return self._error or self.load_warning

    @property
    def current(self) -> Preferences:
        with self._lock:
            return self._last or self.initial

    def save(self, value: Preferences) -> None:
        with self._lock:
            if value == self._last and not self._error:
                return
            self._last = value
            if self.path is None or self._read_only:
                return
            self._pending = value
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._write_pending, daemon=True, name="settings"
                )
                self._thread.start()

    def _write_pending(self) -> None:
        while True:
            with self._lock:
                value, self._pending = self._pending, None
                if value is None:
                    self._thread = None
                    return
            try:
                self._write(value)
                error = ""
            except (OSError, ValueError) as failure:
                error = tr("Could not save settings: {failure}", failure=failure)
            with self._lock:
                self._error = error

    @staticmethod
    def _read(path: Path) -> tuple[dict[str, object], Preferences]:
        with path.open("rb") as stream:
            data = stream.read(_MAX_BYTES + 1)
        if len(data) > _MAX_BYTES:
            raise ValueError(tr("Settings file is too large; defaults are being used."))
        record = json.loads(data.decode("utf-8"))
        return record, decode_preferences(record)

    def _write(self, value: Preferences) -> None:
        assert self.path is not None
        previous = asdict(self._written or self.initial)
        fields = asdict(value)
        with file_lock(self.path.with_name(self.path.name + ".lock")):
            try:
                record, current = self._read(self.path)
            except FileNotFoundError:
                record = {"version": 1, **fields}
            else:
                # Fill absent migrated fields; another version may know values we cannot decode.
                record = {**asdict(current), **record}
                record.update((k, v) for k, v in fields.items() if v != previous[k])
                before, after = (
                    set(previous["favorite_executables"]),
                    set(value.favorite_executables),
                )
                if before != after:
                    record["favorite_executables"] = sorted(
                        (set(current.favorite_executables) - (before - after)) | (after - before)
                    )
            data = json.dumps(record, indent=2).encode("utf-8")
            if len(data) > _MAX_BYTES:
                raise ValueError(tr("Settings file is too large; defaults are being used."))
            replace_file(self.path, data)
            self._written = value
