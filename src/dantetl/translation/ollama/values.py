"""Persisted choices and server-scoped identities for the native Ollama API."""

import hashlib
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from dantetl.i18n import tr

DEFAULT_SERVER = "http://127.0.0.1:11434"
MAX_ENABLED = 32


def server_address(value: str) -> str:
    try:
        value = value.strip()
        if len(value) > 512 or any(ord(c) < 33 or ord(c) > 126 for c in value):
            raise ValueError
        url = urlsplit(value)
        host = (url.hostname or "").rstrip(".")
        if (
            url.scheme not in ("http", "https")
            or not host
            or url.username is not None
            or url.password is not None
            or url.path not in ("", "/")
            or url.query
            or url.fragment
            or not re.fullmatch(r"[a-zA-Z0-9.:-]+", host)
            or "%" in value
            or "\\" in value
            or host in ("ollama.com", "www.ollama.com", "0.0.0.0", "::")
        ):
            raise ValueError
        port = url.port
        if port == (443 if url.scheme == "https" else 80):
            port = None
        host = f"[{host}]" if ":" in host else host
        return f"{url.scheme}://{host}" + (f":{port}" if port is not None else "")
    except ValueError:
        raise ValueError(tr("Enter the HTTP or HTTPS address of your Ollama server.")) from None


@dataclass(frozen=True, slots=True)
class OllamaModel:
    name: str
    digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not re.fullmatch(r"[\w./:-]{1,160}", self.name, re.ASCII)
            or self.name.lower().endswith((":cloud", "-cloud"))
            or not isinstance(self.digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", self.digest)
        ):
            raise ValueError(tr("Invalid Ollama model."))

    @property
    def display_name(self) -> str:
        path, _, tag = (
            self.name.rpartition(":")
            if ":" in self.name.rsplit("/", 1)[-1]
            else (self.name, "", "")
        )
        namespace, _, base = path.rpartition("/")
        # Keep custom namespaces and tags recognizable when formatting the base name.
        base = re.sub(r"(?<=[a-zA-Z])(?=\d)", " ", base.replace("-", " ").replace("_", " "))
        title = base[:1].upper() + base[1:]
        if namespace:
            title = f"{namespace}/{title}"
        if tag:
            size = re.fullmatch(r"(\d+(?:\.\d+)?)([bm])", tag, re.I)
            title += " · " + (tag.upper() if size else tag)
        return title

    def reference(self, server: str) -> str:
        return "ollama:" + hashlib.sha256(f"{server}\0{self.name}".encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class OllamaListing:
    model: OllamaModel
    parameter_size: str = ""
    quantization: str = ""
    size_bytes: int = 0

    @classmethod
    def from_metadata(cls, model: OllamaModel, details: object, size: object) -> "OllamaListing":
        def label(key: str) -> str:
            value = details.get(key) if isinstance(details, dict) else None
            if (
                isinstance(value, str)
                and value.isascii()
                and value.isprintable()
                and len(value) <= 32
            ):
                return value.strip()
            return ""

        return cls(
            model,
            label("parameter_size"),
            label("quantization_level"),
            size if type(size) is int and 0 < size < 2**63 else 0,
        )


@dataclass(frozen=True, slots=True)
class OllamaConfig:
    server: str = DEFAULT_SERVER
    listings: tuple[OllamaListing, ...] = ()

    @property
    def models(self) -> tuple[OllamaModel, ...]:
        return tuple(entry.model for entry in self.listings)

    @classmethod
    def decode(cls, record: object) -> "OllamaConfig":
        if not isinstance(record, dict):
            raise ValueError
        server = record.get("server")
        current = "listings" in record
        rows = record.get("listings" if current else "models")
        if not isinstance(server, str) or not isinstance(rows, list) or len(rows) > MAX_ENABLED:
            raise ValueError
        entries = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError
            if current:
                model = OllamaModel(**row["model"])
                details = {
                    "parameter_size": row.get("parameter_size"),
                    "quantization_level": row.get("quantization"),
                }
                entries.append(OllamaListing.from_metadata(model, details, row.get("size_bytes")))
            else:
                entries.append(OllamaListing(OllamaModel(**row)))
        if len({entry.model.name for entry in entries}) != len(entries):
            raise ValueError
        return cls(server_address(server), tuple(entries))


@dataclass(frozen=True, slots=True)
class OllamaIdentity:
    server: str
    model: OllamaModel
    game: str
    attempt: int = 0


@dataclass(frozen=True, slots=True)
class OllamaSnapshot:
    config: OllamaConfig
    server: str = ""
    listings: tuple[OllamaListing, ...] = ()
    busy: bool = False
    saved: bool = False
    error: str = ""

    @property
    def models(self) -> tuple[OllamaModel, ...]:
        return tuple(entry.model for entry in self.listings)
