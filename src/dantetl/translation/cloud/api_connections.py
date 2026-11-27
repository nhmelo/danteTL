"""Reusable provider credentials, separate from model entries. Worker-only operations."""

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from uuid import uuid4

from dantetl.i18n import mark, tr
from dantetl.translation.cloud.credentials import Secrets
from dantetl.translation.cloud.profiles import CloudProfiles
from dantetl.translation.cloud.settings_file import SettingsFile
from dantetl.translation.cloud.values import (
    MAX_CLOUD_PROFILES,
    ApiConnection,
    CloudProfile,
    CloudProvider,
)

# Includes full version-1 catalogs with JSON-escaped Unicode names.
_FILE = SettingsFile(24 * 1024, mark("API connection settings are too large."))


@dataclass(frozen=True)
class ApiInventory:
    profiles: tuple[CloudProfile, ...]
    connections: tuple[ApiConnection, ...]
    ready: frozenset[str]


class ApiConnections:
    def __init__(self, catalog: CloudProfiles) -> None:
        self.catalog = catalog
        self.path = catalog.path.with_name(catalog.path.stem + "-connections.json")

    def discover(
        self, vault: Callable[[], Secrets], on_problem: Callable[[str], None]
    ) -> ApiInventory:
        with self.catalog.transaction():
            profiles = self.catalog.read()
            try:
                secrets = vault()
                connections = self._read(profiles, secrets, on_problem)
            except (OSError, ValueError):
                on_problem(tr("Invalid API connection settings."))
                return ApiInventory(profiles, (), frozenset())
            ready = set()
            for connection in connections:
                try:
                    if secrets.read(connection.key):
                        ready.add(connection.key)
                except (OSError, ValueError):
                    on_problem(tr("Windows could not read the saved API key."))
            return ApiInventory(profiles, connections, frozenset(ready))

    def _read(
        self,
        profiles: tuple[CloudProfile, ...],
        secrets: Secrets,
        on_problem: Callable[[str], None] = lambda _: None,
    ) -> tuple[ApiConnection, ...]:
        connections: tuple[ApiConnection, ...] = ()
        if self.path.exists():
            value = _FILE.read(self.path)
            if not isinstance(value, dict) or value.get("version") != 1:
                raise ValueError(tr("Unsupported API connection settings."))
            rows = value.get("connections")
            if not isinstance(rows, list) or len(rows) > 16:
                raise ValueError(tr("Invalid API connection settings."))
            try:
                connections = tuple(ApiConnection(**row) for row in rows)
            except TypeError:
                raise ValueError(tr("Invalid API connection settings.")) from None
            if len({c.key for c in connections}) != len(connections):
                raise ValueError(tr("Duplicate API connection IDs."))
        # Adopt previous model-owned credentials in place, without copying secret material.
        legacy = []
        for p in profiles:
            if (
                p.provider == CloudProvider.CHATGPT
                or p.credential
                or any(c.key == p.key for c in connections)
            ):
                continue
            try:
                if secrets.read(p.key):
                    legacy.append(ApiConnection(p.key, p.provider, p.name + " key"))
            except (OSError, ValueError):
                on_problem(tr("Windows could not read the saved API key."))
        if legacy:
            connections += tuple(legacy)
            self._write(connections)
        return connections

    def _write(self, connections: tuple[ApiConnection, ...]) -> None:
        if len(connections) > 16:
            raise ValueError(tr("Remove an unused API connection first (limit: sixteen)."))
        _FILE.write(self.path, {"version": 1, "connections": [asdict(c) for c in connections]})

    def secret(self, profile: CloudProfile, secrets: Secrets) -> str:
        with self.catalog.transaction():
            connections = self._read(self.catalog.read(), secrets)
            if not any(
                c.key == profile.credential_key and c.provider == profile.provider
                for c in connections
            ):
                return ""
            return secrets.read(profile.credential_key) or ""

    def save_model(
        self, profile: CloudProfile, connection: str, secret: str, secrets: Secrets
    ) -> tuple[CloudProfile, bool]:
        with self.catalog.transaction():
            profiles = self.catalog.read()
            connections = self._read(profiles, secrets)
            existing = next(
                (c for c in connections if c.key == connection and c.provider == profile.provider),
                None,
            )
            if connection and existing is None:
                raise ValueError(
                    tr("Choose a saved connection for this provider, or add a new API key.")
                )
            selected = existing or ApiConnection(
                "api-" + uuid4().hex, profile.provider, profile.name + " key"
            )
            previous = secrets.read(selected.key)
            if not secret and not previous:
                raise ValueError(tr("Enter an API key or choose a saved connection."))
            saved = replace(profile, credential=selected.key)
            updated = tuple(p for p in profiles if p.key != profile.key) + (saved,)
            if len(updated) > MAX_CLOUD_PROFILES or (existing is None and len(connections) >= 16):
                raise ValueError(tr("Remove an unused model or connection before adding another."))
            added_connection = False
            try:
                if secret:
                    secrets.write(selected.key, secret)
                if existing is None:
                    self._write(connections + (selected,))
                    added_connection = True
                self.catalog.write(updated)
            except Exception:
                if secret:
                    if previous:
                        secrets.write(selected.key, previous)
                    else:
                        secrets.delete(selected.key)
                if added_connection:
                    self._write(connections)
                raise
            return saved, bool(secret and secret != previous)

    def forget(self, key: str, secrets: Secrets) -> None:
        with self.catalog.transaction():
            connections = self._read(self.catalog.read(), secrets)
            if not any(c.key == key for c in connections):
                raise ValueError(tr("This connection is no longer available."))
            secrets.delete(key)
            self._write(tuple(c for c in connections if c.key != key))
