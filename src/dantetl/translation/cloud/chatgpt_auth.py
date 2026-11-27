"""Official public-client ChatGPT plan authorization, independent of API-key billing."""

import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
import socket
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlparse
from uuid import uuid4

from dantetl.i18n import tr
from dantetl.paths import state_root
from dantetl.translation.cloud.auth_work import AuthInterrupted, AuthWork
from dantetl.translation.cloud.credentials import valid_credential
from dantetl.translation.cloud.protected_store import ProtectedStore
from dantetl.translation.cloud.values import ChatGPTAccount

logger = logging.getLogger(__name__)

ISSUER = "https://auth.openai.com"
AUTHORIZE = ISSUER + "/api/accounts/authorize"
TOKEN = ISSUER + "/api/accounts/oauth/token"
RESOURCE = "https://api.openai.com/v1"
SCOPE = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
MODEL_CATALOG_RESPONSE_LIMIT = 4 * 1024 * 1024


class AuthResponseError(ValueError):
    """Sanitized protocol diagnostic suitable for UI/logs."""


def _client_id(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"oaiapp_[A-Za-z0-9_-]{1,200}", value):
        raise ValueError(tr("ChatGPT did not return a valid client registration."))
    return value


def _json_request(work: AuthWork, method: str, url: str, **kwargs) -> dict:
    try:
        status, data = work.request(method, url, **kwargs)
        try:
            value = json.loads(data)
        except ValueError:
            value = {}
        if not isinstance(value, dict):
            raise AuthResponseError(tr("ChatGPT setup returned an unsupported response."))
        if status != 200:
            code = value.get("error")
            if isinstance(code, dict):
                code = code.get("code")
            if code in {
                "invalid_grant",
                "invalid_refresh_token",
                "token_expired",
                "refresh_token_expired",
                "refresh_token_invalidated",
                "refresh_token_reused",
            }:
                raise SessionExpired(tr("ChatGPT session expired. Continue with ChatGPT again."))
            raise AuthResponseError(
                tr("ChatGPT setup failed (HTTP {status}). Try again later.", status=status)
            )
        return value
    except (AuthResponseError, SessionExpired, AuthInterrupted):
        raise
    except Exception as error:
        # Record only the type: messages can quote request bodies or tokens.
        logger.warning("ChatGPT request failed: %s", type(error).__name__)
        raise OSError(tr("Could not connect to ChatGPT. Check your internet connection.")) from None


class SessionExpired(ValueError):
    pass


def authorization_callback(path: str, state: str) -> dict[str, str]:
    parsed = urlparse(path)
    if (
        parsed.scheme
        or parsed.netloc
        or parsed.fragment
        or parsed.path != "/auth/callback"
        or len(path) > 16_384
    ):
        raise ValueError(tr("Invalid authorization callback."))
    query = parse_qs(parsed.query, max_num_fields=12)
    if any(len(value) != 1 for value in query.values()):
        raise ValueError(tr("Invalid authorization callback."))
    supplied = query.get("state", [""])[0]
    if not supplied.isascii() or not hmac.compare_digest(supplied, state):
        raise ValueError(tr("Invalid authorization callback state."))
    if "iss" in query and query["iss"] != [ISSUER]:
        raise ValueError(tr("Unexpected authorization issuer."))
    return {key: values[0] for key, values in query.items()}


def validate_identity(token: str, jwks: dict, client_id: str, nonce: str) -> dict:
    """Only keys fetched from OpenAI's fixed JWKS endpoint are trusted."""
    require_sign_in_dependencies()
    import jwt

    try:
        if not isinstance(token, str) or not 1 <= len(token) <= 16_384:
            raise ValueError
        header = jwt.get_unverified_header(token)
        # This OIDC flow implements no critical JWS extensions. Keep this boundary
        # explicit even in older optional runtimes awaiting a dependency update.
        if header.get("alg") != "RS256" or "crit" in header:
            raise ValueError
        matches = [
            key
            for key in jwks.get("keys", [])
            if isinstance(key, dict)
            and key.get("kid") == header.get("kid")
            and key.get("use", "sig") == "sig"
            and key.get("alg", "RS256") == "RS256"
        ]
        if len(matches) != 1:
            raise ValueError
        key = jwt.PyJWK.from_dict(matches[0], algorithm="RS256")
        claims = jwt.decode(
            token,
            key.key,
            algorithms=["RS256"],
            audience=client_id,
            issuer=ISSUER,
            options={"require": ["exp", "iat", "iss", "aud", "sub", "nonce"]},
        )
        if not isinstance(claims.get("nonce"), str) or not hmac.compare_digest(
            claims["nonce"], nonce
        ):
            raise ValueError
        if not isinstance(claims.get("sub"), str) or not 1 <= len(claims["sub"]) <= 256:
            raise ValueError
        audience = claims.get("aud")
        if isinstance(audience, list) and len(audience) > 1 and claims.get("azp") != client_id:
            raise ValueError
        if claims.get("azp") and claims["azp"] != client_id:
            raise ValueError
        return claims
    except (ValueError, TypeError, KeyError, jwt.PyJWTError) as error:
        raise ValueError(
            tr("ChatGPT identity validation failed; no account was activated.")
        ) from error


def require_sign_in_dependencies() -> None:
    # Older isolated GPU environments can still run local/API-key translation.
    # Never load a native module from a different interpreter to repair them.
    try:
        import jwt  # noqa: F401
        from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: F401
    except ImportError as error:
        raise ValueError(
            tr(
                "This runtime predates ChatGPT sign-in. Repair it in Settings → Acceleration "
                "and restart, or select the bundled CPU runtime."
            )
        ) from error


def token_record(previous: dict, response: dict) -> dict:
    access, refresh = response.get("access_token"), response.get("refresh_token")
    expiry = response.get("expires_in")
    if (
        not valid_credential(access)
        or not isinstance(response.get("token_type"), str)
        or response["token_type"].lower() != "bearer"
        or type(expiry) is not int
        or not 1 <= expiry <= 604_800
        or (refresh is not None and not valid_credential(refresh))
    ):
        raise ValueError(tr("ChatGPT returned an invalid token response."))
    scope = response.get("scope")
    if scope is not None and not isinstance(scope, str):
        raise ValueError(tr("ChatGPT returned invalid permissions."))
    return {
        **previous,
        "access_token": access,
        "refresh_token": refresh or previous.get("refresh_token", ""),
        "id_token": response.get("id_token", previous.get("id_token", "")),
        "expires_at": time.time() + expiry,
        "scopes": scope.split() if scope is not None else previous.get("scopes", []),
    }


class ChatGPTConnections:
    def __init__(self, store: ProtectedStore | None = None) -> None:
        self.store = store or ProtectedStore(state_root() / "data/chatgpt")

    def accounts(
        self, on_problem: Callable[[str], None] = lambda _: None
    ) -> tuple[ChatGPTAccount, ...]:
        values = []
        for key in self.store.account_keys():
            try:
                record = self.store.read(key)
                rows, scopes = record.get("models", []), record.get("scopes", [])
                if (
                    not isinstance(scopes, (list, tuple))
                    or not isinstance(rows, (list, tuple))
                    or not all(
                        isinstance(row, (list, tuple))
                        and len(row) == 2
                        and all(isinstance(value, str) for value in row)
                        for row in rows
                    )
                ):
                    raise ValueError
                label = record.get("email") or "ChatGPT account"
                values.append(
                    ChatGPTAccount(
                        key,
                        f"{label} · {key[-6:]}",
                        bool(record.get("access_token") and PLAN_SCOPE in scopes),
                        tuple((model, name) for model, name in rows),
                    )
                )
            except (OSError, ValueError):
                on_problem(tr("Invalid ChatGPT connection record."))
        return tuple(values)

    def _host(self, work: AuthWork) -> str:
        with self.store.lock("host", check=work.check):
            record = self.store.read("host")
            if not record:
                record = {"id": "urn:uuid:" + str(uuid4())}
                self.store.write("host", record)
            return record["id"]

    def sign_in(
        self, account: str, open_browser: Callable[[str], None], cancelled: threading.Event
    ) -> str:
        """User-initiated operation. Never reads another app's credentials or auth cache."""
        require_sign_in_dependencies()
        work = AuthWork(cancelled, deadline=time.monotonic() + 210)
        work.check()
        previous = self.store.read(account) if account else {}
        if not account and len(self.store.account_keys()) >= 8:
            raise ValueError(tr("Eight ChatGPT accounts are already registered."))
        key = account or "account-" + uuid4().hex
        state, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(3))
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )
        callback: dict[str, str] = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:
                pass  # Callback query contains a one-use authorization code.

            def do_GET(self) -> None:
                try:
                    query = authorization_callback(self.path, state)
                except ValueError:
                    self.send_error(400, "Invalid authorization callback")
                    return
                callback.update(query)
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Content-Security-Policy", "default-src 'none'")
                self.end_headers()
                self.wfile.write(b"Return to DanteTL. It will finish checking your connection.")

        class Listener(HTTPServer):
            allow_reuse_address = False

            def server_bind(self):
                # Windows SO_REUSEADDR allows another process to steal a callback port.
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                super().server_bind()

            def get_request(self):
                connection, address = super().get_request()
                connection.settimeout(0.5)
                return connection, address

        with Listener(("127.0.0.1", 0), Handler) as server:
            server.timeout = 0.1
            redirect = f"http://127.0.0.1:{server.server_port}/auth/callback"
            params = {
                "client_id": previous.get("client_id", "dynamic_agent_client"),
                "ext_agent_host_id": self._host(work),
                "response_type": "code",
                "redirect_uri": redirect,
                "scope": SCOPE,
                "resource": RESOURCE,
                "state": state,
                "nonce": nonce,
                "code_challenge_method": "S256",
                "code_challenge": challenge,
            }
            if previous.get("client_id"):
                if previous.get("id_token"):
                    params["id_token_hint"] = previous["id_token"]
                if PLAN_SCOPE not in previous.get("scopes", []):
                    params["prompt"] = "consent"
            else:
                params["agent_name_hint"] = "DanteTL"
            open_browser(AUTHORIZE + "?" + urlencode(params))
            deadline = time.monotonic() + 180
            while not callback and not cancelled.is_set() and time.monotonic() < deadline:
                server.handle_request()
        if cancelled.is_set() or not callback:
            raise ValueError(tr("ChatGPT sign-in cancelled or timed out."))
        if callback.get("error"):
            raise ValueError(
                tr("ChatGPT sign-in was declined. Your previous connection is unchanged.")
            )
        client_id = _client_id(callback.get("client_id", previous.get("client_id")))
        if previous.get("client_id") and client_id != previous["client_id"]:
            raise ValueError(tr("ChatGPT returned a different registration; account unchanged."))
        code = callback.get("code", "")
        if not code or len(code) > 8192:
            raise ValueError(tr("ChatGPT did not return an authorization code."))
        if not previous:
            # Preserve the issued registration if exchange fails; do not activate it yet.
            previous = {"client_id": client_id, "host": params["ext_agent_host_id"]}
            self.store.write(key, previous)
        # Browser interaction has its own 180-second bound; exchange/discovery share 30 seconds.
        work.deadline = min(work.deadline, time.monotonic() + 30)
        response = _json_request(
            work,
            "POST",
            TOKEN,
            data={
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect,
                "resource": RESOURCE,
            },
        )
        jwks = _json_request(work, "GET", ISSUER + "/.well-known/jwks.json")
        claims = validate_identity(response.get("id_token", ""), jwks, client_id, nonce)
        if previous.get("subject") and previous["subject"] != claims["sub"]:
            raise ValueError(tr("The signed-in account does not match this registration."))
        record = token_record(previous, response)
        record.update(subject=claims["sub"], email=str(claims.get("email", ""))[:254])
        work.check()
        with self.store.lock(key, check=work.check):
            self.store.write(key, record)
        if PLAN_SCOPE in record["scopes"]:
            self.refresh_models(key, work)
        return key

    def access_token(self, account: str, work: AuthWork | None = None) -> str:
        work = work or AuthWork()
        work.check()
        with self.store.lock(account, check=work.check):
            record = self.store.read(account)
            if not record.get("access_token") or PLAN_SCOPE not in record.get("scopes", []):
                raise ValueError(tr("Enable ChatGPT plan use in Models → Continue with ChatGPT."))
            if not valid_credential(record["access_token"]):
                raise ValueError(
                    tr("Saved ChatGPT connection is malformed. Sign in again in Models.")
                )
            if record.get("expires_at", 0) > time.time() + 60:
                return record["access_token"]
            try:
                response = _json_request(
                    work,
                    "POST",
                    TOKEN,
                    data={
                        "grant_type": "refresh_token",
                        "client_id": record["client_id"],
                        "refresh_token": record.get("refresh_token", ""),
                        "resource": RESOURCE,
                    },
                )
                record = token_record(
                    record, {k: v for k, v in response.items() if k != "id_token"}
                )
                self.store.write(account, record)
            except SessionExpired:
                self.store.write(account, self._without_tokens(record))
                raise
            if PLAN_SCOPE not in record["scopes"]:
                raise ValueError(
                    tr("ChatGPT plan permission was removed. Sign in again in Models.")
                )
            return record["access_token"]

    def refresh_models(
        self, account: str, work: AuthWork | None = None
    ) -> tuple[tuple[str, str], ...]:
        work = work or AuthWork()
        token = self.access_token(account, work)
        value = _json_request(
            work,
            "GET",
            RESOURCE + "/models",
            headers={"Authorization": "Bearer " + token},
            max_response_bytes=MODEL_CATALOG_RESPONSE_LIMIT,
        )
        rows = value.get("models")
        if not isinstance(rows, list) or len(rows) > 256:
            raise ValueError(tr("ChatGPT did not return an account model catalog."))
        models = tuple(
            (row["slug"], row["display_name"])
            for row in rows
            if isinstance(row, dict)
            and row.get("visibility") == "list"
            and isinstance(row.get("slug"), str)
            and re.fullmatch(r"[A-Za-z0-9._:-]{1,120}", row["slug"])
            and isinstance(row.get("display_name"), str)
            and len(row["display_name"]) <= 160
        )[:64]
        with self.store.lock(account, check=work.check):
            record = self.store.read(account)
            self.store.write(account, {**record, "models": models})
        return models

    @staticmethod
    def _without_tokens(record: dict) -> dict:
        return {
            k: v
            for k, v in record.items()
            if k
            not in (
                "access_token",
                "refresh_token",
                "id_token",
                "expires_at",
                "scopes",
                "models",
            )
        }

    def sign_out(self, account: str, work: AuthWork | None = None) -> str:
        work = work or AuthWork()
        confirmed = False
        with self.store.lock(account, check=work.check):
            record = self.store.read(account)
            try:
                if record.get("refresh_token"):
                    discovery = _json_request(
                        work, "GET", ISSUER + "/.well-known/openid-configuration"
                    )
                    endpoint = discovery.get("revocation_endpoint")
                    if endpoint != ISSUER + "/api/accounts/oauth/revoke":
                        raise ValueError(tr("Unrecognized ChatGPT revocation endpoint."))
                    _json_request(
                        work,
                        "POST",
                        endpoint,
                        data={
                            "token": record["refresh_token"],
                            "token_type_hint": "refresh_token",
                            "client_id": record["client_id"],
                        },
                    )
                    confirmed = True
            except Exception as error:
                # Revocation/network/parser failures must never retain local bearer tokens.
                logger.warning("ChatGPT token revocation failed: %s", type(error).__name__)
            finally:
                self.store.write(account, self._without_tokens(record))
        return (
            tr("Signed out of ChatGPT.")
            if confirmed
            else tr(
                "Signed out locally. Remote revocation was not confirmed; "
                "disconnect DanteTL in ChatGPT usage settings."
            )
        )
