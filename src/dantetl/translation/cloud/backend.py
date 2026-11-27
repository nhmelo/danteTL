"""Bounded, cancellable text-only API translation. No capture or Qt dependencies."""

import asyncio
import hashlib
import json
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

import httpx2

from dantetl.i18n import tr
from dantetl.languages.selection import DEFAULT_SOURCE
from dantetl.languages.source import SourceLanguage
from dantetl.languages.target import ENGLISH, TargetLanguage
from dantetl.translation.backend import BatchResult, RequestRejected, Source
from dantetl.translation.cloud.auth_work import AuthInterrupted, AuthWork
from dantetl.translation.cloud.credentials import valid_credential
from dantetl.translation.cloud.protocol import (
    account_error,
    decoding_identity,
    parse_answer,
    prompt_body,
    request_body,
    request_headers,
    request_url,
    response_usage,
)
from dantetl.translation.cloud.responses import PlanUnavailable, ResponseStream
from dantetl.translation.cloud.values import CloudProfile, CloudProvider, CloudUsage
from dantetl.translation.dialogue.classifier import (
    Classification,
    ClassificationInput,
    classification_instruction,
    parse_classification,
)
from dantetl.translation.dialogue.context import DialogueTurn
from dantetl.translation.prompt import translation_instruction


class CloudAllowance:
    """Per-profile app-run limits survive capture restarts and backend recreation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._usage = CloudUsage()
        self._next_call = 0.0
        self._waiting: deque[object] = deque()

    def snapshot(self) -> CloudUsage:
        with self._lock:
            return self._usage

    async def admit(
        self,
        profile: CloudProfile,
        *,
        clock: Callable[[], float] = time.monotonic,
        check: Callable[[], None] = lambda: None,
    ) -> None:
        """Pace FIFO dispatch across owners; unsent cancellations cost no allowance.

        The lock protects shared counters across independent backend I/O loops.
        Only pending calls poll, normally every 50 ms; asyncio cancellation
        interrupts the wait immediately. No lock is held over an await or HTTP.
        """
        ticket = object()
        with self._lock:
            if len(self._waiting) >= 16:
                raise RequestRejected(
                    tr("Cloud request queue is full. Retry after pending work finishes.")
                )
            self._waiting.append(ticket)
        try:
            while True:
                check()
                with self._lock:
                    if self._usage.requests >= profile.request_limit:
                        raise RequestRejected(
                            tr("Cloud request limit reached for this app run. Open Models.")
                        )
                    delay = self._next_call - clock()
                    if self._waiting[0] is ticket and delay <= 0:
                        # This is the dispatch boundary, not the start of a wait.
                        check()
                        self._usage = CloudUsage(
                            self._usage.requests + 1,
                            self._usage.input_tokens,
                            self._usage.output_tokens,
                        )
                        self._next_call = clock() + 60 / profile.requests_per_minute
                        return
                    wait = min(0.05, max(0.005, delay)) if self._waiting[0] is ticket else 0.05
                await asyncio.sleep(wait)
        finally:
            with self._lock:
                self._waiting.remove(ticket)

    def record(self, value: object) -> None:
        if not isinstance(value, dict):
            return
        counts = [
            value.get("prompt_tokens", value.get("input_tokens", 0)),
            value.get("completion_tokens", value.get("output_tokens", 0)),
        ]
        if not all(type(n) is int and 0 <= n <= 1_000_000 for n in counts):
            return
        with self._lock:
            self._usage = CloudUsage(
                self._usage.requests,
                self._usage.input_tokens + counts[0],
                self._usage.output_tokens + counts[1],
            )


class CloudBackend:
    """One owned I/O loop; at most two independent requests; reused HTTPS connections.

    The translation owner keeps working on cache hits while this loop waits. It
    cancels obsolete generations and joins this owner before advertising shutdown.
    Error text deliberately excludes provider bodies, input text and credentials.
    """

    batch_size = 1
    max_in_flight = 2

    def __init__(
        self,
        profile: CloudProfile,
        secret: str,
        *,
        game: str = "",
        language: SourceLanguage = DEFAULT_SOURCE,
        allowance: CloudAllowance | None = None,
        transport: httpx2.AsyncBaseTransport | None = None,
        timeout: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
        access_token: Callable[[AuthWork], str] | None = None,
    ) -> None:
        self._language = language
        self._profile = profile
        self._access_token = access_token
        self._secret = secret
        self._game = " ".join(game.split())[:120]
        self._allowance = allowance or CloudAllowance()
        self._transport, self._timeout, self._clock = transport, timeout, clock
        self._blocked = (
            "" if secret else tr("API key is missing. Open Settings → Models → Cloud models.")
        )
        if secret and not valid_credential(secret):
            self._blocked = tr("Cloud credential is malformed. Reconnect in Models.")
        self.cache_identity = (
            "cloud:"
            + hashlib.sha256(
                json.dumps(
                    [
                        request_url(profile),
                        profile.model,
                        translation_instruction(language),
                        self._game,
                        language.code,
                        ENGLISH.code,
                        *decoding_identity(profile),
                    ],
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
        )
        self._loop = asyncio.new_event_loop()
        self._client: httpx2.AsyncClient | None = None
        self._pending: list[tuple[Future[BatchResult], AuthWork]] = []
        self._pending_lock = threading.Lock()
        self._auth_owner = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cloud-auth")
        self._thread = threading.Thread(target=self._serve, name="cloud-io", daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.run_until_complete(self._loop.shutdown_default_executor())
            self._loop.close()

    def submit_translation(
        self, sources: tuple[Source, ...], *, target: TargetLanguage = ENGLISH
    ) -> Future[BatchResult]:
        with self._pending_lock:
            self._pending = [(p, w) for p, w in self._pending if not p.done()]
            if len(sources) != 1 or len(self._pending) >= self.max_in_flight:
                raise RuntimeError(tr("Cloud backend accepts at most two independent requests."))
            work = AuthWork(deadline=time.monotonic() + self._timeout)
            pending = asyncio.run_coroutine_threadsafe(
                self._batch(sources[0], work, target), self._loop
            )
            self._pending.append((pending, work))
            return pending

    async def _batch(self, source: Source, work: AuthWork, target: TargetLanguage) -> BatchResult:
        try:
            async with asyncio.timeout(max(0, work.deadline - time.monotonic())):
                body = request_body(
                    self._profile, self._game, source, language=self._language, target=target
                )
                answer = await self._translate(source, work, body)
            return {source: answer}, {}
        except (TimeoutError, AuthInterrupted):
            return {}, {
                source: tr(
                    "Cloud request timed out or was cancelled. Restart translation to retry."
                )
            }
        except RequestRejected as error:
            return {}, {source: str(error)}
        except Exception:
            # Unanticipated transport exceptions can include headers. Keep the failure
            # fatal, but never let its original text/exception chain enter app logs.
            raise RuntimeError(
                tr("Cloud request failed unexpectedly. Reconnect in Models.")
            ) from None
        finally:
            work.cancelled.set()

    async def _translate(self, source: Source, work: AuthWork, body: dict[str, object]) -> str:
        if self._blocked:
            raise RequestRejected(self._blocked)
        if self._client is None:
            self._client = httpx2.AsyncClient(
                timeout=httpx2.Timeout(8.0, connect=5.0),
                follow_redirects=False,
                trust_env=False,
                limits=httpx2.Limits(max_connections=2, max_keepalive_connections=2),
                transport=self._transport,
            )
        if self._access_token:
            try:
                self._secret = await self._loop.run_in_executor(
                    self._auth_owner, self._access_token, work
                )
                if not valid_credential(self._secret):
                    raise ValueError(tr("ChatGPT returned an invalid credential. Sign in again."))
            except AuthInterrupted:
                raise
            except (ValueError, OSError) as error:
                self._blocked = str(error)
                raise RequestRejected(self._blocked) from error

        def check_dispatch() -> None:
            work.check()
            if self._blocked:
                raise RequestRejected(self._blocked)

        for attempt in range(2):
            await self._allowance.admit(self._profile, clock=self._clock, check=check_dispatch)
            try:
                async with asyncio.timeout(self._timeout):
                    async with self._client.stream(
                        "POST",
                        request_url(self._profile),
                        json=body,
                        headers=request_headers(self._profile, self._secret),
                    ) as response:
                        if (
                            self._profile.provider == CloudProvider.CHATGPT
                            and response.status_code == 200
                        ):
                            stream = ResponseStream()
                            try:
                                async for chunk in response.aiter_bytes():
                                    stream.feed(chunk)
                                english, usage = stream.finish()
                            except PlanUnavailable as error:
                                self._blocked = str(error)
                                raise
                            self._allowance.record(usage)
                            return english
                        content = bytearray()
                        async for chunk in response.aiter_bytes():
                            content.extend(chunk)
                            if len(content) > 65_536:
                                raise RequestRejected(tr("Cloud response exceeded the size limit."))
                        status = response.status_code
                        try:
                            value = json.loads(content)
                        except (ValueError, UnicodeError):
                            value = None
                        code = (
                            value.get("error", {}).get("code", "")
                            if isinstance(value, dict) and isinstance(value.get("error"), dict)
                            else ""
                        )
                        if self._profile.provider == CloudProvider.CHATGPT and (
                            status in (401, 403, 429)
                            or code == "subscription_sharing_usage_unavailable"
                        ):
                            self._blocked = tr(
                                "ChatGPT plan unavailable. Check sign-in and usage in Models. "
                                "No API-key fallback was used."
                            )
                            raise RequestRejected(self._blocked)
                        if status in (401, 402, 403) or account_error(value):
                            self._blocked = tr(
                                "Cloud account needs attention. Check its API key and balance "
                                "in Models, then restart translation."
                            )
                            raise RequestRejected(self._blocked)
                        retryable = status in (429, 502, 503, 504) or (
                            self._profile.provider == CloudProvider.CLAUDE and status == 529
                        )
                        if retryable and attempt == 0:
                            await asyncio.sleep(1.0)
                            continue
                        if status != 200:
                            raise RequestRejected(
                                tr(
                                    "Cloud request failed (HTTP {status}). Toggle "
                                    "translation to retry.",
                                    status=status,
                                )
                            )
                        self._allowance.record(response_usage(self._profile.provider, value))
                        return parse_answer(value, self._profile.provider)
            except (TimeoutError, httpx2.TimeoutException) as error:
                raise RequestRejected(
                    tr("Cloud request timed out. Restart translation to retry.")
                ) from error
            except httpx2.HTTPError as error:
                raise RequestRejected(
                    tr("Cloud connection failed. Check your internet connection.")
                ) from error
        raise RequestRejected(tr("Cloud service is temporarily unavailable."))

    def translate_with_context(
        self, text: str, context: tuple[DialogueTurn, ...], *, target: TargetLanguage = ENGLISH
    ) -> str:
        source = (text, context)
        good, bad = self.submit_translation((source,), target=target).result()
        if source in bad:
            raise RequestRejected(bad[source])
        return good[source]

    def translate(
        self, texts: tuple[str, ...], *, target: TargetLanguage = ENGLISH
    ) -> tuple[str, ...]:
        return tuple(self.translate_with_context(text, (), target=target) for text in texts)

    def cancel_pending(self) -> None:
        with self._pending_lock:
            for pending, work in self._pending:
                work.cancelled.set()
                pending.cancel()

    def classify(self, value: ClassificationInput, cancelled: threading.Event) -> Classification:
        body = prompt_body(
            self._profile,
            classification_instruction(value, language=self._language),
            value.payload(),
        )
        work = AuthWork(deadline=time.monotonic() + self._timeout)

        async def run() -> str:
            async with asyncio.timeout(self._timeout):
                return await self._translate((value.text, ()), work, body)

        pending = asyncio.run_coroutine_threadsafe(run(), self._loop)
        try:
            while not cancelled.is_set():
                try:
                    return parse_classification(pending.result(timeout=0.05))
                except TimeoutError:
                    if pending.done():
                        raise RequestRejected(tr("Classification timed out.")) from None
            raise RequestRejected(tr("Classification cancelled."))
        finally:
            work.cancelled.set()
            pending.cancel()

    async def _shutdown(self) -> None:
        current = asyncio.current_task()
        tasks = [task for task in asyncio.all_tasks() if task is not current]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def close(self) -> None:
        self.cancel_pending()
        try:
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop).result()
        finally:
            self._auth_owner.shutdown(wait=True, cancel_futures=True)
            self._secret = ""
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()
