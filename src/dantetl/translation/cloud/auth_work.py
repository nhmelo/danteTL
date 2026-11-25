"""One cancellable, deadline-bound authentication operation, including streamed HTTP."""

import asyncio
import threading
import time

import httpx2

from dantetl.i18n import tr
from dantetl.translation.http_request import ResponseTooLarge, read_response

AUTH_RESPONSE_LIMIT = 128 * 1024


class AuthInterrupted(ValueError):
    """Safe diagnostic; interruption never invalidates saved credentials."""


class AuthWork:
    def __init__(
        self,
        cancelled: threading.Event | None = None,
        *,
        deadline: float | None = None,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self.cancelled = cancelled if cancelled is not None else threading.Event()
        self.deadline = deadline if deadline is not None else time.monotonic() + 30
        self.transport = transport

    def check(self) -> None:
        if self.cancelled.is_set():
            raise AuthInterrupted(tr("ChatGPT authentication cancelled."))
        if time.monotonic() >= self.deadline:
            raise AuthInterrupted(tr("ChatGPT authentication timed out."))

    def request(
        self, method: str, url: str, *, max_response_bytes: int = AUTH_RESPONSE_LIMIT, **kwargs
    ) -> tuple[int, bytes]:
        """Called on the auth/setup owner. Async I/O allows cancellation during headers/body."""
        try:
            return asyncio.run(
                read_response(
                    method,
                    url,
                    cancel=self.cancelled,
                    deadline=self.deadline,
                    max_bytes=max_response_bytes,
                    transport=self.transport,
                    **kwargs,
                )
            )
        except InterruptedError:
            raise AuthInterrupted(tr("ChatGPT authentication cancelled.")) from None
        except TimeoutError:
            raise AuthInterrupted(tr("ChatGPT authentication timed out.")) from None
        except ResponseTooLarge:
            raise AuthInterrupted(
                tr(
                    "ChatGPT setup response exceeds its size limit ({value:,} KiB).",
                    value=max_response_bytes // 1024,
                )
            ) from None
