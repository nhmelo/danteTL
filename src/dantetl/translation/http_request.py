"""Deadline and cancellation for decoded HTTP responses."""

import asyncio
import threading
import time

import httpx2


class ResponseTooLarge(ValueError):
    """The decoded response crossed the caller's byte limit."""


async def read_response(
    method: str,
    url: str,
    *,
    cancel: threading.Event,
    deadline: float,
    max_bytes: int,
    transport: httpx2.AsyncBaseTransport | None,
    skip_error_body: bool = False,
    read_timeout: float = 8.0,
    **kwargs,
) -> tuple[int, bytes]:
    def check() -> None:
        if cancel.is_set():
            raise InterruptedError
        if time.monotonic() >= deadline:
            raise TimeoutError

    check()
    owner = asyncio.current_task()
    assert owner is not None

    async def watch() -> None:
        while not cancel.is_set():
            await asyncio.sleep(0.025)
        owner.cancel()

    watcher = asyncio.create_task(watch())
    try:
        async with asyncio.timeout(max(0, deadline - time.monotonic())):
            async with (
                httpx2.AsyncClient(
                    timeout=read_timeout,
                    follow_redirects=False,
                    trust_env=False,
                    transport=transport,
                ) as client,
                client.stream(method, url, **kwargs) as response,
            ):
                check()
                if skip_error_body and response.status_code != 200:
                    return response.status_code, b""
                data = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=16 * 1024):
                    check()
                    if len(data) + len(chunk) > max_bytes:
                        raise ResponseTooLarge
                    data.extend(chunk)
                check()
                return response.status_code, bytes(data)
    except asyncio.CancelledError:
        raise InterruptedError from None
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
