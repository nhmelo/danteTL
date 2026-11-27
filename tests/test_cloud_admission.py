"""Dispatch budgets survive cancelled/expired waits and backend replacement."""

import asyncio
import threading
from dataclasses import replace

import httpx2
import pytest

from dantetl.translation.backend import RequestRejected
from dantetl.translation.cloud.backend import CloudAllowance, CloudBackend
from dantetl.translation.cloud.values import CloudProfile, CloudProvider


@pytest.mark.parametrize("cancel", [True, False])
def test_unsent_wait_does_not_consume_budget_or_delay_replacement(cancel):
    profile = replace(
        CloudProfile.new(CloudProvider.OPENAI), request_limit=2, requests_per_minute=10
    )
    allowance, waiting = CloudAllowance(), threading.Event()
    now = 0.0
    calls = []

    def clock():
        waiting.set()
        return now

    def http(request):
        calls.append(request)
        return httpx2.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": "Settings"}}]}
        )

    def create():
        return CloudBackend(
            profile,
            "fixture",
            allowance=allowance,
            clock=clock,
            transport=httpx2.MockTransport(http),
            timeout=0.15 if not cancel else 3,
        )

    source = ("設定", ())
    backend = create()
    try:
        assert backend.submit_translation((source,)).result(timeout=2)[0][source] == "Settings"
        waiting.clear()
        pending = backend.submit_translation((("新しい文字", ()),))
        assert waiting.wait(2)
        assert allowance.snapshot().requests == 1
        if cancel:
            assert pending.cancel()
        else:
            translated, errors = pending.result(timeout=2)
            assert not translated and errors
    finally:
        backend.close()
    assert len(calls) == 1 and allowance.snapshot().requests == 1
    now = 6.0
    replacement = create()
    try:
        assert replacement.submit_translation((source,)).result(timeout=2)[0][source] == "Settings"
        assert len(calls) == 2 and allowance.snapshot().requests == 2
        assert replacement.submit_translation((source,)).result(timeout=2)[1]
        assert len(calls) == 2
    finally:
        replacement.close()


def test_fifo_admission_skips_cancelled_waiter_without_reserving_future_slots():
    async def replay():
        profile = replace(CloudProfile.new(CloudProvider.OPENAI), requests_per_minute=10)
        allowance, order = CloudAllowance(), []
        now = 0.0
        await allowance.admit(profile, clock=lambda: now)

        async def send(index):
            await allowance.admit(profile, clock=lambda: now)
            order.append(index)

        tasks = [asyncio.create_task(send(i)) for i in range(3)]
        await asyncio.sleep(0)
        tasks[1].cancel()
        await asyncio.gather(tasks[1], return_exceptions=True)
        now = 6.0
        await asyncio.wait_for(tasks[0], 1)
        assert order == [0] and not tasks[2].done()
        now = 12.0
        await asyncio.wait_for(tasks[2], 1)
        assert order == [0, 2] and allowance.snapshot().requests == 3

    asyncio.run(replay())


def test_wait_queue_is_bounded_and_cancelled_entries_release_capacity():
    async def replay():
        profile = replace(CloudProfile.new(CloudProvider.OPENAI), requests_per_minute=10)
        allowance = CloudAllowance()
        await allowance.admit(profile, clock=lambda: 0)
        tasks = [asyncio.create_task(allowance.admit(profile, clock=lambda: 0)) for _ in range(16)]
        await asyncio.sleep(0)
        with pytest.raises(RequestRejected, match="queue is full"):
            await allowance.admit(profile, clock=lambda: 0)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert allowance.snapshot().requests == 1
        await allowance.admit(profile, clock=lambda: 6)
        assert allowance.snapshot().requests == 2

    asyncio.run(replay())


def test_dispatch_guard_failure_does_not_consume_allowance():
    def blocked():
        raise RequestRejected("Account unavailable")

    allowance = CloudAllowance()
    with pytest.raises(RequestRejected, match="Account unavailable"):
        asyncio.run(allowance.admit(CloudProfile.new(CloudProvider.OPENAI), check=blocked))
    assert allowance.snapshot().requests == 0
