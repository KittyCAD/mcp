"""Tests for cancellation-safe blocking work."""

import asyncio
import threading

import pytest

from zoo_mcp.utils.async_utils import run_blocking


@pytest.mark.asyncio
async def test_run_blocking_uses_a_worker_thread() -> None:
    event_loop_thread = threading.get_ident()

    assert await run_blocking(threading.get_ident) != event_loop_thread


@pytest.mark.asyncio
async def test_run_blocking_settles_before_repeated_cancellation_propagates() -> None:
    started = threading.Event()
    finish = threading.Event()

    def blocked_work() -> int:
        started.set()
        assert finish.wait(timeout=1)
        return 42

    task = asyncio.create_task(run_blocking(blocked_work))
    assert await asyncio.to_thread(started.wait, 0.5)

    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_run_blocking_retrieves_worker_failure_after_cancellation() -> None:
    started = threading.Event()
    finish = threading.Event()

    def failing_work() -> None:
        started.set()
        assert finish.wait(timeout=1)
        raise RuntimeError("worker failed while cancellation was pending")

    task = asyncio.create_task(run_blocking(failing_work))
    assert await asyncio.to_thread(started.wait, 0.5)

    task.cancel()
    await asyncio.sleep(0)
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
