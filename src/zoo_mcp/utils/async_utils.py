"""Async helpers for blocking work whose resources outlive cancellation."""

import asyncio
from collections.abc import Callable
from contextlib import suppress
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


async def _finish_cancelled_await(task: asyncio.Task[T]) -> None:
    """Wait for a shielded worker despite repeated cancellation requests."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except BaseException:
            break
    if task.done() and not task.cancelled():
        # Cancellation remains authoritative. Retrieve a simultaneous worker
        # failure to prevent an unhandled-task warning.
        with suppress(BaseException):
            task.result()


async def run_blocking(
    function: Callable[P, T],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Run blocking work without abandoning its thread on cancellation.

    Cancelling ``asyncio.to_thread`` cancels only the await; its worker keeps
    running. Shield the worker and settle it before propagating cancellation so
    callers can safely release locks, semaphores, files, and memory afterward.
    """
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await _finish_cancelled_await(task)
        raise
