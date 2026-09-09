"""Context-local, privacy-safe correlation for Zoo backend operations."""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from functools import wraps
from typing import Literal, ParamSpec, TypeVar
from uuid import uuid4

ApiCallSource = Literal[
    "invocation", "kcl", "rest", "file_operation", "websocket", "session", "command"
]
ApiCallOutcome = Literal["observed", "sent", "succeeded", "failed", "cancelled"]


@dataclass(frozen=True, slots=True)
class ApiCallEvent:
    """One observation, not necessarily a new backend request.

    A reused modeling connection has one backend ID and many command IDs.
    ``None`` means Zoo has not supplied an API call ID; local IDs never replace it.
    """

    operation: str
    invocation_id: str
    api_call_id: str | None
    source: ApiCallSource
    attempt: int
    outcome: ApiCallOutcome
    session_id: str | None = None
    command_id: str | None = None
    async_operation_id: str | None = None
    status_code: int | None = None


@dataclass
class _Invocation:
    operation: str
    owner: asyncio.Task | None
    invocation_id: str = field(default_factory=lambda: str(uuid4()))
    observations: dict[tuple[str | None, str | None, str | None], int] = field(
        default_factory=dict
    )
    result_failed: bool = False
    pending_requests: dict[object, tuple[int, str | None]] = field(default_factory=dict)


@dataclass
class _Attempt:
    number: int
    api_call_ids: list[str] = field(default_factory=list)


_invocation: ContextVar[_Invocation | None] = ContextVar("api_invocation", default=None)
_attempt: ContextVar[_Attempt | None] = ContextVar("api_attempt", default=None)
_buffers: ContextVar[tuple[list[ApiCallEvent], ...]] = ContextVar(
    "api_buffers", default=()
)
_async_operation: ContextVar[str | None] = ContextVar(
    "api_async_operation", default=None
)
_logger = logging.getLogger("zoo_mcp")


@contextmanager
def capture_api_call_events() -> Iterator[list[ApiCallEvent]]:
    """Collect events in this context, including nested captures and child tasks.

    The list remains available after failure or cancellation. Each nested
    collector receives each event once. Finish child tasks before leaving the
    context if a complete snapshot is needed.
    """
    events: list[ApiCallEvent] = []
    token = _buffers.set((*_buffers.get(), events))
    try:
        yield events
    finally:
        _buffers.reset(token)


@contextmanager
def api_call_attempt(number: int) -> Iterator[None]:
    token = _attempt.set(_Attempt(number))
    try:
        yield
    finally:
        _attempt.reset(token)


def attempt_api_call_ids() -> tuple[str, ...] | None:
    attempt = _attempt.get()
    return tuple(attempt.api_call_ids) if attempt and attempt.api_call_ids else None


def mark_api_call_failed() -> None:
    """Mark APIs that return a failure value instead of raising an exception."""
    invocation = _invocation.get()
    if invocation is not None:
        invocation.result_failed = True


def start_rest_request(key: object) -> None:
    """Remember only correlation fields, never the request or its contents."""
    invocation = _invocation.get()
    if invocation is not None:
        attempt = _attempt.get()
        invocation.pending_requests[key] = (
            attempt.number if attempt else 1,
            _async_operation.get(),
        )


def finish_rest_request(key: object) -> None:
    invocation = _invocation.get()
    if invocation is not None:
        invocation.pending_requests.pop(key, None)


@contextmanager
def async_operation_scope(operation_id: str) -> Iterator[None]:
    token = _async_operation.set(operation_id)
    try:
        yield
    finally:
        _async_operation.reset(token)


def record_api_call_event(
    source: ApiCallSource,
    outcome: ApiCallOutcome,
    api_call_id: str | None = None,
    *,
    session_id: str | None = None,
    command_id: str | None = None,
    async_operation_id: str | None = None,
    status_code: int | None = None,
) -> None:
    invocation = _invocation.get()
    if invocation is None:
        return
    attempt = _attempt.get()
    operation_id = async_operation_id or _async_operation.get()
    event = ApiCallEvent(
        operation=invocation.operation,
        invocation_id=invocation.invocation_id,
        api_call_id=api_call_id,
        source=source,
        attempt=attempt.number if attempt else 1,
        outcome=outcome,
        session_id=session_id,
        command_id=command_id,
        async_operation_id=operation_id,
        status_code=status_code,
    )
    if source != "invocation":
        invocation.observations[(api_call_id, session_id, operation_id)] = event.attempt
    if api_call_id and attempt and api_call_id not in attempt.api_call_ids:
        attempt.api_call_ids.append(api_call_id)
    _emit(event)


def _emit(event: ApiCallEvent) -> None:
    for events in _buffers.get():
        events.append(event)
    fields = asdict(event)
    # Include fields in the rendered message and LogRecord for both text search
    # and structured handlers. Never log request URLs, bodies, or exception text.
    _logger.info(
        "Zoo API call %s",
        " ".join(f"{key}={value}" for key, value in fields.items()),
        extra={"api_call_event": fields},
    )


_P = ParamSpec("_P")
_T = TypeVar("_T")


def api_invocation(fn: Callable[_P, Awaitable[_T]]) -> Callable[_P, Awaitable[_T]]:
    """Share a scope with same-task helpers; give concurrent calls their own IDs."""

    @wraps(fn)
    async def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        current = _invocation.get()
        owner = asyncio.current_task()
        if current is not None and current.owner is owner:
            return await fn(*args, **kwargs)
        invocation = _Invocation(getattr(fn, "__name__", type(fn).__name__), owner)
        token = _invocation.set(invocation)
        attempt_token = _attempt.set(None)
        operation_token = _async_operation.set(None)
        outcome: ApiCallOutcome = "succeeded"
        try:
            result = await fn(*args, **kwargs)
            if invocation.result_failed:
                outcome = "failed"
            return result
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except BaseException:
            outcome = "failed"
            raise
        finally:
            try:
                for attempt, operation_id in invocation.pending_requests.values():
                    _emit(
                        ApiCallEvent(
                            operation=invocation.operation,
                            invocation_id=invocation.invocation_id,
                            api_call_id=None,
                            source="rest",
                            attempt=attempt,
                            outcome="cancelled" if outcome == "cancelled" else "failed",
                            async_operation_id=operation_id,
                        )
                    )
                    invocation.observations[(None, None, operation_id)] = attempt
                observations = invocation.observations or {(None, None, None): 1}
                for (api_id, session_id, operation_id), attempt in observations.items():
                    _emit(
                        ApiCallEvent(
                            operation=invocation.operation,
                            invocation_id=invocation.invocation_id,
                            api_call_id=api_id,
                            source="invocation",
                            attempt=attempt,
                            outcome=outcome,
                            session_id=session_id,
                            async_operation_id=operation_id,
                        )
                    )
            finally:
                _async_operation.reset(operation_token)
                _attempt.reset(attempt_token)
                _invocation.reset(token)

    return wrapped
