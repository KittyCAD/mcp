"""Context-local observations of Zoo backend identifiers."""

import logging
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import wraps
from typing import Literal, ParamSpec, TypeVar

ApiCallSource = Literal["kcl", "rest", "file_operation", "websocket", "session"]


@dataclass(frozen=True, slots=True)
class ApiCallEvent:
    """Identifiers supplied by Zoo, rather than locally generated trace IDs."""

    operation: str
    source: ApiCallSource
    api_call_id: str | None
    websocket_upgrade_request_id: str | None = None
    session_id: str | None = None


_operation: ContextVar[str | None] = ContextVar("api_operation", default=None)
_buffers: ContextVar[tuple[list[ApiCallEvent], ...]] = ContextVar(
    "api_buffers", default=()
)
_logger = logging.getLogger("zoo_mcp")


@contextmanager
def capture_api_call_events() -> Iterator[list[ApiCallEvent]]:
    """Collect observations, including nested captures and child tasks.

    The list remains available after failure or cancellation. Finish child tasks
    before leaving the context if a complete snapshot is needed.
    """
    events: list[ApiCallEvent] = []
    token = _buffers.set((*_buffers.get(), events))
    try:
        yield events
    finally:
        _buffers.reset(token)


def record_api_call_event(
    source: ApiCallSource,
    api_call_id: str | None = None,
    *,
    websocket_upgrade_request_id: str | None = None,
    session_id: str | None = None,
) -> None:
    operation = _operation.get()
    if operation is None or not (api_call_id or websocket_upgrade_request_id):
        return
    event = ApiCallEvent(
        operation=operation,
        source=source,
        api_call_id=api_call_id,
        websocket_upgrade_request_id=websocket_upgrade_request_id,
        session_id=session_id,
    )
    for events in _buffers.get():
        events.append(event)
    fields = asdict(event)
    _logger.info(
        "Zoo API call %s",
        " ".join(f"{key}={value}" for key, value in fields.items()),
        extra={"api_call_event": fields},
    )


_P = ParamSpec("_P")
_T = TypeVar("_T")


def api_operation(fn: Callable[_P, Awaitable[_T]]) -> Callable[_P, Awaitable[_T]]:
    """Label observations with their Python tool, without changing its result."""

    @wraps(fn)
    async def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        token = _operation.set(getattr(fn, "__name__", type(fn).__name__))
        try:
            return await fn(*args, **kwargs)
        finally:
            _operation.reset(token)

    return wrapped
