"""Caller-owned native results for sketch inspection across tool calls."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal

import kcl


@dataclass(frozen=True)
class RetainedSketchExecution:
    fingerprint: str
    stage: Literal["mock_preflight", "real_execution"]
    outcome: kcl.ExecOutcome | kcl.KclError = field(repr=False)


@dataclass
class SketchExecutionUse:
    execution: RetainedSketchExecution | None = field(default=None, repr=False)
    reused: bool = False

    def retain(
        self,
        fingerprint: str,
        stage: Literal["mock_preflight", "real_execution"],
        outcome: kcl.ExecOutcome | kcl.KclError,
    ) -> None:
        if isinstance(outcome, kcl.KclError):
            # Keep native geometry, not Python frames holding the caller's context.
            outcome.__traceback__ = None
            outcome.__context__ = None
            outcome.__cause__ = None
        self.execution = RetainedSketchExecution(fingerprint, stage, outcome)


_current: ContextVar[SketchExecutionUse | None] = ContextVar(
    "sketch_execution_use", default=None
)


def current_sketch_execution() -> SketchExecutionUse | None:
    return _current.get()


@contextmanager
def reuse_sketch_execution(
    previous: RetainedSketchExecution | None = None,
) -> Iterator[SketchExecutionUse]:
    """Capture or reuse one result without putting native objects on the wire."""
    use = SketchExecutionUse(previous)
    token = _current.set(use)
    try:
        yield use
    finally:
        _current.reset(token)
