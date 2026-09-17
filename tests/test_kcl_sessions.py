"""Lifecycle and partial-result regressions for the local KCL bindings."""

from unittest.mock import AsyncMock, Mock

import pytest

from zoo_mcp import zoo_tools


class Session:
    def __init__(self, *, error=None, retryable=False):
        self.outcome = Mock()
        self.outcome.raise_for_error.side_effect = error
        self.outcome.is_retryable.return_value = retryable
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True


@pytest.mark.asyncio
async def test_local_operation_executes_once_and_closes(monkeypatch):
    session = Session()
    execute = AsyncMock(return_value=session)
    monkeypatch.setattr(zoo_tools.kcl, "execute_code", execute)
    operation = AsyncMock(return_value="exported")
    assert await zoo_tools._run_kcl_operation("code", None, operation) == "exported"
    execute.assert_awaited_once_with("code")
    operation.assert_awaited_once_with(session)
    assert session.closed


@pytest.mark.asyncio
async def test_partial_constraint_results_survive_execution_failure(monkeypatch):
    session = Session(error=ValueError("invalid KCL"))
    monkeypatch.setattr(zoo_tools.kcl, "execute_code", AsyncMock(return_value=session))
    outcome = await zoo_tools._run_kcl_operation(
        "code", None, zoo_tools._session_outcome, allow_partial=True
    )
    assert outcome is session.outcome
    outcome.raise_for_error.assert_not_called()
    assert session.closed


@pytest.mark.asyncio
async def test_complete_model_required_before_export(monkeypatch):
    session = Session(error=ValueError("invalid KCL"))
    monkeypatch.setattr(zoo_tools.kcl, "execute_code", AsyncMock(return_value=session))
    operation = AsyncMock()
    with pytest.raises(ValueError, match="invalid KCL"):
        await zoo_tools._run_kcl_operation("code", None, operation)
    operation.assert_not_awaited()
    assert session.closed


@pytest.mark.asyncio
async def test_operation_failure_closes_session(monkeypatch):
    session = Session()
    monkeypatch.setattr(zoo_tools.kcl, "execute_code", AsyncMock(return_value=session))
    with pytest.raises(ValueError, match="export failed"):
        await zoo_tools._run_kcl_operation(
            "code", None, AsyncMock(side_effect=ValueError("export failed"))
        )
    assert session.closed


@pytest.mark.asyncio
async def test_retry_closes_previous_session_before_executing_again(monkeypatch):
    class RetryableError(Exception):
        def is_retryable(self):
            return True

    failed = Session(error=RetryableError(), retryable=True)
    succeeded = Session()
    sessions = iter([failed, succeeded])

    async def execute(code):
        current = next(sessions)
        if current is succeeded:
            assert failed.closed
        return current

    monkeypatch.setattr(zoo_tools.kcl, "execute_code", execute)
    assert (
        await zoo_tools._run_kcl_operation(
            "code", None, zoo_tools._session_outcome, allow_partial=True
        )
        is succeeded.outcome
    )
    assert succeeded.closed
