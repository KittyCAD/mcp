"""Job operations shared by MCP and embedded tool hosts."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from zoo_mcp.execution_jobs import (
    ExecutionError,
    ExecutionJob,
    ExecutionJobs,
    ExecutionLogPage,
    ExecutionObservation,
    execution_client,
    observe_execution,
    read_execution_log,
)


async def get_job_execution(
    job_id: UUID,
    wait_seconds: Annotated[int, Field(ge=0, le=20)] = 0,
    artifact_page_token: str | None = None,
) -> ExecutionObservation:
    """Check a job's status, archived log previews, and outputs. Wait up to 20 seconds.
    Pass results.artifacts.next_page as artifact_page_token for more outputs.
    """
    if not 0 <= wait_seconds <= 20:
        raise ExecutionError("wait_seconds must be between 0 and 20.")
    async with execution_client() as client:
        job = await client.job(job_id)
        return await observe_execution(
            client, job, artifact_page_token, wait_seconds=wait_seconds
        )


async def list_job_executions(
    page_token: str | None = None,
    runtime_id: str | None = None,
    status: str | None = None,
) -> ExecutionJobs:
    """List this account's jobs across conversations, newest first. Use next_page to continue."""
    async with execution_client() as client:
        return await client.list_jobs(
            page_token=page_token, runtime_id=runtime_id, status=status
        )


async def read_job_execution_logs(
    job_id: UUID,
    stream: Literal["stdout", "stderr"] = "stdout",
    offset: Annotated[int, Field(ge=0)] = 0,
    limit: Annotated[int, Field(ge=1)] = 8192,
) -> ExecutionLogPage:
    """Read archived stdout or stderr. Check available and use next_offset to page through stored bytes."""
    if offset < 0 or limit < 1:
        raise ExecutionError(
            "Log offset must be nonnegative and limit must be positive."
        )
    async with execution_client() as client:
        return await read_execution_log(client, job_id, stream, offset, limit)


async def cancel_job_execution(job_id: UUID) -> ExecutionJob:
    """Request cancellation. Check the job's status to confirm how it ended."""
    async with execution_client() as client:
        return await client.cancel(job_id)
