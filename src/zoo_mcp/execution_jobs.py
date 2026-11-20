"""Authenticated access to Zoo's durable execution-job API."""

import asyncio
import base64
import codecs
import hashlib
import os
import tempfile
import weakref
import zlib
from collections.abc import AsyncGenerator
from contextlib import aclosing, asynccontextmanager, suppress
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import Annotated, BinaryIO, Literal
from uuid import UUID

import httpx
from kittycad import AsyncExecutorAPI, AsyncKittyCAD
from kittycad.models import (
    ExecutionJobCreateRequest,
    ExecutionLogStream,
    Uuid,
)
from kittycad.models.error import Error as ZooAPIError
from kittycad.pagination import create_async_page_iterator
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic_core import to_jsonable_python

from zoo_mcp import ZooMCPException, ctx
from zoo_mcp.client_config import get_client_config
from zoo_mcp.utils.async_utils import run_blocking

MAX_EXECUTION_INPUTS = 256
MAX_LOG_PAGE_BYTES = 65536
MAX_ERROR_BODY_BYTES = 16 * 1024
ERROR_BODY_TIMEOUT = 2.0
TERMINAL_STATUSES = {"succeeded", "failed", "timed_out", "canceled"}
_download_slots: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, asyncio.Semaphore
] = weakref.WeakKeyDictionary()


def _download_slot() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    if loop not in _download_slots:
        _download_slots[loop] = asyncio.Semaphore(2)
    return _download_slots[loop]


class ExecutionError(ZooMCPException):
    """A safe error suitable for a model-facing tool result."""

    def __init__(self, message: str, code: str = "execution_error") -> None:
        super().__init__(message)
        self.code = code


class ExecutionAPIError(ExecutionError):
    """Expose the API's public error fields to tool callers."""

    def __init__(self, status: int, detail: ZooAPIError | None) -> None:
        self.status = status
        code = f"execution_http_{status}"
        message = f"The execution API returned HTTP {status}."
        if detail:
            reference = f"request ID {detail.request_id}"
            if detail.error_code:
                code = detail.error_code
                reference = f"{code}, {reference}"
            message = f"{detail.message} ({reference})"
        super().__init__(message, code)

    @classmethod
    async def raise_for_status(cls, response: httpx.Response) -> None:
        if response.is_success:
            return
        detail = None
        if (
            response.headers.get("content-type", "").partition(";")[0].lower()
            == "application/json"
        ):
            # Preserve the HTTP status even when the optional diagnostic stalls
            # or the gateway sends an incomplete or malformed body.
            with suppress(
                ValueError, httpx.HTTPError, httpx.StreamError, TimeoutError, zlib.error
            ):
                async with asyncio.timeout(ERROR_BODY_TIMEOUT):
                    if (body := await _read_error_body(response)) is not None:
                        detail = ZooAPIError.model_validate_json(body, extra="ignore")
        raise cls(response.status_code, detail) from None


async def _read_error_body(response: httpx.Response) -> bytes | bytearray | None:
    """Bound both wire bytes and decompressed JSON before parsing the diagnostic."""
    length_header = response.headers.get("content-length")
    length = int(length_header) if length_header is not None else None
    if length is not None and not 0 <= length <= MAX_ERROR_BODY_BYTES:
        return None
    if response.is_stream_consumed:
        return (
            response.content if len(response.content) <= MAX_ERROR_BODY_BYTES else None
        )

    encoding = response.headers.get("content-encoding", "identity").lower().strip()
    if encoding not in {"identity", "gzip", "deflate"}:
        return None
    body = bytearray()
    async for chunk in response.aiter_raw():
        if len(body) + len(chunk) > MAX_ERROR_BODY_BYTES:
            return None
        body.extend(chunk)
        if len(body) == MAX_ERROR_BODY_BYTES:
            if length != len(body):
                return None
            break
    if encoding == "identity":
        return body
    window_bits = zlib.MAX_WBITS + 16 if encoding == "gzip" else zlib.MAX_WBITS
    decoder = zlib.decompressobj(window_bits)
    decoded = decoder.decompress(body, MAX_ERROR_BODY_BYTES + 1)
    if decoder.eof and not decoder.unused_data and len(decoded) <= MAX_ERROR_BODY_BYTES:
        return decoded
    return None


def workspace_path(path: str, *, glob: bool = False) -> str:
    """Accept normalized relative paths using the execution API's convention."""
    parts = PurePosixPath(path).parts
    if (
        not parts
        or path.startswith("/")
        or PurePosixPath(path).as_posix() != path
        or any(part in {".", ".."} for part in parts)
        or any(ord(char) < 32 or ord(char) == 127 for char in path)
        or "\\" in path
        or (not glob and any(char in path for char in "*?[]"))
    ):
        raise ExecutionError(
            "File paths must be normalized relative paths without traversal, "
            "backslashes, or control characters.",
            "invalid_execution_path",
        )
    return path


class ExecutionJob(BaseModel):
    id: UUID
    status: str
    runtime_id: str
    compute_profile_id: str
    timeout_seconds: int
    state_version: int = 0
    created_at: str | None = None
    started_at: str | None = None
    execution_ended_at: str | None = None
    completed_at: str | None = None


class ExecutionJobs(BaseModel):
    items: list[ExecutionJob]
    next_page: str | None = None


class ExecutionArtifact(BaseModel):
    id: UUID
    path: str
    byte_length: Annotated[int, Field(ge=0)]
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class ArtifactPage(BaseModel):
    items: list[ExecutionArtifact]
    next_page: str | None = None


class ExecutionArtifacts(BaseModel):
    state: str
    failure_code: str | None = None
    artifacts: ArtifactPage


class ExecutionLog(BaseModel):
    model_config = ConfigDict(frozen=True)
    state: str
    available: bool
    stream: Literal["stdout", "stderr"]
    data_base64: str
    offset: int
    next_offset: int
    stored_bytes: int | None = None
    complete: bool
    truncated: bool
    failure_code: str | None = None

    @cached_property
    def data(self) -> bytes:
        data = base64.b64decode(self.data_base64, validate=True)
        if self.next_offset != self.offset + len(data):
            raise ExecutionError("The execution API returned an invalid log page.")
        return data


class UploadedInput(BaseModel):
    id: UUID
    byte_length: int
    sha256: str


class _UploadBody:
    """Rewind the snapshot when HTTPX resends an upload after an auth challenge."""

    def __init__(self, snapshot: BinaryIO) -> None:
        self.snapshot = snapshot
        self._started = False
        self.size_and_digest: tuple[int, str] | None = None

    async def __aiter__(self) -> AsyncGenerator[bytes]:
        if self._started:
            self.snapshot.seek(0)
        self._started = True
        self.size_and_digest = None
        digest = hashlib.sha256()
        length = 0
        while chunk := await run_blocking(self.snapshot.read, 65536):
            length += len(chunk)
            digest.update(chunk)
            yield chunk
        self.size_and_digest = length, digest.hexdigest()


class ExecutionInput(BaseModel):
    """An uploaded input and its destination in the remote workspace."""

    model_config = ConfigDict(extra="forbid")
    artifact_id: UUID
    path: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return workspace_path(value)


class ExecutionOutput(BaseModel):
    """One relative output path or glob to retain after execution."""

    model_config = ConfigDict(extra="forbid")
    path: str
    required: bool = True

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return workspace_path(value, glob=True)


class ExecutionSubmission(BaseModel):
    """A runtime-neutral job request; the API admits runtime and compute IDs."""

    model_config = ConfigDict(extra="forbid")
    runtime_id: str = Field(min_length=1)
    compute_profile_id: str = Field(min_length=1)
    command: list[str] = Field(
        min_length=1,
        max_length=128,
        description="Executable and arguments. Invoke a shell explicitly for shell syntax.",
    )
    inputs: list[ExecutionInput] = Field(
        default_factory=list, max_length=MAX_EXECUTION_INPUTS
    )
    outputs: list[ExecutionOutput] = Field(default_factory=list, max_length=256)
    environment: dict[str, str] = Field(
        default_factory=dict,
        description="Non-secret environment variables for the job.",
    )
    timeout_seconds: int = Field(default=300, ge=1, le=86400)

    @model_validator(mode="after")
    def validate_manifest(self) -> "ExecutionSubmission":
        if any("\0" in argument for argument in self.command) or not self.command[0]:
            raise ValueError(
                "command requires an executable and cannot contain NUL bytes."
            )
        validate_input_paths([item.path for item in self.inputs])
        return self


def validate_input_paths(paths: list[str]) -> None:
    """Validate normalized workspace paths and reject file/directory collisions."""
    if len(paths) > MAX_EXECUTION_INPUTS:
        raise ExecutionError(f"A job accepts at most {MAX_EXECUTION_INPUTS} inputs.")
    destinations = {workspace_path(path) for path in paths}
    if len(paths) != len(destinations) or any(
        parent.as_posix() in destinations
        for path in paths
        for parent in PurePosixPath(path).parents
    ):
        raise ExecutionError(
            "Input destinations must be unique and must not overlap.",
            "invalid_execution_path",
        )


class _ExecutionTransport(httpx.AsyncBaseTransport):
    """Inspect the final authenticated response before the SDK can buffer it."""

    def __init__(self, caller: httpx.AsyncClient) -> None:
        self.caller = caller

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self.caller.send(request, stream=True, follow_redirects=False)
        if not response.is_success:
            async with aclosing(response):
                await ExecutionAPIError.raise_for_status(response)
        return response


class _ExecutionHTTPClient(httpx.AsyncClient):
    """Use the caller's request defaults, authentication, hooks and connection pool."""

    def __init__(self, caller: httpx.AsyncClient) -> None:
        super().__init__(transport=_ExecutionTransport(caller), timeout=caller.timeout)
        self.caller = caller

    def build_request(
        self, method: str, url: httpx.URL | str, **kwargs
    ) -> httpx.Request:
        return self.caller.build_request(method, url, **kwargs)


class ExecutionClient:
    """Runtime-neutral operations on the authenticated Zoo execution-jobs API."""

    def __init__(self, client: AsyncKittyCAD) -> None:
        self.http = client.get_http_client()
        self.base_url = client.base_url.rstrip("/")
        self.headers = client.get_headers()
        self._sdk = client
        self._execution_http = _ExecutionHTTPClient(self.http)

    @property
    def _executor(self) -> AsyncExecutorAPI:
        # Pick up credentials refreshed by an embedded caller between requests.
        config = self._sdk.with_base_url(self._sdk.base_url.rstrip("/"))
        config.http_client = self._execution_http
        return AsyncExecutorAPI(config)

    async def upload(self, path: Path) -> UploadedInput:
        """Stream a private snapshot; check that the API retained the same bytes."""
        with path.open("rb") as source:
            body = _UploadBody(source)
            result = await self._executor.upload_execution_input(body, timeout=300)
        uploaded = UploadedInput.model_validate(to_jsonable_python(result))
        if body.size_and_digest != (uploaded.byte_length, uploaded.sha256):
            raise ExecutionError(
                "The uploaded input failed byte-integrity verification."
            )
        return uploaded

    async def prepare_submission(
        self, submission: ExecutionSubmission, files: list[tuple[str, Path]]
    ) -> ExecutionSubmission:
        """Upload snapshots and return the exact request a host can save for replay."""
        validate_input_paths(
            [item.path for item in submission.inputs]
            + [destination for destination, _ in files]
        )
        inputs = list(submission.inputs)
        for destination, path in files:
            uploaded = await self.upload(path)
            inputs.append(ExecutionInput(path=destination, artifact_id=uploaded.id))
        return submission.model_copy(update={"inputs": inputs})

    async def submit_staged(
        self, submission: ExecutionSubmission, files: list[tuple[str, Path]], key: str
    ) -> ExecutionJob:
        """Prepare and submit in one call. Replayable hosts save prepare_submission first."""
        return await self.submit(await self.prepare_submission(submission, files), key)

    async def submit(
        self, request: ExecutionSubmission | dict, key: str
    ) -> ExecutionJob:
        """Submit a frozen API request with its caller-owned idempotency key.

        The caller selects runtime_id, compute_profile_id, command, inputs,
        outputs, environment, and timeout_seconds. No runtime defaults or
        command rewriting are applied by this transport.
        """
        # Repeating this exact request is safe even when the first response was
        # lost. Never regenerate upload IDs or the key inside this retry loop.
        if isinstance(request, ExecutionSubmission):
            request = request.model_dump(mode="json")
        body = ExecutionJobCreateRequest.model_validate(request)
        detail = ""
        for _ in range(2):
            try:
                result = await self._executor.create_execution_job(
                    idempotency_key=key, body=body
                )
                return ExecutionJob.model_validate(to_jsonable_python(result))
            except httpx.TransportError:
                detail = ""
                continue
            except ExecutionAPIError as error:
                if error.status < 500:
                    raise
                detail = f" {error}"
                continue
            except ValueError:
                detail = " The API returned an invalid job."
                break
        raise ExecutionError(
            "Job submission may have succeeded. Do not run it again. "
            "List existing executions to recover the job." + detail,
            "execution_submission_unknown",
        )

    async def job(self, job_id: UUID) -> ExecutionJob:
        job = await self._executor.get_execution_job(Uuid(str(job_id)))
        return ExecutionJob.model_validate(to_jsonable_python(job))

    async def list_jobs(
        self,
        *,
        runtime_id: str | None = None,
        status: str | None = None,
        page_token: str | None = None,
        limit: int = 25,
    ) -> ExecutionJobs:
        """List owner-scoped jobs, optionally restricted to a runtime or status."""
        page = await self._executor.list_execution_jobs_page(
            limit=limit, runtime_id=runtime_id, status=status, page_token=page_token
        )
        return ExecutionJobs.model_validate(to_jsonable_python(page))

    async def cancel(self, job_id: UUID) -> ExecutionJob:
        """Request cancellation and return the observed, possibly pending state."""
        job = await self._executor.cancel_execution_job(Uuid(str(job_id)))
        return ExecutionJob.model_validate(to_jsonable_python(job))

    async def wait(self, job: ExecutionJob, seconds: int) -> ExecutionJob:
        """The caller's wait deadline includes status requests, not only sleeps."""
        try:
            async with asyncio.timeout(seconds):
                delay = 0.5
                while job.status not in TERMINAL_STATUSES:
                    await asyncio.sleep(delay)
                    job = await self.job(job.id)
                    delay = min(delay * 1.5, 3)
        except TimeoutError:
            pass
        return job

    async def artifacts(
        self, job_id: UUID, page_token: str | None = None
    ) -> ExecutionArtifacts:
        page = await self._executor.list_execution_job_artifacts(
            Uuid(str(job_id)), limit=25, page_token=page_token
        )
        return ExecutionArtifacts.model_validate(page, from_attributes=True)

    async def artifact(self, job_id: UUID, artifact_id: UUID) -> ExecutionArtifact:
        async def fetch_page(page_token: str | None = None) -> ArtifactPage:
            return (await self.artifacts(job_id, page_token)).artifacts

        async for artifact in create_async_page_iterator(
            fetch_page, {}, ExecutionArtifact
        ):
            if artifact.id == artifact_id:
                workspace_path(artifact.path, glob=True)
                return artifact
        raise ExecutionError(
            "The artifact is unavailable in this job's retained results.",
            "execution_artifact_unavailable",
        )

    async def logs(
        self, job_id: UUID, stream: Literal["stdout", "stderr"], offset: int, limit: int
    ) -> ExecutionLog:
        limit = min(limit, MAX_LOG_PAGE_BYTES)
        result = await self._executor.get_execution_job_logs(
            Uuid(str(job_id)),
            stream=ExecutionLogStream(stream),
            offset=offset,
            limit=limit,
        )
        page = ExecutionLog.model_validate(result, from_attributes=True)
        if page.offset != offset or page.stream != stream or len(page.data) > limit:
            raise ExecutionError("The execution API returned an invalid log cursor.")
        return page

    @asynccontextmanager
    async def open_artifact(
        self, job_id: UUID, artifact_id: UUID
    ) -> AsyncGenerator[tuple[ExecutionArtifact, Path], None]:
        """Yield the metadata and verified local bytes for one retained output."""
        artifact = await self.artifact(job_id, artifact_id)
        async with self.download(job_id, artifact) as path:
            yield artifact, path

    @asynccontextmanager
    async def download(
        self, job_id: UUID, artifact: ExecutionArtifact
    ) -> AsyncGenerator[Path, None]:
        async with _download_slot():
            with tempfile.TemporaryDirectory(prefix="execution-result-") as directory:
                path = Path(directory) / "artifact"
                digest = hashlib.sha256()
                length = 0
                async with (
                    asyncio.timeout(300),
                    self._executor.download_execution_job_artifact(
                        Uuid(str(job_id)), Uuid(str(artifact.id))
                    ) as response,
                ):
                    with path.open("wb") as target:
                        async for chunk in response.aiter_bytes(65536):
                            length += len(chunk)
                            if length > artifact.byte_length:
                                raise ExecutionError(
                                    "The artifact exceeds its declared size."
                                )
                            digest.update(chunk)
                            await run_blocking(target.write, chunk)
                if (
                    length != artifact.byte_length
                    or digest.hexdigest() != artifact.sha256
                ):
                    raise ExecutionError(
                        "The artifact failed byte-integrity verification.",
                        "execution_artifact_integrity_failed",
                    )
                yield path


@asynccontextmanager
async def execution_client() -> AsyncGenerator[ExecutionClient, None]:
    """Use the current caller's Zoo configuration, including host and headers."""
    try:
        config = get_client_config()
        token = config.token.removeprefix("Bearer ").strip()
        if not token or token.lower() == "bearer":
            raise ExecutionError(
                "Zoo authentication is required for execution.",
                "execution_auth_required",
            )
        async with AsyncKittyCAD(
            token=token,
            base_url=config.base_url.rstrip("/"),
            headers=dict(config.headers or {}),
            verify_ssl=ctx,
            timeout=30,
        ) as client:
            yield ExecutionClient(client)
    except (httpx.HTTPError, TimeoutError):
        raise ExecutionError(
            "The execution API could not be reached. Retry reads later.",
            "execution_transport_error",
        ) from None
    except (ValueError, OSError):
        raise ExecutionError(
            "Execution requires valid Zoo client configuration and available input files.",
            "invalid_execution_data",
        ) from None


class ExecutionLogPage(BaseModel):
    stream: str
    text: str
    offset: int
    next_offset: int
    stored_bytes: int | None
    available: bool
    complete: bool
    truncated: bool
    failure_code: str | None
    utf8_replaced: bool
    data_base64: str | None = Field(
        default=None,
        description="Exact bytes of this page when its text alone cannot reproduce them.",
    )


class ExecutionObservation(BaseModel):
    """A durable job and partial, paginated reads of its retained outputs."""

    ok: bool
    msg: str
    error_code: str | None = None
    job_id: UUID
    job: ExecutionJob
    logs: list[ExecutionLogPage] | None = None
    results: ExecutionArtifacts | None = None


async def read_execution_log(
    client: ExecutionClient,
    job_id: UUID,
    stream: Literal["stdout", "stderr"],
    offset: int = 0,
    limit: int = 8192,
) -> ExecutionLogPage:
    page = await client.logs(job_id, stream, offset, limit)
    data = page.data
    # A UTF-8 scalar may start up to three bytes before this byte cursor.
    # Emit each scalar on the page that completes it, even with a one-byte limit.
    prefix = b""
    if offset and data:
        size = min(3, offset)
        previous = await client.logs(job_id, stream, offset - size, size)
        if previous.next_offset != offset:
            raise ExecutionError("The execution API returned incomplete log context.")
        context = previous.data
        _, consumed = codecs.utf_8_decode(context, "replace", False)
        prefix = context[consumed:]
    window = prefix + data
    final = page.stored_bytes is None or page.next_offset >= page.stored_bytes
    try:
        text, consumed = codecs.utf_8_decode(window, "strict", final)
        replaced = False
    except UnicodeDecodeError:
        text, consumed = codecs.utf_8_decode(window, "replace", final)
        replaced = True
    return ExecutionLogPage(
        **page.model_dump(exclude={"data_base64"}),
        text=text,
        utf8_replaced=replaced,
        data_base64=page.data_base64
        if replaced or prefix or consumed < len(window)
        else None,
    )


async def observe_execution(
    client: ExecutionClient,
    job: ExecutionJob,
    page_token: str | None = None,
    *,
    wait_seconds: int = 0,
) -> ExecutionObservation:
    if wait_seconds:
        try:
            job = await client.wait(job, wait_seconds)
        except (ExecutionError, httpx.HTTPError, ValueError, TimeoutError):
            return ExecutionObservation(
                ok=job.status not in {"failed", "timed_out", "canceled"},
                msg=f"Job {job.id} could not be refreshed. Use get_job_execution to check it again.",
                job_id=job.id,
                job=job,
                error_code="execution_observation_unavailable",
            )
    result = ExecutionObservation(
        ok=job.status not in {"failed", "timed_out", "canceled"},
        msg=f"Execution job {job.id}: {job.status}. "
        "Use this job ID for subsequent reads; do not launch it again to retrieve results.",
        job_id=job.id,
        job=job,
    )
    artifacts, stdout, stderr = await asyncio.gather(
        client.artifacts(job.id, page_token),
        read_execution_log(client, job.id, "stdout"),
        read_execution_log(client, job.id, "stderr"),
        return_exceptions=True,
    )
    result.results = artifacts if isinstance(artifacts, ExecutionArtifacts) else None
    result.logs = [
        page for page in (stdout, stderr) if isinstance(page, ExecutionLogPage)
    ]
    if result.results is None or len(result.logs) != 2:
        result.msg += " Some result reads failed; check this job again later."
        result.error_code = "execution_results_unavailable"
    return result


def read_text_page(path: Path, offset: int, limit: int = 65536) -> tuple[str, int]:
    """Read one UTF-8 page without loading an entire artifact into memory."""
    if offset < 0 or not 1 <= limit <= 65536:
        raise ExecutionError("Invalid text offset or limit (1..65536 bytes).")
    if offset > path.stat().st_size:
        raise ExecutionError("offset must be within the file.")
    with path.open("rb") as source:
        source.seek(offset)
        data = source.read(limit + 4)
    if data and data[0] & 0xC0 == 0x80:
        raise ExecutionError("offset must be a UTF-8 character boundary.")
    end = min(limit, len(data))
    while end < len(data) and data[end] & 0xC0 == 0x80:
        end -= 1
    try:
        text = data[:end].decode("utf-8")
    except UnicodeDecodeError:
        raise ExecutionError("This output is not UTF-8 text; use download.") from None
    return text, offset + end


@dataclass(frozen=True)
class ArtifactImage:
    mime_type: str
    data: str  # base64-encoded image for inline tool output


_IMAGE_FORMATS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _artifact_image_mime(artifact_name: str) -> str | None:
    return _IMAGE_FORMATS.get(PurePosixPath(artifact_name).suffix.lower())


def can_read_artifact_inline(artifact: ExecutionArtifact, offset: int) -> bool:
    """Check the host's raw image byte budget before fetching a retained output."""
    if _artifact_image_mime(artifact.path) is None:
        return True
    if offset:
        raise ExecutionError("Image reads do not support offsets.")
    configured = os.getenv("ZOO_MCP_MAX_INLINE_IMAGE_BYTES")
    if configured is None:
        return True
    if not configured.isascii() or not configured.isdecimal():
        raise ExecutionError(
            "ZOO_MCP_MAX_INLINE_IMAGE_BYTES must be a nonnegative integer."
        )
    return artifact.byte_length <= int(configured)


def read_artifact_content(
    path: Path, artifact_name: str, offset: int = 0
) -> ArtifactImage | tuple[str, int]:
    """Read a verified file as an inline image or a page of UTF-8 text."""
    mime = _artifact_image_mime(artifact_name)
    if mime is None:
        return read_text_page(path, offset)
    if offset:
        raise ExecutionError("Image reads do not support offsets.")
    return ArtifactImage(mime, base64.b64encode(path.read_bytes()).decode("ascii"))
