"""The shared execution transport accepts solver jobs without Python policy."""

import asyncio
import base64
import hashlib
import json
from uuid import uuid4

import httpx
import pytest
from kittycad import AsyncKittyCAD
from kittycad.exceptions import KittyCADClientError
from kittycad.models import Uuid

from zoo_mcp.execution_jobs import (
    ExecutionArtifact,
    ExecutionClient,
    ExecutionError,
    ExecutionJob,
    ExecutionSubmission,
    observe_execution,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime_id", "command"),
    [
        ("calculix-2.23", ["ccx", "-i", "model"]),
        ("openfoam-14", ["/bin/sh", "run-case.sh"]),
        ("ansys-mapdl-2026-r1", ["/bin/sh", "run-mapdl.sh"]),
    ],
)
async def test_solver_job_lifecycle_uses_shared_transport(
    tmp_path, runtime_id, command
):
    job_id, input_id, artifact_id = uuid4(), uuid4(), uuid4()
    source_bytes = b"solver input\x00\xff"
    result_bytes = b"stress\n55\n"
    source = tmp_path / "solver-input"
    source.write_bytes(source_bytes)
    submitted = []
    requests = []
    job = {
        "state_version": 1,
        "active_attempt": 1,
        "created_at": "2026-11-17T00:00:00Z",
        "updated_at": "2026-11-17T00:00:00Z",
        "id": str(job_id),
        "runtime_id": runtime_id,
        "compute_profile_id": "cpu-standard-4",
        "timeout_seconds": 900,
        "status": "succeeded",
    }
    metadata = {
        "id": str(artifact_id),
        "path": "stress.csv",
        "byte_length": len(result_bytes),
        "sha256": hashlib.sha256(result_bytes).hexdigest(),
    }

    async def handle(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer solver-user"
        path = request.url.path
        if path == "/execution/inputs":
            assert await request.aread() == source_bytes
            return httpx.Response(
                200,
                json={
                    "id": str(input_id),
                    "byte_length": len(source_bytes),
                    "sha256": hashlib.sha256(source_bytes).hexdigest(),
                },
            )
        if path == "/execution/jobs" and request.method == "POST":
            assert request.headers["Idempotency-Key"] == "solver-submission"
            submitted.append(json.loads(await request.aread()))
            return httpx.Response(202, json={**job, "status": "running"})
        if path == f"/execution/jobs/{job_id}":
            return httpx.Response(200, json=job)
        if path == "/execution/jobs":
            return httpx.Response(
                200, json={"items": [job], "next_page": "another-page"}
            )
        if path.endswith("/cancel"):
            return httpx.Response(200, json={**job, "status": "canceling"})
        if path.endswith("/artifacts"):
            return httpx.Response(
                200,
                json={
                    "state": "complete",
                    "failure_code": None,
                    "artifacts": {"items": [metadata], "next_page": None},
                },
            )
        if path.endswith("/logs"):
            stream = request.url.params["stream"]
            data = b"solver completed\n" if stream == "stdout" else b""
            return httpx.Response(
                200,
                json={
                    "state": "complete",
                    "available": True,
                    "stream": stream,
                    "data_base64": base64.b64encode(data).decode(),
                    "offset": 0,
                    "next_offset": len(data),
                    "stored_bytes": len(data),
                    "complete": True,
                    "truncated": False,
                },
            )
        if path.endswith(str(artifact_id)):
            return httpx.Response(200, content=result_bytes)
        raise AssertionError(f"Unexpected endpoint: {path}")

    async with AsyncKittyCAD(
        token="solver-user",
        base_url="https://execution.example.test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    ) as sdk:
        client = ExecutionClient(sdk)
        uploaded = await client.upload(source)
        request = {
            "runtime_id": runtime_id,
            "compute_profile_id": "cpu-standard-4",
            "command": command,
            "timeout_seconds": 900,
            "environment": {"OMP_NUM_THREADS": "2"},
            "inputs": [{"artifact_id": str(uploaded.id), "path": "model.inp"}],
            "outputs": [{"path": "stress.csv", "required": True}],
        }
        admitted = await client.submit(request, "solver-submission")
        assert submitted == [request]
        assert admitted.runtime_id == runtime_id
        completed = await client.wait(admitted, 2)
        assert completed.status == "succeeded"

        observed = await observe_execution(client, completed)
        assert observed.ok and observed.job_id == job_id
        assert observed.logs is not None
        assert observed.logs[0].text == "solver completed\n"
        assert "Python" not in observed.msg
        artifact = await client.artifact(job_id, artifact_id)
        async with client.download(job_id, artifact) as output:
            assert output.read_bytes() == result_bytes

        listed = await client.list_jobs(
            runtime_id=runtime_id, status="succeeded", page_token="cursor", limit=10
        )
        assert listed.items[0].runtime_id == runtime_id
        assert listed.next_page == "another-page"
        assert dict(requests[-1].url.params) == {
            "runtime_id": runtime_id,
            "status": "succeeded",
            "page_token": "cursor",
            "limit": "10",
        }
        await client.list_jobs()
        assert dict(requests[-1].url.params) == {"limit": "25"}
        canceled = await client.cancel(job_id)
        assert canceled.status == "canceling" and canceled.runtime_id == runtime_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure, attempts_expected", [("lost", 2), ("unavailable", 2), ("malformed", 1)]
)
async def test_uncertain_solver_submission_has_runtime_neutral_recovery_message(
    failure, attempts_expected
):
    attempts = []

    def respond(request):
        attempts.append((request.headers["Idempotency-Key"], request.content))
        if failure == "lost":
            raise httpx.ReadError("connection lost", request=request)
        if failure == "unavailable":
            return httpx.Response(
                503, json={"message": "Deployment is offline", "request_id": "req-9"}
            )
        return httpx.Response(202, json={})

    async with AsyncKittyCAD(
        token="solver-user",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as sdk:
        client = ExecutionClient(sdk)
        with pytest.raises(ExecutionError) as error:
            await client.submit(
                {
                    "runtime_id": "calculix-2.23",
                    "compute_profile_id": "cpu",
                    "timeout_seconds": 300,
                    "command": ["ccx"],
                },
                "solver-submission",
            )
    assert len(attempts) == attempts_expected and len(set(attempts)) == 1
    assert error.value.code == "execution_submission_unknown"
    assert "python" not in str(error.value).lower()
    if failure == "unavailable":
        assert "Deployment is offline" in str(error.value) and "req-9" in str(
            error.value
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["submit", "download"])
async def test_api_rejection_preserves_public_error_fields(operation):
    requests = []
    detail = {"message": "The requested artifact has expired", "request_id": "req-7"}
    if operation == "submit":
        detail.update(
            message="Key already used for a different request",
            error_code="IdempotencyConflict",
        )

    def respond(request):
        requests.append(request)
        return httpx.Response(
            409 if operation == "submit" else 404,
            stream=httpx.ByteStream(json.dumps(detail).encode()),
            headers={"content-type": "application/json; charset=utf-8"},
        )

    async with AsyncKittyCAD(
        token="caller",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as sdk:
        client = ExecutionClient(sdk)
        with pytest.raises(ExecutionError) as error:
            if operation == "submit":
                await client.submit(
                    {
                        "runtime_id": "calculix",
                        "compute_profile_id": "cpu",
                        "timeout_seconds": 300,
                        "command": ["ccx"],
                    },
                    "original-key",
                )
            else:
                artifact = ExecutionArtifact(
                    id=uuid4(), path="out", byte_length=0, sha256="0" * 64
                )
                async with client.download(uuid4(), artifact):
                    pytest.fail("An error response cannot be used as artifact content")
    assert len(requests) == 1
    assert error.value.code == detail.get("error_code", "execution_http_404")
    assert all(value in str(error.value) for value in detail.values())
    assert str(requests[0].url) not in str(error.value)


@pytest.mark.asyncio
async def test_tools_and_sdk_calls_keep_their_own_errors_on_a_shared_client():
    both_started = asyncio.Event()
    requests = 0

    async def respond(request):
        nonlocal requests
        requests += 1
        if requests == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=2)
        return httpx.Response(403, text="private gateway response")

    async with AsyncKittyCAD(
        token="caller",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as sdk:
        job_id = uuid4()
        tools = ExecutionClient(sdk)
        tool_error, sdk_error = await asyncio.gather(
            tools.job(job_id),
            sdk.executor.get_execution_job(Uuid(str(job_id))),
            return_exceptions=True,
        )
    assert isinstance(tool_error, ExecutionError)
    assert tool_error.code == "execution_http_403"
    assert str(job_id) not in str(tool_error)
    assert isinstance(sdk_error, KittyCADClientError)
    assert sdk_error.request_url and str(job_id) in sdk_error.request_url


@pytest.mark.asyncio
async def test_artifact_lookup_follows_the_api_cursor():
    artifact = ExecutionArtifact(id=uuid4(), path="out", byte_length=0, sha256="0" * 64)

    def respond(request):
        page = int(request.url.params.get("page_token", 0))
        items = [artifact.model_dump(mode="json")] if page == 42 else []
        return httpx.Response(
            200,
            json={
                "state": "complete",
                "artifacts": {
                    "items": items,
                    "next_page": None if items else str(page + 1),
                },
            },
        )

    async with AsyncKittyCAD(
        token="caller",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as sdk:
        assert await ExecutionClient(sdk).artifact(uuid4(), artifact.id) == artifact


def test_default_execution_error_is_runtime_neutral():
    assert ExecutionError("Solver error").code == "execution_error"


@pytest.mark.asyncio
async def test_download_larger_than_former_limit_is_streamed_and_verified():
    chunk = b"x" * (1024 * 1024)
    digest = hashlib.sha256()
    for _ in range(100):
        digest.update(chunk)
    digest.update(b"x")
    length = 100 * len(chunk) + 1

    class ArtifactStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(100):
                yield chunk
            yield b"x"

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, stream=ArtifactStream())
    )
    artifact = ExecutionArtifact(
        id=uuid4(),
        path="large.bin",
        byte_length=length,
        sha256=digest.hexdigest(),
    )
    async with AsyncKittyCAD(
        token="test-user-token", http_client=httpx.AsyncClient(transport=transport)
    ) as sdk:
        client = ExecutionClient(sdk)
        async with client.download(uuid4(), artifact) as path:
            assert path.stat().st_size == length


@pytest.mark.asyncio
async def test_wait_deadline_covers_a_slow_status_read(monkeypatch):
    async def slow_read(job_id):
        await asyncio.sleep(10)
        pytest.fail("The wait deadline should cancel this read")

    async with AsyncKittyCAD(token="caller") as sdk:
        client = ExecutionClient(sdk)
        monkeypatch.setattr(client, "job", slow_read)
        job = ExecutionJob(
            id=uuid4(),
            status="running",
            runtime_id="r",
            compute_profile_id="c",
            timeout_seconds=10,
        )
        assert await asyncio.wait_for(client.wait(job, 1), timeout=2) == job


@pytest.mark.asyncio
async def test_staged_submission_retries_without_reupload(tmp_path):
    input_id, job_id = uuid4(), uuid4()
    uploads, admissions = [], []
    source = tmp_path / "input.dat"
    source.write_bytes(b"input")

    def respond(request):
        if request.url.path.endswith("/inputs"):
            uploads.append(request.content)
            return httpx.Response(
                201,
                json={
                    "id": str(input_id),
                    "byte_length": len(request.content),
                    "sha256": hashlib.sha256(request.content).hexdigest(),
                },
            )
        admissions.append(
            (request.headers["Idempotency-Key"], json.loads(request.content))
        )
        if len(admissions) == 1:
            raise httpx.ReadError("lost response", request=request)
        return httpx.Response(
            202,
            json={
                "id": str(job_id),
                "status": "queued",
                "active_attempt": 0,
                "state_version": 1,
                "created_at": "2026-11-17T00:00:00Z",
                "updated_at": "2026-11-17T00:00:00Z",
                **admissions[-1][1],
            },
        )

    submission = ExecutionSubmission(
        runtime_id="solver", compute_profile_id="cpu", command=["solve"]
    )
    async with AsyncKittyCAD(
        token="caller",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as sdk:
        client = ExecutionClient(sdk)
        with pytest.raises(ExecutionError, match="overlap"):
            await client.submit_staged(
                submission, [("input", source), ("input/sub", source)], "key"
            )
        with pytest.raises(ExecutionError, match="at most 256"):
            existing = ExecutionSubmission(
                **submission.model_dump(exclude={"inputs"}),
                inputs=[{"path": "existing", "artifact_id": input_id}],
            )
            await client.prepare_submission(
                existing, [(str(i), source) for i in range(256)]
            )
        assert not uploads and not admissions
        job = await client.submit_staged(submission, [("input", source)], "key")
    assert job.id == job_id and uploads == [b"input"]
    assert len(admissions) == 2 and admissions[0] == admissions[1]
    assert admissions[0][0] == "key"
    assert admissions[0][1]["inputs"] == [
        {"artifact_id": str(input_id), "path": "input"}
    ]


@pytest.mark.asyncio
async def test_observation_preserves_job_when_wait_fails(monkeypatch):
    async def failed_wait(job, seconds):
        raise httpx.ReadError("private detail")

    async with AsyncKittyCAD(token="caller") as sdk:
        client = ExecutionClient(sdk)
        monkeypatch.setattr(client, "wait", failed_wait)
        job = ExecutionJob(
            id=uuid4(),
            status="running",
            runtime_id="r",
            compute_profile_id="c",
            timeout_seconds=10,
        )
        result = await observe_execution(client, job, wait_seconds=20)
    assert result.job == job and result.job_id == job.id
    assert result.error_code == "execution_observation_unavailable"
    assert "private detail" not in result.msg
