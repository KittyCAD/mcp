"""The shared execution transport accepts solver jobs without Python policy."""

import base64
import hashlib
import json
from uuid import uuid4

import httpx
import pytest
from kittycad import AsyncKittyCAD

from zoo_mcp.execution_jobs import ExecutionClient, ExecutionError, observe_execution


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
async def test_uncertain_solver_submission_has_runtime_neutral_recovery_message():
    attempts = []

    def fail(request):
        attempts.append((request.headers["Idempotency-Key"], request.content))
        raise httpx.ReadError("connection lost", request=request)

    async with AsyncKittyCAD(
        token="solver-user",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(fail)),
    ) as sdk:
        client = ExecutionClient(sdk)
        with pytest.raises(ExecutionError) as failure:
            await client.submit(
                {"runtime_id": "calculix-2.23", "command": ["ccx"]}, "solver-submission"
            )
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert failure.value.code == "execution_submission_unknown"
    assert "python" not in str(failure.value).lower()


def test_default_execution_error_is_runtime_neutral():
    assert ExecutionError("Solver error").code == "execution_error"
