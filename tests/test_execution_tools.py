"""Durable execution is available through MCP without application dependencies."""

import asyncio
import base64
import hashlib
import json
from uuid import uuid4

import httpx
import pytest
from kittycad import AsyncKittyCAD
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, ImageContent

from zoo_mcp import execution_jobs
from zoo_mcp.client_config import ZooClientConfig, get_client_config, use_client_config
from zoo_mcp.execution_jobs import ExecutionClient, ExecutionError, execution_client
from zoo_mcp.server import mcp


async def call_tool(name: str, arguments: dict) -> CallToolResult:
    result = await mcp.call_tool(name, arguments)
    assert isinstance(result, CallToolResult)
    assert result.structured_content is not None
    return result


@pytest.fixture
def api(monkeypatch):
    class API:
        job_id = uuid4()
        input_id = uuid4()
        artifact_id = uuid4()
        status = "succeeded"
        archival = "complete"
        cancel_status = "canceling"
        name = "result.csv"
        data = b"value\n55\n"
        log_data = b"done\xff\n"

        def __init__(self):
            self.requests = []
            self.submissions = []

        @property
        def job(self):
            return {
                "id": str(self.job_id),
                "status": self.status,
                "runtime_id": "openfoam-14",
                "compute_profile_id": "cpu-standard-2",
                "timeout_seconds": 300,
            }

        @property
        def artifact(self):
            return {
                "id": str(self.artifact_id),
                "path": self.name,
                "byte_length": len(self.data),
                "sha256": hashlib.sha256(self.data).hexdigest(),
            }

        async def handle(self, request):
            self.requests.append(request)
            assert request.headers["Authorization"] == "Bearer caller-token"
            assert request.headers["X-Caller"] == "request-context"
            assert request.url.host == "execution.example.test"
            path = request.url.path
            if path == "/execution/inputs":
                content = await request.aread()
                return httpx.Response(
                    201,
                    json={
                        "id": str(self.input_id),
                        "byte_length": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    },
                )
            if path == "/execution/jobs" and request.method == "POST":
                submission = json.loads(await request.aread())
                self.submissions.append(
                    (request.headers["Idempotency-Key"], submission)
                )
                return httpx.Response(
                    202,
                    json={
                        **self.job,
                        **{
                            key: submission[key]
                            for key in [
                                "runtime_id",
                                "compute_profile_id",
                                "timeout_seconds",
                            ]
                        },
                    },
                )
            if path == "/execution/jobs":
                return httpx.Response(
                    200, json={"items": [self.job], "next_page": "next-jobs"}
                )
            if path.endswith("/cancel"):
                return httpx.Response(
                    200, json={**self.job, "status": self.cancel_status}
                )
            if path == f"/execution/jobs/{self.job_id}":
                return httpx.Response(200, json=self.job)
            if path.endswith("/artifacts"):
                return httpx.Response(
                    200,
                    json={
                        "state": self.archival,
                        "failure_code": "output_missing"
                        if self.archival == "failed"
                        else None,
                        "artifacts": {
                            "items": [self.artifact]
                            if self.archival != "pending"
                            else [],
                            "next_page": "next-artifacts",
                        },
                    },
                )
            if path.endswith("/logs"):
                stream = request.url.params["stream"]
                offset = int(request.url.params["offset"])
                content = (
                    self.log_data[offset : offset + int(request.url.params["limit"])]
                    if stream == "stdout" and self.archival != "pending"
                    else b""
                )
                return httpx.Response(
                    200,
                    json={
                        "state": self.archival,
                        "available": self.archival != "pending",
                        "stream": stream,
                        "data_base64": base64.b64encode(content).decode(),
                        "offset": offset,
                        "next_offset": offset + len(content),
                        "stored_bytes": None
                        if self.archival == "pending"
                        else (len(self.log_data) if stream == "stdout" else 0),
                        "complete": self.archival != "pending",
                        "truncated": False,
                    },
                )
            if path.endswith(str(self.artifact_id)):
                return httpx.Response(200, content=self.data)
            raise AssertionError(path)

    instance = API()
    monkeypatch.setattr(
        execution_jobs,
        "AsyncKittyCAD",
        lambda **kwargs: AsyncKittyCAD(
            **kwargs,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(instance.handle)
            ),
        ),
    )
    with use_client_config(
        ZooClientConfig(
            "caller-token",
            "https://execution.example.test/",
            {"X-Caller": "request-context"},
        )
    ):
        yield instance


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runtime,command",
    [
        ("python-3.12", ["python", "script.py"]),
        ("calculix-2.23", ["ccx", "-i", "beam"]),
        ("openfoam-14", ["/bin/sh", "run-case.sh"]),
        ("ansys-mapdl-2026-r1", ["/bin/sh", "run-mapdl.sh"]),
    ],
)
async def test_mcp_submission_upload_and_job_management(
    api, tmp_path, runtime, command
):
    source = tmp_path / "input.bin"
    source.write_bytes(b"input\x00\xff")
    upload = await call_tool("upload_job_execution_input", {"input_file": str(source)})
    assert upload.structured_content["id"] == str(api.input_id)
    submission = {
        "runtime_id": runtime,
        "compute_profile_id": "cpu-standard-4",
        "command": command,
        "inputs": [{"artifact_id": str(api.input_id), "path": "inputs/model.inp"}],
        "outputs": [{"path": "results/*.csv", "required": False}],
        "environment": {"OMP_NUM_THREADS": "2"},
        "timeout_seconds": 900,
    }
    admitted = await call_tool(
        "submit_job_execution",
        {"submission": submission, "idempotency_key": "stable-key"},
    )
    assert admitted.structured_content["id"] == str(api.job_id)
    assert admitted.structured_content["runtime_id"] == runtime
    assert api.submissions == [("stable-key", submission)]
    observed = await call_tool(
        "get_job_execution",
        {"job_id": str(api.job_id), "artifact_page_token": "cursor"},
    )
    data = observed.structured_content
    assert data["ok"] and data["job_id"] == str(api.job_id)
    assert data["results"]["artifacts"]["next_page"] == "next-artifacts"
    assert (
        data["logs"][0]["text"] == "done\ufffd\n" and data["logs"][0]["utf8_replaced"]
    )
    log = await call_tool(
        "read_job_execution_logs", {"job_id": str(api.job_id), "offset": 4, "limit": 1}
    )
    assert (
        log.structured_content["text"] == "\ufffd"
        and log.structured_content["next_offset"] == 5
    )
    listing = await call_tool(
        "list_job_executions",
        {"runtime_id": runtime, "page_token": "more", "status": "running"},
    )
    assert listing.structured_content["next_page"] == "next-jobs"
    assert dict(api.requests[-1].url.params) == {
        "runtime_id": runtime,
        "page_token": "more",
        "status": "running",
        "limit": "25",
    }
    canceled = await call_tool("cancel_job_execution", {"job_id": str(api.job_id)})
    assert canceled.structured_content["status"] == "canceling"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "archival"), [("queued", "pending"), ("failed", "failed")]
)
async def test_job_observation_preserves_archive_state(api, status, archival):
    api.status, api.archival = status, archival
    response = await call_tool("get_job_execution", {"job_id": str(api.job_id)})
    data = response.structured_content
    assert data["job"]["status"] == status and data["results"]["state"] == archival
    if archival == "pending":
        assert data["ok"] and not data["logs"][0]["available"]
        assert data["logs"][0]["stored_bytes"] is None
    else:
        assert not data["ok"] and data["results"]["failure_code"] == "output_missing"
        assert data["logs"][0]["text"] and data["results"]["artifacts"]["items"]


@pytest.mark.asyncio
async def test_repeated_cancel_returns_the_observed_status(api):
    for status in ["canceling", "canceling", "succeeded"]:
        api.cancel_status = status
        result = await call_tool("cancel_job_execution", {"job_id": str(api.job_id)})
        assert result.structured_content["status"] == status
    assert not api.submissions


@pytest.mark.asyncio
async def test_submission_forwards_more_than_256_inputs(api):
    inputs = [
        {"artifact_id": str(api.input_id), "path": f"inputs/{index}.dat"}
        for index in range(300)
    ]
    result = await call_tool(
        "submit_job_execution",
        {
            "idempotency_key": "many-inputs",
            "submission": {
                "runtime_id": "openfoam-14",
                "compute_profile_id": "cpu-standard-2",
                "command": ["sh", "run.sh"],
                "inputs": inputs,
            },
        },
    )
    assert result.structured_content["id"] == str(api.job_id)
    assert api.submissions[0][1]["inputs"] == inputs


@pytest.mark.asyncio
async def test_log_reads_forward_and_decode_pages_larger_than_64k(api):
    limit = 128 * 1024
    api.log_data = b"prefix" + b"x" * limit + b"tail"
    result = await call_tool(
        "read_job_execution_logs",
        {"job_id": str(api.job_id), "offset": 6, "limit": limit},
    )
    assert result.structured_content["text"] == "x" * limit
    assert result.structured_content["next_offset"] == 6 + limit
    assert result.structured_content["stored_bytes"] == len(api.log_data)
    assert api.requests[-1].url.params["limit"] == str(limit)


@pytest.mark.asyncio
async def test_mcp_text_image_and_verified_download(api, tmp_path):
    api.name = "result.txt"
    api.data = ("x" * 65535 + "éend").encode()
    arguments = {"job_id": str(api.job_id), "artifact_id": str(api.artifact_id)}
    first = await call_tool("use_job_execution_artifact", arguments)
    assert first.structured_content["text"] == "x" * 65535
    assert first.structured_content["next_offset"] == 65535
    second = await call_tool(
        "use_job_execution_artifact", {**arguments, "offset": 65535}
    )
    assert second.structured_content["text"] == "éend"
    with pytest.raises(ToolError, match="UTF-8 character boundary"):
        await call_tool("use_job_execution_artifact", {**arguments, "offset": 65536})
    api.name = "result.png"
    api.data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aV1sAAAAASUVORK5CYII="
    )
    image = await call_tool("use_job_execution_artifact", arguments)
    content = next(part for part in image.content if isinstance(part, ImageContent))
    assert (
        content.mime_type == "image/png" and base64.b64decode(content.data) == api.data
    )
    destination = tmp_path / "download.bin"
    downloaded = await call_tool(
        "use_job_execution_artifact",
        {**arguments, "action": "download", "destination": str(destination)},
    )
    assert destination.read_bytes() == api.data
    assert downloaded.structured_content["destination"] == str(destination)
    with pytest.raises(ToolError, match="already exists"):
        await call_tool(
            "use_job_execution_artifact",
            {**arguments, "action": "download", "destination": str(destination)},
        )
    assert destination.read_bytes() == api.data


@pytest.mark.asyncio
async def test_invalid_submission_does_not_contact_api(api):
    submission = {
        "runtime_id": "openfoam-14",
        "compute_profile_id": "cpu-standard-2",
        "command": ["sh", "run.sh"],
    }
    with pytest.raises(ToolError):
        await call_tool("submit_job_execution", {"submission": submission})
    for paths in [["../escape"], ["input", "input/child"], ["input", "input"]]:
        with pytest.raises(ToolError):
            await call_tool(
                "submit_job_execution",
                {
                    "idempotency_key": "key",
                    "submission": {
                        **submission,
                        "inputs": [
                            {"artifact_id": str(uuid4()), "path": p} for p in paths
                        ],
                    },
                },
            )
    assert not api.requests


@pytest.mark.asyncio
async def test_partial_observation_keeps_admitted_job(api, monkeypatch):
    async def unavailable(*args, **kwargs):
        raise httpx.ReadError("untrusted raw error")

    monkeypatch.setattr(ExecutionClient, "artifacts", unavailable)
    result = await call_tool("get_job_execution", {"job_id": str(api.job_id)})
    assert result.structured_content["job_id"] == str(api.job_id)
    assert result.structured_content["ok"]
    assert result.structured_content["error_code"] == "execution_results_unavailable"
    assert "untrusted raw error" not in str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["upload", "download"])
async def test_integrity_failures_never_publish_unverified_files(
    tmp_path, monkeypatch, kind
):
    data = b"contents"
    job_id, artifact_id = uuid4(), uuid4()
    metadata = {
        "id": str(artifact_id),
        "path": "result.bin",
        "byte_length": len(data),
        "sha256": "0" * 64,
    }

    def respond(request):
        if request.url.path.endswith("/inputs"):
            return httpx.Response(200, json=metadata)
        if request.url.path.endswith("/artifacts"):
            return httpx.Response(
                200, json={"state": "complete", "artifacts": {"items": [metadata]}}
            )
        return httpx.Response(200, content=data)

    monkeypatch.setattr(
        execution_jobs,
        "AsyncKittyCAD",
        lambda **kwargs: AsyncKittyCAD(
            **kwargs,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ),
    )
    destination = tmp_path / "download.bin"
    with (
        use_client_config(ZooClientConfig("caller-token")),
        pytest.raises(ToolError, match="integrity"),
    ):
        if kind == "upload":
            source = tmp_path / "source.bin"
            source.write_bytes(data)
            await call_tool("upload_job_execution_input", {"input_file": str(source)})
        else:
            await call_tool(
                "use_job_execution_artifact",
                {
                    "job_id": str(job_id),
                    "artifact_id": str(artifact_id),
                    "action": "download",
                    "destination": str(destination),
                },
            )
    assert not destination.exists()


@pytest.mark.asyncio
async def test_execution_clients_isolate_authentication_and_redact_failures(
    monkeypatch,
):
    records = []

    def respond(request):
        records.append(
            (
                request.url.host,
                request.headers["Authorization"],
                request.headers["X-Caller"],
            )
        )
        return httpx.Response(
            403,
            text="secret response body",
            headers={"Location": "https://secret.example"},
        )

    monkeypatch.setattr(
        execution_jobs,
        "AsyncKittyCAD",
        lambda **kwargs: AsyncKittyCAD(
            **kwargs,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ),
    )

    async def request(name):
        config = ZooClientConfig(
            name + "-token", "https://" + name + ".example", {"X-Caller": name}
        )
        with use_client_config(config):
            with pytest.raises(ExecutionError) as failure:
                async with execution_client() as client:
                    await client.list_jobs()
            assert failure.value.code == "execution_http_403"
            assert "secret" not in str(failure.value) and name + "-token" not in str(
                failure.value
            )
            assert get_client_config() is config

    await asyncio.gather(request("first"), request("second"))
    assert set(records) == {
        ("first.example", "Bearer first-token", "first"),
        ("second.example", "Bearer second-token", "second"),
    }
    with use_client_config(ZooClientConfig("")), pytest.raises(ExecutionError):
        async with execution_client():
            pytest.fail("An explicit empty credential must not use the local token")
    assert len(records) == 2
