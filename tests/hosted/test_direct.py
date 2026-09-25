import asyncio
import io
import json
import socket
import zipfile
from uuid import uuid4

import httpx
import pytest
import uvicorn
from sse_starlette.sse import EventSourceResponse

from tests.hosted.test_hosted import MemoryBackend, principal
from zoo_mcp.hosted.app import create_app
from zoo_mcp.hosted.backend import ServiceError
from zoo_mcp.hosted.catalog import catalog
from zoo_mcp.hosted.runtime import Runtime


@pytest.mark.asyncio
async def test_foundation_catalog_and_direct_results():
    tools = {tool.name: tool for tool in await catalog()}
    assert (
        not {
            "open_zoo_workspace",
            "import_attachment",
        }
        & tools.keys()
    )
    assert "idempotency_key" not in tools["format_kcl"].input_schema["required"]
    assert (
        tools["format_kcl"].input_schema["properties"]["execution_mode"]["default"]
        == "direct"
    )
    backend = MemoryBackend()
    app = create_app(backend=backend)
    try:
        result = await app.state.call(backend.owner, "format_kcl", {"kcl_code": "x=1"})
        assert "x = 1" in json.dumps(result)
        assert "job_id" not in result
        assert not await backend.list(backend.owner, "job")
    finally:
        await app.state.runtime.close()
        await backend.http.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["format_kcl", "lint_and_fix_kcl"])
@pytest.mark.parametrize("archive", [True, False])
async def test_fixes_return_updated_source_without_mutating_input(tool, archive):
    backend = MemoryBackend()
    runtime = Runtime(backend)
    p = backend.owner
    code = (
        "x=1"
        if tool == "format_kcl"
        else (
            "c = startSketchOn(XY)\n"
            "  |> circle(center = [0, 0], radius = 1)\n"
            "  |> circle(center = [5, 0], radius = 1)\n"
            "  |> circle(center = [0, 5], radius = 1)\n"
            "  |> circle(center = [5, 5], radius = 1)\n"
        )
    )
    if archive:
        source = await runtime.artifacts.write_source(
            p, {"main.kcl": code, "parts/leg.kcl": "// preserved dependency\n"}
        )
    else:
        source = runtime.artifacts.describe(
            p, await runtime.artifacts.store(p, "main.kcl", code.encode())
        )
    original = backend.blobs[source["artifact_id"]]
    try:
        result = await runtime.call(
            p, tool, {"project_artifact_id": source["artifact_id"]}
        )
        updated = result["source"]["artifact_id"]
        assert updated != source["artifact_id"]
        assert backend.blobs[source["artifact_id"]] == original
        with zipfile.ZipFile(io.BytesIO(backend.blobs[updated])) as files:
            assert files.read("main.kcl").decode() != code
            if tool == "format_kcl":
                assert files.read("main.kcl").decode().strip() == "x = 1"
            if archive:
                assert (
                    files.read("parts/leg.kcl").decode() == "// preserved dependency\n"
                )
        assert "zoo-mcp-" not in json.dumps(result)
        with pytest.raises(ServiceError):
            await runtime.artifacts.read(principal(), updated)
    finally:
        await runtime.close()
        await backend.http.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("disconnect", [False, True])
async def test_sse_sends_keepalives_and_disconnect_does_not_cancel(
    monkeypatch, disconnect
):
    backend = MemoryBackend()
    app = create_app(backend=backend)
    completed = asyncio.Event()
    monkeypatch.setattr(EventSourceResponse, "DEFAULT_PING_INTERVAL", 0.03)

    async def slow_call(p, name, arguments):
        await asyncio.sleep(0.25)
        completed.set()
        return {"answer": 42}

    monkeypatch.setattr(app.state.runtime, "call", slow_call)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                await asyncio.sleep(0.01)
        async with (
            httpx.AsyncClient() as client,
            client.stream(
                "POST",
                f"http://127.0.0.1:{port}/mcp",
                headers={
                    "Host": "mcp.test",
                    "Authorization": f"Bearer {backend.owner.token}",
                    "Accept": "application/json, text/event-stream",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "format_kcl", "arguments": {"kcl_code": "x=1"}},
                },
            ) as response,
        ):
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            lines = response.aiter_lines()
            async for line in lines:
                if line.startswith(": ping"):
                    assert not completed.is_set()
                    break
            else:
                pytest.fail("No keepalive received before the result")
            if not disconnect:
                messages = [
                    json.loads(line[6:])
                    async for line in lines
                    if line.startswith("data: {")
                ]
                assert messages[-1]["result"]["structuredContent"] == {"answer": 42}
        await asyncio.wait_for(completed.wait(), 2)
    finally:
        server.should_exit = True
        await asyncio.wait_for(serving, 5)
        listener.close()


@pytest.mark.asyncio
async def test_explicit_cancellation_stops_direct_work():
    backend = MemoryBackend()
    runtime = Runtime(backend)
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def operation():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            stopped.set()

    task = asyncio.create_task(runtime.run_direct(operation, asyncio.Event()))
    await started.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.wait_for(stopped.wait(), 1)
    await runtime.close()
    assert not runtime.direct_tasks
    await backend.http.aclose()


@pytest.mark.asyncio
async def test_modeling_session_continuity_and_grant_isolation(monkeypatch):
    backend = MemoryBackend()
    runtime = Runtime(backend)
    session = str(uuid4())
    workers = []

    async def invoke(p, worker, name, arguments):
        workers.append(worker)
        return {"data": {"result": session}, "content": []}

    monkeypatch.setattr(runtime, "invoke", invoke)
    try:
        assert (await runtime.call(backend.owner, "start_modeling_session", {}))[
            "session_id"
        ] == session
        assert (await runtime.call(backend.owner, "get_modeling_sessions", {}))[
            "sessions"
        ] == [session]
        await runtime.call(
            backend.owner, "execute_kcl", {"session_id": session, "kcl_code": "x=1"}
        )
        assert workers[0] is workers[1]
        with pytest.raises(ServiceError):
            await runtime.call(principal(), "snapshot", {"session_id": session})
        await runtime.call(
            backend.owner, "stop_modeling_session", {"session_id": session}
        )
        assert not runtime.workers
        assert (await runtime.call(backend.owner, "get_modeling_sessions", {}))[
            "sessions"
        ] == []
    finally:
        await runtime.close()
        await backend.http.aclose()
