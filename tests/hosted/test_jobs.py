import asyncio
import io
import time
import zipfile

import jsonschema
import pytest

from tests.hosted.test_hosted import MemoryBackend
from zoo_mcp.hosted.app import create_app
from zoo_mcp.hosted.backend import ServiceError
from zoo_mcp.hosted.runtime import Runtime


@pytest.mark.asyncio
async def test_jobs_retry_conflict_restart_and_cancel():
    backend = MemoryBackend()
    runtime = Runtime(backend)
    p = backend.owner
    calls = []

    async def operation():
        calls.append(1)
        return {"ok": True}

    args = {"idempotency_key": "one"}
    first, second = await asyncio.gather(
        runtime.submit(p, "example", args, operation),
        runtime.submit(p, "example", args, operation),
    )
    assert len(calls) == 1 and first["job_id"] == second["job_id"]
    assert (await runtime.job(p, first["job_id"]))["status"] == "completed"
    with pytest.raises(ServiceError, match="different request"):
        await runtime.submit(p, "other", args, operation)
    record = backend.rows[first["job_id"]]
    record["data"].update(status="running", deadline=0)
    restarted = Runtime(backend)
    assert (await restarted.submit(p, "example", args, operation))[
        "status"
    ] == "interrupted"
    assert len(calls) == 1
    started = asyncio.Event()

    async def forever():
        started.set()
        await asyncio.Event().wait()

    pending = asyncio.create_task(
        runtime.submit(p, "example", {"idempotency_key": "two"}, forever)
    )
    await started.wait()
    job_id = next(iter(runtime.jobs))
    assert (await runtime.cancel(p, job_id))["status"] == "cancelled"
    await pending
    await runtime.close()
    await backend.http.aclose()


@pytest.mark.asyncio
async def test_complete_source_and_fresh_job_download_links(monkeypatch):
    backend = MemoryBackend()
    runtime = Runtime(backend)
    p = backend.owner
    source = await runtime.artifacts.write_source(
        p, {"main.kcl": 'import "parts/leg.kcl"', "parts/leg.kcl": "// part"}
    )
    with zipfile.ZipFile(io.BytesIO(backend.blobs[source["artifact_id"]])) as archive:
        assert set(archive.namelist()) == {"main.kcl", "parts/leg.kcl"}

    async def result():
        return source

    job = await runtime.submit(p, "write", {"idempotency_key": "one"}, result)
    original = job["result"]["download_url"]
    monkeypatch.setattr(time, "time", lambda: 1900000000)
    fresh = await runtime.job(p, job["job_id"])
    assert fresh["result"]["download_url"] != original
    token = fresh["result"]["download_url"].split("capability=")[1]
    runtime.artifacts.capabilities.verify(token, source["artifact_id"], "download")
    await backend.http.aclose()


@pytest.mark.asyncio
async def test_background_is_opt_in_and_only_background_requires_a_key():
    backend = MemoryBackend()
    app = create_app(backend=backend)
    p = backend.owner
    try:
        direct = await app.state.call(p, "format_kcl", {"kcl_code": "x=1"})
        explicit = await app.state.call(
            p,
            "format_kcl",
            {
                "kcl_code": "x=1",
                "execution_mode": "direct",
                "idempotency_key": "ignored",
            },
        )
        assert direct == explicit
        assert not await backend.list(p, "job")
        with pytest.raises(jsonschema.ValidationError):
            await app.state.call(
                p, "format_kcl", {"kcl_code": "x=1", "execution_mode": "background"}
            )
        args = {
            "kcl_code": "x=1",
            "execution_mode": "background",
            "idempotency_key": "one",
        }
        job = await app.state.call(p, "format_kcl", args)
        assert job["status"] == "completed"
        assert job["result"] == direct
        assert (await app.state.call(p, "format_kcl", args))["job_id"] == job["job_id"]
        with pytest.raises(ServiceError, match="different request"):
            await app.state.call(p, "format_kcl", {**args, "kcl_code": "x=2"})
    finally:
        await app.state.runtime.close()
        await backend.http.aclose()
