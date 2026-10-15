import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent

from zoo_mcp import kcl_docs, kcl_samples, server
from zoo_mcp.kcl_docs import KCLDocs
from zoo_mcp.kcl_samples import KCLSamples, SampleMetadata


@pytest.fixture(autouse=True)
def reset_indexes(monkeypatch: pytest.MonkeyPatch):
    for index in (KCLDocs, KCLSamples):
        monkeypatch.setattr(index, "_instance", None)
        monkeypatch.setattr(index, "_init_lock", None)
    monkeypatch.setattr(server, "_kcl_index_tasks", {})
    monkeypatch.setattr(server, "zoo_stop_all_modeling_sessions", AsyncMock())


def _samples() -> KCLSamples:
    return KCLSamples(
        manifest={
            "public-project": SampleMetadata(
                title="Gear", description="A gear", multipleFiles=False
            )
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable", ["docs", "samples"])
async def test_startup_and_retries_do_not_block_the_other_index(
    monkeypatch: pytest.MonkeyPatch, unavailable: str
):
    path = "docs/kcl-lang/functions"
    docs = KCLDocs(docs={path: "# Functions"})
    docs.index["kcl-lang"].append(path)
    docs_fetch = AsyncMock(return_value=docs)
    samples_fetch = AsyncMock(return_value=_samples())
    failed_fetch = docs_fetch if unavailable == "docs" else samples_fetch
    healthy_fetch = samples_fetch if unavailable == "docs" else docs_fetch
    failed_index = KCLDocs if unavailable == "docs" else KCLSamples
    healthy_tool = "list_kcl_samples" if unavailable == "docs" else "list_kcl_docs"
    failed_tool = "list_kcl_docs" if unavailable == "docs" else "list_kcl_samples"
    fetch_started = asyncio.Event()
    release_fetch = asyncio.Event()

    async def fail():
        fetch_started.set()
        await release_fetch.wait()
        raise RuntimeError("Unrelated service unavailable")

    failed_fetch.side_effect = fail
    monkeypatch.setattr(kcl_docs, "_fetch_docs_from_zoo_dev", docs_fetch)
    monkeypatch.setattr(kcl_samples, "_fetch_index_from_api", samples_fetch)

    async with asyncio.timeout(5), server._lifespan(server.mcp):
        try:
            await fetch_started.wait()
            # The healthy tool must finish while the other fetch is still blocked.
            first = await server.mcp.call_tool(healthy_tool, arguments={})
            assert isinstance(first, CallToolResult)
            assert not first.is_error
            assert first.content
            assert isinstance(first.content[0], TextContent)
            result = json.loads(first.content[0].text)
            if unavailable == "docs":
                assert result["name"] == "public-project"
            else:
                assert result["kcl-lang"] == [path]

            release_fetch.set()
            await asyncio.gather(
                server._kcl_index_tasks[failed_index], return_exceptions=True
            )
            cached = await server.mcp.call_tool(healthy_tool, arguments={})
            assert isinstance(cached, CallToolResult)
            assert cached.content == first.content
            assert healthy_fetch.await_count == 1
            assert failed_fetch.await_count == 1

            # Only a request for the unavailable index starts another attempt.
            with pytest.raises(ToolError, match="Unrelated service unavailable"):
                await server.mcp.call_tool(failed_tool, arguments={})
            assert failed_fetch.await_count == 2
            assert healthy_fetch.await_count == 1
        finally:
            release_fetch.set()

    assert not server._kcl_index_tasks
    assert KCLDocs._instance is None
    assert KCLSamples._instance is None


@pytest.mark.asyncio
async def test_cancelling_one_caller_preserves_the_shared_fetch(
    monkeypatch: pytest.MonkeyPatch,
):
    fetch_started = asyncio.Event()
    release_fetch = asyncio.Event()

    async def fetch() -> KCLSamples:
        fetch_started.set()
        await release_fetch.wait()
        return _samples()

    fetch_mock = AsyncMock(side_effect=fetch)
    monkeypatch.setattr(kcl_samples, "_fetch_index_from_api", fetch_mock)
    cancelled = asyncio.create_task(
        server.mcp.call_tool("list_kcl_samples", arguments={})
    )
    waiting = asyncio.create_task(
        server.mcp.call_tool("search_kcl_samples", arguments={"query": "gear"})
    )
    try:
        async with asyncio.timeout(5):
            await fetch_started.wait()
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            release_fetch.set()
            result = await waiting
        assert isinstance(result, CallToolResult)
        assert not result.is_error
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)
        assert json.loads(result.content[0].text)["name"] == "public-project"
        assert fetch_mock.await_count == 1
    finally:
        release_fetch.set()
        await asyncio.gather(cancelled, waiting, return_exceptions=True)
