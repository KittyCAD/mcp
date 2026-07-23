import os
import subprocess
import sys
from typing import Any, cast

import pytest
from starlette.requests import Request

from zoo_mcp.api_docs.server import healthz, mcp


@pytest.mark.asyncio
async def test_docs_profile_exposes_only_four_read_only_tools():
    tools = await mcp.list_tools()
    assert [tool.name for tool in tools] == [
        "search_zoo_api",
        "get_zoo_api_operation",
        "get_zoo_api_schema",
        "get_zoo_api_guide",
    ]
    for tool in tools:
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False
        assert tool.annotations.idempotentHint is True
        assert tool.annotations.openWorldHint is False


@pytest.mark.asyncio
async def test_docs_tools_return_structured_results():
    response = cast(
        tuple[list[object], dict[str, Any]],
        await mcp.call_tool("search_zoo_api", {"query": "convert CAD file"}),
    )
    result = response[1]
    assert isinstance(result, dict)
    assert result["results_count"] > 0

    operation_response = cast(
        tuple[list[object], dict[str, Any]],
        await mcp.call_tool(
            "get_zoo_api_operation",
            {"operation_id": "create_file_conversion_options"},
        ),
    )
    operation = operation_response[1]
    assert isinstance(operation, dict)
    assert operation["canonical_curl"]
    assert operation["openapi_source_commit"]
    assert operation["guide_source_revision"]


@pytest.mark.asyncio
async def test_health_exposes_revisions_but_no_documents():
    response = await healthz(cast(Request, None))
    body = response.body.decode()

    assert response.status_code == 200
    assert '"ready":true' in body
    assert '"index_revision"' in body
    assert '"operations"' not in body
    assert '"guides"' not in body


def test_docs_import_starts_without_token_or_authenticated_runtime():
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"ZOO_API_TOKEN", "ZOO_TOKEN"}
    }
    code = """
import asyncio
import sys
from zoo_mcp.api_docs.server import mcp
assert 'kittycad' not in sys.modules
assert 'zoo_mcp.zoo_tools' not in sys.modules
async def check():
    assert len(await mcp.list_tools()) == 4
asyncio.run(check())
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_docs_entrypoint_help_requires_no_token():
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"ZOO_API_TOKEN", "ZOO_TOKEN"}
    }
    result = subprocess.run(
        [sys.executable, "-m", "zoo_mcp.api_docs.server", "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "streamable-http" in result.stdout


@pytest.mark.asyncio
async def test_tool_logging_does_not_include_search_text(caplog):
    query = "private-customer-search-text"
    with caplog.at_level("INFO", logger="zoo_mcp"):
        await mcp.call_tool("search_zoo_api", {"query": query})

    assert query not in caplog.text
    assert "tool=search_zoo_api" in caplog.text
    assert "latency_ms=" in caplog.text
    assert "result_count=" in caplog.text
