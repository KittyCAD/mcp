"""Exercise the Zoo API Docs MCP contract over stdio or Streamable HTTP.

Usage:
    python scripts/smoke_api_docs.py
    python scripts/smoke_api_docs.py zoo-api-docs
    python scripts/smoke_api_docs.py --url https://mcp.zoo.dev/api-docs/mcp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

EXPECTED_TOOLS = {
    "get_zoo_api_guide",
    "get_zoo_api_operation",
    "get_zoo_api_schema",
    "search_zoo_api",
}


def _structured(result: Any) -> dict[str, Any]:
    if result.isError:
        raise RuntimeError(f"MCP tool returned an error: {result.content!r}")
    structured = result.structuredContent
    if not isinstance(structured, dict):
        raise RuntimeError("MCP tool did not return structured content")
    return structured


async def _exercise(read_stream: Any, write_stream: Any) -> dict[str, str]:
    async with ClientSession(
        read_stream,
        write_stream,
        read_timeout_seconds=timedelta(seconds=30),
    ) as session:
        await session.initialize()
        listed = await session.list_tools()
        names = {tool.name for tool in listed.tools}
        if names != EXPECTED_TOOLS:
            raise RuntimeError(f"unexpected Zoo API Docs tool set: {sorted(names)!r}")

        search = _structured(
            await session.call_tool(
                "search_zoo_api",
                {"query": "convert a STEP file to OBJ", "limit": 3},
            )
        )
        result_ids = {
            result.get("operation_id") or result.get("guide_id")
            for result in search.get("results", [])
        }
        if not result_ids.intersection(
            {
                "create_file_conversion",
                "create_file_conversion_options",
                "file-api-beginner",
            }
        ):
            raise RuntimeError(
                f"conversion search missed expected results: {result_ids}"
            )

        operation = _structured(
            await session.call_tool(
                "get_zoo_api_operation",
                {"operation_id": "create_file_conversion_options"},
            )
        )
        curl = operation.get("canonical_curl")
        if (
            not isinstance(curl, str)
            or "https://api.zoo.dev/file/conversion" not in curl
        ):
            raise RuntimeError("operation response has no canonical Zoo API curl")
        if "$ZOO_API_TOKEN" not in curl:
            raise RuntimeError("authenticated curl must use the token placeholder")

        schema_names = operation.get("referenced_schema_names", [])
        if not schema_names:
            raise RuntimeError("operation response has no referenced schemas")
        schema = _structured(
            await session.call_tool(
                "get_zoo_api_schema",
                {"schema_name": schema_names[0]},
            )
        )
        if schema.get("schema_name") != schema_names[0]:
            raise RuntimeError("schema lookup did not return the requested schema")

        guide = _structured(
            await session.call_tool(
                "get_zoo_api_guide",
                {"guide_id": "authentication"},
            )
        )
        if not guide.get("content") or guide.get("truncated") is not False:
            raise RuntimeError("authentication guide response is incomplete")

        return {
            "index_revision": str(operation["index_revision"]),
            "openapi_source_commit": str(operation["openapi_source_commit"]),
            "guide_source_revision": str(operation["guide_source_revision"]),
        }


@asynccontextmanager
async def _session_streams(
    url: str | None, command: list[str]
) -> AsyncIterator[tuple[Any, Any]]:
    if url:
        async with streamable_http_client(url) as (read_stream, write_stream, _):
            yield read_stream, write_stream
        return

    active_command = command or [sys.executable, "-m", "zoo_mcp.api_docs.server"]
    if active_command[0] == "--":
        active_command = active_command[1:]
    if not active_command:
        raise ValueError("stdio command must not be empty")

    child_environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"ZOO_API_TOKEN", "ZOO_TOKEN"}
    }
    parameters = StdioServerParameters(
        command=active_command[0],
        args=active_command[1:],
        env=child_environment,
    )
    async with stdio_client(parameters) as (read_stream, write_stream):
        yield read_stream, write_stream


async def _main(args: argparse.Namespace) -> None:
    if args.url and args.command:
        raise ValueError("pass either --url or a stdio command, not both")
    async with _session_streams(args.url, args.command) as streams:
        revisions = await _exercise(*streams)
    print(json.dumps({"status": "ok", **revisions}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Streamable HTTP MCP URL to smoke test.")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
