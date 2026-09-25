"""Exercise the entry point through real stdio and HTTP MCP connections."""

import asyncio
import sys

import pytest
from mcp import Client, StdioServerParameters
from mcp.types import CallToolResult, TextContent

from zoo_mcp.server import mcp

# Keep these transport checks offline while using the real module entry point,
# tool registrations, and SDK lifespan/shutdown handling.
_SERVER = """
import runpy
from unittest.mock import AsyncMock
from zoo_mcp import server

server._init_kcl_indexes = AsyncMock()
runpy.run_module("zoo_mcp", run_name="__main__")
"""


async def _check_existing_tools(client: Client) -> None:
    tools = await client.list_tools()
    assert {tool.name for tool in tools.tools} == {
        tool.name for tool in await mcp.list_tools()
    }
    result = await client.call_tool("format_kcl", {"kcl_code": "x=1"})
    assert isinstance(result, CallToolResult)
    assert not result.is_error
    assert any(
        isinstance(content, TextContent) and "x = 1" in content.text
        for content in result.content
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [[], ["--transport", "stdio"]])
async def test_stdio_default_and_explicit_transport(arguments):
    command = StdioServerParameters(
        command=sys.executable, args=["-c", _SERVER, *arguments]
    )
    async with asyncio.timeout(15), Client(command) as client:
        await _check_existing_tools(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_streamable_http_serves_existing_tools(unused_tcp_port, mode):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _SERVER,
        "--transport",
        "streamable-http",
        "--host",
        "127.0.0.1",
        "--port",
        str(unused_tcp_port),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(15):
            while True:
                if process.returncode is not None:
                    _, stderr = await process.communicate()
                    pytest.fail(f"HTTP server exited: {stderr.decode()}")
                try:
                    _, writer = await asyncio.open_connection(
                        "127.0.0.1", unused_tcp_port
                    )
                except OSError:
                    await asyncio.sleep(0.05)
                else:
                    writer.close()
                    await writer.wait_closed()
                    break
            async with Client(
                f"http://127.0.0.1:{unused_tcp_port}/mcp", mode=mode
            ) as client:
                await _check_existing_tools(client)
    finally:
        if process.returncode is None:
            process.terminate()
        try:
            await asyncio.wait_for(process.communicate(), 5)
        except TimeoutError:
            process.kill()
            await process.communicate()
            pytest.fail("HTTP server did not shut down after termination")
