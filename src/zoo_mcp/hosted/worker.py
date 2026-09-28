"""One persistent event loop, tool catalog and modeling session per owner."""

import asyncio
import os
import sys
from pathlib import Path

from zoo_mcp.hosted.protocol import encode, read_message
from zoo_mcp.hosted.sandbox import confine


async def serve() -> None:
    # Keep protocol output independent of Python and native-library stdout.
    output = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    confine(Path.cwd())

    from mcp.shared.exceptions import MCPError
    from mcp.types import CallToolResult, TextContent

    from zoo_mcp.server import mcp
    from zoo_mcp.zoo_tools import zoo_stop_all_modeling_sessions

    output.write(encode({"ready": True}))
    try:
        while message := await asyncio.to_thread(read_message, sys.stdin.buffer):
            if message.get("shutdown"):
                break
            credential = message["credential"]
            if not isinstance(credential, str) or not credential:
                raise ValueError("Missing delegated credential")
            # These libraries read process environment. This process belongs to
            # exactly one verified owner, and calls are serialized by its parent.
            os.environ["ZOO_API_TOKEN"] = credential
            os.environ["ZOO_TOKEN"] = credential
            try:
                result = await mcp.call_tool(message["name"], message["arguments"])
                response = {"result": result.model_dump(mode="json", by_alias=True)}
            except MCPError as error:
                response = {"error": error.error.model_dump(mode="json", by_alias=True)}
            except Exception as error:
                # Match MCPServer's tools/call exception conversion.
                result = CallToolResult(
                    is_error=True, content=[TextContent(type="text", text=str(error))]
                )
                response = {"result": result.model_dump(mode="json", by_alias=True)}
            # Credentials must never appear in SDK diagnostics returned to clients.
            wire = encode(response)
            if credential.encode() in wire:
                import json

                response = json.loads(
                    json.dumps(response).replace(credential, "<credential>")
                )
            output.write(encode(response))
    finally:
        await zoo_stop_all_modeling_sessions()
        output.close()


if __name__ == "__main__":
    asyncio.run(serve())
