"""Run one existing tool with one delegated credential in a confined workspace."""

import asyncio
import json
import os
import sys
from pathlib import Path


async def main():
    from zoo_mcp.utils.sandbox import confine

    workspace = Path.cwd()
    confine(workspace)
    message = json.load(sys.stdin)
    credential = message["credential"]
    os.environ["ZOO_API_TOKEN"] = credential
    os.environ["ZOO_TOKEN"] = credential

    from zoo_mcp.server import mcp

    result = await mcp.call_tool(message["name"], message["arguments"])
    # Results may contain SDK diagnostics; never return the delegated credential.
    output = result.model_dump_json(by_alias=True).replace(credential, "<credential>")
    sys.stdout.write(output.replace(str(workspace), "<workspace>"))


if __name__ == "__main__":
    asyncio.run(main())
