"""One grant/scene per subprocess; stdout is a private framed JSON channel."""

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp.types import CallToolResult, TextContent
from pydantic_core import to_jsonable_python


async def run() -> None:
    from .sandbox import confine

    workspace = Path.cwd()
    confine(
        workspace, unsafe_local_dev=os.environ.get("ZOO_MCP_UNSAFE_LOCAL_DEV") == "true"
    )
    from zoo_mcp.server import mcp
    from zoo_mcp.zoo_tools import zoo_stop_all_modeling_sessions

    try:
        while line := await asyncio.to_thread(sys.stdin.readline):
            try:
                message = json.loads(line)
                # This worker is permanently assigned to one grant. Renewing its
                # credential never changes identity or mutates the frontend's env.
                os.environ["ZOO_API_TOKEN"] = message["credential"]
                os.environ["ZOO_TOKEN"] = message["credential"]
                if message["tool"] == "export_kcl":
                    # Preserve every file in multi-file exports (OBJ/GLTF, etc.).
                    import zipfile
                    from uuid import uuid4

                    import kcl

                    from zoo_mcp.zoo_tools import KCLExportFormat

                    from .files import relative_path

                    args = message["arguments"]
                    format_name = args.get("export_format") or "step"
                    export_format = KCLExportFormat.formats.value[format_name]
                    if args.get("kcl_code"):
                        files = await kcl.execute_code_and_export(
                            args["kcl_code"], export_format
                        )
                    else:
                        files = await kcl.execute_and_export(
                            args["kcl_path"], export_format
                        )
                    output = workspace / str(uuid4())
                    output.mkdir()
                    if len(files) == 1:
                        target = output / relative_path(files[0].name)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(bytes(files[0].contents))
                    else:
                        target = output / "export.zip"
                        with zipfile.ZipFile(
                            target, "w", zipfile.ZIP_DEFLATED
                        ) as archive:
                            for file in files:
                                archive.writestr(
                                    relative_path(file.name), bytes(file.contents)
                                )
                    result = CallToolResult(
                        content=[TextContent(type="text", text=str(target))],
                        structured_content={"result": str(target)},
                    )
                else:
                    result = await mcp.call_tool(message["tool"], message["arguments"])
                if not isinstance(result, CallToolResult) or result.is_error:
                    raise ValueError("Unexpected modeling result")
                payload = {
                    "content": to_jsonable_python(result.content),
                    "structured": to_jsonable_python(result.structured_content),
                }
            except Exception:
                # Raw exception strings can include source code and local paths.
                payload = {
                    "error": "The CAD operation failed. Check the input and retry with a new session if needed."
                }
            sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
            sys.stdout.flush()
    finally:
        await zoo_stop_all_modeling_sessions()


if __name__ == "__main__":
    asyncio.run(run())
