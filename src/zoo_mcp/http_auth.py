"""Optional Zoo OAuth bridge for the SDK's Streamable HTTP server.

Inline tools reuse the local implementations in a fresh credential-isolated
process. Persistent scenes and remote file transfers belong to later slices.
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.lowlevel import Server
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, ListToolsResult, TextContent
from pydantic import AnyHttpUrl
from starlette.responses import JSONResponse
from starlette.routing import Route

DOC_TOOLS = {
    "list_kcl_docs",
    "search_kcl_docs",
    "get_kcl_doc",
    "list_kcl_samples",
    "search_kcl_samples",
    "get_kcl_sample",
}
DATASET_TOOLS = {"list_org_datasets", "list_org_skills", "search_org_dataset_semantic"}
INLINE_TOOLS = {
    "format_kcl",
    "lint_and_fix_kcl",
    "mock_execute_kcl",
    "execute_kcl",
    "calculate_kcl_physical_properties",
    "calculate_bounding_box_kcl",
    "get_sketch_constraint_status",
}
SCOPES = ["user:read", "modeling", "datasets:read"]


class ZooOAuth:
    def __init__(self, resource: str, secret: str):
        url = urlsplit(resource)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.path != "/mcp"
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("ZOO_MCP_RESOURCE must be an HTTPS /mcp URL")
        if len(secret) < 32:
            raise ValueError(
                "ZOO_MCP_SERVICE_SECRET must contain at least 32 characters"
            )
        self.resource = resource
        self.origin = f"https://{url.netloc}"
        self.secret = secret

    async def request(self, path: str, token: str) -> dict:
        async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
            response = await client.post(
                self.origin + path,
                headers={"X-Zoo-Mcp-Service-Token": self.secret},
                json={"token": token},
            )
            response.raise_for_status()
            return response.json()

    async def verify_token(self, token: str) -> AccessToken | None:
        info = await self.request("/mcp/introspect", token)
        if (
            not info.get("active")
            or info.get("aud") != self.resource
            or info.get("exp", 0) <= time.time()
        ):
            return None
        return AccessToken(
            token=token,
            client_id=info["client_id"],
            subject=info["sub"],
            scopes=info["scope"].split(),
            expires_at=info["exp"],
            resource=info["aud"],
        )

    async def call(self, token: str, name: str, arguments: dict) -> CallToolResult:
        credential = (await self.request("/mcp/delegate", token))["access_token"]
        with tempfile.TemporaryDirectory(prefix="zoo-mcp-") as directory:
            env = {
                k: v
                for k, v in os.environ.items()
                if k in {"PATH", "SSL_CERT_FILE", "SSL_CERT_DIR"}
            }
            env.update(
                {
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "TMPDIR": directory,
                    "ZOO_HOST": self.origin,
                    "ZOO_API_BASE_URL": self.origin,
                }
            )
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "zoo_mcp.http_worker",
                cwd=directory,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                async with asyncio.timeout(290):
                    output, _ = await process.communicate(
                        json.dumps(
                            {
                                "credential": credential,
                                "name": name,
                                "arguments": arguments,
                            }
                        ).encode()
                    )
                if process.returncode or len(output) > 8 * 1024 * 1024:
                    raise RuntimeError("The isolated tool process failed")
                return CallToolResult.model_validate_json(output)
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()


def error(message: str) -> CallToolResult:
    return CallToolResult(
        is_error=True, content=[TextContent(type="text", text=message)]
    )


def create_app(auth: ZooOAuth | None = None):
    from zoo_mcp.server import mcp

    auth = auth or ZooOAuth(
        os.environ["ZOO_MCP_RESOURCE"], os.environ["ZOO_MCP_SERVICE_SECRET"]
    )
    capacity = asyncio.Semaphore(4)
    tools = {}

    async def list_tools(_ctx, _params):
        if not tools:
            for original in await mcp.list_tools():
                if original.name not in DOC_TOOLS | DATASET_TOOLS | INLINE_TOOLS:
                    continue
                tool = original.model_copy(deep=True)
                schema = deepcopy(tool.input_schema)
                properties = schema["properties"]
                for key in ("kcl_path", "session_id"):
                    properties.pop(key, None)
                schema["additionalProperties"] = False
                if tool.name in INLINE_TOOLS:
                    schema["required"] = list(
                        dict.fromkeys([*schema.get("required", []), "kcl_code"])
                    )
                    tool.description = (
                        (tool.description or tool.name)
                        .split("\n\n")[0]
                        .split(" Either ")[0]
                    )
                    tool.description += " Supply inline kcl_code. This connection does not accept local paths or retain modeling sessions."
                tool.input_schema = schema
                scope = "datasets:read" if tool.name in DATASET_TOOLS else "modeling"
                scopes = (
                    ["user:read"] if tool.name in DOC_TOOLS else ["user:read", scope]
                )
                tool.meta = {
                    **(tool.meta or {}),
                    "securitySchemes": [{"type": "oauth2", "scopes": scopes}],
                }
                tools[tool.name] = tool
        return ListToolsResult(tools=list(tools.values()))

    async def call_tool(ctx, params):
        await list_tools(ctx, None)
        tool = tools.get(params.name)
        if tool is None:
            return error(
                "This tool requires the later persistent-session or file-transfer support."
            )
        arguments = params.arguments or {}
        if set(arguments) - tool.input_schema["properties"].keys():
            return error(
                "Only the advertised arguments are accepted; local paths and session IDs are unavailable."
            )
        if params.name in INLINE_TOOLS and (
            not isinstance(arguments.get("kcl_code"), str)
            or not arguments["kcl_code"].strip()
        ):
            return error("Supply nonempty inline kcl_code.")
        access = get_access_token()
        if access is None:
            return error("Reconnect Zoo to authenticate.")
        if capacity.locked():
            return error("Zoo tool capacity is busy. Try again shortly.")
        async with capacity:
            try:
                async with asyncio.timeout(295):
                    if params.name in DOC_TOOLS:
                        result = await mcp.call_tool(params.name, arguments)
                        assert isinstance(result, CallToolResult)
                        return result
                    return await auth.call(access.token, params.name, arguments)
            except (httpx.HTTPError, RuntimeError, ValueError, TimeoutError):
                return error(
                    "Zoo could not complete the tool call. Reconnect if authorization expired, then retry."
                )

    async def health(_request):
        return JSONResponse({"status": "ok"})

    server = Server(
        "Zoo MCP Server",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        instructions="Use inline KCL to format, lint, execute, and measure models. Documentation and organization dataset tools are available. File transfers, persistent scenes, and browser previews require later support.",
    )
    app = server.streamable_http_app(
        stateless_http=True,
        max_request_body_size=1024 * 1024,
        auth=AuthSettings(
            issuer_url=AnyHttpUrl(auth.origin),
            resource_server_url=AnyHttpUrl(auth.resource),
            required_scopes=SCOPES,
            validate_token_resource=True,
        ),
        token_verifier=auth,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[urlsplit(auth.resource).netloc],
            allowed_origins=["https://chatgpt.com", "https://claude.ai"],
        ),
        custom_starlette_routes=[Route("/healthz", health), Route("/readyz", health)],
    )
    metadata = next(
        route
        for route in app.routes
        if isinstance(route, Route)
        and route.path == "/.well-known/oauth-protected-resource/mcp"
    )
    app.routes.append(
        Route(
            "/.well-known/oauth-protected-resource",
            metadata.endpoint,
            methods=["GET", "OPTIONS"],
        )
    )
    return app
