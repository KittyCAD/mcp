"""Authenticated Streamable HTTP, transfer capabilities, and MCP Apps resources."""

import asyncio
import contextvars
import hmac
import json
from contextlib import asynccontextmanager
from urllib.parse import urlsplit
from uuid import UUID

import httpx
import jsonschema
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import (
    CallToolResult,
    ImageContent,
    ListToolsResult,
    ResourceLink,
    TextContent,
)
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from zoo_mcp import __version__

from .backend import Backend, Principal, ServiceError
from .catalog import background, catalog, scope_for, scopes_for
from .config import Settings
from .runtime import Runtime

current_principal: contextvars.ContextVar[Principal] = contextvars.ContextVar(
    "zoo_mcp_principal"
)
current_disconnect: contextvars.ContextVar[asyncio.Event | None] = (
    contextvars.ContextVar("zoo_mcp_disconnect", default=None)
)


def create_app(
    settings: Settings | None = None, *, backend: Backend | None = None
) -> Starlette:
    settings = settings or (backend.settings if backend else Settings.from_env())
    http = (
        backend.http
        if backend
        else httpx.AsyncClient(
            timeout=httpx.Timeout(60, connect=5), follow_redirects=False
        )
    )
    backend = backend or Backend(settings, http)
    runtime = Runtime(backend)
    tools = {}

    async def tool_list():
        if not tools:
            tools.update({tool.name: tool for tool in await catalog()})
        return list(tools.values())

    async def list_tools(ctx, params):
        return ListToolsResult(tools=await tool_list())

    async def dispatch(p: Principal, name: str, arguments: dict) -> dict:
        p.require(*scopes_for(name))
        if name == "get_job":
            return await runtime.job(p, arguments["job_id"])
        if name == "cancel_job":
            return await runtime.cancel(p, arguments["job_id"])
        if name == "create_upload":
            row = await runtime.artifacts.create(
                p, arguments["name"], arguments["size_bytes"]
            )
            cap = runtime.artifacts.capabilities.issue(p, row["id"], "upload")
            return {
                "artifact_id": row["id"],
                "upload_url": f"{settings.origin}/mcp/files/{row['id']}?capability={cap}",
                "method": "PUT",
                "expires_in": settings.capability_seconds,
            }
        if name == "list_artifacts":
            return {
                "artifacts": [
                    runtime.artifacts.describe(p, r)
                    for r in await backend.list(p, "artifact")
                    if r["data"].get("status") == "ready"
                ]
            }
        if name == "get_artifact":
            row = await backend.get(p, str(UUID(arguments["artifact_id"])))
            if row["kind"] != "artifact":
                raise ServiceError("not_found", "Artifact not found.")
            return runtime.artifacts.describe(p, row)
        if name == "delete_artifact":
            row = await backend.get(p, str(UUID(arguments["artifact_id"])))
            if row["kind"] != "artifact":
                raise ServiceError("not_found", "Artifact not found.")
            await backend.delete(p, row["id"])
            return {"deleted": True}
        if name == "write_kcl_project":
            return await runtime.artifacts.write_source(p, arguments["files"])
        # Source is retained even when callers submit inline KCL.
        source = None
        if arguments.get("kcl_code") and name in {
            "execute_kcl",
            "exec_kcl_project",
            "export_kcl",
        }:
            source = await runtime.artifacts.write_source(
                p, {"main.kcl": arguments["kcl_code"]}
            )
        result = await runtime.call(p, name, arguments)
        if source:
            result["source"] = source
        return result

    async def call(
        p: Principal, name: str, arguments: dict, *, forwarded: bool = False
    ) -> dict:
        await tool_list()
        if name not in tools:
            raise ServiceError("unknown_tool", "Unknown Zoo tool.")
        jsonschema.validate(
            arguments,
            tools[name].input_schema,
            format_checker=jsonschema.FormatChecker(),
        )
        p.require(*scopes_for(name))
        operation_arguments = {
            key: value
            for key, value in arguments.items()
            if key not in {"execution_mode", "idempotency_key"}
        }
        if (
            arguments.get("execution_mode", "direct") == "background"
            and background(name)
            and not forwarded
        ):
            return await runtime.submit(
                p, name, arguments, lambda: dispatch(p, name, operation_arguments)
            )
        return await runtime.run_direct(
            lambda: dispatch(p, name, operation_arguments), current_disconnect.get()
        )

    async def call_tool(ctx, params) -> CallToolResult:
        try:
            data = await call(
                current_principal.get(), params.name, params.arguments or {}
            )
            content = [TextContent(type="text", text=json.dumps(data))]
            result = data.get("result", data)
            # Surface images and downloads as native MCP content, alongside the
            # structured direct result consumed by remote clients.
            for artifact in (
                result.get("artifacts", []) if isinstance(result, dict) else []
            ):
                if artifact.get("unavailable"):
                    continue
                mime = artifact.get("mime_type", "application/octet-stream")
                if (
                    mime in {"image/png", "image/jpeg"}
                    and artifact.get("size_bytes", 0) <= 1024 * 1024
                ):
                    import base64

                    _, image_bytes = await runtime.artifacts.read(
                        current_principal.get(), artifact["artifact_id"]
                    )
                    content.append(
                        ImageContent(
                            type="image",
                            mime_type=mime,
                            data=base64.b64encode(image_bytes).decode(),
                        )
                    )
                else:
                    content.append(
                        ResourceLink(
                            type="resource_link",
                            uri=artifact["download_url"],
                            name=artifact["name"],
                            mime_type=mime,
                        )
                    )
            return CallToolResult(content=content, structured_content=data)
        except ServiceError as error:
            return CallToolResult(
                is_error=True,
                content=[TextContent(type="text", text=str(error))],
                structured_content={"error": error.code, "message": str(error)},
            )
        except (ValueError, jsonschema.ValidationError):
            return CallToolResult(
                is_error=True,
                content=[TextContent(type="text", text="Invalid tool arguments.")],
            )
        except Exception:
            return CallToolResult(
                is_error=True,
                content=[
                    TextContent(
                        type="text",
                        text="Zoo could not complete the operation. Its outcome may be unknown; inspect the scene before retrying.",
                    )
                ],
            )

    server = Server(
        "Zoo",
        version=__version__,
        instructions=(
            "Use Zoo artifact IDs for remote files. Save editable KCL with write_kcl_project before execution. "
            "Tools return direct results by default. Eligible tools accept execution_mode=background with an idempotency_key; get_job polls persisted outcomes. Direct calls have no durable recovery and are never automatically retried. "
            "Restore expired scenes explicitly from saved source."
        ),
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )

    manager = StreamableHTTPSessionManager(
        server,
        stateless=True,
        json_response=False,
        max_request_body_size=2 * 1024 * 1024,
        security_settings=TransportSecuritySettings(
            allowed_hosts=[urlsplit(settings.resource).netloc],
            allowed_origins=[settings.origin, *settings.allowed_origins],
        ),
    )

    async def authenticate(request: Request) -> Principal:
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            raise ServiceError("reauthorize", "Connect your Zoo account to continue.")
        return await backend.principal(auth[7:])

    class MCP:
        async def __call__(self, scope, receive, send):
            disconnected = asyncio.Event()
            upstream_receive = receive

            async def observe_disconnect():
                message = await upstream_receive()
                if message["type"] == "http.disconnect":
                    disconnected.set()
                return message

            receive = observe_disconnect
            request = Request(scope, receive)
            try:
                p = await authenticate(request)
            except ServiceError as error:
                if error.code != "reauthorize":
                    response = JSONResponse(
                        {"error": "temporarily_unavailable"}, status_code=503
                    )
                    await response(scope, receive, send)
                    return
                response = JSONResponse(
                    {"error": "unauthorized"},
                    status_code=401,
                    headers={
                        "WWW-Authenticate": f'Bearer resource_metadata="{settings.origin}/.well-known/oauth-protected-resource/mcp"',
                        "Cache-Control": "no-store",
                    },
                )
                await response(scope, receive, send)
                return
            if request.method == "POST":
                raw = await request.body()
                try:
                    payload = json.loads(raw)
                except ValueError:
                    payload = {}
                if isinstance(payload, dict) and payload.get("method") == "tools/call":
                    params = payload.get("params")
                    name = params.get("name", "") if isinstance(params, dict) else ""
                    await tool_list()
                    if isinstance(name, str) and name in tools:
                        required = set(scopes_for(name))
                        if not required.issubset(p.scopes):
                            scopes = " ".join(sorted(p.scopes | required))
                            response = JSONResponse(
                                {"error": "insufficient_scope"},
                                status_code=403,
                                headers={
                                    "WWW-Authenticate": f'Bearer error="insufficient_scope", scope="{scopes}"',
                                    "Cache-Control": "no-store",
                                },
                            )
                            await response(scope, receive, send)
                            return
                original_receive = receive
                delivered = False

                async def replay():
                    nonlocal delivered
                    if not delivered:
                        delivered = True
                        return {"type": "http.request", "body": raw, "more_body": False}
                    return await original_receive()

                receive = replay
            disconnect_token = current_disconnect.set(disconnected)
            token = current_principal.set(p)
            try:
                await manager.handle_request(scope, receive, send)
            finally:
                current_principal.reset(token)
                current_disconnect.reset(disconnect_token)

    async def metadata(request):
        await tool_list()
        scopes = sorted({scope_for(name) for name in tools if scope_for(name)})
        return JSONResponse(
            {
                "resource": settings.resource,
                "authorization_servers": [settings.api_url],
                "scopes_supported": scopes,
                "resource_name": "Zoo",
                "resource_documentation": "https://zoo.dev/docs",
            }
        )

    async def internal_call(request: Request):
        supplied = request.headers.get("x-zoo-mcp-service-token", "")
        if not hmac.compare_digest(supplied, settings.service_secret):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        p = await authenticate(request)
        payload = await request.json()
        return JSONResponse(
            await call(p, payload["name"], payload["arguments"], forwarded=True)
        )

    async def transfer(request: Request):
        cors = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, PUT, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "600",
        }
        if request.method == "OPTIONS":
            return Response(status_code=204, headers=cors)
        artifact_id = str(UUID(request.path_params["artifact_id"]))
        action = "upload" if request.method == "PUT" else "download"
        capability = runtime.artifacts.capabilities.verify(
            request.query_params.get("capability", ""), artifact_id, action
        )
        params = {"grant_id": capability["grant"]}
        if action == "upload":
            chunks = bytearray()
            async for chunk in request.stream():
                if len(chunks) + len(chunk) > settings.max_file_bytes:
                    return JSONResponse({"error": "file_too_large"}, status_code=413)
                chunks.extend(chunk)
            response = await backend._request(
                "PUT",
                f"/mcp/transfers/{artifact_id}",
                params=params,
                content=bytes(chunks),
            )
            return JSONResponse(response.json(), headers=cors)
        response = await backend._request(
            "GET", f"/mcp/transfers/{artifact_id}", params=params
        )
        return Response(
            response.content,
            media_type=response.headers.get("content-type", "application/octet-stream"),
            headers={
                **cors,
                "Content-Disposition": response.headers.get(
                    "content-disposition", "attachment"
                ),
                "Cache-Control": "no-store",
                "Referrer-Policy": "no-referrer",
            },
        )

    async def metrics(request: Request):
        if not hmac.compare_digest(
            request.headers.get("x-zoo-mcp-service-token", ""), settings.service_secret
        ):
            return Response(status_code=403)
        lines = [
            f"zoo_mcp_workers {len(runtime.workers)}",
            f"zoo_mcp_running_jobs {len(runtime.jobs)}",
        ]
        lines.extend(
            f'zoo_mcp_outcomes_total{{outcome="{key}"}} {value}'
            for key, value in sorted(runtime.outcomes.items())
        )
        return Response("\n".join(lines) + "\n", media_type="text/plain")

    async def health(request):
        return JSONResponse(
            {"ready": not runtime.closing}, status_code=503 if runtime.closing else 200
        )

    async def service_error(request, error: Exception):
        assert isinstance(error, ServiceError)
        status = {
            "not_found": 404,
            "conflict": 409,
            "reauthorize": 401,
            "invalid_capability": 403,
            "insufficient_scope": 403,
        }.get(error.code, 400)
        return JSONResponse(
            {"error": error.code, "message": str(error)}, status_code=status
        )

    @asynccontextmanager
    async def lifespan(app):
        # Refuse readiness on a node where credential-bearing workers cannot be
        # confined. Run this in a child so the HTTP frontend is not restricted.
        if not settings.unsafe_local_dev:
            import os
            import sys

            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                "from pathlib import Path; from tempfile import TemporaryDirectory; from zoo_mcp.hosted.sandbox import confine; temp = TemporaryDirectory(); confine(Path(temp.name))",
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                code = await asyncio.wait_for(process.wait(), 15)
            except TimeoutError:
                process.kill()
                await process.wait()
                raise RuntimeError("Worker sandbox startup check timed out") from None
            if code:
                raise RuntimeError(
                    "Hosted MCP requires native Linux with Landlock ABI 3 or later"
                )
        task = asyncio.create_task(runtime.maintain())
        async with manager.run():
            try:
                yield
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await runtime.close()
                await http.aclose()

    app = Starlette(
        routes=[
            Route("/mcp", MCP(), methods=["GET", "POST", "DELETE"]),
            Route("/.well-known/oauth-protected-resource", metadata),
            Route("/.well-known/oauth-protected-resource/mcp", metadata),
            Route(
                "/mcp/files/{artifact_id}", transfer, methods=["GET", "PUT", "OPTIONS"]
            ),
            Route("/_internal/call", internal_call, methods=["POST"]),
            Route("/_internal/metrics", metrics),
            Route("/healthz", health),
            Route("/readyz", health),
        ],
        lifespan=lifespan,
        exception_handlers={ServiceError: service_error},
    )

    async def protect_requests(request: Request, call_next):
        if request.method in {"POST", "PUT"} and not request.url.path.startswith(
            "/mcp/files/"
        ):
            chunks = bytearray()
            async for chunk in request.stream():
                if len(chunks) + len(chunk) > 2 * 1024 * 1024:
                    return JSONResponse({"error": "request_too_large"}, status_code=413)
                chunks.extend(chunk)
            request._body = bytes(chunks)
        try:
            return await call_next(request)
        except (ValueError, KeyError, jsonschema.ValidationError):
            return JSONResponse({"error": "invalid_request"}, status_code=400)

    app.add_middleware(BaseHTTPMiddleware, dispatch=protect_requests)
    app.state.runtime = runtime
    app.state.call = call
    app.state.backend = backend
    return app


def main():
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8080, access_log=False)
