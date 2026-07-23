"""Read-only MCP server for the pinned public Zoo API documentation index."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from zoo_mcp import logger
from zoo_mcp.api_docs.index import ZooApiDocsIndex

JsonObject = dict[str, Any]

# The SDK otherwise emits one INFO log per request.
# Keep documentation request content out of framework and access logs; the
# aggregate records in _invoke are the only per-tool application logs.
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

INDEX = ZooApiDocsIndex.load_bundled()
READ_ONLY_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)

mcp = FastMCP(
    name="Zoo API Docs",
    instructions=(
        "Search and retrieve pinned public Zoo API reference documentation. "
        "This server is read-only and never executes Zoo API operations."
    ),
    log_level="WARNING",
    host="127.0.0.1",
    port=8000,
    streamable_http_path="/api-docs/mcp",
    json_response=True,
    stateless_http=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "127.0.0.1:*",
            "localhost:*",
            "[::1]:*",
            "mcp.zoo.dev",
            "mcp.zoo.dev:*",
        ],
        allowed_origins=[
            "http://127.0.0.1:*",
            "http://localhost:*",
            "http://[::1]:*",
            "https://mcp.zoo.dev",
        ],
    ),
)


def _invoke(tool_name: str, call: Callable[[], JsonObject]) -> JsonObject:
    started = time.perf_counter()
    status = "ok"
    result_count = 1
    try:
        result = call()
        if "error" in result:
            status = "not_found"
            result_count = 0
        elif "results_count" in result:
            result_count = int(result["results_count"])
        return result
    except Exception:
        status = "error"
        result_count = 0
        raise
    finally:
        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "tool=%s status=%s latency_ms=%d result_count=%d source_revision=%s",
            tool_name,
            status,
            latency_ms,
            result_count,
            INDEX.index_revision,
        )


@mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
async def search_zoo_api(
    query: Annotated[
        str,
        Field(
            min_length=1,
            max_length=500,
            description="Terms describing an endpoint, path, schema, or guide.",
        ),
    ],
    tag: Annotated[
        str | None,
        Field(description="Optional exact OpenAPI or guide tag filter."),
    ] = None,
    limit: Annotated[
        int,
        Field(ge=1, le=10, description="Maximum number of ranked results."),
    ] = 5,
    include_deprecated: Annotated[
        bool,
        Field(description="Include deprecated operations in search results."),
    ] = False,
) -> JsonObject:
    """Search public Zoo operations and allowlisted developer guides.

    Exact operation IDs and paths rank first.
    Search is deterministic and lexical; it does not use embeddings or an LLM.
    """
    return _invoke(
        "search_zoo_api",
        lambda: INDEX.search(
            query,
            tag=tag,
            limit=limit,
            include_deprecated=include_deprecated,
        ),
    )


@mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
async def get_zoo_api_operation(
    operation_id: Annotated[
        str,
        Field(description="Exact OpenAPI operationId returned by search_zoo_api."),
    ],
) -> JsonObject:
    """Get one public operation and a placeholder-only canonical curl example."""
    return _invoke("get_zoo_api_operation", lambda: INDEX.get_operation(operation_id))


@mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
async def get_zoo_api_schema(
    schema_name: Annotated[
        str,
        Field(description="Exact public OpenAPI component schema name."),
    ],
) -> JsonObject:
    """Get one OpenAPI schema without recursively expanding its references."""
    return _invoke("get_zoo_api_schema", lambda: INDEX.get_schema(schema_name))


@mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
async def get_zoo_api_guide(
    guide_id: Annotated[
        str,
        Field(description="Exact guide ID returned by search_zoo_api."),
    ],
) -> JsonObject:
    """Get an allowlisted official guide, capped at 30,000 characters."""
    return _invoke("get_zoo_api_guide", lambda: INDEX.get_guide(guide_id))


@mcp.custom_route("/healthz", methods=["GET"], include_in_schema=False)
async def healthz(_request: Request) -> Response:
    """Return readiness and revisions without exposing document contents."""
    health = INDEX.health()
    return JSONResponse(health, status_code=200 if health["ready"] else 503)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Alias for --transport streamable-http.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    transport = "streamable-http" if args.http else args.transport
    if transport == "streamable-http":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
    try:
        mcp.run(transport=transport)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
