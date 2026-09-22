import asyncio
import json
import socket

import pytest
from starlette.testclient import TestClient

from tests.hosted.test_hosted import MemoryBackend, settings
from zoo_mcp.hosted.app import create_app
from zoo_mcp.hosted.backend import ServiceError
from zoo_mcp.hosted.files import Artifacts


def rpc(response):
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        return next(
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: {")
        )
    return response.json()


@pytest.mark.asyncio
async def test_attachment_rejects_untrusted_and_private_dns(monkeypatch):
    backend = MemoryBackend(settings(file_hosts=("files.example.test",)))
    p = backend.owner
    with pytest.raises(ServiceError):
        await Artifacts(backend).import_openai_file(
            p, {"download_url": "https://evil.test/file", "file_id": "x"}
        )

    async def private(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", private)
    with pytest.raises(ServiceError, match="public addresses"):
        await Artifacts(backend).import_openai_file(
            p, {"download_url": "https://files.example.test/file", "file_id": "x"}
        )
    await backend.http.aclose()


def test_modern_http_routing_and_ui_resource():
    backend = MemoryBackend()
    with TestClient(create_app(backend=backend), base_url="https://mcp.test") as client:
        headers = {
            "Authorization": "Bearer " + backend.owner.token,
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2026-07-28",
            "Mcp-Method": "tools/call",
            "Mcp-Name": "open_zoo_workspace",
        }
        meta = {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientCapabilities": {},
        }
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "open_zoo_workspace", "arguments": {}, "_meta": meta},
        }
        response = client.post("/mcp", headers=headers, json=payload)
        assert response.status_code == 200
        assert rpc(response)["result"]["structuredContent"]["workspace_url"].endswith(
            "/mcp/workspace"
        )
        # The proxy must preserve routing headers so mismatches are rejected.
        invalid = client.post(
            "/mcp", headers={**headers, "Mcp-Name": "delete_project"}, json=payload
        )
        assert "error" in rpc(invalid)
        headers["Mcp-Method"] = "resources/read"
        headers["Mcp-Name"] = "ui://zoo/workspace-v1.html"
        resource = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "resources/read",
                "params": {"uri": "ui://zoo/workspace-v1.html", "_meta": meta},
            },
        )
        content = rpc(resource)["result"]["contents"][0]
        assert content["mimeType"] == "text/html;profile=mcp-app"
        assert "<script>" in content["text"]
        assert content["_meta"]["ui"]["csp"]["connectDomains"] == ["https://mcp.test"]
