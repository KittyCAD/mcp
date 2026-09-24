"""Exercise SDK authentication, discovery, and the restricted remote tool surface."""

import json
import sys
import time
from contextlib import asynccontextmanager

import httpx
import pytest
from mcp.types import TextContent

from zoo_mcp.http_auth import SCOPES, ZooOAuth, create_app
from zoo_mcp.server import mcp

RESOURCE = "https://api.example.test/mcp"


class FakeOAuth(ZooOAuth):
    def __init__(self):
        super().__init__(RESOURCE, "s" * 32)
        self.calls = []

    async def request(self, path, token):
        if path == "/mcp/delegate":
            return {"access_token": "delegated-test-credential"}
        return {
            "active": token != "invalid",
            "aud": RESOURCE if token != "wrong-audience" else "https://other.test/mcp",
            "exp": int(time.time()) + (600 if token != "expired" else -1),
            "client_id": "chatgpt",
            "sub": token,
            "scope": " ".join(SCOPES) if token != "missing-scopes" else "user:read",
        }

    async def call(self, token, name, arguments):
        self.calls.append((token, name, arguments))
        return await mcp.call_tool(name, arguments)


@asynccontextmanager
async def connection(auth):
    app = create_app(auth)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://api.example.test"
        ) as client,
    ):
        yield client


async def rpc(client, method, params=None, token="alice"):
    response = await client.post(
        "/mcp",
        headers={
            "authorization": f"Bearer {token}",
            "accept": "application/json, text/event-stream",
            "mcp-protocol-version": "2025-11-25",
        },
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}},
    )
    assert response.status_code == 200, response.text
    if response.headers["content-type"].startswith("text/event-stream"):
        return json.loads(
            next(
                line[6:]
                for line in response.text.splitlines()
                if line.startswith("data: ")
            )
        )
    return response.json()


@pytest.mark.asyncio
async def test_oauth_challenge_discovery_and_rejected_credentials():
    async with connection(FakeOAuth()) as client:
        response = await client.post("/mcp")
        assert response.status_code == 401
        assert (
            RESOURCE.replace("/mcp", "/.well-known/oauth-protected-resource/mcp")
            in response.headers["www-authenticate"]
        )
        metadata = (
            await client.get("/.well-known/oauth-protected-resource/mcp")
        ).json()
        assert metadata["resource"] == RESOURCE
        assert metadata["authorization_servers"] == ["https://api.example.test/"]
        assert (
            await client.get("/.well-known/oauth-protected-resource")
        ).json() == metadata
        for token in ("invalid", "expired", "wrong-audience"):
            assert (
                await client.post("/mcp", headers={"authorization": f"Bearer {token}"})
            ).status_code == 401
        assert (
            await client.post(
                "/mcp", headers={"authorization": "Bearer missing-scopes"}
            )
        ).status_code == 403


@pytest.mark.asyncio
async def test_authenticated_initialize_discovery_and_real_tool_call():
    auth = FakeOAuth()
    async with connection(auth) as client:
        initialized = await rpc(
            client,
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "ChatGPT", "version": "test"},
            },
        )
        assert "tools" in initialized["result"]["capabilities"]
        catalog = (await rpc(client, "tools/list"))["result"]["tools"]
        tools = {tool["name"]: tool for tool in catalog}
        assert {
            "execute_kcl",
            "format_kcl",
            "search_kcl_docs",
            "list_org_datasets",
        } <= tools.keys()
        assert (
            not {
                "start_modeling_session",
                "import_cad_file",
                "save_image",
                "export_kcl",
            }
            & tools.keys()
        )
        for name in ("execute_kcl", "format_kcl"):
            schema = tools[name]["inputSchema"]
            assert not {"kcl_path", "session_id"} & schema["properties"].keys()
            assert "kcl_code" in schema["required"]
        for token in ("alice", "bob"):
            result = await rpc(
                client,
                "tools/call",
                {"name": "format_kcl", "arguments": {"kcl_code": "x=1"}},
                token,
            )
            assert not result["result"].get("isError", False)
            assert "x = 1" in result["result"]["content"][0]["text"]
        assert [call[0] for call in auth.calls] == ["alice", "bob"]
        for name, arguments in (
            ("format_kcl", {"kcl_path": "/etc/passwd"}),
            ("execute_kcl", {"kcl_code": "x=1", "session_id": "someone-else"}),
            ("format_kcl", {}),
            ("save_image", {}),
        ):
            result = await rpc(
                client, "tools/call", {"name": name, "arguments": arguments}
            )
            assert result["result"]["isError"]
        assert len(auth.calls) == 2


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform != "linux", reason="Credential-bearing workers require Linux Landlock"
)
async def test_real_confined_worker_reuses_existing_tool_without_parent_credentials(
    monkeypatch,
):
    auth = FakeOAuth()
    monkeypatch.setenv("ZOO_API_TOKEN", "parent-credential-must-not-be-used")
    result = await ZooOAuth.call(auth, "alice", "format_kcl", {"kcl_code": "x=1"})
    assert not result.is_error
    assert isinstance(result.content[0], TextContent)
    assert "x = 1" in result.content[0].text


@pytest.mark.parametrize(
    "resource",
    [
        "http://api.test/mcp",
        "https://user:pass@api.test/mcp",
        "https://api.test/mcp?other=1",
        "https://api.test/other",
    ],
)
def test_oauth_configuration_rejects_unsafe_resource_urls(resource):
    with pytest.raises(ValueError):
        ZooOAuth(resource, "s" * 32)
