"""Hosted contracts at the authentication, transport, artifact and worker boundaries."""

import asyncio
import copy
import io
import json
import stat
import time
import zipfile
from dataclasses import replace
from uuid import uuid4

import httpx
import pytest
from starlette.testclient import TestClient

from zoo_mcp.hosted.app import create_app
from zoo_mcp.hosted.backend import Backend, Principal, ServiceError
from zoo_mcp.hosted.catalog import catalog
from zoo_mcp.hosted.config import Settings
from zoo_mcp.hosted.files import Artifacts, Capabilities, unpack_project
from zoo_mcp.hosted.runtime import Runtime

SCOPES = frozenset(
    {
        "modeling",
        "files:read",
        "files:write",
        "projects:read",
        "projects:write",
        "projects:manage",
        "datasets:read",
    }
)


def settings(**kwargs):
    return Settings(
        resource="https://mcp.test/mcp",
        api_url="https://api.test",
        service_secret="s" * 32,
        capability_secret="c" * 32,
        **kwargs,
    )


def principal():
    return Principal(
        str(uuid4()), str(uuid4()), str(uuid4()), None, SCOPES, "mcp-test-token"
    )


class MemoryBackend(Backend):
    def __init__(self, config=None):
        super().__init__(config or settings(unsafe_local_dev=True), httpx.AsyncClient())
        self.owner = principal()
        self.rows = {}
        self.blobs = {}
        self.revoked = False
        self.delegations = 0

    async def principal(self, token):
        if token != self.owner.token or self.revoked:
            raise ServiceError("reauthorize", "Reconnect Zoo.")
        return self.owner

    async def delegate(self, principal):
        p = principal
        await self.principal(p.token)
        self.delegations += 1
        return "delegated-test-token"

    async def get(self, p, record_id):
        row = self.rows.get(record_id)
        if row is None or row["grant_id"] != p.grant_id:
            raise ServiceError("not_found", "Not found.")
        return copy.deepcopy(row)

    async def list(self, p, kind):
        return [
            copy.deepcopy(r)
            for r in self.rows.values()
            if r["grant_id"] == p.grant_id and r["kind"] == kind
        ]

    async def put(self, p, record_id, kind, data, revision=0):
        row = self.rows.get(record_id)
        if (row and (row["revision"] != revision or row["grant_id"] != p.grant_id)) or (
            not row and revision
        ):
            raise ServiceError("conflict", "Revision conflict.")
        self.rows[record_id] = {
            "id": record_id,
            "grant_id": p.grant_id,
            "kind": kind,
            "data": copy.deepcopy(data),
            "revision": revision + 1,
            "expires": "2099-01-01T00:00:00Z",
        }
        return await self.get(p, record_id)

    async def delete(self, p, record_id):
        await self.get(p, record_id)
        del self.rows[record_id]

    async def api(self, principal, method, path, **kwargs):
        p = principal
        record_id = path.rsplit("/", 1)[-1]
        await self.get(p, record_id)
        if method == "PUT":
            self.blobs[record_id] = kwargs["content"]
            return httpx.Response(204)
        return httpx.Response(200, content=self.blobs[record_id])


@pytest.mark.asyncio
async def test_audience_expiry_and_delegation_boundary():
    p = principal()
    info = {
        "active": True,
        "sub": p.user_id,
        "client_id": p.client_id,
        "grant_id": p.grant_id,
        "scope": "modeling",
        "aud": "https://mcp.test/mcp",
        "exp": int(time.time()) + 60,
    }
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.path == "/mcp/introspect":
            assert json.loads(request.content) == {"token": p.token}
            assert "authorization" not in request.headers
            return httpx.Response(200, json=info)
        if request.url.path == "/mcp/delegate":
            assert "authorization" not in request.headers
            return httpx.Response(200, json={"access_token": "delegated-token"})
        assert request.headers["authorization"] == "Bearer delegated-token"
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        backend = Backend(settings(), client)
        assert (await backend.principal(p.token)).grant_id == p.grant_id
        await backend.api(p, "GET", "/user/projects")
        info["aud"] = "https://api.test"
        assert await backend.verify_token(p.token) is None
        info["aud"] = "https://mcp.test/mcp"
        info["exp"] = 0
        assert await backend.verify_token(p.token) is None
    assert all(r.headers["x-zoo-mcp-service-token"] == "s" * 32 for r in requests)


def test_capabilities_bind_action_artifact_and_expiry(monkeypatch):
    caps = Capabilities(settings())
    p = principal()
    artifact_id = str(uuid4())
    token = caps.issue(p, artifact_id, "upload")
    assert p.token not in token
    assert caps.verify(token, artifact_id, "upload")["grant"] == p.grant_id
    for value, ident, action in [
        (token, artifact_id, "download"),
        (token, str(uuid4()), "upload"),
        (token + "x", artifact_id, "upload"),
    ]:
        with pytest.raises(ServiceError, match="invalid or expired"):
            caps.verify(value, ident, action)
    monkeypatch.setattr(time, "time", lambda: 99999999999)
    with pytest.raises(ServiceError):
        caps.verify(token, artifact_id, "upload")


@pytest.mark.parametrize(
    "name",
    [
        "../escape.kcl",
        "/escape.kcl",
        "a/../../escape",
        "a\\escape",
        "a/./escape",
        ".env",
        "a/.secret",
        "C:/secret",
    ],
)
def test_archive_rejects_paths_before_writing(tmp_path, name):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("main.kcl", "safe")
        archive.writestr(name, "unsafe")
    with pytest.raises(ServiceError):
        unpack_project(buf.getvalue(), tmp_path, settings())
    assert list(tmp_path.iterdir()) == []


def test_archive_symlink_duplicates_and_expansion_limit(tmp_path):
    for entries in [[("Main.kcl", "a"), ("main.kcl", "b")], [("main.kcl", "x" * 100)]]:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            for name, data in entries:
                archive.writestr(name, data)
        with pytest.raises(ServiceError):
            unpack_project(buf.getvalue(), tmp_path, settings(max_project_bytes=20))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        symlink = zipfile.ZipInfo("main.kcl")
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(symlink, "/etc/passwd")
    with pytest.raises(ServiceError):
        unpack_project(buf.getvalue(), tmp_path, settings())


@pytest.mark.asyncio
async def test_upload_cannot_be_replaced_and_cross_grant_is_hidden():
    backend = MemoryBackend()
    artifacts = Artifacts(backend)
    p = backend.owner
    row = await artifacts.create(p, "cube.kcl", 3)
    first, second = await asyncio.gather(
        artifacts.complete(p, row, b"abc"),
        artifacts.complete(p, row, b"xyz"),
        return_exceptions=True,
    )
    assert isinstance(first, dict) and isinstance(second, ServiceError)
    assert (await artifacts.read(p, row["id"]))[1] == b"abc"
    with pytest.raises(ServiceError):
        await artifacts.read(principal(), row["id"])
    with pytest.raises(ServiceError):
        await artifacts.create(replace(p, scopes=frozenset()), "cube", 1)
    await backend.http.aclose()


@pytest.mark.asyncio
async def test_catalog_preserves_tools_without_exposing_filesystem():
    from zoo_mcp.server import mcp

    hosted = {t.name: t for t in await catalog()}
    assert {t.name for t in await mcp.list_tools()} <= hosted.keys()
    for tool in hosted.values():
        assert (
            not {
                "input_file",
                "kcl_path",
                "path_artifact_graph",
                "output_path",
                "export_path",
            }
            & tool.input_schema["properties"].keys()
        )


def test_streamable_http_discovery_auth_and_initialize():
    backend = MemoryBackend()
    app = create_app(backend=backend)
    with TestClient(app, base_url="https://mcp.test") as client:
        response = client.get("/mcp")
        assert response.status_code == 401
        assert "oauth-protected-resource/mcp" in response.headers["www-authenticate"]
        metadata = client.get("/.well-known/oauth-protected-resource/mcp").json()
        assert metadata["resource"] == "https://mcp.test/mcp"
        headers = {
            "Authorization": "Bearer " + backend.owner.token,
            "Accept": "application/json, text/event-stream",
        }
        response = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
            },
        )
        assert response.status_code == 200 and "serverInfo" in response.text
        response = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "list_artifacts", "arguments": {}},
            },
        )
        assert response.status_code == 200 and "artifacts" in response.text
        assert (
            client.post(
                "/mcp",
                headers=headers,
                content=b"x" * (2 * 1024 * 1024 + 1),
            ).status_code
            == 413
        )
        assert client.post("/_internal/call", json={}).status_code == 403
        backend.owner = replace(backend.owner, scopes=frozenset({"modeling"}))
        step_up = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "create_upload",
                    "arguments": {"name": "x.kcl", "size_bytes": 1},
                },
            },
        )
        assert step_up.status_code == 403
        assert "insufficient_scope" in step_up.headers["www-authenticate"]
        assert "files:write" in step_up.headers["www-authenticate"]
        assert "modeling" in step_up.headers["www-authenticate"]
        backend.revoked = True
        assert client.post("/mcp", headers=headers, json={}).status_code == 401


@pytest.mark.asyncio
async def test_isolated_worker_does_not_inherit_service_credentials(monkeypatch):
    backend = MemoryBackend(settings(unsafe_local_dev=True))
    runtime = Runtime(backend)
    monkeypatch.setenv("ZOO_MCP_SERVICE_SECRET", "private-service-key")
    monkeypatch.setenv("KITTYCAD_API_TOKEN", "private-legacy-key")
    from zoo_mcp.hosted import runtime as module

    real = asyncio.create_subprocess_exec
    environments = []

    async def spawn(*args, **kwargs):
        environments.append(kwargs["env"])
        return await real(*args, **kwargs)

    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", spawn)
    worker = await runtime.start_worker(backend.owner)
    try:
        result = await runtime.invoke(
            backend.owner, worker, "format_kcl", {"kcl_code": "x=1"}
        )
        assert "x" in json.dumps(result)
        assert (
            "ZOO_MCP_SERVICE_SECRET" not in environments[0]
            and "KITTYCAD_API_TOKEN" not in environments[0]
        )
        with pytest.raises(ServiceError):
            await runtime.invoke(principal(), worker, "format_kcl", {"kcl_code": "x=1"})
        with pytest.raises(ServiceError):
            await runtime.invoke(
                backend.owner, worker, "format_kcl", {"kcl_path": "/etc/passwd"}
            )
    finally:
        await runtime.close()
        await backend.http.aclose()
