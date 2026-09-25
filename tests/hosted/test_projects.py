import json
from dataclasses import replace
from uuid import uuid4

import httpx
import pytest

from tests.hosted.test_hosted import MemoryBackend
from zoo_mcp.hosted.app import create_app
from zoo_mcp.hosted.backend import ServiceError


class ProjectBackend(MemoryBackend):
    def __init__(self):
        super().__init__()
        self.requests = []

    async def api(self, principal, method, path, **kwargs):
        if path.startswith("/mcp/"):
            return await super().api(principal, method, path, **kwargs)
        self.requests.append((method, path, kwargs))
        return httpx.Response(200, json={"id": path.rsplit("/", 1)[-1]})


@pytest.mark.asyncio
async def test_project_updates_preserve_revision_and_source_tree():
    backend = ProjectBackend()
    app = create_app(backend=backend)
    p = backend.owner
    source = await app.state.runtime.artifacts.write_source(
        p, {"main.kcl": "x=1", "parts/leg.kcl": "y=2"}
    )
    project_id = str(uuid4())
    args = {
        "project_id": project_id,
        "project_artifact_id": source["artifact_id"],
        "title": "Desk",
        "expected_revision": "revision-one",
        "deleted_paths": ["old.kcl"],
    }
    try:
        await app.state.call(p, "update_project", args)
        method, path, request = backend.requests[-1]
        assert method == "PUT" and path == f"/user/projects/{project_id}"
        body = json.loads(request["files"][0][1][1])
        assert body["expected_revision"] == "revision-one"
        assert body["deleted_paths"] == ["old.kcl"]
        assert {item[1][0] for item in request["files"][1:]} == {
            "main.kcl",
            "parts/leg.kcl",
        }
        with pytest.raises(ServiceError):
            await app.state.call(
                p, "update_project", {**args, "deleted_paths": ["../outside.kcl"]}
            )
        with pytest.raises(ServiceError, match="permissions"):
            await app.state.call(
                replace(p, scopes=frozenset({"projects:read"})), "update_project", args
            )
    finally:
        await app.state.runtime.close()
        await backend.http.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "method", "suffix"),
    [
        ("get_project", "GET", ""),
        ("publish_project", "POST", "/publish"),
        ("delete_project", "DELETE", ""),
        ("list_project_share_links", "GET", "/share-links"),
        ("create_project_share_link", "POST", "/share-links"),
        ("move_project_to_organization", "PUT", "/organization"),
        ("move_project_to_personal", "DELETE", "/organization"),
    ],
)
async def test_project_operations_use_existing_authorized_endpoints(
    tool, method, suffix
):
    backend = ProjectBackend()
    app = create_app(backend=backend)
    ident = str(uuid4())
    try:
        result = await app.state.call(backend.owner, tool, {"project_id": ident})
        assert "job_id" not in result
        assert backend.requests[-1][:2] == (method, f"/user/projects/{ident}{suffix}")
    finally:
        await app.state.runtime.close()
        await backend.http.aclose()
