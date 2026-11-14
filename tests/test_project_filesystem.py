"""Cloud filesystem replacements against an in-memory HTTP Project API."""

import asyncio
import base64
import copy
import hashlib
import json
from email import policy
from email.parser import BytesParser
from uuid import NAMESPACE_URL, uuid5

import httpx
import pytest
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent

from zoo_mcp import ZooMCPException, client_config, zoo_tools
from zoo_mcp.client_config import ZooClientConfig, use_client_config
from zoo_mcp.server import mcp
from zoo_mcp.zoo_tools import ProjectFileSystem

PROJECT_ID = "cloud:00000000-0000-4000-8000-000000000001"
CLOUD_STAMP = (
    '\n[cloud."example.test"]\nproject_id = "' + PROJECT_ID[6:] + '"\n'
).encode()


class ProjectAPI:
    def __init__(self):
        self.files: dict[str, bytes] = {
            "main.kcl": b"// main",
            "project.toml": b"[settings]\n" + CLOUD_STAMP,
            "parts/a.bin": b"\xff\x00",
        }
        self.revision = 1
        self.requests = []
        self.uploads = []
        self.failure = None
        self.editable = True
        self.download_hook = None
        self.versions = {}
        self.download_contents: bytes | None = None

    def file_id(self, path):
        return str(uuid5(NAMESPACE_URL, f"{self.revision}/{path}"))

    def metadata(self):
        self.versions[str(self.revision)] = {
            self.file_id(path): data for path, data in self.files.items()
        }
        return {
            "id": PROJECT_ID[6:],
            "revision": str(self.revision),
            "title": "My project",
            "description": "Keep this description",
            "category_ids": ["category"],
            "publication_status": "published",
            "entrypoint_path": "main.kcl",
            "project_toml_path": "project.toml",
            "access": {"can_edit": self.editable, "can_delete": False},
            "files": [
                {
                    "relative_path": path,
                    "byte_size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "id": self.file_id(path),
                    "created_at": "created",
                    "updated_at": "updated",
                    "content_type": "application/octet-stream",
                    "file_role": "preview_image"
                    if path == "thumbnail.png"
                    else "project_file",
                }
                for path, data in self.files.items()
            ],
        }

    def handle(self, request):
        self.requests.append(request)
        assert request.headers["Authorization"] == "Bearer test-token"
        assert request.url.path.startswith("/custom/user/projects/")
        if request.method == "GET" and "/versions/" in request.url.path:
            project_path, revision, files_segment, file_id = request.url.path.rsplit(
                "/", 3
            )
            assert project_path.endswith("/versions")
            assert files_segment == "files"
            content = self.versions.get(revision, {}).get(file_id)
            if content is None:
                return httpx.Response(404)
            if self.download_hook:
                hook, self.download_hook = self.download_hook, None
                hook()
            return httpx.Response(
                200,
                content=content
                if self.download_contents is None
                else self.download_contents,
                headers={"X-Project-Revision": revision, "X-Artifact-Id": file_id},
            )
        if request.method == "GET":
            assert request.url.path == f"/custom/user/projects/{PROJECT_ID[6:]}"
            return httpx.Response(200, json=self.metadata())
        assert request.method == "PUT"
        mime = BytesParser(policy=policy.default).parsebytes(
            b"Content-Type: "
            + request.headers["Content-Type"].encode()
            + b"\r\n\r\n"
            + request.content
        )
        files: dict[str, bytes] = {}
        body = None
        for part in mime.iter_parts():
            payload = part.get_payload(decode=True)
            assert isinstance(payload, bytes)
            if part.get_param("name", header="content-disposition") == "body":
                body = json.loads(payload)
            else:
                filename = part.get_filename()
                assert isinstance(filename, str)
                files[filename] = payload
        assert body is not None
        self.uploads.append((body, files))
        if self.failure == "conflict":
            self.failure = None
            self.files["concurrent.txt"] = b"preserve concurrent edit"
            self.revision += 1
            return httpx.Response(409)
        if body["expected_revision"] != str(self.revision):
            return httpx.Response(409)
        assert body["title"] == "My project"
        assert body["description"] == "Keep this description"
        assert body["category_ids"] == ["category"]
        assert set(body["deleted_paths"]) == set(self.files) - set(files) - {
            "thumbnail.png"
        }
        if self.failure == "timeout_before":
            self.failure = None
            raise httpx.ReadTimeout("response lost", request=request)
        if self.failure == "ambiguous":
            self.failure = None
            self.files["parts/a.bin"] = b"someone else's edit"
            self.revision += 1
            raise httpx.ReadTimeout("response lost", request=request)
        if self.failure == "forbidden":
            return httpx.Response(403)
        self.files = files
        self.revision += 1
        if self.failure == "timeout_after":
            self.failure = None
            raise httpx.ReadTimeout("response lost", request=request)
        return httpx.Response(200, json=self.metadata())


@pytest.fixture
def api(monkeypatch, httpx_mock):
    monkeypatch.setattr(
        client_config,
        "_local_config",
        ZooClientConfig("test-token", "https://api.example.test/custom/"),
    )
    remote = ProjectAPI()
    httpx_mock.add_callback(remote.handle, is_reusable=True, is_optional=True)
    return remote


@pytest.fixture
def project_transport(monkeypatch):
    def install(handler):
        monkeypatch.setattr(
            ProjectFileSystem,
            "_client",
            lambda _: httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
                headers={"Authorization": "Bearer test-token"},
                follow_redirects=False,
            ),
        )

    return install


@pytest.mark.asyncio
async def test_write_preserves_binary_metadata_and_arbitrary_field_names(api):
    fs = ProjectFileSystem(PROJECT_ID)
    await fs.write_file("body", "new contents")
    assert api.files["parts/a.bin"] == b"\xff\x00"
    assert api.files["body"] == b"new contents"
    assert len(api.uploads) == 1
    assert api.uploads[0][0]["deleted_paths"] == []


@pytest.mark.asyncio
async def test_empty_directory_and_marker_survive_addition(api):
    fs = ProjectFileSystem(PROJECT_ID)
    await fs.mkdir("empty")
    await fs.write_file("empty/child.kcl", "// child")
    assert api.files["empty/.empty"] == b""
    assert (await fs.stat("empty"))["kind"] == "directory"
    assert [
        entry["relative_path"] for entry in (await fs.readdir("empty"))["files"]
    ] == ["empty/.empty", "empty/child.kcl"]
    with pytest.raises(ZooMCPException, match="already exists"):
        await fs.mkdir("empty")


@pytest.mark.asyncio
async def test_reads_and_metadata(api):
    fs = ProjectFileSystem(PROJECT_ID)
    assert await fs.read_file("parts/a.bin") == b"\xff\x00"
    assert await fs.read_file("main.kcl", "utf-8") == "// main"
    assert (await fs.access("parts"))["access"] == {
        "can_edit": True,
        "can_delete": False,
    }
    assert (await fs.stat("parts/a.bin"))["sha256"] == hashlib.sha256(
        b"\xff\x00"
    ).hexdigest()
    assert (await fs.stat("parts"))["id"] is None
    assert len((await fs.readdir())["files"]) == 3
    assert not api.uploads


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["cp", "mv", "rm"])
async def test_recursive_operations_are_one_replacement(api, operation):
    fs = ProjectFileSystem(PROJECT_ID)
    api.files.update({"parts/nested/.empty": b"", "parts/nested/x": b"x"})
    original = copy.copy(api.files)
    if operation == "rm":
        await fs.rm("parts")
        assert all(not path.startswith("parts/") for path in api.files)
    else:
        await getattr(fs, operation)("parts", "new-parts")
        assert api.files["new-parts/a.bin"] == original["parts/a.bin"]
        assert api.files["new-parts/nested/.empty"] == b""
        assert ("parts/a.bin" in api.files) == (operation == "cp")
    assert api.files["main.kcl"] == original["main.kcl"]
    assert len(api.uploads) == 1


@pytest.mark.asyncio
async def test_destination_directory_and_file_overwrite(api):
    fs = ProjectFileSystem(PROJECT_ID)
    await fs.cp("main.kcl", "parts")
    assert api.files["parts/main.kcl"] == b"// main"
    await fs.mv("parts/main.kcl", "parts/a.bin")
    assert api.files["parts/a.bin"] == b"// main"
    assert "parts/main.kcl" not in api.files


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op,args",
    [
        ("rm", ["main.kcl"]),
        ("rm", ["."]),
        ("mv", ["main.kcl", "other.kcl"]),
        ("mv", ["parts/a.bin", "main.kcl"]),
        ("cp", ["parts/a.bin", "main.kcl"]),
        ("cp", ["parts", "parts/nested"]),
        ("cp", ["parts/a.bin", "parts/a.bin"]),
        ("write_file", ["parts", "x"]),
        ("write_file", ["missing/file", "x"]),
        ("mkdir", ["missing/dir"]),
        ("mkdir", ["main.kcl"]),
        ("rm", ["missing"]),
        ("cp", ["parts/a.bin", "project.toml/nested"]),
    ],
)
async def test_invalid_mutations_never_upload(api, op, args):
    with pytest.raises(ZooMCPException):
        await getattr(ProjectFileSystem(PROJECT_ID), op)(*args)
    assert not api.uploads


@pytest.mark.parametrize(
    "path", ["../a", "parts/../a", "/etc/passwd", "a\x00b", "a\\..\\b", "\\absolute"]
)
def test_unsafe_paths(path):
    with pytest.raises(ZooMCPException):
        zoo_tools._project_path(path)


@pytest.mark.parametrize(
    "project_id", ["local:foo", "cloud:../foo", "cloud:", "not-a-project"]
)
def test_requires_cloud_uuid(project_id):
    with pytest.raises(ZooMCPException):
        ProjectFileSystem(project_id)


@pytest.mark.asyncio
async def test_conflict_reapplies_without_losing_unrelated_edit(api):
    api.failure = "conflict"
    result = await ProjectFileSystem(PROJECT_ID).write_file(
        "parts/a.bin", b"latest requested write"
    )
    assert len(api.uploads) == 2
    assert api.files["concurrent.txt"] == b"preserve concurrent edit"
    assert api.files["parts/a.bin"] == b"latest requested write"
    assert api.uploads[1][0]["expected_revision"] == "2"
    assert result == {
        **api.metadata(),
        "path": "parts/a.bin",
        "contents": base64.b64encode(b"latest requested write").decode(),
        "encoding": "base64",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout_before", "timeout_after"])
async def test_upload_timeout_propagates_without_retry_or_reload(api, failure):
    api.failure = failure
    original = api.files.copy()
    with pytest.raises(httpx.ReadTimeout, match="response lost"):
        await ProjectFileSystem(PROJECT_ID).write_file("parts/a.bin", b"changed")
    assert len(api.uploads) == 1
    assert api.requests[-1].method == "PUT"
    assert (api.files == original) == (failure == "timeout_before")


@pytest.mark.asyncio
async def test_upload_timeout_leaves_concurrent_edit_for_caller(api):
    api.failure = "ambiguous"
    with pytest.raises(httpx.ReadTimeout, match="response lost"):
        await ProjectFileSystem(PROJECT_ID).write_file("parts/a.bin", b"mine")
    assert len(api.uploads) == 1
    assert api.requests[-1].method == "PUT"
    assert api.files["parts/a.bin"] == b"someone else's edit"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,error",
    [
        ("disconnect", httpx.RemoteProtocolError),
        ("server_error", httpx.HTTPStatusError),
        ("invalid_json", ValueError),
    ],
)
async def test_upload_response_errors_propagate_without_retry_or_reload(
    api, project_transport, failure, error
):
    def handle(request):
        response = api.handle(request)
        if request.method != "PUT":
            return response
        if failure == "disconnect":
            raise httpx.RemoteProtocolError("connection closed", request=request)
        if failure == "server_error":
            return httpx.Response(503)
        return httpx.Response(200, content=b"invalid json")

    project_transport(handle)
    with pytest.raises(error):
        await ProjectFileSystem(PROJECT_ID).write_file("parts/a.bin", b"changed")
    assert len(api.uploads) == 1
    assert api.requests[-1].method == "PUT"
    assert api.files["parts/a.bin"] == b"changed"


@pytest.mark.asyncio
async def test_mcp_upload_timeout_surfaces_as_tool_error(api):
    api.failure = "timeout_before"
    with pytest.raises(ToolError, match="response lost"):
        await mcp.call_tool("project_mkdir", {"project_id": PROJECT_ID, "path": "new"})
    assert len(api.uploads) == 1
    assert api.requests[-1].method == "PUT"


@pytest.mark.asyncio
async def test_read_fetches_only_requested_file_from_pinned_revision(api):
    def edit():
        api.files["main.kcl"] = b"// newer revision"
        api.revision += 1

    api.download_hook = edit
    assert await ProjectFileSystem(PROJECT_ID).read_file("main.kcl") == b"// main"
    assert api.files["main.kcl"] == b"// newer revision"
    downloads = [
        request for request in api.requests if "/versions/" in request.url.path
    ]
    assert len(downloads) == 1
    assert "/versions/1/files/" in downloads[0].url.path
    assert (
        await ProjectFileSystem(PROJECT_ID).read_file("main.kcl")
        == b"// newer revision"
    )


@pytest.mark.asyncio
async def test_save_reloads_after_edit_during_pinned_file_download(api):
    def edit():
        api.files["parts/a.bin"] = b"concurrent binary edit"
        api.revision += 1

    api.download_hook = edit

    await ProjectFileSystem(PROJECT_ID).write_file("new.kcl", "// new")
    assert api.uploads[0][1]["parts/a.bin"] == b"\xff\x00"
    assert api.files["parts/a.bin"] == b"concurrent binary edit"
    assert api.files["new.kcl"] == b"// new"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["missing", "parts", "."])
async def test_read_missing_file_or_directory_does_not_download_contents(api, path):
    with pytest.raises(ZooMCPException, match="Not a file"):
        await ProjectFileSystem(PROJECT_ID).read_file(path)
    assert len(api.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["metadata", "mutation"])
async def test_project_size_limit_before_download_and_after_mutation(api, stage):
    current_size = sum(map(len, api.files.values()))
    fs = ProjectFileSystem(
        PROJECT_ID, current_size if stage == "mutation" else current_size - 1
    )
    with pytest.raises(ZooMCPException, match="size"):
        await fs.write_file("new", b"larger")
    assert not api.uploads
    if stage == "metadata":
        assert len(api.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("contents", [b"// oops", b"short", b""])
async def test_file_download_checks_size_and_hash(api, contents):
    api.download_contents = contents
    with pytest.raises(ZooMCPException, match="size|hash"):
        await ProjectFileSystem(PROJECT_ID).read_file("main.kcl")
    assert not api.uploads


@pytest.mark.asyncio
async def test_oversized_file_stream_stops_before_reading_remaining_chunks(
    api, project_transport
):
    class OversizedFile(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"// main"
            yield b"extra"
            raise AssertionError("Must stop when the declared byte size is exceeded")

    def handle(request):
        response = api.handle(request)
        if "/versions/" in request.url.path:
            return httpx.Response(200, headers=response.headers, stream=OversizedFile())
        return response

    project_transport(handle)
    with pytest.raises(ZooMCPException, match="exceeds its declared byte size"):
        await ProjectFileSystem(PROJECT_ID).write_file("new", "x")
    assert not api.uploads


@pytest.mark.asyncio
async def test_unsafe_manifest_path_is_rejected_before_downloading(api):
    api.files["../escape"] = b"unsafe"
    with pytest.raises(ZooMCPException, match="project-relative"):
        await ProjectFileSystem(PROJECT_ID).read_file("main.kcl")
    assert len(api.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("header", ["X-Project-Revision", "X-Artifact-Id"])
async def test_file_from_wrong_version_or_id_is_rejected(
    api, project_transport, header
):
    def handle(request):
        response = api.handle(request)
        if "/versions/" in request.url.path:
            response.headers[header] = "wrong"
        return response

    project_transport(handle)
    with pytest.raises(ZooMCPException, match="does not match its version"):
        await ProjectFileSystem(PROJECT_ID).read_file("main.kcl")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [403, 404, 500])
async def test_file_download_failure_does_not_upload_or_retry(
    api, project_transport, status
):
    downloads = []

    def handle(request):
        if "/versions/" in request.url.path:
            downloads.append(request)
            return httpx.Response(status)
        return api.handle(request)

    project_transport(handle)
    with pytest.raises(httpx.HTTPStatusError):
        await ProjectFileSystem(PROJECT_ID).write_file("new", "x")
    assert len(downloads) == 1
    assert not api.uploads


@pytest.mark.asyncio
async def test_slow_file_download_allows_metadata_queries(
    api, project_transport, monkeypatch
):
    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowFile(httpx.AsyncByteStream):
        async def __aiter__(self):
            entered.set()
            await release.wait()
            yield b"// main"

    def handle(request):
        response = api.handle(request)
        if "/versions/" in request.url.path:
            return httpx.Response(200, headers=response.headers, stream=SlowFile())
        return response

    project_transport(handle)
    loop = asyncio.get_running_loop()
    real_time = loop.time
    elapsed = 0.0
    monkeypatch.setattr(loop, "time", lambda: real_time() + elapsed)
    fs = ProjectFileSystem(PROJECT_ID)
    task = asyncio.create_task(fs.read_file("main.kcl"))
    try:
        await entered.wait()
        elapsed = 7.0
        assert (await fs.stat("parts/a.bin"))["byte_size"] == 2
        assert not task.done()
        release.set()
        assert await task == b"// main"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_permission_denied_and_http_failure_never_retry(api):
    api.editable = False
    with pytest.raises(ZooMCPException, match="cannot edit"):
        await ProjectFileSystem(PROJECT_ID).write_file("new", "x")
    assert not api.uploads
    api.editable = True
    api.failure = "forbidden"
    with pytest.raises(httpx.HTTPStatusError):
        await ProjectFileSystem(PROJECT_ID).write_file("new", "x")
    assert len(api.uploads) == 1


@pytest.mark.asyncio
async def test_same_project_writes_are_serialized(api):
    await asyncio.gather(
        *(
            ProjectFileSystem(PROJECT_ID).write_file(f"file-{n}", str(n))
            for n in range(3)
        )
    )
    assert all(api.files[f"file-{n}"] == str(n).encode() for n in range(3))
    assert len(api.uploads) == 3


@pytest.mark.asyncio
async def test_last_child_preserves_parent_but_explicit_marker_removal_works(api):
    fs = ProjectFileSystem(PROJECT_ID)
    await fs.rm("parts/a.bin")
    assert api.files["parts/.empty"] == b""
    await fs.rm("parts/.empty")
    assert all(not path.startswith("parts/") for path in api.files)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation,args",
    [
        ("rm", ["project.toml"]),
        ("mv", ["project.toml", "renamed.toml"]),
    ],
)
async def test_manifest_cannot_be_removed_or_moved(api, operation, args):
    with pytest.raises(ZooMCPException, match="Project API requires"):
        await getattr(ProjectFileSystem(PROJECT_ID), operation)(*args)
    assert not api.uploads


@pytest.mark.asyncio
async def test_retry_budget(api, project_transport):
    original = api.handle
    attempts = 0

    def conflict_forever(request):
        nonlocal attempts
        if request.method == "PUT":
            attempts += 1
            return httpx.Response(409)
        return original(request)

    project_transport(conflict_forever)
    with pytest.raises(ZooMCPException, match="three attempts"):
        await ProjectFileSystem(PROJECT_ID).rm("parts")
    assert attempts == 3


@pytest.mark.asyncio
async def test_conflicting_remove_includes_new_descendants(api, project_transport):
    original = api.handle
    conflict = True

    def handle(request):
        nonlocal conflict
        if request.method == "PUT" and conflict:
            conflict = False
            api.files["parts/new.bin"] = b"added before retry"
            api.revision += 1
            return httpx.Response(409)
        return original(request)

    project_transport(handle)
    await ProjectFileSystem(PROJECT_ID).rm("parts")
    assert "parts/new.bin" in api.uploads[-1][0]["deleted_paths"]
    assert all(not name.startswith("parts/") for name in api.files)


@pytest.mark.asyncio
async def test_credentials_are_captured_and_locks_are_scoped(api):
    fs = ProjectFileSystem(PROJECT_ID)
    with use_client_config(
        ZooClientConfig("different-token", "https://other.example.test")
    ):
        other = ProjectFileSystem(PROJECT_ID)
        lock = fs._lock()
        assert other._lock() is not lock
        assert ProjectFileSystem(PROJECT_ID)._lock() is other._lock()
        assert await fs.read_file("main.kcl") == b"// main"


@pytest.mark.asyncio
async def test_file_requests_keep_captured_custom_headers(api):
    with use_client_config(
        ZooClientConfig(
            "test-token",
            "https://api.example.test/custom/",
            headers={"X-Caller": "project-owner"},
        )
    ):
        fs = ProjectFileSystem(PROJECT_ID)

    assert await fs.read_file("parts/a.bin") == b"\xff\x00"
    assert all(
        request.headers["X-Caller"] == "project-owner" for request in api.requests
    )


@pytest.mark.asyncio
async def test_cancellation_releases_mutation_lock(api, monkeypatch):
    fs = ProjectFileSystem(PROJECT_ID)
    original = fs._snapshot
    entered = asyncio.Event()

    async def blocked(client):
        entered.set()
        await asyncio.Event().wait()
        return await original(client)

    monkeypatch.setattr(fs, "_snapshot", blocked)
    task = asyncio.create_task(fs.mkdir("cancelled"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await ProjectFileSystem(PROJECT_ID).mkdir("after-cancellation")
    assert "after-cancellation/.empty" in api.files
    assert "cancelled/.empty" not in api.files


@pytest.mark.asyncio
async def test_total_deadline_has_actionable_error(api, monkeypatch):
    monkeypatch.setattr(zoo_tools, "_PROJECT_TIMEOUT", 0)

    async def blocked(self, client):
        await asyncio.sleep(0)
        raise AssertionError("Deadline should cancel snapshot loading")

    monkeypatch.setattr(ProjectFileSystem, "_metadata", blocked)
    with pytest.raises(ZooMCPException, match="deadline exceeded"):
        await ProjectFileSystem(PROJECT_ID).read_file("main.kcl")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool,args",
    [
        ("project_mkdir", {"path": "new"}),
        ("project_cp", {"source": "parts/a.bin", "destination": "copied.bin"}),
        ("project_mv", {"source": "parts/a.bin", "destination": "moved.bin"}),
        ("project_rm", {"path": "parts"}),
        ("project_access", {"path": "parts"}),
        ("project_stat", {"path": "parts/a.bin"}),
        ("project_readdir", {"path": "./"}),
    ],
)
async def test_remaining_mcp_tools(api, tool, args):
    result = await mcp.call_tool(tool, {"project_id": PROJECT_ID, **args})
    assert isinstance(result, CallToolResult)
    assert not result.is_error


@pytest.mark.asyncio
async def test_mcp_bad_inputs_return_actionable_errors(api):
    with pytest.raises(ToolError, match="main.kcl"):
        await mcp.call_tool(
            "project_rm", {"project_id": PROJECT_ID, "path": "main.kcl"}
        )
    with pytest.raises(ToolError, match="cloud:"):
        await mcp.call_tool("project_stat", {"project_id": "local:bad"})
    with pytest.raises(ToolError, match="base64|padding|digit|characters"):
        await mcp.call_tool(
            "project_write_file",
            {
                "project_id": PROJECT_ID,
                "path": "file",
                "contents": "!!!",
                "encoding": "base64",
            },
        )


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limits(api, limit):
    with pytest.raises(ZooMCPException, match="positive"):
        ProjectFileSystem(PROJECT_ID, limit)


@pytest.mark.asyncio
async def test_file_download_follows_no_redirects(api, project_transport):
    original = api.handle

    def handle(request):
        if "/versions/" in request.url.path:
            return httpx.Response(
                302, headers={"Location": "https://other.example.test/archive.zip"}
            )
        return original(request)

    project_transport(handle)
    with pytest.raises(httpx.HTTPStatusError):
        await ProjectFileSystem(PROJECT_ID).read_file("main.kcl")
    assert not api.uploads


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "contents,options,expected",
    [
        (b"\xff\x00\x80", {}, b"\xff\x00\x80"),
        (b"", {}, b""),
        ("", {}, b""),
        ("", {"encoding": "base64"}, b""),
        ("text \u00e9", {}, "text \u00e9".encode()),
        ("text \u00e9", {"encoding": "utf-8"}, "text \u00e9".encode()),
        ("/wCA", {"encoding": "base64"}, b"\xff\x00\x80"),
    ],
)
async def test_write_tool_accepts_bytes_without_encoding(
    api, contents, options, expected
):
    result = await mcp.call_tool(
        "project_write_file",
        {
            "project_id": PROJECT_ID,
            "path": "written.bin",
            "contents": contents,
            **options,
        },
    )
    assert isinstance(result, CallToolResult)
    assert not result.is_error
    assert api.files["written.bin"] == expected
    assert isinstance(result.content[0], TextContent)
    binary = isinstance(contents, bytes) or options.get("encoding") == "base64"
    assert json.loads(result.content[0].text) == {
        **api.metadata(),
        "path": "written.bin",
        "contents": base64.b64encode(expected).decode()
        if binary
        else expected.decode("utf-8"),
        "encoding": "base64" if binary else "utf-8",
    }
    read = await mcp.call_tool(
        "project_read_file",
        {
            "project_id": PROJECT_ID,
            "path": "written.bin",
            "encoding": "base64" if binary else "utf-8",
        },
    )
    assert isinstance(read, CallToolResult)
    assert isinstance(read.content[0], TextContent)
    assert json.loads(read.content[0].text) == {
        key: value
        for key, value in json.loads(result.content[0].text).items()
        if key in ("contents", "encoding")
    }


@pytest.mark.asyncio
async def test_write_result_returns_normalized_path(api):
    result = await mcp.call_tool(
        "project_write_file",
        {
            "project_id": PROJECT_ID,
            "path": "  .\\parts\\new.kcl  ",
            "contents": "// committed contents\n",
        },
    )
    assert isinstance(result, CallToolResult)
    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text) == {
        **api.metadata(),
        "path": "parts/new.kcl",
        "contents": "// committed contents\n",
        "encoding": "utf-8",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", ["utf-8", "base64"])
async def test_raw_bytes_require_omitted_encoding(api, encoding):
    with pytest.raises(ToolError, match="Omit encoding"):
        await mcp.call_tool(
            "project_write_file",
            {
                "project_id": PROJECT_ID,
                "path": "written.bin",
                "contents": b"\xff",
                "encoding": encoding,
            },
        )
    assert not api.uploads


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    ["part:v1.kcl", "/".join(["long-name" * 20] * 8) + "/model.kcl"],
    ids=["colon", "long"],
)
async def test_api_valid_paths_do_not_disable_other_files(api, path):
    api.files[path] = b"// valid API file"
    fs = ProjectFileSystem(PROJECT_ID)
    assert (await fs.stat("main.kcl"))["kind"] == "file"
    assert await fs.read_file(path) == b"// valid API file"
    await fs.write_file(path, b"// updated")
    assert api.files[path] == b"// updated"


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
async def test_large_directory_listing_scans_metadata_linearly(
    api, monkeypatch, nested
):
    count = 2000
    api.files = {
        f"child-{n}/file" if nested else f"child-{n}": b"" for n in range(count)
    }
    metadata = api.metadata()

    class CountingEntries(list):
        visits = 0

        def __iter__(self):
            for entry in super().__iter__():
                self.visits += 1
                yield entry

    entries = CountingEntries(metadata["files"])
    metadata["files"] = entries
    fs = ProjectFileSystem(PROJECT_ID)

    async def load_metadata(client):
        return metadata

    monkeypatch.setattr(fs, "_metadata", load_metadata)
    children = (await fs.readdir())["files"]
    assert len(children) == count
    assert {entry["kind"] for entry in children} == {"directory" if nested else "file"}
    # Count work instead of asserting wall-clock timing on CI machines.
    assert entries.visits <= count * 4


@pytest.mark.asyncio
async def test_write_path_normalization_matches_api(api):
    fs = ProjectFileSystem(PROJECT_ID)
    await fs.write_file("  .\\parts\\part:v1.kcl  ", "// normalized")
    assert api.files["parts/part:v1.kcl"] == b"// normalized"
    assert await fs.read_file(" parts\\part:v1.kcl ") == b"// normalized"


@pytest.mark.asyncio
async def test_quoted_and_literal_percent_filenames_remain_distinct(api):
    api.files["part%22v1.kcl"] = b"// literal percent name"
    fs = ProjectFileSystem(PROJECT_ID)
    await fs.write_file('part"v1.kcl', b"// quoted name")
    assert api.files['part"v1.kcl'] == b"// quoted name"
    assert api.files["part%22v1.kcl"] == b"// literal percent name"
    assert await fs.read_file('part"v1.kcl') == b"// quoted name"
    request = next(request for request in api.requests if request.method == "PUT")
    assert b'filename="part\\"v1.kcl"\r\n' in request.content
    assert b'filename="part%22v1.kcl"\r\n' in request.content


@pytest.mark.asyncio
async def test_unrelated_write_preserves_existing_quotes_unicode_and_empty_files(api):
    originals = {
        'part"v1.kcl': b"// quoted",
        "part%22v1.kcl": b"// percent",
        'pi\u00e8ce"2.bin': b"\xff\x00",
        'empty"file': b"",
    }
    api.files.update(originals)
    await ProjectFileSystem(PROJECT_ID).write_file("unrelated", b"new")
    assert all(api.files[name] == content for name, content in originals.items())


@pytest.mark.asyncio
async def test_streamed_multipart_preserves_binary_across_chunks(api):
    data = b"\x00\xff\r\n--not-a-boundary\r\n" * 10_000
    await ProjectFileSystem(PROJECT_ID).write_file('large"binary', data)
    assert api.files['large"binary'] == data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "line\nbreak",
        "carriage\rreturn",
        "tab\tname",
        "delete\x7fname",
        "null\x00name",
        "\ntrimmed",
        'injection"\r\nX-Injected: yes',
    ],
)
async def test_control_characters_are_rejected_before_any_request(api, path):
    with pytest.raises(ZooMCPException, match="control characters"):
        await ProjectFileSystem(PROJECT_ID).write_file(path, b"data")
    assert not api.requests
