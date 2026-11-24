"""Cloud filesystem replacements against an in-memory HTTP Project API."""

import asyncio
import base64
import copy
import hashlib
import io
import json
import zipfile
from email import policy
from email.parser import BytesParser
from uuid import NAMESPACE_URL, uuid5

import httpx
import pytest
from kittycad import AsyncKittyCAD
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent

from zoo_mcp import ZooMCPException, client_config, zoo_tools
from zoo_mcp.client_config import ZooClientConfig, use_client_config
from zoo_mcp.server import mcp, project_stat, project_write_file
from zoo_mcp.zoo_tools import ProjectFileSystem

PROJECT_ID = "cloud:00000000-0000-4000-8000-000000000001"
CLOUD_STAMP = (
    '\n[cloud."example.test"]\nproject_id = "' + PROJECT_ID[6:] + '"\n'
).encode()


def project_zip(entries):
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, data in entries:
            archive.writestr(path, data)
    return content.getvalue()


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
        self.entrypoint = "main.kcl"
        self.editable = True
        self.download_hook = None
        self.versions = {}
        self.download_contents: bytes | None = None

    def file_id(self, path):
        return str(uuid5(NAMESPACE_URL, f"{self.revision}/{path}"))

    def metadata(self):
        self.versions[str(self.revision)] = {
            self.file_id(path): (path, data) for path, data in self.files.items()
        }
        return {
            "id": PROJECT_ID[6:],
            "revision": str(self.revision),
            "metadata_revision": str(self.revision),
            "created_at": "2026-11-20T00:00:00Z",
            "updated_at": "2026-11-20T00:00:00Z",
            "title": "My project",
            "description": "Keep this description",
            "category_ids": ["category"],
            "publication_status": "published",
            "publication": {"has_unpublished_changes": False},
            "preview_status": "pending",
            "entrypoint_path": self.entrypoint,
            "project_toml_path": "project.toml",
            "access": {
                "can_edit": self.editable,
                "can_delete": False,
                "can_manage_organization": False,
                "organization_id": None,
                "scope": "personal",
            },
            "files": [
                {
                    "relative_path": path,
                    "byte_size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "id": self.file_id(path),
                    "created_at": "2026-11-20T00:00:00Z",
                    "updated_at": "2026-11-20T00:00:00Z",
                    "sort_order": order,
                    "content_type": "application/octet-stream",
                    "file_role": "preview_image"
                    if path == "thumbnail.png"
                    else "project_file",
                }
                for order, (path, data) in enumerate(self.files.items())
            ],
        }

    def handle(self, request):
        self.requests.append(request)
        assert request.headers["Authorization"] == "Bearer test-token"
        assert request.url.path.startswith("/custom/user/projects/")
        if request.method == "GET" and "/versions/" in request.url.path:
            project_path, revision, endpoint = request.url.path.rsplit("/", 2)
            assert project_path == f"/custom/user/projects/{PROJECT_ID[6:]}/versions"
            assert endpoint == "download"
            assert request.url.params["format"] == "zip"
            file_id = request.url.params["file_ids"]
            selected = self.versions.get(revision, {}).get(file_id)
            if selected is None:
                return httpx.Response(
                    400, json={"error_code": "InvalidProjectFileSelection"}
                )
            path, content = selected
            if self.download_hook:
                hook, self.download_hook = self.download_hook, None
                hook()
            return httpx.Response(
                200,
                content=project_zip(
                    [
                        (
                            path,
                            content
                            if self.download_contents is None
                            else self.download_contents,
                        )
                    ]
                ),
                headers={"Content-Type": "application/zip"},
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
        if CLOUD_STAMP not in files["project.toml"]:
            files["project.toml"] += CLOUD_STAMP
        self.files = files
        self.revision += 1
        if self.failure == "timeout_after":
            self.failure = None
            raise httpx.ReadTimeout("response lost", request=request)
        return httpx.Response(200, json=self.metadata())

    def downloaded_paths(self, requests):
        return [
            self.versions[request.url.path.split("/")[-2]][
                request.url.params["file_ids"]
            ][0]
            for request in requests
            if "/versions/" in request.url.path
        ]


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
            lambda fs: AsyncKittyCAD(
                token=fs.config.token,
                base_url=fs.config.base_url.rstrip("/"),
                headers=dict(fs.config.headers or {}),
                http_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(handler),
                    follow_redirects=False,
                    event_hooks={"response": [zoo_tools._capture_rest_response]},
                ),
            ),
        )

    return install


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["body", "event"])
async def test_write_preserves_binary_metadata_and_arbitrary_field_names(api, path):
    fs = ProjectFileSystem(PROJECT_ID)
    await fs.write_file(path, "new contents")
    assert api.files["parts/a.bin"] == b"\xff\x00"
    assert api.files[path] == b"new contents"
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
    assert (await fs.read_file("parts/a.bin"))[0] == b"\xff\x00"
    assert (await fs.read_file("main.kcl", "utf-8"))[0] == "// main"
    assert (await fs.access("parts"))["access"] == {
        "can_edit": True,
        "can_delete": False,
        "can_manage_organization": False,
        "organization_id": None,
        "scope": "personal",
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
    with pytest.raises(ZooMCPException, match="upload may have completed"):
        await ProjectFileSystem(PROJECT_ID).write_file("parts/a.bin", b"changed")
    assert len(api.uploads) == 1
    assert api.requests[-1].method == "PUT"
    assert (api.files == original) == (failure == "timeout_before")


@pytest.mark.asyncio
async def test_upload_timeout_leaves_concurrent_edit_for_caller(api):
    api.failure = "ambiguous"
    with pytest.raises(ZooMCPException, match="upload may have completed"):
        await ProjectFileSystem(PROJECT_ID).write_file("parts/a.bin", b"mine")
    assert len(api.uploads) == 1
    assert api.requests[-1].method == "PUT"
    assert api.files["parts/a.bin"] == b"someone else's edit"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["disconnect", "server_error", "invalid_json", "invalid_model", "empty"]
)
async def test_upload_response_errors_propagate_without_retry_or_reload(
    api, project_transport, failure
):
    def handle(request):
        response = api.handle(request)
        if request.method != "PUT":
            return response
        if failure == "disconnect":
            raise httpx.RemoteProtocolError("connection closed", request=request)
        if failure == "server_error":
            return httpx.Response(503)
        if failure == "empty":
            return httpx.Response(200)
        if failure == "invalid_model":
            return httpx.Response(200, json={})
        return httpx.Response(200, content=b"invalid json")

    project_transport(handle)
    with pytest.raises(ZooMCPException, match="Inspect the latest project"):
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
    assert (await ProjectFileSystem(PROJECT_ID).read_file("main.kcl"))[0] == b"// main"
    assert api.files["main.kcl"] == b"// newer revision"
    downloads = [
        request for request in api.requests if "/versions/" in request.url.path
    ]
    assert len(downloads) == 1
    assert downloads[0].url.path.endswith("/versions/1/download")
    assert downloads[0].url.params["file_ids"] == str(
        uuid5(NAMESPACE_URL, "1/main.kcl")
    )
    assert (await ProjectFileSystem(PROJECT_ID).read_file("main.kcl"))[
        0
    ] == b"// newer revision"


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
@pytest.mark.parametrize("contents", [b"// oops", b"short", b"", b"x" * 1_000_000])
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
            yield b"x" * (256 * 1024)
            raise AssertionError("Must stop when the archive size limit is exceeded")

    def handle(request):
        response = api.handle(request)
        if "/versions/" in request.url.path:
            return httpx.Response(200, headers=response.headers, stream=OversizedFile())
        return response

    project_transport(handle)
    with pytest.raises(ZooMCPException, match="archive exceeds"):
        await ProjectFileSystem(PROJECT_ID).write_file("new", "x")
    assert not api.uploads


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,message", [("../escape", "project-relative"), ("parts//a.kcl", "canonical")]
)
async def test_unsafe_manifest_path_is_rejected_before_downloading(api, path, message):
    api.files[path] = b"unsafe"
    with pytest.raises(ZooMCPException, match=message):
        await ProjectFileSystem(PROJECT_ID).read_file("main.kcl")
    assert len(api.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entries",
    [
        [],
        [("other.kcl", b"// main")],
        [("../main.kcl", b"// main")],
        [("main.kcl", b"// main"), ("extra.kcl", b"extra")],
        [("main.kcl", b"// main"), ("main.kcl", b"// main")],
    ],
)
async def test_download_archive_contains_only_the_requested_path(
    api, project_transport, entries
):
    def handle(request):
        if "/versions/" in request.url.path:
            return httpx.Response(200, content=project_zip(entries))
        return api.handle(request)

    project_transport(handle)
    with pytest.raises(ZooMCPException, match="requested file"):
        await ProjectFileSystem(PROJECT_ID).read_file("main.kcl")


@pytest.mark.asyncio
async def test_invalid_download_archive_is_an_actionable_error(api, project_transport):
    def handle(request):
        if "/versions/" in request.url.path:
            return httpx.Response(200, content=b"not a zip")
        return api.handle(request)

    project_transport(handle)
    with pytest.raises(ZooMCPException, match="Invalid project file archive"):
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
    with pytest.raises(ZooMCPException):
        await ProjectFileSystem(PROJECT_ID).write_file("new", "x")
    assert len(downloads) == 1
    assert not api.uploads


@pytest.mark.asyncio
async def test_slow_file_download_allows_metadata_queries(
    api, project_transport, monkeypatch
):
    fs = ProjectFileSystem(PROJECT_ID)
    assert (await fs.read_file("parts/a.bin"))[0] == b"\xff\x00"
    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowFile(httpx.AsyncByteStream):
        async def __aiter__(self):
            entered.set()
            await release.wait()
            yield project_zip([("main.kcl", b"// main")])

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
    task = asyncio.create_task(fs.read_file("main.kcl"))
    try:
        await entered.wait()
        elapsed = 7.0
        assert (await fs.stat("parts/a.bin"))["byte_size"] == 2
        assert (await fs.read_file("parts/a.bin"))[0] == b"\xff\x00"
        assert not task.done()
        release.set()
        assert await task == (b"// main", "1")
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_unsupported_entrypoint_is_rejected_before_downloading(api):
    api.entrypoint = "parts/model.kcl"
    with pytest.raises(ZooMCPException, match="require main.kcl"):
        await ProjectFileSystem(PROJECT_ID).write_file("new", "x")
    assert len(api.requests) == 1


@pytest.mark.asyncio
async def test_permission_denied_and_http_failure_never_retry(api):
    api.editable = False
    with pytest.raises(ZooMCPException, match="cannot edit"):
        await ProjectFileSystem(PROJECT_ID).write_file("new", "x")
    assert len(api.requests) == 1
    api.editable = True
    api.failure = "forbidden"
    with pytest.raises(ZooMCPException):
        await ProjectFileSystem(PROJECT_ID).write_file("new", "x")
    assert len(api.uploads) == 1


@pytest.mark.asyncio
async def test_concurrent_writes_retry_conflicts_without_losing_files(
    api, project_transport
):
    ready = asyncio.Event()
    initial_uploads = 0
    conflicts = 0

    async def handle(request):
        nonlocal initial_uploads, conflicts
        if request.method == "PUT" and initial_uploads < 3:
            initial_uploads += 1
            if initial_uploads == 3:
                ready.set()
            # All writers submit the same revision before any upload commits.
            await ready.wait()
        response = api.handle(request)
        if response.status_code == 409:
            conflicts += 1
        return response

    project_transport(handle)
    async with asyncio.timeout(5):
        await asyncio.gather(
            *(
                ProjectFileSystem(PROJECT_ID).write_file(f"file-{n}", str(n))
                for n in range(3)
            )
        )
    assert all(api.files[f"file-{n}"] == str(n).encode() for n in range(3))
    assert api.files["parts/a.bin"] == b"\xff\x00"
    assert conflicts == 2


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
async def test_credentials_are_captured_at_construction(api):
    fs = ProjectFileSystem(PROJECT_ID)
    with use_client_config(
        ZooClientConfig("different-token", "https://other.example.test")
    ):
        assert (await fs.read_file("main.kcl"))[0] == b"// main"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["new.kcl", "body", "event", 'quoted"file.kcl'])
async def test_file_requests_keep_captured_custom_headers(api, path):
    with use_client_config(
        ZooClientConfig(
            "test-token",
            "https://api.example.test/custom/",
            headers={
                "X-Caller": "project-owner",
                "User-Agent": "ExampleClient/1.0",
                "content-type": "application/json",
            },
        )
    ):
        fs = ProjectFileSystem(PROJECT_ID)

    assert (await fs.read_file("parts/a.bin"))[0] == b"\xff\x00"
    await fs.write_file(path, b"// changed")
    assert api.files[path] == b"// changed"
    assert all(
        request.headers["X-Caller"] == "project-owner" for request in api.requests
    )
    assert all(
        request.headers.get_list("User-Agent") == ["ExampleClient/1.0"]
        for request in api.requests
    )
    upload = next(request for request in api.requests if request.method == "PUT")
    assert len(upload.headers.get_list("Content-Type")) == 1
    assert upload.headers["Content-Type"].startswith("multipart/form-data;")


@pytest.mark.asyncio
async def test_cancelled_mutation_does_not_write_files(api, monkeypatch):
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
async def test_inner_timeout_keeps_its_original_error(api, monkeypatch):
    expected = TimeoutError("inner library timeout")

    async def timed_out(self, client):
        raise expected

    monkeypatch.setattr(ProjectFileSystem, "_metadata", timed_out)
    with pytest.raises(TimeoutError) as caught:
        await ProjectFileSystem(PROJECT_ID).read_file("main.kcl")
    assert caught.value is expected
    assert not api.uploads


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
    with pytest.raises(ZooMCPException):
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
        if key in ("contents", "encoding", "revision")
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
    assert (await fs.read_file(path))[0] == b"// valid API file"
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
    assert (await fs.read_file(" parts\\part:v1.kcl "))[0] == b"// normalized"


@pytest.mark.asyncio
async def test_quoted_and_literal_percent_filenames_remain_distinct(api):
    api.files["part%22v1.kcl"] = b"// literal percent name"
    fs = ProjectFileSystem(PROJECT_ID)
    await fs.write_file('part"v1.kcl', b"// quoted name")
    assert api.files['part"v1.kcl'] == b"// quoted name"
    assert api.files["part%22v1.kcl"] == b"// literal percent name"
    assert (await fs.read_file('part"v1.kcl'))[0] == b"// quoted name"
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


@pytest.mark.asyncio
@pytest.mark.parametrize("binary", [False, True])
async def test_manifest_write_returns_stamped_contents_from_acknowledged_revision(
    api, binary
):
    contents = "[settings]\n"
    result = await ProjectFileSystem(PROJECT_ID).write_file(
        "project.toml", contents.encode() if binary else contents
    )
    stored = contents.encode() + CLOUD_STAMP
    returned = (
        base64.b64decode(result["contents"]) if binary else result["contents"].encode()
    )
    assert returned == stored
    entry = next(
        entry for entry in result["files"] if entry["relative_path"] == "project.toml"
    )
    assert hashlib.sha256(returned).hexdigest() == entry["sha256"]
    assert len(returned) == entry["byte_size"]
    assert (await ProjectFileSystem(PROJECT_ID).read_file("project.toml"))[
        0
    ] == returned


@pytest.mark.asyncio
async def test_manifest_write_result_stays_on_acknowledged_revision(
    api, project_transport
):
    def handle(request):
        response = api.handle(request)
        if request.method == "PUT":
            api.files["project.toml"] = b"[settings]\n# concurrent edit\n" + CLOUD_STAMP
            api.revision += 1
        return response

    project_transport(handle)
    result = await ProjectFileSystem(PROJECT_ID).write_file(
        "project.toml", "[settings]\n"
    )
    assert result["contents"].encode() == b"[settings]\n" + CLOUD_STAMP
    assert result["revision"] == "2"
    assert api.revision == 3


@pytest.mark.asyncio
async def test_binary_read_explains_how_to_use_base64(api):
    with pytest.raises(ZooMCPException, match="base64"):
        await ProjectFileSystem(PROJECT_ID).read_file("parts/a.bin", "utf-8")
    assert not api.uploads


@pytest.mark.asyncio
async def test_bad_base64_explains_expected_input(api):
    with pytest.raises(ZooMCPException, match="valid base64"):
        await project_write_file(PROJECT_ID, "bad.bin", "!", "base64")
    assert not api.requests


def test_missing_project_credentials_explain_how_to_authenticate(monkeypatch):
    monkeypatch.setattr(client_config, "_saved_login", None)
    monkeypatch.setattr(client_config, "_local_config", ZooClientConfig(""))
    with pytest.raises(ZooMCPException, match="auth login|ZOO_API_TOKEN"):
        ProjectFileSystem(PROJECT_ID)


@pytest.mark.asyncio
async def test_http_rejection_preserves_status_and_api_details(api, project_transport):
    def handle(request):
        if request.method == "PUT":
            return httpx.Response(
                422,
                json={
                    "message": "project.toml is invalid",
                    "error_code": "InvalidManifest",
                },
            )
        return api.handle(request)

    project_transport(handle)
    with pytest.raises(ZooMCPException, match="422.*InvalidManifest"):
        await ProjectFileSystem(PROJECT_ID).write_file("new", "x")


@pytest.mark.asyncio
async def test_unexpected_value_error_is_not_reclassified(api, monkeypatch):
    async def broken(self, path):
        raise ValueError("unexpected bug")

    monkeypatch.setattr(ProjectFileSystem, "stat", broken)
    with pytest.raises(ValueError, match="unexpected bug"):
        await project_stat(PROJECT_ID)


@pytest.mark.asyncio
async def test_manifest_read_failure_reports_successful_upload(api, project_transport):
    def handle(request):
        if "/versions/2/" in request.url.path:
            return httpx.Response(503)
        return api.handle(request)

    project_transport(handle)
    with pytest.raises(ZooMCPException, match="upload succeeded at revision 2"):
        await ProjectFileSystem(PROJECT_ID).write_file(
            "project.toml", "[settings]\n# changed\n"
        )
    assert api.files["project.toml"] == b"[settings]\n# changed\n" + CLOUD_STAMP
    assert len(api.uploads) == 1


@pytest.mark.asyncio
async def test_download_disconnect_is_an_actionable_tool_error(api, project_transport):
    def handle(request):
        if "/versions/" in request.url.path:
            raise httpx.ReadError("connection closed", request=request)
        return api.handle(request)

    project_transport(handle)
    with pytest.raises(ZooMCPException, match="Check your connection"):
        await ProjectFileSystem(PROJECT_ID).read_file("main.kcl")
    assert not api.uploads


@pytest.mark.asyncio
async def test_expected_project_error_propagates_unchanged(api, monkeypatch):
    expected = ZooMCPException("Permission denied")

    async def denied(self, path):
        raise expected

    monkeypatch.setattr(ProjectFileSystem, "stat", denied)
    with pytest.raises(ZooMCPException) as caught:
        await project_stat(PROJECT_ID)
    assert caught.value is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict", [False, True])
@pytest.mark.parametrize(
    "operation,args",
    [
        ("write_file", ["main.kcl", "// changed model"]),
        ("cp", ["main.kcl", "parts/model.kcl"]),
        ("mv", ["parts/a.bin", "parts/moved.bin"]),
        ("rm", ["parts"]),
        ("mkdir", ["new"]),
    ],
)
async def test_mutations_omit_existing_previews_so_api_renders_again(
    api, operation, args, conflict
):
    api.files["thumbnail.png"] = b"old preview"
    if conflict:
        api.failure = "conflict"
    await getattr(ProjectFileSystem(PROJECT_ID), operation)(*args)
    assert "thumbnail.png" not in api.files
    for body, files in api.uploads:
        assert "thumbnail.png" not in files
        # Preview assets are server-owned and cannot be listed as deleted paths.
        assert "thumbnail.png" not in body["deleted_paths"]


@pytest.mark.asyncio
@pytest.mark.parametrize("contents", [b"old preview", b"new preview"])
@pytest.mark.parametrize("conflict", [False, True])
async def test_explicit_preview_write_is_preserved_even_when_unchanged(
    api, contents, conflict
):
    api.files["thumbnail.png"] = b"old preview"
    if conflict:
        api.failure = "conflict"
    fs = ProjectFileSystem(PROJECT_ID)
    result = await fs.write_file("thumbnail.png", contents)
    assert base64.b64decode(result["contents"]) == contents
    assert any(entry["relative_path"] == "thumbnail.png" for entry in result["files"])
    assert (await fs.read_file("thumbnail.png"))[0] == contents
    assert all(files["thumbnail.png"] == contents for _, files in api.uploads)


@pytest.mark.asyncio
async def test_archive_symlink_is_not_returned_as_a_project_file(
    api, project_transport
):
    member = zipfile.ZipInfo("main.kcl")
    member.create_system = 3
    member.external_attr = 0o120777 << 16
    archive = project_zip([(member, b"// main")])

    def handle(request):
        if "/versions/" in request.url.path:
            return httpx.Response(200, content=archive)
        return api.handle(request)

    project_transport(handle)
    with pytest.raises(ZooMCPException, match="requested file"):
        await ProjectFileSystem(PROJECT_ID).read_file("main.kcl")


@pytest.mark.asyncio
async def test_archive_checksum_failure_is_a_tool_error(api, project_transport):
    archive = bytearray(project_zip([("main.kcl", b"// main")]))
    directory_offset = archive.index(b"PK\x01\x02")
    archive[directory_offset + 16 : directory_offset + 20] = b"\x00" * 4

    def handle(request):
        if "/versions/" in request.url.path:
            return httpx.Response(200, content=bytes(archive))
        return api.handle(request)

    project_transport(handle)
    with pytest.raises(ZooMCPException, match="Invalid project file archive"):
        await ProjectFileSystem(PROJECT_ID).read_file("main.kcl")


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["cp", "mv"])
async def test_explicit_copy_to_preview_preserves_identical_bytes(api, operation):
    api.files["thumbnail.png"] = b"same preview"
    api.files["parts/replacement.png"] = b"same preview"
    fs = ProjectFileSystem(PROJECT_ID)
    await getattr(fs, operation)("parts/replacement.png", "thumbnail.png")
    assert (await fs.read_file("thumbnail.png"))[0] == b"same preview"
    assert ("parts/replacement.png" in api.files) == (operation == "cp")


@pytest.mark.asyncio
async def test_consecutive_writes_reuse_cached_files_across_instances(api):
    await ProjectFileSystem(PROJECT_ID).write_file("first.kcl", "// first")
    assert set(api.downloaded_paths(api.requests)) == {
        "main.kcl",
        "project.toml",
        "parts/a.bin",
    }
    start = len(api.requests)
    await ProjectFileSystem(PROJECT_ID).write_file("second.kcl", "// second")
    assert api.downloaded_paths(api.requests[start:]) == []
    assert api.files["first.kcl"] == b"// first"
    assert api.files["second.kcl"] == b"// second"
    assert api.files["parts/a.bin"] == b"\xff\x00"
    assert [request.method for request in api.requests[start:]] == ["GET", "PUT"]


@pytest.mark.asyncio
async def test_cached_project_downloads_only_external_additions_and_edits(api):
    await ProjectFileSystem(PROJECT_ID).write_file("first.kcl", "// first")
    api.files["main.kcl"] = b"// external edit"
    api.files["external.bin"] = b"external addition"
    del api.files["parts/a.bin"]
    api.revision += 1
    start = len(api.requests)
    await ProjectFileSystem(PROJECT_ID).write_file("second.kcl", "// second")
    assert set(api.downloaded_paths(api.requests[start:])) == {
        "main.kcl",
        "external.bin",
    }
    assert api.files["main.kcl"] == b"// external edit"
    assert api.files["external.bin"] == b"external addition"
    assert "parts/a.bin" not in api.files
    assert api.files["first.kcl"] == b"// first"


@pytest.mark.asyncio
async def test_conflict_refreshes_only_new_files_in_a_cached_project(api):
    await ProjectFileSystem(PROJECT_ID).write_file("first.kcl", "// first")
    api.failure = "conflict"
    start = len(api.requests)
    await ProjectFileSystem(PROJECT_ID).write_file("second.kcl", "// second")
    assert api.downloaded_paths(api.requests[start:]) == ["concurrent.txt"]
    assert api.files["concurrent.txt"] == b"preserve concurrent edit"
    assert api.files["second.kcl"] == b"// second"


@pytest.mark.asyncio
async def test_manifest_cache_uses_stored_contents_instead_of_submitted_text(api):
    fs = ProjectFileSystem(PROJECT_ID)
    await fs.write_file("project.toml", "[settings]\n# new settings\n")
    expected = b"[settings]\n# new settings\n" + CLOUD_STAMP
    assert api.files["project.toml"] == expected
    start = len(api.requests)
    assert (await ProjectFileSystem(PROJECT_ID).read_file("project.toml"))[
        0
    ] == expected
    await ProjectFileSystem(PROJECT_ID).write_file("new.kcl", "// new")
    assert api.files["project.toml"] == expected
    assert api.downloaded_paths(api.requests[start:]) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout_before", "timeout_after"])
async def test_failed_upload_does_not_replace_cached_bytes_with_unacknowledged_input(
    api, failure
):
    fs = ProjectFileSystem(PROJECT_ID)
    await fs.write_file("warm.kcl", "// warm")
    api.failure = failure
    with pytest.raises(ZooMCPException, match="upload may have completed"):
        await fs.write_file("main.kcl", "// requested change")
    expected = b"// main" if failure == "timeout_before" else b"// requested change"
    assert (await ProjectFileSystem(PROJECT_ID).read_file("main.kcl"))[0] == expected


@pytest.mark.asyncio
async def test_cached_read_still_checks_access_with_the_api(api, project_transport):
    fs = ProjectFileSystem(PROJECT_ID)
    assert (await fs.read_file("main.kcl"))[0] == b"// main"

    def forbidden(request):
        return httpx.Response(403, json={"message": "Access revoked"})

    project_transport(forbidden)
    with pytest.raises(ZooMCPException, match="403.*Access revoked"):
        await fs.read_file("main.kcl")


@pytest.mark.asyncio
async def test_preview_metadata_changes_refresh_bytes_without_a_new_revision(api):
    api.files["thumbnail.png"] = b"old preview"
    fs = ProjectFileSystem(PROJECT_ID)
    assert (await fs.read_file("thumbnail.png"))[0] == b"old preview"
    api.files["thumbnail.png"] = b"generated preview"
    assert (await fs.read_file("thumbnail.png"))[0] == b"generated preview"


@pytest.mark.asyncio
async def test_files_without_hashes_are_downloaded_again(api, project_transport):
    def handle(request):
        response = api.handle(request)
        if "/versions/" not in request.url.path:
            metadata = response.json()
            for entry in metadata["files"]:
                entry["sha256"] = None
            return httpx.Response(200, json=metadata)
        return response

    project_transport(handle)
    fs = ProjectFileSystem(PROJECT_ID)
    assert (await fs.read_file("main.kcl"))[0] == b"// main"
    assert (await fs.read_file("main.kcl"))[0] == b"// main"
    assert api.downloaded_paths(api.requests) == ["main.kcl", "main.kcl"]


@pytest.mark.asyncio
async def test_cache_evicts_old_file_bytes_when_full(api, monkeypatch):
    monkeypatch.setattr(zoo_tools, "_PROJECT_CACHE_BYTES", 8)
    fs = ProjectFileSystem(PROJECT_ID)
    assert (await fs.read_file("main.kcl"))[0] == b"// main"
    assert (await fs.read_file("parts/a.bin"))[0] == b"\xff\x00"
    assert (await fs.read_file("main.kcl"))[0] == b"// main"
    assert api.downloaded_paths(api.requests) == ["main.kcl", "parts/a.bin", "main.kcl"]


@pytest.mark.asyncio
async def test_cache_limits_empty_entries_and_reuses_recent_request_scopes(
    api, monkeypatch
):
    monkeypatch.setattr(zoo_tools, "_PROJECT_CACHE_FILES", 2)
    api.files["empty"] = b""
    for scope in ["a", "b", "a", "c", "a", "b"]:
        with use_client_config(
            ZooClientConfig(
                "test-token", "https://api.example.test/custom/", {"X-Cache-Tag": scope}
            )
        ):
            assert (await ProjectFileSystem(PROJECT_ID).read_file("empty"))[0] == b""
    assert [
        request.headers["X-Cache-Tag"]
        for request in api.requests
        if "/versions/" in request.url.path
    ] == ["a", "b", "c", "b"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "second",
    [
        ZooClientConfig(
            "different-token", "https://api.example.test/custom/", {"X-Scope": "one"}
        ),
        ZooClientConfig(
            "test-token", "https://other.example.test/custom/", {"X-Scope": "one"}
        ),
        ZooClientConfig(
            "test-token", "https://api.example.test/custom/", {"X-Scope": "two"}
        ),
    ],
)
async def test_cache_is_scoped_to_host_credentials_and_headers(httpx_mock, second):
    remote = ProjectAPI()
    metadata = remote.metadata()
    downloads = []

    def handle(request):
        if request.url.path.endswith("/download"):
            downloads.append(request)
            return httpx.Response(200, content=project_zip([("main.kcl", b"// main")]))
        return httpx.Response(200, json=metadata)

    httpx_mock.add_callback(handle, is_reusable=True)
    first = ZooClientConfig(
        "test-token", "https://api.example.test/custom/", {"X-Scope": "one"}
    )
    for config in [first, first, second]:
        with use_client_config(config):
            assert (await ProjectFileSystem(PROJECT_ID).read_file("main.kcl"))[
                0
            ] == b"// main"
    assert len(downloads) == 2
    assert downloads[1].headers["Authorization"] == f"Bearer {second.token}"
    assert downloads[1].headers["X-Scope"] == second.headers["X-Scope"]


@pytest.mark.asyncio
async def test_files_larger_than_the_cache_budget_are_not_retained(api, monkeypatch):
    monkeypatch.setattr(zoo_tools, "_PROJECT_CACHE_BYTES", 8)
    api.files["large.bin"] = b"123456789"
    fs = ProjectFileSystem(PROJECT_ID)
    for _ in range(2):
        assert (await fs.read_file("large.bin"))[0] == b"123456789"
    assert api.downloaded_paths(api.requests) == ["large.bin", "large.bin"]


@pytest.mark.asyncio
async def test_cached_bytes_do_not_bypass_a_lower_project_size_limit(api):
    assert (await ProjectFileSystem(PROJECT_ID).read_file("main.kcl"))[0] == b"// main"
    size = sum(map(len, api.files.values()))
    with pytest.raises(ZooMCPException, match="size limit"):
        await ProjectFileSystem(PROJECT_ID, size - 1).read_file("main.kcl")


@pytest.mark.asyncio
async def test_cache_is_separate_for_projects_on_the_same_host(httpx_mock):
    remote = ProjectAPI()
    downloads = []

    def handle(request):
        if request.url.path.endswith("/download"):
            downloads.append(request.url.path)
            return httpx.Response(200, content=project_zip([("main.kcl", b"// main")]))
        metadata = remote.metadata()
        metadata["id"] = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, json=metadata)

    httpx_mock.add_callback(handle, is_reusable=True)
    with use_client_config(
        ZooClientConfig("test-token", "https://api.example.test/custom/")
    ):
        for project_id in [
            PROJECT_ID,
            PROJECT_ID,
            "cloud:00000000-0000-4000-8000-000000000002",
        ]:
            assert (await ProjectFileSystem(project_id).read_file("main.kcl"))[
                0
            ] == b"// main"
    assert len(downloads) == 2
    assert downloads[0] != downloads[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_operation", ["stat", "readdir"])
async def test_shared_filesystem_returns_each_reads_own_revision(
    api, project_transport, metadata_operation
):
    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowFile(httpx.AsyncByteStream):
        async def __aiter__(self):
            entered.set()
            await release.wait()
            yield project_zip([("main.kcl", b"// main")])

    def handle(request):
        response = api.handle(request)
        if "/versions/" in request.url.path:
            return httpx.Response(200, headers=response.headers, stream=SlowFile())
        return response

    project_transport(handle)
    fs = ProjectFileSystem(PROJECT_ID)
    reading = asyncio.create_task(fs.read_file("main.kcl"))
    try:
        await entered.wait()
        api.files["main.kcl"] = b"// user edit"
        api.revision += 1
        latest = await getattr(fs, metadata_operation)()
        release.set()
        contents, revision = await reading
        assert contents == b"// main"
        assert revision == "1"
        assert latest["revision"] == "2"
        with pytest.raises(ZooMCPException, match="changed"):
            await project_write_file(
                PROJECT_ID, "main.kcl", "// stale edit", expected_revision=revision
            )
        assert api.files["main.kcl"] == b"// user edit"
        assert not api.uploads
    finally:
        reading.cancel()
        await asyncio.gather(reading, return_exceptions=True)


@pytest.mark.asyncio
async def test_read_returns_the_revision_the_agent_can_use_for_its_write(api):
    from zoo_mcp.server import project_read_file

    result = await project_read_file(PROJECT_ID, "main.kcl")
    assert result["contents"] == "// main"
    assert result["revision"] == "1"
    saved = await project_write_file(
        PROJECT_ID, "main.kcl", "// edited", expected_revision=result["revision"]
    )
    assert saved["revision"] == "2"
    assert api.files["main.kcl"] == b"// edited"


@pytest.mark.asyncio
@pytest.mark.parametrize("during_upload", [False, True])
@pytest.mark.parametrize(
    "tool,arguments",
    [
        (
            "project_write_file",
            {"path": "main.kcl", "contents": "// stale replacement"},
        ),
        ("project_mkdir", {"path": "new"}),
        ("project_rm", {"path": "parts/a.bin"}),
        ("project_cp", {"source": "parts/a.bin", "destination": "copied.bin"}),
        ("project_mv", {"source": "parts/a.bin", "destination": "moved.bin"}),
    ],
)
async def test_observed_revision_rejects_changes_without_reapplying_the_write(
    api, during_upload, tool, arguments
):
    from zoo_mcp import server

    if during_upload:
        api.failure = "conflict"
    else:
        api.files["main.kcl"] = b"// user edit"
        api.revision += 1
    with pytest.raises(ZooMCPException, match="changed"):
        await getattr(server, tool)(PROJECT_ID, **arguments, expected_revision="1")
    assert api.files["main.kcl"] == (b"// main" if during_upload else b"// user edit")
    assert len(api.uploads) == (1 if during_upload else 0)


@pytest.mark.asyncio
async def test_stale_write_reports_conflict_before_downloading_files(
    api, project_transport
):
    api.files["main.kcl"] = b"// user edit"
    api.revision += 1

    def handle(request):
        if "/versions/" in request.url.path:
            raise httpx.ReadTimeout("file download timed out", request=request)
        return api.handle(request)

    project_transport(handle)
    with pytest.raises(ZooMCPException, match="Project changed since it was read"):
        await project_write_file(
            PROJECT_ID, "main.kcl", "// stale replacement", expected_revision="1"
        )
    assert api.files["main.kcl"] == b"// user edit"
    assert not api.uploads


@pytest.mark.asyncio
async def test_read_revision_matches_its_contents_when_the_cloud_changes_during_download(
    api,
):
    from zoo_mcp.server import project_read_file

    def edit():
        api.files["main.kcl"] = b"// user edit"
        api.revision += 1

    api.download_hook = edit
    result = await project_read_file(PROJECT_ID, "main.kcl")
    assert result["contents"] == "// main"
    assert result["revision"] == "1"
    with pytest.raises(ZooMCPException, match="changed"):
        await project_write_file(
            PROJECT_ID,
            "main.kcl",
            "// stale edit",
            expected_revision=result["revision"],
        )
    assert api.files["main.kcl"] == b"// user edit"
    assert not api.uploads
