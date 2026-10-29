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

import httpx
import pytest
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent

from zoo_mcp import ZooMCPException, credentials, zoo_tools
from zoo_mcp.credentials import ZooCredentials
from zoo_mcp.server import mcp
from zoo_mcp.zoo_tools import ProjectFileSystem

PROJECT_ID = "cloud:00000000-0000-4000-8000-000000000001"


class ProjectAPI:
    def __init__(self):
        self.files: dict[str, bytes] = {
            "main.kcl": b"// main",
            "project.toml": b"[settings]",
            "parts/a.bin": b"\xff\x00",
        }
        self.revision = 1
        self.requests = []
        self.uploads = []
        self.failure = None
        self.editable = True
        self.download_hook = None
        self.archive_files: dict[str, bytes] | None = None

    def metadata(self):
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
                    "id": path,
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
        if request.method == "GET" and request.url.path.endswith("/download"):
            assert request.url.params["format"] == "zip"
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for path, content in (
                    self.archive_files if self.archive_files is not None else self.files
                ).items():
                    z.writestr(path, content)
            if self.download_hook:
                hook, self.download_hook = self.download_hook, None
                hook()
            return httpx.Response(200, content=archive.getvalue())
        if request.method == "GET":
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
        assert body["expected_revision"] == str(self.revision)
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
        credentials,
        "_local_credentials",
        ZooCredentials("test-token", "https://api.example.test/custom/"),
    )
    remote = ProjectAPI()
    httpx_mock.add_callback(remote.handle, is_reusable=True, is_optional=True)
    return remote


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
    "path", ["../a", "parts/../a", "/etc/passwd", "C:/a", "a\\b", "a\x00b", " a", "a "]
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
    await ProjectFileSystem(PROJECT_ID).write_file(
        "parts/a.bin", b"latest requested write"
    )
    assert len(api.uploads) == 2
    assert api.files["concurrent.txt"] == b"preserve concurrent edit"
    assert api.files["parts/a.bin"] == b"latest requested write"
    assert api.uploads[1][0]["expected_revision"] == "2"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,uploads", [("timeout_before", 2), ("timeout_after", 1)]
)
@pytest.mark.parametrize("operation", ["mkdir", "mv", "rm", "write_file"])
async def test_uncertain_upload_is_reconciled(api, failure, uploads, operation):
    api.failure = failure
    fs = ProjectFileSystem(PROJECT_ID)
    args = {
        "mkdir": ["new"],
        "mv": ["parts/a.bin", "moved.bin"],
        "rm": ["parts/a.bin"],
        "write_file": ["parts/a.bin", b"changed"],
    }[operation]
    await getattr(fs, operation)(*args)
    assert len(api.uploads) == uploads


@pytest.mark.asyncio
async def test_uncertain_upload_does_not_clobber_ambiguous_edit(api):
    api.failure = "ambiguous"
    with pytest.raises(ZooMCPException, match="uncertain"):
        await ProjectFileSystem(PROJECT_ID).write_file("parts/a.bin", b"mine")
    assert len(api.uploads) == 1
    assert api.files["parts/a.bin"] == b"someone else's edit"


@pytest.mark.asyncio
async def test_snapshot_retries_download_racing_metadata(api):
    def edit():
        api.files["main.kcl"] = b"// newer revision"
        api.revision += 1

    api.download_hook = edit
    assert (
        await ProjectFileSystem(PROJECT_ID).read_file("main.kcl")
        == b"// newer revision"
    )
    assert sum(request.url.path.endswith("/download") for request in api.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["metadata", "archive", "mutation"])
async def test_limit_at_each_boundary(api, stage):
    current_size = sum(map(len, api.files.values()))
    fs = ProjectFileSystem(
        PROJECT_ID, current_size if stage == "mutation" else current_size - 1
    )
    if stage == "archive":
        fs.limit = current_size
        api.archive_files = {"main.kcl": b"x" * (current_size + 1)}
    with pytest.raises(ZooMCPException, match="size limit"):
        await fs.write_file("new", b"larger")
    assert not api.uploads
    if stage == "metadata":
        assert len(api.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_archive", [{"../escape": b"x"}, {"main.kcl": b"wrong hash"}]
)
async def test_archive_path_and_integrity_validation(api, bad_archive):
    api.archive_files = bad_archive
    with pytest.raises(ZooMCPException):
        await ProjectFileSystem(PROJECT_ID).read_file("main.kcl")
    assert not api.uploads


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
async def test_mcp_binary_round_trip_and_errors(api):
    encoded = base64.b64encode(b"\xff\x00").decode()
    result = await mcp.call_tool(
        "project_write_file",
        {
            "project_id": PROJECT_ID,
            "path": "binary",
            "contents": encoded,
            "encoding": "base64",
        },
    )
    assert isinstance(result, CallToolResult)
    assert not result.is_error
    result = await mcp.call_tool(
        "project_read_file",
        {"project_id": PROJECT_ID, "path": "binary", "encoding": "base64"},
    )
    assert isinstance(result, CallToolResult)
    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text) == {
        "contents": encoded,
        "encoding": "base64",
    }
    with pytest.raises(ToolError, match="main.kcl"):
        await mcp.call_tool(
            "project_rm", {"project_id": PROJECT_ID, "path": "main.kcl"}
        )


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
async def test_retry_budget(api, monkeypatch):
    original = api.handle
    attempts = 0

    def conflict_forever(request):
        nonlocal attempts
        if request.method == "PUT":
            attempts += 1
            return httpx.Response(409)
        return original(request)

    monkeypatch.setattr(
        ProjectFileSystem,
        "_client",
        lambda _: httpx.AsyncClient(
            transport=httpx.MockTransport(conflict_forever),
            headers={"Authorization": "Bearer test-token"},
        ),
    )
    with pytest.raises(ZooMCPException, match="three attempts"):
        await ProjectFileSystem(PROJECT_ID).rm("parts")
    assert attempts == 3


@pytest.mark.asyncio
async def test_conflicting_remove_includes_new_descendants(api, monkeypatch):
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

    monkeypatch.setattr(
        ProjectFileSystem,
        "_client",
        lambda _: httpx.AsyncClient(
            transport=httpx.MockTransport(handle),
            headers={"Authorization": "Bearer test-token"},
        ),
    )
    await ProjectFileSystem(PROJECT_ID).rm("parts")
    assert "parts/new.bin" in api.uploads[-1][0]["deleted_paths"]
    assert all(not name.startswith("parts/") for name in api.files)


@pytest.mark.asyncio
async def test_credentials_are_captured_and_locks_are_scoped(api):
    fs = ProjectFileSystem(PROJECT_ID)
    from zoo_mcp.credentials import use_credentials

    with use_credentials(
        ZooCredentials("different-token", "https://other.example.test")
    ):
        other = ProjectFileSystem(PROJECT_ID)
        lock = fs._lock()
        assert other._lock() is not lock
        assert ProjectFileSystem(PROJECT_ID)._lock() is other._lock()
        assert await fs.read_file("main.kcl") == b"// main"


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

    monkeypatch.setattr(ProjectFileSystem, "_snapshot", blocked)
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
@pytest.mark.parametrize(
    "operation,args", [("mkdir", ["new"]), ("mv", ["parts/a.bin", "moved.bin"])]
)
async def test_timed_out_upload_commits_while_retry_is_in_flight(
    api, monkeypatch, operation, args
):
    original = api.handle
    attempts = 0
    api.failure = "timeout_before"

    def handle(request):
        nonlocal attempts
        if request.method == "PUT":
            attempts += 1
            if attempts == 2:
                # The first request finished after reconciliation fetched the old
                # revision, but before the second request checked its revision.
                api.files = api.uploads[0][1]
                api.revision += 1
                return httpx.Response(409)
        return original(request)

    monkeypatch.setattr(
        ProjectFileSystem,
        "_client",
        lambda _: httpx.AsyncClient(
            transport=httpx.MockTransport(handle),
            headers={"Authorization": "Bearer test-token"},
        ),
    )
    await getattr(ProjectFileSystem(PROJECT_ID), operation)(*args)
    assert attempts == 2


@pytest.mark.asyncio
async def test_unreconcilable_upload_has_explicit_uncertain_error(api, monkeypatch):
    original = api.handle
    api.failure = "timeout_after"

    def handle(request):
        if api.uploads and request.method == "GET":
            return httpx.Response(503)
        return original(request)

    monkeypatch.setattr(
        ProjectFileSystem,
        "_client",
        lambda _: httpx.AsyncClient(
            transport=httpx.MockTransport(handle),
            headers={"Authorization": "Bearer test-token"},
        ),
    )
    with pytest.raises(ZooMCPException, match="uncertain; could not reload"):
        await ProjectFileSystem(PROJECT_ID).mkdir("new")
    assert len(api.uploads) == 1


@pytest.mark.asyncio
async def test_project_download_follows_no_redirects(api, monkeypatch):
    original = api.handle

    def handle(request):
        if request.url.path.endswith("/download"):
            return httpx.Response(
                302, headers={"Location": "https://other.example.test/archive.zip"}
            )
        return original(request)

    monkeypatch.setattr(
        ProjectFileSystem,
        "_client",
        lambda _: httpx.AsyncClient(
            transport=httpx.MockTransport(handle),
            headers={"Authorization": "Bearer test-token"},
            follow_redirects=False,
        ),
    )
    with pytest.raises(httpx.HTTPStatusError):
        await ProjectFileSystem(PROJECT_ID).read_file("main.kcl")
    assert not api.uploads


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "contents,options,expected",
    [
        (b"\xff\x00\x80", {}, b"\xff\x00\x80"),
        (b"", {}, b""),
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
