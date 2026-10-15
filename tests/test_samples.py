import asyncio
import io
import json
import stat
import warnings
import zipfile

import httpx
import pytest
from kittycad.exceptions import KittyCADServerError
from mcp.types import CallToolResult, TextContent
from pytest_httpx import HTTPXMock

from zoo_mcp import kcl_samples, server
from zoo_mcp.credentials import ZooCredentials, use_credentials
from zoo_mcp.kcl_docs import KCLDocs
from zoo_mcp.kcl_samples import KCLSamples, SampleMetadata

_PROJECT_ID = "e9996d92-3ec1-447d-b91e-a68188b41b07"
_OTHER_ID = "e11a54c2-7bb8-4bbe-bba9-e6b75d3a67ed"
_LIST_URL = "https://api.zoo.dev/projects/public?limit=100"
_DOWNLOAD_URL = f"https://api.zoo.dev/projects/public/{_PROJECT_ID}/download?format=zip"


def _project(project_id: str = _PROJECT_ID, title: str = "Cycloidal Gear") -> dict:
    return {
        "id": project_id,
        "title": title,
        "description": "A gear (see https://example.com) with several parts.",
        "categories": [],
        "owner": {"username": "creator"},
        "published_at": "2026-10-09T00:00:00Z",
        "like_count": 0,
        "new_backend_field": "ignored",
    }


def _archive(files: list[tuple[str | zipfile.ZipInfo, str | bytes]]) -> bytes:
    content = io.BytesIO()
    with (
        zipfile.ZipFile(content, "w", compression=zipfile.ZIP_DEFLATED) as archive,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", UserWarning)
        for filename, source in files:
            archive.writestr(filename, source)
    return content.getvalue()


@pytest.fixture(autouse=True)
def reset_samples(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(KCLSamples, "_instance", None)
    monkeypatch.setattr(KCLSamples, "_init_lock", None)


@pytest.fixture
def samples_index(monkeypatch: pytest.MonkeyPatch) -> KCLSamples:
    samples = KCLSamples(
        manifest={
            _PROJECT_ID: SampleMetadata(
                title="Cycloidal Gear",
                description="A gear assembly",
                multipleFiles=False,
            )
        }
    )
    monkeypatch.setattr(KCLSamples, "_instance", samples)
    return samples


@pytest.mark.asyncio
async def test_index_follows_all_pages_without_credentials(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ZOO_HOST", "https://alternate.example")
    httpx_mock.add_response(
        url=_LIST_URL,
        json={"items": [_project()], "next_page": "second"},
    )
    # An empty page with a cursor must not stop discovery.
    httpx_mock.add_response(
        url=f"{_LIST_URL}&page_token=second",
        json={"items": [], "next_page": "third"},
    )
    httpx_mock.add_response(
        url=f"{_LIST_URL}&page_token=third",
        json={"items": [_project(_OTHER_ID, "Axial Fan")], "next_page": None},
    )

    with use_credentials(
        ZooCredentials(token="caller-token", base_url="https://private.example")
    ):
        await KCLSamples.initialize()

    samples = KCLSamples.get()
    assert set(samples.manifest) == {_PROJECT_ID, _OTHER_ID}
    assert samples.manifest[_PROJECT_ID]["description"] == _project()["description"]
    assert samples.manifest[_OTHER_ID]["title"] == "Axial Fan"
    assert samples.file_index == {}
    assert all("authorization" not in r.headers for r in httpx_mock.get_requests())
    assert {r.url.host for r in httpx_mock.get_requests()} == {"api.zoo.dev"}


@pytest.mark.asyncio
async def test_initialize_retries_an_empty_listing(httpx_mock: HTTPXMock):
    httpx_mock.add_response(url=_LIST_URL, json={"items": []})
    httpx_mock.add_response(url=_LIST_URL, json={"items": [_project()]})

    await KCLSamples.initialize()
    assert KCLSamples._instance is None
    await KCLSamples.initialize()
    assert _PROJECT_ID in KCLSamples.get().manifest


@pytest.mark.asyncio
async def test_failed_later_page_does_not_cache_partial_index(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url=_LIST_URL, json={"items": [_project()], "next_page": "second"}
    )
    httpx_mock.add_response(
        url=f"{_LIST_URL}&page_token=second", status_code=503, json={"message": "Down"}
    )

    with pytest.raises(KittyCADServerError):
        await KCLSamples.initialize()
    assert KCLSamples._instance is None
    assert kcl_samples.list_available_samples() == []


@pytest.fixture
def initialized_docs(monkeypatch: pytest.MonkeyPatch) -> KCLDocs:
    docs = KCLDocs(docs={"docs/kcl-lang/functions": "# Functions"})
    monkeypatch.setattr(KCLDocs, "_instance", docs)
    monkeypatch.setattr(server, "_kcl_index_task", None)
    return docs


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["later_page", "empty_listing"])
async def test_sample_tools_retry_an_uninitialized_index(
    httpx_mock: HTTPXMock, initialized_docs: KCLDocs, failure: str
):
    if failure == "later_page":
        httpx_mock.add_response(
            url=_LIST_URL, json={"items": [_project()], "next_page": "second"}
        )
        httpx_mock.add_response(
            url=f"{_LIST_URL}&page_token=second",
            status_code=503,
            json={"message": "Down"},
        )
    else:
        httpx_mock.add_response(url=_LIST_URL, json={"items": []})
    httpx_mock.add_response(
        url=_LIST_URL, json={"items": [_project(_OTHER_ID, "Axial Fan")]}
    )

    first = await server.mcp.call_tool("list_kcl_samples", arguments={})
    assert isinstance(first, CallToolResult)
    assert not first.is_error
    assert not first.content
    assert KCLSamples._instance is None

    recovered = await server.mcp.call_tool(
        "search_kcl_samples", arguments={"query": "fan"}
    )
    assert isinstance(recovered, CallToolResult)
    assert not recovered.is_error
    assert len(recovered.content) == 1
    assert isinstance(recovered.content[0], TextContent)
    assert json.loads(recovered.content[0].text)["name"] == _OTHER_ID
    assert set(KCLSamples.get().manifest) == {_OTHER_ID}
    assert KCLDocs.get() is initialized_docs

    # A successful retry is reused without fetching the index again.
    await server.mcp.call_tool("list_kcl_samples", arguments={})
    assert len(httpx_mock.get_requests()) == (3 if failure == "later_page" else 2)


@pytest.mark.asyncio
async def test_concurrent_tools_share_an_index_retry(
    monkeypatch: pytest.MonkeyPatch, initialized_docs: KCLDocs
):
    attempts = 0
    retry_started = asyncio.Event()
    release_retry = asyncio.Event()

    async def fetch() -> KCLSamples:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("Temporary listing failure")
        retry_started.set()
        await release_retry.wait()
        return KCLSamples(
            manifest={
                _PROJECT_ID: SampleMetadata(
                    title="Cycloidal Gear", description="", multipleFiles=False
                )
            }
        )

    monkeypatch.setattr(kcl_samples, "_fetch_index_from_api", fetch)
    await server.mcp.call_tool("list_kcl_samples", arguments={})
    calls = [
        asyncio.create_task(server.mcp.call_tool("list_kcl_samples", arguments={}))
        for _ in range(5)
    ]
    try:
        async with asyncio.timeout(5):
            await retry_started.wait()
            assert attempts == 2
            release_retry.set()
            results = await asyncio.gather(*calls)
        for result in results:
            assert isinstance(result, CallToolResult)
            assert not result.is_error
            assert len(result.content) == 1
            assert isinstance(result.content[0], TextContent)
            assert json.loads(result.content[0].text)["name"] == _PROJECT_ID
        assert attempts == 2
        assert KCLDocs.get() is initialized_docs
    finally:
        release_retry.set()
        await asyncio.gather(*calls, return_exceptions=True)


@pytest.mark.asyncio
async def test_concurrent_initialization_fetches_once(httpx_mock: HTTPXMock):
    httpx_mock.add_response(url=_LIST_URL, json={"items": [_project()]})

    await asyncio.gather(KCLSamples.initialize(), KCLSamples.initialize())
    await KCLSamples.initialize()

    assert len(httpx_mock.get_requests()) == 1
    assert _PROJECT_ID in KCLSamples.get().manifest


def test_archive_preserves_nested_paths_and_ignores_other_files():
    files = {
        "main.kcl": 'import "parts/fan.kcl" as fan\n',
        "parts/fan.kcl": "blades = 5\n",
        "parts/부품 name.kcl": "radius = 10\n",
    }
    archive = _archive(
        list(files.items())
        + [
            ("project.toml", "[settings]\n"),
            ("thumbnail.png", b"\x89PNG"),
            ("parts/", ""),
        ]
    )

    assert kcl_samples._parse_project_archive(archive) == files


@pytest.mark.parametrize(
    "filename",
    [
        "../evil.kcl",
        "/absolute.kcl",
        "C:/evil.kcl",
        "parts\\evil.kcl",
        "parts/../../evil.kcl",
        "./main.kcl",
        "parts//fan.kcl",
        "%2e%2e/evil.kcl",
        "%2fabsolute.kcl",
    ],
)
def test_archive_rejects_unsafe_paths(filename: str):
    with pytest.raises(ValueError, match="Unsafe"):
        kcl_samples._parse_project_archive(_archive([(filename, "x = 1")]))


def test_archive_rejects_symlink():
    member = zipfile.ZipInfo("link.kcl")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16

    with pytest.raises(ValueError, match="Unsafe"):
        kcl_samples._parse_project_archive(_archive([(member, "main.kcl")]))


def test_archive_rejects_duplicate_paths():
    with pytest.raises(ValueError, match="duplicate"):
        kcl_samples._parse_project_archive(
            _archive([("main.kcl", "x = 1"), ("main.kcl", "x = 2")])
        )


@pytest.mark.parametrize("content", [b"not a zip", _archive([("main.kcl", b"\xff")])])
def test_archive_rejects_invalid_zip_or_source(content: bytes):
    with pytest.raises((zipfile.BadZipFile, UnicodeDecodeError)):
        kcl_samples._parse_project_archive(content)


@pytest.mark.asyncio
async def test_get_sample_downloads_and_caches_kcl_files(
    httpx_mock: HTTPXMock, samples_index: KCLSamples
):
    httpx_mock.add_response(
        url=_DOWNLOAD_URL,
        content=_archive(
            [
                ("main.kcl", 'import "parts/fan.kcl" as fan\n'),
                ("parts/fan.kcl", "x = 1"),
            ]
        ),
    )
    assert kcl_samples.list_available_samples()[0]["multipleFiles"] is False

    with use_credentials(
        ZooCredentials(token="caller-token", base_url="https://private.example")
    ):
        result = await kcl_samples.get_sample_content(_PROJECT_ID)
    assert result is not None
    assert result["name"] == _PROJECT_ID
    assert result["multipleFiles"] is True
    assert [file["filename"] for file in result["files"]] == [
        "main.kcl",
        "parts/fan.kcl",
    ]
    assert await kcl_samples.get_sample_content(_PROJECT_ID) == result
    assert kcl_samples.list_available_samples()[0]["multipleFiles"] is True
    assert kcl_samples.search_samples("gear")[0]["multipleFiles"] is True
    assert len(httpx_mock.get_requests()) == 1
    assert "authorization" not in httpx_mock.get_requests()[0].headers


@pytest.mark.asyncio
async def test_deleted_sample_is_not_cached(
    httpx_mock: HTTPXMock, samples_index: KCLSamples
):
    httpx_mock.add_response(url=_DOWNLOAD_URL, status_code=404)

    assert await kcl_samples.get_sample_content(_PROJECT_ID) is None
    assert samples_index.file_index == {}


@pytest.mark.asyncio
async def test_download_failure_does_not_cache_files(
    httpx_mock: HTTPXMock, samples_index: KCLSamples
):
    httpx_mock.add_response(url=_DOWNLOAD_URL, status_code=503)

    with pytest.raises(httpx.HTTPStatusError):
        await kcl_samples.get_sample_content(_PROJECT_ID)
    assert samples_index.file_index == {}


@pytest.mark.asyncio
async def test_download_rejects_redirect(
    httpx_mock: HTTPXMock, samples_index: KCLSamples
):
    httpx_mock.add_response(
        url=_DOWNLOAD_URL, status_code=302, headers={"location": "https://example.com"}
    )

    with pytest.raises(httpx.HTTPStatusError):
        await kcl_samples.get_sample_content(_PROJECT_ID)
    assert samples_index.file_index == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["unknown-project", "../main", "%2e%2e", ""])
async def test_unknown_or_unsafe_name_makes_no_request(
    name: str, samples_index: KCLSamples
):
    assert await kcl_samples.get_sample_content(name) is None


def test_empty_queries_and_search_by_title(samples_index: KCLSamples):
    assert "error" in kcl_samples.search_samples("")[0]
    assert "error" in kcl_samples.search_samples("   ")[0]
    assert kcl_samples.search_samples("missing") == []
    assert kcl_samples.search_samples("gear")[0]["name"] == _PROJECT_ID
