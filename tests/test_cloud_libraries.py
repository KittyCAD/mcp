import asyncio
import json

import httpx
import pytest
from mcp.types import CallToolResult, TextContent

from zoo_mcp import ZooMCPException, client_config, cloud_libraries
from zoo_mcp.api_call_events import capture_api_call_events
from zoo_mcp.client_config import ZooClientConfig, use_client_config
from zoo_mcp.server import mcp

LIBRARY = "00000000-0000-4000-8000-000000000001"
PROJECT = "00000000-0000-4000-8000-000000000002"
VERSION = "00000000-0000-4000-8000-000000000003"


@pytest.fixture(autouse=True)
def caller():
    with use_client_config(
        ZooClientConfig(
            "caller-token",
            "https://api.example.test/custom/",
            headers={
                "User-Agent": "ExampleClient/1.0",
                "X-Request-Source": "example",
            },
        )
    ):
        yield


@pytest.mark.asyncio
async def test_library_pages_keep_cursor_auth_headers_and_backend_ids(httpx_mock):
    httpx_mock.add_response(
        json={"items": [], "next_page": "next-page"},
        headers={"x-api-call-id": "first-request"},
    )
    library = {
        "id": LIBRARY,
        "name": "Converted parts",
        "source": "org_dataset",
        "org_id": PROJECT,
    }
    httpx_mock.add_response(json={"items": [library], "next_page": None})
    with capture_api_call_events() as events:
        first = await cloud_libraries.list_cloud_libraries(limit=5)
    assert first == {"items": [], "next_page": "next-page"}
    second = await cloud_libraries.list_cloud_libraries(
        limit=5, page_token=first["next_page"]
    )
    assert second == {"items": [library], "next_page": None}
    requests = httpx_mock.get_requests()
    assert [dict(r.url.params) for r in requests] == [
        {"limit": "5"},
        {"limit": "5", "page_token": "next-page"},
    ]
    for request in requests:
        assert request.url.path == "/custom/user/project-libraries"
        assert request.headers["authorization"] == "Bearer caller-token"
        assert request.headers["user-agent"] == "ExampleClient/1.0"
        assert request.headers["X-Request-Source"] == "example"
    assert events[0].api_call_id == "first-request"
    assert events[0].operation == "list_cloud_libraries"


@pytest.mark.asyncio
@pytest.mark.parametrize("library", [None, LIBRARY])
async def test_browse_projects_returns_filesystem_ids(httpx_mock, library):
    httpx_mock.add_response(
        json={
            "items": [
                {
                    "id": PROJECT,
                    "title": "Bracket",
                    "library_id": library,
                    "new_field": "ignored",
                }
            ],
            "next_page": "opaque",
        }
    )
    result = await cloud_libraries.list_cloud_projects(library_id=library)
    assert result["items"] == [
        {
            "id": PROJECT,
            "title": "Bracket",
            "library_id": library,
            "project_id": f"cloud:{PROJECT}",
        }
    ]
    assert result["next_page"] == "opaque"
    assert httpx_mock.get_request().url.path == (
        f"/custom/user/project-libraries/{LIBRARY}/projects"
        if library
        else "/custom/user/projects"
    )


@pytest.mark.asyncio
async def test_unassigned_personal_projects(httpx_mock):
    httpx_mock.add_response(json={"items": [], "next_page": None})
    await cloud_libraries.list_cloud_projects(
        scope="personal", unassigned=True, page_token="cursor"
    )
    assert dict(httpx_mock.get_request().url.params) == {
        "limit": "20",
        "scope": "personal",
        "unassigned": "true",
        "page_token": "cursor",
    }


@pytest.mark.asyncio
async def test_semantic_search_stays_in_active_org_and_preserves_version(httpx_mock):
    match = {
        "project_id": PROJECT,
        "version_id": VERSION,
        "title": "Bracket",
        "description": "Mounting bracket",
        "similarity": 0.9,
    }
    httpx_mock.add_response(json=[match])
    result = await cloud_libraries.search_cloud_projects(
        "  mounting bracket  ", LIBRARY, limit=3
    )
    assert result == {"matches": [{**match, "project_id": f"cloud:{PROJECT}"}]}
    request = httpx_mock.get_request()
    assert request.method == "POST"
    assert request.url.path == "/custom/org/projects/search/semantic"
    assert json.loads(request.content) == {
        "q": "mounting bracket",
        "limit": 3,
        "library_id": LIBRARY,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 404, 429, 500, 302])
async def test_http_errors_are_actionable_and_redacted(httpx_mock, status):
    httpx_mock.add_response(
        status_code=status,
        text="caller-token private-server-error",
        headers={"location": "https://elsewhere.test"},
    )
    with pytest.raises(ZooMCPException) as exc:
        await cloud_libraries.search_cloud_projects("bracket")
    assert "caller-token" not in str(exc.value)
    assert "private-server-error" not in str(exc.value)
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"limit": 0},
        {"limit": 101},
        {"limit": 1.5},
        {"limit": "2"},
        {"limit": True},
        {"library_id": "../other"},
        {"scope": "other"},
        {"library_id": LIBRARY, "unassigned": True},
    ],
)
async def test_bad_browse_arguments_do_not_make_requests(httpx_mock, kwargs):
    with pytest.raises(ZooMCPException):
        await cloud_libraries.list_cloud_projects(**kwargs)
    assert not httpx_mock.get_requests()


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [" ", "a" * 2049])
async def test_bad_queries_do_not_make_requests(httpx_mock, query):
    with pytest.raises(ZooMCPException, match="query must"):
        await cloud_libraries.search_cloud_projects(query)
    assert not httpx_mock.get_requests()


@pytest.mark.asyncio
async def test_explicit_empty_credentials_do_not_use_local_login(httpx_mock):
    with use_client_config(ZooClientConfig("")), pytest.raises(ZooMCPException):
        await cloud_libraries.list_cloud_libraries()
    assert not httpx_mock.get_requests()


@pytest.mark.asyncio
async def test_missing_startup_token_preserves_login_guidance(httpx_mock, monkeypatch):
    monkeypatch.setattr(client_config, "_local_config", ZooClientConfig(""))
    monkeypatch.setattr(client_config, "_saved_login", None)
    monkeypatch.setattr(client_config, "_login_message", "Run zoo-mcp auth login.")
    previous = client_config._client_config.set(None)
    try:
        with pytest.raises(ZooMCPException, match="auth login"):
            await cloud_libraries.list_cloud_libraries()
    finally:
        client_config._client_config.reset(previous)
    assert not httpx_mock.get_requests()


@pytest.mark.asyncio
async def test_parallel_callers_keep_their_credentials(httpx_mock):
    async def respond(request):
        await asyncio.sleep(0)
        return httpx.Response(
            200,
            json={
                "items": [{"id": LIBRARY, "name": request.headers["authorization"]}],
                "next_page": None,
            },
        )

    httpx_mock.add_callback(respond, is_reusable=True)

    async def call(token):
        with use_client_config(ZooClientConfig(token, "https://api.example.test")):
            return await cloud_libraries.list_cloud_libraries()

    results = await asyncio.gather(call("first"), call("second"))
    assert [result["items"][0]["name"] for result in results] == [
        "Bearer first",
        "Bearer second",
    ]


@pytest.mark.asyncio
async def test_bounded_response(httpx_mock, monkeypatch):
    monkeypatch.setattr(cloud_libraries, "_MAX_RESPONSE_BYTES", 8)
    httpx_mock.add_response(content=b"x" * 9)
    with pytest.raises(ZooMCPException, match="too large"):
        await cloud_libraries.list_cloud_libraries()


@pytest.mark.asyncio
async def test_malformed_project_identity_is_an_actionable_error(httpx_mock):
    httpx_mock.add_response(
        json={"items": [{"title": "Missing identity"}], "next_page": None}
    )
    with pytest.raises(ZooMCPException, match="invalid project ID"):
        await cloud_libraries.list_cloud_projects()


@pytest.mark.asyncio
async def test_mcp_tools_are_registered_as_read_only(httpx_mock):
    tools = {tool.name: tool for tool in await mcp.list_tools()}
    for name in [
        "list_cloud_libraries",
        "list_cloud_projects",
        "search_cloud_projects",
    ]:
        annotations = tools[name].annotations
        assert annotations is not None and annotations.read_only_hint is True
    httpx_mock.add_response(json={"items": [], "next_page": None})
    result = await mcp.call_tool("list_cloud_libraries", arguments={})
    assert isinstance(result, CallToolResult) and not result.is_error
    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text) == {"items": [], "next_page": None}
