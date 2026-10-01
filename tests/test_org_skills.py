import httpx
import pytest
from kittycad import AsyncKittyCAD
from mcp.types import CallToolResult
from pytest_httpx import HTTPXMock

from zoo_mcp import zoo_tools
from zoo_mcp.server import mcp


@pytest.fixture
def org_skills() -> list[dict[str, str]]:
    return [
        {
            "id": "00000000-0000-4000-8000-000000000001",
            "name": "alpha",
            "description": "first skill",
            "markdown": "# Alpha",
        },
        {
            "id": "00000000-0000-4000-8000-000000000002",
            "name": "beta",
            "description": "second skill",
            "markdown": "# Beta",
        },
    ]


@pytest.fixture
def skill_client(monkeypatch: pytest.MonkeyPatch) -> AsyncKittyCAD:
    client = AsyncKittyCAD(
        token="test-token",
        base_url="https://api.example.test/custom/",
        headers={"X-Client": "org-skills-test"},
        cookies={"session": "configured-session"},
        timeout=4.25,
    )
    monkeypatch.setattr(zoo_tools, "AsyncKittyCAD", lambda **kwargs: client)
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("paginated", [False, True], ids=["legacy", "paginated"])
async def test_list_org_skills_complete(
    skill_client: AsyncKittyCAD,
    org_skills: list[dict[str, str]],
    httpx_mock: HTTPXMock,
    paginated: bool,
) -> None:
    # Additive API fields must remain compatible, as they are in the SDK.
    wire_skills = [{**skill, "new_field": True} for skill in org_skills]
    if paginated:
        httpx_mock.add_response(
            json={"items": wire_skills[:1], "next_page": "cursor+/="}
        )
        httpx_mock.add_response(json={"items": [], "next_page": "last-page"})
        httpx_mock.add_response(
            json={"items": wire_skills[1:], "next_page": None, "new_field": True}
        )
    else:
        httpx_mock.add_response(json=wire_skills)

    transport = skill_client.get_http_client()
    response = await mcp.call_tool("list_org_skills", arguments={})
    assert isinstance(response, CallToolResult)
    assert response.structured_content == {"result": org_skills}
    assert transport.is_closed

    requests = httpx_mock.get_requests()
    assert [dict(request.url.params) for request in requests] == (
        [
            {"limit": "100"},
            {"limit": "100", "page_token": "cursor+/="},
            {"limit": "100", "page_token": "last-page"},
        ]
        if paginated
        else [{"limit": "100"}]
    )
    for request in requests:
        assert request.url.host == "api.example.test"
        assert request.url.path == "/custom/org/skills"
        assert request.headers["Authorization"] == "Bearer test-token"
        assert request.headers["X-Client"] == "org-skills-test"
        assert request.headers["Cookie"] == "session=configured-session"
        assert request.extensions["timeout"] == {
            "connect": 4.25,
            "read": 4.25,
            "write": 4.25,
            "pool": 4.25,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 204, 404])
async def test_list_org_skills_empty_response(
    skill_client: AsyncKittyCAD, httpx_mock: HTTPXMock, status: int
) -> None:
    httpx_mock.add_response(status_code=status)
    response = await mcp.call_tool("list_org_skills", arguments={})
    assert isinstance(response, CallToolResult)
    assert response.structured_content == {"result": []}
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [403, 404, 429, 500])
async def test_list_org_skills_later_http_error_never_returns_partial_skills(
    skill_client: AsyncKittyCAD,
    org_skills: list[dict[str, str]],
    httpx_mock: HTTPXMock,
    status: int,
) -> None:
    httpx_mock.add_response(json={"items": org_skills[:1], "next_page": "next"})
    httpx_mock.add_response(status_code=status, json={"message": "failed"})
    response = await mcp.call_tool("list_org_skills", arguments={})
    assert isinstance(response, CallToolResult)
    assert response.structured_content is not None
    result = response.structured_content["result"]
    assert isinstance(result, str)
    assert result.startswith("There was an error listing org skills")
    assert len(httpx_mock.get_requests()) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"items": []},
        {"items": "invalid", "next_page": None},
        {"items": [{"name": "missing-fields"}], "next_page": None},
        {"items": [], "next_page": 1},
        {"items": [], "next_page": " "},
        {"items": [], "next_page": "next"},
    ],
)
async def test_list_org_skills_invalid_continuation_is_an_error(
    skill_client: AsyncKittyCAD,
    org_skills: list[dict[str, str]],
    httpx_mock: HTTPXMock,
    payload: object,
) -> None:
    httpx_mock.add_response(json={"items": org_skills[:1], "next_page": "next"})
    httpx_mock.add_response(json=payload)
    response = await mcp.call_tool("list_org_skills", arguments={})
    assert isinstance(response, CallToolResult)
    assert response.structured_content is not None
    result = response.structured_content["result"]
    assert isinstance(result, str)
    assert result.startswith("There was an error listing org skills")
    assert len(httpx_mock.get_requests()) == 2


@pytest.mark.asyncio
async def test_list_org_skills_redirect_is_not_followed(
    skill_client: AsyncKittyCAD, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        status_code=302, headers={"Location": "https://another.example/org/skills"}
    )
    response = await mcp.call_tool("list_org_skills", arguments={})
    assert isinstance(response, CallToolResult)
    assert response.structured_content is not None
    assert isinstance(response.structured_content["result"], str)
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.asyncio
async def test_list_org_skills_timeout_closes_client(
    skill_client: AsyncKittyCAD,
    org_skills: list[dict[str, str]],
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(json={"items": org_skills[:1], "next_page": "next"})
    httpx_mock.add_exception(httpx.ReadTimeout("timed out"))
    transport = skill_client.get_http_client()
    response = await mcp.call_tool("list_org_skills", arguments={})
    assert isinstance(response, CallToolResult)
    assert response.structured_content is not None
    assert isinstance(response.structured_content["result"], str)
    assert len(httpx_mock.get_requests()) == 2
    assert transport.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 429, 500])
async def test_list_org_skills_initial_http_error_is_not_an_empty_catalog(
    skill_client: AsyncKittyCAD, httpx_mock: HTTPXMock, status: int
) -> None:
    httpx_mock.add_response(status_code=status)
    response = await mcp.call_tool("list_org_skills", arguments={})
    assert isinstance(response, CallToolResult)
    assert response.structured_content is not None
    result = response.structured_content["result"]
    assert isinstance(result, str)
    assert result.startswith("There was an error listing org skills")
    assert len(httpx_mock.get_requests()) == 1
