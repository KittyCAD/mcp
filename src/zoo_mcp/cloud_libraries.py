"""Discover the caller's cloud libraries and projects without modifying them."""

import asyncio
import json
from typing import Any, Literal
from uuid import UUID

import httpx

from zoo_mcp import ZooMCPException
from zoo_mcp.api_call_events import api_operation
from zoo_mcp.zoo_tools import _new_zoo_client

_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_PROJECT_FIELDS = (
    "id",
    "library_id",
    "title",
    "description",
    "revision",
    "entrypoint_path",
    "project_toml_path",
    "updated_at",
    "access",
)


def _library_id(value: str) -> str:
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise ZooMCPException(
            "library_id must be a library UUID returned by list_cloud_libraries"
        ) from exc


def _limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise ZooMCPException("limit must be between 1 and 100")
    return value


def _project_id(value: Any) -> str:
    try:
        return f"cloud:{UUID(str(value))}"
    except ValueError as exc:
        raise ZooMCPException("Cloud API returned an invalid project ID") from exc


async def _request(
    method: str,
    path: str,
    *,
    params: dict[str, str | int] | None = None,
    body: dict[str, Any] | None = None,
) -> Any:
    try:
        client = _new_zoo_client()
    except ValueError as exc:
        raise ZooMCPException(str(exc)) from exc
    try:
        async with (
            asyncio.timeout(45),
            client,
            client.get_http_client().stream(
                method,
                client.base_url + path,
                params=params,
                json=body,
                headers=client.get_headers(),
                follow_redirects=False,
            ) as response,
        ):
            response.raise_for_status()
            content = bytearray()
            async for chunk in response.aiter_bytes():
                if len(content) + len(chunk) > _MAX_RESPONSE_BYTES:
                    raise ZooMCPException(
                        "Cloud listing is too large; request a smaller page"
                    )
                content.extend(chunk)
            return json.loads(content)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        message = {
            401: "Cloud access requires a valid user session; reconnect and try again",
            403: "The caller cannot access this resource or has no active organization",
            404: "The cloud resource does not exist or is not accessible to the caller",
            429: "Cloud requests are rate limited; try again later",
        }.get(
            status,
            f"Cloud API returned HTTP {status}; refresh the listing before retrying",
        )
        raise ZooMCPException(message) from exc
    except (TimeoutError, httpx.TimeoutException) as exc:
        raise ZooMCPException("Cloud request timed out; try again later") from exc
    except httpx.RequestError as exc:
        raise ZooMCPException(
            "Cloud API could not be reached; try again later"
        ) from exc
    except ValueError as exc:
        raise ZooMCPException("Cloud response is invalid") from exc


def _fields(value: Any, names: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ZooMCPException("Cloud API returned an unexpected item")
    return {name: value[name] for name in names if name in value}


def _page(data: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ZooMCPException("Cloud API returned an unexpected listing")
    return {
        "items": [_fields(item, fields) for item in data["items"]],
        "next_page": data.get("next_page"),
    }


@api_operation
async def list_cloud_libraries(
    limit: int = 20, page_token: str | None = None
) -> dict[str, Any]:
    """List personal libraries and libraries in the caller's active organization.

    Returns library IDs, names, ownership, and creation sources such as zds,
    factory_portal, and org_dataset. Continue with next_page as page_token,
    even when a page has no items. Unassigned projects have no named library;
    find them with list_cloud_projects(unassigned=True).
    """
    params: dict[str, str | int] = {"limit": _limit(limit)}
    if page_token:
        params["page_token"] = page_token
    return _page(
        await _request("GET", "/user/project-libraries", params=params),
        ("id", "name", "source", "user_id", "org_id", "can_manage"),
    )


@api_operation
async def list_cloud_projects(
    library_id: str | None = None,
    scope: Literal["personal", "organization"] | None = None,
    unassigned: bool = False,
    limit: int = 20,
    page_token: str | None = None,
) -> dict[str, Any]:
    """Browse accessible cloud projects, optionally within one named library.

    Without library_id, scope selects personal or organization projects and
    unassigned=True selects projects with no named library. With library_id,
    omit scope and unassigned. Keep the same filters when continuing with
    next_page as page_token. Use project_id with the cloud project filesystem
    tools; projects can move between libraries while browsing.
    """
    params: dict[str, str | int] = {"limit": _limit(limit)}
    if library_id is not None:
        if scope is not None or unassigned:
            raise ZooMCPException(
                "Use library_id alone, or scope/unassigned without library_id"
            )
        path = f"/user/project-libraries/{_library_id(library_id)}/projects"
    else:
        path = "/user/projects"
        if scope is not None:
            if scope not in ("personal", "organization"):
                raise ZooMCPException("scope must be personal or organization")
            params["scope"] = scope
        if unassigned:
            params["unassigned"] = "true"
    if page_token:
        params["page_token"] = page_token
    page = _page(await _request("GET", path, params=params), _PROJECT_FIELDS)
    for item in page["items"]:
        item["project_id"] = _project_id(item.get("id"))
    return page


@api_operation
async def search_cloud_projects(
    query: str, library_id: str | None = None, limit: int = 10
) -> dict[str, Any]:
    """Search current projects shared with the caller's active organization by meaning.

    Searches titles, descriptions, and bounded source filenames, not file
    contents. Personal projects are excluded; browse them with list_cloud_projects.
    Optionally restrict to an organization library_id. Indexing is asynchronous,
    so an empty result does not prove a newly saved project is absent. A match
    describes version_id at search time; filesystem tools read current files.
    """
    query = query.strip()
    if not 1 <= len(query) <= 2048:
        raise ZooMCPException("query must contain between 1 and 2048 characters")
    body: dict[str, Any] = {"q": query, "limit": _limit(limit)}
    if library_id is not None:
        body["library_id"] = _library_id(library_id)
    data = await _request("POST", "/org/projects/search/semantic", body=body)
    if not isinstance(data, list):
        raise ZooMCPException("Cloud API returned an unexpected search response")
    matches = [
        _fields(
            item, ("project_id", "version_id", "title", "description", "similarity")
        )
        for item in data
    ]
    for match in matches:
        match["project_id"] = _project_id(match.get("project_id"))
    return {"matches": matches}
