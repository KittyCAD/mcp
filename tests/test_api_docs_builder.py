import copy
from typing import Any

import httpx
import pytest

from zoo_mcp.api_docs.builder import (
    _documentation_slug,
    _fetch_without_redirects,
    build_index,
)
from zoo_mcp.api_docs.index import MAX_GUIDE_CHARACTERS, ZooApiDocsIndex


def _manifest() -> dict[str, Any]:
    return {
        "version": 1,
        "allowed_host": "zoo.dev",
        "guides": [
            {
                "id": "guide",
                "title": "Guide",
                "url": "https://zoo.dev/guide",
                "tags": ["guide"],
            }
        ],
    }


def _policy() -> dict[str, Any]:
    return {
        "canonical_api_base_url": "https://api.zoo.dev",
        "canonical_docs_base_url": "https://zoo.dev/docs/developer-tools/api",
        "default_authentication": "bearer",
        "exclude_tags": ["hidden"],
        "exclude_path_prefixes": ["/internal/"],
        "unauthenticated_operation_ids": ["get_public"],
    }


def _document() -> dict[str, Any]:
    response = {
        "200": {
            "description": "ok",
            "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/Payload"}}
            },
        }
    }
    return {
        "openapi": "3.0.3",
        "paths": {
            "/public/{id}": {
                "get": {
                    "operationId": "get_public",
                    "summary": "Get a public object.",
                    "tags": ["objects"],
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string", "format": "uuid"},
                        }
                    ],
                    "responses": response,
                }
            },
            "/deprecated": {
                "get": {
                    "operationId": "old_operation",
                    "summary": "Old operation.",
                    "tags": ["objects"],
                    "deprecated": True,
                    "responses": response,
                }
            },
            "/hidden": {
                "get": {
                    "operationId": "hidden_operation",
                    "summary": "Hidden operation.",
                    "tags": ["hidden"],
                    "responses": response,
                }
            },
            "/internal/secret": {
                "get": {
                    "operationId": "internal_operation",
                    "summary": "Internal operation.",
                    "tags": ["objects"],
                    "responses": response,
                }
            },
        },
        "components": {
            "schemas": {
                "Payload": {
                    "type": "object",
                    "properties": {"child": {"$ref": "#/components/schemas/Payload"}},
                }
            }
        },
    }


def _build(document: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_index(
        document or _document(),
        {"guide": "# Guide\n\nOfficial guide text.\n"},
        api_commit="a" * 40,
        policy=_policy(),
        guide_manifest=_manifest(),
    )


def test_build_index_applies_publication_policy_and_counts_every_operation():
    payload = _build()

    assert payload["source_operation_count"] == 4
    assert payload["published_operation_count"] == 2
    assert payload["excluded_operation_count"] == 2
    assert "hidden_operation" not in payload["operations"]
    assert "internal_operation" not in payload["operations"]
    assert payload["operations"]["get_public"]["authentication_required"] is False
    assert payload["operations"]["old_operation"]["deprecated"] is True


def test_exact_lookup_cannot_recover_excluded_operations():
    index = ZooApiDocsIndex(_build())

    assert "error" in index.get_operation("hidden_operation")
    assert "error" in index.get_operation("internal_operation")
    assert index.search("hidden_operation")["results"] == []
    assert index.search("internal_operation")["results"] == []


def test_exact_operation_id_and_path_rank_first():
    index = ZooApiDocsIndex(_build())

    assert index.search("get_public")["results"][0]["operation_id"] == "get_public"
    assert index.search("/public/{id}")["results"][0]["operation_id"] == "get_public"


def test_deprecated_operations_are_search_opt_in_but_exact_lookup_warns():
    index = ZooApiDocsIndex(_build())

    assert index.search("old_operation")["results"] == []
    results = index.search("old_operation", include_deprecated=True)["results"]
    assert results[0]["operation_id"] == "old_operation"
    assert index.get_operation("old_operation")["warning"].startswith(
        "This operation is deprecated"
    )


def test_recursive_and_missing_schema_references_are_not_expanded():
    document = _document()
    document["components"]["schemas"]["Broken"] = {
        "type": "object",
        "properties": {"missing": {"$ref": "#/components/schemas/Ghost"}},
    }
    index = ZooApiDocsIndex(_build(document))

    recursive = index.get_schema("Payload")
    assert recursive["schema"]["properties"]["child"] == {
        "$ref": "#/components/schemas/Payload"
    }
    assert recursive["referenced_schema_names"] == ["Payload"]

    broken = index.get_schema("Broken")
    assert broken["referenced_schema_names"] == ["Ghost"]
    assert broken["missing_schema_references"] == ["Ghost"]
    assert index.health()["ready"] is False


def test_guide_content_is_capped_and_marked_truncated():
    payload = _build()
    payload = copy.deepcopy(payload)
    payload["guides"]["guide"]["content"] = "x" * (MAX_GUIDE_CHARACTERS + 9)
    guide = ZooApiDocsIndex(payload).get_guide("guide")

    assert len(guide["content"]) == MAX_GUIDE_CHARACTERS
    assert guide["original_characters"] == MAX_GUIDE_CHARACTERS + 9
    assert guide["truncated"] is True


def test_source_fetch_rejects_unlisted_hosts_and_redirects():
    def redirect(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "https://example.com"})

    with httpx.Client(
        transport=httpx.MockTransport(redirect), follow_redirects=False
    ) as client:
        with pytest.raises(ValueError, match="redirect rejected"):
            _fetch_without_redirects(
                "https://zoo.dev/guide",
                allowed_host="zoo.dev",
                accept="text/markdown",
                client=client,
            )
        with pytest.raises(ValueError, match="not allowlisted"):
            _fetch_without_redirects(
                "https://example.com/guide",
                allowed_host="zoo.dev",
                accept="text/markdown",
                client=client,
            )


def test_documentation_slug_preserves_reference_title_punctuation():
    assert _documentation_slug("Get a user's org.") == "get-a-user%27s-org"
    assert (
        _documentation_slug("List projects for the website/gallery.")
        == "list-projects-for-the-website%2Fgallery"
    )
    assert (
        _documentation_slug("Owned by the caller’s organization.")
        == "owned-by-the-caller%E2%80%99s-organization"
    )


def test_bundled_index_matches_the_pinned_api_snapshot():
    index = ZooApiDocsIndex.load_bundled()

    assert index.payload["source_operation_count"] == 219
    assert index.payload["published_operation_count"] == 155
    assert index.payload["source_schema_count"] == 553
    assert index.payload["unresolved_schema_references"] == []
    assert sum(operation["deprecated"] for operation in index.operations.values()) == 3
    assert "get_ipinfo" not in index.operations
    assert "internal_get_api_token_for_discord_user" not in index.operations
    assert (
        index.operations["get_user_org"]["documentation_url"]
        == "https://zoo.dev/docs/developer-tools/api/orgs/get-a-user%27s-org"
    )
    assert index.health()["ready"] is True
