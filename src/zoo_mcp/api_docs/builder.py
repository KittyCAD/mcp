"""Build the immutable Zoo API documentation index from allowlisted sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from importlib.resources import files
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote, urlparse

import httpx

JsonObject = dict[str, Any]

INDEX_FORMAT_VERSION = 1
OPENAPI_REPOSITORY = "KittyCAD/api"
OPENAPI_PATH = "openapi/api.json"
OPENAPI_RAW_URL = (
    "https://raw.githubusercontent.com/KittyCAD/api/{commit}/openapi/api.json"
)
HTTP_METHODS = ("delete", "get", "head", "patch", "post", "put", "trace")
SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


def load_publication_policy() -> JsonObject:
    """Load the checked-in operation publication policy."""
    resource = files("zoo_mcp.api_docs").joinpath("data/publication_policy.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def load_guide_manifest() -> JsonObject:
    """Load the checked-in guide allowlist."""
    resource = files("zoo_mcp.api_docs").joinpath("data/guide_sources.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def validate_commit(commit: str) -> str:
    """Require a full lowercase Git commit SHA for reproducible fetches."""
    if not SHA_PATTERN.fullmatch(commit):
        raise ValueError("source commit must be a full 40-character lowercase SHA")
    return commit


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")


def _documentation_slug(value: str) -> str:
    """Match the API reference's title-based, percent-encoded route slugs."""
    normalized = re.sub(r"\s+", "-", value.strip().rstrip(".").lower())
    return quote(normalized, safe="-")


def _schema_name_from_ref(ref: str) -> str | None:
    prefix = "#/components/schemas/"
    if not ref.startswith(prefix):
        return None
    return ref.removeprefix(prefix).replace("~1", "/").replace("~0", "~")


def referenced_schema_names(value: Any) -> list[str]:
    """Return direct schema references without expanding recursive schemas."""
    names: set[str] = set()

    def visit(current: Any) -> None:
        if isinstance(current, dict):
            ref = current.get("$ref")
            if isinstance(ref, str):
                name = _schema_name_from_ref(ref)
                if name is not None:
                    names.add(name)
            for child in current.values():
                visit(child)
        elif isinstance(current, list):
            for child in current:
                visit(child)

    visit(value)
    return sorted(names)


def _resolve_component(
    document: JsonObject,
    value: Any,
    section: str,
) -> JsonObject:
    if not isinstance(value, dict):
        return {}
    ref = value.get("$ref")
    if not isinstance(ref, str):
        return value
    prefix = f"#/components/{section}/"
    if not ref.startswith(prefix):
        return value
    name = ref.removeprefix(prefix).replace("~1", "/").replace("~0", "~")
    resolved = document.get("components", {}).get(section, {}).get(name)
    return resolved if isinstance(resolved, dict) else value


def _normalize_parameter(document: JsonObject, value: Any) -> JsonObject:
    parameter = _resolve_component(document, value, "parameters")
    return {
        "name": str(parameter.get("name", "")),
        "in": str(parameter.get("in", "")),
        "required": bool(parameter.get("required", False)),
        "description": str(parameter.get("description", "")),
        "deprecated": bool(parameter.get("deprecated", False)),
        "schema": parameter.get("schema", {}),
        "content": parameter.get("content", {}),
    }


def _normalize_request_body(document: JsonObject, value: Any) -> JsonObject | None:
    if not value:
        return None
    request_body = _resolve_component(document, value, "requestBodies")
    content: JsonObject = {}
    for content_type, media in sorted(request_body.get("content", {}).items()):
        if not isinstance(media, dict):
            continue
        content[content_type] = {
            "schema": media.get("schema", {}),
            "encoding": media.get("encoding", {}),
        }
    return {
        "required": bool(request_body.get("required", False)),
        "description": str(request_body.get("description", "")),
        "content": content,
    }


def _normalize_responses(document: JsonObject, value: Any) -> JsonObject:
    responses: JsonObject = {}
    if not isinstance(value, dict):
        return responses
    for status, response_or_ref in sorted(value.items()):
        response = _resolve_component(document, response_or_ref, "responses")
        content = response.get("content", {})
        responses[str(status)] = {
            "description": str(response.get("description", "")),
            "content_types": sorted(content) if isinstance(content, dict) else [],
            "referenced_schema_names": referenced_schema_names(response),
        }
    return responses


def _documentation_url(
    docs_base_url: str,
    tags: list[str],
    summary: str,
    operation_id: str,
) -> str:
    visible_tags = [tag for tag in tags if tag != "hidden"]
    tag = _slugify(visible_tags[0]) if visible_tags else "meta"
    slug = _documentation_slug(summary) or _slugify(operation_id)
    return f"{docs_base_url}/{tag}/{slug}"


def _normalize_operation(
    document: JsonObject,
    path: str,
    method: str,
    path_item: JsonObject,
    operation: JsonObject,
    policy: JsonObject,
    schema_names: set[str],
) -> JsonObject:
    operation_id = operation.get("operationId")
    if not isinstance(operation_id, str) or not operation_id:
        raise ValueError(f"{method.upper()} {path} has no operationId")

    tags = [str(tag) for tag in operation.get("tags", [])]
    summary = str(operation.get("summary", ""))
    parameters = [
        _normalize_parameter(document, parameter)
        for parameter in [
            *path_item.get("parameters", []),
            *operation.get("parameters", []),
        ]
    ]
    request_body = _normalize_request_body(document, operation.get("requestBody"))
    responses = _normalize_responses(document, operation.get("responses", {}))
    refs = set(
        referenced_schema_names(
            {
                "parameters": parameters,
                "request_body": request_body,
            }
        )
    )
    for response in responses.values():
        refs.update(response.get("referenced_schema_names", []))
    sorted_refs = sorted(refs)
    missing_refs = sorted(refs - schema_names)
    unauthenticated = set(policy.get("unauthenticated_operation_ids", []))
    multipart_fields = policy.get("curl_multipart_fields", {}).get(operation_id)

    return {
        "operation_id": operation_id,
        "method": method.upper(),
        "path": path,
        "summary": summary,
        "description": str(operation.get("description", "")),
        "tags": tags,
        "parameters": parameters,
        "request_body": request_body,
        "request_content_types": (
            sorted(request_body["content"]) if request_body is not None else []
        ),
        "responses": responses,
        "referenced_schema_names": sorted_refs,
        "missing_schema_references": missing_refs,
        "authentication_required": operation_id not in unauthenticated,
        "authentication": (
            None
            if operation_id in unauthenticated
            else str(policy.get("default_authentication", "bearer"))
        ),
        "deprecated": bool(operation.get("deprecated", False)),
        "beta": "beta" in tags,
        "websocket": bool(operation.get("x-dropshot-websocket") is not None)
        or path.startswith("/ws/")
        or "101" in operation.get("responses", {}),
        "documentation_url": _documentation_url(
            str(policy["canonical_docs_base_url"]), tags, summary, operation_id
        ),
        "curl_multipart_fields": multipart_fields,
    }


def _is_excluded(path: str, operation: JsonObject, policy: JsonObject) -> bool:
    excluded_tags = set(policy.get("exclude_tags", []))
    if excluded_tags.intersection(operation.get("tags", [])):
        return True
    return any(
        path.startswith(prefix) for prefix in policy.get("exclude_path_prefixes", [])
    )


def _strip_frontmatter(content: str) -> str:
    normalized = content.replace("\r\n", "\n").strip()
    if normalized.startswith("---\n"):
        closing = normalized.find("\n---\n", 4)
        if closing != -1:
            normalized = normalized[closing + 5 :].lstrip()
    return normalized


def normalize_guide_content(content: str) -> str:
    """Normalize line endings and token placeholders without rewriting the guide."""
    normalized = _strip_frontmatter(content)
    normalized = normalized.replace("Bearer $TOKEN", "Bearer $ZOO_API_TOKEN")
    if not normalized:
        raise ValueError("guide content is empty")
    return normalized.rstrip() + "\n"


def _fetch_without_redirects(
    url: str,
    *,
    allowed_host: str,
    accept: str,
    client: httpx.Client | None = None,
) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != allowed_host:
        raise ValueError(f"source URL is not allowlisted: {url}")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError(f"source URL contains unsupported components: {url}")

    owns_client = client is None
    active_client = client or httpx.Client(follow_redirects=False, timeout=30.0)
    try:
        response = active_client.get(url, headers={"Accept": accept})
        if 300 <= response.status_code < 400:
            raise ValueError(f"redirect rejected for allowlisted source: {url}")
        response.raise_for_status()
        if response.url != httpx.URL(url):
            raise ValueError(f"source URL changed unexpectedly: {url}")
        return response.content
    finally:
        if owns_client:
            active_client.close()


def fetch_openapi(commit: str, client: httpx.Client | None = None) -> JsonObject:
    """Fetch OpenAPI only from the pinned KittyCAD/api raw path."""
    validate_commit(commit)
    url = OPENAPI_RAW_URL.format(commit=commit)
    content = _fetch_without_redirects(
        url,
        allowed_host="raw.githubusercontent.com",
        accept="application/json",
        client=client,
    )
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("OpenAPI source must be a JSON object")
    return parsed


def fetch_guides(client: httpx.Client | None = None) -> dict[str, str]:
    """Fetch only the URLs in the checked-in guide manifest."""
    manifest = load_guide_manifest()
    allowed_host = str(manifest["allowed_host"])
    guides: dict[str, str] = {}
    for guide in manifest["guides"]:
        content = _fetch_without_redirects(
            str(guide["url"]),
            allowed_host=allowed_host,
            accept="text/markdown",
            client=client,
        ).decode("utf-8")
        guides[str(guide["id"])] = normalize_guide_content(content)
    return guides


def load_guides_from_directory(directory: Path) -> dict[str, str]:
    """Load pre-fetched guide bodies by allowlisted guide ID."""
    manifest = load_guide_manifest()
    guides: dict[str, str] = {}
    for guide in manifest["guides"]:
        guide_id = str(guide["id"])
        source = directory / f"{guide_id}.md"
        guides[guide_id] = normalize_guide_content(source.read_text(encoding="utf-8"))
    return guides


def _guide_revision(guides: dict[str, str]) -> str:
    content = b"".join(
        guide_id.encode() + b"\0" + guides[guide_id].encode()
        for guide_id in sorted(guides)
    )
    return f"sha256:{_sha256(content)}"


def build_index(
    document: JsonObject,
    guides: dict[str, str],
    *,
    api_commit: str,
    policy: JsonObject | None = None,
    guide_manifest: JsonObject | None = None,
) -> JsonObject:
    """Build a deterministic runtime index from OpenAPI and guide content."""
    validate_commit(api_commit)
    active_policy = policy or load_publication_policy()
    active_manifest = guide_manifest or load_guide_manifest()
    schemas_value = document.get("components", {}).get("schemas", {})
    if not isinstance(schemas_value, dict):
        raise ValueError("OpenAPI components.schemas must be an object")
    schemas = cast(JsonObject, schemas_value)
    schema_name_set = set(schemas)

    manifest_ids = {str(guide["id"]) for guide in active_manifest["guides"]}
    if set(guides) != manifest_ids:
        missing = sorted(manifest_ids - set(guides))
        unexpected = sorted(set(guides) - manifest_ids)
        raise ValueError(
            f"guide set does not match allowlist; missing={missing}, unexpected={unexpected}"
        )

    operations: JsonObject = {}
    source_operation_count = 0
    excluded_operation_count = 0
    seen_operation_ids: set[str] = set()
    paths_value = document.get("paths", {})
    if not isinstance(paths_value, dict):
        raise ValueError("OpenAPI paths must be an object")
    paths = cast(JsonObject, paths_value)

    for path, path_item_value in sorted(paths.items()):
        if not isinstance(path_item_value, dict):
            continue
        path_item = cast(JsonObject, path_item_value)
        for method in HTTP_METHODS:
            operation_value = path_item.get(method)
            if not isinstance(operation_value, dict):
                continue
            operation = cast(JsonObject, operation_value)
            source_operation_count += 1
            operation_id = operation.get("operationId")
            if not isinstance(operation_id, str) or not operation_id:
                raise ValueError(f"{method.upper()} {path} has no operationId")
            if operation_id in seen_operation_ids:
                raise ValueError(f"duplicate operationId: {operation_id}")
            seen_operation_ids.add(operation_id)
            if _is_excluded(path, operation, active_policy):
                excluded_operation_count += 1
                continue
            operations[operation_id] = _normalize_operation(
                document,
                path,
                method,
                path_item,
                operation,
                active_policy,
                schema_name_set,
            )

    normalized_schemas: JsonObject = {}
    for name, schema in sorted(schemas.items()):
        refs = referenced_schema_names(schema)
        normalized_schemas[name] = {
            "schema": schema,
            "referenced_schema_names": refs,
            "missing_schema_references": sorted(set(refs) - schema_name_set),
        }

    all_schema_refs = referenced_schema_names(document)
    unresolved_refs = sorted(set(all_schema_refs) - schema_name_set)

    revision = _guide_revision(guides)
    guide_records: JsonObject = {}
    for guide in active_manifest["guides"]:
        guide_id = str(guide["id"])
        content = guides[guide_id]
        guide_records[guide_id] = {
            "guide_id": guide_id,
            "title": str(guide["title"]),
            "tags": [str(tag) for tag in guide.get("tags", [])],
            "beta": bool(guide.get("beta", False)),
            "documentation_url": str(guide["url"]),
            "content": content,
            "content_sha256": _sha256(content.encode()),
        }

    payload: JsonObject = {
        "format_version": INDEX_FORMAT_VERSION,
        "openapi_source": {
            "repository": OPENAPI_REPOSITORY,
            "path": OPENAPI_PATH,
            "commit": api_commit,
        },
        "guide_source": {
            "manifest": "zoo_mcp/api_docs/data/guide_sources.json",
            "revision": revision,
        },
        "canonical_api_base_url": str(active_policy["canonical_api_base_url"]),
        "canonical_docs_base_url": str(active_policy["canonical_docs_base_url"]),
        "source_operation_count": source_operation_count,
        "published_operation_count": len(operations),
        "excluded_operation_count": excluded_operation_count,
        "source_schema_count": len(schemas),
        "unresolved_schema_references": unresolved_refs,
        "operations": dict(sorted(operations.items())),
        "schemas": normalized_schemas,
        "guides": dict(sorted(guide_records.items())),
    }
    payload["index_revision"] = f"sha256:{_sha256(_canonical_json(payload))}"
    return payload


def write_index(index: JsonObject, output: Path) -> None:
    """Write the deterministic index atomically."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-commit", required=True, type=validate_commit)
    parser.add_argument(
        "--openapi-file",
        type=Path,
        help="Use a local OpenAPI file while retaining the required source commit.",
    )
    parser.add_argument(
        "--guides-dir",
        type=Path,
        help="Use pre-fetched <guide-id>.md files instead of the allowlisted URLs.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.openapi_file:
        document = json.loads(args.openapi_file.read_text(encoding="utf-8"))
    else:
        document = fetch_openapi(args.api_commit)
    guides = (
        load_guides_from_directory(args.guides_dir)
        if args.guides_dir
        else fetch_guides()
    )
    index = build_index(
        document,
        guides,
        api_commit=args.api_commit,
    )
    write_index(index, args.output)
    print(
        json.dumps(
            {
                "index_revision": index["index_revision"],
                "openapi_source_commit": args.api_commit,
                "guide_source_revision": index["guide_source"]["revision"],
                "source_operation_count": index["source_operation_count"],
                "published_operation_count": index["published_operation_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
