"""Runtime access to the immutable Zoo API documentation index."""

from __future__ import annotations

import copy
import json
import re
from importlib.resources import files
from typing import Any

from zoo_mcp.api_docs.builder import INDEX_FORMAT_VERSION
from zoo_mcp.api_docs.curl import generate_curl

JsonObject = dict[str, Any]

MAX_GUIDE_CHARACTERS = 30_000
MAX_QUERY_CHARACTERS = 500
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "can",
    "do",
    "does",
    "for",
    "get",
    "how",
    "i",
    "in",
    "is",
    "me",
    "my",
    "new",
    "of",
    "on",
    "the",
    "to",
    "what",
    "with",
    "zoo",
    "api",
    "create",
    "list",
}
SYNONYMS = {
    "agent": ("zookeeper", "ml"),
    "agents": ("zookeeper", "ml"),
    "authenticate": ("authentication", "auth", "bearer", "token"),
    "authenticated": ("authentication", "auth", "bearer", "token"),
    "authentication": ("auth", "token", "bearer"),
    "convert": ("conversion",),
    "conversion": ("convert",),
    "organization": ("org", "orgs"),
    "organizations": ("org", "orgs"),
    "usage": ("api-calls", "billing", "minutes"),
}


def _tokens(value: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall(value.lower())
    filtered = [token for token in tokens if token not in STOP_WORDS]
    return filtered or tokens


def _token_set(value: Any) -> set[str]:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    return set(_tokens(str(value)))


def _plain_text(value: str) -> str:
    text = re.sub(r"```.*?```", " ", value, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", text)
    text = re.sub(r"[#>*_|~-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _excerpt(value: str, query_tokens: list[str], limit: int = 280) -> str:
    text = _plain_text(value)
    if len(text) <= limit:
        return text
    lowered = text.lower()
    positions = [lowered.find(token) for token in query_tokens]
    positions = [position for position in positions if position >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - limit // 3)
    end = min(len(text), start + limit)
    prefix = "..." if start else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix


def _weighted_matches(
    query_tokens: list[str],
    fields: list[tuple[set[str], int]],
) -> tuple[int, int]:
    score = 0
    matched = 0
    for token in query_tokens:
        token_score = 0
        candidates = (token, *SYNONYMS.get(token, ()))
        for candidate_number, candidate in enumerate(candidates):
            discount = 1.0 if candidate_number == 0 else 0.55
            for field_tokens, weight in fields:
                if candidate in field_tokens:
                    token_score = max(token_score, int(weight * discount))
                elif any(item.startswith(candidate) for item in field_tokens):
                    token_score = max(token_score, int(weight * 0.45 * discount))
        if token_score:
            matched += 1
            score += token_score
    if query_tokens:
        score += int(200 * matched / len(query_tokens))
    return score, matched


class ZooApiDocsIndex:
    """Search and retrieve documentation from a validated immutable index."""

    def __init__(self, payload: JsonObject):
        if payload.get("format_version") != INDEX_FORMAT_VERSION:
            raise ValueError("unsupported Zoo API Docs index format")
        self.payload = payload
        self.operations: JsonObject = payload["operations"]
        self.schemas: JsonObject = payload["schemas"]
        self.guides: JsonObject = payload["guides"]

    @classmethod
    def load_bundled(cls) -> ZooApiDocsIndex:
        resource = files("zoo_mcp.api_docs").joinpath("data/index.json")
        return cls(json.loads(resource.read_text(encoding="utf-8")))

    @property
    def index_revision(self) -> str:
        return str(self.payload["index_revision"])

    @property
    def openapi_source_commit(self) -> str:
        return str(self.payload["openapi_source"]["commit"])

    @property
    def guide_source_revision(self) -> str:
        return str(self.payload["guide_source"]["revision"])

    def _metadata(self, documentation_url: str) -> JsonObject:
        return {
            "index_revision": self.index_revision,
            "openapi_source_commit": self.openapi_source_commit,
            "guide_source_revision": self.guide_source_revision,
            "documentation_url": documentation_url,
        }

    def _error(self, message: str) -> JsonObject:
        return {
            **self._metadata(str(self.payload["canonical_docs_base_url"])),
            "error": message,
            "deprecated": False,
            "beta": False,
        }

    def search(
        self,
        query: str,
        *,
        tag: str | None = None,
        limit: int = 5,
        include_deprecated: bool = False,
    ) -> JsonObject:
        if not 1 <= limit <= 10:
            raise ValueError("limit must be between 1 and 10")
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if len(normalized_query) > MAX_QUERY_CHARACTERS:
            raise ValueError(f"query must be at most {MAX_QUERY_CHARACTERS} characters")

        query_lower = normalized_query.lower()
        query_tokens = _tokens(normalized_query)
        raw_query_tokens = set(TOKEN_PATTERN.findall(query_lower))
        tag_lower = tag.strip().lower() if tag else None
        ranked: list[tuple[int, int, str, JsonObject]] = []

        for operation_id, operation in self.operations.items():
            if operation["deprecated"] and not include_deprecated:
                continue
            operation_tags = [str(value).lower() for value in operation["tags"]]
            if tag_lower and tag_lower not in operation_tags:
                continue

            exact_tier = 0
            if query_lower == operation_id.lower():
                exact_tier = 2
            elif query_lower == str(operation["path"]).lower():
                exact_tier = 1

            score, matched = _weighted_matches(
                query_tokens,
                [
                    (_token_set(operation_id), 420),
                    (_token_set(operation["path"]), 380),
                    (_token_set(operation["summary"]), 260),
                    (_token_set(operation["tags"]), 190),
                    (_token_set(operation["referenced_schema_names"]), 130),
                    (_token_set(operation["description"]), 70),
                ],
            )
            summary_lower = str(operation["summary"]).lower()
            if query_lower in summary_lower:
                score += 300
            operation_id_tokens = set(operation_id.lower().split("_"))
            for action in ("create", "delete", "get", "list", "update"):
                if action in raw_query_tokens and action in operation_id_tokens:
                    score += 500
            if exact_tier:
                score += 10_000 * exact_tier
            if score <= 0 or (matched == 0 and not exact_tier):
                continue

            result = {
                "type": "operation",
                "operation_id": operation_id,
                "method": operation["method"],
                "path": operation["path"],
                "title": operation["summary"] or operation_id,
                "excerpt": _excerpt(
                    f"{operation['summary']} {operation['description']}", query_tokens
                ),
                "tags": operation["tags"],
                "deprecated": operation["deprecated"],
                "beta": operation["beta"],
                "documentation_url": operation["documentation_url"],
                "source_revision": self.openapi_source_commit,
            }
            ranked.append((exact_tier, score, f"operation:{operation_id}", result))

        for guide_id, guide in self.guides.items():
            guide_tags = [str(value).lower() for value in guide["tags"]]
            if tag_lower and tag_lower not in guide_tags:
                continue
            exact_tier = 2 if query_lower == guide_id.lower() else 0
            score, matched = _weighted_matches(
                query_tokens,
                [
                    (_token_set(guide_id), 400),
                    (_token_set(guide["title"]), 280),
                    (_token_set(guide["tags"]), 190),
                    (_token_set(guide["content"]), 35),
                ],
            )
            if query_lower in str(guide["title"]).lower():
                score += 300
            if guide_id == "authentication" and raw_query_tokens.intersection(
                {"auth", "authenticate", "authenticated", "authentication", "bearer"}
            ):
                score += 1_200
            if exact_tier:
                score += 10_000 * exact_tier
            if score <= 0 or (matched == 0 and not exact_tier):
                continue

            result = {
                "type": "guide",
                "guide_id": guide_id,
                "title": guide["title"],
                "excerpt": _excerpt(guide["content"], query_tokens),
                "tags": guide["tags"],
                "deprecated": False,
                "beta": guide["beta"],
                "documentation_url": guide["documentation_url"],
                "source_revision": self.guide_source_revision,
            }
            ranked.append((exact_tier, score, f"guide:{guide_id}", result))

        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        results = [item[3] for item in ranked[:limit]]
        return {
            **self._metadata(str(self.payload["canonical_docs_base_url"])),
            "results_count": len(results),
            "results": results,
        }

    def get_operation(self, operation_id: str) -> JsonObject:
        operation = self.operations.get(operation_id)
        if operation is None:
            return self._error(
                "Operation not found or excluded by the public documentation policy."
            )
        curl = generate_curl(
            operation,
            self.schemas,
            base_url=str(self.payload["canonical_api_base_url"]),
            docs_base_url=str(self.payload["canonical_docs_base_url"]),
        )
        response = {
            **self._metadata(str(operation["documentation_url"])),
            "operation_id": operation_id,
            "method": operation["method"],
            "path": operation["path"],
            "summary": operation["summary"],
            "description": operation["description"],
            "tags": copy.deepcopy(operation["tags"]),
            "parameters": copy.deepcopy(operation["parameters"]),
            "request_content_types": copy.deepcopy(operation["request_content_types"]),
            "responses": copy.deepcopy(operation["responses"]),
            "referenced_schema_names": copy.deepcopy(
                operation["referenced_schema_names"]
            ),
            "missing_schema_references": copy.deepcopy(
                operation["missing_schema_references"]
            ),
            "authentication_required": operation["authentication_required"],
            "authentication": operation["authentication"],
            "deprecated": operation["deprecated"],
            "beta": operation["beta"],
            "warning": (
                "This operation is deprecated; prefer a non-deprecated replacement."
                if operation["deprecated"]
                else curl["warning"]
            ),
            "canonical_curl": curl["canonical_curl"],
            "connection_metadata": curl["connection_metadata"],
            "curl_guide_url": curl["guide_url"],
        }
        return response

    def get_schema(self, schema_name: str) -> JsonObject:
        record = self.schemas.get(schema_name)
        if record is None:
            return self._error("Schema not found.")
        return {
            **self._metadata(str(self.payload["canonical_docs_base_url"])),
            "schema_name": schema_name,
            "schema": copy.deepcopy(record["schema"]),
            "referenced_schema_names": copy.deepcopy(record["referenced_schema_names"]),
            "missing_schema_references": copy.deepcopy(
                record["missing_schema_references"]
            ),
            "deprecated": False,
            "beta": False,
        }

    def get_guide(self, guide_id: str) -> JsonObject:
        guide = self.guides.get(guide_id)
        if guide is None:
            return self._error("Guide not found.")
        content = str(guide["content"])
        truncated = len(content) > MAX_GUIDE_CHARACTERS
        return {
            **self._metadata(str(guide["documentation_url"])),
            "guide_id": guide_id,
            "title": guide["title"],
            "content": content[:MAX_GUIDE_CHARACTERS],
            "original_characters": len(content),
            "truncated": truncated,
            "deprecated": False,
            "beta": guide["beta"],
        }

    def health(self) -> JsonObject:
        unresolved_count = len(self.payload.get("unresolved_schema_references", []))
        return {
            "status": "ok" if unresolved_count == 0 else "not_ready",
            "ready": unresolved_count == 0,
            "index_revision": self.index_revision,
            "openapi_source_commit": self.openapi_source_commit,
            "guide_source_revision": self.guide_source_revision,
            "unresolved_schema_reference_count": unresolved_count,
        }
