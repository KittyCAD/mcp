"""Deterministic, placeholder-only curl generation for published operations."""

from __future__ import annotations

import json
import re
from typing import Any, cast

JsonObject = dict[str, Any]
MAX_SCHEMA_EXAMPLE_DEPTH = 10


def _schema_ref_name(schema: JsonObject) -> str | None:
    ref = schema.get("$ref")
    prefix = "#/components/schemas/"
    if not isinstance(ref, str) or not ref.startswith(prefix):
        return None
    return ref.removeprefix(prefix).replace("~1", "/").replace("~0", "~")


def _resolve_schema(
    schema: JsonObject, schemas: JsonObject
) -> tuple[JsonObject, str | None]:
    name = _schema_ref_name(schema)
    if name is None:
        return schema, None
    record = schemas.get(name, {})
    resolved = record.get("schema", {}) if isinstance(record, dict) else {}
    return (resolved if isinstance(resolved, dict) else {}), name


def _placeholder(name: str, schema: JsonObject) -> str:
    schema_format = schema.get("format")
    if schema_format == "uuid":
        return "<uuid>"
    if schema_format in {"date", "date-time", "time"}:
        return f"<{schema_format}>"
    if schema_format in {"email", "idn-email"}:
        return "<email>"
    if schema_format in {"uri", "url"}:
        return "https://example.com"
    if schema_format in {"binary", "byte"}:
        return "<path-to-file>"
    return f"<{name or 'value'}>"


def _example_value(
    schema: JsonObject,
    name: str,
    schemas: JsonObject,
    *,
    depth: int = 0,
    seen: frozenset[str] = frozenset(),
) -> Any:
    if depth >= MAX_SCHEMA_EXAMPLE_DEPTH:
        return _placeholder(name, schema)

    resolved, ref_name = _resolve_schema(schema, schemas)
    if ref_name is not None:
        if ref_name in seen:
            return f"<recursive-{ref_name}>"
        return _example_value(
            resolved,
            name,
            schemas,
            depth=depth + 1,
            seen=seen | {ref_name},
        )

    if "const" in resolved:
        return resolved["const"]
    enum = resolved.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]

    for composite in ("oneOf", "anyOf"):
        variants = resolved.get(composite)
        if isinstance(variants, list) and variants and isinstance(variants[0], dict):
            return _example_value(
                variants[0], name, schemas, depth=depth + 1, seen=seen
            )

    all_of = resolved.get("allOf")
    if isinstance(all_of, list) and all_of:
        values: list[Any] = []
        merged: JsonObject = {}
        for part in all_of:
            if not isinstance(part, dict):
                continue
            value = _example_value(part, name, schemas, depth=depth + 1, seen=seen)
            values.append(value)
            if isinstance(value, dict):
                merged.update(value)
        if merged:
            return merged
        if values:
            return values[0]

    schema_type = resolved.get("type")
    if schema_type is None and "properties" in resolved:
        schema_type = "object"
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), "string")

    if schema_type == "object":
        properties_value = resolved.get("properties", {})
        if not isinstance(properties_value, dict) or not properties_value:
            return {"key": "<value>"}
        properties = cast(JsonObject, properties_value)
        required = [item for item in resolved.get("required", []) if item in properties]
        selected = required or list(properties)[:3]
        selected = selected[:8]
        return {
            property_name: _example_value(
                property_schema if isinstance(property_schema, dict) else {},
                property_name,
                schemas,
                depth=depth + 1,
                seen=seen,
            )
            for property_name in selected
            if (property_schema := properties.get(property_name)) is not None
        }
    if schema_type == "array":
        items = resolved.get("items", {})
        return [
            _example_value(
                items if isinstance(items, dict) else {},
                name,
                schemas,
                depth=depth + 1,
                seen=seen,
            )
        ]
    if schema_type == "boolean":
        return False
    if schema_type == "integer":
        return 0
    if schema_type == "number":
        return 0.0
    return _placeholder(name, resolved)


def _parameter_value(parameter: JsonObject) -> str:
    schema = parameter.get("schema", {})
    if not isinstance(schema, dict):
        schema = {}
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return str(enum[0])
    return _placeholder(str(parameter.get("name", "value")), schema)


def _is_binary(schema: JsonObject, schemas: JsonObject) -> bool:
    resolved, _ = _resolve_schema(schema, schemas)
    if resolved.get("format") == "binary":
        return True
    items = resolved.get("items")
    return isinstance(items, dict) and items.get("format") == "binary"


def _request_url(operation: JsonObject, base_url: str) -> str:
    path = str(operation["path"])
    path_parameters = {
        str(parameter["name"]): _parameter_value(parameter)
        for parameter in operation.get("parameters", [])
        if parameter.get("in") == "path"
    }
    for placeholder in re.findall(r"\{([^}]+)\}", path):
        path = path.replace(
            "{" + placeholder + "}",
            path_parameters.get(placeholder, f"<{placeholder}>"),
        )

    query_parts = [
        f"{parameter['name']}={_parameter_value(parameter)}"
        for parameter in operation.get("parameters", [])
        if parameter.get("in") == "query" and not parameter.get("deprecated")
    ]
    query = f"?{'&'.join(query_parts)}" if query_parts else ""
    return f"{base_url.rstrip('/')}{path}{query}"


def _unsupported(
    operation: JsonObject,
    url: str,
    reason: str,
    guide_url: str,
) -> JsonObject:
    return {
        "canonical_curl": None,
        "connection_metadata": {
            "protocol": "http",
            "method": operation["method"],
            "url": url,
            "authentication_required": operation["authentication_required"],
        },
        "guide_url": guide_url,
        "warning": reason,
    }


def generate_curl(
    operation: JsonObject,
    schemas: JsonObject,
    *,
    base_url: str,
    docs_base_url: str,
) -> JsonObject:
    """Generate a canonical placeholder curl or explicit connection metadata."""
    url = _request_url(operation, base_url)
    if operation.get("websocket"):
        websocket_url = "wss://" + url.removeprefix("https://")
        return {
            "canonical_curl": None,
            "connection_metadata": {
                "protocol": "websocket",
                "url": websocket_url,
                "authentication_required": operation["authentication_required"],
                "authentication_message": (
                    {
                        "type": "headers",
                        "headers": {"Authorization": "Bearer $ZOO_API_TOKEN"},
                    }
                    if operation["authentication_required"]
                    else None
                ),
            },
            "guide_url": f"{docs_base_url}/authentication",
            "warning": (
                "This operation uses WebSocket protocol; curl would not describe "
                "the Zoo message exchange accurately."
            ),
        }

    unsupported_locations = sorted(
        {
            str(parameter.get("in"))
            for parameter in operation.get("parameters", [])
            if parameter.get("in") not in {"path", "query", "header"}
        }
    )
    if unsupported_locations:
        return _unsupported(
            operation,
            url,
            f"Unsupported parameter locations: {', '.join(unsupported_locations)}",
            str(operation["documentation_url"]),
        )

    parts = [f'curl -X {operation["method"]} "{url}"']
    if operation.get("authentication_required"):
        parts.append('--header "Authorization: Bearer $ZOO_API_TOKEN"')

    for parameter in operation.get("parameters", []):
        if parameter.get("in") != "header":
            continue
        name = str(parameter.get("name", ""))
        if name.lower() == "authorization":
            continue
        parts.append(f'--header "{name}: {_parameter_value(parameter)}"')

    request_body = operation.get("request_body")
    if request_body:
        content = request_body.get("content", {})
        if not isinstance(content, dict):
            content = {}
        json_types = sorted(
            content_type
            for content_type in content
            if content_type == "application/json" or content_type.endswith("+json")
        )
        if json_types:
            content_type = json_types[0]
            media = content[content_type]
            schema = media.get("schema", {}) if isinstance(media, dict) else {}
            example = _example_value(
                schema if isinstance(schema, dict) else {}, "body", schemas
            )
            parts.append(f'--header "Content-Type: {content_type}"')
            parts.append("--data '" + json.dumps(example, separators=(",", ":")) + "'")
        elif "multipart/form-data" in content:
            media = content["multipart/form-data"]
            schema = media.get("schema", {}) if isinstance(media, dict) else {}
            multipart_override = operation.get("curl_multipart_fields")
            if isinstance(multipart_override, list):
                for field in multipart_override:
                    if not isinstance(field, dict):
                        continue
                    field_name = str(field.get("name", "field"))
                    if field.get("kind") == "file":
                        parts.append(f'--form "{field_name}=@<path-to-file>"')
                    elif field.get("kind") == "json":
                        schema_name = str(field.get("schema", ""))
                        schema_record = schemas.get(schema_name, {})
                        field_schema = (
                            schema_record.get("schema", {})
                            if isinstance(schema_record, dict)
                            else {}
                        )
                        field_value = _example_value(
                            field_schema if isinstance(field_schema, dict) else {},
                            field_name,
                            schemas,
                        )
                        serialized = json.dumps(field_value, separators=(",", ":"))
                        parts.append(
                            f"--form '{field_name}={serialized};type=application/json'"
                        )
            else:
                resolved, _ = _resolve_schema(
                    schema if isinstance(schema, dict) else {}, schemas
                )
                properties = resolved.get("properties", {})
                if not isinstance(properties, dict) or not properties:
                    return _unsupported(
                        operation,
                        url,
                        "Multipart request schema has no named fields.",
                        str(operation["documentation_url"]),
                    )
                for field_name, field_schema_value in properties.items():
                    field_schema = (
                        field_schema_value
                        if isinstance(field_schema_value, dict)
                        else {}
                    )
                    if _is_binary(field_schema, schemas):
                        parts.append(f'--form "{field_name}=@<path-to-file>"')
                        continue
                    field_value = _example_value(field_schema, str(field_name), schemas)
                    if isinstance(field_value, (dict, list)):
                        serialized = json.dumps(field_value, separators=(",", ":"))
                        parts.append(
                            f"--form '{field_name}={serialized};type=application/json'"
                        )
                    else:
                        parts.append(f'--form "{field_name}={field_value}"')
        elif "application/x-www-form-urlencoded" in content:
            media = content["application/x-www-form-urlencoded"]
            schema = media.get("schema", {}) if isinstance(media, dict) else {}
            resolved, _ = _resolve_schema(
                schema if isinstance(schema, dict) else {}, schemas
            )
            properties = resolved.get("properties", {})
            if not isinstance(properties, dict) or not properties:
                return _unsupported(
                    operation,
                    url,
                    "Form request schema has no named fields.",
                    str(operation["documentation_url"]),
                )
            parts.append('--header "Content-Type: application/x-www-form-urlencoded"')
            for field_name, field_schema_value in properties.items():
                field_schema = (
                    field_schema_value if isinstance(field_schema_value, dict) else {}
                )
                value = _example_value(field_schema, str(field_name), schemas)
                parts.append(f'--data-urlencode "{field_name}={value}"')
        elif content:
            return _unsupported(
                operation,
                url,
                "Unsupported request content types: " + ", ".join(sorted(content)),
                str(operation["documentation_url"]),
            )

    return {
        "canonical_curl": " \\\n  ".join(parts),
        "connection_metadata": None,
        "guide_url": str(operation["documentation_url"]),
        "warning": None,
    }
