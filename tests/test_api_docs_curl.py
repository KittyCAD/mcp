from typing import Any

from zoo_mcp.api_docs.curl import generate_curl


def _operation(**overrides: Any) -> dict[str, Any]:
    operation = {
        "operation_id": "example",
        "method": "POST",
        "path": "/objects/{id}",
        "parameters": [],
        "request_body": None,
        "authentication_required": True,
        "websocket": False,
        "documentation_url": "https://zoo.dev/docs/developer-tools/api/example",
    }
    operation.update(overrides)
    return operation


def _generate(operation: dict[str, Any], schemas: dict[str, Any] | None = None):
    return generate_curl(
        operation,
        schemas or {},
        base_url="https://api.zoo.dev",
        docs_base_url="https://zoo.dev/docs/developer-tools/api",
    )


def test_json_query_path_header_and_bearer_curl():
    operation = _operation(
        parameters=[
            {
                "name": "id",
                "in": "path",
                "schema": {"type": "string", "format": "uuid"},
            },
            {"name": "limit", "in": "query", "schema": {"type": "integer"}},
            {"name": "X-Trace", "in": "header", "schema": {"type": "string"}},
        ],
        request_body={
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    }
                }
            }
        },
    )

    curl = _generate(operation)["canonical_curl"]
    assert "https://api.zoo.dev/objects/<uuid>?limit=<limit>" in curl
    assert "Authorization: Bearer $ZOO_API_TOKEN" in curl
    assert "X-Trace: <X-Trace>" in curl
    assert "Content-Type: application/json" in curl
    assert '--data \'{"name":"<name>"}\'' in curl


def test_urlencoded_form_curl():
    operation = _operation(
        request_body={
            "content": {
                "application/x-www-form-urlencoded": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "client_id": {"type": "string"},
                            "active": {"type": "boolean"},
                        },
                    }
                }
            }
        }
    )

    curl = _generate(operation)["canonical_curl"]
    assert "application/x-www-form-urlencoded" in curl
    assert '--data-urlencode "client_id=<client_id>"' in curl
    assert '--data-urlencode "active=False"' in curl


def test_multipart_json_and_file_curl():
    operation = _operation(
        request_body={
            "content": {
                "multipart/form-data": {
                    "schema": {"$ref": "#/components/schemas/Upload"}
                }
            }
        }
    )
    schemas = {
        "Upload": {
            "schema": {
                "type": "object",
                "properties": {
                    "body": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                    "file": {"type": "string", "format": "binary"},
                },
            }
        }
    }

    curl = _generate(operation, schemas)["canonical_curl"]
    assert '--form \'body={"name":"<name>"};type=application/json\'' in curl
    assert '--form "file=@<path-to-file>"' in curl


def test_nested_refs_and_all_of_produce_a_bounded_valid_shape():
    operation = _operation(
        request_body={
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/Envelope"}
                }
            }
        }
    )
    schemas = {
        "Envelope": {
            "schema": {
                "type": "object",
                "properties": {
                    "mode": {"allOf": [{"$ref": "#/components/schemas/Mode"}]}
                },
                "required": ["mode"],
            }
        },
        "Mode": {
            "schema": {
                "oneOf": [
                    {"type": "string", "enum": ["safe"]},
                    {"$ref": "#/components/schemas/Mode"},
                ]
            }
        },
    }

    curl = _generate(operation, schemas)["canonical_curl"]
    assert '--data \'{"mode":"safe"}\'' in curl


def test_public_curl_has_no_authorization_header():
    curl = _generate(_operation(authentication_required=False))["canonical_curl"]
    assert "Authorization" not in curl


def test_websocket_returns_metadata_instead_of_invented_curl():
    result = _generate(_operation(websocket=True, path="/ws/modeling/commands"))

    assert result["canonical_curl"] is None
    assert result["connection_metadata"]["protocol"] == "websocket"
    assert result["connection_metadata"]["url"].startswith("wss://api.zoo.dev/")
    assert result["connection_metadata"]["authentication_message"] == {
        "type": "headers",
        "headers": {"Authorization": "Bearer $ZOO_API_TOKEN"},
    }


def test_unusual_body_returns_metadata_instead_of_invented_curl():
    result = _generate(
        _operation(
            request_body={
                "content": {
                    "application/vnd.zoo.unusual": {"schema": {"type": "string"}}
                }
            }
        )
    )

    assert result["canonical_curl"] is None
    assert result["connection_metadata"]["protocol"] == "http"
    assert "Unsupported request content types" in result["warning"]
