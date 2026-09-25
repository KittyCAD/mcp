"""Remote tool contracts. Local path-based tool schemas remain unchanged."""

from copy import deepcopy
from typing import Any

from mcp.types import Tool, ToolAnnotations

from .runtime import INPUT_PATHS, OUTPUT_PATHS

TEXT = {"type": "string"}
UUID = {"type": "string", "format": "uuid"}
OUTPUT = {"type": "object", "additionalProperties": True}
DOC_TOOLS = {
    "list_kcl_docs",
    "search_kcl_docs",
    "get_kcl_doc",
    "list_kcl_samples",
    "search_kcl_samples",
    "get_kcl_sample",
}
DATASET_TOOLS = {"list_org_datasets", "list_org_skills", "search_org_dataset_semantic"}
SCENE_READ_TOOLS = {
    "get_modeling_sessions",
    "get_face_info",
    "entity_distance",
    "curve_get_end_points",
    "engine_util_evaluate_path",
    "curve_get_type",
    "edge_get_length",
    "entity_get_all_child_uuids",
    "entity_get_index",
    "entity_get_parent_id",
    "entity_get_sketch_paths",
}

# name: (description, properties, required, scope, mutates)
CUSTOM: dict[str, tuple[str, dict[str, Any], list[str], str, bool]] = {
    "list_projects": (
        "List Zoo projects accessible to your connected account.",
        {},
        [],
        "projects:read",
        False,
    ),
    "get_project": (
        "Read current project metadata, permissions, file list, and revision.",
        {"project_id": UUID},
        ["project_id"],
        "projects:read",
        False,
    ),
    "open_project": (
        "Copy a Zoo project's current source into a temporary ZIP for modeling or editing.",
        {"project_id": UUID},
        ["project_id"],
        "projects:read",
        True,
    ),
    "create_project": (
        "Create a private Zoo project from a complete temporary project ZIP.",
        {
            "project_artifact_id": UUID,
            "title": TEXT,
            "description": TEXT,
            "entrypoint_path": TEXT,
        },
        ["project_artifact_id", "title"],
        "projects:write",
        True,
    ),
    "update_project": (
        "Replace a project's draft with a complete source ZIP; requires its acknowledged revision and intentionally removed file paths.",
        {
            "project_id": UUID,
            "project_artifact_id": UUID,
            "title": TEXT,
            "description": TEXT,
            "entrypoint_path": TEXT,
            "expected_revision": TEXT,
            "deleted_paths": {"type": "array", "items": TEXT},
        },
        [
            "project_id",
            "project_artifact_id",
            "title",
            "expected_revision",
            "deleted_paths",
        ],
        "projects:write",
        True,
    ),
    "publish_project": (
        "Submit a Zoo project for public review using the existing publication workflow.",
        {"project_id": UUID},
        ["project_id"],
        "projects:manage",
        True,
    ),
    "delete_project": (
        "Delete a Zoo project and its stored content according to Zoo's existing deletion policy.",
        {"project_id": UUID},
        ["project_id"],
        "projects:manage",
        True,
    ),
    "list_project_share_links": (
        "List a project's current share links.",
        {"project_id": UUID},
        ["project_id"],
        "projects:read",
        False,
    ),
    "create_project_share_link": (
        "Create a public download share link for a Zoo project.",
        {"project_id": UUID},
        ["project_id"],
        "projects:manage",
        True,
    ),
    "delete_project_share_link": (
        "Revoke a project's share link.",
        {"project_id": UUID, "key": TEXT},
        ["project_id", "key"],
        "projects:manage",
        True,
    ),
    "move_project_to_organization": (
        "Move a personal project into the connected account's active organization library.",
        {"project_id": UUID},
        ["project_id"],
        "projects:manage",
        True,
    ),
    "move_project_to_personal": (
        "Move an organization project back to personal ownership when permitted.",
        {"project_id": UUID},
        ["project_id"],
        "projects:manage",
        True,
    ),
    "get_job": (
        "Retrieve operation status and results. Interrupted work is never replayed automatically.",
        {"job_id": UUID},
        ["job_id"],
        "",
        False,
    ),
    "cancel_job": (
        "Cancel a running job. A completed upstream change cannot be undone by cancellation.",
        {"job_id": UUID},
        ["job_id"],
        "",
        True,
    ),
    "create_upload": (
        "Reserve a temporary file and obtain a short-lived upload URL. PUT exactly size_bytes bytes, then use the artifact ID.",
        {
            "name": TEXT,
            "size_bytes": {"type": "integer", "minimum": 0, "maximum": 268435456},
        },
        ["name", "size_bytes"],
        "files:write",
        True,
    ),
    "list_artifacts": (
        "List your temporary files; files expire seven days after creation.",
        {},
        [],
        "files:read",
        False,
    ),
    "get_artifact": (
        "Get file details and a fresh download URL.",
        {"artifact_id": UUID},
        ["artifact_id"],
        "files:read",
        False,
    ),
    "delete_artifact": (
        "Delete a temporary artifact. Saved Zoo projects are independent copies.",
        {"artifact_id": UUID},
        ["artifact_id"],
        "files:write",
        True,
    ),
    "write_kcl_project": (
        "Save editable KCL source and text dependencies as a temporary project ZIP before executing or exporting it.",
        {
            "files": {
                "type": "object",
                "additionalProperties": TEXT,
                "maxProperties": 1000,
            }
        },
        ["files"],
        "files:write",
        True,
    ),
}


def scope_for(name: str) -> str:
    if name in CUSTOM:
        return CUSTOM[name][3]
    if name in DOC_TOOLS:
        return ""
    if name in DATASET_TOOLS:
        return "datasets:read"
    return "files:write" if name == "save_image" else "modeling"


def scopes_for(name: str) -> list[str]:
    scopes = {scope_for(name)} - {""}
    if name == "open_project":
        scopes.add("files:write")
    if name in {"create_project", "update_project"}:
        scopes.add("files:read")
    if name not in CUSTOM and name not in DOC_TOOLS | DATASET_TOOLS | SCENE_READ_TOOLS:
        scopes.update({"files:read", "files:write"})
    return sorted(scopes)


def background(name: str) -> bool:
    if name in CUSTOM:
        return CUSTOM[name][4] and name not in {"create_upload", "cancel_job"}
    return name not in DOC_TOOLS | DATASET_TOOLS | {"get_modeling_sessions"}


def execution_options(name: str, schema: dict) -> None:
    if not background(name):
        return
    schema["properties"].update(
        {
            "execution_mode": {
                "type": "string",
                "enum": ["direct", "background"],
                "default": "direct",
                "description": "Direct results by default; background returns a durable job handle.",
            },
            "idempotency_key": {
                "type": "string",
                "minLength": 1,
                "maxLength": 128,
                "description": "Required only for background execution; reuse only for identical arguments.",
            },
        }
    )
    schema.setdefault("allOf", []).append(
        {
            "if": {
                "properties": {"execution_mode": {"const": "background"}},
                "required": ["execution_mode"],
            },
            "then": {"required": ["idempotency_key"]},
        }
    )


async def catalog() -> list[Tool]:
    from zoo_mcp.server import mcp

    tools = []
    for original in await mcp.list_tools():
        schema = deepcopy(original.input_schema)
        properties = schema.setdefault("properties", {})
        required = schema.get("required", [])
        for local, remote in INPUT_PATHS.items():
            if local in properties:
                properties.pop(local)
                properties[remote] = {
                    "anyOf": [UUID, {"type": "null"}],
                    "description": "Private Zoo artifact ID returned by upload, project, or execution tools.",
                }
                required = [remote if n == local else n for n in required]
        for output in OUTPUT_PATHS:
            properties.pop(output, None)
            required = [n for n in required if n != output]
        schema["required"] = required
        schema["additionalProperties"] = False
        description = (original.description or original.name).split("\n\n")[0]
        for local, remote in INPUT_PATHS.items():
            description = description.replace(local, remote)
        description += " Hosted file inputs use Zoo artifact IDs; outputs return downloadable artifacts."
        readonly = original.name in DOC_TOOLS | DATASET_TOOLS | SCENE_READ_TOOLS
        tools.append(
            Tool(
                name=original.name,
                title=original.name.replace("_", " ").title(),
                description=description,
                input_schema=schema,
                output_schema=OUTPUT,
                annotations=ToolAnnotations(
                    read_only_hint=readonly,
                    destructive_hint=False,
                    open_world_hint=original.name in DOC_TOOLS,
                ),
                _meta={
                    "securitySchemes": [
                        {
                            "type": "oauth2",
                            "scopes": scopes_for(original.name),
                        }
                    ]
                },
            )
        )
    for name, (description, properties, required, scope, mutates) in CUSTOM.items():
        properties = deepcopy(properties)
        required = list(required)
        meta: dict = {
            "securitySchemes": [{"type": "oauth2", "scopes": scopes_for(name)}]
        }
        tools.append(
            Tool(
                name=name,
                title=name.replace("_", " ").title(),
                description=description,
                input_schema={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
                output_schema=OUTPUT,
                annotations=ToolAnnotations(
                    read_only_hint=not mutates,
                    destructive_hint=name.startswith(("delete_", "update_", "move_")),
                    open_world_hint=name
                    in {
                        "publish_project",
                        "create_project_share_link",
                        "import_attachment",
                    },
                ),
                _meta=meta,
            )
        )
    for tool in tools:
        execution_options(tool.name, tool.input_schema)
    return tools
