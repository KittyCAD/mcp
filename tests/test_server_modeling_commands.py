from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from kittycad.models import (
    CurveGetEndPoints,
    CurveGetType,
    EdgeGetLength,
    EngineUtilEvaluatePath,
    EntityGetAllChildUuids,
    EntityGetDistance,
    EntityGetIndex,
    EntityGetParentId,
    EntityGetSketchPaths,
    EntityReference,
    HighlightSetEntities,
    ModelingCmd,
    Point3d,
    SelectEntity,
    SetSelectionFilter,
)
from kittycad.models.entity_reference import OptionFace
from kittycad.models.modeling_cmd import (
    OptionCurveGetEndPoints,
    OptionCurveGetType,
    OptionEdgeGetLength,
    OptionEngineUtilEvaluatePath,
    OptionEntityGetAllChildUuids,
    OptionEntityGetDistance,
    OptionEntityGetIndex,
    OptionEntityGetParentId,
    OptionEntityGetSketchPaths,
    OptionHighlightSetEntities,
    OptionSelectEntity,
    OptionSetSelectionFilter,
)
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ImageContent

from zoo_mcp import server
from zoo_mcp.server import mcp


def _result(response: Sequence[Any] | dict[str, Any]) -> Any:
    assert isinstance(response, Sequence)
    meta = response[1]
    assert isinstance(meta, dict)
    return cast(dict[str, Any], meta)["result"]


def _structured_result(response: Sequence[Any] | dict[str, Any]) -> dict[str, Any]:
    assert isinstance(response, Sequence)
    result = response[1]
    assert isinstance(result, dict)
    return cast(dict[str, Any], result)


@pytest.mark.asyncio
async def test_modeling_tools_are_registered():
    names = {tool.name for tool in await mcp.list_tools()}
    assert {
        "entity_distance",
        "set_selection_filter",
        "select_entity",
        "curve_get_end_points",
        "engine_util_evaluate_path",
        "curve_get_type",
        "edge_get_length",
        "entity_get_all_child_uuids",
        "entity_get_index",
        "entity_get_parent_id",
        "entity_get_sketch_paths",
        "highlight_set_entities",
        "snapshot",
        "start_modeling_session",
        "stop_modeling_session",
    } <= names


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments", "request_type", "request_fields", "response_data"),
    [
        (
            "entity_distance",
            {"entity_id1": "entity-1", "entity_id2": "entity-2", "on_axis": "x"},
            OptionEntityGetDistance,
            {
                "entity_id1": "entity-1",
                "entity_id2": "entity-2",
                "distance_type": {"axis": "x", "type": "on_axis"},
            },
            EntityGetDistance(min_distance=1, max_distance=2),
        ),
        (
            # The only command builder with a branch: omitting on_axis must
            # produce a Euclidean distance_type, not an on-axis one.
            "entity_distance",
            {"entity_id1": "entity-1", "entity_id2": "entity-2"},
            OptionEntityGetDistance,
            {
                "entity_id1": "entity-1",
                "entity_id2": "entity-2",
                "distance_type": {"type": "euclidean"},
            },
            EntityGetDistance(min_distance=0, max_distance=5),
        ),
        (
            "set_selection_filter",
            {"entity_types": ["face", "edge"], "session_id": "session-id"},
            OptionSetSelectionFilter,
            {"filter": ["face", "edge"]},
            SetSelectionFilter(),
        ),
        (
            "select_entity",
            {
                "entities": [{"type": "face", "face_id": "face-id"}],
                "session_id": "session-id",
            },
            OptionSelectEntity,
            {
                "entities": [
                    {"face_id": "face-id", "topology_fallback": None, "type": "face"}
                ]
            },
            SelectEntity(),
        ),
        (
            "highlight_set_entities",
            {"entity_ids": ["entity-1", "entity-2"], "session_id": "session-id"},
            OptionHighlightSetEntities,
            {"entities": ["entity-1", "entity-2"]},
            HighlightSetEntities(),
        ),
        (
            "curve_get_end_points",
            {"curve_id": "curve-id"},
            OptionCurveGetEndPoints,
            {"curve_id": "curve-id"},
            CurveGetEndPoints(
                start=Point3d(x=0, y=0, z=0),
                end=Point3d(x=1, y=2, z=3),
            ),
        ),
        (
            "engine_util_evaluate_path",
            {"path_json": '{"type":"line"}', "t": 0.25},
            OptionEngineUtilEvaluatePath,
            {"path_json": '{"type":"line"}', "t": 0.25},
            EngineUtilEvaluatePath(pos=Point3d(x=1, y=2, z=3)),
        ),
        (
            "curve_get_type",
            {"curve_id": "curve-id"},
            OptionCurveGetType,
            {"curve_id": "curve-id"},
            CurveGetType(curve_type="line"),
        ),
        (
            "edge_get_length",
            {"edge_id": "edge-id"},
            OptionEdgeGetLength,
            {"edge_id": "edge-id"},
            EdgeGetLength(length=12.5),
        ),
        (
            "entity_get_all_child_uuids",
            {"entity_id": "entity-id"},
            OptionEntityGetAllChildUuids,
            {"entity_id": "entity-id"},
            EntityGetAllChildUuids(entity_ids=["child-id"]),
        ),
        (
            "entity_get_index",
            {"entity_id": "entity-id"},
            OptionEntityGetIndex,
            {"entity_id": "entity-id"},
            EntityGetIndex(entity_index=3),
        ),
        (
            "entity_get_parent_id",
            {"entity_id": "entity-id"},
            OptionEntityGetParentId,
            {"entity_id": "entity-id"},
            EntityGetParentId(entity_id="parent-id"),
        ),
        (
            "entity_get_sketch_paths",
            {"entity_id": "entity-id"},
            OptionEntityGetSketchPaths,
            {"entity_id": "entity-id"},
            EntityGetSketchPaths(entity_ids=["path-id"]),
        ),
    ],
)
async def test_modeling_tool_builds_expected_command(
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    arguments: dict[str, object],
    request_type: type,
    request_fields: dict[str, object],
    response_data: object,
):
    mock = MagicMock(return_value=SimpleNamespace(data=response_data))
    monkeypatch.setattr(server, "zoo_execute_modeling_command", mock)

    response = await mcp.call_tool(tool_name, arguments=arguments)

    assert _structured_result(response) == cast(Any, response_data).model_dump()
    command = cast(ModelingCmd, mock.call_args.args[2])
    assert isinstance(command.root, request_type)
    assert command.root.model_dump(exclude={"type"}) == request_fields


@pytest.mark.asyncio
async def test_selection_tools_require_a_session():
    """Selection state is discarded without a session, so it must be required."""
    for tool_name, arguments in (
        ("set_selection_filter", {"entity_types": ["face"]}),
        ("select_entity", {"entities": [{"type": "face", "face_id": "face-id"}]}),
        ("highlight_set_entities", {"entity_ids": ["entity-1"]}),
    ):
        with pytest.raises(ToolError):
            await mcp.call_tool(tool_name, arguments=arguments)

    tools = {tool.name: tool for tool in await mcp.list_tools()}
    for tool_name in (
        "set_selection_filter",
        "select_entity",
        "highlight_set_entities",
    ):
        schema = tools[tool_name].inputSchema
        assert "session_id" in schema.get("required", []), tool_name
        assert "kcl_code" not in schema.get("properties", {}), tool_name


@pytest.mark.asyncio
async def test_query_tool_forwards_kcl_and_session(monkeypatch: pytest.MonkeyPatch):
    mock = MagicMock(
        return_value=SimpleNamespace(data=EntityGetIndex(entity_index=3)),
    )
    monkeypatch.setattr(server, "zoo_execute_modeling_command", mock)

    await mcp.call_tool(
        "entity_get_index",
        arguments={"entity_id": "entity-id", "kcl_code": "code"},
    )
    assert mock.call_args.args[0] == "code"
    assert mock.call_args.args[5] is None

    await mcp.call_tool(
        "entity_get_index",
        arguments={"entity_id": "entity-id", "session_id": "session-id"},
    )
    assert mock.call_args.args[0] is None
    assert mock.call_args.args[5] == "session-id"


@pytest.mark.asyncio
async def test_modeling_tool_returns_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        server,
        "zoo_execute_modeling_command",
        MagicMock(side_effect=RuntimeError("boom")),
    )

    with pytest.raises(ToolError, match="Error executing tool entity_get_index: boom"):
        await mcp.call_tool(
            "entity_get_index",
            arguments={"entity_id": "entity-id", "kcl_code": "code"},
        )


@pytest.mark.asyncio
async def test_start_and_stop_modeling_session_tools(
    monkeypatch: pytest.MonkeyPatch,
):
    start = MagicMock(return_value="session-id")
    stop = MagicMock()
    monkeypatch.setattr(server, "zoo_start_modeling_session", start)
    monkeypatch.setattr(server, "zoo_stop_modeling_session", stop)

    start_response = await mcp.call_tool("start_modeling_session", arguments={})
    stop_response = await mcp.call_tool(
        "stop_modeling_session",
        arguments={"session_id": "session-id"},
    )

    assert _result(start_response) == "session-id"
    assert _result(stop_response) is None
    start.assert_called_once_with()
    stop.assert_called_once_with("session-id")


@pytest.mark.asyncio
async def test_snapshot_tool_forwards_session_and_zoom(
    monkeypatch: pytest.MonkeyPatch,
):
    mock = MagicMock(return_value=b"jpeg")
    monkeypatch.setattr(server, "zoo_snapshot", mock)

    response = await mcp.call_tool(
        "snapshot",
        arguments={"session_id": "session-id", "max_image_dimension": 256},
    )

    assert isinstance(response, Sequence)
    content = response[0]
    assert isinstance(content, list)
    assert len(content) == 1
    assert isinstance(content[0], ImageContent)
    mock.assert_called_once_with(
        kcl_code=None,
        kcl_path=None,
        session_id="session-id",
        max_image_dimension=256,
        zoom=True,
        highlight_edges=False,
    )

    await mcp.call_tool(
        "snapshot",
        arguments={"session_id": "session-id", "zoom": False},
    )
    assert mock.call_args.kwargs["zoom"] is False

    await mcp.call_tool(
        "snapshot",
        arguments={"session_id": "session-id", "highlight_edges": True},
    )
    assert mock.call_args.kwargs["highlight_edges"] is True


@pytest.mark.asyncio
async def test_snapshot_tool_writes_to_output_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """output_path returns the saved path instead of an inline image."""
    monkeypatch.setattr(server, "zoo_snapshot", MagicMock(return_value=b"jpeg-bytes"))
    output_path = tmp_path / "snap.jpg"

    response = await mcp.call_tool(
        "snapshot",
        arguments={"session_id": "session-id", "output_path": str(output_path)},
    )

    result = _result(response)
    assert Path(result) == output_path.resolve()
    assert output_path.read_bytes() == b"jpeg-bytes"


@pytest.mark.asyncio
async def test_snapshot_tool_output_path_accepts_a_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    monkeypatch.setattr(server, "zoo_snapshot", MagicMock(return_value=b"jpeg-bytes"))

    response = await mcp.call_tool(
        "snapshot",
        arguments={"session_id": "session-id", "output_path": str(tmp_path)},
    )

    result = _result(response)
    assert Path(result) == (tmp_path / "image.jpg").resolve()
    assert (tmp_path / "image.jpg").read_bytes() == b"jpeg-bytes"


@pytest.mark.asyncio
async def test_snapshot_tool_returns_image_when_output_path_omitted(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(server, "zoo_snapshot", MagicMock(return_value=b"jpeg-bytes"))

    response = await mcp.call_tool("snapshot", arguments={"session_id": "session-id"})

    assert isinstance(response, Sequence)
    content = response[0]
    assert isinstance(content, list)
    assert isinstance(content[0], ImageContent)


@pytest.mark.asyncio
async def test_kcl_execution_tools_forward_session_id(
    monkeypatch: pytest.MonkeyPatch,
):
    execute = AsyncMock(return_value=(True, "KCL code executed successfully"))
    exec_project = MagicMock(return_value={"item_count": 1})
    monkeypatch.setattr(server, "zoo_execute_kcl", execute)
    monkeypatch.setattr(server, "zoo_exec_kcl_project", exec_project)

    execute_response = await mcp.call_tool(
        "execute_kcl",
        arguments={"kcl_code": "code", "session_id": "session-id"},
    )
    project_response = await mcp.call_tool(
        "exec_kcl_project",
        arguments={"kcl_code": "code", "session_id": "session-id"},
    )

    assert _result(execute_response) == [True, "KCL code executed successfully"]
    assert _structured_result(project_response) == {"item_count": 1}
    execute.assert_awaited_once_with(
        kcl_code="code",
        kcl_path=None,
        session_id="session-id",
    )
    exec_project.assert_called_once_with(
        kcl_code="code",
        kcl_path=None,
        session_id="session-id",
    )


@pytest.mark.asyncio
async def test_query_tools_document_their_arguments():
    """Docstrings are the LLM-facing contract; every parameter needs one."""
    tools = {tool.name: tool for tool in await mcp.list_tools()}
    for tool_name in (
        "curve_get_type",
        "edge_get_length",
        "entity_get_all_child_uuids",
        "entity_get_index",
        "entity_get_parent_id",
        "entity_get_sketch_paths",
        "curve_get_end_points",
        "entity_distance",
    ):
        description = tools[tool_name].description or ""
        assert "Args:" in description, tool_name
        assert "session_id:" in description, tool_name


@pytest.mark.asyncio
async def test_entity_reference_is_still_exported():
    """select_entity's schema depends on this alias."""
    assert EntityReference(OptionFace(face_id="face-id")) is not None
