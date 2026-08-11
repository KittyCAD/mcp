from collections.abc import Sequence
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
    EntityType,
    GlobalAxis,
    HighlightSetEntities,
    Point3d,
    SelectEntity,
    SetSelectionFilter,
)
from kittycad.models.entity_reference import OptionFace
from kittycad.models.uuid import Uuid
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
    ("tool_name", "zoo_tool", "arguments", "expected_kwargs", "response_data"),
    [
        (
            "entity_distance",
            "zoo_entity_distance",
            {
                "entity_id1": "entity-1",
                "entity_id2": "entity-2",
                "on_axis": "x",
            },
            {
                "kcl_code": None,
                "kcl_path": None,
                "entity_id1": Uuid("entity-1"),
                "entity_id2": Uuid("entity-2"),
                "on_axis": GlobalAxis.X,
            },
            EntityGetDistance(min_distance=1, max_distance=2),
        ),
        (
            "set_selection_filter",
            "zoo_set_selection_filter",
            {"entity_types": ["face", "edge"]},
            {
                "kcl_code": None,
                "kcl_path": None,
                "entity_types": [EntityType.FACE, EntityType.EDGE],
            },
            SetSelectionFilter(),
        ),
        (
            "select_entity",
            "zoo_select_entity",
            {"entities": [{"type": "face", "face_id": "face-id"}]},
            {
                "kcl_code": None,
                "kcl_path": None,
                "entities": [EntityReference(OptionFace(face_id="face-id"))],
            },
            SelectEntity(),
        ),
        (
            "curve_get_end_points",
            "zoo_curve_get_end_points",
            {"curve_id": "curve-id"},
            {
                "kcl_code": None,
                "kcl_path": None,
                "curve_id": Uuid("curve-id"),
            },
            CurveGetEndPoints(
                start=Point3d(x=0, y=0, z=0),
                end=Point3d(x=1, y=2, z=3),
            ),
        ),
        (
            "engine_util_evaluate_path",
            "zoo_engine_util_evaluate_path",
            {"path_json": '{"type":"line"}', "t": 0.5},
            {
                "kcl_code": None,
                "kcl_path": None,
                "path_json": '{"type":"line"}',
                "t": 0.5,
            },
            EngineUtilEvaluatePath(pos=Point3d(x=1, y=2, z=3)),
        ),
        (
            "curve_get_type",
            "zoo_curve_get_type",
            {"curve_id": "curve-id"},
            {
                "kcl_code": None,
                "kcl_path": None,
                "curve_id": Uuid("curve-id"),
            },
            CurveGetType(curve_type="arc"),
        ),
        (
            "edge_get_length",
            "zoo_edge_get_length",
            {"edge_id": "edge-id"},
            {
                "kcl_code": None,
                "kcl_path": None,
                "edge_id": Uuid("edge-id"),
            },
            EdgeGetLength(length=12.5),
        ),
        (
            "entity_get_all_child_uuids",
            "zoo_entity_get_all_child_uuids",
            {"entity_id": "entity-id"},
            {
                "kcl_code": None,
                "kcl_path": None,
                "entity_id": Uuid("entity-id"),
            },
            EntityGetAllChildUuids(entity_ids=["child-id"]),
        ),
        (
            "entity_get_index",
            "zoo_entity_get_index",
            {"entity_id": "entity-id"},
            {
                "kcl_code": None,
                "kcl_path": None,
                "entity_id": Uuid("entity-id"),
            },
            EntityGetIndex(entity_index=3),
        ),
        (
            "entity_get_parent_id",
            "zoo_entity_get_parent_id",
            {"entity_id": "entity-id"},
            {
                "kcl_code": None,
                "kcl_path": None,
                "entity_id": Uuid("entity-id"),
            },
            EntityGetParentId(entity_id="parent-id"),
        ),
        (
            "entity_get_sketch_paths",
            "zoo_entity_get_sketch_paths",
            {"entity_id": "entity-id"},
            {
                "kcl_code": None,
                "kcl_path": None,
                "entity_id": Uuid("entity-id"),
            },
            EntityGetSketchPaths(entity_ids=["path-id"]),
        ),
        (
            "highlight_set_entities",
            "zoo_highlight_set_entities",
            {"entity_ids": ["entity-1", "entity-2"]},
            {
                "kcl_code": None,
                "kcl_path": None,
                "entity_ids": ["entity-1", "entity-2"],
            },
            HighlightSetEntities(),
        ),
    ],
)
async def test_modeling_tool_maps_arguments_and_returns_structured_response(
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    zoo_tool: str,
    arguments: dict[str, object],
    expected_kwargs: dict[str, object],
    response_data: object,
):
    mock = MagicMock(return_value=response_data)
    monkeypatch.setattr(server, zoo_tool, mock)

    response = await mcp.call_tool(tool_name, arguments=arguments)

    assert _structured_result(response) == cast(Any, response_data).model_dump()
    mock.assert_called_once_with(**expected_kwargs, session_id=None)


@pytest.mark.asyncio
async def test_modeling_tool_returns_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        server,
        "zoo_entity_get_index",
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

    start_response = await mcp.call_tool(
        "start_modeling_session",
        arguments={},
    )
    stop_response = await mcp.call_tool(
        "stop_modeling_session",
        arguments={"session_id": "session-id"},
    )

    assert _result(start_response) == "session-id"
    assert _result(stop_response) is None
    start.assert_called_once_with()
    stop.assert_called_once_with("session-id")


@pytest.mark.asyncio
async def test_modeling_tool_forwards_session_id(monkeypatch: pytest.MonkeyPatch):
    mock = MagicMock(return_value=EntityGetIndex(entity_index=3))
    monkeypatch.setattr(server, "zoo_entity_get_index", mock)

    response = await mcp.call_tool(
        "entity_get_index",
        arguments={"entity_id": "entity-id", "session_id": "session-id"},
    )

    assert _structured_result(response) == {"entity_index": 3}
    mock.assert_called_once_with(
        kcl_code=None,
        kcl_path=None,
        entity_id=Uuid("entity-id"),
        session_id="session-id",
    )


@pytest.mark.asyncio
async def test_snapshot_tool_forwards_session_id(monkeypatch: pytest.MonkeyPatch):
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
    )


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
