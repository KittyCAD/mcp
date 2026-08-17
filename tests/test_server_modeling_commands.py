from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from kittycad.models import (
    CurveGetEndPoints,
    CurveGetType,
    DefaultCameraCenterToSelection,
    EdgeGetLength,
    EngineUtilEvaluatePath,
    EntityGetAllChildUuids,
    EntityGetDistance,
    EntityGetIndex,
    EntityGetParentId,
    EntityGetSketchPaths,
    HighlightSetEntities,
    ModelingCmd,
    Point3d,
    SelectReplace,
    SetSelectionFilter,
)
from kittycad.models.modeling_cmd import (
    OptionCurveGetEndPoints,
    OptionCurveGetType,
    OptionDefaultCameraCenterToSelection,
    OptionEdgeGetLength,
    OptionEngineUtilEvaluatePath,
    OptionEntityGetAllChildUuids,
    OptionEntityGetDistance,
    OptionEntityGetIndex,
    OptionEntityGetParentId,
    OptionEntityGetSketchPaths,
    OptionHighlightSetEntities,
    OptionSelectReplace,
    OptionSetSelectionFilter,
)
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ImageContent, TextContent

from zoo_mcp import server
from zoo_mcp.server import mcp
from zoo_mcp.zoo_tools import CameraView, ResultZooExecuteKclRemote


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
        "select_entities",
        "center_camera_on_selection",
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
        "get_sessions",
        "stop_modeling_session",
    } <= names


@pytest.mark.asyncio
async def test_get_sessions_tool(monkeypatch: pytest.MonkeyPatch):
    get_sessions = MagicMock(return_value=["session-id"])
    monkeypatch.setattr(server, "zoo_get_modeling_sessions", get_sessions)

    response = await mcp.call_tool("get_sessions", arguments={})

    assert _result(response) == ["session-id"]
    get_sessions.assert_called_once_with()


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
            "highlight_set_entities",
            {"entity_ids": ["entity-1", "entity-2"], "session_id": "session-id"},
            OptionHighlightSetEntities,
            {"entities": ["entity-1", "entity-2"]},
            HighlightSetEntities(),
        ),
        (
            "select_entities",
            {"entity_ids": ["entity-1", "entity-2"], "session_id": "session-id"},
            OptionSelectReplace,
            {"entities": ["entity-1", "entity-2"]},
            SelectReplace(),
        ),
        (
            "center_camera_on_selection",
            {"session_id": "session-id"},
            OptionDefaultCameraCenterToSelection,
            {"camera_movement": "vantage"},
            DefaultCameraCenterToSelection(),
        ),
        (
            "center_camera_on_selection",
            {"session_id": "session-id", "move_vantage": False},
            OptionDefaultCameraCenterToSelection,
            {"camera_movement": "none"},
            DefaultCameraCenterToSelection(),
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
    arguments["session_id"] = "session-id"

    response = await mcp.call_tool(tool_name, arguments=arguments)

    assert _structured_result(response) == cast(Any, response_data).model_dump()
    command = cast(ModelingCmd, mock.call_args.args[0])
    assert isinstance(command.root, request_type)
    assert command.root.model_dump(exclude={"type"}) == request_fields
    assert mock.call_args.args[3] == "session-id"


@pytest.mark.asyncio
async def test_scene_tools_require_a_session():
    for tool_name, arguments in (
        ("get_face_info", {"face_id": "face-id"}),
        ("entity_distance", {"entity_id1": "a", "entity_id2": "b"}),
        ("curve_get_end_points", {"curve_id": "curve-id"}),
        ("engine_util_evaluate_path", {"path_json": "{}", "t": 0.5}),
        ("curve_get_type", {"curve_id": "curve-id"}),
        ("edge_get_length", {"edge_id": "edge-id"}),
        ("entity_get_all_child_uuids", {"entity_id": "entity-id"}),
        ("entity_get_index", {"entity_id": "entity-id"}),
        ("entity_get_parent_id", {"entity_id": "entity-id"}),
        ("entity_get_sketch_paths", {"entity_id": "entity-id"}),
        ("set_selection_filter", {"entity_types": ["face"]}),
        ("select_entities", {"entity_ids": ["entity-1"]}),
        ("highlight_set_entities", {"entity_ids": ["entity-1"]}),
        ("center_camera_on_selection", {}),
        ("snapshot", {}),
        ("exec_kcl_project", {"kcl_code": "code"}),
    ):
        with pytest.raises(ToolError):
            await mcp.call_tool(tool_name, arguments=arguments)

    tools = {tool.name: tool for tool in await mcp.list_tools()}
    for tool_name in (
        "set_selection_filter",
        "select_entities",
        "highlight_set_entities",
        "center_camera_on_selection",
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
        "snapshot",
        "exec_kcl_project",
    ):
        schema = tools[tool_name].inputSchema
        assert "session_id" in schema.get("required", []), tool_name
        if tool_name != "exec_kcl_project":
            properties = schema.get("properties", {})
            assert "kcl_code" not in properties, tool_name
            assert "kcl_path" not in properties, tool_name
            assert "input_file" not in properties, tool_name


@pytest.mark.asyncio
async def test_emphasis_guidance_is_documented():
    """The relative strength of highlight vs selection is easy to get wrong."""
    tools = {tool.name: tool for tool in await mcp.list_tools()}
    highlight = tools["highlight_set_entities"].description or ""
    assert "select_entities" in highlight
    assert "center_camera_on_selection" in highlight

    select = tools["select_entities"].description or ""
    assert "highlight_set_entities" in select


@pytest.mark.asyncio
async def test_only_one_selection_tool_is_exposed():
    """Two selection tools with different argument shapes invite the wrong pick."""
    names = {tool.name for tool in await mcp.list_tools()}
    assert "select_entities" in names
    assert "select_entity" not in names


@pytest.mark.asyncio
async def test_query_tool_forwards_session(monkeypatch: pytest.MonkeyPatch):
    mock = MagicMock(
        return_value=SimpleNamespace(data=EntityGetIndex(entity_index=3)),
    )
    monkeypatch.setattr(server, "zoo_execute_modeling_command", mock)

    await mcp.call_tool(
        "entity_get_index",
        arguments={"entity_id": "entity-id", "session_id": "session-id"},
    )
    assert mock.call_args.args[3] == "session-id"


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
            arguments={"entity_id": "entity-id", "session_id": "session-id"},
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
        session_id="session-id",
        views=None,
        max_image_dimension=256,
        padding=0.1,
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
@pytest.mark.parametrize(
    ("camera_view", "expected_views"),
    [
        (None, None),
        ("front", ["front"]),
        (["front", "top"], ["front", "top"]),
        ("multiview", ["front", "right", "top", "isometric"]),
        (
            "multi_isometric",
            [
                "isometric_front_right",
                "isometric_front_left",
                "isometric_back_right",
                "isometric_back_left",
            ],
        ),
    ],
    ids=["default", "named", "list", "multiview", "multi_isometric"],
)
async def test_snapshot_tool_resolves_camera_views(
    monkeypatch: pytest.MonkeyPatch,
    camera_view: object,
    expected_views: list[str] | None,
):
    mock = MagicMock(return_value=b"jpeg")
    monkeypatch.setattr(server, "zoo_snapshot", mock)

    arguments: dict[str, Any] = {"session_id": "session-id"}
    if camera_view is not None:
        arguments["camera_view"] = camera_view
    await mcp.call_tool("snapshot", arguments=arguments)

    views = mock.call_args.kwargs["views"]
    if expected_views is None:
        assert views is None
    else:
        assert views == [
            CameraView.to_kittycad_camera(CameraView.views.value[name])
            for name in expected_views
        ]


@pytest.mark.parametrize(
    ("name", "expected_vantage"),
    [
        ("front", (0.0, -1.0, 0.0)),
        ("back", (0.0, 1.0, 0.0)),
        ("left", (-1.0, 0.0, 0.0)),
        ("right", (1.0, 0.0, 0.0)),
        ("top", (0.0, 0.0, 1.0)),
        ("bottom", (0.0, 0.0, -1.0)),
        ("isometric_front_right", (1.0, -1.0, 1.0)),
        ("isometric_front_left", (-1.0, -1.0, 1.0)),
        ("isometric_back_right", (1.0, 1.0, 1.0)),
        ("isometric_back_left", (-1.0, 1.0, 1.0)),
    ],
)
def test_named_views_sit_on_the_expected_axis(
    name: str,
    expected_vantage: tuple[float, float, float],
):
    """Named views reach the engine on the app's axes, unmirrored.

    'front' must look from -Y so it shows the -Y face, and every isometric
    must look down from +Z rather than up from below.
    """
    view = CameraView.to_kittycad_camera(CameraView.views.value[name])

    assert (view.vantage.x, view.vantage.y, view.vantage.z) == expected_vantage


@pytest.mark.asyncio
async def test_snapshot_tool_accepts_an_explicit_camera(
    monkeypatch: pytest.MonkeyPatch,
):
    """An explicit camera reaches the engine in the frame the caller gave it."""
    mock = MagicMock(return_value=b"jpeg")
    monkeypatch.setattr(server, "zoo_snapshot", mock)

    await mcp.call_tool(
        "snapshot",
        arguments={
            "session_id": "session-id",
            "camera_view": {
                "up": [0, 0, 1],
                "vantage": [0, -1, 0],
                "center": [0, 0, 0],
            },
        },
    )

    (view,) = mock.call_args.kwargs["views"]
    assert (view.up.x, view.up.y, view.up.z) == (0, 0, 1)
    assert (view.vantage.x, view.vantage.y, view.vantage.z) == (0, -1, 0)
    assert (view.center.x, view.center.y, view.center.z) == (0, 0, 0)


@pytest.mark.asyncio
async def test_snapshot_tool_rejects_an_unknown_camera_view():
    with pytest.raises(ToolError, match="Invalid camera view"):
        await mcp.call_tool(
            "snapshot",
            arguments={"session_id": "session-id", "camera_view": "asdf"},
        )


@pytest.mark.asyncio
async def test_snapshot_tool_rejects_a_malformed_camera_view():
    with pytest.raises(ToolError, match="Invalid camera view"):
        await mcp.call_tool(
            "snapshot",
            arguments={"session_id": "session-id", "camera_view": {"hello": [0, 0, 0]}},
        )


@pytest.mark.asyncio
async def test_snapshot_tool_rejects_more_views_than_fit_a_collage():
    with pytest.raises(ToolError, match="At most 4 camera views"):
        await mcp.call_tool(
            "snapshot",
            arguments={
                "session_id": "session-id",
                "camera_view": ["front", "back", "left", "right", "top"],
            },
        )


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
    tmp_path,
):
    artifact_graph_path = tmp_path / "artifact-graph.json"
    execute = AsyncMock(
        return_value=ResultZooExecuteKclRemote(
            ok=True,
            message="KCL code executed successfully",
            path_artifact_graph=artifact_graph_path,
        )
    )
    exec_project = MagicMock(return_value=artifact_graph_path)
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

    assert _result(execute_response) == {
        "ok": True,
        "message": "KCL code executed successfully",
        "path_artifact_graph": str(artifact_graph_path),
    }
    assert isinstance(project_response, Sequence)
    project_content = project_response[0]
    assert isinstance(project_content, TextContent)
    assert project_content.text == f'"{artifact_graph_path}"'
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
        "engine_util_evaluate_path",
        "get_face_info",
    ):
        description = tools[tool_name].description or ""
        assert "Args:" in description, tool_name
        assert "session_id:" in description, tool_name
