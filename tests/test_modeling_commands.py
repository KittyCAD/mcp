from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from kittycad.models import (
    CurveGetEndPoints,
    CurveGetType,
    EdgeGetLength,
    EngineUtilEvaluatePath,
    EntityGetAllChildUuids,
    EntityGetIndex,
    EntityGetParentId,
    EntityGetSketchPaths,
    EntityReference,
    EntityType,
    HighlightSetEntities,
    ModelingCmd,
    Point2d,
    Point3d,
    SceneSelectionType,
    SelectEntity,
    SelectWithPoint,
    SetSelectionFilter,
)
from kittycad.models.entity_reference import OptionFace
from kittycad.models.modeling_cmd import (
    OptionCurveGetEndPoints,
    OptionCurveGetType,
    OptionEdgeGetLength,
    OptionEngineUtilEvaluatePath,
    OptionEntityGetAllChildUuids,
    OptionEntityGetIndex,
    OptionEntityGetParentId,
    OptionEntityGetSketchPaths,
    OptionHighlightSetEntities,
    OptionSelectEntity,
    OptionSelectWithPoint,
    OptionSetSelectionFilter,
)
from kittycad.models.ok_modeling_cmd_response import (
    OkModelingCmdResponse,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionEntityGetIndex as ResponseEntityGetIndex,
)
from kittycad.models.ok_web_socket_response_data import (
    ModelingData,
    OkWebSocketResponseData,
    OptionModeling,
)
from kittycad.models.success_web_socket_response import SuccessWebSocketResponse
from kittycad.models.uuid import Uuid

from zoo_mcp import ZooMCPException, zoo_tools


def test_send_modeling_command_returns_matching_typed_response():
    websocket = MagicMock()
    response_data = EntityGetIndex(entity_index=4)
    expected_response = ResponseEntityGetIndex(data=response_data)

    def recv() -> SimpleNamespace:
        request = websocket.send.call_args.args[0].root
        response = SuccessWebSocketResponse(
            request_id=request.cmd_id,
            resp=OkWebSocketResponseData(
                OptionModeling(
                    data=ModelingData(
                        modeling_response=OkModelingCmdResponse(expected_response)
                    )
                )
            ),
            success=True,
        )
        return SimpleNamespace(root=response)

    websocket.recv.side_effect = recv
    result = zoo_tools._send_modeling_command(
        cast(Any, websocket),
        ModelingCmd(OptionEntityGetIndex(entity_id="entity-id")),
        ResponseEntityGetIndex,
        "entity index",
    )

    assert result == expected_response
    request = websocket.send.call_args.args[0].root
    assert request.cmd.root.model_dump() == {
        "entity_id": "entity-id",
        "type": "entity_get_index",
    }


def test_modeling_session_starts_empty_then_executes_and_reuses_websocket(
    monkeypatch: pytest.MonkeyPatch,
):
    zoo_tools.zoo_stop_all_modeling_sessions()
    context = MagicMock()
    websocket = MagicMock()
    context.__enter__.return_value = websocket
    execute_project = MagicMock(return_value={})
    expected_response = ResponseEntityGetIndex(data=EntityGetIndex(entity_index=4))
    send_command = MagicMock(return_value=expected_response)
    monkeypatch.setattr(zoo_tools, "_modeling_websocket_context", lambda: context)
    monkeypatch.setattr(zoo_tools, "_exec_kcl_project", execute_project)
    monkeypatch.setattr(zoo_tools, "_send_modeling_command", send_command)

    session_id = zoo_tools.zoo_start_modeling_session()
    execute_project.assert_not_called()

    artifact_graph = zoo_tools.zoo_exec_kcl_project(
        kcl_code="code",
        session_id=session_id,
    )
    result = zoo_tools.zoo_entity_get_index(
        kcl_code=None,
        kcl_path=None,
        entity_id=Uuid("entity-id"),
        session_id=session_id,
    )

    assert result == expected_response.data
    assert artifact_graph == {}
    execute_project.assert_called_once()
    assert execute_project.call_args.args[0] is websocket
    assert send_command.call_args.args[0] is websocket

    zoo_tools.zoo_stop_modeling_session(session_id)
    context.__exit__.assert_called_once_with(None, None, None)
    with pytest.raises(ZooMCPException, match="Unknown modeling session"):
        zoo_tools.zoo_entity_get_index(
            kcl_code=None,
            kcl_path=None,
            entity_id=Uuid("entity-id"),
            session_id=session_id,
        )


def test_modeling_session_start_does_not_execute_kcl(monkeypatch: pytest.MonkeyPatch):
    zoo_tools.zoo_stop_all_modeling_sessions()
    context = MagicMock()
    context.__enter__.return_value = MagicMock()
    execute_project = MagicMock()
    monkeypatch.setattr(zoo_tools, "_modeling_websocket_context", lambda: context)
    monkeypatch.setattr(zoo_tools, "_exec_kcl_project", execute_project)

    session_id = zoo_tools.zoo_start_modeling_session()

    execute_project.assert_not_called()
    zoo_tools.zoo_stop_modeling_session(session_id)
    context.__exit__.assert_called_once_with(None, None, None)


@pytest.mark.asyncio
async def test_execute_kcl_executes_in_modeling_session(
    monkeypatch: pytest.MonkeyPatch,
):
    execute_project = MagicMock(return_value={})
    monkeypatch.setattr(zoo_tools, "zoo_exec_kcl_project", execute_project)

    result = await zoo_tools.zoo_execute_kcl(
        kcl_code="code",
        session_id="session-id",
    )

    assert result == (True, "KCL code executed successfully")
    execute_project.assert_called_once_with(
        kcl_code="code",
        kcl_path=None,
        session_id="session-id",
    )


@pytest.mark.parametrize(
    ("call", "request_type", "request_fields", "response_type", "response_data"),
    [
        (
            lambda: zoo_tools.zoo_select_with_point(
                "code",
                None,
                Point2d(x=10, y=20),
                SceneSelectionType.ADD,
            ),
            OptionSelectWithPoint,
            {
                "selected_at_window": {"x": 10.0, "y": 20.0},
                "selection_type": "add",
            },
            zoo_tools.ResponseSelectWithPoint,
            SelectWithPoint(entity_id="selected-id"),
        ),
        (
            lambda: zoo_tools.zoo_set_selection_filter(
                "code", None, [EntityType.FACE, EntityType.EDGE]
            ),
            OptionSetSelectionFilter,
            {"filter": ["face", "edge"]},
            zoo_tools.ResponseSetSelectionFilter,
            SetSelectionFilter(),
        ),
        (
            lambda: zoo_tools.zoo_select_entity(
                "code",
                None,
                [EntityReference(OptionFace(face_id="face-id"))],
            ),
            OptionSelectEntity,
            {
                "entities": [
                    {
                        "face_id": "face-id",
                        "topology_fallback": None,
                        "type": "face",
                    }
                ]
            },
            zoo_tools.ResponseSelectEntity,
            SelectEntity(),
        ),
        (
            lambda: zoo_tools.zoo_curve_get_end_points("code", None, Uuid("curve-id")),
            OptionCurveGetEndPoints,
            {"curve_id": "curve-id"},
            zoo_tools.ResponseCurveGetEndPoints,
            CurveGetEndPoints(
                start=Point3d(x=0, y=0, z=0),
                end=Point3d(x=1, y=2, z=3),
            ),
        ),
        (
            lambda: zoo_tools.zoo_engine_util_evaluate_path(
                "code", None, '{"type":"line"}', 0.25
            ),
            OptionEngineUtilEvaluatePath,
            {"path_json": '{"type":"line"}', "t": 0.25},
            zoo_tools.ResponseEngineUtilEvaluatePath,
            EngineUtilEvaluatePath(pos=Point3d(x=1, y=2, z=3)),
        ),
        (
            lambda: zoo_tools.zoo_curve_get_type("code", None, Uuid("curve-id")),
            OptionCurveGetType,
            {"curve_id": "curve-id"},
            zoo_tools.ResponseCurveGetType,
            CurveGetType(curve_type="line"),
        ),
        (
            lambda: zoo_tools.zoo_edge_get_length("code", None, Uuid("edge-id")),
            OptionEdgeGetLength,
            {"edge_id": "edge-id"},
            zoo_tools.ResponseEdgeGetLength,
            EdgeGetLength(length=12.5),
        ),
        (
            lambda: zoo_tools.zoo_entity_get_all_child_uuids(
                "code", None, Uuid("entity-id")
            ),
            OptionEntityGetAllChildUuids,
            {"entity_id": "entity-id"},
            zoo_tools.ResponseEntityGetAllChildUuids,
            EntityGetAllChildUuids(entity_ids=["child-id"]),
        ),
        (
            lambda: zoo_tools.zoo_entity_get_index("code", None, Uuid("entity-id")),
            OptionEntityGetIndex,
            {"entity_id": "entity-id"},
            zoo_tools.ResponseEntityGetIndex,
            EntityGetIndex(entity_index=3),
        ),
        (
            lambda: zoo_tools.zoo_entity_get_parent_id("code", None, Uuid("entity-id")),
            OptionEntityGetParentId,
            {"entity_id": "entity-id"},
            zoo_tools.ResponseEntityGetParentId,
            EntityGetParentId(entity_id="parent-id"),
        ),
        (
            lambda: zoo_tools.zoo_entity_get_sketch_paths(
                "code", None, Uuid("entity-id")
            ),
            OptionEntityGetSketchPaths,
            {"entity_id": "entity-id"},
            zoo_tools.ResponseEntityGetSketchPaths,
            EntityGetSketchPaths(entity_ids=["path-id"]),
        ),
        (
            lambda: zoo_tools.zoo_highlight_set_entities(
                "code", None, ["entity-1", "entity-2"]
            ),
            OptionHighlightSetEntities,
            {"entities": ["entity-1", "entity-2"]},
            zoo_tools.ResponseHighlightSetEntities,
            HighlightSetEntities(),
        ),
    ],
)
def test_modeling_tool_constructs_expected_command(
    monkeypatch: pytest.MonkeyPatch,
    call: Callable[[], object],
    request_type: type,
    request_fields: dict[str, object],
    response_type: type,
    response_data: object,
):
    captured: dict[str, object] = {}

    def execute(
        kcl_code: str | None,
        kcl_path: object,
        command: ModelingCmd,
        expected_response: type,
        response_description: str,
        session_id: str | None,
    ) -> SimpleNamespace:
        captured.update(
            kcl_code=kcl_code,
            kcl_path=kcl_path,
            command=command,
            expected_response=expected_response,
            response_description=response_description,
            session_id=session_id,
        )
        return SimpleNamespace(data=response_data)

    monkeypatch.setattr(zoo_tools, "_execute_project_modeling_command", execute)

    assert call() is response_data
    assert captured["kcl_code"] == "code"
    assert captured["kcl_path"] is None
    assert captured["session_id"] is None
    command = cast(ModelingCmd, captured["command"])
    assert isinstance(command.root, request_type)
    assert command.root.model_dump(exclude={"type"}) == request_fields
    assert captured["expected_response"] is response_type
