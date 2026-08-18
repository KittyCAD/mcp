import re
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from kittycad.models import (
    EntityGetIndex,
    ModelingCmd,
    TakeSnapshot,
)
from kittycad.models.api_error import ApiError
from kittycad.models.failure_web_socket_response import FailureWebSocketResponse
from kittycad.models.modeling_cmd import (
    OptionDefaultCameraLookAt,
    OptionDefaultCameraSetOrthographic,
    OptionEdgeLinesVisible,
    OptionEntityGetIndex,
    OptionTakeSnapshot,
    OptionViewIsometric,
    OptionZoomToFit,
)
from kittycad.models.modeling_session_data import ModelingSessionData
from kittycad.models.ok_modeling_cmd_response import (
    OkModelingCmdResponse,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionEntityGetIndex as ResponseEntityGetIndex,
)
from kittycad.models.ok_web_socket_response_data import (
    ModelingData,
    ModelingSessionDataData,
    OkWebSocketResponseData,
    OptionModeling,
    OptionModelingSessionData,
)
from kittycad.models.success_web_socket_response import SuccessWebSocketResponse
from kittycad.models.uuid import Uuid
from websockets.exceptions import ConnectionClosedError

from zoo_mcp import ZooMCPException, zoo_tools


@pytest.fixture(autouse=True)
def _clear_modeling_sessions():
    """Keep module-level session state from leaking between tests."""
    zoo_tools.zoo_stop_all_modeling_sessions()
    yield
    zoo_tools.zoo_stop_all_modeling_sessions()


def _ok_frame(request_id: object, response: object) -> SimpleNamespace:
    return SimpleNamespace(
        root=SuccessWebSocketResponse(
            request_id=cast(Any, request_id),
            resp=OkWebSocketResponseData(
                OptionModeling(
                    data=ModelingData(
                        modeling_response=OkModelingCmdResponse(cast(Any, response))
                    )
                )
            ),
            success=True,
        )
    )


def _failure_frame(request_id: object, message: str) -> SimpleNamespace:
    return SimpleNamespace(
        root=FailureWebSocketResponse(
            request_id=cast(Any, request_id),
            errors=[ApiError(error_code=cast(Any, "bad_request"), message=message)],
            success=False,
        )
    )


def test_send_modeling_command_returns_matching_typed_response():
    websocket = MagicMock()
    response_data = EntityGetIndex(entity_index=4)
    expected_response = ResponseEntityGetIndex(data=response_data)

    def recv() -> SimpleNamespace:
        request = websocket.send.call_args.args[0].root
        return _ok_frame(request.cmd_id, expected_response)

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


def test_send_modeling_command_skips_another_commands_failure_frame():
    """A stale failure frame must not be reported as this command's error."""
    websocket = MagicMock()
    expected_response = ResponseEntityGetIndex(data=EntityGetIndex(entity_index=7))
    frames: list[SimpleNamespace] = []

    def recv() -> SimpleNamespace:
        if not frames:
            request = websocket.send.call_args.args[0].root
            frames.append(
                _failure_frame("00000000-0000-0000-0000-000000000000", "stale error")
            )
            frames.append(_ok_frame(request.cmd_id, expected_response))
        return frames.pop(0)

    websocket.recv.side_effect = recv

    result = zoo_tools._send_modeling_command(
        cast(Any, websocket),
        ModelingCmd(OptionEntityGetIndex(entity_id="entity-id")),
        ResponseEntityGetIndex,
        "entity index",
    )

    assert result == expected_response


def test_send_modeling_command_skips_unsolicited_session_frames():
    """A live session interleaves informational frames that carry no request_id."""
    websocket = MagicMock()
    expected_response = ResponseEntityGetIndex(data=EntityGetIndex(entity_index=9))
    frames: list[SimpleNamespace] = []

    def recv() -> SimpleNamespace:
        if not frames:
            request = websocket.send.call_args.args[0].root
            frames.append(
                SimpleNamespace(
                    root=SuccessWebSocketResponse(
                        request_id=None,
                        resp=OkWebSocketResponseData(
                            OptionModelingSessionData(
                                data=ModelingSessionDataData(
                                    session=ModelingSessionData(
                                        api_call_id="api-call-id"
                                    )
                                )
                            )
                        ),
                        success=True,
                    )
                )
            )
            frames.append(_ok_frame(request.cmd_id, expected_response))
        return frames.pop(0)

    websocket.recv.side_effect = recv

    result = zoo_tools._send_modeling_command(
        cast(Any, websocket),
        ModelingCmd(OptionEntityGetIndex(entity_id="entity-id")),
        ResponseEntityGetIndex,
        "entity index",
    )

    assert result == expected_response


def test_send_modeling_command_raises_readable_message_for_own_failure():
    websocket = MagicMock()

    def recv() -> SimpleNamespace:
        request = websocket.send.call_args.args[0].root
        return _failure_frame(request.cmd_id, "No such entity exists")

    websocket.recv.side_effect = recv

    with pytest.raises(ZooMCPException) as exc_info:
        zoo_tools._send_modeling_command(
            cast(Any, websocket),
            ModelingCmd(OptionEntityGetIndex(entity_id="entity-id")),
            ResponseEntityGetIndex,
            "entity index",
        )

    message = str(exc_info.value)
    assert "bad_request: No such entity exists" in message
    assert "errors=" not in message


def test_modeling_session_starts_empty_then_executes_and_reuses_websocket(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    context = MagicMock()
    websocket = MagicMock()
    context.__enter__.return_value = websocket
    artifact_graph_path = tmp_path / "artifact-graph.json"
    artifact_graph_path.write_text("{}")
    execute_project = MagicMock(return_value=artifact_graph_path)
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
    result = zoo_tools.zoo_execute_modeling_command(
        ModelingCmd(OptionEntityGetIndex(entity_id="entity-id")),
        ResponseEntityGetIndex,
        "entity index",
        session_id,
    )

    assert result == expected_response
    assert artifact_graph == artifact_graph_path
    assert artifact_graph_path.exists()
    execute_project.assert_called_once()
    assert execute_project.call_args.args[0] is websocket
    assert send_command.call_args.args[0] is websocket

    zoo_tools.zoo_stop_modeling_session(session_id)
    assert not artifact_graph_path.exists()
    context.__exit__.assert_called_once_with(None, None, None)
    with pytest.raises(ZooMCPException, match="Unknown modeling session"):
        zoo_tools.zoo_execute_modeling_command(
            ModelingCmd(OptionEntityGetIndex(entity_id="entity-id")),
            ResponseEntityGetIndex,
            "entity index",
            session_id,
        )


def test_modeling_session_start_does_not_execute_kcl(monkeypatch: pytest.MonkeyPatch):
    context = MagicMock()
    context.__enter__.return_value = MagicMock()
    execute_project = MagicMock()
    monkeypatch.setattr(zoo_tools, "_modeling_websocket_context", lambda: context)
    monkeypatch.setattr(zoo_tools, "_exec_kcl_project", execute_project)

    session_id = zoo_tools.zoo_start_modeling_session()

    execute_project.assert_not_called()
    zoo_tools.zoo_stop_modeling_session(session_id)
    context.__exit__.assert_called_once_with(None, None, None)


def test_start_handshake_does_not_hold_state_lock_and_shutdown_cancels_it(
    monkeypatch: pytest.MonkeyPatch,
):
    context = MagicMock()
    websocket = MagicMock()
    handshake_started = threading.Event()
    release_handshake = threading.Event()
    results: list[str] = []
    errors: list[Exception] = []

    def enter_websocket() -> MagicMock:
        handshake_started.set()
        release_handshake.wait(timeout=5)
        return websocket

    def start_session() -> None:
        try:
            results.append(zoo_tools.zoo_start_modeling_session())
        except Exception as error:
            errors.append(error)

    context.__enter__.side_effect = enter_websocket
    monkeypatch.setattr(zoo_tools, "_modeling_websocket_context", lambda: context)

    starter = threading.Thread(target=start_session)
    starter.start()
    assert handshake_started.wait(timeout=5)

    started = time.monotonic()
    assert zoo_tools.zoo_get_modeling_sessions() == []
    with pytest.raises(ZooMCPException, match="already open or starting"):
        zoo_tools.zoo_start_modeling_session()
    zoo_tools.zoo_stop_all_modeling_sessions()
    elapsed = time.monotonic() - started

    release_handshake.set()
    starter.join(timeout=5)

    assert elapsed < 0.5
    assert results == []
    assert len(errors) == 1
    assert isinstance(errors[0], ZooMCPException)
    assert "start was canceled" in str(errors[0])
    assert zoo_tools._modeling_session is None
    context.__exit__.assert_called_once_with(None, None, None)


def test_stop_recovers_the_slot_from_a_start_whose_handshake_hangs(
    monkeypatch: pytest.MonkeyPatch,
):
    """The whole recovery loop: the rejection names the pending start, stopping
    it frees the slot immediately, and a new session opens while the hung
    handshake is still outstanding."""
    hung_context = MagicMock()
    live_context = MagicMock()
    live_websocket = MagicMock()
    live_context.__enter__.return_value = live_websocket
    contexts = iter([hung_context, live_context])
    handshake_started = threading.Event()
    release_handshake = threading.Event()
    errors: list[Exception] = []

    def enter_hung_websocket() -> MagicMock:
        handshake_started.set()
        release_handshake.wait(timeout=5)
        return MagicMock()

    def start_session() -> None:
        try:
            zoo_tools.zoo_start_modeling_session()
        except Exception as error:
            errors.append(error)

    hung_context.__enter__.side_effect = enter_hung_websocket
    monkeypatch.setattr(
        zoo_tools, "_modeling_websocket_context", lambda: next(contexts)
    )

    starter = threading.Thread(target=start_session)
    starter.start()
    assert handshake_started.wait(timeout=5)

    # A blocked caller learns the pending session's ID from the rejection, which
    # is its only way to name something get_modeling_sessions does not list.
    assert zoo_tools.zoo_get_modeling_sessions() == []
    with pytest.raises(ZooMCPException, match="already open or starting") as rejection:
        zoo_tools.zoo_start_modeling_session()
    pending_id = re.search(r"'([^']+)'", str(rejection.value))
    assert pending_id is not None

    zoo_tools.zoo_stop_modeling_session(pending_id.group(1))

    # Still hung, but the slot is free, so the next start no longer waits on it.
    assert not release_handshake.is_set()
    recovered_id = zoo_tools.zoo_start_modeling_session()
    assert zoo_tools.zoo_get_modeling_sessions() == [recovered_id]

    release_handshake.set()
    starter.join(timeout=5)

    # The canceled starter closes the socket it opened and leaves the
    # recovered session alone.
    assert len(errors) == 1
    assert "start was canceled" in str(errors[0])
    hung_context.__exit__.assert_called_once_with(None, None, None)
    live_context.__exit__.assert_not_called()
    assert zoo_tools.zoo_get_modeling_sessions() == [recovered_id]

    zoo_tools.zoo_stop_modeling_session(recovered_id)


def test_stop_rejects_an_id_that_is_neither_open_nor_starting(
    monkeypatch: pytest.MonkeyPatch,
):
    context = MagicMock()
    context.__enter__.return_value = MagicMock()
    monkeypatch.setattr(zoo_tools, "_modeling_websocket_context", lambda: context)

    session_id = zoo_tools.zoo_start_modeling_session()
    with pytest.raises(ZooMCPException, match="Unknown modeling session"):
        zoo_tools.zoo_stop_modeling_session("some-other-id")

    # The real session survives a mismatched stop.
    assert zoo_tools.zoo_get_modeling_sessions() == [session_id]
    zoo_tools.zoo_stop_modeling_session(session_id)


def test_busy_session_does_not_block_rejection_of_another_session(
    monkeypatch: pytest.MonkeyPatch,
):
    """Starting another session must not wait for the active session's lock."""
    context = MagicMock()
    context.__enter__.return_value = MagicMock()
    monkeypatch.setattr(zoo_tools, "_modeling_websocket_context", lambda: context)

    session_id = zoo_tools.zoo_start_modeling_session()
    holding = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with zoo_tools._modeling_websocket(session_id):
            holding.set()
            release.wait(timeout=5)

    holder = threading.Thread(target=hold)
    holder.start()
    assert holding.wait(timeout=5)

    started = time.monotonic()
    with pytest.raises(ZooMCPException, match="already open"):
        zoo_tools.zoo_start_modeling_session()
    elapsed = time.monotonic() - started

    release.set()
    holder.join(timeout=5)
    zoo_tools.zoo_stop_modeling_session(session_id)

    assert elapsed < 0.5, (
        f"rejecting another session blocked for {elapsed:.2f}s behind a busy one"
    )


def test_session_is_evicted_when_its_websocket_dies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    context = MagicMock()
    context.__enter__.return_value = MagicMock()
    monkeypatch.setattr(zoo_tools, "_modeling_websocket_context", lambda: context)

    session_id = zoo_tools.zoo_start_modeling_session()
    artifact_graph_path = tmp_path / "artifact-graph.json"
    artifact_graph_path.write_text("{}")
    assert isinstance(zoo_tools._modeling_session, zoo_tools._ModelingSession)
    zoo_tools._modeling_session.artifact_graph_paths.add(artifact_graph_path)

    with (
        pytest.raises(ZooMCPException, match="no longer connected"),
        zoo_tools._modeling_websocket(session_id),
    ):
        raise ConnectionClosedError(None, None)

    assert zoo_tools._modeling_session is None
    assert not artifact_graph_path.exists()
    with (
        pytest.raises(ZooMCPException, match="Unknown modeling session"),
        zoo_tools._modeling_websocket(session_id),
    ):
        pass


def test_only_one_session_can_be_open(monkeypatch: pytest.MonkeyPatch):
    context = MagicMock()
    context.__enter__.return_value = MagicMock()
    monkeypatch.setattr(zoo_tools, "_modeling_websocket_context", lambda: context)

    first = zoo_tools.zoo_start_modeling_session()

    with pytest.raises(ZooMCPException, match="already open"):
        zoo_tools.zoo_start_modeling_session()

    zoo_tools.zoo_stop_modeling_session(first)
    second = zoo_tools.zoo_start_modeling_session()
    zoo_tools.zoo_stop_modeling_session(second)


def test_get_modeling_sessions(monkeypatch: pytest.MonkeyPatch):
    context = MagicMock()
    context.__enter__.return_value = MagicMock()
    monkeypatch.setattr(zoo_tools, "_modeling_websocket_context", lambda: context)

    assert zoo_tools.zoo_get_modeling_sessions() == []

    session_id = zoo_tools.zoo_start_modeling_session()
    assert zoo_tools.zoo_get_modeling_sessions() == [session_id]

    zoo_tools.zoo_stop_modeling_session(session_id)
    assert zoo_tools.zoo_get_modeling_sessions() == []


@pytest.mark.asyncio
async def test_execute_kcl_executes_in_modeling_session(
    monkeypatch: pytest.MonkeyPatch,
):
    artifact_graph_path = Path("artifact-graph.json")
    execute_project = MagicMock(return_value=artifact_graph_path)
    monkeypatch.setattr(zoo_tools, "zoo_exec_kcl_project", execute_project)

    result = await zoo_tools.zoo_execute_kcl(
        kcl_code="code",
        session_id="session-id",
    )

    assert isinstance(result, zoo_tools.ResultZooExecuteKclRemote)
    assert result.ok is True
    assert result.path_artifact_graph == artifact_graph_path
    execute_project.assert_called_once_with(
        kcl_code="code",
        kcl_path=None,
        session_id="session-id",
    )


@pytest.mark.asyncio
async def test_execute_kcl_session_message_states_diagnostics_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    """The engine's exec response carries no non_fatal list, so say so."""
    monkeypatch.setattr(
        zoo_tools,
        "zoo_exec_kcl_project",
        MagicMock(return_value=Path("artifact-graph.json")),
    )

    result = await zoo_tools.zoo_execute_kcl(
        kcl_code="code",
        session_id="session-id",
    )

    assert isinstance(result, zoo_tools.ResultZooExecuteKclRemote)
    assert "Non-fatal diagnostics" in result.message
    assert "not" in result.message


def test_exec_kcl_project_reads_with_a_timeout(monkeypatch: pytest.MonkeyPatch):
    """Reading the raw socket without a timeout would hang forever."""
    websocket = MagicMock()
    websocket._recv_timeout = 12.5
    sent: dict[str, object] = {}

    def send(payload: str) -> None:
        import json

        sent.update(json.loads(payload))

    websocket.ws.send.side_effect = send

    def recv(timeout: float | None = None) -> str:
        import json

        assert timeout == 12.5
        return json.dumps(
            {
                "success": True,
                "request_id": sent["request_id"],
                "resp": {
                    "type": "exec_kcl_project",
                    "data": {"result": {"Ok": {"artifact_graph": {"nodes": {}}}}},
                },
            }
        )

    websocket.ws.recv.side_effect = recv

    result = zoo_tools._exec_kcl_project(cast(Any, websocket), "main.kcl", [])

    try:
        assert result.suffix == ".json"
        assert result.exists()
        assert result.read_text() == '{"nodes": {}}'
    finally:
        result.unlink(missing_ok=True)


def _capture_snapshot_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> list[ModelingCmd]:
    """Record the command sequence zoo_snapshot sends, stubbing the transport."""
    commands: list[ModelingCmd] = []
    responses = {
        zoo_tools.ResponseDefaultCameraLookAt: SimpleNamespace(data=None),
        zoo_tools.ResponseDefaultCameraSetOrthographic: SimpleNamespace(data=None),
        zoo_tools.ResponseViewIsometric: SimpleNamespace(data=None),
        zoo_tools.ResponseZoomToFit: SimpleNamespace(data=None),
        zoo_tools.ResponseEdgeLinesVisible: SimpleNamespace(data=None),
        zoo_tools.ResponseTakeSnapshot: SimpleNamespace(
            data=TakeSnapshot(contents="anBlZw==")
        ),
    }

    def send_command(
        ws: object,
        command: ModelingCmd,
        expected_response: type,
        description: str,
    ) -> object:
        commands.append(command)
        return responses[cast(Any, expected_response)]

    monkeypatch.setattr(zoo_tools, "_send_modeling_command", send_command)
    monkeypatch.setattr(
        zoo_tools,
        "_modeling_websocket",
        lambda session_id: nullcontext(MagicMock()),
    )
    return commands


def _camera_view(name: str) -> OptionDefaultCameraLookAt:
    return zoo_tools.CameraView.to_kittycad_camera(
        zoo_tools.CameraView.views.value[name]
    )


def test_snapshot_frames_the_scene_before_capturing(monkeypatch: pytest.MonkeyPatch):
    commands = _capture_snapshot_commands(monkeypatch)
    resize = MagicMock(return_value=b"resized-jpeg")
    monkeypatch.setattr(zoo_tools, "resize_image", resize)

    result = zoo_tools.zoo_snapshot(
        session_id="session-id",
        max_image_dimension=256,
    )

    assert result == b"resized-jpeg"
    assert [type(command.root) for command in commands] == [
        OptionDefaultCameraSetOrthographic,
        OptionEdgeLinesVisible,
        OptionViewIsometric,
        OptionZoomToFit,
        OptionTakeSnapshot,
    ]
    assert cast(Any, commands[-1].root).format == "jpeg"
    resize.assert_called_once_with(b"jpeg", 256)


def test_snapshot_hides_edge_lines_by_default(monkeypatch: pytest.MonkeyPatch):
    """Edge outlines otherwise compete with highlight_set_entities."""
    commands = _capture_snapshot_commands(monkeypatch)
    monkeypatch.setattr(zoo_tools, "resize_image", MagicMock(return_value=b"jpeg"))

    zoo_tools.zoo_snapshot(session_id="session-id")

    edge_commands = [
        command.root
        for command in commands
        if isinstance(command.root, OptionEdgeLinesVisible)
    ]
    assert len(edge_commands) == 1
    assert cast(Any, edge_commands[0]).hidden is True


def test_snapshot_can_show_edge_lines(monkeypatch: pytest.MonkeyPatch):
    commands = _capture_snapshot_commands(monkeypatch)
    monkeypatch.setattr(zoo_tools, "resize_image", MagicMock(return_value=b"jpeg"))

    zoo_tools.zoo_snapshot(session_id="session-id", highlight_edges=True)

    edge_commands = [
        command.root
        for command in commands
        if isinstance(command.root, OptionEdgeLinesVisible)
    ]
    assert len(edge_commands) == 1
    assert cast(Any, edge_commands[0]).hidden is False


def test_snapshot_sets_edge_visibility_once_across_views(
    monkeypatch: pytest.MonkeyPatch,
):
    """It is a scene-wide setting, so re-sending it per view is wasted work."""
    commands = _capture_snapshot_commands(monkeypatch)
    monkeypatch.setattr(zoo_tools, "resize_image", MagicMock(return_value=b"jpeg"))
    monkeypatch.setattr(
        zoo_tools, "create_image_collage", MagicMock(return_value=b"collage")
    )

    zoo_tools.zoo_snapshot(
        session_id="session-id",
        views=[_camera_view("front"), _camera_view("top")],
    )

    assert (
        len(
            [
                command
                for command in commands
                if isinstance(command.root, OptionEdgeLinesVisible)
            ]
        )
        == 1
    )


def test_snapshot_always_uses_orthographic_projection(
    monkeypatch: pytest.MonkeyPatch,
):
    """Perspective projection distorts measurements read off the image."""
    monkeypatch.setattr(zoo_tools, "resize_image", MagicMock(return_value=b"jpeg"))

    for zoom in (True, False):
        commands = _capture_snapshot_commands(monkeypatch)
        zoo_tools.zoo_snapshot(session_id="session-id", zoom=zoom)
        assert any(
            isinstance(command.root, OptionDefaultCameraSetOrthographic)
            for command in commands
        ), f"zoom={zoom} did not set an orthographic camera"


def test_snapshot_can_preserve_the_current_camera(monkeypatch: pytest.MonkeyPatch):
    commands = _capture_snapshot_commands(monkeypatch)
    monkeypatch.setattr(zoo_tools, "resize_image", MagicMock(return_value=b"jpeg"))

    zoo_tools.zoo_snapshot(session_id="session-id", zoom=False)

    # No isometric/zoom-to-fit: the caller's framing is left alone.
    assert not any(
        isinstance(command.root, OptionViewIsometric | OptionZoomToFit)
        for command in commands
    )


def test_snapshot_captures_one_image_per_view(monkeypatch: pytest.MonkeyPatch):
    commands = _capture_snapshot_commands(monkeypatch)
    monkeypatch.setattr(zoo_tools, "resize_image", MagicMock(return_value=b"resized"))
    collage = MagicMock(return_value=b"collage")
    monkeypatch.setattr(zoo_tools, "create_image_collage", collage)

    views = [_camera_view(name) for name in ("front", "right", "top")]
    result = zoo_tools.zoo_snapshot(session_id="session-id", views=views)

    assert result == b"resized"
    # Each requested view is aimed explicitly, so no isometric fallback.
    assert [
        command.root
        for command in commands
        if isinstance(command.root, OptionDefaultCameraLookAt)
    ] == views
    assert not any(
        isinstance(command.root, OptionViewIsometric) for command in commands
    )
    assert (
        len(
            [
                command
                for command in commands
                if isinstance(command.root, OptionTakeSnapshot)
            ]
        )
        == 3
    )
    collage.assert_called_once_with([b"jpeg"] * 3)


def test_snapshot_fits_the_whole_session(
    monkeypatch: pytest.MonkeyPatch,
):
    commands = _capture_snapshot_commands(monkeypatch)
    monkeypatch.setattr(zoo_tools, "resize_image", MagicMock(return_value=b"jpeg"))

    zoo_tools.zoo_snapshot(session_id="session-id")

    zoom_commands = [
        command.root
        for command in commands
        if isinstance(command.root, OptionZoomToFit)
    ]
    assert len(zoom_commands) == 1
    assert cast(Any, zoom_commands[0]).object_ids == []


def test_face_info_uuid_type_is_accepted():
    """Guard the Uuid alias the tools pass through to the engine."""
    assert str(Uuid("face-id")) == "face-id"
