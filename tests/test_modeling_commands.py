import asyncio
import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from kittycad.models import (
    EntityGetIndex,
    ModelingCmd,
    StepImportTargetRepresentation,
    TakeSnapshot,
    WebSocketRequest,
    WebSocketResponse,
)
from kittycad.models.api_error import ApiError
from kittycad.models.failure_web_socket_response import FailureWebSocketResponse
from kittycad.models.input_format3d import OptionStep
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


@pytest_asyncio.fixture(autouse=True)
async def _clear_modeling_sessions():
    """Keep module-level session state from leaking between tests."""
    await zoo_tools.zoo_stop_all_modeling_sessions()
    yield
    await zoo_tools.zoo_stop_all_modeling_sessions()


@asynccontextmanager
async def _async_context(value: Any) -> AsyncIterator[Any]:
    yield value


def _sent_request(websocket: AsyncMock) -> Any:
    return WebSocketRequest.model_validate_json(websocket.send.call_args.args[0]).root


class _FakeClock:
    """A monotonic clock the tests advance by hand.

    Lets a 300s budget be exercised without a 300s test, and makes the
    frame-by-frame accounting exact rather than timing-dependent.
    """

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _session_frame() -> str:
    """An unsolicited frame of the kind a live session interleaves."""
    return WebSocketResponse(
        SuccessWebSocketResponse(
            request_id=None,
            resp=OkWebSocketResponseData(
                OptionModelingSessionData(
                    data=ModelingSessionDataData(
                        session=ModelingSessionData(api_call_id="api-call-id")
                    )
                )
            ),
            success=True,
        )
    ).model_dump_json()


def _ok_frame(request_id: object, response: object) -> str:
    return WebSocketResponse(
        SuccessWebSocketResponse(
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
    ).model_dump_json()


def _failure_frame(request_id: object, message: str) -> str:
    return WebSocketResponse(
        FailureWebSocketResponse(
            request_id=cast(Any, request_id),
            errors=[ApiError(error_code=cast(Any, "bad_request"), message=message)],
            success=False,
        )
    ).model_dump_json()


@pytest.mark.asyncio
async def test_modeling_websocket_uses_async_client_configuration(
    monkeypatch: pytest.MonkeyPatch,
):
    connection = AsyncMock()
    open_websocket = AsyncMock(return_value=connection)
    monkeypatch.setattr(zoo_tools, "connect", open_websocket)
    client = SimpleNamespace(
        base_url="https://api.example.test",
        get_headers=MagicMock(return_value={"Authorization": "Bearer token"}),
        verify_ssl=True,
    )

    result = await zoo_tools._open_modeling_websocket(cast(Any, client))

    assert result is connection
    open_websocket.assert_awaited_once_with(
        "wss://api.example.test/ws/modeling/commands?"
        "fps=30&post_effect=ssao&show_grid=false&unlocked_framerate=false&"
        "video_res_height=1024&video_res_width=1024&webrtc=false",
        additional_headers={"Authorization": "Bearer token"},
        close_timeout=120,
        max_size=None,
        ssl=True,
    )


@pytest.mark.asyncio
async def test_modeling_websocket_omits_tls_for_an_http_host(
    monkeypatch: pytest.MonkeyPatch,
):
    connection = AsyncMock()
    open_websocket = AsyncMock(return_value=connection)
    monkeypatch.setattr(zoo_tools, "connect", open_websocket)
    client = SimpleNamespace(
        base_url="http://localhost:8080",
        get_headers=MagicMock(return_value={"Authorization": "Bearer token"}),
        verify_ssl=zoo_tools.ctx,
    )

    result = await zoo_tools._open_modeling_websocket(cast(Any, client))

    assert result is connection
    await_args = open_websocket.await_args
    assert await_args is not None
    assert await_args.kwargs["ssl"] is None
    assert await_args.args[0].startswith("ws://localhost:8080/")


@pytest.mark.asyncio
async def test_send_modeling_command_returns_matching_typed_response():
    websocket = AsyncMock()
    response_data = EntityGetIndex(entity_index=4)
    expected_response = ResponseEntityGetIndex(data=response_data)

    async def recv() -> str:
        request = _sent_request(websocket)
        return _ok_frame(request.cmd_id, expected_response)

    websocket.recv.side_effect = recv
    result = await zoo_tools._send_modeling_command(
        cast(Any, websocket),
        ModelingCmd(OptionEntityGetIndex(entity_id="entity-id")),
        ResponseEntityGetIndex,
        "entity index",
    )

    assert result == expected_response
    request = _sent_request(websocket)
    assert request.cmd.root.model_dump() == {
        "entity_id": "entity-id",
        "type": "entity_get_index",
    }


@pytest.mark.asyncio
async def test_send_modeling_command_skips_another_commands_failure_frame():
    """A stale failure frame must not be reported as this command's error."""
    websocket = AsyncMock()
    expected_response = ResponseEntityGetIndex(data=EntityGetIndex(entity_index=7))
    frames: list[str] = []

    async def recv() -> str:
        if not frames:
            request = _sent_request(websocket)
            frames.append(
                _failure_frame("00000000-0000-0000-0000-000000000000", "stale error")
            )
            frames.append(_ok_frame(request.cmd_id, expected_response))
        return frames.pop(0)

    websocket.recv.side_effect = recv

    result = await zoo_tools._send_modeling_command(
        cast(Any, websocket),
        ModelingCmd(OptionEntityGetIndex(entity_id="entity-id")),
        ResponseEntityGetIndex,
        "entity index",
    )

    assert result == expected_response


@pytest.mark.asyncio
async def test_send_modeling_command_skips_unsolicited_session_frames():
    """A live session interleaves informational frames that carry no request_id."""
    websocket = AsyncMock()
    expected_response = ResponseEntityGetIndex(data=EntityGetIndex(entity_index=9))
    frames: list[str] = []

    async def recv() -> str:
        if not frames:
            request = _sent_request(websocket)
            frames.append(_session_frame())
            frames.append(_ok_frame(request.cmd_id, expected_response))
        return frames.pop(0)

    websocket.recv.side_effect = recv

    result = await zoo_tools._send_modeling_command(
        cast(Any, websocket),
        ModelingCmd(OptionEntityGetIndex(entity_id="entity-id")),
        ResponseEntityGetIndex,
        "entity index",
    )

    assert result == expected_response


@pytest.mark.asyncio
async def test_send_modeling_command_raises_readable_message_for_own_failure():
    websocket = AsyncMock()

    async def recv() -> str:
        request = _sent_request(websocket)
        return _failure_frame(request.cmd_id, "No such entity exists")

    websocket.recv.side_effect = recv

    with pytest.raises(ZooMCPException) as exc_info:
        await zoo_tools._send_modeling_command(
            cast(Any, websocket),
            ModelingCmd(OptionEntityGetIndex(entity_id="entity-id")),
            ResponseEntityGetIndex,
            "entity index",
        )

    message = str(exc_info.value)
    assert "bad_request: No such entity exists" in message
    assert "errors=" not in message


@pytest.mark.asyncio
async def test_modeling_session_starts_empty_then_executes_and_reuses_websocket(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    websocket = AsyncMock()
    artifact_graph_path = tmp_path / "artifact-graph.json"
    artifact_graph_path.write_text("{}")
    execute_project = AsyncMock(return_value=artifact_graph_path)
    expected_response = ResponseEntityGetIndex(data=EntityGetIndex(entity_index=4))
    send_command = AsyncMock(return_value=expected_response)
    monkeypatch.setattr(
        zoo_tools, "_open_modeling_websocket", AsyncMock(return_value=websocket)
    )
    monkeypatch.setattr(zoo_tools, "_exec_kcl_project", execute_project)
    monkeypatch.setattr(zoo_tools, "_send_modeling_command", send_command)

    session_id = await zoo_tools.zoo_start_modeling_session()
    execute_project.assert_not_called()

    artifact_graph = await zoo_tools.zoo_exec_kcl_project(
        kcl_code="code",
        session_id=session_id,
    )
    result = await zoo_tools.zoo_execute_modeling_command(
        ModelingCmd(OptionEntityGetIndex(entity_id="entity-id")),
        ResponseEntityGetIndex,
        "entity index",
        session_id,
    )

    assert result == expected_response
    assert artifact_graph == artifact_graph_path
    assert artifact_graph_path.exists()
    execute_project.assert_awaited_once()
    assert execute_project.call_args.args[0] is websocket
    assert send_command.call_args.args[0] is websocket

    await zoo_tools.zoo_stop_modeling_session(session_id)
    assert not artifact_graph_path.exists()
    websocket.close.assert_awaited_once_with()
    with pytest.raises(ZooMCPException, match="Unknown modeling session"):
        await zoo_tools.zoo_execute_modeling_command(
            ModelingCmd(OptionEntityGetIndex(entity_id="entity-id")),
            ResponseEntityGetIndex,
            "entity index",
            session_id,
        )


@pytest.mark.asyncio
async def test_modeling_session_start_does_not_execute_kcl(
    monkeypatch: pytest.MonkeyPatch,
):
    websocket = AsyncMock()
    execute_project = AsyncMock()
    monkeypatch.setattr(
        zoo_tools, "_open_modeling_websocket", AsyncMock(return_value=websocket)
    )
    monkeypatch.setattr(zoo_tools, "_exec_kcl_project", execute_project)

    session_id = await zoo_tools.zoo_start_modeling_session()

    execute_project.assert_not_called()
    await zoo_tools.zoo_stop_modeling_session(session_id)
    websocket.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_modeling_client_construction_failure_releases_the_session_slot(
    monkeypatch: pytest.MonkeyPatch,
):
    class BrokenClient:
        def __init__(self, **kwargs: object) -> None:
            raise ValueError("missing token")

    monkeypatch.setattr(zoo_tools, "AsyncKittyCAD", BrokenClient)

    with pytest.raises(ValueError, match="missing token"):
        await zoo_tools.zoo_start_modeling_session()

    assert zoo_tools._modeling_session is None


@pytest.mark.asyncio
async def test_start_handshake_does_not_hold_state_lock_and_shutdown_cancels_it(
    monkeypatch: pytest.MonkeyPatch,
):
    websocket = AsyncMock()
    handshake_started = asyncio.Event()
    release_handshake = asyncio.Event()

    async def open_websocket(client: object) -> AsyncMock:
        handshake_started.set()
        await release_handshake.wait()
        return websocket

    monkeypatch.setattr(zoo_tools, "_open_modeling_websocket", open_websocket)

    starter = asyncio.create_task(zoo_tools.zoo_start_modeling_session())
    await asyncio.wait_for(handshake_started.wait(), timeout=1)

    assert zoo_tools.zoo_get_modeling_sessions() == []
    with pytest.raises(ZooMCPException, match="already open or starting"):
        await zoo_tools.zoo_start_modeling_session()
    await asyncio.wait_for(zoo_tools.zoo_stop_all_modeling_sessions(), timeout=0.5)

    release_handshake.set()
    with pytest.raises(ZooMCPException, match="start was canceled"):
        await starter

    assert zoo_tools._modeling_session is None
    websocket.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_stop_recovers_the_slot_from_a_start_whose_handshake_hangs(
    monkeypatch: pytest.MonkeyPatch,
):
    """The whole recovery loop: the rejection names the pending start, stopping
    it frees the slot immediately, and a new session opens while the hung
    handshake is still outstanding."""
    hung_websocket = AsyncMock()
    live_websocket = AsyncMock()
    handshake_started = asyncio.Event()
    release_handshake = asyncio.Event()
    calls = 0

    async def open_websocket(client: object) -> AsyncMock:
        nonlocal calls
        calls += 1
        if calls == 1:
            handshake_started.set()
            await release_handshake.wait()
            return hung_websocket
        return live_websocket

    monkeypatch.setattr(zoo_tools, "_open_modeling_websocket", open_websocket)

    starter = asyncio.create_task(zoo_tools.zoo_start_modeling_session())
    await asyncio.wait_for(handshake_started.wait(), timeout=1)

    # A blocked caller learns the pending session's ID from the rejection, which
    # is its only way to name something get_modeling_sessions does not list.
    assert zoo_tools.zoo_get_modeling_sessions() == []
    with pytest.raises(ZooMCPException, match="already open or starting") as rejection:
        await zoo_tools.zoo_start_modeling_session()
    pending_id = re.search(r"'([^']+)'", str(rejection.value))
    assert pending_id is not None

    await zoo_tools.zoo_stop_modeling_session(pending_id.group(1))

    # Still hung, but the slot is free, so the next start no longer waits on it.
    assert not release_handshake.is_set()
    recovered_id = await zoo_tools.zoo_start_modeling_session()
    assert zoo_tools.zoo_get_modeling_sessions() == [recovered_id]

    release_handshake.set()
    with pytest.raises(ZooMCPException, match="start was canceled"):
        await starter

    # The canceled starter closes the socket it opened and leaves the
    # recovered session alone.
    hung_websocket.close.assert_awaited_once_with()
    live_websocket.close.assert_not_awaited()
    assert zoo_tools.zoo_get_modeling_sessions() == [recovered_id]

    await zoo_tools.zoo_stop_modeling_session(recovered_id)


@pytest.mark.asyncio
async def test_stop_rejects_an_id_that_is_neither_open_nor_starting(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(zoo_tools, "_open_modeling_websocket", AsyncMock())

    session_id = await zoo_tools.zoo_start_modeling_session()
    with pytest.raises(ZooMCPException, match="Unknown modeling session"):
        await zoo_tools.zoo_stop_modeling_session("some-other-id")

    # The real session survives a mismatched stop.
    assert zoo_tools.zoo_get_modeling_sessions() == [session_id]
    await zoo_tools.zoo_stop_modeling_session(session_id)


@pytest.mark.asyncio
async def test_busy_session_does_not_block_rejection_of_another_session(
    monkeypatch: pytest.MonkeyPatch,
):
    """Starting another session must not wait for the active session's lock."""
    monkeypatch.setattr(zoo_tools, "_open_modeling_websocket", AsyncMock())

    session_id = await zoo_tools.zoo_start_modeling_session()
    holding = asyncio.Event()
    release = asyncio.Event()

    async def hold() -> None:
        async with zoo_tools._modeling_websocket(session_id):
            holding.set()
            await release.wait()

    holder = asyncio.create_task(hold())
    await asyncio.wait_for(holding.wait(), timeout=1)

    with pytest.raises(ZooMCPException, match="already open"):
        await asyncio.wait_for(zoo_tools.zoo_start_modeling_session(), timeout=0.5)

    release.set()
    await holder
    await zoo_tools.zoo_stop_modeling_session(session_id)


@pytest.mark.asyncio
async def test_session_is_evicted_when_its_websocket_dies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(zoo_tools, "_open_modeling_websocket", AsyncMock())

    session_id = await zoo_tools.zoo_start_modeling_session()
    artifact_graph_path = tmp_path / "artifact-graph.json"
    artifact_graph_path.write_text("{}")
    assert isinstance(zoo_tools._modeling_session, zoo_tools._ModelingSession)
    zoo_tools._modeling_session.artifact_graph_paths.add(artifact_graph_path)

    with pytest.raises(ZooMCPException, match="no longer connected"):
        async with zoo_tools._modeling_websocket(session_id):
            raise ConnectionClosedError(None, None)

    assert zoo_tools._modeling_session is None
    assert not artifact_graph_path.exists()
    with pytest.raises(ZooMCPException, match="Unknown modeling session"):
        async with zoo_tools._modeling_websocket(session_id):
            pass


@pytest.mark.asyncio
async def test_only_one_session_can_be_open(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(zoo_tools, "_open_modeling_websocket", AsyncMock())

    first = await zoo_tools.zoo_start_modeling_session()

    with pytest.raises(ZooMCPException, match="already open"):
        await zoo_tools.zoo_start_modeling_session()

    await zoo_tools.zoo_stop_modeling_session(first)
    second = await zoo_tools.zoo_start_modeling_session()
    await zoo_tools.zoo_stop_modeling_session(second)


@pytest.mark.asyncio
async def test_get_modeling_sessions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(zoo_tools, "_open_modeling_websocket", AsyncMock())

    assert zoo_tools.zoo_get_modeling_sessions() == []

    session_id = await zoo_tools.zoo_start_modeling_session()
    assert zoo_tools.zoo_get_modeling_sessions() == [session_id]

    await zoo_tools.zoo_stop_modeling_session(session_id)
    assert zoo_tools.zoo_get_modeling_sessions() == []


@pytest.mark.asyncio
async def test_execute_kcl_executes_in_modeling_session(
    monkeypatch: pytest.MonkeyPatch,
):
    artifact_graph_path = Path("artifact-graph.json")
    execute_project = AsyncMock(return_value=artifact_graph_path)
    monkeypatch.setattr(zoo_tools, "zoo_exec_kcl_project", execute_project)

    result = await zoo_tools.zoo_execute_kcl(
        kcl_code="code",
        session_id="session-id",
    )

    assert isinstance(result, zoo_tools.ResultZooExecuteKclRemote)
    assert result.ok is True
    assert result.path_artifact_graph == artifact_graph_path
    execute_project.assert_awaited_once_with(
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
        AsyncMock(return_value=Path("artifact-graph.json")),
    )

    result = await zoo_tools.zoo_execute_kcl(
        kcl_code="code",
        session_id="session-id",
    )

    assert isinstance(result, zoo_tools.ResultZooExecuteKclRemote)
    assert "Non-fatal diagnostics" in result.message
    assert "not" in result.message


@pytest.mark.asyncio
async def test_exec_kcl_project_reads_within_the_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
):
    """Each read is bounded by what is left of the call, not a fixed per-read value.

    A fixed per-read timeout cannot bound this loop: the frame skipped below
    would restart it, which is what let a stalled engine hang the caller.
    """
    clock = _FakeClock()
    monkeypatch.setattr(zoo_tools, "monotonic", clock)
    monkeypatch.setattr(zoo_tools, "MODELING_COMMAND_TIMEOUT", 100.0)

    websocket = AsyncMock()
    sent: dict[str, object] = {}
    timeouts: list[float | None] = []

    async def send(payload: str) -> None:
        sent.update(json.loads(payload))

    websocket.send.side_effect = send

    async def recv() -> str:
        clock.advance(10.0)
        if len(timeouts) == 2:
            # A frame for someone else's request must consume the budget.
            return json.dumps({"success": True, "request_id": "other"})
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

    websocket.recv.side_effect = recv
    wait_for = asyncio.wait_for

    async def record_timeout(awaitable: Any, timeout: float) -> Any:
        timeouts.append(timeout)
        return await awaitable

    monkeypatch.setattr(zoo_tools.asyncio, "wait_for", record_timeout)

    result = await zoo_tools._exec_kcl_project(cast(Any, websocket), "main.kcl", [])
    monkeypatch.setattr(zoo_tools.asyncio, "wait_for", wait_for)

    assert timeouts == [100.0, 100.0, 90.0]

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

    async def send_command(
        ws: object,
        command: ModelingCmd,
        expected_response: type,
        description: str,
        deadline: object = None,
    ) -> object:
        commands.append(command)
        return responses[cast(Any, expected_response)]

    monkeypatch.setattr(zoo_tools, "_send_modeling_command", send_command)
    monkeypatch.setattr(
        zoo_tools,
        "_modeling_websocket",
        lambda session_id: _async_context(AsyncMock()),
    )
    return commands


def _camera_view(name: str) -> OptionDefaultCameraLookAt:
    return zoo_tools.CameraView.to_kittycad_camera(
        zoo_tools.CameraView.views.value[name]
    )


@pytest.mark.asyncio
async def test_snapshot_frames_the_scene_before_capturing(
    monkeypatch: pytest.MonkeyPatch,
):
    commands = _capture_snapshot_commands(monkeypatch)
    resize = MagicMock(return_value=b"resized-jpeg")
    monkeypatch.setattr(zoo_tools, "resize_image", resize)

    result = await zoo_tools.zoo_snapshot(
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


@pytest.mark.asyncio
async def test_snapshot_hides_edge_lines_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    """Edge outlines otherwise compete with highlight_set_entities."""
    commands = _capture_snapshot_commands(monkeypatch)
    monkeypatch.setattr(zoo_tools, "resize_image", MagicMock(return_value=b"jpeg"))
    monkeypatch.setattr(
        zoo_tools, "create_image_collage", MagicMock(return_value=b"collage")
    )

    await zoo_tools.zoo_snapshot(session_id="session-id")

    edge_commands = [
        command.root
        for command in commands
        if isinstance(command.root, OptionEdgeLinesVisible)
    ]
    assert len(edge_commands) == 1
    assert cast(Any, edge_commands[0]).hidden is True


@pytest.mark.asyncio
async def test_snapshot_can_show_edge_lines(monkeypatch: pytest.MonkeyPatch):
    commands = _capture_snapshot_commands(monkeypatch)
    monkeypatch.setattr(zoo_tools, "resize_image", MagicMock(return_value=b"jpeg"))

    await zoo_tools.zoo_snapshot(session_id="session-id", highlight_edges=True)

    edge_commands = [
        command.root
        for command in commands
        if isinstance(command.root, OptionEdgeLinesVisible)
    ]
    assert len(edge_commands) == 1
    assert cast(Any, edge_commands[0]).hidden is False


@pytest.mark.asyncio
async def test_snapshot_sets_edge_visibility_once_across_views(
    monkeypatch: pytest.MonkeyPatch,
):
    """It is a scene-wide setting, so re-sending it per view is wasted work."""
    commands = _capture_snapshot_commands(monkeypatch)
    monkeypatch.setattr(zoo_tools, "resize_image", MagicMock(return_value=b"jpeg"))
    monkeypatch.setattr(
        zoo_tools, "create_image_collage", MagicMock(return_value=b"collage")
    )

    await zoo_tools.zoo_snapshot(
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


@pytest.mark.asyncio
async def test_snapshot_always_uses_orthographic_projection(
    monkeypatch: pytest.MonkeyPatch,
):
    """Perspective projection distorts measurements read off the image."""
    monkeypatch.setattr(zoo_tools, "resize_image", MagicMock(return_value=b"jpeg"))

    for zoom in (True, False):
        commands = _capture_snapshot_commands(monkeypatch)
        await zoo_tools.zoo_snapshot(session_id="session-id", zoom=zoom)
        assert any(
            isinstance(command.root, OptionDefaultCameraSetOrthographic)
            for command in commands
        ), f"zoom={zoom} did not set an orthographic camera"


@pytest.mark.asyncio
async def test_snapshot_can_preserve_the_current_camera(
    monkeypatch: pytest.MonkeyPatch,
):
    commands = _capture_snapshot_commands(monkeypatch)
    monkeypatch.setattr(zoo_tools, "resize_image", MagicMock(return_value=b"jpeg"))

    await zoo_tools.zoo_snapshot(session_id="session-id", zoom=False)

    # No isometric/zoom-to-fit: the caller's framing is left alone.
    assert not any(
        isinstance(command.root, OptionViewIsometric | OptionZoomToFit)
        for command in commands
    )


@pytest.mark.asyncio
async def test_snapshot_captures_one_image_per_view(monkeypatch: pytest.MonkeyPatch):
    commands = _capture_snapshot_commands(monkeypatch)
    monkeypatch.setattr(zoo_tools, "resize_image", MagicMock(return_value=b"resized"))
    collage = MagicMock(return_value=b"collage")
    monkeypatch.setattr(zoo_tools, "create_image_collage", collage)

    views = [_camera_view(name) for name in ("front", "right", "top")]
    result = await zoo_tools.zoo_snapshot(session_id="session-id", views=views)

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


@pytest.mark.asyncio
async def test_snapshot_fits_the_whole_session(
    monkeypatch: pytest.MonkeyPatch,
):
    commands = _capture_snapshot_commands(monkeypatch)
    monkeypatch.setattr(zoo_tools, "resize_image", MagicMock(return_value=b"jpeg"))

    await zoo_tools.zoo_snapshot(session_id="session-id")

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


@pytest.mark.asyncio
async def test_unmatched_frames_consume_the_budget_instead_of_resetting_it(
    monkeypatch: pytest.MonkeyPatch,
):
    """The regression this whole deadline exists for.

    A live modeling session heartbeats every ~10s. Each of those frames used to
    start a fresh per-read timeout, so the 300s bound could never elapse and a
    stalled import blocked its caller for as long as the connection lived.
    """
    clock = _FakeClock()
    monkeypatch.setattr(zoo_tools, "monotonic", clock)
    monkeypatch.setattr(zoo_tools, "MODELING_COMMAND_TIMEOUT", 300.0)

    websocket = AsyncMock()
    timeouts: list[float] = []

    async def recv() -> str:
        clock.advance(10.0)
        return _session_frame()

    websocket.recv.side_effect = recv
    wait_for = asyncio.wait_for

    async def record_timeout(awaitable: Any, timeout: float) -> Any:
        timeouts.append(timeout)
        return await awaitable

    monkeypatch.setattr(zoo_tools.asyncio, "wait_for", record_timeout)

    with pytest.raises(zoo_tools.ZooMCPTimeoutError) as exc_info:
        await zoo_tools._send_modeling_command(
            cast(Any, websocket),
            ModelingCmd(OptionEntityGetIndex(entity_id="entity-id")),
            ResponseEntityGetIndex,
            "entity index",
        )
    monkeypatch.setattr(zoo_tools.asyncio, "wait_for", wait_for)

    # Each read is offered only what is left, so the budget runs out.
    assert timeouts[:4] == [300.0, 300.0, 290.0, 280.0]
    assert len(timeouts) == 31
    message = str(exc_info.value)
    assert "entity index" in message
    assert "300s" in message
    assert "waited 300.0s" in message


@pytest.mark.asyncio
async def test_command_times_out_when_the_engine_never_answers(
    monkeypatch: pytest.MonkeyPatch,
):
    clock = _FakeClock()
    monkeypatch.setattr(zoo_tools, "monotonic", clock)
    monkeypatch.setattr(zoo_tools, "MODELING_COMMAND_TIMEOUT", 300.0)

    websocket = AsyncMock()

    async def recv() -> str:
        clock.advance(300.0)
        raise TimeoutError

    websocket.recv.side_effect = recv

    with pytest.raises(zoo_tools.ZooMCPTimeoutError, match="entity index"):
        await zoo_tools._send_modeling_command(
            cast(Any, websocket),
            ModelingCmd(OptionEntityGetIndex(entity_id="entity-id")),
            ResponseEntityGetIndex,
            "entity index",
        )

    websocket.recv.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_command_send_uses_the_same_deadline_as_its_response():
    websocket = AsyncMock()

    async def send(payload: object) -> None:
        await asyncio.sleep(60)

    websocket.send.side_effect = send

    with pytest.raises(zoo_tools.ZooMCPTimeoutError, match="entity index"):
        await zoo_tools._send_modeling_command(
            cast(Any, websocket),
            ModelingCmd(OptionEntityGetIndex(entity_id="entity-id")),
            ResponseEntityGetIndex,
            "entity index",
            zoo_tools._Deadline(timeout=0.01),
        )

    websocket.recv.assert_not_awaited()


def test_a_timeout_is_a_zoo_mcp_exception_subclass():
    """Callers that only catch ZooMCPException must still see a timeout."""
    assert issubclass(zoo_tools.ZooMCPTimeoutError, ZooMCPException)


@pytest.mark.asyncio
async def test_timeout_aborts_and_evicts_the_session(
    monkeypatch: pytest.MonkeyPatch,
):
    """A session whose engine still owes a response cannot be handed back."""
    clock = _FakeClock()
    monkeypatch.setattr(zoo_tools, "monotonic", clock)
    monkeypatch.setattr(zoo_tools, "MODELING_COMMAND_TIMEOUT", 300.0)

    websocket = AsyncMock()
    websocket.transport = MagicMock()

    async def recv() -> str:
        clock.advance(300.0)
        raise TimeoutError

    websocket.recv.side_effect = recv
    monkeypatch.setattr(
        zoo_tools, "_open_modeling_websocket", AsyncMock(return_value=websocket)
    )

    session_id = await zoo_tools.zoo_start_modeling_session()

    with pytest.raises(zoo_tools.ZooMCPTimeoutError):
        await zoo_tools.zoo_execute_modeling_command(
            ModelingCmd(OptionEntityGetIndex(entity_id="entity-id")),
            ResponseEntityGetIndex,
            "entity index",
            session_id,
        )

    websocket.transport.abort.assert_called_once_with()
    websocket.close.assert_not_awaited()
    assert zoo_tools._modeling_session is None
    with pytest.raises(ZooMCPException, match="Unknown modeling session"):
        await zoo_tools.zoo_execute_modeling_command(
            ModelingCmd(OptionEntityGetIndex(entity_id="entity-id")),
            ResponseEntityGetIndex,
            "entity index",
            session_id,
        )


@pytest.mark.asyncio
async def test_cancellation_aborts_and_evicts_the_session(
    monkeypatch: pytest.MonkeyPatch,
):
    websocket = AsyncMock()
    websocket.transport = MagicMock()
    response_started = asyncio.Event()

    async def recv() -> str:
        response_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    websocket.recv.side_effect = recv
    monkeypatch.setattr(
        zoo_tools, "_open_modeling_websocket", AsyncMock(return_value=websocket)
    )
    session_id = await zoo_tools.zoo_start_modeling_session()

    command = asyncio.create_task(
        zoo_tools.zoo_execute_modeling_command(
            ModelingCmd(OptionEntityGetIndex(entity_id="entity-id")),
            ResponseEntityGetIndex,
            "entity index",
            session_id,
        )
    )
    await asyncio.wait_for(response_started.wait(), timeout=1)
    command.cancel()

    with pytest.raises(asyncio.CancelledError):
        await command

    websocket.transport.abort.assert_called_once_with()
    assert zoo_tools._modeling_session is None


@pytest.mark.asyncio
async def test_connection_close_mid_command_reports_a_dead_session(
    monkeypatch: pytest.MonkeyPatch,
):
    websocket = AsyncMock()
    websocket.recv.side_effect = ConnectionClosedError(None, None)
    monkeypatch.setattr(
        zoo_tools, "_open_modeling_websocket", AsyncMock(return_value=websocket)
    )

    session_id = await zoo_tools.zoo_start_modeling_session()

    with pytest.raises(ZooMCPException, match="no longer connected") as exc_info:
        await zoo_tools.zoo_execute_modeling_command(
            ModelingCmd(OptionEntityGetIndex(entity_id="entity-id")),
            ResponseEntityGetIndex,
            "entity index",
            session_id,
        )

    # A dead socket is not a timeout, so it must not be reported as retryable.
    assert not isinstance(exc_info.value, zoo_tools.ZooMCPTimeoutError)
    assert zoo_tools._modeling_session is None


@pytest.mark.asyncio
async def test_snapshot_shares_one_budget_across_all_its_commands(
    monkeypatch: pytest.MonkeyPatch,
):
    """Otherwise a four-view capture gets fourteen separate budgets."""
    seen: list[object] = []

    async def send_command(
        ws: object,
        command: ModelingCmd,
        expected_response: type,
        description: str,
        deadline: object = None,
    ) -> object:
        seen.append(deadline)
        if expected_response is zoo_tools.ResponseTakeSnapshot:
            return SimpleNamespace(data=TakeSnapshot(contents="anBlZw=="))
        return SimpleNamespace(data=None)

    monkeypatch.setattr(zoo_tools, "_send_modeling_command", send_command)
    monkeypatch.setattr(
        zoo_tools,
        "_modeling_websocket",
        lambda session_id: _async_context(AsyncMock()),
    )
    monkeypatch.setattr(zoo_tools, "resize_image", MagicMock(return_value=b"jpeg"))
    monkeypatch.setattr(
        zoo_tools, "create_image_collage", MagicMock(return_value=b"collage")
    )

    await zoo_tools.zoo_snapshot(
        session_id="session-id",
        views=[_camera_view("front"), _camera_view("right")],
    )

    assert len(seen) > 1
    assert all(isinstance(deadline, zoo_tools._Deadline) for deadline in seen)
    assert len({id(deadline) for deadline in seen}) == 1


def _stub_import_transport(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    websocket = AsyncMock()
    monkeypatch.setattr(
        zoo_tools,
        "_modeling_websocket",
        lambda session_id: _async_context(websocket),
    )
    return websocket


@pytest.mark.parametrize("extension", ["step", "stp", "STEP", "STP"])
def test_step_import_uses_mesh_without_brep_fallback(extension: str) -> None:
    input_format = zoo_tools._get_input_format(extension)

    assert input_format is not None
    assert isinstance(input_format.root, OptionStep)
    assert (
        input_format.root.target_representation == StepImportTargetRepresentation.MESH
    )
    assert input_format.root.split_closed_faces is False


@pytest.mark.asyncio
async def test_import_logs_safe_metadata_and_outcome(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
):
    """Enough to correlate a stalled import server-side, and no file contents."""
    step = tmp_path / "part.step"
    step.write_text("ISO-10303-21; SECRET-CUSTOMER-GEOMETRY")
    _stub_import_transport(monkeypatch)
    monkeypatch.setattr(
        zoo_tools,
        "_await_modeling_response",
        AsyncMock(
            return_value=SimpleNamespace(data=SimpleNamespace(object_id="object-id"))
        ),
    )

    with caplog.at_level("INFO", logger="zoo_mcp"):
        assert await zoo_tools.zoo_import_cad_file("session-id", step) == "object-id"

    records = [r.getMessage() for r in caplog.records if "CAD import" in r.getMessage()]
    assert len(records) == 2
    sent, finished = records
    assert "CAD import sent" in sent
    assert "outcome=awaiting-engine-response" in sent
    assert "CAD import finished" in finished
    assert "outcome=imported" in finished
    for record in records:
        assert "session=session-id" in record
        assert "ext=step" in record
        assert f"bytes={step.stat().st_size}" in record
        assert "request_id=" in record
        assert "elapsed=" in record
        assert "SECRET-CUSTOMER-GEOMETRY" not in record


@pytest.mark.asyncio
async def test_import_timeout_is_logged_and_raised(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
):
    step = tmp_path / "part.step"
    step.write_bytes(b"ISO-10303-21;")
    _stub_import_transport(monkeypatch)

    async def timeout(*args: object, **kwargs: object) -> None:
        raise zoo_tools.ZooMCPTimeoutError("engine went quiet")

    monkeypatch.setattr(zoo_tools, "_await_modeling_response", timeout)

    with (
        caplog.at_level("INFO", logger="zoo_mcp"),
        pytest.raises(zoo_tools.ZooMCPTimeoutError),
    ):
        await zoo_tools.zoo_import_cad_file("session-id", step)

    assert any(
        "CAD import finished" in r.getMessage() and "outcome=timeout" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_import_waits_within_one_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The import's own read loop is the one the production hang sat in."""
    clock = _FakeClock()
    monkeypatch.setattr(zoo_tools, "monotonic", clock)
    monkeypatch.setattr(zoo_tools, "MODELING_COMMAND_TIMEOUT", 120.0)

    step = tmp_path / "part.step"
    step.write_bytes(b"ISO-10303-21;")
    websocket = _stub_import_transport(monkeypatch)

    async def recv() -> str:
        clock.advance(10.0)
        return _session_frame()

    websocket.recv.side_effect = recv

    with pytest.raises(zoo_tools.ZooMCPTimeoutError, match="CAD file import"):
        await zoo_tools.zoo_import_cad_file("session-id", step)

    assert websocket.recv.await_count == 12
