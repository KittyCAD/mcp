import asyncio
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import kcl
import pytest
from kittycad import AsyncKittyCAD
from kittycad.exceptions import KittyCADServerError
from kittycad.models import ApiCallStatus, FileVolume

from zoo_mcp import ZooMCPException, zoo_tools
from zoo_mcp.api_call_events import (
    api_operation,
    capture_api_call_events,
    record_api_call_event,
)


@pytest.mark.asyncio
async def test_nested_captures_include_child_tasks_once():
    @api_operation
    async def call(api_id):
        await asyncio.sleep(0)
        record_api_call_event("rest", api_id)

    with capture_api_call_events() as outer:
        with capture_api_call_events() as inner:
            await asyncio.gather(call("one"), call("two"))
        await call("three")
    assert [e.api_call_id for e in outer] == ["one", "two", "three"]
    assert inner == outer[:2]
    assert all(e.operation == "call" for e in outer)


@pytest.mark.asyncio
async def test_separate_concurrent_captures_do_not_mix():
    @api_operation
    async def call(api_id):
        await asyncio.sleep(0)
        record_api_call_event("rest", api_id)

    async def capture(api_id):
        with capture_api_call_events() as events:
            await call(api_id)
        return events

    one, two = await asyncio.gather(capture("one"), capture("two"))
    assert [e.api_call_id for e in one] == ["one"]
    assert [e.api_call_id for e in two] == ["two"]


@pytest.mark.asyncio
async def test_logs_without_capture_omit_inputs_and_exception_text(caplog):
    @api_operation
    async def call(secret):
        record_api_call_event("rest", "backend")
        raise ValueError(secret)

    with caplog.at_level("INFO", logger="zoo_mcp"), pytest.raises(ValueError):
        await call("private query and credentials")
    records = [r for r in caplog.records if hasattr(r, "api_call_event")]
    assert len(records) == 1
    assert records[0].api_call_event["api_call_id"] == "backend"
    assert "private query" not in caplog.text
    assert "api_call_id=backend" in caplog.text


class _Outcome:
    def issues(self):
        return []

    def sketch_constraint_report(self):
        return SimpleNamespace(
            fully_constrained=[],
            under_constrained=[],
            over_constrained=[],
            errors=[],
            warnings=[],
            execution_errors=[],
            execution_fatals=[],
            is_complete=True,
            kcl_error=None,
        )

    def render_sketch_png(self, name, *, instance_index=None):
        assert name == "profile"
        assert instance_index is None
        return b"png"


class _Session:
    def __init__(self, api_id="backend", upgrade_id="upgrade"):
        self.api_call_id = api_id
        self.websocket_upgrade_request_id = upgrade_id
        self.outcome = _Outcome()
        self.close = AsyncMock()
        self.export = AsyncMock(return_value=[SimpleNamespace(contents=b"step")])
        self.measure = AsyncMock()


@pytest.mark.asyncio
@pytest.mark.parametrize("from_file", [False, True])
async def test_execute_captures_ids_before_close(monkeypatch, tmp_path, from_file):
    session = _Session()

    async def close():
        session.api_call_id = None
        session.websocket_upgrade_request_id = None

    session.close.side_effect = close
    monkeypatch.setattr(kcl, "mock_execute_code", AsyncMock(return_value=_Outcome()))
    monkeypatch.setattr(kcl, "mock_execute", AsyncMock(return_value=_Outcome()))
    monkeypatch.setattr(kcl, "new_kcl_session_code", AsyncMock(return_value=session))
    monkeypatch.setattr(kcl, "new_kcl_session", AsyncMock(return_value=session))
    path = tmp_path / "main.kcl"
    path.write_text("x = 1")
    with capture_api_call_events() as events:
        result = await zoo_tools.zoo_execute_kcl(
            kcl_path=path if from_file else None,
            kcl_code=None if from_file else "x = 1",
        )
    assert result.ok
    session.close.assert_awaited_once()
    assert session.api_call_id is None
    assert [
        (e.operation, e.api_call_id, e.websocket_upgrade_request_id) for e in events
    ] == [("zoo_execute_kcl", "backend", "upgrade")]


@pytest.mark.asyncio
@pytest.mark.parametrize("from_file", [False, True])
@pytest.mark.parametrize(
    "operation", ["properties", "bounding_box", "export", "constraints", "visualize"]
)
async def test_standalone_tools_capture_ids_from_one_execution(
    monkeypatch, tmp_path, from_file, operation
):
    session = _Session()
    point = SimpleNamespace(x=1, y=2, z=3)
    bbox = SimpleNamespace(get_center=lambda: point, get_dimensions=lambda: point)
    session.measure.return_value = SimpleNamespace(
        get_volume=lambda: 10,
        get_mass=lambda: 20,
        get_surface_area=lambda: 30,
        get_center_of_mass=lambda: point,
        get_bounding_box=lambda: bbox,
    )
    open_code = AsyncMock(return_value=session)
    open_file = AsyncMock(return_value=session)
    monkeypatch.setattr(kcl, "new_kcl_session_code", open_code)
    monkeypatch.setattr(kcl, "new_kcl_session", open_file)
    path = tmp_path / "part.kcl"
    path.write_text("x = 1")
    arguments = {"kcl_path": path} if from_file else {"kcl_code": "x = 1"}
    with capture_api_call_events() as events:
        if operation == "properties":
            result = await zoo_tools.zoo_calculate_kcl_physical_properties(
                arguments.get("kcl_code"),
                arguments.get("kcl_path"),
                "mm",
                "g",
                "kg:m3",
                1000,
                "mm2",
                "mm3",
            )
            assert result["volume"] == 10
        elif operation == "bounding_box":
            result = await zoo_tools.zoo_calculate_bounding_box_kcl("mm", **arguments)
            assert result["center"] == {"x": 1, "y": 2, "z": 3}
        elif operation == "export":
            output = await zoo_tools.zoo_export_kcl(
                export_path=tmp_path / "part.step", **arguments
            )
            assert output.read_bytes() == b"step"
        elif operation == "constraints":
            result = await zoo_tools.zoo_get_sketch_constraint_status(**arguments)
            assert result["kcl_executes_successfully"]
        else:
            assert (
                await zoo_tools.zoo_visualize_sketch(
                    "profile",
                    kcl_code=None if from_file else "x = 1",
                    kcl_path=path if from_file else None,
                )
                == b"png"
            )
    assert open_code.await_count == int(not from_file)
    assert open_file.await_count == int(from_file)
    session.close.assert_awaited_once()
    assert [
        (e.source, e.api_call_id, e.websocket_upgrade_request_id) for e in events
    ] == [("kcl", "backend", "upgrade")]


@pytest.mark.asyncio
async def test_followup_retry_retains_ids_and_closes_each_session(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(zoo_tools, "_execution_retry_delay", lambda _: 0)
    sessions = [_Session("backend-1", "upgrade-1"), _Session("backend-2", "upgrade-2")]
    sessions[0].export.side_effect = kcl.KclError("retry", True)
    monkeypatch.setattr(kcl, "new_kcl_session_code", AsyncMock(side_effect=sessions))
    with (
        capture_api_call_events() as events,
        zoo_tools.capture_execution_retry_events() as retries,
    ):
        output = await zoo_tools.zoo_export_kcl(
            kcl_code="x = 1", export_path=tmp_path / "part.step"
        )
    assert output.read_bytes() == b"step"
    for session in sessions:
        session.close.assert_awaited_once()
    assert [(e.api_call_id, e.websocket_upgrade_request_id) for e in events] == [
        ("backend-1", "upgrade-1"),
        ("backend-2", "upgrade-2"),
    ]
    assert [e.outcome for e in retries] == ["retry_scheduled", "recovered"]


@pytest.mark.asyncio
async def test_execution_retry_does_not_repeat_mock_or_execute_for_inspection(
    monkeypatch,
):
    session = _Session()
    mock = AsyncMock(return_value=_Outcome())
    open_session = AsyncMock(side_effect=[kcl.KclError("retry", True), session])
    monkeypatch.setattr(kcl, "mock_execute_code", mock)
    monkeypatch.setattr(kcl, "new_kcl_session_code", open_session)
    monkeypatch.setattr(zoo_tools, "_execution_retry_delay", lambda _: 0)
    legacy = AsyncMock(side_effect=AssertionError("duplicate execution"))
    monkeypatch.setattr(kcl, "execute_code", legacy)
    session.measure.return_value = SimpleNamespace(get_volume=lambda: 12.5)
    with capture_api_call_events() as events:
        result = await zoo_tools.zoo_execute_kcl(
            kcl_code="x = 1",
            physical_properties_request=zoo_tools.KclPhysicalPropertiesRequest(
                ("volume",)
            ),
        )
    assert result.ok
    mock.assert_awaited_once()
    legacy.assert_not_called()
    assert open_session.await_count == 2
    session.close.assert_awaited_once()
    session.measure.assert_awaited_once()
    assert [e.api_call_id for e in events] == ["backend"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_inspection_failure_retains_ids_and_closes_session(monkeypatch, cancel):
    session = _Session()
    ready = asyncio.Event()

    async def measure(_request):
        ready.set()
        if cancel:
            await asyncio.Event().wait()
        raise ValueError("failed inspection")

    session.measure.side_effect = measure
    monkeypatch.setattr(kcl, "mock_execute_code", AsyncMock(return_value=_Outcome()))
    monkeypatch.setattr(kcl, "new_kcl_session_code", AsyncMock(return_value=session))
    with capture_api_call_events() as events:
        task = asyncio.create_task(
            zoo_tools.zoo_execute_kcl(
                kcl_code="x = 1",
                physical_properties_request=zoo_tools.KclPhysicalPropertiesRequest(
                    ("volume",)
                ),
            )
        )
        await ready.wait()
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await task
            assert result.ok
            assert result.inspection.physical_analysis_status == "failed"
    session.close.assert_awaited_once()
    assert [(e.api_call_id, e.websocket_upgrade_request_id) for e in events] == [
        ("backend", "upgrade")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mock_failure", [False, True])
async def test_failures_before_session_returns_have_no_ids(
    monkeypatch, mock_failure, tmp_path
):
    failure = ValueError("api_call_id=untrusted")
    monkeypatch.setattr(kcl, "mock_execute_code", AsyncMock(side_effect=failure))
    open_session = AsyncMock(side_effect=failure)
    monkeypatch.setattr(kcl, "new_kcl_session_code", open_session)
    with capture_api_call_events() as events:
        if mock_failure:
            assert not (await zoo_tools.zoo_execute_kcl(kcl_code="code")).ok
            open_session.assert_not_called()
        else:
            with pytest.raises(ValueError):
                await zoo_tools.zoo_export_kcl(
                    kcl_code="code", export_path=tmp_path / "part.step"
                )
    assert events == []


@pytest.mark.asyncio
async def test_missing_backend_ids_are_valid_and_emit_no_observation(monkeypatch):
    session = _Session(None, None)
    monkeypatch.setattr(kcl, "mock_execute_code", AsyncMock(return_value=_Outcome()))
    monkeypatch.setattr(kcl, "new_kcl_session_code", AsyncMock(return_value=session))
    with capture_api_call_events() as events:
        assert (await zoo_tools.zoo_execute_kcl(kcl_code="x = 1")).ok
    session.close.assert_awaited_once()
    assert events == []


@pytest.fixture
def client(monkeypatch):
    client = AsyncKittyCAD(token="private-token")
    monkeypatch.setattr(zoo_tools, "AsyncKittyCAD", lambda **kwargs: client)
    return client


@pytest.mark.asyncio
async def test_sdk_failure_raw_fallback_and_pagination_capture_each_response(
    client, httpx_mock
):
    first = {
        "items": [
            {"id": "dataset-1", "name": "dataset", "status": "new-backend-status"}
        ],
        "next_page": "private-page-token",
    }
    for api_id, payload in [
        ("sdk-page", first),
        ("raw-page-1", first),
        ("raw-page-2", {"items": [], "next_page": None}),
    ]:
        httpx_mock.add_response(headers={"X-Api-Call-Id": api_id}, json=payload)
    with capture_api_call_events() as events:
        result = await zoo_tools.zoo_list_org_datasets()
    assert result == [{"id": "dataset-1", "name": "dataset", "description": None}]
    assert [e.api_call_id for e in events] == ["sdk-page", "raw-page-1", "raw-page-2"]
    assert "private-page-token" not in str([asdict(e) for e in events])
    assert len(httpx_mock.get_requests()) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["http", "parse", "network"])
async def test_rest_ids_survive_http_and_parse_failures(client, httpx_mock, failure):
    if failure == "network":
        httpx_mock.add_exception(httpx.ConnectError("unavailable"))
    else:
        httpx_mock.add_response(
            status_code=503 if failure == "http" else 200,
            headers={"X-Api-Call-Id": "response-id"},
            text="invalid JSON",
        )
    with (
        capture_api_call_events() as events,
        pytest.raises(
            (ZooMCPException, KittyCADServerError, ValueError, httpx.ConnectError)
        ),
    ):
        await zoo_tools.zoo_list_org_skills()
    assert [e.api_call_id for e in events] == (
        [] if failure == "network" else ["response-id"]
    )


@pytest.mark.asyncio
async def test_polling_captures_request_ids_and_original_operation_id(
    client, monkeypatch, httpx_mock, cube_stl
):
    operation_id = "d4154735-9cf8-4bc4-98a4-7c7af077388f"
    monkeypatch.setattr(zoo_tools, "FILE_API_CALL_POLL_INTERVAL", 0)

    async def create(**kwargs):
        await client.get_http_client().post("https://example.test/create")
        return FileVolume.model_construct(
            id=operation_id, status=ApiCallStatus.QUEUED, volume=None
        )

    async def poll(**kwargs):
        response = await client.get_http_client().get("https://example.test/poll")
        response.raise_for_status()
        return SimpleNamespace(
            root=SimpleNamespace(
                type="file_volume",
                model_dump=lambda **_: {
                    "id": operation_id,
                    "status": ApiCallStatus.COMPLETED,
                    "volume": 42,
                },
            )
        )

    monkeypatch.setattr(client.file, "create_file_volume", create)
    monkeypatch.setattr(client.api_calls, "get_async_operation", poll)
    httpx_mock.add_response(headers={"X-Api-Call-Id": "create-request"})
    httpx_mock.add_response(headers={"X-Api-Call-Id": "poll-request"})
    with capture_api_call_events() as events:
        assert await zoo_tools.zoo_calculate_volume(cube_stl, "cm3") == 42
    assert [(e.source, e.api_call_id) for e in events] == [
        ("rest", "create-request"),
        ("file_operation", operation_id),
        ("rest", "poll-request"),
    ]


@pytest.mark.asyncio
async def test_request_without_response_does_not_reuse_an_earlier_id(
    client, httpx_mock
):
    httpx_mock.add_response(
        headers={"X-Api-Call-Id": "first-request"},
        json={"items": [], "next_page": "next"},
    )
    httpx_mock.add_exception(httpx.ConnectError("no response"))
    with capture_api_call_events() as events, pytest.raises(httpx.ConnectError):
        await zoo_tools.zoo_list_org_datasets()
    assert [e.api_call_id for e in events] == ["first-request"]


@pytest.mark.live
@pytest.mark.xdist_group(name="engine")
@pytest.mark.asyncio
async def test_live_kcl_execution_measurement_and_export_capture_backend_ids(
    cube_kcl, tmp_path
):
    with capture_api_call_events() as events:
        execution = await zoo_tools.zoo_execute_kcl(kcl_path=cube_kcl)
        properties = await zoo_tools.zoo_calculate_kcl_physical_properties(
            None, cube_kcl, "mm", "g", "kg:m3", 1000, "mm2", "mm3"
        )
        output = await zoo_tools.zoo_export_kcl(
            kcl_path=cube_kcl, export_path=tmp_path / "cube.step"
        )
    assert execution.ok
    assert isinstance(properties["volume"], float)
    assert output.read_bytes()
    assert {e.operation for e in events} == {
        "zoo_execute_kcl",
        "zoo_calculate_kcl_physical_properties",
        "zoo_export_kcl",
    }
    assert len(events) == 3
    assert all(e.api_call_id and e.websocket_upgrade_request_id for e in events)
