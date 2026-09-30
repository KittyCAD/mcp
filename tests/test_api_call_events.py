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
    api_invocation,
    capture_api_call_events,
    record_api_call_event,
)


@pytest.mark.asyncio
async def test_nested_captures_and_same_task_helpers_share_invocation():
    @api_invocation
    async def helper():
        record_api_call_event("rest", "observed", "backend")

    @api_invocation
    async def outer_call():
        with capture_api_call_events() as inner:
            await helper()
        return inner

    with capture_api_call_events() as outer:
        inner = await outer_call()
    assert len(inner) == 1
    assert inner[0] is outer[0]
    assert len(outer) == 2
    assert {e.operation for e in outer} == {"outer_call"}
    assert len({e.invocation_id for e in outer}) == 1


@pytest.mark.asyncio
async def test_concurrent_calls_inherit_capture_but_have_separate_invocations():
    @api_invocation
    async def call(api_id):
        await asyncio.sleep(0)
        record_api_call_event("rest", "observed", api_id)

    @api_invocation
    async def parent():
        await asyncio.gather(call("one"), call("two"))

    with capture_api_call_events() as events:
        await parent()
    invocations = {
        api_id: {e.invocation_id for e in events if e.api_call_id == api_id}
        for api_id in ("one", "two")
    }
    assert all(len(ids) == 1 for ids in invocations.values())
    assert invocations["one"].isdisjoint(invocations["two"])
    assert all(e.api_call_id is None for e in events if e.operation == "parent")


@pytest.mark.asyncio
async def test_separate_concurrent_captures_do_not_mix():
    @api_invocation
    async def call(api_id):
        await asyncio.sleep(0)
        record_api_call_event("rest", "observed", api_id)

    async def capture(api_id):
        with capture_api_call_events() as events:
            await call(api_id)
        return events

    one, two = await asyncio.gather(capture("one"), capture("two"))
    assert {e.api_call_id for e in one} == {"one"}
    assert {e.api_call_id for e in two} == {"two"}


@pytest.mark.asyncio
async def test_logs_without_capture_omit_inputs(caplog):
    @api_invocation
    async def call(secret):
        record_api_call_event("rest", "observed", "backend")
        raise ValueError(secret)

    with caplog.at_level("INFO", logger="zoo_mcp"), pytest.raises(ValueError):
        await call("private query and credentials")
    records = [r for r in caplog.records if hasattr(r, "api_call_event")]
    assert records[-1].api_call_event["outcome"] == "failed"
    assert records[-1].api_call_event["api_call_id"] == "backend"
    assert "private query" not in caplog.text
    assert "api_call_id=backend" in caplog.text


class _Outcome:
    def __init__(self, report=None):
        self.report_value = report or SimpleNamespace(
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

    def issues(self):
        return []

    def sketch_constraint_report(self):
        return self.report_value

    def render_sketch_png(self, name):
        assert name == "profile"
        return b"png"


class _Session:
    def __init__(self, api_id="backend", upgrade_id="upgrade", outcome=None):
        self.api_call_id = api_id
        self.websocket_upgrade_request_id = upgrade_id
        self.outcome = outcome or _Outcome()
        self.close = AsyncMock()
        self.export = AsyncMock(return_value=[SimpleNamespace(contents=b"step")])
        self.measure = AsyncMock()


def test_installed_kcl_provides_session_correlation_properties():
    assert hasattr(kcl.KclSession, "api_call_id")
    assert hasattr(kcl.KclSession, "websocket_upgrade_request_id")


@pytest.mark.asyncio
@pytest.mark.parametrize("from_file", [False, True])
async def test_execute_kcl_captures_engine_ids_before_session_closes(
    monkeypatch, tmp_path, from_file
):
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
        (event.source, event.api_call_id, event.websocket_upgrade_request_id)
        for event in events
    ] == [
        ("kcl", "backend", "upgrade"),
        ("invocation", "backend", "upgrade"),
    ]
    assert all(event.operation == "zoo_execute_kcl" for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("from_file", [False, True])
@pytest.mark.parametrize(
    "operation", ["properties", "bounding_box", "export", "constraints", "visualize"]
)
async def test_standalone_tools_use_one_session(
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
            assert result == {
                "volume": 10,
                "mass": 20,
                "surface_area": 30,
                "center_of_mass": {"x": 1, "y": 2, "z": 3},
                "bounding_box": {
                    "center": {"x": 1, "y": 2, "z": 3},
                    "dimensions": {"x": 1, "y": 2, "z": 3},
                },
            }
        elif operation == "bounding_box":
            assert await zoo_tools.zoo_calculate_bounding_box_kcl(
                "mm", **arguments
            ) == {
                "center": {"x": 1, "y": 2, "z": 3},
                "dimensions": {"x": 1, "y": 2, "z": 3},
            }
        elif operation == "export":
            output = await zoo_tools.zoo_export_kcl(
                export_path=tmp_path / "part.step", **arguments
            )
            assert output.read_bytes() == b"step"
            session.export.assert_awaited_once_with(kcl.FileExportFormat.Step)
        elif operation == "constraints":
            result = await zoo_tools.zoo_get_sketch_constraint_status(**arguments)
            assert result == {
                "fully_constrained": [],
                "under_constrained": [],
                "over_constrained": [],
                "errors": [],
                "total_sketches": 0,
                "kcl_executes_successfully": True,
                "kcl_error": None,
            }
        else:
            assert (
                await zoo_tools.zoo_visualize_sketch("profile", **arguments) == b"png"
            )
    assert open_code.await_count == int(not from_file)
    assert open_file.await_count == int(from_file)
    session.close.assert_awaited_once()
    assert session.measure.await_count == int(
        operation in ("properties", "bounding_box")
    )
    assert {e.api_call_id for e in events} == {"backend"}
    assert {e.websocket_upgrade_request_id for e in events} == {"upgrade"}
    assert events[-1].source == "invocation" and events[-1].outcome == "succeeded"


@pytest.mark.asyncio
async def test_cancellation_during_inspection_retains_successful_attempt(monkeypatch):
    session = _Session()
    ready = asyncio.Event()

    async def measure(_request):
        ready.set()
        await asyncio.Event().wait()

    session.measure.side_effect = measure
    monkeypatch.setattr(kcl, "mock_execute_code", AsyncMock(return_value=_Outcome()))
    monkeypatch.setattr(
        kcl,
        "new_kcl_session_code",
        AsyncMock(side_effect=[kcl.KclError("retry", True), session]),
    )
    monkeypatch.setattr(zoo_tools, "_execution_retry_delay", lambda _: 0)
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
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    session.close.assert_awaited_once()
    summary = events[-1]
    assert (summary.source, summary.attempt, summary.outcome) == (
        "invocation",
        2,
        "cancelled",
    )
    assert (summary.api_call_id, summary.websocket_upgrade_request_id) == (
        "backend",
        "upgrade",
    )


@pytest.mark.asyncio
async def test_kcl_followup_retry_keeps_both_ids_and_closes_each_attempt(monkeypatch):
    monkeypatch.setattr(zoo_tools, "_execution_retry_delay", lambda _: 0)
    sessions = [_Session("backend-1", "upgrade-1"), _Session("backend-2", "upgrade-2")]
    sessions[0].export.side_effect = kcl.KclError("retry", True)
    open_session = AsyncMock(side_effect=sessions)
    monkeypatch.setattr(kcl, "new_kcl_session_code", open_session)

    async def export(session):
        return await session.export(kcl.FileExportFormat.Step)

    with (
        capture_api_call_events() as events,
        zoo_tools.capture_execution_retry_events() as retries,
    ):
        actual = await zoo_tools._execute_kcl_with_retries(
            export, "code", None, _operation="export_kcl"
        )
    assert actual is sessions[1].export.return_value
    assert open_session.await_count == 2
    for session in sessions:
        session.close.assert_awaited_once()
    kcl_events = [e for e in events if e.source == "kcl"]
    assert [
        (e.attempt, e.api_call_id, e.websocket_upgrade_request_id) for e in kcl_events
    ] == [
        (1, "backend-1", "upgrade-1"),
        (2, "backend-2", "upgrade-2"),
    ]
    assert [e.api_call_ids for e in retries] == [("backend-1",), ("backend-2",)]
    assert [e.outcome for e in retries] == ["retry_scheduled", "recovered"]
    assert [
        (e.attempt, e.api_call_id, e.websocket_upgrade_request_id)
        for e in events
        if e.source == "invocation"
    ] == [(1, "backend-1", "upgrade-1"), (2, "backend-2", "upgrade-2")]


@pytest.mark.asyncio
@pytest.mark.parametrize("inspections", [False, True])
async def test_execution_session_retry_does_not_repeat_mock_or_execution(
    monkeypatch, inspections
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
    with (
        capture_api_call_events() as events,
        zoo_tools.capture_execution_retry_events() as retries,
    ):
        result = await zoo_tools.zoo_execute_kcl(
            kcl_code="x = 1",
            physical_properties_request=zoo_tools.KclPhysicalPropertiesRequest(
                ("volume",)
            )
            if inspections
            else None,
        )
    assert result.ok
    mock.assert_awaited_once()
    legacy.assert_not_called()
    assert open_session.await_count == 2
    session.close.assert_awaited_once()
    assert session.measure.await_count == int(inspections)
    assert [(e.attempt, e.api_call_ids) for e in retries] == [
        (1, None),
        (2, ("backend",)),
    ]
    assert [
        (e.attempt, e.api_call_id, e.websocket_upgrade_request_id)
        for e in events
        if e.source == "kcl"
    ] == [(1, None, None), (2, "backend", "upgrade")]
    summaries = [e for e in events if e.source == "invocation"]
    assert [(e.attempt, e.api_call_id) for e in summaries] == [
        (1, None),
        (2, "backend"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_kcl_followup_failure_closes_session_and_retains_ids(
    monkeypatch, caplog, cancel
):
    session = _Session()
    monkeypatch.setattr(kcl, "new_kcl_session_code", AsyncMock(return_value=session))
    ready = asyncio.Event()

    async def followup(session):
        ready.set()
        if cancel:
            await asyncio.Event().wait()
        raise ValueError("private follow-up error")

    with capture_api_call_events() as events, caplog.at_level("INFO", logger="zoo_mcp"):
        task = asyncio.create_task(
            zoo_tools._execute_kcl_with_retries(
                followup, "code", None, _operation="execute"
            )
        )
        await ready.wait()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else ValueError):
            await task
    session.close.assert_awaited_once()
    summary = [e for e in events if e.source == "invocation"]
    assert len(summary) == 1
    assert (summary[0].api_call_id, summary[0].websocket_upgrade_request_id) == (
        "backend",
        "upgrade",
    )
    assert summary[0].outcome == ("cancelled" if cancel else "failed")
    record = [r.api_call_event for r in caplog.records if hasattr(r, "api_call_event")][
        -1
    ]
    assert record == asdict(summary[0])
    assert "api_call_id=backend" in caplog.text
    assert "websocket_upgrade_request_id=upgrade" in caplog.text
    assert "private follow-up error" not in caplog.text


@pytest.mark.asyncio
async def test_pre_session_failure_does_not_parse_ids_from_error(monkeypatch):
    monkeypatch.setattr(
        kcl,
        "new_kcl_session_code",
        AsyncMock(side_effect=ValueError("api_call_id=untrusted")),
    )
    with capture_api_call_events() as events, pytest.raises(ValueError):
        await zoo_tools._execute_kcl_with_retries(
            AsyncMock(), "code", None, _operation="execute"
        )
    assert events
    assert all(
        e.api_call_id is None and e.websocket_upgrade_request_id is None for e in events
    )
    assert all(e.outcome == "failed" for e in events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing_property", [None, "api_call_id", "websocket_upgrade_request_id"]
)
async def test_session_properties_are_required_but_none_is_valid(
    monkeypatch, missing_property
):
    session = _Session(None, None)
    if missing_property:
        delattr(session, missing_property)
    monkeypatch.setattr(kcl, "new_kcl_session_code", AsyncMock(return_value=session))
    use_session = AsyncMock(return_value=42)
    with capture_api_call_events() as events:
        if missing_property:
            with pytest.raises(AttributeError, match=missing_property):
                await zoo_tools._execute_kcl_with_retries(
                    use_session, "code", None, _operation="execute"
                )
            use_session.assert_not_called()
        else:
            assert (
                await zoo_tools._execute_kcl_with_retries(
                    use_session, "code", None, _operation="execute"
                )
                == 42
            )
    session.close.assert_awaited_once()
    assert all(
        e.api_call_id is None and e.websocket_upgrade_request_id is None for e in events
    )


@pytest.mark.asyncio
async def test_mock_preflight_failure_records_no_backend_ids(monkeypatch):
    monkeypatch.setattr(
        kcl, "mock_execute_code", AsyncMock(side_effect=ValueError("invalid"))
    )
    open_session = AsyncMock()
    monkeypatch.setattr(kcl, "new_kcl_session_code", open_session)
    with capture_api_call_events() as events:
        result = await zoo_tools.zoo_execute_kcl(kcl_code="code")
    assert not result.ok
    open_session.assert_not_called()
    assert len(events) == 1
    assert events[0].source == "invocation" and events[0].outcome == "failed"
    assert (
        events[0].api_call_id is None and events[0].websocket_upgrade_request_id is None
    )


@pytest.mark.asyncio
async def test_invocation_preserves_distinct_upgrade_ids_and_attempts():
    @api_invocation
    async def call():
        for attempt, upgrade in ((1, "first"), (2, "second"), (3, "second")):
            with zoo_tools.api_call_attempt(attempt):
                record_api_call_event(
                    "kcl", "observed", "backend", websocket_upgrade_request_id=upgrade
                )

    with capture_api_call_events() as events:
        await call()
    assert [
        (e.attempt, e.api_call_id, e.websocket_upgrade_request_id)
        for e in events
        if e.source == "invocation"
    ] == [(1, "backend", "first"), (2, "backend", "second"), (3, "backend", "second")]


@pytest.mark.asyncio
async def test_generic_retries_do_not_inject_session_arguments():
    async def arbitrary(value):
        return value

    with zoo_tools.capture_execution_retry_events() as retries:
        assert await zoo_tools._execute_with_retries(arbitrary, 42) == 42
    assert retries[0].api_call_ids is None


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
    rest = [e for e in events if e.source == "rest"]
    assert [e.api_call_id for e in rest] == ["sdk-page", "raw-page-1", "raw-page-2"]
    assert len({e.invocation_id for e in events}) == 1
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
    assert events[-1].outcome == "failed"
    assert {e.api_call_id for e in events} == (
        {None} if failure == "network" else {"response-id"}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_poll_requests_and_original_operation_have_separate_ids(
    client, monkeypatch, httpx_mock, cube_stl, failure
):
    operation_id = "d4154735-9cf8-4bc4-98a4-7c7af077388f"
    monkeypatch.setattr(zoo_tools, "FILE_API_CALL_POLL_INTERVAL", 0)

    async def create(**kwargs):
        await client.get_http_client().post("https://example.test/create")
        return FileVolume.model_construct(
            id=operation_id, status=ApiCallStatus.QUEUED, volume=None
        )

    polls = 0

    async def poll(**kwargs):
        nonlocal polls
        polls += 1
        response = await client.get_http_client().get("https://example.test/poll")
        response.raise_for_status()
        return SimpleNamespace(
            root=SimpleNamespace(
                type="file_volume",
                model_dump=lambda **_: {
                    "id": operation_id,
                    "status": ApiCallStatus.COMPLETED
                    if polls == 2
                    else ApiCallStatus.IN_PROGRESS,
                    "volume": 42,
                },
            )
        )

    monkeypatch.setattr(client.file, "create_file_volume", create)
    monkeypatch.setattr(client.api_calls, "get_async_operation", poll)
    httpx_mock.add_response(headers={"X-Api-Call-Id": "create-request"})
    httpx_mock.add_response(headers={"X-Api-Call-Id": "poll-request-1"})
    httpx_mock.add_response(
        headers={"X-Api-Call-Id": "poll-request-2"}, status_code=503 if failure else 200
    )
    with capture_api_call_events() as events:
        if failure:
            with pytest.raises(httpx.HTTPStatusError):
                await zoo_tools.zoo_calculate_volume(cube_stl, "cm3")
        else:
            assert await zoo_tools.zoo_calculate_volume(cube_stl, "cm3") == 42
    assert [
        (e.api_call_id, e.async_operation_id) for e in events if e.source == "rest"
    ] == [
        ("create-request", None),
        ("poll-request-1", operation_id),
        ("poll-request-2", operation_id),
    ]
    assert [
        (e.api_call_id, e.async_operation_id)
        for e in events
        if e.source == "file_operation"
    ] == [(operation_id, operation_id)]
    assert any(
        e.api_call_id == operation_id
        and e.outcome == ("failed" if failure else "succeeded")
        for e in events
        if e.source == "invocation"
    )


@pytest.mark.asyncio
async def test_successful_sdk_pagination_captures_every_request(client, httpx_mock):
    httpx_mock.add_response(
        headers={"X-Api-Call-Id": "page-1"}, json={"items": [], "next_page": "next"}
    )
    httpx_mock.add_response(
        headers={"X-Api-Call-Id": "page-2"}, json={"items": [], "next_page": None}
    )
    with capture_api_call_events() as events:
        assert await zoo_tools.zoo_list_org_datasets() == []
    assert [e.api_call_id for e in events if e.source == "rest"] == ["page-1", "page-2"]


@pytest.mark.asyncio
async def test_execute_failure_value_has_failed_invocation(monkeypatch):
    class FailedOutcome(_Outcome):
        def sketch_constraint_report(self):
            raise ValueError("failed")

    session = _Session(outcome=FailedOutcome())
    monkeypatch.setattr(kcl, "mock_execute_code", AsyncMock(return_value=_Outcome()))
    monkeypatch.setattr(kcl, "new_kcl_session_code", AsyncMock(return_value=session))
    with capture_api_call_events() as events:
        result = await zoo_tools.zoo_execute_kcl(kcl_code="x = 1")
    assert not result.ok
    session.close.assert_awaited_once()
    assert [
        (e.api_call_id, e.websocket_upgrade_request_id, e.outcome)
        for e in events
        if e.source == "invocation"
    ] == [("backend", "upgrade", "failed")]


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
    assert output == tmp_path / "cube.step"
    assert output.read_bytes()
    succeeded = [e for e in events if e.source == "kcl" and e.outcome == "observed"]
    assert {e.operation for e in succeeded} == {
        "zoo_execute_kcl",
        "zoo_calculate_kcl_physical_properties",
        "zoo_export_kcl",
    }
    assert all(e.api_call_id and e.websocket_upgrade_request_id for e in succeeded)
    assert len({e.invocation_id for e in succeeded}) == 3


@pytest.mark.asyncio
async def test_request_without_response_does_not_inherit_an_earlier_id(
    client, httpx_mock
):
    httpx_mock.add_response(
        headers={"X-Api-Call-Id": "first-request"},
        json={"items": [], "next_page": "next"},
    )
    httpx_mock.add_exception(httpx.ConnectError("no response"))
    with capture_api_call_events() as events, pytest.raises(httpx.ConnectError):
        await zoo_tools.zoo_list_org_datasets()
    assert [(e.api_call_id, e.outcome) for e in events if e.source == "rest"] == [
        ("first-request", "observed"),
        (None, "failed"),
    ]
