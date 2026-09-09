import asyncio
from dataclasses import asdict
from types import SimpleNamespace

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


class _Trace:
    def __init__(self):
        self.api_call_ids = []


@pytest.mark.asyncio
async def test_kcl_failed_attempt_then_success_keeps_both_ids(monkeypatch):
    monkeypatch.setattr(kcl, "ApiCallTrace", _Trace)
    monkeypatch.setattr(zoo_tools, "_execution_retry_delay", lambda _: 0)
    traces = []
    result = [b"unchanged export bytes"]

    async def execute(*, trace):
        traces.append(trace)
        trace.api_call_ids.append(f"backend-{len(traces)}")
        if len(traces) == 1:
            raise kcl.KclError("retry", True)
        return result

    with (
        capture_api_call_events() as events,
        zoo_tools.capture_execution_retry_events() as retries,
    ):
        actual = await zoo_tools._execute_kcl_with_retries(
            execute, _operation="export_kcl"
        )
    assert actual is result
    assert traces[0] is not traces[1]
    kcl_events = [e for e in events if e.source == "kcl"]
    assert [(e.attempt, e.api_call_id, e.outcome) for e in kcl_events] == [
        (1, "backend-1", "failed"),
        (2, "backend-2", "succeeded"),
    ]
    assert [e.api_call_ids for e in retries] == [("backend-1",), ("backend-2",)]
    assert [e.outcome for e in retries] == ["retry_scheduled", "recovered"]


@pytest.mark.asyncio
async def test_kcl_cancellation_and_pre_response_failure(monkeypatch):
    monkeypatch.setattr(kcl, "ApiCallTrace", _Trace)
    ready = asyncio.Event()

    async def execute(*, trace):
        trace.api_call_ids.append("cancelled-backend")
        ready.set()
        await asyncio.Event().wait()

    with capture_api_call_events() as events:
        task = asyncio.create_task(
            zoo_tools._execute_kcl_with_retries(execute, _operation="execute")
        )
        await ready.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert [(e.api_call_id, e.outcome) for e in events if e.source == "kcl"] == [
        ("cancelled-backend", "cancelled")
    ]

    async def invalid(*, trace):
        raise ValueError("no response")

    with capture_api_call_events() as failed, pytest.raises(ValueError):
        await zoo_tools._execute_kcl_with_retries(invalid, _operation="execute")
    assert all(e.api_call_id is None for e in failed)
    assert all(e.outcome == "failed" for e in failed)
    assert {e.invocation_id for e in events}.isdisjoint(e.invocation_id for e in failed)


@pytest.mark.asyncio
async def test_generic_retries_do_not_inject_a_trace():
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
    monkeypatch.setattr(kcl, "ApiCallTrace", _Trace)

    async def execute(code, *, trace):
        trace.api_call_ids.append("failed-execution")
        raise ValueError("failed")

    monkeypatch.setattr(kcl, "execute_code", execute)
    with capture_api_call_events() as events:
        result = await zoo_tools.zoo_execute_kcl(kcl_code="x = 1")
    assert not result.ok
    assert [(e.api_call_id, e.outcome) for e in events if e.source == "invocation"] == [
        ("failed-execution", "failed")
    ]


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_kcl_measurement_and_export_capture_backend_ids(cube_kcl, tmp_path):
    with capture_api_call_events() as events:
        properties = await zoo_tools.zoo_calculate_kcl_physical_properties(
            None, cube_kcl, "mm", "g", "kg:m3", 1000, "mm2", "mm3"
        )
        output = await zoo_tools.zoo_export_kcl(
            kcl_path=cube_kcl, export_path=tmp_path / "cube.step"
        )
    assert isinstance(properties["volume"], float)
    assert output == tmp_path / "cube.step"
    assert output.read_bytes()
    succeeded = [e for e in events if e.source == "kcl" and e.outcome == "succeeded"]
    assert {e.operation for e in succeeded} == {
        "zoo_calculate_kcl_physical_properties",
        "zoo_export_kcl",
    }
    assert all(e.api_call_id for e in succeeded)
    assert len({e.invocation_id for e in succeeded}) == 2


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
