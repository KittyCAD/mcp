import asyncio
import base64
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import kcl
import pytest
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
from mcp.types import CallToolResult, ImageContent, TextContent
from PIL import Image

from tests.test_execution_preflight import Outcome, Session
from zoo_mcp.credentials import ZooCredentials, use_credentials
from zoo_mcp.server import mcp
from zoo_mcp.zoo_tools import zoo_evaluate_kcl


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "removed", ["calculate_kcl_physical_properties", "calculate_bounding_box_kcl"]
)
async def test_measurements_use_evaluate_kcl(removed):
    names = {tool.name for tool in await mcp.list_tools()}
    assert "evaluate_kcl" in names and removed not in names
    with pytest.raises(ToolError, match=f"Unknown tool: {removed}"):
        await mcp.call_tool(removed, {})


@pytest.fixture
def bindings(monkeypatch):
    buffer = io.BytesIO()
    Image.new("RGB", (10, 10), color="red").save(buffer, format="JPEG")
    session = Session(Outcome())
    session.snapshots = AsyncMock(
        side_effect=lambda _fmt, views, **_kwargs: [
            list(buffer.getvalue()) for _ in views
        ]
    )
    session.measure = AsyncMock(wraps=session.measure)
    session.close = AsyncMock(wraps=session.close)
    mock = AsyncMock(return_value=Outcome())
    opened = AsyncMock(return_value=session)
    monkeypatch.setattr(kcl, "mock_execute_code", mock)
    monkeypatch.setattr(kcl, "mock_execute", mock)
    monkeypatch.setattr(kcl, "new_kcl_session_code", opened)
    monkeypatch.setattr(kcl, "new_kcl_session", opened)
    with use_credentials(ZooCredentials("test-token")):
        yield SimpleNamespace(session=session, mock=mock, opened=opened)


async def call(arguments):
    result = await mcp.call_tool("evaluate_kcl", arguments)
    assert isinstance(result, CallToolResult)
    assert result.structured_content is not None
    return (result, result.structured_content)


@pytest.mark.asyncio
async def test_all_outputs_share_execution_and_have_wire_safe_content(
    bindings, tmp_path
):
    (response, result) = await call(
        {
            "kcl_code": "x = 1",
            "snapshot_request": {"views": ["front", "right"]},
            "physical_properties_request": {"properties": ["volume", "surface_area"]},
        }
    )
    assert result["ok"]
    assert result["real_execution"]["status"] == "succeeded"
    assert result["inspection"]["physical_properties"]["volume"] == {
        "value": 12.5,
        "unit": "mm3",
    }
    assert result["inspection"]["completed_snapshot_views"] == ["front", "right"]
    assert "rendered_snapshot" not in result["inspection"]
    assert isinstance(response.content[0], TextContent)
    assert json.loads(response.content[0].text) == result
    assert isinstance(response.content[1], ImageContent)
    Image.open(io.BytesIO(base64.b64decode(response.content[1].data))).verify()
    bindings.mock.assert_awaited_once()
    bindings.opened.assert_awaited_once()
    bindings.session.snapshots.assert_awaited_once()
    bindings.session.measure.assert_awaited_once()
    bindings.session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_execution_only_does_not_inspect(bindings):
    (_, result) = await call({"kcl_code": "x = 1"})
    assert result["ok"]
    assert result["inspection"]["physical_analysis_status"] == "not_requested"
    assert result["inspection"]["rendered_snapshots_status"] == "not_requested"
    bindings.session.snapshots.assert_not_awaited()
    bindings.session.measure.assert_not_awaited()
    bindings.session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_preflight_failure_has_no_hosted_work(bindings, tmp_path):
    bindings.mock.return_value = Outcome("error")
    (response, result) = await call({"kcl_code": "x = 1"})
    assert response.is_error
    assert result["real_execution"]["status"] == "not_run"
    bindings.opened.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_outputs_do_not_retry_geometry(bindings, tmp_path):
    bindings.session.snapshots.side_effect = RuntimeError("render failed")
    bindings.session.measure.side_effect = RuntimeError("measure failed")
    (_, result) = await call(
        {
            "kcl_code": "x = 1",
            "snapshot_request": {},
            "physical_properties_request": {"properties": ["volume"]},
        }
    )
    assert result["ok"]
    assert result["inspection"]["rendered_snapshots_status"] == "failed"
    assert result["inspection"]["physical_analysis_status"] == "failed"
    bindings.opened.assert_awaited_once()
    bindings.session.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["snapshots", "measure"])
async def test_cancellation_closes_session(bindings, tmp_path, stage):
    getattr(bindings.session, stage).side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await call(
            {
                "kcl_code": "x = 1",
                "snapshot_request": {},
                "physical_properties_request": {"properties": ["volume"]},
            }
        )
    bindings.opened.assert_awaited_once()
    bindings.session.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        {"kcl_path": "/also.kcl"},
        {"snapshot_request": {"views": []}},
        {"snapshot_request": {"views": ["unknown"]}},
        {"snapshot_request": {"views": [{"up": [1, 2]}]}},
        {"snapshot_request": {"padding": -1}},
        {"physical_properties_request": {"properties": ["mass"]}},
        {
            "physical_properties_request": {
                "properties": ["surface_area"],
                "unit_area": "wrong",
            }
        },
        {
            "physical_properties_request": {
                "properties": ["volume"],
                "unit_volume": "wrong",
            }
        },
    ],
)
async def test_invalid_requests_do_not_execute(bindings, extra):
    with pytest.raises(ToolError) as error:
        await call({"kcl_code": "x = 1", **extra})
    assert type(error.value) is ToolError
    bindings.opened.assert_not_awaited()
    bindings.mock.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments, message",
    [
        ({}, "Provide exactly one of kcl_code or kcl_path"),
        (
            {"kcl_code": "x = 1", "kcl_path": "/also.kcl"},
            "Provide exactly one of kcl_code or kcl_path",
        ),
        (
            {
                "kcl_code": "x = 1",
                "physical_properties_request": {"properties": ["mass"]},
            },
            "density is required when requesting mass",
        ),
        (
            {
                "kcl_code": "x = 1",
                "physical_properties_request": {"properties": ["mass"], "density": 1},
            },
            "unit_density is required when requesting mass",
        ),
        (
            {
                "kcl_code": "x = 1",
                "snapshot_request": {"views": ["front"], "zoom": False},
            },
            "Named snapshot views require zoom-to-fit",
        ),
        (
            {
                "kcl_code": "x = 1",
                "snapshot_request": {
                    "views": [
                        {"up": [0, 0, 1], "vantage": [0, 0, 1], "center": [0, 0, 0]}
                    ]
                },
            },
            "Camera direction and up must be nonzero and nonparallel",
        ),
    ],
)
async def test_invalid_requests_expose_corrective_messages(
    bindings, arguments, message
):
    with pytest.raises(ToolError, match=message) as error:
        await mcp.call_tool("evaluate_kcl", arguments)
    assert type(error.value) is ToolError
    bindings.mock.assert_not_awaited()
    bindings.opened.assert_not_awaited()


@pytest.mark.asyncio
async def test_unexpected_errors_remain_masked(monkeypatch):
    monkeypatch.setattr(
        "zoo_mcp.server.zoo_evaluate_kcl",
        AsyncMock(side_effect=ValueError("internal failure details")),
    )
    with pytest.raises(UnexpectedToolError) as error:
        await mcp.call_tool("evaluate_kcl", {"kcl_code": "x = 1"})
    assert str(error.value) == "Error executing tool evaluate_kcl"


@pytest.mark.asyncio
async def test_captured_project_is_shared(bindings, tmp_path):
    source = tmp_path / "main.kcl"
    source.write_text("x = 1")
    result = await zoo_evaluate_kcl(kcl_path=source)
    assert result.ok
    assert bindings.mock.await_args.args[0] == bindings.opened.await_args.args[0]
    bindings.opened.assert_awaited_once()
