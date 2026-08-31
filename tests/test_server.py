import asyncio
import base64
import io
import json
import os
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import kcl
import pytest
import pytest_asyncio
from kittycad import AsyncKittyCAD
from kittycad.exceptions import KittyCADClientError
from kittycad.models import (
    ApiCallStatus,
    FaceGetCenter,
    FaceGetGradient,
    FaceGetPosition,
    FileVolume,
    OrgDataset,
    Point3d,
)
from kittycad.models.async_api_call_output import OptionFileMass, OptionFileVolume
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ImageContent, TextContent
from PIL import Image as PILImage

import zoo_mcp
import zoo_mcp.zoo_tools
from zoo_mcp.kcl_docs import KCLDocs
from zoo_mcp.kcl_samples import KCLSamples, SampleMetadata
from zoo_mcp.server import mcp


def _async_items(items: Sequence[Any]) -> AsyncIterator[Any]:
    async def iterate() -> AsyncIterator[Any]:
        for item in items:
            yield item

    return iterate()


def _async_error(error: Exception) -> AsyncIterator[Any]:
    async def iterate() -> AsyncIterator[Any]:
        raise error
        yield  # pragma: no cover - makes this an async iterator

    return iterate()


@pytest.fixture
def async_kittycad_client(monkeypatch: pytest.MonkeyPatch) -> AsyncKittyCAD:
    client = AsyncKittyCAD(token="test-token")
    monkeypatch.setattr(zoo_mcp.zoo_tools, "AsyncKittyCAD", lambda **kwargs: client)
    return client


def _meta_result(response: Sequence[Any] | dict[str, Any]) -> Any:
    """Extract response[1]["result"] with proper typing for ty."""
    assert isinstance(response, Sequence)
    meta = response[1]
    assert isinstance(meta, dict)
    return cast(dict[str, Any], meta)["result"]


def _structured_result(response: Sequence[Any] | dict[str, Any]) -> dict[str, Any]:
    """Extract structured content with proper typing for ty."""
    assert isinstance(response, Sequence)
    result = response[1]
    assert isinstance(result, dict)
    return cast(dict[str, Any], result)


def _content_list(response: Sequence[Any] | dict[str, Any]) -> list[Any]:
    """Extract response[0] as a typed list for ty."""
    assert isinstance(response, Sequence)
    content = response[0]
    assert isinstance(content, list)
    return cast(list[Any], content)


@pytest_asyncio.fixture
async def populated_modeling_session(cube_kcl: str):
    session_id = _meta_result(
        await mcp.call_tool("start_modeling_session", arguments={})
    )
    artifact_graph_path: Path | None = None
    failure: BaseException | None = None
    try:
        response = await mcp.call_tool(
            "exec_kcl_project",
            arguments={"kcl_path": cube_kcl, "session_id": session_id},
        )
        artifact_graph_path = Path(_meta_result(response))
        yield session_id
    except BaseException as error:
        failure = error
        raise
    finally:
        if artifact_graph_path is not None:
            artifact_graph_path.unlink(missing_ok=True)
        try:
            await mcp.call_tool(
                "stop_modeling_session", arguments={"session_id": session_id}
            )
        except ToolError:
            if failure is None:
                raise


@pytest.mark.asyncio
async def test_calculate_center_of_mass(cube_stl: str):
    response = await mcp.call_tool(
        "calculate_center_of_mass",
        arguments={
            "input_file": cube_stl,
            "unit_length": "mm",
        },
    )
    result = _meta_result(response)
    assert isinstance(result, dict)
    assert "x" in result and "y" in result and "z" in result
    assert result["x"] == pytest.approx(5.0)
    assert result["y"] == pytest.approx(5.0)
    assert result["z"] == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_calculate_center_of_mass_error(cube_stl: str):
    response = await mcp.call_tool(
        "calculate_center_of_mass",
        arguments={
            "input_file": cube_stl,
            "unit_length": "asdf",
        },
    )
    result = _meta_result(response)
    assert "not a valid UnitLength" in result


@pytest.mark.asyncio
async def test_calculate_mass(cube_stl: str):
    response = await mcp.call_tool(
        "calculate_mass",
        arguments={
            "input_file": cube_stl,
            "unit_mass": "g",
            "unit_density": "kg:m3",
            "density": 1000.0,
        },
    )
    result = _meta_result(response)
    assert isinstance(result, float)
    assert result == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_calculate_mass_error(cube_stl: str):
    response = await mcp.call_tool(
        "calculate_mass",
        arguments={
            "input_file": cube_stl,
            "unit_mass": "asdf",
            "unit_density": "kg:m3",
            "density": 1000.0,
        },
    )
    result = _meta_result(response)
    assert "not a valid UnitMass" in result


@pytest.mark.asyncio
async def test_calculate_surface_area(cube_stl: str):
    response = await mcp.call_tool(
        "calculate_surface_area", arguments={"input_file": cube_stl, "unit_area": "mm2"}
    )
    result = _meta_result(response)
    assert isinstance(result, float)
    assert result == pytest.approx(600.0)


@pytest.mark.asyncio
async def test_calculate_surface_area_error(cube_stl: str):
    response = await mcp.call_tool(
        "calculate_surface_area",
        arguments={
            "input_file": cube_stl,
            "unit_area": "asdf",
        },
    )
    result = _meta_result(response)
    assert "not a valid UnitArea" in result


@pytest.mark.asyncio
async def test_calculate_volume(cube_stl: str):
    response = await mcp.call_tool(
        "calculate_volume", arguments={"input_file": cube_stl, "unit_volume": "cm3"}
    )
    result = _meta_result(response)
    assert isinstance(result, float)
    assert result == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_calculate_volume_polls_async_operation(
    monkeypatch: pytest.MonkeyPatch,
    cube_stl: str,
    async_kittycad_client: AsyncKittyCAD,
):
    operation_id = "d4154735-9cf8-4bc4-98a4-7c7af077388f"
    monkeypatch.setattr(zoo_mcp.zoo_tools, "FILE_API_CALL_POLL_INTERVAL", 0.0)
    monkeypatch.setattr(
        async_kittycad_client.file,
        "create_file_volume",
        AsyncMock(
            return_value=FileVolume.model_construct(
                id=operation_id,
                status=ApiCallStatus.UPLOADED,
                volume=None,
            )
        ),
    )
    get_async_operation = AsyncMock(
        side_effect=[
            SimpleNamespace(
                root=OptionFileVolume.model_construct(
                    id=operation_id,
                    status=ApiCallStatus.IN_PROGRESS,
                    volume=None,
                )
            ),
            SimpleNamespace(
                root=OptionFileVolume.model_construct(
                    id=operation_id,
                    status=ApiCallStatus.COMPLETED,
                    volume=42.0,
                )
            ),
        ]
    )
    monkeypatch.setattr(
        async_kittycad_client.api_calls,
        "get_async_operation",
        get_async_operation,
    )

    response = await mcp.call_tool(
        "calculate_volume",
        arguments={"input_file": cube_stl, "unit_volume": "cm3"},
    )

    assert _meta_result(response) == pytest.approx(42.0)
    assert get_async_operation.await_count == 2
    get_async_operation.assert_awaited_with(id=operation_id)


@pytest.mark.asyncio
async def test_pending_async_file_operation_does_not_starve_other_tools(
    monkeypatch: pytest.MonkeyPatch,
    cube_stl: str,
    async_kittycad_client: AsyncKittyCAD,
):
    operation_id = "533575ce-740a-4101-9094-a59a26d377c1"
    entered = asyncio.Event()
    release = asyncio.Event()
    monkeypatch.setattr(
        async_kittycad_client.file,
        "create_file_volume",
        AsyncMock(
            return_value=FileVolume.model_construct(
                id=operation_id,
                status=ApiCallStatus.UPLOADED,
                volume=None,
            )
        ),
    )

    async def delayed_async_result(*args: Any, **kwargs: Any) -> SimpleNamespace:
        entered.set()
        await release.wait()
        return SimpleNamespace(
            root=OptionFileVolume.model_construct(
                id=operation_id,
                status=ApiCallStatus.COMPLETED,
                volume=42.0,
            )
        )

    monkeypatch.setattr(
        async_kittycad_client.api_calls,
        "get_async_operation",
        delayed_async_result,
    )

    pending = asyncio.create_task(
        mcp.call_tool(
            "calculate_volume",
            arguments={"input_file": cube_stl, "unit_volume": "cm3"},
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)

    other = await asyncio.wait_for(
        mcp.call_tool("get_modeling_sessions", arguments={}), timeout=1
    )
    assert _meta_result(other) == []
    assert not pending.done()

    release.set()
    assert _meta_result(await pending) == pytest.approx(42.0)


@pytest.mark.asyncio
async def test_calculate_volume_surfaces_async_worker_failure(
    monkeypatch: pytest.MonkeyPatch,
    cube_stl: str,
    async_kittycad_client: AsyncKittyCAD,
):
    operation_id = "06f632ae-4da3-45fa-ad6e-d24566aba7e3"
    monkeypatch.setattr(
        async_kittycad_client.file,
        "create_file_volume",
        AsyncMock(
            return_value=FileVolume.model_construct(
                id=operation_id,
                status=ApiCallStatus.QUEUED,
                volume=None,
            )
        ),
    )
    monkeypatch.setattr(
        async_kittycad_client.api_calls,
        "get_async_operation",
        AsyncMock(
            return_value=SimpleNamespace(
                root=OptionFileVolume.model_construct(
                    id=operation_id,
                    status=ApiCallStatus.FAILED,
                    error="unsupported topology",
                    volume=None,
                )
            )
        ),
    )

    response = await mcp.call_tool(
        "calculate_volume",
        arguments={"input_file": cube_stl, "unit_volume": "cm3"},
    )

    result = _meta_result(response)
    assert "operation 06f632ae-4da3-45fa-ad6e-d24566aba7e3 failed" in result
    assert "unsupported topology" in result


@pytest.mark.asyncio
async def test_calculate_volume_rejects_wrong_async_result_variant(
    monkeypatch: pytest.MonkeyPatch,
    cube_stl: str,
    async_kittycad_client: AsyncKittyCAD,
):
    operation_id = "35bf3830-9f6d-49fc-b2d0-bc74a18c1680"
    monkeypatch.setattr(
        async_kittycad_client.file,
        "create_file_volume",
        AsyncMock(
            return_value=FileVolume.model_construct(
                id=operation_id,
                status=ApiCallStatus.UPLOADED,
                volume=None,
            )
        ),
    )
    monkeypatch.setattr(
        async_kittycad_client.api_calls,
        "get_async_operation",
        AsyncMock(
            return_value=SimpleNamespace(
                root=OptionFileMass.model_construct(
                    id=operation_id,
                    status=ApiCallStatus.COMPLETED,
                    mass=42.0,
                )
            )
        ),
    )

    response = await mcp.call_tool(
        "calculate_volume",
        arguments={"input_file": cube_stl, "unit_volume": "cm3"},
    )

    assert "returned file_mass" in _meta_result(response)


@pytest.mark.asyncio
async def test_calculate_volume_bounds_async_polling(
    monkeypatch: pytest.MonkeyPatch,
    cube_stl: str,
    async_kittycad_client: AsyncKittyCAD,
):
    operation_id = "d1c29e7e-0f6b-460b-b871-3d976510ff20"
    monkeypatch.setattr(zoo_mcp.zoo_tools, "FILE_API_CALL_TIMEOUT", 0.0)
    monkeypatch.setattr(
        async_kittycad_client.file,
        "create_file_volume",
        AsyncMock(
            return_value=FileVolume.model_construct(
                id=operation_id,
                status=ApiCallStatus.IN_PROGRESS,
                volume=None,
            )
        ),
    )
    get_async_operation = AsyncMock()
    monkeypatch.setattr(
        async_kittycad_client.api_calls,
        "get_async_operation",
        get_async_operation,
    )

    response = await mcp.call_tool(
        "calculate_volume",
        arguments={"input_file": cube_stl, "unit_volume": "cm3"},
    )

    assert "Timed out waiting for calculate volume operation" in _meta_result(response)
    get_async_operation.assert_not_awaited()


@pytest.mark.asyncio
async def test_slow_file_request_does_not_starve_other_tools(
    monkeypatch: pytest.MonkeyPatch,
    cube_stl: str,
    async_kittycad_client: AsyncKittyCAD,
):
    """A pending REST request must yield instead of freezing MCP dispatch."""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_volume(*args: Any, **kwargs: Any) -> FileVolume:
        entered.set()
        await release.wait()
        return FileVolume.model_construct(volume=1.0)

    monkeypatch.setattr(
        async_kittycad_client.file,
        "create_file_volume",
        slow_volume,
    )

    stalled = asyncio.create_task(
        mcp.call_tool(
            "calculate_volume",
            arguments={"input_file": cube_stl, "unit_volume": "cm3"},
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)

    other = await asyncio.wait_for(
        mcp.call_tool("get_modeling_sessions", arguments={}), timeout=1
    )
    assert _meta_result(other) == []
    assert not stalled.done()

    release.set()
    assert _meta_result(await stalled) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_calculate_volume_error(cube_stl: str):
    response = await mcp.call_tool(
        "calculate_volume", arguments={"input_file": cube_stl, "unit_volume": "asdf"}
    )
    result = _meta_result(response)
    assert "not a valid UnitVolume" in result


@pytest.mark.asyncio
async def test_calculate_volume_uppercase_step_extension(cube2_step_uppercase: str):
    """Test that CAD files with uppercase extensions (e.g., .STEP) are handled correctly."""
    response = await mcp.call_tool(
        "calculate_volume",
        arguments={"input_file": cube2_step_uppercase, "unit_volume": "cm3"},
    )
    result = _meta_result(response)
    assert isinstance(result, float)
    # The cube2.STEP file should have a valid volume
    assert result > 0


@pytest.mark.asyncio
async def test_calculate_volume_stp_extension(cube_stp: str):
    """Test that CAD files with .stp extension (alias for .step) are handled correctly."""
    response = await mcp.call_tool(
        "calculate_volume",
        arguments={"input_file": cube_stp, "unit_volume": "cm3"},
    )
    result = _meta_result(response)
    assert isinstance(result, float)
    # The .stp file should have a valid volume
    assert result > 0


@pytest.mark.asyncio
async def test_calculate_cad_physical_properties(cube_stl: str):
    response = await mcp.call_tool(
        "calculate_cad_physical_properties",
        arguments={
            "input_file": cube_stl,
            "unit_length": "mm",
            "unit_mass": "g",
            "unit_density": "kg:m3",
            "density": 1000.0,
            "unit_area": "mm2",
            "unit_volume": "cm3",
        },
    )
    result = _meta_result(response)
    assert isinstance(result, dict)
    assert result["volume"] == pytest.approx(1.0)
    assert result["mass"] == pytest.approx(1.0)
    assert result["surface_area"] == pytest.approx(600.0)
    com = result["center_of_mass"]
    assert com["x"] == pytest.approx(5.0)
    assert com["y"] == pytest.approx(5.0)
    assert com["z"] == pytest.approx(5.0)
    bbox = result["bounding_box"]
    assert "center" in bbox and "dimensions" in bbox
    assert bbox["dimensions"]["x"] == pytest.approx(10.0, abs=0.1)
    assert bbox["dimensions"]["y"] == pytest.approx(10.0, abs=0.1)
    assert bbox["dimensions"]["z"] == pytest.approx(10.0, abs=0.1)


@pytest.mark.asyncio
async def test_calculate_cad_physical_properties_error(cube_stl: str):
    response = await mcp.call_tool(
        "calculate_cad_physical_properties",
        arguments={
            "input_file": cube_stl,
            "unit_length": "mm",
            "unit_mass": "bad",
            "unit_density": "kg:m3",
            "density": 1000.0,
            "unit_area": "mm2",
            "unit_volume": "cm3",
        },
    )
    result = _meta_result(response)
    assert "error calculating physical properties" in result


@pytest.mark.asyncio
async def test_calculate_kcl_physical_properties(cube_kcl: str):
    response = await mcp.call_tool(
        "calculate_kcl_physical_properties",
        arguments={
            "kcl_code": None,
            "kcl_path": cube_kcl,
            "unit_length": "mm",
            "unit_mass": "g",
            "unit_density": "kg:m3",
            "density": 1000.0,
            "unit_area": "mm2",
            "unit_volume": "cm3",
        },
    )
    result = _meta_result(response)
    assert isinstance(result, dict)
    # 10mm cube = 1 cm³
    assert result["volume"] == pytest.approx(1.0, abs=1e-3)
    assert result["mass"] == pytest.approx(1.0, abs=1e-3)
    assert result["surface_area"] == pytest.approx(600.0, abs=1e-1)
    com = result["center_of_mass"]
    assert com["x"] == pytest.approx(5.0, abs=1e-1)
    assert com["y"] == pytest.approx(5.0, abs=1e-1)
    assert com["z"] == pytest.approx(5.0, abs=1e-1)
    bbox = result["bounding_box"]
    assert "center" in bbox and "dimensions" in bbox
    assert com == pytest.approx(bbox["center"], abs=0.1)
    assert bbox["dimensions"]["x"] == pytest.approx(10.0, abs=0.1)
    assert bbox["dimensions"]["y"] == pytest.approx(10.0, abs=0.1)
    assert bbox["dimensions"]["z"] == pytest.approx(10.0, abs=0.1)


@pytest.mark.asyncio
async def test_calculate_kcl_physical_properties_error():
    response = await mcp.call_tool(
        "calculate_kcl_physical_properties",
        arguments={
            "kcl_code": None,
            "kcl_path": None,
            "unit_length": "mm",
            "unit_mass": "g",
            "unit_density": "kg:m3",
            "density": 1000.0,
            "unit_area": "mm2",
            "unit_volume": "cm3",
        },
    )
    result = _meta_result(response)
    assert "error calculating physical properties" in result


@pytest.mark.asyncio
async def test_calculate_kcl_physical_properties_invalid_unit(cube_kcl: str):
    response = await mcp.call_tool(
        "calculate_kcl_physical_properties",
        arguments={
            "kcl_code": None,
            "kcl_path": cube_kcl,
            "unit_length": "mm",
            "unit_mass": "g",
            "unit_density": "kg:m3",
            "density": 1000.0,
            "unit_area": "bad",
            "unit_volume": "cm3",
        },
    )
    result = _meta_result(response)
    assert "Invalid unit_area" in result


@pytest.mark.asyncio
async def test_calculate_bounding_box_kcl(cube_kcl: str):
    response = await mcp.call_tool(
        "calculate_bounding_box_kcl",
        arguments={
            "kcl_code": None,
            "kcl_path": cube_kcl,
            "unit_length": "mm",
        },
    )
    result = _meta_result(response)
    assert isinstance(result, dict)
    assert "center" in result
    assert "dimensions" in result
    center = result["center"]
    dimensions = result["dimensions"]
    assert "x" in center and "y" in center and "z" in center
    assert "x" in dimensions and "y" in dimensions and "z" in dimensions
    # 10mm cube: dimensions should be ~10 in each direction
    assert dimensions["x"] == pytest.approx(10.0, abs=0.1)
    assert dimensions["y"] == pytest.approx(10.0, abs=0.1)
    assert dimensions["z"] == pytest.approx(10.0, abs=0.1)


@pytest.mark.asyncio
async def test_calculate_bounding_box_kcl_error():
    response = await mcp.call_tool(
        "calculate_bounding_box_kcl",
        arguments={
            "kcl_code": None,
            "kcl_path": None,
            "unit_length": "mm",
        },
    )
    result = _meta_result(response)
    assert "error calculating bounding box" in result


@pytest.mark.asyncio
async def test_calculate_bounding_box_cad(cube_stl: str):
    response = await mcp.call_tool(
        "calculate_bounding_box_cad",
        arguments={
            "input_file": cube_stl,
        },
    )
    result = _meta_result(response)
    assert isinstance(result, dict)
    assert "center" in result
    assert "dimensions" in result
    center = result["center"]
    dimensions = result["dimensions"]
    assert "x" in center and "y" in center and "z" in center
    assert "x" in dimensions and "y" in dimensions and "z" in dimensions
    assert dimensions["x"] == pytest.approx(10.0, abs=0.1)
    assert dimensions["y"] == pytest.approx(10.0, abs=0.1)
    assert dimensions["z"] == pytest.approx(10.0, abs=0.1)
    assert center["x"] == pytest.approx(5.0, abs=0.1)
    assert center["y"] == pytest.approx(5.0, abs=0.1)
    assert center["z"] == pytest.approx(5.0, abs=0.1)


@pytest.mark.asyncio
async def test_calculate_bounding_box_cad_error(empty_step: str):
    response = await mcp.call_tool(
        "calculate_bounding_box_cad",
        arguments={
            "input_file": empty_step,
        },
    )
    result = _meta_result(response)
    assert "error calculating the bounding box" in result


@pytest.mark.asyncio
async def test_calculate_bounding_box_cad_step(cube_stp: str):
    """Test bounding box calculation for STEP files with uppercase extension."""
    response = await mcp.call_tool(
        "calculate_bounding_box_cad",
        arguments={
            "input_file": cube_stp,
        },
    )
    result = _meta_result(response)
    assert isinstance(result, dict)
    center = result["center"]
    dimensions = result["dimensions"]
    assert "x" in center and "y" in center and "z" in center
    assert "x" in dimensions and "y" in dimensions and "z" in dimensions
    assert dimensions["x"] == pytest.approx(10.0, abs=0.1)
    assert dimensions["y"] == pytest.approx(10.0, abs=0.1)
    assert dimensions["z"] == pytest.approx(10.0, abs=0.1)
    assert center["x"] == pytest.approx(5.0, abs=0.1)
    assert center["y"] == pytest.approx(5.0, abs=0.1)
    assert center["z"] == pytest.approx(-5.0, abs=0.1)


@pytest.mark.asyncio
async def test_convert_cad_file(cube_stl: str):
    response = await mcp.call_tool(
        "convert_cad_file",
        arguments={
            "input_file": cube_stl,
            "export_path": None,
            "export_format": "obj",
        },
    )
    result = _meta_result(response)
    assert Path(result).exists()
    assert Path(result).stat().st_size != 0


@pytest.mark.asyncio
async def test_convert_cad_file_error(empty_step: str):
    response = await mcp.call_tool(
        "convert_cad_file",
        arguments={
            "input_file": empty_step,
            "export_path": None,
            "export_format": "asdf",
        },
    )
    result = _meta_result(response)
    assert "error converting the CAD" in result


@pytest.mark.asyncio
async def test_execute_kcl(cube_kcl: str):
    response = await mcp.call_tool(
        "execute_kcl",
        arguments={
            "kcl_code": None,
            "kcl_path": cube_kcl,
        },
    )
    result = _meta_result(response)
    assert result["ok"] is True
    assert "KCL code executed successfully" in result["message"]


@pytest.mark.asyncio
async def test_execute_kcl_error():
    response = await mcp.call_tool(
        "execute_kcl",
        arguments={
            "kcl_code": "asdf = asdf",
            "kcl_path": None,
        },
    )
    result = _meta_result(response)
    assert result["ok"] is False
    assert "Failed to execute KCL code" in result["message"]


@pytest.mark.asyncio
async def test_exec_kcl_project_extracts_artifact_graph():
    raw_socket = AsyncMock()

    async def recv():
        request = json.loads(raw_socket.send.call_args.args[0])
        return json.dumps(
            {
                "success": True,
                "request_id": request["request_id"],
                "resp": {
                    "type": "exec_kcl_project",
                    "data": {
                        "result": {
                            "ok": {
                                "artifact_graph": {
                                    "map": {"artifact-id": {"type": "solid2d"}},
                                    "item_count": 1,
                                }
                            }
                        }
                    },
                },
            }
        )

    raw_socket.recv.side_effect = recv
    result = await zoo_mcp.zoo_tools._exec_kcl_project(
        cast(Any, raw_socket),
        "main.kcl",
        [{"path": "main.kcl", "contents": list(b"sketch = startSketchOn(XY)")}],
    )

    try:
        assert json.loads(result.read_text()) == {
            "map": {"artifact-id": {"type": "solid2d"}},
            "item_count": 1,
        }
    finally:
        result.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_exec_kcl_project_tool(monkeypatch, tmp_path):
    artifact_graph = {
        "map": {"artifact-id": {"type": "solid2d"}},
        "item_count": 1,
    }
    artifact_graph_path = tmp_path / "artifact-graph.json"
    artifact_graph_path.write_text(json.dumps(artifact_graph))
    mock = AsyncMock(return_value=artifact_graph_path)
    monkeypatch.setattr("zoo_mcp.server.zoo_exec_kcl_project", mock)

    response = await mcp.call_tool(
        "exec_kcl_project",
        arguments={
            "kcl_code": "sketch = startSketchOn(XY)",
            "kcl_path": None,
            "session_id": "session-id",
        },
    )

    assert _meta_result(response) == str(artifact_graph_path)
    mock.assert_awaited_once_with(
        kcl_code="sketch = startSketchOn(XY)",
        kcl_path=None,
        session_id="session-id",
    )


@pytest.mark.asyncio
async def test_execute_kcl_surfaces_warning_issue(warning_kcl: str):
    """A non-fatal warning (disjoint union) succeeds but is reported."""
    response = await mcp.call_tool(
        "execute_kcl",
        arguments={"kcl_code": None, "kcl_path": warning_kcl},
    )
    result = _meta_result(response)
    assert result["ok"] is True
    assert "KCL code execution completed with the following issues" in result["message"]
    assert "Warnings:" in result["message"]
    assert "no overlap" in result["message"]
    # A pure warning must not be mislabelled as an error/fatal.
    assert "Errors:" not in result["message"]
    assert "Fatal issues:" not in result["message"]


@pytest.mark.asyncio
async def test_execute_kcl_surfaces_error_issues(error_kcl: str):
    """Non-fatal errors (labelled `extrude` arg) succeed but are reported."""
    response = await mcp.call_tool(
        "execute_kcl",
        arguments={"kcl_code": None, "kcl_path": error_kcl},
    )
    result = _meta_result(response)
    assert result["ok"] is True
    assert "KCL code execution completed with the following issues" in result["message"]
    assert "Errors:" in result["message"]


@pytest.mark.asyncio
async def test_execute_kcl_reports_fatal_error(fatal_error_kcl: str):
    """A fatal error aborts execution and is reported as a failure."""
    response = await mcp.call_tool(
        "execute_kcl",
        arguments={"kcl_code": None, "kcl_path": fatal_error_kcl},
    )
    result = _meta_result(response)
    assert result["ok"] is False
    assert "Failed to execute KCL code" in result["message"]


class _FakeIssue:
    """Stand-in for kcl.CompilationIssue."""

    def __init__(self, *, severity: str) -> None:
        self.severity = severity

    def is_warning(self) -> bool:
        return self.severity == "warning"

    def is_err(self) -> bool:
        # kcl reports fatal issues as errors too.
        return self.severity in ("error", "fatal")

    def is_fatal(self) -> bool:
        return self.severity == "fatal"


class _FakeOutcome:
    """Stand-in for kcl.ExecOutcome."""

    def __init__(self, issues: list[_FakeIssue]) -> None:
        self._issues = issues

    def issues(self) -> list[_FakeIssue]:
        return self._issues

    def report(self, issue: _FakeIssue) -> str:
        return f"{issue.severity} report"


def test_format_execution_issues_groups_by_severity():
    outcome = _FakeOutcome(
        [
            _FakeIssue(severity="warning"),
            _FakeIssue(severity="error"),
            _FakeIssue(severity="fatal"),
            _FakeIssue(severity="warning"),
        ]
    )
    issues = zoo_mcp.zoo_tools._format_execution_issues(cast(Any, outcome))
    assert issues == {
        "warning": ["warning report", "warning report"],
        "error": ["error report"],
        "fatal": ["fatal report"],
    }


def test_format_execution_issues_empty_when_no_issues():
    assert zoo_mcp.zoo_tools._format_execution_issues(cast(Any, _FakeOutcome([]))) == {}


@pytest.mark.asyncio
async def test_execute_kcl_surfaces_all_issue_severities(monkeypatch):
    outcome = _FakeOutcome(
        [
            _FakeIssue(severity="warning"),
            _FakeIssue(severity="error"),
            _FakeIssue(severity="fatal"),
        ]
    )

    async def fake_execute_code(code: str):
        return outcome

    monkeypatch.setattr(zoo_mcp.zoo_tools.kcl, "execute_code", fake_execute_code)

    result = await zoo_mcp.zoo_tools.zoo_execute_kcl(kcl_code="anything")
    assert isinstance(result, zoo_mcp.zoo_tools.ResultZooExecuteKclLocal)
    assert result.ok is True
    assert result.message.startswith(
        "KCL code execution completed with the following issues:"
    )
    assert "Fatal issues:\n\nfatal report" in result.message
    assert "Errors:\n\nerror report" in result.message
    assert "Warnings:\n\nwarning report" in result.message
    # Severities are rendered fatal -> error -> warning.
    assert (
        result.message.index("Fatal issues:")
        < result.message.index("Errors:")
        < result.message.index("Warnings:")
    )


class _RetryableError(Exception):
    def __init__(self, message: str, retryable: bool) -> None:
        super().__init__(message)
        self._retryable = retryable

    def is_retryable(self) -> bool:
        return self._retryable


@pytest.fixture
def retry_delays(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(zoo_mcp.zoo_tools.asyncio, "sleep", record_sleep)
    monkeypatch.setattr(zoo_mcp.zoo_tools.random, "uniform", lambda _a, _b: 0.0)
    return delays


@pytest.mark.asyncio
async def test_execute_with_retries_succeeds_first_try():
    calls = 0
    events: list[zoo_mcp.zoo_tools.ExecutionRetryEvent] = []

    async def fn(value: str) -> str:
        nonlocal calls
        calls += 1
        return value

    with zoo_mcp.zoo_tools.capture_execution_retry_events(events.append):
        result = await zoo_mcp.zoo_tools._execute_with_retries(
            fn,
            "ok",
            _operation="test_operation",
        )

    assert result == "ok"
    assert calls == 1
    assert len(events) == 1
    assert events[0].operation == "test_operation"
    assert events[0].outcome == "succeeded"
    assert events[0].attempt == 1
    assert events[0].max_attempts == zoo_mcp.zoo_tools.MAX_EXECUTION_ATTEMPTS
    assert events[0].error_family is None
    assert events[0].fresh_connection is True


@pytest.mark.asyncio
async def test_execute_with_retries_retries_then_succeeds(
    retry_delays: list[float],
):
    calls = 0
    events: list[zoo_mcp.zoo_tools.ExecutionRetryEvent] = []

    async def fn() -> str:
        nonlocal calls
        calls += 1
        if calls < zoo_mcp.zoo_tools.MAX_EXECUTION_ATTEMPTS:
            raise _RetryableError("hangup", retryable=True)
        return "recovered"

    with zoo_mcp.zoo_tools.capture_execution_retry_events(events.append):
        result = await zoo_mcp.zoo_tools._execute_with_retries(fn)

    assert result == "recovered"
    assert calls == zoo_mcp.zoo_tools.MAX_EXECUTION_ATTEMPTS
    assert retry_delays == [0.25, 0.5]
    assert [event.outcome for event in events] == [
        "retry_scheduled",
        "retry_scheduled",
        "recovered",
    ]
    assert [event.attempt for event in events] == [1, 2, 3]


@pytest.mark.asyncio
async def test_execute_with_retries_exhausts_attempts(
    retry_delays: list[float],
):
    calls = 0
    events: list[zoo_mcp.zoo_tools.ExecutionRetryEvent] = []

    async def fn() -> str:
        nonlocal calls
        calls += 1
        raise _RetryableError("hangup", retryable=True)

    with (
        zoo_mcp.zoo_tools.capture_execution_retry_events(events.append),
        pytest.raises(_RetryableError),
    ):
        await zoo_mcp.zoo_tools._execute_with_retries(fn)

    assert calls == zoo_mcp.zoo_tools.MAX_EXECUTION_ATTEMPTS
    assert retry_delays == [0.25, 0.5]
    assert [event.outcome for event in events] == [
        "retry_scheduled",
        "retry_scheduled",
        "exhausted",
    ]


@pytest.mark.asyncio
async def test_execute_with_retries_does_not_retry_non_retryable():
    calls = 0
    events: list[zoo_mcp.zoo_tools.ExecutionRetryEvent] = []

    async def fn() -> str:
        nonlocal calls
        calls += 1
        raise _RetryableError("bad code", retryable=False)

    with (
        zoo_mcp.zoo_tools.capture_execution_retry_events(events.append),
        pytest.raises(_RetryableError),
    ):
        await zoo_mcp.zoo_tools._execute_with_retries(fn)

    assert calls == 1
    assert len(events) == 1
    assert events[0].outcome == "terminal_non_retryable"
    assert events[0].error_family == "_RetryableError"


@pytest.mark.asyncio
async def test_execute_with_retries_does_not_retry_plain_exception():
    calls = 0
    events: list[zoo_mcp.zoo_tools.ExecutionRetryEvent] = []

    async def fn() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("no is_retryable method")

    with (
        zoo_mcp.zoo_tools.capture_execution_retry_events(events.append),
        pytest.raises(ValueError),
    ):
        await zoo_mcp.zoo_tools._execute_with_retries(fn)

    assert calls == 1
    assert len(events) == 1
    assert events[0].outcome == "terminal_non_retryable"
    assert events[0].error_family == "ValueError"


@pytest.mark.asyncio
async def test_execute_with_retries_sanitizes_engine_hangup(
    retry_delays: list[float],
):
    secret = "customer-code-and-filename"
    events: list[zoo_mcp.zoo_tools.ExecutionRetryEvent] = []

    async def fn() -> None:
        raise kcl.KclError(f"KCL EngineHangup error\n{secret}", True)

    with (
        zoo_mcp.zoo_tools.capture_execution_retry_events(events.append),
        pytest.raises(kcl.KclError),
    ):
        await zoo_mcp.zoo_tools._execute_with_retries(fn)

    assert retry_delays == [0.25, 0.5]
    assert [event.error_family for event in events] == [
        "EngineHangup",
        "EngineHangup",
        "EngineHangup",
    ]
    assert secret not in repr(events)


@pytest.mark.asyncio
async def test_execute_with_retries_supports_async_event_handler():
    events: list[zoo_mcp.zoo_tools.ExecutionRetryEvent] = []

    async def handler(event: zoo_mcp.zoo_tools.ExecutionRetryEvent) -> None:
        events.append(event)

    async def fn() -> str:
        return "ok"

    with zoo_mcp.zoo_tools.capture_execution_retry_events(handler):
        assert await zoo_mcp.zoo_tools._execute_with_retries(fn) == "ok"

    assert [event.outcome for event in events] == ["succeeded"]


@pytest.mark.asyncio
async def test_execute_with_retries_ignores_event_handler_failure():
    def handler(_event: zoo_mcp.zoo_tools.ExecutionRetryEvent) -> None:
        raise RuntimeError("telemetry unavailable")

    async def fn() -> str:
        return "ok"

    with zoo_mcp.zoo_tools.capture_execution_retry_events(handler):
        assert await zoo_mcp.zoo_tools._execute_with_retries(fn) == "ok"


@pytest.mark.asyncio
async def test_execute_with_retries_enforces_total_time_budget(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[zoo_mcp.zoo_tools.ExecutionRetryEvent] = []
    never_finishes = asyncio.Event()
    monkeypatch.setattr(
        zoo_mcp.zoo_tools,
        "EXECUTION_RETRY_TOTAL_TIMEOUT_SECONDS",
        0.01,
    )

    async def fn() -> None:
        await never_finishes.wait()

    with (
        zoo_mcp.zoo_tools.capture_execution_retry_events(events.append),
        pytest.raises(zoo_mcp.ZooMCPTimeoutError),
    ):
        await zoo_mcp.zoo_tools._execute_with_retries(fn)

    assert len(events) == 1
    assert events[0].outcome == "exhausted"
    assert events[0].error_family == "RetryBudgetExceeded"


@pytest.mark.asyncio
async def test_export_kcl(cube_kcl: str):
    response = await mcp.call_tool(
        "export_kcl",
        arguments={
            "kcl_code": None,
            "kcl_path": cube_kcl,
            "export_path": None,
            "export_format": "step",
        },
    )
    result = _meta_result(response)
    assert Path(result).exists()
    assert Path(result).stat().st_size != 0


@pytest.mark.asyncio
async def test_export_kcl_error():
    response = await mcp.call_tool(
        "export_kcl",
        arguments={
            "kcl_code": "asdf",
            "kcl_path": None,
            "export_path": None,
            "export_format": "step",
        },
    )
    result = _meta_result(response)
    assert "error exporting the CAD" in result


@pytest.mark.asyncio
async def test_format_kcl_path_success(cube_kcl: str):
    response = await mcp.call_tool(
        "format_kcl",
        arguments={
            "kcl_code": None,
            "kcl_path": cube_kcl,
        },
    )
    result = _meta_result(response)
    assert "Successfully formatted KCL code at" in result


@pytest.mark.asyncio
async def test_format_kcl_project_success(kcl_project: str):
    response = await mcp.call_tool(
        "format_kcl",
        arguments={
            "kcl_code": None,
            "kcl_path": kcl_project,
        },
    )
    result = _meta_result(response)
    assert "Successfully formatted KCL code at" in result


@pytest.mark.asyncio
async def test_format_kcl_str_success(cube_kcl: str):
    response = await mcp.call_tool(
        "format_kcl",
        arguments={
            "kcl_code": Path(cube_kcl).read_text(),
            "kcl_path": None,
        },
    )
    result = _meta_result(response)
    assert "|>" in result


@pytest.mark.asyncio
async def test_format_kcl_error(cube_stl: str):
    response = await mcp.call_tool(
        "format_kcl",
        arguments={
            "kcl_code": None,
            "kcl_path": cube_stl,
        },
    )
    result = _meta_result(response)
    assert "error formatting the KCL" in result


@pytest.mark.asyncio
async def test_lint_and_fix_kcl_str_success():
    code = """c = startSketchOn(XY)
  |> circle(center = [0, 0], radius = 1)
  |> circle(center = [5, 0], radius = 1)
  |> circle(center = [0,  5], radius = 1)
  |> circle(center = [5, 5], radius = 1)
"""
    response = await mcp.call_tool(
        "lint_and_fix_kcl",
        arguments={
            "kcl_code": code,
            "kcl_path": None,
        },
    )
    fixed_code, _ = _meta_result(response)
    assert fixed_code != code


@pytest.mark.asyncio
async def test_lint_and_fix_kcl_path_success(kcl_project: str):
    response = await mcp.call_tool(
        "lint_and_fix_kcl",
        arguments={
            "kcl_code": None,
            "kcl_path": kcl_project,
        },
    )
    fixed_code_msg, _ = _meta_result(response)
    assert "Successfully linted and fixed KCL code" in fixed_code_msg


@pytest.mark.asyncio
async def test_lint_and_fix_kcl_error(cube_stl: str):
    response = await mcp.call_tool(
        "lint_and_fix_kcl",
        arguments={
            "kcl_code": None,
            "kcl_path": cube_stl,
        },
    )
    fixed_code_msg, _ = _meta_result(response)
    assert "error linting and fixing" in fixed_code_msg


@pytest.mark.asyncio
async def test_get_sketch_constraint_status_fully_constrained_code():
    kcl_code = """
sketch(on = YZ) {
  line1 = line(start = [var 2mm, var 8mm], end = [var 5mm, var 7mm])
  line1.start.at[0] == 2
  line1.start.at[1] == 8
  line1.end.at[0] == 5
  line1.end.at[1] == 7
}
"""
    response = await mcp.call_tool(
        "get_sketch_constraint_status",
        arguments={"kcl_code": kcl_code, "kcl_path": None},
    )
    result = _meta_result(response)
    assert isinstance(result, dict)
    assert len(result["fully_constrained"]) == 1
    assert len(result["under_constrained"]) == 0
    assert len(result["over_constrained"]) == 0
    assert result["total_sketches"] == 1
    sketch = result["fully_constrained"][0]
    assert sketch["status"] == "FullyConstrained"
    assert sketch["free_count"] == 0
    assert sketch["conflict_count"] == 0


@pytest.mark.asyncio
async def test_get_sketch_constraint_status_under_constrained_code():
    kcl_code = """
sketch(on = YZ) {
  line1 = line(start = [var 1.32mm, var -1.93mm], end = [var 6.08mm, var 2.51mm])
}
"""
    response = await mcp.call_tool(
        "get_sketch_constraint_status",
        arguments={"kcl_code": kcl_code, "kcl_path": None},
    )
    result = _meta_result(response)
    assert isinstance(result, dict)
    assert len(result["under_constrained"]) == 1
    assert len(result["fully_constrained"]) == 0
    assert result["total_sketches"] == 1
    sketch = result["under_constrained"][0]
    assert sketch["status"] == "UnderConstrained"
    assert sketch["free_count"] > 0


@pytest.mark.asyncio
async def test_get_sketch_constraint_status_over_constrained_code():
    kcl_code = """
sketch(on = YZ) {
  line1 = line(start = [var 2mm, var 8mm], end = [var 5mm, var 7mm])
  line1.start.at[0] == 2
  line1.start.at[1] == 8
  line1.end.at[0] == 5
  line1.end.at[1] == 7
  distance([line1.start, line1.end]) == 100mm
}
"""
    response = await mcp.call_tool(
        "get_sketch_constraint_status",
        arguments={"kcl_code": kcl_code, "kcl_path": None},
    )
    result = _meta_result(response)
    assert isinstance(result, dict)
    assert len(result["over_constrained"]) == 1
    assert len(result["fully_constrained"]) == 0
    assert result["total_sketches"] == 1
    sketch = result["over_constrained"][0]
    assert sketch["status"] == "OverConstrained"
    assert sketch["conflict_count"] > 0


@pytest.mark.asyncio
async def test_get_sketch_constraint_status_path(fully_constrained_kcl: str):
    response = await mcp.call_tool(
        "get_sketch_constraint_status",
        arguments={"kcl_code": None, "kcl_path": fully_constrained_kcl},
    )
    result = _meta_result(response)
    assert isinstance(result, dict)
    assert len(result["fully_constrained"]) == 1
    assert result["total_sketches"] == 1


@pytest.mark.asyncio
async def test_get_sketch_constraint_status_error():
    response = await mcp.call_tool(
        "get_sketch_constraint_status",
        arguments={"kcl_code": "asdf = asdf", "kcl_path": None},
    )
    result = _meta_result(response)
    assert isinstance(result, dict)
    assert result["kcl_executes_successfully"] is False
    assert result["kcl_error"] is not None
    assert result["kcl_error"]["phase"] in {"parse", "execution"}
    assert isinstance(result["kcl_error"]["text"], str)
    assert result["kcl_error"]["text"] != ""


SKETCH_VISUALIZER_KCL = """
@settings(experimentalFeatures = allow)

s1 = sketch(on = YZ) {
  line1 = line(start = [var 2mm, var 8mm], end = [var 5mm, var 7mm])
  line1.start.at[0] == 2
  line1.start.at[1] == 8
  line1.end.at[0] == 5
  line1.end.at[1] == 7
}

s2 = sketch(on = XZ) {
  line1 = line(start = [var 1mm, var 2mm], end = [var 3mm, var 4mm])
}
"""


@pytest.mark.asyncio
async def test_visualize_sketch_returns_png():
    response = await mcp.call_tool(
        "visualize_sketch",
        arguments={
            "sketch_name": "s1",
            "kcl_code": SKETCH_VISUALIZER_KCL,
            "kcl_path": None,
        },
    )

    image = _content_list(response)[0]
    assert isinstance(image, ImageContent)
    assert image.mimeType == "image/png"
    png_bytes = base64.b64decode(image.data)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    with PILImage.open(io.BytesIO(png_bytes)) as png:
        assert png.format == "PNG"


@pytest.mark.asyncio
async def test_visualize_sketch_path_writes_png(tmp_path: Path):
    kcl_path = tmp_path / "main.kcl"
    kcl_path.write_text(SKETCH_VISUALIZER_KCL)
    output_dir = tmp_path / "renders"
    output_dir.mkdir()

    response = await mcp.call_tool(
        "visualize_sketch",
        arguments={
            "sketch_name": "s2",
            "kcl_code": None,
            "kcl_path": str(kcl_path),
            "output_path": str(output_dir),
        },
    )

    result = Path(_meta_result(response))
    assert result == output_dir / "image.png"
    assert result.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.asyncio
async def test_visualize_sketch_reports_missing_name():
    response = await mcp.call_tool(
        "visualize_sketch",
        arguments={
            "sketch_name": "missingSketch",
            "kcl_code": SKETCH_VISUALIZER_KCL,
            "kcl_path": None,
        },
    )

    result = _meta_result(response)
    assert isinstance(result, str)
    assert "no sketch named `missingSketch`" in result


@pytest.mark.asyncio
async def test_get_face_info(monkeypatch):
    face_info = zoo_mcp.zoo_tools.FaceInfo(
        face_get_position=FaceGetPosition(pos=Point3d(x=1.0, y=2.0, z=3.0)),
        face_get_gradient=FaceGetGradient(
            df_du=Point3d(x=1.0, y=0.0, z=0.0),
            df_dv=Point3d(x=0.0, y=1.0, z=0.0),
            normal=Point3d(x=0.0, y=0.0, z=1.0),
        ),
        face_get_center=FaceGetCenter(pos=Point3d(x=0.5, y=0.5, z=0.0)),
    )
    mock = AsyncMock(return_value=face_info)
    monkeypatch.setattr("zoo_mcp.server.zoo_face_info", mock)

    response = await mcp.call_tool(
        "get_face_info",
        arguments={
            "face_id": "face-id",
            "session_id": "session-id",
        },
    )

    result = _structured_result(response)
    assert result == {
        "face_get_position": {"pos": {"x": 1.0, "y": 2.0, "z": 3.0}},
        "face_get_gradient": {
            "df_du": {"x": 1.0, "y": 0.0, "z": 0.0},
            "df_dv": {"x": 0.0, "y": 1.0, "z": 0.0},
            "normal": {"x": 0.0, "y": 0.0, "z": 1.0},
        },
        "face_get_center": {"pos": {"x": 0.5, "y": 0.5, "z": 0.0}},
    }
    mock.assert_awaited_once_with(
        face_id="face-id",
        session_id="session-id",
    )


@pytest.mark.asyncio
async def test_mock_execute_kcl(cube_kcl: str):
    response = await mcp.call_tool(
        "mock_execute_kcl",
        arguments={
            "kcl_code": None,
            "kcl_path": cube_kcl,
        },
    )
    result = _meta_result(response)
    assert isinstance(result, (tuple, list))
    assert result[0] is True
    assert "KCL code mock executed successfully" in result[1]


@pytest.mark.asyncio
async def test_mock_execute_kcl_error():
    response = await mcp.call_tool(
        "mock_execute_kcl",
        arguments={
            "kcl_code": "asdf = asdf",
            "kcl_path": None,
        },
    )
    result = _meta_result(response)
    assert isinstance(result, (tuple, list))
    assert result[0] is False
    assert "Failed to mock execute KCL code" in result[1]


@pytest.mark.asyncio
async def test_mock_execute_kcl_rejects_unknown_keyword():
    response = await mcp.call_tool(
        "mock_execute_kcl",
        arguments={
            "kcl_code": """
profile = startSketchOn(XY)
  |> startProfile(at = [0, 0])
  |> xLine(length = 10)
  |> yLine(length = 10)
  |> xLine(length = -10)
  |> close()

part = extrude(profile, length = 10, symmetry = true)
""",
            "kcl_path": None,
        },
    )
    result = _meta_result(response)
    assert isinstance(result, (tuple, list))
    assert result[0] is False
    assert "Errors:" in result[1]
    assert "`symmetry` is not an argument of `extrude`" in result[1]


@pytest.mark.asyncio
async def test_mock_execute_kcl_preserves_warning_success(monkeypatch):
    outcome = _FakeOutcome([_FakeIssue(severity="warning")])

    async def fake_mock_execute_code(code: str):
        return outcome

    monkeypatch.setattr(
        zoo_mcp.zoo_tools.kcl, "mock_execute_code", fake_mock_execute_code
    )

    ok, message = await zoo_mcp.zoo_tools.zoo_mock_execute_kcl(kcl_code="anything")
    assert ok is True
    assert "Warnings:" in message
    assert "warning report" in message


@pytest.mark.asyncio
async def test_snapshot_from_a_modeling_session(populated_modeling_session: str):
    response = await mcp.call_tool(
        "snapshot", arguments={"session_id": populated_modeling_session}
    )
    assert isinstance(_content_list(response)[0], ImageContent)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "camera_view",
    [
        "front",
        "isometric",
        "multiview",
        "multi_isometric",
        ["front", "top"],
        {"up": [0, 0, 1], "vantage": [0, -1, 0], "center": [0, 0, 0]},
    ],
    ids=["named", "isometric", "multiview", "multi_isometric", "list", "explicit"],
)
async def test_snapshot_camera_views(
    camera_view: object, populated_modeling_session: str
):
    response = await mcp.call_tool(
        "snapshot",
        arguments={
            "session_id": populated_modeling_session,
            "camera_view": camera_view,
        },
    )

    assert isinstance(_content_list(response)[0], ImageContent)


@pytest.mark.asyncio
async def test_snapshot_rejects_an_unknown_camera_view():
    with pytest.raises(ToolError, match="Invalid camera view"):
        await mcp.call_tool(
            "snapshot",
            arguments={"session_id": "session-id", "camera_view": "asdf"},
        )


@pytest.mark.asyncio
async def test_snapshot_rejects_a_malformed_camera_view():
    with pytest.raises(ToolError, match="Invalid camera view"):
        await mcp.call_tool(
            "snapshot",
            arguments={
                "session_id": "session-id",
                "camera_view": {"hello": [0, 0, 0]},
            },
        )


@pytest.mark.asyncio
async def test_snapshot_output_path(populated_modeling_session: str, tmp_path):
    """When output_path is provided, the tool writes to disk and returns the path."""
    output_path = tmp_path / "snap.jpg"
    response = await mcp.call_tool(
        "snapshot",
        arguments={
            "session_id": populated_modeling_session,
            "output_path": str(output_path),
        },
    )

    result = _meta_result(response)
    assert Path(result) == output_path.resolve()
    assert Path(result).stat().st_size > 0


@pytest.mark.asyncio
async def test_snapshot_output_path_directory(
    populated_modeling_session: str, tmp_path
):
    """A directory output_path writes image.jpg into that directory."""
    response = await mcp.call_tool(
        "snapshot",
        arguments={
            "session_id": populated_modeling_session,
            "output_path": str(tmp_path),
        },
    )

    result = _meta_result(response)
    assert Path(result).name == "image.jpg"
    assert Path(result).stat().st_size > 0


@pytest.mark.asyncio
async def test_snapshot_respects_max_image_dimension(populated_modeling_session: str):
    response = await mcp.call_tool(
        "snapshot",
        arguments={
            "session_id": populated_modeling_session,
            "max_image_dimension": 128,
        },
    )

    image = _content_list(response)[0]
    assert isinstance(image, ImageContent)
    with PILImage.open(io.BytesIO(base64.b64decode(image.data))) as rendered:
        assert max(rendered.size) <= 128


@pytest.mark.asyncio
async def test_collapsed_snapshot_tools_are_gone():
    """The per-source/per-layout snapshot tools folded into `snapshot`."""
    tool_names = {tool.name for tool in await mcp.list_tools()}

    assert "snapshot" in tool_names
    assert tool_names.isdisjoint(
        {
            "snapshot_of_kcl",
            "snapshot_of_cad",
            "multiview_snapshot_of_kcl",
            "multiview_snapshot_of_cad",
            "multi_isometric_snapshot_of_kcl",
            "multi_isometric_snapshot_of_cad",
        }
    )


@pytest.mark.asyncio
async def test_text_to_cad_tools_are_not_registered():
    tools = await mcp.list_tools()
    tool_names = {tool.name for tool in tools}

    assert "text_to_cad" not in tool_names
    assert "edit_kcl_project" not in tool_names


_FAKE_DOC_CONTENT = {
    "docs/kcl-lang/functions": (
        "# Functions\n\n"
        "Functions in KCL let you reuse logic. A function takes named "
        "parameters and returns a value. Functions can be defined inline or "
        "imported from other modules. Calling a function uses the standard "
        "`name(arg = value)` syntax. This page describes how functions "
        "behave inside KCL programs and how they interact with sketches.\n"
    ),
    "docs/kcl-lang/sketches": (
        "# Sketches\n\n"
        "A sketch is a 2D profile drawn on a plane. Sketches are the "
        "starting point for most modeling operations.\n"
    ),
    "docs/kcl-std/functions/std-sketch-extrude": (
        "# extrude\n\n"
        "Extrude a sketch into a 3D solid. The extrude function takes a "
        "sketch and a length and returns a solid.\n"
    ),
    "docs/kcl-std/types/Sketch": (
        "# Sketch\n\nThe Sketch type represents a 2D sketch on a plane.\n"
    ),
    "docs/kcl-std/consts/PI": ("# PI\n\nThe mathematical constant pi.\n"),
    "docs/kcl-std/modules/math": ("# math\n\nMath utility module.\n"),
}


def _build_fake_docs() -> KCLDocs:
    docs = KCLDocs(docs=dict(_FAKE_DOC_CONTENT))
    for path in docs.docs:
        if path.startswith("docs/kcl-lang/"):
            docs.index["kcl-lang"].append(path)
        elif path.startswith("docs/kcl-std/functions/"):
            docs.index["kcl-std-functions"].append(path)
        elif path.startswith("docs/kcl-std/types/"):
            docs.index["kcl-std-types"].append(path)
        elif path.startswith("docs/kcl-std/consts/"):
            docs.index["kcl-std-consts"].append(path)
        elif path.startswith("docs/kcl-std/modules/"):
            docs.index["kcl-std-modules"].append(path)
    for category in docs.index:
        docs.index[category].sort()
    return docs


@pytest_asyncio.fixture(scope="module")
async def live_docs_index():
    """Populate the docs index with synthetic data.

    Fully offline: avoids hitting zoo.dev. The fetch and parse pipeline is
    covered by ``tests/test_docs.py`` and ``tests/test_data_retrieval_utils.py``.
    """
    saved = KCLDocs._instance
    KCLDocs._instance = _build_fake_docs()
    try:
        yield KCLDocs._instance
    finally:
        KCLDocs._instance = saved


@pytest.mark.xdist_group(name="docs")
@pytest.mark.asyncio
async def test_list_kcl_docs(live_docs_index):
    """Test that list_kcl_docs returns categorized documentation."""
    response = await mcp.call_tool("list_kcl_docs", arguments={})
    inner_list = _content_list(response)
    assert len(inner_list) == 1
    assert isinstance(inner_list[0], TextContent)
    result = json.loads(inner_list[0].text)

    assert isinstance(result, dict)
    # Check all expected categories exist
    assert "kcl-lang" in result
    assert "kcl-std-functions" in result
    assert "kcl-std-types" in result
    assert "kcl-std-consts" in result
    assert "kcl-std-modules" in result

    # Verify we have docs in each major category
    assert len(result["kcl-lang"]) > 0, "Should have KCL language docs"
    assert len(result["kcl-std-functions"]) > 0, "Should have std function docs"
    assert len(result["kcl-std-types"]) > 0, "Should have std type docs"


@pytest.mark.xdist_group(name="docs")
@pytest.mark.asyncio
async def test_search_kcl_docs(live_docs_index):
    """Test that search_kcl_docs returns relevant excerpts for 'extrude'."""
    response = await mcp.call_tool(
        "search_kcl_docs", arguments={"query": "extrude", "max_results": 5}
    )
    # FastMCP returns list results as [list_of_TextContent]
    inner_list = _content_list(response)
    assert len(inner_list) > 0, "Should find results for 'extrude'"

    # Parse all results
    result = [json.loads(tc.text) for tc in inner_list]

    # Check result structure
    first_result = result[0]
    assert "path" in first_result
    assert "title" in first_result
    assert "excerpt" in first_result
    assert "match_count" in first_result

    # The extrude function doc should be in the results
    paths = [r["path"] for r in result]
    assert any("extrude" in p.lower() for p in paths), (
        "Should find extrude-related docs"
    )


@pytest.mark.xdist_group(name="docs")
@pytest.mark.asyncio
async def test_search_kcl_docs_sketch(live_docs_index):
    """Test searching for 'sketch' returns relevant results."""
    response = await mcp.call_tool(
        "search_kcl_docs", arguments={"query": "sketch", "max_results": 10}
    )
    inner_list = _content_list(response)
    assert len(inner_list) > 0, "Should find results for 'sketch'"

    result = [json.loads(tc.text) for tc in inner_list]

    # Should find sketch-related docs
    all_text = " ".join([r["title"] + r["excerpt"] for r in result]).lower()
    assert "sketch" in all_text, "Results should contain 'sketch'"


@pytest.mark.xdist_group(name="docs")
@pytest.mark.asyncio
async def test_search_kcl_docs_no_results(live_docs_index):
    """Test that search_kcl_docs handles queries with no matches."""
    response = await mcp.call_tool(
        "search_kcl_docs",
        arguments={"query": "xyznonexistentterm12345abc", "max_results": 5},
    )
    inner_list = _content_list(response)
    assert len(inner_list) == 0, "Should find no results for gibberish query"


@pytest.mark.xdist_group(name="docs")
@pytest.mark.asyncio
async def test_search_kcl_docs_empty_query(live_docs_index):
    """Test that search_kcl_docs handles empty queries."""
    response = await mcp.call_tool(
        "search_kcl_docs", arguments={"query": "", "max_results": 5}
    )
    inner_list = _content_list(response)
    assert len(inner_list) == 1

    result = json.loads(inner_list[0].text)
    assert "error" in result


@pytest.mark.xdist_group(name="docs")
@pytest.mark.asyncio
async def test_get_kcl_doc_functions(live_docs_index):
    """Test that get_kcl_doc retrieves the functions documentation."""
    response = await mcp.call_tool(
        "get_kcl_doc", arguments={"doc_path": "docs/kcl-lang/functions"}
    )
    inner_list = _content_list(response)
    assert len(inner_list) == 1

    result = inner_list[0].text
    assert isinstance(result, str)
    # Should contain content about functions
    assert "function" in result.lower(), "Should mention functions"
    assert len(result) > 100, "Should have substantial content"


@pytest.mark.xdist_group(name="docs")
@pytest.mark.asyncio
async def test_get_kcl_doc_extrude(live_docs_index):
    """Test that get_kcl_doc retrieves the extrude function documentation."""
    response = await mcp.call_tool(
        "get_kcl_doc",
        arguments={"doc_path": "docs/kcl-std/functions/std-sketch-extrude"},
    )
    inner_list = _content_list(response)
    assert len(inner_list) == 1

    result = inner_list[0].text
    assert isinstance(result, str)
    assert "extrude" in result.lower(), "Should mention extrude"


@pytest.mark.xdist_group(name="docs")
@pytest.mark.asyncio
async def test_get_kcl_doc_not_found(live_docs_index):
    """Test that get_kcl_doc handles missing documentation."""
    response = await mcp.call_tool(
        "get_kcl_doc", arguments={"doc_path": "docs/nonexistent/fake"}
    )
    inner_list = _content_list(response)
    assert len(inner_list) == 1

    result = inner_list[0].text
    assert isinstance(result, str)
    assert "Documentation not found" in result


@pytest.mark.xdist_group(name="docs")
@pytest.mark.asyncio
async def test_get_kcl_doc_path_traversal(live_docs_index):
    """Test that get_kcl_doc rejects path traversal attempts."""
    response = await mcp.call_tool(
        "get_kcl_doc", arguments={"doc_path": "../../../etc/passwd"}
    )
    inner_list = _content_list(response)
    assert len(inner_list) == 1

    result = inner_list[0].text
    assert isinstance(result, str)
    assert "Documentation not found" in result


_FAKE_SAMPLE_FILES: dict[str, dict[str, str]] = {
    "ball-bearing": {"main.kcl": "// ball-bearing\nradius = 10\n"},
    "spur-gear": {"main.kcl": "// spur gear\nteeth = 24\n"},
    "axial-fan": {
        "main.kcl": 'import "parameters.kcl" as p\n',
        "parameters.kcl": "blades = 5\n",
        "fan.kcl": "// fan body\n",
    },
}

_FAKE_SAMPLE_META: dict[str, tuple[str, str]] = {
    "ball-bearing": ("Ball Bearing", "A rolling-element bearing."),
    "spur-gear": ("Spur Gear", "A gear with straight teeth."),
    "axial-fan": ("Axial Fan", "An axial-flow fan with multiple parts."),
}


def _build_fake_samples() -> KCLSamples:
    samples = KCLSamples()
    for name, (title, description) in _FAKE_SAMPLE_META.items():
        files = _FAKE_SAMPLE_FILES[name]
        samples.manifest[name] = SampleMetadata(
            title=title,
            description=description,
            multipleFiles=len(files) > 1,
        )
        samples.file_index[name] = dict(files)
    return samples


@pytest_asyncio.fixture(scope="module")
async def live_samples_index():
    """Populate the samples index with synthetic data.

    Fully offline. Parse and fetch behavior is covered by
    ``tests/test_samples.py`` and ``tests/test_data_retrieval_utils.py``.
    """
    saved = KCLSamples._instance
    KCLSamples._instance = _build_fake_samples()
    try:
        yield KCLSamples._instance
    finally:
        KCLSamples._instance = saved


@pytest.mark.xdist_group(name="samples")
@pytest.mark.asyncio
async def test_list_kcl_samples(live_samples_index):
    """Test that list_kcl_samples returns sample information."""
    response = await mcp.call_tool("list_kcl_samples", arguments={})
    inner_list = _content_list(response)
    assert len(inner_list) > 0, "Should have samples in the list"

    # Parse first result and check structure
    first_result = json.loads(inner_list[0].text)
    assert "name" in first_result
    assert "title" in first_result
    assert "description" in first_result
    assert "multipleFiles" in first_result


@pytest.mark.xdist_group(name="samples")
@pytest.mark.asyncio
async def test_search_kcl_samples_gear(live_samples_index):
    """Test searching for 'gear' returns relevant results."""
    response = await mcp.call_tool(
        "search_kcl_samples", arguments={"query": "gear", "max_results": 5}
    )
    inner_list = _content_list(response)
    assert len(inner_list) > 0, "Should find results for 'gear'"

    result = [json.loads(tc.text) for tc in inner_list]

    # Check result structure
    first_result = result[0]
    assert "name" in first_result
    assert "title" in first_result
    assert "description" in first_result
    assert "match_count" in first_result
    assert "excerpt" in first_result

    # Should find gear-related samples
    all_text = " ".join([r["title"] + r["description"] for r in result]).lower()
    assert "gear" in all_text, "Results should contain 'gear'"


@pytest.mark.xdist_group(name="samples")
@pytest.mark.asyncio
async def test_search_kcl_samples_bearing(live_samples_index):
    """Test searching for 'bearing' returns relevant results."""
    response = await mcp.call_tool(
        "search_kcl_samples", arguments={"query": "bearing", "max_results": 5}
    )
    inner_list = _content_list(response)
    assert len(inner_list) > 0, "Should find results for 'bearing'"

    result = [json.loads(tc.text) for tc in inner_list]

    # Should find bearing-related samples
    names = [r["name"] for r in result]
    assert any("bearing" in n for n in names), "Should find bearing samples"


@pytest.mark.xdist_group(name="samples")
@pytest.mark.asyncio
async def test_search_kcl_samples_no_results(live_samples_index):
    """Test that search_kcl_samples handles queries with no matches."""
    response = await mcp.call_tool(
        "search_kcl_samples",
        arguments={"query": "xyznonexistentterm12345abc", "max_results": 5},
    )
    inner_list = _content_list(response)
    assert len(inner_list) == 0, "Should find no results for gibberish query"


@pytest.mark.xdist_group(name="samples")
@pytest.mark.asyncio
async def test_search_kcl_samples_empty_query(live_samples_index):
    """Test that search_kcl_samples handles empty queries."""
    response = await mcp.call_tool(
        "search_kcl_samples", arguments={"query": "", "max_results": 5}
    )
    inner_list = _content_list(response)
    assert len(inner_list) == 1

    result = json.loads(inner_list[0].text)
    assert "error" in result


@pytest.mark.xdist_group(name="samples")
@pytest.mark.asyncio
async def test_get_kcl_sample_single_file(live_samples_index):
    """Test that get_kcl_sample retrieves a single-file sample."""
    response = await mcp.call_tool(
        "get_kcl_sample", arguments={"sample_name": "ball-bearing"}
    )
    result = _meta_result(response)

    assert isinstance(result, dict)
    assert result["name"] == "ball-bearing"
    assert "title" in result
    assert "description" in result
    assert "files" in result
    assert len(result["files"]) >= 1

    # Check file structure
    main_file = next((f for f in result["files"] if f["filename"] == "main.kcl"), None)
    assert main_file is not None, "Should have main.kcl"
    assert len(main_file["content"]) > 0, "Should have content"


@pytest.mark.xdist_group(name="samples")
@pytest.mark.asyncio
async def test_get_kcl_sample_multi_file(live_samples_index):
    """Test that get_kcl_sample retrieves a multi-file sample."""
    response = await mcp.call_tool(
        "get_kcl_sample", arguments={"sample_name": "axial-fan"}
    )
    result = _meta_result(response)

    assert isinstance(result, dict)
    assert result["name"] == "axial-fan"
    assert result["multipleFiles"] is True
    assert len(result["files"]) > 1, "Should have multiple files"

    # Check expected files exist
    filenames = [f["filename"] for f in result["files"]]
    assert "main.kcl" in filenames
    assert "parameters.kcl" in filenames or "fan.kcl" in filenames


@pytest.mark.xdist_group(name="samples")
@pytest.mark.asyncio
async def test_get_kcl_sample_not_found(live_samples_index):
    """Test that get_kcl_sample handles missing samples."""
    response = await mcp.call_tool(
        "get_kcl_sample", arguments={"sample_name": "nonexistent-sample-xyz"}
    )
    inner_list = _content_list(response)
    assert len(inner_list) == 1

    result = inner_list[0].text
    assert isinstance(result, str)
    assert "Sample not found" in result


@pytest.mark.xdist_group(name="samples")
@pytest.mark.asyncio
async def test_get_kcl_sample_path_traversal(live_samples_index):
    """Test that get_kcl_sample rejects path traversal attempts."""
    response = await mcp.call_tool(
        "get_kcl_sample", arguments={"sample_name": "../../../etc/passwd"}
    )
    inner_list = _content_list(response)
    assert len(inner_list) == 1

    result = inner_list[0].text
    assert isinstance(result, str)
    assert "Sample not found" in result


@pytest.mark.asyncio
async def test_save_image(populated_modeling_session: str, tmp_path):
    """Test saving an image to disk."""
    # First get an image from snapshot
    snapshot_response = await mcp.call_tool(
        "snapshot",
        arguments={
            "session_id": populated_modeling_session,
            "camera_view": "isometric",
        },
    )
    image = _content_list(snapshot_response)[0]
    assert isinstance(image, ImageContent)

    # Now save the image to disk
    output_path = tmp_path / "test_image.png"
    response = await mcp.call_tool(
        "save_image",
        arguments={
            "image": image.model_dump(),
            "output_path": str(output_path),
        },
    )
    result = _meta_result(response)
    assert Path(result).exists()
    assert Path(result).stat().st_size > 0


@pytest.mark.asyncio
async def test_save_image_to_directory(populated_modeling_session: str, tmp_path):
    """Test saving an image to a directory creates image.jpg."""
    # First get an image from snapshot
    snapshot_response = await mcp.call_tool(
        "snapshot",
        arguments={
            "session_id": populated_modeling_session,
            "camera_view": "isometric",
        },
    )
    image = _content_list(snapshot_response)[0]
    assert isinstance(image, ImageContent)

    # Save to directory
    response = await mcp.call_tool(
        "save_image",
        arguments={
            "image": image.model_dump(),
            "output_path": str(tmp_path),
        },
    )
    result = _meta_result(response)
    assert Path(result).exists()
    assert Path(result).name == "image.jpg"
    assert Path(result).stat().st_size > 0


@pytest.mark.asyncio
async def test_save_png_image_to_directory(tmp_path: Path):
    png_buffer = io.BytesIO()
    PILImage.new("RGB", (2, 2)).save(png_buffer, format="PNG")
    image = ImageContent(
        type="image",
        data=base64.b64encode(png_buffer.getvalue()).decode(),
        mimeType="image/png",
    )

    response = await mcp.call_tool(
        "save_image",
        arguments={"image": image.model_dump(), "output_path": str(tmp_path)},
    )

    result = Path(_meta_result(response))
    assert result == tmp_path / "image.png"
    assert result.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.asyncio
async def test_save_image_creates_parent_dirs(
    populated_modeling_session: str, tmp_path
):
    """Test that save_image creates parent directories if they don't exist."""
    # First get an image from snapshot
    snapshot_response = await mcp.call_tool(
        "snapshot",
        arguments={
            "session_id": populated_modeling_session,
            "camera_view": "isometric",
        },
    )
    image = _content_list(snapshot_response)[0]
    assert isinstance(image, ImageContent)

    # Save to a nested path that doesn't exist
    output_path = tmp_path / "nested" / "dirs" / "test_image.png"
    response = await mcp.call_tool(
        "save_image",
        arguments={
            "image": image.model_dump(),
            "output_path": str(output_path),
        },
    )
    result = _meta_result(response)
    assert Path(result).exists()
    assert Path(result).stat().st_size > 0


@pytest.mark.asyncio
async def test_save_image_to_temp_file(populated_modeling_session: str):
    """Test that save_image creates a temp file when no path is provided."""
    # First get an image from snapshot
    snapshot_response = await mcp.call_tool(
        "snapshot",
        arguments={
            "session_id": populated_modeling_session,
            "camera_view": "isometric",
        },
    )
    image = _content_list(snapshot_response)[0]
    assert isinstance(image, ImageContent)

    # Save without specifying a path
    response = await mcp.call_tool(
        "save_image",
        arguments={
            "image": image.model_dump(),
        },
    )
    result = _meta_result(response)
    assert Path(result).exists()
    assert Path(result).suffix == ".jpg"
    assert Path(result).stat().st_size > 0


@pytest.mark.asyncio
async def test_list_org_datasets_success(
    monkeypatch: pytest.MonkeyPatch, async_kittycad_client: AsyncKittyCAD
):
    fake_datasets = [
        SimpleNamespace(id="uuid-1", name="alpha", description="first dataset"),
        SimpleNamespace(id="uuid-2", name="beta", description=None),
    ]
    mock = MagicMock(return_value=_async_items(fake_datasets))
    monkeypatch.setattr(
        async_kittycad_client.orgs,
        "list_org_datasets",
        mock,
    )

    response = await mcp.call_tool("list_org_datasets", arguments={})
    result = _meta_result(response)
    assert result == [
        {"id": "uuid-1", "name": "alpha", "description": "first dataset"},
        {"id": "uuid-2", "name": "beta", "description": None},
    ]
    # Datasets an org excluded from lookup must be filtered out server-side.
    mock.assert_called_once_with(limit=None, page_token=None, lookup_enabled=True)


@pytest.mark.asyncio
async def test_list_org_datasets_falls_back_for_unknown_status(
    monkeypatch: pytest.MonkeyPatch, async_kittycad_client: AsyncKittyCAD
):
    with pytest.raises(ValueError) as exc_info:
        OrgDataset.model_validate({"status": "paused"})

    monkeypatch.setattr(
        async_kittycad_client.orgs,
        "list_org_datasets",
        MagicMock(return_value=_async_error(exc_info.value)),
    )
    raw_response = SimpleNamespace(
        is_success=True,
        json=MagicMock(
            return_value={
                "items": [
                    {
                        "id": "uuid-paused",
                        "name": "paused dataset",
                        "description": "temporarily paused",
                        "status": "paused",
                    }
                ],
                "next_page": None,
            }
        ),
    )
    http_client = MagicMock()
    http_client.get = AsyncMock(return_value=raw_response)
    monkeypatch.setattr(
        async_kittycad_client,
        "get_http_client",
        MagicMock(return_value=http_client),
    )

    response = await mcp.call_tool("list_org_datasets", arguments={})

    assert _meta_result(response) == [
        {
            "id": "uuid-paused",
            "name": "paused dataset",
            "description": "temporarily paused",
        }
    ]
    # The raw fallback has to filter on lookup too, not just the SDK path.
    assert http_client.get.call_args.kwargs["params"] == {"lookup_enabled": "true"}


@pytest.mark.asyncio
async def test_list_org_datasets_empty_when_404(
    monkeypatch: pytest.MonkeyPatch, async_kittycad_client: AsyncKittyCAD
):
    monkeypatch.setattr(
        async_kittycad_client.orgs,
        "list_org_datasets",
        MagicMock(
            return_value=_async_error(
                KittyCADClientError(message="No org found", status_code=404)
            )
        ),
    )

    response = await mcp.call_tool("list_org_datasets", arguments={})
    result = _meta_result(response)
    assert result == []


@pytest.mark.asyncio
async def test_list_org_datasets_empty_when_fallback_hits_404(
    monkeypatch: pytest.MonkeyPatch, async_kittycad_client: AsyncKittyCAD
):
    """Schema drift plus a 404 must still be empty, not an uncaught client error."""
    with pytest.raises(ValueError) as exc_info:
        OrgDataset.model_validate({"status": "paused"})

    monkeypatch.setattr(
        async_kittycad_client.orgs,
        "list_org_datasets",
        MagicMock(return_value=_async_error(exc_info.value)),
    )
    http_client = MagicMock()
    http_client.get = AsyncMock(return_value=SimpleNamespace(is_success=False))
    monkeypatch.setattr(
        async_kittycad_client,
        "get_http_client",
        MagicMock(return_value=http_client),
    )
    monkeypatch.setattr(
        "kittycad.response_helpers.raise_for_status",
        MagicMock(
            side_effect=KittyCADClientError(message="No org found", status_code=404)
        ),
    )

    response = await mcp.call_tool("list_org_datasets", arguments={})
    assert _meta_result(response) == []


@pytest.mark.asyncio
async def test_search_org_dataset_semantic_success(
    monkeypatch: pytest.MonkeyPatch, async_kittycad_client: AsyncKittyCAD
):
    fake_matches = [
        SimpleNamespace(
            chunk_index=0,
            content="first chunk",
            conversion_id="conv-uuid-1",
            similarity=0.91,
            source_file_path="path/one.kcl",
        ),
        SimpleNamespace(
            chunk_index=2,
            content="second chunk",
            conversion_id="conv-uuid-2",
            similarity=0.74,
            source_file_path="path/two.kcl",
        ),
    ]
    mock = AsyncMock(return_value=fake_matches)
    monkeypatch.setattr(
        async_kittycad_client.orgs,
        "search_org_dataset_semantic",
        mock,
    )

    response = await mcp.call_tool(
        "search_org_dataset_semantic",
        arguments={
            "dataset_id": "dataset-uuid",
            "query": "find the gear",
            "limit": 5,
        },
    )
    result = _meta_result(response)
    assert result == [
        {
            "source_file_path": "path/one.kcl",
            "content": "first chunk",
            "similarity": 0.91,
            "chunk_index": 0,
            "conversion_id": "conv-uuid-1",
        },
        {
            "source_file_path": "path/two.kcl",
            "content": "second chunk",
            "similarity": 0.74,
            "chunk_index": 2,
            "conversion_id": "conv-uuid-2",
        },
    ]
    mock.assert_awaited_once_with(id="dataset-uuid", q="find the gear", limit=5)


@pytest.mark.asyncio
async def test_search_org_dataset_semantic_error(
    monkeypatch: pytest.MonkeyPatch, async_kittycad_client: AsyncKittyCAD
):
    monkeypatch.setattr(
        async_kittycad_client.orgs,
        "search_org_dataset_semantic",
        AsyncMock(side_effect=RuntimeError("boom")),
    )

    response = await mcp.call_tool(
        "search_org_dataset_semantic",
        arguments={
            "dataset_id": "dataset-uuid",
            "query": "anything",
        },
    )
    result = _meta_result(response)
    assert isinstance(result, str)
    assert result.startswith("There was an error searching dataset dataset-uuid")


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=3, reruns_delay=1)
@pytest.mark.skipif(
    not os.environ.get("ZOO_DATASET_TOKEN"),
    reason="ZOO_DATASET_TOKEN not set",
)
async def test_list_and_search_org_dataset_live(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZOO_HOST", "https://api.dev.zoo.dev")
    dev_client = AsyncKittyCAD(token=os.environ["ZOO_DATASET_TOKEN"])
    monkeypatch.setattr(zoo_mcp.zoo_tools, "AsyncKittyCAD", lambda **kwargs: dev_client)

    try:
        list_response = await mcp.call_tool("list_org_datasets", arguments={})
        datasets = _meta_result(list_response)
        assert isinstance(datasets, list)
        assert len(datasets) >= 1, "expected at least one dataset in the dev org"
        dataset_id = datasets[0]["id"]

        search_response = await mcp.call_tool(
            "search_org_dataset_semantic",
            arguments={
                "dataset_id": dataset_id,
                "query": "PVC DWV Straight Reducer",
            },
        )
        matches = _meta_result(search_response)
        assert isinstance(matches, list)
        assert len(matches) >= 1, "expected at least one semantic-search match"
    finally:
        await dev_client.aclose()


@pytest.mark.asyncio
async def test_list_org_skills_success(
    monkeypatch: pytest.MonkeyPatch, async_kittycad_client: AsyncKittyCAD
):
    fake_skills = [
        SimpleNamespace(
            id="uuid-1",
            name="alpha",
            description="first skill",
            markdown="# Alpha",
        ),
        SimpleNamespace(
            id="uuid-2",
            name="beta",
            description="second skill",
            markdown="# Beta",
        ),
    ]
    monkeypatch.setattr(
        async_kittycad_client.orgs,
        "list_org_skills",
        AsyncMock(return_value=fake_skills),
    )

    response = await mcp.call_tool("list_org_skills", arguments={})
    result = _meta_result(response)
    assert result == [
        {
            "id": "uuid-1",
            "name": "alpha",
            "description": "first skill",
            "markdown": "# Alpha",
        },
        {
            "id": "uuid-2",
            "name": "beta",
            "description": "second skill",
            "markdown": "# Beta",
        },
    ]


@pytest.mark.asyncio
async def test_list_org_skills_empty_when_404(
    monkeypatch: pytest.MonkeyPatch, async_kittycad_client: AsyncKittyCAD
):
    monkeypatch.setattr(
        async_kittycad_client.orgs,
        "list_org_skills",
        AsyncMock(
            side_effect=KittyCADClientError(message="No org found", status_code=404)
        ),
    )

    response = await mcp.call_tool("list_org_skills", arguments={})
    result = _meta_result(response)
    assert result == []


@pytest.mark.asyncio
async def test_list_org_skills_empty_when_none(
    monkeypatch: pytest.MonkeyPatch, async_kittycad_client: AsyncKittyCAD
):
    monkeypatch.setattr(
        async_kittycad_client.orgs,
        "list_org_skills",
        AsyncMock(return_value=None),
    )

    response = await mcp.call_tool("list_org_skills", arguments={})
    result = _meta_result(response)
    assert result == []


@pytest.mark.asyncio
async def test_list_org_skills_error(
    monkeypatch: pytest.MonkeyPatch, async_kittycad_client: AsyncKittyCAD
):
    monkeypatch.setattr(
        async_kittycad_client.orgs,
        "list_org_skills",
        AsyncMock(side_effect=RuntimeError("boom")),
    )

    response = await mcp.call_tool("list_org_skills", arguments={})
    result = _meta_result(response)
    assert isinstance(result, str)
    assert result.startswith("There was an error listing org skills")
