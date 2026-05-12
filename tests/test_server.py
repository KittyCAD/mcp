import json
import os
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from kittycad import KittyCAD
from kittycad.exceptions import KittyCADClientError
from mcp.types import ImageContent, TextContent

import zoo_mcp
import zoo_mcp.zoo_tools
from zoo_mcp.kcl_docs import KCLDocs
from zoo_mcp.kcl_samples import KCLSamples, SampleMetadata
from zoo_mcp.server import mcp


def _meta_result(response: Sequence[Any] | dict[str, Any]) -> Any:
    """Extract response[1]["result"] with proper typing for ty."""
    assert isinstance(response, Sequence)
    meta = response[1]
    assert isinstance(meta, dict)
    return cast(dict[str, Any], meta)["result"]


def _content_list(response: Sequence[Any] | dict[str, Any]) -> list[Any]:
    """Extract response[0] as a typed list for ty."""
    assert isinstance(response, Sequence)
    content = response[0]
    assert isinstance(content, list)
    return cast(list[Any], content)


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
    assert result["z"] == pytest.approx(-5.0)


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
    assert com["z"] == pytest.approx(-5.0)
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
    assert com["z"] == pytest.approx(-5.0, abs=1e-1)
    bbox = result["bounding_box"]
    assert "center" in bbox and "dimensions" in bbox
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
            "input_path": cube_stl,
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
            "input_path": empty_step,
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
    assert isinstance(result, (tuple, list))
    assert result[0] is True
    assert "KCL code executed successfully" in result[1]


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
    assert isinstance(result, (tuple, list))
    assert result[0] is False
    assert "Failed to execute KCL code" in result[1]


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
async def test_multiview_snapshot_of_cad(cube_stl: str):
    response = await mcp.call_tool(
        "multiview_snapshot_of_cad",
        arguments={
            "input_file": cube_stl,
        },
    )
    result = _content_list(response)[0]
    assert isinstance(result, ImageContent)


@pytest.mark.asyncio
async def test_multiview_snapshot_of_cad_error(empty_step: str):
    response = await mcp.call_tool(
        "multiview_snapshot_of_cad",
        arguments={
            "input_file": empty_step,
        },
    )
    result = _meta_result(response)
    assert "error creating the multiview snapshot" in result


@pytest.mark.asyncio
async def test_multiview_snapshot_of_kcl(cube_kcl: str):
    response = await mcp.call_tool(
        "multiview_snapshot_of_kcl",
        arguments={
            "kcl_code": None,
            "kcl_path": cube_kcl,
        },
    )
    result = _content_list(response)[0]
    assert isinstance(result, ImageContent)


@pytest.mark.asyncio
async def test_multiview_snapshot_of_kcl_error(empty_step: str):
    response = await mcp.call_tool(
        "multiview_snapshot_of_kcl",
        arguments={
            "kcl_code": None,
            "kcl_path": empty_step,
        },
    )
    result = _meta_result(response)
    assert "error creating the multiview snapshot" in result


@pytest.mark.asyncio
async def test_multi_isometric_snapshot_of_cad(cube_stl: str):
    response = await mcp.call_tool(
        "multi_isometric_snapshot_of_cad",
        arguments={
            "input_file": cube_stl,
        },
    )
    result = _content_list(response)[0]
    assert isinstance(result, ImageContent)


@pytest.mark.asyncio
async def test_multi_isometric_snapshot_of_cad_error(empty_step: str):
    response = await mcp.call_tool(
        "multi_isometric_snapshot_of_cad",
        arguments={
            "input_file": empty_step,
        },
    )
    result = _meta_result(response)
    assert "error creating the multi-isometric snapshot" in result


@pytest.mark.asyncio
async def test_multi_isometric_snapshot_of_kcl(cube_kcl: str):
    response = await mcp.call_tool(
        "multi_isometric_snapshot_of_kcl",
        arguments={
            "kcl_code": None,
            "kcl_path": cube_kcl,
        },
    )
    result = _content_list(response)[0]
    assert isinstance(result, ImageContent)


@pytest.mark.asyncio
async def test_multi_isometric_snapshot_of_kcl_error(empty_step: str):
    response = await mcp.call_tool(
        "multi_isometric_snapshot_of_kcl",
        arguments={
            "kcl_code": None,
            "kcl_path": empty_step,
        },
    )
    result = _meta_result(response)
    assert "error creating the multi-isometric snapshot" in result


@pytest.mark.asyncio
async def test_snapshot_of_cad(cube_stl: str):
    response = await mcp.call_tool(
        "snapshot_of_cad",
        arguments={
            "input_file": cube_stl,
            "camera_view": "isometric",
        },
    )
    result = _content_list(response)[0]
    assert isinstance(result, ImageContent)


@pytest.mark.asyncio
async def test_snapshot_of_cad_error(empty_step: str):
    response = await mcp.call_tool(
        "snapshot_of_cad",
        arguments={
            "input_file": empty_step,
            "camera_view": "isometric",
        },
    )
    result = _meta_result(response)
    assert "error creating the snapshot" in result


@pytest.mark.asyncio
async def test_snapshot_of_cad_camera(cube_stl: str):
    response = await mcp.call_tool(
        "snapshot_of_cad",
        arguments={
            "input_file": cube_stl,
            "camera_view": {
                "up": [0, 0, 1],
                "vantage": [0, -1, 0],
                "center": [0, 0, 0],
            },
        },
    )
    result = _content_list(response)[0]
    assert isinstance(result, ImageContent)


@pytest.mark.asyncio
async def test_snapshot_of_cad_camera_error(empty_step: str):
    response = await mcp.call_tool(
        "snapshot_of_cad",
        arguments={
            "input_file": empty_step,
            "camera_view": {
                "hello": [0, 0, 0],
            },
        },
    )
    result = _meta_result(response)
    assert "error creating the snapshot" in result


@pytest.mark.asyncio
async def test_snapshot_of_cad_view(cube_stl: str):
    response = await mcp.call_tool(
        "snapshot_of_cad",
        arguments={
            "input_file": cube_stl,
            "camera_view": "front",
        },
    )
    result = _content_list(response)[0]
    assert isinstance(result, ImageContent)


@pytest.mark.asyncio
async def test_snapshot_of_cad_view_error(cube_stl: str):
    response = await mcp.call_tool(
        "snapshot_of_cad",
        arguments={
            "input_file": cube_stl,
            "camera_view": "asdf",
        },
    )
    result = _meta_result(response)
    assert "Invalid camera view" in result


@pytest.mark.asyncio
async def test_snapshot_of_kcl(cube_kcl: str):
    response = await mcp.call_tool(
        "snapshot_of_kcl",
        arguments={
            "kcl_code": None,
            "kcl_path": cube_kcl,
            "camera_view": "isometric",
        },
    )
    result = _content_list(response)[0]
    assert isinstance(result, ImageContent)


@pytest.mark.asyncio
async def test_snapshot_of_kcl_error(empty_step: str):
    response = await mcp.call_tool(
        "snapshot_of_kcl",
        arguments={
            "kcl_code": None,
            "kcl_path": empty_step,
            "camera_view": "isometric",
        },
    )
    result = _meta_result(response)
    assert "error creating the snapshot" in result


@pytest.mark.asyncio
async def test_snapshot_of_kcl_camera(cube_kcl: str):
    response = await mcp.call_tool(
        "snapshot_of_kcl",
        arguments={
            "kcl_code": None,
            "kcl_path": cube_kcl,
            "camera_view": {
                "up": [0, 0, 1],
                "vantage": [0, -1, 0],
                "center": [0, 0, 0],
            },
        },
    )
    result = _content_list(response)[0]
    assert isinstance(result, ImageContent)


@pytest.mark.asyncio
async def test_snapshot_of_kcl_camera_error(empty_kcl: str):
    response = await mcp.call_tool(
        "snapshot_of_kcl",
        arguments={
            "kcl_code": None,
            "kcl_path": empty_kcl,
            "camera_view": {
                "hello": [0, 0, 0],
            },
        },
    )
    result = _meta_result(response)
    assert "error creating the snapshot" in result


@pytest.mark.asyncio
async def test_snapshot_of_kcl_view(cube_kcl: str):
    response = await mcp.call_tool(
        "snapshot_of_kcl",
        arguments={
            "kcl_code": None,
            "kcl_path": cube_kcl,
            "camera_view": "front",
        },
    )
    result = _content_list(response)[0]
    assert isinstance(result, ImageContent)


@pytest.mark.asyncio
async def test_snapshot_of_kcl_view_error(cube_kcl: str):
    response = await mcp.call_tool(
        "snapshot_of_kcl",
        arguments={
            "kcl_code": None,
            "kcl_path": cube_kcl,
            "camera_view": "asdf",
        },
    )
    result = _meta_result(response)
    assert "Invalid camera view" in result


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
async def live_docs_cache():
    """Populate the docs cache with synthetic data.

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
async def test_list_kcl_docs(live_docs_cache):
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
async def test_search_kcl_docs(live_docs_cache):
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
async def test_search_kcl_docs_sketch(live_docs_cache):
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
async def test_search_kcl_docs_no_results(live_docs_cache):
    """Test that search_kcl_docs handles queries with no matches."""
    response = await mcp.call_tool(
        "search_kcl_docs",
        arguments={"query": "xyznonexistentterm12345abc", "max_results": 5},
    )
    inner_list = _content_list(response)
    assert len(inner_list) == 0, "Should find no results for gibberish query"


@pytest.mark.xdist_group(name="docs")
@pytest.mark.asyncio
async def test_search_kcl_docs_empty_query(live_docs_cache):
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
async def test_get_kcl_doc_functions(live_docs_cache):
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
async def test_get_kcl_doc_extrude(live_docs_cache):
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
async def test_get_kcl_doc_not_found(live_docs_cache):
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
async def test_get_kcl_doc_path_traversal(live_docs_cache):
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
        samples.file_cache[name] = dict(files)
    return samples


@pytest_asyncio.fixture(scope="module")
async def live_samples_cache():
    """Populate the samples cache with synthetic data.

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
async def test_list_kcl_samples(live_samples_cache):
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
async def test_search_kcl_samples_gear(live_samples_cache):
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
async def test_search_kcl_samples_bearing(live_samples_cache):
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
async def test_search_kcl_samples_no_results(live_samples_cache):
    """Test that search_kcl_samples handles queries with no matches."""
    response = await mcp.call_tool(
        "search_kcl_samples",
        arguments={"query": "xyznonexistentterm12345abc", "max_results": 5},
    )
    inner_list = _content_list(response)
    assert len(inner_list) == 0, "Should find no results for gibberish query"


@pytest.mark.xdist_group(name="samples")
@pytest.mark.asyncio
async def test_search_kcl_samples_empty_query(live_samples_cache):
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
async def test_get_kcl_sample_single_file(live_samples_cache):
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
async def test_get_kcl_sample_multi_file(live_samples_cache):
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
async def test_get_kcl_sample_not_found(live_samples_cache):
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
async def test_get_kcl_sample_path_traversal(live_samples_cache):
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
async def test_save_image(cube_stl: str, tmp_path):
    """Test saving an image to disk."""
    # First get an image from snapshot_of_cad
    snapshot_response = await mcp.call_tool(
        "snapshot_of_cad",
        arguments={
            "input_file": cube_stl,
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
async def test_save_image_to_directory(cube_stl: str, tmp_path):
    """Test saving an image to a directory creates image.png."""
    # First get an image from snapshot_of_cad
    snapshot_response = await mcp.call_tool(
        "snapshot_of_cad",
        arguments={
            "input_file": cube_stl,
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
    assert Path(result).name == "image.png"
    assert Path(result).stat().st_size > 0


@pytest.mark.asyncio
async def test_save_image_creates_parent_dirs(cube_stl: str, tmp_path):
    """Test that save_image creates parent directories if they don't exist."""
    # First get an image from snapshot_of_cad
    snapshot_response = await mcp.call_tool(
        "snapshot_of_cad",
        arguments={
            "input_file": cube_stl,
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
async def test_save_image_to_temp_file(cube_stl: str):
    """Test that save_image creates a temp file when no path is provided."""
    # First get an image from snapshot_of_cad
    snapshot_response = await mcp.call_tool(
        "snapshot_of_cad",
        arguments={
            "input_file": cube_stl,
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
    assert Path(result).suffix == ".png"
    assert Path(result).stat().st_size > 0


@pytest.mark.asyncio
async def test_list_org_datasets_success(monkeypatch: pytest.MonkeyPatch):
    fake_datasets = [
        SimpleNamespace(id="uuid-1", name="alpha", description="first dataset"),
        SimpleNamespace(id="uuid-2", name="beta", description=None),
    ]
    monkeypatch.setattr(
        zoo_mcp.kittycad_client.orgs,
        "list_org_datasets",
        MagicMock(return_value=iter(fake_datasets)),
    )

    response = await mcp.call_tool("list_org_datasets", arguments={})
    result = _meta_result(response)
    assert result == [
        {"id": "uuid-1", "name": "alpha", "description": "first dataset"},
        {"id": "uuid-2", "name": "beta", "description": None},
    ]


@pytest.mark.asyncio
async def test_list_org_datasets_empty_when_404(monkeypatch: pytest.MonkeyPatch):
    def raise_404(*args: Any, **kwargs: Any):
        raise KittyCADClientError(message="No org found", status_code=404)

    monkeypatch.setattr(
        zoo_mcp.kittycad_client.orgs,
        "list_org_datasets",
        raise_404,
    )

    response = await mcp.call_tool("list_org_datasets", arguments={})
    result = _meta_result(response)
    assert result == []


@pytest.mark.asyncio
async def test_search_org_dataset_semantic_success(monkeypatch: pytest.MonkeyPatch):
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
    mock = MagicMock(return_value=fake_matches)
    monkeypatch.setattr(
        zoo_mcp.kittycad_client.orgs,
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
    mock.assert_called_once_with(id="dataset-uuid", q="find the gear", limit=5)


@pytest.mark.asyncio
async def test_search_org_dataset_semantic_error(monkeypatch: pytest.MonkeyPatch):
    def raise_500(*args: Any, **kwargs: Any):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        zoo_mcp.kittycad_client.orgs,
        "search_org_dataset_semantic",
        raise_500,
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
    dev_client = KittyCAD(token=os.environ["ZOO_DATASET_TOKEN"])
    monkeypatch.setattr(zoo_mcp, "kittycad_client", dev_client)
    monkeypatch.setattr(zoo_mcp.zoo_tools, "kittycad_client", dev_client)

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
