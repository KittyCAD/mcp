import asyncio
import base64
import io
from pathlib import Path

import pytest
from mcp.types import CallToolResult, ImageContent
from PIL import Image

from zoo_mcp.server import mcp

pytestmark = [
    pytest.mark.live,
    pytest.mark.asyncio,
    pytest.mark.xdist_group(name="engine"),
]


@pytest.mark.parametrize("inline_code", [False, True], ids=["path", "code"])
async def test_all_physical_properties_match_cube(cube_kcl, inline_code):
    async with asyncio.timeout(120):
        response = await mcp.call_tool(
            "evaluate_kcl",
            {
                "kcl_code": Path(cube_kcl).read_text() if inline_code else None,
                "kcl_path": None if inline_code else cube_kcl,
                "physical_properties_request": {
                    "properties": [
                        "volume",
                        "mass",
                        "surface_area",
                        "center_of_mass",
                        "bounding_box",
                    ],
                    "unit_length": "mm",
                    "unit_volume": "cm3",
                    "unit_mass": "g",
                    "unit_area": "mm2",
                    "density": 1000,
                    "unit_density": "kg:m3",
                },
            },
        )
    assert isinstance(response, CallToolResult)
    result = response.structured_content
    assert result is not None and result["ok"], result
    inspection = result["inspection"]
    assert inspection["physical_analysis_status"] == "succeeded", inspection
    properties = inspection["physical_properties"]
    for name, value, unit in (
        ("volume", 1.0, "cm3"),
        ("mass", 1.0, "g"),
        ("surface_area", 600.0, "mm2"),
    ):
        assert properties[name]["value"] == pytest.approx(value, abs=1e-3)
        assert properties[name]["unit"] == unit
    assert properties["center_of_mass"]["value"] == pytest.approx(
        {"x": 5, "y": 5, "z": 5}, abs=0.1
    )
    assert properties["bounding_box"]["dimensions"] == pytest.approx(
        {"x": 10, "y": 10, "z": 10}, abs=0.1
    )


async def test_combined_inspection_matches_cube(cube_kcl, tmp_path):
    async with asyncio.timeout(120):
        response = await mcp.call_tool(
            "evaluate_kcl",
            {
                "kcl_path": cube_kcl,
                "snapshot_request": {"views": ["front", "isometric"]},
                "physical_properties_request": {
                    "properties": ["volume", "bounding_box"]
                },
            },
        )
        assert isinstance(response, CallToolResult)
        result = response.structured_content
        assert result is not None and result["ok"], result
        inspection = result["inspection"]
        assert inspection["physical_analysis_status"] == "succeeded", inspection
        properties = inspection["physical_properties"]
        assert properties["volume"]["value"] == pytest.approx(1000)
        assert properties["bounding_box"]["dimensions"] == pytest.approx(
            {"x": 10, "y": 10, "z": 10}
        )
        assert inspection["rendered_snapshots_status"] == "succeeded", inspection
        image = next(c for c in response.content if isinstance(c, ImageContent))
        Image.open(io.BytesIO(base64.b64decode(image.data))).verify()
