import asyncio
import base64
import io

import pytest
from mcp.types import CallToolResult, ImageContent
from PIL import Image

from zoo_mcp.server import mcp

pytestmark = [
    pytest.mark.live,
    pytest.mark.asyncio,
    pytest.mark.xdist_group(name="engine"),
]


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
