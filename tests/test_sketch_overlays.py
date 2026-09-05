"""Exercise region-overlay options through the real MCP and KCL renderer."""

import base64
from io import BytesIO
from pathlib import Path

import pytest
from mcp.types import ImageContent
from PIL import Image

from zoo_mcp import ZooMCPException
from zoo_mcp.server import mcp
from zoo_mcp.zoo_tools import zoo_visualize_sketch

# Same region fixture as modeling-app's sketch_visualizer/region_overlay test.
REGION_KCL = """
profile = sketch(on = XY) {
  bottom = line(start = [var 0mm, var 0mm], end = [var 40mm, var 0mm])
  right = line(start = [var 40mm, var 0mm], end = [var 40mm, var 24mm])
  top = line(start = [var 40mm, var 24mm], end = [var 0mm, var 24mm])
  left = line(start = [var 0mm, var 24mm], end = [var 0mm, var 0mm])
  coincident([bottom.end, right.start])
  coincident([right.end, top.start])
  coincident([top.end, left.start])
  coincident([left.end, bottom.start])
}
selectedRegion = region(segments = [profile.bottom, profile.right])
"""


def colors(png: bytes) -> set[tuple[int, int, int]]:
    with Image.open(BytesIO(png)) as image:
        pixels = image.convert("RGB").tobytes()
    return set(zip(pixels[0::3], pixels[1::3], pixels[2::3], strict=True))


@pytest.mark.asyncio
@pytest.mark.parametrize("downstream_error", [False, True])
async def test_region_overlay_survives_downstream_failure(
    tmp_path: Path, downstream_error: bool
) -> None:
    source = REGION_KCL + ("\nbroken = missingValue\n" if downstream_error else "")
    path = tmp_path / "main.kcl"
    path.write_text(source)
    output = tmp_path / "overlay.png"

    response = await mcp.call_tool(
        "visualize_sketch",
        arguments={
            "sketch_name": "profile",
            "kcl_path": str(path),
            "highlighted_segments": ["bottom", "right"],
            "resolved_region": "selectedRegion",
            "output_path": str(output),
        },
    )

    assert output.is_file(), response
    assert {(255, 79, 216), (43, 72, 43), (60, 115, 255)} <= colors(output.read_bytes())
    assert path.read_text() == source


@pytest.mark.asyncio
async def test_failed_region_can_show_seeds_without_inventing_boundary() -> None:
    source = REGION_KCL.replace("profile.right])", "profile.missing])")
    with pytest.raises(ZooMCPException):
        await zoo_visualize_sketch(
            "profile",
            kcl_code=source,
            highlighted_segments=["bottom"],
            resolved_region="selectedRegion",
        )

    response = await mcp.call_tool(
        "visualize_sketch",
        arguments={
            "sketch_name": "profile",
            "kcl_code": source,
            "highlighted_segments": ["bottom"],
        },
    )
    assert isinstance(response, tuple)
    content = response[0]
    assert isinstance(content, list)
    assert isinstance(content[0], ImageContent)
    pixel_colors = colors(base64.b64decode(content[0].data))
    assert (255, 79, 216) in pixel_colors
    assert (43, 72, 43) not in pixel_colors


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("segments", "region", "error"),
    [
        (["missing"], None, "no segment named `missing`"),
        (["profile.bottom"], None, "no segment named `profile.bottom`"),
        (None, "missing", "no region named `missing`"),
        (None, "profile", "not a resolved region"),
    ],
)
async def test_overlay_rejects_unknown_or_wrong_names(
    segments: list[str] | None, region: str | None, error: str
) -> None:
    with pytest.raises(ZooMCPException, match=error):
        await zoo_visualize_sketch(
            "profile",
            kcl_code=REGION_KCL,
            highlighted_segments=segments,
            resolved_region=region,
        )
