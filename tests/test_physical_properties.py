from pathlib import Path
from typing import Any

import pytest
import trimesh
from kittycad.models import (
    FileCenterOfMass,
    FileConversion,
    FileMass,
    FileSurfaceArea,
    FileVolume,
    Point3d,
)

from zoo_mcp import zoo_tools

_KITTYCAD_COORDINATE_SYSTEM = {
    "name": "kittycad",
    "up_axis": "+z",
    "forward_axis": "-y",
}


def _translated_asymmetric_stl() -> bytes:
    mesh = trimesh.creation.box(extents=(4.0, 6.0, 8.0))
    mesh.apply_translation((12.0, 23.0, 34.0))
    return mesh.export(file_type="stl")


def _mock_physical_property_api(
    monkeypatch: pytest.MonkeyPatch,
    *,
    converted_stl: bytes | None,
) -> dict[str, int]:
    conversion_calls = {"count": 0}

    monkeypatch.setattr(
        zoo_tools.kittycad_client.file,
        "create_file_volume",
        lambda **_kwargs: FileVolume.model_construct(volume=192.0),
    )
    monkeypatch.setattr(
        zoo_tools.kittycad_client.file,
        "create_file_mass",
        lambda **_kwargs: FileMass.model_construct(mass=192.0),
    )
    monkeypatch.setattr(
        zoo_tools.kittycad_client.file,
        "create_file_surface_area",
        lambda **_kwargs: FileSurfaceArea.model_construct(surface_area=208.0),
    )
    monkeypatch.setattr(
        zoo_tools.kittycad_client.file,
        "create_file_center_of_mass",
        lambda **_kwargs: FileCenterOfMass.model_construct(
            # The API currently returns the source center (12, 23, 34) in its
            # internal OpenGL frame: (x, y, z) -> (x, z, -y).
            center_of_mass=Point3d(x=12.0, y=34.0, z=-23.0)
        ),
    )

    def convert_file(**_kwargs: Any) -> FileConversion:
        conversion_calls["count"] += 1
        if converted_stl is None:
            pytest.fail("STL input should not be converted")
        return FileConversion.model_construct(outputs={"output.stl": converted_stl})

    monkeypatch.setattr(
        zoo_tools.kittycad_client.file,
        "create_file_conversion",
        convert_file,
    )
    return conversion_calls


def _assert_spatial_result(result: dict[str, Any]) -> None:
    assert result["coordinate_system"] == _KITTYCAD_COORDINATE_SYSTEM
    assert result["center_of_mass"] == pytest.approx({"x": 12.0, "y": 23.0, "z": 34.0})
    assert result["bounding_box"]["center"] == pytest.approx(
        result["center_of_mass"], abs=1e-6
    )
    assert result["bounding_box"]["dimensions"] == pytest.approx(
        {"x": 4.0, "y": 6.0, "z": 8.0}, abs=1e-6
    )


@pytest.mark.asyncio
async def test_cad_physical_properties_stl_uses_one_coordinate_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stl_data = _translated_asymmetric_stl()
    input_file = tmp_path / "translated-asymmetric.stl"
    input_file.write_bytes(stl_data)
    conversion_calls = _mock_physical_property_api(monkeypatch, converted_stl=None)

    result = await zoo_tools.zoo_calculate_cad_physical_properties(
        file_path=input_file,
        unit_length="mm",
        unit_mass="g",
        unit_density="kg:m3",
        density=1000.0,
        unit_area="mm2",
        unit_vol="mm3",
    )

    _assert_spatial_result(result)
    assert conversion_calls["count"] == 0


@pytest.mark.asyncio
async def test_cad_physical_properties_converted_model_uses_one_coordinate_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stl_data = _translated_asymmetric_stl()
    input_file = tmp_path / "translated-asymmetric.step"
    input_file.write_bytes(b"mock STEP data")
    conversion_calls = _mock_physical_property_api(monkeypatch, converted_stl=stl_data)

    result = await zoo_tools.zoo_calculate_cad_physical_properties(
        file_path=input_file,
        unit_length="mm",
        unit_mass="g",
        unit_density="kg:m3",
        density=1000.0,
        unit_area="mm2",
        unit_vol="mm3",
    )

    _assert_spatial_result(result)
    assert conversion_calls["count"] == 1
