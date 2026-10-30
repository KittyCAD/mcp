from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import pytest_asyncio
from mcp.server.mcpserver.exceptions import ToolError

from tests.test_evaluate_kcl import bindings as bindings  # noqa: PLC0414 -- shared pytest fixture
from tests.test_evaluate_kcl import call
from zoo_mcp.server import mcp
from zoo_mcp.zoo_tools import (
    UNIT_AREA_MAP,
    UNIT_DENSITY_MAP,
    UNIT_LENGTH_MAP,
    UNIT_MASS_MAP,
    UNIT_VOLUME_MAP,
    CameraView,
    KclPhysicalPropertiesRequest,
    zoo_evaluate_kcl,
)


@pytest_asyncio.fixture
async def input_schema():
    tool = next(tool for tool in await mcp.list_tools() if tool.name == "evaluate_kcl")
    return tool.model_dump(by_alias=True)["inputSchema"]


def test_schema_describes_accepted_camera_and_measurement_inputs(input_schema):
    definitions = input_schema["$defs"]
    snapshot = definitions["EvaluationSnapshotRequest"]
    camera = definitions["EvaluationCamera"]
    measurements = definitions["EvaluationPhysicalPropertiesRequest"]
    for request in (snapshot, camera, measurements):
        assert request["description"]
        assert request["additionalProperties"] is False
        assert all(field["description"] for field in request["properties"].values())

    views = snapshot["properties"]["views"]
    assert (views["minItems"], views["maxItems"]) == (1, 4)
    assert views["default"] == ["isometric"]
    named, custom = views["items"]["anyOf"]
    assert set(named["enum"]) == set(CameraView.views.value)
    assert custom["$ref"] == "#/$defs/EvaluationCamera"
    assert set(camera["required"]) == {"up", "vantage", "center"}
    for coordinates in camera["properties"].values():
        assert coordinates["minItems"] == coordinates["maxItems"] == 3
        assert coordinates["items"]["type"] == "number"
    for name in ("vantage", "center"):
        assert "millimeters" in camera["properties"][name]["description"]
        assert "regardless of" in camera["properties"][name]["description"]
    assert "Dimensionless" in camera["properties"]["up"]["description"]
    assert snapshot["properties"]["padding"]["minimum"] == 0
    assert snapshot["properties"]["max_image_dimension"]["exclusiveMinimum"] == 0

    fields = measurements["properties"]
    assert measurements["required"] == ["properties"]
    assert fields["properties"]["minItems"] == 1
    assert set(fields["properties"]["items"]["enum"]) == {
        "volume",
        "mass",
        "surface_area",
        "center_of_mass",
        "bounding_box",
    }
    for name, accepted in (
        ("unit_length", UNIT_LENGTH_MAP),
        ("unit_mass", UNIT_MASS_MAP),
        ("unit_area", UNIT_AREA_MAP),
        ("unit_volume", UNIT_VOLUME_MAP),
    ):
        assert set(fields[name]["enum"]) == set(accepted)
        assert fields[name]["default"] in accepted
    assert set(fields["unit_density"]["anyOf"][0]["enum"]) == set(UNIT_DENSITY_MAP)
    for name in ("density", "unit_density"):
        assert "mass" in fields[name]["description"]
    for name in ("snapshot_request", "physical_properties_request"):
        assert input_schema["properties"][name]["description"]
        assert input_schema["properties"][name]["default"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_name", ["snapshot_request", "physical_properties_request"]
)
async def test_schema_examples_work_through_mcp(bindings, input_schema, request_name):
    point = SimpleNamespace(x=1, y=2, z=3)
    bindings.session.measure.side_effect = None
    bindings.session.measure.return_value = SimpleNamespace(
        get_volume=lambda: 12.5,
        get_mass=lambda: 1,
        get_bounding_box=lambda: SimpleNamespace(
            get_center=lambda: point, get_dimensions=lambda: point
        ),
    )
    examples = input_schema["properties"][request_name]["examples"]
    assert examples
    status = (
        "rendered_snapshots_status"
        if request_name == "snapshot_request"
        else "physical_analysis_status"
    )
    for example in examples:
        _, result = await call({"kcl_code": "x = 1", request_name: example})
        assert result["ok"]
        assert result["inspection"][status] == "succeeded"


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["python", "mcp"])
@pytest.mark.parametrize(
    "unit_length, unit_mass, unit_density, unit_area, unit_volume",
    [("in", "lb", "lb:ft3", "ft2", "usgal"), ("cm", "kg", "kg:m3", "m2", "l")],
)
async def test_string_units_remain_compatible_for_python_clients(
    bindings,
    route: str,
    unit_length: str,
    unit_mass: str,
    unit_density: str,
    unit_area: str,
    unit_volume: str,
):
    # Zookeeper passes string parameters to the public Python request class.
    request = KclPhysicalPropertiesRequest(
        ("volume", "mass", "surface_area", "center_of_mass", "bounding_box"),
        unit_length=unit_length,
        unit_mass=unit_mass,
        unit_density=unit_density,
        density=1000,
        unit_area=unit_area,
        unit_volume=unit_volume,
    )
    point = SimpleNamespace(x=1, y=2, z=3)
    bindings.session.measure.side_effect = None
    bindings.session.measure.return_value = SimpleNamespace(
        get_volume=lambda: 12.5,
        get_mass=lambda: 1,
        get_surface_area=lambda: 42,
        get_center_of_mass=lambda: point,
        get_bounding_box=lambda: SimpleNamespace(
            get_center=lambda: point, get_dimensions=lambda: point
        ),
    )
    if route == "python":
        result = asdict(
            await zoo_evaluate_kcl(
                kcl_code="x = 1", physical_properties_request=request
            )
        )
    else:
        _, result = await call(
            {"kcl_code": "x = 1", "physical_properties_request": asdict(request)}
        )
    assert result["inspection"]["physical_analysis_status"] == "succeeded"
    properties = result["inspection"]["physical_properties"]
    assert {name: value["unit"] for name, value in properties.items()} == {
        "volume": unit_volume,
        "mass": unit_mass,
        "surface_area": unit_area,
        "center_of_mass": unit_length,
        "bounding_box": unit_length,
    }
    bindings.session.measure.assert_awaited_once()


@pytest.mark.asyncio
async def test_custom_camera_preserves_coordinates_and_automatic_framing(
    bindings, monkeypatch
):
    camera = {"up": [0, 0, 1], "vantage": [150, -150, 120], "center": [10, 0, 0]}
    convert = Mock(wraps=CameraView.to_kcl_camera)
    monkeypatch.setattr(CameraView, "to_kcl_camera", convert)
    _, result = await call(
        {
            "kcl_code": "x = 1",
            "snapshot_request": {"views": ["front", camera]},
        }
    )
    assert result["inspection"]["rendered_snapshots_status"] == "succeeded"
    assert result["inspection"]["completed_snapshot_views"] == ["front", "custom_2"]
    convert.assert_any_call(camera)
    assert [
        call.kwargs["zoom"] for call in bindings.session.snapshots.await_args_list
    ] == [True, False]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"views": ["front"] * 5},
        {"max_image_dimension": 0},
        {"padding": float("nan")},
        {"view": "front"},
        {"views": [{"up": [0, 0, 1], "vantage": [1, 0, 1]}]},
        {"views": [{"up": [0, 0, 1], "vantage": [1, 0, 1], "center": [0, 0, 0, 0]}]},
        {
            "views": [
                {"up": [0, 0, 1], "vantage": [1, 0, float("inf")], "center": [0, 0, 0]}
            ]
        },
        {
            "views": [
                {
                    "up": [0, 0, 1],
                    "vantage": [1, 0, 1],
                    "center": [0, 0, 0],
                    "zoom": True,
                }
            ]
        },
        {"views": [{"up": [0, 0, 1], "vantage": [0, 0, 0], "center": [0, 0, 0]}]},
        {"views": [{"up": [0, 0, 1], "vantage": [0, 0, 1], "center": [0, 0, 0]}]},
    ],
)
async def test_invalid_snapshot_objects_fail_before_execution(bindings, options):
    with pytest.raises(ToolError) as error:
        await call({"kcl_code": "x = 1", "snapshot_request": options})
    assert type(error.value) is ToolError
    bindings.mock.assert_not_awaited()
    bindings.opened.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"properties": []},
        {"properties": ["volume"], "units_volume": "mm3"},
        {"properties": ["mass"], "density": 1000},
        {"properties": ["mass"], "unit_density": "kg:m3"},
        {"properties": ["mass"], "density": float("inf"), "unit_density": "kg:m3"},
        *[
            {"properties": ["volume"], name: "unsupported"}
            for name in (
                "unit_length",
                "unit_mass",
                "unit_density",
                "unit_area",
                "unit_volume",
            )
        ],
    ],
)
async def test_invalid_measurement_objects_fail_before_execution(bindings, options):
    with pytest.raises(ToolError) as error:
        await call({"kcl_code": "x = 1", "physical_properties_request": options})
    assert type(error.value) is ToolError
    bindings.mock.assert_not_awaited()
    bindings.opened.assert_not_awaited()
