import io
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar, cast
from uuid import uuid4

import aiofiles
import kcl
import trimesh

if TYPE_CHECKING:

    class FixedLintsProtocol(Protocol):
        """Protocol for kcl.FixedLints - the stub file is missing these attributes."""

        @property
        def new_code(self) -> str: ...
        @property
        def unfixed_lints(self) -> list[kcl.Discovered]: ...


from kittycad.exceptions import KittyCADClientError
from kittycad.models import (
    Axis,
    AxisDirectionPair,
    Direction,
    FileCenterOfMass,
    FileConversion,
    FileExportFormat,
    FileImportFormat,
    FileMass,
    FileSurfaceArea,
    FileVolume,
    ImageFormat,
    ImportFile,
    InputFormat3d,
    ModelingCmd,
    ModelingCmdId,
    Point3d,
    PostEffectType,
    System,
    UnitArea,
    UnitDensity,
    UnitLength,
    UnitMass,
    UnitVolume,
    WebSocketRequest,
)
from kittycad.models.input_format3d import (
    OptionFbx,
    OptionGltf,
    OptionObj,
    OptionPly,
    OptionSldprt,
    OptionStep,
    OptionStl,
)
from kittycad.models.modeling_cmd import (
    OptionDefaultCameraLookAt,
    OptionDefaultCameraSetOrthographic,
    OptionImportFiles,
    OptionTakeSnapshot,
    OptionViewIsometric,
    OptionZoomToFit,
)
from kittycad.models.uuid import Uuid
from kittycad.models.web_socket_request import OptionModelingCmdReq

from zoo_mcp import ZooMCPException, kittycad_client, logger
from zoo_mcp.utils.image_utils import create_image_collage, resize_image

SUPPORTED_EXTS = {x.value.lower() for x in FileImportFormat} | {"stp"}

# Map alternative extensions to their canonical FileImportFormat values
_EXT_ALIASES = {
    "stp": "step",
}

# Mappings from user-facing short strings to kcl PyO3 enum members.
# The kcl unit enums cannot be constructed from strings directly.
UNIT_AREA_MAP: dict[str, kcl.UnitArea] = {
    "cm2": kcl.UnitArea.SquareCentimeters,
    "dm2": kcl.UnitArea.SquareDecimeters,
    "ft2": kcl.UnitArea.SquareFeet,
    "in2": kcl.UnitArea.SquareInches,
    "km2": kcl.UnitArea.SquareKilometers,
    "m2": kcl.UnitArea.SquareMeters,
    "mm2": kcl.UnitArea.SquareMillimeters,
    "yd2": kcl.UnitArea.SquareYards,
}

UNIT_VOLUME_MAP: dict[str, kcl.UnitVolume] = {
    "cm3": kcl.UnitVolume.CubicCentimeters,
    "ft3": kcl.UnitVolume.CubicFeet,
    "in3": kcl.UnitVolume.CubicInches,
    "m3": kcl.UnitVolume.CubicMeters,
    "yd3": kcl.UnitVolume.CubicYards,
    "usfloz": kcl.UnitVolume.FluidOunces,
    "usgal": kcl.UnitVolume.Gallons,
    "l": kcl.UnitVolume.Liters,
    "ml": kcl.UnitVolume.Milliliters,
}

UNIT_LENGTH_MAP: dict[str, kcl.UnitLength] = {
    "cm": kcl.UnitLength.Centimeters,
    "ft": kcl.UnitLength.Feet,
    "in": kcl.UnitLength.Inches,
    "m": kcl.UnitLength.Meters,
    "mm": kcl.UnitLength.Millimeters,
    "yd": kcl.UnitLength.Yards,
}

UNIT_MASS_MAP: dict[str, kcl.UnitMass] = {
    "g": kcl.UnitMass.Grams,
    "kg": kcl.UnitMass.Kilograms,
    "lb": kcl.UnitMass.Pounds,
}

UNIT_DENSITY_MAP: dict[str, kcl.UnitDensity] = {
    "lb:ft3": kcl.UnitDensity.PoundsPerCubicFeet,
    "kg:m3": kcl.UnitDensity.KilogramsPerCubicMeter,
}


_T = TypeVar("_T")


def _parse_unit(value: str, mapping: dict[str, _T], unit_type_name: str) -> _T:
    """Look up a unit enum member from a user-provided string."""
    result = mapping.get(value)
    if result is None:
        valid = ", ".join(f"'{k}'" for k in mapping)
        raise ZooMCPException(
            f"Invalid {unit_type_name} '{value}'. Valid options: {valid}"
        )
    return result


def _normalize_ext(ext: str) -> str:
    """Normalize a file extension to its canonical FileImportFormat value.

    Args:
        ext: The file extension (without the leading dot), case-insensitive.

    Returns:
        The normalized extension that can be used with FileImportFormat.
    """
    ext_lower = ext.lower()
    return _EXT_ALIASES.get(ext_lower, ext_lower)


def _check_kcl_code_or_path(
    kcl_code: str | None,
    kcl_path: Path | str | None,
    require_main_file: bool = True,
) -> None:
    """This is a helper function to check the provided kcl_code or kcl_path for various functions.
        If both are provided, kcl_code is used.
        If kcl_path is a file, it checks if the path is a .kcl file, otherwise raises an exception.
        If kcl_path is a directory, it checks if it contains a main.kcl file in the root, otherwise raises an exception.
        If neither are provided, it raises an exception.

    Args:
        kcl_code (str | None): KCL code
        kcl_path (Path | str | None): KCL path, the path should point to a .kcl file or a directory containing a main.kcl file.
        require_main_file (bool): Whether to require a main.kcl file in the directory if kcl_path is a directory. Default is True.

    Returns:
        None
    """

    # default to using the code if both are provided
    if kcl_code and kcl_path:
        logger.warning("Both code and kcl_path provided, using code")
        kcl_path = None

    if kcl_path:
        kcl_path = Path(kcl_path)
        if not kcl_path.exists():
            logger.error("The provided kcl_path does not exist")
            raise ZooMCPException("The provided kcl_path does not exist")
        if kcl_path.is_file() and kcl_path.suffix != ".kcl":
            logger.error("The provided kcl_path is not a .kcl file")
            raise ZooMCPException("The provided kcl_path is not a .kcl file")
        if (
            kcl_path.is_dir()
            and require_main_file
            and not (kcl_path / "main.kcl").is_file()
        ):
            logger.error(
                "The provided kcl_path directory does not contain a main.kcl file"
            )
            raise ZooMCPException(
                "The provided kcl_path does not contain a main.kcl file"
            )

    if not kcl_code and not kcl_path:
        logger.error("Neither code nor kcl_path provided")
        raise ZooMCPException("Neither code nor kcl_path provided")


class KCLExportFormat(Enum):
    formats = {
        "fbx": kcl.FileExportFormat.Fbx,
        "gltf": kcl.FileExportFormat.Gltf,
        "glb": kcl.FileExportFormat.Glb,
        "obj": kcl.FileExportFormat.Obj,
        "ply": kcl.FileExportFormat.Ply,
        "step": kcl.FileExportFormat.Step,
        "stl": kcl.FileExportFormat.Stl,
    }


class CameraView(Enum):
    views = {
        "front": {
            "up": [0.0, 0.0, 1.0],
            "vantage": [0.0, -1.0, 0.0],
            "center": [0.0, 0.0, 0.0],
        },
        "back": {
            "up": [0.0, 0.0, 1.0],
            "vantage": [0.0, 1.0, 0.0],
            "center": [0.0, 0.0, 0.0],
        },
        "left": {
            "up": [0.0, 0.0, 1.0],
            "vantage": [-1.0, 0.0, 0.0],
            "center": [0.0, 0.0, 0.0],
        },
        "right": {
            "up": [0.0, 0.0, 1.0],
            "vantage": [1.0, 0.0, 0.0],
            "center": [0.0, 0.0, 0.0],
        },
        "top": {
            "up": [0.0, 1.0, 0.0],
            "vantage": [0.0, 0.0, 1.0],
            "center": [0.0, 0.0, 0.0],
        },
        "bottom": {
            "up": [0.0, -1.0, 0.0],
            "vantage": [0.0, 0.0, -1.0],
            "center": [0.0, 0.0, 0.0],
        },
        "isometric": {
            "up": [0.0, 0.0, 1.0],
            "vantage": [1.0, -1.0, 1.0],
            "center": [0.0, 0.0, 0.0],
        },
        "isometric_front_right": {
            "up": [0.0, 0.0, 1.0],
            "vantage": [1.0, -1.0, 1.0],
            "center": [0.0, 0.0, 0.0],
        },
        "isometric_front_left": {
            "up": [0.0, 0.0, 1.0],
            "vantage": [-1.0, -1.0, 1.0],
            "center": [0.0, 0.0, 0.0],
        },
        "isometric_back_right": {
            "up": [0.0, 0.0, 1.0],
            "vantage": [1.0, 1.0, -1.0],
            "center": [0.0, 0.0, 0.0],
        },
        "isometric_back_left": {
            "up": [0.0, 0.0, 1.0],
            "vantage": [-1.0, 1.0, -1.0],
            "center": [0.0, 0.0, 0.0],
        },
    }

    @staticmethod
    def to_kcl_camera(view: dict[str, list[float]]) -> kcl.CameraLookAt:
        return kcl.CameraLookAt(
            up=kcl.Point3d(
                x=view["up"][0],
                y=view["up"][1],
                z=view["up"][2],
            ),
            vantage=kcl.Point3d(
                x=view["vantage"][0],
                y=-view["vantage"][1],
                z=view["vantage"][2],
            ),
            center=kcl.Point3d(
                x=view["center"][0],
                y=view["center"][1],
                z=view["center"][2],
            ),
        )

    @staticmethod
    def to_kittycad_camera(view: dict[str, list[float]]) -> OptionDefaultCameraLookAt:
        return OptionDefaultCameraLookAt(
            up=Point3d(
                x=view["up"][0],
                y=view["up"][1],
                z=view["up"][2],
            ),
            vantage=Point3d(
                x=view["vantage"][0],
                y=-view["vantage"][1],
                z=view["vantage"][2],
            ),
            center=Point3d(
                x=view["center"][0],
                y=view["center"][1],
                z=view["center"][2],
            ),
        )


def _get_input_format(ext: str) -> InputFormat3d | None:
    match ext.lower():
        case "fbx":
            return InputFormat3d(OptionFbx())
        case "gltf":
            return InputFormat3d(OptionGltf())
        case "obj":
            return InputFormat3d(
                OptionObj(
                    coords=System(
                        forward=AxisDirectionPair(
                            axis=Axis.Y, direction=Direction.NEGATIVE
                        ),
                        up=AxisDirectionPair(axis=Axis.Z, direction=Direction.POSITIVE),
                    ),
                    units=UnitLength.MM,
                )
            )
        case "ply":
            return InputFormat3d(
                OptionPly(
                    coords=System(
                        forward=AxisDirectionPair(
                            axis=Axis.Y, direction=Direction.NEGATIVE
                        ),
                        up=AxisDirectionPair(axis=Axis.Z, direction=Direction.POSITIVE),
                    ),
                    units=UnitLength.MM,
                )
            )
        case "sldprt":
            return InputFormat3d(OptionSldprt(split_closed_faces=True))
        case "step" | "stp":
            return InputFormat3d(OptionStep(split_closed_faces=True))
        case "stl":
            return InputFormat3d(
                OptionStl(
                    coords=System(
                        forward=AxisDirectionPair(
                            axis=Axis.Y, direction=Direction.NEGATIVE
                        ),
                        up=AxisDirectionPair(axis=Axis.Z, direction=Direction.POSITIVE),
                    ),
                    units=UnitLength.MM,
                )
            )
    return None


async def zoo_calculate_center_of_mass(
    file_path: Path | str,
    unit_length: str,
) -> dict[str, float]:
    """Calculate the center of mass of the file

    Args:
        file_path(Path | str): The path to the file. The file should be one of the supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp, .stl (case-insensitive)
        unit_length(str): The unit length to return. This should be one of 'cm', 'ft', 'in', 'm', 'mm', 'yd'

    Returns:
        dict[str]: If the center of mass can be calculated return the center of mass as a dictionary with x, y, and z keys
    """
    file_path = Path(file_path)

    logger.info("Calculating center of mass for %s", str(file_path.resolve()))

    async with aiofiles.open(file_path, "rb") as inp:
        data = await inp.read()

    src_format = FileImportFormat(_normalize_ext(file_path.suffix.split(".")[1]))

    result = kittycad_client.file.create_file_center_of_mass(
        src_format=src_format,
        body=data,
        output_unit=UnitLength(unit_length),
    )

    if not isinstance(result, FileCenterOfMass):
        logger.info(
            "Failed to calculate center of mass, incorrect return type %s",
            type(result),
        )
        raise ZooMCPException(
            "Failed to calculate center of mass, incorrect return type %s",
            type(result),
        )

    com = result.center_of_mass.to_dict() if result.center_of_mass is not None else None

    if com is None:
        raise ZooMCPException(
            "Failed to calculate center of mass, no center of mass returned"
        )

    return com


async def zoo_calculate_mass(
    file_path: Path | str,
    unit_mass: str,
    unit_density: str,
    density: float,
) -> float:
    """Calculate the mass of the file in the requested unit

    Args:
        file_path(Path | str): The path to the file. The file should be one of the supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stl  (case-insensitive)
        unit_mass(str): The unit mass to return. This should be one of 'g', 'kg', 'lb'.
        unit_density(str): The unit density of the material. This should be one of 'lb:ft3', 'kg:m3'.
        density(float): The density of the material.

    Returns:
        float | None: If the mass of the file can be calculated, return the mass in the requested unit
    """

    file_path = Path(file_path)

    logger.info("Calculating mass for %s", str(file_path.resolve()))

    async with aiofiles.open(file_path, "rb") as inp:
        data = await inp.read()

    src_format = FileImportFormat(_normalize_ext(file_path.suffix.split(".")[1]))

    result = kittycad_client.file.create_file_mass(
        output_unit=UnitMass(unit_mass),
        src_format=src_format,
        body=data,
        material_density_unit=UnitDensity(unit_density),
        material_density=density,
    )

    if not isinstance(result, FileMass):
        logger.info("Failed to calculate mass, incorrect return type %s", type(result))
        raise ZooMCPException(
            "Failed to calculate mass, incorrect return type %s", type(result)
        )

    mass = result.mass

    if mass is None:
        raise ZooMCPException("Failed to calculate mass, no mass returned")

    return mass


async def zoo_calculate_surface_area(file_path: Path | str, unit_area: str) -> float:
    """Calculate the surface area of the file in the requested unit

    Args:
        file_path (Path | str): The path to the file. The file should be one of the supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp, .stl (case-insensitive)
        unit_area (str): The unit area to return. This should be one of 'cm2', 'dm2', 'ft2', 'in2', 'km2', 'm2', 'mm2', 'yd2'.

    Returns:
        float: If the surface area can be calculated return the surface area
    """

    file_path = Path(file_path)

    logger.info("Calculating surface area for %s", str(file_path.resolve()))

    async with aiofiles.open(file_path, "rb") as inp:
        data = await inp.read()

    src_format = FileImportFormat(_normalize_ext(file_path.suffix.split(".")[1]))

    result = kittycad_client.file.create_file_surface_area(
        output_unit=UnitArea(unit_area),
        src_format=src_format,
        body=data,
    )

    if not isinstance(result, FileSurfaceArea):
        logger.error(
            "Failed to calculate surface area, incorrect return type %s",
            type(result),
        )
        raise ZooMCPException(
            "Failed to calculate surface area, incorrect return type %s",
        )

    surface_area = result.surface_area

    if surface_area is None:
        raise ZooMCPException(
            "Failed to calculate surface area, no surface area returned"
        )

    return surface_area


async def zoo_calculate_volume(file_path: Path | str, unit_vol: str) -> float:
    """Calculate the volume of the file in the requested unit

    Args:
        file_path (Path | str): The path to the file. The file should be one of the supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp, .stl (case-insensitive)
        unit_vol (str): The unit volume to return. This should be one of 'cm3', 'ft3', 'in3', 'm3', 'yd3', 'usfloz', 'usgal', 'l', 'ml'.

    Returns:
        float: If the volume of the file can be calculated, return the volume in the requested unit
    """

    file_path = Path(file_path)

    logger.info("Calculating volume for %s", str(file_path.resolve()))

    async with aiofiles.open(file_path, "rb") as inp:
        data = await inp.read()

    src_format = FileImportFormat(_normalize_ext(file_path.suffix.split(".")[1]))

    result = kittycad_client.file.create_file_volume(
        output_unit=UnitVolume(unit_vol),
        src_format=src_format,
        body=data,
    )

    if not isinstance(result, FileVolume):
        logger.info(
            "Failed to calculate volume, incorrect return type %s", type(result)
        )
        raise ZooMCPException(
            "Failed to calculate volume, incorrect return type %s", type(result)
        )

    volume = result.volume

    if volume is None:
        raise ZooMCPException("Failed to calculate volume, no volume returned")

    return volume


async def zoo_calculate_cad_physical_properties(
    file_path: Path | str,
    unit_length: str,
    unit_mass: str,
    unit_density: str,
    density: float,
    unit_area: str,
    unit_vol: str,
) -> dict:
    """Calculate physical properties (volume, mass, surface area, center of mass, bounding box) of a CAD file.

    NOTE: The bounding box will be returned in the same unit length as the original CAD file.

    Args:
        file_path (Path | str): The path to the file. The file should be one of the supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp, .stl (case-insensitive)
        unit_length (str): The unit of length for center of mass. One of 'cm', 'ft', 'in', 'm', 'mm', 'yd'.
        unit_mass (str): The unit of mass for the mass result. One of 'g', 'kg', 'lb'.
        unit_density (str): The unit of density for the material. One of 'lb:ft3', 'kg:m3'.
        density (float): The density of the material.
        unit_area (str): The unit of area for surface area. One of 'cm2', 'dm2', 'ft2', 'in2', 'km2', 'm2', 'mm2', 'yd2'.
        unit_vol (str): The unit of volume. One of 'cm3', 'ft3', 'in3', 'm3', 'yd3', 'usfloz', 'usgal', 'l', 'ml'.

    Returns:
        dict: A dictionary with keys 'volume', 'mass', 'surface_area', 'center_of_mass', and 'bounding_box'.
    """
    file_path = Path(file_path)

    logger.info("Calculating physical properties for %s", str(file_path.resolve()))

    async with aiofiles.open(file_path, "rb") as inp:
        data = await inp.read()

    normalized_ext = _normalize_ext(file_path.suffix.split(".")[1])
    src_format = FileImportFormat(normalized_ext)

    volume_result = kittycad_client.file.create_file_volume(
        output_unit=UnitVolume(unit_vol),
        src_format=src_format,
        body=data,
    )
    if not isinstance(volume_result, FileVolume) or volume_result.volume is None:
        raise ZooMCPException("Failed to calculate volume")

    mass_result = kittycad_client.file.create_file_mass(
        output_unit=UnitMass(unit_mass),
        src_format=src_format,
        body=data,
        material_density_unit=UnitDensity(unit_density),
        material_density=density,
    )
    if not isinstance(mass_result, FileMass) or mass_result.mass is None:
        raise ZooMCPException("Failed to calculate mass")

    sa_result = kittycad_client.file.create_file_surface_area(
        output_unit=UnitArea(unit_area),
        src_format=src_format,
        body=data,
    )
    if not isinstance(sa_result, FileSurfaceArea) or sa_result.surface_area is None:
        raise ZooMCPException("Failed to calculate surface area")

    com_result = kittycad_client.file.create_file_center_of_mass(
        src_format=src_format,
        body=data,
        output_unit=UnitLength(unit_length),
    )
    if (
        not isinstance(com_result, FileCenterOfMass)
        or com_result.center_of_mass is None
    ):
        raise ZooMCPException("Failed to calculate center of mass")

    # Compute bounding box from mesh data
    if normalized_ext == "stl":
        bbox = _compute_stl_bounding_box(data)
    else:
        stl_result = kittycad_client.file.create_file_conversion(
            src_format=src_format,
            output_format=FileExportFormat.STL,
            body=data,
        )
        if not isinstance(stl_result, FileConversion):
            raise ZooMCPException("Failed to convert file for bounding box calculation")
        if stl_result.outputs is None or len(stl_result.outputs) == 0:
            raise ZooMCPException(
                "Failed to convert file for bounding box calculation, no output"
            )
        bbox = _compute_stl_bounding_box(list(stl_result.outputs.values())[0])

    physical_properties = {
        "volume": volume_result.volume,
        "mass": mass_result.mass,
        "surface_area": sa_result.surface_area,
        "center_of_mass": com_result.center_of_mass.to_dict(),
        "bounding_box": bbox,
    }

    return physical_properties


async def zoo_calculate_kcl_physical_properties(
    kcl_code: str | None,
    kcl_path: Path | str | None,
    unit_length: str,
    unit_mass: str,
    unit_density: str,
    density: float,
    unit_area: str,
    unit_vol: str,
) -> dict:
    """Calculate physical properties (volume, mass, surface area, center of mass, bounding box) of a KCL model.

    Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point
    to a .kcl file or a directory containing a main.kcl file.

    Args:
        kcl_code (str | None): KCL code to evaluate.
        kcl_path (Path | str | None): Path to a .kcl file or a directory containing a main.kcl file.
        unit_length (str): The unit of length for center of mass and bounding box. One of 'cm', 'ft', 'in', 'm', 'mm', 'yd'.
        unit_mass (str): The unit of mass for the mass result. One of 'g', 'kg', 'lb'.
        unit_density (str): The unit of density for the material. One of 'lb:ft3', 'kg:m3'.
        density (float): The density of the material.
        unit_area (str): The unit of area for surface area. One of 'cm2', 'dm2', 'ft2', 'in2', 'km2', 'm2', 'mm2', 'yd2'.
        unit_vol (str): The unit of volume. One of 'cm3', 'ft3', 'in3', 'm3', 'yd3', 'usfloz', 'usgal', 'l', 'ml'.

    Returns:
        dict: A dictionary with keys 'volume', 'mass', 'surface_area', 'center_of_mass', and 'bounding_box'.
    """
    logger.info("Calculating physical properties of KCL")

    _check_kcl_code_or_path(kcl_code, kcl_path)

    request = kcl.PhysicalPropertiesRequest()
    request.set_surface_area(_parse_unit(unit_area, UNIT_AREA_MAP, "unit_area"))
    request.set_volume(_parse_unit(unit_vol, UNIT_VOLUME_MAP, "unit_volume"))
    request.set_center_of_mass(_parse_unit(unit_length, UNIT_LENGTH_MAP, "unit_length"))
    request.set_bounding_box(_parse_unit(unit_length, UNIT_LENGTH_MAP, "unit_length"))
    request.set_mass(
        output_unit=_parse_unit(unit_mass, UNIT_MASS_MAP, "unit_mass"),
        material_density=density,
        material_density_unit=_parse_unit(
            unit_density, UNIT_DENSITY_MAP, "unit_density"
        ),
    )

    if kcl_code:
        response = await kcl.execute_code_and_measure(kcl_code, request)
    else:
        response = await kcl.execute_and_measure(str(kcl_path), request)

    volume = response.get_volume()
    com = response.get_center_of_mass()
    sa = response.get_surface_area()
    mass = response.get_mass()
    bbox = response.get_bounding_box()
    bbox_center = bbox.get_center()
    bbox_dims = bbox.get_dimensions()

    physical_properties = {
        "volume": volume,
        "mass": mass,
        "surface_area": sa,
        "center_of_mass": {"x": com.x, "y": com.y, "z": com.z},
        "bounding_box": {
            "center": {"x": bbox_center.x, "y": bbox_center.y, "z": bbox_center.z},
            "dimensions": {"x": bbox_dims.x, "y": bbox_dims.y, "z": bbox_dims.z},
        },
    }

    return physical_properties


def _compute_stl_bounding_box(stl_data: bytes) -> dict:
    """Load an STL file with trimesh and compute the bounding box.

    Args:
        stl_data: Raw bytes of an STL file (binary or ASCII).

    Returns:
        dict with 'center' (dict with x,y,z) and 'dimensions' (dict with x,y,z).
    """
    if len(stl_data) == 0:
        raise ZooMCPException("STL data is empty")

    mesh = trimesh.load(io.BytesIO(stl_data), file_type="stl")

    if not hasattr(mesh, "bounds") or mesh.bounds is None:
        raise ZooMCPException("Failed to compute bounding box from STL data")

    bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    center = (bounds[0] + bounds[1]) / 2
    dimensions = bounds[1] - bounds[0]

    return {
        "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
        "dimensions": {
            "x": float(dimensions[0]),
            "y": float(dimensions[1]),
            "z": float(dimensions[2]),
        },
    }


async def zoo_calculate_bounding_box_kcl(
    unit_length: str,
    kcl_code: str | None = None,
    kcl_path: Path | str | None = None,
) -> dict:
    """Calculate the bounding box of a KCL model.

    Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point
    to a .kcl file or a directory containing a main.kcl file.

    Args:
        kcl_code (str | None): KCL code to evaluate.
        kcl_path (Path | str | None): Path to a .kcl file or a directory containing a main.kcl file.
        unit_length(str): The unit length to return. This should be one of 'cm', 'ft', 'in', 'm', 'mm', 'yd'

    Returns:
        dict: A dictionary with 'center' (dict with x,y,z) and 'dimensions' (dict with x,y,z).
    """
    logger.info("Calculating bounding box of KCL")

    _check_kcl_code_or_path(kcl_code, kcl_path)

    if kcl_code:
        response = await kcl.execute_code_and_bounding_box(
            kcl_code,
            output_unit=_parse_unit(unit_length, UNIT_LENGTH_MAP, "unit_length"),
        )
    else:
        response = await kcl.execute_and_bounding_box(
            str(kcl_path),
            output_unit=_parse_unit(unit_length, UNIT_LENGTH_MAP, "unit_length"),
        )

    center = response.get_center()
    dims = response.get_dimensions()

    return {
        "center": {"x": center.x, "y": center.y, "z": center.z},
        "dimensions": {"x": dims.x, "y": dims.y, "z": dims.z},
    }


async def zoo_calculate_bounding_box_cad(
    file_path: Path | str,
) -> dict:
    """Calculate the bounding box of a CAD file.

    Converts the CAD file to STL via the Zoo API, then parses the mesh to compute the bounding box.

    Args:
        file_path (Path | str): The path to the CAD file. Supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp, .stl (case-insensitive)

    Returns:
        dict: A dictionary with 'center' (dict with x,y,z) and 'dimensions' (dict with x,y,z). The unit of the center and dimensions is the same as original unit of the CAD file.
    """
    file_path = Path(file_path)

    logger.info("Calculating bounding box for %s", str(file_path.resolve()))

    async with aiofiles.open(file_path, "rb") as inp:
        data = await inp.read()

    normalized_ext = _normalize_ext(file_path.suffix.split(".")[1])

    # If the file is already STL, parse it directly
    if normalized_ext == "stl":
        return _compute_stl_bounding_box(data)

    src_format = FileImportFormat(normalized_ext)

    # Convert to STL to get mesh data for bounding box computation
    stl_result = kittycad_client.file.create_file_conversion(
        src_format=src_format,
        output_format=FileExportFormat.STL,
        body=data,
    )

    if not isinstance(stl_result, FileConversion):
        raise ZooMCPException(
            "Failed to convert file for bounding box calculation, incorrect return type %s",
            type(stl_result),
        )

    if stl_result.outputs is None or len(stl_result.outputs) == 0:
        raise ZooMCPException(
            "Failed to convert file for bounding box calculation, no output"
        )

    stl_data = list(stl_result.outputs.values())[0]

    return _compute_stl_bounding_box(stl_data)


async def zoo_convert_cad_file(
    input_path: Path | str,
    export_path: Path | str | None = None,
    export_format: FileExportFormat | str | None = FileExportFormat.STEP,
) -> Path:
    """Convert a cad file to another cad file

    Args:
        input_path (Path | str): path to the CAD file to convert. The file should be one of the supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp, .stl (case-insensitive)
        export_path (Path | str | None): The path to save the cad file. If no path is provided, a temporary file will be created. If the path is a directory, a temporary file will be created in the directory. If the path is a file, it will be overwritten if the extension is valid.
        export_format (FileExportFormat | str | None): format to export the KCL code to. This should be one of 'fbx', 'glb', 'gltf', 'obj', 'ply', 'step', 'stl'. If no format is provided, the default is 'step'.

    Returns:
        Path: Return the path to the exported model if successful
    """

    input_path = Path(input_path)
    input_ext = input_path.suffix.split(".")[1].lower()
    if input_ext not in SUPPORTED_EXTS:
        logger.error("The provided input path does not have a valid extension")
        raise ZooMCPException("The provided input path does not have a valid extension")
    logger.info("Converting the cad file %s", str(input_path.resolve()))

    # check the export format
    if not export_format:
        logger.warning("No export format provided, defaulting to step")
        export_format = FileExportFormat.STEP
    else:
        if export_format not in FileExportFormat:
            logger.warning(
                "Invalid export format %s provided, defaulting to step", export_format
            )
            export_format = FileExportFormat.STEP
        else:
            export_format = FileExportFormat(export_format)

    if export_path is None:
        logger.warning("No export path provided, creating a temporary file")
        export_path = await aiofiles.tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{export_format.value.lower()}"
        )
        export_path = Path(export_path.name)
    else:
        export_path = Path(export_path)
        if export_path.suffix:
            ext = export_path.suffix.split(".")[1]
            if ext not in [i.value for i in FileExportFormat]:
                logger.warning(
                    "The provided export path does not have a valid extension, using a temporary file instead"
                )
                export_path = await aiofiles.tempfile.NamedTemporaryFile(
                    dir=export_path.parent.resolve(),
                    delete=False,
                    suffix=f".{export_format.value.lower()}",
                )
            else:
                logger.warning("The provided export path is a file, overwriting")
        else:
            export_path = await aiofiles.tempfile.NamedTemporaryFile(
                dir=export_path.resolve(),
                delete=False,
                suffix=f".{export_format.value.lower()}",
            )
            logger.info("Using provided export path: %s", str(export_path.name))

    async with aiofiles.open(input_path, "rb") as inp:
        data = await inp.read()

    export_response = kittycad_client.file.create_file_conversion(
        src_format=FileImportFormat(_normalize_ext(input_ext)),
        output_format=FileExportFormat(export_format),
        body=data,
    )

    if not isinstance(export_response, FileConversion):
        logger.error(
            "Failed to convert file, incorrect return type %s",
            type(export_response),
        )
        raise ZooMCPException(
            "Failed to convert file, incorrect return type %s",
        )

    if export_response.outputs is None:
        logger.error("Failed to convert file")
        raise ZooMCPException("Failed to convert file no output response")

    async with aiofiles.open(export_path, "wb") as out:
        await out.write(list(export_response.outputs.values())[0])

    logger.info("KCL project exported successfully to %s", str(export_path.resolve()))

    return export_path


async def zoo_execute_kcl(
    kcl_code: str | None = None,
    kcl_path: Path | str | None = None,
) -> tuple[bool, str]:
    """Execute KCL code given a string of KCL code or a path to a KCL project. Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point to a .kcl file or a directory containing a main.kcl file.

    Args:
        kcl_code (str | None): KCL code
        kcl_path (Path | str | None): KCL path, the path should point to a .kcl file or a directory containing a main.kcl file.

    Returns:
        tuple(bool, str): Returns True if the KCL code executed successfully and a success message, False otherwise and the error message.
    """
    logger.info("Executing KCL code")

    _check_kcl_code_or_path(kcl_code, kcl_path)

    try:
        if kcl_code:
            await kcl.execute_code(kcl_code)
        else:
            await kcl.execute(str(kcl_path))
        logger.info("KCL code executed successfully")
        return True, "KCL code executed successfully"
    except Exception as e:
        logger.info("Failed to execute KCL code: %s", e)
        return False, f"Failed to execute KCL code: {e}"


async def zoo_export_kcl(
    kcl_code: str | None = None,
    kcl_path: Path | str | None = None,
    export_path: Path | str | None = None,
    export_format: kcl.FileExportFormat | str | None = kcl.FileExportFormat.Step,
) -> Path:
    """Export KCL code to a CAD file. Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point to a .kcl file or a directory containing a main.kcl file.

    Args:
        kcl_code (str | None): KCL code
        kcl_path (Path | str | None): KCL path, the path should point to a .kcl file or a directory containing a main.kcl file.
        export_path (Path | str | None): path to save the step file, this should be a directory or a file with the appropriate extension. If no path is provided, a temporary file will be created.
        export_format (kcl.FileExportFormat | str | None): format to export the KCL code to. This should be one of 'fbx', 'glb', 'gltf', 'obj', 'ply', 'step', 'stl'. If no format is provided, the default is 'step'.

    Returns:
        Path: Return the path to the exported model if successful
    """

    logger.info("Exporting KCL to Step")

    _check_kcl_code_or_path(kcl_code, kcl_path)

    # check the export format
    if not export_format:
        logger.warning("No export format provided, defaulting to step")
        export_format = kcl.FileExportFormat.Step
    else:
        if export_format not in KCLExportFormat.formats.value.keys():
            logger.warning(
                "Invalid export format %s provided, defaulting to step", export_format
            )
            export_format = kcl.FileExportFormat.Step
        else:
            export_format = KCLExportFormat.formats.value[export_format]

    if export_path is None:
        logger.warning("No export path provided, creating a temporary file")
        export_path = await aiofiles.tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{str(export_format).split('.')[1].lower()}"
        )
        export_path = Path(export_path.name)
    else:
        export_path = Path(export_path)
        if export_path.suffix:
            ext = export_path.suffix.split(".")[1]
            if ext not in [i.value for i in FileExportFormat]:
                logger.warning(
                    "The provided export path does not have a valid extension, using a temporary file instead"
                )
                export_path = await aiofiles.tempfile.NamedTemporaryFile(
                    dir=export_path.parent.resolve(),
                    delete=False,
                    suffix=f".{str(export_format).split('.')[1].lower()}",
                )
            else:
                logger.warning("The provided export path is a file, overwriting")
        else:
            export_path = await aiofiles.tempfile.NamedTemporaryFile(
                dir=export_path.resolve(),
                delete=False,
                suffix=f".{str(export_format).split('.')[1].lower()}",
            )
            logger.info("Using provided export path: %s", str(export_path.name))

    async with aiofiles.open(export_path, "wb") as out:
        if kcl_code:
            logger.info("Exporting KCL code to %s", str(kcl_code))
            export_response = await kcl.execute_code_and_export(kcl_code, export_format)
        else:
            logger.info("Exporting KCL project to %s", str(kcl_path))
            assert kcl_path is not None  # _check_kcl_code_or_path ensures this
            kcl_path_resolved = Path(kcl_path)
            export_response = await kcl.execute_and_export(
                str(kcl_path_resolved.resolve()), export_format
            )
        await out.write(bytes(export_response[0].contents))

    logger.info("KCL exported successfully to %s", str(export_path))
    return Path(export_path)


async def zoo_format_kcl(
    kcl_code: str | None,
    kcl_path: Path | str | None,
) -> str | None:
    """Format KCL given a string of KCL code or a path to a KCL project. Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point to a .kcl file or a directory containing .kcl files.

    Args:
        kcl_code (str | None): KCL code to format.
        kcl_path (Path | str | None): KCL path, the path should point to a .kcl file or a directory containing a main.kcl file.

    Returns:
        str | None: Returns the formatted kcl code if the kcl_code is used otherwise returns None, the KCL in the kcl_path will be formatted in place
    """

    logger.info("Formatting the KCL")

    _check_kcl_code_or_path(kcl_code, kcl_path, require_main_file=False)

    try:
        if kcl_code:
            formatted_code = kcl.format(kcl_code)
            return formatted_code
        else:
            # _check_kcl_code_or_path ensures kcl_path is valid when kcl_code is None
            assert kcl_path is not None
            path = Path(kcl_path)
            if path.is_file():
                code = path.read_text()
                formatted = kcl.format(code)
                path.write_text(formatted)
            else:
                await kcl.format_dir(str(kcl_path))
            return None
    except Exception as e:
        logger.error(e)
        raise ZooMCPException(f"Failed to format the KCL: {e}")


def zoo_lint_and_fix_kcl(
    kcl_code: str | None,
    kcl_path: Path | str | None,
) -> tuple[str | None, list[str]]:
    """Lint and fix KCL given a string of KCL code or a path to a KCL project. Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point to a .kcl file or a directory containing .kcl files.

    Args:
        kcl_code (str | None): KCL code to lint and fix.
        kcl_path (Path | str | None): KCL path, the path should point to a .kcl file or a directory containing a main.kcl file.

    Returns:
        tuple[str | None, list[str]]: If kcl_code is provided, it returns a tuple of the fixed kcl code and a list of unfixed lints.
                                      If kcl_path is provided, it returns None and a list of unfixed lints for each file in the project.
    """

    logger.info("Linting and fixing the KCL")

    _check_kcl_code_or_path(kcl_code, kcl_path, require_main_file=False)

    try:
        if kcl_code:
            linted_kcl = cast(
                "FixedLintsProtocol",
                kcl.lint_and_fix_families(
                    kcl_code,
                    [kcl.FindingFamily.Correctness, kcl.FindingFamily.Simplify],
                ),
            )
            if len(linted_kcl.unfixed_lints) > 0:
                unfixed_lints = [
                    f"{lint.description}, {lint.finding.description}"
                    for lint in linted_kcl.unfixed_lints
                ]
            else:
                unfixed_lints = ["All lints fixed"]
            return linted_kcl.new_code, unfixed_lints
        else:
            # _check_kcl_code_or_path ensures kcl_path is valid when kcl_code is None
            assert kcl_path is not None
            kcl_path_resolved = Path(kcl_path)
            unfixed_lints = []
            for kcl_file in kcl_path_resolved.rglob("*.kcl"):
                linted_kcl = cast(
                    "FixedLintsProtocol",
                    kcl.lint_and_fix_families(
                        kcl_file.read_text(),
                        [kcl.FindingFamily.Correctness, kcl.FindingFamily.Simplify],
                    ),
                )
                kcl_file.write_text(linted_kcl.new_code)
                if len(linted_kcl.unfixed_lints) > 0:
                    unfixed_lints.extend(
                        [
                            f"In file {kcl_file.name}, {lint.description}, {lint.finding.description}"
                            for lint in linted_kcl.unfixed_lints
                        ]
                    )
                else:
                    unfixed_lints.append(f"In file {kcl_file.name}, All lints fixed")
            return None, unfixed_lints
    except Exception as e:
        logger.error(e)
        raise ZooMCPException(f"Failed to lint and fix the KCL: {e}")


def _format_constraint_status(status: kcl.SketchConstraintStatus) -> dict:
    """Format a single SketchConstraintStatus into a dict."""
    return {
        "name": status.name,
        "status": str(status.status).removeprefix("ConstraintKind."),
        "free_count": status.free_count,
        "conflict_count": status.conflict_count,
        "total_count": status.total_count,
    }


def _format_constraint_report(report: kcl.SketchConstraintReport) -> dict:
    """Format a SketchConstraintReport into a dict."""
    result: dict = {
        "fully_constrained": [
            _format_constraint_status(s) for s in report.fully_constrained
        ],
        "under_constrained": [
            _format_constraint_status(s) for s in report.under_constrained
        ],
        "over_constrained": [
            _format_constraint_status(s) for s in report.over_constrained
        ],
        "errors": [_format_constraint_status(s) for s in report.errors],
        "total_sketches": (
            len(report.fully_constrained)
            + len(report.under_constrained)
            + len(report.over_constrained)
            + len(report.errors)
        ),
        "kcl_executes_successfully": report.is_complete,
        "kcl_error": None,
    }
    if report.kcl_error is not None:
        result["kcl_error"] = {
            "phase": report.kcl_error.phase,
            "text": report.kcl_error.text,
        }
    return result


async def zoo_get_sketch_constraint_status(
    kcl_code: str | None = None,
    kcl_path: Path | str | None = None,
) -> dict:
    """Execute KCL and return a report of sketch constraint status. Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point to a .kcl file or a directory containing a main.kcl file.

    Args:
        kcl_code (str | None): KCL code to check constraints for.
        kcl_path (Path | str | None): KCL path, the path should point to a .kcl file or a directory containing a main.kcl file.

    Returns:
        dict: A report grouping sketches by constraint status (fully_constrained, under_constrained, over_constrained, errors). Also includes kcl_executes_successfully and kcl_error; when KCL parse/execution fails, kcl_executes_successfully is False and kcl_error contains phase and text describing the failure.
    """

    logger.info("Getting sketch constraint status")

    _check_kcl_code_or_path(kcl_code, kcl_path)

    try:
        if kcl_code:
            report = await kcl.get_sketch_constraint_status_code(kcl_code)
        else:
            assert kcl_path is not None
            report = await kcl.get_sketch_constraint_status(str(kcl_path))
        return _format_constraint_report(report)
    except Exception as e:
        logger.error(e)
        raise ZooMCPException(f"Failed to get sketch constraint status: {e}")


async def zoo_mock_execute_kcl(
    kcl_code: str | None = None,
    kcl_path: Path | str | None = None,
) -> tuple[bool, str]:
    """Mock execute KCL code given a string of KCL code or a path to a KCL project. Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point to a .kcl file or a directory containing a main.kcl file.

    Args:
        kcl_code (str | None): KCL code
        kcl_path (Path | str | None): KCL path, the path should point to a .kcl file or a directory containing a main.kcl file.

    Returns:
        tuple(bool, str): Returns True if the KCL code executed successfully and a success message, False otherwise and the error message.
    """
    logger.info("Executing KCL code")

    _check_kcl_code_or_path(kcl_code, kcl_path)

    try:
        if kcl_code:
            await kcl.mock_execute_code(kcl_code)
        else:
            await kcl.mock_execute(str(kcl_path))
        logger.info("KCL mock executed successfully")
        return True, "KCL code mock executed successfully"
    except Exception as e:
        logger.info("Failed to mock execute KCL code: %s", e)
        return False, f"Failed to mock execute KCL code: {e}"


def zoo_multiview_snapshot_of_cad(
    input_path: Path | str,
    padding: float = 0.1,
    max_image_dimension: int = 512,
    zoom: bool = True,
) -> bytes:
    """Save a multiview snapshot of a CAD file.

    Args:
        input_path (Path | str): Path to the CAD file to save a multiview snapshot. The file should be one of the supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp, .stl (case-insensitive)
        padding (float): The padding to apply to the snapshot. Default is 0.1.
        max_image_dimension (int): The maximum width or height of the returned image in pixels. Default is 512.
        zoom (bool): Whether to zoom-to-fit the model before each snapshot. Default is True.

    Returns:
        bytes or None: The JPEG image contents if successful
    """

    input_path = Path(input_path)

    # Connect to the websocket.
    with (
        kittycad_client.modeling.modeling_commands_ws(
            fps=30,
            post_effect=PostEffectType.SSAO,
            show_grid=False,
            unlocked_framerate=False,
            video_res_height=1024,
            video_res_width=1024,
            webrtc=False,
        ) as ws,
        open(input_path, "rb") as data,
    ):
        # Import files request must be sent as binary, because the file contents might be binary.
        import_id = ModelingCmdId(uuid4())

        input_ext = input_path.suffix.split(".")[1].lower()
        if input_ext not in SUPPORTED_EXTS:
            logger.error("The provided input path does not have a valid extension")
            raise ZooMCPException(
                "The provided input path does not have a valid extension"
            )

        input_format = _get_input_format(input_ext)
        if input_format is None:
            logger.error("The provided extension is not supported for import")
            raise ZooMCPException("The provided extension is not supported for import")

        ws.send_binary(
            WebSocketRequest(
                OptionModelingCmdReq(
                    cmd=ModelingCmd(
                        OptionImportFiles(
                            files=[ImportFile(data=data.read(), path=input_path.name)],
                            format=input_format,
                        )
                    ),
                    cmd_id=ModelingCmdId(import_id),
                )
            )
        )

        # Wait for the import to succeed.
        while True:
            message = ws.recv().model_dump()
            if message["request_id"] == import_id:
                break
        if message["success"] is not True:
            logger.error("Failed to import CAD file")
            raise ZooMCPException("Failed to import CAD file")
        object_id = message["resp"]["data"]["modeling_response"]["data"]["object_id"]

        # set camera to ortho
        ortho_cam_id = ModelingCmdId(uuid4())
        ws.send(
            WebSocketRequest(
                OptionModelingCmdReq(
                    cmd=ModelingCmd(OptionDefaultCameraSetOrthographic()),
                    cmd_id=ModelingCmdId(ortho_cam_id),
                )
            )
        )

        views = [
            OptionDefaultCameraLookAt(
                up=Point3d(x=0, y=0, z=1),
                vantage=Point3d(x=0, y=-1, z=0),
                center=Point3d(x=0, y=0, z=0),
            ),
            OptionDefaultCameraLookAt(
                up=Point3d(x=0, y=0, z=1),
                vantage=Point3d(x=1, y=0, z=0),
                center=Point3d(x=0, y=0, z=0),
            ),
            OptionDefaultCameraLookAt(
                up=Point3d(x=0, y=1, z=0),
                vantage=Point3d(x=0, y=0, z=1),
                center=Point3d(x=0, y=0, z=0),
            ),
            OptionViewIsometric(),
        ]

        jpeg_contents_list = []

        for view in views:
            # change camera look at
            camera_look_id = ModelingCmdId(uuid4())
            ws.send(
                WebSocketRequest(
                    OptionModelingCmdReq(
                        cmd=ModelingCmd(view),
                        cmd_id=ModelingCmdId(camera_look_id),
                    )
                )
            )

            if zoom:
                focus_id = ModelingCmdId(uuid4())
                ws.send(
                    WebSocketRequest(
                        OptionModelingCmdReq(
                            cmd=ModelingCmd(
                                OptionZoomToFit(object_ids=[object_id], padding=padding)
                            ),
                            cmd_id=ModelingCmdId(focus_id),
                        )
                    )
                )

                # Wait for success message.
                while True:
                    message = ws.recv().model_dump()
                    if message["request_id"] == focus_id:
                        break
                if message["success"] is not True:
                    logger.error("Failed to move camera to fit object")
                    raise ZooMCPException("Failed to move camera to fit object")

            # Take a snapshot as a JPEG.
            snapshot_id = ModelingCmdId(uuid4())
            ws.send(
                WebSocketRequest(
                    OptionModelingCmdReq(
                        cmd=ModelingCmd(OptionTakeSnapshot(format=ImageFormat.JPEG)),
                        cmd_id=ModelingCmdId(snapshot_id),
                    )
                )
            )

            # Wait for success message.
            while True:
                message = ws.recv().model_dump()
                if message["request_id"] == snapshot_id:
                    break
            if message["success"] is not True:
                logger.error("Failed to capture snapshot")
                raise ZooMCPException("Failed to capture snapshot")
            jpeg_contents = message["resp"]["data"]["modeling_response"]["data"][
                "contents"
            ]

            jpeg_contents_list.append(jpeg_contents)

        collage = create_image_collage(jpeg_contents_list)

        return resize_image(collage, max_image_dimension)


def zoo_multi_isometric_snapshot_of_cad(
    input_path: Path | str,
    padding: float = 0.1,
    max_image_dimension: int = 512,
    zoom: bool = True,
) -> bytes:
    """Save a multi-isometric snapshot of a CAD file showing 4 isometric views.

    Args:
        input_path (Path | str): Path to the CAD file to save a multi-isometric snapshot. The file should be one of the supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp, .stl (case-insensitive)
        padding (float): The padding to apply to the snapshot. Default is 0.1.
        max_image_dimension (int): The maximum width or height of the returned image in pixels. Default is 512.
        zoom (bool): Whether to zoom-to-fit the model before each snapshot. Default is True.

    Returns:
        bytes or None: The JPEG image contents if successful
    """

    input_path = Path(input_path)

    # Connect to the websocket.
    with (
        kittycad_client.modeling.modeling_commands_ws(
            fps=30,
            post_effect=PostEffectType.SSAO,
            show_grid=False,
            unlocked_framerate=False,
            video_res_height=1024,
            video_res_width=1024,
            webrtc=False,
        ) as ws,
        open(input_path, "rb") as data,
    ):
        # Import files request must be sent as binary, because the file contents might be binary.
        import_id = ModelingCmdId(uuid4())

        input_ext = input_path.suffix.split(".")[1].lower()
        if input_ext not in SUPPORTED_EXTS:
            logger.error("The provided input path does not have a valid extension")
            raise ZooMCPException(
                "The provided input path does not have a valid extension"
            )

        input_format = _get_input_format(input_ext)
        if input_format is None:
            logger.error("The provided extension is not supported for import")
            raise ZooMCPException("The provided extension is not supported for import")

        ws.send_binary(
            WebSocketRequest(
                OptionModelingCmdReq(
                    cmd=ModelingCmd(
                        OptionImportFiles(
                            files=[ImportFile(data=data.read(), path=input_path.name)],
                            format=input_format,
                        )
                    ),
                    cmd_id=ModelingCmdId(import_id),
                )
            )
        )

        # Wait for the import to succeed.
        while True:
            message = ws.recv().model_dump()
            if message["request_id"] == import_id:
                break
        if message["success"] is not True:
            logger.error("Failed to import CAD file")
            raise ZooMCPException("Failed to import CAD file")
        object_id = message["resp"]["data"]["modeling_response"]["data"]["object_id"]

        # set camera to ortho
        ortho_cam_id = ModelingCmdId(uuid4())
        ws.send(
            WebSocketRequest(
                OptionModelingCmdReq(
                    cmd=ModelingCmd(OptionDefaultCameraSetOrthographic()),
                    cmd_id=ModelingCmdId(ortho_cam_id),
                )
            )
        )

        # Use 4 isometric views from different corners
        views = [
            CameraView.to_kittycad_camera(
                CameraView.views.value["isometric_front_right"]
            ),
            CameraView.to_kittycad_camera(
                CameraView.views.value["isometric_front_left"]
            ),
            CameraView.to_kittycad_camera(
                CameraView.views.value["isometric_back_right"]
            ),
            CameraView.to_kittycad_camera(
                CameraView.views.value["isometric_back_left"]
            ),
        ]

        jpeg_contents_list = []

        for view in views:
            # change camera look at
            camera_look_id = ModelingCmdId(uuid4())
            ws.send(
                WebSocketRequest(
                    OptionModelingCmdReq(
                        cmd=ModelingCmd(view),
                        cmd_id=ModelingCmdId(camera_look_id),
                    )
                )
            )

            if zoom:
                focus_id = ModelingCmdId(uuid4())
                ws.send(
                    WebSocketRequest(
                        OptionModelingCmdReq(
                            cmd=ModelingCmd(
                                OptionZoomToFit(object_ids=[object_id], padding=padding)
                            ),
                            cmd_id=ModelingCmdId(focus_id),
                        )
                    )
                )

                # Wait for success message.
                while True:
                    message = ws.recv().model_dump()
                    if message["request_id"] == focus_id:
                        break
                if message["success"] is not True:
                    logger.error("Failed to move camera to fit object")
                    raise ZooMCPException("Failed to move camera to fit object")

            # Take a snapshot as a JPEG.
            snapshot_id = ModelingCmdId(uuid4())
            ws.send(
                WebSocketRequest(
                    OptionModelingCmdReq(
                        cmd=ModelingCmd(OptionTakeSnapshot(format=ImageFormat.JPEG)),
                        cmd_id=ModelingCmdId(snapshot_id),
                    )
                )
            )

            # Wait for success message.
            while True:
                message = ws.recv().model_dump()
                if message["request_id"] == snapshot_id:
                    break
            if message["success"] is not True:
                logger.error("Failed to capture snapshot")
                raise ZooMCPException("Failed to capture snapshot")
            jpeg_contents = message["resp"]["data"]["modeling_response"]["data"][
                "contents"
            ]

            jpeg_contents_list.append(jpeg_contents)

        collage = create_image_collage(jpeg_contents_list)

        return resize_image(collage, max_image_dimension)


async def zoo_multi_isometric_snapshot_of_kcl(
    kcl_code: str | None,
    kcl_path: Path | str | None,
    padding: float = 0.1,
    max_image_dimension: int = 512,
    zoom: bool = True,
) -> bytes:
    """Execute the KCL code and save a multi-isometric snapshot showing 4 isometric views. Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point to a .kcl file or a directory containing a main.kcl file.

    Args:
        kcl_code (str | None): KCL code
        kcl_path (Path | str | None): KCL path, the path should point to a .kcl file or a directory containing a main.kcl file.
        padding (float): The padding to apply to the snapshot. Default is 0.1.
        max_image_dimension (int): The maximum width or height of the returned image in pixels. Default is 512.
        zoom (bool): Whether to zoom-to-fit the model before each snapshot. Default is True.

    Returns:
        bytes or None: The JPEG image contents if successful
    """

    logger.info("Taking a multi-isometric snapshot of KCL")

    _check_kcl_code_or_path(kcl_code, kcl_path)

    try:
        # Use 4 isometric views from different corners
        camera_list = [
            CameraView.to_kcl_camera(CameraView.views.value["isometric_front_right"]),
            CameraView.to_kcl_camera(CameraView.views.value["isometric_front_left"]),
            CameraView.to_kcl_camera(CameraView.views.value["isometric_back_right"]),
            CameraView.to_kcl_camera(CameraView.views.value["isometric_back_left"]),
        ]

        views = [
            kcl.SnapshotOptions(camera=camera, padding=padding)
            for camera in camera_list
        ]

        if kcl_code:
            # The stub says list[list[int]] but it actually returns list[bytes]
            jpeg_contents_list: list[bytes] = cast(
                list[bytes],
                cast(
                    object,
                    await kcl.execute_code_and_snapshot_views(
                        kcl_code,
                        kcl.ImageFormat.Jpeg,
                        snapshot_options=views,
                        zoom=zoom,
                    ),
                ),
            )
        else:
            # _check_kcl_code_or_path ensures kcl_path is valid when kcl_code is None
            assert kcl_path is not None
            kcl_path_resolved = Path(kcl_path)
            # The stub says list[list[int]] but it actually returns list[bytes]
            jpeg_contents_list = cast(
                list[bytes],
                cast(
                    object,
                    await kcl.execute_and_snapshot_views(
                        str(kcl_path_resolved),
                        kcl.ImageFormat.Jpeg,
                        snapshot_options=views,
                        zoom=zoom,
                    ),
                ),
            )

        collage = create_image_collage(jpeg_contents_list)

        return resize_image(collage, max_image_dimension)

    except Exception as e:
        logger.error("Failed to take multi-isometric snapshot: %s", e)
        raise ZooMCPException(f"Failed to take multi-isometric snapshot: {e}")


async def zoo_multiview_snapshot_of_kcl(
    kcl_code: str | None,
    kcl_path: Path | str | None,
    padding: float = 0.1,
    max_image_dimension: int = 512,
    zoom: bool = True,
) -> bytes:
    """Execute the KCL code and save a multiview snapshot of the resulting CAD model. Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point to a .kcl file or a directory containing a main.kcl file.

    Args:
        kcl_code (str | None): KCL code
        kcl_path (Path | str | None): KCL path, the path should point to a .kcl file or a directory containing a main.kcl file.
        padding (float): The padding to apply to the snapshot. Default is 0.1.
        max_image_dimension (int): The maximum width or height of the returned image in pixels. Default is 512.
        zoom (bool): Whether to zoom-to-fit the model before each snapshot. Default is True.

    Returns:
        bytes or None: The JPEG image contents if successful
    """

    logger.info("Taking a multiview snapshot of KCL")

    _check_kcl_code_or_path(kcl_code, kcl_path)

    try:
        # None in the camera list means isometric view
        # https://github.com/KittyCAD/modeling-app/blob/main/rust/kcl-python-bindings/tests/tests.py#L192
        camera_list = [
            kcl.CameraLookAt(
                up=kcl.Point3d(x=0, y=0, z=1),
                vantage=kcl.Point3d(x=0, y=-1, z=0),
                center=kcl.Point3d(x=0, y=0, z=0),
            ),
            kcl.CameraLookAt(
                up=kcl.Point3d(x=0, y=0, z=1),
                vantage=kcl.Point3d(x=1, y=0, z=0),
                center=kcl.Point3d(x=0, y=0, z=0),
            ),
            kcl.CameraLookAt(
                up=kcl.Point3d(x=0, y=1, z=0),
                vantage=kcl.Point3d(x=0, y=0, z=1),
                center=kcl.Point3d(x=0, y=0, z=0),
            ),
            None,
        ]

        views = [
            kcl.SnapshotOptions(camera=camera, padding=padding)
            for camera in camera_list
        ]

        if kcl_code:
            # The stub says list[list[int]] but it actually returns list[bytes]
            jpeg_contents_list: list[bytes] = cast(
                list[bytes],
                cast(
                    object,
                    await kcl.execute_code_and_snapshot_views(
                        kcl_code,
                        kcl.ImageFormat.Jpeg,
                        snapshot_options=views,
                        zoom=zoom,
                    ),
                ),
            )
        else:
            # _check_kcl_code_or_path ensures kcl_path is valid when kcl_code is None
            assert kcl_path is not None
            kcl_path_resolved = Path(kcl_path)
            # The stub says list[list[int]] but it actually returns list[bytes]
            jpeg_contents_list = cast(
                list[bytes],
                cast(
                    object,
                    await kcl.execute_and_snapshot_views(
                        str(kcl_path_resolved),
                        kcl.ImageFormat.Jpeg,
                        snapshot_options=views,
                        zoom=zoom,
                    ),
                ),
            )

        collage = create_image_collage(jpeg_contents_list)

        return resize_image(collage, max_image_dimension)

    except Exception as e:
        logger.error("Failed to take multiview snapshot: %s", e)
        raise ZooMCPException(f"Failed to take multiview snapshot: {e}")


def zoo_snapshot_of_cad(
    input_path: Path | str,
    camera: OptionDefaultCameraLookAt | OptionViewIsometric | None = None,
    padding: float = 0.1,
    max_image_dimension: int = 512,
    zoom: bool = True,
) -> bytes:
    """Save a single view snapshot of a CAD file.

    Args:
        input_path (Path | str): Path to the CAD file to save a snapshot. The file should be one of the supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp, .stl (case-insensitive)
        camera (OptionDefaultCameraLookAt | OptionViewIsometric | None): The camera to use for the snapshot. If None, a default camera (isometric) will be used.
        padding (float): The padding to apply to the snapshot. Default is 0.1.
        max_image_dimension (int): The maximum width or height of the returned image in pixels. Default is 512.
        zoom (bool): Whether to zoom-to-fit the model before the snapshot. Default is True.

    Returns:
        bytes or None: The JPEG image contents if successful
    """

    input_path = Path(input_path)

    # Connect to the websocket.
    with (
        kittycad_client.modeling.modeling_commands_ws(
            fps=30,
            post_effect=PostEffectType.SSAO,
            show_grid=False,
            unlocked_framerate=False,
            video_res_height=1024,
            video_res_width=1024,
            webrtc=False,
        ) as ws,
        open(input_path, "rb") as data,
    ):
        # Import files request must be sent as binary, because the file contents might be binary.
        import_id = ModelingCmdId(uuid4())

        input_ext = input_path.suffix.split(".")[1].lower()
        if input_ext not in SUPPORTED_EXTS:
            logger.error("The provided input path does not have a valid extension")
            raise ZooMCPException(
                "The provided input path does not have a valid extension"
            )

        input_format = _get_input_format(input_ext)
        if input_format is None:
            logger.error("The provided extension is not supported for import")
            raise ZooMCPException("The provided extension is not supported for import")

        ws.send_binary(
            WebSocketRequest(
                OptionModelingCmdReq(
                    cmd=ModelingCmd(
                        OptionImportFiles(
                            files=[ImportFile(data=data.read(), path=input_path.name)],
                            format=input_format,
                        )
                    ),
                    cmd_id=ModelingCmdId(import_id),
                )
            )
        )

        # Wait for the import to succeed.
        while True:
            message = ws.recv().model_dump()
            if message["request_id"] == import_id:
                break
        if message["success"] is not True:
            raise ZooMCPException("Failed to import CAD file")
        object_id = message["resp"]["data"]["modeling_response"]["data"]["object_id"]

        # set camera to ortho
        ortho_cam_id = ModelingCmdId(uuid4())
        ws.send(
            WebSocketRequest(
                OptionModelingCmdReq(
                    cmd=ModelingCmd(OptionDefaultCameraSetOrthographic()),
                    cmd_id=ModelingCmdId(ortho_cam_id),
                )
            )
        )

        camera_look_id = ModelingCmdId(uuid4())
        if camera is None:
            camera = OptionViewIsometric()
        ws.send(
            WebSocketRequest(
                OptionModelingCmdReq(
                    cmd=ModelingCmd(camera),
                    cmd_id=ModelingCmdId(camera_look_id),
                )
            )
        )

        if zoom:
            focus_id = ModelingCmdId(uuid4())
            ws.send(
                WebSocketRequest(
                    OptionModelingCmdReq(
                        cmd=ModelingCmd(
                            OptionZoomToFit(object_ids=[object_id], padding=padding)
                        ),
                        cmd_id=ModelingCmdId(focus_id),
                    )
                )
            )

            # Wait for success message.
            while True:
                message = ws.recv().model_dump()
                if message["request_id"] == focus_id:
                    break
            if message["success"] is not True:
                raise ZooMCPException("Failed to zoom to fit on CAD file")

        # Take a snapshot as a JPEG.
        snapshot_id = ModelingCmdId(uuid4())
        ws.send(
            WebSocketRequest(
                OptionModelingCmdReq(
                    cmd=ModelingCmd(OptionTakeSnapshot(format=ImageFormat.JPEG)),
                    cmd_id=ModelingCmdId(snapshot_id),
                )
            )
        )

        # Wait for success message.
        while True:
            message = ws.recv().model_dump()
            if message["request_id"] == snapshot_id:
                break
        if message["success"] is not True:
            raise ZooMCPException("Failed to take snapshot of CAD file")
        jpeg_contents = message["resp"]["data"]["modeling_response"]["data"]["contents"]

        return resize_image(jpeg_contents, max_image_dimension)


async def zoo_snapshot_of_kcl(
    kcl_code: str | None,
    kcl_path: Path | str | None,
    camera: kcl.CameraLookAt | None = None,
    padding: float = 0.1,
    max_image_dimension: int = 512,
    zoom: bool = True,
) -> bytes:
    """Execute the KCL code and save a single view snapshot of the resulting CAD model. Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point to a .kcl file or a directory containing a main.kcl file.

    Args:
        kcl_code (str | None): KCL code
        kcl_path (Path | str | None): KCL path, the path should point to a .kcl file or a directory containing a main.kcl file.
        camera (kcl.CameraLookAt | None): The camera to use for the snapshot. If None, a default camera (isometric) will be used.
        padding (float): The padding to apply to the snapshot. Default is 0.1.
        max_image_dimension (int): The maximum width or height of the returned image in pixels. Default is 512.
        zoom (bool): Whether to zoom-to-fit the model before the snapshot. Default is True.

    Returns:
        bytes or None: The JPEG image contents if successful
    """

    logger.info("Taking a snapshot of KCL")

    _check_kcl_code_or_path(kcl_code, kcl_path)

    view = kcl.SnapshotOptions(camera=camera, padding=padding)

    if kcl_code:
        # The stub says list[list[int]] but it actually returns list[bytes]
        jpeg_contents_list: list[bytes] = cast(
            list[bytes],
            cast(
                object,
                await kcl.execute_code_and_snapshot_views(
                    kcl_code,
                    kcl.ImageFormat.Jpeg,
                    snapshot_options=[view],
                    zoom=zoom,
                ),
            ),
        )
    else:
        # _check_kcl_code_or_path ensures kcl_path is valid when kcl_code is None
        assert kcl_path is not None
        kcl_path_resolved = Path(kcl_path)
        # The stub says list[list[int]] but it actually returns list[bytes]
        jpeg_contents_list = cast(
            list[bytes],
            cast(
                object,
                await kcl.execute_and_snapshot_views(
                    str(kcl_path_resolved),
                    kcl.ImageFormat.Jpeg,
                    snapshot_options=[view],
                    zoom=zoom,
                ),
            ),
        )

    return resize_image(jpeg_contents_list[0], max_image_dimension)


def zoo_list_org_datasets() -> list[dict[str, str]]:
    """List all datasets visible to the org tied to the current ZOO_API_TOKEN.

    Returns:
        A list of {"id": <uuid str>, "name": <str>} entries, possibly empty.
    """
    logger.info("Listing org datasets")
    try:
        datasets = list(
            kittycad_client.orgs.list_org_datasets(limit=None, page_token=None)
        )
    except KittyCADClientError as exc:
        if exc.status_code == 404:
            return []
        raise ZooMCPException(f"Failed to list org datasets: {exc}") from exc

    return [{"id": str(d.id), "name": d.name} for d in datasets]


def zoo_search_org_dataset_semantic(
    dataset_id: str,
    query: str,
    limit: int | None = None,
) -> list[dict]:
    """Semantic-search a dataset and return the top matching chunks.

    Args:
        dataset_id: UUID of the dataset (as returned by zoo_list_org_datasets).
        query: Natural-language query to embed and search with.
        limit: Optional max number of matches to return.

    Returns:
        A list of dicts with keys: source_file_path, content, similarity,
        chunk_index, conversion_id.
    """
    logger.info(
        "Semantic search in dataset %s for query of length %d", dataset_id, len(query)
    )
    try:
        matches = kittycad_client.orgs.search_org_dataset_semantic(
            id=Uuid(dataset_id), q=query, limit=limit
        )
    except KittyCADClientError as exc:
        raise ZooMCPException(f"Failed to search dataset {dataset_id}: {exc}") from exc

    return [
        {
            "source_file_path": m.source_file_path,
            "content": m.content,
            "similarity": m.similarity,
            "chunk_index": m.chunk_index,
            "conversion_id": str(m.conversion_id),
        }
        for m in matches
    ]
