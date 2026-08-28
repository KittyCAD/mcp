import asyncio
import io
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, TypeVar, cast
from urllib.parse import urlencode, urlsplit, urlunsplit
from uuid import uuid4

import aiofiles
import bson
import kcl
import trimesh
from kittycad import AsyncKittyCAD

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
    FaceGetCenter,
    FaceGetGradient,
    FaceGetPosition,
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
    Point2d,
    Point3d,
    PostEffectType,
    System,
    UnitArea,
    UnitDensity,
    UnitLength,
    UnitMass,
    UnitVolume,
    WebSocketRequest,
    WebSocketResponse,
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
    OptionEdgeLinesVisible,
    OptionFaceGetCenter,
    OptionFaceGetGradient,
    OptionFaceGetPosition,
    OptionImportFiles,
    OptionTakeSnapshot,
    OptionViewIsometric,
    OptionZoomToFit,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionDefaultCameraLookAt as ResponseDefaultCameraLookAt,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionDefaultCameraSetOrthographic as ResponseDefaultCameraSetOrthographic,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionEdgeLinesVisible as ResponseEdgeLinesVisible,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionFaceGetCenter as ResponseFaceGetCenter,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionFaceGetGradient as ResponseFaceGetGradient,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionFaceGetPosition as ResponseFaceGetPosition,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionImportFiles as ResponseImportFiles,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionTakeSnapshot as ResponseTakeSnapshot,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionViewIsometric as ResponseViewIsometric,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionZoomToFit as ResponseZoomToFit,
)
from kittycad.models.ok_web_socket_response_data import OptionModeling
from kittycad.models.success_web_socket_response import SuccessWebSocketResponse
from kittycad.models.uuid import Uuid
from kittycad.models.web_socket_request import OptionModelingCmdReq
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import WebSocketException

from zoo_mcp import (
    ZooMCPException,
    ZooMCPTimeoutError,
    ctx,
    logger,
)
from zoo_mcp.utils.image_utils import create_image_collage, resize_image

SUPPORTED_EXTS = {x.value.lower() for x in FileImportFormat} | {"stp"}

# The wall-clock budget for one modeling tool call, measured across every
# websocket send and read. It has to stay under the reconnect budget of
# the conversations driving these tools, so a stalled engine surfaces as a
# retryable error instead of an abandoned request.
MODELING_COMMAND_TIMEOUT = 300.0

# Map alternative extensions to their canonical FileImportFormat values
_EXT_ALIASES = {
    "stp": "step",
}


def load_kcl_project(path: Path | str) -> tuple[str, list[dict[str, str | list[int]]]]:
    """Load a KCL project into the shape expected by exec_kcl_project."""
    path = Path(path).resolve()
    root = path if path.is_dir() else path.parent
    entrypoint = "main.kcl" if path.is_dir() else path.name

    files: list[dict[str, str | list[int]]] = [
        {
            "path": file.relative_to(root).as_posix(),
            "contents": list(file.read_bytes()),
        }
        for file in sorted(root.rglob("*"))
        if file.is_file()
    ]

    return entrypoint, files


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
    "mm3": kcl.UnitVolume.CubicMillimeters,
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


# KCL engine connections can hang up transiently. The kcl bindings flag such
# errors via ``KclError.is_retryable()`` so callers can retry them. Mirror the
# retry behavior the bindings' own tests use.
MAX_EXECUTION_ATTEMPTS = 3

# The native zoo-kcl client supplies its bearer token on the WebSocket HTTP
# upgrade. The API can nevertheless intermittently report that this particular
# socket did not receive the header. zoo-kcl does not currently classify that
# response as retryable, but a fresh execution creates a fresh authenticated
# socket and recovers. Match the server's distinctive instruction rather than
# broad authentication text so invalid or expired credentials still fail fast.
_TRANSIENT_WEBSOCKET_AUTH_ERROR_MARKERS = (
    "Authorization",
    "Bearer <token>",
    "over this websocket",
)

_KclCoro = Callable[..., Awaitable[_T]]


def _is_retryable_execution_error(error: Exception) -> bool:
    """Return whether a KCL execution should be retried on a fresh socket."""
    message = str(error)
    if all(marker in message for marker in _TRANSIENT_WEBSOCKET_AUTH_ERROR_MARKERS):
        return True

    is_retryable = getattr(error, "is_retryable", None)
    return callable(is_retryable) and is_retryable()


async def _execute_with_retries(
    async_fn: _KclCoro[_T], *args: object, **kwargs: object
) -> _T:
    """Await a KCL execution coroutine, retrying on retryable engine errors.

    The kcl bindings raise ``kcl.KclError`` for execution failures and expose
    ``is_retryable()`` so transient errors (e.g. an engine hangup) can be
    retried instead of bubbling up. Non-retryable errors are re-raised
    immediately.

    Args:
        async_fn: The kcl coroutine function to call (e.g. ``kcl.execute_code``).
        *args: Positional arguments forwarded to ``async_fn``.
        **kwargs: Keyword arguments forwarded to ``async_fn``.

    Returns:
        Whatever ``async_fn`` returns on success.
    """
    retries_remaining = MAX_EXECUTION_ATTEMPTS - 1
    while True:
        try:
            return await async_fn(*args, **kwargs)
        except Exception as error:
            if retries_remaining > 0 and _is_retryable_execution_error(error):
                logger.warning(
                    "Retryable KCL execution error, retrying (%d attempt(s) left): %s",
                    retries_remaining,
                    error,
                )
                retries_remaining -= 1
                continue
            raise


# Issue severities surfaced from an execution outcome, in descending order of
# severity. Each entry maps a ``kcl.CompilationIssue`` predicate to the label
# used when rendering that issue's report. ``is_fatal`` is checked before
# ``is_err`` because a fatal issue may also report as an error.
_EXECUTION_ISSUE_SEVERITIES = (
    ("fatal", "is_fatal"),
    ("error", "is_err"),
    ("warning", "is_warning"),
)


def _format_execution_issues(outcome: "kcl.ExecOutcome") -> dict[str, list[str]]:
    """Render compilation issues from an execution outcome, grouped by severity.

    ``kcl.execute`` / ``kcl.execute_code`` return an ``ExecOutcome`` whose
    ``issues()`` may include warning-, error-, and fatal-level
    ``CompilationIssue``s (e.g. a CSG subtract with no overlap surfaces as a
    warning). Each issue is rendered to a miette report string with the
    relevant source snippet and bucketed by its severity.

    Args:
        outcome: The outcome returned by a kcl execution call.

    Returns:
        A mapping of severity label (``"fatal"``, ``"error"``, ``"warning"``)
        to the rendered reports at that level. Severity levels with no issues
        are omitted.
    """
    issues: dict[str, list[str]] = {}
    for issue in outcome.issues():
        for severity, predicate in _EXECUTION_ISSUE_SEVERITIES:
            if getattr(issue, predicate)():
                issues.setdefault(severity, []).append(outcome.report(issue))
                break
    return issues


def _execution_issues_message(issues: dict[str, list[str]]) -> str:
    """Render grouped KCL execution issues using the public tool message format."""
    sections = [
        f"{label}:\n\n" + "\n\n".join(issues[severity])
        for severity, label in (
            ("fatal", "Fatal issues"),
            ("error", "Errors"),
            ("warning", "Warnings"),
        )
        if severity in issues
    ]
    return "KCL code execution completed with the following issues:\n\n" + "\n\n".join(
        sections
    )


class KCLExportFormat(Enum):
    formats = {  # noqa: RUF012
        "fbx": kcl.FileExportFormat.Fbx,
        "gltf": kcl.FileExportFormat.Gltf,
        "glb": kcl.FileExportFormat.Glb,
        "obj": kcl.FileExportFormat.Obj,
        "ply": kcl.FileExportFormat.Ply,
        "step": kcl.FileExportFormat.Step,
        "stl": kcl.FileExportFormat.Stl,
    }


class CameraView(Enum):
    views = {  # noqa: RUF012
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
            "vantage": [1.0, 1.0, 1.0],
            "center": [0.0, 0.0, 0.0],
        },
        "isometric_back_left": {
            "up": [0.0, 0.0, 1.0],
            "vantage": [-1.0, 1.0, 1.0],
            "center": [0.0, 0.0, 0.0],
        },
    }

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
                y=view["vantage"][1],
                z=view["vantage"][2],
            ),
            center=Point3d(
                x=view["center"][0],
                y=view["center"][1],
                z=view["center"][2],
            ),
        )


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

    async with AsyncKittyCAD(verify_ssl=ctx) as client:
        result = await client.file.create_file_center_of_mass(
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

    async with AsyncKittyCAD(verify_ssl=ctx) as client:
        result = await client.file.create_file_mass(
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

    async with AsyncKittyCAD(verify_ssl=ctx) as client:
        result = await client.file.create_file_surface_area(
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
        unit_vol (str): The unit volume to return. This should be one of 'cm3', 'ft3', 'in3', 'm3', 'mm3', 'yd3', 'usfloz', 'usgal', 'l', 'ml'.

    Returns:
        float: If the volume of the file can be calculated, return the volume in the requested unit
    """

    file_path = Path(file_path)

    logger.info("Calculating volume for %s", str(file_path.resolve()))

    async with aiofiles.open(file_path, "rb") as inp:
        data = await inp.read()

    src_format = FileImportFormat(_normalize_ext(file_path.suffix.split(".")[1]))

    async with AsyncKittyCAD(verify_ssl=ctx) as client:
        result = await client.file.create_file_volume(
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


async def zoo_import_cad_file(session_id: str, input_file: Path | str) -> str:
    """Import a CAD file into the scene and return the imported object's id.

    Sent as a binary frame rather than through ``_send_modeling_command``
    because the file contents are binary in MsgPack encoding.
    """
    input_file = Path(input_file)

    input_ext = input_file.suffix.split(".")[-1].lower()
    if input_ext not in SUPPORTED_EXTS:
        raise ZooMCPException(
            f"'{input_file.name}' does not have a supported CAD extension; "
            f"expected one of {sorted(SUPPORTED_EXTS)}"
        )

    input_format = _get_input_format(input_ext)
    if input_format is None:
        raise ZooMCPException(f"'{input_ext}' files cannot be imported")

    command_id = ModelingCmdId(uuid4())
    # Extension and byte size only; the file contents are customer data.
    file_size = input_file.stat().st_size
    deadline = _Deadline()

    def log_stage(stage: str, outcome: str) -> None:
        logger.info(
            "CAD import %s: session=%s request_id=%s ext=%s bytes=%d "
            "elapsed=%.3fs outcome=%s",
            stage,
            session_id,
            command_id,
            input_ext,
            file_size,
            deadline.elapsed,
            outcome,
        )

    async with aiofiles.open(input_file, "rb") as data:
        contents = await data.read()

    async with _modeling_websocket(session_id) as ws:
        request = WebSocketRequest(
            OptionModelingCmdReq(
                cmd=ModelingCmd(
                    OptionImportFiles(
                        files=[ImportFile(data=contents, path=input_file.name)],
                        format=input_format,
                    )
                ),
                cmd_id=command_id,
            )
        )
        try:
            await _send_modeling_frame(
                ws,
                bson.encode(request.model_dump(exclude_none=True)),
                deadline,
                "CAD file import",
            )
            log_stage("sent", "awaiting-engine-response")
            response = await _await_modeling_response(
                ws,
                command_id,
                ResponseImportFiles,
                "CAD file import",
                deadline,
            )
        except ZooMCPTimeoutError:
            log_stage("finished", "timeout")
            raise
        except ZooMCPException as error:
            log_stage("finished", f"failed ({error})")
            raise

        log_stage("finished", "imported")
        return response.data.object_id


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
        unit_vol (str): The unit of volume. One of 'cm3', 'ft3', 'in3', 'm3', 'mm3', 'yd3', 'usfloz', 'usgal', 'l', 'ml'.

    Returns:
        dict: A dictionary with keys 'volume', 'mass', 'surface_area', 'center_of_mass', and 'bounding_box'.
    """
    file_path = Path(file_path)

    logger.info("Calculating physical properties for %s", str(file_path.resolve()))

    async with aiofiles.open(file_path, "rb") as inp:
        data = await inp.read()

    normalized_ext = _normalize_ext(file_path.suffix.split(".")[1])
    src_format = FileImportFormat(normalized_ext)

    async with AsyncKittyCAD(verify_ssl=ctx) as client:
        volume_result = await client.file.create_file_volume(
            output_unit=UnitVolume(unit_vol),
            src_format=src_format,
            body=data,
        )
        if not isinstance(volume_result, FileVolume) or volume_result.volume is None:
            raise ZooMCPException("Failed to calculate volume")

        mass_result = await client.file.create_file_mass(
            output_unit=UnitMass(unit_mass),
            src_format=src_format,
            body=data,
            material_density_unit=UnitDensity(unit_density),
            material_density=density,
        )
        if not isinstance(mass_result, FileMass) or mass_result.mass is None:
            raise ZooMCPException("Failed to calculate mass")

        sa_result = await client.file.create_file_surface_area(
            output_unit=UnitArea(unit_area),
            src_format=src_format,
            body=data,
        )
        if not isinstance(sa_result, FileSurfaceArea) or sa_result.surface_area is None:
            raise ZooMCPException("Failed to calculate surface area")

        com_result = await client.file.create_file_center_of_mass(
            src_format=src_format,
            body=data,
            output_unit=UnitLength(unit_length),
        )
        if (
            not isinstance(com_result, FileCenterOfMass)
            or com_result.center_of_mass is None
        ):
            raise ZooMCPException("Failed to calculate center of mass")

        if normalized_ext == "stl":
            stl_data = data
        else:
            stl_result = await client.file.create_file_conversion(
                src_format=src_format,
                output_format=FileExportFormat.STL,
                body=data,
            )
            if not isinstance(stl_result, FileConversion):
                raise ZooMCPException(
                    "Failed to convert file for bounding box calculation"
                )
            if stl_result.outputs is None or len(stl_result.outputs) == 0:
                raise ZooMCPException(
                    "Failed to convert file for bounding box calculation, no output"
                )
            stl_data = next(iter(stl_result.outputs.values()))

    bbox = await asyncio.to_thread(_compute_stl_bounding_box, stl_data)

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
        unit_vol (str): The unit of volume. One of 'cm3', 'ft3', 'in3', 'm3', 'mm3', 'yd3', 'usfloz', 'usgal', 'l', 'ml'.

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
        response = await _execute_with_retries(
            kcl.execute_code_and_measure, kcl_code, request
        )
    else:
        response = await _execute_with_retries(
            kcl.execute_and_measure, str(kcl_path), request
        )

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
        response = await _execute_with_retries(
            kcl.execute_code_and_bounding_box,
            kcl_code,
            output_unit=_parse_unit(unit_length, UNIT_LENGTH_MAP, "unit_length"),
        )
    else:
        response = await _execute_with_retries(
            kcl.execute_and_bounding_box,
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
        return await asyncio.to_thread(_compute_stl_bounding_box, data)

    src_format = FileImportFormat(normalized_ext)

    # Convert to STL to get mesh data for bounding box computation
    async with AsyncKittyCAD(verify_ssl=ctx) as client:
        stl_result = await client.file.create_file_conversion(
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

    stl_data = next(iter(stl_result.outputs.values()))

    return await asyncio.to_thread(_compute_stl_bounding_box, stl_data)


async def zoo_convert_cad_file(
    input_file: Path | str,
    export_path: Path | str | None = None,
    export_format: FileExportFormat | str | None = FileExportFormat.STEP,
) -> Path:
    """Convert a cad file to another cad file

    Args:
        input_file (Path | str): path to the CAD file to convert. The file should be one of the supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp, .stl (case-insensitive)
        export_path (Path | str | None): The path to save the cad file. If no path is provided, a temporary file will be created. If the path is a directory, a temporary file will be created in the directory. If the path is a file, it will be overwritten if the extension is valid.
        export_format (FileExportFormat | str | None): format to export the KCL code to. This should be one of 'fbx', 'glb', 'gltf', 'obj', 'ply', 'step', 'stl'. If no format is provided, the default is 'step'.

    Returns:
        Path: Return the path to the exported model if successful
    """

    input_file = Path(input_file)
    input_ext = input_file.suffix.split(".")[1].lower()
    if input_ext not in SUPPORTED_EXTS:
        logger.error("The provided input path does not have a valid extension")
        raise ZooMCPException("The provided input path does not have a valid extension")
    logger.info("Converting the cad file %s", str(input_file.resolve()))

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

    async with aiofiles.open(input_file, "rb") as inp:
        data = await inp.read()

    async with AsyncKittyCAD(verify_ssl=ctx) as client:
        export_response = await client.file.create_file_conversion(
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
        await out.write(next(iter(export_response.outputs.values())))

    logger.info("KCL project exported successfully to %s", str(export_path.resolve()))

    return export_path


@dataclass
class ResultZooExecuteKclLocal:
    ok: bool
    message: str


@dataclass
class ResultZooExecuteKclRemote:
    ok: bool
    message: str
    path_artifact_graph: Path


ResultZooExecuteKcl: TypeAlias = ResultZooExecuteKclLocal | ResultZooExecuteKclRemote


async def zoo_execute_kcl(
    kcl_code: str | None = None,
    kcl_path: Path | str | None = None,
    session_id: str | None = None,
) -> ResultZooExecuteKcl:
    """Execute KCL code given a string of KCL code or a path to a KCL project. Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point to a .kcl file or a directory containing a main.kcl file.

    Args:
        kcl_code (str | None): KCL code
        kcl_path (Path | str | None): KCL path, the path should point to a .kcl file or a directory containing a main.kcl file.
        session_id (str | None): An open modeling session in which to execute the KCL.

    Returns:
        ResultZooExecuteKcl: The execution status and message. Session executions
        also include the artifact graph's temporary JSON file path. When a local
        run completes, compilation issues are appended to the message rather than
        treated as a hard failure. Session runs cannot report non-fatal diagnostics.
    """
    logger.info("Executing KCL code")

    _check_kcl_code_or_path(kcl_code, kcl_path)

    try:
        if session_id is not None:
            path_artifact_graph = await zoo_exec_kcl_project(
                kcl_code=kcl_code,
                kcl_path=kcl_path,
                session_id=session_id,
            )
            logger.info("KCL code executed in modeling session")
            # The engine's exec_kcl_project response carries only an artifact
            # graph, so warnings the local compiler would surface are not
            # available here. Say so rather than report an unqualified success.
            return ResultZooExecuteKclRemote(
                ok=True,
                message=(
                    "KCL code executed successfully in the modeling session. "
                    "Non-fatal diagnostics (warnings and non-fatal errors) are not "
                    "reported for session runs; re-run without session_id to check "
                    "them."
                ),
                path_artifact_graph=path_artifact_graph,
            )

        if kcl_code:
            outcome = await _execute_with_retries(kcl.execute_code, kcl_code)
        else:
            outcome = await _execute_with_retries(kcl.execute, str(kcl_path))

        issues = _format_execution_issues(outcome)
        if issues:
            total = sum(len(reports) for reports in issues.values())
            logger.info("KCL code execution reported %d issue(s)", total)
            message = _execution_issues_message(issues)
            return ResultZooExecuteKclLocal(ok=True, message=message)

        logger.info("KCL code executed successfully")
        return ResultZooExecuteKclLocal(
            ok=True, message="KCL code executed successfully"
        )
    except Exception as e:
        logger.info("Failed to execute KCL code: %s", e)
        return ResultZooExecuteKclLocal(
            ok=False, message=f"Failed to execute KCL code: {e}"
        )


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
    elif isinstance(export_format, str):
        if export_format not in KCLExportFormat.formats.value:
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
            export_response = await _execute_with_retries(
                kcl.execute_code_and_export, kcl_code, export_format
            )
        else:
            logger.info("Exporting KCL project to %s", str(kcl_path))
            assert kcl_path is not None  # _check_kcl_code_or_path ensures this
            kcl_path_resolved = Path(kcl_path)
            export_response = await _execute_with_retries(
                kcl.execute_and_export, str(kcl_path_resolved.resolve()), export_format
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
            report = await _execute_with_retries(
                kcl.get_sketch_constraint_status_code, kcl_code
            )
        else:
            assert kcl_path is not None
            report = await _execute_with_retries(
                kcl.get_sketch_constraint_status, str(kcl_path)
            )
        return _format_constraint_report(report)
    except Exception as e:
        logger.error(e)
        raise ZooMCPException(f"Failed to get sketch constraint status: {e}")


async def zoo_visualize_sketch(
    sketch_name: str,
    kcl_code: str | None = None,
    kcl_path: Path | str | None = None,
) -> bytes:
    """Execute KCL and render one named sketch as a PNG.

    The renderer is provided by ``zoo-kcl`` on ``ExecOutcome``. Sketch names
    are the variable names assigned to sketch expressions and are also exposed
    by :func:`zoo_get_sketch_constraint_status`.

    Args:
        sketch_name: Variable name of the sketch to render.
        kcl_code: KCL source code to execute.
        kcl_path: Path to a KCL file or project containing ``main.kcl``.

    Returns:
        Raw PNG bytes for the requested sketch.
    """
    logger.info("Visualizing sketch %s", sketch_name)

    _check_kcl_code_or_path(kcl_code, kcl_path)

    try:
        if kcl_code:
            outcome = await _execute_with_retries(kcl.execute_code, kcl_code)
        else:
            assert kcl_path is not None
            outcome = await _execute_with_retries(kcl.execute, str(kcl_path))
        return bytes(outcome.render_sketch_png(sketch_name))
    except Exception as e:
        logger.error(e)
        raise ZooMCPException(f"Failed to visualize sketch: {e}")


async def zoo_mock_execute_kcl(
    kcl_code: str | None = None,
    kcl_path: Path | str | None = None,
) -> tuple[bool, str]:
    """Mock execute KCL code given a string of KCL code or a path to a KCL project. Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point to a .kcl file or a directory containing a main.kcl file.

    Args:
        kcl_code (str | None): KCL code
        kcl_path (Path | str | None): KCL path, the path should point to a .kcl file or a directory containing a main.kcl file.

    Returns:
        tuple(bool, str): Returns ``False`` when execution aborts or reports an
        error/fatal compilation issue. Warning-only outcomes remain successful
        and include their rendered diagnostics in the message.
    """
    logger.info("Executing KCL code")

    _check_kcl_code_or_path(kcl_code, kcl_path)

    try:
        if kcl_code:
            outcome = await kcl.mock_execute_code(kcl_code)
        else:
            outcome = await kcl.mock_execute(str(kcl_path))

        issues = _format_execution_issues(outcome)
        if issues:
            total = sum(len(reports) for reports in issues.values())
            logger.info("KCL mock execution reported %d issue(s)", total)
            has_blocking_issues = "fatal" in issues or "error" in issues
            return not has_blocking_issues, _execution_issues_message(issues)

        logger.info("KCL mock executed successfully")
        return True, "KCL code mock executed successfully"
    except Exception as e:
        logger.info("Failed to mock execute KCL code: %s", e)
        return False, f"Failed to mock execute KCL code: {e}"


@dataclass
class FaceInfo:
    face_get_position: FaceGetPosition
    face_get_gradient: FaceGetGradient
    face_get_center: FaceGetCenter


def _prepare_kcl_project(
    kcl_code: str | None,
    kcl_path: Path | str | None,
) -> tuple[str, list[dict[str, str | list[int]]]]:
    _check_kcl_code_or_path(kcl_code, kcl_path)

    if kcl_code:
        return "main.kcl", [
            {
                "path": "main.kcl",
                "contents": list(kcl_code.encode()),
            }
        ]

    assert kcl_path is not None
    return load_kcl_project(kcl_path)


async def _exec_kcl_project(
    ws: ClientConnection,
    entrypoint: str,
    files: list[dict[str, str | list[int]]],
) -> Path:
    """Execute a KCL project through the server-side websocket protocol."""
    request_id = ModelingCmdId(uuid4())
    request = {
        "type": "exec_kcl_project",
        "request_id": request_id,
        "project": {
            "entrypoint": entrypoint,
            "files": files,
        },
    }
    # Bounded by what is left of the whole call rather than per read, so the
    # frames skipped below cannot keep the loop alive indefinitely.
    deadline = _Deadline()
    await _send_modeling_frame(
        ws,
        json.dumps(request),
        deadline,
        "KCL project execution",
    )

    # This response is not represented in the generated SDK yet.
    while True:
        remaining = deadline.remaining
        if remaining <= 0:
            raise _modeling_timeout(deadline, "KCL project execution")
        try:
            raw_response = await asyncio.wait_for(ws.recv(), timeout=remaining)
        except TimeoutError as error:
            raise _modeling_timeout(deadline, "KCL project execution") from error
        try:
            response = json.loads(raw_response)
        except (TypeError, json.JSONDecodeError):
            continue
        if response.get("request_id") != request_id:
            continue
        if not response.get("success", False):
            raise ZooMCPException("Failed to execute KCL project")
        payload = response.get("resp", {})
        if payload.get("type") != "exec_kcl_project":
            continue

        result = payload.get("data", {}).get("result")
        if not isinstance(result, dict):
            raise ZooMCPException("KCL project execution returned no result")
        lowered_result = {str(key).lower(): value for key, value in result.items()}
        if "err" in lowered_result or "error" in lowered_result:
            error = lowered_result.get("err", lowered_result.get("error"))
            raise ZooMCPException(f"Failed to execute KCL project: {error}")

        success = lowered_result.get("ok")
        if not isinstance(success, dict):
            raise ZooMCPException("KCL project execution returned no success result")
        artifact_graph = success.get("artifact_graph")
        if not isinstance(artifact_graph, dict):
            raise ZooMCPException("KCL project execution returned no artifact graph")

        with NamedTemporaryFile(
            mode="w", delete=False, suffix=".json", encoding="utf-8"
        ) as artifact_graph_file:
            json.dump(artifact_graph, artifact_graph_file)
            return Path(artifact_graph_file.name)


@dataclass
class _ModelingSession:
    session_id: str
    client: AsyncKittyCAD
    websocket: ClientConnection
    lock: asyncio.Lock
    artifact_graph_paths: set[Path] = field(default_factory=set)


@dataclass
class _StartingModelingSession:
    """A claim on the single session slot while its websocket is connecting.

    A starter owns the slot only for as long as the slot still holds its own
    sentinel. Whoever removes it has canceled the start, so the starter closes
    the socket it opened instead of publishing it.
    """

    session_id: str


_modeling_session: _ModelingSession | _StartingModelingSession | None = None


async def _open_modeling_websocket(client: AsyncKittyCAD) -> ClientConnection:
    """Open the async modeling transport using an async client's credentials.

    KittyCAD 1.5.0's generated ``AsyncModelingAPI.modeling_commands_ws``
    currently returns ``None`` and constructs a relative websocket URL. Keep
    connection setup here until that generated method is usable.
    """
    query = urlencode(
        {
            "fps": 30,
            "post_effect": PostEffectType.SSAO,
            "show_grid": "false",
            "unlocked_framerate": "false",
            "video_res_height": 1024,
            "video_res_width": 1024,
            "webrtc": "false",
        }
    )
    url = f"{client.base_url.rstrip('/')}/ws/modeling/commands?{query}"
    parsed_url = urlsplit(url)
    if parsed_url.scheme == "https":
        websocket_scheme = "wss"
        websocket_ssl = client.verify_ssl
    elif parsed_url.scheme == "http":
        websocket_scheme = "ws"
        websocket_ssl = None
    else:
        raise ZooMCPException(f"Unsupported Zoo API URL scheme '{parsed_url.scheme}'")
    websocket_url = urlunsplit(parsed_url._replace(scheme=websocket_scheme))
    return await connect(
        websocket_url,
        additional_headers=client.get_headers(),
        close_timeout=120,
        max_size=None,
        ssl=websocket_ssl,
    )


async def _close_modeling_session(session: _ModelingSession) -> None:
    """Close one session's websocket, waiting for any in-flight command."""
    try:
        async with session.lock:
            await session.websocket.close()
    finally:
        try:
            await session.client.aclose()
        finally:
            _unlink_modeling_session_artifact_graphs(session)


def _abort_modeling_websocket(session: _ModelingSession) -> None:
    """Abort a socket immediately when awaiting its close isn't safe."""
    try:
        session.websocket.transport.abort()
    except Exception as error:
        logger.warning(
            "Failed to abort modeling session %s websocket: %s",
            session.session_id,
            error,
        )


async def _discard_modeling_websocket(
    session: _ModelingSession,
    *,
    abort: bool = False,
) -> None:
    """Close a session's websocket from inside its own command.

    Deliberately not _close_modeling_session: that waits on ``session.lock``,
    which the command calling this still holds, so it would deadlock.
    """
    if abort:
        _abort_modeling_websocket(session)
        resources = (("async client", session.client.aclose),)
    else:
        resources = (
            ("websocket", session.websocket.close),
            ("async client", session.client.aclose),
        )

    for resource, close in resources:
        try:
            await close()
        except Exception as error:
            # The socket is being abandoned anyway; a failure closing it must not
            # replace the timeout the caller is about to see.
            logger.warning(
                "Failed to close modeling session %s %s: %s",
                session.session_id,
                resource,
                error,
            )


def _unlink_modeling_session_artifact_graphs(session: _ModelingSession) -> None:
    for path in session.artifact_graph_paths:
        path.unlink(missing_ok=True)
    session.artifact_graph_paths.clear()


async def zoo_start_modeling_session() -> str:
    global _modeling_session

    session_id = str(uuid4())
    starting = _StartingModelingSession(session_id=session_id)
    if _modeling_session is not None:
        # Name the blocker so a caller that lost track of it, or that is
        # blocked by a start whose handshake is hung, can stop it.
        raise ZooMCPException(
            f"A modeling session is already open or starting "
            f"('{_modeling_session.session_id}'); "
            "stop it before starting another"
        )
    _modeling_session = starting

    client: AsyncKittyCAD | None = None
    try:
        client = AsyncKittyCAD(verify_ssl=ctx)
        websocket = await _open_modeling_websocket(client)
    except BaseException:
        if _modeling_session is starting:
            _modeling_session = None
        if client is not None:
            try:
                await client.aclose()
            except Exception as error:
                logger.warning("Failed to close modeling client: %s", error)
        raise

    session = _ModelingSession(
        session_id=session_id,
        client=client,
        websocket=websocket,
        lock=asyncio.Lock(),
    )
    if _modeling_session is starting:
        _modeling_session = session
        return session_id

    await websocket.close()
    await client.aclose()
    raise ZooMCPException("Modeling session start was canceled")


def zoo_get_modeling_sessions() -> list[str]:
    """Return the IDs of modeling sessions owned by this server process."""
    if not isinstance(_modeling_session, _ModelingSession):
        return []
    return [_modeling_session.session_id]


async def zoo_stop_modeling_session(session_id: str) -> None:
    global _modeling_session

    if _modeling_session is None or _modeling_session.session_id != session_id:
        raise ZooMCPException(f"Unknown modeling session '{session_id}'")
    session = _modeling_session
    _modeling_session = None

    if isinstance(session, _StartingModelingSession):
        logger.info("Canceled starting modeling session %s", session_id)
        return

    await _close_modeling_session(session)


async def zoo_stop_all_modeling_sessions() -> None:
    """Close the persistent modeling session, normally during server shutdown."""
    global _modeling_session

    session = _modeling_session
    _modeling_session = None

    if session is None:
        return
    if isinstance(session, _StartingModelingSession):
        # Its own starter closes the half-open socket; see zoo_start_modeling_session.
        return
    try:
        await _close_modeling_session(session)
    except Exception as error:
        logger.warning("Failed to close modeling session: %s", error)


def _abort_all_modeling_sessions() -> None:
    """Synchronously abandon a session during interpreter teardown.

    Async websocket and client objects belong to the server's event loop and
    cannot be awaited from the new loop an ``atexit`` hook would have to create.
    The normal lifespan shutdown remains responsible for graceful cleanup.
    """
    global _modeling_session

    session = _modeling_session
    _modeling_session = None
    if not isinstance(session, _ModelingSession):
        return

    try:
        _abort_modeling_websocket(session)
    finally:
        _unlink_modeling_session_artifact_graphs(session)


def _evict_modeling_session(session_id: str) -> None:
    """Drop a session whose websocket is no longer usable."""
    global _modeling_session

    if (
        isinstance(_modeling_session, _ModelingSession)
        and _modeling_session.session_id == session_id
    ):
        session = _modeling_session
        _modeling_session = None
    else:
        session = None

    if session is not None:
        _unlink_modeling_session_artifact_graphs(session)


# Allows external agents to reuse engine connections so the Zoo infrastructure
# isn't spinning up and down engines for each command, when many times the same
# engine scene has the same model.


@asynccontextmanager
async def _modeling_websocket(session_id: str) -> AsyncIterator[ClientConnection]:
    active_session = _modeling_session
    if (
        not isinstance(active_session, _ModelingSession)
        or active_session.session_id != session_id
    ):
        raise ZooMCPException(f"Unknown modeling session '{session_id}'")
    session = active_session

    # Session state has no awaits around its reads and writes, while this lock
    # serializes commands that yield during websocket I/O.
    async with session.lock:
        try:
            yield session.websocket
        except ZooMCPTimeoutError:
            # The engine still owes a response to the command that gave up. Its
            # request_id no longer matches anything a later command waits for, so
            # the backlog would silently eat the next command's budget, and the
            # engine keeps working on the abandoned command either way. Drop the
            # session rather than hand it back.
            _evict_modeling_session(session_id)
            await _discard_modeling_websocket(session, abort=True)
            raise
        except asyncio.CancelledError:
            # A canceled send or receive can leave a command outstanding. Do not
            # hand that transport to a later call whose view of scene state may
            # no longer match the engine's.
            _evict_modeling_session(session_id)
            await _discard_modeling_websocket(session, abort=True)
            raise
        except (WebSocketException, OSError) as error:
            _evict_modeling_session(session_id)
            await _discard_modeling_websocket(session)
            raise ZooMCPException(
                f"Modeling session '{session_id}' is no longer connected "
                f"({error}); start a new session"
            ) from error


async def zoo_exec_kcl_project(
    session_id: str,
    kcl_code: str | None = None,
    kcl_path: Path | str | None = None,
) -> Path:
    entrypoint, files = await asyncio.to_thread(
        _prepare_kcl_project, kcl_code, kcl_path
    )

    async with _modeling_websocket(session_id) as ws:
        path = await _exec_kcl_project(ws, entrypoint, files)
        if (
            not isinstance(_modeling_session, _ModelingSession)
            or _modeling_session.session_id != session_id
        ):
            path.unlink(missing_ok=True)
            raise ZooMCPException(f"Unknown modeling session '{session_id}'")
        _modeling_session.artifact_graph_paths.add(path)
        return path


async def zoo_face_info(
    face_id: Uuid,
    session_id: str,
) -> FaceInfo:
    async with _modeling_websocket(session_id) as ws:
        # One budget covers all three requests and replies.
        deadline = _Deadline()
        cmd_id_face_get_position = ModelingCmdId(uuid4())
        await _send_modeling_frame(
            ws,
            WebSocketRequest(
                OptionModelingCmdReq(
                    cmd=ModelingCmd(
                        OptionFaceGetPosition(
                            object_id=face_id,
                            uv=Point2d(x=0.5, y=0.5),
                        )
                    ),
                    cmd_id=cmd_id_face_get_position,
                )
            ).model_dump_json(exclude_none=True),
            deadline,
            "face info",
        )

        cmd_id_face_get_gradient = ModelingCmdId(uuid4())
        await _send_modeling_frame(
            ws,
            WebSocketRequest(
                OptionModelingCmdReq(
                    cmd=ModelingCmd(
                        OptionFaceGetGradient(
                            object_id=face_id,
                            uv=Point2d(x=0.5, y=0.5),
                        )
                    ),
                    cmd_id=cmd_id_face_get_gradient,
                )
            ).model_dump_json(exclude_none=True),
            deadline,
            "face info",
        )

        cmd_id_face_get_center = ModelingCmdId(uuid4())
        await _send_modeling_frame(
            ws,
            WebSocketRequest(
                OptionModelingCmdReq(
                    cmd=ModelingCmd(OptionFaceGetCenter(object_id=face_id)),
                    cmd_id=cmd_id_face_get_center,
                )
            ).model_dump_json(exclude_none=True),
            deadline,
            "face info",
        )

        face_get_position: FaceGetPosition | Literal[False] = False
        face_get_gradient: FaceGetGradient | Literal[False] = False
        face_get_center: FaceGetCenter | Literal[False] = False

        while (
            face_get_position is False
            or face_get_gradient is False
            or face_get_center is False
        ):
            message = (await _recv_modeling_frame(ws, deadline, "face info")).root

            # Match on request_id first; see _send_modeling_command.
            request_id = getattr(message, "request_id", None)
            expected_ids = {
                cmd_id_face_get_position,
                cmd_id_face_get_gradient,
                cmd_id_face_get_center,
            }

            if not isinstance(message, SuccessWebSocketResponse):
                if request_id is None or request_id in expected_ids:
                    raise ZooMCPException(_format_websocket_failure(message))
                continue

            if request_id not in expected_ids:
                continue

            response = message.resp.root
            if not isinstance(response, OptionModeling):
                raise ZooMCPException("Received an unexpected websocket response")
            modeling_response = response.data.modeling_response.root

            if message.request_id == cmd_id_face_get_position:
                if not isinstance(modeling_response, ResponseFaceGetPosition):
                    raise ZooMCPException(
                        "Received an unexpected face position response"
                    )
                face_get_position = modeling_response.data
            elif message.request_id == cmd_id_face_get_gradient:
                if not isinstance(modeling_response, ResponseFaceGetGradient):
                    raise ZooMCPException(
                        "Received an unexpected face gradient response"
                    )
                face_get_gradient = modeling_response.data
            elif message.request_id == cmd_id_face_get_center:
                if not isinstance(modeling_response, ResponseFaceGetCenter):
                    raise ZooMCPException("Received an unexpected face center response")
                face_get_center = modeling_response.data

        return FaceInfo(
            face_get_position=face_get_position,
            face_get_gradient=face_get_gradient,
            face_get_center=face_get_center,
        )


_ModelingResponseT = TypeVar("_ModelingResponseT")


class _Deadline:
    """A monotonic wall-clock budget shared by websocket I/O in one tool call.

    Monotonic so a clock adjustment mid-import cannot extend or collapse the
    budget, and shared so that a multi-command call like a multiview snapshot
    is bounded as a whole rather than per command.
    """

    __slots__ = ("_started", "timeout")

    def __init__(self, timeout: float | None = None) -> None:
        # Read at call time, not bound as a default, so a deployment can lower
        # the budget under its own reconnect window.
        self.timeout = MODELING_COMMAND_TIMEOUT if timeout is None else timeout
        self._started = monotonic()

    @property
    def elapsed(self) -> float:
        return monotonic() - self._started

    @property
    def remaining(self) -> float:
        return self.timeout - self.elapsed


def _modeling_timeout(deadline: _Deadline, description: str) -> ZooMCPTimeoutError:
    return ZooMCPTimeoutError(
        f"The modeling engine did not complete {description} within "
        f"{deadline.timeout:g}s (waited {deadline.elapsed:.1f}s). The modeling "
        "session was closed; start a new one and retry, and simplify or shrink "
        "the input if it fails again."
    )


async def _recv_modeling_frame(
    ws: ClientConnection,
    deadline: _Deadline,
    description: str,
) -> WebSocketResponse:
    """Read one frame, bounded by what is left of the whole call's budget.

    A live session interleaves unsolicited frames every few seconds. Applying
    the remaining whole-call budget to each read makes those frames consume it
    instead of restarting a fixed per-read timeout.
    """
    remaining = deadline.remaining
    if remaining <= 0:
        raise _modeling_timeout(deadline, description)

    try:
        raw_response = await asyncio.wait_for(ws.recv(), timeout=remaining)
    except TimeoutError as error:
        raise _modeling_timeout(deadline, description) from error
    return WebSocketResponse.model_validate_json(raw_response)


async def _send_modeling_frame(
    ws: ClientConnection,
    payload: str | bytes,
    deadline: _Deadline,
    description: str,
) -> None:
    """Send one frame within the same wall-clock budget as its response."""
    remaining = deadline.remaining
    if remaining <= 0:
        raise _modeling_timeout(deadline, description)

    try:
        await asyncio.wait_for(ws.send(payload), timeout=remaining)
    except TimeoutError as error:
        raise _modeling_timeout(deadline, description) from error


def _format_websocket_failure(message: object) -> str:
    """Render an engine failure frame as a readable message, not a model repr."""
    errors = getattr(message, "errors", None)
    if not errors:
        return f"The modeling engine reported a failure: {message}"

    rendered = "; ".join(
        f"{getattr(error, 'error_code', 'error')}: {getattr(error, 'message', error)}"
        for error in errors
    )
    return f"The modeling engine reported a failure: {rendered}"


async def _await_modeling_response(
    ws: ClientConnection,
    command_id: ModelingCmdId,
    expected_response: type[_ModelingResponseT],
    response_description: str,
    deadline: _Deadline,
) -> _ModelingResponseT:
    """Read until the typed response for ``command_id`` arrives or time runs out."""
    while True:
        message = (await _recv_modeling_frame(ws, deadline, response_description)).root

        request_id = getattr(message, "request_id", None)

        if not isinstance(message, SuccessWebSocketResponse):
            # Only raise on a failure that is actually ours. A failure carrying
            # no request_id is connection-level so it does apply; one for
            # another command is stale and would report a bogus cause here.
            if request_id is None or request_id == command_id:
                raise ZooMCPException(_format_websocket_failure(message))
            continue

        # Successes must match exactly. A persistent session also carries
        # unsolicited informational frames (session data, ICE) with no
        # request_id, and those are not responses to this command.
        if request_id != command_id:
            continue

        response = message.resp.root
        if not isinstance(response, OptionModeling):
            raise ZooMCPException("Received an unexpected websocket response")
        modeling_response = response.data.modeling_response.root

        if not isinstance(modeling_response, expected_response):
            raise ZooMCPException(
                f"Received an unexpected {response_description} response"
            )
        return modeling_response


async def _send_modeling_command(
    ws: ClientConnection,
    command: ModelingCmd,
    expected_response: type[_ModelingResponseT],
    response_description: str,
    deadline: _Deadline | None = None,
) -> _ModelingResponseT:
    """Send one modeling command and return its matching typed response.

    Without a ``deadline`` the command gets a budget of its own; callers that
    send several commands pass one in so the whole sequence is bounded.
    """
    if deadline is None:
        deadline = _Deadline()
    command_id = ModelingCmdId(uuid4())
    await _send_modeling_frame(
        ws,
        WebSocketRequest(
            OptionModelingCmdReq(
                cmd=command,
                cmd_id=command_id,
            )
        ).model_dump_json(exclude_none=True),
        deadline,
        response_description,
    )

    return await _await_modeling_response(
        ws,
        command_id,
        expected_response,
        response_description,
        deadline,
    )


async def zoo_execute_modeling_command(
    command: ModelingCmd,
    expected_response: type[_ModelingResponseT],
    response_description: str,
    session_id: str,
) -> _ModelingResponseT:
    """Run one modeling command in an existing session."""
    async with _modeling_websocket(session_id) as ws:
        return await _send_modeling_command(
            ws,
            command,
            expected_response,
            response_description,
            _Deadline(),
        )


async def zoo_snapshot(
    session_id: str,
    views: list[OptionDefaultCameraLookAt] | None = None,
    max_image_dimension: int = 512,
    padding: float = 0.1,
    zoom: bool = True,
    highlight_edges: bool = False,
) -> bytes:
    """Capture an existing modeling session as a JPEG, from one camera or several.

    The camera is always switched to an orthographic projection, since
    perspective distorts measurements read off the image. ``views`` gives the
    cameras to capture from; each is rendered in turn on the same connection
    and the results are tiled into one image. With no views, the scene is
    pointed isometrically when ``zoom`` is set, and otherwise captured from
    whatever camera it already has.

    Unless ``zoom`` is False each view is zoomed to fit first; a raw
    take_snapshot uses whatever camera the scene happens to have, which on a
    freshly executed project leaves the model a few pixels wide. Passing
    ``zoom=False`` is how a caller keeps framing it set up itself.

    Edge lines are hidden unless ``highlight_edges`` is True, so that entities
    highlighted via highlight_set_entities stand out instead of competing with
    the outline drawn on every edge in the scene.
    """
    # One budget for the whole capture. A per-command budget would multiply by
    # the command count, which for a four-view capture is fourteen commands.
    deadline = _Deadline()

    async with _modeling_websocket(session_id) as ws:
        await _send_modeling_command(
            ws,
            ModelingCmd(OptionDefaultCameraSetOrthographic()),
            ResponseDefaultCameraSetOrthographic,
            "orthographic camera",
            deadline,
        )

        # Scene-wide, so it is set once rather than per view.
        await _send_modeling_command(
            ws,
            ModelingCmd(OptionEdgeLinesVisible(hidden=not highlight_edges)),
            ResponseEdgeLinesVisible,
            "edge line visibility",
            deadline,
        )

        zoom_to_fit = OptionZoomToFit(object_ids=[], padding=padding)

        jpeg_contents_list: list[bytes] = []
        for view in views or [None]:
            if view is not None:
                await _send_modeling_command(
                    ws,
                    ModelingCmd(view),
                    ResponseDefaultCameraLookAt,
                    "camera view",
                    deadline,
                )
            elif zoom:
                await _send_modeling_command(
                    ws,
                    ModelingCmd(OptionViewIsometric()),
                    ResponseViewIsometric,
                    "isometric view",
                    deadline,
                )

            if zoom:
                await _send_modeling_command(
                    ws,
                    ModelingCmd(zoom_to_fit),
                    ResponseZoomToFit,
                    "zoom to fit",
                    deadline,
                )

            response = await _send_modeling_command(
                ws,
                ModelingCmd(OptionTakeSnapshot(format=ImageFormat.JPEG)),
                ResponseTakeSnapshot,
                "snapshot",
                deadline,
            )
            jpeg_contents_list.append(response.data.contents)

    return await asyncio.to_thread(
        lambda: resize_image(
            create_image_collage(jpeg_contents_list),
            max_image_dimension,
        )
    )


async def _list_org_datasets_raw(
    client: AsyncKittyCAD,
    lookup_enabled: bool | None,
) -> list[dict[str, str | None]]:
    """List datasets without validating fields unrelated to this tool's output."""
    http_client = client.get_http_client()
    url = f"{client.base_url}/org/datasets"
    page_token: str | None = None
    seen_page_tokens: set[str] = set()
    datasets: list[dict[str, str | None]] = []

    while True:
        # A page token carries the filter forward, so only the first request has
        # to spell it out; sending both is harmless.
        params: dict[str, str] = {}
        if lookup_enabled is not None:
            params["lookup_enabled"] = "true" if lookup_enabled else "false"
        if page_token is not None:
            params["page_token"] = page_token
        response = await http_client.get(
            url=url,
            headers=client.get_headers(),
            params=params,
        )
        if not response.is_success:
            from kittycad.response_helpers import raise_for_status

            raise_for_status(response)

        payload = response.json()
        items = payload.get("items", [])
        if not isinstance(items, list):
            raise ZooMCPException("Failed to list org datasets: invalid response")

        for item in items:
            if not isinstance(item, dict):
                continue
            dataset_id = item.get("id")
            name = item.get("name")
            description = item.get("description")
            if isinstance(dataset_id, str) and isinstance(name, str):
                datasets.append(
                    {
                        "id": dataset_id,
                        "name": name,
                        "description": description
                        if isinstance(description, str)
                        else None,
                    }
                )

        page_token = payload.get("next_page")
        if not isinstance(page_token, str) or not page_token:
            return datasets
        if page_token in seen_page_tokens:
            logger.warning(
                "Org datasets pagination repeated page token; stopping early"
            )
            return datasets
        seen_page_tokens.add(page_token)


def _org_datasets_empty_or_raise(
    exc: KittyCADClientError,
) -> list[dict[str, str | None]]:
    """Treat a missing datasets endpoint as empty; re-raise anything else."""
    if exc.status_code == 404:
        return []
    raise ZooMCPException(f"Failed to list org datasets: {exc}") from exc


async def zoo_list_org_datasets(
    lookup_enabled: bool | None = True,
) -> list[dict[str, str | None]]:
    """List all datasets visible to the org tied to the current ZOO_API_TOKEN.

    Args:
        lookup_enabled: When True (the default), only datasets an org has left
            enabled for lookup are returned. Pass None to list every dataset
            regardless of its lookup setting, or False for only the excluded ones.

    Returns:
        A list of {"id": <uuid str>, "name": <str>, "description": <str | None>}
        entries, possibly empty.
    """
    logger.info("Listing org datasets (lookup_enabled=%s)", lookup_enabled)
    async with AsyncKittyCAD(verify_ssl=ctx) as client:
        use_raw_fallback = False
        datasets = []
        try:
            datasets = [
                dataset
                async for dataset in client.orgs.list_org_datasets(
                    limit=None, page_token=None, lookup_enabled=lookup_enabled
                )
            ]
        except ValueError as exc:
            logger.warning(
                "SDK could not validate org datasets; falling back to raw JSON: %s",
                exc,
            )
            use_raw_fallback = True
        except KittyCADClientError as exc:
            return _org_datasets_empty_or_raise(exc)

        # Run the fallback outside the handler above so its own client errors still
        # get the 404-means-empty treatment instead of escaping uncaught.
        if use_raw_fallback:
            try:
                return await _list_org_datasets_raw(client, lookup_enabled)
            except KittyCADClientError as exc:
                return _org_datasets_empty_or_raise(exc)

    return [
        {"id": str(d.id), "name": d.name, "description": d.description}
        for d in datasets
    ]


async def zoo_list_org_skills() -> list[dict[str, str]]:
    """List all skills visible to the org tied to the current ZOO_API_TOKEN.

    Returns:
        A list of {"id": <uuid str>, "name": <str>, "description": <str>,
        "markdown": <str>} entries, possibly empty.
    """
    logger.info("Listing org skills")
    async with AsyncKittyCAD(verify_ssl=ctx) as client:
        try:
            skills = await client.orgs.list_org_skills()
        except KittyCADClientError as exc:
            if exc.status_code == 404:
                return []
            raise ZooMCPException(f"Failed to list org skills: {exc}") from exc

    return [
        {
            "id": str(s.id),
            "name": s.name,
            "description": s.description,
            "markdown": s.markdown,
        }
        for s in (skills or [])
    ]


async def zoo_search_org_dataset_semantic(
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
    async with AsyncKittyCAD(verify_ssl=ctx) as client:
        try:
            matches = await client.orgs.search_org_dataset_semantic(
                id=Uuid(dataset_id), q=query, limit=limit
            )
        except KittyCADClientError as exc:
            raise ZooMCPException(
                f"Failed to search dataset {dataset_id}: {exc}"
            ) from exc

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
