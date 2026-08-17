import asyncio
import atexit
import signal
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import FrameType

from kittycad.models import (
    CameraMovement,
    CurveGetEndPoints,
    CurveGetType,
    DefaultCameraCenterToSelection,
    DistanceType,
    EdgeGetLength,
    EngineUtilEvaluatePath,
    EntityGetAllChildUuids,
    EntityGetDistance,
    EntityGetIndex,
    EntityGetParentId,
    EntityGetSketchPaths,
    EntityType,
    GlobalAxis,
    HighlightSetEntities,
    ModelingCmd,
    SelectReplace,
    SetSelectionFilter,
)
from kittycad.models.distance_type import OptionEuclidean, OptionOnAxis
from kittycad.models.modeling_cmd import (
    OptionCurveGetEndPoints,
    OptionCurveGetType,
    OptionDefaultCameraCenterToSelection,
    OptionDefaultCameraLookAt,
    OptionEdgeGetLength,
    OptionEngineUtilEvaluatePath,
    OptionEntityGetAllChildUuids,
    OptionEntityGetDistance,
    OptionEntityGetIndex,
    OptionEntityGetParentId,
    OptionEntityGetSketchPaths,
    OptionHighlightSetEntities,
    OptionSelectReplace,
    OptionSetSelectionFilter,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionCurveGetEndPoints as ResponseCurveGetEndPoints,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionCurveGetType as ResponseCurveGetType,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionDefaultCameraCenterToSelection as ResponseDefaultCameraCenterToSelection,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionEdgeGetLength as ResponseEdgeGetLength,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionEngineUtilEvaluatePath as ResponseEngineUtilEvaluatePath,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionEntityGetAllChildUuids as ResponseEntityGetAllChildUuids,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionEntityGetDistance as ResponseEntityGetDistance,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionEntityGetIndex as ResponseEntityGetIndex,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionEntityGetParentId as ResponseEntityGetParentId,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionEntityGetSketchPaths as ResponseEntityGetSketchPaths,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionHighlightSetEntities as ResponseHighlightSetEntities,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionSelectReplace as ResponseSelectReplace,
)
from kittycad.models.ok_modeling_cmd_response import (
    OptionSetSelectionFilter as ResponseSetSelectionFilter,
)
from kittycad.models.uuid import Uuid
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent

from zoo_mcp import ZooMCPException, logger
from zoo_mcp.kcl_docs import (
    KCLDocs,
    get_doc_content,
    list_available_docs,
    search_docs,
)
from zoo_mcp.kcl_samples import (
    KCLSamples,
    SampleData,
    get_sample_content,
    list_available_samples,
    search_samples,
)
from zoo_mcp.utils.image_utils import (
    encode_image,
    save_image_bytes_to_disk,
    save_image_to_disk,
)
from zoo_mcp.zoo_tools import (
    CameraView,
    FaceInfo,
    ResultZooExecuteKcl,
    ResultZooExecuteKclLocal,
    zoo_calculate_bounding_box_cad,
    zoo_calculate_bounding_box_kcl,
    zoo_calculate_cad_physical_properties,
    zoo_calculate_center_of_mass,
    zoo_calculate_kcl_physical_properties,
    zoo_calculate_mass,
    zoo_calculate_surface_area,
    zoo_calculate_volume,
    zoo_convert_cad_file,
    zoo_exec_kcl_project,
    zoo_execute_kcl,
    zoo_execute_modeling_command,
    zoo_export_kcl,
    zoo_face_info,
    zoo_format_kcl,
    zoo_get_modeling_sessions,
    zoo_get_sketch_constraint_status,
    zoo_import_cad_file,
    zoo_lint_and_fix_kcl,
    zoo_list_org_datasets,
    zoo_list_org_skills,
    zoo_mock_execute_kcl,
    zoo_search_org_dataset_semantic,
    zoo_snapshot,
    zoo_start_modeling_session,
    zoo_stop_all_modeling_sessions,
    zoo_stop_modeling_session,
)


async def _init_kcl_indexes() -> None:
    """Populate the KCL docs and samples indexes from zoo.dev concurrently.

    Each index is initialized independently — a failure in one does not
    prevent the other from succeeding.
    """

    async def _safe(coro, label: str) -> None:
        try:
            await coro
        except Exception as e:
            logger.warning(f"Failed to initialize {label} index: {e}")

    await asyncio.gather(
        _safe(KCLDocs.initialize(), "KCL docs"),
        _safe(KCLSamples.initialize(), "KCL samples"),
    )


_kcl_index_task: asyncio.Task[None] | None = None


async def _ensure_kcl_indexes() -> None:
    """Ensure KCL indexes are initialized, kicking off the fetch on first call.

    Subsequent calls await the same in-flight task, so initialization
    runs at most once per process.
    """
    global _kcl_index_task
    if _kcl_index_task is None:
        _kcl_index_task = asyncio.create_task(_init_kcl_indexes())
    await _kcl_index_task


@asynccontextmanager
async def _lifespan(_server: FastMCP) -> AsyncIterator[None]:
    """Eagerly start index population when the server starts.

    Tools still ``await _ensure_kcl_indexes()`` so they wait for completion
    if invoked before the background fetch finishes.
    """
    global _kcl_index_task
    if _kcl_index_task is None:
        _kcl_index_task = asyncio.create_task(_init_kcl_indexes())
    try:
        yield
    finally:
        zoo_stop_all_modeling_sessions()
        if _kcl_index_task is not None and not _kcl_index_task.done():
            _kcl_index_task.cancel()
        _kcl_index_task = None
        KCLDocs._instance = None
        KCLSamples._instance = None


mcp = FastMCP(
    name="Zoo MCP Server",
    log_level="INFO",
    lifespan=_lifespan,
)


@mcp.tool()
async def calculate_center_of_mass(input_file: str, unit_length: str) -> dict | str:
    """Calculate the center of mass of a 3d object represented by the input file.

    Args:
        input_file (str): The path of the file to get the mass from. The file should be one of the supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp, .stl (case-insensitive)
        unit_length (str): The unit of length to return the result in. One of 'cm', 'ft', 'in', 'm', 'mm', 'yd'

    Returns:
        str: The center of mass of the file in the specified unit of length, or an error message if the operation fails.
    """

    logger.info("calculate_center_of_mass tool called for file: %s", input_file)

    try:
        com = await zoo_calculate_center_of_mass(
            file_path=input_file, unit_length=unit_length
        )
        return com
    except Exception as e:
        return f"There was an error calculating the center of mass of the file: {e}"


@mcp.tool()
async def calculate_mass(
    input_file: str, unit_mass: str, unit_density: str, density: float
) -> float | str:
    """Calculate the mass of a 3d object represented by the input file.

    Args:
        input_file (str): The path of the file to get the mass from. The file should be one of the supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp, .stl (case-insensitive)
        unit_mass (str): The unit of mass to return the result in. One of 'g', 'kg', 'lb'.
        unit_density (str): The unit of density to calculate the mass. One of 'lb:ft3', 'kg:m3'.
        density (float): The density of the material.

    Returns:
        str: The mass of the file in the specified unit of mass, or an error message if the operation fails.
    """

    logger.info("calculate_mass tool called for file: %s", input_file)

    try:
        mass = await zoo_calculate_mass(
            file_path=input_file,
            unit_mass=unit_mass,
            unit_density=unit_density,
            density=density,
        )
        return mass
    except Exception as e:
        return f"There was an error calculating the mass of the file: {e}"


@mcp.tool()
async def calculate_surface_area(input_file: str, unit_area: str) -> float | str:
    """Calculate the surface area of a 3d object represented by the input file.

    Args:
        input_file (str): The path of the file to get the surface area from. The file should be one of the supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp, .stl (case-insensitive)
        unit_area (str): The unit of area to return the result in. One of 'cm2', 'dm2', 'ft2', 'in2', 'km2', 'm2', 'mm2', 'yd2'.

    Returns:
        str: The surface area of the file in the specified unit of area, or an error message if the operation fails.
    """

    logger.info("calculate_surface_area tool called for file: %s", input_file)

    try:
        surface_area = await zoo_calculate_surface_area(
            file_path=input_file, unit_area=unit_area
        )
        return surface_area
    except Exception as e:
        return f"There was an error calculating the surface area of the file: {e}"


@mcp.tool()
async def calculate_volume(input_file: str, unit_volume: str) -> float | str:
    """Calculate the volume of a 3d object represented by the input file.

    Args:
        input_file (str): The path of the file to get the volume from. The file should be one of the supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp, .stl (case-insensitive)
        unit_volume (str): The unit of volume to return the result in. One of 'cm3', 'ft3', 'in3', 'm3', 'mm3', 'yd3', 'usfloz', 'usgal', 'l', 'ml'.

    Returns:
        str: The volume of the file in the specified unit of volume, or an error message if the operation fails.
    """

    logger.info("calculate_volume tool called for file: %s", input_file)

    try:
        volume = await zoo_calculate_volume(file_path=input_file, unit_vol=unit_volume)
        return volume
    except Exception as e:
        return f"There was an error calculating the volume of the file: {e}"


@mcp.tool()
async def calculate_cad_physical_properties(
    input_file: str,
    unit_length: str,
    unit_mass: str,
    unit_density: str,
    density: float,
    unit_area: str,
    unit_volume: str,
) -> dict | str:
    """Calculate physical properties (volume, mass, surface area, center of mass, bounding box) of a CAD file.

    Args:
        input_file (str): The path of the file. The file should be one of the supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp, .stl (case-insensitive)
        unit_length (str): The unit of length for center of mass. One of 'cm', 'ft', 'in', 'm', 'mm', 'yd'.
        unit_mass (str): The unit of mass for the mass result. One of 'g', 'kg', 'lb'.
        unit_density (str): The unit of density for the material. One of 'lb:ft3', 'kg:m3'.
        density (float): The density of the material.
        unit_area (str): The unit of area for surface area. One of 'cm2', 'dm2', 'ft2', 'in2', 'km2', 'm2', 'mm2', 'yd2'.
        unit_volume (str): The unit of volume. One of 'cm3', 'ft3', 'in3', 'm3', 'mm3', 'yd3', 'usfloz', 'usgal', 'l', 'ml'.

    Returns:
        dict | str: A dictionary with keys 'volume', 'mass', 'surface_area', 'center_of_mass', and 'bounding_box', or an error message if the operation fails.
    """

    logger.info(
        "calculate_cad_physical_properties tool called for file: %s", input_file
    )

    try:
        return await zoo_calculate_cad_physical_properties(
            file_path=input_file,
            unit_length=unit_length,
            unit_mass=unit_mass,
            unit_density=unit_density,
            density=density,
            unit_area=unit_area,
            unit_vol=unit_volume,
        )
    except Exception as e:
        return f"There was an error calculating physical properties of the file: {e}"


@mcp.tool()
async def calculate_kcl_physical_properties(
    kcl_code: str | None = None,
    kcl_path: str | None = None,
    unit_length: str = "mm",
    unit_mass: str = "g",
    unit_density: str = "kg:m3",
    density: float = 1000.0,
    unit_area: str = "mm2",
    unit_volume: str = "cm3",
) -> dict | str:
    """Calculate physical properties (volume, mass, surface area, center of mass, bounding box) of a KCL model.

    Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point
    to a .kcl file or a directory containing a main.kcl file.

    Args:
        kcl_code (str | None): The KCL code to evaluate.
        kcl_path (str | None): Path to a .kcl file or a directory containing a main.kcl file.
        unit_length (str): The unit of length for center of mass. One of 'cm', 'ft', 'in', 'm', 'mm', 'yd'.
        unit_mass (str): The unit of mass for the mass result. One of 'g', 'kg', 'lb'.
        unit_density (str): The unit of density for the material. One of 'lb:ft3', 'kg:m3'.
        density (float): The density of the material.
        unit_area (str): The unit of area for surface area. One of 'cm2', 'dm2', 'ft2', 'in2', 'km2', 'm2', 'mm2', 'yd2'.
        unit_volume (str): The unit of volume. One of 'cm3', 'ft3', 'in3', 'm3', 'mm3', 'yd3', 'usfloz', 'usgal', 'l', 'ml'.

    Returns:
        dict | str: A dictionary with keys 'volume', 'mass', 'surface_area', 'center_of_mass', and 'bounding_box', or an error message if the operation fails.
    """

    logger.info("calculate_kcl_physical_properties tool called")

    try:
        return await zoo_calculate_kcl_physical_properties(
            kcl_code=kcl_code,
            kcl_path=kcl_path,
            unit_length=unit_length,
            unit_mass=unit_mass,
            unit_density=unit_density,
            density=density,
            unit_area=unit_area,
            unit_vol=unit_volume,
        )
    except Exception as e:
        return (
            f"There was an error calculating physical properties of the KCL model: {e}"
        )


@mcp.tool()
async def calculate_bounding_box_kcl(
    unit_length: str,
    kcl_code: str | None = None,
    kcl_path: str | None = None,
) -> dict | str:
    """Calculate the bounding box of a KCL model.

    Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point
    to a .kcl file or a directory containing a main.kcl file.

    Args:
        unit_length (str): The unit of length to return the result in. One of 'cm', 'ft', 'in', 'm', 'mm', 'yd'
        kcl_code (str | None): The KCL code to evaluate.
        kcl_path (str | None): Path to a .kcl file or a directory containing a main.kcl file.

    Returns:
        dict | str: A dictionary with 'center' (dict with x,y,z) and 'dimensions' (dict with x,y,z),
                    or an error message if the operation fails.
    """

    logger.info("calculate_bounding_box_kcl tool called")

    try:
        return await zoo_calculate_bounding_box_kcl(
            unit_length=unit_length,
            kcl_code=kcl_code,
            kcl_path=kcl_path,
        )
    except Exception as e:
        return f"There was an error calculating bounding box of the KCL model: {e}"


@mcp.tool()
async def calculate_bounding_box_cad(
    input_file: str,
) -> dict | str:
    """Calculate the bounding box of a CAD file.

    Args:
        input_file (str): The path of the CAD file. The file should be one of the supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp, .stl (case-insensitive)

    Returns:
        dict | str: A dictionary with 'center' (dict with x,y,z) and 'dimensions' (dict with x,y,z),
                    or an error message if the operation fails.
    """

    logger.info("calculate_bounding_box_cad tool called for file: %s", input_file)

    try:
        return await zoo_calculate_bounding_box_cad(file_path=input_file)
    except Exception as e:
        return f"There was an error calculating the bounding box of the file: {e}"


@mcp.tool()
async def convert_cad_file(
    input_path: str,
    export_path: str | None,
    export_format: str | None,
) -> str:
    """Convert a CAD file from one format to another CAD file format.

    Args:
        input_path (str): The input cad file to convert. The file should be one of the supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp, .stl (case-insensitive)
        export_path (str | None): The path to save the converted CAD file to. If the path is a directory, a temporary file will be created in the directory. If the path is a file, it will be overwritten if the extension is valid.
        export_format (str | None): The format of the exported CAD file. This should be one of 'fbx', 'glb', 'gltf', 'obj', 'ply', 'step', 'stl'. If no format is provided, the default is 'step'.

    Returns:
        str: The path to the converted CAD file, or an error message if the operation fails.
    """

    logger.info("convert_cad_file tool called")

    try:
        step_path = await zoo_convert_cad_file(
            input_path=input_path, export_path=export_path, export_format=export_format
        )
        return str(step_path)
    except Exception as e:
        return f"There was an error converting the CAD file: {e}"


@mcp.tool()
async def execute_kcl(
    kcl_code: str | None = None,
    kcl_path: str | None = None,
    session_id: str | None = None,
) -> ResultZooExecuteKcl:
    """Execute KCL code given a string of KCL code or a path to a KCL project. Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point to a .kcl file or a directory containing a main.kcl file.

      Session executions save the artifact graph to a temporary JSON file and
      return its path. Local executions do not produce an artifact graph and can
      have large network overhead depending on the model.

    Args:
        kcl_code (str | None): The KCL code to execute.
        kcl_path (str | None): The path to a KCL file to execute. The path should point to a .kcl file or a directory containing a main.kcl file.
        session_id: An open modeling session in which to execute the KCL.

    Returns:
        ResultZooExecuteKcl: The execution status and message. Session executions
                            also include the artifact graph's JSON file path.
    """

    logger.info("execute_kcl tool called")

    try:
        return await zoo_execute_kcl(
            kcl_code=kcl_code,
            kcl_path=kcl_path,
            session_id=session_id,
        )
    except Exception as e:
        return ResultZooExecuteKclLocal(
            ok=False, message=f"Failed to execute KCL code: {e}"
        )


@mcp.tool()
async def exec_kcl_project(
    session_id: str,
    kcl_code: str | None = None,
    kcl_path: str | None = None,
) -> str:
    """Run a KCL project on the server side and save its artifact graph.

    Args:
        kcl_code (str | None): KCL code to run as a single-file project.
        kcl_path (str | None): A .kcl file or project directory containing main.kcl.
        session_id: The modeling session in which to execute the project.

    Returns:
        str: The path to the JSON file containing the artifact graph.
    """
    logger.info("exec_kcl_project tool called")

    return str(
        zoo_exec_kcl_project(
            kcl_code=kcl_code,
            kcl_path=kcl_path,
            session_id=session_id,
        )
    )


@mcp.tool()
async def start_modeling_session() -> str:
    """Open an empty modeling websocket for subsequent tools.

    Only one modeling session can be open at a time. Stop the current session
    before starting another.

    The server does not expire sessions after an idle or lifetime timeout.
    Callers are responsible for tracking and enforcing their desired timeout.

    Pass the returned session_id to execute_kcl or exec_kcl_project to populate
    the scene, then reuse it with modeling query, selection, and highlight tools.
    Stop the session explicitly with stop_modeling_session when finished.

    Returns:
        str: The session ID to pass to session-aware modeling tools.
    """
    logger.info("start_modeling_session tool called")
    return zoo_start_modeling_session()


@mcp.tool()
async def get_modeling_sessions() -> list[str]:
    """List modeling sessions owned by the current MCP server process.

    Use this to recover the active session ID after reconnecting to a server
    process. A restarted server has no sessions. The current implementation
    supports at most one session, but the list return type allows future
    support for multiple sessions.

    Returns:
        list[str]: Active modeling session IDs, currently empty or one item.
    """
    logger.info("get_modeling_sessions tool called")
    return zoo_get_modeling_sessions()


@mcp.tool()
async def import_cad_file(session_id: str, input_path: str) -> str:
    """Import a CAD file into an existing modeling session.

    Args:
        session_id: The ID returned by start_modeling_session.
        input_path: Path to a .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp,
                    or .stl file.

    Returns:
        str: The modeling engine ID of the imported object.
    """
    logger.info("import_cad_file tool called for file: %s", input_path)
    return zoo_import_cad_file(session_id=session_id, input_path=input_path)


@mcp.tool()
async def stop_modeling_session(session_id: str) -> None:
    """Close a persistent modeling websocket session.

    Args:
        session_id: The ID returned by start_modeling_session.

    Returns:
        None
    """
    logger.info("stop_modeling_session tool called")
    zoo_stop_modeling_session(session_id)


@mcp.tool()
async def export_kcl(
    kcl_code: str | None = None,
    kcl_path: str | None = None,
    export_path: str | None = None,
    export_format: str | None = None,
) -> str:
    """Export KCL code to a CAD file. Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point to a .kcl file or a directory containing a main.kcl file.

    Args:
        kcl_code (str | None): The KCL code to export to a CAD file.
        kcl_path (str | None): The path to a KCL file to export to a CAD file. The path should point to a .kcl file or a directory containing a main.kcl file.
        export_path (str | None): The path to export the CAD file. If no path is provided, a temporary file will be created.
        export_format (str | None): The format to export the file as. This should be one of 'fbx', 'glb', 'gltf', 'obj', 'ply', 'step', 'stl'. If no format is provided, the default is 'step'.

    Returns:
        str: The path to the converted CAD file, or an error message if the operation fails.
    """

    logger.info("convert_kcl_to_step tool called")

    try:
        cad_path = await zoo_export_kcl(
            kcl_code=kcl_code,
            kcl_path=kcl_path,
            export_path=export_path,
            export_format=export_format,
        )
        return str(cad_path)
    except Exception as e:
        return f"There was an error exporting the CAD file: {e}"


@mcp.tool()
async def format_kcl(
    kcl_code: str | None = None,
    kcl_path: str | None = None,
) -> str:
    """Format KCL code given a string of KCL code or a path to a KCL project. Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point to a .kcl file or a directory containing .kcl files.

    Args:
        kcl_code (str | None): The KCL code to format.
        kcl_path (str | None): The path to a KCL file to format. The path should point to a .kcl file or a directory containing a main.kcl file.

    Returns:
        str | None: Returns the formatted kcl code if the kcl_code is used otherwise returns None, the KCL in the kcl_path will be formatted in place
    """

    logger.info("format_kcl tool called")

    try:
        res = await zoo_format_kcl(kcl_code=kcl_code, kcl_path=kcl_path)
        if isinstance(res, str):
            return res
        else:
            return f"Successfully formatted KCL code at: {kcl_path}"
    except Exception as e:
        return f"There was an error formatting the KCL: {e}"


@mcp.tool()
async def get_sketch_constraint_status(
    kcl_code: str | None = None,
    kcl_path: str | None = None,
) -> dict | str:
    """Execute KCL and return a report of sketch constraint status. Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point to a .kcl file or a directory containing a main.kcl file.

    Sketches are grouped by constraint status: fully_constrained, under_constrained, over_constrained, and errors.
    Each sketch entry includes the sketch name, status, free_count (under-constrained segments), conflict_count (over-constrained segments), and total_count (total segments analyzed).

    The report also includes kcl_executes_successfully (False when KCL parsing/execution failed before all sketches were analyzed) and kcl_error (None on success, otherwise a dict with phase ("parse" or "execution") and text fields describing the failure). A partial report may still contain sketch entries analyzed prior to the failure.

    Args:
        kcl_code (str | None): The KCL code to check constraints for.
        kcl_path (str | None): The path to a KCL file or directory containing a main.kcl file.

    Returns:
        dict | str: A report grouping sketches by constraint status, or an error message if the operation fails.
    """

    logger.info("get_sketch_constraint_status tool called")

    try:
        return await zoo_get_sketch_constraint_status(
            kcl_code=kcl_code, kcl_path=kcl_path
        )
    except Exception as e:
        return f"Failed to get sketch constraint status: {e}"


@mcp.tool()
async def get_face_info(
    face_id: str,
    session_id: str,
) -> FaceInfo:
    """Get the position, gradient, normal, and center of a face in a modeling session.

    Args:
        face_id (str): Usually a user or LLM-selected face id.
        session_id: A modeling session populated by execute_kcl or exec_kcl_project.

    Returns:
        FaceInfo: The face position, gradient, normal, and center. The position is
                  the starting point of the face's outside perimeter in KittyCAD
                  engine space. The center is the geometric center of the face and
                  should generally be preferred over the position.
    """
    logger.info("get_face_info tool called for face_id=%s", face_id)

    return zoo_face_info(
        face_id=Uuid(face_id),
        session_id=session_id,
    )


@mcp.tool()
async def entity_distance(
    entity_id1: str,
    entity_id2: str,
    session_id: str,
    on_axis: GlobalAxis | None = None,
) -> EntityGetDistance:
    """Get the minimum and maximum distance between two model entities.

    Args:
        entity_id1: The first entity UUID, typically obtained from an artifact graph.
        entity_id2: The second entity UUID.
        session_id: A modeling session populated by execute_kcl or exec_kcl_project.
        on_axis: Optional global axis for projected distance; omit for Euclidean distance.

    Returns:
        EntityGetDistance: The minimum and maximum distance between the entities.
    """
    logger.info("entity_distance tool called")
    return zoo_execute_modeling_command(
        ModelingCmd(
            OptionEntityGetDistance(
                entity_id1=Uuid(entity_id1),
                entity_id2=Uuid(entity_id2),
                distance_type=DistanceType(
                    OptionOnAxis(axis=on_axis)
                    if on_axis is not None
                    else OptionEuclidean()
                ),
            )
        ),
        ResponseEntityGetDistance,
        "entity distance",
        session_id,
    ).data


@mcp.tool()
async def set_selection_filter(
    entity_types: list[EntityType],
    session_id: str,
) -> SetSelectionFilter:
    """Set which model entity types can be added to the "selection set".

    Args:
        entity_types: Entity types permitted by the selection filter.
        session_id: An open modeling session, from start_modeling_session. Required:
                    the filter is scene state and would be discarded without one.

    Returns:
        SetSelectionFilter: Confirmation that the filter was set.
    """
    logger.info("set_selection_filter tool called")
    return zoo_execute_modeling_command(
        ModelingCmd(OptionSetSelectionFilter(filter=entity_types)),
        ResponseSetSelectionFilter,
        "selection filter",
        session_id,
    ).data


@mcp.tool()
async def select_entities(
    entity_ids: list[str],
    session_id: str,
) -> SelectReplace:
    """Replace the "selection set" with the given entities.

    Takes UUIDs exactly as they appear in an artifact graph, and works for faces,
    edges, and solids alike.

    Selected entities render with a distinct color tint and an outline, which is
    considerably more obvious in a `snapshot` than the subtler brightening that
    highlight_set_entities applies. For the strongest emphasis on a face, call
    both this and highlight_set_entities; for an edge use just one, as selection
    overrides the highlight. Follow with `snapshot` (passing zoom=False to keep
    the current camera) to see the result.

    Selections draw through the solid. An entity facing away from the camera
    still renders tinted, seen through the body, so a tinted entity in a
    snapshot is not evidence that it faces the camera: an occluded edge shows
    up as an interior diagonal rather than on the silhouette, and an occluded
    face looks washed out rather than solid. Pick a camera on the same side as
    the entity to see it properly.

    Args:
        entity_ids: Entity UUIDs to select; pass an empty list to clear the selection.
        session_id: An open modeling session, from start_modeling_session. Required:
                    the selection is scene state and would be discarded without one.

    Returns:
        SelectReplace: Confirmation that the selection was replaced.
    """
    logger.info("select_entities tool called")
    return zoo_execute_modeling_command(
        ModelingCmd(OptionSelectReplace(entities=entity_ids)),
        ResponseSelectReplace,
        "select entities",
        session_id,
    ).data


@mcp.tool()
async def center_camera_on_selection(
    session_id: str,
    move_vantage: bool = True,
) -> DefaultCameraCenterToSelection:
    """Point the camera at the current "selection set".

    Useful for making a small selection legible, especially an edge: an edge is
    only a pixel or two wide, so centring it helps far more than any change of
    colour. Set the selection first with select_entities.

    Centring moves the camera without re-framing the scene, so parts of the model
    may fall outside the image. Follow with `snapshot` passing zoom=False, since
    zoom=True re-frames the whole scene and undoes the centring.

    Args:
        session_id: An open modeling session, from start_modeling_session. Required:
                    camera position is scene state and would be discarded without one.
        move_vantage: Move the camera's vantage point as well as its target.
                      True (the default) centres the selection most precisely;
                      False re-aims the camera from where it already is.

    Returns:
        DefaultCameraCenterToSelection: Confirmation that the camera was centred.
    """
    logger.info("center_camera_on_selection tool called")
    return zoo_execute_modeling_command(
        ModelingCmd(
            OptionDefaultCameraCenterToSelection(
                camera_movement=CameraMovement.VANTAGE
                if move_vantage
                else CameraMovement.NONE
            )
        ),
        ResponseDefaultCameraCenterToSelection,
        "camera centering",
        session_id,
    ).data


@mcp.tool()
async def highlight_set_entities(
    entity_ids: list[str],
    session_id: str,
) -> HighlightSetEntities:
    """Replace the currently highlighted entities. Does NOT modify the "selection set".

    This is a visual command: follow it with the `snapshot` tool, passing the same
    session_id and zoom=False, to see the highlight without moving the camera. It
    highlights edges, surfaces and bodies.

    Highlighting only brightens an entity slightly. select_entities is markedly
    more obvious, tinting the entity and drawing an outline around it. For the
    strongest emphasis on a face, call both; on an edge call only one, because
    selection overrides the highlight. An edge is a pixel or two wide whichever
    you choose, so also consider center_camera_on_selection or a larger
    max_image_dimension to make one legible.

    Highlights draw through the solid, so an entity facing away from the camera
    still renders, seen through the body. A highlighted entity in a snapshot is
    therefore not evidence that it faces the camera; choose a camera on the same
    side as the entity to see it properly.

    Args:
        entity_ids: Entity UUIDs to highlight; pass an empty list to clear highlights.
        session_id: An open modeling session, from start_modeling_session. Required:
                    highlights are scene state and would be discarded without one.

    Returns:
        HighlightSetEntities: Confirmation that the highlights were replaced.
    """
    logger.info("highlight_set_entities tool called")
    return zoo_execute_modeling_command(
        ModelingCmd(OptionHighlightSetEntities(entities=entity_ids)),
        ResponseHighlightSetEntities,
        "highlight entities",
        session_id,
    ).data


@mcp.tool()
async def curve_get_end_points(
    curve_id: str,
    session_id: str,
) -> CurveGetEndPoints:
    """Get the start and end points of a curve entity.

    Args:
        curve_id: Curve UUID, typically obtained from an artifact graph.
        session_id: A modeling session populated by execute_kcl or exec_kcl_project.

    Returns:
        CurveGetEndPoints: The curve's start and end points.
    """
    logger.info("curve_get_end_points tool called")
    return zoo_execute_modeling_command(
        ModelingCmd(OptionCurveGetEndPoints(curve_id=Uuid(curve_id))),
        ResponseCurveGetEndPoints,
        "curve endpoints",
        session_id,
    ).data


@mcp.tool()
async def engine_util_evaluate_path(
    path_json: str,
    t: float,
    session_id: str,
) -> EngineUtilEvaluatePath:
    """Evaluate a serialized KCL path at parameter t.

    Examples of `path_json`:

    ```json
    {
      "start": {
        "from": [10, 0]
      },
      "value": [
        {
          "type": "Arc",
          "center": [0, 0],
          "radius": 10,
          "angle_range": [0, 90]
        }
      ]
    }
    ```

    ```json
    {
      "start": {
        "from": [0, 0]
      },
      "value": [
        {
          "type": "ToPoint",
          "to": [10, 0]
        },
        {
          "type": "ToPoint",
          "to": [10, 10]
        }
      ]
    }
    ```

    Args:
        path_json: The serialized JSON representation of the KCL sketch or path.
        t: Normalized path parameter, conventionally between 0 and 1.
        session_id: A modeling session populated by execute_kcl or exec_kcl_project.

    Returns:
        EngineUtilEvaluatePath: The position on the path at parameter t.
    """
    logger.info("engine_util_evaluate_path tool called")
    return zoo_execute_modeling_command(
        ModelingCmd(OptionEngineUtilEvaluatePath(path_json=path_json, t=t)),
        ResponseEngineUtilEvaluatePath,
        "path evaluation",
        session_id,
    ).data


@mcp.tool()
async def curve_get_type(
    curve_id: str,
    session_id: str,
) -> CurveGetType:
    """Get whether a curve is a line, arc, or NURBS curve.

    Args:
        curve_id: Curve UUID, typically obtained from an artifact graph.
        session_id: A modeling session populated by execute_kcl or exec_kcl_project.

    Returns:
        CurveGetType: The curve's geometric type.
    """
    logger.info("curve_get_type tool called")
    return zoo_execute_modeling_command(
        ModelingCmd(OptionCurveGetType(curve_id=Uuid(curve_id))),
        ResponseCurveGetType,
        "curve type",
        session_id,
    ).data


@mcp.tool()
async def edge_get_length(
    edge_id: str,
    session_id: str,
) -> EdgeGetLength:
    """Get the length of an edge entity in the current scene units.

    Args:
        edge_id: Edge UUID, typically obtained from an artifact graph.
        session_id: A modeling session populated by execute_kcl or exec_kcl_project.

    Returns:
        EdgeGetLength: The edge length in the current scene units.
    """
    logger.info("edge_get_length tool called")
    return zoo_execute_modeling_command(
        ModelingCmd(OptionEdgeGetLength(edge_id=Uuid(edge_id))),
        ResponseEdgeGetLength,
        "edge length",
        session_id,
    ).data


@mcp.tool()
async def entity_get_all_child_uuids(
    entity_id: str,
    session_id: str,
) -> EntityGetAllChildUuids:
    """Get all child UUIDs belonging to an entity.

    Args:
        entity_id: Entity UUID, typically obtained from an artifact graph.
        session_id: A modeling session populated by execute_kcl or exec_kcl_project.

    Returns:
        EntityGetAllChildUuids: All child entity UUIDs.
    """
    logger.info("entity_get_all_child_uuids tool called")
    return zoo_execute_modeling_command(
        ModelingCmd(OptionEntityGetAllChildUuids(entity_id=Uuid(entity_id))),
        ResponseEntityGetAllChildUuids,
        "entity child IDs",
        session_id,
    ).data


@mcp.tool()
async def entity_get_index(
    entity_id: str,
    session_id: str,
) -> EntityGetIndex:
    """Get an entity's index within its parent.

    Args:
        entity_id: Entity UUID, typically obtained from an artifact graph.
        session_id: A modeling session populated by execute_kcl or exec_kcl_project.

    Returns:
        EntityGetIndex: The entity's index within its parent.
    """
    logger.info("entity_get_index tool called")
    return zoo_execute_modeling_command(
        ModelingCmd(OptionEntityGetIndex(entity_id=Uuid(entity_id))),
        ResponseEntityGetIndex,
        "entity index",
        session_id,
    ).data


@mcp.tool()
async def entity_get_parent_id(
    entity_id: str,
    session_id: str,
) -> EntityGetParentId:
    """Get the UUID of an entity's parent.

    Args:
        entity_id: Entity UUID, typically obtained from an artifact graph.
        session_id: A modeling session populated by execute_kcl or exec_kcl_project.

    Returns:
        EntityGetParentId: The parent entity's UUID.
    """
    logger.info("entity_get_parent_id tool called")
    return zoo_execute_modeling_command(
        ModelingCmd(OptionEntityGetParentId(entity_id=Uuid(entity_id))),
        ResponseEntityGetParentId,
        "entity parent ID",
        session_id,
    ).data


@mcp.tool()
async def entity_get_sketch_paths(
    entity_id: str,
    session_id: str,
) -> EntityGetSketchPaths:
    """Get the sketch path UUIDs belonging to an entity.

    Args:
        entity_id: Entity UUID, typically obtained from an artifact graph.
        session_id: A modeling session populated by execute_kcl or exec_kcl_project.

    Returns:
        EntityGetSketchPaths: The sketch path UUIDs belonging to the entity.
    """
    logger.info("entity_get_sketch_paths tool called")
    return zoo_execute_modeling_command(
        ModelingCmd(OptionEntityGetSketchPaths(entity_id=Uuid(entity_id))),
        ResponseEntityGetSketchPaths,
        "entity sketch paths",
        session_id,
    ).data


# Shorthands for the two view sets worth asking for by name. Each tiles into a
# 2x2 collage in the order listed.
_CAMERA_VIEW_PRESETS: dict[str, list[str]] = {
    "multiview": ["front", "right", "top", "isometric"],
    "multi_isometric": [
        "isometric_front_right",
        "isometric_front_left",
        "isometric_back_right",
        "isometric_back_left",
    ],
}

# One snapshot per view on a 2x2 grid; more would not fit the collage.
_MAX_CAMERA_VIEWS = 4


def _resolve_camera_view(
    view: str | dict[str, list[float]],
) -> OptionDefaultCameraLookAt:
    """Turn a single named or explicit camera view into a modeling command.

    Named and explicit views share one conversion so the two paths cannot
    disagree about the coordinate frame.
    """
    if isinstance(view, dict):
        try:
            return CameraView.to_kittycad_camera(view)
        except (KeyError, IndexError, TypeError) as e:
            raise ZooMCPException(
                f"Invalid camera view {view}: expected 'up', 'vantage' and "
                f"'center' keys, each a list of 3 numbers ({e})"
            ) from e

    if view not in CameraView.views.value:
        raise ZooMCPException(
            f"Invalid camera view: {view}. Must be one of "
            f"{list(CameraView.views.value.keys())}"
        )
    return CameraView.to_kittycad_camera(CameraView.views.value[view])


def _resolve_camera_views(
    camera_view: str
    | dict[str, list[float]]
    | list[str | dict[str, list[float]]]
    | None,
) -> list[OptionDefaultCameraLookAt] | None:
    """Expand the camera_view argument into the list of views to capture.

    None means the scene's own framing decides, which zoo_snapshot renders
    isometrically when zooming and leaves alone otherwise.
    """
    if camera_view is None:
        return None

    if isinstance(camera_view, str) and camera_view in _CAMERA_VIEW_PRESETS:
        requested: list[str | dict[str, list[float]]] = list(
            _CAMERA_VIEW_PRESETS[camera_view]
        )
    elif isinstance(camera_view, list):
        requested = camera_view
    else:
        requested = [camera_view]

    if not requested:
        return None
    if len(requested) > _MAX_CAMERA_VIEWS:
        raise ZooMCPException(
            f"At most {_MAX_CAMERA_VIEWS} camera views can be captured at "
            f"once, got {len(requested)}"
        )
    return [_resolve_camera_view(view) for view in requested]


@mcp.tool()
async def snapshot(
    session_id: str,
    camera_view: str
    | dict[str, list[float]]
    | list[str | dict[str, list[float]]]
    | None = None,
    zoom: bool = True,
    highlight_edges: bool = False,
    max_image_dimension: int = 512,
    padding: float = 0.1,
    output_path: str | None = None,
) -> ImageContent | str:
    """Render an open modeling session as an image.

    Populate the scene first by passing this session_id to execute_kcl or
    exec_kcl_project. The camera always uses an orthographic projection, so
    measurements read off the image are not distorted by perspective.

    Args:
        session_id: A modeling session populated by execute_kcl or exec_kcl_project.
        camera_view: Which view or views to capture. Omit it for a single
                     isometric view. Otherwise one of:

            1. A named view: 'front', 'back', 'left', 'right', 'top',
               'bottom', 'isometric', 'isometric_front_right',
               'isometric_front_left', 'isometric_back_right',
               'isometric_back_left'.

               A named view puts the camera on that axis looking back at the
               origin, matching the app's standard views: 'front' is -Y,
               'back' is +Y, 'left' is -X, 'right' is +X, 'top' is +Z and
               'bottom' is -Z.

               So 'front' shows the face whose outward normal is -Y. The four
               isometric views all look down from above, from (+X, -Y, +Z)
               for 'isometric_front_right', (-X, -Y, +Z) for
               'isometric_front_left', (+X, +Y, +Z) for
               'isometric_back_right' and (-X, +Y, +Z) for
               'isometric_back_left'. Plain 'isometric' is front-right.

            2. A dict with "up", "vantage" and "center" keys, each a list of 3
               floats in model space: "vantage" is the camera position and
               "center" the point it looks at. For example
               {"up": [0, 0, 1], "vantage": [0, -1, 0], "center": [0, 0, 0]}
               looks at the origin from the front, showing the -Y face.

               With zoom=True only the direction from "center" to "vantage"
               matters, because zooming to fit sets the distance: [0, -1, 0]
               and [0, -200, 0] frame identically. "up" must not be parallel
               to that direction.

            3. 'multiview' for a 2x2 collage of front (top left), right (top
               right), top (bottom left) and isometric (bottom right).

            4. 'multi_isometric' for a 2x2 collage of the front-right (top
               left), front-left (top right), back-right (bottom left) and
               back-left (bottom right) isometric views.

            5. A list of up to 4 names and/or dicts, tiled in the order given.

        zoom: Zoom to fit before capturing each view. Leave True unless you
              have positioned the session's camera yourself; a freshly
              executed scene is not framed, so zoom=False renders the model
              only a few pixels wide.
        highlight_edges: Whether rendered edges should be outlined. Default is
                         False so that entities highlighted with
                         highlight_set_entities are obvious in the image.
        max_image_dimension: Maximum width or height of the returned JPEG.
        padding: Fraction of the frame left as margin when zooming to fit.
        output_path: If provided, the snapshot is written to disk and the
                     absolute file path is returned instead of the image. May
                     be a file path (e.g. '/path/to/image.jpg') or a directory
                     (in which case the file is named 'image.jpg'). If
                     omitted, the image is returned inline as an ImageContent.
                     The file is always JPEG data whatever extension is given,
                     so prefer '.jpg' to avoid writing a JPEG named '.png'.

    Returns:
        ImageContent | str: The snapshot as an inline image when output_path is
                            omitted; otherwise the absolute path to the saved file.
    """
    logger.info("snapshot tool called")

    image = zoo_snapshot(
        session_id=session_id,
        views=_resolve_camera_views(camera_view),
        max_image_dimension=max_image_dimension,
        padding=padding,
        zoom=zoom,
        highlight_edges=highlight_edges,
    )
    if output_path is not None:
        return save_image_bytes_to_disk(image, output_path)
    return encode_image(image)


@mcp.tool()
async def lint_and_fix_kcl(
    kcl_code: str | None = None,
    kcl_path: str | None = None,
) -> tuple[str, list[str]]:
    """Lint and fix KCL code given a string of KCL code or a path to a KCL project. Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point to a .kcl file or a directory containing .kcl files.

    Args:
        kcl_code (str | None): The KCL code to lint and fix.
        kcl_path (str | None): The path to a KCL file to lint and fix. The path should point to a .kcl file or a directory containing a main.kcl file.

    Returns:
        tuple[str, list[str]]: If kcl_code is provided, it returns a tuple containing the fixed KCL code and a list of unfixed lints.
                               If kcl_path is provided, it returns a tuple containing a success message and a list of unfixed lints for each file in the project.
    """

    logger.info("lint_and_fix_kcl tool called")

    try:
        res, lints = zoo_lint_and_fix_kcl(kcl_code=kcl_code, kcl_path=kcl_path)
        if isinstance(res, str):
            return res, lints
        else:
            return f"Successfully linted and fixed KCL code at: {kcl_path}", lints
    except Exception as e:
        return f"There was an error linting and fixing the KCL: {e}", []


@mcp.tool()
async def mock_execute_kcl(
    kcl_code: str | None = None,
    kcl_path: str | None = None,
) -> tuple[bool, str]:
    """Mock execute KCL code given a string of KCL code or a path to a KCL project. Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point to a .kcl file or a directory containing a main.kcl file.

    Args:
        kcl_code (str | None): The KCL code to mock execute.
        kcl_path (str | None): The path to a KCL file to mock execute. The path should point to a .kcl file or a directory containing a main.kcl file.

    Returns:
        tuple(bool, str): Returns True if the KCL code executed successfully and a success message, False otherwise and the error message.
    """

    logger.info("mock_execute_kcl tool called")

    try:
        return await zoo_mock_execute_kcl(kcl_code=kcl_code, kcl_path=kcl_path)
    except Exception as e:
        return False, f"Failed to mock execute KCL code: {e}"


@mcp.tool()
async def save_image(
    image: ImageContent,
    output_path: str | None = None,
) -> str:
    """Save an ImageContent object to disk. This allows a human to review images locally that an LLM has requested.

    Args:
        image (ImageContent): The ImageContent object to save. This is typically returned by the snapshot tool. Note that snapshot can write straight to disk via its own output_path argument; this tool is for images you already hold.
        output_path (str | None): The path where the image should be saved. Can be a file path (e.g., '/path/to/image.jpg') or a directory (e.g., '/path/to/dir'). If a directory is provided, the file will be named 'image.jpg'. If not provided, a temporary file will be created.

    Returns:
        str: The absolute path to the saved image file, or an error message if the operation fails.
    """

    logger.info("save_image tool called with output_path: %s", output_path)

    try:
        saved_path = save_image_to_disk(image=image, output_path=output_path)
        return saved_path
    except Exception as e:
        return f"There was an error saving the image: {e}"


@mcp.tool()
async def list_org_datasets() -> list[dict] | str:
    """List the datasets available to the user's organization.

    Each dataset has a UUID `id`, a human-readable `name`, and an optional
    `description`. Use the `id` as the `dataset_id` argument to
    `search_org_dataset_semantic`.

    Returns:
        A list of {"id": str, "name": str, "description": str | None}
        entries, or an error message if the operation fails.
    """

    logger.info("list_org_datasets tool called")

    try:
        return zoo_list_org_datasets()
    except Exception as e:
        return f"There was an error listing org datasets: {e}"


@mcp.tool()
async def list_org_skills() -> list[dict] | str:
    """List the skills available to the user's organization.

    Each skill has a UUID `id`, a human-readable `name`, a `description`, and a
    `markdown` body containing the skill's full content.

    Returns:
        A list of {"id": str, "name": str, "description": str, "markdown": str}
        entries, or an error message if the operation fails.
    """

    logger.info("list_org_skills tool called")

    try:
        return zoo_list_org_skills()
    except Exception as e:
        return f"There was an error listing org skills: {e}"


@mcp.tool()
async def search_org_dataset_semantic(
    dataset_id: str,
    query: str,
    limit: int | None = None,
) -> list[dict] | str:
    """Semantic-search a dataset for samples relevant to the query.

    Embeds the query with the org-dataset embedding model and returns the top
    chunk matches ranked by cosine similarity.

    Args:
        dataset_id (str): The UUID of the dataset to search (from `list_org_datasets`).
        query (str): The natural-language query to embed and search with.
        limit (int | None): Optional max number of matches to return.

    Returns:
        A list of match dicts (source_file_path, content, similarity, chunk_index, conversion_id),
        or an error message if the operation fails.
    """

    logger.info("search_org_dataset_semantic tool called for dataset_id=%s", dataset_id)

    try:
        return zoo_search_org_dataset_semantic(
            dataset_id=dataset_id, query=query, limit=limit
        )
    except Exception as e:
        return f"There was an error searching dataset {dataset_id}: {e}"


@mcp.tool()
async def list_kcl_docs() -> dict | str:
    """List all available KCL documentation topics organized by category.

    Returns a dictionary with the following categories:
    - kcl-lang: KCL language documentation (syntax, types, functions, etc.)
    - kcl-std-functions: Standard library function documentation
    - kcl-std-types: Standard library type documentation
    - kcl-std-consts: Standard library constants documentation
    - kcl-std-modules: Standard library module documentation

    Each category contains a list of documentation file paths that can be
    retrieved using get_kcl_doc().

    Returns:
        dict | str: Categories mapped to lists of available documentation paths.
        If there was an error, returns an error message string.
    """
    logger.info("list_kcl_docs tool called")
    try:
        await _ensure_kcl_indexes()
        return list_available_docs()
    except Exception as e:
        logger.error("list_kcl_docs tool called with error: %s", e)
        return f"There was an error listing KCL documentation: {e}"


@mcp.tool()
async def search_kcl_docs(query: str, max_results: int = 5) -> list[dict] | str:
    """Search KCL documentation by keyword.

    Searches across all KCL language and standard library documentation
    for the given query. Returns relevant excerpts with surrounding context.

    Args:
        query (str): The search query (case-insensitive).
        max_results (int): Maximum number of results to return (default: 5).

    Returns:
        list[dict] | str: List of search results, each containing:
            - path: The documentation file path
            - title: The document title (from first heading)
            - excerpt: A relevant excerpt with the match highlighted in context
            - match_count: Number of times the query appears in the document
            If there was an error, returns an error message string.
    """
    logger.info("search_kcl_docs tool called with query: %s", query)
    try:
        await _ensure_kcl_indexes()
        return search_docs(query, max_results)
    except Exception as e:
        logger.error("search_kcl_docs tool called with error: %s", e)
        return f"There was an error searching KCL documentation: {e}"


@mcp.tool()
async def get_kcl_doc(doc_path: str) -> str:
    """Get the full content of a specific KCL documentation file.

    Use list_kcl_docs() to see available documentation paths, or
    search_kcl_docs() to find relevant documentation by keyword.

    Args:
        doc_path (str): The path to the documentation file
            (e.g., "docs/kcl-lang/functions" or
            "docs/kcl-std/functions/std-sketch-extrude")

    Returns:
        str: The full Markdown content of the documentation file,
            or an error message if not found. If there was an error, returns an error message string.
    """
    logger.info("get_kcl_doc tool called for path: %s", doc_path)
    try:
        await _ensure_kcl_indexes()
        content = get_doc_content(doc_path)
        if content is None:
            return f"Documentation not found: {doc_path}. Use list_kcl_docs() to see available paths."
        return content
    except Exception as e:
        logger.error("get_kcl_doc tool called with error: %s", e)
        return f"There was an error retrieving KCL documentation: {e}"


@mcp.tool()
async def list_kcl_samples() -> list[dict] | str:
    """List all available KCL sample projects.

    Returns a list of all available KCL code samples from the Zoo samples
    repository. Each sample demonstrates a specific CAD modeling technique
    or creates a particular 3D model.

    Returns:
        list[dict] | str: List of sample information, each containing:
            - name: The sample directory name (use with get_kcl_sample)
            - title: Human-readable title
            - description: Brief description of what the sample creates
            - multipleFiles: Whether the sample contains multiple KCL files.
              Best-effort hint only. The /aquarium index doesn't expose file
              counts, so this is ``False`` for any sample that has not yet
              been fetched via get_kcl_sample. Call get_kcl_sample if you
              need a reliable answer.
            If there was an error, returns an error message string.
    """
    logger.info("list_kcl_samples tool called")
    try:
        await _ensure_kcl_indexes()
        return list_available_samples()
    except Exception as e:
        logger.error("list_kcl_samples tool called with error: %s", e)
        return f"There was an error listing KCL samples: {e}"


@mcp.tool()
async def search_kcl_samples(query: str, max_results: int = 5) -> list[dict] | str:
    """Search KCL samples by keyword.

    Searches across all KCL sample titles and descriptions
    for the given query. Returns matching samples ranked by relevance.

    Args:
        query (str): The search query (case-insensitive).
        max_results (int): Maximum number of results to return (default: 5).

    Returns:
        list[dict] | str: List of search results, each containing:
            - name: The sample directory name (use with get_kcl_sample)
            - title: Human-readable title
            - description: Brief description of the sample
            - multipleFiles: Whether the sample contains multiple KCL files.
              Best-effort hint only — see list_kcl_samples for the full
              caveat. Call get_kcl_sample if you need a reliable answer.
            - match_count: Number of times the query appears in title/description
            - excerpt: A relevant excerpt with the match in context
            If there was an error, returns an error message string.
    """
    logger.info("search_kcl_samples tool called with query: %s", query)
    try:
        await _ensure_kcl_indexes()
        return search_samples(query, max_results)
    except Exception as e:
        logger.error("search_kcl_samples tool called with error: %s", e)
        return f"There was an error searching KCL samples: {e}"


@mcp.tool()
async def get_kcl_sample(sample_name: str) -> SampleData | str:
    """Get the full content of a specific KCL sample including all files.

    Retrieves all KCL files that make up a sample project. Some samples
    consist of a single main.kcl file, while others have multiple files
    (e.g., parameters.kcl, components, etc.).

    Use list_kcl_samples() to see available sample names, or
    search_kcl_samples() to find samples by keyword.

    Args:
        sample_name (str): The sample directory name
            (e.g., "ball-bearing", "axial-fan", "gear")

    Returns:
        SampleData | str: A SampleData dictionary containing:
            - name: The sample directory name
            - title: Human-readable title
            - description: Brief description
            - multipleFiles: Whether the sample contains multiple files.
              Reliable here (unlike in list_kcl_samples / search_kcl_samples),
              because this tool fetches the per-sample page and counts the
              parsed files.
            - files: List of SampleFile dictionaries, each with 'filename' and 'content'
        Returns an error message string if the sample is not found. If there was an error, returns an error message string.
    """
    logger.info("get_kcl_sample tool called for sample: %s", sample_name)
    try:
        await _ensure_kcl_indexes()
        sample = await get_sample_content(sample_name)
        if sample is None:
            return f"Sample not found: {sample_name}. Use list_kcl_samples() to see available samples."
        return sample
    except Exception as e:
        logger.error("get_kcl_sample tool called with error: %s", e)
        return f"There was an error retrieving KCL sample: {e}"


def _shutdown_on_signal(signum: int, _frame: FrameType | None) -> None:
    """Turn a termination signal into the interrupt the server unwinds on."""
    logger.info("Received signal %s, shutting down MCP server...", signum)
    raise KeyboardInterrupt


def install_shutdown_handlers() -> None:
    """Make the server close its modeling sessions on the way out."""
    try:
        signal.signal(signal.SIGTERM, _shutdown_on_signal)
    except ValueError:
        # Only the main thread may install handlers; elsewhere the embedding
        # application owns signal disposition.
        logger.debug("Not on the main thread, leaving signal handlers alone")

    atexit.register(zoo_stop_all_modeling_sessions)


def main():
    logger.info("Starting MCP server...")
    install_shutdown_handlers()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
