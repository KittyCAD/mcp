import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import kcl
from kittycad.models.modeling_cmd import OptionDefaultCameraLookAt, Point3d
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
from zoo_mcp.utils.image_utils import encode_image, save_image_to_disk
from zoo_mcp.zoo_tools import (
    CameraView,
    zoo_calculate_bounding_box_cad,
    zoo_calculate_bounding_box_kcl,
    zoo_calculate_cad_physical_properties,
    zoo_calculate_center_of_mass,
    zoo_calculate_kcl_physical_properties,
    zoo_calculate_mass,
    zoo_calculate_surface_area,
    zoo_calculate_volume,
    zoo_convert_cad_file,
    zoo_execute_kcl,
    zoo_export_kcl,
    zoo_format_kcl,
    zoo_get_sketch_constraint_status,
    zoo_lint_and_fix_kcl,
    zoo_list_org_datasets,
    zoo_mock_execute_kcl,
    zoo_multi_isometric_snapshot_of_cad,
    zoo_multi_isometric_snapshot_of_kcl,
    zoo_multiview_snapshot_of_cad,
    zoo_multiview_snapshot_of_kcl,
    zoo_search_org_dataset_semantic,
    zoo_snapshot_of_cad,
    zoo_snapshot_of_kcl,
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
        unit_volume (str): The unit of volume to return the result in. One of 'cm3', 'ft3', 'in3', 'm3', 'yd3', 'usfloz', 'usgal', 'l', 'ml'.

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
        unit_volume (str): The unit of volume. One of 'cm3', 'ft3', 'in3', 'm3', 'yd3', 'usfloz', 'usgal', 'l', 'ml'.

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
        unit_volume (str): The unit of volume. One of 'cm3', 'ft3', 'in3', 'm3', 'yd3', 'usfloz', 'usgal', 'l', 'ml'.

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
) -> tuple[bool, str]:
    """Execute KCL code given a string of KCL code or a path to a KCL project. Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point to a .kcl file or a directory containing a main.kcl file.

    Args:
        kcl_code (str | None): The KCL code to execute.
        kcl_path (str | None): The path to a KCL file to execute. The path should point to a .kcl file or a directory containing a main.kcl file.

    Returns:
        tuple(bool, str): Returns True if the KCL code executed successfully and a success message, False otherwise and the error message.
    """

    logger.info("execute_kcl tool called")

    try:
        return await zoo_execute_kcl(kcl_code=kcl_code, kcl_path=kcl_path)
    except Exception as e:
        return False, f"Failed to execute KCL code: {e}"


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
async def multiview_snapshot_of_cad(
    input_file: str,
    zoom: bool = True,
) -> ImageContent | str:
    """Save a multiview snapshot of a CAD file. The input file should be one of the supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp, .stl (case-insensitive)

    This multiview image shows the render of the model from 4 different views:
        The top left images is a front view.
        The top right image is a right side view.
        The bottom left image is a top view.
        The bottom right image is an isometric view

    Args:
        input_file (str): The path of the file to get the mass from. The file should be one of the supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp, .stl (case-insensitive)
        zoom (bool): Whether to zoom-to-fit the model before each snapshot. Default is True.

    Returns:
        ImageContent | str: The multiview snapshot of the CAD file as an image, or an error message if the operation fails.
    """

    logger.info("multiview_snapshot_of_cad tool called for file: %s", input_file)

    try:
        image = zoo_multiview_snapshot_of_cad(
            input_path=input_file,
            zoom=zoom,
        )
        return encode_image(image)
    except Exception as e:
        return f"There was an error creating the multiview snapshot: {e}"


@mcp.tool()
async def multiview_snapshot_of_kcl(
    kcl_code: str | None = None,
    kcl_path: str | None = None,
    zoom: bool = True,
) -> ImageContent | str:
    """Save a multiview snapshot of KCL code. Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point to a .kcl file or a directory containing a main.kcl file.

    This multiview image shows the render of the model from 4 different views:
        The top left images is a front view.
        The top right image is a right side view.
        The bottom left image is a top view.
        The bottom right image is an isometric view

    Args:
        kcl_code (str | None): The KCL code to export to a CAD file.
        kcl_path (str | None): The path to a KCL file to export to a CAD file. The path should point to a .kcl file or a directory containing a main.kcl file.
        zoom (bool): Whether to zoom-to-fit the model before each snapshot. Default is True.

    Returns:
        ImageContent | str: The multiview snapshot of the KCL code as an image, or an error message if the operation fails.
    """

    logger.info("multiview_snapshot_of_kcl tool called")

    try:
        image = await zoo_multiview_snapshot_of_kcl(
            kcl_code=kcl_code,
            kcl_path=kcl_path,
            zoom=zoom,
        )
        return encode_image(image)
    except Exception as e:
        return f"There was an error creating the multiview snapshot: {e}"


@mcp.tool()
async def multi_isometric_snapshot_of_cad(
    input_file: str,
    zoom: bool = True,
) -> ImageContent | str:
    """Save a multi-isometric snapshot of a CAD file showing 4 isometric views. The input file should be one of the supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp, .stl (case-insensitive)

    This multi-isometric image shows the render of the model from 4 different isometric views:
        The top left image is an isometric view from the front-right corner.
        The top right image is an isometric view from the front-left corner.
        The bottom left image is an isometric view from the back-right corner.
        The bottom right image is an isometric view from the back-left corner.

    Args:
        input_file (str): The path of the file to snapshot. The file should be one of the supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp, .stl (case-insensitive)
        zoom (bool): Whether to zoom-to-fit the model before each snapshot. Default is True.

    Returns:
        ImageContent | str: The multi-isometric snapshot of the CAD file as an image, or an error message if the operation fails.
    """

    logger.info("multi_isometric_snapshot_of_cad tool called for file: %s", input_file)

    try:
        image = zoo_multi_isometric_snapshot_of_cad(
            input_path=input_file,
            zoom=zoom,
        )
        return encode_image(image)
    except Exception as e:
        return f"There was an error creating the multi-isometric snapshot: {e}"


@mcp.tool()
async def multi_isometric_snapshot_of_kcl(
    kcl_code: str | None = None,
    kcl_path: str | None = None,
    zoom: bool = True,
) -> ImageContent | str:
    """Save a multi-isometric snapshot of KCL code showing 4 isometric views. Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point to a .kcl file or a directory containing a main.kcl file.

    This multi-isometric image shows the render of the model from 4 different isometric views:
        The top left image is an isometric view from the front-right corner.
        The top right image is an isometric view from the front-left corner.
        The bottom left image is an isometric view from the back-right corner.
        The bottom right image is an isometric view from the back-left corner.

    Args:
        kcl_code (str | None): The KCL code to export to a CAD file.
        kcl_path (str | None): The path to a KCL file to export to a CAD file. The path should point to a .kcl file or a directory containing a main.kcl file.
        zoom (bool): Whether to zoom-to-fit the model before each snapshot. Default is True.

    Returns:
        ImageContent | str: The multi-isometric snapshot of the KCL code as an image, or an error message if the operation fails.
    """

    logger.info("multi_isometric_snapshot_of_kcl tool called")

    try:
        image = await zoo_multi_isometric_snapshot_of_kcl(
            kcl_code=kcl_code,
            kcl_path=kcl_path,
            zoom=zoom,
        )
        return encode_image(image)
    except Exception as e:
        return f"There was an error creating the multi-isometric snapshot: {e}"


@mcp.tool()
async def snapshot_of_cad(
    input_file: str,
    camera_view: dict[str, list[float]] | str = "isometric",
    zoom: bool = True,
) -> ImageContent | str:
    """Save a snapshot of a CAD file.

    Args:
        input_file (str): The path of the file to get the mass from. The file should be one of the supported formats: .fbx, .gltf, .obj, .ply, .sldprt, .step, .stp, .stl (case-insensitive)
        camera_view (dict | str): The camera to use for the snapshot.

            1. If a string is provided, it should be one of 'front', 'back', 'left', 'right', 'top', 'bottom', 'isometric', 'isometric_front_right', 'isometric_front_left', 'isometric_back_right', 'isometric_back_left' to set the camera to a predefined view.

            2. If a dict is provided, supply a dict with the following keys and values:
               "up" (list of 3 floats) defining the up vector of the camera, "vantage" (list of 3 floats), and "center" (list of 3 floats).
               For example camera = {"up": [0, 0, 1], "vantage": [0, -1, 0], "center": [0, 0, 0]} would set the camera to be looking at the origin from the front side (-y direction).
        zoom (bool): Whether to zoom-to-fit the model before the snapshot. Default is True.

    Returns:
        ImageContent | str: The snapshot of the CAD file as an image, or an error message if the operation fails.
    """

    logger.info("snapshot_of_cad tool called for file: %s", input_file)

    try:
        if isinstance(camera_view, dict):
            camera = OptionDefaultCameraLookAt(
                up=Point3d(
                    x=camera_view["up"][0],
                    y=camera_view["up"][1],
                    z=camera_view["up"][2],
                ),
                vantage=Point3d(
                    x=camera_view["vantage"][0],
                    y=-camera_view["vantage"][1],
                    z=camera_view["vantage"][2],
                ),
                center=Point3d(
                    x=camera_view["center"][0],
                    y=camera_view["center"][1],
                    z=camera_view["center"][2],
                ),
            )
        else:
            if camera_view not in CameraView.views.value:
                raise ZooMCPException(
                    f"Invalid camera view: {camera_view}. Must be one of {list(CameraView.views.value.keys())}"
                )
            camera = CameraView.to_kittycad_camera(CameraView.views.value[camera_view])

        image = zoo_snapshot_of_cad(
            input_path=input_file,
            camera=camera,
            zoom=zoom,
        )
        return encode_image(image)
    except Exception as e:
        return f"There was an error creating the snapshot: {e}"


@mcp.tool()
async def snapshot_of_kcl(
    kcl_code: str | None = None,
    kcl_path: str | None = None,
    camera_view: dict[str, list[float]] | str = "isometric",
    zoom: bool = True,
) -> ImageContent | str:
    """Save a snapshot of a model represented by KCL. Either kcl_code or kcl_path must be provided. If kcl_path is provided, it should point to a .kcl file or a directory containing a main.kcl file.

    Args:
        kcl_code (str | None): The KCL code to export to a CAD file.
        kcl_path (str | None): The path to a KCL file to export to a CAD file. The path should point to a .kcl file or a directory containing a main.kcl file.
        camera_view (dict | str): The camera to use for the snapshot.

            1. If a string is provided, it should be one of 'front', 'back', 'left', 'right', 'top', 'bottom', 'isometric', 'isometric_front_right', 'isometric_front_left', 'isometric_back_right', 'isometric_back_left' to set the camera to a predefined view.

            2. If a dict is provided, supply a dict with the following keys and values:
               "up" (list of 3 floats) defining the up vector of the camera, "vantage" (list of 3 floats), and "center" (list of 3 floats).
               For example camera = {"up": [0, 0, 1], "vantage": [0, -1, 0], "center": [0, 0, 0]} would set the camera to be looking at the origin from the front side (-y direction).
        zoom (bool): Whether to zoom-to-fit the model before the snapshot. Default is True.

    Returns:
        ImageContent | str: The snapshot of the CAD file as an image, or an error message if the operation fails.
    """

    logger.info("snapshot_of_kcl tool called")

    try:
        if isinstance(camera_view, dict):
            camera = kcl.CameraLookAt(
                up=kcl.Point3d(
                    x=camera_view["up"][0],
                    y=camera_view["up"][1],
                    z=camera_view["up"][2],
                ),
                vantage=kcl.Point3d(
                    x=camera_view["vantage"][0],
                    y=-camera_view["vantage"][1],
                    z=camera_view["vantage"][2],
                ),
                center=kcl.Point3d(
                    x=camera_view["center"][0],
                    y=camera_view["center"][1],
                    z=camera_view["center"][2],
                ),
            )
        else:
            if camera_view not in CameraView.views.value:
                raise ZooMCPException(
                    f"Invalid camera view: {camera_view}. Must be one of {list(CameraView.views.value.keys())}"
                )
            camera = CameraView.to_kcl_camera(CameraView.views.value[camera_view])

        image = await zoo_snapshot_of_kcl(
            kcl_code=kcl_code,
            kcl_path=kcl_path,
            camera=camera,
            zoom=zoom,
        )
        return encode_image(image)
    except Exception as e:
        return f"There was an error creating the snapshot: {e}"


@mcp.tool()
async def save_image(
    image: ImageContent,
    output_path: str | None = None,
) -> str:
    """Save an ImageContent object to disk. This allows a human to review images locally that an LLM has requested.

    Args:
        image (ImageContent): The ImageContent object to save. This is typically returned by snapshot tools like snapshot_of_kcl, snapshot_of_cad, multiview_snapshot_of_kcl, or multiview_snapshot_of_cad.
        output_path (str | None): The path where the image should be saved. Can be a file path (e.g., '/path/to/image.png') or a directory (e.g., '/path/to/dir'). If a directory is provided, the file will be named 'image.png'. If not provided, a temporary file will be created.

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


def main():
    logger.info("Starting MCP server...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
