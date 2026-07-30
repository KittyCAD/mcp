"""Live physical-properties tests against a heavy multi-module KCL assembly.

The fixture project is the one from KittyCAD/text-to-cad#3928: the engine
executes it but cannot measure it as a whole, while its individual sub-assemblies
measure fine.

These tests exist to keep that defect visible and to pin the shape of the failure
we report for it. ``test_heavy_assembly_measures`` is the desired behavior and is
expected to fail until the engine can measure the assembly; it will start passing
(as XPASS) when that lands, which is the signal to delete the xfail marker and
this note.

Marked ``live`` so they can be deselected with ``pytest -m 'not live'`` when
offline, and ``slow`` for ``pytest -m 'not slow'``. One whole-project attempt
normally costs ~25s but has been seen to run for ~10 minutes before the engine
reports a timed-out modeling command, so the measurement is made once per module
and shared. Sharing it only holds if the tests that use it run in the same
process, so both carry an ``xdist_group`` marker -- under ``-n`` with the default
``--dist load`` they would land on different workers and each pay for its own
measurement.
"""

import asyncio

import pytest
import pytest_asyncio

from zoo_mcp import ZooKclEngineError
from zoo_mcp.zoo_tools import zoo_calculate_kcl_physical_properties

pytestmark = [
    pytest.mark.live,
    pytest.mark.slow,
    pytest.mark.asyncio,
    # Keeps the whole-assembly measurement on one worker; see the module note.
    pytest.mark.xdist_group("heavy_assembly"),
]

# Generous enough for a slow-but-working engine, short enough that a hung
# modeling command cannot stall the suite for ten minutes.
MEASURE_TIMEOUT_SECONDS = 240

STEEL_DENSITY_KG_M3 = 7850


async def _measure(kcl_path: str) -> dict:
    return await asyncio.wait_for(
        zoo_calculate_kcl_physical_properties(
            kcl_code=None,
            kcl_path=kcl_path,
            unit_length="mm",
            unit_mass="kg",
            unit_density="kg:m3",
            density=STEEL_DENSITY_KG_M3,
            unit_area="mm2",
            unit_vol="cm3",
        ),
        timeout=MEASURE_TIMEOUT_SECONDS,
    )


@pytest_asyncio.fixture(scope="module")
async def whole_assembly_measurement(heavy_assembly_project: str):
    """Measure the whole assembly once, returning the properties or the failure."""
    try:
        return await _measure(heavy_assembly_project)
    except (ZooKclEngineError, TimeoutError) as error:
        return error


async def test_heavy_assembly_measurement_failure_is_reported_in_full(
    whole_assembly_measurement: dict | BaseException,
):
    """However the engine fails here, we must report it usefully.

    This is the contract the engine defect is currently exercising: a failure has
    to arrive as ``ZooKclEngineError`` naming the operation, carrying the engine's
    own message rather than the PyO3 tuple repr, and quoting the engine's
    identifiers whenever the engine supplied them.
    """
    if isinstance(whole_assembly_measurement, TimeoutError):
        pytest.fail(
            f"measuring the assembly exceeded {MEASURE_TIMEOUT_SECONDS}s; the "
            "engine neither answered nor reported a timed-out modeling command"
        )
    if not isinstance(whole_assembly_measurement, ZooKclEngineError):
        # The engine measured it. Nothing to assert about the failure path.
        assert isinstance(whole_assembly_measurement, dict)
        return

    error = whole_assembly_measurement
    assert error.operation == "execute_and_measure"
    assert error.message, "the engine's message must not be empty"
    assert not str(error).startswith("("), (
        f"the PyO3 tuple repr leaked into the message: {error}"
    )
    assert error.attempts >= 1
    # The engine quotes IDs for some failures and not others, but whenever it
    # does they must reach the caller in both the fields and the string form.
    for identifier in (error.modeling_command_id, error.api_call_id):
        if identifier is not None:
            assert identifier in str(error)


async def test_heavy_assembly_sub_module_measures(heavy_assembly_sub_module: str):
    """One sub-assembly must measure, which localizes the defect to assembly size.

    If this ever starts failing too, the problem is broader than the known
    whole-assembly defect and the xfail below is no longer the whole story.
    """
    properties = await _measure(heavy_assembly_sub_module)

    assert properties["volume"] > 0
    assert properties["mass"] > 0
    assert properties["surface_area"] > 0
    dimensions = properties["bounding_box"]["dimensions"]
    assert all(dimensions[axis] > 0 for axis in ("x", "y", "z"))


@pytest.mark.xfail(
    reason=(
        "the engine cannot measure this assembly as a whole: it returns "
        '"internal error: unknown"'
    ),
    strict=False,
)
async def test_heavy_assembly_measures(
    whole_assembly_measurement: dict | BaseException,
):
    """The whole assembly should measure. It does not, yet."""
    if isinstance(whole_assembly_measurement, BaseException):
        raise whole_assembly_measurement

    assert whole_assembly_measurement["volume"] > 0
    assert whole_assembly_measurement["mass"] > 0
