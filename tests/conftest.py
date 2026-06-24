import shutil
from pathlib import Path

import pytest


@pytest.fixture
def cube_kcl():
    test_file = Path(__file__).parent / "data" / "cube.kcl"
    yield f"{test_file.resolve()}"


@pytest.fixture
def cube_stl():
    test_file = Path(__file__).parent / "data" / "cube.stl"
    yield f"{test_file.resolve()}"


@pytest.fixture
def empty_kcl():
    test_file = Path(__file__).parent / "data" / "empty.kcl"
    yield f"{test_file.resolve()}"


@pytest.fixture
def empty_step():
    test_file = Path(__file__).parent / "data" / "empty.step"
    yield f"{test_file.resolve()}"


@pytest.fixture
def kcl_project():
    project_path = Path(__file__).parent / "data" / "test_kcl_project"
    yield f"{project_path.resolve()}"


@pytest.fixture
def box_with_linter_errors():
    test_file = Path(__file__).parent / "data" / "box_with_linter_errors.kcl"
    yield f"{test_file.resolve()}"


@pytest.fixture
def cube2_step_uppercase():
    """Fixture for a STEP file with uppercase extension to test case insensitivity."""
    test_file = Path(__file__).parent / "data" / "cube2.STEP"
    yield f"{test_file.resolve()}"


@pytest.fixture
def cube_stp(tmp_path):
    """Fixture for a STEP file with .stp extension to test extension alias handling."""
    source_file = Path(__file__).parent / "data" / "cube2.STEP"
    dest_file = tmp_path / "cube.stp"
    shutil.copy(source_file, dest_file)
    yield f"{dest_file.resolve()}"


@pytest.fixture
def warning_kcl():
    """KCL that executes successfully but emits a warning-level issue."""
    test_file = Path(__file__).parent / "data" / "execute_with_warning.kcl"
    yield f"{test_file.resolve()}"


@pytest.fixture
def error_kcl():
    """KCL that executes successfully but emits error-level (non-fatal) issues."""
    test_file = Path(__file__).parent / "data" / "execute_with_error.kcl"
    yield f"{test_file.resolve()}"


@pytest.fixture
def fatal_error_kcl():
    """KCL that aborts execution with a fatal error (raised by the engine)."""
    test_file = Path(__file__).parent / "data" / "execute_with_fatal_error.kcl"
    yield f"{test_file.resolve()}"


@pytest.fixture
def fully_constrained_kcl():
    test_file = Path(__file__).parent / "data" / "fully_constrained_sketch.kcl"
    yield f"{test_file.resolve()}"


@pytest.fixture
def under_constrained_kcl():
    test_file = Path(__file__).parent / "data" / "under_constrained_sketch.kcl"
    yield f"{test_file.resolve()}"
