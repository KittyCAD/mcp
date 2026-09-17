import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import kcl
import pytest
from mcp.types import CallToolResult

from zoo_mcp import zoo_tools
from zoo_mcp.server import mcp


@dataclass
class Issue:
    severity: str

    def message(self):
        return "PRIVATE_CUSTOMER_CONTENT"

    def is_fatal(self):
        return self.severity == "fatal"

    def is_err(self):
        return self.severity in ("error", "fatal")

    def is_warning(self):
        return self.severity == "warning"


class Outcome:
    def __init__(self, *severities: str):
        self.severities = severities

    def issues(self):
        return [Issue(severity) for severity in self.severities]

    def report(self, issue):
        return f"{issue.severity}: PRIVATE_CUSTOMER_CONTENT"


@pytest.fixture(params=["local", "session", "project"])
def execution_route(request):
    return request.param


@pytest.fixture(params=["code", "file", "directory"])
def execution_input(request, tmp_path):
    if request.param == "code":
        return {"kcl_code": "x = 1"}
    project = tmp_path / "project"
    project.mkdir()
    entrypoint = project / ("part.kcl" if request.param == "file" else "main.kcl")
    entrypoint.write_text("import x from 'library.kcl'\ny = x\n")
    (project / "library.kcl").write_text("export x = 1\n")
    (project / "project.toml").write_text("[settings]\n")
    return {"kcl_path": str(entrypoint if request.param == "file" else project)}


async def execute(route: str, arguments: dict[str, Any]):
    if route == "project":
        return await zoo_tools.zoo_exec_kcl_project(
            session_id="session-id", **arguments
        )
    return await zoo_tools.zoo_execute_kcl(
        session_id="session-id" if route == "session" else None, **arguments
    )


def mock_bindings(monkeypatch, mock, real):
    monkeypatch.setattr(kcl, "mock_execute_code", mock)
    monkeypatch.setattr(kcl, "mock_execute", mock)
    monkeypatch.setattr(kcl, "execute_code", real)
    monkeypatch.setattr(kcl, "execute", real)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "fatal", "thrown"])
async def test_mock_failure_returns_before_waiting_for_real_execution(
    monkeypatch,
    execution_route,
    execution_input,
    failure,
):
    """Even an indefinitely stalled real execution must not delay mock failure."""

    async def never_finishes(*args, **kwargs):
        await asyncio.Event().wait()

    real = AsyncMock(side_effect=never_finishes)
    mock = AsyncMock(
        side_effect=ValueError("mock aborted") if failure == "thrown" else None,
        return_value=Outcome(failure),
    )
    mock_bindings(monkeypatch, mock, real)
    monkeypatch.setattr(zoo_tools, "_execute_resolved_kcl_project", real)
    transport = AsyncMock(side_effect=never_finishes)
    monkeypatch.setattr(zoo_tools, "_exec_kcl_project", transport)
    websocket = MagicMock(side_effect=AssertionError("must not acquire session"))
    monkeypatch.setattr(zoo_tools, "_modeling_websocket", websocket)

    result = await asyncio.wait_for(
        execute(execution_route, execution_input), timeout=1
    )

    assert not result.ok
    assert result.mock_preflight.status == "failed"
    assert result.real_execution.status == "not_run"
    mock.assert_awaited_once()
    real.assert_not_called()
    real.assert_not_awaited()
    transport.assert_not_called()
    websocket.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("warning", [False, True])
@pytest.mark.parametrize("real_failure", [False, True])
async def test_preflight_order_counts_and_separate_outcomes(
    monkeypatch,
    execution_route,
    execution_input,
    warning,
    real_failure,
):
    calls = []

    async def preflight(*args):
        calls.append("mock")
        return Outcome(*(["warning"] if warning else []))

    async def run(*args):
        calls.append("real")
        if real_failure:
            raise ValueError("engine rejected execution")
        return Outcome() if execution_route == "local" else Path("artifact.json")

    mock = AsyncMock(side_effect=preflight)
    real = AsyncMock(side_effect=run)
    mock_bindings(monkeypatch, mock, real)
    monkeypatch.setattr(zoo_tools, "_execute_resolved_kcl_project", real)

    with zoo_tools.capture_execution_stage_events() as events:
        result = await execute(execution_route, execution_input)

    assert calls == ["mock", "real"]
    mock.assert_awaited_once()
    real.assert_awaited_once()
    assert result.ok is not real_failure
    assert result.mock_preflight.status == "succeeded"
    assert bool(result.mock_preflight.diagnostics.get("warning")) is warning
    assert result.real_execution.status == ("failed" if real_failure else "succeeded")
    assert result.real_execution.diagnostics == {}
    assert [event.stage for event in events] == ["mock_preflight", "real_execution"]
    assert [event.attempts for event in events] == [1, 1]
    if execution_route != "local" and not real_failure:
        assert result.path_artifact_graph == Path("artifact.json")
        assert (
            "Real-execution diagnostics are not reported"
            in result.real_execution.message
        )


@pytest.mark.asyncio
async def test_real_execution_cannot_start_while_preflight_is_pending(
    monkeypatch,
    execution_route,
):
    entered, release = asyncio.Event(), asyncio.Event()

    async def preflight(*args):
        entered.set()
        await release.wait()
        return Outcome()

    mock = AsyncMock(side_effect=preflight)
    real = AsyncMock(
        return_value=Outcome() if execution_route == "local" else Path("graph.json")
    )
    mock_bindings(monkeypatch, mock, real)
    monkeypatch.setattr(zoo_tools, "_execute_resolved_kcl_project", real)
    task = asyncio.create_task(execute(execution_route, {"kcl_code": "x = 1"}))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        real.assert_not_called()
    finally:
        release.set()
        await asyncio.wait_for(task, timeout=1)
    mock.assert_awaited_once()
    real.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["main.kcl", "part.kcl"])
async def test_both_stages_use_captured_project_after_original_files_change(
    monkeypatch,
    tmp_path,
    execution_route,
    entrypoint,
):
    project = tmp_path / "project"
    project.mkdir()
    original = {
        entrypoint: b"import x from 'library.kcl'\nimport 'assets/model.stl' as model\ny = x\n",
        "library.kcl": b"export x = 1\n",
        "project.toml": b"[settings]\n",
        "assets/model.stl": b"\x00\xff",
    }
    for name, contents in original.items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    captured = []

    async def preflight(path):
        captured.append(Path(path))
        assert Path(path).name == entrypoint
        assert Path(path).parent != project
        for name, contents in original.items():
            assert (Path(path).parent / name).read_bytes() == contents
            (project / name).write_bytes(b"changed after capture")
        return Outcome()

    async def local(path):
        assert Path(path) == captured[0]
        for name, contents in original.items():
            assert (Path(path).parent / name).read_bytes() == contents
        return Outcome()

    async def remote(session_id, actual_entrypoint, files):
        assert session_id == "session-id"
        assert actual_entrypoint == entrypoint
        assert {file["path"]: bytes(file["contents"]) for file in files} == original
        return Path("artifact.json")

    mock = AsyncMock(side_effect=preflight)
    real = AsyncMock(side_effect=local)
    mock_bindings(monkeypatch, mock, real)
    monkeypatch.setattr(zoo_tools, "_execute_resolved_kcl_project", remote)
    result = await execute(
        execution_route,
        {
            "kcl_path": str(
                project if entrypoint == "main.kcl" else project / entrypoint
            )
        },
    )
    assert result.ok
    assert not captured[0].parent.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("exhausted", [False, True])
async def test_transient_retries_reuse_preflight_and_captured_input(
    monkeypatch,
    tmp_path,
    exhausted,
):
    source = tmp_path / "main.kcl"
    source.write_text("x = 1")
    calls = []
    captured = []

    async def preflight(path):
        calls.append("mock")
        captured.append(path)
        return Outcome("warning")

    async def run(path):
        calls.append("real")
        assert path == captured[0]
        assert Path(path).read_text() == "x = 1"
        source.write_text("changed")
        if exhausted or calls.count("real") == 1:
            raise kcl.KclError("KCL EngineHangup error", True)
        return Outcome()

    mock_bindings(
        monkeypatch, AsyncMock(side_effect=preflight), AsyncMock(side_effect=run)
    )
    monkeypatch.setattr(zoo_tools, "_execution_retry_delay", lambda attempt: 0)
    with (
        zoo_tools.capture_execution_retry_events() as retries,
        zoo_tools.capture_execution_stage_events() as stages,
    ):
        result = await zoo_tools.zoo_execute_kcl(kcl_path=source)
    attempts = zoo_tools.MAX_EXECUTION_ATTEMPTS if exhausted else 2
    assert calls == ["mock"] + ["real"] * attempts
    assert len(retries) == attempts
    assert [stage.attempts for stage in stages] == [1, attempts]
    assert result.ok is not exhausted
    assert result.mock_preflight.diagnostics["warning"]
    assert not Path(captured[0]).parent.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["mock", "real"])
async def test_cancellation_cleans_up_captured_project(monkeypatch, tmp_path, phase):
    source = tmp_path / "main.kcl"
    source.write_text("x = 1")
    entered = asyncio.Event()
    captured = []

    async def wait(path):
        captured.append(Path(path))
        entered.set()
        await asyncio.Event().wait()

    mock = (
        AsyncMock(side_effect=wait)
        if phase == "mock"
        else AsyncMock(return_value=Outcome())
    )
    real = AsyncMock(side_effect=wait)
    mock_bindings(monkeypatch, mock, real)
    with zoo_tools.capture_execution_stage_events() as events:
        task = asyncio.create_task(zoo_tools.zoo_execute_kcl(kcl_path=source))
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert not captured[0].parent.exists()
    assert events[-1].outcome == "cancelled"
    if phase == "mock":
        real.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "mock", "real"])
async def test_stage_telemetry_times_each_phase_without_contents(
    monkeypatch, caplog, failure
):
    clock = [100.0]
    monkeypatch.setattr(zoo_tools, "monotonic", lambda: clock[0])

    async def preflight(code):
        clock[0] += 2
        if failure == "mock":
            raise ValueError("PRIVATE_CUSTOMER_CONTENT")
        return Outcome("warning")

    async def run(code):
        clock[0] += 7
        if failure == "real":
            raise ValueError("PRIVATE_CUSTOMER_CONTENT")
        return Outcome()

    mock_bindings(
        monkeypatch, AsyncMock(side_effect=preflight), AsyncMock(side_effect=run)
    )
    with (
        zoo_tools.capture_execution_stage_events() as events,
        caplog.at_level("INFO", logger="zoo_mcp"),
    ):
        result = await zoo_tools.zoo_execute_kcl(kcl_code="PRIVATE_CUSTOMER_CONTENT")
    assert [event.elapsed_seconds for event in events] == [
        2,
        0 if failure == "mock" else 7,
    ]
    assert events[0].outcome == ("failed" if failure == "mock" else "succeeded")
    assert (
        events[1].outcome
        == {None: "succeeded", "mock": "not_run", "real": "failed"}[failure]
    )
    assert all(event.operation == "execute_kcl" for event in events)
    assert "PRIVATE_CUSTOMER_CONTENT" not in repr(events) + caplog.text
    assert "PRIVATE_CUSTOMER_CONTENT" in repr(result.mock_preflight)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["x = (", "x = undefined_variable"])
async def test_actual_mock_parse_and_semantic_failures_block_real_execution(
    monkeypatch,
    execution_route,
    code,
):
    real = AsyncMock(side_effect=AssertionError("must not execute"))
    monkeypatch.setattr(kcl, "execute_code", real)
    monkeypatch.setattr(zoo_tools, "_execute_resolved_kcl_project", real)
    result = await asyncio.wait_for(
        execute(execution_route, {"kcl_code": code}), timeout=2
    )
    assert not result.ok
    assert result.mock_preflight.status == "failed"
    assert result.real_execution.status == "not_run"
    real.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocking", [False, True])
async def test_mcp_tools_serialize_separate_stage_results(
    monkeypatch,
    execution_route,
    blocking,
):
    mock = AsyncMock(return_value=Outcome("error" if blocking else "warning"))
    real = AsyncMock(return_value=Outcome())
    mock_bindings(monkeypatch, mock, real)
    remote = AsyncMock(return_value=Path("artifact.json"))
    monkeypatch.setattr(zoo_tools, "_execute_resolved_kcl_project", remote)
    arguments = {"kcl_code": "x = 1"}
    if execution_route != "local":
        arguments["session_id"] = "session-id"
    response = await mcp.call_tool(
        "exec_kcl_project" if execution_route == "project" else "execute_kcl",
        arguments=arguments,
    )
    assert isinstance(response, CallToolResult)
    assert response.structured_content is not None
    result = response.structured_content["result"]
    assert result["ok"] is not blocking
    assert result["mock_preflight"]["status"] == ("failed" if blocking else "succeeded")
    assert result["real_execution"]["status"] == (
        "not_run" if blocking else "succeeded"
    )
    assert result["mock_preflight"]["diagnostics"]["error" if blocking else "warning"]
    if execution_route != "local" and not blocking:
        assert result["path_artifact_graph"] == "artifact.json"


@pytest.mark.asyncio
async def test_invalid_input_returns_failed_preflight(monkeypatch, execution_route):
    mock = AsyncMock(side_effect=AssertionError("must not execute"))
    real = AsyncMock(side_effect=AssertionError("must not execute"))
    mock_bindings(monkeypatch, mock, real)
    result = await execute(execution_route, {})
    assert result.mock_preflight.status == "failed"
    assert result.real_execution.status == "not_run"
    mock.assert_not_called()
    real.assert_not_called()


@pytest.mark.asyncio
async def test_actual_mock_resolves_imports_from_captured_project(
    monkeypatch,
    tmp_path,
    execution_route,
):
    library = tmp_path / "library.kcl"
    library.write_text("export x = 1\n")
    (tmp_path / "main.kcl").write_text('import x from "library.kcl"\ny = x + 1\n')
    real = AsyncMock(
        return_value=Outcome() if execution_route == "local" else Path("graph.json")
    )
    monkeypatch.setattr(kcl, "execute", real)
    monkeypatch.setattr(zoo_tools, "_execute_resolved_kcl_project", real)
    result = await execute(execution_route, {"kcl_path": str(tmp_path)})
    assert result.ok, result
    assert result.mock_preflight.status == "succeeded"
    real.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("has_code_error", [False, True])
async def test_plane_of_mock_limitation_allows_real_execution(
    monkeypatch, execution_route, cube_kcl, has_code_error
):
    code = Path(cube_kcl).read_text() + "\np = planeOf(cube, face = END)\n"
    if has_code_error:
        code += "x = sqrt(4, unexpected = 1)\n"
    real = AsyncMock(
        return_value=Outcome() if execution_route == "local" else Path("graph.json")
    )
    monkeypatch.setattr(kcl, "execute_code", real)
    monkeypatch.setattr(zoo_tools, "_execute_resolved_kcl_project", real)

    result = await execute(execution_route, {"kcl_code": code})

    assert result.ok is not has_code_error
    assert (
        "The engine isn't available" in result.mock_preflight.diagnostics["warning"][0]
    )
    assert real.await_count == (0 if has_code_error else 1)
    if has_code_error:
        assert result.mock_preflight.status == "failed"
        assert "`unexpected`" in result.mock_preflight.diagnostics["error"][0]
        assert result.real_execution.status == "not_run"
    else:
        assert result.mock_preflight.status == "succeeded"
        assert result.real_execution.status == "succeeded"


@pytest.mark.asyncio
async def test_mock_limitation_text_in_source_cannot_hide_an_error(
    monkeypatch, execution_route
):
    code = (
        "x = sqrt(4, unexpected = \"The engine isn't available, "
        'so returning an arbitrary incorrect plane")'
    )
    real = AsyncMock(side_effect=AssertionError("must not execute"))
    monkeypatch.setattr(kcl, "execute_code", real)
    monkeypatch.setattr(zoo_tools, "_execute_resolved_kcl_project", real)
    result = await execute(execution_route, {"kcl_code": code})
    assert not result.ok
    assert result.mock_preflight.diagnostics["error"]
    assert result.real_execution.status == "not_run"
    real.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("input_is_directory", [False, True])
async def test_capture_ignores_unrelated_files(
    monkeypatch, tmp_path, execution_route, input_is_directory
):
    entry = tmp_path / "main.kcl"
    entry.write_text("x = 1\n")
    # Even a supported CAD extension must not cause an unrelated file read.
    unrelated = tmp_path / "unrelated.step"
    unrelated.write_bytes(b"not a dependency")
    read_bytes = Path.read_bytes

    def guarded_read(path):
        if path == unrelated:
            raise PermissionError("unrelated file is unreadable")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    real = AsyncMock(
        return_value=Outcome() if execution_route == "local" else Path("graph.json")
    )
    monkeypatch.setattr(kcl, "execute", real)
    monkeypatch.setattr(zoo_tools, "_execute_resolved_kcl_project", real)
    result = await execute(
        execution_route,
        {"kcl_path": str(tmp_path if input_is_directory else entry)},
    )
    assert result.ok, result
    real.assert_awaited_once()


@pytest.mark.asyncio
async def test_capture_materializes_linked_modules(
    monkeypatch, tmp_path, execution_route
):
    shared = tmp_path / "shared"
    shared.mkdir()
    library = shared / "library.kcl"
    library.write_text("export x = 1\n")
    (shared / "main.kcl").write_text('export import x from "library.kcl"\n')
    project = tmp_path / "project"
    project.mkdir()
    try:
        (project / "shared").symlink_to(shared, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    (project / "main.kcl").write_text('import x from "shared/main.kcl"\ny = x + 1\n')
    original_mock = kcl.mock_execute
    captured = []

    async def preflight(path):
        captured.append(Path(path).parent)
        outcome = await original_mock(path)
        library.write_text("changed after preflight")
        return outcome

    async def run(*args):
        assert (captured[0] / "shared/library.kcl").read_text() == "export x = 1\n"
        assert not (captured[0] / "shared").is_symlink()
        if execution_route != "local":
            files = {file["path"]: bytes(file["contents"]) for file in args[2]}
            assert files["shared/library.kcl"] == b"export x = 1\n"
            return Path("graph.json")
        return Outcome()

    monkeypatch.setattr(kcl, "mock_execute", preflight)
    monkeypatch.setattr(kcl, "execute", run)
    monkeypatch.setattr(zoo_tools, "_execute_resolved_kcl_project", run)
    result = await execute(execution_route, {"kcl_path": str(project)})
    assert result.ok, result
    assert not captured[0].exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_failure", [False, True])
async def test_cancel_during_project_capture_waits_for_worker_cleanup(
    monkeypatch,
    tmp_path,
    worker_failure,
):
    (tmp_path / "main.kcl").write_text("x = 1")
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    destinations = []
    capture = zoo_tools._capture_execution_project

    def worker(path, destination):
        destinations.append(destination)
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(timeout=2)
        if worker_failure:
            raise OSError("snapshot read failed")
        return capture(path, destination)

    monkeypatch.setattr(zoo_tools, "_capture_execution_project", worker)
    mock = AsyncMock(side_effect=AssertionError("must not mock execute"))
    real = AsyncMock(side_effect=AssertionError("must not execute"))
    mock_bindings(monkeypatch, mock, real)
    task = asyncio.create_task(zoo_tools.zoo_execute_kcl(kcl_path=tmp_path))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        if not destinations:
            task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not destinations[0].exists()
    mock.assert_not_called()
    real.assert_not_called()
