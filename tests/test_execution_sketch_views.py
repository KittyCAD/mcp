"""Bundled sketch rendering uses native outcomes, not another execution."""

from dataclasses import dataclass, field
from pathlib import Path

import kcl
import pytest

from zoo_mcp import ZooMCPException, zoo_tools
from zoo_mcp.sketch_execution import current_sketch_execution, reuse_sketch_execution
from zoo_mcp.zoo_tools import KclSketchViewRequest

SKETCHES = """
@settings(defaultLengthUnit = mm, kclVersion = 2.0)

profile = sketch(on = YZ) {
  edge = line(start = [var 2mm, var 8mm], end = [var 5mm, var 7mm])
  coincident([edge.start, [2mm, 8mm]])
  coincident([edge.end, [5mm, 7mm]])
}
fn makeProfile() {
  profile = sketch(on = XY) {
    edge = line(start = [0mm, 0mm], end = [10mm, 0mm])
  }
  return profile
}
second = makeProfile()
"""


@dataclass
class NativeCalls:
    preflight: int = 0
    execution: int = 0
    outcomes: list[kcl.ExecOutcome] = field(default_factory=list)


@pytest.fixture
def native_calls(monkeypatch: pytest.MonkeyPatch) -> NativeCalls:
    """Run the actual interpreter/renderer with its offline engine backend."""
    calls = NativeCalls()
    mock_execute_code = kcl.mock_execute_code

    async def preflight(code: str) -> kcl.ExecOutcome:
        calls.preflight += 1
        return await mock_execute_code(code)

    async def open_session(
        kcl_code: str | None,
        kcl_path: Path | str | None,
        *,
        highlight_edges: bool | None = None,
        video_res_width: int | None = None,
        video_res_height: int | None = None,
    ) -> kcl.KclSession:
        calls.execution += 1
        if kcl_code is not None:
            session = await kcl.new_kcl_session_code(kcl_code, mock=True)
        else:
            session = await kcl.new_kcl_session(str(kcl_path), mock=True)
        calls.outcomes.append(session.outcome)
        return session

    monkeypatch.setattr(kcl, "mock_execute_code", preflight)
    monkeypatch.setattr(zoo_tools, "_open_kcl_session", open_session)
    return calls


@pytest.mark.asyncio
async def test_sketch_views_reuse_execution_and_select_instances(
    native_calls: NativeCalls,
) -> None:
    requests = tuple(KclSketchViewRequest("profile", index) for index in (0, 1))
    result = await zoo_tools.zoo_execute_kcl(kcl_code=SKETCHES, sketch_views=requests)

    assert result.ok
    assert (native_calls.preflight, native_calls.execution) == (1, 1)
    assert result.inspection.sketch_constraints_status == "succeeded"
    for view, request in zip(result.inspection.sketch_views, requests, strict=True):
        assert view.request == request
        assert view.error is None
        assert view.png == bytes(
            native_calls.outcomes[0].render_sketch_png(
                request.sketch_name, instance_index=request.instance_index
            )
        )
    assert (
        result.inspection.sketch_views[0].png != result.inspection.sketch_views[1].png
    )


@pytest.mark.asyncio
async def test_bad_sketch_selection_does_not_change_execution_success(
    native_calls: NativeCalls,
) -> None:
    result = await zoo_tools.zoo_execute_kcl(
        kcl_code=SKETCHES,
        sketch_views=(
            KclSketchViewRequest("profile"),
            KclSketchViewRequest("profile", 2),
            KclSketchViewRequest("absent"),
            KclSketchViewRequest("profile", 1),
        ),
    )

    assert result.ok
    assert (native_calls.preflight, native_calls.execution) == (1, 1)
    ambiguous, out_of_range, absent, valid = result.inspection.sketch_views
    for view, message in (
        (ambiguous, "found 2 sketches named"),
        (out_of_range, "out of range"),
        (absent, "no sketch named"),
    ):
        assert view.png is None
        assert message in (view.error or "")
    assert valid.png is not None


@pytest.mark.asyncio
async def test_preflight_failure_returns_completed_sketch_without_real_execution(
    native_calls: NativeCalls,
) -> None:
    result = await zoo_tools.zoo_execute_kcl(
        kcl_code=SKETCHES + "\nlate = missingValue\n",
        sketch_views=(KclSketchViewRequest("profile", 0),),
    )

    assert not result.ok
    assert "missingValue" in result.message
    assert result.mock_preflight.status == "failed"
    assert result.real_execution.status == "not_run"
    assert (native_calls.preflight, native_calls.execution) == (1, 0)
    assert result.inspection.sketch_views[0].png is not None


@pytest.mark.asyncio
async def test_no_requested_sketches_preserves_default(
    native_calls: NativeCalls,
) -> None:
    result = await zoo_tools.zoo_execute_kcl(kcl_code=SKETCHES)
    assert result.ok
    assert result.inspection.sketch_views == []
    assert (native_calls.preflight, native_calls.execution) == (1, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selector, message",
    [
        (KclSketchViewRequest("profile", -1), "instance_index must be non-negative"),
        (KclSketchViewRequest(" "), "sketch_name must not be empty"),
    ],
)
async def test_invalid_selector_rejected_before_execution(
    native_calls: NativeCalls, selector: KclSketchViewRequest, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        await zoo_tools.zoo_execute_kcl(kcl_code=SKETCHES, sketch_views=(selector,))
    assert (native_calls.preflight, native_calls.execution) == (0, 0)


@pytest.mark.asyncio
async def test_remote_sketch_request_rejected_before_execution(
    native_calls: NativeCalls,
) -> None:
    with pytest.raises(ValueError, match="only available for local execution"):
        await zoo_tools.zoo_execute_kcl(
            kcl_code=SKETCHES,
            session_id="remote-session",
            sketch_views=(KclSketchViewRequest("profile", 0),),
        )
    assert (native_calls.preflight, native_calls.execution) == (0, 0)


@pytest.mark.asyncio
async def test_later_views_reuse_closed_execution_and_keep_selection_errors(
    native_calls: NativeCalls,
) -> None:
    requests = tuple(KclSketchViewRequest("profile", index) for index in (0, 1))
    with reuse_sketch_execution() as first:
        result = await zoo_tools.zoo_execute_kcl(
            kcl_code=SKETCHES, sketch_views=requests
        )
    assert result.ok
    assert first.execution is not None
    assert current_sketch_execution() is None

    for index, view in enumerate(result.inspection.sketch_views):
        with reuse_sketch_execution(first.execution) as later:
            png = await zoo_tools.zoo_visualize_sketch(
                "profile", kcl_code=SKETCHES, instance_index=index
            )
        assert later.reused
        assert later.execution is first.execution
        assert png == view.png

    for name, index in (("profile", None), ("profile", 2), ("absent", None)):
        with (
            reuse_sketch_execution(first.execution) as later,
            pytest.raises(ZooMCPException),
        ):
            await zoo_tools.zoo_visualize_sketch(
                name, kcl_code=SKETCHES, instance_index=index
            )
        assert later.reused
    assert (native_calls.preflight, native_calls.execution) == (1, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("downstream_error", [False, True])
async def test_standalone_keeps_native_result_for_later_views(
    native_calls: NativeCalls, downstream_error: bool
) -> None:
    code = SKETCHES + ("\nlate = missingValue\n" if downstream_error else "")
    with reuse_sketch_execution() as first:
        png = await zoo_tools.zoo_visualize_sketch(
            "profile", kcl_code=code, instance_index=0
        )
    assert first.execution is not None
    assert not first.reused
    with reuse_sketch_execution(first.execution) as later:
        assert png == await zoo_tools.zoo_visualize_sketch(
            "profile", kcl_code=code, instance_index=0
        )
    assert later.reused
    assert native_calls.execution == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["source", "import", "settings", "entrypoint"])
async def test_project_changes_require_fresh_execution(
    native_calls: NativeCalls, tmp_path: Path, change: str
) -> None:
    code = SKETCHES.replace(
        "profile = sketch(on = YZ)",
        'import offset from "dimensions.kcl"\nprofile = sketch(on = YZ)',
        1,
    ).replace("2mm, 8mm", "2mm, offset")
    entry = tmp_path / "main.kcl"
    entry.write_text(code)
    dependency = tmp_path / "dimensions.kcl"
    dependency.write_text("export offset = 8mm\n")
    with reuse_sketch_execution() as first:
        result = await zoo_tools.zoo_execute_kcl(kcl_path=tmp_path)
    assert result.ok, result.message
    assert first.execution is not None

    (tmp_path / "notes.md").write_text("This is not an execution input.\n")
    with reuse_sketch_execution(first.execution) as unchanged:
        await zoo_tools.zoo_visualize_sketch(
            "profile", kcl_path=entry, instance_index=0
        )
    assert unchanged.reused
    assert native_calls.execution == 1

    if change == "source":
        entry.write_text(code.replace("10mm", "11mm"))
    elif change == "import":
        dependency.write_text("export offset = 9mm\n")
    elif change == "settings":
        (tmp_path / "project.toml").write_text("[settings]\n")
    else:
        entry = tmp_path / "other.kcl"
        entry.write_text(code)
    with reuse_sketch_execution(first.execution) as changed:
        png = await zoo_tools.zoo_visualize_sketch(
            "profile", kcl_path=entry, instance_index=0
        )
    assert not changed.reused
    assert changed.execution is not None
    assert changed.execution.fingerprint != first.execution.fingerprint
    assert png == bytes(
        native_calls.outcomes[-1].render_sketch_png("profile", instance_index=0)
    )
    assert native_calls.execution == 2


@pytest.mark.asyncio
async def test_explicit_execution_never_uses_saved_validation(
    native_calls: NativeCalls,
) -> None:
    with reuse_sketch_execution() as first:
        await zoo_tools.zoo_execute_kcl(kcl_code=SKETCHES)
    with reuse_sketch_execution(first.execution) as second:
        result = await zoo_tools.zoo_execute_kcl(kcl_code=SKETCHES)
    assert result.ok
    assert not second.reused
    assert (native_calls.preflight, native_calls.execution) == (2, 2)

    code = SKETCHES + "\nlate = missingValue\n"
    with reuse_sketch_execution(second.execution) as failed:
        result = await zoo_tools.zoo_execute_kcl(kcl_code=code)
    assert not result.ok
    assert result.real_execution.status == "not_run"
    assert failed.execution is not None
    assert failed.execution.stage == "mock_preflight"
    with reuse_sketch_execution(failed.execution) as later:
        png = await zoo_tools.zoo_visualize_sketch(
            "profile", kcl_code=code, instance_index=0
        )
    assert png.startswith(b"\x89PNG")
    assert later.reused
    assert (native_calls.preflight, native_calls.execution) == (3, 2)


@pytest.mark.asyncio
async def test_independent_callers_do_not_share_saved_results(
    native_calls: NativeCalls,
) -> None:
    with reuse_sketch_execution() as first:
        await zoo_tools.zoo_execute_kcl(kcl_code=SKETCHES)
    with reuse_sketch_execution() as unrelated:
        await zoo_tools.zoo_visualize_sketch(
            "profile", kcl_code=SKETCHES, instance_index=0
        )
    assert not unrelated.reused
    assert unrelated.execution is not first.execution
    assert native_calls.execution == 2


@pytest.mark.asyncio
async def test_fresh_capture_keeps_original_diagnostic_paths(
    native_calls: NativeCalls, tmp_path: Path
) -> None:
    entry = tmp_path / "main.kcl"
    entry.write_text("profile = missingValue\n")
    with reuse_sketch_execution(), pytest.raises(ZooMCPException) as failure:
        await zoo_tools.zoo_visualize_sketch("profile", kcl_path=entry)
    assert str(entry) in str(failure.value)
    assert "zoo-mcp-preflight-" not in str(failure.value)


@pytest.mark.live
@pytest.mark.xdist_group(name="engine")
@pytest.mark.asyncio
async def test_engine_failure_returns_png_and_keeps_partial_report() -> None:
    code = SKETCHES + "\nbadRegion = region(segments = [profile.edge])\n"
    with (
        zoo_tools.capture_execution_stage_events() as events,
        reuse_sketch_execution() as first,
    ):
        result = await zoo_tools.zoo_execute_kcl(
            kcl_code=code,
            sketch_views=(KclSketchViewRequest("profile", 0),),
        )

    assert not result.ok
    assert result.mock_preflight.status == "succeeded"
    assert result.real_execution.status == "failed"
    assert "Unable create a region" in result.message
    assert result.inspection.sketch_constraints_status == "partial"
    assert result.inspection.sketch_views[0].png is not None
    assert [(event.stage, event.attempts) for event in events] == [
        ("mock_preflight", 1),
        ("real_execution", 1),
    ]
    assert first.execution is not None
    assert first.execution.stage == "real_execution"
    with (
        reuse_sketch_execution(first.execution) as later,
        zoo_tools.capture_execution_retry_events() as retries,
    ):
        png = await zoo_tools.zoo_visualize_sketch(
            "profile", kcl_code=code, instance_index=0
        )
    assert later.reused
    assert png == result.inspection.sketch_views[0].png
    assert retries == []
