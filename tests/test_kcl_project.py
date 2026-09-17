import json
from pathlib import Path
from typing import cast

import kcl
import pytest

from zoo_mcp import ZooMCPException, zoo_tools
from zoo_mcp.utils.kcl_project import load_kcl_project


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "statement",
    [
        'import x as first, y as second from "library.kcl"',
        "import x as first,\n  y as second from 'library.kcl'",
        'import * from "library.kcl"',
        'import "library.kcl" as library',
    ],
)
async def test_capture_follows_imports_but_ignores_comments_and_strings(
    tmp_path, statement
):
    source = tmp_path / "project"
    source.mkdir()
    code = (
        '// import "unrelated.step"\n'
        '/* import z from "missing.kcl" */\n'
        "example = \"import 'also-missing.kcl'\"\n" + statement + "\n"
    )
    (source / "main.kcl").write_text(code)
    (source / "library.kcl").write_text(
        'export import x from "nested/main.kcl"\nexport y = 2\n'
    )
    (source / "nested").mkdir()
    (source / "nested/main.kcl").write_text("export x = 1\n")
    (source / "unrelated.step").write_bytes(b"not referenced")
    destination = tmp_path / "captured"
    resolved = zoo_tools._capture_execution_project(source, destination)

    assert {file["path"] for file in resolved.files} == {
        "main.kcl",
        "library.kcl",
        "nested/main.kcl",
    }
    assert (destination / "main.kcl").read_text() == code
    assert resolved.path is not None
    outcome = await kcl.mock_execute(resolved.path)
    assert not any(issue.is_err() for issue in outcome.issues())


@pytest.mark.asyncio
@pytest.mark.parametrize("absolute_buffer", [False, True])
async def test_capture_includes_gltf_buffers_without_unrelated_assets(
    tmp_path, absolute_buffer
):
    source = tmp_path / "project"
    assets = source / "assets"
    assets.mkdir(parents=True)
    (source / "main.kcl").write_text('import "assets/model.gltf" as model\n')
    buffer = assets / "mesh.bin"
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [
            {
                "uri": buffer.as_posix() if absolute_buffer else "mesh.bin",
                "byteLength": 4,
            },
            {"uri": "data:application/octet-stream;base64,AAAAAA==", "byteLength": 4},
        ],
    }
    (assets / "model.gltf").write_text(json.dumps(gltf))
    buffer.write_bytes(b"\x00\x01\xfe\xff")
    (assets / "unrelated.bin").write_bytes(b"not referenced")
    resolved = zoo_tools._capture_execution_project(source, tmp_path / "captured")
    buffer.unlink()
    files = {
        file["path"]: bytes(cast(list[int], file["contents"]))
        for file in resolved.files
    }

    assert set(files) == {"main.kcl", "assets/model.gltf", "assets/mesh.bin"}
    assert files["assets/mesh.bin"] == b"\x00\x01\xfe\xff"
    assert resolved.path is not None
    outcome = await kcl.mock_execute(resolved.path)
    assert not any(issue.is_err() for issue in outcome.issues())


@pytest.mark.asyncio
@pytest.mark.parametrize("absolute_import", [False, True])
async def test_capture_preserves_cad_imports_outside_entrypoint_directory(
    tmp_path, cube_stl, absolute_import
):
    source = tmp_path / "project"
    source.mkdir()
    asset = tmp_path / "cube.stl"
    asset.write_bytes(Path(cube_stl).read_bytes())
    imported = asset.as_posix() if absolute_import else "../cube.stl"
    (source / "main.kcl").write_text(f'import "{imported}" as cube\n')
    resolved = zoo_tools._capture_execution_project(source, tmp_path / "captured")
    asset.unlink()

    assert resolved.entrypoint == "project/main.kcl"
    assert {file["path"] for file in resolved.files} == {"project/main.kcl", "cube.stl"}
    assert resolved.path is not None
    outcome = await kcl.mock_execute(resolved.path)
    assert not any(issue.is_err() for issue in outcome.issues())


@pytest.mark.asyncio
async def test_linked_entrypoint_uses_imports_from_its_logical_directory(tmp_path):
    original = tmp_path / "original.kcl"
    original.write_text('import x from "library.kcl"\ny = x\n')
    project = tmp_path / "project"
    project.mkdir()
    (project / "library.kcl").write_text("export x = 1\n")
    entry = project / "main.kcl"
    try:
        entry.symlink_to(original)
    except OSError:
        pytest.skip("file symlinks are unavailable")
    resolved = zoo_tools._capture_execution_project(entry, tmp_path / "captured")

    assert resolved.entrypoint == "main.kcl"
    assert {file["path"] for file in resolved.files} == {"main.kcl", "library.kcl"}
    assert resolved.path is not None
    outcome = await kcl.mock_execute(resolved.path)
    assert not any(issue.is_err() for issue in outcome.issues())


@pytest.mark.parametrize("linked", [False, True])
def test_import_cycles_do_not_recurse_indefinitely(tmp_path, linked):
    if linked:
        try:
            (tmp_path / "again").symlink_to(tmp_path, target_is_directory=True)
        except OSError:
            pytest.skip("directory symlinks are unavailable")
        imported = "again/main.kcl"
    else:
        imported = "main.kcl"
    (tmp_path / "main.kcl").write_text(f'import "{imported}" as recursive\n')
    with pytest.raises(ZooMCPException, match="Circular KCL import"):
        load_kcl_project(tmp_path)


@pytest.mark.asyncio
async def test_capture_leaves_missing_import_diagnostics_to_kcl(tmp_path):
    source = tmp_path / "project"
    source.mkdir()
    (source / "main.kcl").write_text('import x from "missing.kcl"\n')
    resolved = zoo_tools._capture_execution_project(source, tmp_path / "captured")
    assert resolved.path is not None
    with pytest.raises(kcl.KclError, match="missing.kcl"):
        await kcl.mock_execute(resolved.path)
