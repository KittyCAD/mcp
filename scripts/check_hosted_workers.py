"""Exercise real Linux confinement and the production worker without API secrets.

Run with the container's Python, mounting this script read-only outside runtime
read roots. Failure is fatal: this check never skips unsupported confinement.
"""

import asyncio
import base64
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from zoo_mcp.hosted.pool import Owner, WorkerPool
from zoo_mcp.hosted.sandbox import confine


def check_filesystem(workspace: Path, outside_file: Path) -> None:
    confine(workspace)
    local = workspace / "allowed.kcl"
    local.write_text("x = 1\n")
    assert local.read_text() == "x = 1\n"
    for path in (
        outside_file,
        workspace / "escape",
        Path("/proc/self/environ"),
        Path("/etc/passwd"),
    ):
        try:
            path.read_bytes()
        except PermissionError:
            pass
        else:
            raise AssertionError(f"Read escaped confinement: {path}")
    for path in (outside_file, Path("/tmp/escaped-hosted-worker")):
        try:
            path.write_text("must fail")
        except PermissionError:
            pass
        else:
            raise AssertionError(f"Write escaped confinement: {path}")
    # Exercise a real native KCL import, which must obey the same kernel policy.
    from zoo_mcp.zoo_tools import zoo_mock_execute_kcl

    code = f'import protectedValue from "{outside_file}"\nx = protectedValue\n'
    entry = workspace / "main.kcl"
    entry.write_text(code)
    success, message = asyncio.run(zoo_mock_execute_kcl(kcl_path=entry))
    assert not success, "KCL import escaped confinement"
    assert "synthetic-fixture-value" not in message


async def check_workers(root: Path) -> None:
    from PIL import Image

    from zoo_mcp.server import mcp

    pool = WorkerPool(root, "https://api.example")
    a = Owner("https://issuer.example", "user", "org", "api_key", "a")
    b = Owner("https://issuer.example", "user", "org", "api_key", "b")

    async def credential():
        return "synthetic-downstream-credential"

    async def call(owner, name, arguments):
        return await pool.call(owner, credential, name, arguments)

    try:
        await pool.probe()
        for name, arguments in (
            ("format_kcl", {"kcl_code": "x=1"}),
            ("get_modeling_sessions", {}),
        ):
            expected = await mcp.call_tool(name, arguments)
            actual = await call(a, name, arguments)
            assert actual == expected, (actual, expected)
        await asyncio.gather(*(call(a, "get_modeling_sessions", {}) for _ in range(10)))
        assert len(pool.entries) == 1
        process = pool.entries[a].worker.process
        assert process is not None
        first_pid = process.pid
        await call(a, "format_kcl", {"kcl_code": "x=2"})
        assert pool.entries[a].worker.process is process

        buf = io.BytesIO()
        Image.new("RGB", (2, 2)).save(buf, format="PNG")
        result = await call(
            a,
            "save_image",
            {
                "image": {
                    "type": "image",
                    "data": base64.b64encode(buf.getvalue()).decode(),
                    "mimeType": "image/png",
                },
                "output_path": "owned.png",
            },
        )
        assert not result.is_error, result
        workspace = pool.entries[a].worker.workspace
        assert (workspace / "owned.png").exists()
        kcl_path = workspace / "owned.kcl"
        kcl_path.write_text("secret = 123456789\n")
        same_owner = await call(a, "format_kcl", {"kcl_path": str(kcl_path)})
        assert "Successfully formatted" in same_owner.model_dump_json()
        other_owner = await call(b, "format_kcl", {"kcl_path": str(kcl_path)})
        assert "123456789" not in other_owner.model_dump_json()
        assert "Permission denied" in other_owner.model_dump_json()

        # A kernel-killed process is evicted, and a reconnect gets a fresh worker.
        process.kill()
        await process.wait()
        try:
            await call(a, "get_modeling_sessions", {})
        except (OSError, asyncio.IncompleteReadError):
            pass
        else:
            raise AssertionError("Dead worker accepted a call")
        assert a not in pool.entries and b in pool.entries
        assert not workspace.exists()
        await call(a, "get_modeling_sessions", {})
        replacement = pool.entries[a].worker.process
        assert replacement is not None and replacement.pid != first_pid
    finally:
        await pool.close()
    assert not list(root.iterdir())


def main() -> None:
    if len(sys.argv) > 1:
        check_filesystem(Path(sys.argv[1]), Path(sys.argv[2]))
        return
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        workspace = root / "workspace"
        workspace.mkdir()
        outside_file = root / "protected.kcl"
        outside_file.write_text('export protectedValue = "synthetic-fixture-value"\n')
        (workspace / "escape").symlink_to(outside_file)
        subprocess.run(
            [sys.executable, __file__, str(workspace), str(outside_file)],
            check=True,
            cwd=workspace,
        )
        os.environ["PARENT_ONLY_MARKER"] = "synthetic-parent-only-value"
        asyncio.run(check_workers(root / "workers"))
    print("Linux confinement and persistent worker checks passed")


if __name__ == "__main__":
    main()
