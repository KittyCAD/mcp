from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, patch

import pytest

from zoo_mcp import server, zoo_tools

SnapshotFunction = Callable[..., Awaitable[bytes]]
ServerSnapshotFunction = Callable[..., Awaitable[object]]

KCL_SNAPSHOT_FUNCTIONS: tuple[SnapshotFunction, ...] = (
    zoo_tools.zoo_snapshot_of_kcl,
    zoo_tools.zoo_multiview_snapshot_of_kcl,
    zoo_tools.zoo_multi_isometric_snapshot_of_kcl,
)

SERVER_SNAPSHOT_FUNCTIONS: tuple[tuple[ServerSnapshotFunction, str], ...] = (
    (server.snapshot_of_kcl, "zoo_snapshot_of_kcl"),
    (server.multiview_snapshot_of_kcl, "zoo_multiview_snapshot_of_kcl"),
    (
        server.multi_isometric_snapshot_of_kcl,
        "zoo_multi_isometric_snapshot_of_kcl",
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "snapshot_function",
    KCL_SNAPSHOT_FUNCTIONS,
    ids=lambda function: function.__name__,
)
@pytest.mark.parametrize("use_code", [True, False], ids=["code", "path"])
@pytest.mark.parametrize("highlight_edges", [False, True], ids=["hidden", "shown"])
async def test_kcl_snapshot_forwards_highlight_edges(
    snapshot_function: SnapshotFunction,
    use_code: bool,
    highlight_edges: bool,
    cube_kcl: str,
):
    code_snapshot = AsyncMock(return_value=[b"view"] * 4)
    path_snapshot = AsyncMock(return_value=[b"view"] * 4)

    with (
        patch.object(
            zoo_tools.kcl,
            "execute_code_and_snapshot_views",
            new=code_snapshot,
        ),
        patch.object(
            zoo_tools.kcl,
            "execute_and_snapshot_views",
            new=path_snapshot,
        ),
        patch.object(zoo_tools, "create_image_collage", return_value=b"collage"),
        patch.object(zoo_tools, "resize_image", return_value=b"snapshot"),
    ):
        result = await snapshot_function(
            kcl_code="cube()" if use_code else None,
            kcl_path=None if use_code else cube_kcl,
            highlight_edges=highlight_edges,
        )

    assert result == b"snapshot"
    selected_snapshot = code_snapshot if use_code else path_snapshot
    unused_snapshot = path_snapshot if use_code else code_snapshot
    selected_snapshot.assert_awaited_once()
    unused_snapshot.assert_not_awaited()
    await_args = selected_snapshot.await_args
    assert await_args is not None
    assert await_args.kwargs["highlight_edges"] is highlight_edges


@pytest.mark.asyncio
async def test_kcl_snapshot_omits_highlight_edges_by_default(cube_kcl: str):
    path_snapshot = AsyncMock(return_value=[b"view"])

    with (
        patch.object(
            zoo_tools.kcl,
            "execute_and_snapshot_views",
            new=path_snapshot,
        ),
        patch.object(zoo_tools, "resize_image", return_value=b"snapshot"),
    ):
        await zoo_tools.zoo_snapshot_of_kcl(kcl_code=None, kcl_path=cube_kcl)

    path_snapshot.assert_awaited_once()
    await_args = path_snapshot.await_args
    assert await_args is not None
    assert "highlight_edges" not in await_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("snapshot_function", "implementation_name"),
    SERVER_SNAPSHOT_FUNCTIONS,
    ids=lambda value: value.__name__ if callable(value) else value,
)
async def test_server_snapshot_tool_disables_highlight_edges_by_default(
    snapshot_function: ServerSnapshotFunction,
    implementation_name: str,
):
    implementation = AsyncMock(return_value=b"snapshot")

    with (
        patch.object(server, implementation_name, new=implementation),
        patch.object(server, "encode_image", return_value="encoded"),
    ):
        result = await snapshot_function(kcl_code="cube()")

    assert result == "encoded"
    implementation.assert_awaited_once()
    await_args = implementation.await_args
    assert await_args is not None
    assert await_args.kwargs["highlight_edges"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("snapshot_function", "implementation_name"),
    SERVER_SNAPSHOT_FUNCTIONS,
    ids=lambda value: value.__name__ if callable(value) else value,
)
async def test_server_snapshot_tool_allows_highlight_edges_override(
    snapshot_function: ServerSnapshotFunction,
    implementation_name: str,
):
    implementation = AsyncMock(return_value=b"snapshot")

    with (
        patch.object(server, implementation_name, new=implementation),
        patch.object(server, "encode_image", return_value="encoded"),
    ):
        result = await snapshot_function(kcl_code="cube()", highlight_edges=True)

    assert result == "encoded"
    implementation.assert_awaited_once()
    await_args = implementation.await_args
    assert await_args is not None
    assert await_args.kwargs["highlight_edges"] is True
