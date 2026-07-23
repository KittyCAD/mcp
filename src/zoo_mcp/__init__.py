"""Shared definitions for the Zoo Model Context Protocol servers.

The authenticated CAD runtime is intentionally loaded lazily.
This keeps the documentation-only server independent from the KittyCAD client and
from Zoo credentials while preserving the existing public package attributes.
"""

import logging
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import ssl

    from kittycad import KittyCAD

FORMAT = "%(asctime)s | %(levelname)-7s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s"

logging.basicConfig(
    level=logging.INFO, format=FORMAT, handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("zoo_mcp")


try:
    __version__ = version("zoo_mcp")
except PackageNotFoundError:
    __version__ = "0.0.0"


class ZooMCPException(Exception):
    """Custom exception for Zoo MCP Server."""


def __getattr__(name: str) -> Any:
    """Lazily expose the authenticated server's legacy runtime attributes."""
    if name not in {"ctx", "kittycad_client"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from zoo_mcp import authenticated_runtime

    value = getattr(authenticated_runtime, name)
    globals()[name] = value
    return value


if TYPE_CHECKING:
    ctx: ssl.SSLContext
    kittycad_client: KittyCAD

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
