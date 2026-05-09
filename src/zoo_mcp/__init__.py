"""Zoo Model Context Protocol (MCP) Server.

A lightweight service that enables AI assistants to execute Zoo commands through the Model Context Protocol (MCP).
"""

import logging
import ssl
import sys
from importlib.metadata import PackageNotFoundError, version

import truststore
from kittycad import KittyCAD

FORMAT = "%(asctime)s | %(levelname)-7s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s"

logging.basicConfig(
    level=logging.INFO, format=FORMAT, handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("zoo_mcp")


try:
    __version__ = version("zoo_mcp")
except PackageNotFoundError:
    # package is not installed
    logger.error("zoo-mcp package is not installed.")


class ZooMCPException(Exception):
    """Custom exception for Zoo MCP Server."""


ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
kittycad_client = KittyCAD(verify_ssl=ctx)
# set the websocket receive timeout to 5 minutes
kittycad_client.websocket_recv_timeout = 300

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
