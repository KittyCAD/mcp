"""Zoo Model Context Protocol (MCP) Server.

A lightweight service that enables AI assistants to execute Zoo commands through the Model Context Protocol (MCP).
"""

import logging
import ssl
import sys
from importlib.metadata import PackageNotFoundError, version

import truststore

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


class ZooMCPTimeoutError(ZooMCPException):
    """A modeling command exceeded its wall-clock budget.

    Separate from ZooMCPException so callers can tell "the engine never
    answered in time", which is worth retrying, apart from "the engine
    rejected this command", which is not.
    """


ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
