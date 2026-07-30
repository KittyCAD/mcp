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


class ZooKclEngineError(ZooMCPException):
    """A KCL engine failure, with the details callers need to react to it.

    The kcl bindings raise ``kcl.KclError``, which stringifies as a PyO3 tuple
    (``("engine: KclErrorDetails { ... }", False)``) and exposes its retryable
    flag only through an ``is_retryable()`` method. Every retried kcl operation
    re-raises its terminal failure as this instead, so callers read fields rather
    than parsing that repr: whether a retry is worthwhile, which operation
    failed, and the engine's own identifiers when it supplied them. Tools that
    report failures in-band (``zoo_execute_kcl`` returning ``(False, message)``)
    get the identifiers too, because ``str()`` appends them.

    Attributes:
        message: The engine's message, unwrapped from the PyO3 tuple repr.
        operation: The kcl entry point that failed, e.g. ``execute_and_measure``.
        retryable: Whether the bindings flagged the failure as retryable. This is
            the engine's own opinion; it is ``False`` for generic engine errors
            that nonetheless come back clean on a later attempt.
        attempts: How many attempts were made before giving up.
        modeling_command_id: The engine's modeling command ID, when quoted.
        api_call_id: The engine's API call ID, when quoted.
    """

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        retryable: bool = False,
        attempts: int = 1,
        modeling_command_id: str | None = None,
        api_call_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.operation = operation
        self.retryable = retryable
        self.attempts = attempts
        self.modeling_command_id = modeling_command_id
        self.api_call_id = api_call_id

    def is_retryable(self) -> bool:
        """Return the engine's retryable flag, matching ``kcl.KclError``."""
        return self.retryable

    def __str__(self) -> str:
        """Render the engine's message, ending with any identifiers it supplied.

        The engine buries its IDs mid-prose when it quotes them at all, so repeat
        them in a fixed trailing position. Callers that can only pass a string
        along -- a tool returning ``(False, message)``, a log line -- then report
        them too, instead of only the callers that read the fields.
        """
        identifiers = " ".join(
            f"{name}={value}"
            for name, value in (
                ("modeling_command_id", self.modeling_command_id),
                ("api_call_id", self.api_call_id),
            )
            if value is not None
        )
        if not identifiers:
            return self.message
        return f"{self.message} [{identifiers}]"


ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
kittycad_client = KittyCAD(verify_ssl=ctx)
# set the websocket receive timeout to 5 minutes
kittycad_client.websocket_recv_timeout = 300

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
