"""Runtime objects used only by the authenticated Zoo MCP server."""

import ssl

import truststore
from kittycad import KittyCAD

ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
kittycad_client = KittyCAD(verify_ssl=ctx)
kittycad_client.websocket_recv_timeout = 300
