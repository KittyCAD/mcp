# Hosting Zoo MCP

Zoo MCP uses the official MCP Python SDK. Its local Streamable HTTP entry point
serves `/mcp` using the process's Zoo API credential and modeling session. Keep
that process private to one owner. A transport session is not a user or credential
boundary.

An embedding application can import `mcp` from `zoo_mcp.server` and use the SDK's
`streamable_http_app()` to obtain an ASGI application. Configure `AuthSettings`,
a `TokenVerifier`, the public resource URL and issuer, required scopes, and
explicit host/origin allowlists through the SDK. It supplies the MCP transport,
protected-resource metadata, authentication challenges, and serialization.
See the SDK's [ASGI guide](https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/run/asgi.md)
and [authorization guide](https://modelcontextprotocol.io/docs/2026-07-28/tutorials/security/authorization#python).

A service hosting multiple owners must keep their credentials, modeling state,
and filesystem access isolated. The current KCL bindings read process credentials,
and Zoo MCP has one active modeling session per process. Run tool calls in an
isolated process owned by a verified identity, with a private workspace and
serialized credential updates. Keep application ownership independent of MCP
transport sessions so reconnects and OAuth refresh can recover the same worker.

A hosting wrapper can forward `mcp.list_tools()` and the owning process's
`mcp.call_tool()` results directly, preserving descriptions, schemas, structured
content, images, and errors. Deployment-specific authentication, worker
supervision, container builds, and infrastructure configuration belong to that
wrapper. Existing local commands and tool implementations remain unchanged.
