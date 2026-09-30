# Zoo Model Context Protocol (MCP) Server

An [MCP server](https://modelcontextprotocol.io/docs/getting-started/intro) housing various Zoo built utilities

<!-- mcp-name: io.github.KittyCAD/zoo-mcp -->

## Prerequisites

1. An API key for Zoo, get one [here](https://zoo.dev/account)
2. An environment variable `ZOO_API_TOKEN` set to your API key
    ```bash
    export ZOO_API_TOKEN="your_api_key_here"
    ```

## Installation

1. [Ensure uv has been installed](https://docs.astral.sh/uv/getting-started/installation/)

2. [Create a uv environment](https://docs.astral.sh/uv/pip/environments/)
    ```bash
    uv venv
    ```

3. [Activate your uv environment (Optional)](https://docs.astral.sh/uv/pip/environments/#using-a-virtual-environment)

4. Install the package from GitHub
    ```bash
    uv pip install git+ssh://git@github.com/KittyCAD/mcp.git
    ```

## Running the Server

The server can be started by using [uvx](https://docs.astral.sh/uv/guides/tools/#running-tools)
```bash
uvx zoo-mcp
```

The server can be started locally by using uv and the zoo_mcp module
```bash
uv run -m zoo_mcp
```

The server can also be run with the [mcp package](https://github.com/modelcontextprotocol/python-sdk)
```bash
uv run mcp run src/zoo_mcp/server.py
```

### Streamable HTTP

The server uses stdio by default. To serve the same tools over Streamable HTTP:

```bash
uvx zoo-mcp --transport streamable-http --host 127.0.0.1 --port 8000
# From a local checkout:
uv run -m zoo_mcp --transport streamable-http
```

Connect an MCP client to `http://127.0.0.1:8000/mcp`. The SDK manages HTTP
sessions and streaming responses. `--host` and `--port` configure the HTTP
listener; their defaults are `127.0.0.1` and `8000`.

### Prebuilt binaries

Each [GitHub release](https://github.com/KittyCAD/mcp/releases) also attaches standalone executables (built with PyInstaller) for Linux (`x86_64`, `arm64`), macOS (`arm64`, `x86_64`), and Windows (`x86_64`) — no Python toolchain required. Download the binary for your platform, set `ZOO_API_TOKEN`, and run it directly, e.g.:
```bash
ZOO_API_TOKEN="your_api_key_here" ./zoo-mcp-linux-x86_64
```
> The binaries are not code-signed, so macOS Gatekeeper and Windows SmartScreen may warn on first run.

## Capturing backend API call IDs in Python

Python callers can collect tracing events without changing a tool's return value:

```python
from zoo_mcp.zoo_tools import capture_api_call_events, zoo_execute_kcl

with capture_api_call_events() as events:
    result = await zoo_execute_kcl(kcl_path="/path/to/project/main.kcl")

api_call_ids = list(
    dict.fromkeys(
        event.api_call_id for event in events if event.api_call_id is not None
    )
)
```

The list remains available if a call raises or is canceled. Nested capture contexts
each receive an event once. Concurrent tool invocations have separate invocation IDs;
child tasks inherit their parent's capture contexts, so await them before consuming
the completed event list.

For Zookeeper's `execute_project` integration, wrap the existing
`zoo_execute_kcl(...)` call in this context and attach the collected backend IDs
to the execution trace. Capture also works when the call requests snapshots or
physical properties. The IDs remain available after the execution session closes;
they identify the engine session for log correlation, not a reusable zoo-mcp
`session_id`. Execution result objects do not include these tracing fields.

`ApiCallEvent` contains `operation`, `invocation_id`, `api_call_id`, `source`,
`attempt`, and `outcome`, with optional `websocket_upgrade_request_id`,
`session_id`, `command_id`, `async_operation_id`, and HTTP `status_code`.
Missing identifiers are `None`. The backend `api_call_id` and HTTP WebSocket
upgrade request ID are distinct; neither is replaced by a local session or
command ID. Invocation summaries preserve both connection identifiers and their
retry attempt.

Every real local KCL execution opens a `KclSession`. Capture reads its
`api_call_id` and `websocket_upgrade_request_id` properties immediately after
creation, inside the owning retry attempt. Measurements, exports, snapshots,
constraints, and sketch rendering reuse that execution, and the session closes
after its requested work, including on errors or cancellation. Mock preflight
produces no backend IDs and runs once before real-execution retries.

An `observed` event records information already received; a later operation can
still fail. Invocation completion describes the whole Python call. Existing
execution retry events expose `api_call_ids` for that attempt, containing only
backend API call IDs. A failure before the KCL binding returns a session can have
no IDs, even if the backend received the request. Capturing IDs on these failures
is intentionally deferred; IDs are never parsed from error messages.

Events are observations, not a count of backend requests. A persistent modeling
session reuses its backend and upgrade request IDs across many command IDs. Its
backend ID becomes available from session metadata; the handshake ID remains in
`websocket_upgrade_request_id`. A file operation's `id` is also retained as
`async_operation_id`; each polling HTTP request has its own `api_call_id`.
Repeated observations of an ID are expected.

The same events are logged at INFO even without a capture context. Log records
include searchable identifiers and a structured `api_call_event` attribute;
tracing does not include credentials, source code, request bodies, or query text.
MCP tool response schemas are unchanged.

This integration requires `zoo-kcl>=0.3.188`, which includes the session
properties from [modeling-app PR #14156](https://github.com/KittyCAD/modeling-app/pull/14156).
Missing session properties are errors; a property whose value is `None` remains
valid.

## Integrations

The server can be used as is by [running the server](#running-the-server) or importing directly into your python code.
```python
from zoo_mcp.server import mcp

mcp.run()
```

Individual tools can be used in your own python code as well. At Zoo we use
zoo-mcp like this with ZooKeeper to save on resources. Instead of spinning up
one MCP server per agent, each agent in a sense "embeds" the server in their own
runtime. It has the additional benefit of preventing shared state.

```python
from mcp.server.mcpserver import MCPServer
from zoo_mcp.zoo_tools import ResultZooExecuteKcl, zoo_execute_kcl

mcp = MCPServer(name="My Example Server")


@mcp.tool()
async def my_execute_kcl(kcl_code: str) -> ResultZooExecuteKcl:
    """
    Example tool that uses the zoo_execute_kcl function from zoo_mcp.zoo_tools
    """
    return await zoo_execute_kcl(kcl_code=kcl_code)
```

The server can be integrated with [Claude desktop](https://claude.ai/download) using the following command
```bash 
uv run mcp install src/zoo_mcp/server.py
```

The server can also be integrated with [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) using the following command
```bash
claude mcp add --scope project "Zoo-MCP" uv -- --directory "$PWD"/src/zoo_mcp run server.py
```

The server can also be tested using the [MCP Inspector](https://modelcontextprotocol.io/legacy/tools/inspector#python)
```bash
uv run mcp dev src/zoo_mcp/server.py
```

For running with [codex-cli](https://github.com/openai/codex)
```bash
codex \
  -c 'mcp_servers.zoo.command="uvx"' \
  -c 'mcp_servers.zoo.args=["zoo-mcp"]' \
  -c mcp_servers.zoo.env.ZOO_API_TOKEN="$ZOO_API_TOKEN"
```

You can also use the helper script included in this repo:
```bash
./codex-zoo.sh
```
The script prompts for a request, runs Codex with the Zoo MCP server, and saves a JSONL transcript (including token usage) to `codex-run-<timestamp>.jsonl`.

## Architecture

Tools are defined in `src/zoo_mcp/*.py`, where they are then imported into
`src/zoo_mcp/server.py` and tied to actual `@mcp.tool()` decorated functions.

`src/zoo_mcp/zoo_tools.py` acts as a large toolset to interact with Zoo's KCL and
engine facilities. This source file houses other utilities like `parse_unit` or
`normalize_ext` (normalizing file extensions).

Modeling scenes use explicit persistent sessions, with at most one session open
per server process. Call `get_modeling_sessions` to recover its ID after a client
reconnect, or call `start_modeling_session` when none exists. Populate the
session with `execute_kcl`, `exec_kcl_project`, or `import_cad_file`; pass the
same `session_id` to `snapshot` and modeling tools; then call
`stop_modeling_session` when finished.

As of 0.28.0, `execute_kcl` and `exec_kcl_project` run mock execution before real
execution and return separate `mock_preflight` and `real_execution` objects.
Each contains `status` (`succeeded`, `failed`, or `not_run`), `message`, and
`diagnostics` grouped by severity. Stage messages are short summaries; the
top-level `message` retains the full report for existing callers. Failed stages
also expose `error_family`, including `ZooMCPTimeoutError` for session timeouts.
Mock errors or an aborted mock execution
return immediately with `ok: false` and `real_execution.status: "not_run"`.
Mock warnings remain in `mock_preflight.diagnostics` even if real execution fails.
The known `planeOf` mock-engine limitation is reported as a warning so the real
engine can evaluate it; other mock errors still block execution.
Session responses expose mock diagnostics; the engine does not return real-stage
diagnostics for session execution.

Path inputs capture the entrypoint, its transitive imports (including linked
modules and glTF buffers), and `project.toml` once. Both stages use that copy
without scanning unrelated files in the containing directory. Dependencies and
symlink targets must stay inside the entrypoint's directory; external paths are
rejected before file reads or execution. Transient local real-execution failures
retain their bounded retries using the same copy without repeating mock execution.
Diagnostics refer to the original source paths. Inline `kcl_code` accepts
self-contained code and standard-library imports; filesystem imports require
`kcl_path` so their dependencies can be captured within an explicit directory.
`exec_kcl_project` now returns
this structured result instead of a path string: check `ok`, then read
`path_artifact_graph` on session success. The standalone `mock_execute_kcl` tool
continues to return its existing boolean/message pair.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on the [GitHub repository](https://github.com/KittyCAD/mcp)

PRs will need to pass tests and linting before being merged.

### [ruff](https://docs.astral.sh/ruff/) is used for linting and formatting.
```bash
uvx ruff check
uvx ruff format
```

### [ty](https://docs.astral.sh/ty/) is used for type checking.
```bash
uvx ty check
```

## Testing

The server includes tests located in [`tests`](`tests`). To run the tests, use the following command:
```bash
uv run pytest -n auto
```
