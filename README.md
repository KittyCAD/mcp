# Zoo Model Context Protocol (MCP) Server

An [MCP server](https://modelcontextprotocol.io/docs/getting-started/intro) housing various Zoo built utilities

<!-- mcp-name: io.github.KittyCAD/zoo-mcp -->

## Prerequisites

A Zoo account. Run `uvx zoo-mcp auth login` once before starting the server.
The command opens your browser and prints a verification URL and code.
The server reuses your saved login; public tools also work without login.

You can also use an [API key](https://zoo.dev/account):
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

Manage device login explicitly:
```bash
uvx zoo-mcp auth login
uvx zoo-mcp auth login --no-browser
uvx zoo-mcp auth status
uvx zoo-mcp auth logout
```

Logins are saved per API origin in your system credential store using
[`keyring`](https://keyring.readthedocs.io/): macOS Keychain, Windows Credential
Locker, or a supported Linux keyring. On headless systems without an available
keyring, set `ZOO_API_TOKEN` instead. Status never displays tokens. Logout removes
only the saved login for the current origin; restart running servers afterward.
If your login expires or is revoked, run `auth login` again and restart the server.
If the login command fails because of a network error, run it again to retry.
Server startup never opens a browser. Device authorization and polling use
[`requests-oauth2client`](https://github.com/guillp/requests_oauth2client).
Device login requires the API's dedicated Zoo MCP device client registration.

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

### Supplying client configuration from Python

Local entry points capture credentials when Zoo MCP is imported at startup.
Token precedence is `KITTYCAD_API_TOKEN`, `ZOO_API_TOKEN`, then the registry's
`ZOO_TOKEN` alias. Any explicit environment token bypasses saved/device login.
API origin precedence is `ZOO_HOST`, `KITTYCAD_HOST`, then `https://api.zoo.dev`.
Saved logins never cross API origins.

Applications can supply client configuration explicitly without changing the process
environment or adding authentication arguments to MCP tools:

```python
from zoo_mcp.client_config import ZooClientConfig, use_client_config
from zoo_mcp.server import mcp

with use_client_config(ZooClientConfig(token=api_token, base_url=api_origin)):
    result = await mcp.call_tool("list_org_datasets", {})
```

The context supplies the same token, origin, and headers to the KittyCAD SDK and KCL
session constructors and restores the previous configuration on exit. New
calls can use refreshed configuration. Finish child tasks before leaving their
configuration context.
An explicit configuration context also prevents the server lifespan from loading a
saved local login. Hosted workers and embeddings should always use this interface.

To forward custom HTTP headers, pass `headers={"User-Agent": "MyApp/1.0",
"X-Request-Source": "example"}` to `ZooClientConfig`. These headers reach REST
requests and modeling WebSocket handshakes, including KCL sessions. A supplied
`User-Agent` replaces the default value. Tokens and headers are excluded from
configuration representations.

Client configuration applies to downstream calls. Applications serving different
owners must also isolate their process-wide modeling sessions and workspaces.

### Cloud project filesystem

The `project_write_file`, `project_mkdir`, `project_rm`, `project_cp`,
`project_mv`, `project_access`, `project_stat`, `project_readdir`, and
`project_read_file` tools operate on an existing `cloud:<uuid>` project using
its caller's Zoo credentials. Paths use project-relative POSIX syntax; `.`
selects the root for metadata and directory reads. Quoted filenames are preserved;
control characters in paths are rejected. Operations stay within one
project.

Every mutation fetches current metadata and complete file contents, applies its
changes in memory, then uploads the whole project with `expected_revision` and
explicit `deleted_paths`. Copies and moves use one upload, including recursive
directory operations. A revision conflict reloads the latest project and
reapplies the operation, up to three upload attempts. Thus a write replaces the
target's latest contents, and recursive removal includes files added before a
retry. Unrelated files and project metadata are preserved. Connection failures,
timeouts, other HTTP errors, and unreadable responses surface to the caller
without an automatic retry. After an upload error, the caller can inspect the
project before deciding whether to retry.

The default limit is **256 MiB of total uncompressed project contents**, based on
four times text-to-CAD's 64 MiB DFM capture limit. Callers can override
`max_project_bytes`. The limit applies before downloading, while reading the
archive, and after mutations. A complete operation has a 300-second deadline.
There is no persistent content cache.

- `project_write_file` accepts raw Python bytes when encoding is omitted;
  strings default to UTF-8. Explicit `encoding="utf-8"` or `encoding="base64"`
  applies to strings. Use base64 for binary contents over JSON MCP transports.
  Its parent directory must exist. A successful result includes the acknowledged
  project metadata plus `path` (normalized and project-relative), `contents`, and
  `encoding`. Text returns with `encoding="utf-8"`; raw bytes and base64 inputs
  return base64-encoded contents with `encoding="base64"`.
- `project_mkdir` requires an existing parent and a nonexistent target. It
  persists the directory using a visible, zero-byte `.empty` file. That marker
  can stay after files are added. Removing or moving a directory's last child
  also creates a marker to preserve the parent directory. Explicitly removing
  the `.empty` marker allows an otherwise empty directory to disappear.
- `project_rm` removes a file or directory recursively. Paths are literal; there
  is no glob expansion.
- `project_cp` and `project_mv` append the source basename when the destination
  is a directory. Existing files can be replaced; directory collisions are
  errors. Moves remove the source in the same upload that creates the target.
- `main.kcl` must remain the entrypoint. It can be edited but cannot be removed,
  moved, or overwritten through copy/move. The project root cannot be removed,
  copied, or moved. The API also requires `project.toml`, so removing or moving
  that manifest is rejected.
- `project_access` returns the API's current project capabilities for an existing
  path. `project_stat` returns file metadata or an inferred directory record;
  directory IDs, timestamps, sizes, and hashes are null.
- `project_readdir` returns `{ "files": [...] }` for immediate children.
- `project_read_file` returns `{ "contents": "...", "encoding": "utf-8" }`.
  Set `encoding="base64"` for binary files.

Python callers can use the implementation directly:

```python
from zoo_mcp.credentials import ZooCredentials, use_credentials
from zoo_mcp.zoo_tools import ProjectFileSystem

with use_credentials(ZooCredentials(token=api_token)):
    project = ProjectFileSystem("cloud:00000000-0000-4000-8000-000000000001")
    await project.mkdir("parts")
    await project.write_file("parts/bracket.kcl", "// KCL contents")
    contents = await project.read_file("parts/bracket.kcl", encoding="utf-8")
```

`read_file` returns bytes when no encoding is supplied. A `ProjectFileSystem`
instance retains the credentials supplied when it was constructed; construct a
new instance when changing the token or API host.

### Prebuilt binaries

Each [GitHub release](https://github.com/KittyCAD/mcp/releases) also attaches standalone executables (built with PyInstaller) for Linux (`x86_64`, `arm64`), macOS (`arm64`, `x86_64`), and Windows (`x86_64`) — no Python toolchain required. Download the binary for your platform, set `ZOO_API_TOKEN`, and run it directly, e.g.:
```bash
ZOO_API_TOKEN="your_api_key_here" ./zoo-mcp-linux-x86_64
```
> The binaries are not code-signed, so macOS Gatekeeper and Windows SmartScreen may warn on first run.

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
session with `execute_kcl` or `import_cad_file`, passing `session_id` to either
tool. Reuse the same `session_id` with `snapshot` and modeling tools; then call
`stop_modeling_session` when finished.

Sessions remain available across client reconnects while the server and engine
connection stay alive. Idle sessions continue receiving engine status messages
so keepalive checks remain responsive. Disconnected sessions are removed from
`get_modeling_sessions`; start a new session and execute the model again to recover.

KCL execution and persistent modeling sessions use the CPU engine pool. Isolated
KCL snapshots temporarily enable CPU rendering. Persistent sessions initialize
rendering on the first snapshot, selection, highlight, or camera tool and retain
it until the session is stopped, preserving visual state across calls. Exports,
measurements, and geometry queries do not initialize graphics.

The KCL samples tools list and search public Aquarium projects from the Zoo API.
Pass the returned `name` (a public project ID) to `get_kcl_sample` to download its
KCL sources. Filenames preserve project-relative paths, including subdirectories.
These public tools do not require an API token. The `multipleFiles` field in
list/search results is a hint until the project has been downloaded.

`execute_kcl` runs mock execution before real execution and returns separate
`mock_preflight` and `real_execution` objects.
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
When calling `execute_kcl` with `session_id`, check `ok`, then read
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
