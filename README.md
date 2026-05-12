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
    uv pip install git+ssh://git@github.com/KittyCAD/zoo-mcp.git
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

## Integrations

The server can be used as is by [running the server](#running-the-server) or importing directly into your python code.
```python
from zoo_mcp.server import mcp

mcp.run()
```

Individual tools can be used in your own python code as well

```python
from mcp.server.fastmcp import FastMCP
from zoo_mcp.zoo_tools import zoo_execute_kcl

mcp = FastMCP(name="My Example Server")


@mcp.tool()
async def my_execute_kcl(kcl_code: str) -> tuple[bool, str]:
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
