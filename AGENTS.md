## Development Commands
Before committing any code, ensure all tests pass, formatting is good, linting is clean, and type checks succeed.

### Environment Setup
- `uv venv` - Create virtual environment
- `uv pip install -e .` - Install package in development mode
- `export ZOO_API_TOKEN="your_api_key_here"` - Set required Zoo API token

### Running the Server
- `uv run -m zoo_mcp` - Start the MCP server locally
- `uv run mcp run src/zoo_mcp/server.py` - Alternative method using mcp package
- `uv run mcp dev src/zoo_mcp/server.py` - Run server with MCP Inspector for testing

### Testing and Quality
- `uv run -n auto pytest` - Run all tests
- `uv run pytest tests/test_server.py` - Run specific test file
- `uv run ruff check` - Run linter
- `uv run ruff format` - Format code
- `uv run ty check` - Type check source code

### Integration Commands
- `uv run mcp install src/zoo_mcp/server.py` - Install server for Claude Desktop integration

## Architecture

This is a Model Context Protocol (MCP) server that exposes Zoo CAD and KCL utility tools to AI assistants. The architecture consists of:

### Core Components
- `src/zoo_mcp/server.py` - FastMCP server that defines the MCP interface and registers the tools
- `src/zoo_mcp/zoo_tools.py` - Contains the Zoo and KCL tool implementations that interface with the KittyCAD API and local KCL runtime
- `src/zoo_mcp/__init__.py` - Package initialization with logging configuration

### Key Dependencies
- `kittycad` - Official Zoo API client for accessing Zoo file, modeling, and org-dataset functionality
- `mcp[cli]` - Model Context Protocol framework for AI assistant integration
- `pytest-asyncio` - For testing async functions

### API Integration
The server connects to Zoo's APIs using the KittyCAD client. Requests that hit Zoo APIs require a valid `ZOO_API_TOKEN` environment variable. The tool implementations cover:
- CAD file conversion and physical-property calculations
- KCL execution, formatting, linting, export, and snapshots
- Org dataset listing and semantic search

### Testing Strategy  
Tests are located in `tests/test_server.py` and cover:
- Basic tool functionality
- Success scenarios
- Failure scenarios
- All tests are async and use pytest-asyncio

## Package Structure
Built as a standard Python package using setuptools with source code in `src/zoo_mcp/`. The package can be installed via pip/uv or used directly as a module.
