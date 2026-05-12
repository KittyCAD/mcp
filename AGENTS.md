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
- `uv run -n auto pytest` - Run all tests (includes `live` tests that hit zoo.dev)
- `uv run -n auto pytest -m "not live"` - Run all tests except those that hit live external services (use this when offline or to avoid network calls)
- `uv run pytest tests/test_server.py` - Run specific test file
- `uv run ruff check` - Run linter
- `uv run ruff format` - Format code
- `uv run ty check` - Type check source code

### Integration Commands
- `uv run mcp install src/zoo_mcp/server.py` - Install server for Claude Desktop integration

## Architecture

This is a Model Context Protocol (MCP) server that exposes Zoo CAD and KCL utility tools to AI assistants. The architecture consists of:

### Core Components
- `src/zoo_mcp/server.py` - FastMCP server that defines the MCP interface and registers all `@mcp.tool()` entry points (KCL execution, snapshots, physical-property calculations, KCL docs/samples lookup, org datasets, etc.). Owns the lifespan hook that lazily populates the KCL docs/samples indexes.
- `src/zoo_mcp/zoo_tools.py` - Implementations of the CAD-oriented tools that talk to Zoo's KittyCAD API (executing/exporting KCL, file conversion, snapshots, physical properties, sketch constraint status, lint-and-fix, org dataset listing/semantic search, etc.).
- `src/zoo_mcp/kcl_docs.py` - Fetches and indexes KCL documentation from `zoo.dev` (via the sitemap, with `Accept: text/markdown`) and exposes list/search/get helpers backed by a lazily-initialized index.
- `src/zoo_mcp/kcl_samples.py` - Fetches and indexes KCL samples from `zoo.dev/aquarium` and exposes list/search/get helpers; per-sample file contents are parsed from the per-sample markdown pages on demand.
- `src/zoo_mcp/utils/data_retrieval_utils.py` - Shared helpers for fetching `zoo.dev` pages safely (path validation, redirect-blocking fetches, markdown excerpting) used by `kcl_docs.py` and `kcl_samples.py`.
- `src/zoo_mcp/utils/image_utils.py` - Image utilities used by snapshot tools (encoding to MCP `ImageContent`, saving to disk, collage building, resizing).
- `src/zoo_mcp/__init__.py` - Package initialization: configures logging, builds the shared `kittycad_client` (with TLS / websocket settings) used across the modules, defines `ZooMCPException`.
- `src/zoo_mcp/__main__.py` - `python -m zoo_mcp` entry point; delegates to `server.main`.

### Key Dependencies
- `kittycad` - Official Zoo API client for accessing KCL execution and org-dataset endpoints
- `kcl` - Python bindings for the KCL language used by execute/format/lint/snapshot tools
- `mcp[cli]` - Model Context Protocol framework for AI assistant integration
- `httpx` - Async HTTP client used to fetch KCL docs/samples from `zoo.dev`
- `pytest-asyncio` - For testing async functions

### API Integration
The server connects to Zoo's KCL execution APIs using the KittyCAD client, and to `zoo.dev` markdown pages via `httpx` for the docs/samples indexes. All KittyCAD requests require a valid `ZOO_API_TOKEN` environment variable. Notable flows:
- KCL execution / export / snapshot tools (in `zoo_tools.py`) use the `kcl` bindings and the KittyCAD client's modeling/execution endpoints.
- Org datasets (`list_org_datasets`, `search_org_dataset_semantic`) call KittyCAD's org-datasets endpoints, with a raw-HTTP fallback when the SDK's pydantic models reject newly-added backend fields.
- KCL docs/samples (`list_kcl_docs`, `search_kcl_docs`, `get_kcl_doc`, `list_kcl_samples`, `search_kcl_samples`, `get_kcl_sample`) read from the in-memory indexes populated lazily from `zoo.dev` (sitemap-driven for docs, `/aquarium` index for samples).

### Testing Strategy
Tests live in `tests/` and are split across:
- `tests/test_server.py` - Exercises every MCP tool end-to-end through `mcp.call_tool`, mixing real KCL/CAD calls with mocked KittyCAD responses for org-dataset tools, and synthetic in-memory indexes for KCL docs/samples tools.
- `tests/test_docs.py`, `tests/test_samples.py` - Unit tests for the docs categorization / title extraction and the samples markdown index/page parsers.
- `tests/test_data_retrieval_utils.py` - Unit tests for the shared `zoo.dev` fetch helpers (path safety, excerpt extraction, redirect-blocking fetch, markdown `Accept` header).
- `tests/test_live_zoo_dev.py` - Marked `live`; hits `zoo.dev` end-to-end for the docs and samples tools so breakages in the upstream markdown shape are caught. Deselect with `-m "not live"` when offline.
- All async tests use `pytest-asyncio`.

## Package Structure
Built as a standard Python package using setuptools with source code in `src/zoo_mcp/`. The package can be installed via pip/uv or used directly as a module.
