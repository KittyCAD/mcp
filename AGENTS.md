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
- `uv run pytest -n auto` - Run all tests (includes `live` tests that hit zoo.dev)
- `uv run pytest -n auto -m "not live"` - Run all tests except those that hit live external services (use this when offline or to avoid network calls)
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
- `src/zoo_mcp/utils/image_utils.py` - Image utilities used by the `snapshot` tool (encoding to MCP `ImageContent`, saving to disk, tiling up to four views into a collage, resizing).
- `src/zoo_mcp/__init__.py` - Package initialization: configures logging and the shared TLS context, and defines `ZooMCPException` and `ZooMCPTimeoutError`.
- `src/zoo_mcp/__main__.py` - `python -m zoo_mcp` entry point; delegates to `server.main`.

### Key Dependencies
- `kittycad` - Official Zoo API client for accessing KCL execution and org-dataset endpoints
- `kcl` - Python bindings for the KCL language used by the execute/format/lint tools
- `mcp[cli]` - Model Context Protocol framework for AI assistant integration
- `httpx` - Async HTTP client used to fetch KCL docs/samples from `zoo.dev`
- `pytest-asyncio` - For testing async functions

### API Integration
The server connects to Zoo's KCL execution APIs using the KittyCAD client, and to `zoo.dev` markdown pages via `httpx` for the docs/samples indexes. All KittyCAD requests require a valid `ZOO_API_TOKEN` environment variable. Notable flows:
- KCL execution / export tools (in `zoo_tools.py`) use the `kcl` bindings and the KittyCAD client's modeling/execution endpoints.
- Modeling tools operate on persistent sessions, with at most one session open per server process. Callers use `get_modeling_sessions` to recover the active ID after reconnecting or `start_modeling_session` when none exists, then populate the session through `execute_kcl`, `exec_kcl_project`, or `import_cad_file`. They pass its `session_id` to `snapshot` and query tools and call `stop_modeling_session` when finished. `zoo_snapshot` then loops camera → zoom-to-fit → take-snapshot on that session and tiles the results.
- Every modeling tool call is bounded by `zoo_tools.MODELING_COMMAND_TIMEOUT`, a monotonic wall-clock budget shared by every websocket send and read the call makes. It cannot be a per-read timeout: a live session interleaves unsolicited frames every ~10s, and each one would restart a per-read timeout, so a stalled import blocked its caller for as long as the connection lived. Expiry raises `ZooMCPTimeoutError` (a `ZooMCPException` subclass, so it is retryable but distinguishable from an engine rejection), aborts the socket, and evicts the session. Modeling sessions use an asynchronous websocket authenticated from the same loop-local `AsyncKittyCAD` client configuration as the REST tools, so pending engine responses yield to other MCP calls and cancellation.
- Org datasets (`list_org_datasets`, `search_org_dataset_semantic`) call KittyCAD's org-datasets endpoints, with a raw-HTTP fallback when the SDK's pydantic models reject newly-added backend fields. Listing passes `lookup_enabled=true`, so datasets an org has excluded from lookup are never surfaced or searched.
- KCL docs/samples (`list_kcl_docs`, `search_kcl_docs`, `get_kcl_doc`, `list_kcl_samples`, `search_kcl_samples`, `get_kcl_sample`) read from the in-memory indexes populated lazily from `zoo.dev` (sitemap-driven for docs, `/aquarium` index for samples).

### Testing Strategy
Tests live in `tests/` and are split across:
- `tests/test_server.py` - Exercises every MCP tool end-to-end through `mcp.call_tool`, mixing real KCL/CAD calls with mocked KittyCAD responses for org-dataset tools, and synthetic in-memory indexes for KCL docs/samples tools.
- `tests/test_docs.py`, `tests/test_samples.py` - Unit tests for the docs categorization / title extraction and the samples markdown index/page parsers.
- `tests/test_data_retrieval_utils.py` - Unit tests for the shared `zoo.dev` fetch helpers (path safety, excerpt extraction, redirect-blocking fetch, markdown `Accept` header).
- `tests/test_modeling_commands.py`, `tests/test_server_modeling_commands.py` - Mocked-transport tests for the modeling websocket layer, including the exact command sequence `zoo_snapshot` sends, how the `snapshot` tool resolves its `camera_view` argument, and the command deadline (success, engine failure, connection close, a stream of unmatched frames, and no-response timeout). The deadline tests drive a fake monotonic clock so a 300s budget is exercised without a 300s test.
- `tests/test_live_zoo_dev.py` - Marked `live`; hits `zoo.dev` end-to-end for the docs and samples tools so breakages in the upstream markdown shape are caught. Deselect with `-m "not live"` when offline.
- All async tests use `pytest-asyncio`.

## Package Structure
Built as a standard Python package using setuptools with source code in `src/zoo_mcp/`. The package can be installed via pip/uv or used directly as a module.

### Releasing and versioning
`server.json` is the manifest `mcp-publisher` publishes to the MCP registry at
https://registry.modelcontextprotocol.io. It is checked in and read from the repo at release time, so
its contents — name, description, repository, and the `packages[0]` entry describing the PyPI package,
its transport, and its environment variables — are what the registry serves to anyone installing this
server. Treat it as a release artifact, not as documentation.

`release.yml` runs on a pushed tag: build wheels and binaries, publish to PyPI, then publish
`server.json` to the registry. Bump the version in the same commit in all of:
- `pyproject.toml` - `version`. `release.yml` builds the wheel straight from it, so this is what lands on PyPI.
- `server.json` - `packages[0].version`. Nothing in CI rewrites this one, and the registry resolves the PyPI package at exactly this version, which is why the workflow sleeps 60s waiting for the PyPI publish to land first. Leave it stale and the registry entry points at the previous release.
- `server.json` - the top-level `version`. The workflow overwrites this from the tag (`jq '.version = $v'`) just before publishing, so a stale value will not reach the registry, but keep it in step so the checked-in manifest is not self-contradicting.
- `uv.lock` - pins this package's own version. Run `uv lock` last and commit the result rather than hand-editing it.

Nothing validates the tag against `pyproject.toml`. Tag `vX.Y.Z` must match the version bumped here, or
PyPI gets one version while the registry advertises another.
