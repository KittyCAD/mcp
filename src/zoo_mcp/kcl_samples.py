"""KCL Samples fetching and search.

This module fetches the KCL samples index from zoo.dev and provides search
functionality for LLMs. The index is loaded lazily — server.py kicks off the
fetch in the background when the MCP server's lifespan starts, and tools
``await KCLSamples.initialize()`` before serving so the first call also
bootstraps if the lifespan hook hasn't run yet. Repeat calls are no-ops once
the index is loaded.

Both the index page (``/aquarium``) and per-sample pages
(``/aquarium/<sample>``) are requested with ``Accept: text/markdown``.
Per-sample pages embed each KCL file as a ``### <filename>.kcl`` section
with a fenced ```kcl`` ... ``` `` block, which we parse to recover file
contents on demand.
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import ClassVar, TypedDict

import httpx

from zoo_mcp import ctx, logger
from zoo_mcp.utils.data_retrieval_utils import (
    ZOO_BASE_URL,
    extract_excerpt,
    fetch_markdown,
    is_safe_path_component,
)

_AQUARIUM_BASE_URL = f"{ZOO_BASE_URL}/aquarium"

# Only allow safe characters in sample names and filenames
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_-]+\.kcl$")

# Index entries look like:
#   - [Title](/aquarium/<slug>) - <description> (<Categories>)
# The ``- <description>`` chunk and the trailing categories chunk are both
# optional — if a future zoo.dev page lists a slug with no description we
# still want to capture it. Description text may itself contain parens (e.g.
# embedded URLs), so we strip the trailing ``(...)`` group anchored at
# end-of-line in a separate pass.
_INDEX_LINE_RE = re.compile(
    r"^- \[(?P<title>[^\]]+)\]"
    r"\(/aquarium/(?P<name>[A-Za-z0-9_-]+)\)"
    r"(?:\s*-\s*(?P<rest>.*))?\s*$"
)

# Strips a trailing ``(Cat)`` or ``(Cat1, Cat2, ...)`` group from a
# description. Each token must start with an uppercase letter and contain
# only letters, so acronyms (``(API)``, ``(CAD)``) and ``(Manufacturing)``
# are recognized but prose parentheticals (``(see foo)``,
# ``(https://...)``) are deliberately left intact. Categories containing
# digits (``(3D Models)``), hyphens (``(KCL-Std)``), or internal spaces
# (``(Power Tools)``) will NOT be stripped and will leak into the
# description verbatim — extend this pattern if zoo.dev introduces such
# labels.
_TRAILING_CATEGORIES_RE = re.compile(
    r"\s*\((?P<cats>[A-Z][A-Za-z]*(?:,\s*[A-Z][A-Za-z]*)*)\)\s*$"
)

# Matches each "### <filename>.kcl" header followed by a ```kcl ... ``` block
# inside an /aquarium/<sample> page.
_FILE_BLOCK_RE = re.compile(
    r"^###[ \t]+(?P<filename>[^\s/\\]+\.kcl)[ \t]*\n+"
    r"```kcl[ \t]*\n(?P<content>.*?)\n```",
    re.MULTILINE | re.DOTALL,
)


class SampleMetadata(TypedDict):
    """Metadata for a single KCL sample, derived from the /aquarium index."""

    title: str
    description: str
    # ``multipleFiles`` is best-effort: the /aquarium index doesn't expose
    # file counts, so it starts ``False`` for every sample and is corrected
    # in-place by ``get_sample_content`` once the per-sample page has been
    # fetched and parsed. Callers reading this field from a list/search
    # response should treat it as a hint and re-check via ``get_kcl_sample``
    # if a reliable value is needed.
    multipleFiles: bool


class SampleFile(TypedDict):
    """A single file within a KCL sample."""

    filename: str
    content: str


class SampleData(TypedDict):
    """Complete data for a KCL sample including all files."""

    name: str
    title: str
    description: str
    multipleFiles: bool
    files: list[SampleFile]


@dataclass
class KCLSamples:
    """Container for KCL samples data."""

    # Sample metadata indexed by sample directory name
    manifest: dict[str, SampleMetadata] = field(default_factory=dict)
    # Parsed file contents indexed by sample name -> filename -> content
    file_index: dict[str, dict[str, str]] = field(default_factory=dict)

    _instance: ClassVar["KCLSamples | None"] = None
    _init_lock: ClassVar[asyncio.Lock | None] = None

    @classmethod
    def get(cls) -> "KCLSamples":
        """Get the samples index, or an empty one if not initialized."""
        return cls._instance if cls._instance is not None else cls()

    @classmethod
    async def initialize(cls) -> None:
        """Initialize the samples index from zoo.dev. Repeat and concurrent calls are safe."""
        if cls._instance is not None:
            return
        if cls._init_lock is None:
            cls._init_lock = asyncio.Lock()
        async with cls._init_lock:
            if cls._instance is None:
                cls._instance = await _fetch_index_from_zoo_dev()


def _parse_index_markdown(markdown: str) -> dict[str, SampleMetadata]:
    """Parse the /aquarium markdown index into a {name: metadata} mapping."""
    manifest: dict[str, SampleMetadata] = {}
    for line in markdown.splitlines():
        m = _INDEX_LINE_RE.match(line)
        if m is None:
            continue

        name = m.group("name")
        if not is_safe_path_component(name, _SAFE_NAME_RE):
            logger.warning(f"Rejected unsafe sample name from index: {name!r}")
            continue

        raw_rest = m.group("rest")
        rest = raw_rest.strip() if raw_rest else ""
        cats_match = _TRAILING_CATEGORIES_RE.search(rest)
        if cats_match:
            description = rest[: cats_match.start()].rstrip()
        else:
            description = rest

        manifest[name] = SampleMetadata(
            title=m.group("title").strip(),
            description=description,
            multipleFiles=False,
        )

    return manifest


def _parse_aquarium_markdown(markdown: str) -> dict[str, str]:
    """Extract ``filename -> content`` pairs from an /aquarium/<sample> page.

    The page contains a ``## Files`` section followed by repeated
    ``### <filename>.kcl`` headers, each immediately followed by a fenced
    ```kcl`` ... ``` `` block holding that file's source.
    """
    files: dict[str, str] = {}
    for match in _FILE_BLOCK_RE.finditer(markdown):
        filename = match.group("filename")
        if is_safe_path_component(filename, _SAFE_FILENAME_RE):
            files[filename] = match.group("content")
        else:
            logger.warning(f"Rejected unsafe filename in markdown: {filename!r}")
    return files


async def _fetch_sample_files(
    client: httpx.AsyncClient,
    sample_name: str,
) -> dict[str, str]:
    """Fetch a sample's files from /aquarium/<sample_name> as markdown."""
    url = f"{_AQUARIUM_BASE_URL}/{sample_name}"
    markdown = await fetch_markdown(client, url, sample_name)
    if markdown is None:
        return {}
    return _parse_aquarium_markdown(markdown)


async def _fetch_index_from_zoo_dev() -> KCLSamples:
    """Fetch the samples index from zoo.dev and return a KCLSamples instance."""
    samples = KCLSamples()

    logger.info(f"Fetching KCL samples index from {ZOO_BASE_URL}...")

    async with httpx.AsyncClient(timeout=30.0, verify=ctx) as client:
        markdown = await fetch_markdown(client, _AQUARIUM_BASE_URL, "/aquarium")
        if markdown is None:
            return samples

        samples.manifest = _parse_index_markdown(markdown)

    logger.info(f"KCL samples index loaded with {len(samples.manifest)} samples")
    return samples


async def initialize_samples_index() -> None:
    """Initialize the samples index from zoo.dev."""
    await KCLSamples.initialize()


def list_available_samples() -> list[dict]:
    """Return a list of all available KCL samples with basic info.

    Returns a list of dictionaries, each containing:
    - name: The sample directory name (used to retrieve the sample)
    - title: Human-readable title
    - description: Brief description of the sample
    - multipleFiles: Whether the sample contains multiple KCL files.

    Note on ``multipleFiles``: the /aquarium index page does not expose
    file counts, so this field is ``False`` for any sample whose per-sample
    page has not yet been fetched. It becomes accurate only after
    ``get_kcl_sample`` has been called for that sample (which caches the
    parsed file list and updates the metadata in-place). Treat
    ``multipleFiles`` as a best-effort hint here; call ``get_kcl_sample``
    if you need a reliable answer.

    Use get_kcl_sample() with the name to retrieve the full sample content.

    Returns:
        list[dict]: List of sample information dictionaries.
    """
    samples = KCLSamples.get()
    result = []

    for name, metadata in sorted(samples.manifest.items()):
        result.append(
            {
                "name": name,
                "title": metadata.get("title", name),
                "description": metadata.get("description", ""),
                "multipleFiles": metadata.get("multipleFiles", False),
            }
        )

    return result


def search_samples(query: str, max_results: int = 5) -> list[dict]:
    """Search samples by keyword in title and description.

    Searches across all KCL sample titles and descriptions
    for the given query. Returns matching samples ranked by relevance.

    Args:
        query (str): The search query (case-insensitive).
        max_results (int): Maximum number of results to return (default: 5).

    Returns:
        list[dict]: List of search results, each containing:
            - name: The sample directory name (used to retrieve the sample)
            - title: Human-readable title
            - description: Brief description of the sample
            - multipleFiles: Whether the sample contains multiple KCL files.
              Best-effort hint only — see ``list_available_samples`` for the
              full caveat. Call ``get_kcl_sample`` if you need a reliable
              answer.
            - match_count: Number of times the query appears in title/description
            - excerpt: A relevant excerpt with the match in context
    """
    if not query or not query.strip():
        return [{"error": "Empty search query"}]

    query = query.strip()
    query_lower = query.lower()
    results: list[dict] = []

    samples = KCLSamples.get()

    for name, metadata in samples.manifest.items():
        title = metadata.get("title", name)
        description = metadata.get("description", "")
        searchable = f"{title} {description} {name}"
        searchable_lower = searchable.lower()

        match_count = searchable_lower.count(query_lower)
        if match_count > 0:
            title_matches = title.lower().count(query_lower)
            score = match_count + (title_matches * 3)

            excerpt = extract_excerpt(searchable, query, context_chars=150)

            results.append(
                {
                    "name": name,
                    "title": title,
                    "description": description,
                    "multipleFiles": metadata.get("multipleFiles", False),
                    "match_count": match_count,
                    "excerpt": excerpt,
                    "_score": score,
                }
            )

    results.sort(key=lambda x: x["_score"], reverse=True)

    for r in results:
        del r["_score"]

    return results[:max_results]


async def get_sample_content(sample_name: str) -> SampleData | None:
    """Get the full content of a specific KCL sample including all files.

    Use list_kcl_samples() to see available sample names, or
    search_kcl_samples() to find samples by keyword.

    Args:
        sample_name (str): The sample directory name
            (e.g., "ball-bearing", "axial-fan")

    Returns:
        SampleData | None: A dictionary containing:
            - name: The sample directory name
            - title: Human-readable title
            - description: Brief description
            - multipleFiles: Whether the sample contains multiple files
            - files: List of file dictionaries, each with 'filename' and 'content'
        Returns None if the sample is not found.
    """
    samples = KCLSamples.get()

    if not is_safe_path_component(sample_name, _SAFE_NAME_RE):
        return None

    metadata = samples.manifest.get(sample_name)
    if metadata is None:
        return None

    if sample_name in samples.file_index:
        file_contents = samples.file_index[sample_name]
    else:
        async with httpx.AsyncClient(timeout=30.0, verify=ctx) as client:
            file_contents = await _fetch_sample_files(client, sample_name)

        if not file_contents:
            return None

        samples.file_index[sample_name] = file_contents
        # The index doesn't tell us file counts; record the real value now
        # that we've parsed the per-sample page.
        metadata["multipleFiles"] = len(file_contents) > 1

    files_list: list[SampleFile] = []
    for filename, content in sorted(file_contents.items()):
        files_list.append(SampleFile(filename=filename, content=content))

    return SampleData(
        name=sample_name,
        title=metadata.get("title", sample_name),
        description=metadata.get("description", ""),
        multipleFiles=metadata.get("multipleFiles", False),
        files=files_list,
    )
