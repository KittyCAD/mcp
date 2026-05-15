"""KCL Documentation fetching and search.

This module fetches KCL documentation from zoo.dev and provides search
functionality for LLMs. The index is loaded lazily — server.py kicks off the
fetch in the background when the MCP server's lifespan starts, and tools
``await KCLDocs.initialize()`` before serving so the first call also
bootstraps if the lifespan hook hasn't run yet. Repeat calls are no-ops once
the index is loaded.

Pages are requested with ``Accept: text/markdown`` so we get clean markdown
rather than rendered HTML.
"""

import asyncio
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from posixpath import normpath
from typing import ClassVar
from urllib.parse import unquote, urlparse

import httpx

from zoo_mcp import ctx, logger
from zoo_mcp.utils.data_retrieval_utils import (
    ZOO_BASE_URL,
    ZOO_SITEMAP_URL,
    extract_excerpt,
    fetch_markdown,
    is_safe_path_component,
)

# Doc identifiers look like "docs/kcl-lang/<page>" or
# "docs/kcl-std/<group>/<page>" (no file extension; these are zoo.dev URL paths).
_SAFE_DOC_PATH_RE = re.compile(r"^docs/[A-Za-z0-9/_-]+$")

_SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# Cap concurrent zoo.dev requests so startup doesn't hammer the site.
_FETCH_CONCURRENCY = 16


def _is_safe_doc_path(path: str) -> bool:
    """Validate that a doc path is safe and does not contain traversal sequences."""
    if not is_safe_path_component(path, _SAFE_DOC_PATH_RE):
        return False

    normalized = normpath(unquote(path))
    if not normalized.startswith("docs/"):
        return False

    return True


@dataclass
class KCLDocs:
    """Container for documentation data."""

    docs: dict[str, str] = field(default_factory=dict)
    index: dict[str, list[str]] = field(
        default_factory=lambda: {
            "kcl-lang": [],
            "kcl-std-functions": [],
            "kcl-std-types": [],
            "kcl-std-consts": [],
            "kcl-std-modules": [],
        }
    )

    _instance: ClassVar["KCLDocs | None"] = None
    _init_lock: ClassVar[asyncio.Lock | None] = None

    @classmethod
    def get(cls) -> "KCLDocs":
        """Get the docs index, or an empty one if not initialized."""
        return cls._instance if cls._instance is not None else cls()

    @classmethod
    async def initialize(cls) -> None:
        """Initialize the docs index from zoo.dev. Repeat and concurrent calls are safe."""
        if cls._instance is not None:
            return
        if cls._init_lock is None:
            cls._init_lock = asyncio.Lock()
        async with cls._init_lock:
            if cls._instance is None:
                cls._instance = await _fetch_docs_from_zoo_dev()


def _categorize_doc_path(path: str) -> str | None:
    """Categorize a doc path into one of the index categories."""
    if path.startswith("docs/kcl-lang/"):
        return "kcl-lang"
    elif path.startswith("docs/kcl-std/functions/"):
        return "kcl-std-functions"
    elif path.startswith("docs/kcl-std/types/"):
        return "kcl-std-types"
    elif path.startswith("docs/kcl-std/consts/"):
        return "kcl-std-consts"
    elif path.startswith("docs/kcl-std/modules/"):
        return "kcl-std-modules"
    return None


def _extract_title(content: str) -> str:
    """Extract the title from Markdown content (first # heading)."""
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return ""


async def _discover_doc_paths(client: httpx.AsyncClient) -> list[str]:
    """Walk zoo.dev's sitemap and return KCL doc identifiers."""

    base_netloc = urlparse(ZOO_BASE_URL).netloc

    try:
        response = await client.get(ZOO_SITEMAP_URL, follow_redirects=False)
        response.raise_for_status()
        index_root = ET.fromstring(response.text)
    except (httpx.HTTPError, ET.ParseError) as e:
        logger.warning(f"Failed to fetch sitemap index from {ZOO_BASE_URL}: {e}")
        return []

    child_urls: list[str] = []
    for loc in index_root.findall("sm:sitemap/sm:loc", _SITEMAP_NS):
        if loc.text and urlparse(loc.text).netloc == base_netloc:
            child_urls.append(loc.text)

    paths: set[str] = set()
    for sitemap_url in child_urls:
        try:
            response = await client.get(sitemap_url, follow_redirects=False)
            response.raise_for_status()
            root = ET.fromstring(response.text)
        except (httpx.HTTPError, ET.ParseError) as e:
            logger.warning(f"Failed to fetch sitemap {sitemap_url}: {e}")
            continue

        for url in root.findall("sm:url/sm:loc", _SITEMAP_NS):
            if url.text is None:
                continue
            parsed = urlparse(url.text)
            if parsed.netloc != base_netloc:
                continue
            ident = parsed.path.lstrip("/")
            if _categorize_doc_path(ident) and _is_safe_doc_path(ident):
                paths.add(ident)

    return sorted(paths)


async def _fetch_docs_from_zoo_dev() -> KCLDocs:
    """Fetch all KCL docs from zoo.dev."""
    docs = KCLDocs()

    logger.info(f"Fetching KCL documentation from {ZOO_BASE_URL}...")

    async with httpx.AsyncClient(timeout=30.0, verify=ctx) as client:
        doc_paths = await _discover_doc_paths(client)
        if not doc_paths:
            return docs

        logger.info(f"Found {len(doc_paths)} documentation pages")

        sem = asyncio.Semaphore(_FETCH_CONCURRENCY)

        async def fetch_one(path: str) -> tuple[str, str | None]:
            async with sem:
                content = await fetch_markdown(client, f"{ZOO_BASE_URL}/{path}", path)
            return path, content

        results = await asyncio.gather(*(fetch_one(p) for p in doc_paths))

        for path, content in results:
            if content is None:
                continue
            docs.docs[path] = content
            category = _categorize_doc_path(path)
            if category and category in docs.index:
                docs.index[category].append(path)

    for category in docs.index:
        docs.index[category].sort()

    logger.info(f"KCL documentation index initialized with {len(docs.docs)} files")
    return docs


async def initialize_docs_index() -> None:
    """Initialize the docs index from zoo.dev."""
    await KCLDocs.initialize()


def list_available_docs() -> dict[str, list[str]]:
    """Return categorized list of available documentation.

    Returns a dictionary with the following categories:
    - kcl-lang: KCL language documentation (syntax, types, functions, etc.)
    - kcl-std-functions: Standard library function documentation
    - kcl-std-types: Standard library type documentation
    - kcl-std-consts: Standard library constants documentation
    - kcl-std-modules: Standard library module documentation

    Each category contains a list of documentation paths that can be
    retrieved using get_kcl_doc().

    Returns:
        dict: Categories mapped to lists of available documentation paths.
    """
    return KCLDocs.get().index


def search_docs(query: str, max_results: int = 5) -> list[dict]:
    """Search docs by keyword.

    Searches across all KCL language and standard library documentation
    for the given query. Returns relevant excerpts with surrounding context.

    Args:
        query (str): The search query (case-insensitive).
        max_results (int): Maximum number of results to return (default: 5).

    Returns:
        list[dict]: List of search results, each containing:
            - path: The documentation path
            - title: The document title (from first heading)
            - excerpt: A relevant excerpt with the match highlighted in context
            - match_count: Number of times the query appears in the document
    """

    if not query or not query.strip():
        return [{"error": "Empty search query"}]

    query = query.strip()
    query_lower = query.lower()
    results: list[dict] = []

    for path, content in KCLDocs.get().docs.items():
        content_lower = content.lower()

        match_count = content_lower.count(query_lower)
        if match_count > 0:
            title = _extract_title(content)
            excerpt = extract_excerpt(content, query)

            results.append(
                {
                    "path": path,
                    "title": title,
                    "excerpt": excerpt,
                    "match_count": match_count,
                }
            )

    results.sort(key=lambda x: x["match_count"], reverse=True)

    return results[:max_results]


def get_doc_content(doc_path: str) -> str | None:
    """Get the full content of a specific KCL documentation file.

    Use list_kcl_docs() to see available documentation paths, or
    search_kcl_docs() to find relevant documentation by keyword.

    Args:
        doc_path (str): The path to the documentation file
            (e.g., "docs/kcl-lang/functions" or
            "docs/kcl-std/functions/std-sketch-extrude").

    Returns:
        str: The full Markdown content of the documentation file,
            or None if not found or the path is unsafe.
    """

    if not _is_safe_doc_path(doc_path):
        return None

    return KCLDocs.get().docs.get(doc_path)
