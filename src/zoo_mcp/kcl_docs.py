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
from markdown_it import MarkdownIt

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
    return normalized.startswith("docs/")


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
    _parsed_docs: dict[str, tuple[str, "_ParsedDoc"]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def parsed_doc(self, path: str, content: str) -> "_ParsedDoc":
        """Parse each page once, refreshing it if its source changes."""
        cached = self._parsed_docs.get(path)
        if cached is None or cached[0] != content:
            cached = (content, _parse_doc(content))
            self._parsed_docs[path] = cached
        return cached[1]

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
                result = await _fetch_docs_from_zoo_dev()
                if result.docs:
                    cls._instance = result


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


_MARKDOWN = MarkdownIt("commonmark").enable("table")
_MODEL_VIEWER_START = re.compile(r"^\s*<model-viewer(?:\s|>)", re.IGNORECASE)
_MODEL_VIEWER_END = re.compile(r"</model-viewer\s*>", re.IGNORECASE)


@dataclass
class _MarkdownBlock:
    text: str
    kind: str
    has_code: bool = False
    complete: bool = True
    table_header: str = ""
    table_rows: list[str] = field(default_factory=list)


@dataclass
class _MarkdownSection:
    heading: str = ""
    heading_source: str = ""
    blocks: list[_MarkdownBlock] = field(default_factory=list)
    heading_tokens: set[str] = field(default_factory=set)
    tokens: set[str] = field(default_factory=set)


@dataclass
class _ParsedDoc:
    title: str
    sections: list[_MarkdownSection]
    title_tokens: set[str]
    heading_tokens: set[str]
    tokens: set[str]


def _parse_doc(content: str) -> _ParsedDoc:
    """Use Markdown source maps to retain whole blocks in their original syntax."""
    # Match the parser's newline normalization, without stripping indentation.
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    tokens = _MARKDOWN.parse(content)
    roots = [i for i, token in enumerate(tokens) if token.level == 0 and token.map]
    sections: list[_MarkdownSection] = []
    section = _MarkdownSection()
    title = ""
    cursor = 0

    def source(start: int, end: int) -> str:
        return "\n".join(lines[start:end]).strip("\n")

    def keep_gap(end: int) -> None:
        # Reference definitions have source lines but no block tokens.
        text = source(cursor, end)
        if text.strip():
            section.blocks.append(_MarkdownBlock(text, "reference"))

    for position, token_index in enumerate(roots):
        token = tokens[token_index]
        assert token.map is not None
        start, end = token.map
        if start < cursor:
            continue
        keep_gap(start)
        text = source(start, end)
        if token.type == "heading_open":
            if section.heading_source or section.blocks:
                sections.append(section)
            heading = tokens[token_index + 1].content
            section = _MarkdownSection(heading=heading, heading_source=text)
            if token.tag == "h1" and not title:
                title = heading
        else:
            next_index = (
                roots[position + 1] if position + 1 < len(roots) else len(tokens)
            )
            children = tokens[token_index:next_index]
            code = [
                child for child in children if child.type in {"fence", "code_block"}
            ]
            # CommonMark accepts EOF as a fence terminator. Our excerpts require
            # an explicit closing fence, including inside lists and blockquotes.
            complete = all(
                child.type != "fence"
                or (
                    child.map is not None
                    and child.map[1] - child.map[0] == child.content.count("\n") + 2
                    and (not child.content or child.content.endswith("\n"))
                )
                for child in code
            )
            block = _MarkdownBlock(text, token.type, bool(code), complete)
            if token.type in {
                "paragraph_open",
                "html_block",
            } and _MODEL_VIEWER_START.match(text):
                # The site's multiline custom element can span several CommonMark
                # blocks. Keep the entire widget attached to its code example.
                block.kind = "model_viewer"
                closing = next(
                    (
                        i + 1
                        for i in range(start, len(lines))
                        if _MODEL_VIEWER_END.search(lines[i])
                    ),
                    None,
                )
                block.complete = closing is not None
                if closing is not None:
                    end = max(end, closing)
                    block.text = source(start, end)
            if token.type == "table_open":
                rows = [
                    child.map
                    for child in children
                    if child.type == "tr_open"
                    and child.map
                    and child.map[0] > start + 1
                ]
                if rows:
                    block.table_header = source(start, rows[0][0])
                    block.table_rows = [source(*row) for row in rows]
            section.blocks.append(block)
        cursor = end
    keep_gap(len(lines))
    if section.heading_source or section.blocks:
        sections.append(section)
    for section in sections:
        section.heading_tokens = _lookup_tokens(section.heading)
        section.tokens = _lookup_tokens(
            "\n".join(
                [section.heading_source, *(block.text for block in section.blocks)]
            )
        )
    return _ParsedDoc(
        title,
        sections,
        _lookup_tokens(title),
        set().union(*(section.heading_tokens for section in sections)),
        _lookup_tokens(content),
    )


def _lookup_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _compact_block(block: _MarkdownBlock, budget: int, tokens: set[str]) -> str:
    """Keep whole blocks, or a table header with complete, relevant data rows."""
    if not block.complete:
        return ""
    if len(block.text) <= budget:
        return block.text
    if not block.table_rows:
        return ""
    header = block.table_header
    used = len(header)
    kept: dict[int, str] = {}
    rows = sorted(
        enumerate(block.table_rows),
        key=lambda item: (-len(tokens & _lookup_tokens(item[1])), item[0]),
    )
    for index, row in rows:
        if used + len(row) + 1 <= budget:
            kept[index] = row
            used += len(row) + 1
    if not kept:
        return ""
    return "\n".join([header, *(kept[index] for index in sorted(kept))])


def lookup_docs(query: str, max_results: int = 3, max_chars: int = 6000) -> dict:
    """Return ranked source sections within one Markdown-content budget."""
    if max_results < 1 or max_chars < 1:
        raise ValueError("max_results and max_chars must be positive")
    query = query.strip().lower()
    tokens = _lookup_tokens(query)
    if not tokens:
        return {"results": [], "message": "Provide a symbol or documentation topic."}

    candidates = []
    docs = KCLDocs.get()
    for path, content in docs.docs.items():
        parsed = docs.parsed_doc(path, content)
        # A qualified std symbol and its unqualified name both count as exact.
        symbol = parsed.title.strip("` ").lower()
        exact = query in {
            symbol,
            symbol.rsplit("::", 1)[-1],
            path.rsplit("/", 1)[-1].lower(),
        }
        title_matches = len(tokens & parsed.title_tokens)
        heading_matches = len(tokens & parsed.heading_tokens)
        body_matches = len(tokens & parsed.tokens)
        if exact or title_matches or heading_matches or body_matches:
            score = (exact, title_matches, heading_matches, body_matches)
            candidates.append((score, path, parsed))
    candidates.sort(key=lambda item: (tuple(-n for n in item[0]), item[1]))
    selected = candidates[:max_results]
    results = []
    remaining = max_chars
    for position, (_, path, parsed) in enumerate(selected):
        # Share the remaining budget so the first long page cannot hide alternatives.
        budget = remaining // (len(selected) - position)
        ranked = sorted(
            enumerate(parsed.sections),
            key=lambda item: (
                -len(tokens & item[1].heading_tokens),
                -len(tokens & item[1].tokens),
                item[0],
            ),
        )
        excerpts: dict[int, dict[str, str]] = {}
        used = 0
        omitted = False
        for index, section in ranked:
            # A heading is only useful when some of its content fits too. Reserve
            # its cost without charging the page for sections we cannot include.
            has_heading = bool(section.heading_source)
            kept = [section.heading_source] if has_heading else []
            section_used = len(kept[0]) if kept else 0
            body = section.blocks
            example_included = False
            for block_index, block in enumerate(body):
                # Generated pages can repeat their introductory paragraph. Keep
                # one copy without treating the duplicate as lost information.
                if (
                    index == 0
                    and block_index > 0
                    and block.text == body[block_index - 1].text
                    and block.kind == body[block_index - 1].kind == "paragraph_open"
                ):
                    continue
                # The viewer belongs to the preceding code example. Returning
                # just its HTML would make an omitted example look populated.
                if block.kind == "model_viewer" and not example_included:
                    omitted = True
                    continue
                separator_cost = 2 if kept else 0
                excerpt = _compact_block(
                    block, budget - used - section_used - separator_cost, tokens
                )
                if block.has_code:
                    example_included = bool(excerpt)
                if excerpt != block.text:
                    omitted = True
                if excerpt:
                    kept.append(excerpt)
                    section_used += len(excerpt) + separator_cost
            if len(kept) > int(has_heading):
                used += section_used
                excerpts[index] = {
                    "heading": section.heading,
                    "content": "\n\n".join(kept),
                }
            else:
                omitted = True
        remaining -= used
        results.append(
            {
                "path": path,
                "title": parsed.title,
                "sections": [excerpts[index] for index in sorted(excerpts)],
                "truncated": omitted,
            }
        )
    return {
        "results": results,
        "message": (
            "Use get_kcl_doc(path) for full pages when a result is truncated."
            if results
            else "No matching KCL documentation. Try a symbol or fewer keywords."
        ),
    }


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
