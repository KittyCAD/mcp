"""Shared utilities for fetching KCL docs and samples from zoo.dev.

Provides path validation, URL fetching, and text extraction helpers used by
both kcl_docs and kcl_samples modules.
"""

import posixpath
import re
from urllib.parse import unquote

import httpx

from zoo_mcp import logger

ZOO_BASE_URL = "https://zoo.dev"
ZOO_SITEMAP_URL = f"{ZOO_BASE_URL}/sitemap.xml"


def is_safe_path_component(value: str, pattern: re.Pattern[str]) -> bool:
    """Validate that a path component is safe."""
    if not value:
        return False

    if not pattern.match(value):
        return False

    decoded = unquote(value)
    if not pattern.match(decoded):
        return False

    normalized = posixpath.normpath(decoded)
    if normalized != decoded or normalized.startswith(".."):
        return False

    return True


async def fetch_url(
    client: httpx.AsyncClient,
    url: str,
    label: str,
    *,
    accept: str | None = None,
) -> str | None:
    """Fetch a URL and return the response text.

    Uses follow_redirects=False to prevent silent resolution of paths
    outside the intended endpoint.

    Args:
        client: The HTTP client to use.
        url: The full URL to fetch.
        label: A human-readable label for log messages.
        accept: Optional value for the Accept header. zoo.dev serves
            markdown when ``Accept: text/markdown`` is sent.

    Returns:
        The response body as a string, or None if the fetch failed.
    """
    headers = {"Accept": accept} if accept else None
    try:
        response = await client.get(url, follow_redirects=False, headers=headers)
        if response.is_redirect:
            logger.warning(
                f"Rejected redirect for {label}: {response.headers.get('location')}"
            )
            return None
        response.raise_for_status()
        return response.text
    except httpx.HTTPError as e:
        logger.warning(f"Failed to fetch {label}: {e}")
        return None


async def fetch_markdown(client: httpx.AsyncClient, url: str, label: str) -> str | None:
    """Fetch a zoo.dev page as markdown."""
    return await fetch_url(client, url, label, accept="text/markdown")


def extract_excerpt(content: str, query: str, context_chars: int = 200) -> str:
    """Extract an excerpt around the first match of query in content."""
    query_lower = query.lower()
    content_lower = content.lower()

    pos = content_lower.find(query_lower)
    if pos == -1:
        return content[:context_chars].strip() + "..."

    start = max(0, pos - context_chars // 2)
    end = min(len(content), pos + len(query) + context_chars // 2)

    if start > 0:
        while start > 0 and content[start - 1] not in " \n\t":
            start -= 1

    if end < len(content):
        while end < len(content) and content[end] not in " \n\t":
            end += 1

    excerpt = content[start:end].strip()

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(content) else ""

    return f"{prefix}{excerpt}{suffix}"
