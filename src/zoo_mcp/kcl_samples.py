"""KCL Samples fetching and search.

Public Aquarium project metadata is indexed lazily from the Zoo API. Sample
names are public project IDs. KCL files are read on demand from ZIP downloads,
preserving project-relative paths. These endpoints do not need credentials;
the shared index always uses production public projects rather than a caller's
authenticated or alternate-host data.
"""

import asyncio
import io
import re
import stat
import zipfile
from dataclasses import dataclass, field
from typing import ClassVar

import httpx
from kittycad import AsyncKittyCAD
from kittycad.client import DEFAULT_BASE_URL

# pydantic requires typing_extensions.TypedDict on Python < 3.12.
from typing_extensions import TypedDict

from zoo_mcp import ctx, logger
from zoo_mcp.utils.data_retrieval_utils import extract_excerpt, is_safe_path_component

_PUBLIC_PROJECTS_URL = f"{DEFAULT_BASE_URL}/projects/public"
_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
_MAX_KCL_BYTES = 16 * 1024 * 1024

# Project files may have nested paths, spaces, and Unicode names, but must
# remain project-relative. Sample identifiers must be safe URL components.
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_SAFE_FILENAME_RE = re.compile(r"^[^\\\x00-\x1f\x7f:%]+\.[kK][cC][lL]$")


class SampleMetadata(TypedDict):
    """Metadata for a public Aquarium project."""

    title: str
    description: str
    # The public listing doesn't expose file counts. This hint is corrected
    # after get_sample_content downloads and counts the project's KCL files.
    multipleFiles: bool


class SampleFile(TypedDict):
    """A KCL source with its project-relative path."""

    filename: str
    content: str


class SampleData(TypedDict):
    """Complete data for a KCL sample including its KCL files."""

    name: str
    title: str
    description: str
    multipleFiles: bool
    files: list[SampleFile]


@dataclass
class KCLSamples:
    """Container for public Aquarium projects and cached KCL sources."""

    manifest: dict[str, SampleMetadata] = field(default_factory=dict)
    file_index: dict[str, dict[str, str]] = field(default_factory=dict)

    _instance: ClassVar["KCLSamples | None"] = None
    _init_lock: ClassVar[asyncio.Lock | None] = None

    @classmethod
    def get(cls) -> "KCLSamples":
        """Get the samples index, or an empty one if not initialized."""
        return cls._instance if cls._instance is not None else cls()

    @classmethod
    async def initialize(cls) -> None:
        """Initialize from public projects. Repeat and concurrent calls are safe."""
        if cls._instance is not None:
            return
        if cls._init_lock is None:
            cls._init_lock = asyncio.Lock()
        async with cls._init_lock:
            if cls._instance is None:
                result = await _fetch_index_from_api()
                if result.manifest:
                    cls._instance = result


def _parse_project_archive(content: bytes) -> dict[str, str]:
    """Read KCL sources from a ZIP in memory, without extracting any files."""
    files: dict[str, str] = {}
    total_bytes = 0
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for member in archive.infolist():
            filename = member.orig_filename
            if member.is_dir() or not filename.lower().endswith(".kcl"):
                continue
            if (
                filename.startswith("/")
                or not is_safe_path_component(filename, _SAFE_FILENAME_RE)
                or stat.S_ISLNK(member.external_attr >> 16)
                or filename in files
            ):
                raise ValueError(
                    f"Unsafe or duplicate KCL path in sample: {filename!r}"
                )
            total_bytes += member.file_size
            if total_bytes > _MAX_KCL_BYTES:
                raise ValueError("Sample KCL files exceed the size limit")
            files[filename] = archive.read(member).decode("utf-8")
    return files


async def _fetch_sample_files(
    client: httpx.AsyncClient,
    sample_name: str,
) -> dict[str, str]:
    """Download a public project's KCL files as a bounded ZIP response."""
    # The SDK's download method currently JSON-decodes archive responses.
    # Read the binary endpoint directly until it supports binary downloads.
    async with client.stream(
        "GET",
        f"{_PUBLIC_PROJECTS_URL}/{sample_name}/download",
        params={"format": "zip"},
        follow_redirects=False,
    ) as response:
        if response.status_code == 404:
            return {}
        response.raise_for_status()
        content = bytearray()
        async for chunk in response.aiter_bytes():
            content.extend(chunk)
            if len(content) > _MAX_ARCHIVE_BYTES:
                raise ValueError("Sample archive exceeds the size limit")
    return _parse_project_archive(bytes(content))


async def _fetch_index_from_api() -> KCLSamples:
    """Index every page of production public Aquarium project metadata."""
    samples = KCLSamples()
    logger.info("Fetching public Aquarium projects from %s...", DEFAULT_BASE_URL)

    async with AsyncKittyCAD(token="", timeout=30.0, verify_ssl=ctx) as client:
        async for project in client.projects.list_public_projects(limit=100):
            samples.manifest[str(project.id)] = SampleMetadata(
                title=project.title,
                description=project.description,
                multipleFiles=False,
            )

    logger.info("KCL samples index loaded with %d samples", len(samples.manifest))
    return samples


async def initialize_samples_index() -> None:
    """Initialize the samples index from public Aquarium projects."""
    await KCLSamples.initialize()


def list_available_samples() -> list[dict]:
    """Return a list of all available KCL samples with basic info.

    Returns a list of dictionaries, each containing:
    - name: The public project ID (used to retrieve the sample)
    - title: Human-readable title
    - description: Brief description of the sample
    - multipleFiles: Whether the sample contains multiple KCL files.

    Note on ``multipleFiles``: the public projects listing does not expose
    file counts, so this field is ``False`` for any sample whose project
    has not yet been downloaded. It becomes accurate only after
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
            - name: The public project ID (used to retrieve the sample)
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
        sample_name (str): The public project ID
            returned by list_kcl_samples() or search_kcl_samples()

    Returns:
        SampleData | None: A dictionary containing:
            - name: The public project ID
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
        # The listing doesn't tell us file counts; record the real value now
        # that we've downloaded the project.
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
