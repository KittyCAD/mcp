"""Live integration tests for KCL docs and public Aquarium samples.

These exercise real fetch + search/get paths against zoo.dev documentation
and the Zoo public projects API.

Marked ``live`` so they can be deselected locally with ``pytest -m 'not live'``
when offline.
"""

import json
from typing import Any

import pytest
import pytest_asyncio
from markdown_it import MarkdownIt
from mcp.types import CallToolResult, InputRequiredResult, TextContent

from zoo_mcp.kcl_docs import KCLDocs
from zoo_mcp.kcl_samples import KCLSamples
from zoo_mcp.server import mcp
from zoo_mcp.utils.data_retrieval_utils import ZOO_BASE_URL

pytestmark = [pytest.mark.live, pytest.mark.asyncio]


def _content_list(response: CallToolResult | InputRequiredResult) -> list[Any]:
    assert isinstance(response, CallToolResult)
    return list(response.content)


def _meta_result(response: CallToolResult | InputRequiredResult) -> Any:
    assert isinstance(response, CallToolResult)
    assert response.structured_content is not None
    return response.structured_content["result"]


@pytest_asyncio.fixture(scope="module")
async def real_docs():
    """Fetch the docs index from zoo.dev once for the module."""
    saved = KCLDocs._instance
    KCLDocs._instance = None
    try:
        await KCLDocs.initialize()
        assert KCLDocs._instance is not None
        assert len(KCLDocs._instance.docs) > 0, (
            f"{ZOO_BASE_URL} returned no docs — sitemap or markdown shape likely changed"
        )
        yield KCLDocs._instance
    finally:
        KCLDocs._instance = saved


@pytest_asyncio.fixture(scope="module")
async def real_samples():
    """Fetch the public Aquarium project index once for the module."""
    saved = KCLSamples._instance
    KCLSamples._instance = None
    try:
        await KCLSamples.initialize()
        assert KCLSamples._instance is not None, (
            "Zoo public projects API returned no samples"
        )
        assert len(KCLSamples._instance.manifest) > 0, (
            "Zoo public projects API returned no samples"
        )
        yield KCLSamples._instance
    finally:
        KCLSamples._instance = saved


# ---------------------------------------------------------------------------
# Docs: list / search / get
# ---------------------------------------------------------------------------


@pytest.mark.xdist_group(name="live-docs")
async def test_live_list_kcl_docs(real_docs):
    response = await mcp.call_tool("list_kcl_docs", arguments={})
    inner = _content_list(response)
    assert len(inner) == 1
    assert isinstance(inner[0], TextContent)
    result = json.loads(inner[0].text)

    assert set(result.keys()) == {
        "kcl-lang",
        "kcl-std-functions",
        "kcl-std-types",
        "kcl-std-consts",
        "kcl-std-modules",
    }
    # Each major category should have at least one entry on a healthy zoo.dev.
    for category in ("kcl-lang", "kcl-std-functions", "kcl-std-types"):
        assert len(result[category]) > 0, f"empty category: {category}"


@pytest.mark.xdist_group(name="live-docs")
async def test_live_search_kcl_docs_extrude(real_docs):
    response = await mcp.call_tool(
        "search_kcl_docs", arguments={"query": "extrude", "max_results": 5}
    )
    inner = _content_list(response)
    assert len(inner) > 0, "expected matches for 'extrude'"

    results = [json.loads(tc.text) for tc in inner]
    first = results[0]
    assert {"path", "title", "excerpt", "match_count"} <= set(first.keys())
    assert any("extrude" in r["path"].lower() for r in results)


@pytest.mark.xdist_group(name="live-docs")
async def test_live_get_kcl_doc_from_listing(real_docs):
    """Pick a path from the live listing and fetch it — avoids hard-coding."""
    listing = await mcp.call_tool("list_kcl_docs", arguments={})
    paths = json.loads(_content_list(listing)[0].text)["kcl-std-functions"]
    assert paths, "no kcl-std-functions paths to test against"

    target = paths[0]
    response = await mcp.call_tool("get_kcl_doc", arguments={"doc_path": target})
    inner = _content_list(response)
    assert len(inner) == 1
    text = inner[0].text
    assert isinstance(text, str)
    assert len(text) > 50, f"unexpectedly small doc payload for {target!r}"
    assert "Documentation not found" not in text


@pytest.mark.xdist_group(name="live-docs")
async def test_live_lookup_extrude_includes_argument_descriptions(real_docs):
    response = await mcp.call_tool(
        "lookup_kcl_docs", {"query": "extrude", "max_results": 1, "max_chars": 2000}
    )
    assert isinstance(response, CallToolResult)
    assert response.structured_content is not None
    page = response.structured_content["results"][0]
    assert page["path"].endswith("/std-sketch-extrude")
    sections = {section["heading"]: section["content"] for section in page["sections"]}
    assert "extrude(" in sections["extrude"]
    # Compare the complete argument description with today's published docs.
    length_row = next(
        line
        for line in real_docs.docs[page["path"]].splitlines()
        if line.startswith("| `length` |")
    )
    assert length_row in sections["Arguments"]
    assert sum(map(len, sections.values())) <= 2000
    assert page["truncated"]


@pytest.mark.xdist_group(name="live-docs")
async def test_live_lookup_all_kcl_functions(real_docs):
    """Exercise each published function through the MCP get and lookup tools."""
    markdown = MarkdownIt("commonmark").enable("table")
    for path in real_docs.index["kcl-std-functions"]:
        fetched = await mcp.call_tool("get_kcl_doc", {"doc_path": path})
        source = _meta_result(fetched)
        assert source == real_docs.docs[path], path
        assert _content_list(fetched)[0].text == source, path
        full_budget = max(6000, 4 * len(source))
        arguments = [
            {"query": path.rsplit("/", 1)[-1], "max_results": 1, "max_chars": budget}
            for budget in sorted({1, 15, 80, 400, 1000, 2000, 6000, 10000, full_budget})
        ]
        arguments.append({"query": source.splitlines()[0].removeprefix("# ")})
        for args in arguments:
            response = await mcp.call_tool("lookup_kcl_docs", args)
            assert isinstance(response, CallToolResult)
            payload = response.structured_content
            assert payload is not None, (path, args)
            assert json.loads(_content_list(response)[0].text) == payload, (path, args)
            pages = payload["results"]
            assert pages, (path, args)
            if args.get("max_results") == 1:
                assert len(pages) == 1 and pages[0]["path"] == path, (path, args)
            assert sum(
                len(section["content"])
                for page in pages
                for section in page["sections"]
            ) <= args.get("max_chars", 6000), (path, args)
            for page in pages:
                original = real_docs.docs[page["path"]]
                original_lines = original.splitlines()
                code_blocks = {
                    "\n".join(original_lines[token.map[0] : token.map[1]])
                    for token in markdown.parse(original)
                    if token.map and token.type in {"fence", "code_block"}
                }
                if args.get("max_chars", 6000) >= 2000:
                    assert page["sections"], (path, args, page)
                if not page["sections"]:
                    assert page["truncated"], (path, args, page)
                for section in page["sections"]:
                    heading, separator, body = section["content"].partition("\n\n")
                    assert separator and body.strip(), (path, args, section)
                    assert heading in original.splitlines(), (path, args, section)
                    assert heading.lstrip("# ").rstrip(" #") == section["heading"], (
                        path,
                        args,
                        section,
                    )
                    lines = section["content"].splitlines()
                    for token in markdown.parse(section["content"]):
                        if token.map is None:
                            continue
                        block = lines[token.map[0] : token.map[1]]
                        if token.type in {"fence", "code_block"}:
                            assert "\n".join(block) in code_blocks, (
                                path,
                                args,
                                section,
                            )
                        elif token.type == "table_open":
                            assert len(block) >= 3, (path, args, section)
                            assert "\n".join(block[:2]) in original, (
                                path,
                                args,
                                section,
                            )
                            assert all(row in original_lines for row in block[2:]), (
                                path,
                                args,
                                section,
                            )
            if args.get("max_chars") == full_budget:
                page = pages[0]
                assert not page["truncated"], (path, args)
                # All source lines must survive a sufficient budget. Comparing
                # sets allows duplicate intro paragraphs to collapse.
                expected = {line for line in source.splitlines() if line.strip()}
                actual = {
                    line
                    for section in page["sections"]
                    for line in section["content"].splitlines()
                    if line.strip()
                }
                assert actual == expected, (path, args, expected - actual)


# ---------------------------------------------------------------------------
# Samples: list / search / get
# ---------------------------------------------------------------------------


@pytest.mark.xdist_group(name="live-samples")
async def test_live_list_kcl_samples(real_samples):
    response = await mcp.call_tool("list_kcl_samples", arguments={})
    inner = _content_list(response)
    assert len(inner) > 10, "expected many samples in the live aquarium index"

    first = json.loads(inner[0].text)
    assert {"name", "title", "description", "multipleFiles"} <= set(first.keys())


@pytest.mark.xdist_group(name="live-samples")
async def test_live_search_kcl_samples_gear(real_samples):
    response = await mcp.call_tool(
        "search_kcl_samples", arguments={"query": "gear", "max_results": 5}
    )
    inner = _content_list(response)
    assert len(inner) > 0, "expected matches for 'gear'"

    results = [json.loads(tc.text) for tc in inner]
    first = results[0]
    assert {"name", "title", "description", "match_count", "excerpt"} <= set(
        first.keys()
    )
    blob = " ".join(r["title"] + r["description"] + r["name"] for r in results).lower()
    assert "gear" in blob


@pytest.mark.xdist_group(name="live-samples")
async def test_live_get_kcl_sample_from_listing(real_samples):
    """Fetch the first sample from the live listing and check its files."""
    listing = await mcp.call_tool("list_kcl_samples", arguments={})
    listed = [json.loads(tc.text) for tc in _content_list(listing)]
    assert listed, "no samples listed"

    name = listed[0]["name"]
    response = await mcp.call_tool("get_kcl_sample", arguments={"sample_name": name})
    result = _meta_result(response)

    assert isinstance(result, dict)
    assert result["name"] == name
    assert result["files"], f"no files returned for sample {name!r}"
    assert all(f["filename"].lower().endswith(".kcl") for f in result["files"])
    assert any(f["content"] for f in result["files"])
