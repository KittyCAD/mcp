"""Live integration tests against zoo.dev for the KCL docs/samples pipeline.

These exercise the real fetch + parse + search/get path end-to-end so that
breakage in zoo.dev's markdown shape (or our parsers) is caught.

Marked ``live`` so they can be deselected locally with ``pytest -m 'not live'``
when offline.
"""

import json
from collections.abc import Sequence
from typing import Any, cast

import pytest
import pytest_asyncio
from mcp.types import TextContent

from zoo_mcp.kcl_docs import KCLDocs
from zoo_mcp.kcl_samples import KCLSamples
from zoo_mcp.server import mcp

pytestmark = [pytest.mark.live, pytest.mark.asyncio]


def _content_list(response: Sequence[Any] | dict[str, Any]) -> list[Any]:
    assert isinstance(response, Sequence)
    content = response[0]
    assert isinstance(content, list)
    return cast(list[Any], content)


def _meta_result(response: Sequence[Any] | dict[str, Any]) -> Any:
    assert isinstance(response, Sequence)
    meta = response[1]
    assert isinstance(meta, dict)
    return cast(dict[str, Any], meta)["result"]


@pytest_asyncio.fixture(scope="module")
async def real_docs():
    """Fetch the docs index from zoo.dev once for the module."""
    saved = KCLDocs._instance
    KCLDocs._instance = None
    try:
        await KCLDocs.initialize()
        assert KCLDocs._instance is not None
        assert len(KCLDocs._instance.docs) > 0, (
            "zoo.dev returned no docs — sitemap or markdown shape likely changed"
        )
        yield KCLDocs._instance
    finally:
        KCLDocs._instance = saved


@pytest_asyncio.fixture(scope="module")
async def real_samples():
    """Fetch the samples index from zoo.dev once for the module."""
    saved = KCLSamples._instance
    KCLSamples._instance = None
    try:
        await KCLSamples.initialize()
        assert KCLSamples._instance is not None
        assert len(KCLSamples._instance.manifest) > 0, (
            "zoo.dev returned no samples — /aquarium markdown shape likely changed"
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
    main = next((f for f in result["files"] if f["filename"] == "main.kcl"), None)
    assert main is not None, f"sample {name!r} is missing main.kcl"
    assert len(main["content"]) > 0
