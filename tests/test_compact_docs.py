import pytest
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult

from zoo_mcp.kcl_docs import KCLDocs, lookup_docs
from zoo_mcp.server import mcp


@pytest.fixture
def docs(monkeypatch):
    pages = {
        "docs/kcl-std/functions/extrude": "# extrude\n\nExtrude a sketch.\n\n## Signature\n\n```kcl\nextrude(sketch, length = 6mm)\n```\n\n## Examples\n\n```kcl\npart = extrude(profile, length = 6mm)\n```",
        "docs/kcl-std/functions/hole": "# hole\n\nCreate holes.\n\n## Examples\n\n```kcl\nhole(part, diameter = 4.5mm)\n```",
        "docs/kcl-std/functions/fillet": "# fillet\n\nRound edges.\n\n## Examples\n\n```kcl\nfillet(part, radius = 5mm)\n```",
        "docs/kcl-lang/tutorial": "# Tutorial\n\n" + "extrude hole fillet " * 200,
    }
    monkeypatch.setattr(KCLDocs, "_instance", KCLDocs(docs=pages))
    return pages


@pytest.mark.parametrize("query", ["extrude", "hole", "fillet"])
def test_exact_reference_beats_repeated_body_mentions(docs, query):
    result = lookup_docs(query)
    first = result["results"][0]
    assert first["path"].endswith("/" + query)
    assert any("```kcl" in section["content"] for section in first["sections"])
    assert not first["truncated"]


def test_multiword_query_and_deterministic_order(docs, monkeypatch):
    first = lookup_docs("extrude sketch")
    assert first["results"][0]["title"] == "extrude"
    monkeypatch.setattr(
        KCLDocs, "_instance", KCLDocs(docs=dict(reversed(list(docs.items()))))
    )
    assert lookup_docs("extrude sketch") == first


@pytest.mark.parametrize("budget", [1, 45, 120, 6000])
def test_one_budget_preserves_fences(docs, budget):
    result = lookup_docs("extrude hole fillet", max_chars=budget)
    contents = [s["content"] for page in result["results"] for s in page["sections"]]
    assert sum(map(len, contents)) <= budget
    assert all(content.count("```") % 2 == 0 for content in contents)
    if budget < 120:
        assert any(page["truncated"] for page in result["results"])


def test_fenced_headings_and_oversized_example(monkeypatch):
    content = (
        "# extrude\n\nShort description.\n\n~~~kcl\n# not a heading\n"
        + "x" * 500
        + "\n~~~\n\n## Notes\n\nextrude notes"
    )
    monkeypatch.setattr(
        KCLDocs, "_instance", KCLDocs(docs={"docs/kcl-lang/example": content})
    )
    page = lookup_docs("extrude", max_chars=150)["results"][0]
    assert page["truncated"]
    assert all("~~~" not in s["content"] for s in page["sections"])
    assert "not a heading" not in [s["heading"] for s in page["sections"]]
    assert any("description" in s["content"] for s in page["sections"])


def test_incomplete_source_fence_is_not_returned(monkeypatch):
    monkeypatch.setattr(
        KCLDocs,
        "_instance",
        KCLDocs(docs={"docs/kcl-lang/example": "# Example\n\n```kcl\nunfinished"}),
    )
    page = lookup_docs("example")["results"][0]
    assert page["truncated"]
    assert "```" not in str(page["sections"])


@pytest.mark.parametrize("query", ["", "  ", "??", "unfindable"])
def test_empty_and_unmatched_queries(docs, query):
    result = lookup_docs(query)
    assert not result["results"]
    assert result["message"]


@pytest.mark.asyncio
async def test_lookup_through_mcp(docs):
    result = await mcp.call_tool(
        "lookup_kcl_docs", {"query": "extrude", "max_results": 1}
    )
    assert isinstance(result, CallToolResult)
    assert result.structured_content == lookup_docs("extrude", max_results=1)
    with pytest.raises(ToolError, match="positive"):
        await mcp.call_tool("lookup_kcl_docs", {"query": "extrude", "max_chars": 0})
