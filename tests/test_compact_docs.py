from pathlib import Path

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


@pytest.fixture
def extrude_doc(monkeypatch):
    # zoo.dev's actual layout: an inline signature, duplicate intro, and a long
    # argument table. The fixture ends after the first complete example.
    content = (Path(__file__).parent / "data" / "extrude-doc.md").read_text()
    monkeypatch.setattr(
        KCLDocs,
        "_instance",
        KCLDocs(docs={"docs/kcl-std/functions/std-sketch-extrude": content}),
    )
    return content


def test_extrude_includes_argument_descriptions_within_compact_budget(extrude_doc):
    page = lookup_docs("extrude", max_results=1, max_chars=2000)["results"][0]
    sections = {s["heading"]: s["content"] for s in page["sections"]}
    assert "```kcl\nextrude(" in sections["extrude"]
    assert sections["extrude"].count("Extend a 2-dimensional sketch") == 1
    assert "| Name | Type | Description | Required |" in sections["Arguments"]
    assert "| `length` |" in sections["Arguments"]
    assert "Incompatible with `to`." in sections["Arguments"]
    assert sum(map(len, sections.values())) <= 2000
    assert page["truncated"]


@pytest.mark.parametrize("budget", [1, 15, 80, 400, 1000, 2000, 6000, 10000])
def test_extrude_excerpts_have_bodies_and_complete_source_rows(extrude_doc, budget):
    page = lookup_docs("extrude", max_chars=budget)["results"][0]
    assert sum(len(s["content"]) for s in page["sections"]) <= budget
    for section in page["sections"]:
        content = section["content"]
        assert len(content.split("\n\n")) > 1
        assert content.count("```") % 2 == 0
        if section["heading"] == "Arguments":
            table = content.split("\n\n", 1)[1].splitlines()
            assert len(table) >= 3
            assert table[0] == "| Name | Type | Description | Required |"
            assert table[1] == "|----------|------|-------------|----------|"
            assert all(row in extrude_doc.splitlines() for row in table[2:])
    assert page["truncated"] == (budget < 10000)


def test_partial_table_prioritizes_matching_rows(monkeypatch):
    header = "| Name | Description |\n| --- | --- |"
    first = "| length | Extrusion length. |"
    target = "| twistAngle | Twist the sketch; accepts `a \\| b`. |"
    content = f"# extrude\n\n## Arguments\n\n{header}\n{first}\n{target}"
    monkeypatch.setattr(
        KCLDocs, "_instance", KCLDocs(docs={"docs/kcl-lang/example": content})
    )
    expected = f"## Arguments\n\n{header}\n{target}"
    page = lookup_docs("twistAngle", max_chars=len(expected))["results"][0]
    assert page["sections"] == [{"heading": "Arguments", "content": expected}]
    assert page["truncated"]


@pytest.mark.parametrize(
    "body",
    [
        "x" * 200,
        "```kcl\n" + "x" * 200 + "\n```",
        "| Name | Description |\n| --- | --- |\n| length | " + "x" * 200 + " |",
    ],
)
def test_omitted_section_does_not_consume_heading_budget(monkeypatch, body):
    notes = "## Notes\n\nUseful."
    content = f"# extrude\n\n## Arguments\n\n{body}\n\n{notes}"
    monkeypatch.setattr(
        KCLDocs, "_instance", KCLDocs(docs={"docs/kcl-lang/example": content})
    )
    page = lookup_docs("arguments", max_chars=len(notes))["results"][0]
    assert page["sections"] == [{"heading": "Notes", "content": notes}]
    assert page["truncated"]


def test_table_is_omitted_when_only_its_header_fits(monkeypatch):
    header = "## Arguments\n\n| Name | Description |\n| --- | --- |"
    content = header + "\n| length | " + "x" * 200 + " |"
    monkeypatch.setattr(
        KCLDocs, "_instance", KCLDocs(docs={"docs/kcl-lang/example": content})
    )
    page = lookup_docs("arguments", max_chars=len(header))["results"][0]
    assert not page["sections"]
    assert page["truncated"]


def test_only_adjacent_intro_prose_is_deduplicated(monkeypatch):
    code = "```kcl\nexample = 1\n```"
    content = f"# Example\n\nA.\n\nA.\n\nB.\n\nA.\n\n{code}\n\n{code}"
    monkeypatch.setattr(
        KCLDocs, "_instance", KCLDocs(docs={"docs/kcl-lang/example": content})
    )
    page = lookup_docs("example")["results"][0]
    assert page["sections"][0]["content"] == content.replace("A.\n\nA.", "A.", 1)
    assert not page["truncated"]


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
