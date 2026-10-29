from pathlib import Path
from unittest.mock import Mock

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult

from zoo_mcp import kcl_docs
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


@pytest.mark.parametrize(
    "query", ["std-math-legAngX", "std-math-legangx", "STD-MATH-LEGANGX"]
)
def test_mixed_case_path_beats_module_mentions(monkeypatch, query):
    path = "docs/kcl-std/functions/std-math-legAngX"
    monkeypatch.setattr(
        KCLDocs,
        "_instance",
        KCLDocs(
            docs={
                path: "# legAngX\n\nCompute the angle of a triangle's leg.",
                "docs/kcl-std/modules/std-math": "# std::math\n\n## legAngX\n\n"
                "See [legAngX](/docs/kcl-std/functions/std-math-legAngX).",
            }
        ),
    )
    assert lookup_docs(query, max_results=1)["results"][0]["path"] == path


@pytest.mark.parametrize("indent", [" ", "  ", "   "])
def test_indented_fence_keeps_source_indentation(monkeypatch, indent):
    example = f"{indent}```kcl\n{indent}x = 1\n{indent}  + 2\n{indent}```"
    content = f"# Example\n\n1. Evaluate the expression.\n{example}"
    monkeypatch.setattr(
        KCLDocs, "_instance", KCLDocs(docs={"docs/kcl-lang/example": content})
    )
    page = lookup_docs("example")["results"][0]
    assert example in page["sections"][0]["content"]
    assert not page["truncated"]


@pytest.mark.parametrize(
    "viewer",
    [
        '<model-viewer src="/model.glb">\n</model-viewer>',
        '<model-viewer\n  src="/model.glb"\n  camera-controls\n>\n</model-viewer>',
    ],
)
def test_omitted_example_does_not_leave_viewer_only_section(monkeypatch, viewer):
    content = "# Example\n\nIntro.\n\n## Examples\n\n```kcl\n" + "x" * 500
    content += f"\n```\n\n{viewer}"
    monkeypatch.setattr(
        KCLDocs, "_instance", KCLDocs(docs={"docs/kcl-lang/example": content})
    )
    compact = lookup_docs("example", max_chars=200)["results"][0]
    assert compact["sections"] == [
        {"heading": "Example", "content": "# Example\n\nIntro."}
    ]
    assert compact["truncated"]
    full = lookup_docs("example")["results"][0]
    assert viewer in full["sections"][1]["content"]
    assert not full["truncated"]


def test_viewer_tracks_its_own_example(monkeypatch):
    first = "```kcl\nx = 1\n```"
    second = "```kcl\n" + "x" * 500 + "\n```"
    viewer = '<model-viewer src="/second.glb">\n</model-viewer>'
    content = f"# Example\n\n{first}\n\n{second}\n\n{viewer}"
    monkeypatch.setattr(
        KCLDocs, "_instance", KCLDocs(docs={"docs/kcl-lang/example": content})
    )
    page = lookup_docs("example", max_chars=200)["results"][0]
    assert page["sections"][0]["content"] == "# Example\n\n" + first
    assert page["truncated"]


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


def test_setext_headings_keep_original_source(monkeypatch):
    intro = "extrude\n=======\n\nExtend a sketch."
    arguments = "Arguments\n---------\n\nSpecify the extrusion length."
    monkeypatch.setattr(
        KCLDocs,
        "_instance",
        KCLDocs(docs={"docs/kcl-lang/example": intro + "\n\n" + arguments}),
    )
    page = lookup_docs("extrude")["results"][0]
    assert page["title"] == "extrude"
    assert page["sections"] == [
        {"heading": "extrude", "content": intro},
        {"heading": "Arguments", "content": arguments},
    ]
    assert not page["truncated"]


@pytest.mark.parametrize(
    "container",
    [
        "- Follow these steps:\n\n  # Nested heading\n\n  ```kcl\n  x = 1\n  ```\n\n- Finish.",
        "> # Nested heading\n>\n> A quoted example.\n>\n> ```kcl\n> x = 1\n> ```",
        "    # Indented code, not a heading\n\n    x = 1",
    ],
)
def test_nested_blocks_are_atomic_and_do_not_create_sections(monkeypatch, container):
    content = "# Example\n\nIntro.\n\n" + container
    monkeypatch.setattr(
        KCLDocs, "_instance", KCLDocs(docs={"docs/kcl-lang/example": content})
    )
    full = lookup_docs("example")["results"][0]
    assert full["sections"] == [{"heading": "Example", "content": content}]
    assert not full["truncated"]
    compact = lookup_docs("example", max_chars=len(content) - 1)["results"][0]
    assert compact["sections"] == [
        {"heading": "Example", "content": "# Example\n\nIntro."}
    ]
    assert compact["truncated"]


@pytest.mark.parametrize("prefix", ["", "> ", "  "])
@pytest.mark.parametrize("ending", ["", "\n", "\n``` trailing text"])
def test_unclosed_fences_inside_containers_are_omitted(monkeypatch, prefix, ending):
    incomplete = "```kcl\nx = 1" + ending
    incomplete = "\n".join(prefix + line for line in incomplete.split("\n"))
    if prefix == "  ":
        incomplete = "- Example:\n\n" + incomplete
    content = "# Example\n\nIntro.\n\n" + incomplete
    monkeypatch.setattr(
        KCLDocs, "_instance", KCLDocs(docs={"docs/kcl-lang/example": content})
    )
    page = lookup_docs("example")["results"][0]
    assert page["sections"] == [
        {"heading": "Example", "content": "# Example\n\nIntro."}
    ]
    assert page["truncated"]


@pytest.mark.parametrize(
    "code",
    [
        "```kcl\n```",
        "~~~~kcl\n\n~~~~~",
        "```kcl\nx = 1\n````  ",
        "```kcl\nx = 1\n\n```",
        "```kcl\nx = 1\n```\n",
    ],
)
def test_closed_fences_remain_complete_with_crlf_source(monkeypatch, code):
    content = "# Example\n\n" + code
    monkeypatch.setattr(
        KCLDocs,
        "_instance",
        KCLDocs(docs={"docs/kcl-lang/example": content.replace("\n", "\r\n")}),
    )
    page = lookup_docs("example")["results"][0]
    assert page["sections"] == [{"heading": "Example", "content": content.rstrip("\n")}]
    assert not page["truncated"]


def test_reference_definitions_are_preserved(monkeypatch):
    content = (
        '[first]: /first "A reference"\n\n# Example\n\n'
        "Read [first] and [second].\n\n[second]: /second"
    )
    monkeypatch.setattr(
        KCLDocs, "_instance", KCLDocs(docs={"docs/kcl-lang/example": content})
    )
    page = lookup_docs("example")["results"][0]
    assert "\n\n".join(section["content"] for section in page["sections"]) == content
    assert not page["truncated"]


def test_parsed_pages_are_reused_and_refresh_when_source_changes(docs, monkeypatch):
    parse = Mock(wraps=kcl_docs._MARKDOWN.parse)
    monkeypatch.setattr(kcl_docs._MARKDOWN, "parse", parse)
    first = lookup_docs("extrude")
    assert parse.call_count == len(docs)
    assert lookup_docs("extrude") == first
    lookup_docs("hole", max_chars=2000)
    assert parse.call_count == len(docs)

    docs["docs/kcl-std/functions/extrude"] = "# extrude\n\nUpdated description."
    page = lookup_docs("extrude")["results"][0]
    assert "Updated description." in page["sections"][0]["content"]
    assert parse.call_count == len(docs) + 1

    monkeypatch.setattr(KCLDocs, "_instance", KCLDocs(docs=docs.copy()))
    lookup_docs("extrude")
    assert parse.call_count == 2 * len(docs) + 1


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


@pytest.mark.parametrize("outer_pipes", [True, False])
def test_partial_table_prioritizes_matching_rows(monkeypatch, outer_pipes):
    header = "| Name | Description |\n| :--- | ---: |"
    first = "| length | Extrusion length. |"
    target = "| twistAngle | Twist the sketch; accepts `a \\| b`. |"
    if not outer_pipes:
        header = "\n".join(line.strip("| ") for line in header.splitlines())
        first = first.strip("| ")
        target = target.strip("| ")
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
