from zoo_mcp import kcl_samples
from zoo_mcp.utils.data_retrieval_utils import extract_excerpt


def test_parse_index_markdown_basic():
    """Index lines are split into name/title/description, dropping categories."""
    markdown = (
        "# CAD Samples Gallery\n"
        "\n"
        "## Samples\n"
        "\n"
        "- [Angle Gauge](/aquarium/angle-gauge)"
        " - Simple angle gauge function to generate a gauge given an angle"
        " (Construction)\n"
        "- [Ball Bearing](/aquarium/ball-bearing)"
        " - A rolling-element bearing (Manufacturing, Automotive)\n"
    )

    parsed = kcl_samples._parse_index_markdown(markdown)

    assert set(parsed.keys()) == {"angle-gauge", "ball-bearing"}
    assert parsed["angle-gauge"]["title"] == "Angle Gauge"
    assert parsed["angle-gauge"]["description"] == (
        "Simple angle gauge function to generate a gauge given an angle"
    )
    assert parsed["angle-gauge"]["multipleFiles"] is False
    assert parsed["ball-bearing"]["description"] == "A rolling-element bearing"


def test_parse_index_markdown_handles_missing_description():
    """Entries without a ``- description`` chunk are still captured."""
    markdown = "- [Bare Slug](/aquarium/bare-slug)\n"

    parsed = kcl_samples._parse_index_markdown(markdown)

    assert parsed["bare-slug"]["title"] == "Bare Slug"
    assert parsed["bare-slug"]["description"] == ""


def test_parse_index_markdown_strips_acronym_categories():
    """All-caps acronym categories like ``(API)`` / ``(CAD)`` are stripped."""
    markdown = (
        "- [Widget](/aquarium/widget) - A widget (API, CAD)\n"
        "- [Gizmo](/aquarium/gizmo) - A gizmo (CNC)\n"
    )

    parsed = kcl_samples._parse_index_markdown(markdown)

    assert parsed["widget"]["description"] == "A widget"
    assert parsed["gizmo"]["description"] == "A gizmo"


def test_parse_index_markdown_keeps_url_parens_in_description():
    """Trailing ``(...)`` is only stripped when it looks like categories."""
    markdown = (
        "- [Bench](/aquarium/bench)"
        " - This is a slight remix of original 3D Boaty (https://example.com)."
        " Park bench shape. (Maker)\n"
    )

    parsed = kcl_samples._parse_index_markdown(markdown)

    assert "bench" in parsed
    desc = parsed["bench"]["description"]
    assert "(https://example.com)" in desc
    assert desc.endswith("Park bench shape.")


def test_parse_aquarium_markdown_extracts_files():
    """Per-sample markdown yields filename -> content for each ``### *.kcl`` block."""
    markdown = (
        "# Coilover\n\n"
        "## Files\n\n"
        "### main.kcl\n\n"
        "```kcl\n"
        'import "part.kcl" as part\n'
        "```\n\n"
        "### part.kcl\n\n"
        "```kcl\n"
        "x = 1\n"
        "```\n"
    )

    files = kcl_samples._parse_aquarium_markdown(markdown)

    assert files == {"main.kcl": 'import "part.kcl" as part', "part.kcl": "x = 1"}


def test_extract_excerpt():
    """Test excerpt extraction with context."""
    content = "This is some text before. The keyword appears here in the middle. And this is text after."
    excerpt = extract_excerpt(content, "keyword", context_chars=40)

    assert "keyword" in excerpt
    assert len(excerpt) < len(content) + 10  # Account for ellipsis


def test_extract_excerpt_no_match():
    """Test excerpt extraction when query is not found."""
    content = "Some content without the search term."
    excerpt = extract_excerpt(content, "nonexistent", context_chars=50)

    # Should return beginning of content as fallback
    assert excerpt.startswith("Some content")


def test_list_available_samples_returns_list():
    """Test that list_available_samples returns a list."""
    # Without initialization, should return empty list
    result = kcl_samples.list_available_samples()
    assert isinstance(result, list)


def test_search_samples_empty_query():
    """Test search with empty query returns error."""
    result = kcl_samples.search_samples("")
    assert len(result) == 1
    assert "error" in result[0]

    result = kcl_samples.search_samples("   ")
    assert len(result) == 1
    assert "error" in result[0]


def test_search_samples_returns_list():
    """Test that search_samples returns a list."""
    result = kcl_samples.search_samples("gear")
    assert isinstance(result, list)
