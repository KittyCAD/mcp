from zoo_mcp import kcl_docs
from zoo_mcp.utils.data_retrieval_utils import extract_excerpt


def test_extract_title():
    """Test title extraction from Markdown content."""
    content = "# My Title\n\nSome content here."
    assert kcl_docs._extract_title(content) == "My Title"

    # Test with no title
    content_no_title = "Some content without a heading."
    assert kcl_docs._extract_title(content_no_title) == ""


def test_extract_excerpt():
    """Test excerpt extraction with context."""
    content = "This is some text before. The keyword appears here in the middle. And this is text after."
    excerpt = extract_excerpt(content, "keyword", context_chars=40)

    assert "keyword" in excerpt
    assert len(excerpt) < len(content) + 10  # Account for ellipsis


def test_categorize_doc_path():
    """Test documentation path categorization."""
    assert kcl_docs._categorize_doc_path("docs/kcl-lang/functions.md") == "kcl-lang"
    assert (
        kcl_docs._categorize_doc_path("docs/kcl-std/functions/extrude.md")
        == "kcl-std-functions"
    )
    assert (
        kcl_docs._categorize_doc_path("docs/kcl-std/types/Sketch.md") == "kcl-std-types"
    )
    assert (
        kcl_docs._categorize_doc_path("docs/kcl-std/consts/PI.md") == "kcl-std-consts"
    )
    assert (
        kcl_docs._categorize_doc_path("docs/kcl-std/modules/math.md")
        == "kcl-std-modules"
    )
    assert kcl_docs._categorize_doc_path("docs/other/file.md") is None
