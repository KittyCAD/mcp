import httpx
import pytest

from zoo_mcp.kcl_docs import _SAFE_DOC_PATH_RE
from zoo_mcp.kcl_samples import _SAFE_FILENAME_RE, _SAFE_NAME_RE
from zoo_mcp.utils.data_retrieval_utils import (
    extract_excerpt,
    fetch_markdown,
    fetch_url,
    is_safe_path_component,
)

# ---------------------------------------------------------------------------
# is_safe_path_component
# ---------------------------------------------------------------------------


class TestIsSafePathComponent:
    """Tests for is_safe_path_component."""

    def test_valid_simple_name(self):
        assert is_safe_path_component("axial-fan", _SAFE_NAME_RE) is True

    def test_valid_name_with_underscore(self):
        assert is_safe_path_component("my_sample", _SAFE_NAME_RE) is True

    def test_valid_name_alphanumeric(self):
        assert is_safe_path_component("gear123", _SAFE_NAME_RE) is True

    def test_empty_string(self):
        assert is_safe_path_component("", _SAFE_NAME_RE) is False

    def test_dot_dot_traversal(self):
        assert is_safe_path_component("..", _SAFE_NAME_RE) is False

    def test_slash_in_name(self):
        assert is_safe_path_component("a/b", _SAFE_NAME_RE) is False

    def test_url_encoded_dot_dot(self):
        """URL-encoded '..' (%2e%2e) should be rejected."""
        assert is_safe_path_component("%2e%2e", _SAFE_NAME_RE) is False

    def test_url_encoded_slash(self):
        """URL-encoded '/' (%2f) should be rejected."""
        assert is_safe_path_component("%2f", _SAFE_NAME_RE) is False

    def test_double_encoded_dot_dot(self):
        """Double-encoded '..' (%252e%252e) should be rejected."""
        assert is_safe_path_component("%252e%252e", _SAFE_NAME_RE) is False

    def test_percent_sign(self):
        assert is_safe_path_component("a%b", _SAFE_NAME_RE) is False

    def test_space(self):
        assert is_safe_path_component("a b", _SAFE_NAME_RE) is False

    def test_valid_kcl_filename(self):
        assert is_safe_path_component("main.kcl", _SAFE_FILENAME_RE) is True

    def test_valid_kcl_filename_with_hyphens(self):
        assert is_safe_path_component("my-part.kcl", _SAFE_FILENAME_RE) is True

    def test_kcl_filename_with_path_traversal(self):
        assert is_safe_path_component("../main.kcl", _SAFE_FILENAME_RE) is False

    def test_kcl_filename_with_slash(self):
        assert is_safe_path_component("dir/main.kcl", _SAFE_FILENAME_RE) is False

    def test_kcl_filename_wrong_extension(self):
        assert is_safe_path_component("main.py", _SAFE_FILENAME_RE) is False

    def test_valid_doc_path(self):
        assert (
            is_safe_path_component("docs/kcl-lang/functions", _SAFE_DOC_PATH_RE) is True
        )

    def test_valid_kcl_std_doc_path(self):
        assert (
            is_safe_path_component(
                "docs/kcl-std/functions/std-sketch-extrude", _SAFE_DOC_PATH_RE
            )
            is True
        )

    def test_doc_path_with_md_extension_rejected(self):
        """The new regex no longer matches the legacy ``.md`` suffix."""
        assert (
            is_safe_path_component("docs/kcl-lang/functions.md", _SAFE_DOC_PATH_RE)
            is False
        )

    def test_doc_path_with_encoded_traversal(self):
        """The regex blocks '%' so encoded traversal is rejected."""
        assert (
            is_safe_path_component(
                "docs/kcl-lang/%2e%2e%2f%2e%2e%2fREADME", _SAFE_DOC_PATH_RE
            )
            is False
        )

    def test_doc_path_with_literal_dot_dot(self):
        assert is_safe_path_component("docs/../etc/passwd", _SAFE_DOC_PATH_RE) is False


# ---------------------------------------------------------------------------
# extract_excerpt
# ---------------------------------------------------------------------------


class TestExtractExcerpt:
    """Tests for extract_excerpt."""

    def test_match_found(self):
        content = "The quick brown fox jumps over the lazy dog."
        excerpt = extract_excerpt(content, "fox", context_chars=20)
        assert "fox" in excerpt

    def test_no_match_returns_beginning(self):
        content = "Some content without the search term."
        excerpt = extract_excerpt(content, "nonexistent", context_chars=50)
        assert excerpt.startswith("Some content")
        assert excerpt.endswith("...")

    def test_case_insensitive(self):
        content = "Hello World"
        excerpt = extract_excerpt(content, "hello", context_chars=50)
        assert "Hello" in excerpt

    def test_short_content(self):
        content = "Short."
        excerpt = extract_excerpt(content, "Short", context_chars=200)
        assert "Short" in excerpt

    def test_match_at_beginning(self):
        content = "keyword is at the start of this long text."
        excerpt = extract_excerpt(content, "keyword", context_chars=20)
        assert "keyword" in excerpt
        assert not excerpt.startswith("...")

    def test_match_at_end(self):
        content = "This is a long text that ends with keyword"
        excerpt = extract_excerpt(content, "keyword", context_chars=20)
        assert "keyword" in excerpt
        assert not excerpt.endswith("...")

    def test_ellipsis_when_truncated(self):
        content = "word " * 20 + "keyword" + " word" * 20
        excerpt = extract_excerpt(content, "keyword", context_chars=20)
        assert "keyword" in excerpt
        assert excerpt.startswith("...")
        assert excerpt.endswith("...")


# ---------------------------------------------------------------------------
# fetch_url / fetch_markdown
# ---------------------------------------------------------------------------


class TestFetchUrl:
    """Tests for fetch_url and fetch_markdown."""

    @pytest.mark.asyncio
    async def test_successful_fetch(self, httpx_mock):
        httpx_mock.add_response(
            url="https://example.com/file.txt",
            text="file content",
        )
        async with httpx.AsyncClient() as client:
            result = await fetch_url(client, "https://example.com/file.txt", "file.txt")
        assert result == "file content"

    @pytest.mark.asyncio
    async def test_redirect_rejected(self, httpx_mock):
        httpx_mock.add_response(
            url="https://example.com/file.txt",
            status_code=302,
            headers={"location": "https://evil.com/payload"},
        )
        async with httpx.AsyncClient() as client:
            result = await fetch_url(client, "https://example.com/file.txt", "file.txt")
        assert result is None

    @pytest.mark.asyncio
    async def test_http_error_returns_none(self, httpx_mock):
        httpx_mock.add_response(
            url="https://example.com/file.txt",
            status_code=404,
        )
        async with httpx.AsyncClient() as client:
            result = await fetch_url(client, "https://example.com/file.txt", "file.txt")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_markdown_sends_accept_header(self, httpx_mock):
        httpx_mock.add_response(
            url="https://example.com/page",
            text="# Heading",
            match_headers={"Accept": "text/markdown"},
        )
        async with httpx.AsyncClient() as client:
            result = await fetch_markdown(client, "https://example.com/page", "page")
        assert result == "# Heading"
