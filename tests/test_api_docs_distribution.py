import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
DOCS_MCP_URL = "https://mcp.zoo.dev/api-docs/mcp"


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(
            _contains_key(child, key) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False


def test_docs_registry_metadata_is_remote_unauthenticated_and_separate():
    authenticated = json.loads((ROOT / "server.json").read_text())
    docs = json.loads((ROOT / "registry" / "zoo-api-docs" / "server.json").read_text())

    assert authenticated["name"] == "io.github.KittyCAD/zoo-mcp"
    assert docs["name"] == "io.github.KittyCAD/zoo-api-docs"
    assert docs["title"] == "Zoo API Docs"
    assert len(docs["description"]) <= 100
    assert docs["remotes"] == [
        {
            "type": "streamable-http",
            "url": "https://mcp.zoo.dev/api-docs/mcp",
        }
    ]
    assert "packages" not in docs
    assert not _contains_key(docs, "environmentVariables")
    assert not _contains_key(docs, "headers")


def test_container_runs_only_the_docs_entrypoint():
    dockerfile = (ROOT / "Dockerfile.api-docs").read_text()

    assert 'ENTRYPOINT ["zoo-api-docs"]' in dockerfile
    assert "zoo-mcp" not in dockerfile
    assert "ZOO_API_TOKEN" not in dockerfile
    assert 'io.modelcontextprotocol.server.name="io.github.KittyCAD/zoo-api-docs"' in (
        dockerfile
    )


def test_cross_client_bundle_uses_one_shared_skill_and_docs_server():
    codex_plugin = json.loads((ROOT / ".codex-plugin" / "plugin.json").read_text())
    codex_marketplace = json.loads(
        (ROOT / ".agents" / "plugins" / "marketplace.json").read_text()
    )
    claude_plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
    claude_marketplace = json.loads(
        (ROOT / ".claude-plugin" / "marketplace.json").read_text()
    )
    gemini_extension = json.loads((ROOT / "gemini-extension.json").read_text())
    skill = (ROOT / "skills" / "zoo-api-docs" / "SKILL.md").read_text()

    expected_http_server = {
        "zoo-api-docs": {
            "type": "http",
            "url": DOCS_MCP_URL,
        }
    }
    assert codex_plugin["skills"] == "./skills/"
    assert codex_plugin["mcpServers"] == expected_http_server
    assert codex_marketplace["plugins"][0]["name"] == "zoo-api-docs"
    assert claude_plugin["skills"] == "./skills/"
    assert claude_plugin["mcpServers"] == expected_http_server
    assert claude_marketplace["plugins"][0]["name"] == "zoo-api-docs"
    assert gemini_extension["mcpServers"] == {"zoo-api-docs": {"httpUrl": DOCS_MCP_URL}}
    assert skill.startswith("---\nname: zoo-api-docs\n")
    assert "Set `limit` to `3`" in skill
    assert "Fetch one schema at a time" in skill
    assert "mcp__" not in skill
