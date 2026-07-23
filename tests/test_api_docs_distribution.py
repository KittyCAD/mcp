import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


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
