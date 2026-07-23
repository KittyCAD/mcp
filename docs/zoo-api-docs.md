# Zoo API Docs

Zoo API Docs is a separate, public MCP server for finding Zoo endpoints, schemas, official guides, and placeholder-only curl examples.
It never calls the Zoo API and does not need a Zoo account or API token.

## Hosted installation

Use this Streamable HTTP endpoint:

```text
https://mcp.zoo.dev/api-docs/mcp
```

After the `io.github.KittyCAD/zoo-api-docs` listing reaches the official MCP Registry, compatible marketplaces can install it directly.
VS Code also supports this [one-click install link](vscode:mcp/install?%7B%22name%22%3A%22zoo-api-docs%22%2C%22type%22%3A%22http%22%2C%22url%22%3A%22https%3A%2F%2Fmcp.zoo.dev%2Fapi-docs%2Fmcp%22%7D).

### Codex

```bash
codex mcp add zoo-api-docs --url https://mcp.zoo.dev/api-docs/mcp
```

Equivalent `config.toml`:

```toml
[mcp_servers.zoo-api-docs]
url = "https://mcp.zoo.dev/api-docs/mcp"
```

### Claude Code

```bash
claude mcp add --transport http zoo-api-docs https://mcp.zoo.dev/api-docs/mcp
```

### Gemini CLI

```bash
gemini mcp add --transport http zoo-api-docs https://mcp.zoo.dev/api-docs/mcp
```

### VS Code

Add this to your user or workspace `mcp.json`:

```json
{
  "servers": {
    "zoo-api-docs": {
      "type": "http",
      "url": "https://mcp.zoo.dev/api-docs/mcp"
    }
  }
}
```

## Install with the lookup skill

The raw MCP installation commands above receive the server's concise lookup instructions.
The platform bundles below also install the shared `zoo-api-docs` Agent Skill for more detailed, on-demand guidance.

### Codex plugin

```bash
codex plugin marketplace add KittyCAD/mcp
codex plugin add zoo-api-docs@zoo
```

### Claude Code plugin

```bash
claude plugin marketplace add KittyCAD/mcp
claude plugin install zoo-api-docs@zoo
```

### Gemini CLI extension

```bash
gemini extensions install https://github.com/KittyCAD/mcp
```

The bundle uses one platform-neutral `skills/zoo-api-docs/SKILL.md`.
Only the installation manifests and Streamable HTTP configuration differ by client.

## Local fallback

Run the packaged stdio profile without credentials:

```bash
uvx --from zoo_mcp zoo-api-docs
```

For a local Codex configuration:

```bash
codex mcp add zoo-api-docs -- uvx --from zoo_mcp zoo-api-docs
```

## Pick the right Zoo server

| Server | Use it for | Authentication | Can execute Zoo operations |
| --- | --- | --- | --- |
| Zoo API Docs | Search public API operations, schemas, guides, and curl examples | None | No |
| Zoo MCP (`zoo-mcp`) | Execute CAD, KCL, conversion, snapshot, and organization tools | Zoo API token | Yes |

Do not configure a Zoo API token for Zoo API Docs.
Use the existing `zoo-mcp` profile when you intend to execute an operation.

## Available tools

- `search_zoo_api` searches operation IDs, paths, summaries, descriptions, tags, referenced schema names, and allowlisted guides.
- `get_zoo_api_operation` returns one exact public operation and its canonical request metadata.
- `get_zoo_api_schema` returns one exact schema without recursively expanding references.
- `get_zoo_api_guide` returns one allowlisted guide and caps content at 30,000 characters.

Every result identifies the pinned OpenAPI commit, guide revision, index revision, and canonical Zoo documentation URL.
