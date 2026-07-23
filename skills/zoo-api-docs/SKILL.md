---
name: zoo-api-docs
description: Find Zoo API endpoints, operation IDs, parameters, authentication requirements, request and response schemas, official guides, and canonical curl examples through the read-only Zoo API Docs MCP server. Use when answering questions about the Zoo API reference or determining the exact HTTP method, path, or payload for a Zoo API operation.
---

# Zoo API Docs

Use the connected Zoo API Docs MCP server directly.
Tool names below are unqualified; use the Zoo API Docs server namespace required by the current client.

## Lookup workflow

1. Start with `search_zoo_api`.
   Use a query containing two to six distinctive terms.
   Set `limit` to `3` unless the user requests broader discovery.
   Set `tag` only when the exact tag is known.
2. Stop if the search result already answers the question.
   A result containing the method, path, title, and documentation URL is sufficient for a simple endpoint lookup.
3. Call `get_zoo_api_operation` only when exact parameters, authentication, content types, responses, or a curl example are required.
4. Call `get_zoo_api_schema` only for an exact schema name returned by a search or operation result.
   Fetch one schema at a time unless the user explicitly asks for a comparison.
5. Call `get_zoo_api_guide` only for tutorials or conceptual guidance.

## Context discipline

- Do not prefetch operation details, schemas, or guides.
- Do not use web search when the pinned Zoo documentation answers the question.
- Do not wrap MCP calls in shell, Python, JavaScript, or another code-execution tool when direct calls are available.
- Cite the returned `documentation_url`.
- If the first search fails, rephrase or narrow it once.
  Report that no documented match was found rather than inventing an endpoint or request field.
