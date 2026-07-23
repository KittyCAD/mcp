# Zoo API Docs operations

The `zoo-api-docs` package profile and `ghcr.io/kittycad/zoo-api-docs` image contain the same checked-in immutable index.
Zoo infrastructure owns routing `https://mcp.zoo.dev/api-docs/mcp` and promoting image digests between environments.

## Source update contract

After `openapi/api.json` merges in `KittyCAD/api`, its `api-update-spec-for-repos.yml` workflow dispatches the full merge commit SHA to this repository.
Use this command only for manual recovery or backfill:

```bash
gh api repos/KittyCAD/mcp/dispatches \
  --method POST \
  -f event_type=zoo-api-docs-source-update \
  -f 'client_payload[api_commit]=0123456789abcdef0123456789abcdef01234567'
```

The token used by that workflow needs permission to dispatch events to `KittyCAD/mcp`.
Do not put the token in the payload or logs.

Documentation changes use the same event without `api_commit`.
That retains the OpenAPI commit already pinned in the bundled index and refreshes only the allowlisted guide URLs.
The builder records a deterministic digest of the exact fetched Markdown as the guide source revision.

The update workflow rejects non-commit OpenAPI revisions, redirects, unlisted hosts, missing guides, malformed OpenAPI operations, and unresolved test expectations.
It opens an index-update pull request only after the policy tests and retrieval evaluation pass.

## Image build

Merges that change the docs server or index publish this immutable candidate tag:

```text
ghcr.io/kittycad/zoo-api-docs:sha-<KittyCAD/mcp commit>
```

Release tags also publish `ghcr.io/kittycad/zoo-api-docs:<release tag>`.
Use the digest emitted by the image workflow for every deployment and promotion; do not deploy a mutable tag.

## Promotion

1. Deploy the candidate digest to staging.
2. Check `/healthz` and require `ready: true` plus the expected `index_revision`.
3. Run `python scripts/smoke_api_docs.py --url <staging-mcp-url>`.
4. Promote the same digest to production.
5. Run the health and MCP smoke tests against `https://mcp.zoo.dev/api-docs/mcp`.
6. Run the `publish-zoo-api-docs-registry` workflow with the deployed digest and index revision.

The Registry workflow refuses to publish before the production health and MCP checks succeed.
ChatGPT directory submission and other marketplace listings happen after the official Registry launch and are not release gates.

## Failure and rollback

Fetch, validation, evaluation, image-build, staging-smoke, or production-smoke failure must stop promotion and leave the current production digest in place.
Rollback means redeploying the previously recorded production digest, checking `/healthz`, and rerunning the MCP smoke script.

## Observability

Application logs contain only tool name, status, latency, result count, and index revision.
Do not log search text, tool arguments, returned documents, client headers, tokens, or other client-provided content at the proxy or application layer.
