# Authenticated Streamable HTTP

[MCP #271](https://github.com/KittyCAD/mcp/pull/271) and
[API #4671](https://github.com/KittyCAD/api/pull/4671) form the first testable slice.
The API provides OAuth, consent, short-lived delegated Zoo credentials, public
`/mcp` routing, and the preview deployment. The Python SDK provides MCP discovery,
authentication challenges, request handling, and streaming.

The ordinary stdio and trusted HTTP modes retain the complete local tool surface.
Authenticated HTTP exposes inline KCL execution, formatting, linting, mock
execution, physical properties, bounding boxes, sketch constraints, documentation,
and organization dataset tools. Each credential-bearing call runs in a fresh
process and temporary directory; credentials are never shared through the frontend
process environment. Linux Landlock confines file access. No job queue, artifact
store, persistent worker registry, or custom MCP protocol implementation is needed.

## Deployment

Use the immutable `ghcr.io/kittycad/zoo-mcp-http` image produced by the
Authenticated MCP HTTP workflow. The companion API preview pins its exact digest.
The image requires Linux amd64 with Landlock ABI 3 or later. The workflow checks
the filesystem boundary on native Linux before publishing. Authenticated workers
fail closed if confinement is unavailable; ordinary stdio and trusted HTTP remain
usable on other platforms.

The API and MCP container share `ZOO_MCP_RESOURCE` (the exact public HTTPS `/mcp`
URL) and `ZOO_MCP_SERVICE_SECRET` (at least 32 characters). The API generates and
validates the OAuth credentials. The MCP process carries no shared Zoo API key.
The container command is:

```sh
zoo-mcp --transport streamable-http --zoo-oauth --host 0.0.0.0 --port 8080
```

Expose the API's HTTPS endpoint, with the MCP container reachable only by the API.
Requests are limited to 1 MiB, four concurrent tool calls, and a 295-second
deadline. Calls finish directly in their original MCP response.

## Connect and exercise the first slice

After API #4671's preview becomes ready, use
`https://api-pr-4671.dev.zoo.dev/mcp` with OAuth in the ChatGPT desktop app.
In Settings → MCP servers, add a Streamable HTTP server, save/restart, and select
Authenticate. In clients that expose developer-mode plugins instead, add the same
URL in the plugin's MCP connection settings and choose OAuth. Sign in to Zoo and
approve the requested scopes. The API must allow the exact OpenAI client metadata
URL shown by the client; its preview also permits the narrowly scoped desktop
Codex metadata pattern.

These steps follow the official [desktop MCP setup](https://learn.chatgpt.com/docs/extend/mcp)
and [ChatGPT plugin connection guide](https://developers.openai.com/plugins/deploy/connect-chatgpt).
Workspace permissions and client availability still determine which interface is
available to your account.

Use these prompts one at a time:

1. **Discovery and local tool execution:** “Use Zoo's `format_kcl` tool to format
   the inline code `x=1`. Return the tool result.”
2. **Documentation:** “Use Zoo's `search_kcl_docs` tool to find documentation for
   `extrude`. Summarize the matching result.”
3. **Authenticated account access:** “Use Zoo's `list_org_datasets` tool. List the
   datasets available to my account; do not run a dataset search yet.”
4. **Modeling:** “Use Zoo's KCL samples to obtain a small, self-contained solid,
   then call `execute_kcl` with inline `kcl_code`. Report the mock preflight and
   real execution outcomes.”
5. **Measurement:** “Use `calculate_kcl_physical_properties` with that same inline
   KCL and report its volume.”

File uploads/downloads, persistent modeling sessions, exports, and the browser
workspace are not advertised in this first authenticated catalog. They are later
stack additions, so STEP uploads and interactive previews cannot be used as
acceptance criteria for the first slice.

## Remaining stack

- MCP #280, based on #271: deferred persistent scenes, artifact-backed tools,
  and optional durable jobs; paired with API #4730 and #4731.
- MCP #281, based on #280: deferred project management.
- MCP #282, based on #281: deferred browser workspace; paired with API #4733.
- MCP #283, based on #282: deferred distribution/catalog materials.

These drafts remain stacked on their immediate predecessor. Their deployment
images and contracts are separate from the first slice's `zoo-mcp-http` image.
