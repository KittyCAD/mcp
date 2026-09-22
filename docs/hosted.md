# Hosted tools foundation

`zoo-mcp-hosted` serves the same tool catalog as local stdio over authenticated
Streamable HTTP. The API owns OAuth, API-key exchange, consent, delegated
credentials, and grant-owned storage. This private service requires the
authentication and artifact/session contracts in the API stack rooted at #4671.

Tools return direct results by default. The SDK sends SSE responses and periodic
keepalives; the public API transport slice must stream them without ingress
buffering. Calls have a 300-second wall-clock deadline. A dropped connection does
not cancel work, and an interrupted call has no durable recovery or automatic
retry. Inspect the scene or saved files before retrying an uncertain operation.

Hosted file parameters use `artifact_id`, `project_artifact_id`, or
`artifact_graph_id` in place of host paths. `create_upload` reserves a file and
returns a short-lived PUT URL. Upload exactly the declared number of bytes before
using the artifact. `write_kcl_project` saves editable source and dependencies as
a ZIP; `get_artifact` refreshes download links. Formatting and lint fixes return
the updated project as `source`, leaving the original input unchanged. Exports,
snapshots, graphs, and other file outputs become grant-owned artifacts.

Each modeling session has its own subprocess, temporary directory, and renewed
delegated credential. Calls to that session serialize in its worker. Session
records identify the owning pod, allowing later requests on another pod to route
there without sharing credentials between grants. Grant revocation, idle expiry,
deadlines, worker limits, and shutdown bound worker lifetime. Expired scenes must
be restored from saved source.

Run `uv sync --frozen`, then set:

```sh
ZOO_MCP_RESOURCE=https://your-api.example/mcp
ZOO_API_URL=https://your-api.example
ZOO_MCP_SERVICE_SECRET=<same private service secret as the API>
ZOO_MCP_CAPABILITY_SECRET=<private transfer signing secret>
ZOO_MCP_NODE_URL=http://<pod-ip>:8080
ZOO_MCP_ALLOWED_ORIGINS=https://chatgpt.com,https://claude.ai
```

Secrets must contain at least 32 characters. Use deployment-derived URLs.
`ZOO_MCP_MAX_WORKERS` defaults to 16. The service runs on port 8080 and exposes
`/healthz` and `/readyz`. Deploy it privately with a read-only filesystem, a bounded
writable `/tmp`, no service-account token, dropped capabilities, and controlled
egress. Grant storage permits 1,000 artifacts / 10 GiB per grant and expires
temporary files after seven days. Uploads are limited to 256 MiB; expanded
projects to 512 MiB and 1,000 files.

The minimal `Dockerfile.hosted` requires native Linux with Landlock ABI 3 or newer.
Startup fails closed when confinement is unavailable. The Hosted MCP workflow
publishes a commit-tagged image only after its Linux sandbox check succeeds.
The companion API preview must pin the published image digest. For local
development only, `ZOO_MCP_UNSAFE_LOCAL_DEV=true` permits a host without Landlock.

The remaining MCP stack adds optional background execution, project management,
the browser workspace with the existing viewer, and distribution materials, in
that order. API public transport and automatic previews complete the initial
hosted milestone; project/workspace authorization follows separately. Local
stdio behavior, dependency versions, and main's KCL execution improvements remain
unchanged. The GLB conversion fix is a separate API change.

Original review threads remain in MCP #271 and API #4671. The original heads are
preserved under `backup/hosted-mcp-20260922` in each repository. Child PRs target
their immediate parent and should be retargeted/restacked after parent merges.
Production enablement and catalog publication are separate rollout steps.

## Optional durable execution

Eligible tools additionally accept `execution_mode: "background"`. Only this
mode requires an `idempotency_key` (1–128 characters). Reusing the key with the
same tool and arguments returns the same job; a different request conflicts.
Omitting the mode, or selecting `"direct"`, preserves the original result shape.

Background calls return `job_id` and status. `get_job` reads persisted outcomes
and refreshes artifact links; `cancel_job` requests cancellation on the owning
worker. Completed side effects cannot be undone. A worker lost before recording
an outcome becomes `interrupted` after its deadline and is never replayed
automatically. Durable records retain outcomes, not credentials or running
processes. There are at most eight active jobs per grant.

## Project management

Project tools list, open, create, update, publish, delete, share, and move Zoo
projects through the existing API endpoints. This slice requires the API
project/workspace authorization child and its `projects:read`, `projects:write`,
and `projects:manage` scopes. Opening a project makes a temporary source copy;
updating a project submits its complete source tree, the acknowledged revision,
and explicit deleted paths. Existing API ownership and revision checks remain
authoritative. Direct results remain the default; eligible operations can opt
into background execution.

## Browser workspace and previews

`/mcp/workspace` and the `ui://zoo/workspace-v2.html` MCP app resource serve the
same bundled workspace. It uses deployment-derived OAuth endpoints and the
reserved workspace client supplied by the API authorization slice. Browser
operations explicitly request background execution and poll persisted results.
The existing Three.js viewer, source editor, and model previews are retained;
GLB API changes and a different viewer remain separate decisions.

Build browser assets with `npm ci --ignore-scripts && npm run build` in
`src/zoo_mcp/hosted/web` before building a wheel or starting this service.
Hosted image and release workflows include this step. Attachment import accepts
only explicitly configured `ZOO_MCP_FILE_HOSTS`, resolves and pins public IPs,
and rejects redirects. Configure that allowlist for each deployment.

## Distribution and rollout

`server.json` retains the local PyPI package and adds the planned remote endpoint.
`distribution/zoo` contains the catalog plugin package; archive its contents,
including dotfiles, when submission is authorized. The current package version
remains 0.28.1. Follow the repository release versioning instructions before a
new release. Catalog descriptions and fixture screenshots are under
[`docs/catalog`](catalog/submission.md).

`deploy/hosted.yaml` is a deployment example, with an intentionally incomplete
image digest. Select a reviewed immutable image, configure matching API contracts
and secrets, and use the deployment's own issuer/resource/callback URLs before
applying it. Production enablement and external publication require separate
rollout steps; this stack does not perform either.

Before that rollout, exercise a public API PR preview with the pinned image:

1. Complete OAuth consent, PKCE exchange, refresh, and revocation using the
   supported Claude and Codex metadata clients. Check API-token exchange and
   revocation of its originating API key.
2. Upload and download a file, format a source project, and export a model.
   Verify another grant cannot read its files, jobs, or modeling sessions.
3. Reuse a modeling session across calls and through another service pod. Keep
   a direct call open beyond the normal proxy idle interval and confirm SSE
   keepalives and the final result reach the client through public ingress.
4. Compare a direct operation with the completed result of its background form;
   retry the same idempotency key, cancel a running job, and interrupt a worker.
5. Open, modify, save, and preview a project in the browser and MCP app resource.
   Capture final catalog screenshots from the deployed service before submission.
