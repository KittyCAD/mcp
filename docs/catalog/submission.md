# Zoo catalog submission package

Public MCP URL: `https://api.zoo.dev/mcp`.
Authentication: OAuth authorization code + PKCE S256, CIMD public clients.
Workspace: `https://api.zoo.dev/mcp/workspace`.
OpenAI package: `distribution/zoo` (zip the directory contents, including dotfiles).
The MCP Registry remote is in `server.json`; the local stdio package remains available.

Suggested description: **Create editable CAD models, inspect geometry, convert
files, and manage Zoo projects.** Requires a Zoo account. CAD operations use the
account's existing Zoo API credits and billing settings. Temporary uploads and generated
files expire after seven days; projects saved to Zoo are independent durable copies.

## OpenAI catalog

Upload the package through the OpenAI plugin submission workflow and configure
the MCP server with CIMD. Confirm the production client metadata and callbacks
shown by the management page, allow them on Zoo, and exercise both ChatGPT and
Codex. The manifest includes the shared MCP service and UI metadata. No personal
marketplace installation is required to prepare this public catalog package.

Positive examples:

1. “Create a 40 mm mounting bracket with two 5 mm holes and export STEP.”
2. “Preview this uploaded STL and convert it to GLB.”
3. “Open my Zoo project and show it in 3D.”
4. “Find the volume and surface area of this STEP part.”
5. “Save the revised KCL source as a private Zoo project.”

Negative examples:

1. “Find a restaurant for tonight.” — do not select Zoo.
2. “Send this design to a supplier and purchase 200 parts.” — purchasing is outside this plugin.
3. “Give me another organization's private project.” — access is not available.

## Anthropic Directory

Submit a remote MCP connector with the same URL and OAuth/CIMD configuration.
Choose the applicable engineering/productivity category in the current portal.
Demonstrate uploads, CAD execution/export, project browsing, explicit destructive
operations, and the MCP Apps workspace in Claude. Include the browser workspace
as the fallback where the host cannot render an interactive resource. Directory
review is independent of the MCP Registry and OpenAI submission.

## Reviewer preparation

Create a dedicated, least-privileged Zoo review account with credits and a small
sample project. Provide access through the portal's secure reviewer-credentials
field; never store passwords or tokens in this repository. Include a short screen
recording of login, file upload, GLB preview, saving a private project, and
revocation. Capture final screenshots from the deployed service using a real
sample CAD model; the automated UI screenshots use a local bracket fixture and simulated API responses.

Before submission, replace any staging callback with the actual deployment
origin, verify the publisher profile/contact and public policy URLs, confirm
OAuth client documents, and finish the staging exercises in `docs/hosted.md`.
Uploading these materials and receiving approval are external publication steps.

Reference contracts:

- [OpenAI authentication](https://developers.openai.com/plugins/build/auth)
- [OpenAI submission](https://developers.openai.com/plugins/deploy/submission)
- [OpenAI UI and file reference](https://developers.openai.com/plugins/reference)
- [Anthropic authentication](https://claude.com/docs/connectors/building/authentication)
- [Anthropic submission](https://claude.com/docs/connectors/building/submission)
