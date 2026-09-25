import { readFileSync } from "node:fs";
import { test, expect } from "@playwright/test";
const bracket = readFileSync(new URL("./bracket.glb", import.meta.url));
test("connect, upload, preview, and preserve project revision", async ({
  page,
}) => {
  const origin = "http://127.0.0.1:8088",
    errors: string[] = [],
    calls: any[] = [];
  let uploaded = false;
  const artifact = {
    artifact_id: "d9de48b0-0ec1-4eae-8e66-f8bd599a1760",
    name: "bracket.glb",
    size_bytes: bracket.length,
    download_url: origin + "/mcp/files/example?capability=test",
  };
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/mcp/workspace/config", (route) =>
    route.fulfill({
      json: {
        api_url: origin,
        resource: origin + "/mcp",
        client_id: "test",
        scopes: ["modeling", "files:read", "files:write"],
      },
    }),
  );
  await page.route("**/oauth2/authorize?**", (route) => {
    const url = new URL(route.request().url());
    return route.fulfill({
      status: 302,
      headers: {
        location:
          origin +
          "/mcp/workspace?code=test&state=" +
          url.searchParams.get("state") +
          "&iss=" +
          encodeURIComponent(origin),
      },
    });
  });
  await page.route("**/oauth2/token", (route) =>
    route.fulfill({
      json: { access_token: "test-access", refresh_token: "test-refresh" },
    }),
  );
  await page.route("**/mcp/files/**", (route) => {
    if (route.request().method() === "PUT") {
      uploaded = true;
      return route.fulfill({ json: { ok: true } });
    }
    return route.fulfill({
      contentType: "model/gltf-binary",
      body: bracket,
    });
  });
  await page.route("**/mcp/workspace/call", (route) => {
    const body = route.request().postDataJSON();
    calls.push(body);
    let result: any = {};
    switch (body.name) {
      case "list_artifacts":
        result = { artifacts: uploaded ? [artifact] : [] };
        break;
      case "create_upload":
        result = {
          artifact_id: artifact.artifact_id,
          upload_url: artifact.download_url,
        };
        break;
      case "get_artifact":
        result = artifact;
        break;
      case "list_projects":
        result = {
          projects: [
            {
              id: "8f95c984-faf4-428a-81fb-52337ebf4fb6",
              title: "Mounting bracket",
              revision: "revision-shown-to-user",
            },
          ],
        };
        break;
      case "open_project":
        result = {
          status: "completed",
          result: { ...artifact, name: "project.zip" },
        };
        break;
      case "update_project":
        result = {
          status: "completed",
          result: {
            id: body.arguments.project_id,
            title: body.arguments.title,
            revision: "new-revision",
          },
        };
        break;
    }
    return route.fulfill({ json: result });
  });
  await page.goto("/mcp/workspace");
  expect(errors).toEqual([]);
  await page.getByRole("button", { name: "Connect Zoo", exact: true }).click();
  await expect(page.getByText("No temporary files yet.")).toBeVisible();
  await page.getByLabel("Upload CAD or KCL file").setInputFiles({
    name: "bracket.glb",
    mimeType: "model/gltf-binary",
    buffer: bracket,
  });
  await expect(page.getByRole("button", { name: "bracket.glb" })).toBeVisible();
  await page.getByRole("button", { name: "Preview in 3D" }).click();
  await expect(page.getByRole("status")).toHaveText(
    "Drag to orbit · Scroll to zoom",
  );
  await page.screenshot({
    path: "../../../../docs/catalog/workspace-desktop.png",
    fullPage: true,
  });
  await page.getByLabel("File source").selectOption("projects");
  await page.getByRole("button", { name: "Mounting bracket" }).click();
  await expect(page.getByLabel("Project title")).toHaveValue(
    "Mounting bracket",
  );
  await page.getByRole("button", { name: "Save to Zoo" }).click();
  await expect(page.getByRole("status")).toHaveText("Saved to Zoo.");
  expect(
    calls.find((c) => c.name === "update_project").arguments.expected_revision,
  ).toBe("revision-shown-to-user");
  expect(errors).toEqual([]);
  await page.setViewportSize({ width: 390, height: 850 });
  await page.screenshot({
    path: "../../../../docs/catalog/workspace-mobile.png",
    fullPage: true,
  });
});
