import { readFileSync } from "node:fs";
import { test, expect } from "@playwright/test";

const bracket = readFileSync(new URL("./bracket.glb", import.meta.url));
const jsonLength = bracket.readUInt32LE(12);
const gltf = JSON.parse(bracket.subarray(20, 20 + jsonLength).toString());
const binary = bracket.subarray(
  28 + jsonLength,
  28 + jsonLength + gltf.buffers[0].byteLength,
);
const ready = "Drag to orbit · Scroll to zoom";
for (const scenario of [
  {
    name: "embedded geometry",
    uri: "data:application/octet-stream;base64," + binary.toString("base64"),
    status: ready,
  },
  {
    name: "glTF buffer media type",
    uri: "data:application/gltf-buffer;base64," + binary.toString("base64"),
    status: ready,
  },
  {
    name: "truncated geometry",
    uri:
      "data:application/octet-stream;base64," +
      binary.subarray(1).toString("base64"),
    status: "The preview contains an incomplete embedded buffer.",
  },
  {
    name: "invalid base64",
    uri: "data:application/octet-stream;base64,!!!!",
    status: "The preview contains an invalid embedded buffer.",
  },
  {
    name: "external geometry",
    uri: "https://external.example.test/geometry.bin",
    status: "Preview files must embed their textures and buffers.",
  },
]) {
  test(`converted STEP with ${scenario.name} under the workspace CSP`, async ({
    page,
  }, testInfo) => {
    const model = structuredClone(gltf);
    model.buffers[0].uri = scenario.uri;
    const origin = "http://127.0.0.1:8088";
    const source = {
      artifact_id: "step-source",
      name: "bracket.step",
      size_bytes: 100,
    };
    const preview = {
      artifact_id: "step-preview",
      name: "bracket.glb",
      download_url: origin + "/mcp/files/preview",
    };
    const calls: any[] = [];
    const errors: string[] = [];
    const externalRequests: string[] = [];
    page.on("request", (request) => {
      if (!request.url().startsWith(origin + "/"))
        externalRequests.push(request.url());
    });
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    await page.addInitScript(() => {
      sessionStorage.setItem(
        "zoo-oauth",
        JSON.stringify({ state: "test-state", verifier: "test-verifier" }),
      );
    });
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
    await page.route("**/oauth2/token", (route) =>
      route.fulfill({ json: { access_token: "test-access" } }),
    );
    await page.route("**/mcp/files/preview", (route) =>
      route.fulfill({
        contentType: "model/gltf+json",
        body: JSON.stringify(model),
      }),
    );
    await page.route("**/mcp/workspace/call", (route) => {
      const body = route.request().postDataJSON();
      calls.push(body);
      switch (body.name) {
        case "list_artifacts":
          return route.fulfill({ json: { artifacts: [source] } });
        case "convert_cad_file":
          return route.fulfill({
            json: { status: "completed", result: { artifacts: [preview] } },
          });
        case "get_artifact":
          return route.fulfill({ json: preview });
        default:
          throw new Error("Unexpected tool: " + body.name);
      }
    });
    await page.goto(
      "/mcp/workspace?code=test&state=test-state&iss=" +
        encodeURIComponent(origin),
    );
    await page.getByRole("button", { name: "bracket.step" }).click();
    await page.getByRole("button", { name: "Preview in 3D" }).click();
    await expect(page.getByRole("status")).toHaveText(scenario.status);
    if (scenario.status === ready) {
      await expect(page.locator("#placeholder")).toBeHidden();
      await expect
        .poll(() =>
          page.locator("#canvas").evaluate(
            (canvas: HTMLCanvasElement) =>
              new Promise<boolean>((resolve) =>
                requestAnimationFrame(() => {
                  const sample = document.createElement("canvas");
                  sample.width = canvas.width;
                  sample.height = canvas.height;
                  const context = sample.getContext("2d")!;
                  context.drawImage(canvas, 0, 0);
                  resolve(
                    context
                      .getImageData(0, 0, sample.width, sample.height)
                      .data.some(
                        (value, index) => index % 4 === 3 && value > 0,
                      ),
                  );
                }),
              ),
          ),
        )
        .toBe(true);
      await page.screenshot({ path: testInfo.outputPath("step-preview.png") });
    }
    expect(
      calls.find((call) => call.name === "convert_cad_file").arguments,
    ).toMatchObject({
      artifact_id: source.artifact_id,
      export_format: "glb",
      execution_mode: "background",
    });
    expect(errors).toEqual([]);
    expect(externalRequests).toEqual([]);
  });
}
