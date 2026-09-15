import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "e2e",
  use: {
    baseURL: "http://127.0.0.1:8088",
    viewport: { width: 1100, height: 850 },
    launchOptions: { executablePath: process.env.ZOO_PLAYWRIGHT_EXECUTABLE },
  },
  webServer: {
    command: "node preview-server.mjs",
    url: "http://127.0.0.1:8088/mcp/workspace",
    reuseExistingServer: false,
  },
  reporter: "list",
});
