import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
createServer(async (req, res) => {
  if (req.url?.startsWith("/mcp/workspace")) {
    const html = await readFile(
      new URL("./dist/index.html", import.meta.url),
      "utf8",
    );
    const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
    const digest = createHash("sha256").update(script).digest("base64");
    res.setHeader("Content-Type", "text/html");
    // Match the deployed workspace: embedded buffers cannot use fetch(data:...).
    res.setHeader(
      "Content-Security-Policy",
      `default-src 'none'; script-src 'sha256-${digest}'; style-src 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'`,
    );
    res.end(html);
  } else {
    res.writeHead(404);
    res.end();
  }
}).listen(8088, "127.0.0.1");
