import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
createServer(async (req, res) => {
  if (req.url?.startsWith("/mcp/workspace")) {
    res.setHeader("Content-Type", "text/html");
    res.end(await readFile(new URL("./dist/index.html", import.meta.url)));
  } else {
    res.writeHead(404);
    res.end();
  }
}).listen(8088, "127.0.0.1");
