import { build } from "esbuild";
import { mkdir, readFile, writeFile } from "node:fs/promises";
const result = await build({
  entryPoints: ["main.ts"],
  bundle: true,
  write: false,
  format: "iife",
  target: "es2022",
  minify: true,
});
const template = await readFile("index.html", "utf8");
await mkdir("dist", { recursive: true });
await writeFile(
  "dist/index.html",
  template.replace(
    "<!-- APPLICATION -->",
    () =>
      "<script>" +
      result.outputFiles[0].text.replaceAll("</script", "<\\/script") +
      "</script>",
  ),
);
