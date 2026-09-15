import { App } from "@modelcontextprotocol/ext-apps";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

type Value = Record<string, any>;
type HostFiles = {
  selectFiles?: () => Promise<
    { fileId: string; fileName: string; mimeType: string }[]
  >;
  getFileDownloadUrl?: (input: {
    fileId: string;
  }) => Promise<{ downloadUrl: string }>;
};
declare global {
  interface Window {
    openai?: HostFiles;
  }
}
const $ = <T extends HTMLElement>(id: string) =>
  document.getElementById(id) as T;
let host: App | undefined,
  config: Value = {},
  accessToken = "",
  refreshToken = "",
  selected: Value | undefined,
  project: Value | undefined;
let rows: Value[] = [];
const status = (text: string, error = false) => {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
};
const key = () => crypto.randomUUID();
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
async function browserToken(form: URLSearchParams) {
  const response = await fetch(config.api_url + "/oauth2/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: form,
  });
  if (!response.ok) throw new Error("Reconnect Zoo to continue.");
  const value = await response.json();
  accessToken = value.access_token;
  refreshToken = value.refresh_token || refreshToken;
}
async function call(name: string, args: Value = {}): Promise<Value> {
  let value: Value;
  if (host) {
    const result = await host.callServerTool({ name, arguments: args });
    if (result.isError)
      throw new Error(
        result.content
          ?.filter((c) => c.type === "text")
          .map((c) => c.text)
          .join(" ") || "Zoo request failed.",
      );
    value = result.structuredContent as Value;
  } else {
    if (!accessToken) throw new Error("Connect Zoo to continue.");
    let response = await fetch("/mcp/workspace/call", {
      method: "POST",
      headers: {
        Authorization: "Bearer " + accessToken,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ name, arguments: args }),
    });
    if (response.status === 401 && refreshToken) {
      await browserToken(
        new URLSearchParams({
          grant_type: "refresh_token",
          refresh_token: refreshToken,
          client_id: config.client_id,
          resource: config.resource,
        }),
      );
      response = await fetch("/mcp/workspace/call", {
        method: "POST",
        headers: {
          Authorization: "Bearer " + accessToken,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ name, arguments: args }),
      });
    }
    value = await response.json();
    if (!response.ok) throw new Error(value.message || "Zoo request failed.");
  }
  if (value?.error) throw new Error(value.message || value.error);
  return value;
}
async function operation(name: string, args: Value): Promise<Value> {
  let result = await call(name, { ...args, idempotency_key: key() });
  for (let n = 0; result.status === "running" && n < 165; n++) {
    status("Zoo is working…");
    await sleep(2000);
    result = await call("get_job", { job_id: result.job_id });
  }
  if (result.status && result.status !== "completed")
    throw new Error(
      result.error ||
        "The operation did not complete. Check its status before trying again.",
    );
  return result.result || result;
}
async function refresh() {
  status("Loading…");
  const mode = $<HTMLSelectElement>("mode").value;
  const value = await call(
    mode === "projects" ? "list_projects" : "list_artifacts",
  );
  rows = value.projects || value.artifacts || [];
  const list = $("files");
  list.replaceChildren();
  list.className = "";
  if (!rows.length) {
    list.className = "empty";
    list.textContent =
      "No " + (mode === "projects" ? "projects" : "temporary files") + " yet.";
  }
  for (const row of rows) {
    const button = document.createElement("button");
    button.className = "row";
    button.textContent = row.title || row.name;
    const small = document.createElement("small");
    small.textContent =
      row.size_bytes !== undefined
        ? (row.size_bytes / 1024).toFixed(0) + " KB"
        : row.publication_status || "Zoo project";
    button.append(small);
    button.onclick = () =>
      run(async () => {
        list
          .querySelectorAll(".selected")
          .forEach((e) => e.classList.remove("selected"));
        button.classList.add("selected");
        project = mode === "projects" ? row : undefined;
        selected =
          mode === "projects"
            ? await operation("open_project", { project_id: row.id })
            : row;
        updateSelection();
      });
    list.append(button);
  }
  status("");
}
function updateSelection() {
  $("filename").textContent =
    project?.title || selected?.name || "No file selected";
  for (const id of ["preview", "download", "save"])
    $<HTMLButtonElement>(id).disabled = !selected;
  for (const id of ["publish", "share", "transfer", "delete"])
    $(id).hidden = !project;
  $<HTMLInputElement>("sourceTitle").value = project?.title || "";
  $("links").replaceChildren();
}
async function upload(file: File) {
  if (file.size > 268435456)
    throw new Error("Files must be no larger than 256 MiB.");
  status("Uploading " + file.name + "…");
  const slot = await call("create_upload", {
    name: file.name,
    size_bytes: file.size,
  });
  const response = await fetch(slot.upload_url, { method: "PUT", body: file });
  if (!response.ok)
    throw new Error("Upload failed. Select the file again to retry.");
  selected = await call("get_artifact", { artifact_id: slot.artifact_id });
  project = undefined;
  updateSelection();
  await refresh();
}
let renderer: THREE.WebGLRenderer | undefined,
  scene: THREE.Scene,
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls;
async function view(url: string) {
  if (!renderer) {
    const canvas = $<HTMLCanvasElement>("canvas");
    renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
    });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(40, 1, 0.001, 100000);
    controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    scene.add(new THREE.HemisphereLight(0xffffff, 0x556644, 3));
    const sun = new THREE.DirectionalLight(0xffffff, 3);
    sun.position.set(3, 6, 4);
    scene.add(sun);
    new ResizeObserver(() => {
      const w = canvas.clientWidth,
        h = canvas.clientHeight;
      renderer!.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }).observe(canvas);
    renderer.setAnimationLoop(() => {
      controls.update();
      renderer!.render(scene, camera);
    });
  }
  const response = await fetch(url);
  if (!response.ok)
    throw new Error("The preview link expired. Select Preview again.");
  const loading = new THREE.LoadingManager();
  loading.setURLModifier((uri) => {
    if (!uri.startsWith("blob:") && !uri.startsWith("data:"))
      throw new Error("Preview files must embed their textures and buffers.");
    return uri;
  });
  const model = await new GLTFLoader(loading).parseAsync(
    await response.arrayBuffer(),
    "",
  );
  const previous = scene.getObjectByName("model");
  if (previous) {
    scene.remove(previous);
    previous.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        object.geometry.dispose();
        for (const material of Array.isArray(object.material)
          ? object.material
          : [object.material])
          material.dispose();
      }
    });
  }
  model.scene.name = "model";
  scene.add(model.scene);
  const box = new THREE.Box3().setFromObject(model.scene),
    center = box.getCenter(new THREE.Vector3()),
    size = Math.max(...box.getSize(new THREE.Vector3()).toArray(), 0.01);
  controls.target.copy(center);
  camera.position
    .copy(center)
    .add(new THREE.Vector3(size * 1.4, size, size * 1.4));
  camera.near = size / 1000;
  camera.far = size * 100;
  camera.updateProjectionMatrix();
  controls.update();
  $("placeholder").style.display = "none";
}
async function preview() {
  if (!selected) return;
  status("Preparing preview…");
  let artifact = selected;
  if (!selected.name.toLowerCase().endsWith(".glb")) {
    const kcl = /\.(kcl|zip)$/i.test(selected.name);
    const result = await operation(
      kcl ? "export_kcl" : "convert_cad_file",
      kcl
        ? { project_artifact_id: selected.artifact_id, export_format: "glb" }
        : { artifact_id: selected.artifact_id, export_format: "glb" },
    );
    artifact = result.artifacts?.find((a: Value) =>
      a.name.toLowerCase().endsWith(".glb"),
    );
    if (!artifact)
      throw new Error(
        "A 3D preview is unavailable for this model. You can still download the output.",
      );
  }
  artifact = await call("get_artifact", { artifact_id: artifact.artifact_id });
  await view(artifact.download_url);
  status("Drag to orbit · Scroll to zoom");
}
async function save() {
  if (!selected) return;
  let artifact = selected;
  if (!/\.zip$/i.test(artifact.name)) {
    if (!/\.kcl$/i.test(artifact.name))
      throw new Error("Save editable KCL source or a project ZIP to Zoo.");
    const fresh = await call("get_artifact", {
      artifact_id: artifact.artifact_id,
    });
    const response = await fetch(fresh.download_url);
    artifact = await operation("write_kcl_project", {
      files: { "main.kcl": await response.text() },
    });
  }
  const title = $<HTMLInputElement>("sourceTitle").value.trim();
  if (!title) throw new Error("Enter a project title.");
  const args: Value = { project_artifact_id: artifact.artifact_id, title };
  if (project) {
    args.project_id = project.id;
    args.expected_revision = project.revision;
    args.deleted_paths = [];
  }
  const result = await operation(
    project ? "update_project" : "create_project",
    args,
  );
  project = result;
  status("Saved to Zoo.");
  updateSelection();
}
function link(url: string) {
  const parsed = new URL(url);
  if (parsed.protocol !== "https:") throw new Error("Invalid download link.");
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.textContent = "Open share link";
  anchor.className = "link";
  anchor.target = "_blank";
  anchor.rel = "noreferrer";
  $("links").append(anchor);
}
async function run(fn: () => Promise<unknown>) {
  try {
    await fn();
  } catch (error) {
    status(
      error instanceof Error ? error.message : "Zoo request failed.",
      true,
    );
  }
}
$("refresh").onclick = () => run(refresh);
$("mode").onchange = () => run(refresh);
$("preview").onclick = () => run(preview);
$("save").onclick = () => run(save);
$<HTMLInputElement>("upload").onchange = (event) => {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (file) run(() => upload(file));
};
$("download").onclick = () =>
  run(async () => {
    if (!selected) return;
    const artifact = await call("get_artifact", {
      artifact_id: selected.artifact_id,
    });
    if (host) await host.openLink({ url: artifact.download_url });
    else location.assign(artifact.download_url);
  });
$("publish").onclick = () =>
  run(async () => {
    await operation("publish_project", { project_id: project!.id });
    status("Submitted for publication review.");
  });
$("share").onclick = () =>
  run(async () => {
    const result = await operation("create_project_share_link", {
      project_id: project!.id,
    });
    link(result.url);
    status("Share link created.");
  });
$("transfer").onclick = () =>
  run(async () => {
    await operation("move_project_to_organization", {
      project_id: project!.id,
    });
    status("Moved to your organization.");
    await refresh();
  });
$("delete").onclick = () =>
  run(async () => {
    const button = $("delete");
    if (button.textContent !== "Confirm delete") {
      button.textContent = "Confirm delete";
      return;
    }
    await operation("delete_project", { project_id: project!.id });
    selected = undefined;
    project = undefined;
    button.textContent = "Delete project";
    updateSelection();
    await refresh();
  });
async function connectBrowser() {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  const encode = (b: Uint8Array) =>
    btoa(String.fromCharCode(...b))
      .replaceAll("+", "-")
      .replaceAll("/", "_")
      .replaceAll("=", "");
  const verifier = encode(bytes),
    state = key();
  sessionStorage.setItem("zoo-oauth", JSON.stringify({ verifier, state }));
  const challenge = encode(
    new Uint8Array(
      await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier)),
    ),
  );
  const params = new URLSearchParams({
    response_type: "code",
    client_id: config.client_id,
    redirect_uri: location.origin + "/mcp/workspace",
    scope: config.scopes.join(" "),
    state,
    code_challenge: challenge,
    code_challenge_method: "S256",
    resource: config.resource,
  });
  location.assign(config.api_url + "/oauth2/authorize?" + params);
}
$("login").onclick = () => run(connectBrowser);
async function start() {
  if (window.parent !== window) {
    host = new App({ name: "Zoo", version: "1.0.0" }, {});
    await host.connect();
    config = await call("open_zoo_workspace");
  } else {
    config = await (await fetch("/mcp/workspace/config")).json();
    const params = new URLSearchParams(location.search);
    if (params.has("code")) {
      const saved = JSON.parse(sessionStorage.getItem("zoo-oauth") || "null");
      sessionStorage.removeItem("zoo-oauth");
      if (
        !saved ||
        params.get("state") !== saved.state ||
        params.get("iss") !== config.api_url
      )
        throw new Error(
          "Authorization could not be verified. Connect Zoo again.",
        );
      history.replaceState(null, "", "/mcp/workspace");
      await browserToken(
        new URLSearchParams({
          grant_type: "authorization_code",
          client_id: config.client_id,
          code: params.get("code")!,
          redirect_uri: location.origin + "/mcp/workspace",
          code_verifier: saved.verifier,
          resource: config.resource,
        }),
      );
    } else {
      $("login").style.display = "block";
      status("Connect your Zoo account to get started.");
    }
  }
  $<HTMLAnchorElement>("account").href =
    config.account_url || config.api_url + "/oauth2/mcp/connections";
  if (window.openai?.selectFiles && window.openai.getFileDownloadUrl) {
    $("native").hidden = false;
    $("native").onclick = () =>
      run(async () => {
        const files = await window.openai!.selectFiles!();
        for (const file of files) {
          const { downloadUrl } = await window.openai!.getFileDownloadUrl!({
            fileId: file.fileId,
          });
          await operation("import_attachment", {
            file: {
              download_url: downloadUrl,
              file_id: file.fileId,
              mime_type: file.mimeType,
              file_name: file.fileName,
            },
          });
        }
        await refresh();
      });
  }
  if (host || accessToken) {
    $("login").style.display = "none";
    await refresh();
  }
}
run(start);
