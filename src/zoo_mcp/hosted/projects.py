"""Project tool adapters over Zoo's existing ownership and revision checks."""

import io
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import quote
from uuid import UUID

from .backend import Backend, Principal, ServiceError
from .files import Artifacts, relative_path, unpack_project


class Projects:
    def __init__(self, backend: Backend):
        self.backend = backend
        self.artifacts = Artifacts(backend)

    async def call(self, p: Principal, name: str, args: dict) -> dict:
        project_id = str(UUID(args["project_id"])) if "project_id" in args else None
        path = f"/user/projects/{project_id}" if project_id else "/user/projects"
        if name == "list_projects":
            return {"projects": (await self.backend.api(p, "GET", path)).json()}
        if name == "get_project":
            return (await self.backend.api(p, "GET", path)).json()
        if name == "open_project":
            p.require("files:write")
            response = await self.backend.api(
                p, "GET", path + "/download", params={"format": "zip"}
            )
            if len(response.content) > self.backend.settings.max_file_bytes:
                raise ServiceError(
                    "project_too_large",
                    "This project's archive exceeds the hosted upload limit.",
                )
            return self.artifacts.describe(
                p, await self.artifacts.store(p, f"{project_id}.zip", response.content)
            )
        if name in {"create_project", "update_project"}:
            row, data = await self.artifacts.read(p, args["project_artifact_id"])
            if not row["data"]["name"].endswith(".zip"):
                raise ServiceError(
                    "invalid_project",
                    "Save the complete source tree as a project ZIP first.",
                )
            with tempfile.TemporaryDirectory(prefix="zoo-project-") as directory:
                root = Path(directory)
                unpack_project(data, root, self.backend.settings)
                body = {
                    "title": args["title"],
                    "description": args.get("description", ""),
                    "entrypoint_path": relative_path(
                        args.get("entrypoint_path", "main.kcl")
                    ),
                    "publication_status": "private",
                }
                if name == "update_project":
                    body.update(
                        expected_revision=args["expected_revision"],
                        deleted_paths=[relative_path(n) for n in args["deleted_paths"]],
                    )
                import json

                files = [("body", (None, json.dumps(body), "application/json"))]
                files.extend(
                    (
                        "files",
                        (
                            str(f.relative_to(root)),
                            f.read_bytes(),
                            "application/octet-stream",
                        ),
                    )
                    for f in sorted(root.rglob("*"))
                    if f.is_file()
                )
                response = await self.backend.api(
                    p, "POST" if name == "create_project" else "PUT", path, files=files
                )
                return response.json()
        mapping = {
            "publish_project": ("POST", "/publish"),
            "delete_project": ("DELETE", ""),
            "list_project_share_links": ("GET", "/share-links"),
            "create_project_share_link": ("POST", "/share-links"),
            "delete_project_share_link": (
                "DELETE",
                "/share-links/" + quote(args.get("key", ""), safe=""),
            ),
            "move_project_to_organization": ("PUT", "/organization"),
            "move_project_to_personal": ("DELETE", "/organization"),
        }
        method, suffix = mapping[name]
        kwargs = {"json": {}} if name == "create_project_share_link" else {}
        response = await self.backend.api(p, method, path + suffix, **kwargs)
        value = response.json() if response.content else {"ok": True}
        return value if isinstance(value, dict) else {"share_links": value}

    async def write_source(self, p: Principal, files: dict[str, str]) -> dict:
        if not files or len(files) > self.backend.settings.max_project_files:
            raise ServiceError(
                "invalid_project", "Supply a project with at most 1,000 text files."
            )
        total = sum(len(value.encode()) for value in files.values())
        if total > self.backend.settings.max_project_bytes:
            raise ServiceError(
                "project_too_large", "Project source exceeds the expanded size limit."
            )
        buf = io.BytesIO()
        seen = set()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, text in files.items():
                name = relative_path(name)
                if name.casefold() in seen:
                    raise ServiceError(
                        "invalid_project", "Project paths must be unique."
                    )
                seen.add(name.casefold())
                archive.writestr(name, text)
        return self.artifacts.describe(
            p, await self.artifacts.store(p, "project.zip", buf.getvalue())
        )
