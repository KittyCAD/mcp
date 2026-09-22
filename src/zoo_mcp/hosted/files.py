"""Artifact capabilities, bounded transfer, and project path validation."""

import base64
import hashlib
import hmac
import io
import ipaddress
import json
import mimetypes
import socket
import stat
import time
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx

from .backend import Backend, Principal, ServiceError
from .config import Settings


def relative_path(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or "\x00" in name
        or ":" in name
        or path.is_absolute()
        or any(p in ("..", ".") for p in name.split("/"))
    ):
        raise ServiceError(
            "invalid_path",
            "Use a relative project filename without parent-directory components.",
        )
    if any(part.startswith(".") for part in path.parts) or len(name) > 512:
        raise ServiceError(
            "invalid_path", "Hidden files and overlong paths are not supported."
        )
    return str(path)


def unpack_project(data: bytes, directory: Path, settings: Settings) -> None:
    """Reject unsafe archives before creating any extracted files."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        files = archive.infolist()
        if (
            len(files) > settings.max_project_files
            or sum(f.file_size for f in files) > settings.max_project_bytes
        ):
            raise ServiceError(
                "project_too_large",
                "The project exceeds the file count or expanded size limit.",
            )
        seen: set[str] = set()
        for info in files:
            name = relative_path(info.filename.rstrip("/"))
            if (
                name.casefold() in seen
                or stat.S_ISLNK(info.external_attr >> 16)
                or info.flag_bits & 1
            ):
                raise ServiceError(
                    "invalid_archive",
                    "Duplicate paths, symlinks, and encrypted files are not supported.",
                )
            seen.add(name.casefold())
        for info in files:
            target = directory / relative_path(info.filename.rstrip("/"))
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("xb") as dest:
                    remaining = info.file_size
                    while chunk := source.read(min(1024 * 1024, remaining + 1)):
                        remaining -= len(chunk)
                        if remaining < 0:
                            raise ServiceError(
                                "invalid_archive", "Incorrect expanded file size."
                            )
                        dest.write(chunk)


class Capabilities:
    """Authenticated encryption is unnecessary: payloads contain no credentials."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def issue(self, p: Principal, artifact_id: str, action: str) -> str:
        payload = {
            "grant": p.grant_id,
            "id": str(UUID(artifact_id)),
            "action": action,
            "exp": int(time.time()) + self.settings.capability_seconds,
            "nonce": str(uuid4()),
        }
        raw = base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":")).encode()
        ).rstrip(b"=")
        signature = hmac.new(
            self.settings.capability_secret.encode(), raw, hashlib.sha256
        ).digest()
        return (
            raw.decode()
            + "."
            + base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
        )

    def verify(self, token: str, artifact_id: str, action: str) -> dict:
        try:
            raw, signature = token.encode().split(b".")
            actual = base64.urlsafe_b64decode(signature + b"=" * (-len(signature) % 4))
            expected = hmac.new(
                self.settings.capability_secret.encode(), raw, hashlib.sha256
            ).digest()
            if not hmac.compare_digest(actual, expected):
                raise ValueError()
            payload = json.loads(base64.urlsafe_b64decode(raw + b"=" * (-len(raw) % 4)))
            if (
                payload["id"] != artifact_id
                or payload["action"] != action
                or payload["exp"] <= time.time()
            ):
                raise ValueError()
            return payload
        except (ValueError, KeyError, TypeError):
            raise ServiceError(
                "invalid_capability",
                "This transfer link is invalid or expired. Request a new link.",
            ) from None


class Artifacts:
    def __init__(self, backend: Backend):
        self.backend = backend
        self.settings = backend.settings
        self.capabilities = Capabilities(self.settings)

    async def create(self, p: Principal, name: str, size: int) -> dict:
        p.require("files:write")
        name = relative_path(name)
        if size < 0 or size > self.settings.max_file_bytes:
            raise ServiceError(
                "file_too_large", "Files must be no larger than 256 MiB."
            )
        records = await self.backend.list(p, "artifact")
        if (
            len(records) >= 1000
            or sum(r["data"].get("size_bytes", 0) for r in records) + size
            > 10 * 1024**3
        ):
            raise ServiceError(
                "storage_limit", "Delete temporary files before uploading more."
            )
        return await self.backend.put(
            p,
            str(uuid4()),
            "artifact",
            {
                "name": name,
                "size_bytes": size,
                "mime_type": mimetypes.guess_type(name)[0]
                or "application/octet-stream",
                "status": "uploading",
            },
        )

    async def store(self, p: Principal, name: str, data: bytes) -> dict:
        row = await self.create(p, name, len(data))
        return await self.complete(p, row, data)

    async def complete(self, p: Principal, row: dict, data: bytes) -> dict:
        p.require("files:write")
        if (
            row["kind"] != "artifact"
            or row["data"]["status"] != "uploading"
            or row["data"].get("claimed")
            or len(data) != row["data"]["size_bytes"]
        ):
            raise ServiceError(
                "invalid_upload",
                "The upload is complete already or its size does not match.",
            )
        # Claim before writing so simultaneous PUTs cannot replace the same blob.
        row = await self.backend.put(
            p,
            row["id"],
            "artifact",
            {**row["data"], "status": "uploading", "claimed": True},
            row["revision"],
        )
        await self.backend.api(p, "PUT", f"/mcp/blobs/{row['id']}", content=data)
        return await self.backend.put(
            p,
            row["id"],
            "artifact",
            {
                **row["data"],
                "status": "ready",
                "sha256": hashlib.sha256(data).hexdigest(),
            },
            row["revision"],
        )

    async def read(self, p: Principal, artifact_id: str) -> tuple[dict, bytes]:
        p.require("files:read")
        row = await self.backend.get(p, str(UUID(artifact_id)))
        if row["kind"] != "artifact" or row["data"].get("status") != "ready":
            raise ServiceError(
                "not_ready", "This file is unavailable or its upload is incomplete."
            )
        data = (await self.backend.api(p, "GET", f"/mcp/blobs/{row['id']}")).content
        return row, data

    def describe(self, p: Principal, row: dict) -> dict:
        token = self.capabilities.issue(p, row["id"], "download")
        return {
            "artifact_id": row["id"],
            **row["data"],
            "expires_at": row["expires"],
            "download_url": f"{self.settings.origin}/mcp/files/{row['id']}?capability={token}",
        }

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
        return self.describe(p, await self.store(p, "project.zip", buf.getvalue()))

    async def store_project(self, p: Principal, source: Path) -> dict:
        root = source if source.is_dir() else source.parent
        files = sorted(root.rglob("*")) if source.is_dir() else [source]
        files = [path for path in files if path.is_file()]
        if (
            len(files) > self.settings.max_project_files
            or sum(path.stat().st_size for path in files)
            > self.settings.max_project_bytes
        ):
            raise ServiceError(
                "project_too_large", "The updated project exceeds the source limits."
            )
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                if path.is_symlink() or not path.resolve().is_relative_to(
                    root.resolve()
                ):
                    raise ServiceError(
                        "invalid_path",
                        "Project files must remain inside the workspace.",
                    )
                archive.write(path, relative_path(path.relative_to(root).as_posix()))
        return self.describe(p, await self.store(p, "project.zip", output.getvalue()))

    async def import_openai_file(self, p: Principal, value: dict) -> dict:
        p.require("files:write")
        url = urlsplit(value["download_url"])
        if (
            url.scheme != "https"
            or url.hostname not in self.settings.file_hosts
            or url.username
            or url.password
            or url.port not in (None, 443)
        ):
            raise ServiceError(
                "invalid_file_url",
                "This attachment host is not supported. Use the Zoo upload picker.",
            )
        # Exact host allowlist plus public DNS targets; deployments also restrict egress.
        import asyncio

        addresses = await asyncio.get_running_loop().getaddrinfo(
            url.hostname, 443, type=socket.SOCK_STREAM
        )
        if not addresses or any(
            not ipaddress.ip_address(a[4][0]).is_global for a in addresses
        ):
            raise ServiceError(
                "invalid_file_url", "Attachment URL must resolve to public addresses."
            )
        # Pin the vetted address while preserving TLS SNI and hostname verification.
        # A dedicated client prevents proxy/environment routing and connection reuse
        # across different hostnames that happen to resolve to the same address.
        pinned = httpx.URL(value["download_url"]).copy_with(host=addresses[0][4][0])
        async with (
            httpx.AsyncClient(
                trust_env=False, timeout=60, follow_redirects=False
            ) as client,
            client.stream(
                "GET",
                pinned,
                headers={"Host": url.hostname},
                extensions={"sni_hostname": url.hostname},
            ) as response,
        ):
            if response.status_code != 200:
                raise ServiceError(
                    "attachment_expired",
                    "Select the attachment again to obtain a fresh download link.",
                )
            data = bytearray()
            async for chunk in response.aiter_bytes():
                if len(data) + len(chunk) > self.settings.max_file_bytes:
                    raise ServiceError(
                        "file_too_large", "The attachment exceeds the upload limit."
                    )
                data.extend(chunk)
        return self.describe(
            p,
            await self.store(
                p, value.get("file_name") or "attachment.bin", bytes(data)
            ),
        )
