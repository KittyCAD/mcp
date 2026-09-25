"""Explicit deployment configuration and bounded resource policies."""

import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    resource: str
    api_url: str
    service_secret: str = field(repr=False)
    capability_secret: str = field(repr=False)
    node_url: str = "http://127.0.0.1:8080"
    file_hosts: tuple[str, ...] = ()
    allowed_origins: tuple[str, ...] = ()
    max_file_bytes: int = 256 * 1024 * 1024
    max_project_bytes: int = 512 * 1024 * 1024
    max_project_files: int = 1000
    max_workers: int = 16
    max_sessions_per_grant: int = 4
    max_jobs_per_grant: int = 8
    idle_seconds: int = 1800
    operation_seconds: int = 300
    capability_seconds: int = 300
    unsafe_local_dev: bool = False

    def __post_init__(self):
        for value in (self.resource, self.api_url):
            u = urlsplit(value)
            local = self.unsafe_local_dev and u.hostname in ("localhost", "127.0.0.1")
            if (
                (u.scheme != "https" and not local)
                or not u.hostname
                or u.username
                or u.password
                or u.query
                or u.fragment
            ):
                raise ValueError(
                    "MCP resource and API URLs must be canonical HTTPS URLs"
                )
        if urlsplit(self.resource).path != "/mcp" or urlsplit(self.api_url).path:
            raise ValueError("Use /mcp for the MCP resource and the bare API origin")
        if (
            min(
                self.max_workers,
                self.max_sessions_per_grant,
                self.max_jobs_per_grant,
                self.idle_seconds,
                self.operation_seconds,
            )
            < 1
        ):
            raise ValueError("Worker, session, and timeout limits must be positive")
        if len(self.service_secret) < 32 or len(self.capability_secret) < 32:
            raise ValueError("Hosted MCP secrets must contain at least 32 characters")
        if self.max_file_bytes > 256 * 1024 * 1024 or self.max_file_bytes < 1:
            raise ValueError(
                "File size must be between 1 byte and the API's 256 MiB ceiling"
            )

    @property
    def origin(self) -> str:
        u = urlsplit(self.resource)
        return f"{u.scheme}://{u.netloc}"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            resource=os.environ.get("ZOO_MCP_RESOURCE", ""),
            api_url=os.environ.get("ZOO_API_URL", "").rstrip("/"),
            service_secret=os.environ.get("ZOO_MCP_SERVICE_SECRET", ""),
            capability_secret=os.environ.get("ZOO_MCP_CAPABILITY_SECRET", ""),
            node_url=os.environ.get("ZOO_MCP_NODE_URL", "http://127.0.0.1:8080"),
            allowed_origins=tuple(
                filter(None, os.environ.get("ZOO_MCP_ALLOWED_ORIGINS", "").split(","))
            ),
            file_hosts=tuple(
                filter(None, os.environ.get("ZOO_MCP_FILE_HOSTS", "").split(","))
            ),
            max_workers=int(os.environ.get("ZOO_MCP_MAX_WORKERS", "16")),
            unsafe_local_dev=os.environ.get("ZOO_MCP_UNSAFE_LOCAL_DEV") == "true",
        )
