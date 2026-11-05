"""Explicit, context-local credentials for Zoo SDK and KCL calls."""

import os
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from kittycad.client import DEFAULT_BASE_URL


@dataclass(frozen=True, slots=True)
class ZooCredentials:
    """Downstream API configuration; tokens and headers are excluded from repr."""

    token: str = field(repr=False)
    base_url: str = DEFAULT_BASE_URL
    headers: Mapping[str, str] | None = field(default=None, repr=False)

    @classmethod
    def from_environment(cls) -> "ZooCredentials":
        """Capture local environment aliases and the KittyCAD SDK's default URL."""
        token = (
            os.getenv("KITTYCAD_API_TOKEN")
            or os.getenv("ZOO_API_TOKEN")
            or os.getenv("ZOO_TOKEN")
            or ""
        )
        base_url = (
            os.getenv("ZOO_HOST") or os.getenv("KITTYCAD_HOST") or DEFAULT_BASE_URL
        )
        return cls(token=token, base_url=base_url)


# All entry points import this module at startup. An unconfigured local server
# can still use tools that do not need Zoo credentials.
_local_credentials = ZooCredentials.from_environment()
_saved_login: ZooCredentials | None = None
_login_message: str = (
    "No API token configured. Run zoo-mcp auth login or set ZOO_API_TOKEN."
)
_credentials: ContextVar[ZooCredentials | None] = ContextVar(
    "zoo_credentials", default=None
)


def get_credentials() -> ZooCredentials:
    """Return explicit call credentials or the local startup settings."""
    explicit = _credentials.get()
    credentials = explicit if explicit is not None else _local_credentials
    if explicit is None and not credentials.token and _saved_login is not None:
        return _saved_login
    if not credentials.token:
        raise ValueError(
            "Explicit ZooCredentials requires a token."
            if explicit is not None
            else _login_message
        )
    return credentials


@contextmanager
def use_credentials(credentials: ZooCredentials) -> Generator[None, None, None]:
    """Scope credentials to this context and its child tasks without changing env."""
    previous = _credentials.set(credentials)
    try:
        yield
    finally:
        _credentials.reset(previous)
