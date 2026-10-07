"""Explicit, context-local credentials for Zoo SDK and KCL calls."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from kittycad.client import DEFAULT_BASE_URL


@dataclass(frozen=True, slots=True)
class ZooCredentials:
    """Downstream API configuration; tokens are excluded from representations."""

    token: str = field(repr=False)
    base_url: str = DEFAULT_BASE_URL

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
_credentials: ContextVar[ZooCredentials | None] = ContextVar(
    "zoo_credentials", default=None
)


def get_credentials() -> ZooCredentials:
    """Return explicit call credentials or the local startup settings."""
    credentials = _credentials.get() or _local_credentials
    if not credentials.token:
        raise ValueError(
            "No API token configured. Use ZooCredentials or set "
            "KITTYCAD_API_TOKEN or ZOO_API_TOKEN before starting the server."
        )
    return credentials


@contextmanager
def use_credentials(credentials: ZooCredentials) -> Iterator[None]:
    """Scope credentials to this context and its child tasks without changing env."""
    previous = _credentials.set(credentials)
    try:
        yield
    finally:
        _credentials.reset(previous)
