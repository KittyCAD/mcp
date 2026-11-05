"""Explicit, context-local client configuration for Zoo SDK and KCL calls."""

import os
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from kittycad.client import DEFAULT_BASE_URL


@dataclass(frozen=True, slots=True)
class ZooClientConfig:
    """Downstream API configuration; tokens and headers are excluded from repr."""

    token: str = field(repr=False)
    base_url: str = DEFAULT_BASE_URL
    headers: Mapping[str, str] | None = field(default=None, repr=False)

    @classmethod
    def from_environment(cls) -> "ZooClientConfig":
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
_local_config = ZooClientConfig.from_environment()
_saved_login: ZooClientConfig | None = None
_login_message: str = (
    "No API token configured. Run zoo-mcp auth login or set ZOO_API_TOKEN."
)
_client_config: ContextVar[ZooClientConfig | None] = ContextVar(
    "zoo_client_config", default=None
)


def get_client_config() -> ZooClientConfig:
    """Return explicit call configuration or the local startup settings."""
    explicit = _client_config.get()
    config = explicit if explicit is not None else _local_config
    if explicit is None and not config.token and _saved_login is not None:
        return _saved_login
    if not config.token:
        raise ValueError(
            "Explicit ZooClientConfig requires a token."
            if explicit is not None
            else _login_message
        )
    return config


@contextmanager
def use_client_config(config: ZooClientConfig) -> Generator[None, None, None]:
    """Scope configuration to this context and its child tasks without changing env."""
    previous = _client_config.set(config)
    try:
        yield
    finally:
        _client_config.reset(previous)
