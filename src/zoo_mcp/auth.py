"""Local device login. Hosted callers supply ZooCredentials explicitly."""

import asyncio
import sys
import time
import webbrowser
from contextlib import suppress
from datetime import UTC, datetime
from urllib.parse import urlsplit

import anyio.to_thread
import keyring
import requests
from keyring.errors import KeyringError
from kittycad import AsyncKittyCAD
from requests.adapters import HTTPAdapter
from requests_oauth2client import (
    AccessDenied,
    DeviceAuthorizationPollingJob,
    ExpiredToken,
    OAuth2Client,
    OAuth2Error,
)
from requests_oauth2client.utils import validate_endpoint_uri

from zoo_mcp import credentials as credentials_module
from zoo_mcp import ctx
from zoo_mcp.credentials import ZooCredentials

# Dedicated first-party public device client, registered by the API seed migration.
MCP_CLIENT_ID = "c1bc3f76-8578-4ff1-9f64-0e909c164670"


class LoginError(ValueError):
    """A safe, user-facing authentication error (never includes response bodies)."""


def api_origin(value: str) -> str:
    try:
        url = urlsplit(value)
        port = url.port
    except ValueError:
        raise LoginError("Zoo login requires a valid API origin.") from None
    if (
        url.scheme not in ("https", "http")
        or not url.hostname
        or url.username
        or url.password
        or url.path not in ("", "/")
        or url.query
        or url.fragment
        or (
            url.scheme == "http"
            and url.hostname not in ("localhost", "127.0.0.1", "::1")
        )
    ):
        raise LoginError(
            "Zoo login requires an HTTPS API origin (HTTP is allowed on localhost)."
        )
    host = f"[{url.hostname}]" if ":" in url.hostname else url.hostname
    if port and port != (443 if url.scheme == "https" else 80):
        host += f":{port}"
    return f"{url.scheme}://{host}"


class LoginStore:
    """Store each API origin's login in the operating system credential store."""

    service = "zoo-mcp"

    def load(self, origin: str) -> ZooCredentials | None:
        origin = api_origin(origin)
        try:
            token = keyring.get_password(self.service, origin)
            return ZooCredentials(token, origin) if token else None
        except KeyringError:
            raise LoginError(
                "Could not access the system credential store. Unlock it or set ZOO_API_TOKEN."
            ) from None

    def save(self, login: ZooCredentials) -> None:
        try:
            keyring.set_password(self.service, api_origin(login.base_url), login.token)
        except KeyringError:
            raise LoginError(
                "Could not save Zoo login in the system credential store. Unlock it or set ZOO_API_TOKEN."
            ) from None

    def delete(self, origin: str) -> None:
        try:
            origin = api_origin(origin)
            if keyring.get_password(self.service, origin) is not None:
                keyring.delete_password(self.service, origin)
        except KeyringError:
            raise LoginError(
                "Could not remove Zoo login from the system credential store."
            ) from None


class _SystemTrustAdapter(HTTPAdapter):
    def build_connection_pool_key_attributes(self, request, verify, cert=None):
        host, settings = super().build_connection_pool_key_attributes(
            request, verify, cert
        )
        settings["ssl_context"] = ctx
        return host, settings


def device_login(origin: str, *, browser: bool = True) -> ZooCredentials:
    """Run interactive login in the CLI; OAuth polling belongs to the library."""
    origin = api_origin(origin)
    request_options = {"timeout": 15, "allow_redirects": False}
    try:
        with requests.Session() as session:
            session.mount("https://", _SystemTrustAdapter())
            client = OAuth2Client(
                token_endpoint=origin + "/oauth2/device/token",
                device_authorization_endpoint=origin + "/oauth2/device/auth",
                client_id=MCP_CLIENT_ID,
                session=session,
                testing=origin.startswith("http://"),
            )
            info = client.authorize_device(requests_kwargs=request_options)
            url = validate_endpoint_uri(
                info.verification_uri_complete or info.verification_uri, path=False
            )
            if info.expires_at is None:
                raise LoginError("Zoo returned an invalid device login lifetime.")
            poll = DeviceAuthorizationPollingJob(
                client, info, requests_kwargs=request_options
            )
            if poll.interval <= 0:
                raise LoginError("Zoo returned an invalid polling interval.")
            print(
                f"Zoo MCP login: open {url}\nEnter code: {info.user_code}",
                file=sys.stderr,
                flush=True,
            )
            if browser:
                with suppress(webbrowser.Error, OSError):
                    webbrowser.open(url)
            lifetime = (info.expires_at - datetime.now(UTC)).total_seconds()
            # The library handles polling intervals, pending approval, and slow
            # down responses. Only the device-code deadline belongs to this loop.
            deadline = time.monotonic() + lifetime
            while time.monotonic() + poll.interval < deadline:
                token = poll()
                if token is not None:
                    if (
                        not isinstance(token.access_token, str)
                        or not token.access_token
                    ):
                        raise LoginError("Zoo returned an invalid device token.")
                    return ZooCredentials(token.access_token, origin)
            raise LoginError("Zoo device login expired. Run zoo-mcp auth login again.")
    except AccessDenied:
        raise LoginError(
            "Zoo device login was denied. Run zoo-mcp auth login again."
        ) from None
    except ExpiredToken:
        raise LoginError(
            "Zoo device login expired. Run zoo-mcp auth login again."
        ) from None
    except (
        OAuth2Error,
        requests.RequestException,
        ValueError,
        TypeError,
        KeyError,
    ) as error:
        if isinstance(error, LoginError):
            raise
        raise LoginError(
            "Zoo device login failed. Check connectivity and run zoo-mcp auth login again."
        ) from None


async def validate_login(credentials: ZooCredentials) -> bool:
    try:
        async with (
            asyncio.timeout(15),
            AsyncKittyCAD(
                token=credentials.token,
                base_url=api_origin(credentials.base_url),
                verify_ssl=ctx,
            ) as client,
        ):
            user = await client.users.get_user_self()
            return (
                user is not None and user.block is None and not user.deletion_scheduled
            )
    except Exception as error:
        # SDK errors intentionally do not reach stderr: some contain request details.
        status = getattr(error, "status_code", None)
        if status in (401, 403):
            return False
        raise LoginError(
            "Could not verify Zoo login. Check connectivity and retry."
        ) from None


async def _load_saved_login() -> None:
    store = LoginStore()
    try:
        origin = api_origin(credentials_module._local_credentials.base_url)
        login = await anyio.to_thread.run_sync(
            store.load, origin, abandon_on_cancel=True
        )
        if login and await validate_login(login):
            credentials_module._saved_login = login
    except (LoginError, OSError):
        credentials_module._login_message = "Zoo login unavailable. Run zoo-mcp auth login and restart the server, or set ZOO_API_TOKEN."
        print(credentials_module._login_message, file=sys.stderr, flush=True)


def start_saved_login() -> asyncio.Task[None] | None:
    if (
        credentials_module._credentials.get() is not None
        or credentials_module._local_credentials.token
    ):
        return None
    credentials_module._saved_login = None
    credentials_module._login_message = (
        "Run zoo-mcp auth login and restart the server, or set ZOO_API_TOKEN."
    )
    return asyncio.create_task(_load_saved_login(), name="zoo-saved-login")


async def auth_command(command: str, *, browser: bool = True) -> None:
    config = ZooCredentials.from_environment()
    origin = api_origin(config.base_url)
    store = LoginStore()
    if command == "logout":
        store.delete(origin)
        print(
            f"Removed saved Zoo MCP login for {origin}. Environment credentials still take precedence.",
            file=sys.stderr,
        )
        return
    if config.token:
        if not await validate_login(config):
            raise LoginError(f"Zoo MCP {origin}: environment credentials invalid.")
        print(f"Zoo MCP {origin}: environment credentials valid.", file=sys.stderr)
        return
    if command == "login":
        login = device_login(origin, browser=browser)
        store.save(login)
        print(f"Zoo MCP {origin}: logged in.", file=sys.stderr)
        return
    login = store.load(origin)
    valid = login is not None and await validate_login(login)
    if not valid:
        raise LoginError(f"Zoo MCP {origin}: no valid saved login.")
    print(f"Zoo MCP {origin}: logged in.", file=sys.stderr)
