"""Local device login. Hosted callers supply ZooClientConfig explicitly."""

import asyncio
import sys
import threading
import time
import webbrowser
from concurrent.futures import Future
from contextlib import suppress
from datetime import UTC, datetime
from urllib.parse import urlsplit

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

from zoo_mcp import client_config, ctx, logger
from zoo_mcp.client_config import ZooClientConfig

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

    def load(self, origin: str) -> ZooClientConfig | None:
        origin = api_origin(origin)
        try:
            token = keyring.get_password(self.service, origin)
            return ZooClientConfig(token, origin) if token else None
        except KeyringError:
            raise LoginError(
                "Could not access the system credential store. Unlock it or set ZOO_API_TOKEN."
            ) from None

    def save(self, login: ZooClientConfig) -> None:
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


def device_login(origin: str, *, browser: bool = True) -> ZooClientConfig:
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
            verification_url = info.verification_uri_complete or info.verification_uri
            verification_origin = api_origin(
                urlsplit(verification_url)
                ._replace(path="", query="", fragment="")
                .geturl()
            )
            url = validate_endpoint_uri(
                verification_url,
                https=not (
                    origin.startswith("http://") and verification_origin == origin
                ),
                path=False,
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
                    return ZooClientConfig(token.access_token, origin)
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


async def validate_login(config: ZooClientConfig) -> bool:
    try:
        async with (
            asyncio.timeout(15),
            AsyncKittyCAD(
                token=config.token,
                base_url=api_origin(config.base_url),
                headers=dict(config.headers or {}),
                verify_ssl=ctx,
            ) as client,
        ):
            user = await client.users.get_user_self()
            # Leave billing and access restrictions to the API for each operation.
            return user is not None and not user.deletion_scheduled
    except Exception as error:
        # SDK errors intentionally do not reach stderr: some contain request details.
        status = getattr(error, "status_code", None)
        if status in (401, 403):
            return False
        raise LoginError(
            "Could not verify Zoo login. Check connectivity and retry."
        ) from None


async def _read_saved_login(origin: str) -> ZooClientConfig | None:
    """A keyring unlock prompt must not keep the process alive at shutdown."""
    result: Future[ZooClientConfig | None] = Future()

    def read() -> None:
        if not result.set_running_or_notify_cancel():
            return
        try:
            result.set_result(LoginStore().load(origin))
        except Exception as error:
            result.set_exception(error)

    threading.Thread(target=read, name="zoo-keyring", daemon=True).start()
    return await asyncio.wrap_future(result)


async def _load_saved_login() -> None:
    message = "Run zoo-mcp auth login and restart the server, or set ZOO_API_TOKEN."
    try:
        origin = api_origin(client_config._local_config.base_url)
        login = await _read_saved_login(origin)
        if login and await validate_login(login):
            client_config._saved_login = login
    except (LoginError, OSError):
        message = "Zoo login unavailable. Run zoo-mcp auth login and restart the server, or set ZOO_API_TOKEN."
        logger.warning("%s", message)
    finally:
        client_config._login_message = message


def start_saved_login() -> asyncio.Task[None] | None:
    if (
        client_config._client_config.get() is not None
        or client_config._local_config.token
    ):
        return None
    client_config._saved_login = None
    client_config._login_message = "Checking saved Zoo login; retry shortly."
    return asyncio.create_task(_load_saved_login(), name="zoo-saved-login")


def auth_command(command: str, *, browser: bool = True) -> None:
    """Keep synchronous device polling outside asyncio's SIGINT handler."""
    config = ZooClientConfig.from_environment()
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
        if not asyncio.run(validate_login(config)):
            raise LoginError(f"Zoo MCP {origin}: environment credentials invalid.")
        print(f"Zoo MCP {origin}: environment credentials valid.", file=sys.stderr)
        return
    if command == "login":
        login = device_login(origin, browser=browser)
        store.save(login)
        print(f"Zoo MCP {origin}: logged in.", file=sys.stderr)
        return
    login = store.load(origin)
    valid = login is not None and asyncio.run(validate_login(login))
    if not valid:
        raise LoginError(f"Zoo MCP {origin}: no valid saved login.")
    print(f"Zoo MCP {origin}: logged in.", file=sys.stderr)
