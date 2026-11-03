"""Local device login. Hosted callers supply ZooCredentials explicitly."""

import asyncio
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from kittycad import AsyncKittyCAD

from zoo_mcp import credentials as credentials_module
from zoo_mcp import ctx
from zoo_mcp.credentials import ZooCredentials

# Dedicated first-party public device client, registered by the API seed migration.
MCP_CLIENT_ID = "c1bc3f76-8578-4ff1-9f64-0e909c164670"
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"


class LoginError(ValueError):
    """A safe, user-facing authentication error (never includes response bodies)."""


def api_origin(value: str) -> str:
    url = urlsplit(value)
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
    port = url.port
    host = f"[{url.hostname}]" if ":" in url.hostname else url.hostname
    if port and port != (443 if url.scheme == "https" else 80):
        host += f":{port}"
    return f"{url.scheme}://{host}"


@dataclass(frozen=True)
class SavedLogin:
    origin: str
    token: str = field(repr=False)
    expires_at: float

    def valid(self) -> bool:
        return (
            bool(self.token)
            and math.isfinite(self.expires_at)
            and self.expires_at > time.time()
        )


class LoginStore:
    def __init__(self, directory: Path | None = None):
        self.directory = directory or (
            Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
            / "zoo-mcp"
            / "logins"
        )

    def path(self, origin: str) -> Path:
        return self.directory / (
            hashlib.sha256(api_origin(origin).encode()).hexdigest() + ".json"
        )

    def load(self, origin: str) -> SavedLogin | None:
        path = self.path(origin)
        try:
            with path.open() as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or (
                    os.name != "nt" and info.st_mode & 0o077
                ):
                    raise LoginError(
                        "Saved Zoo MCP login must have user-only permissions. Run zoo-mcp auth login again."
                    )
                data = json.load(stream)
            login = SavedLogin(**data)
            if login.origin != api_origin(origin) or not isinstance(login.token, str):
                return None
            return login if login.valid() else None
        except FileNotFoundError:
            return None
        except (TypeError, KeyError, json.JSONDecodeError):
            return None

    def save(self, login: SavedLogin) -> None:
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.directory, 0o700)
        descriptor, temporary = tempfile.mkstemp(prefix=".login-", dir=self.directory)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(
                    {
                        "origin": api_origin(login.origin),
                        "token": login.token,
                        "expires_at": login.expires_at,
                    },
                    stream,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path(login.origin))
        finally:
            Path(temporary).unlink(missing_ok=True)

    def delete(self, origin: str) -> None:
        self.path(origin).unlink(missing_ok=True)


async def open_browser(url: str) -> None:
    # Isolate browser helper output from the MCP stdio stream, including on Linux.
    if getattr(sys, "frozen", False):
        import shutil

        executable = shutil.which("open" if sys.platform == "darwin" else "xdg-open")
        if not executable:
            return
        command = [executable, url]
    else:
        command = [sys.executable, "-m", "webbrowser", "-t", url]
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            async with asyncio.timeout(5):
                await process.wait()
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
    except OSError:
        pass


def positive_seconds(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise LoginError("Zoo returned an invalid login lifetime or polling interval.")
    return float(value)


async def device_login(origin: str, *, browser: bool = True) -> SavedLogin:
    origin = api_origin(origin)
    async with httpx.AsyncClient(
        verify=ctx, follow_redirects=False, timeout=15
    ) as client:
        try:
            response = await client.post(
                origin + "/oauth2/device/auth", data={"client_id": MCP_CLIENT_ID}
            )
            if not response.is_success:
                raise LoginError(
                    "Zoo could not start device login. Retry with zoo-mcp auth login."
                )
            info = response.json()
            device_code, user_code = info["device_code"], info["user_code"]
            url = info.get("verification_uri_complete") or info["verification_uri"]
            if not all(
                isinstance(item, str) and item for item in (device_code, user_code, url)
            ):
                raise LoginError("Zoo returned an invalid device login response.")
            # Do not launch non-web protocols supplied by a misconfigured endpoint.
            verification = urlsplit(url)
            if (
                verification.scheme != "https"
                or not verification.hostname
                or verification.username
                or verification.password
            ):
                raise LoginError("Zoo returned an invalid verification URL.")
            deadline = time.monotonic() + positive_seconds(info["expires_in"])
            interval = positive_seconds(info.get("interval", 5))
            print(
                f"Zoo MCP login: open {url}\nEnter code: {user_code}",
                file=sys.stderr,
                flush=True,
            )
            if browser:
                await open_browser(url)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= interval:
                    await asyncio.sleep(max(0, remaining))
                    raise LoginError(
                        "Zoo device login expired. Run zoo-mcp auth login to retry."
                    )
                await asyncio.sleep(interval)
                try:
                    async with asyncio.timeout(deadline - time.monotonic()):
                        response = await client.post(
                            origin + "/oauth2/device/token",
                            data={
                                "client_id": MCP_CLIENT_ID,
                                "device_code": device_code,
                                "grant_type": DEVICE_GRANT,
                            },
                        )
                except (httpx.TransportError, TimeoutError):
                    interval *= 2
                    continue
                if response.status_code == 429 or response.status_code >= 500:
                    interval += 5
                    continue
                data = response.json()
                if response.is_success:
                    token = data["access_token"]
                    if (
                        not isinstance(token, str)
                        or not token
                        or str(data.get("token_type", "")).lower() != "bearer"
                    ):
                        raise LoginError("Zoo returned an invalid device token.")
                    return SavedLogin(
                        origin,
                        token,
                        time.time() + positive_seconds(data["expires_in"]),
                    )
                error = data.get("error")
                if error == "authorization_pending":
                    continue
                if error == "slow_down":
                    interval += 5
                    continue
                if error == "access_denied":
                    raise LoginError(
                        "Zoo device login was denied. Run zoo-mcp auth login to retry."
                    )
                if error == "expired_token":
                    raise LoginError(
                        "Zoo device login expired. Run zoo-mcp auth login to retry."
                    )
                raise LoginError(
                    "Zoo device login failed. Run zoo-mcp auth login to retry."
                )
        except (
            httpx.HTTPError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as from_error:
            raise LoginError(
                "Zoo device login is unavailable. Run zoo-mcp auth login to retry."
            ) from from_error


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


async def _maintain_login() -> None:
    store = LoginStore()
    try:
        origin = api_origin(credentials_module._local_credentials.base_url)
        while True:
            login = store.load(origin)
            if login and not await validate_login(ZooCredentials(login.token, origin)):
                store.delete(origin)
                login = None
            if login is None:
                login = await device_login(origin)
                store.save(login)
            credentials_module._saved_login = login
            credentials_module._login_message = (
                "Zoo login expired. Run zoo-mcp auth login and restart the server."
            )
            await asyncio.sleep(max(0, login.expires_at - time.time()))
            credentials_module._saved_login = None
            credentials_module._login_message = (
                "Zoo login pending. Complete device login in your browser, then retry."
            )
    except asyncio.CancelledError:
        raise
    except (LoginError, OSError):
        credentials_module._login_message = (
            "Zoo login unavailable. Run zoo-mcp auth login, then restart the server."
        )
        print(credentials_module._login_message, file=sys.stderr, flush=True)


def start_local_login() -> asyncio.Task[None] | None:
    if (
        credentials_module._credentials.get() is not None
        or credentials_module._local_credentials.token
    ):
        return None
    credentials_module._login_message = (
        "Zoo login pending. Complete device login in your browser, then retry."
    )
    return asyncio.create_task(_maintain_login(), name="zoo-device-login")


async def auth_command(command: str, *, browser: bool = True) -> int:
    config = ZooCredentials.from_environment()
    origin = api_origin(config.base_url)
    store = LoginStore()
    if command == "logout":
        store.delete(origin)
        print(
            f"Removed saved Zoo MCP login for {origin}. Environment credentials still take precedence.",
            file=sys.stderr,
        )
        return 0
    if config.token:
        valid = await validate_login(config)
        print(
            f"Zoo MCP {origin}: environment credentials {'valid' if valid else 'invalid'}.",
            file=sys.stderr,
        )
        return 0 if valid else 1
    if command == "login":
        login = await device_login(origin, browser=browser)
        store.save(login)
        print(f"Zoo MCP {origin}: logged in.", file=sys.stderr)
        return 0
    login = store.load(origin)
    valid = login is not None and await validate_login(
        ZooCredentials(login.token, origin)
    )
    print(
        f"Zoo MCP {origin}: {'logged in' if valid else 'no valid saved login'}.",
        file=sys.stderr,
    )
    return 0 if valid else 1
