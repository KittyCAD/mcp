import asyncio
import os
import time
from unittest.mock import AsyncMock
from urllib.parse import parse_qs

import httpx
import pytest
from mcp.types import CallToolResult

from zoo_mcp import auth, credentials, server
from zoo_mcp.credentials import ZooCredentials, get_credentials, use_credentials

ORIGIN = "https://api.example"


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(credentials, "_local_credentials", ZooCredentials("", ORIGIN))
    monkeypatch.setattr(credentials, "_saved_login", None)
    monkeypatch.setattr(credentials, "_login_message", "No API token configured")
    monkeypatch.setattr(auth, "open_browser", AsyncMock())
    for name in ("KITTYCAD_API_TOKEN", "ZOO_API_TOKEN", "ZOO_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ZOO_HOST", ORIGIN)


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    sleeps = []
    real_sleep = asyncio.sleep

    async def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
        await real_sleep(0)

    monkeypatch.setattr(auth.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(auth.asyncio, "sleep", sleep)
    return sleeps


def device_response(httpx_mock, **changes):
    httpx_mock.add_response(
        method="POST",
        url=ORIGIN + "/oauth2/device/auth",
        json={
            "device_code": "device",
            "user_code": "ABCD-EFGH",
            "verification_uri_complete": "https://api.example/verify?user_code=ABCD-EFGH",
            "expires_in": 300,
        }
        | changes,
    )


@pytest.mark.asyncio
async def test_form_requests_polling_and_atomic_host_specific_storage(
    httpx_mock, clock, capsys
):
    device_response(httpx_mock, interval=2)
    for status, data in [
        (400, {"error": "authorization_pending"}),
        (400, {"error": "slow_down"}),
        (503, {}),
        (
            200,
            {"access_token": "secret-token", "token_type": "Bearer", "expires_in": 120},
        ),
    ]:
        httpx_mock.add_response(
            method="POST",
            url=ORIGIN + "/oauth2/device/token",
            status_code=status,
            json=data,
        )
    login = await auth.device_login(ORIGIN)
    assert clock == [2, 2, 7, 12]
    store = auth.LoginStore()
    store.save(login)
    assert store.load(ORIGIN + "/") == login
    assert store.load("https://other.example") is None
    if os.name != "nt":
        assert store.path(ORIGIN).stat().st_mode & 0o777 == 0o600
        assert store.directory.stat().st_mode & 0o777 == 0o700
    assert "secret-token" not in repr(login)
    assert len(list(store.directory.iterdir())) == 1
    for request in httpx_mock.get_requests():
        assert request.headers["content-type"] == "application/x-www-form-urlencoded"
        form = parse_qs(request.content.decode())
        assert form["client_id"] == [auth.MCP_CLIENT_ID]
        if request.url.path.endswith("token"):
            assert form["grant_type"] == [auth.DEVICE_GRANT]
            assert form["device_code"] == ["device"]
    out = capsys.readouterr()
    assert out.out == ""
    assert "ABCD-EFGH" in out.err and "secret-token" not in out.err


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,match",
    [
        ("access_denied", "denied"),
        ("expired_token", "expired"),
        ("invalid_client", "failed"),
    ],
)
async def test_terminal_polling_errors(httpx_mock, clock, error, match):
    device_response(httpx_mock)
    httpx_mock.add_response(
        status_code=400, json={"error": error, "error_description": "sensitive-details"}
    )
    with pytest.raises(auth.LoginError, match=match) as raised:
        await auth.device_login(ORIGIN)
    assert "sensitive-details" not in str(raised.value)
    assert auth.LoginStore().load(ORIGIN) is None


@pytest.mark.asyncio
async def test_expiry_transport_backoff_and_cancellation(
    httpx_mock, clock, monkeypatch
):
    device_response(httpx_mock, expires_in=20)
    httpx_mock.add_exception(httpx.ReadTimeout("sensitive-token"))
    httpx_mock.add_response(status_code=400, json={"error": "authorization_pending"})
    with pytest.raises(auth.LoginError, match="expired"):
        await auth.device_login(ORIGIN, browser=False)
    assert clock == [5, 10, 5]
    assert isinstance(auth.open_browser, AsyncMock)
    auth.open_browser.assert_not_called()

    async def cancel(_seconds):
        raise asyncio.CancelledError

    device_response(httpx_mock)
    monkeypatch.setattr(auth.asyncio, "sleep", cancel)
    with pytest.raises(asyncio.CancelledError):
        await auth.device_login(ORIGIN)
    assert auth.LoginStore().load(ORIGIN) is None


def test_store_failed_replace_preserves_old_login(tmp_path, monkeypatch):
    store = auth.LoginStore(tmp_path / "logins")
    old = auth.SavedLogin(ORIGIN, "saved", time.time() + 120)
    store.save(old)

    def fail(*_args):
        raise OSError("failure")

    monkeypatch.setattr(auth.os, "replace", fail)
    with pytest.raises(OSError):
        store.save(auth.SavedLogin(ORIGIN, "new", time.time() + 120))
    assert store.load(ORIGIN) == old
    assert len(list(store.directory.iterdir())) == 1
    store.path(ORIGIN).write_text("invalid JSON")
    assert store.load(ORIGIN) is None


@pytest.mark.asyncio
async def test_environment_and_explicit_credentials_never_start_login(monkeypatch):
    login = AsyncMock()
    monkeypatch.setattr(auth, "_maintain_login", login)
    monkeypatch.setattr(
        credentials, "_local_credentials", ZooCredentials("env", ORIGIN)
    )
    assert auth.start_local_login() is None
    monkeypatch.setattr(credentials, "_local_credentials", ZooCredentials("", ORIGIN))
    with use_credentials(ZooCredentials("explicit", ORIGIN)):
        assert auth.start_local_login() is None
        assert get_credentials().token == "explicit"
    login.assert_not_called()


@pytest.mark.asyncio
async def test_saved_login_survives_restart_and_revocation_reauthenticates(monkeypatch):
    saved = auth.SavedLogin(ORIGIN, "saved", time.time() + 120)
    auth.LoginStore().save(saved)
    validate = AsyncMock(side_effect=[True, True, False])
    device = AsyncMock(
        return_value=auth.SavedLogin(ORIGIN, "replacement", time.time() + 120)
    )
    monkeypatch.setattr(auth, "validate_login", validate)
    monkeypatch.setattr(auth, "device_login", device)
    for expected in ("saved", "saved", "replacement"):
        task = auth.start_local_login()
        assert task is not None
        for _ in range(20):
            await asyncio.sleep(0)
            if credentials._saved_login is not None:
                break
        assert get_credentials().token == expected
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        credentials._saved_login = None
    assert device.await_count == 1


@pytest.mark.asyncio
async def test_lifespan_initializes_while_login_pending_and_cancels_on_shutdown(
    monkeypatch,
):
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def wait():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(auth, "_maintain_login", wait)
    monkeypatch.setattr(server, "_init_kcl_indexes", AsyncMock())
    monkeypatch.setattr(server, "zoo_stop_all_modeling_sessions", AsyncMock())
    async with asyncio.timeout(2), server._lifespan(server.mcp):
        await started.wait()
        with pytest.raises(ValueError, match="login pending"):
            get_credentials()
        result = await server.mcp.call_tool("format_kcl", {"kcl_code": "x=1"})
        assert isinstance(result, CallToolResult)
        assert not result.is_error
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_status_and_logout_never_print_tokens(monkeypatch, capsys):
    store = auth.LoginStore()
    store.save(auth.SavedLogin(ORIGIN, "secret-token", time.time() + 120))
    other = auth.SavedLogin("https://other.example", "other-secret", time.time() + 120)
    store.save(other)
    monkeypatch.setattr(auth, "validate_login", AsyncMock(return_value=True))
    assert await auth.auth_command("status") == 0
    assert await auth.auth_command("logout") == 0
    assert await auth.auth_command("status") == 1
    assert store.load(other.origin) == other
    captured = capsys.readouterr()
    assert captured.out == "" and "secret" not in captured.err
