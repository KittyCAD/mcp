import asyncio
import subprocess
import sys
import threading
import time
from unittest.mock import AsyncMock, Mock
from urllib.parse import parse_qs

import pytest
import requests
from keyring.errors import NoKeyringError
from mcp.types import CallToolResult

from zoo_mcp import auth, credentials, server
from zoo_mcp.credentials import ZooCredentials, get_credentials, use_credentials

ORIGIN = "https://api.example"


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    passwords = {}
    monkeypatch.setattr(auth.keyring, "get_password", lambda *key: passwords.get(key))
    monkeypatch.setattr(
        auth.keyring,
        "set_password",
        lambda service, origin, value: passwords.__setitem__((service, origin), value),
    )
    monkeypatch.setattr(
        auth.keyring, "delete_password", lambda *key: passwords.pop(key)
    )
    monkeypatch.setattr(credentials, "_local_credentials", ZooCredentials("", ORIGIN))
    monkeypatch.setattr(credentials, "_saved_login", None)
    monkeypatch.setattr(credentials, "_login_message", "No API token configured")
    for name in ("KITTYCAD_API_TOKEN", "ZOO_API_TOKEN", "ZOO_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ZOO_HOST", ORIGIN)
    return passwords


@pytest.fixture(autouse=True)
def browser(monkeypatch) -> Mock:
    browser = Mock(return_value=True)
    monkeypatch.setattr(auth.webbrowser, "open", browser)
    return browser


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        if seconds:
            sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(auth.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(auth.time, "sleep", sleep)
    return sleeps


def device_response(requests_mock, **changes):
    requests_mock.post(
        ORIGIN + "/oauth2/device/auth",
        json={
            "device_code": "device",
            "user_code": "ABCD-EFGH",
            "verification_uri": "https://api.example/verify",
            "verification_uri_complete": "https://api.example/verify?user_code=ABCD-EFGH",
            "expires_in": 300,
        }
        | changes,
    )


@pytest.mark.asyncio
async def test_login_uses_device_grant_and_saves_in_keyring(
    requests_mock, clock, capsys, browser
):
    device_response(requests_mock, interval=2)
    requests_mock.post(
        ORIGIN + "/oauth2/device/token",
        [
            {"status_code": 400, "json": {"error": "authorization_pending"}},
            {"status_code": 400, "json": {"error": "slow_down"}},
            {
                "json": {
                    "access_token": "secret-token",
                    "token_type": "Bearer",
                    "expires_in": 120,
                }
            },
        ],
    )
    await auth.auth_command("login")
    assert clock == [2, 2, 7]
    saved = auth.LoginStore().load(ORIGIN)
    assert saved is not None and saved.token == "secret-token"
    assert "secret-token" not in repr(saved)
    browser.assert_called_once_with("https://api.example/verify?user_code=ABCD-EFGH")
    for request in requests_mock.request_history:
        assert request.headers["content-type"] == "application/x-www-form-urlencoded"
        form = parse_qs(request.text)
        assert form["client_id"] == [auth.MCP_CLIENT_ID]
        assert request.timeout == 15
        if request.path.endswith("token"):
            assert form["grant_type"] == [
                "urn:ietf:params:oauth:grant-type:device_code"
            ]
            assert form["device_code"] == ["device"]
    output = capsys.readouterr()
    assert output.out == ""
    assert "ABCD-EFGH" in output.err and "secret-token" not in output.err


@pytest.mark.parametrize(
    "error,match",
    [
        ("access_denied", "denied"),
        ("expired_token", "expired"),
        ("invalid_client", "failed"),
    ],
)
def test_terminal_errors_are_safe(requests_mock, clock, error, match):
    device_response(requests_mock)
    requests_mock.post(
        ORIGIN + "/oauth2/device/token",
        status_code=400,
        json={"error": error, "error_description": "sensitive-details"},
    )
    with pytest.raises(auth.LoginError, match=match) as raised:
        auth.device_login(ORIGIN)
    assert "sensitive-details" not in str(raised.value)
    assert auth.LoginStore().load(ORIGIN) is None
    assert requests_mock.call_count == 2


def test_local_expiry_stops_pending_requests(requests_mock, clock, browser):
    device_response(requests_mock, expires_in=16)
    requests_mock.post(
        ORIGIN + "/oauth2/device/token",
        status_code=400,
        json={"error": "authorization_pending"},
    )
    with pytest.raises(auth.LoginError, match="expired"):
        auth.device_login(ORIGIN, browser=False)
    assert clock == [5, 5, 5]
    browser.assert_not_called()
    assert requests_mock.call_count == 4


@pytest.mark.parametrize("endpoint", ["auth", "token"])
def test_device_requests_never_follow_redirects(requests_mock, clock, endpoint):
    device_response(requests_mock)
    requests_mock.post(
        ORIGIN + "/oauth2/device/" + endpoint,
        status_code=302,
        headers={"Location": "https://other.example/credentials"},
    )
    with pytest.raises(auth.LoginError):
        auth.device_login(ORIGIN)
    assert requests_mock.last_request.url == ORIGIN + "/oauth2/device/" + endpoint


@pytest.mark.parametrize(
    "url", ["file:///tmp/login", "https://user:password@api.example/verify"]
)
def test_invalid_verification_url_never_opens_browser(requests_mock, url, browser):
    device_response(requests_mock, verification_uri_complete=url)
    with pytest.raises(auth.LoginError):
        auth.device_login(ORIGIN)
    browser.assert_not_called()


@pytest.mark.parametrize("browser_failure", [False, True])
def test_browser_failure_and_no_browser_allow_manual_login(
    requests_mock, clock, browser, browser_failure
):
    device_response(requests_mock, verification_uri_complete=None)
    requests_mock.post(
        ORIGIN + "/oauth2/device/token",
        json={"access_token": "token", "token_type": "Bearer", "expires_in": 120},
    )
    if browser_failure:
        browser.side_effect = OSError("no browser")
    assert auth.device_login(ORIGIN, browser=browser_failure).token == "token"
    assert browser.call_count == int(browser_failure)


def test_interrupt_and_network_failure_do_not_save_login(
    requests_mock, clock, monkeypatch
):
    device_response(requests_mock, expires_in=16)
    requests_mock.post(
        ORIGIN + "/oauth2/device/token", exc=requests.ReadTimeout("sensitive-details")
    )
    with pytest.raises(auth.LoginError, match="Check connectivity") as raised:
        auth.device_login(ORIGIN)
    assert "sensitive-details" not in str(raised.value)
    assert clock == [5]
    monkeypatch.setattr(auth.webbrowser, "open", Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        auth.device_login(ORIGIN)
    assert auth.LoginStore().load(ORIGIN) is None


def test_keyring_stores_opaque_tokens_per_origin(isolate):
    store = auth.LoginStore()
    saved = ZooCredentials("saved", ORIGIN)
    store.save(saved)
    other = ZooCredentials("other", "https://other.example")
    store.save(other)
    assert isolate[(store.service, ORIGIN)] == "saved"
    assert store.load(ORIGIN + "/") == saved
    assert store.load("https://api.example:443") == saved
    assert store.load("https://api.example:8443") is None
    store.delete(ORIGIN)
    store.delete(ORIGIN)
    assert store.load(ORIGIN) is None
    assert store.load(other.base_url) == other


@pytest.mark.parametrize("reported_lifetime", [0, 1, 36096])
@pytest.mark.asyncio
async def test_saved_token_validity_comes_from_api_not_reported_lifetime(
    requests_mock, clock, monkeypatch, reported_lifetime
):
    device_response(requests_mock)
    requests_mock.post(
        ORIGIN + "/oauth2/device/token",
        json={
            "access_token": "saved",
            "token_type": "Bearer",
            "expires_in": reported_lifetime,
        },
    )
    await auth.auth_command("login")
    tomorrow = time.time() + 86400
    monkeypatch.setattr(auth.time, "time", lambda: tomorrow)
    validate = AsyncMock(return_value=True)
    monkeypatch.setattr(auth, "validate_login", validate)
    task = auth.start_saved_login()
    assert task is not None
    await task
    expected = ZooCredentials("saved", ORIGIN)
    validate.assert_awaited_once_with(expected)
    assert get_credentials() == expected


def test_unavailable_keyring_has_actionable_safe_error(monkeypatch):
    monkeypatch.setattr(
        auth.keyring,
        "get_password",
        Mock(side_effect=NoKeyringError("sensitive-details")),
    )
    with pytest.raises(auth.LoginError, match="set ZOO_API_TOKEN") as raised:
        auth.LoginStore().load(ORIGIN)
    assert "sensitive-details" not in str(raised.value)


def test_device_requests_use_system_certificate_store():
    request = requests.Request("POST", ORIGIN + "/oauth2/device/auth").prepare()
    _, settings = auth._SystemTrustAdapter().build_connection_pool_key_attributes(
        request, True
    )
    assert settings["ssl_context"] is auth.ctx
    assert settings["cert_reqs"] == "CERT_REQUIRED"


@pytest.mark.asyncio
async def test_environment_and_explicit_credentials_bypass_keyring(monkeypatch):
    load = Mock(side_effect=AssertionError("must not read local credentials"))
    monkeypatch.setattr(auth.keyring, "get_password", load)
    monkeypatch.setattr(
        credentials, "_local_credentials", ZooCredentials("env", ORIGIN)
    )
    assert auth.start_saved_login() is None
    monkeypatch.setattr(credentials, "_local_credentials", ZooCredentials("", ORIGIN))
    with use_credentials(ZooCredentials("explicit", ORIGIN)):
        assert auth.start_saved_login() is None
        assert get_credentials().token == "explicit"
    load.assert_not_called()


@pytest.mark.asyncio
async def test_saved_login_survives_restart_but_revocation_never_starts_device_login(
    monkeypatch,
):
    saved = ZooCredentials("saved", ORIGIN)
    auth.LoginStore().save(saved)
    monkeypatch.setattr(
        auth, "validate_login", AsyncMock(side_effect=[True, True, False])
    )
    device = Mock(side_effect=AssertionError("server must not start device login"))
    monkeypatch.setattr(auth, "device_login", device)
    for valid in (True, True, False):
        task = auth.start_saved_login()
        assert task is not None
        await task
        if valid:
            assert get_credentials().token == "saved"
        else:
            with pytest.raises(ValueError, match="auth login"):
                get_credentials()
    device.assert_not_called()


@pytest.mark.asyncio
async def test_startup_without_saved_login_never_opens_browser(browser):
    task = auth.start_saved_login()
    assert task is not None
    await task
    browser.assert_not_called()
    with pytest.raises(ValueError, match="auth login"):
        get_credentials()


@pytest.mark.asyncio
async def test_lifespan_stays_responsive_and_cancels_blocked_keyring(monkeypatch):
    started, release = threading.Event(), threading.Event()

    def load(_service, _origin):
        started.set()
        release.wait(5)

    monkeypatch.setattr(auth.keyring, "get_password", load)
    monkeypatch.setattr(server, "_init_kcl_indexes", AsyncMock())
    monkeypatch.setattr(server, "zoo_stop_all_modeling_sessions", AsyncMock())
    try:
        async with asyncio.timeout(2), server._lifespan(server.mcp):
            while not started.is_set():
                await asyncio.sleep(0.001)
            result = await server.mcp.call_tool("format_kcl", {"kcl_code": "x=1"})
            assert isinstance(result, CallToolResult)
            assert not result.is_error
    finally:
        release.set()
    assert credentials._saved_login is None


@pytest.mark.asyncio
async def test_status_and_logout_never_print_tokens(monkeypatch, capsys):
    store = auth.LoginStore()
    store.save(ZooCredentials("secret-token", ORIGIN))
    other = ZooCredentials("other-secret", "https://other.example")
    store.save(other)
    monkeypatch.setattr(auth, "validate_login", AsyncMock(return_value=True))
    await auth.auth_command("status")
    await auth.auth_command("logout")
    with pytest.raises(auth.LoginError, match="no valid saved login"):
        await auth.auth_command("status")
    assert store.load(other.base_url) == other
    output = capsys.readouterr()
    assert output.out == "" and "secret" not in output.err


@pytest.mark.parametrize("command", ["login", "status"])
@pytest.mark.parametrize("valid", [True, False])
@pytest.mark.asyncio
async def test_auth_command_with_environment_credentials(
    monkeypatch, capsys, command, valid
):
    monkeypatch.setenv("ZOO_API_TOKEN", "secret-token")
    monkeypatch.setattr(auth, "validate_login", AsyncMock(return_value=valid))
    if valid:
        await auth.auth_command(command)
        assert "environment credentials valid" in capsys.readouterr().err
    else:
        with pytest.raises(auth.LoginError, match="environment credentials invalid"):
            await auth.auth_command(command)
        assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "outcome,exit_code,stderr",
    [
        ("success", 0, ""),
        ("login_error", 1, "Login failed.\n"),
        ("os_error", 1, "Could not update saved Zoo MCP login.\n"),
        ("interrupt", 130, ""),
    ],
)
def test_auth_cli_exit_status_and_error_output(outcome, exit_code, stderr):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from unittest.mock import AsyncMock
from zoo_mcp import server
from zoo_mcp.auth import LoginError

error = {
    "success": None,
    "login_error": LoginError("Login failed."),
    "os_error": OSError("sensitive-details"),
    "interrupt": KeyboardInterrupt(),
}[sys.argv.pop()]
server.auth_command = AsyncMock(side_effect=error)
server.main()
""",
            "auth",
            "status",
            outcome,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == exit_code
    assert result.stdout == ""
    assert result.stderr == stderr
