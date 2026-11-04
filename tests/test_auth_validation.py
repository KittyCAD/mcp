import asyncio

import httpx
import pytest
from kittycad.models import BlockReason

from zoo_mcp import auth, credentials
from zoo_mcp.credentials import ZooCredentials, get_credentials

CREDENTIALS = ZooCredentials("secret-token", "https://api.example")
USER_URL = CREDENTIALS.base_url + "/user"


@pytest.fixture
def user_response():
    return {
        "id": "00000000-0000-4000-8000-000000000001",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "image": "https://api.example/avatar.png",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes,valid",
    [
        ({}, True),
        *[({"block": reason.value}, True) for reason in BlockReason],
        ({"deletion_scheduled": True}, False),
    ],
)
async def test_validate_login_checks_user_through_sdk(
    httpx_mock, user_response, changes, valid
):
    httpx_mock.add_response(url=USER_URL, json=user_response | changes)
    assert await auth.validate_login(CREDENTIALS) is valid
    request = httpx_mock.get_request()
    assert request.method == "GET"
    assert request.headers["Authorization"] == "Bearer secret-token"


@pytest.mark.parametrize("entrypoint", ["startup", "status"])
def test_billing_block_does_not_disable_saved_login(
    monkeypatch, httpx_mock, user_response, capsys, entrypoint
):
    for name in ("KITTYCAD_API_TOKEN", "ZOO_API_TOKEN", "ZOO_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ZOO_HOST", CREDENTIALS.base_url)
    monkeypatch.setattr(
        credentials, "_local_credentials", ZooCredentials("", CREDENTIALS.base_url)
    )
    monkeypatch.setattr(credentials, "_saved_login", None)
    monkeypatch.setattr(credentials, "_login_message", "No API token configured")
    monkeypatch.setattr(auth.LoginStore, "load", lambda self, origin: CREDENTIALS)
    httpx_mock.add_response(
        url=USER_URL, json=user_response | {"block": "payment_method_failed"}
    )
    if entrypoint == "startup":

        async def start():
            task = auth.start_saved_login()
            assert task is not None
            await task
            assert get_credentials() == CREDENTIALS

        asyncio.run(start())
    else:
        auth.auth_command("status")
        assert "logged in" in capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_validate_login_rejects_unauthorized_credentials(
    httpx_mock, status, capsys, caplog
):
    httpx_mock.add_response(
        url=USER_URL, status_code=status, json={"message": "sensitive-upstream-details"}
    )
    assert await auth.validate_login(CREDENTIALS) is False
    output = capsys.readouterr()
    assert output.out == output.err == ""
    assert "sensitive-upstream-details" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["server", "connection", "timeout"])
async def test_validate_login_keeps_upstream_errors_private(
    httpx_mock, failure, capsys, caplog
):
    details = "sensitive-upstream-details secret-token"
    if failure == "server":
        httpx_mock.add_response(
            url=USER_URL, status_code=503, json={"message": details}
        )
    else:
        error = httpx.ConnectError if failure == "connection" else httpx.ReadTimeout
        httpx_mock.add_exception(error(details), url=USER_URL)
    with pytest.raises(auth.LoginError, match="Check connectivity and retry") as raised:
        await auth.validate_login(CREDENTIALS)
    assert "sensitive" not in str(raised.value)
    assert "secret-token" not in str(raised.value)
    output = capsys.readouterr()
    assert output.out == output.err == ""
    assert "sensitive" not in caplog.text
    assert "secret-token" not in caplog.text


@pytest.mark.asyncio
async def test_validate_login_bounds_stalled_sdk_transport(httpx_mock, monkeypatch):
    started = asyncio.Event()

    async def stalled_transport(request):
        started.set()
        await asyncio.Event().wait()

    httpx_mock.add_callback(stalled_transport, url=USER_URL)
    timeout = asyncio.timeout

    def immediate_timeout(seconds):
        assert seconds == 15
        return timeout(0)

    monkeypatch.setattr(auth.asyncio, "timeout", immediate_timeout)
    with pytest.raises(auth.LoginError, match="Check connectivity and retry"):
        await auth.validate_login(CREDENTIALS)
    assert started.is_set()
