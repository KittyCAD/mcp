"""Credentials stay in their call context and reach both downstream SDKs."""

import asyncio
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest.mock import AsyncMock

import pytest
from kittycad.client import DEFAULT_BASE_URL

from zoo_mcp import credentials, zoo_tools
from zoo_mcp.credentials import ZooCredentials, get_credentials, use_credentials


@pytest.fixture(autouse=True)
def local_credentials(monkeypatch):
    local = ZooCredentials("local-token", "https://local.example")
    monkeypatch.setattr(credentials, "_local_credentials", local)
    return local


@pytest.mark.parametrize(
    "token_variable", ["ZOO_API_TOKEN", "KITTYCAD_API_TOKEN", "ZOO_TOKEN"]
)
@pytest.mark.parametrize("host_variable", ["ZOO_HOST", "KITTYCAD_HOST", None])
def test_local_settings_are_captured_once(monkeypatch, token_variable, host_variable):
    for name in (
        "ZOO_API_TOKEN",
        "KITTYCAD_API_TOKEN",
        "ZOO_TOKEN",
        "ZOO_HOST",
        "KITTYCAD_HOST",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(token_variable, "startup-token")
    if host_variable:
        monkeypatch.setenv(host_variable, "https://startup.example")
    captured = ZooCredentials.from_environment()
    monkeypatch.setattr(credentials, "_local_credentials", captured)

    monkeypatch.setenv(token_variable, "changed-token")
    monkeypatch.setenv("ZOO_HOST", "https://changed.example")
    assert get_credentials() == ZooCredentials(
        "startup-token",
        "https://startup.example" if host_variable else DEFAULT_BASE_URL,
    )


def test_missing_startup_token_cannot_fall_back_to_later_environment(monkeypatch):
    monkeypatch.setattr(credentials, "_local_credentials", ZooCredentials(""))
    monkeypatch.setenv("ZOO_API_TOKEN", "later-token")
    with pytest.raises(ValueError, match="No API token configured"):
        get_credentials()


@pytest.mark.asyncio
async def test_scopes_isolate_concurrent_calls_and_restore_after_failure(
    local_credentials,
):
    first = ZooCredentials("first-token", "https://first.example")
    second = ZooCredentials("second-token", "https://second.example")

    async def observe(configuration):
        with use_credentials(configuration):
            await asyncio.sleep(0)
            async with zoo_tools._new_zoo_client() as client:
                assert (client.token, client.base_url) == (
                    configuration.token,
                    configuration.base_url,
                )
            with (
                pytest.raises(RuntimeError, match="synthetic failure"),
                use_credentials(local_credentials),
            ):
                raise RuntimeError("synthetic failure")
            assert get_credentials() is configuration
        assert get_credentials() is local_credentials

    await asyncio.gather(observe(first), observe(second))
    assert get_credentials() is local_credentials
    assert "first-token" not in repr(first)


@pytest.mark.asyncio
async def test_cancellation_resets_credentials(local_credentials):
    async def cancel_inside_scope():
        try:
            with use_credentials(ZooCredentials("cancelled-token")):
                raise asyncio.CancelledError
        finally:
            assert get_credentials() is local_credentials

    with pytest.raises(asyncio.CancelledError):
        await cancel_inside_scope()


@pytest.fixture
def modeling_api():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *_args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("from_file", [False, True])
async def test_explicit_credentials_reach_native_kcl_without_environment_changes(
    monkeypatch, tmp_path, modeling_api, from_file
):
    monkeypatch.setenv("ZOO_API_TOKEN", "conflicting-environment-token")
    monkeypatch.setenv("ZOO_HOST", "http://127.0.0.1:1")
    before = (os.environ["ZOO_API_TOKEN"], os.environ["ZOO_HOST"])
    base_url, requests = modeling_api
    code = "@settings(kclVersion = 2.0)\nvalue = 1"
    path = tmp_path / "main.kcl"
    path.write_text(code)

    async def connect(token):
        with use_credentials(ZooCredentials(token, base_url)):
            async with asyncio.timeout(10):
                with pytest.raises(Exception, match="401"):
                    await zoo_tools._open_kcl_session(
                        None if from_file else code, path if from_file else None
                    )

    await asyncio.gather(connect("first-token"), connect("refreshed-token"))
    assert len(requests) == 2
    assert all(path.startswith("/ws/modeling/commands?") for path, _ in requests)
    assert {authorization for _, authorization in requests} == {
        "Bearer first-token",
        "Bearer refreshed-token",
    }
    assert (os.environ["ZOO_API_TOKEN"], os.environ["ZOO_HOST"]) == before


@pytest.mark.asyncio
async def test_refresh_reaches_kcl_on_the_next_call(monkeypatch):
    session = AsyncMock(api_call_id=None, websocket_upgrade_request_id=None)
    constructor = AsyncMock(return_value=session)
    monkeypatch.setattr(zoo_tools.kcl, "new_kcl_session_code", constructor)
    for token in ("first-token", "refreshed-token"):
        with use_credentials(ZooCredentials(token, "https://api.example")):
            await zoo_tools._open_kcl_session("value = 1", None)
    assert [call.kwargs["token"] for call in constructor.await_args_list] == [
        "first-token",
        "refreshed-token",
    ]
