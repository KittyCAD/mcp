"""Client configuration stays in its call context and reaches both downstream SDKs."""

import asyncio
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlsplit

import pytest
from kittycad.client import DEFAULT_BASE_URL
from websockets.exceptions import InvalidStatus

from zoo_mcp import client_config, zoo_tools
from zoo_mcp.client_config import ZooClientConfig, get_client_config, use_client_config


@pytest.fixture(autouse=True)
def local_config(monkeypatch):
    local = ZooClientConfig("local-token", "https://local.example")
    monkeypatch.setattr(client_config, "_local_config", local)
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
    captured = ZooClientConfig.from_environment()
    monkeypatch.setattr(client_config, "_local_config", captured)

    monkeypatch.setenv(token_variable, "changed-token")
    monkeypatch.setenv("ZOO_HOST", "https://changed.example")
    assert get_client_config() == ZooClientConfig(
        "startup-token",
        "https://startup.example" if host_variable else DEFAULT_BASE_URL,
    )


def test_missing_startup_token_cannot_fall_back_to_later_environment(monkeypatch):
    monkeypatch.setattr(client_config, "_local_config", ZooClientConfig(""))
    monkeypatch.setenv("ZOO_API_TOKEN", "later-token")
    with pytest.raises(ValueError, match="No API token configured"):
        get_client_config()


@pytest.mark.asyncio
async def test_scopes_isolate_concurrent_calls_and_restore_after_failure(
    local_config,
):
    first = ZooClientConfig(
        "first-token", "https://first.example", headers={"X-Caller": "first"}
    )
    second = ZooClientConfig(
        "second-token", "https://second.example", headers={"X-Caller": "second"}
    )

    async def observe(configuration):
        with use_client_config(configuration):
            await asyncio.sleep(0)
            async with zoo_tools._new_zoo_client() as client:
                assert (client.token, client.base_url) == (
                    configuration.token,
                    configuration.base_url,
                )
                assert (
                    client.get_headers()["X-Caller"]
                    == configuration.headers["X-Caller"]
                )
            with (
                pytest.raises(RuntimeError, match="synthetic failure"),
                use_client_config(local_config),
            ):
                raise RuntimeError("synthetic failure")
            assert get_client_config() is configuration
        assert get_client_config() is local_config

    await asyncio.gather(observe(first), observe(second))
    assert get_client_config() is local_config
    assert "first-token" not in repr(first)


def test_config_repr_omits_headers():
    configuration = ZooClientConfig(
        "private-token",
        headers={
            "User-Agent": "ExampleClient/1.0",
            "X-Request-Source": "example-source",
        },
    )
    representation = repr(configuration)
    assert "private-token" not in representation
    assert "headers" not in representation
    assert configuration.headers is not None
    for name, value in configuration.headers.items():
        assert name not in representation
        assert value not in representation


@pytest.mark.asyncio
async def test_sdk_clients_do_not_share_mutable_headers():
    headers = {"X-Caller": "original"}
    with use_client_config(ZooClientConfig("caller-token", headers=headers)):
        async with zoo_tools._new_zoo_client() as first:
            first.headers["X-Caller"] = "changed"
            async with zoo_tools._new_zoo_client() as second:
                assert second.get_headers()["X-Caller"] == "original"
    assert headers == {"X-Caller": "original"}


@pytest.mark.asyncio
async def test_rest_requests_use_context_headers_and_restore_defaults(
    httpx_mock, local_config
):
    httpx_mock.add_response(json={"items": [], "next_page": None}, is_reusable=True)
    configurations = [
        ZooClientConfig(
            "first-token",
            headers={
                "User-Agent": "ExampleClient/1.0",
                "X-Request-Source": "first-source",
                "X-Caller": "first",
            },
        ),
        ZooClientConfig(
            "second-token",
            headers={
                "user-agent": "AnotherCaller/2.0",
                "X-Request-Source": "second-source",
                "X-Caller": "second",
            },
        ),
        ZooClientConfig("plain-token"),
        ZooClientConfig("empty-headers-token", headers={}),
    ]

    async def call(configuration):
        with use_client_config(configuration):
            await asyncio.sleep(0)
            assert await zoo_tools.zoo_list_org_datasets() == []

    await asyncio.gather(*(call(configuration) for configuration in configurations))
    assert await zoo_tools.zoo_list_org_datasets() == []
    configurations.append(local_config)
    requests = httpx_mock.get_requests()
    assert len(requests) == len(configurations)
    by_token = {request.headers["Authorization"]: request for request in requests}
    for configuration in configurations:
        request = by_token[f"Bearer {configuration.token}"]
        assert str(request.url).startswith(configuration.base_url)
        assert len(request.headers.get_list("User-Agent")) == 1
        if configuration.headers:
            for name, value in configuration.headers.items():
                assert request.headers.get_list(name) == [value]
        else:
            assert request.headers["User-Agent"].startswith("python-httpx/")
            assert "X-Request-Source" not in request.headers
            assert "X-Caller" not in request.headers


@pytest.mark.asyncio
async def test_cancellation_resets_config(local_config):
    async def cancel_inside_scope():
        try:
            with use_client_config(ZooClientConfig("cancelled-token")):
                raise asyncio.CancelledError
        finally:
            assert get_client_config() is local_config

    with pytest.raises(asyncio.CancelledError):
        await cancel_inside_scope()


@pytest.fixture
def modeling_api():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            requests.append((self.path, self.headers))
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
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
async def test_explicit_config_reaches_native_kcl_without_environment_changes(
    monkeypatch, tmp_path, modeling_api, from_file
):
    monkeypatch.setenv("ZOO_API_TOKEN", "conflicting-environment-token")
    monkeypatch.setenv("ZOO_HOST", "http://127.0.0.1:1")
    before = (os.environ["ZOO_API_TOKEN"], os.environ["ZOO_HOST"])
    base_url, requests = modeling_api
    code = "@settings(kclVersion = 2.0)\nvalue = 1"
    path = tmp_path / "main.kcl"
    path.write_text(code)

    configurations = [
        ZooClientConfig(
            "first-token",
            base_url,
            headers={
                "User-Agent": "ExampleClient/1.0",
                "X-Request-Source": "first-source",
            },
        ),
        ZooClientConfig(
            "refreshed-token",
            base_url,
            headers={
                "user-agent": "AnotherCaller/2.0",
                "X-Request-Source": "refreshed-source",
            },
        ),
        ZooClientConfig("plain-token", base_url),
        ZooClientConfig("empty-headers-token", base_url, headers={}),
    ]

    async def connect(configuration):
        with use_client_config(configuration):
            async with asyncio.timeout(10):
                with pytest.raises(Exception, match="401"):
                    await zoo_tools._open_kcl_session(
                        None if from_file else code, path if from_file else None
                    )

    await asyncio.gather(*(connect(configuration) for configuration in configurations))
    assert len(requests) == len(configurations)
    assert all(path.startswith("/ws/modeling/commands?") for path, _ in requests)
    for path, _ in requests:
        query = parse_qs(urlsplit(path).query)
        assert query["geometry_only"] == ["true"]
        # The API routes geometry-only requests to CPU without an explicit pool.
        assert query["webrtc"] == ["false"]
    by_token = {headers["Authorization"]: headers for _, headers in requests}
    for configuration in configurations:
        headers = by_token[f"Bearer {configuration.token}"]
        assert len(headers.get_all("User-Agent")) == 1
        if configuration.headers:
            for name, value in configuration.headers.items():
                assert headers.get_all(name) == [value]
        else:
            assert "ExampleClient" not in headers["User-Agent"]
            assert headers.get_all("X-Request-Source") is None
    assert (os.environ["ZOO_API_TOKEN"], os.environ["ZOO_HOST"]) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("user_agent_name", ["User-Agent", "user-agent", "USER-AGENT"])
async def test_modeling_websocket_forwards_headers_once(modeling_api, user_agent_name):
    base_url, requests = modeling_api
    headers = {
        user_agent_name: "ExampleClient/1.0",
        "X-Request-Source": "modeling-source",
        "X-Caller": "modeling",
    }
    with use_client_config(
        ZooClientConfig("modeling-token", base_url, headers=headers)
    ):
        async with zoo_tools._new_zoo_client() as client:
            with pytest.raises(InvalidStatus, match="401"):
                await zoo_tools._open_modeling_websocket(client)
    assert len(requests) == 1
    path, received = requests[0]
    assert path.startswith("/ws/modeling/commands?")
    assert received.get_all("Authorization") == ["Bearer modeling-token"]
    for name, value in headers.items():
        assert received.get_all(name) == [value]


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [None, {}])
async def test_modeling_websocket_preserves_default_headers(modeling_api, headers):
    base_url, requests = modeling_api
    with use_client_config(
        ZooClientConfig("modeling-token", base_url, headers=headers)
    ):
        async with zoo_tools._new_zoo_client() as client:
            with pytest.raises(InvalidStatus, match="401"):
                await zoo_tools._open_modeling_websocket(client)
    assert len(requests) == 1
    _, received = requests[0]
    assert len(received.get_all("User-Agent")) == 1
    assert "websockets/" in received["User-Agent"]
    assert received.get_all("X-Request-Source") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("from_file", [False, True])
async def test_refresh_reaches_kcl_on_the_next_call(monkeypatch, from_file):
    session = AsyncMock(api_call_id=None, websocket_upgrade_request_id=None)
    constructor = AsyncMock(return_value=session)
    monkeypatch.setattr(
        zoo_tools.kcl,
        "new_kcl_session" if from_file else "new_kcl_session_code",
        constructor,
    )
    configurations = [
        ZooClientConfig("first-token", headers={"X-Caller": "first"}),
        ZooClientConfig("refreshed-token", headers={"X-Caller": "refreshed"}),
        ZooClientConfig("plain-token"),
        ZooClientConfig("empty-headers-token", headers={}),
    ]
    for configuration in configurations:
        with use_client_config(configuration):
            await zoo_tools._open_kcl_session(
                None if from_file else "value = 1", "main.kcl" if from_file else None
            )
    assert [call.kwargs["token"] for call in constructor.await_args_list] == [
        "first-token",
        "refreshed-token",
        "plain-token",
        "empty-headers-token",
    ]
    assert [call.kwargs["headers"] for call in constructor.await_args_list] == [
        configuration.headers for configuration in configurations
    ]
