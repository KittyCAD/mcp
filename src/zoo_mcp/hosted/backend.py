"""Zoo API boundary: incoming MCP tokens only go to the authorization server."""

import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from mcp.server.auth.provider import AccessToken

from .config import Settings


class ServiceError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Principal:
    user_id: str
    client_id: str
    grant_id: str
    org_id: str | None
    scopes: frozenset[str]
    token: str = field(repr=False)

    def require(self, *scopes: str) -> None:
        if not set(scopes).issubset(self.scopes):
            raise ServiceError(
                "insufficient_scope",
                "Reconnect Zoo and grant the permissions required for this action.",
            )


class Backend:
    def __init__(self, settings: Settings, http: httpx.AsyncClient):
        self.settings = settings
        self.http = http

    async def _request(
        self, method: str, path: str, *, credential: str | None = None, **kwargs
    ) -> httpx.Response:
        headers = {"X-Zoo-Mcp-Service-Token": self.settings.service_secret}
        if credential:
            headers["Authorization"] = f"Bearer {credential}"
        try:
            response = await self.http.request(
                method, self.settings.api_url + path, headers=headers, **kwargs
            )
        except httpx.HTTPError:
            raise ServiceError(
                "upstream_unavailable",
                "Zoo is temporarily unavailable. The operation outcome may be unknown.",
            ) from None
        if response.is_error:
            codes = {
                401: "reauthorize",
                403: "forbidden",
                404: "not_found",
                409: "conflict",
                429: "rate_limited",
                402: "credits_exhausted",
            }
            code = codes.get(response.status_code, "upstream_error")
            # API error bodies may contain supplied source, URLs, or credentials.
            raise ServiceError(code, f"Zoo could not complete this request ({code}).")
        return response

    async def verify_token(self, token: str) -> AccessToken | None:
        response = await self._request("POST", "/mcp/introspect", json={"token": token})
        info = response.json()
        if (
            not info.get("active")
            or info.get("aud") != self.settings.resource
            or info.get("exp", 0) <= time.time()
        ):
            return None
        return AccessToken(
            token=token,
            client_id=info["client_id"],
            subject=info["sub"],
            scopes=info["scope"].split(),
            expires_at=info["exp"],
            resource=info["aud"],
            claims={
                "iss": self.settings.api_url,
                "grant_id": info["grant_id"],
                "org_id": info.get("org_id"),
            },
        )

    async def principal(self, token: str) -> Principal:
        access = await self.verify_token(token)
        if access is None or not access.subject or not access.claims:
            raise ServiceError("reauthorize", "Reconnect Zoo to continue.")
        return Principal(
            access.subject,
            access.client_id,
            access.claims["grant_id"],
            access.claims.get("org_id"),
            frozenset(access.scopes),
            token,
        )

    async def delegate(self, principal: Principal) -> str:
        response = await self._request(
            "POST", "/mcp/delegate", json={"token": principal.token}
        )
        return response.json()["access_token"]

    async def api(
        self, principal: Principal, method: str, path: str, **kwargs
    ) -> httpx.Response:
        return await self._request(
            method, path, credential=await self.delegate(principal), **kwargs
        )

    async def get(self, p: Principal, record_id: str) -> dict[str, Any]:
        return (await self.api(p, "GET", f"/mcp/records/{record_id}")).json()

    async def list(self, p: Principal, kind: str) -> list[dict[str, Any]]:
        return (await self.api(p, "GET", "/mcp/records", params={"kind": kind})).json()

    async def put(
        self, p: Principal, record_id: str, kind: str, data: dict, revision: int = 0
    ) -> dict[str, Any]:
        return (
            await self.api(
                p,
                "PUT",
                f"/mcp/records/{record_id}",
                json={"kind": kind, "data": data, "expected_revision": revision},
            )
        ).json()

    async def delete(self, p: Principal, record_id: str) -> None:
        await self.api(p, "DELETE", f"/mcp/records/{record_id}")
