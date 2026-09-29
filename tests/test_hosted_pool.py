import asyncio
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import AsyncMock

import anyio
import pytest
import pytest_asyncio
from mcp.types import CallToolResult, TextContent

from zoo_mcp.hosted.pool import Owner, Worker, WorkerPool, WorkerUnavailable


class FakeWorker(Worker):
    instances: ClassVar[list["FakeWorker"]] = []

    def __init__(self, root, origin):
        super().__init__(root, origin)
        self.calls = []
        self.active = False
        self.started = 0
        self.closed = 0
        self.block: asyncio.Event | None = None
        self.entered = asyncio.Event()
        FakeWorker.instances.append(self)

    async def start(self):
        self.started += 1
        await asyncio.sleep(0)

    async def call(self, credential, name, arguments):
        assert not self.active
        self.active = True
        self.entered.set()
        try:
            if self.block:
                await self.block.wait()
            self.calls.append((credential, name, arguments))
            return CallToolResult(
                content=[TextContent(type="text", text=str(len(self.calls)))]
            )
        finally:
            self.active = False

    async def close(self):
        self.closed += 1
        await super().close()


def owner(key="a", **changes):
    values: dict[str, Any] = {
        "issuer": "https://issuer.example",
        "user_id": "user",
        "org_id": "org",
        "credential_kind": "api_key",
        "credential_id": key,
    }
    return Owner(**(values | changes))


async def credential():
    return "delegated-secret"


@pytest_asyncio.fixture
async def pool(tmp_path):
    FakeWorker.instances = []
    pool = WorkerPool(tmp_path, "https://api.example", factory=FakeWorker)
    yield pool
    await pool.close()
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_concurrent_start_and_reconnect_keep_one_worker(pool):
    results = await asyncio.gather(
        *(pool.call(owner(), credential, "tool", {}) for _ in range(20))
    )
    assert len(FakeWorker.instances) == 1
    assert FakeWorker.instances[0].started == 1
    assert {r.content[0].text for r in results} == {str(i) for i in range(1, 21)}
    await pool.call(owner(), credential, "recover", {})
    assert len(FakeWorker.instances[0].calls) == 21


@pytest.mark.asyncio
async def test_distinct_credential_and_identity_components_are_isolated(pool):
    owners = [
        owner(),
        owner("b"),
        owner(credential_kind="oauth_grant"),
        owner(org_id="other"),
    ]
    for identity in owners:
        await pool.call(identity, credential, "tool", {})
    assert len(pool.entries) == 4
    with pytest.raises(WorkerUnavailable, match="capacity"):
        await pool.call(owner(user_id="other"), credential, "tool", {})
    assert len(FakeWorker.instances) == 4


@pytest.mark.asyncio
async def test_blocked_owner_does_not_block_other_owners_or_queued_cancellation(pool):
    await pool.call(owner(), credential, "first", {})
    worker = FakeWorker.instances[0]
    worker.entered.clear()
    worker.block = asyncio.Event()
    first = asyncio.create_task(pool.call(owner(), credential, "blocked", {}))
    await worker.entered.wait()
    queued = asyncio.create_task(pool.call(owner(), credential, "queued", {}))
    await asyncio.sleep(0)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    await asyncio.wait_for(pool.call(owner("other"), credential, "tool", {}), 1)
    assert worker.closed == 0
    worker.block.set()
    await first


@pytest.mark.asyncio
async def test_cancel_active_call_closes_only_its_worker(pool):
    for key in ("a", "b"):
        await pool.call(owner(key), credential, "first", {})
    worker = FakeWorker.instances[0]
    worker.entered.clear()
    worker.block = asyncio.Event()
    call = asyncio.create_task(pool.call(owner(), credential, "blocked", {}))
    await worker.entered.wait()
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert worker.closed == 1
    assert not worker.workspace.exists()
    assert list(pool.entries) == [owner("b")]
    await pool.call(owner(), credential, "reconnect", {})
    assert len(FakeWorker.instances) == 3


@pytest.mark.asyncio
async def test_idle_expiry_and_authorization_revocation(pool):
    await pool.call(owner(), credential, "first", {})
    entry = pool.entries[owner()]
    entry.last_used -= 1801
    await pool.expire_idle()
    assert not pool.entries
    await pool.call(owner(), credential, "reconnect", {})
    await pool.evict(owner())
    assert not pool.entries


@pytest.mark.asyncio
async def test_failed_start_releases_capacity_and_workspace(pool, monkeypatch):
    async def fail(_self):
        raise RuntimeError("failed confinement")

    monkeypatch.setattr(FakeWorker, "start", fail)
    with pytest.raises(RuntimeError, match="confinement"):
        await pool.call(owner(), credential, "tool", {})
    assert not pool.entries
    assert not FakeWorker.instances[0].workspace.exists()


@pytest.mark.asyncio
async def test_delegation_happens_after_queue_wait(pool):
    await pool.call(owner(), credential, "first", {})
    entry = pool.entries[owner()]
    delegate = AsyncMock(return_value="fresh")

    async with entry.lock:
        queued = asyncio.create_task(pool.call(owner(), delegate, "tool", {}))
        await asyncio.sleep(0)
        delegate.assert_not_awaited()
    await queued
    delegate.assert_awaited_once_with()
    assert FakeWorker.instances[0].calls[-1][0] == "fresh"


@pytest.mark.asyncio
async def test_closed_pool_rejects_new_work(pool):
    await pool.close()
    with pytest.raises(WorkerUnavailable, match="shutting down"):
        await pool.call(owner(), credential, "tool", {})


@pytest.mark.asyncio
async def test_revoke_during_start_joins_start_before_cleanup(pool, monkeypatch):
    starting = asyncio.Event()
    cancelled = asyncio.Event()

    async def delayed_start(self):
        starting.set()
        try:
            await asyncio.Event().wait()
        finally:
            assert self.workspace.exists()
            cancelled.set()

    monkeypatch.setattr(FakeWorker, "start", delayed_start)
    call = asyncio.create_task(pool.call(owner(), credential, "tool", {}))
    await starting.wait()
    await asyncio.wait_for(pool.evict(owner()), 1)
    assert cancelled.is_set()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert not pool.entries
    assert FakeWorker.instances[0].closed == 1


@pytest.mark.asyncio
async def test_deadline_evicts_and_releases_capacity(pool):
    await pool.call(owner(), credential, "first", {})
    worker = FakeWorker.instances[0]
    worker.block = asyncio.Event()
    pool.call_timeout = 0.01
    with pytest.raises(TimeoutError):
        await pool.call(owner(), credential, "blocked", {})
    assert worker.closed == 1
    assert not pool.entries


@pytest.mark.asyncio
async def test_http_cancel_scope_cannot_interrupt_cleanup(pool):
    await pool.call(owner(), credential, "first", {})
    worker = FakeWorker.instances[0]
    worker.block = asyncio.Event()
    worker.entered.clear()
    async with anyio.create_task_group() as group:
        group.start_soon(pool.call, owner(), credential, "blocked", {})
        await worker.entered.wait()
        group.cancel_scope.cancel()
    assert not pool.entries
    assert not worker.workspace.exists()


@pytest.mark.asyncio
async def test_idle_recheck_does_not_evict_a_newly_busy_owner(pool):
    await pool.call(owner(), credential, "first", {})
    entry = pool.entries[owner()]
    entry.last_used -= 1801
    entry.users = 1
    await pool.evict(owner(), entry, idle_only=True)
    assert pool.entries[owner()] is entry
    entry.users = 0


def test_modeling_deadline_must_fit_in_worker_budget(tmp_path: Path):
    with pytest.raises(ValueError):
        WorkerPool(tmp_path, "https://api.example", call_timeout=300)
