"""Bounded persistent workers selected only by verified authorization identity."""

import asyncio
import shutil
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, ErrorData

from zoo_mcp.hosted.protocol import encode, receive


@dataclass(frozen=True)
class Owner:
    issuer: str
    user_id: str
    org_id: str | None
    credential_kind: Literal["api_key", "oauth_grant"]
    credential_id: str


class WorkerUnavailable(RuntimeError):
    """Retryable infrastructure error; the previous call may have executed."""


class Worker:
    def __init__(self, root: Path, api_origin: str):
        self.workspace = Path(tempfile.mkdtemp(prefix="worker-", dir=root))
        self.process: asyncio.subprocess.Process | None = None
        self.api_origin = api_origin

    async def start(self) -> None:
        # No inherited tokens, proxy credentials, service secrets or config paths.
        env = {
            "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
            "HOME": str(self.workspace),
            "TMPDIR": str(self.workspace),
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            "PYTHONDONTWRITEBYTECODE": "1",
            "ZOO_HOST": self.api_origin,
            "ZOO_API_BASE_URL": self.api_origin,
        }
        starting = asyncio.create_task(
            asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "zoo_mcp.hosted.worker",
                cwd=self.workspace,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        )
        try:
            self.process = await asyncio.shield(starting)
        except asyncio.CancelledError:
            self.process = await starting
            raise
        assert self.process.stdout is not None
        async with asyncio.timeout(30):
            if await receive(self.process.stdout) != {"ready": True}:
                raise WorkerUnavailable("Worker confinement failed")

    async def call(self, credential: str, name: str, arguments: dict) -> CallToolResult:
        assert self.process and self.process.stdin and self.process.stdout
        self.process.stdin.write(
            encode({"credential": credential, "name": name, "arguments": arguments})
        )
        await self.process.stdin.drain()
        response = await receive(self.process.stdout)
        if "error" in response:
            error = ErrorData.model_validate(response["error"])
            raise MCPError(error.code, error.message, error.data)
        return CallToolResult.model_validate(response["result"])

    async def close(self) -> None:
        try:
            if self.process is not None and self.process.returncode is None:
                try:
                    if self.process.stdin:
                        self.process.stdin.write(encode({"shutdown": True}))
                        await self.process.stdin.drain()
                        self.process.stdin.close()
                    async with asyncio.timeout(3):
                        await self.process.wait()
                except (OSError, TimeoutError):
                    if self.process.returncode is None:
                        self.process.kill()
                    await self.process.wait()
        finally:
            shutil.rmtree(self.workspace)


@dataclass
class Entry:
    worker: Worker
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0
    last_used: float = field(default_factory=time.monotonic)
    valid: bool = True
    started: bool = False
    running: asyncio.Task | None = None
    closing: asyncio.Task | None = None


class WorkerPool:
    def __init__(
        self,
        root: Path,
        api_origin: str,
        *,
        capacity: int = 4,
        idle_timeout: float = 1800,
        call_timeout: float = 310,
        factory: Callable[[Path, str], Worker] = Worker,
    ):
        if capacity < 1 or idle_timeout <= 0 or call_timeout <= 300:
            raise ValueError("Invalid worker limits (call timeout must exceed 300s)")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root = root
        self.api_origin = api_origin
        self.capacity = capacity
        self.idle_timeout = idle_timeout
        self.call_timeout = call_timeout
        self.factory = factory
        self.entries: dict[Owner, Entry] = {}
        self.lock = asyncio.Lock()
        self.closed = False

    async def probe(self) -> None:
        """Fail service startup unless a real confined worker can initialize."""
        worker = self.factory(self.root, self.api_origin)
        try:
            await worker.start()
        finally:
            await worker.close()

    async def call(
        self,
        owner: Owner,
        delegate: Callable[[], Awaitable[str]],
        name: str,
        arguments: dict,
    ) -> CallToolResult:
        async with self.lock:
            if self.closed:
                raise WorkerUnavailable("Hosted runtime is shutting down. Reconnect.")
            entry = self.entries.get(owner)
            if entry is None:
                if len(self.entries) >= self.capacity:
                    raise WorkerUnavailable(
                        "Hosted worker capacity is full. Retry later."
                    )
                entry = Entry(self.factory(self.root, self.api_origin))
                self.entries[owner] = entry
            entry.users += 1
        try:
            async with entry.lock:
                if not entry.valid:
                    raise WorkerUnavailable("Worker expired. Reconnect and retry.")
                entry.running = asyncio.current_task()
                try:
                    # Delegate after acquiring the owner lock, so a queued call
                    # never receives a credential that expired while waiting.
                    credential = await delegate()
                    if not entry.started:
                        await entry.worker.start()
                        entry.started = True
                    async with asyncio.timeout(self.call_timeout):
                        result = await entry.worker.call(credential, name, arguments)
                    if not entry.valid:
                        raise WorkerUnavailable(
                            "Worker authorization expired. Reconnect."
                        )
                    return result
                except MCPError:
                    raise
                except BaseException:
                    if entry.valid:
                        await self.evict(owner, entry)
                    raise
                finally:
                    entry.running = None
        finally:
            entry.users -= 1
            entry.last_used = time.monotonic()

    async def evict(
        self, owner: Owner, expected: Entry | None = None, *, idle_only: bool = False
    ) -> None:
        async with self.lock:
            entry = self.entries.get(owner)
            if entry is None or (expected is not None and entry is not expected):
                return
            if idle_only and (
                entry.users or time.monotonic() - entry.last_used < self.idle_timeout
            ):
                return
            if entry.closing is None:
                entry.valid = False
                initiator = asyncio.current_task()

                async def close_entry() -> None:
                    # Cancellation finishes any in-flight spawn before closing it.
                    # Keep the capacity slot until sockets/workspace are gone.
                    if entry.running is not None and entry.running is not initiator:
                        entry.running.cancel()
                        await asyncio.gather(entry.running, return_exceptions=True)
                    try:
                        await entry.worker.close()
                    finally:
                        async with self.lock:
                            if self.entries.get(owner) is entry:
                                del self.entries[owner]

                entry.closing = asyncio.create_task(close_entry())
            task = entry.closing
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def expire_idle(self) -> None:
        for owner, entry in list(self.entries.items()):
            if (
                not entry.users
                and time.monotonic() - entry.last_used >= self.idle_timeout
            ):
                await self.evict(owner, entry, idle_only=True)

    async def close(self) -> None:
        self.closed = True
        for owner in list(self.entries):
            await self.evict(owner)
