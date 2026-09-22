"""Grant-isolated scene workers with bounded direct execution."""

import asyncio
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4, uuid5

from .backend import Backend, Principal, ServiceError
from .files import Artifacts, unpack_project

INPUT_PATHS = {
    "input_file": "artifact_id",
    "kcl_path": "project_artifact_id",
    "path_artifact_graph": "artifact_graph_id",
}
logger = logging.getLogger(__name__)

OUTPUT_PATHS = {"output_path", "export_path"}


@dataclass
class Worker:
    principal: Principal = field(repr=False)
    process: asyncio.subprocess.Process = field(repr=False)
    directory: tempfile.TemporaryDirectory = field(repr=False)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    touched: float = field(default_factory=time.monotonic)
    scene_id: str | None = None

    async def close(self):
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 5)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        self.directory.cleanup()


class Runtime:
    def __init__(self, backend: Backend):
        self.backend = backend
        self.settings = backend.settings
        self.artifacts = Artifacts(backend)
        self.workers: dict[str, Worker] = {}
        self.jobs: dict[str, asyncio.Task] = {}
        self.direct_tasks: set[asyncio.Task] = set()
        self.closing = False
        self.outcomes: Counter[str] = Counter()
        self._allocation_lock = asyncio.Lock()

    async def start_worker(self, p: Principal) -> Worker:
        async with self._allocation_lock:
            if self.closing or len(self.workers) >= self.settings.max_workers:
                raise ServiceError(
                    "busy", "Zoo modeling capacity is busy. Try again shortly."
                )
            directory = tempfile.TemporaryDirectory(prefix="zoo-mcp-")
            env = {
                k: v
                for k, v in os.environ.items()
                if k in {"PATH", "SSL_CERT_FILE", "SSL_CERT_DIR"}
            }
            env.update(
                {
                    "ZOO_HOST": self.settings.api_url,
                    "ZOO_API_BASE_URL": self.settings.api_url,
                    "TMPDIR": directory.name,
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                    "ZOO_MCP_UNSAFE_LOCAL_DEV": str(
                        self.settings.unsafe_local_dev
                    ).lower(),
                }
            )
            try:
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "zoo_mcp.hosted.worker",
                    cwd=directory.name,
                    env=env,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    limit=16 * 1024 * 1024,
                )
            except BaseException:
                directory.cleanup()
                raise
            worker = Worker(p, process, directory)
            self.workers[str(process.pid)] = worker
            return worker

    async def discard(self, worker: Worker):
        self.workers.pop(str(worker.process.pid), None)
        await worker.close()
        if worker.scene_id:
            try:
                await self.backend.delete(worker.principal, worker.scene_id)
            except ServiceError:
                # Revoked credentials cannot remove the row; expiry/cleanup does.
                pass

    async def invoke(
        self, p: Principal, worker: Worker, name: str, arguments: dict
    ) -> dict:
        if worker.principal.grant_id != p.grant_id:
            raise ServiceError("not_found", "Modeling session not found.")
        async with worker.lock:
            worker.touched = time.monotonic()
            worker.principal = p
            args = dict(arguments)
            if any(key in args for key in INPUT_PATHS) or any(
                key in args for key in OUTPUT_PATHS
            ):
                raise ServiceError(
                    "invalid_path",
                    "Hosted tools accept artifact IDs, not filesystem paths.",
                )
            root = Path(worker.directory.name)
            for original, remote in INPUT_PATHS.items():
                artifact_id = args.pop(remote, None)
                if artifact_id:
                    row, data = await self.artifacts.read(p, artifact_id)
                    target = root / "inputs" / str(UUID(artifact_id))
                    target.mkdir(parents=True, exist_ok=True)
                    if row["data"]["name"].lower().endswith(".zip"):
                        # A fresh directory avoids mutation of already acknowledged inputs.
                        target = target / str(uuid4())
                        target.mkdir()
                        unpack_project(data, target, self.settings)
                        args[original] = str(target)
                    else:
                        target = target / Path(row["data"]["name"]).name
                        target.write_bytes(data)
                        args[original] = str(
                            target.parent
                            if name in {"format_kcl", "lint_and_fix_kcl"}
                            and original == "kcl_path"
                            else target
                        )
            from zoo_mcp.server import mcp

            tool = mcp._tool_manager.get_tool(name)
            if not tool:
                raise ServiceError("unknown_tool", "Unknown Zoo tool.")
            for output in OUTPUT_PATHS:
                args.pop(output, None)
                if output in tool.parameters.get("properties", {}):
                    args[output] = None
            credential = await self.backend.delegate(p)
            assert (
                worker.process.stdin is not None and worker.process.stdout is not None
            )
            worker.process.stdin.write(
                (
                    json.dumps(
                        {"credential": credential, "tool": name, "arguments": args}
                    )
                    + "\n"
                ).encode()
            )
            try:
                async with asyncio.timeout(self.settings.operation_seconds):
                    await worker.process.stdin.drain()
                    line = await worker.process.stdout.readline()
                    if not line:
                        raise ServiceError(
                            "worker_failed",
                            "The modeling worker stopped. Restore the scene from saved source.",
                        )
                    result = json.loads(line)
            except (TimeoutError, asyncio.CancelledError):
                await self.discard(worker)
                raise
            if "error" in result:
                raise ServiceError("operation_failed", result["error"])
            worker.touched = time.monotonic()
            collected = await self._collect(p, root, result)
            if name in {"format_kcl", "lint_and_fix_kcl"} and args.get("kcl_path"):
                source = Path(args["kcl_path"])
                collected["source"] = await self.artifacts.store_project(p, source)
            return collected

    async def _collect(self, p: Principal, root: Path, result: dict) -> dict:
        saved: dict[str, dict] = {}

        async def convert(value):
            if isinstance(value, str):
                if value.startswith(str(root)):
                    path = Path(value).resolve()
                    if not path.is_relative_to(root.resolve()) or not path.is_file():
                        raise ServiceError(
                            "invalid_output",
                            "The worker returned an invalid output file.",
                        )
                    if path.stat().st_size > self.settings.max_file_bytes:
                        raise ServiceError(
                            "file_too_large",
                            "The generated output exceeds the file limit.",
                        )
                    if value not in saved:
                        saved[value] = self.artifacts.describe(
                            p,
                            await self.artifacts.store(p, path.name, path.read_bytes()),
                        )
                    return saved[value]
                return value.replace(str(root), "<workspace>")
            if isinstance(value, dict):
                if value.get("type") == "image" and "data" in value:
                    import base64

                    image_data = base64.b64decode(value["data"], validate=True)
                    extension = "png" if value.get("mimeType") == "image/png" else "jpg"
                    artifact = self.artifacts.describe(
                        p,
                        await self.artifacts.store(
                            p, f"snapshot.{extension}", image_data
                        ),
                    )
                    saved[artifact["artifact_id"]] = artifact
                    return artifact
                return {k: await convert(v) for k, v in value.items()}
            if isinstance(value, list):
                return [await convert(v) for v in value]
            return value

        structured = await convert(result.get("structured"))
        content = []
        for block in result.get("content", []):
            if block.get("type") == "image":
                import base64

                image = base64.b64decode(block["data"], validate=True)
                extension = "png" if block.get("mimeType") == "image/png" else "jpg"
                artifact = self.artifacts.describe(
                    p, await self.artifacts.store(p, f"snapshot.{extension}", image)
                )
                saved[artifact["artifact_id"]] = artifact
                block = {
                    "type": "resource_link",
                    "uri": artifact["download_url"],
                    "name": artifact["name"],
                    "mimeType": block.get("mimeType"),
                }
            if block.get("type") == "text":
                value = block["text"]
                try:
                    value = json.loads(value)
                except (ValueError, TypeError):
                    pass
                value = await convert(value)
                block = {
                    "type": "text",
                    "text": value if isinstance(value, str) else json.dumps(value),
                }
            content.append(block)
        return {
            "content": content,
            "data": structured,
            "artifacts": list(saved.values()),
        }

    async def call(self, p: Principal, name: str, arguments: dict) -> dict:
        if name == "get_modeling_sessions":
            rows = await self.backend.list(p, "session")
            return {
                "sessions": [
                    r["data"]["scene_id"]
                    for r in rows
                    if r["data"].get("expires_at", 0) > time.time()
                ]
            }
        session_id = arguments.get("session_id")
        worker = None
        if session_id:
            row = await self.backend.get(p, str(UUID(session_id)))
            if (
                row["kind"] != "session"
                or row["data"].get("expires_at", 0) <= time.time()
            ):
                raise ServiceError(
                    "session_expired",
                    "Restore this scene from saved KCL or a Zoo project.",
                )
            if row["data"]["node"] != self.settings.node_url:
                return await self.forward(p, row["data"]["node"], name, arguments)
            worker = next(
                (
                    w
                    for w in self.workers.values()
                    if w.scene_id == session_id and w.principal.grant_id == p.grant_id
                ),
                None,
            )
            if worker is None:
                raise ServiceError(
                    "session_expired",
                    "Restore this scene from saved source; the worker has restarted.",
                )
        if worker is None:
            if name == "start_modeling_session":
                rows = await self.backend.list(p, "session")
                if (
                    sum(r["data"].get("expires_at", 0) > time.time() for r in rows)
                    >= self.settings.max_sessions_per_grant
                ):
                    raise ServiceError(
                        "session_limit",
                        "Close an existing modeling session before opening another.",
                    )
            worker = await self.start_worker(p)
        transient = not session_id and name != "start_modeling_session"
        try:
            result = await self.invoke(p, worker, name, arguments)
            if name == "start_modeling_session":
                data = result.get("data")
                scene_id = data.get("result") if isinstance(data, dict) else None
                if not scene_id:
                    scene_id = result["content"][0]["text"].strip('"')
                worker.scene_id = str(UUID(scene_id))
                await self.backend.put(
                    p,
                    worker.scene_id,
                    "session",
                    {
                        "scene_id": worker.scene_id,
                        "node": self.settings.node_url,
                        "expires_at": time.time() + self.settings.idle_seconds,
                    },
                )
                result["session_id"] = worker.scene_id
            elif name == "stop_modeling_session":
                assert session_id is not None
                await self.backend.delete(p, session_id)
                await self.discard(worker)
            elif session_id:
                row = await self.backend.get(p, session_id)
                await self.backend.put(
                    p,
                    session_id,
                    "session",
                    {
                        **row["data"],
                        "expires_at": time.time() + self.settings.idle_seconds,
                    },
                    row["revision"],
                )
            return result
        except BaseException:
            if name == "start_modeling_session":
                await self.discard(worker)
            raise
        finally:
            if transient:
                await self.discard(worker)

    async def forward(
        self, p: Principal, node: str, name: str, arguments: dict
    ) -> dict:
        # Destinations originate only in service-authenticated records. Restrict them
        # further to the configured cluster worker CIDR in production ingress/egress.
        import ipaddress
        from urllib.parse import urlsplit

        url = urlsplit(node)
        try:
            valid = (
                url.scheme == "http"
                and url.port == 8080
                and not url.username
                and not url.password
                and not url.query
                and not url.fragment
                and url.path in {"", "/"}
                and ipaddress.ip_address(url.hostname or "").is_private
            )
        except ValueError:
            valid = False
        if not valid:
            raise ServiceError(
                "invalid_worker", "The modeling worker address is invalid."
            )
        try:
            response = await self.backend.http.post(
                node + "/_internal/call",
                headers={
                    "X-Zoo-Mcp-Service-Token": self.settings.service_secret,
                    "Authorization": f"Bearer {p.token}",
                },
                json={"name": name, "arguments": arguments},
                timeout=self.settings.operation_seconds + 10,
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            raise ServiceError(
                "session_expired",
                "The owning worker is unavailable. Restore the scene from saved source.",
            ) from None

    async def run_direct(
        self, operation, disconnected: asyncio.Event | None = None
    ) -> dict:
        if self.closing or len(self.direct_tasks) >= self.settings.max_workers * 4:
            raise ServiceError(
                "busy", "Zoo modeling capacity is busy. Try again shortly."
            )

        async def bounded():
            async with asyncio.timeout(self.settings.operation_seconds):
                return await operation()

        task = asyncio.create_task(bounded())
        self.direct_tasks.add(task)

        def complete(done):
            self.direct_tasks.discard(done)
            if not done.cancelled():
                done.exception()  # Consume failures even when the caller disconnected.

        task.add_done_callback(complete)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Losing the SSE connection is not an explicit cancellation. Work may
            # finish within its deadline, but there is no replay or durable result.
            if disconnected is None or not disconnected.is_set():
                task.cancel()
            raise

    async def submit(self, p: Principal, name: str, arguments: dict, operation) -> dict:
        key = arguments.get("idempotency_key")
        if not isinstance(key, str) or not key or len(key) > 128:
            raise ServiceError(
                "idempotency_required",
                "Supply a unique idempotency_key for this operation; reuse it only when retrying the same request.",
            )
        job_id = str(uuid5(UUID(p.grant_id), key))
        digest = hashlib.sha256(
            json.dumps([name, arguments], sort_keys=True).encode()
        ).hexdigest()
        try:
            existing = await self.backend.get(p, job_id)
        except ServiceError as error:
            if error.code != "not_found":
                raise
        else:
            if existing["kind"] != "job" or existing["data"]["digest"] != digest:
                raise ServiceError(
                    "idempotency_conflict",
                    "This idempotency key belongs to a different request.",
                )
            return await self.job(p, job_id)
        rows = await self.backend.list(p, "job")
        if (
            sum(
                r["data"].get("status") == "running"
                and r["data"].get("deadline", 0) > time.time()
                for r in rows
            )
            >= self.settings.max_jobs_per_grant
        ):
            raise ServiceError("job_limit", "Wait for an existing operation to finish.")
        try:
            row = await self.backend.put(
                p,
                job_id,
                "job",
                {
                    "status": "running",
                    "digest": digest,
                    "node": self.settings.node_url,
                    "deadline": time.time() + self.settings.operation_seconds + 30,
                },
            )
        except ServiceError as error:
            if error.code != "conflict":
                raise
            # A simultaneous retry may have won the create on another pod.
            existing = await self.backend.get(p, job_id)
            if existing["kind"] != "job" or existing["data"].get("digest") != digest:
                raise ServiceError(
                    "idempotency_conflict", "This key belongs to another operation."
                ) from None
            return await self.job(p, job_id)

        async def execute():
            try:
                # Revalidate grant just before any work; never persist credentials.
                await self.backend.principal(p.token)
                async with asyncio.timeout(self.settings.operation_seconds):
                    result = await operation()
                data = {**row["data"], "status": "completed", "result": result}
            except asyncio.CancelledError:
                data = {**row["data"], "status": "cancelled"}
            except Exception as exc:
                data = {
                    **row["data"],
                    "status": "failed",
                    "error": exc.code
                    if isinstance(exc, ServiceError)
                    else "operation_failed",
                }
            self.outcomes[data["status"]] += 1
            try:
                # Large graphs and reports belong in artifact storage, not a database row.
                if len(json.dumps(data).encode()) > 900_000:
                    artifact = self.artifacts.describe(
                        p,
                        await self.artifacts.store(
                            p, "result.json", json.dumps(data["result"]).encode()
                        ),
                    )
                    data["result"] = {"result_artifact": artifact}
                await self.backend.put(p, job_id, "job", data, row["revision"])
            except Exception:
                self.outcomes["unrecorded"] += 1
                logger.warning("job_outcome_unrecorded")
            finally:
                self.jobs.pop(job_id, None)

        task = asyncio.create_task(execute())
        self.jobs[job_id] = task
        await asyncio.wait({task}, timeout=2)
        return await self.job(p, job_id)

    async def job(self, p: Principal, job_id: str) -> dict:
        row = await self.backend.get(p, str(UUID(job_id)))
        if row["kind"] != "job":
            raise ServiceError("not_found", "Job not found.")
        data = row["data"]
        if data["status"] == "running" and data["deadline"] < time.time():
            data = {
                **data,
                "status": "interrupted",
                "error": "Worker stopped before recording an outcome. Inspect the project or scene before retrying.",
            }

        async def fresh(value):
            if isinstance(value, dict):
                if "artifact_id" in value and "download_url" in value:
                    try:
                        return self.artifacts.describe(
                            p, await self.backend.get(p, value["artifact_id"])
                        )
                    except ServiceError as error:
                        if error.code != "not_found":
                            raise
                        return {
                            "artifact_id": value["artifact_id"],
                            "unavailable": True,
                        }
                return {k: await fresh(v) for k, v in value.items()}
            if isinstance(value, list):
                return [await fresh(v) for v in value]
            return value

        return await fresh(
            {
                "job_id": job_id,
                **{k: v for k, v in data.items() if k not in {"node", "digest"}},
            }
        )

    async def cancel(self, p: Principal, job_id: str) -> dict:
        row = await self.backend.get(p, str(UUID(job_id)))
        if row["kind"] != "job":
            raise ServiceError("not_found", "Job not found.")
        if row["data"]["node"] != self.settings.node_url:
            return await self.forward(
                p, row["data"]["node"], "cancel_job", {"job_id": job_id}
            )
        task = self.jobs.get(job_id)
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return await self.job(p, job_id)

    async def maintain(self):
        while not self.closing:
            await asyncio.sleep(30)
            for worker in list(self.workers.values()):
                try:
                    response = await self.backend._request(
                        "POST",
                        "/mcp/lease",
                        json={"grant_id": worker.principal.grant_id},
                    )
                    if not response.json().get("active"):
                        await self.discard(worker)
                        continue
                    if (
                        not worker.lock.locked()
                        and time.monotonic() - worker.touched
                        > self.settings.idle_seconds
                    ):
                        await self.discard(worker)
                except Exception:
                    await self.discard(worker)
            try:
                await self.backend._request("POST", "/mcp/cleanup")
            except Exception:
                self.outcomes["cleanup_failed"] += 1
                logger.warning("artifact_cleanup_failed")

    async def close(self):
        self.closing = True
        for task in list(self.jobs.values()):
            task.cancel()
        await asyncio.gather(*self.jobs.values(), return_exceptions=True)
        for task in list(self.direct_tasks):
            task.cancel()
        await asyncio.gather(*self.direct_tasks, return_exceptions=True)
        await asyncio.gather(
            *(w.close() for w in self.workers.values()), return_exceptions=True
        )
        self.workers.clear()
