from __future__ import annotations

import asyncio
import json
import logging
from itertools import count
from typing import Any


class MCPClientError(Exception):
    pass


class MCPInactiveError(MCPClientError):
    pass


class MCPProtocolError(MCPClientError):
    pass


class MCPClient:
    def __init__(
        self,
        command: list[str] | None = None,
        timeout_sec: float = 20.0,
        startup_timeout_sec: float = 30.0,
        message_format: str = "ndjson",
        log_stderr: bool = False,
        logger: logging.Logger | None = None,
    ):
        self.command = command or ["docker", "mcp", "gateway", "run"]
        self.timeout_sec = timeout_sec
        self.startup_timeout_sec = startup_timeout_sec
        fmt = (message_format or "ndjson").strip().lower()
        self.message_format = fmt if fmt in {"ndjson", "framed"} else "ndjson"
        self.log_stderr = bool(log_stderr)
        self.log = logger or logging.getLogger("mcp_client")

        self._proc: asyncio.subprocess.Process | None = None
        self._active = False
        self._id_counter = count(1)
        self._pending: dict[int, asyncio.Future] = {}
        self._pending_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._stdout_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._start_lock = asyncio.Lock()

    @property
    def active(self) -> bool:
        return self._active

    async def start(self) -> None:
        async with self._start_lock:
            if self._active:
                return
            if self._proc and self._proc.returncode is None:
                self._active = True
                return

            self._proc = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._stdout_task = asyncio.create_task(self._stdout_loop(), name="mcp-stdout-loop")
            self._stderr_task = asyncio.create_task(self._stderr_loop(), name="mcp-stderr-loop")

            try:
                await asyncio.wait_for(self._rpc("tools/list", {}, _skip_start=True), timeout=self.startup_timeout_sec)
            except Exception as e:
                await self._mark_inactive(f"startup_health_check_failed: {e}")
                raise MCPInactiveError(f"MCP start failed: {e}") from e
            self._active = True

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self._rpc("tools/list", {})
        tools = self._extract_tools(result)
        if not isinstance(tools, list):
            raise MCPProtocolError("tools/list result missing tools array")
        return tools

    async def call_tool(self, name: str, arguments: dict) -> Any:
        result = await self._rpc("tools/call", {"name": name, "arguments": arguments or {}})
        content = result.get("content") if isinstance(result, dict) else None
        if content is None:
            return result
        return content

    async def close(self) -> None:
        self._active = False
        for task in (self._stdout_task, self._stderr_task):
            if task and not task.done():
                task.cancel()
        self._stdout_task = None
        self._stderr_task = None

        async with self._pending_lock:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(MCPInactiveError("MCP client closed"))
            self._pending.clear()

        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        self._proc = None

    async def _rpc(self, method: str, params: dict[str, Any], _skip_start: bool = False) -> Any:
        if not self._active and method != "tools/list":
            raise MCPInactiveError("MCP is inactive")
        if not _skip_start:
            proc_dead = self._proc is None or self._proc.returncode is not None
            if proc_dead or (not self._active and method == "tools/list"):
                await self.start()
        req_id = next(self._id_counter)

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        async with self._pending_lock:
            self._pending[req_id] = fut

        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        payload = json.dumps(request, ensure_ascii=False).encode("utf-8")
        encoded = (
            (
                f"Content-Length: {len(payload)}\r\n"
                "Content-Type: application/json\r\n\r\n"
            ).encode("ascii")
            + payload
        ) if self.message_format == "framed" else payload + b"\n"

        try:
            async with self._write_lock:
                if not self._proc or not self._proc.stdin:
                    raise MCPInactiveError("MCP process is not running")
                self._proc.stdin.write(encoded)
                await self._proc.stdin.drain()
            response = await asyncio.wait_for(fut, timeout=self.timeout_sec)
            if "error" in response and response["error"] is not None:
                raise MCPClientError(f"MCP error: {response['error']}")
            return response.get("result")
        except asyncio.TimeoutError as e:
            raise MCPClientError(f"MCP timeout for method={method}") from e
        finally:
            async with self._pending_lock:
                self._pending.pop(req_id, None)

    async def _stdout_loop(self) -> None:
        if not self._proc or not self._proc.stdout:
            return
        stream = self._proc.stdout
        buffer = b""

        try:
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    await self._mark_inactive("stdout_closed")
                    return
                buffer += chunk

                # Parse framed JSON-RPC messages (Content-Length)
                while True:
                    head_end = buffer.find(b"\r\n\r\n")
                    if head_end < 0:
                        break
                    headers_blob = buffer[:head_end].decode("ascii", errors="ignore")
                    content_length = None
                    for line in headers_blob.split("\r\n"):
                        if ":" not in line:
                            continue
                        k, v = line.split(":", 1)
                        if k.strip().lower() == "content-length":
                            try:
                                content_length = int(v.strip())
                            except ValueError:
                                content_length = None
                    if content_length is None:
                        # Not a valid framed message, try line fallback below.
                        break
                    total = head_end + 4 + content_length
                    if len(buffer) < total:
                        break
                    body = buffer[head_end + 4:total]
                    buffer = buffer[total:]
                    await self._handle_message_bytes(body)

                # Fallback for line-delimited JSON-RPC
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if not (line.startswith(b"{") and line.endswith(b"}")):
                        continue
                    await self._handle_message_bytes(line)
        except asyncio.CancelledError:
            return
        except Exception as e:
            await self._mark_inactive(f"stdout_loop_error: {e}")

    async def _handle_message_bytes(self, body: bytes) -> None:
        try:
            msg = json.loads(body.decode("utf-8", errors="replace"))
        except Exception:
            await self._mark_inactive("invalid_json_on_stdout")
            raise MCPProtocolError("invalid JSON on MCP stdout")

        if not isinstance(msg, dict):
            return

        rid = msg.get("id")
        if rid is None:
            return
        try:
            rid_int = int(rid)
        except Exception:
            return

        async with self._pending_lock:
            fut = self._pending.get(rid_int)
        if fut and not fut.done():
            fut.set_result(msg)

    async def _stderr_loop(self) -> None:
        if not self._proc or not self._proc.stderr:
            return
        stream = self._proc.stderr
        try:
            while True:
                line = await stream.readline()
                if not line:
                    return
                msg = line.decode("utf-8", errors="replace").rstrip()
                if msg and self.log_stderr:
                    self.log.warning("[MCP STDERR] %s", msg)
        except asyncio.CancelledError:
            return
        except Exception as e:
            self.log.warning("stderr loop stopped: %s", e)

    async def _mark_inactive(self, reason: str) -> None:
        self._active = False
        async with self._pending_lock:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(MCPInactiveError(reason))
            self._pending.clear()

    @staticmethod
    def _extract_tools(result: Any) -> list[dict[str, Any]]:
        if isinstance(result, dict):
            if isinstance(result.get("tools"), list):
                return result["tools"]
            if isinstance(result.get("items"), list):
                return result["items"]
        if isinstance(result, list):
            return result
        return []
