from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


def _norm(name: str) -> str:
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


@dataclass
class MCPProbeStatus:
    active: bool
    tools: list[str]
    reason: str | None = None


@dataclass
class _RpcWaiter:
    event: threading.Event
    response: Any = None
    error: Any = None


class _MCPStdioClient:
    def __init__(
        self,
        command: str,
        args: list[str],
        cwd: str | None,
        env_overrides: dict[str, str],
        timeout_sec: int,
        message_format: str = "ndjson",
    ):
        self.command = command
        self.args = args
        self.cwd = cwd
        self.env_overrides = env_overrides
        self.timeout_sec = timeout_sec
        fmt = (message_format or "ndjson").strip().lower()
        self.message_format = fmt if fmt in {"ndjson", "framed"} else "ndjson"
        self._proc: subprocess.Popen | None = None
        self._write_lock = threading.Lock()
        self._pending: dict[str, _RpcWaiter] = {}
        self._pending_lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail: list[str] = []
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        cmd = [self.command] + list(self.args or [])
        env = os.environ.copy()
        env.update(self.env_overrides or {})
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
            env=env,
            bufsize=0,
        )
        self._running = True
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_thread.start()

    def close(self) -> None:
        self._running = False
        proc = self._proc
        self._proc = None
        if proc:
            try:
                proc.terminate()
            except Exception:
                pass
        with self._pending_lock:
            for waiter in self._pending.values():
                waiter.error = {"code": -1, "message": "mcp_stdio_closed"}
                waiter.event.set()
            self._pending.clear()

    def rpc(self, method: str, params: dict[str, Any]) -> Any:
        self.start()
        rid = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": rid,
            "method": method,
            "params": params,
        }
        encoded = self._encode_outbound(payload)
        waiter = _RpcWaiter(event=threading.Event())
        with self._pending_lock:
            self._pending[rid] = waiter
        try:
            with self._write_lock:
                if not self._proc or not self._proc.stdin:
                    raise RuntimeError("mcp_stdio_not_running")
                self._proc.stdin.write(encoded)
                self._proc.stdin.flush()
            if not waiter.event.wait(timeout=self.timeout_sec):
                raise TimeoutError(f"mcp_stdio_timeout:{method}")
            if waiter.error:
                raise RuntimeError(json.dumps(waiter.error, ensure_ascii=False))
            return waiter.response
        finally:
            with self._pending_lock:
                self._pending.pop(rid, None)

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self.start()
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }
        encoded = self._encode_outbound(payload)
        with self._write_lock:
            if self._proc and self._proc.stdin:
                self._proc.stdin.write(encoded)
                self._proc.stdin.flush()

    def _encode_outbound(self, payload: dict[str, Any]) -> bytes:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if self.message_format == "framed":
            return (
                f"Content-Length: {len(body)}\r\n"
                f"Content-Type: application/json\r\n\r\n"
            ).encode("ascii") + body
        return body + b"\n"

    def tail_stderr(self) -> str:
        return "\n".join(self._stderr_tail[-8:]).strip()

    def _stderr_loop(self) -> None:
        proc = self._proc
        if not proc or not proc.stderr:
            return
        try:
            while self._running:
                line = proc.stderr.readline()
                if not line:
                    break
                txt = line.decode("utf-8", errors="replace").strip()
                if not txt:
                    continue
                self._stderr_tail.append(txt)
                if len(self._stderr_tail) > 200:
                    self._stderr_tail = self._stderr_tail[-200:]
        except Exception:
            return

    def _reader_loop(self) -> None:
        proc = self._proc
        if not proc or not proc.stdout:
            return
        buf = b""
        try:
            while self._running:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                buf += chunk
                # framed parser first
                while True:
                    head_end = buf.find(b"\r\n\r\n")
                    if head_end == -1:
                        break
                    head = buf[:head_end].decode("ascii", errors="ignore")
                    content_length = None
                    for line in head.split("\r\n"):
                        if ":" not in line:
                            continue
                        k, v = line.split(":", 1)
                        if k.strip().lower() == "content-length":
                            try:
                                content_length = int(v.strip())
                            except Exception:
                                content_length = None
                    if content_length is None:
                        break
                    total_len = head_end + 4 + content_length
                    if len(buf) < total_len:
                        break
                    body = buf[head_end + 4:total_len]
                    buf = buf[total_len:]
                    self._handle_message_bytes(body)

                # fallback: line-delimited json
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(b"{") and line.endswith(b"}"):
                        self._handle_message_bytes(line)
        finally:
            self._running = False
            with self._pending_lock:
                for waiter in self._pending.values():
                    waiter.error = {"code": -1, "message": "mcp_stdio_reader_stopped"}
                    waiter.event.set()
                self._pending.clear()

    def _handle_message_bytes(self, body: bytes) -> None:
        try:
            msg = json.loads(body.decode("utf-8", errors="replace"))
        except Exception:
            return
        if not isinstance(msg, dict):
            return
        rid = msg.get("id")
        if rid is None:
            return
        rid = str(rid)
        with self._pending_lock:
            waiter = self._pending.get(rid)
        if not waiter:
            return
        if "error" in msg and msg.get("error") is not None:
            waiter.error = msg.get("error")
        else:
            waiter.response = msg.get("result")
        waiter.event.set()


class MCPToolBackend:
    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        tb_cfg = self.cfg.get("tool_backend") or {}
        mcp_cfg = self.cfg.get("mcp") or {}

        self.mode = str(tb_cfg.get("mode", "auto")).lower()
        if self.mode not in {"auto", "mcp", "local"}:
            self.mode = "auto"

        self.enabled = bool(mcp_cfg.get("enabled", False))
        self.transport = str(mcp_cfg.get("transport", "auto")).lower()
        if self.transport not in {"auto", "http", "stdio"}:
            self.transport = "auto"

        self.endpoint = str(mcp_cfg.get("endpoint") or "").strip()
        self.timeout_sec = int(mcp_cfg.get("timeout_sec", 20))
        self.list_method = str(mcp_cfg.get("list_method") or "tools/list")
        self.call_method = str(mcp_cfg.get("call_method") or "tools/call")

        stdio_cfg = mcp_cfg.get("stdio") or {}
        self.stdio_command = str(stdio_cfg.get("command") or "").strip()
        self.stdio_args = [str(a) for a in (stdio_cfg.get("args") or [])]
        self.stdio_cwd = str(stdio_cfg.get("cwd") or "").strip()
        stdio_env = stdio_cfg.get("env") or {}
        self.stdio_env = {str(k): str(v) for k, v in stdio_env.items()} if isinstance(stdio_env, dict) else {}
        self.initialize_on_connect = bool(stdio_cfg.get("initialize_on_connect", True))
        self.initialize_method = str(stdio_cfg.get("initialize_method") or "initialize")
        self.initialized_notify_method = str(
            stdio_cfg.get("initialized_notify_method") or "notifications/initialized"
        )
        self.tool_call_param_order = str(stdio_cfg.get("tool_call_param_order") or "name_arguments")
        self.stdio_message_format = str(stdio_cfg.get("message_format") or "ndjson").strip().lower()
        if self.stdio_message_format not in {"ndjson", "framed"}:
            self.stdio_message_format = "ndjson"
        self._stdio_client: _MCPStdioClient | None = None
        self._active_transport: str | None = None

        default_aliases = {
            "weather.forecast": [
                "openweather.get_forecast",
                "weather.get_forecast",
                "weather.forecast",
                "weather",
            ],
            "weather.get_forecast": [
                "openweather.get_forecast",
                "weather.get_forecast",
                "weather.forecast",
                "weather",
            ],
            "websearch.search": [
                "duckduckgo.search",
                "duckduckgo.web_search",
                "websearch.search",
                "search",
                "wikipedia.search",
            ],
            "filesystem.list": [
                "filesystem.list",
                "filesystem.list_dir",
                "list_directory",
                "directory_tree",
                "obsidian.list",
            ],
            "filesystem.read": [
                "filesystem.read",
                "filesystem.read_file",
                "read_file",
                "obsidian.read",
            ],
            "filesystem.tail": [
                "filesystem.tail",
                "filesystem.read_tail",
                "read_file",
                "filesystem.read_file",
            ],
            "filesystem.find": [
                "filesystem.find",
                "filesystem.search_files",
                "find_files",
                "search_files",
            ],
            "filesystem.grep": [
                "filesystem.grep",
                "filesystem.search_text",
                "search_text",
                "search_content",
            ],
            "filesystem.write": [
                "filesystem.write",
                "filesystem.write_file",
                "write_file",
                "create_new_file_with_text",
                "edit_file",
            ],
            "filesystem.mkdir": [
                "filesystem.mkdir",
                "filesystem.make_dir",
                "make_directory",
                "create_directory",
            ],
            "filesystem.delete": [
                "filesystem.delete",
                "filesystem.delete_path",
                "delete_file",
                "delete_path",
            ],
            "obsidian.list": [
                "obsidian_list_files_in_vault",
                "obsidian_list_files_in_dir",
            ],
            "obsidian.read": [
                "obsidian_get_file_contents",
            ],
            "obsidian.write": [
                "obsidian_append_content",
            ],
            "obsidian.delete": [
                "obsidian_delete_file",
            ],
            "time.now": [
                "time.now",
                "time.get_time",
            ],
            "wikipedia.search": [
                "wikipedia.search",
                "wiki.search",
            ],
        }
        user_aliases = mcp_cfg.get("aliases") or {}
        for key, values in user_aliases.items():
            if isinstance(values, list):
                default_aliases[str(key)] = [str(v) for v in values]
        self.aliases = default_aliases

        self._active = False
        self._reason = "not_probed"
        self._tool_names: list[str] = []
        self._tool_norm_to_name: dict[str, str] = {}

    def close(self) -> None:
        if self._stdio_client:
            self._stdio_client.close()
            self._stdio_client = None
        self._active = False

    def probe(self) -> MCPProbeStatus:
        if not self.enabled:
            self._active = False
            self._reason = "disabled"
            return MCPProbeStatus(active=False, tools=[], reason=self._reason)

        transports = self._transport_order()
        last_error = "no_transport_available"
        for t in transports:
            try:
                if t == "http":
                    if not self.endpoint:
                        raise RuntimeError("missing_endpoint")
                    result = self._rpc_http(self.list_method, {})
                else:
                    result = self._rpc_stdio(self.list_method, {}, initialize=True)
                tools = self._extract_tool_names(result)
                if not tools:
                    raise RuntimeError("no_tools")
                self._active = True
                self._reason = None
                self._active_transport = t
                self._tool_names = tools
                self._tool_norm_to_name = {_norm(n): n for n in tools}
                return MCPProbeStatus(active=True, tools=tools, reason=None)
            except Exception as e:
                last_error = str(e)
                if t == "stdio":
                    self.close()
                continue

        self._active = False
        self._reason = last_error
        self._active_transport = None
        self._tool_names = []
        self._tool_norm_to_name = {}
        return MCPProbeStatus(active=False, tools=[], reason=self._reason)

    @property
    def active(self) -> bool:
        return self._active

    @property
    def reason(self) -> str | None:
        return self._reason

    @property
    def tools(self) -> list[str]:
        return list(self._tool_names)

    @property
    def active_transport(self) -> str | None:
        return self._active_transport

    def select_backend(self, tool_name: str) -> tuple[str, str | None]:
        if self.mode == "local":
            return "local", None
        if self.mode == "mcp":
            if not self.active:
                return "mcp", f"mcp_inactive:{self.reason or 'unknown'}"
            if self.can_handle(tool_name):
                return "mcp", None
            return "mcp", "mcp_tool_unavailable"
        if self.can_handle(tool_name):
            return "mcp", None
        if self.active:
            return "local", "mcp_tool_unavailable"
        return "local", f"mcp_inactive:{self.reason or 'unknown'}"

    def can_handle(self, tool_name: str) -> bool:
        if not self.active:
            return False
        return self.resolve_tool_name(tool_name) is not None

    def resolve_tool_name(self, tool_name: str) -> str | None:
        if not self.active:
            return None
        candidates = [tool_name]
        candidates.extend(self.aliases.get(tool_name, []))
        for c in candidates:
            if c in self._tool_names:
                return c
            n = _norm(c)
            if n in self._tool_norm_to_name:
                return self._tool_norm_to_name[n]
        return None

    def call_tool(self, tool_name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        if not self.active:
            return {"ok": False, "error": f"mcp_inactive:{self.reason or 'unknown'}", "result": None}
        resolved = self.resolve_tool_name(tool_name)
        if not resolved:
            return {"ok": False, "error": f"mcp_tool_unavailable:{tool_name}", "result": None}
        args = arguments or {}
        call_shapes = self._call_shapes(resolved, args)
        last_error = None
        for params in call_shapes:
            try:
                if self._active_transport == "http":
                    result = self._rpc_http(self.call_method, params)
                else:
                    result = self._rpc_stdio(self.call_method, params, initialize=False)
                return {
                    "ok": True,
                    "error": None,
                    "result": self._normalize_call_result(result),
                    "mcp_tool": resolved,
                    "transport": self._active_transport,
                }
            except Exception as e:
                last_error = str(e)
                continue
        return {
            "ok": False,
            "error": last_error or "mcp_call_failed",
            "result": None,
            "mcp_tool": resolved,
            "transport": self._active_transport,
        }

    def _transport_order(self) -> list[str]:
        if self.transport == "http":
            return ["http"]
        if self.transport == "stdio":
            return ["stdio"]
        # auto: try HTTP first, then stdio.
        return ["http", "stdio"]

    def _rpc_http(self, method: str, params: dict[str, Any]) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params,
        }
        r = requests.post(self.endpoint, json=payload, timeout=self.timeout_sec)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(json.dumps(data.get("error"), ensure_ascii=False))
        if isinstance(data, dict) and "result" in data:
            return data.get("result")
        return data

    def _rpc_stdio(self, method: str, params: dict[str, Any], initialize: bool) -> Any:
        if not self.stdio_command:
            raise RuntimeError("missing_stdio_command")
        if self._stdio_client is None:
            cwd = self.stdio_cwd or None
            if cwd and not Path(cwd).is_absolute():
                cwd = str((Path(__file__).resolve().parents[2] / cwd).resolve())
            self._stdio_client = _MCPStdioClient(
                command=self.stdio_command,
                args=self.stdio_args,
                cwd=cwd,
                env_overrides=self.stdio_env,
                timeout_sec=self.timeout_sec,
                message_format=self.stdio_message_format,
            )
        try:
            if initialize and self.initialize_on_connect:
                self._initialize_stdio()
            return self._stdio_client.rpc(method, params)
        except Exception as e:
            tail = ""
            try:
                tail = self._stdio_client.tail_stderr() if self._stdio_client else ""
            except Exception:
                tail = ""
            if tail:
                raise RuntimeError(f"{e}; stderr_tail={tail}") from e
            raise

    def _initialize_stdio(self) -> None:
        if not self._stdio_client:
            return
        init_params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "ALON", "version": "1.0"},
        }
        try:
            self._stdio_client.rpc(self.initialize_method, init_params)
        except Exception:
            # Some servers may not require/accept initialize; continue.
            return
        try:
            self._stdio_client.notify(self.initialized_notify_method, {})
        except Exception:
            return

    def _call_shapes(self, resolved: str, args: dict[str, Any]) -> list[dict[str, Any]]:
        if self.tool_call_param_order == "tool_args":
            return [
                {"tool": resolved, "args": args},
                {"name": resolved, "arguments": args},
                {"tool_name": resolved, "arguments": args},
            ]
        return [
            {"name": resolved, "arguments": args},
            {"tool": resolved, "args": args},
            {"tool_name": resolved, "arguments": args},
        ]

    @staticmethod
    def _extract_tool_names(result: Any) -> list[str]:
        tools = []
        if isinstance(result, dict):
            if isinstance(result.get("tools"), list):
                tools = result.get("tools") or []
            elif isinstance(result.get("items"), list):
                tools = result.get("items") or []
        elif isinstance(result, list):
            tools = result
        out: list[str] = []
        for t in tools:
            if isinstance(t, str):
                out.append(t)
            elif isinstance(t, dict):
                name = t.get("name") or t.get("tool") or t.get("id")
                if name:
                    out.append(str(name))
        seen = set()
        uniq = []
        for n in out:
            if n in seen:
                continue
            seen.add(n)
            uniq.append(n)
        return uniq

    @staticmethod
    def _normalize_call_result(result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            out = {"raw": result}
            text_chunks: list[str] = []
            content = result.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        txt = item.get("text")
                        if txt:
                            text_chunks.append(str(txt))
            if text_chunks:
                out["text"] = "\n".join(text_chunks).strip()
            if isinstance(result.get("structuredContent"), dict):
                out["structured"] = result.get("structuredContent")
            return out
        if isinstance(result, list):
            return {"raw": result, "text": json.dumps(result, ensure_ascii=False)}
        return {"raw": result, "text": str(result)}
