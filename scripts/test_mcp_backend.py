from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from services.orchestrator.mcp_backend import MCPToolBackend  # noqa: E402


def load_cfg() -> dict:
    cfg_path = ROOT / "config" / "default.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Test ALON MCP backend connectivity.")
    parser.add_argument(
        "--tool",
        help="Optional canonical tool name to call after probe (e.g. websearch.search).",
    )
    parser.add_argument(
        "--args-json",
        default="{}",
        help='JSON args for --tool, e.g. \'{"query":"USD AMD"}\'',
    )
    parser.add_argument(
        "--require-active",
        action="store_true",
        help="Exit non-zero if MCP is not active.",
    )
    args = parser.parse_args()

    cfg = load_cfg()
    backend = MCPToolBackend(cfg)

    status = backend.probe()
    print(f"mode={backend.mode}")
    print(f"active={status.active}")
    print(f"transport={backend.active_transport}")
    print(f"reason={status.reason}")
    print(f"tools_count={len(status.tools)}")
    if status.tools:
        preview = ", ".join(status.tools[:20])
        print(f"tools_preview={preview}")
    else:
        if status.reason and "timeout" in status.reason.lower():
            print("hint=stdio server started but did not answer tools/list in time; check command/method/transport")
        elif status.reason and "missing_endpoint" in status.reason.lower():
            print("hint=set mcp.endpoint for HTTP transport, or use stdio transport")
        elif status.reason and "missing_stdio_command" in status.reason.lower():
            print("hint=set mcp.stdio.command + mcp.stdio.args")

    if args.require_active and not status.active:
        return 2

    if not args.tool:
        return 0 if status.active or not args.require_active else 2

    try:
        tool_args = json.loads(args.args_json or "{}")
        if not isinstance(tool_args, dict):
            raise ValueError("args-json must decode to object")
    except Exception as e:
        print(f"invalid_args_json={e}")
        return 3

    selected, reason = backend.select_backend(args.tool)
    resolved = backend.resolve_tool_name(args.tool)
    print(f"tool={args.tool}")
    print(f"selected_backend={selected}")
    print(f"resolve_reason={reason}")
    print(f"resolved_tool={resolved}")

    if selected != "mcp":
        print("tool_call_skipped=selected_backend_not_mcp")
        return 0

    result = backend.call_tool(args.tool, tool_args)
    print("call_ok=", bool(result.get("ok")))
    print("call_transport=", result.get("transport"))
    print("call_error=", result.get("error"))
    payload = result.get("result")
    if payload is not None:
        text = (payload.get("text") if isinstance(payload, dict) else str(payload)) or ""
        if len(text) > 500:
            text = text[:500] + "..."
        print("result_text_preview=", text)
    return 0 if result.get("ok") else 4


if __name__ == "__main__":
    raise SystemExit(main())
