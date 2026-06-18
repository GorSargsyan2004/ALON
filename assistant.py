from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from llm_client import LMStudioClient, mcp_tools_to_openai
from mcp_client import MCPClient, MCPClientError, MCPInactiveError


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("assistant")


class ToolAssistant:
    def __init__(
        self,
        llm: LMStudioClient,
        mcp: MCPClient,
        max_tool_rounds: int = 6,
    ):
        self.llm = llm
        self.mcp = mcp
        self.max_tool_rounds = max_tool_rounds
        self.mcp_active = False
        self._openai_tools: list[dict[str, Any]] = []

    async def start(self) -> None:
        await self.llm.start()
        try:
            await self.mcp.start()
            tools = await self.mcp.list_tools()
            self._openai_tools = mcp_tools_to_openai(tools)
            self.mcp_active = True
            log.info("MCP active with %d tools", len(self._openai_tools))
        except Exception as e:
            self.mcp_active = False
            self._openai_tools = []
            log.warning("MCP unavailable, fallback to no-tools mode: %s", e)

    async def close(self) -> None:
        await self.mcp.close()
        await self.llm.close()

    async def chat_with_tools(self, messages: list[dict[str, Any]]) -> str:
        rounds = 0
        while rounds < self.max_tool_rounds:
            rounds += 1
            tools = self._openai_tools if self.mcp_active else None
            assistant_message = await self.llm.chat_completion(
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            tool_calls = assistant_message.get("tool_calls") or []
            content = assistant_message.get("content")

            # No tool calls: final assistant output.
            if not tool_calls or not self.mcp_active:
                final = (content or "").strip()
                return final if final else "I don't have a response."

            messages.append(
                {
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": tool_calls,
                }
            )

            for call in tool_calls:
                call_id = call.get("id")
                func = (call.get("function") or {})
                tool_name = func.get("name")
                args_raw = func.get("arguments") or "{}"

                try:
                    tool_args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                    if not isinstance(tool_args, dict):
                        tool_args = {}
                except Exception:
                    tool_args = {}

                try:
                    result = await self.mcp.call_tool(tool_name, tool_args)
                    tool_content = self._serialize_tool_result(result)
                except (MCPInactiveError, MCPClientError, Exception) as e:
                    self.mcp_active = False
                    self._openai_tools = []
                    log.warning("MCP failed; switching to fallback no-tools mode: %s", e)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": tool_name,
                            "content": f"Tool execution failed: {e}",
                        }
                    )
                    # Continue chat gracefully without tools on next loop
                    break

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": tool_content,
                    }
                )

        return "I reached the tool-call iteration limit. Please refine the request."

    @staticmethod
    def _serialize_tool_result(result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, list):
            chunks = []
            for item in result:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        chunks.append(str(item.get("text") or ""))
                    else:
                        chunks.append(json.dumps(item, ensure_ascii=False))
                else:
                    chunks.append(str(item))
            return "\n".join(c for c in chunks if c).strip()
        return json.dumps(result, ensure_ascii=False)


async def _main() -> None:
    llm = LMStudioClient(base_url="http://localhost:1234/v1")
    mcp = MCPClient(command=["docker", "mcp", "gateway", "run"])
    assistant = ToolAssistant(llm=llm, mcp=mcp)

    await assistant.start()
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Use tools when needed. Keep answers concise. "
                "If tool data is unavailable, say that clearly."
            ),
        }
    ]

    try:
        while True:
            user = input("You: ").strip()
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                break
            messages.append({"role": "user", "content": user})
            reply = await assistant.chat_with_tools(messages)
            messages.append({"role": "assistant", "content": reply})
            print(f"Assistant: {reply}")
    finally:
        await assistant.close()


if __name__ == "__main__":
    asyncio.run(_main())
