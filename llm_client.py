from __future__ import annotations

import asyncio
from typing import Any

import aiohttp


class LMStudioClient:
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 600,
        timeout_sec: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_sec = float(timeout_sec)
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        if self._session and not self._session.closed:
            return
        timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
        self._session = aiohttp.ClientSession(timeout=timeout)
        if not self.model:
            self.model = await self._detect_model()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> dict[str, Any]:
        await self.start()
        if not self.model:
            raise RuntimeError("No LM Studio model configured and none detected")
        assert self._session is not None

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        url = f"{self.base_url}/chat/completions"
        async with self._session.post(url, json=payload) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"LM Studio error {resp.status}: {text}")
            data = await resp.json()

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("LM Studio returned no choices")
        message = (choices[0] or {}).get("message") or {}
        if not isinstance(message, dict):
            raise RuntimeError("Invalid LM Studio response: message is not object")
        return message

    async def _detect_model(self) -> str:
        assert self._session is not None
        url = f"{self.base_url}/models"
        async with self._session.get(url) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"LM Studio /models error {resp.status}: {text}")
            data = await resp.json()
        models = data.get("data") or []
        if not models:
            raise RuntimeError("No models loaded in LM Studio")
        first = models[0] or {}
        model_id = first.get("id")
        if not model_id:
            raise RuntimeError("Unable to detect model id from LM Studio")
        return str(model_id)


def mcp_tools_to_openai(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in mcp_tools or []:
        if isinstance(t, str):
            name = t
            desc = f"MCP tool: {t}"
            schema: dict[str, Any] = {"type": "object", "properties": {}}
        elif isinstance(t, dict):
            name = str(t.get("name") or t.get("tool") or "")
            if not name:
                continue
            desc = str(t.get("description") or f"MCP tool: {name}")
            schema = t.get("inputSchema") or t.get("parameters") or {}
            if not isinstance(schema, dict):
                schema = {"type": "object", "properties": {}}
        else:
            continue

        if "type" not in schema:
            schema = {"type": "object", "properties": {}, **schema}

        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": schema,
                },
            }
        )
    return out

