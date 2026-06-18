from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from services.orchestrator.lmstudio_client import chat_completion
from services.orchestrator.prompt_builder import build_router_messages


@dataclass
class RouterDecision:
    intent: str
    confidence: float
    need_context: dict | None
    tool_calls: list[dict]
    user_rewrite: str
    context_policy: str | None = None
    router_context_turns: int | None = None
    brain_context_turns: int | None = None
    debug_reason: str | None = None

    @classmethod
    def fallback(cls, user_text: str, reason: str = "router_fallback") -> "RouterDecision":
        return cls(
            intent="assist",
            confidence=0.0,
            need_context=None,
            tool_calls=[],
            user_rewrite=user_text or "",
            context_policy=None,
            router_context_turns=None,
            brain_context_turns=None,
            debug_reason=reason,
        )


def decide(
    user_text: str,
    now_iso: str,
    tool_catalog: str,
    memory_note: str,
    cfg: dict,
    memory_transcript: str = "",
) -> RouterDecision:
    router_cfg = (cfg.get("llm") or {}).get("router") or {}
    base_url = router_cfg.get("base_url", "http://localhost:1234/v1")
    model = router_cfg.get("model", "qwen2.5-3b-instruct")
    temperature = float(router_cfg.get("temperature", 0.2))
    max_tokens = int(router_cfg.get("max_tokens", 350))

    system_prompt = (
        "You are the ROUTER. Decide the user's intent and tool calls. "
        "Return STRICT JSON only, no markdown, no prose. "
        "Hard rules (must follow): "
        "1) If user mentions weather/forecast/temperature/rain/snow or asks what to wear -> intent=weather with one weather tool_call. "
        "2) If user asks to use Codex (\"tell codex\", \"use codex\", \"run codex\", \"/codex\") -> intent=codex with one codex tool_call. "
        "3) If user asks to search the web or look up info -> intent=search with one web-search tool_call. "
        "4) Filesystem intent only for explicit file/folder actions and must include one filesystem tool_call. "
        "4.1) If user explicitly asks about Obsidian/vault/note creation in Obsidian, use one obsidian tool_call "
        "(obsidian.write/obsidian.read/obsidian.list) instead of generic filesystem.write. "
        "Filesystem rules: do not emit variable placeholders like ${...}; do not chain dependent calls in one turn unless explicitly requested. "
        "Use concrete arguments only. Prefer a single call for simple actions. "
        "For create file on Desktop, use filesystem.write with args {path, content, mode}. "
        "Use argument names: path, content, mode, lines, pattern, query. Avoid custom keys like file_path/lines_count. "
        "If a tool is required by these rules, tool_calls MUST NOT be empty. "
        "If the user references time (yesterday, last week, Feb 6, continue), "
        "set need_context with ISO start/end. "
        "Default location is Yerevan if user doesn't specify. "
        "Only include clothing advice if user asks what to wear. "
        "If ambiguous and no rule applies, choose intent=assist and tool_calls empty. "
        "Reminder: URLs should be placed in parentheses in display text because TTS ignores parentheses."
    )

    messages = build_router_messages(
        user_text=user_text,
        now_iso=now_iso,
        tool_catalog=tool_catalog,
        memory_transcript_or_empty=memory_transcript or memory_note or "",
    )

    schema = {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": ["assist", "weather", "search", "filesystem", "executor", "codex"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "need_context": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "properties": {
                            "start": {"type": ["string", "null"]},
                            "end": {"type": ["string", "null"]},
                            "reason": {"type": ["string", "null"]},
                        },
                        "required": ["start", "end", "reason"],
                        "additionalProperties": False,
                    },
                ]
            },
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            },
        },
        "user_rewrite": {"type": "string"},
        "context_policy": {"type": ["string", "null"], "enum": ["none", "default", "range"]},
        "router_context_turns": {"type": ["integer", "null"]},
        "brain_context_turns": {"type": ["integer", "null"]},
    },
    "required": ["intent", "confidence", "need_context", "tool_calls", "user_rewrite"],
    "additionalProperties": False,
}

    extra_payload = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "router_decision",
                "schema": schema,
                "strict": True,
            },
        }
    }

    try:
        raw = chat_completion(
            base_url=base_url,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                messages[1],
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_payload=extra_payload,
        )
    except Exception as e:
        return RouterDecision.fallback(user_text, reason=f"router_llm_error:{e}")

    data = _safe_json_load(raw)
    if not _is_valid_decision(data):
        return RouterDecision.fallback(user_text, reason="router_invalid_json")

    intent = data.get("intent") or "assist"
    confidence = float(data.get("confidence") or 0)
    need_context = data.get("need_context")
    tool_calls = data.get("tool_calls") or []
    user_rewrite = data.get("user_rewrite") or user_text or ""
    context_policy = data.get("context_policy")
    router_context_turns = data.get("router_context_turns")
    brain_context_turns = data.get("brain_context_turns")

    if confidence < 0.35:
        return RouterDecision.fallback(user_text, reason="router_low_confidence")

    return RouterDecision(
        intent=str(intent),
        confidence=confidence,
        need_context=need_context,
        tool_calls=tool_calls,
        user_rewrite=user_rewrite,
        context_policy=context_policy,
        router_context_turns=router_context_turns,
        brain_context_turns=brain_context_turns,
        debug_reason=None,
    )


def _safe_json_load(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    # best-effort extraction of JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def _is_valid_decision(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get("intent") not in {"assist", "weather", "search", "filesystem", "executor", "codex"}:
        return False
    if "confidence" not in data:
        return False
    if "tool_calls" not in data or not isinstance(data.get("tool_calls"), list):
        return False
    if "user_rewrite" not in data:
        return False
    return True
