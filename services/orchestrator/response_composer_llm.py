from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


_LLM_GENERATE = None


class ComposeError(RuntimeError):
    def __init__(self, message: str, raw: str | None = None):
        super().__init__(message)
        self.raw = raw or ""


def configure(llm_generate) -> None:
    global _LLM_GENERATE
    _LLM_GENERATE = llm_generate


def compose_reply(
    user_text: str,
    intent: str,
    tool_result: Optional[dict],
    recent_context: str,
    assistant_name: str,
    system_style: str,
    now_iso: str,
    constraints: dict,
) -> dict:
    if _LLM_GENERATE is None:
        raise ComposeError("Composer LLM not configured")
    max_chars = int(constraints.get("composer_max_chars", 500))
    recent_context = preprocess_context(recent_context, max_chars)
    prompt = _build_prompt(
        user_text=user_text,
        intent=intent,
        tool_result=_trim_tool_result(tool_result, constraints),
        recent_context=recent_context,
        assistant_name=assistant_name,
        system_style=system_style,
        now_iso=now_iso,
        constraints=constraints,
    )
    try:
        raw = _LLM_GENERATE(prompt)
    except Exception as e:
        raise ComposeError(f"LLM generate failed: {e}") from e
    try:
        return _parse_response_strict(raw)
    except Exception:
        # One repair attempt to coerce JSON
        try:
            repaired = _LLM_GENERATE(_repair_prompt(raw))
            return _parse_response_strict(repaired)
        except Exception as e:
            raise ComposeError(f"Composer JSON parse failed: {e}", raw=raw) from e


def _build_prompt(
    user_text: str,
    intent: str,
    tool_result: Optional[dict],
    recent_context: str,
    assistant_name: str,
    system_style: str,
    now_iso: str,
    constraints: dict,
) -> str:
    default_location = constraints.get("default_location", "Yerevan")
    return f"""
You are {assistant_name}. Compose a natural reply using the user's tone and any tool data.
Output STRICT JSON only with keys: spoken_text, display_text, memory_events, followups.
Do not include any extra keys or text outside the JSON object.

Rules:
- spoken_text: 1-4 sentences, concise, no URLs, no long file paths, no citations.
- display_text: may include URLs/paths ONLY inside parentheses.
- spoken_text must include the core message; do NOT put the entire reply only inside parentheses.
- If location is not specified for weather, default to {default_location}; do not ask for location.
- If search is about a person, present "possible matches" and ask for confirmation.
- For weather, interpret the numbers (e.g., "looks like a wet day") instead of just repeating metrics.
- For weather intent, put ALL key facts in spoken_text (not inside parentheses).
- For calendar intent, date + weekday + holiday/event must be in spoken_text.
- If user is frustrated, empathize gently without swearing back.
- If tool_result has an error, explain briefly and ask a short follow-up.
- Parentheses are ONLY for URLs, citations/sources, long file paths, or optional metadata.
- If you are unsure, still return valid JSON with a short safe reply (but never empty fields).

Now: {now_iso}
Default location: {default_location}
Constraints: {json.dumps(constraints)}

System style:
{system_style}

Recent context:
{recent_context}

Intent: {intent}
User: {user_text}
Tool result (JSON):
{json.dumps(tool_result or {}, ensure_ascii=False)}
""".strip()


def _trim_tool_result(tool_result: Optional[dict], constraints: dict) -> Optional[dict]:
    if not tool_result:
        return tool_result
    max_chars = int(constraints.get("max_tool_content_chars", 1500))
    out = dict(tool_result)
    def _norm(s: str) -> str:
        return _normalize_text(s, max_chars)
    # trim any long content fields
    for key in ["content", "text", "summary"]:
        if key in out and isinstance(out[key], str):
            out[key] = _norm(out[key])
    if "result" in out and isinstance(out["result"], dict):
        inner = dict(out["result"])
        for key in ["content", "text", "summary"]:
            if key in inner and isinstance(inner[key], str):
                inner[key] = _norm(inner[key])
        out["result"] = inner
    # trim search results
    if "results" in out and isinstance(out["results"], list):
        trimmed = []
        for r in out["results"][:5]:
            trimmed.append({
                "title": _normalize_text(r.get("title") or "", 120),
                "url": r.get("url"),
                "snippet": _normalize_text(r.get("snippet") or "", 240),
                "source_name": r.get("source_name"),
            })
        out["results"] = trimmed
    if "result" in out and isinstance(out["result"], dict) and isinstance(out["result"].get("results"), list):
        trimmed = []
        for r in out["result"]["results"][:5]:
            trimmed.append({
                "title": _normalize_text(r.get("title") or "", 120),
                "url": r.get("url"),
                "snippet": _normalize_text(r.get("snippet") or "", 240),
                "source_name": r.get("source_name"),
            })
        out["result"]["results"] = trimmed
    return out


def _parse_response_strict(text: str) -> dict:
    if not text:
        raise ValueError("Empty composer response")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Composer returned no JSON object")
    blob = text[start:end + 1]
    data = json.loads(blob)
    spoken = (data.get("spoken_text") or "").strip()
    display = (data.get("display_text") or "").strip()
    memory_events = data.get("memory_events") or []
    followups = data.get("followups") or []
    if not spoken and not display:
        raise ValueError("Composer JSON missing spoken/display text")
    return {
        "spoken_text": spoken,
        "display_text": display,
        "memory_events": memory_events if isinstance(memory_events, list) else [],
        "followups": followups if isinstance(followups, list) else [],
    }


def _repair_prompt(raw: str) -> str:
    # Keep repair prompt small and context-free to avoid slow retries
    raw = (raw or "")[:3000]
    return f"""
Return STRICT JSON only with keys: spoken_text, display_text, memory_events, followups.
Do not add any extra text. Fix the output below into valid JSON.

Output to fix:
{raw}
""".strip()


def _normalize_text(text: str, max_chars: int | None = None) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if max_chars and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip()
    return cleaned


def preprocess_context(text: str, max_chars: int) -> str:
    return _normalize_text(text, max_chars)
