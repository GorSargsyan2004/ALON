from __future__ import annotations

import json
import re
from typing import List, Dict, Any


def build_router_messages(
    user_text: str,
    now_iso: str,
    tool_catalog: str,
    memory_transcript_or_empty: str,
) -> List[Dict[str, str]]:
    system = (
        "You are the ROUTER. Output strict JSON only. "
        "Decide intent and tool calls. URLs should be placed in parentheses in display text."
    )
    user = (
        f"Now: {now_iso}\n"
        f"Tool catalog (canonical names):\n{tool_catalog}\n\n"
        f"Memory (if any):\n{memory_transcript_or_empty}\n\n"
        f"User message:\n{user_text}\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_brain_messages(
    system_style: str,
    user_text: str,
    memory_transcript_or_empty: str,
    tool_results_or_empty: str,
    tool_backend_mode: str = "local",
    mcp_active: bool = False,
    local_fallback_used: bool = False,
) -> List[Dict[str, str]]:
    if mcp_active and not local_fallback_used:
        backend_note = (
            "Runtime mode: MCP active. Prefer MCP tool outputs as authoritative for this turn; "
            "if important fields are missing, state that explicitly."
        )
    else:
        backend_note = (
            "Runtime mode: local deterministic fallback. Use local tool outputs as authoritative for this turn; "
            "do not invent missing facts."
        )
    if tool_backend_mode == "mixed":
        backend_note += " Some tools used MCP and some used local fallback in this turn."
    elif tool_backend_mode == "mcp":
        backend_note += " Tool outputs in this turn came from MCP."
    elif tool_backend_mode == "local" and tool_results_or_empty:
        backend_note += " Tool outputs in this turn came from local fallback."

    system = (
        f"{system_style}\n"
        "Your response will be spoken aloud. "
        "Put URLs and long source lists in parentheses so TTS won't read them. "
        "If the user asked for weather/search, present key facts plainly (not hidden in parentheses). "
        f"{backend_note}"
    )
    blocks = []
    if memory_transcript_or_empty:
        blocks.append(f"MEMORY:\n{memory_transcript_or_empty}")
    blocks.append(f"USER_REQUEST:\n{user_text}")
    if tool_results_or_empty:
        blocks.append(f"TOOL_RESULTS:\n{tool_results_or_empty}")
        blocks.append(
            "INSTRUCTIONS:\n"
            "- Answer naturally and helpfully.\n"
            "- Integrate tool results with the user's tone.\n"
            "- If tool results include errors, acknowledge the failure and do not hallucinate missing data.\n"
            "- Never claim an action was completed unless a tool result explicitly succeeded.\n"
            "- Put URLs and long source lists in parentheses."
        )
    else:
        blocks.append(
            "INSTRUCTIONS:\n"
            "- Answer naturally and helpfully.\n"
            "- Put URLs and long source lists in parentheses."
        )
    user = "\n\n".join(blocks)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def format_tool_results(tool_results: List[Dict[str, Any]]) -> str:
    if not tool_results:
        return ""
    parts = []
    for t in tool_results:
        name = t.get("tool") or ""
        ok = t.get("ok", False)
        result = t.get("result") or {}
        if name.startswith("weather"):
            parts.append(_format_weather(result, ok, t.get("error")))
        elif name.startswith("websearch"):
            parts.append(_format_search(result, ok))
        elif name.startswith("filesystem"):
            parts.append(_format_filesystem(result, ok, t.get("error")))
        elif name.startswith("obsidian"):
            parts.append(_format_obsidian(result, ok, t.get("error")))
        elif name.startswith("codex"):
            parts.append(_format_codex(result, ok))
        else:
            parts.append(f"{name}: {'ok' if ok else 'error'}")
    return "\n\n".join([p for p in parts if p])


def enforce_budget(
    messages: List[Dict[str, str]],
    max_chars_total: int,
    max_chars_tool_results: int,
    max_chars_memory: int,
) -> List[Dict[str, str]]:
    if max_chars_total <= 0:
        return messages
    msg = [m.copy() for m in messages]
    total = sum(len(m.get("content") or "") for m in msg)
    if total <= max_chars_total:
        return msg

    # truncate tool results first
    msg = _truncate_section(msg, "TOOL_RESULTS:", max_chars_tool_results)
    total = sum(len(m.get("content") or "") for m in msg)
    if total <= max_chars_total:
        return msg

    # then truncate memory
    msg = _truncate_section(msg, "MEMORY:", max_chars_memory)
    total = sum(len(m.get("content") or "") for m in msg)
    if total <= max_chars_total:
        return msg

    # finally drop memory entirely
    msg = _remove_section(msg, "MEMORY:")
    return msg


def _truncate_section(messages: List[Dict[str, str]], header: str, max_chars: int) -> List[Dict[str, str]]:
    if max_chars <= 0:
        return messages
    out = []
    for m in messages:
        content = m.get("content") or ""
        if header in content:
            content = _truncate_block(content, header, max_chars)
        out.append({**m, "content": content})
    return out


def _remove_section(messages: List[Dict[str, str]], header: str) -> List[Dict[str, str]]:
    out = []
    for m in messages:
        content = m.get("content") or ""
        if header in content:
            content = _truncate_block(content, header, 0)
        out.append({**m, "content": content})
    return out


def _truncate_block(text: str, header: str, max_chars: int) -> str:
    pattern = re.compile(rf"{re.escape(header)}\n(.*?)(\n\n[A-Z_]+:|\Z)", re.S)
    def repl(match: re.Match) -> str:
        body = match.group(1)
        tail = match.group(2)
        if max_chars <= 0:
            return tail.strip() and tail or ""
        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + "..."
        return f"{header}\n{body}{tail}"
    return pattern.sub(repl, text)


def _format_weather(result: dict, ok: bool, error: str | None = None) -> str:
    if not ok or not result:
        msg = "forecast failed"
        if error:
            msg = error
        return f"WEATHER: error ({msg})"
    if result.get("summary"):
        return f"WEATHER: {result.get('summary')}"
    if result.get("text"):
        return f"WEATHER: {result.get('text')}"
    loc = result.get("location") or "Unknown"
    when = result.get("when") or "today"
    t_min = result.get("t_min")
    t_max = result.get("t_max")
    precip = result.get("precip_prob_max")
    wind = result.get("wind_max")
    parts = [f"WEATHER ({when} in {loc}):"]
    if t_min is not None and t_max is not None:
        parts.append(f"low {int(t_min)}°C, high {int(t_max)}°C")
    if precip is not None:
        parts.append(f"precip up to {int(precip)}%")
    if wind is not None:
        parts.append(f"wind up to {int(float(wind))} km/h")
    if result.get("wear_advice"):
        parts.append(f"wear: {result.get('wear_advice')}")
    return " ".join(parts)


def _format_search(result: dict, ok: bool) -> str:
    if not ok or not result:
        return "SEARCH: error"
    if result.get("text") and not result.get("final_answer"):
        return "SEARCH: " + _shorten_text(str(result.get("text")), 600)
    final = result.get("final_answer") or {}
    spoken = (final.get("answer_spoken") or "").strip()
    sources = (final.get("sources") or [])[:3]
    src_text = []
    for s in sources:
        title = s.get("title") or "Source"
        url = s.get("url") or ""
        if url:
            src_text.append(f"{title} — {url}")
    out = "SEARCH: " + (spoken or "Summary unavailable.")
    if src_text:
        out += " (Sources: " + "; ".join(src_text) + ")"
    return out


def _format_filesystem(result: dict, ok: bool, error: str | None = None) -> str:
    if not ok or not result:
        msg = error or "error"
        return f"FILESYSTEM: error ({msg})"
    if result.get("text") and not result.get("op"):
        return f"FILESYSTEM: {_shorten_text(str(result.get('text')), 500)}"
    op = result.get("op") or result.get("status") or "op"
    path = result.get("path") or ""
    content = result.get("content") or result.get("preview") or ""
    if content:
        content = _shorten_text(content, 500)
    return f"FILESYSTEM ({op}) {path}: {content}".strip()


def _format_obsidian(result: dict, ok: bool, error: str | None = None) -> str:
    if not ok or not result:
        msg = error or "error"
        return f"OBSIDIAN: error ({msg})"
    op = result.get("op") or "op"
    path = result.get("path") or result.get("filepath") or ""
    content = result.get("content") or result.get("text") or ""
    if content:
        content = _shorten_text(str(content), 500)
    return f"OBSIDIAN ({op}) {path}: {content}".strip()


def _format_codex(result: dict, ok: bool) -> str:
    if not ok or not result:
        return "CODEX: error"
    exit_code = result.get("exit_code")
    out_path = result.get("output_path")
    files = result.get("files_changed") or []
    files_str = ", ".join(files[:6])
    summary = f"CODEX: exit_code={exit_code}, output={out_path}"
    if files_str:
        summary += f" (files: {files_str})"
    return summary


def _shorten_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text
