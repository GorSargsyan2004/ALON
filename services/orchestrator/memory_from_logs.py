from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from datetime import datetime

_LOG_PATH: Optional[Path] = None


def configure(log_path: Path) -> None:
    global _LOG_PATH
    _LOG_PATH = log_path


def get_recent_transcript(turns: int = 20, log_path: Optional[Path] = None, max_tool_content_chars: int = 4000) -> str:
    log_path = log_path or _LOG_PATH
    if log_path is None:
        return ""
    entries = _read_last_entries(log_path, max_lines=turns * 6)
    lines: List[str] = []
    for e in entries:
        t = _format_ts(e.get("ts"))
        if e.get("type") == "turn_start":
            user = e.get("user_text")
            if user:
                lines.append(f"{t} Gor: {user}")
        elif e.get("type") == "turn_end":
            assistant = e.get("assistant_text")
            if assistant:
                lines.append(f"{t} Alon: {assistant}")
        elif e.get("type") == "tool_result":
            tr = e.get("tool_result") or {}
            if tr.get("tool") == "filesystem" and tr.get("ok"):
                result = tr.get("result") or {}
                content = result.get("content")
                if content and result.get("memory_store"):
                    content = content[:max_tool_content_chars]
                    op = tr.get("op") or ""
                    path = result.get("path") or ""
                    lines.append(f"{t} Alon (tool filesystem {op} {path}): \"{content}\"")
    return "\n".join(lines[-turns * 2:]).strip()


def load_recent(
    default_recent_turns: int = 20,
    log_path: Optional[Path] = None,
    max_tool_content_chars: int = 4000,
) -> str:
    return get_recent_transcript(
        turns=default_recent_turns,
        log_path=log_path,
        max_tool_content_chars=max_tool_content_chars,
    )


def render_recent(
    turns: int,
    max_chars: int,
    log_path: Optional[Path] = None,
    max_tool_content_chars: int = 4000,
) -> str:
    text = get_recent_transcript(
        turns=turns,
        log_path=log_path,
        max_tool_content_chars=max_tool_content_chars,
    )
    if max_chars and len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


def get_range_transcript(start_iso: str, end_iso: str, log_path: Optional[Path] = None,
                         max_tool_content_chars: int = 4000) -> str:
    log_path = log_path or _LOG_PATH
    if log_path is None:
        return ""
    entries = _read_last_entries(log_path, max_lines=2000)
    start_dt = _parse_iso(start_iso)
    end_dt = _parse_iso(end_iso)
    if not start_dt or not end_dt:
        return ""
    lines: List[str] = []
    for e in entries:
        ts = e.get("ts")
        dt = _parse_iso(ts)
        if not dt or dt < start_dt or dt > end_dt:
            continue
        t = _format_ts(ts)
        if e.get("type") == "turn_start":
            user = e.get("user_text")
            if user:
                lines.append(f"{t} Gor: {user}")
        elif e.get("type") == "turn_end":
            assistant = e.get("assistant_text")
            if assistant:
                lines.append(f"{t} Alon: {assistant}")
        elif e.get("type") == "tool_result":
            tr = e.get("tool_result") or {}
            if tr.get("tool") == "filesystem" and tr.get("ok"):
                result = tr.get("result") or {}
                content = result.get("content")
                if content and result.get("memory_store"):
                    content = content[:max_tool_content_chars]
                    op = tr.get("op") or ""
                    path = result.get("path") or ""
                    lines.append(f"{t} Alon (tool filesystem {op} {path}): \"{content}\"")
    return "\n".join(lines).strip()


def load_range(
    start_iso: str,
    end_iso: str,
    max_turns: int = 200,
    log_path: Optional[Path] = None,
    max_tool_content_chars: int = 4000,
) -> str:
    log_path = log_path or _LOG_PATH
    if log_path is None:
        return ""
    entries = _read_last_entries(log_path, max_lines=max_turns * 8)
    start_dt = _parse_iso(start_iso)
    end_dt = _parse_iso(end_iso)
    if not start_dt or not end_dt:
        return ""
    lines: List[str] = []
    for e in entries:
        ts = e.get("ts")
        dt = _parse_iso(ts)
        if not dt or dt < start_dt or dt > end_dt:
            continue
        t = _format_ts(ts)
        if e.get("type") == "turn_start":
            user = e.get("user_text")
            if user:
                lines.append(f"{t} Gor: {user}")
        elif e.get("type") == "turn_end":
            assistant = e.get("assistant_text")
            if assistant:
                lines.append(f"{t} Alon: {assistant}")
        elif e.get("type") == "tool_result":
            tr = e.get("tool_result") or {}
            if tr.get("tool") == "filesystem" and tr.get("ok"):
                result = tr.get("result") or {}
                content = result.get("content")
                if content and result.get("memory_store"):
                    content = content[:max_tool_content_chars]
                    op = tr.get("op") or ""
                    path = result.get("path") or ""
                    lines.append(f"{t} Alon (tool filesystem {op} {path}): \"{content}\"")
        if len(lines) >= max_turns * 2:
            break
    return "\n".join(lines).strip()


def render_range(
    start_iso: str,
    end_iso: str,
    max_turns: int,
    max_chars: int,
    log_path: Optional[Path] = None,
    max_tool_content_chars: int = 4000,
) -> str:
    text = load_range(
        start_iso=start_iso,
        end_iso=end_iso,
        max_turns=max_turns,
        log_path=log_path,
        max_tool_content_chars=max_tool_content_chars,
    )
    if max_chars and len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


def get_relevant_transcript(query: str, log_path: Optional[Path] = None,
                            start_iso: Optional[str] = None, end_iso: Optional[str] = None,
                            max_items: int = 10, max_tool_content_chars: int = 4000) -> str:
    log_path = log_path or _LOG_PATH
    if log_path is None:
        return ""
    entries = _read_last_entries(log_path, max_lines=800)
    words = set(re.findall(r"\w+", (query or "").lower()))
    start_dt = _parse_iso(start_iso) if start_iso else None
    end_dt = _parse_iso(end_iso) if end_iso else None
    scored = []
    for e in entries:
        ts = e.get("ts")
        dt = _parse_iso(ts)
        if start_dt and dt and dt < start_dt:
            continue
        if end_dt and dt and dt > end_dt:
            continue
        text = None
        t = _format_ts(e.get("ts"))
        if e.get("type") == "turn_start":
            text = e.get("user_text")
            who = "Gor"
        elif e.get("type") == "turn_end":
            text = e.get("assistant_text")
            who = "Alon"
        elif e.get("type") == "tool_result":
            tr = e.get("tool_result") or {}
            if tr.get("tool") == "filesystem" and tr.get("ok"):
                res = tr.get("result") or {}
                content = res.get("content")
                if content:
                    content = content[:max_tool_content_chars]
                    op = tr.get("op") or ""
                    path = res.get("path") or ""
                    text = f"{op} {path} {content}"
                    who = "Alon (tool)"
        if not text:
            continue
        score = sum(1 for w in words if w in text.lower())
        if score > 0 or (e.get("type") == "tool_result" and (e.get("tool_result") or {}).get("result", {}).get("memory_store")):
            scored.append((score, f"{t} {who}: {text}"))
    scored.sort(key=lambda x: x[0], reverse=True)
    return "\n".join([s for _, s in scored[:max_items]])


def _read_last_entries(log_path: Path, max_lines: int = 200) -> List[dict]:
    if not log_path.exists():
        return []
    lines = _tail_lines(log_path, max_lines)
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _tail_lines(path: Path, n: int) -> List[str]:
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        block = 8192
        data = b""
        while size > 0 and data.count(b"\n") <= n:
            read_size = min(block, size)
            size -= read_size
            f.seek(size)
            data = f.read(read_size) + data
    lines = data.splitlines()
    return [l.decode("utf-8", errors="replace") for l in lines[-n:]]


def _format_ts(ts: str | None) -> str:
    if not ts:
        return "<unknown>"
    if "T" in ts:
        date, time = ts.split("T", 1)
        time = time.split("+", 1)[0]
        time = time.split("-", 1)[0]
        return f"<{date} {time}>"
    return f"<{ts}>"


def _parse_iso(ts: str | None):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        try:
            if "T" in ts:
                return datetime.fromisoformat(ts.split("+", 1)[0])
        except Exception:
            return None
    return None
