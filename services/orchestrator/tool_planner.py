from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Optional


_WEAR_RE = re.compile(r"\b(what to wear|what should i wear|wear|clothes|jacket|coat|outfit)\b", re.I)
_TODAY_RE = re.compile(r"\b(today|now|current)\b", re.I)
_TOMORROW_RE = re.compile(r"\b(tomorrow)\b", re.I)
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def plan_weather(
    llm_generate: Callable[[str], str],
    user_text: str,
    default_location: str,
    timezone: str,
) -> Dict[str, Any]:
    prompt = _weather_prompt(user_text, default_location, timezone)
    raw = llm_generate(prompt)
    data = _safe_json(raw) or {}

    location = (data.get("args") or {}).get("location") or _extract_location(user_text) or default_location
    day = (data.get("args") or {}).get("day") or _extract_day(user_text) or "today"
    day = _normalize_day(day, user_text)

    include_wear = bool(_WEAR_RE.search(user_text))

    plan = {
        "tool": "weather.get_forecast",
        "args": {
            "location": location,
            "day": day,
            "include_wear_advice": include_wear,
        },
    }
    return plan


def plan_search(
    llm_generate: Callable[[str], str],
    user_text: str,
) -> Dict[str, Any]:
    prompt = _search_prompt(user_text)
    raw = llm_generate(prompt)
    data = _safe_json(raw) or {}

    args = data.get("args") or {}
    query = (args.get("query") or "").strip() or user_text.strip()
    focus = (args.get("focus") or "").strip()
    if not focus:
        focus = _short_focus(user_text)

    plan = {
        "tool": "websearch.search",
        "args": {
            "query": query,
            "max_results": int(args.get("max_results") or 5),
            "focus": focus,
        },
    }
    return plan


def plan_filesystem(
    llm_generate: Callable[[str], str],
    user_text: str,
    allowed_roots: list[str],
) -> Dict[str, Any]:
    prompt = _filesystem_prompt(user_text, allowed_roots)
    raw = llm_generate(prompt)
    data = _safe_json(raw) or {}

    tool = (data.get("tool") or "").strip()
    args = data.get("args") or {}

    if tool not in {
        "filesystem.list_dir",
        "filesystem.read_file",
        "filesystem.search_text",
        "filesystem.summarize",
    }:
        return _fallback_filesystem_plan(user_text)

    if tool == "filesystem.list_dir":
        path = _sanitize_rel_path(args.get("path") or _extract_path(user_text) or ".")
        return {
            "tool": tool,
            "args": {
                "path": path,
                "recursive": bool(args.get("recursive", False)),
                "pattern": args.get("pattern"),
                "max_entries": int(args.get("max_entries") or 100),
            },
        }

    if tool == "filesystem.read_file":
        path = _sanitize_rel_path(args.get("path") or _extract_path(user_text) or "")
        return {
            "tool": tool,
            "args": {
                "path": path or ".",
                "start_line": int(args.get("start_line") or 1),
                "end_line": args.get("end_line"),
            },
        }

    if tool == "filesystem.search_text":
        path = _sanitize_rel_path(args.get("path") or _extract_path(user_text) or ".")
        return {
            "tool": tool,
            "args": {
                "path": path,
                "query": (args.get("query") or _extract_query(user_text) or "").strip(),
                "file_glob": args.get("file_glob") or "*.py",
                "max_files": int(args.get("max_files") or 50),
                "max_hits": int(args.get("max_hits") or 50),
            },
        }

    if tool == "filesystem.summarize":
        path = _sanitize_rel_path(args.get("path") or _extract_path(user_text) or ".")
        mode = (args.get("mode") or "folder").strip().lower()
        if mode not in {"file", "folder"}:
            mode = "folder"
        return {
            "tool": tool,
            "args": {
                "path": path,
                "mode": mode,
                "focus": args.get("focus"),
            },
        }

    return _fallback_filesystem_plan(user_text)


def summarize_search(
    llm_generate: Callable[[str], str],
    user_text: str,
    query: str,
    results: list[dict],
    pages: list[dict],
) -> str:
    prompt = _summarize_prompt(user_text, query, results, pages)
    raw = llm_generate(prompt)
    text = (raw or "").strip()
    if not text:
        return "I found some results, but I couldn't summarize them clearly."
    return _limit_sentences(text, 2)


def _safe_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _normalize_day(day: str, user_text: str) -> str:
    day = (day or "").strip().lower()
    if day in {"today", "tomorrow"}:
        return day
    if _DATE_RE.search(day):
        return _DATE_RE.search(day).group(1)
    extracted = _extract_day(user_text)
    if extracted:
        return extracted
    return "today"


def _extract_day(text: str) -> Optional[str]:
    if _TOMORROW_RE.search(text):
        return "tomorrow"
    if _TODAY_RE.search(text):
        return "today"
    m = _DATE_RE.search(text or "")
    if m:
        return m.group(1)
    return None


def _extract_location(text: str) -> Optional[str]:
    if " in " not in (text or "").lower():
        return None
    try:
        return text.split(" in ", 1)[1].strip()
    except Exception:
        return None


def _short_focus(text: str, max_words: int = 8) -> str:
    words = re.findall(r"[A-Za-z0-9']+", text or "")
    return " ".join(words[:max_words])


def _weather_prompt(user_text: str, default_location: str, timezone: str) -> str:
    return f"""
You are a strict JSON generator for a weather tool. Output ONLY valid JSON and nothing else.
Schema:
{{
  "tool": "weather.get_forecast",
  "args": {{
    "location": "city or place",
    "day": "today" | "tomorrow" | "YYYY-MM-DD",
    "include_wear_advice": true/false
  }}
}}
Rules:
- include_wear_advice=true ONLY if the user asks what to wear, clothes, jacket, or coat.
- If the user asks about today/now/current -> day="today".
- If they ask about tomorrow -> day="tomorrow".
- If they ask for a specific date, use YYYY-MM-DD.
- If no day is specified, use "today".
- If no location is specified, use the default location.
Default location: {default_location}
Timezone: {timezone}

Examples:
User: "What's the weather today in Yerevan?"
{{"tool":"weather.get_forecast","args":{{"location":"Yerevan","day":"today","include_wear_advice":false}}}}
User: "What should I wear tomorrow in Gyumri?"
{{"tool":"weather.get_forecast","args":{{"location":"Gyumri","day":"tomorrow","include_wear_advice":true}}}}
User: "Weather on 2026-02-10 in London"
{{"tool":"weather.get_forecast","args":{{"location":"London","day":"2026-02-10","include_wear_advice":false}}}}

User: "{user_text}"
""".strip()


def _search_prompt(user_text: str) -> str:
    return f"""
You are a strict JSON generator for a web search tool. Output ONLY valid JSON and nothing else.
Schema:
{{
  "tool": "websearch.search",
  "args": {{
    "query": "clean query string",
    "max_results": 5,
    "focus": "short phrase of what the user is trying to find out"
  }}
}}
Rules:
- query should be concise and search-engine friendly.
- focus should be short and specific.

Examples:
User: "search the web for latest NVIDIA earnings"
{{"tool":"websearch.search","args":{{"query":"NVIDIA earnings latest report","max_results":5,"focus":"latest NVIDIA earnings"}}}}
User: "look up OpenAI Sora release date"
{{"tool":"websearch.search","args":{{"query":"OpenAI Sora release date","max_results":5,"focus":"Sora release date"}}}}

User: "{user_text}"
""".strip()


def _summarize_prompt(user_text: str, query: str, results: list[dict], pages: list[dict]) -> str:
    trimmed_results = []
    for r in (results or [])[:5]:
        trimmed_results.append({
            "title": r.get("title") or "",
            "snippet": (r.get("snippet") or "")[:300],
            "source": r.get("source_name") or "",
            "url": r.get("url") or "",
            "date": r.get("date") or r.get("publish_date") or "",
        })

    trimmed_pages = []
    for p in (pages or [])[:3]:
        trimmed_pages.append({
            "title": p.get("title") or "",
            "source": p.get("source_name") or "",
            "publish_date": p.get("publish_date") or "",
            "text": (p.get("text") or "")[:800],
        })

    data = {
        "user_question": user_text,
        "clean_query": query,
        "results": trimmed_results,
        "page_extracts": trimmed_pages,
    }

    return f"""
You are a concise assistant. Write a 1-3 sentence spoken answer based only on the data below.
Do NOT include URLs. Do NOT include a sources list. Avoid long quotes.
If the user asked for latest/recent, include publish dates if available; otherwise say "recent reports".
If data is insufficient, say that briefly.

Data (JSON):
{json.dumps(data, ensure_ascii=False)}
""".strip()


def _filesystem_prompt(user_text: str, allowed_roots: list[str]) -> str:
    roots = ", ".join(allowed_roots) if allowed_roots else "."
    return f"""
You are a strict JSON generator for a filesystem tool. Output ONLY valid JSON and nothing else.
Actions:
1) list_dir
{{
  "tool": "filesystem.list_dir",
  "args": {{
    "path": "relative/path",
    "recursive": false,
    "pattern": "*.py" | null,
    "max_entries": 100
  }}
}}
2) read_file
{{
  "tool": "filesystem.read_file",
  "args": {{
    "path": "relative/path/to/file.py",
    "start_line": 1,
    "end_line": 200
  }}
}}
3) search_text
{{
  "tool": "filesystem.search_text",
  "args": {{
    "path": "relative/path",
    "query": "text to search",
    "file_glob": "*.py",
    "max_files": 50,
    "max_hits": 50
  }}
}}
4) summarize
{{
  "tool": "filesystem.summarize",
  "args": {{
    "path": "relative/path",
    "mode": "file" | "folder",
    "focus": "short string or null"
  }}
}}

Rules:
- All paths must be relative to allowed roots.
- If unsure, use list_dir on a relevant folder.
Allowed roots: {roots}

User: "{user_text}"
""".strip()


def _fallback_filesystem_plan(user_text: str) -> Dict[str, Any]:
    text = user_text.lower()
    path = _sanitize_rel_path(_extract_path(user_text) or ".")

    if "summarize" in text:
        mode = "file" if _looks_like_file(path) else "folder"
        return {
            "tool": "filesystem.summarize",
            "args": {"path": path, "mode": mode, "focus": None},
        }

    if "search" in text or "find" in text:
        return {
            "tool": "filesystem.search_text",
            "args": {
                "path": path if path != "." else "services",
                "query": _extract_query(user_text) or "",
                "file_glob": "*.py",
                "max_files": 50,
                "max_hits": 50,
            },
        }

    if "open" in text or "read" in text:
        if _looks_like_file(path):
            return {
                "tool": "filesystem.read_file",
                "args": {"path": path, "start_line": 1, "end_line": 200},
            }

    return {
        "tool": "filesystem.list_dir",
        "args": {"path": path, "recursive": False, "pattern": None, "max_entries": 100},
    }


def _sanitize_rel_path(path: str) -> str:
    path = (path or "").strip().strip("\"'").replace("\\", "/")
    if not path:
        return "."
    if ":" in path or path.startswith("/") or path.startswith("\\"):
        path = path.split(":")[-1].lstrip("/\\")
    if ".." in path.split("/"):
        return "."
    return path.strip()


def _extract_path(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"([A-Za-z]:\\\\[^\\s]+|[\\w./\\\\-]+\\.[A-Za-z0-9]{1,6}|[\\w./\\\\-]+/[^\\s]+)", text)
    if m:
        return m.group(1).strip().strip("\"'")
    return None


def _extract_query(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\"([^\"]+)\"", text)
    if m:
        return m.group(1)
    m = re.search(r"find\\s+(.+)$", text, re.I)
    if m:
        return m.group(1)
    return None


def _looks_like_file(path: str) -> bool:
    return "." in (path or "").split("/")[-1]


def _limit_sentences(text: str, max_sentences: int) -> str:
    parts = re.split(r"(?<=[.!?])\\s+", text.strip())
    if len(parts) <= max_sentences:
        return text.strip()
    return " ".join(parts[:max_sentences]).strip()
