from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional


_LLM_GENERATE = None
_RECENT_TURNS = 20


def configure(llm_generate, recent_turns: Optional[int] = None) -> None:
    global _LLM_GENERATE, _RECENT_TURNS
    _LLM_GENERATE = llm_generate
    if recent_turns is not None:
        _RECENT_TURNS = int(recent_turns)


def plan(user_text: str, now_iso: str, tz: str, recent_context: str = "", recent_state: dict | None = None) -> dict:
    if _LLM_GENERATE is None:
        raise RuntimeError("Planner LLM not configured")
    prompt = _build_prompt(user_text, now_iso, tz, _RECENT_TURNS, recent_context, recent_state or {})
    raw = _LLM_GENERATE(prompt)
    return _parse_plan(raw)


def _build_prompt(user_text: str, now_iso: str, tz: str, recent_turns: int, recent_context: str, recent_state: dict) -> str:
    default_location = "Yerevan"
    return f"""
You are an intent classifier and tool planner.
Output STRICT JSON only. No markdown, no prose.

Now: {now_iso}
Timezone: {tz}
Default location (if needed): {default_location}

Rules:
- Must output keys: intent, context_request, tool_args.
- intent: assist | weather | search | filesystem | calendar | executor
- context_request: null OR {{"mode":"last_turns","turns":N}} OR {{"mode":"range","start":ISO,"end":ISO,"reason":...}}
- Output "topic": one of weather|search|filesystem|calendar|general
- Output "followup": {{"is_followup": true|false, "refers_to":"previous_turn|date|location|person|file|search_results|none"}}
- If user mentions a date/time span:
  - yesterday => previous day in local TZ (00:00..23:59)
  - today => current day (00:00..23:59)
  - "Feb 6" => that day (infer year if missing)
  - "in/after 3 days" => use that future date
  - last week => last 7 days
  - "an hour ago" => now-1h..now
- If no explicit request, set context_request to {{"mode":"last_turns","turns":{recent_turns}}}

Tool args:
- weather: {{ "location": "...", "when": "today|tomorrow|YYYY-MM-DD|this_week", "whatToWear": true/false }}
- search: {{ "query": "...", "max_results": 5, "summarize": true, "citations": true }}
- filesystem: {{ "cwd":"Desktop|Documents|ProjectRoot", "op":"list|read|tail|find|grep|stat", "path":"...", "lines":2, "pattern":"*.txt", "what_to_do":"store|summarize|quote|extract" }}
- calendar: {{ "query": "user's date question or event request" }}
- executor: return intent executor only if user explicitly requests running code

Important:
- If the user asks to remember/recall, "what happened", or refers to yesterday/last week in a memory sense,
  set intent="assist" and set context_request to the appropriate range. tool_args must be null.
  Do NOT choose filesystem for memory/recall questions.
- Use intent="search" ONLY if the user explicitly commands a web search OR the question clearly needs
  up-to-date/precise external information (news, prices, releases, schedules, current leadership, etc.).
  Do NOT choose search just because the word "search" appears as a noun.
- The user may ask about upcoming activities or festivals (e.g., Valentine's Day, March 8); handle those naturally.
- If a message is a follow-up ("and?", "what about that day", "activity then"), inherit the previous topic unless the user clearly switches.
- Never route to weather unless weather is explicit OR the previous topic was weather and user continues it.
- Never route to person-search unless user mentions a person/social network/profile again.

Recent conversation (short):
{recent_context}

Recent state (JSON):
{json.dumps(recent_state, ensure_ascii=False)}

User:
{user_text}
""".strip()


def _parse_plan(text: str) -> dict:
    if not text:
        raise ValueError("Empty planner response")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Planner returned no JSON object")
    blob = text[start:end + 1]
    data = json.loads(blob)
    if not isinstance(data, dict):
        raise ValueError("Planner JSON is not an object")
    if "intent" not in data:
        raise ValueError("Planner JSON missing intent")
    if "context_request" not in data:
        data["context_request"] = None
    if "tool_args" not in data:
        data["tool_args"] = None
    if "topic" not in data:
        data["topic"] = "general"
    if "followup" not in data:
        data["followup"] = {"is_followup": False, "refers_to": "none"}
    return data
