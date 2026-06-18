from __future__ import annotations

import json


def planner_prompt(user_text: str, recent_context: str, allowed_roots: list[str], cwd_options: list[str]) -> str:
    rules = {
        "allowed_roots": allowed_roots,
        "cwd_options": cwd_options,
    }
    return f"""
You are a filesystem planner. Output STRICT JSON only, no markdown.
Never access paths outside allowed_roots. Prefer read-only operations.

Allowed roots: {json.dumps(allowed_roots)}
CWD options: {json.dumps(cwd_options)}

Recent context:
{recent_context}

User request:
{user_text}

Return JSON with this schema:
{{
  "cwd": "Desktop|Documents|ProjectRoot",
  "steps": [
    {{"op":"cd","path":"Desktop"}},
    {{"op":"list","path":".","limit":50}},
    {{"op":"read","path":"file.txt","max_chars":20000}},
    {{"op":"tail","path":"file.txt","lines":20}},
    {{"op":"stat","path":"file.txt"}},
    {{"op":"find","pattern":"*.txt","limit":50}},
    {{"op":"grep","query":"needle","limit_hits":20,"limit_files":20}},
    {{"op":"return"}}
  ],
  "user_intent": "list|read|tail|find|grep|stat|summarize|quote|extract",
  "what_to_do_with_content": "store|summarize|quote|extract",
  "notes": ""
}}
""".strip()


def summarizer_prompt(question: str, file_content: str, recent_context: str) -> str:
    return f"""
You are a concise assistant. Summarize or extract based on the user's request.
Avoid long quotes. If extracting, keep it brief.

Recent context:
{recent_context}

User request:
{question}

Content:
{file_content}
""".strip()
