from __future__ import annotations

from typing import Any, Dict, List, Optional

from .ops import list_dir, read_file
from .safety import FsPolicy, resolve_path


def summarize_path(
    llm_generate,
    policy: FsPolicy,
    path: str,
    mode: str,
    focus: Optional[str] = None,
) -> Dict[str, Any]:
    focus = (focus or "").strip()
    mode = (mode or "").strip().lower()
    if mode not in {"file", "folder"}:
        mode = "folder"

    if mode == "file":
        root, resolved, rel = resolve_path(path, policy, must_exist=True)
        file_result = read_file(rel, policy, start_line=1, end_line=None)
        excerpt = (file_result.get("content") or "")[:4000]
        prompt = _file_prompt(excerpt, focus)
        summary = (llm_generate(prompt) or "").strip()
        return {
            "path": file_result.get("path") or rel,
            "mode": "file",
            "summary": summary,
            "total_lines": file_result.get("total_lines"),
        }

    dir_result = list_dir(path, policy)
    rel = dir_result.get("path") or path
    entries = dir_result.get("entries") or []

    candidates = _pick_candidate_files(entries)
    excerpts = []
    for name in candidates[:3]:
        try:
            file_result = read_file(f"{rel}/{name}".strip("/"), policy, start_line=1, end_line=None)
            content = (file_result.get("content") or "")[:2000]
            excerpts.append({"file": name, "text": content})
        except Exception:
            continue

    prompt = _folder_prompt(rel, entries, excerpts, focus)
    summary = (llm_generate(prompt) or "").strip()

    return {
        "path": rel,
        "mode": "folder",
        "summary": summary,
        "key_files": candidates[:6],
    }


def _pick_candidate_files(entries: List[Dict[str, Any]]) -> List[str]:
    preferred_ext = {".py", ".md", ".txt", ".yaml", ".yml"}
    files = []
    for e in entries:
        if e.get("type") != "file":
            continue
        name = e.get("name") or ""
        for ext in preferred_ext:
            if name.lower().endswith(ext):
                files.append(name)
                break
    return files[:6]


def _file_prompt(excerpt: str, focus: str) -> str:
    focus_line = f"Focus: {focus}" if focus else "Focus: none"
    return f"""
You are a concise assistant. Summarize the file content below with 3 bullet points:
- Purpose of the file
- Key functions or classes
- How it is used
Keep each bullet short. Avoid long quotes.
{focus_line}

File content:
{excerpt}
""".strip()


def _folder_prompt(path: str, entries: List[Dict[str, Any]], excerpts: List[Dict[str, str]], focus: str) -> str:
    focus_line = f"Focus: {focus}" if focus else "Focus: none"
    file_list = ", ".join([e.get("name") or "" for e in entries if e.get("type") == "file"][:20])
    excerpt_text = "\n\n".join([f"{e['file']}:\n{e['text']}" for e in excerpts])
    return f"""
You are a concise assistant. Provide a 1-3 sentence overview of the folder and then a short "Key files:" list (max 6).
Keep it brief and TTS-friendly. Avoid long quotes.
{focus_line}
Folder: {path}
Files: {file_list}

Excerpts:
{excerpt_text}
""".strip()
