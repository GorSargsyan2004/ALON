from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import re
from .fs_tool import (
    resolve_path,
    list_dir,
    read_text,
    read_tail_lines,
    stat,
    find_files,
    grep_text,
)


class PlanError(Exception):
    pass


class AmbiguousPathError(Exception):
    def __init__(self, matches: list[str]):
        super().__init__("Ambiguous path")
        self.matches = matches


def parse_plan(text: str) -> dict:
    if not text:
        raise PlanError("Empty planner response")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise PlanError("No JSON object found")
    blob = text[start:end + 1]
    try:
        return json.loads(blob)
    except Exception as e:
        raise PlanError(f"Invalid JSON: {e}") from e


def validate_plan(plan: dict) -> dict:
    if not isinstance(plan, dict):
        raise PlanError("Plan is not an object")
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise PlanError("Plan steps missing")
    for s in steps:
        if not isinstance(s, dict):
            raise PlanError("Step must be object")
        op = s.get("op")
        if op not in {"cd", "list", "read", "tail", "stat", "find", "grep", "return"}:
            raise PlanError(f"Unknown op: {op}")
    return plan


def run_plan(
    plan: dict,
    cfg: dict,
    logger,
    session_id: str,
    turn_id: str,
    llm_generate,
    recent_context: str,
    user_text: str,
    project_root: Path,
) -> dict:
    fs_cfg = cfg.get("filesystem") or {}
    max_file_bytes = int(fs_cfg.get("max_file_bytes", 1_000_000))
    max_read_chars = int(fs_cfg.get("max_read_chars", 20000))
    max_list_items = int(fs_cfg.get("max_list_items", 50))
    max_tool_chars = int((cfg.get("memory") or {}).get("max_tool_content_chars", 4000))

    allow_roots = _build_allow_roots(fs_cfg.get("allow_roots") or [], project_root)
    roots_map = _cwd_map(project_root)

    cwd_label = plan.get("cwd") or "ProjectRoot"
    cwd = roots_map.get(cwd_label, project_root)

    action = (plan.get("what_to_do_with_content") or "store").lower()
    store_content = action == "store"
    last_content = None
    last_path = None
    last_result = None
    last_op = None
    results = []

    for step in plan["steps"]:
        op = step.get("op")
        if op == "return":
            break
        if op == "cd":
            target = step.get("path") or "."
            cwd = _resolve_cwd(target, cwd, allow_roots, roots_map)
            results.append({"op": "cd", "path": str(cwd)})
            continue

        call = {"type": "tool_call", "session_id": session_id, "turn_id": turn_id,
                "tool_call": {"tool": "filesystem", "op": op, "args": step}}
        logger.write(call)

        try:
            if op == "list":
                p = _resolve_path(step.get("path") or ".", cwd, allow_roots, roots_map)
                res = list_dir(p, limit=int(step.get("limit", max_list_items)))
                last_op = "list"
                last_result = res
            elif op == "stat":
                p = _resolve_path(step.get("path") or ".", cwd, allow_roots, roots_map)
                res = stat(p)
                last_op = "stat"
                last_result = res
            elif op == "read":
                p = _resolve_path(step.get("path") or ".", cwd, allow_roots, roots_map)
                res = {"path": str(p), "content": read_text(p, max_chars=max_read_chars, max_bytes=max_file_bytes)}
                last_content = res["content"]
                last_path = str(p)
                last_op = "read"
                last_result = res
            elif op == "tail":
                p = _resolve_path(step.get("path") or ".", cwd, allow_roots, roots_map)
                lines = int(step.get("lines", 20))
                res = {"path": str(p), "content": read_tail_lines(p, n=lines, max_bytes=max_file_bytes)}
                last_content = res["content"]
                last_path = str(p)
                last_op = "tail"
                last_result = res
            elif op == "find":
                pattern = step.get("pattern") or "*"
                res = {"root": str(cwd), "matches": [str(p) for p in find_files(cwd, pattern, limit=int(step.get("limit", max_list_items)))]}
                last_op = "find"
                last_result = res
            elif op == "grep":
                res = grep_text(cwd, step.get("query") or "", limit_hits=int(step.get("limit_hits", 20)), limit_files=int(step.get("limit_files", 20)))
                last_op = "grep"
                last_result = res
            else:
                raise PlanError("Unsupported op")

            if store_content and op in {"read", "tail"}:
                res["memory_store"] = True
            logger.write({
                "type": "tool_result",
                "session_id": session_id,
                "turn_id": turn_id,
                "tool_result": {
                    "tool": "filesystem",
                    "op": op,
                    "ok": True,
                    "result": _cap_content(res, max_tool_chars),
                    "error": None,
                },
            })
            results.append(res)
        except AmbiguousPathError as e:
            logger.write({
                "type": "tool_result",
                "session_id": session_id,
                "turn_id": turn_id,
                "tool_result": {
                    "tool": "filesystem",
                    "op": op,
                    "ok": False,
                    "result": {"matches": e.matches},
                    "error": "Ambiguous path",
                },
            })
            return {"status": "ambiguous", "choices": e.matches}
        except Exception as e:
            logger.write({
                "type": "tool_result",
                "session_id": session_id,
                "turn_id": turn_id,
                "tool_result": {
                    "tool": "filesystem",
                    "op": op,
                    "ok": False,
                    "result": None,
                    "error": str(e),
                },
            })
            return {"status": "error", "error": str(e)}

    user_intent = plan.get("user_intent") or ""

    return {
        "status": "ok",
        "op": last_op,
        "result": last_result,
        "action": action,
        "memory_store": store_content and bool(last_content),
        "content": last_content,
        "path": last_path,
    }


def _summarizer_prompt(question: str, content: str, recent_context: str) -> str:
    return f"""
You are a concise assistant. Summarize or extract based on the user's request.
Avoid long quotes.

Recent context:
{recent_context}

User request:
{question}

Content:
{content}
""".strip()


def _cap_content(res: dict, max_chars: int) -> dict:
    if not isinstance(res, dict):
        return res
    out = dict(res)
    if "content" in out and isinstance(out["content"], str) and len(out["content"]) > max_chars:
        out["content"] = out["content"][:max_chars]
        out["content_truncated"] = True
    return out


def _build_allow_roots(entries: list[str], project_root: Path) -> list[Path]:
    roots = []
    if not entries:
        entries = [
            "%USERPROFILE%/Desktop",
            "%USERPROFILE%/Documents",
            "<PROJECT_ROOT>",
        ]
    for r in entries:
        if r == "<PROJECT_ROOT>":
            roots.append(project_root)
            continue
        p = Path(os.path.expandvars(r)).expanduser()
        if not p.is_absolute():
            p = (project_root / p).resolve()
        roots.append(p)
    return roots


def _cwd_map(project_root: Path) -> dict:
    user = os.environ.get("USERPROFILE") or str(project_root)
    base = {
        "Desktop": Path(user) / "Desktop",
        "Documents": Path(user) / "Documents",
        "ProjectRoot": project_root,
        "ALON": project_root,
        "project": project_root,
    }
    # add lowercase keys for case-insensitive matching
    base.update({k.lower(): v for k, v in base.items()})
    return base


def _resolve_cwd(label: str, cwd: Path, allow_roots: list[Path], roots_map: dict) -> Path:
    key = (label or "").strip()
    if key in roots_map:
        return roots_map[key]
    low = key.lower()
    if low in roots_map:
        return roots_map[low]
    extracted = _extract_root_and_subpath(label, roots_map)
    if extracted:
        root, sub = extracted
        if not sub:
            return root
        candidate = root / sub
        if candidate.exists() and candidate.is_dir():
            return candidate
        return root
    return _resolve_path(label, cwd, allow_roots, roots_map)


def _resolve_path(path_str: str, cwd: Path, allow_roots: list[Path], roots_map: dict) -> Path:
    key = (path_str or "").strip()
    if key in roots_map:
        return roots_map[key]
    low = key.lower()
    if low in roots_map:
        return roots_map[low]
    extracted = _extract_root_and_subpath(path_str, roots_map)
    if extracted:
        root, sub = extracted
        if not sub:
            return root
        candidate = root / sub
        return resolve_path(str(candidate), root, allow_roots)
    p = resolve_path(path_str, cwd, allow_roots)
    if p.exists():
        return p
    # try fuzzy search in cwd
    name = Path(path_str).name
    matches = [str(p) for p in find_files(cwd, name, limit=10)]
    if len(matches) == 1:
        return Path(matches[0])
    if len(matches) > 1:
        raise AmbiguousPathError(matches)
    raise FileNotFoundError(f"Path not found: {path_str}")


def _extract_root_and_subpath(path_str: str, roots_map: dict) -> Optional[tuple[Path, str]]:
    if not path_str:
        return None
    s = path_str.strip().strip('"').strip("'")
    m = re.search(r"\b(desktop|documents|projectroot|project|alon)\b(.*)", s, re.I)
    if not m:
        return None
    key = m.group(1).lower()
    root = roots_map.get(key)
    if not root:
        return None
    tail = (m.group(2) or "").strip()
    if not tail:
        return root, ""
    tail = re.sub(r"^[\\/:\\s]+", "", tail)
    tail = re.sub(r"^(and|then|to|in|inside|there|the)\\b\\s*", "", tail, flags=re.I)
    file_match = re.search(r"([\\w\\-. ]+\\.[A-Za-z0-9]{1,8})", tail)
    sub = file_match.group(1).strip() if file_match else tail
    return root, sub
