import uuid
import json
import re
import subprocess
import os
import warnings
import logging
from contextlib import contextmanager
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import yaml

# Pre-set ffmpeg path early to avoid pydub warning on import.
_ffmpeg_early = shutil.which("ffmpeg")
if _ffmpeg_early:
    os.environ.setdefault("FFMPEG_BINARY", _ffmpeg_early)

# Allow importing from project root (so `services.*` imports work when running this file directly)
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from logger import JsonlLogger
from services.tts.text_preprocess import preprocess_for_tts

from services.tools.weather import (
    geocode,
    forecast_daily,
    select_day,
    what_to_wear,
    speakable_summary,
)
from services.tools.websearch import search_web
from services.tools.filesystem.runner import (
    parse_plan as fs_parse_plan,
    validate_plan as fs_validate_plan,
    run_plan as fs_run_plan,
    PlanError as FsPlanError,
)
from services.tools.filesystem.prompts import planner_prompt as fs_planner_prompt
from tool_planner import plan_search, summarize_search
from memory_from_logs import (
    configure as mem_configure,
    render_recent,
    render_range,
)
from response_composer import strip_parentheses_for_tts
import threading
from services.orchestrator.lmstudio_client import chat_completion, list_models
from services.orchestrator.router_llm import decide as router_decide
from services.orchestrator.context_manager import (
    check_server_fingerprint,
    router_warmed,
    brain_warmed,
    mark_router_warmed,
    mark_brain_warmed,
)
from services.orchestrator.prompt_builder import (
    build_router_messages,
    build_brain_messages,
    format_tool_results,
    enforce_budget,
)
from services.orchestrator.mcp_backend import MCPToolBackend
from services.orchestrator.ui_console import TerminalUI
from services.tools.codex.codex_tool import run_codex
from services.tools.filesystem.policy import DESKTOP_ROOT
from services.tools.filesystem import fs_tool
from format_validators import (
    extract_urls,
    ensure_no_urls_in_spoken,
    move_urls_to_parentheses,
    enforce_max_links,
    enforce_length_caps,
    enforce_core_facts,
)

_LAST_TOOL_ERROR = None
_LAST_SEARCH_RESULT = None
_LAST_SEARCH_QUERY = None

piper_tts = None
maya1_tts = None
luxtts_tts = None

CONFIG_PATH = ROOT / "config" / "default.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_console_settings(console_verbose: bool):
    if console_verbose:
        return
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.ERROR)
    for name in [
        "transformers",
        "huggingface_hub",
        "tqdm",
        "urllib3",
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        from transformers import logging as tlog
        tlog.set_verbosity_error()
    except Exception:
        pass
    try:
        from huggingface_hub import logging as hlog
        hlog.set_verbosity_error()
    except Exception:
        pass


def _ensure_ffmpeg(console_verbose: bool = False) -> None:
    # Try PATH first, then common install locations
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        candidates = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
            r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        ]
        for p in candidates:
            if Path(p).exists():
                ffmpeg = p
                break
    if ffmpeg:
        os.environ.setdefault("FFMPEG_BINARY", ffmpeg)
        try:
            from pydub import AudioSegment
            AudioSegment.converter = ffmpeg
        except Exception:
            pass
        if console_verbose:
            print(f"[ffmpeg] using {ffmpeg}")
    elif console_verbose:
        print("[ffmpeg] not found on PATH")


def _log_llm_provider(logger: JsonlLogger, session_id: str, turn_id: str, cfg: dict):
    llm_cfg = cfg.get("llm") or {}
    router_cfg = llm_cfg.get("router") or {}
    brain_cfg = llm_cfg.get("brain") or {}
    logger.write({
        "type": "llm_provider",
        "session_id": session_id,
        "turn_id": turn_id,
        "provider": "lmstudio",
        "model": brain_cfg.get("model", ""),
        "router_model": router_cfg.get("model", ""),
        "ts": logger.now_iso(),
    })


def tts_piper(piper_model_path: Path, text: str, out_wav: Path):
    global piper_tts
    if piper_tts is None:
        from services.tts import piper_tts as _piper
        piper_tts = _piper
    piper_tts.speak(text, piper_model_path, out_wav)


class AsyncWavPlayer:
    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        try:
            import winsound
            self._winsound = winsound
        except Exception:
            self._winsound = None

    def play(self, wav_path: Path) -> None:
        with self._lock:
            self.stop()
            if self._winsound:
                self._winsound.PlaySound(
                    str(wav_path),
                    self._winsound.SND_FILENAME | self._winsound.SND_ASYNC,
                )
                return
            ps = f"""
Add-Type -AssemblyName presentationCore
$player = New-Object system.media.soundplayer "{wav_path}"
$player.PlaySync()
"""
            self._proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-Command", ps],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def stop(self) -> None:
        if self._winsound:
            try:
                self._winsound.PlaySound(None, self._winsound.SND_PURGE)
            except Exception:
                pass
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self._proc = None


_AUDIO_PLAYER = None


def _get_audio_player() -> AsyncWavPlayer:
    global _AUDIO_PLAYER
    if _AUDIO_PLAYER is None:
        _AUDIO_PLAYER = AsyncWavPlayer()
    return _AUDIO_PLAYER


def play_wav_windows(wav_path: Path):
    _get_audio_player().play(wav_path)


def tool_stub(intent: str) -> str:
    if intent == "search":
        return "Web search isn’t enabled yet. I can add it next."
    if intent == "filesystem":
        return "Filesystem tool isn’t enabled yet. I can add safe read-only browsing next."
    if intent == "executor":
        return "Execution tool isn’t enabled yet. When we add it, it will be sandboxed for safety."
    return ""


def _collect_source_names(result: dict) -> list[str]:
    names = []
    seen = set()
    for r in (result.get("results") or []):
        name = r.get("source_name") or ""
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    for p in (result.get("pages") or []):
        name = p.get("source_name") or ""
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _fallback_search_summary(result: dict, query: str) -> str:
    results = result.get("results") or []
    if not results:
        return "I couldn't find any results for that."
    top = results[0]
    snippet = (top.get("snippet") or "").strip()
    title = (top.get("title") or "").strip()
    if snippet:
        return snippet if snippet.endswith((".", "!", "?")) else snippet + "."
    if title:
        return f"I found a result about {title}."
    return f"I found results about {query}."


def _strip_sources_line(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s*Sources?:.*$", "", text, flags=re.I).strip()


def _compact_advice(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\\s+", text.strip())
    parts = [p.strip().rstrip(".") for p in parts if p.strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0] + "."
    return f"{parts[0]}. Also, {parts[1]}."


def _limit_sentences(text: str, max_sentences: int = 2) -> str:
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\\s+", text.strip())
    if len(parts) <= max_sentences:
        return text.strip()
    return " ".join(parts[:max_sentences]).strip()


def _wants_citations(user_text: str) -> bool:
    t = (user_text or "").lower()
    return bool(re.search(r"\b(source|sources|cite|citation|reference|refs)\b", t))


def _reply_obj(text: str) -> dict:
    return {"spoken_text": text, "display_text": text}


def _build_final(spoken_text: str, display_text: str, turn_id: str) -> dict:
    return {
        "spoken_text": spoken_text or "",
        "display_text": display_text or "",
        "turn_id": turn_id,
    }

def _wants_links(user_text: str) -> bool:
    t = (user_text or "").lower()
    return bool(re.search(r"\b(link|links|url|website|profile link|facebook link)\b", t))


def _compact_search_answer(text: str, user_text: str) -> str:
    if not text:
        return ""
    cleaned = _strip_sources_line(text)
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\\s+", cleaned) if p.strip()]
    if not parts:
        return cleaned
    t = (user_text or "").lower()
    wants_number = any(w in t for w in ["rate", "price", "cost", "exchange", "usd", "eur", "amd", "gbp"])
    if wants_number:
        for p in parts:
            if re.search(r"\d", p):
                return p if p.endswith((".", "!", "?")) else p + "."
    if len(parts) == 1:
        return parts[0] if parts[0].endswith((".", "!", "?")) else parts[0] + "."
    return " ".join(parts[:2]).strip()


def _extract_rate_from_result(result: dict, query: str) -> str | None:
    text = ""
    for p in (result.get("pages") or [])[:2]:
        text += " " + (p.get("text") or "")
    if not text:
        for r in (result.get("results") or [])[:2]:
            text += " " + (r.get("snippet") or "")
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    codes = re.findall(r"\b[A-Z]{3}\b", (query or "").upper())
    codes = [c for c in codes if c.isalpha()]
    if len(codes) >= 2:
        c1, c2 = codes[0], codes[1]
        pat = rf"1\s*{c1}\s*=?\s*([0-9][0-9,\.]*)\s*{c2}"
        m = re.search(pat, text)
        if m:
            return f"1 {c1} = {m.group(1)} {c2}."
    # fallback: first number near 'rate'
    m = re.search(r"\b([0-9][0-9,\.]*)\b", text)
    if m and codes:
        if len(codes) >= 2:
            return f"1 {codes[0]} = {m.group(1)} {codes[1]}."
    return None


def _format_search_results_list(result: dict, user_text: str, max_items: int = 3) -> str:
    results = result.get("results") or []
    if not results:
        return _reply_obj("I didn't find any results.")
    want_links = _wants_links(user_text)
    items = results[:max_items]
    if want_links:
        pairs = []
        for r in items:
            title = (r.get("title") or "Result").strip()
            url = (r.get("url") or "").strip()
            if url:
                pairs.append(f"{title}: {url}")
        if not pairs:
            return _reply_obj("I couldn't extract direct links from the results.")
        spoken = "Here are a few links."
        display = spoken + f" (Links: " + "; ".join(pairs) + ")"
        return {"spoken_text": spoken, "display_text": display}
    titles = [r.get("title") for r in items if r.get("title")]
    if not titles:
        return _reply_obj("I found a few results. Want links?")
    spoken = "Here are a few options: " + "; ".join(titles) + "."
    display = spoken + " (Say 'give links' if you want the URLs.)"
    return {"spoken_text": spoken, "display_text": display}


def _strip_maya_tags(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<[a-z_]+>", "", text, flags=re.I).strip()


def _pick_voice_mode(text: str) -> str:
    t = (text or "").lower()
    if not t:
        return "neutral"
    exclaims = t.count("!")
    excited_words = ["awesome", "great", "amazing", "love", "excited", "yay", "woo", "let's go"]
    calm_words = ["sorry", "apologize", "no worries", "it's okay", "take care", "gentle", "calm"]
    serious_words = ["important", "note", "warning", "deadline", "official", "policy", "error", "risk"]
    if exclaims >= 2 or any(w in t for w in excited_words):
        return "energetic"
    if any(w in t for w in calm_words):
        return "calm"
    if any(w in t for w in serious_words):
        return "serious"
    return "neutral"


def _parse_voice_command(user_text: str) -> str | None:
    if not user_text:
        return None
    text = user_text.lower()
    if not (text.startswith("/voice") or text.startswith("/voise")):
        return None
    parts = user_text.strip().split()
    if len(parts) < 2:
        return None
    return parts[1].strip().lower()


def _parse_trim_command(user_text: str) -> str | None:
    if not user_text:
        return None
    text = user_text.lower()
    if not (text.startswith("/voice") or text.startswith("/voise")):
        return None
    parts = user_text.strip().split()
    if len(parts) < 2:
        return None
    cmd = parts[1].strip().lower()
    if cmd in {"trim_enable", "trim_disable"}:
        return cmd
    return None


def _router_tool_catalog() -> str:
    return (
        "weather.forecast(location, when, include_clothing)\n"
        "websearch.search(query, recency_days)\n"
        "filesystem.list(path), filesystem.read(path), filesystem.tail(path, lines), "
        "filesystem.find(path, pattern), filesystem.grep(path, query), "
        "filesystem.write(path, content, mode), filesystem.mkdir(path), filesystem.delete(path)\n"
        "obsidian.list(dirpath), obsidian.read(filepath), obsidian.write(filepath, content), obsidian.delete(filepath)\n"
        "codex.run(task, output_filename)\n"
    )


def _has_template_vars(value) -> bool:
    if isinstance(value, str):
        return "${" in value and "}" in value
    if isinstance(value, dict):
        return any(_has_template_vars(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_template_vars(v) for v in value)
    return False


def _extract_filename(user_text: str) -> str | None:
    if not user_text:
        return None
    m = re.search(r"\b([A-Za-z0-9._-]+\.[A-Za-z0-9]{1,10})\b", user_text)
    if not m:
        return None
    return m.group(1)


def _looks_like_obsidian_request(tool_name: str, args: dict, user_text: str) -> bool:
    if tool_name.startswith("obsidian."):
        return True
    if not tool_name.startswith("filesystem."):
        return False
    path_blob = " ".join(
        [
            str(args.get("path") or ""),
            str(args.get("filepath") or ""),
            str(args.get("dirpath") or ""),
        ]
    ).lower().replace("\\", "/")
    if "obsidian" in path_blob:
        return True
    text_blob = str(user_text or "").lower()
    if "obsidian" in text_blob and not any(k in path_blob for k in ("desktop", "documents")):
        return True
    return False


def _obsidian_rel_path(path_value: str | None, user_text: str, default_name: str = "ALON.md") -> str:
    raw = str(path_value or "").strip().replace("\\", "/")
    raw = raw.strip("\"'")

    if raw:
        if raw.lower().startswith("~/"):
            raw = raw[2:]
        if "/obsidian/" in raw.lower():
            idx = raw.lower().rfind("/obsidian/")
            raw = raw[idx + len("/obsidian/"):]
        elif raw.lower().startswith("obsidian/"):
            raw = raw[len("obsidian/"):]
        elif raw.lower() in {"obsidian", "obsidian/"}:
            raw = ""

    if not raw:
        filename = _extract_filename(user_text)
        if filename:
            raw = filename

    if not raw:
        raw = default_name

    parts = [p for p in raw.replace("\\", "/").split("/") if p not in {"", ".", ".."}]
    return "/".join(parts) or default_name


def _rewrite_obsidian_tool_call(tool_name: str, args: dict, user_text: str) -> tuple[str, dict]:
    name = (tool_name or "").strip()
    a = dict(args or {})

    if name in {"obsidian.write", "filesystem.write"}:
        filepath = _obsidian_rel_path(a.get("filepath") or a.get("path"), user_text)
        content = str(a.get("content") or a.get("text") or "")
        return "obsidian.write", {"filepath": filepath, "content": content}

    if name in {"obsidian.read", "filesystem.read", "filesystem.tail"}:
        filepath = _obsidian_rel_path(a.get("filepath") or a.get("path"), user_text)
        return "obsidian.read", {"filepath": filepath}

    if name in {"obsidian.list", "filesystem.list"}:
        dirpath = _obsidian_rel_path(a.get("dirpath") or a.get("path"), user_text, default_name="")
        if not dirpath:
            return "obsidian.list", {}
        return "obsidian.list", {"dirpath": dirpath}

    if name in {"obsidian.delete", "filesystem.delete"}:
        filepath = _obsidian_rel_path(a.get("filepath") or a.get("path"), user_text)
        return "obsidian.delete", {"filepath": filepath, "confirm": True}

    return name, a


def _normalize_fs_args(op: str, args: dict, user_text: str) -> tuple[dict, str | None]:
    args = dict(args or {})

    def _replace_known_templates(s: str) -> str:
        if not isinstance(s, str):
            return s
        repl = {
            "${desktop}": str(DESKTOP_ROOT),
            "${documents}": str(Path.home() / "Documents"),
            "${home}": str(Path.home()),
            "${location}": str(Path.home()),
            "${user_home}": str(Path.home()),
        }
        out = s
        for k, v in repl.items():
            out = out.replace(k, v)
        return out

    for k, v in list(args.items()):
        if isinstance(v, str):
            args[k] = _replace_known_templates(v)

    # Only path-like placeholders are forbidden; content may legitimately contain ${...}.
    path_like_keys = {"path", "file_path", "filepath", "target_path", "cwd", "directory", "root", "dir"}
    path_like_values = {k: v for k, v in args.items() if k in path_like_keys}
    if _has_template_vars(path_like_values):
        # Common recovery: "${path}" means "the referenced file path" -> infer from user text on Desktop.
        raw_path = str(args.get("path") or args.get("file_path") or "").strip()
        if raw_path in {"${path}", "${file_path}", "${target_path}"}:
            filename = _extract_filename(user_text)
            if filename:
                args["path"] = str(DESKTOP_ROOT / filename)
            else:
                return {}, "filesystem tool args contain unresolved template variables"
        else:
            return {}, "filesystem tool args contain unresolved template variables"

    def _first(*keys):
        for k in keys:
            v = args.get(k)
            if v is not None and str(v).strip() != "":
                return v
        return None

    path = _first("path", "file_path", "filepath", "target_path")
    cwd = _first("cwd", "directory", "root", "dir")
    if not path and cwd and op in {"list", "find", "grep"}:
        path = cwd

    if isinstance(path, str):
        raw_path = path.strip()
        token = raw_path.lower().replace("\\", "/")
        if token in {"desktop", "~/desktop", "%userprofile%/desktop"}:
            path = str(DESKTOP_ROOT)
        elif token in {"documents", "~/documents", "%userprofile%/documents"}:
            path = str(Path.home() / "Documents")
        elif token in {"projectroot", "<project_root>", "root"}:
            path = str(ROOT)
        elif token.startswith("desktop/"):
            rel = raw_path.replace("\\", "/").split("/", 1)[1]
            path = str(DESKTOP_ROOT / rel)
        elif token.startswith("documents/"):
            rel = raw_path.replace("\\", "/").split("/", 1)[1]
            path = str((Path.home() / "Documents") / rel)

    if op == "write" and not path:
        filename = _first("filename", "name") or _extract_filename(user_text)
        if filename:
            path = str(DESKTOP_ROOT / str(filename))

    if op in {"list", "find", "grep"} and not path:
        path = str(DESKTOP_ROOT)

    if op in {"read", "tail"} and path:
        p = Path(os.path.expandvars(os.path.expanduser(str(path))))
        if not p.is_absolute():
            path = str(DESKTOP_ROOT / p)

    if op in {"write", "mkdir", "delete"} and path:
        p = Path(os.path.expandvars(os.path.expanduser(str(path))))
        if not p.is_absolute():
            path = str(DESKTOP_ROOT / p)

    norm: dict = {}
    if path is not None:
        norm["path"] = str(path)

    if op == "tail":
        lines = _first("lines", "n_lines", "line_count", "lines_count")
        try:
            norm["lines"] = int(lines) if lines is not None else 50
        except Exception:
            norm["lines"] = 50

    if op == "find":
        norm["pattern"] = str(_first("pattern", "glob", "name", "file_pattern") or "*")

    if op == "grep":
        query = _first("query", "pattern", "text")
        if query is None:
            return norm, "filesystem.grep requires a query"
        norm["query"] = str(query)
        limit_hits = _first("limit_hits", "max_hits")
        limit_files = _first("limit_files", "max_files")
        if limit_hits is not None:
            try:
                norm["limit_hits"] = int(limit_hits)
            except Exception:
                pass
        if limit_files is not None:
            try:
                norm["limit_files"] = int(limit_files)
            except Exception:
                pass

    if op == "write":
        norm["content"] = str(_first("content", "text") or "")
        mode = str(_first("mode") or "overwrite").strip().lower()
        norm["mode"] = "append" if mode == "append" else "overwrite"
        if norm["mode"] == "overwrite" and norm["content"] == "":
            t = (user_text or "").lower()
            allow_empty = any(k in t for k in ["create file", "create empty", "touch", "blank file", "empty file"])
            if not allow_empty:
                return norm, "filesystem.write would overwrite with empty content; provide content explicitly"

    required = {
        "read": ["path"],
        "tail": ["path"],
        "write": ["path"],
        "mkdir": ["path"],
        "delete": ["path"],
    }
    for key in required.get(op, []):
        if not norm.get(key):
            return norm, f"filesystem.{op} requires {key}"
    return norm, None


def _normalize_tool_out(out) -> dict:
    if isinstance(out, dict) and "ok" in out:
        return out
    if isinstance(out, str):
        return {"ok": False, "error": out, "result": None}
    return {"ok": True, "result": out, "error": None}


def _normalize_mcp_tool_out(tool_name: str, args: dict, mcp_result: dict) -> dict:
    payload = mcp_result or {}
    structured = payload.get("structured") if isinstance(payload, dict) else None
    text = payload.get("text") if isinstance(payload, dict) else None
    raw = payload.get("raw") if isinstance(payload, dict) else payload
    raw_is_error = isinstance(raw, dict) and bool(raw.get("isError"))
    if raw_is_error:
        err_text = text or "mcp_tool_error"
        return {"ok": False, "result": None, "error": err_text}

    if tool_name.startswith("weather"):
        if isinstance(structured, dict):
            out = dict(structured)
            out.setdefault("location", args.get("location"))
            out.setdefault("when", args.get("when") or args.get("day") or "today")
            if text and "summary" not in out:
                out["summary"] = text
            return {"ok": True, "result": out, "error": None}
        return {
            "ok": True,
            "result": {
                "location": args.get("location") or "Yerevan",
                "when": args.get("when") or args.get("day") or "today",
                "summary": text or str(raw),
                "text": text or str(raw),
            },
            "error": None,
        }

    if tool_name == "websearch.search":
        sources = []
        if isinstance(structured, dict) and isinstance(structured.get("sources"), list):
            for s in structured.get("sources")[:5]:
                if isinstance(s, dict):
                    sources.append({"title": s.get("title"), "url": s.get("url")})
        spoken = ""
        if isinstance(structured, dict):
            spoken = (
                structured.get("answer_spoken")
                or structured.get("answer")
                or structured.get("summary")
                or ""
            )
        if not spoken:
            spoken = text or str(raw)
        display = spoken
        if sources:
            source_items = [f"{(s.get('title') or 'Source')} — {s.get('url')}" for s in sources if s.get("url")]
            if source_items:
                display = f"{display} (Sources: " + "; ".join(source_items) + ")"
        return {
            "ok": True,
            "result": {
                "query": args.get("query"),
                "provider": "mcp",
                "final_answer": {
                    "answer_spoken": spoken,
                    "answer_display": display,
                    "sources": sources,
                },
                "text": spoken,
                "raw": raw,
            },
            "error": None,
        }

    if tool_name.startswith("filesystem."):
        if isinstance(structured, dict):
            out = dict(structured)
            out.setdefault("op", tool_name.split(".", 1)[1] if "." in tool_name else "op")
            out.setdefault("path", args.get("path") or args.get("cwd") or "")
            if text and not out.get("content"):
                out["content"] = text
            return {"ok": True, "result": out, "error": None}
        return {
            "ok": True,
            "result": {
                "op": tool_name.split(".", 1)[1] if "." in tool_name else "op",
                "path": args.get("path") or args.get("cwd") or "",
                "text": text or str(raw),
                "raw": raw,
            },
            "error": None,
        }

    if tool_name.startswith("obsidian."):
        op = tool_name.split(".", 1)[1] if "." in tool_name else "op"
        filepath = args.get("filepath") or args.get("path") or args.get("dirpath") or ""
        if isinstance(structured, dict):
            out = dict(structured)
            out.setdefault("op", op)
            out.setdefault("filepath", filepath)
            if text and not out.get("text"):
                out["text"] = text
            return {"ok": True, "result": out, "error": None}
        return {
            "ok": True,
            "result": {
                "op": op,
                "filepath": filepath,
                "text": text or str(raw),
                "raw": raw,
            },
            "error": None,
        }

    return {"ok": True, "result": structured or {"text": text or str(raw), "raw": raw}, "error": None}


def _truncate_tool_payload(obj, max_chars: int):
    if max_chars <= 0:
        return obj
    if isinstance(obj, str):
        if len(obj) > max_chars:
            return obj[:max_chars].rstrip() + "..."
        return obj
    if isinstance(obj, list):
        return [_truncate_tool_payload(i, max_chars) for i in obj]
    if isinstance(obj, dict):
        return {k: _truncate_tool_payload(v, max_chars) for k, v in obj.items()}
    return obj


def _is_overflow_error(err: Exception) -> bool:
    msg = str(err).lower()
    triggers = [
        "context length",
        "maximum context",
        "too large",
        "max tokens",
        "prompt is too long",
        "context window",
    ]
    return any(t in msg for t in triggers)


def _snapshot_desktop() -> dict[str, float]:
    snap = {}
    try:
        for p in DESKTOP_ROOT.rglob("*"):
            if p.is_file():
                try:
                    snap[str(p)] = p.stat().st_mtime
                except Exception:
                    continue
    except Exception:
        return snap
    return snap


def _diff_desktop(before: dict[str, float], after: dict[str, float]) -> list[str]:
    changed = []
    for path, mtime in after.items():
        if path not in before or before[path] != mtime:
            changed.append(path)
    return changed


# _nullcontext retained for internal non-UI usage
@contextmanager
def _nullcontext():
    yield


def _looks_like_memory_question(text: str) -> bool:
    t = (text or "").lower()
    if any(w in t for w in ["remember", "recall", "what happened", "yesterday", "last week", "earlier", "before"]):
        if not any(w in t for w in ["file", "folder", "directory", "read", "open", "list", "tail", "find", "search"]):
            return True
    return False


def _looks_like_why_question(text: str) -> bool:
    return bool(re.match(r"^\s*why\b", (text or "").strip(), flags=re.I))


def _set_last_tool_error(tool: str, error: str) -> None:
    global _LAST_TOOL_ERROR
    _LAST_TOOL_ERROR = {"tool": tool, "error": error}


def _clear_last_tool_error() -> None:
    global _LAST_TOOL_ERROR
    _LAST_TOOL_ERROR = None


def _set_last_search(result: dict, query: str) -> None:
    global _LAST_SEARCH_RESULT, _LAST_SEARCH_QUERY
    _LAST_SEARCH_RESULT = result
    _LAST_SEARCH_QUERY = query


def _has_last_search() -> bool:
    return bool(_LAST_SEARCH_RESULT)


def _looks_like_search_request(text: str) -> bool:
    t = (text or "").lower()
    # explicit command patterns (imperative or direct request)
    cmd_patterns = [
        r"^(search|google|lookup|look up|find)\b",
        r"\b(search|google|lookup|look up|find)\s+(for|about)\b",
        r"\bsearch the web\b",
        r"\blook up\b",
        r"\bgoogle\b",
        r"\bfind on the web\b",
        r"\bcan you (search|google|look up|lookup|find)\b",
        r"\bplease (search|google|look up|lookup|find)\b",
        r"\bdo a search\b",
    ]
    if any(re.search(p, t) for p in cmd_patterns):
        return True
    return False


def _needs_fresh_info(text: str) -> bool:
    t = (text or "").lower()
    if any(w in t for w in ["today", "right now", "current", "currently", "latest", "recent", "breaking", "news"]):
        return True
    if any(w in t for w in ["price", "stock", "earnings", "forecast", "release date", "launch date"]):
        return True
    if any(w in t for w in ["schedule", "standings", "score", "results", "flight", "ticket", "availability"]):
        return True
    if any(w in t for w in ["ceo", "president", "prime minister", "leader", "who is the current"]):
        return True
    return False


def _should_search(text: str) -> bool:
    return _looks_like_search_request(text) or _needs_fresh_info(text)


def _looks_like_more_results_request(text: str) -> bool:
    t = (text or "").lower()
    return any(
        phrase in t
        for phrase in [
            "more results",
            "show more",
            "give more",
            "provide more",
            "provide several",
            "several",
            "another",
            "others",
            "more please",
        ]
    )


def _summarize_tool_result(intent: str, tool_out: dict) -> dict | None:
    if not isinstance(tool_out, dict):
        return None
    if not tool_out.get("ok"):
        return {"intent": intent, "error": tool_out.get("error")}
    data = tool_out.get("result") or {}
    if intent == "weather":
        return {
            "location": data.get("location"),
            "when": data.get("when"),
            "t_min": data.get("t_min"),
            "t_max": data.get("t_max"),
            "precip_prob_max": data.get("precip_prob_max"),
            "wind_max": data.get("wind_max"),
        }
    if intent == "calendar":
        return {
            "date": data.get("date"),
            "weekday": data.get("weekday"),
            "holiday_hint": data.get("holiday_hint"),
        }
    if intent == "search":
        results = data.get("results") or []
        return {
            "query": data.get("query") or tool_out.get("query"),
            "top_titles": [r.get("title") for r in results[:3] if r.get("title")],
        }
    if intent == "filesystem":
        res = data.get("result") or data
        return {
            "op": data.get("op") or tool_out.get("op") or res.get("op"),
            "path": data.get("path") or res.get("path"),
        }
    return None


def _clamp_range(start_iso: str, end_iso: str, max_range_days: int) -> tuple[str, str]:
    try:
        start_dt = datetime.fromisoformat(start_iso)
        end_dt = datetime.fromisoformat(end_iso)
    except Exception:
        return start_iso, end_iso
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt
    max_seconds = max_range_days * 86400
    if (end_dt - start_dt).total_seconds() > max_seconds:
        end_dt = start_dt + timedelta(days=max_range_days)
    return start_dt.isoformat(), end_dt.isoformat()


def _normalize_weather_day(when: str) -> tuple[str, bool]:
    if not when:
        return "today", False
    w = str(when).strip().lower()
    if w in {"today", "tomorrow"}:
        return w, False
    if re.match(r"^\d{4}-\d{2}-\d{2}$", w):
        return w, False
    if w in {"this_week", "this week", "week"}:
        return "today", True
    return "today", False


def _extract_relative_days(text: str) -> int | None:
    t = (text or "").lower()
    m = re.search(r"\b(in|after)\s+(\d+)\s+days?\b", t)
    if not m:
        m = re.search(r"\b(\d+)\s+days?\s+(from now|later)\b", t)
    if m:
        try:
            return int(m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1))
        except Exception:
            return None
    return None




def _speak_list_dir(result: dict) -> str:
    entries = result.get("entries") or []
    count = len(entries)
    path = result.get("path") or "."
    top = ", ".join([e.get("name") for e in entries[:5] if e.get("name")])
    truncated = result.get("truncated")
    if top:
        msg = f"I found {count} items in {path}. Top: {top}."
    else:
        msg = f"I found {count} items in {path}."
    if truncated:
        msg = f"{msg} Showing the first {count}."
    return msg


def _speak_search(result: dict) -> str:
    hits = result.get("hits") or []
    total = result.get("hits_returned") or len(hits)
    files = result.get("files_scanned") or 0
    if not hits:
        return "I didn't find any matches."
    top_hits = []
    for h in hits[:3]:
        file = h.get("file") or ""
        line = h.get("line")
        text = h.get("text") or ""
        if file:
            if line:
                top_hits.append(f"{file}:{line} {text}")
            else:
                top_hits.append(f"{file} {text}")
    top_str = " | ".join(top_hits)
    return f"I found {total} matches across {files} files. Top hits: {top_str}."


def _speak_read_file(result: dict, llm_generate) -> str:
    path = result.get("path") or "the file"
    start_line = result.get("start_line")
    end_line = result.get("end_line")
    content = (result.get("content") or "")[:2000]
    prompt = f"""
You are a concise assistant. Summarize the file content in 1-2 sentences.
Avoid long quotes. Do not include any file paths in the summary.
Content:
{content}
""".strip()
    summary = (llm_generate(prompt) or "").strip()
    summary = _limit_sentences(summary, 2) or "I read the file."
    if start_line and end_line:
        return f"{summary} I read lines {start_line} to {end_line} from {path}."
    return f"{summary} I read {path}."


def _speak_summary(result: dict) -> str:
    summary = (result.get("summary") or "").strip()
    summary = _limit_sentences(summary, 2)
    return summary or "Here's a brief summary."


def _handle_weather_from_plan(
    cfg: dict,
    tool_args: dict,
    user_text: str,
    logger: JsonlLogger,
    session_id: str,
    turn_id: str,
    llm_generate,
    timezone: str,
) -> str:
    weather_cfg = (cfg.get("tools") or {}).get("weather") or {}
    default_location = weather_cfg.get("default_location", "Yerevan")
    forecast_days = int(weather_cfg.get("forecast_days", 3))
    max_forecast_days = int(weather_cfg.get("max_forecast_days", 16))
    assistant_name = (cfg.get("assistant") or {}).get("name", "") or ""

    location = (tool_args.get("location") or "").strip() or default_location
    loc_l = location.lower()
    if any(k in loc_l for k in ["user's location", "user location", "infer", "ip", "unknown"]):
        location = default_location
    if assistant_name and location.strip().lower() == assistant_name.strip().lower():
        location = default_location
    when = tool_args.get("when") or tool_args.get("day") or "today"
    include_wear = bool(re.search(r"\b(what to wear|what should i wear|wear|clothes|jacket|coat|outfit)\b", user_text or "", re.I))
    week_note = False
    rel_days = _extract_relative_days(user_text)
    if rel_days is not None:
        today = datetime.now(ZoneInfo(timezone)).date()
        day = (today + timedelta(days=rel_days)).isoformat()
        required_days = rel_days + 1
        if required_days > forecast_days:
            forecast_days = min(max_forecast_days, required_days)
    else:
        day, week_note = _normalize_weather_day(when)
        if re.match(r"^\d{4}-\d{2}-\d{2}$", day):
            try:
                today = datetime.now(ZoneInfo(timezone)).date()
                target = datetime.fromisoformat(day).date()
                diff = (target - today).days
                if diff >= 0:
                    required_days = diff + 1
                    if required_days > forecast_days:
                        forecast_days = min(max_forecast_days, required_days)
            except Exception:
                pass

    call_id = str(uuid.uuid4())
    logger.write({
        "type": "tool_call",
        "session_id": session_id,
        "turn_id": turn_id,
        "tool_call": {
            "id": call_id,
            "tool": "weather.get_forecast",
            "args": {
                "location": location,
                "day": day,
                "include_wear_advice": include_wear,
            },
            "plan_args": tool_args,
        },
    })

    try:
        loc = geocode(location)
        data = forecast_daily(loc, days=forecast_days)
        day_forecast, label = select_day(data, day)
        summary = speakable_summary(loc.name, day_forecast, label=label)

        result = {
            "location": loc.name,
            "when": day,
            "t_min": getattr(day_forecast, "t_min", None),
            "t_max": getattr(day_forecast, "t_max", None),
            "precip_prob_max": getattr(day_forecast, "precip_prob_max", None),
            "wind_max": getattr(day_forecast, "wind_max", None),
            "condition": getattr(day_forecast, "weather_code", None),
            "summary": summary,
        }
        if include_wear:
            result["wear_advice"] = _compact_advice(what_to_wear(day_forecast))

        logger.write({
            "type": "tool_result",
            "session_id": session_id,
            "turn_id": turn_id,
            "tool_result": {
                "id": call_id,
                "tool": "weather.get_forecast",
                "ok": True,
                "result": result,
                "error": None
            },
        })

        _clear_last_tool_error()
        if week_note:
            result["note"] = "I can check a specific day this week if you want."
        return {"ok": True, "result": result}
    except Exception as e:
        _set_last_tool_error("weather", str(e))
        logger.write({
            "type": "tool_result",
            "session_id": session_id,
            "turn_id": turn_id,
            "tool_result": {
                "id": call_id,
                "tool": "weather.get_forecast",
                "ok": False,
                "result": None,
                "error": str(e)
            },
        })
        return {"ok": False, "error": str(e), "result": None}


def _handle_search_from_plan(
    cfg: dict,
    tool_args: dict,
    user_text: str,
    logger: JsonlLogger,
    session_id: str,
    turn_id: str,
    llm_generate,
    ui=None,
) -> str:
    web_cfg = (cfg.get("tools") or {}).get("websearch") or {}
    enabled = web_cfg.get("enabled", False)
    if not enabled:
        return {"ok": False, "error": "search disabled", "result": None}

    query = (tool_args.get("query") or "").strip() or user_text.strip()
    max_results = int(tool_args.get("max_results") or web_cfg.get("max_results", 5))
    fetch_top_n_pages = int(web_cfg.get("fetch_top_n_pages", 2))
    user_agent = web_cfg.get("user_agent", "Alon/0.1 (+local assistant)")
    timeout_sec = int(web_cfg.get("timeout_sec", 20))
    cache_ttl_seconds = int(web_cfg.get("cache_ttl_seconds", 21600))
    summarize = tool_args.get("summarize")
    if summarize is None:
        summarize = True
    citations = _wants_citations(user_text)

    settings = {
        "provider": str(web_cfg.get("provider", "local")).lower(),
        "max_chars_per_page_clean": int(web_cfg.get("max_chars_per_page_clean", 12000)),
        "max_chars_per_chunk": int(web_cfg.get("max_chars_per_chunk", 2500)),
        "max_chunks_per_page": int(web_cfg.get("max_chunks_per_page", 6)),
        "llm_chunk_max_tokens": int(web_cfg.get("llm_chunk_max_tokens", 250)),
        "llm_reduce_max_tokens": int(web_cfg.get("llm_reduce_max_tokens", 300)),
        "llm_final_max_tokens": int(web_cfg.get("llm_final_max_tokens", 350)),
        "request_timeout_sec": int(web_cfg.get("request_timeout_sec", timeout_sec)),
        "max_source_checks": int(web_cfg.get("max_source_checks", max_results)),
        "min_match_score": float(web_cfg.get("min_match_score", 0.35)),
        "gemini": (web_cfg.get("gemini") or {}),
    }

    def _search_progress(evt: dict) -> None:
        stage = evt.get("stage")
        if stage == "source_check":
            idx = evt.get("index")
            total = evt.get("total")
            url = evt.get("url") or ""
            if ui:
                ui.print_status(f"🔎 Checking source {idx}/{total}: {url}")
        elif stage == "search_results":
            if ui:
                ui.print_status(f"🌐 Search results: {evt.get('count', 0)} candidates")
        elif stage == "gemini_start":
            if ui:
                ui.print_status(f"☁ Gemini search… ({evt.get('model','gemini')})")
        elif stage == "gemini_success":
            if ui:
                ui.print_status(f"✅ Gemini grounded answer ({evt.get('sources', 0)} sources)")
        elif stage == "gemini_fallback":
            if ui:
                ui.print_status("↪ Gemini unavailable/weak, switching to local search pipeline…")
        elif stage == "source_match":
            matched = bool(evt.get("matched"))
            if ui and not matched:
                ui.print_status("↪ No direct answer here, trying next source…")
        elif stage == "answer_found":
            if ui:
                ui.print_status("✅ Found relevant source")
        elif stage == "answer_not_found":
            if ui:
                ui.print_status("⚠ No exact source match found; using best available result")
        elif stage == "cache_hit":
            if ui:
                ui.print_status("📦 Using cached search result")
        logger.write({
            "type": "search_progress",
            "session_id": session_id,
            "turn_id": turn_id,
            "progress": evt,
            "ts": logger.now_iso(),
        })

    call_id = str(uuid.uuid4())
    logger.write({
        "type": "tool_call",
        "session_id": session_id,
        "turn_id": turn_id,
        "tool_call": {
            "id": call_id,
            "tool": "websearch.search",
            "args": {
                "query": query,
                "max_results": max_results,
            },
            "plan_args": tool_args,
        },
    })

    try:
        cache_dir = ROOT / "data" / "cache" / "web"
        result = search_web(
            query=query,
            max_results=max_results,
            fetch_top_n_pages=fetch_top_n_pages,
            user_agent=user_agent,
            timeout_sec=timeout_sec,
            cache_ttl_seconds=cache_ttl_seconds,
            cache_dir=cache_dir,
            llm_chat_func=(lambda sys, user, max_tokens=None: llm_generate(sys, user, max_tokens=max_tokens))
            if callable(llm_generate)
            else None,
            turn_id=turn_id,
            include_sources=citations,
            settings=settings,
            user_text=user_text,
            progress_cb=_search_progress,
        )

        logger.write({
            "type": "tool_result",
            "session_id": session_id,
            "turn_id": turn_id,
            "tool_result": {
                "id": call_id,
                "tool": "websearch.search",
                "ok": True,
                "result": result,
                "error": None
            },
        })

        _set_last_search(result, query)
        _clear_last_tool_error()
        return {"ok": True, "result": result, "query": query}
    except Exception as e:
        _set_last_tool_error("search", str(e))
        logger.write({
            "type": "tool_result",
            "session_id": session_id,
            "turn_id": turn_id,
            "tool_result": {
                "id": call_id,
                "tool": "websearch.search",
                "ok": False,
                "result": None,
                "error": str(e)
            },
        })
        return {"ok": False, "error": str(e), "result": None}


def handle_websearch(
    cfg: dict,
    user_text: str,
    logger: JsonlLogger,
    session_id: str,
    turn_id: str,
    llm_generate,
) -> str:
    web_cfg = (cfg.get("tools") or {}).get("websearch") or {}
    enabled = web_cfg.get("enabled", False)
    if not enabled:
        return "Web search isn’t enabled yet. I can add it next."

    planner_enabled = ((cfg.get("orchestrator") or {}).get("llm_tool_planner") or {}).get("enabled", False)
    planner_llm = llm_generate if planner_enabled else (lambda _p: "")

    max_results = int(web_cfg.get("max_results", 5))
    fetch_top_n_pages = int(web_cfg.get("fetch_top_n_pages", 2))
    user_agent = web_cfg.get("user_agent", "Alon/0.1 (+local assistant)")
    timeout_sec = int(web_cfg.get("timeout_sec", 20))
    cache_ttl_seconds = int(web_cfg.get("cache_ttl_seconds", 21600))
    settings = {
        "provider": str(web_cfg.get("provider", "local")).lower(),
        "max_chars_per_page_clean": int(web_cfg.get("max_chars_per_page_clean", 12000)),
        "max_chars_per_chunk": int(web_cfg.get("max_chars_per_chunk", 2500)),
        "max_chunks_per_page": int(web_cfg.get("max_chunks_per_page", 6)),
        "llm_chunk_max_tokens": int(web_cfg.get("llm_chunk_max_tokens", 250)),
        "llm_reduce_max_tokens": int(web_cfg.get("llm_reduce_max_tokens", 300)),
        "llm_final_max_tokens": int(web_cfg.get("llm_final_max_tokens", 350)),
        "request_timeout_sec": int(web_cfg.get("request_timeout_sec", timeout_sec)),
        "max_source_checks": int(web_cfg.get("max_source_checks", max_results)),
        "min_match_score": float(web_cfg.get("min_match_score", 0.35)),
        "gemini": (web_cfg.get("gemini") or {}),
    }

    plan = plan_search(planner_llm, user_text)
    plan_args = plan.get("args") or {}
    plan_args["max_results"] = max_results

    call_id = str(uuid.uuid4())
    logger.write({
        "type": "tool_call",
        "session_id": session_id,
        "turn_id": turn_id,
        "tool_call": {
            "id": call_id,
            "tool": "websearch.search",
            "args": plan_args,
            "plan": plan,
        },
    })

    try:
        cache_dir = ROOT / "data" / "cache" / "web"
        def _search_progress(evt: dict) -> None:
            logger.write({
                "type": "search_progress",
                "session_id": session_id,
                "turn_id": turn_id,
                "progress": evt,
                "ts": logger.now_iso(),
            })
        result = search_web(
            query=plan_args.get("query") or user_text,
            max_results=max_results,
            fetch_top_n_pages=fetch_top_n_pages,
            user_agent=user_agent,
            timeout_sec=timeout_sec,
            cache_ttl_seconds=cache_ttl_seconds,
            cache_dir=cache_dir,
            llm_chat_func=(lambda sys, user, max_tokens=None: llm_generate(sys, user, max_tokens=max_tokens))
            if callable(llm_generate)
            else None,
            turn_id=turn_id,
            include_sources=_wants_citations(user_text),
            settings=settings,
            user_text=user_text,
            progress_cb=_search_progress,
        )

        logger.write({
            "type": "tool_result",
            "session_id": session_id,
            "turn_id": turn_id,
            "tool_result": {
                "id": call_id,
                "tool": "websearch.search",
                "ok": True,
                "result": result,
                "error": None
            },
        })

        _set_last_search(result, plan_args.get("query") or user_text)
        if _wants_links(user_text):
            return _format_search_results_list(result, user_text)

        if planner_enabled:
            summary = summarize_search(
                llm_generate,
                user_text,
                plan_args.get("query") or user_text,
                result.get("results") or [],
                result.get("pages") or [],
            )
        else:
            summary = _fallback_search_summary(result, plan_args.get("query") or user_text)

        rate_answer = _extract_rate_from_result(result, plan_args.get("query") or user_text)
        summary = rate_answer or _compact_search_answer(summary, user_text)
        if _wants_citations(user_text):
            sources = _collect_source_names(result)
            if sources:
                summary = f"{summary} Sources: {', '.join(sources[:5])}."
        return _reply_obj(summary.strip())

    except Exception as e:
        logger.write({
            "type": "tool_result",
            "session_id": session_id,
            "turn_id": turn_id,
            "tool_result": {
                "id": call_id,
                "tool": "websearch.search",
                "ok": False,
                "result": None,
                "error": str(e)
            },
        })
        return _reply_obj("Sorry — I couldn’t search the web right now.")


def handle_weather(
    cfg: dict,
    user_text: str,
    logger: JsonlLogger,
    session_id: str,
    turn_id: str,
    llm_generate,
    timezone: str,
    tool_args: dict | None = None,
) -> str:
    weather_cfg = (cfg.get("tools") or {}).get("weather") or {}
    location = weather_cfg.get("default_location", "Yerevan")
    forecast_days = int(weather_cfg.get("forecast_days", 3))

    plan_args = tool_args or {}
    location = plan_args.get("location") or location
    day = plan_args.get("when") or plan_args.get("day") or "today"
    if isinstance(day, str) and day.strip().lower() in {"now", "current"}:
        day = "today"
    include_wear = bool(
        plan_args.get("include_clothing")
        or plan_args.get("include_wear_advice")
        or plan_args.get("whatToWear")
    )
    if isinstance(location, str) and ("infer" in location.lower() or "user" in location.lower()):
        location = weather_cfg.get("default_location", "Yerevan")

    call_id = str(uuid.uuid4())
    logger.write({
        "type": "tool_call",
        "session_id": session_id,
        "turn_id": turn_id,
        "tool_call": {
            "id": call_id,
            "tool": "weather.get_forecast",
            "args": plan_args,
        },
    })

    try:
        loc = geocode(location)
        data = forecast_daily(loc, days=forecast_days)

        day_forecast, label = select_day(data, day)
        summary = speakable_summary(loc.name, day_forecast, label=label)

        result = {
            "location": loc.name,
            "when": day,
            "t_min": getattr(day_forecast, "t_min", None),
            "t_max": getattr(day_forecast, "t_max", None),
            "precip_prob_max": getattr(day_forecast, "precip_prob_max", None),
            "wind_max": getattr(day_forecast, "wind_max", None),
            "condition": getattr(day_forecast, "weather_code", None),
            "summary": summary,
        }
        if include_wear:
            result["wear_advice"] = _compact_advice(what_to_wear(day_forecast))

        logger.write({
            "type": "tool_result",
            "session_id": session_id,
            "turn_id": turn_id,
            "tool_result": {
                "id": call_id,
                "tool": "weather.get_forecast",
                "ok": True,
                "result": result,
                "error": None
            },
        })

        return {"ok": True, "result": result}

    except Exception as e:
        logger.write({
            "type": "tool_result",
            "session_id": session_id,
            "turn_id": turn_id,
            "tool_result": {
                "id": call_id,
                "tool": "weather.get_forecast",
                "ok": False,
                "result": None,
                "error": str(e)
            },
        })
        return {"ok": False, "error": str(e), "result": None}


def _fs_plan_from_tool_args(tool_args: dict) -> dict:
    op = (tool_args.get("op") or "").strip().lower()
    if op not in {"list", "read", "tail", "find", "grep", "stat"}:
        op = "list"
    cwd = tool_args.get("cwd") or "ProjectRoot"
    path = tool_args.get("path") or "."
    step = {"op": op, "path": path}
    if op == "tail":
        step["lines"] = int(tool_args.get("lines") or tool_args.get("n") or 20)
    if op == "find":
        step["pattern"] = tool_args.get("pattern") or "*"
    if op == "grep":
        step["query"] = tool_args.get("query") or ""
        step["limit_hits"] = int(tool_args.get("limit_hits") or 20)
        step["limit_files"] = int(tool_args.get("limit_files") or 20)
    if op == "list":
        step["limit"] = int(tool_args.get("limit") or tool_args.get("max_items") or 50)
    plan = {
        "cwd": cwd,
        "steps": [step],
        "user_intent": op,
    }
    what_to_do = tool_args.get("what_to_do") or tool_args.get("what_to_do_with_content")
    if what_to_do:
        plan["what_to_do_with_content"] = what_to_do
    return plan


def handle_filesystem(
    cfg: dict,
    user_text: str,
    logger: JsonlLogger,
    session_id: str,
    turn_id: str,
    llm_generate,
    recent_context: str,
    tool_args: dict | None = None,
) -> str:
    fs_cfg = cfg.get("filesystem") or {}
    enabled = fs_cfg.get("enabled", True)
    if not enabled:
        return "Filesystem tool isn’t enabled yet."

    allow_roots = fs_cfg.get("allow_roots") or []
    cwd_options = ["Desktop", "Documents", "ProjectRoot"]
    plan = None
    if tool_args and isinstance(tool_args, dict):
        try:
            plan = fs_validate_plan(_fs_plan_from_tool_args(tool_args))
        except Exception:
            plan = None
    if plan is None:
        planner = fs_planner_prompt(user_text, recent_context, allow_roots, cwd_options)
        try:
            plan_text = llm_generate(planner)
            plan = fs_validate_plan(fs_parse_plan(plan_text))
        except FsPlanError:
            return "I can use Desktop, Documents, or ProjectRoot. What should I open or list?"

    result = fs_run_plan(
        plan,
        cfg,
        logger,
        session_id,
        turn_id,
        llm_generate,
        recent_context,
        user_text,
        ROOT,
    )

    if result.get("status") == "ambiguous":
        choices = result.get("choices") or []
        choices = choices[:5]
        return {"ok": False, "error": "ambiguous", "choices": choices, "result": None}
    if result.get("status") == "error":
        _set_last_tool_error("filesystem", result.get("error") or "unknown error")
        return {"ok": False, "error": result.get("error") or "unknown error", "result": None}
    _clear_last_tool_error()
    return {"ok": True, "result": result}


def main():
    cfg = load_config()
    console_verbose = bool((cfg.get("debug") or {}).get("console_verbose", False))
    apply_console_settings(console_verbose)
    _ensure_ffmpeg(console_verbose)

    # Import TTS modules after console settings to suppress noisy warnings.
    global piper_tts, maya1_tts, luxtts_tts
    if piper_tts is None or maya1_tts is None or luxtts_tts is None:
        from services.tts import piper_tts as _piper, maya1_tts as _maya, luxtts_tts as _lux
        piper_tts = _piper
        maya1_tts = _maya
        luxtts_tts = _lux

    llm_cfg = cfg.get("llm") or {}
    router_cfg = llm_cfg.get("router") or {}
    brain_cfg = llm_cfg.get("brain") or {}
    router_base_url = router_cfg.get("base_url", "http://localhost:1234/v1").rstrip("/")
    brain_base_url = brain_cfg.get("base_url", "http://localhost:1234/v1").rstrip("/")
    router_model = router_cfg.get("model", "qwen2.5-3b-instruct")
    brain_model = brain_cfg.get("model", "qwen2.5-7b-instruct-uncensored")
    router_temp = float(router_cfg.get("temperature", 0.2))
    router_max_tokens = int(router_cfg.get("max_tokens", 350))
    brain_temp = float(brain_cfg.get("temperature", 0.6))
    brain_max_tokens = int(brain_cfg.get("max_tokens", 500))

    tts_cfg = cfg.get("tts", {})
    voice_model = ROOT / tts_cfg["piper_voice_model"]
    tts_dir = ROOT / "data" / "cache" / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    out_wav_default = ROOT / tts_cfg["output_wav"]
    tts_engine = (tts_cfg.get("engine") or "piper").lower()
    luxtts_cfg = tts_cfg.get("luxtts") or {}
    luxtts_model_id = luxtts_cfg.get("model_id") or "YatharthS/LuxTTS"
    luxtts_ref_wav = ROOT / (luxtts_cfg.get("reference_voice_wav") or "")
    luxtts_device = luxtts_cfg.get("device", "cuda")
    luxtts_num_steps = int(luxtts_cfg.get("num_step", luxtts_cfg.get("num_steps", 4)))
    luxtts_timeout = int(luxtts_cfg.get("timeout_sec", 20))
    voice_pack = luxtts_cfg.get("voice_pack") or {}
    voice_default = (voice_pack.get("default") or "neutral").lower()
    voice_keys = set((voice_pack.get("voices") or {}).keys())
    current_voice = voice_default
    voice_override = None
    maya_cfg = tts_cfg.get("maya1") or {}
    maya_model_dir = ROOT / (maya_cfg.get("model_dir") or "models/tts/maya1")
    maya_device = maya_cfg.get("device", "cuda")
    maya_dtype = maya_cfg.get("dtype", "float16")

    if tts_engine == "maya1":
        index_file = maya_model_dir / "model.safetensors.index.json"
        if not maya_model_dir.exists() or not index_file.exists():
            print("[WARN] Maya1 model files missing; falling back to Piper.")
            tts_engine = "piper"
    if tts_engine == "luxtts":
        if not luxtts_ref_wav.exists() or not luxtts_ref_wav.is_file():
            print("[WARN] LuxTTS reference wav missing; falling back to Piper.")
            tts_engine = "piper"

    name = cfg["assistant"]["name"]
    style = cfg["assistant"]["system_style"]
    tz = cfg["assistant"]["timezone"]

    log_path = ROOT / cfg["logging"]["jsonl_path"]
    logger = JsonlLogger(log_path, tz)
    mem_configure(log_path)
    memory_cfg = cfg.get("memory") or {}
    memory_turns = int(memory_cfg.get("default_recent_turns", memory_cfg.get("recent_turns", 20)))
    memory_max_turns = int(memory_cfg.get("max_turns", 200))
    memory_tool_chars = int(memory_cfg.get("max_tool_content_chars", 4000))
    router_cold_turns = int(memory_cfg.get("router_default_turns_cold", 8))
    brain_cold_turns = int(memory_cfg.get("brain_default_turns_cold", 30))
    min_turns_retry = int(memory_cfg.get("min_turns_retry", 6))
    memory_hard = bool(memory_cfg.get("hard_mode", False))
    max_context_chars = int(memory_cfg.get("max_context_chars", 1800))
    max_relevant_chars = int(memory_cfg.get("max_relevant_chars", 1200))

    prompt_budget = ((cfg.get("llm") or {}).get("prompt_budget") or {})
    max_chars_total = int(prompt_budget.get("max_chars_total", 16000))
    max_chars_tool_results = int(prompt_budget.get("max_chars_tool_results", 7000))
    max_chars_memory_router = int(prompt_budget.get("max_chars_memory_router", 2500))
    max_chars_memory_brain = int(prompt_budget.get("max_chars_memory_brain", 8000))
    retry_on_overflow = bool(prompt_budget.get("retry_on_overflow", True))

    tools_cfg = cfg.get("tools", {})
    weather_enabled = (tools_cfg.get("weather", {}) or {}).get("enabled", False)
    websearch_enabled = (tools_cfg.get("websearch", {}) or {}).get("enabled", False)
    filesystem_enabled = (tools_cfg.get("filesystem", {}) or {}).get("enabled", False)
    filesystem_enabled = (cfg.get("filesystem") or {}).get("enabled", filesystem_enabled)
    search_enabled = (cfg.get("search") or {}).get("enabled", websearch_enabled)
    weather_enabled = (cfg.get("weather") or {}).get("enabled", weather_enabled)
    mcp_backend = MCPToolBackend(cfg)
    mcp_status = mcp_backend.probe()

    try:
        models = list_models(brain_base_url, timeout_sec=10)
        want_router = router_model
        want_brain = brain_model
        first_id = (models[0].get("id") if models else "")
        match_router = any(m.get("id") == want_router for m in models)
        match_brain = any(m.get("id") == want_brain for m in models)
        if console_verbose:
            print(
                f"[LM Studio] models ok. router='{want_router}' match={match_router}, "
                f"brain='{want_brain}' match={match_brain}, first='{first_id}'"
            )
    except Exception:
        print("LM Studio server not running. Start Local Server in LM Studio.")

    ui_cfg = cfg.get("ui") or {}
    fancy_terminal = bool(ui_cfg.get("fancy_terminal", True))
    ui = TerminalUI(assistant_name=name, fancy=fancy_terminal)
    if fancy_terminal:
        ui.show_banner()

    if console_verbose:
        ui.print_debug(f"{name} orchestrator. Type a message. Type 'exit' to quit.")
        if mcp_status.active:
            ui.print_debug(f"MCP active ({len(mcp_status.tools)} tools), mode={mcp_backend.mode}")
        else:
            ui.print_debug(f"MCP inactive ({mcp_status.reason}), mode={mcp_backend.mode}")

    session_id = str(uuid.uuid4())
    logger.write({"type": "session_start", "session_id": session_id})
    logger.write({
        "type": "mcp_status",
        "session_id": session_id,
        "active": mcp_status.active,
        "mode": mcp_backend.mode,
        "transport": mcp_backend.active_transport,
        "reason": mcp_status.reason,
        "tools": mcp_status.tools[:30],
        "ts": logger.now_iso(),
    })

    while True:
        user = ui.prompt_input().strip()
        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            break
        if user.strip().lower() == "/stop":
            _get_audio_player().stop()
            if console_verbose:
                ui.print_status("Audio stopped.")
            continue

        trim_cmd = _parse_trim_command(user)
        if trim_cmd:
            if tts_engine != "luxtts":
                reply = "Silence trimming is only available for LuxTTS."
            else:
                trim_cfg = luxtts_cfg.get("silence_trim") or {}
                if trim_cmd == "trim_enable":
                    trim_cfg["enabled"] = True
                    luxtts_cfg["silence_trim"] = trim_cfg
                    reply = "Silence trimming enabled."
                else:
                    trim_cfg["enabled"] = False
                    luxtts_cfg["silence_trim"] = trim_cfg
                    reply = "Silence trimming disabled."
            turn_id = str(uuid.uuid4())
            logger.write({
                "type": "turn_start",
                "session_id": session_id,
                "turn_id": turn_id,
                "user_text": user,
                "route": {"intent": "assist", "confidence": 1.0, "reason": "trim_command"},
            })
            final = _build_final(reply, reply, turn_id)
            tts_text = final["spoken_text"]
            selected_voice = current_voice
            clean_text = final["spoken_text"]
            tags = []
            tts_meta = None
            pre = preprocess_for_tts(final["spoken_text"], current_voice)
            clean_text = pre["clean_text"]
            selected_voice = pre["voice"]
            tags = pre["tags"]
            if tts_engine != "maya1":
                clean_text = _strip_maya_tags(clean_text)
            if tts_engine == "luxtts":
                tts_text, tts_meta = luxtts_tts.preprocess_text_for_tts(clean_text, luxtts_cfg)
            else:
                tts_text = clean_text
            logger.write({
                "type": "turn_end",
                "session_id": session_id,
                "turn_id": turn_id,
                "assistant_text": reply,
                "spoken_text": reply,
                "display_text": reply,
                "tts_engine": tts_engine,
            })
            out_wav = tts_dir / f"tts_{turn_id}.wav"
            logger.write({
                "type": "tts_input",
                "session_id": session_id,
                "turn_id": turn_id,
                "original_text": reply,
                "spoken_text": final["spoken_text"],
                "display_text": final["display_text"],
                "clean_text": clean_text,
                "tts_text": clean_text,
                "tts_processed_text": tts_text,
                "tags": tags,
                "selected_voice": selected_voice,
                "engine": tts_engine,
                "voice_key": selected_voice,
            })
            logger.write({
                "type": "tts_output",
                "session_id": session_id,
                "turn_id": turn_id,
                "wav_path": str(out_wav),
            })
            ui.print_assistant(final["display_text"])
            try:
                fallback = False
                fallback_reason = None
                engine_used = tts_engine
                if tts_engine == "luxtts":
                    try:
                        luxtts_tts.speak(
                            clean_text,
                            model_id_or_dir=str(luxtts_model_id),
                            reference_wav=luxtts_ref_wav,
                            out_wav=out_wav,
                            device=luxtts_device,
                            num_steps=luxtts_num_steps,
                            timeout_sec=luxtts_timeout,
                            tts_cfg=luxtts_cfg,
                            voice_key=selected_voice,
                            verbose=console_verbose,
                        )
                    except Exception as e:
                        fallback = True
                        engine_used = "piper"
                        fallback_reason = f"luxtts_error: {e}"
                        logger.write({
                            "type": "tts_error",
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "error": f"LuxTTS failed: {e}",
                        })
                        tts_piper(voice_model, clean_text, out_wav)
                elif tts_engine == "maya1":
                    try:
                        maya1_tts.speak(
                            clean_text,
                            model_dir=maya_model_dir,
                            out_wav=out_wav,
                            device=maya_device,
                            dtype=maya_dtype,
                        )
                    except Exception as e:
                        fallback = True
                        engine_used = "piper"
                        if "maya1_offload_slow" in str(e):
                            fallback_reason = "maya1_offload_slow"
                        logger.write({
                            "type": "tts_error",
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "error": f"Maya1 failed: {e}",
                        })
                        tts_piper(voice_model, clean_text, out_wav)
                else:
                    tts_piper(voice_model, clean_text, out_wav)

                engine_event = {
                    "type": "tts_engine",
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "engine": engine_used,
                    "fallback": fallback,
                }
                if fallback_reason:
                    engine_event["reason"] = fallback_reason
                logger.write(engine_event)

                play_wav_windows(out_wav)
            except Exception as e:
                logger.write({
                    "type": "tts_error",
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "error": str(e),
                })
                if console_verbose:
                    print(f"[TTS error] {e}\n")
            continue

        voice_cmd = _parse_voice_command(user)
        if voice_cmd:
            if voice_cmd == "auto":
                voice_override = None
                current_voice = voice_default
                reply = "Voice set to auto."
            elif voice_cmd in voice_keys:
                voice_override = voice_cmd
                current_voice = voice_cmd
                reply = f"Voice set to {voice_cmd}."
            else:
                reply = "Unknown voice. Available: " + ", ".join(sorted(voice_keys)) + ", auto."
            turn_id = str(uuid.uuid4())
            logger.write({
                "type": "turn_start",
                "session_id": session_id,
                "turn_id": turn_id,
                "user_text": user,
                "route": {"intent": "assist", "confidence": 1.0, "reason": "voice_command"},
            })
            final = _build_final(reply, reply, turn_id)
            tts_text = final["spoken_text"]
            selected_voice = current_voice
            clean_text = final["spoken_text"]
            tags = []
            tts_meta = None
            pre = preprocess_for_tts(final["spoken_text"], current_voice)
            clean_text = pre["clean_text"]
            selected_voice = pre["voice"]
            tags = pre["tags"]
            if tts_engine != "maya1":
                clean_text = _strip_maya_tags(clean_text)
            if tts_engine == "luxtts":
                tts_text, tts_meta = luxtts_tts.preprocess_text_for_tts(clean_text, luxtts_cfg)
            else:
                tts_text = clean_text
            logger.write({
                "type": "turn_end",
                "session_id": session_id,
                "turn_id": turn_id,
                "assistant_text": reply,
                "spoken_text": reply,
                "display_text": reply,
                "tts_engine": tts_engine,
            })
            out_wav = tts_dir / f"tts_{turn_id}.wav"
            logger.write({
                "type": "tts_input",
                "session_id": session_id,
                "turn_id": turn_id,
                "original_text": reply,
                "spoken_text": final["spoken_text"],
                "display_text": final["display_text"],
                "clean_text": clean_text,
                "tts_text": clean_text,
                "tts_processed_text": tts_text,
                "tags": tags,
                "selected_voice": selected_voice,
                "engine": tts_engine,
                "voice_key": selected_voice,
            })
            logger.write({
                "type": "tts_output",
                "session_id": session_id,
                "turn_id": turn_id,
                "wav_path": str(out_wav),
            })
            ui.print_assistant(final["display_text"])
            try:
                fallback = False
                fallback_reason = None
                engine_used = tts_engine
                if tts_engine == "luxtts":
                    try:
                        luxtts_tts.speak(
                            clean_text,
                            model_id_or_dir=str(luxtts_model_id),
                            reference_wav=luxtts_ref_wav,
                            out_wav=out_wav,
                            device=luxtts_device,
                            num_steps=luxtts_num_steps,
                            timeout_sec=luxtts_timeout,
                            tts_cfg=luxtts_cfg,
                            voice_key=selected_voice,
                            verbose=console_verbose,
                        )
                    except Exception as e:
                        fallback = True
                        engine_used = "piper"
                        fallback_reason = f"luxtts_error: {e}"
                        logger.write({
                            "type": "tts_error",
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "error": f"LuxTTS failed: {e}",
                        })
                        tts_piper(voice_model, clean_text, out_wav)
                elif tts_engine == "maya1":
                    try:
                        maya1_tts.speak(
                            clean_text,
                            model_dir=maya_model_dir,
                            out_wav=out_wav,
                            device=maya_device,
                            dtype=maya_dtype,
                        )
                    except Exception as e:
                        fallback = True
                        engine_used = "piper"
                        if "maya1_offload_slow" in str(e):
                            fallback_reason = "maya1_offload_slow"
                        logger.write({
                            "type": "tts_error",
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "error": f"Maya1 failed: {e}",
                        })
                        tts_piper(voice_model, clean_text, out_wav)
                else:
                    tts_piper(voice_model, clean_text, out_wav)

                engine_event = {
                    "type": "tts_engine",
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "engine": engine_used,
                    "fallback": fallback,
                }
                if fallback_reason:
                    engine_event["reason"] = fallback_reason
                logger.write(engine_event)

                play_wav_windows(out_wav)
            except Exception as e:
                logger.write({
                    "type": "tts_error",
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "error": str(e),
                })
                if console_verbose:
                    print(f"[TTS error] {e}\n")
            continue

        turn_id = str(uuid.uuid4())
        now_iso = datetime.now(ZoneInfo(tz)).isoformat()
        if mcp_backend.mode in {"auto", "mcp"} and not mcp_backend.active:
            prev_active = mcp_status.active
            prev_reason = mcp_status.reason
            mcp_status = mcp_backend.probe()
            if mcp_status.active != prev_active or mcp_status.reason != prev_reason:
                logger.write({
                    "type": "mcp_status_update",
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "active": mcp_status.active,
                    "mode": mcp_backend.mode,
                    "transport": mcp_backend.active_transport,
                    "reason": mcp_status.reason,
                    "tools": mcp_status.tools[:30],
                    "ts": logger.now_iso(),
                })
                if console_verbose:
                    ui.print_debug(
                        f"MCP status update: active={mcp_status.active}, reason={mcp_status.reason}"
                    )

        tool_catalog = _router_tool_catalog()
        memory_note = "Context is available from recent logs when needed."

        check_server_fingerprint(brain_base_url)

        router_memory = ""
        if not router_warmed():
            router_memory = render_recent(
                turns=router_cold_turns,
                max_chars=max_chars_memory_router,
                max_tool_content_chars=memory_tool_chars,
            )

        decision = router_decide(
            user,
            now_iso,
            tool_catalog,
            memory_note,
            cfg,
            memory_transcript=router_memory,
        )
        if decision.intent == "assist" and decision.tool_calls:
            first_tool = str((decision.tool_calls[0] or {}).get("name") or "")
            inferred_intent = "assist"
            if first_tool.startswith("filesystem."):
                inferred_intent = "filesystem"
            elif first_tool.startswith("weather."):
                inferred_intent = "weather"
            elif first_tool.startswith("websearch."):
                inferred_intent = "search"
            elif first_tool.startswith("executor."):
                inferred_intent = "executor"
            elif first_tool.startswith("codex."):
                inferred_intent = "codex"
            logger.write({
                "type": "router_adjustment",
                "session_id": session_id,
                "turn_id": turn_id,
                "reason": "assist_intent_with_tool_calls_promoted",
                "original_intent": "assist",
                "promoted_intent": inferred_intent,
                "original_tool_calls": decision.tool_calls,
                "ts": logger.now_iso(),
            })
            decision.intent = inferred_intent
        mark_router_warmed()
        logger.write({
            "type": "router_decision",
            "session_id": session_id,
            "turn_id": turn_id,
            "decision": {
                "intent": decision.intent,
                "confidence": decision.confidence,
                "need_context": decision.need_context,
                "tool_calls": decision.tool_calls,
                "user_rewrite": decision.user_rewrite,
                "context_policy": decision.context_policy,
                "router_context_turns": decision.router_context_turns,
                "brain_context_turns": decision.brain_context_turns,
                "debug_reason": getattr(decision, "debug_reason", None),
            },
            "ts": logger.now_iso(),
        })
        if getattr(decision, "debug_reason", None):
            logger.write({
                "type": "router_warning",
                "session_id": session_id,
                "turn_id": turn_id,
                "warning": str(decision.debug_reason),
                "ts": logger.now_iso(),
            })

        logger.write({
            "type": "turn_start",
            "session_id": session_id,
            "turn_id": turn_id,
            "user_text": user,
            "route": {"intent": decision.intent, "confidence": decision.confidence, "reason": "router"},
        })

        memory_block = ""
        memory_window = {"mode": "none"}
        if decision.need_context and decision.need_context.get("start") and decision.need_context.get("end"):
            memory_block = render_range(
                decision.need_context.get("start") or "",
                decision.need_context.get("end") or "",
                max_turns=memory_max_turns,
                max_chars=max_chars_memory_brain,
                max_tool_content_chars=memory_tool_chars,
            )
            memory_window = {
                "mode": "range",
                "start": decision.need_context.get("start"),
                "end": decision.need_context.get("end"),
                "reason": decision.need_context.get("reason"),
            }
        elif not brain_warmed():
            memory_block = render_recent(
                turns=brain_cold_turns,
                max_chars=max_chars_memory_brain,
                max_tool_content_chars=memory_tool_chars,
            )
            memory_window = {"mode": "cold_start", "turns": brain_cold_turns}
        elif (decision.context_policy or "none") == "default":
            turns = decision.brain_context_turns or memory_turns
            if turns > memory_max_turns:
                turns = memory_max_turns
            if turns < 1:
                turns = memory_turns
            memory_block = render_recent(
                turns=int(turns),
                max_chars=max_chars_memory_brain,
                max_tool_content_chars=memory_tool_chars,
            )
            memory_window = {"mode": "default", "turns": int(turns)}
        else:
            memory_block = ""
            memory_window = {"mode": "none"}

        logger.write({
            "type": "memory_window",
            "session_id": session_id,
            "turn_id": turn_id,
            "window": memory_window,
            "ts": logger.now_iso(),
        })

        def llm_generate(system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
            return chat_completion(
                base_url=brain_base_url,
                model=brain_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=brain_temp,
                max_tokens=int(max_tokens or brain_max_tokens),
            )

        def brain_chat(messages, max_tokens: int | None = None) -> str:
            return chat_completion(
                base_url=brain_base_url,
                model=brain_model,
                messages=messages,
                temperature=brain_temp,
                max_tokens=int(max_tokens or brain_max_tokens),
            )

        tool_names = [c.get("name") for c in (decision.tool_calls or []) if c.get("name")]
        if tool_names:
            ui.print_status("⚡ Routing…")
            ui.print_status("🧰 Tools: " + ", ".join(tool_names))

        tool_results = []
        with (ui.spinner("⏳ Running tools…") if tool_names else _nullcontext()):
            for call in (decision.tool_calls or []):
                name = (call.get("name") or "").strip()
                args = call.get("arguments") or {}

                if _looks_like_obsidian_request(name, args, user):
                    original_name, original_args = name, dict(args)
                    name, args = _rewrite_obsidian_tool_call(name, args, user)
                    logger.write({
                        "type": "tool_rewrite",
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "from_name": original_name,
                        "to_name": name,
                        "from_args": original_args,
                        "to_args": args,
                        "reason": "obsidian_request",
                        "ts": logger.now_iso(),
                    })

                # Normalize filesystem args early so both MCP/local branches receive concrete safe args.
                fs_arg_error = None
                if name.startswith("filesystem."):
                    op_name = name.split(".", 1)[1]
                    args, fs_arg_error = _normalize_fs_args(op_name, args, user)

                selected_backend, backend_reason = mcp_backend.select_backend(name)
                if name.startswith("obsidian."):
                    # Obsidian tools are MCP-only in this runtime.
                    selected_backend = "mcp"
                    if not mcp_backend.active:
                        backend_reason = f"mcp_inactive:{mcp_backend.reason or 'unknown'}"
                    elif not mcp_backend.can_handle(name):
                        backend_reason = "mcp_tool_unavailable"
                    else:
                        backend_reason = None
                elif name in {"filesystem.write", "filesystem.mkdir", "filesystem.delete"}:
                    # Keep mutating filesystem ops deterministic and policy-enforced in local code.
                    selected_backend = "local"
                    backend_reason = "local_mutation_policy"
                tool_backend_used = "local"
                fallback_reason = backend_reason
                mcp_transport = None
                out = None

                # MCP first when selected; deterministic local fallback in auto mode.
                if selected_backend == "mcp" and not name.startswith("executor") and name != "codex.run":
                    mcp_call = mcp_backend.call_tool(name, args)
                    mcp_transport = mcp_call.get("transport")
                    if mcp_call.get("ok"):
                        out = _normalize_mcp_tool_out(name, args, mcp_call.get("result") or {})
                        if (
                            isinstance(out, dict)
                            and not out.get("ok")
                            and mcp_backend.mode != "mcp"
                            and not name.startswith("obsidian.")
                        ):
                            fallback_reason = out.get("error") or "mcp_tool_error"
                            ui.print_status("↪ MCP tool failed, switching to local fallback…")
                            out = None
                            tool_backend_used = "local"
                        else:
                            tool_backend_used = "mcp"
                            fallback_reason = None
                    elif mcp_backend.mode == "mcp":
                        out = {
                            "ok": False,
                            "error": mcp_call.get("error") or "mcp_call_failed",
                            "result": None,
                        }
                        tool_backend_used = "mcp"
                    else:
                        err = mcp_call.get("error") or fallback_reason or "mcp_call_failed"
                        if name.startswith("obsidian."):
                            out = {"ok": False, "error": err, "result": None}
                            tool_backend_used = "mcp"
                            fallback_reason = None
                        else:
                            fallback_reason = err
                            ui.print_status("↪ MCP tool failed, switching to local fallback…")
                            tool_backend_used = "local"

                if out is None and name in {"weather.forecast", "weather.get_forecast"} and weather_enabled:
                    out = handle_weather(
                        cfg,
                        user,
                        logger,
                        session_id,
                        turn_id,
                        llm_generate,
                        tz,
                        tool_args=args,
                    )
                elif out is None and name == "websearch.search" and search_enabled:
                    out = _handle_search_from_plan(
                        cfg,
                        args,
                        user,
                        logger,
                        session_id,
                        turn_id,
                        llm_generate,
                        ui=ui,
                    )
                elif out is None and name.startswith("filesystem.") and filesystem_enabled:
                    if fs_arg_error:
                        out = {"ok": False, "error": fs_arg_error, "result": None}
                    else:
                        op = name.split(".", 1)[1]
                    try:
                        if out is None:
                            if op == "list":
                                res = fs_tool.list_dir(args.get("path") or str(DESKTOP_ROOT))
                                out = {"ok": True, "result": res, "error": None}
                            elif op == "read":
                                res = fs_tool.read_file(args.get("path") or "")
                                out = {"ok": True, "result": res, "error": None}
                            elif op == "tail":
                                res = fs_tool.tail_file(args.get("path") or "", int(args.get("lines") or 50))
                                out = {"ok": True, "result": res, "error": None}
                            elif op == "find":
                                matches = fs_tool.find_files(
                                    root=args.get("path") or str(DESKTOP_ROOT),
                                    pattern=args.get("pattern") or "*",
                                    limit=50,
                                )
                                res = {"path": args.get("path") or str(DESKTOP_ROOT), "pattern": args.get("pattern") or "*", "matches": [str(m) for m in matches]}
                                out = {"ok": True, "result": res, "error": None}
                            elif op == "grep":
                                res = fs_tool.grep_text(
                                    root=args.get("path") or str(DESKTOP_ROOT),
                                    query=args.get("query") or "",
                                    limit_hits=int(args.get("limit_hits") or 20),
                                    limit_files=int(args.get("limit_files") or 20),
                                )
                                out = {"ok": True, "result": res, "error": None}
                            elif op == "write":
                                res = fs_tool.write_file(args.get("path") or "", args.get("content") or "", args.get("mode") or "overwrite")
                                out = {"ok": True, "result": res, "error": None}
                            elif op == "mkdir":
                                res = fs_tool.make_dir(args.get("path") or "")
                                out = {"ok": True, "result": res, "error": None}
                            elif op == "delete":
                                res = fs_tool.delete_path(args.get("path") or "")
                                out = {"ok": True, "result": res, "error": None}
                            else:
                                out = {"ok": False, "error": f"unknown filesystem op: {op}", "result": None}
                    except PermissionError as e:
                        logger.write({
                            "type": "policy_denied",
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "action": "read" if op in {"list", "read", "tail", "find", "grep"} else "write",
                            "path": args.get("path") or "",
                            "ts": logger.now_iso(),
                        })
                        out = {"ok": False, "error": str(e), "result": None}
                    except Exception as e:
                        out = {"ok": False, "error": str(e), "result": None}
                elif out is None and name.startswith("filesystem") and filesystem_enabled:
                    out = handle_filesystem(
                        cfg,
                        user,
                        logger,
                        session_id,
                        turn_id,
                        llm_generate,
                        memory_block,
                        tool_args=args,
                    )
                elif out is None and name == "codex.run":
                    codex_cfg = (cfg.get("tools") or {}).get("codex") or {}
                    if not codex_cfg.get("enabled", True):
                        out = {"ok": False, "error": "codex disabled", "result": None}
                    else:
                        task = args.get("task") or user
                        output_filename = args.get("output_filename") or codex_cfg.get("output_filename", "codex_result.md")
                        codex_system = (
                            "You must output ONLY a Codex prompt. No explanations. No markdown fences. "
                            "Work only within Desktop. Save results to the provided output file as Markdown."
                        )
                        codex_user = (
                            f"Task: {task}\n"
                            f"Output filename: {output_filename}\n"
                            f"Constraints: Desktop-only, write output as Markdown.\n"
                        )
                        codex_prompt = brain_chat([
                            {"role": "system", "content": codex_system},
                            {"role": "user", "content": codex_user},
                        ]) or ""
                        ui.print_dim_block("Codex Prompt", codex_prompt, max_chars=1200)

                        before = _snapshot_desktop()
                        logger.write({
                            "type": "codex_exec_start",
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "cwd": str(DESKTOP_ROOT),
                            "ts": logger.now_iso(),
                        })
                        try:
                            with ui.spinner("⏳ Running Codex…"):
                                result = run_codex(
                                    codex_prompt=codex_prompt,
                                    output_filename=output_filename,
                                    exec_mode=str(codex_cfg.get("mode", "wsl")),
                                    codex_command=codex_cfg.get("command"),
                                    wsl_prelude=codex_cfg.get("wsl_prelude"),
                                    timeout_sec=int(codex_cfg.get("timeout_sec", 180)),
                                )
                        except PermissionError as e:
                            logger.write({
                                "type": "policy_denied",
                                "session_id": session_id,
                                "turn_id": turn_id,
                                "action": "exec",
                                "path": str(DESKTOP_ROOT),
                                "ts": logger.now_iso(),
                            })
                            result = {"ok": False, "error": str(e), "exit_code": 1}

                        after = _snapshot_desktop()
                        files_changed = _diff_desktop(before, after)
                        result["files_changed"] = files_changed

                        logger.write({
                            "type": "codex_prompt",
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "path": str(DESKTOP_ROOT / "_codex_last_prompt.txt"),
                            "chars": len(codex_prompt),
                            "ts": logger.now_iso(),
                        })
                        logger.write({
                            "type": "codex_exec_end",
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "exit_code": result.get("exit_code"),
                            "ts": logger.now_iso(),
                        })
                        out = {"ok": result.get("ok", False), "result": result, "error": result.get("error")}
                elif out is None and name.startswith("executor"):
                    out = {"ok": False, "error": "executor disabled", "result": None}
                elif out is None:
                    out = {"ok": False, "error": "unknown tool", "result": None}
                logger.write({
                    "type": "tool_backend",
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "tool": name,
                    "selected": selected_backend,
                    "used": tool_backend_used,
                    "fallback_reason": fallback_reason,
                    "transport": mcp_transport if tool_backend_used == "mcp" else None,
                    "ts": logger.now_iso(),
                })
                tool_results.append({
                    "tool": name,
                    "backend": tool_backend_used,
                    "fallback_reason": fallback_reason,
                    "transport": mcp_transport if tool_backend_used == "mcp" else None,
                    **_normalize_tool_out(out),
                })
                if isinstance(out, dict) and not out.get("ok") and out.get("error"):
                    ui.print_error(f"Tool error: {out.get('error')}")
                    logger.write({
                        "type": "tool_error",
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "tool": name,
                        "error": str(out.get("error")),
                        "ts": logger.now_iso(),
                    })

        tool_results_text = format_tool_results(tool_results)
        backend_set = {str(t.get("backend") or "local") for t in tool_results}
        local_fallback_used = any(
            (str(t.get("backend") or "local") == "local") and bool(t.get("fallback_reason"))
            for t in tool_results
        )
        if not backend_set:
            tool_backend_mode = "local"
        elif len(backend_set) == 1:
            tool_backend_mode = next(iter(backend_set))
        else:
            tool_backend_mode = "mixed"
        logger.write({
            "type": "brain_prompt_mode",
            "session_id": session_id,
            "turn_id": turn_id,
            "mcp_active": bool(mcp_backend.active),
            "tool_backend_mode": tool_backend_mode,
            "local_fallback_used": local_fallback_used,
            "ts": logger.now_iso(),
        })
        messages = build_brain_messages(
            system_style=style,
            user_text=user,
            memory_transcript_or_empty=memory_block,
            tool_results_or_empty=tool_results_text,
            tool_backend_mode=tool_backend_mode,
            mcp_active=bool(mcp_backend.active),
            local_fallback_used=local_fallback_used,
        )
        messages = enforce_budget(
            messages,
            max_chars_total=max_chars_total,
            max_chars_tool_results=max_chars_tool_results,
            max_chars_memory=max_chars_memory_brain,
        )

        if decision.user_rewrite and decision.user_rewrite.strip() and decision.user_rewrite.strip() != user.strip():
            # append effective request
            for m in messages:
                if m.get("role") == "user":
                    m["content"] += f"\nEffective request:\n{decision.user_rewrite}\n"
                    break

        reply_text = ""
        try:
            with ui.spinner("🧠 Thinking…"):
                reply_text = brain_chat(messages) or "Sorry — I didn't get that."
            mark_brain_warmed()
        except Exception as e:
            logger.write({
                "type": "llm_error",
                "session_id": session_id,
                "turn_id": turn_id,
                "stage": "brain_chat",
                "error": str(e),
                "ts": logger.now_iso(),
            })
            if retry_on_overflow and _is_overflow_error(e):
                logger.write({
                    "type": "llm_retry",
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "reason": "overflow",
                    "ts": logger.now_iso(),
                })
                retry_memory = render_recent(
                    turns=min_turns_retry,
                    max_chars=max_chars_memory_brain,
                    max_tool_content_chars=memory_tool_chars,
                )
                tool_results_text = tool_results_text[: max(500, max_chars_tool_results // 2)]
                messages = build_brain_messages(
                    system_style=style,
                    user_text=user,
                    memory_transcript_or_empty=retry_memory,
                    tool_results_or_empty=tool_results_text,
                    tool_backend_mode=tool_backend_mode,
                    mcp_active=bool(mcp_backend.active),
                    local_fallback_used=local_fallback_used,
                )
                messages = enforce_budget(
                    messages,
                    max_chars_total=max_chars_total,
                    max_chars_tool_results=max_chars_tool_results // 2,
                    max_chars_memory=max_chars_memory_brain,
                )
                try:
                    with ui.spinner("🧠 Thinking…"):
                        reply_text = brain_chat(messages) or "Sorry — I didn't get that."
                    mark_brain_warmed()
                except Exception as e2:
                    logger.write({
                        "type": "llm_error",
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "stage": "brain_chat_retry",
                        "error": str(e2),
                        "ts": logger.now_iso(),
                    })
                    msg = str(e2).lower()
                    if any(s in msg for s in ["connection", "refused", "failed to establish", "timed out", "model", "404", "503"]):
                        reply_text = (
                            "I couldn't reach the language model right now. "
                            "Please check LM Studio server and loaded model IDs."
                        )
                    else:
                        reply_text = "Sorry — I didn't get that."
            else:
                msg = str(e).lower()
                if any(s in msg for s in ["connection", "refused", "failed to establish", "timed out", "model", "404", "503"]):
                    reply_text = (
                        "I couldn't reach the language model right now. "
                        "Please check LM Studio server and loaded model IDs."
                    )
                else:
                    reply_text = "Sorry — I didn't get that."

        failed_tools = [t for t in tool_results if not bool(t.get("ok"))]
        if failed_tools and all(not bool(t.get("ok")) for t in tool_results):
            first_failed = failed_tools[0]
            err = str(first_failed.get("error") or "the tool call failed").strip()
            tname = str(first_failed.get("tool") or "")
            if tname.startswith("obsidian."):
                reply_text = (
                    f"I couldn't write to Obsidian because {err}. "
                    "Please make sure the Obsidian MCP tool is connected and try again."
                )
            else:
                reply_text = f"I couldn't complete that action because {err}."

        display_text = reply_text.strip()
        spoken_text = display_text

        urls = extract_urls(display_text)
        display_text = move_urls_to_parentheses(display_text, urls)
        display_text = enforce_max_links(display_text, max_links=int((cfg.get("ui") or {}).get("max_links", 5)))
        if re.fullmatch(r"\s*\([^)]*\)\s*", display_text):
            display_text = re.sub(r"^\s*\((.*)\)\s*$", r"\1", display_text).strip()
        if re.fullmatch(r"[\(\)\s]*", display_text or ""):
            display_text = spoken_text or display_text

        spoken_text = strip_parentheses_for_tts(display_text)
        spoken_text = ensure_no_urls_in_spoken(spoken_text)
        spoken_text, display_text = enforce_core_facts(decision.intent, spoken_text, display_text)
        spoken_text = strip_parentheses_for_tts(display_text)
        spoken_text = ensure_no_urls_in_spoken(spoken_text)
        spoken_text, display_text = enforce_length_caps(
            spoken_text,
            display_text,
            spoken_max=700,
            display_max=2000,
        )

        final = _build_final(spoken_text, display_text, turn_id)

        if voice_override:
            current_voice = voice_override
        else:
            current_voice = _pick_voice_mode(final["spoken_text"])

        tts_text = final["spoken_text"]
        selected_voice = current_voice
        clean_text = final["spoken_text"]
        tags = []
        tts_meta = None
        if (cfg.get("tts") or {}).get("ignore_parentheses", True):
            tts_text = strip_parentheses_for_tts(tts_text)
            clean_text = tts_text
        pre = preprocess_for_tts(tts_text, current_voice)
        clean_text = pre["clean_text"]
        selected_voice = pre["voice"]
        tags = pre["tags"]
        if tts_engine != "maya1":
            clean_text = _strip_maya_tags(clean_text)
        if tts_engine == "luxtts":
            tts_text, tts_meta = luxtts_tts.preprocess_text_for_tts(clean_text, luxtts_cfg)
        else:
            tts_text = clean_text

        out_wav = tts_dir / f"tts_{turn_id}.wav"
        tool_summary = tool_results
        logger.write({
            "type": "turn_end",
            "session_id": session_id,
            "turn_id": turn_id,
            "assistant_text": final["display_text"],
            "spoken_text": final["spoken_text"],
            "display_text": final["display_text"],
            "tool_result_summary": tool_summary,
            "tts_engine": tts_engine,
            "ts": logger.now_iso(),
        })
        _log_llm_provider(logger, session_id, turn_id, cfg)

        logger.write({
            "type": "tts_input",
            "session_id": session_id,
            "turn_id": turn_id,
            "original_text": final["display_text"],
            "spoken_text": final["spoken_text"],
            "display_text": final["display_text"],
            "clean_text": clean_text,
            "tts_text": clean_text,
            "tts_processed_text": tts_text,
            "tags": tags,
            "selected_voice": selected_voice,
            "engine": tts_engine,
            "tts_meta": tts_meta,
            "voice_key": selected_voice,
            "ts": logger.now_iso(),
        })

        logger.write({
            "type": "tts_output",
            "session_id": session_id,
            "turn_id": turn_id,
            "wav_path": str(out_wav),
            "ts": logger.now_iso(),
        })

        ui.print_assistant(final["display_text"])

        try:
            fallback = False
            fallback_reason = None
            engine_used = tts_engine
            if tts_engine == "luxtts":
                try:
                    luxtts_tts.speak(
                        clean_text,
                        model_id_or_dir=str(luxtts_model_id),
                        reference_wav=luxtts_ref_wav,
                        out_wav=out_wav,
                        device=luxtts_device,
                        num_steps=luxtts_num_steps,
                        timeout_sec=luxtts_timeout,
                        tts_cfg=luxtts_cfg,
                        voice_key=selected_voice,
                        verbose=console_verbose,
                    )
                except Exception as e:
                    fallback = True
                    engine_used = "piper"
                    fallback_reason = f"luxtts_error: {e}"
                    logger.write({
                        "type": "tts_error",
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "error": f"LuxTTS failed: {e}",
                    })
                    tts_piper(voice_model, clean_text, out_wav)
            elif tts_engine == "maya1":
                try:
                    maya1_tts.speak(
                        clean_text,
                        model_dir=maya_model_dir,
                        out_wav=out_wav,
                        device=maya_device,
                        dtype=maya_dtype,
                    )
                except Exception as e:
                    fallback = True
                    engine_used = "piper"
                    if "maya1_offload_slow" in str(e):
                        fallback_reason = "maya1_offload_slow"
                    logger.write({
                        "type": "tts_error",
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "error": f"Maya1 failed: {e}",
                    })
                    tts_piper(voice_model, clean_text, out_wav)
            else:
                tts_piper(voice_model, clean_text, out_wav)

            engine_event = {
                "type": "tts_engine",
                "session_id": session_id,
                "turn_id": turn_id,
                "engine": engine_used,
                "fallback": fallback,
            }
            if fallback_reason:
                engine_event["reason"] = fallback_reason
            logger.write(engine_event)

            play_wav_windows(out_wav)
        except Exception as e:
            logger.write({
                "type": "tts_error",
                "session_id": session_id,
                "turn_id": turn_id,
                "error": str(e),
            })
            if console_verbose:
                print(f"[TTS error] {e}\n")

    try:
        mcp_backend.close()
    except Exception:
        pass
    logger.write({"type": "session_end", "session_id": session_id})


if __name__ == "__main__":
    main()
