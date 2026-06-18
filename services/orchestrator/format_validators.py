from __future__ import annotations

import re
from typing import List, Tuple


_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_WIN_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s)]+")


def extract_urls(text: str) -> List[str]:
    if not text:
        return []
    # ignore urls already inside parentheses
    scrubbed = re.sub(r"\([^)]*\)", " ", text)
    return _URL_RE.findall(scrubbed)


def ensure_no_urls_in_spoken(spoken_text: str) -> str:
    if not spoken_text:
        return ""
    out = _URL_RE.sub("", spoken_text)
    out = _WIN_PATH_RE.sub("", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def move_urls_to_parentheses(display_text: str, urls: List[str]) -> str:
    if not display_text:
        display_text = ""
    cleaned = _URL_RE.sub("", display_text)
    cleaned = _WIN_PATH_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if urls:
        unique = []
        for u in urls:
            if u not in unique:
                unique.append(u)
        cleaned = cleaned.rstrip(".")
        cleaned = f"{cleaned} (Links: " + "; ".join(unique) + ")"
    return cleaned


def enforce_max_links(display_text: str, max_links: int = 5) -> str:
    if max_links <= 0:
        return re.sub(r"\(Links:[^)]+\)", "", display_text).strip()
    m = re.search(r"\(Links:\s*([^)]+)\)", display_text)
    if not m:
        return display_text
    parts = [p.strip() for p in m.group(1).split(";") if p.strip()]
    parts = parts[:max_links]
    repl = "(Links: " + "; ".join(parts) + ")"
    return display_text[: m.start()] + repl + display_text[m.end():]


def enforce_length_caps(spoken_text: str, display_text: str, spoken_max: int = 700, display_max: int = 2000) -> Tuple[str, str]:
    s = spoken_text or ""
    d = display_text or ""
    if len(s) > spoken_max:
        s = s[:spoken_max].rstrip() + "."
    if len(d) > display_max:
        d = d[:display_max].rstrip() + "..."
    return s, d


def enforce_core_facts(intent: str, spoken_text: str, display_text: str) -> Tuple[str, str]:
    if intent not in {"weather", "calendar"}:
        return spoken_text, display_text
    has_digit_spoken = bool(re.search(r"\d", spoken_text or ""))
    has_digit_display = bool(re.search(r"\d", display_text or ""))
    if not has_digit_spoken and has_digit_display:
        spoken_text = strip_parentheses(display_text)
    # If core facts are inside parentheses, inline them into display for weather/calendar
    display_text = flatten_parenthetical_facts(display_text)
    return spoken_text, display_text


def strip_parentheses(text: str) -> str:
    if not text:
        return ""
    out = re.sub(r"\([^)]*\)", " ", text)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def flatten_parenthetical_facts(text: str) -> str:
    if not text:
        return ""
    # If parentheses contain digits (likely facts), inline them.
    def repl(match: re.Match) -> str:
        inner = match.group(1)
        if re.search(r"\d", inner):
            return " " + inner.strip()
        return " "  # drop non-fact parenthetical
    out = re.sub(r"\(([^)]*)\)", repl, text)
    out = re.sub(r"\s+", " ", out).strip()
    return out
