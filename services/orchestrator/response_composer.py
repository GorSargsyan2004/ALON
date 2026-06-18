from __future__ import annotations

import re
from typing import List, Dict


def strip_parentheses_for_tts(text: str) -> str:
    if not text:
        return ""
    out = text
    # remove parenthetical blocks
    out = re.sub(r"\([^)]*\)", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _sentiment_opener(user_text: str) -> str:
    t = (user_text or "").lower()
    if re.search(r"\b(fuck|shit|damn|ugh|wtf|hate)\b", t):
        return "Yeah, that kind of weather can be frustrating."
    if any(w in t for w in ["cloudy", "gloomy", "not nice", "bad weather", "dreary"]):
        return "Yeah, cloudy days can feel rough."
    if any(w in t for w in ["rain", "raining", "snow", "snowing"]):
        return "Let me check that for you."
    return "Sure — here's what I found."


def _join_sentences(parts: List[str], max_sentences: int = 3) -> str:
    cleaned = [p.strip() for p in parts if p and p.strip()]
    if not cleaned:
        return ""
    out = []
    for p in cleaned:
        if not p.endswith((".", "!", "?")):
            p += "."
        out.append(p)
        if len(out) >= max_sentences:
            break
    return " ".join(out).strip()


def compose_weather_reply(user_text: str, tool_data: Dict, assistant_name: str, style: str, tz: str) -> Dict:
    opener = _sentiment_opener(user_text)
    summary = (tool_data.get("summary") or "").strip()
    wear = (tool_data.get("wear_advice") or "").strip()

    extra = ""
    if re.search(r"\bwinter\b", (user_text or "").lower()):
        extra = "Yeah, winter doesn't always back off when we want."

    parts = [opener]
    if summary:
        parts.append(summary)
    if wear:
        parts.append(wear)
    if extra:
        parts.append(extra)

    spoken = _join_sentences(parts, max_sentences=3)
    display = spoken
    return {"spoken_text": spoken, "display_text": display}


def compose_search_reply(
    user_text: str,
    search_summary: str,
    links: List[Dict],
    assistant_name: str,
    style: str,
    max_links: int = 5,
) -> Dict:
    summary = (search_summary or "").strip()
    summary = re.sub(r"https?://\S+", "", summary)
    sources_text = ""
    m = re.search(r"\bSources?:\s*(.+)$", summary, flags=re.I)
    if m:
        sources_text = m.group(1).strip()
    summary = re.sub(r"\s*Sources?:.*$", "", summary, flags=re.I).strip()

    parts = [summary] if summary else []

    t = (user_text or "").lower()
    if any(x in t for x in ["my sister", "my brother", "my friend", "facebook", "instagram", "profile"]):
        parts.append("These are possible matches, so please verify the right person.")

    spoken = _join_sentences(parts, max_sentences=3)

    display = spoken
    items = []
    for r in (links or [])[:max_links]:
        title = (r.get("title") or "Result").strip()
        url = (r.get("url") or "").strip()
        if url:
            items.append(f"{title} — {url}")
    extras = []
    if items:
        extras.append("Links: " + "; ".join(items))
    if sources_text:
        extras.append("Sources: " + sources_text)
    if extras:
        display = f"{spoken}\n(" + " | ".join(extras) + ")"
    return {"spoken_text": spoken, "display_text": display}
