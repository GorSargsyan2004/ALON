from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Optional


_HOLIDAYS = {
    (2, 14): "Valentine's Day",
    (1, 1): "New Year's Day",
    (12, 25): "Christmas",
}

_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _parse_relative_days(text: str) -> Optional[int]:
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


def _parse_explicit_date(text: str, today: date) -> Optional[date]:
    t = (text or "").strip()
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            return None
    m = re.search(r"\b([A-Za-z]{3,9})\s+(\d{1,2})(?:,\s*(\d{4}))?\b", t)
    if m:
        month = _MONTHS.get(m.group(1).lower())
        if month:
            year = int(m.group(3)) if m.group(3) else today.year
            try:
                d = date(year, month, int(m.group(2)))
                if d < today and not m.group(3):
                    d = date(year + 1, month, int(m.group(2)))
                return d
            except Exception:
                return None
    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]{3,9})(?:\s+(\d{4}))?\b", t)
    if m:
        month = _MONTHS.get(m.group(2).lower())
        if month:
            year = int(m.group(3)) if m.group(3) else today.year
            try:
                d = date(year, month, int(m.group(1)))
                if d < today and not m.group(3):
                    d = date(year + 1, month, int(m.group(1)))
                return d
            except Exception:
                return None
    return None


def resolve_calendar_query(user_text: str, today: date) -> dict:
    rel = _parse_relative_days(user_text)
    if rel is not None:
        target = today + timedelta(days=rel)
    else:
        target = _parse_explicit_date(user_text, today) or today

    holiday = _HOLIDAYS.get((target.month, target.day))
    return {
        "date": target.isoformat(),
        "weekday": target.strftime("%A"),
        "holiday_hint": holiday,
    }
