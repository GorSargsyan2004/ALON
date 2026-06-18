from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DayForecast:
    date: str
    t_max: float
    t_min: float
    precip_prob_max: Optional[int]
    wind_max: Optional[float]
    weather_code: Optional[int]


def _safe_get(arr, i, default=None):
    try:
        return arr[i]
    except Exception:
        return default


def parse_day(data: Dict[str, Any], day_index: int = 0) -> DayForecast:
    daily = data.get("daily") or {}
    return DayForecast(
        date=_safe_get(daily.get("time", []), day_index, "unknown"),
        t_max=float(_safe_get(daily.get("temperature_2m_max", []), day_index, 0)),
        t_min=float(_safe_get(daily.get("temperature_2m_min", []), day_index, 0)),
        precip_prob_max=_safe_get(daily.get("precipitation_probability_max", []), day_index, None),
        wind_max=_safe_get(daily.get("wind_speed_10m_max", []), day_index, None),
        weather_code=_safe_get(daily.get("weather_code", []), day_index, None),
    )


def select_day(data: Dict[str, Any], day: str) -> tuple[DayForecast, str]:
    day = (day or "").strip().lower()
    if day == "today":
        if _safe_get((data.get("daily") or {}).get("time", []), 0, None) is None:
            raise ValueError("No forecast available for today")
        return parse_day(data, day_index=0), "Today"
    if day == "tomorrow":
        if _safe_get((data.get("daily") or {}).get("time", []), 1, None) is None:
            raise ValueError("No forecast available for tomorrow")
        return parse_day(data, day_index=1), "Tomorrow"

    daily = data.get("daily") or {}
    days = daily.get("time") or []
    try:
        idx = days.index(day)
    except ValueError:
        raise ValueError(f"Date not available in forecast: {day}")

    return parse_day(data, day_index=idx), day


def what_to_wear(day: DayForecast) -> str:
    # Simple, practical rules tuned for spoken output
    tips = []

    avg = (day.t_max + day.t_min) / 2.0
    if day.t_max <= 5:
        tips.append("Wear a warm coat, and consider gloves.")
    elif day.t_max <= 12:
        tips.append("A jacket or coat is a good idea.")
    elif day.t_max <= 20:
        tips.append("Light jacket or hoodie should be enough.")
    else:
        tips.append("T-shirt weather; maybe bring a light layer for evening.")

    if day.precip_prob_max is not None and day.precip_prob_max >= 40:
        tips.append("Bring an umbrella or a rain jacket.")

    if day.wind_max is not None and float(day.wind_max) >= 30:
        tips.append("It may be windy—wear something windproof.")

    return " ".join(tips)


def speakable_summary(location_name: str, day: DayForecast, label: str = "Tomorrow") -> str:
    prob = ""
    if day.precip_prob_max is not None:
        prob = f" Precipitation chance up to {int(day.precip_prob_max)} percent."
    wind = ""
    if day.wind_max is not None:
        wind = f" Wind up to {int(float(day.wind_max))} km/h."
    return (
        f"{label} in {location_name}: low {int(day.t_min)}°C, high {int(day.t_max)}°C."
        f"{prob}{wind}"
    )
