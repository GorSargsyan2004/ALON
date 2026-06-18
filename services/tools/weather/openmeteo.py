from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import requests


GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


@dataclass
class Location:
    name: str
    latitude: float
    longitude: float
    timezone: str


def geocode(name: str, language: str = "en") -> Location:
    params = {"name": name, "count": 1, "language": language, "format": "json"}
    r = requests.get(GEOCODE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    results = data.get("results") or []
    if not results:
        raise ValueError(f"Could not find location: {name}")
    top = results[0]
    return Location(
        name=top.get("name") or name,
        latitude=float(top["latitude"]),
        longitude=float(top["longitude"]),
        timezone=top.get("timezone") or "auto",
    )


def forecast_daily(loc: Location, days: int = 2) -> Dict[str, Any]:
    # Daily summary + a current snapshot
    params = {
        "latitude": loc.latitude,
        "longitude": loc.longitude,
        "timezone": loc.timezone,  # or "auto"
        "forecast_days": days,
        "current": "temperature_2m,precipitation,rain,showers,snowfall,wind_speed_10m",
        "daily": ",".join([
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
            "wind_speed_10m_max",
        ]),
    }
    r = requests.get(FORECAST_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()
