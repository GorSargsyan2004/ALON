from __future__ import annotations

import requests


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    extra_payload: dict | None = None,
    timeout_sec: int = 180,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    if extra_payload:
        payload.update(extra_payload)
    r = requests.post(url, json=payload, timeout=timeout_sec)
    r.raise_for_status()
    data = r.json()
    return (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()


def list_models(base_url: str, timeout_sec: int = 10) -> list[dict]:
    url = f"{base_url.rstrip('/')}/models"
    r = requests.get(url, timeout=timeout_sec)
    r.raise_for_status()
    return r.json().get("data") or []
