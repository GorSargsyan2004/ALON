from __future__ import annotations

import requests


def llm_chat(
    cfg: dict,
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    provider = ((cfg.get("llm") or {}).get("provider") or "ollama").lower()
    if provider == "lmstudio":
        return _lmstudio_chat(cfg, system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens)
    return _ollama_generate(cfg, system_prompt, user_prompt)


def _ollama_generate(cfg: dict, system_prompt: str, user_prompt: str) -> str:
    base_url = (cfg.get("ollama") or {}).get("url", "http://localhost:11434")
    model = (cfg.get("ollama") or {}).get("model", "llama3.1:8b")
    url = f"{base_url.rstrip('/')}/api/generate"
    prompt = f"{system_prompt}\n\nUser: {user_prompt}"
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


def _lmstudio_chat(
    cfg: dict,
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    lm = cfg.get("lmstudio") or {}
    base_url = lm.get("base_url", "http://localhost:1234/v1").rstrip("/")
    model = lm.get("model", "")
    temperature = float(temperature if temperature is not None else lm.get("temperature", 0.6))
    max_tokens = int(max_tokens if max_tokens is not None else lm.get("max_tokens", 350))
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
