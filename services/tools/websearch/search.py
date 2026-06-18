from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import requests

# Optional dependency: pip install duckduckgo-search
try:
    from ddgs import DDGS
except Exception:
    DDGS = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

MAP_SYSTEM_PROMPT = (
    "You are a careful web summarizer. Return ONLY valid JSON with keys: "
    "chunk_takeaways, key_facts, notable_quotes, topics. "
    "notable_quotes must be <=25 words each. "
    "Do not include URLs unless inside fields. Extract key facts and quotes faithfully."
)

REDUCE_SYSTEM_PROMPT = (
    "Combine chunk JSONs into a single source summary JSON with keys: "
    "source_title, url, summary_bullets, key_numbers, what_it_means, confidence. "
    "Be concise. Preserve important numbers/dates."
)

FINAL_SYSTEM_PROMPT = (
    "You are ALON. Answer the user using the provided source summaries only. "
    "Do NOT add facts that are not present in the summaries. "
    "Return ONLY valid JSON with keys: answer_spoken, answer_display, sources. "
    "Keep spoken answer clean (no URLs). Put URLs only in display answer parentheses. "
    "If include_sources is false, do not add sources in the display. "
    "If answer_style is 'short', use 1 sentence. If 'brief', use 2-3 sentences max."
)

SEARCH_PIPELINE_VERSION = 3


def search_web(
    query: str,
    max_results: int = 5,
    fetch_top_n_pages: int = 2,
    user_agent: str = "Alon/0.1 (+local assistant)",
    timeout_sec: int = 20,
    cache_ttl_seconds: int = 21600,
    cache_dir: Optional[Path] = None,
    llm_chat_func=None,
    turn_id: Optional[str] = None,
    include_sources: bool = False,
    settings: Optional[Dict[str, Any]] = None,
    user_text: Optional[str] = None,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    query = (query or "").strip()
    if not query:
        raise ValueError("Empty query")

    settings = settings or {}
    max_chars_per_page_clean = int(settings.get("max_chars_per_page_clean", 12000))
    max_chars_per_chunk = int(settings.get("max_chars_per_chunk", 2500))
    max_chunks_per_page = int(settings.get("max_chunks_per_page", 6))
    llm_chunk_max_tokens = int(settings.get("llm_chunk_max_tokens", 250))
    llm_reduce_max_tokens = int(settings.get("llm_reduce_max_tokens", 300))
    llm_final_max_tokens = int(settings.get("llm_final_max_tokens", 350))
    request_timeout_sec = int(settings.get("request_timeout_sec", timeout_sec))
    max_source_checks = int(settings.get("max_source_checks", max_results))
    min_match_score = float(settings.get("min_match_score", 0.35))
    provider = str(settings.get("provider") or "local").strip().lower()
    gemini_cfg = (settings.get("gemini") or {}) if isinstance(settings.get("gemini"), dict) else {}
    gemini_enabled = bool(gemini_cfg.get("enabled", False) or provider == "gemini")
    gemini_base_url = str(gemini_cfg.get("base_url") or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    gemini_model = str(gemini_cfg.get("model") or "gemini-2.0-flash")
    gemini_timeout_sec = int(gemini_cfg.get("timeout_sec", request_timeout_sec))
    gemini_max_output_tokens = int(gemini_cfg.get("max_output_tokens", llm_final_max_tokens))
    gemini_api_key_env = str(gemini_cfg.get("api_key_env") or "GEMINI_API_KEY")
    gemini_api_key = str(gemini_cfg.get("api_key") or os.getenv(gemini_api_key_env, "")).strip()
    gemini_api_key_file = str(gemini_cfg.get("api_key_file") or ".env").strip()
    if not gemini_api_key and gemini_api_key_file:
        key_path = Path(gemini_api_key_file)
        if not key_path.is_absolute():
            key_path = Path(__file__).resolve().parents[3] / key_path
        gemini_api_key = _read_key_from_env_file(key_path, gemini_api_key_env)
    gemini_min_match_score = float(gemini_cfg.get("min_answer_match_score", min_match_score))

    metrics = {
        "sources_fetched": 0,
        "sources_checked": 0,
        "sources_matched": 0,
        "answer_found": False,
        "chars_per_page": [],
        "chunks_per_page": [],
        "total_chunks": 0,
        "llm_calls": 0,
        "truncations": 0,
        "timing_ms": {"search": 0, "fetch": 0, "map": 0, "reduce": 0, "final": 0},
    }

    cache_info = {"hit": False, "age_sec": 0, "path": None, "ttl_sec": cache_ttl_seconds}
    cache_key = _hash_query(f"{provider}:{gemini_model}:{query}")
    cache_path = cache_dir / f"{cache_key}.json" if cache_dir else None

    if cache_dir and cache_path:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = _load_cache(cache_path, cache_ttl_seconds)
        if cached and int(cached.get("pipeline_version") or 0) >= SEARCH_PIPELINE_VERSION:
            cache_info["hit"] = True
            cache_info["age_sec"] = cached.get("cache", {}).get("age_sec")
            cache_info["path"] = str(cache_path)
            cached["cache"] = cache_info
            cached["cache_hit"] = True
            _emit_progress(progress_cb, {
                "stage": "cache_hit",
                "query": query,
                "path": str(cache_path),
                "age_sec": cache_info["age_sec"],
            })
            return cached

    if gemini_enabled:
        _emit_progress(progress_cb, {
            "stage": "gemini_start",
            "query": query,
            "model": gemini_model,
        })
        if not gemini_api_key:
            _emit_progress(progress_cb, {
                "stage": "gemini_fallback",
                "reason": f"missing_api_key:{gemini_api_key_env}",
            })
        else:
            gem_t0 = time.perf_counter()
            try:
                gemini_result = _search_with_gemini_grounding(
                    query=query,
                    user_text=user_text or query,
                    include_sources=include_sources,
                    api_key=gemini_api_key,
                    base_url=gemini_base_url,
                    model=gemini_model,
                    timeout_sec=gemini_timeout_sec,
                    max_output_tokens=gemini_max_output_tokens,
                )
                metrics["timing_ms"]["gemini"] = int((time.perf_counter() - gem_t0) * 1000)
                match_score = _lexical_match_score(query, gemini_result.get("answer_spoken") or "")
                has_sources = bool(gemini_result.get("sources"))
                if match_score >= gemini_min_match_score or has_sources:
                    metrics["answer_found"] = True
                    metrics["sources_checked"] = len(gemini_result.get("sources") or [])
                    metrics["sources_matched"] = len(gemini_result.get("sources") or [])
                    payload = {
                        "query": query,
                        "results": gemini_result.get("results") or [],
                        "sources": gemini_result.get("sources") or [],
                        "engine": "gemini_google_search",
                        "provider": "gemini",
                        "pipeline_version": SEARCH_PIPELINE_VERSION,
                        "cache": cache_info,
                        "cache_hit": False,
                        "ts": int(time.time()),
                        "metrics": metrics,
                        "final_answer": {
                            "answer_spoken": gemini_result.get("answer_spoken") or "",
                            "answer_display": gemini_result.get("answer_display") or "",
                            "sources": gemini_result.get("sources") or [],
                        },
                    }
                    if cache_dir and cache_path:
                        _save_cache(cache_path, payload)
                        payload["cache"]["path"] = str(cache_path)
                    _emit_progress(progress_cb, {
                        "stage": "gemini_success",
                        "sources": len(gemini_result.get("sources") or []),
                        "score": round(match_score, 3),
                    })
                    return payload
                _emit_progress(progress_cb, {
                    "stage": "gemini_fallback",
                    "reason": "weak_answer",
                    "score": round(match_score, 3),
                })
            except Exception as e:
                metrics["timing_ms"]["gemini"] = int((time.perf_counter() - gem_t0) * 1000)
                _emit_progress(progress_cb, {
                    "stage": "gemini_fallback",
                    "reason": str(e),
                })

    t0 = time.perf_counter()
    results, engine = _run_search(query, max_results, user_agent, request_timeout_sec)
    metrics["timing_ms"]["search"] = int((time.perf_counter() - t0) * 1000)
    _emit_progress(progress_cb, {
        "stage": "search_results",
        "query": query,
        "engine": engine,
        "count": len(results),
    })

    if not results:
        payload = {
            "query": query,
            "results": [],
            "sources": [],
            "engine": engine,
            "provider": "local",
            "pipeline_version": SEARCH_PIPELINE_VERSION,
            "cache": cache_info,
            "cache_hit": False,
            "ts": int(time.time()),
            "metrics": metrics,
            "final_answer": _fallback_final_answer(query, [], include_sources),
        }
        return payload

    decision = {
        "depth": "deep",
        "answer_style": "brief",
        "cite": bool(include_sources),
    }
    if llm_chat_func:
        decision = _decide_strategy(
            llm_chat_func,
            query=query,
            user_text=user_text or "",
            results=results,
            explicit_citations=bool(include_sources),
            max_tokens=120,
        )

    sources: List[Dict[str, Any]] = []
    checked_sources: List[Dict[str, Any]] = []
    target_sources = 1 if decision.get("depth") == "quick" else max(1, min(fetch_top_n_pages, 2))
    candidate_results = results[: max(1, min(max_source_checks, len(results)))]

    t1 = time.perf_counter()
    for idx, item in enumerate(candidate_results, start=1):
        event = {
            "stage": "source_check",
            "index": idx,
            "total": len(candidate_results),
            "url": item.get("url"),
            "title": item.get("title"),
        }
        _emit_progress(progress_cb, event)

        src = _fetch_single_source(
            item,
            user_agent=user_agent,
            timeout_sec=request_timeout_sec,
            max_chars=max_chars_per_page_clean,
        )
        checked_sources.append(src)
        metrics["sources_checked"] += 1
        if src.get("text"):
            metrics["sources_fetched"] += 1

        text = src.get("text") or src.get("snippet") or ""
        text = _normalize_text(text, max_chars_per_page_clean)
        if len(text) > max_chars_per_page_clean:
            metrics["truncations"] += 1
            text = text[:max_chars_per_page_clean]
        chunks = _chunk_text(text, max_chars_per_chunk, max_chunks_per_page)
        metrics["chars_per_page"].append(len(text))
        metrics["chunks_per_page"].append(len(chunks))
        metrics["total_chunks"] += len(chunks)
        src["chunks"] = []
        src["chunk_summaries"] = []

        for j, chunk in enumerate(chunks):
            header = f"[Source {idx} | Chunk {j+1}/{len(chunks)} | {src.get('title','')} | {src.get('url','')}]"
            chunk_text = f"{header}\n{chunk}"
            summary = None
            if llm_chat_func and decision.get("depth") != "quick":
                summary = _map_chunk(
                    llm_chat_func,
                    chunk_text,
                    max_tokens=llm_chunk_max_tokens,
                )
                metrics["llm_calls"] += 1 if summary else 0
                if summary is None and len(chunk) > 800:
                    smaller = chunk[:800]
                    summary = _map_chunk(
                        llm_chat_func,
                        f"{header}\n{smaller}",
                        max_tokens=llm_chunk_max_tokens,
                    )
                    metrics["llm_calls"] += 1 if summary else 0
            if summary is None:
                summary = {
                    "chunk_takeaways": [_shorten(src.get("snippet") or "", 160)],
                    "key_facts": [],
                    "notable_quotes": [],
                    "topics": [],
                }
            src["chunk_summaries"].append(summary)
            src["chunks"].append({"len": len(chunk)})

        reduced = None
        if llm_chat_func and src.get("chunk_summaries") and decision.get("depth") != "quick":
            reduced = _reduce_source(llm_chat_func, src, max_tokens=llm_reduce_max_tokens)
            metrics["llm_calls"] += 1 if reduced else 0
        if reduced is None:
            reduced = _fallback_reduce(src)
        src["source_summary"] = reduced

        matched, match_reason, match_score = _source_matches_query(
            query=query,
            source=src,
            llm_chat_func=llm_chat_func,
            min_score=min_match_score,
        )
        src["match"] = {"ok": matched, "score": match_score, "reason": match_reason}
        _emit_progress(progress_cb, {
            "stage": "source_match",
            "index": idx,
            "total": len(candidate_results),
            "url": src.get("url"),
            "matched": bool(matched),
            "score": match_score,
            "reason": match_reason,
        })

        if matched:
            metrics["sources_matched"] += 1
            metrics["answer_found"] = True
            sources.append(src)
            if len(sources) >= target_sources:
                break
        elif not sources:
            # keep first non-match as fallback if we never find a match
            if not any(s.get("fallback_candidate") for s in checked_sources):
                src["fallback_candidate"] = True

    fetch_map_reduce_ms = int((time.perf_counter() - t1) * 1000)
    metrics["timing_ms"]["fetch_map_reduce"] = fetch_map_reduce_ms
    # keep legacy keys for existing dashboards
    metrics["timing_ms"]["fetch"] = fetch_map_reduce_ms

    if not sources:
        fallback_src = next((s for s in checked_sources if s.get("fallback_candidate")), None)
        if fallback_src:
            sources = [fallback_src]
        else:
            sources = checked_sources[:1]
        _emit_progress(progress_cb, {"stage": "answer_not_found", "query": query})
    else:
        _emit_progress(progress_cb, {
            "stage": "answer_found",
            "query": query,
            "source_count": len(sources),
            "top_url": (sources[0].get("url") if sources else None),
        })

    # Final answer composer
    t4 = time.perf_counter()
    final_answer = None
    if llm_chat_func:
        final_answer = _final_answer(
            llm_chat_func,
            query=query,
            sources=sources,
            include_sources=bool(decision.get("cite", include_sources)),
            max_tokens=llm_final_max_tokens,
            answer_style=decision.get("answer_style", "brief"),
        )
        metrics["llm_calls"] += 1 if final_answer else 0
    if final_answer is None:
        final_answer = _fallback_final_answer(query, sources, include_sources)
    metrics["timing_ms"]["final"] = int((time.perf_counter() - t4) * 1000)

    payload = {
        "query": query,
        "results": results,
        "sources": _compact_sources_for_result(sources),
        "engine": engine,
        "provider": "local",
        "pipeline_version": SEARCH_PIPELINE_VERSION,
        "cache": cache_info,
        "cache_hit": False,
        "ts": int(time.time()),
        "metrics": metrics,
        "final_answer": final_answer,
    }

    if cache_dir and cache_path and results:
        _save_cache(cache_path, payload)
        payload["cache"]["path"] = str(cache_path)

    # Debug dump per turn
    if cache_dir and turn_id:
        debug_path = cache_dir / f"{turn_id}_debug.json"
        _save_debug(debug_path, payload, sources)

    return payload


def _search_with_gemini_grounding(
    query: str,
    user_text: str,
    include_sources: bool,
    api_key: str,
    base_url: str,
    model: str,
    timeout_sec: int,
    max_output_tokens: int,
) -> Dict[str, Any]:
    prompt = (
        "Answer the user question using Google Search grounding.\n"
        "Rules:\n"
        "- Give a direct, factual answer.\n"
        "- If uncertain, say uncertainty clearly.\n"
        "- Do not include raw URLs in the answer text.\n"
        f"Question: {user_text or query}\n"
    )
    endpoint = f"{base_url}/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": int(max_output_tokens),
        },
    }
    resp = requests.post(endpoint, json=payload, timeout=timeout_sec)
    if resp.status_code >= 400:
        body = _shorten(resp.text or "", 280)
        raise RuntimeError(f"gemini_http_{resp.status_code}: {body}")
    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError("gemini_no_candidates")
    first = candidates[0] or {}
    content = (first.get("content") or {}).get("parts") or []
    answer = _normalize_text(" ".join([(p.get("text") or "") for p in content]), 1800)
    if not answer:
        raise RuntimeError("gemini_empty_answer")

    grounding = first.get("groundingMetadata") or {}
    raw_chunks = grounding.get("groundingChunks") or []
    sources = []
    seen = set()
    for chunk in raw_chunks:
        web = chunk.get("web") or {}
        url = (web.get("uri") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        title = (web.get("title") or "").strip() or _source_name(url)
        sources.append({
            "title": title,
            "url": url,
            "publish_date": None,
            "source_name": _source_name(url),
            "source_summary": None,
        })
        if len(sources) >= 8:
            break

    spoken = _strip_urls(answer).strip()
    display = spoken
    if include_sources and sources:
        display = _append_sources(display, sources[:5])

    results = []
    for s in sources[:5]:
        results.append({
            "title": s.get("title"),
            "snippet": "",
            "url": s.get("url"),
            "source_name": s.get("source_name"),
            "date": None,
        })

    return {
        "answer_spoken": spoken,
        "answer_display": display or spoken,
        "sources": sources,
        "results": results,
    }


def _run_search(query: str, max_results: int, user_agent: str, timeout_sec: int):
    results: List[Dict[str, Any]] = []
    engine = "ddg_html"

    if DDGS is not None:
        try:
            results = _search_ddg_api(query, max_results)
            engine = "ddgs"
        except Exception:
            results = []

    if not results:
        results = _search_ddg_html(query, max_results, user_agent, timeout_sec)

    return results, engine


def _search_ddg_api(query: str, max_results: int) -> List[Dict[str, Any]]:
    items = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            url = r.get("href") or r.get("url") or ""
            if not url:
                continue
            items.append({
                "title": (r.get("title") or "").strip(),
                "snippet": _clean_snippet(r.get("body") or ""),
                "url": url,
                "source_name": _source_name(url),
                "date": (r.get("date") or "").strip() or None,
            })
    return items


def _search_ddg_html(
    query: str,
    max_results: int,
    user_agent: str,
    timeout_sec: int,
) -> List[Dict[str, Any]]:
    if BeautifulSoup is None:
        return []

    url = "https://duckduckgo.com/html/"
    headers = {"User-Agent": user_agent}
    resp = requests.get(url, params={"q": query}, headers=headers, timeout=timeout_sec)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    results: List[Dict[str, Any]] = []

    for result in soup.select("div.result"):
        if len(results) >= max_results:
            break
        link = result.select_one("a.result__a")
        snippet = result.select_one(".result__snippet")
        if not link or not link.get("href"):
            continue
        href = link.get("href")
        title = link.get_text(strip=True)
        snip = snippet.get_text(" ", strip=True) if snippet else ""
        results.append({
            "title": title,
            "snippet": _clean_snippet(snip),
            "url": href,
            "source_name": _source_name(href),
            "date": None,
        })

    return results


def _fetch_sources(
    results: List[Dict[str, Any]],
    max_pages: int,
    user_agent: str,
    timeout_sec: int,
    max_chars: int,
) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    seen = set()
    headers = {"User-Agent": user_agent}

    for r in results:
        if len(sources) >= max_pages:
            break
        url = r.get("url")
        if not url or url in seen:
            continue
        seen.add(url)

        text = ""
        publish_date = None
        title = r.get("title") or ""
        try:
            resp = requests.get(url, headers=headers, timeout=timeout_sec)
            if resp.status_code < 400:
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" in content_type:
                    html = resp.text or ""
                    text = _extract_readable_text(html, max_chars=max_chars)
                    publish_date = _extract_publish_date(html)
                    title = title or _extract_title(html) or ""
        except Exception:
            text = ""

        sources.append({
            "url": url,
            "title": title,
            "snippet": _clean_snippet(r.get("snippet") or ""),
            "text": text,
            "publish_date": publish_date,
            "source_name": _source_name(url),
        })

    return sources


def _fetch_single_source(
    result: Dict[str, Any],
    user_agent: str,
    timeout_sec: int,
    max_chars: int,
) -> Dict[str, Any]:
    headers = {"User-Agent": user_agent}
    url = result.get("url")
    text = ""
    publish_date = None
    title = result.get("title") or ""
    try:
        resp = requests.get(url, headers=headers, timeout=timeout_sec)
        if resp.status_code < 400:
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type:
                html = resp.text or ""
                text = _extract_readable_text(html, max_chars=max_chars)
                publish_date = _extract_publish_date(html)
                title = title or _extract_title(html) or ""
    except Exception:
        text = ""

    return {
        "url": url,
        "title": title,
        "snippet": _clean_snippet(result.get("snippet") or ""),
        "text": text,
        "publish_date": publish_date or result.get("date"),
        "source_name": _source_name(url or ""),
    }


def _source_matches_query(
    query: str,
    source: Dict[str, Any],
    llm_chat_func=None,
    min_score: float = 0.35,
) -> tuple[bool, str, float]:
    summary_obj = source.get("source_summary") or {}
    summary_text = " ".join(summary_obj.get("summary_bullets") or [])
    numbers = " ".join(summary_obj.get("key_numbers") or [])
    snippet = source.get("snippet") or ""
    candidate_text = _normalize_text(" ".join([summary_text, numbers, snippet]), 1800)
    if not candidate_text:
        return False, "empty_source", 0.0

    if llm_chat_func:
        judge_system = (
            "Return ONLY valid JSON with keys: has_answer, score, reason. "
            "has_answer is true only if the source contains information that directly answers the question. "
            "score is 0..1."
        )
        payload = {
            "question": query,
            "source_title": source.get("title"),
            "source_url": source.get("url"),
            "source_text": candidate_text,
        }
        try:
            raw = llm_chat_func(judge_system, json.dumps(payload, ensure_ascii=False), max_tokens=120)
            parsed = _parse_json(raw)
            score = float(parsed.get("score") or 0.0)
            has_answer = bool(parsed.get("has_answer")) and score >= min_score
            reason = str(parsed.get("reason") or "llm_judge")
            return has_answer, reason, max(0.0, min(1.0, score))
        except Exception:
            pass

    score = _lexical_match_score(query, candidate_text)
    return score >= min_score, "lexical_match", score


def _lexical_match_score(query: str, text: str) -> float:
    q_tokens = {
        t for t in re.findall(r"[a-z0-9]+", (query or "").lower())
        if len(t) > 2 and t not in {"the", "and", "for", "with", "from", "that", "what", "tell", "please"}
    }
    if not q_tokens:
        return 1.0 if text else 0.0
    hay = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    if not hay:
        return 0.0
    overlap = len(q_tokens.intersection(hay))
    return overlap / max(1, len(q_tokens))


def _emit_progress(progress_cb: Optional[Callable[[Dict[str, Any]], None]], event: Dict[str, Any]) -> None:
    if not progress_cb:
        return
    try:
        progress_cb(event)
    except Exception:
        return


def _extract_readable_text(html: str, max_chars: int) -> str:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()
        parts = []
        for tag in soup.find_all(["h1", "h2", "h3", "p", "li"]):
            txt = tag.get_text(" ", strip=True)
            if txt:
                parts.append(txt)
        text = " ".join(parts)
    else:
        text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = _normalize_text(text, max_chars)
    return text


def _extract_title(html: str) -> Optional[str]:
    if BeautifulSoup is None:
        return None
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return None


def _extract_publish_date(html: str) -> Optional[str]:
    if BeautifulSoup is None:
        return _extract_date_from_text(html)

    soup = BeautifulSoup(html, "html.parser")
    candidates: List[str] = []
    for meta in soup.find_all("meta"):
        name = (meta.get("name") or meta.get("property") or "").lower()
        if name in {
            "article:published_time",
            "og:published_time",
            "pubdate",
            "publish-date",
            "publication_date",
            "date",
            "dc.date",
            "dc.date.issued",
            "dc.date.created",
            "dcterms.date",
            "sailthru.date",
        }:
            content = (meta.get("content") or "").strip()
            if content:
                candidates.append(content)

    for t in soup.find_all("time"):
        if t.get("datetime"):
            candidates.append(t.get("datetime"))
        else:
            txt = t.get_text(" ", strip=True)
            if txt:
                candidates.append(txt)

    for c in candidates:
        normalized = _extract_date_from_text(c)
        if normalized:
            return normalized

    return _extract_date_from_text(html)


def _extract_date_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b", text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    month_map = {
        "jan": "01",
        "january": "01",
        "feb": "02",
        "february": "02",
        "mar": "03",
        "march": "03",
        "apr": "04",
        "april": "04",
        "may": "05",
        "jun": "06",
        "june": "06",
        "jul": "07",
        "july": "07",
        "aug": "08",
        "august": "08",
        "sep": "09",
        "sept": "09",
        "september": "09",
        "oct": "10",
        "october": "10",
        "nov": "11",
        "november": "11",
        "dec": "12",
        "december": "12",
    }

    m = re.search(r"\b([A-Za-z]{3,9})\s+(\d{1,2}),\s*(\d{4})\b", text)
    if m:
        month = month_map.get(m.group(1).lower())
        if month:
            return f"{m.group(3)}-{month}-{int(m.group(2)):02d}"

    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})\b", text)
    if m:
        month = month_map.get(m.group(2).lower())
        if month:
            return f"{m.group(3)}-{month}-{int(m.group(1)):02d}"

    return None


def _chunk_text(text: str, max_chars: int, max_chunks: int) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text) and len(chunks) < max_chunks:
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


def _map_chunk(llm_chat_func, chunk_text: str, max_tokens: int) -> Optional[dict]:
    try:
        raw = llm_chat_func(MAP_SYSTEM_PROMPT, chunk_text, max_tokens=max_tokens)
        return _parse_json(raw)
    except Exception:
        return None


def _reduce_source(llm_chat_func, src: dict, max_tokens: int) -> Optional[dict]:
    payload = {
        "source_title": src.get("title"),
        "url": src.get("url"),
        "chunks": src.get("chunk_summaries") or [],
    }
    user_prompt = json.dumps(payload, ensure_ascii=False)
    try:
        raw = llm_chat_func(REDUCE_SYSTEM_PROMPT, user_prompt, max_tokens=max_tokens)
        return _parse_json(raw)
    except Exception:
        return None


def _decide_strategy(
    llm_chat_func,
    query: str,
    user_text: str,
    results: List[Dict[str, Any]],
    explicit_citations: bool,
    max_tokens: int,
) -> Dict[str, Any]:
    snippets = []
    for r in results[:5]:
        snippets.append({
            "title": r.get("title"),
            "snippet": _shorten(_clean_snippet(r.get("snippet") or ""), 200),
        })
    payload = {
        "query": query,
        "user_text": user_text,
        "explicit_citations": explicit_citations,
        "top_snippets": snippets,
    }
    system = (
        "Return ONLY valid JSON with keys: depth, answer_style, cite. "
        "depth: 'quick' or 'deep'. "
        "answer_style: 'short' (1 sentence) or 'brief' (2-3 sentences). "
        "cite: true only if explicit_citations is true. "
        "If the question looks like a single fact/formula, choose depth='quick'. "
        "If the user asked for research/overview/latest, choose depth='deep'."
    )
    try:
        raw = llm_chat_func(system, json.dumps(payload, ensure_ascii=False), max_tokens=max_tokens)
        data = _parse_json(raw)
    except Exception:
        data = {}
    depth = data.get("depth") if data.get("depth") in {"quick", "deep"} else "deep"
    answer_style = data.get("answer_style") if data.get("answer_style") in {"short", "brief"} else "brief"
    cite = bool(data.get("cite")) and explicit_citations
    return {"depth": depth, "answer_style": answer_style, "cite": cite}


def _final_answer(
    llm_chat_func,
    query: str,
    sources: List[Dict[str, Any]],
    include_sources: bool,
    max_tokens: int,
    answer_style: str = "brief",
) -> Optional[dict]:
    reduced = []
    for s in sources:
        reduced.append({
            "title": s.get("title"),
            "url": s.get("url"),
            "publish_date": s.get("publish_date"),
            "source_name": s.get("source_name"),
            "source_summary": s.get("source_summary"),
        })
    payload = {
        "question": query,
        "include_sources": bool(include_sources),
        "answer_style": answer_style,
        "sources": reduced,
    }
    user_prompt = json.dumps(payload, ensure_ascii=False)
    try:
        raw = llm_chat_func(FINAL_SYSTEM_PROMPT, user_prompt, max_tokens=max_tokens)
        data = _parse_json(raw)
    except Exception:
        return None

    spoken = (data.get("answer_spoken") or "").strip()
    display = (data.get("answer_display") or "").strip()
    sources_out = data.get("sources") or []
    spoken = _strip_urls(spoken)
    display = _ensure_urls_in_parentheses(display)
    if not include_sources:
        display = _strip_urls(display)
        display = re.sub(r"\(\s*Source:[^)]*\)", "", display).strip()
        display = re.sub(r"\bLinks?:\s*", "", display, flags=re.I).strip()
    allowed = [s.get("url") for s in sources if s.get("url")]
    display = _filter_display_urls(display, allowed)
    if include_sources and "Source:" not in display:
        display = _append_sources(display, sources)
    if not spoken or _is_link_only(spoken):
        return _fallback_final_answer(query, sources, include_sources)
    return {
        "answer_spoken": spoken,
        "answer_display": display or spoken,
        "sources": sources_out,
    }


def _fallback_reduce(src: dict) -> dict:
    takeaways = []
    for c in src.get("chunk_summaries") or []:
        takeaways.extend(c.get("chunk_takeaways") or [])
    takeaways = [t for t in takeaways if t]
    return {
        "source_title": src.get("title"),
        "url": src.get("url"),
        "summary_bullets": takeaways[:6],
        "key_numbers": _extract_numbers(" ".join(takeaways)),
        "what_it_means": [],
        "confidence": "low",
    }


def _fallback_final_answer(query: str, sources: List[Dict[str, Any]], include_sources: bool) -> dict:
    if not sources:
        spoken = "I couldn't find any results for that."
        return {"answer_spoken": spoken, "answer_display": spoken, "sources": []}
    top = sources[0]
    snippet = top.get("snippet") or ""
    spoken = snippet if snippet.endswith((".", "!", "?")) else snippet + "."
    spoken = _shorten(spoken, 240)
    display = spoken
    if include_sources:
        display = _append_sources(spoken, sources[:1])
    return {"answer_spoken": _strip_urls(spoken), "answer_display": display, "sources": [
        {"title": top.get("title"), "url": top.get("url")}
    ]}


def _compact_sources_for_result(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for s in sources:
        out.append({
            "title": s.get("title"),
            "url": s.get("url"),
            "publish_date": s.get("publish_date"),
            "source_name": s.get("source_name"),
            "source_summary": s.get("source_summary"),
        })
    return out


def _save_debug(path: Path, payload: dict, sources: List[Dict[str, Any]]) -> None:
    try:
        debug = {
            "payload": payload,
            "sources": sources,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(debug, f, ensure_ascii=False, indent=2)
    except Exception:
        return


def _parse_json(text: str) -> dict:
    if not text:
        raise ValueError("Empty JSON")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object")
    blob = text[start:end + 1]
    data = json.loads(blob)
    if not isinstance(data, dict):
        raise ValueError("JSON is not an object")
    return data


def _normalize_text(text: str, max_chars: Optional[int] = None) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if max_chars and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip()
    return cleaned


def _clean_snippet(text: str) -> str:
    return _normalize_text(text, 320)


def _shorten(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."


def _strip_urls(text: str) -> str:
    return re.sub(r"https?://\S+", "", text).strip()


def _ensure_urls_in_parentheses(text: str) -> str:
    urls = re.findall(r"https?://\S+", text)
    if not urls:
        return text
    cleaned = re.sub(r"https?://\S+", "", text).strip()
    tail = " ".join([f"{u}" for u in urls])
    if tail:
        cleaned = f"{cleaned} (Source: {tail})"
    return cleaned


def _append_sources(text: str, sources: List[Dict[str, Any]]) -> str:
    items = []
    for s in sources[:5]:
        title = s.get("title") or ""
        url = s.get("url") or ""
        if title and url:
            items.append(f"{title} — {url}")
        elif url:
            items.append(url)
    if not items:
        return text
    return f"{text} (Source: " + "; ".join(items) + ")"


def _filter_display_urls(text: str, allowed_urls: List[str]) -> str:
    if not allowed_urls:
        return re.sub(r"https?://\S+", "", text).strip()
    urls = re.findall(r"https?://\S+", text)
    for u in urls:
        if u not in allowed_urls:
            text = text.replace(u, "").strip()
    return text


def _is_link_only(text: str) -> bool:
    if not text:
        return True
    t = text.strip()
    if re.search(r"https?://", t):
        stripped = re.sub(r"https?://\S+", "", t)
        stripped = re.sub(r"\bLinks?:\b", "", stripped, flags=re.I).strip()
        return len(stripped) < 12
    return False


def _extract_numbers(text: str) -> List[str]:
    return re.findall(r"\b\d+(?:[\.,]\d+)?\b", text)[:6]


def _source_name(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host.lstrip("www.")


def _hash_query(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _load_cache(cache_path: Path, ttl_seconds: int) -> Optional[Dict[str, Any]]:
    if not cache_path.exists():
        return None
    try:
        age = time.time() - cache_path.stat().st_mtime
        if age > ttl_seconds:
            return None
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data or not data.get("results"):
            return None
        data.setdefault("cache", {})
        data["cache"]["age_sec"] = int(age)
        return data
    except Exception:
        return None


def _save_cache(cache_path: Path, payload: Dict[str, Any]) -> None:
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        return


def _read_key_from_env_file(path: Path, key: str) -> str:
    try:
        if not path.exists():
            return ""
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() != key:
                    continue
                value = v.strip().strip("\"'").strip()
                return value
    except Exception:
        return ""
    return ""
