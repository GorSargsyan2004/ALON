import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from services.orchestrator.llm_client import llm_chat
from services.tools.websearch.search import search_web


def load_config():
    with open(ROOT / "config" / "default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    web_cfg = (cfg.get("tools") or {}).get("websearch") or {}
    query = "formula of celsius to fahrenheit"

    result = search_web(
        query=query,
        max_results=int(web_cfg.get("max_results", 5)),
        fetch_top_n_pages=int(web_cfg.get("fetch_top_n_pages", 2)),
        user_agent=web_cfg.get("user_agent", "ALON/1.0 (+local assistant)"),
        timeout_sec=int(web_cfg.get("timeout_sec", 20)),
        cache_ttl_seconds=int(web_cfg.get("cache_ttl_seconds", 21600)),
        cache_dir=ROOT / "data" / "cache" / "web",
        llm_chat_func=lambda sys_p, user_p, max_tokens=None: llm_chat(cfg, sys_p, user_p, max_tokens=max_tokens),
        turn_id="test_websearch",
        include_sources=True,
        user_text=query,
        settings={
            "provider": str(web_cfg.get("provider", "local")).lower(),
            "max_chars_per_page_clean": int(web_cfg.get("max_chars_per_page_clean", 12000)),
            "max_chars_per_chunk": int(web_cfg.get("max_chars_per_chunk", 2500)),
            "max_chunks_per_page": int(web_cfg.get("max_chunks_per_page", 6)),
            "llm_chunk_max_tokens": int(web_cfg.get("llm_chunk_max_tokens", 250)),
            "llm_reduce_max_tokens": int(web_cfg.get("llm_reduce_max_tokens", 300)),
            "llm_final_max_tokens": int(web_cfg.get("llm_final_max_tokens", 350)),
            "request_timeout_sec": int(web_cfg.get("request_timeout_sec", 15)),
            "max_source_checks": int(web_cfg.get("max_source_checks", 5)),
            "min_match_score": float(web_cfg.get("min_match_score", 0.35)),
            "gemini": (web_cfg.get("gemini") or {}),
        },
    )

    final = result.get("final_answer") or {}
    spoken = final.get("answer_spoken") or ""
    display = final.get("answer_display") or ""

    print("Spoken:", spoken)
    print("Display:", display)
    print("Has URL in spoken:", bool(re.search(r"https?://", spoken)))
    print("Chunks total:", result.get("metrics", {}).get("total_chunks"))


if __name__ == "__main__":
    main()
