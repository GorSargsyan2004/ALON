import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from services.orchestrator.response_composer_llm import configure, compose_reply
from services.orchestrator.format_validators import extract_urls, ensure_no_urls_in_spoken


def load_config():
    with open(ROOT / "config" / "default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mock_llm(prompt: str) -> str:
    # minimal dummy for local testing without calling the model
    return """{
      "spoken_text": "Sure — here is a concise answer.",
      "display_text": "Sure — here is a concise answer. (Links: https://example.com)",
      "memory_events": [],
      "followups": []
    }"""


def run_tests():
    cfg = load_config()
    now_iso = datetime.now(ZoneInfo(cfg["assistant"]["timezone"])).isoformat(timespec="seconds")
    configure(mock_llm)

    cases = [
        ("weather", {"ok": True, "result": {"location": "Yerevan", "when": "today", "t_min": 2, "t_max": 7, "precip_prob_max": 88}}, "Naah, the weather is not nice, cloudy, can you look at the weather today?"),
        ("weather", {"ok": True, "result": {"location": "Yerevan", "when": "today", "t_min": 2, "t_max": 7, "precip_prob_max": 88}}, "chance to rain 88 percent, fuck…"),
        ("search", {"ok": True, "result": {"query": "Gohar Sargsyan facebook", "results": [{"title": "Gohar Sargsyan - Facebook", "url": "https://facebook.com/x", "snippet": "..." }]}}, "search internet facebook link of my sister Gohar Sargsyan"),
        ("filesystem", {"ok": True, "result": {"op": "read", "path": "C:\\\\Users\\\\PC\\\\Desktop\\\\sample.txt", "content": "Hello"}},"read the file"),
    ]

    for intent, tool_result, user_text in cases:
        out = compose_reply(
            user_text=user_text,
            intent=intent,
            tool_result=tool_result,
            recent_context="",
            assistant_name=cfg["assistant"]["name"],
            system_style=cfg["assistant"]["system_style"],
            now_iso=now_iso,
            constraints={"max_links": 5, "max_tool_content_chars": 1500},
        )
        assert out["spoken_text"]
        assert "http" not in ensure_no_urls_in_spoken(out["spoken_text"]).lower()
        urls = extract_urls(out["display_text"])
        if urls:
            assert "(" in out["display_text"] and ")" in out["display_text"]

    print("Composer LLM checks passed.")


if __name__ == "__main__":
    run_tests()
