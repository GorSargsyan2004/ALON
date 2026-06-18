import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from services.orchestrator.planner import configure, plan


def load_config():
    with open(ROOT / "config" / "default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mock_llm(prompt: str) -> str:
    # emulate calendar intent for the three prompts
    return """{
      "intent": "calendar",
      "context_request": {"mode":"last_turns","turns":6},
      "tool_args": {"query":"calendar"},
      "topic": "calendar",
      "followup": {"is_followup": true, "refers_to": "date"}
    }"""


def run_tests():
    cfg = load_config()
    tz = cfg["assistant"]["timezone"]
    now_iso = datetime.now(ZoneInfo(tz)).isoformat(timespec="seconds")
    configure(mock_llm)

    cases = [
        "what day is it after 6 days",
        "and what day is Feb 14",
        "what activity the world has on Feb 14",
    ]

    for text in cases:
        p = plan(text, now_iso, tz, "recent context", {"topic": "calendar"})
        assert p["intent"] == "calendar"
        assert p["topic"] == "calendar"

    print("Planner continuity checks passed.")


if __name__ == "__main__":
    run_tests()
