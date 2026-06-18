import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from services.orchestrator.planner import configure, plan
from services.orchestrator.llm_client import llm_chat


def load_config():
    with open(ROOT / "config" / "default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    tz = cfg["assistant"]["timezone"]
    memory_turns = int((cfg.get("memory") or {}).get("recent_turns", 20))

    configure(lambda prompt: llm_chat(cfg, "", prompt), memory_turns)

    now_iso = datetime.now(ZoneInfo(tz)).isoformat(timespec="seconds")

    tests = [
        "do you remember yesterday?",
        "continue Feb 6",
        "weather tomorrow in Yerevan",
        "what should I wear tomorrow",
        "go to Desktop and tail 2 lines sample.txt",
    ]

    for t in tests:
        p = plan(t, now_iso, tz)
        print(f"\nUser: {t}")
        print(json.dumps(p, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
