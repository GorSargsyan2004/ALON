import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from services.tools.filesystem.runner import run_plan, validate_plan
from services.orchestrator.logger import JsonlLogger


def main():
    cfg_path = ROOT / "config" / "default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    user = os.environ.get("USERPROFILE")
    desktop = Path(user) / "Desktop" if user else None
    target_dir = desktop if desktop and desktop.exists() else (ROOT / "data")
    target_dir.mkdir(parents=True, exist_ok=True)
    sample = target_dir / "sample.txt"
    sample.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")

    plan = {
        "cwd": "Desktop" if desktop and desktop.exists() else "ProjectRoot",
        "steps": [
            {"op": "list", "path": ".", "limit": 5},
            {"op": "tail", "path": str(sample.name), "lines": 2},
        ],
        "user_intent": "tail",
        "what_to_do_with_content": "store",
        "notes": "",
    }
    validate_plan(plan)

    logger = JsonlLogger(ROOT / "data" / "memory" / "alon_log.jsonl", cfg["assistant"]["timezone"])
    result = run_plan(
        plan,
        cfg,
        logger,
        session_id="test-session",
        turn_id="test-turn",
        llm_generate=lambda p: "",
        recent_context="",
        user_text="",
        project_root=ROOT,
    )
    print("Result:", result)
    print("Sample file:", sample)


if __name__ == "__main__":
    main()
