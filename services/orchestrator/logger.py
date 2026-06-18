import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any, Dict, Optional

def _jsonify(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    return obj

class JsonlLogger:
    def __init__(self, file_path: Path, timezone: str):
        self.file_path = file_path
        self.tz = ZoneInfo(timezone)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def now_iso(self) -> str:
        return datetime.now(self.tz).isoformat(timespec="seconds")

    def write(self, event: Dict[str, Any]) -> None:
        event = dict(event)
        event.setdefault("ts", self.now_iso())
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=_jsonify) + "\n")
