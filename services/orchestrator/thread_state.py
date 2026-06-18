from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


def get_recent_state(log_path: Path, turns: int = 3) -> Dict:
    if not log_path.exists():
        return {}
    entries = _tail_jsonl(log_path, turns * 6)
    state = {}
    for e in reversed(entries):
        if e.get("type") == "state":
            state = e.copy()
            break
    return state or {}


def _tail_jsonl(path: Path, n: int) -> list[dict]:
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        block = 8192
        data = b""
        while size > 0 and data.count(b"\n") <= n:
            read_size = min(block, size)
            size -= read_size
            f.seek(size)
            data = f.read(read_size) + data
    lines = data.splitlines()[-n:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line.decode("utf-8", errors="replace")))
        except Exception:
            continue
    return out
