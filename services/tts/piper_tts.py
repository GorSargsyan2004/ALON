from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def speak(text: str, model_path: Path, out_wav: Path) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "piper",
        "--model", str(model_path),
        "--output_file", str(out_wav),
    ]

    proc = subprocess.run(
        cmd,
        input=(text or "").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Piper failed.\nSTDOUT:\n{proc.stdout.decode('utf-8','ignore')}\n\n"
            f"STDERR:\n{proc.stderr.decode('utf-8','ignore')}"
        )
