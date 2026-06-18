from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Dict, Any


def run_codex(
    prompt: str,
    desktop_dir: Path,
    output_filename: str,
    command: list[str],
    timeout_sec: int = 120,
) -> Dict[str, Any]:
    desktop_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = desktop_dir / "codex_prompt.txt"
    out_path = desktop_dir / output_filename

    prompt_path.write_text(prompt, encoding="utf-8")

    result = subprocess.run(
        command,
        input=prompt,
        text=True,
        cwd=str(desktop_dir),
        capture_output=True,
        timeout=timeout_sec,
    )

    stdout_path = desktop_dir / "codex_stdout.txt"
    stderr_path = desktop_dir / "codex_stderr.txt"
    stdout_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_path.write_text(result.stderr or "", encoding="utf-8")

    return {
        "ok": result.returncode == 0,
        "exit_code": result.returncode,
        "prompt_path": str(prompt_path),
        "output_path": str(out_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
    }
