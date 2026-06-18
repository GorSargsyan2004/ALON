from __future__ import annotations

from pathlib import Path
from typing import Optional

from services.tools.executor.exec_tool import run_command
from services.tools.filesystem.policy import DESKTOP_ROOT, assert_write_allowed


def run_codex(
    codex_prompt: str,
    output_filename: str = "codex_result.md",
    exec_mode: str = "wsl",
    codex_command: Optional[list[str]] = None,
    wsl_prelude: Optional[str] = None,
    timeout_sec: int = 120,
) -> dict:
    desktop = DESKTOP_ROOT
    desktop.mkdir(parents=True, exist_ok=True)
    assert_write_allowed(desktop)

    prompt_path = desktop / "_codex_last_prompt.md"
    output_path = desktop / output_filename
    prompt_path.write_text(codex_prompt, encoding="utf-8")

    if not codex_command:
        return {
            "ok": False,
            "error": "codex_not_configured",
            "prompt_path": str(prompt_path),
            "output_path": str(output_path),
            "stdout": "",
            "stderr": "",
            "exit_code": 1,
        }

    result = run_command(
        codex_command,
        cwd=str(desktop),
        timeout_sec=timeout_sec,
        mode=exec_mode,
        wsl_prelude=wsl_prelude,
        input_text=codex_prompt,
    )

    stdout_path = desktop / "_codex_last_output.txt"
    stdout_path.write_text((result.get("stdout") or "") + "\n" + (result.get("stderr") or ""), encoding="utf-8")

    return {
        "ok": result.get("exit_code", 1) == 0,
        "prompt_path": str(prompt_path),
        "output_path": str(output_path),
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
        "exit_code": result.get("exit_code", 1),
    }
