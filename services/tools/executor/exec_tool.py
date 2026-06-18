from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from services.tools.filesystem.policy import resolve_safe, assert_exec_allowed, DESKTOP_ROOT


ALLOWLIST = {"python", "pip", "git", "node", "npm", "powershell", "cmd", "bash", "codex"}
DENYLIST = {"reg", "sc", "net", "shutdown", "format", "diskpart", "bcdedit"}


def run_command(
    command: list[str],
    cwd: str,
    timeout_sec: int = 60,
    mode: str = "windows",
    wsl_prelude: str | None = None,
    input_text: str | None = None,
) -> dict:
    if not command:
        raise ValueError("Empty command")
    exe = command[0].lower()
    if exe in DENYLIST or exe not in ALLOWLIST:
        raise PermissionError(f"Command not allowed: {exe}")

    cwd_path = resolve_safe(cwd)
    assert_exec_allowed(cwd_path)

    if mode == "wsl":
        cmd_str = " ".join(shlex.quote(c) for c in command)
        if wsl_prelude:
            cmd_str = f"{wsl_prelude}; {cmd_str}"
        wsl_cwd = _to_wsl_path(cwd_path)
        full_cmd = ["wsl.exe", "--cd", wsl_cwd, "--", "bash", "-lc", cmd_str]
    else:
        full_cmd = command

    result = subprocess.run(
        full_cmd,
        cwd=str(cwd_path),
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        input=input_text,
    )
    stdout = _cap(result.stdout)
    stderr = _cap(result.stderr)
    return {
        "exit_code": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def _cap(text: str, limit: int = 50_000) -> str:
    if not text:
        return ""
    if len(text) > limit:
        return text[:limit] + "...(truncated)"
    return text


def _to_wsl_path(p: Path) -> str:
    # Convert C:\Users\X\Desktop -> /mnt/c/Users/X/Desktop
    drive = p.drive.rstrip(":").lower()
    tail = str(p).replace("\\", "/").split(":", 1)[-1]
    return f"/mnt/{drive}{tail}"
