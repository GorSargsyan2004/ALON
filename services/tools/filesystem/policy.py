from __future__ import annotations

import os
from pathlib import Path


DESKTOP_ROOT = (Path.home() / "Desktop").resolve()
DOCUMENTS_ROOT = (Path.home() / "Documents").resolve()


def resolve_safe(path_str: str) -> Path:
    if not path_str:
        raise ValueError("Empty path")
    expanded = os.path.expandvars(os.path.expanduser(path_str))
    p = Path(expanded)
    if not p.is_absolute():
        p = (Path.cwd() / p)
    return p.resolve()


def assert_read_allowed(p: Path) -> None:
    if not _is_within(p, DOCUMENTS_ROOT) and not _is_within(p, DESKTOP_ROOT):
        raise PermissionError("Read denied: path is outside Documents/Desktop")


def assert_write_allowed(p: Path) -> None:
    if not _is_within(p, DESKTOP_ROOT):
        raise PermissionError("Write denied: only Desktop is allowed")


def assert_exec_allowed(cwd: Path) -> None:
    if not _is_within(cwd, DESKTOP_ROOT):
        raise PermissionError("Exec denied: only Desktop is allowed")


def _is_within(p: Path, root: Path) -> bool:
    try:
        p = p.resolve()
        root = root.resolve()
        return root == p or root in p.parents
    except Exception:
        return False
