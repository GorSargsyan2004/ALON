from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class FsPolicy:
    allowed_roots: list[Path]
    max_read_bytes: int
    max_list_entries: int
    max_search_files: int
    max_search_hits: int
    deny_extensions: set[str]
    deny_dirs: set[str]
    deny_name_contains: list[str]


def build_policy(repo_root: Path, cfg: dict) -> FsPolicy:
    roots_cfg = cfg.get("allowed_roots") or ["."]
    roots: list[Path] = []
    for r in roots_cfg:
        p = Path(r)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        else:
            p = p.resolve()
        roots.append(p)

    deny_ext = {e.lower() for e in (cfg.get("deny_extensions") or [])}
    deny_dirs = {d.lower() for d in (cfg.get("deny_dirs") or [])}
    deny_contains = [s.lower() for s in (cfg.get("deny_name_contains") or [])]

    return FsPolicy(
        allowed_roots=roots,
        max_read_bytes=int(cfg.get("max_read_bytes", 200000)),
        max_list_entries=int(cfg.get("max_list_entries", 200)),
        max_search_files=int(cfg.get("max_search_files", 200)),
        max_search_hits=int(cfg.get("max_search_hits", 200)),
        deny_extensions=deny_ext,
        deny_dirs=deny_dirs,
        deny_name_contains=deny_contains,
    )


def resolve_path(
    path_str: str,
    policy: FsPolicy,
    must_exist: bool = True,
) -> tuple[Path, Path, str]:
    if not path_str:
        path_str = "."

    if _has_traversal(path_str):
        raise PermissionError("Path traversal is not allowed")

    candidates: list[Path] = []
    p = Path(path_str)
    if p.is_absolute():
        candidates = [p]
    else:
        for root in policy.allowed_roots:
            candidates.append(root / p)

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue

        root = _find_root(resolved, policy.allowed_roots)
        if root is None:
            continue
        if must_exist and not resolved.exists():
            continue

        rel = _relative_to_root(resolved, root)
        _check_deny_parts(rel, policy)
        if resolved.is_file():
            _check_deny_file(resolved, policy)

        return root, resolved, rel.as_posix() or "."

    if p.is_absolute():
        raise PermissionError("Path is outside allowed roots")
    raise FileNotFoundError(f"Path not found: {path_str}")


def is_within_root(path: Path, root: Path) -> bool:
    try:
        path = path.resolve()
        root = root.resolve()
    except Exception:
        return False
    return path == root or root in path.parents


def is_denied_name(name: str, policy: FsPolicy) -> bool:
    lower = (name or "").lower()
    if lower in policy.deny_dirs:
        return True
    for needle in policy.deny_name_contains:
        if needle in lower:
            return True
    return False


def is_denied_extension(path: Path, policy: FsPolicy) -> bool:
    return path.suffix.lower() in policy.deny_extensions


def is_binary_file(path: Path, sniff_bytes: int = 4096) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(sniff_bytes)
        return b"\x00" in chunk
    except Exception:
        return True


def _find_root(path: Path, roots: Iterable[Path]) -> Optional[Path]:
    for root in roots:
        if is_within_root(path, root):
            return root
    return None


def _relative_to_root(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except Exception:
        return Path(".")


def _has_traversal(path_str: str) -> bool:
    parts = Path(path_str).parts
    return any(p == ".." for p in parts)


def _check_deny_parts(rel: Path, policy: FsPolicy) -> None:
    for part in rel.parts:
        lower = part.lower()
        if lower in policy.deny_dirs:
            raise PermissionError(f"Access to '{part}' is blocked")
        for needle in policy.deny_name_contains:
            if needle in lower:
                raise PermissionError(f"Access to '{part}' is blocked")


def _check_deny_file(path: Path, policy: FsPolicy) -> None:
    if is_denied_extension(path, policy):
        raise PermissionError("File type is blocked")
