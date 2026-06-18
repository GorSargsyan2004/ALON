from __future__ import annotations

import fnmatch
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .safety import FsPolicy, is_binary_file, is_denied_extension, is_denied_name, is_within_root, resolve_path


def list_dir(
    path: str,
    policy: FsPolicy,
    recursive: bool = False,
    pattern: Optional[str] = None,
    max_entries: Optional[int] = None,
) -> Dict[str, Any]:
    root, resolved, rel = resolve_path(path, policy, must_exist=True)
    if not resolved.is_dir():
        raise NotADirectoryError("Path is not a directory")

    entries = []
    truncated = False
    limit = int(max_entries or policy.max_list_entries)

    def sort_key(p: Path):
        return (0 if p.is_dir() else 1, p.name.lower())

    if not recursive:
        for entry in sorted(resolved.iterdir(), key=sort_key):
            name = entry.name
            if is_denied_name(name, policy):
                continue
            if entry.is_dir():
                if name.lower() in policy.deny_dirs:
                    continue
            else:
                if is_denied_extension(entry, policy):
                    continue
            if pattern and not fnmatch.fnmatch(name, pattern):
                continue

            try:
                resolved_entry = entry.resolve()
            except Exception:
                continue
            if not is_within_root(resolved_entry, root):
                continue

            stat = entry.stat()
            entries.append({
                "name": name,
                "type": "dir" if entry.is_dir() else "file",
                "size_bytes": int(stat.st_size) if entry.is_file() else 0,
                "modified_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            })

            if len(entries) >= limit:
                truncated = True
                break
    else:
        for dir_path, dir_names, file_names in _walk_dir(resolved, policy):
            dir_path = Path(dir_path)
            for d in sorted(dir_names):
                if pattern and not fnmatch.fnmatch(d, pattern):
                    continue
                full = dir_path / d
                try:
                    if not is_within_root(full.resolve(), root):
                        continue
                except Exception:
                    continue
                rel_name = full.resolve().relative_to(resolved.resolve()).as_posix()
                stat = full.stat()
                entries.append({
                    "name": rel_name,
                    "type": "dir",
                    "size_bytes": 0,
                    "modified_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                })
                if len(entries) >= limit:
                    truncated = True
                    break
            if truncated:
                break
            for f in sorted(file_names):
                if pattern and not fnmatch.fnmatch(f, pattern):
                    continue
                if is_denied_name(f, policy):
                    continue
                full = dir_path / f
                if is_denied_extension(full, policy):
                    continue
                try:
                    if not is_within_root(full.resolve(), root):
                        continue
                except Exception:
                    continue
                rel_name = full.resolve().relative_to(resolved.resolve()).as_posix()
                stat = full.stat()
                entries.append({
                    "name": rel_name,
                    "type": "file",
                    "size_bytes": int(stat.st_size),
                    "modified_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                })
                if len(entries) >= limit:
                    truncated = True
                    break
            if truncated:
                break

    return {
        "path": rel,
        "entries": entries,
        "truncated": truncated,
    }


def read_file(
    path: str,
    policy: FsPolicy,
    start_line: int = 1,
    end_line: Optional[int] = None,
) -> Dict[str, Any]:
    root, resolved, rel = resolve_path(path, policy, must_exist=True)
    if not resolved.is_file():
        raise FileNotFoundError("Path is not a file")
    if is_denied_extension(resolved, policy):
        raise PermissionError("File type is blocked")
    if is_binary_file(resolved):
        raise PermissionError("Binary files are blocked")

    size = resolved.stat().st_size
    if size > policy.max_read_bytes:
        raise ValueError(f"File is too large to read safely ({size} bytes). Try a smaller file.")

    if start_line < 1:
        start_line = 1

    with open(resolved, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    total_lines = len(lines)
    if end_line is None or end_line > total_lines:
        end_line = total_lines
    if end_line < start_line:
        raise ValueError("end_line must be >= start_line")

    content_lines = lines[start_line - 1:end_line]
    content = "\n".join(content_lines)

    return {
        "path": rel,
        "start_line": start_line,
        "end_line": end_line,
        "total_lines": total_lines,
        "content": content,
    }


def search_text(
    path: str,
    query: str,
    policy: FsPolicy,
    file_glob: Optional[str] = None,
    max_files: Optional[int] = None,
    max_hits: Optional[int] = None,
) -> Dict[str, Any]:
    root, resolved, rel = resolve_path(path, policy, must_exist=True)
    if not resolved.is_dir():
        raise NotADirectoryError("Path is not a directory")
    if not query:
        raise ValueError("Query is empty")

    file_glob = file_glob or "*"
    max_files = int(max_files or policy.max_search_files)
    max_hits = int(max_hits or policy.max_search_hits)

    hits = []
    files_scanned = 0
    truncated = False
    query_lower = query.lower()

    for dir_path, dir_names, file_names in _walk_dir(resolved, policy):
        for fname in file_names:
            if files_scanned >= max_files:
                truncated = True
                break
            if not fnmatch.fnmatch(fname, file_glob):
                continue
            if is_denied_name(fname, policy):
                continue
            file_path = Path(dir_path) / fname
            if is_denied_extension(file_path, policy):
                continue
            try:
                if file_path.stat().st_size > policy.max_read_bytes:
                    continue
            except Exception:
                continue
            try:
                if not is_within_root(file_path.resolve(), root):
                    continue
            except Exception:
                continue
            if is_binary_file(file_path):
                continue

            files_scanned += 1
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, start=1):
                        if query_lower in line.lower():
                            hits.append({
                                "file": _rel_to_root(file_path, root),
                                "line": i,
                                "text": line.strip()[:300],
                            })
                            if len(hits) >= max_hits:
                                truncated = True
                                break
                if truncated:
                    break
            except Exception:
                continue
        if truncated:
            break

    return {
        "path": rel,
        "query": query,
        "hits": hits,
        "files_scanned": files_scanned,
        "hits_returned": len(hits),
        "truncated": truncated,
    }


def _walk_dir(root: Path, policy: FsPolicy):
    for dir_path, dir_names, file_names in os_walk(root):
        dir_names[:] = [
            d for d in dir_names
            if not is_denied_name(d, policy) and d.lower() not in policy.deny_dirs
        ]
        yield dir_path, dir_names, file_names


def os_walk(path: Path):
    for dir_path, dir_names, file_names in os.walk(path):
        yield dir_path, dir_names, file_names


def _rel_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.name
