from __future__ import annotations

import os
from pathlib import Path
import fnmatch
from datetime import datetime
from typing import Iterable, List, Optional, Union

from services.tools.filesystem.policy import (
    resolve_safe,
    assert_read_allowed,
    assert_write_allowed,
)

_PathLike = Union[str, Path]


def resolve_path(path: str, cwd: Optional[Path] = None, allow_roots: Optional[Iterable[Path]] = None) -> Path:
    # Backward-compatible resolver used by runner.py; allow_roots is ignored here
    if path is None or path == "":
        base = cwd if isinstance(cwd, Path) else Path.cwd()
        p = base
    else:
        raw = os.path.expandvars(os.path.expanduser(str(path)))
        candidate = Path(raw)
        if not candidate.is_absolute():
            base = cwd if isinstance(cwd, Path) else Path.cwd()
            candidate = base / candidate
        p = candidate
    return resolve_safe(str(p))


def list_dir(path: _PathLike, limit: int = 50) -> dict:
    p = resolve_path(path)
    assert_read_allowed(p)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Not a directory: {p}")
    entries = []
    for child in p.iterdir():
        try:
            stat = child.stat()
        except Exception:
            continue
        entries.append({
            "name": child.name,
            "type": "dir" if child.is_dir() else "file",
            "path": str(child),
            "size_bytes": int(stat.st_size) if child.is_file() else 0,
            "modified_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        })
        if limit and len(entries) >= int(limit):
            break
    return {"path": str(p), "entries": entries}


def read_file(path: _PathLike, max_bytes: int = 200_000) -> dict:
    p = resolve_path(path)
    assert_read_allowed(p)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Not a file: {p}")
    size = p.stat().st_size
    if size > max_bytes:
        raise ValueError(f"File too large: {size} bytes")
    data = p.read_bytes()
    text = _decode_text(data)
    return {"path": str(p), "content": text}


def tail_file(path: _PathLike, n_lines: int = 50, max_bytes: int = 200_000) -> dict:
    p = resolve_path(path)
    assert_read_allowed(p)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Not a file: {p}")
    size = p.stat().st_size
    if size > max_bytes:
        raise ValueError(f"File too large: {size} bytes")
    lines = _tail_lines(p, n_lines)
    return {"path": str(p), "lines": lines}


def write_file(path: _PathLike, content: str, mode: str = "overwrite") -> dict:
    p = resolve_path(path)
    assert_write_allowed(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if mode == "append":
        p.write_text((p.read_text(encoding="utf-8", errors="ignore") if p.exists() else "") + content, encoding="utf-8")
    else:
        p.write_text(content, encoding="utf-8")
    return {"path": str(p), "bytes": len(content.encode("utf-8"))}


def make_dir(path: _PathLike) -> dict:
    p = resolve_path(path)
    assert_write_allowed(p)
    p.mkdir(parents=True, exist_ok=True)
    return {"path": str(p), "created": True}


def delete_path(path: _PathLike) -> dict:
    p = resolve_path(path)
    assert_write_allowed(p)
    if p.is_dir():
        for child in p.iterdir():
            if child.is_dir():
                _delete_dir(child)
            else:
                child.unlink(missing_ok=True)
        p.rmdir()
    else:
        p.unlink(missing_ok=True)
    return {"path": str(p), "deleted": True}


def read_text(path: _PathLike, max_chars: int = 20000, max_bytes: int = 1_000_000) -> str:
    res = read_file(path, max_bytes=max_bytes)
    text = res.get("content", "")
    if max_chars and len(text) > max_chars:
        return text[:max_chars]
    return text


def read_tail_lines(path: _PathLike, n: int = 20, max_bytes: int = 1_000_000) -> str:
    res = tail_file(path, n_lines=n, max_bytes=max_bytes)
    lines = res.get("lines", [])
    return "\n".join(lines)


def stat(path: _PathLike) -> dict:
    p = resolve_path(path)
    assert_read_allowed(p)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")
    st = p.stat()
    return {
        "path": str(p),
        "type": "dir" if p.is_dir() else "file",
        "size_bytes": int(st.st_size) if p.is_file() else 0,
        "modified_iso": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
    }


def find_files(root: _PathLike, pattern: str = "*", limit: int = 50) -> List[Path]:
    r = resolve_path(root)
    assert_read_allowed(r)
    if not r.exists() or not r.is_dir():
        raise NotADirectoryError(f"Not a directory: {r}")
    matches: List[Path] = []
    for dirpath, _, filenames in os.walk(r):
        for name in filenames:
            if not fnmatch.fnmatch(name, pattern):
                continue
            p = Path(dirpath) / name
            try:
                assert_read_allowed(p)
            except Exception:
                continue
            matches.append(p)
            if limit and len(matches) >= int(limit):
                return matches
    return matches


def grep_text(
    root: _PathLike,
    query: str,
    limit_hits: int = 20,
    limit_files: int = 20,
) -> dict:
    r = resolve_path(root)
    assert_read_allowed(r)
    if not r.exists() or not r.is_dir():
        raise NotADirectoryError(f"Not a directory: {r}")
    if not query:
        raise ValueError("Query is empty")
    hits = []
    files_scanned = 0
    query_lower = query.lower()
    for dirpath, _, filenames in os.walk(r):
        for name in filenames:
            if files_scanned >= int(limit_files):
                return {
                    "path": str(r),
                    "query": query,
                    "hits": hits,
                    "files_scanned": files_scanned,
                    "hits_returned": len(hits),
                    "truncated": True,
                }
            p = Path(dirpath) / name
            try:
                assert_read_allowed(p)
            except Exception:
                continue
            try:
                data = p.read_bytes()
            except Exception:
                continue
            text = _decode_text(data)
            files_scanned += 1
            for i, line in enumerate(text.splitlines(), start=1):
                if query_lower in line.lower():
                    hits.append({"file": str(p), "line": i, "text": line.strip()[:300]})
                    if len(hits) >= int(limit_hits):
                        return {
                            "path": str(r),
                            "query": query,
                            "hits": hits,
                            "files_scanned": files_scanned,
                            "hits_returned": len(hits),
                            "truncated": True,
                        }
    return {
        "path": str(r),
        "query": query,
        "hits": hits,
        "files_scanned": files_scanned,
        "hits_returned": len(hits),
        "truncated": False,
    }


def _delete_dir(p: Path) -> None:
    for child in p.iterdir():
        if child.is_dir():
            _delete_dir(child)
        else:
            child.unlink(missing_ok=True)
    p.rmdir()


def _decode_text(data: bytes) -> str:
    for enc in ("utf-8", "utf-16", "cp1252"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def _tail_lines(p: Path, n: int) -> List[str]:
    with p.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        block = 8192
        data = b""
        while size > 0 and data.count(b"\n") <= n:
            read_size = min(block, size)
            size -= read_size
            f.seek(size)
            data = f.read(read_size) + data
    lines = data.splitlines()[-n:]
    return [l.decode("utf-8", errors="replace") for l in lines]
