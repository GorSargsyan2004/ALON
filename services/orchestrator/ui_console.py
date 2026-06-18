from __future__ import annotations

import os
from dataclasses import dataclass
from contextlib import contextmanager
from typing import List

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    from rich.prompt import Prompt
    from rich.control import Control
    _RICH_OK = True
except Exception:
    Console = None
    Panel = None
    Text = None
    box = None
    Prompt = None
    Control = None
    _RICH_OK = False


_BANNER = (
    """
╭─────────────────────────────────────────────────────────────────────────────────╮
│     █████      █████████   █████          ███████    ██████   █████    █████    │ 
│    ░███░      ███░░░░░███ ░░███         ███░░░░░███ ░░██████ ░░███    ░░░███    │
│    ░███      ░███    ░███  ░███        ███     ░░███ ░███░███ ░███      ░███    │
│    ░███      ░███████████  ░███       ░███      ░███ ░███░░███░███      ░███    │
│    ░███      ░███░░░░░███  ░███       ░███      ░███ ░███ ░░██████      ░███    │
│    ░███      ░███    ░███  ░███      █░░███     ███  ░███  ░░█████      ░███    │
│    ░█████    █████   █████ ███████████ ░░░███████░   █████  ░░█████    █████    │
│    ░░░░░    ░░░░░   ░░░░░ ░░░░░░░░░░░    ░░░░░░░    ░░░░░    ░░░░░    ░░░░░     │
│       __          __   __   __      __        __   __   __                      │
│      |__) \ /    / _` /  \ |__)    /__`  /\  |__) / _` /__` \ /  /\  |\ |       │
│      |__)  |     \__> \__/ |  \    .__/ /~~\ |  \ \__> .__/  |  /~~\ | \|       │
╰─────────────────────────────────────────────────────────────────────────────────╯\n\n\n\n\n
"""
)


@dataclass
class Theme:
    banner_start: str = "#ADFFE2"
    banner_end: str = "#C4B5FD"
    pill_border: str = "#BAFFEA"
    assistant_border: str = "#DEB8FF"


class TerminalUI:
    def __init__(self, assistant_name: str = "Alon", fancy: bool = True):
        self.assistant_name = assistant_name
        self.fancy = bool(fancy and _RICH_OK)
        self.console = Console() if self.fancy else None
        self.theme = Theme()

    def show_banner(self) -> None:
        if not self.fancy or not self.console:
            return
        banner = gradient_text(_BANNER, self.theme.banner_start, self.theme.banner_end)
        self.console.print(banner)

    def prompt_input(self) -> str:
        if not self.fancy or not self.console:
            return input("You: ")
        text = self.console.input("> ")
        # attempt to clear the raw input line, then re-render pill with user text
        if Control:
            try:
                self.console.control(Control.cursor_up(1), Control.clear_line())
            except Exception:
                pass
        # re-render pill with user text
        pill = self._pill(f"> {text}", dim=False)
        self.console.print(pill)
        return text

    def print_assistant(self, text: str) -> None:
        if not self.fancy or not self.console:
            print(f"{self.assistant_name}: {strip_bold_markers(text)}\n")
            return
        title = Text(self.assistant_name, style=f"bold {self.theme.assistant_border}")
        content = render_bold_text(text)
        panel = Panel(
            content,
            title=title,
            border_style=self.theme.assistant_border,
            box=box.ROUNDED,
            expand=False,
        )
        self.console.print(panel)

    def print_status(self, text: str) -> None:
        if not self.fancy or not self.console:
            print(text)
            return
        self.console.print(Text(text, style="dim"))

    def print_dim_block(self, label: str, text: str, max_chars: int = 1200) -> None:
        if max_chars and len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        if not self.fancy or not self.console:
            print(f"{label}: {text}")
            return
        content = Text(f"{label}: {text}", style="dim")
        self.console.print(content)

    def spinner(self, label: str):
        if self.fancy and self.console and hasattr(self.console, "status"):
            return self.console.status(Text(label, style="dim"))
        return _nullcontext()

    def print_error(self, text: str) -> None:
        if not self.fancy or not self.console:
            print(f"ERROR: {text}")
            return
        self.console.print(Text(f"⚠ {text}", style="red"))

    def _pill(self, text: str, dim: bool) -> Panel:
        if not self.fancy or not self.console:
            return Panel(text)
        content = Text(f" {text} ")
        if dim:
            content.stylize("dim")
        panel = Panel(
            content,
            border_style=self.theme.pill_border,
            box=box.ROUNDED,
            expand=False,
        )
        return panel

    def print_debug(self, text: str) -> None:
        if os.environ.get("ALON_DEBUG") == "1" and self.console:
            self.console.print(Text(text, style="dim"))


def gradient_text(multiline: str, start_hex: str, end_hex: str) -> Text:
    if not _RICH_OK:
        return Text(multiline)
    lines = multiline.splitlines()
    out = Text()
    for line in lines:
        out.append(_gradient_line(line, start_hex, end_hex))
        out.append("\n")
    return out


def _gradient_line(line: str, start_hex: str, end_hex: str) -> Text:
    text = Text()
    # ignore leading spaces for gradient, but preserve them
    leading = len(line) - len(line.lstrip(" "))
    if leading > 0:
        text.append(" " * leading)
    visible = line[leading:]
    n = max(len(visible), 1)
    for i, ch in enumerate(visible):
        t = i / max(n - 1, 1)
        color = _lerp_hex(start_hex, end_hex, t)
        text.append(ch, style=color)
    return text


def _lerp_hex(a: str, b: str, t: float) -> str:
    ar, ag, ab = _hex_to_rgb(a)
    br, bg, bb = _hex_to_rgb(b)
    r = int(ar + (br - ar) * t)
    g = int(ag + (bg - ag) * t)
    bl = int(ab + (bb - ab) * t)
    return f"#{r:02X}{g:02X}{bl:02X}"


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def render_bold_text(text: str) -> Text:
    if not _RICH_OK:
        return Text(strip_bold_markers(text))
    import re

    out = Text()
    if not text:
        return out
    pos = 0
    for m in re.finditer(r"\*\*(.+?)\*\*", text):
        if m.start() > pos:
            out.append(text[pos:m.start()])
        seg = m.group(1)
        out.append(seg, style="white")
        pos = m.end()
    if pos < len(text):
        out.append(text[pos:])
    return out


def strip_bold_markers(text: str) -> str:
    import re
    return re.sub(r"\*\*(.+?)\*\*", r"\1", text or "")


@contextmanager
def _nullcontext():
    yield
