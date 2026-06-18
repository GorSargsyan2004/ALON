from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from services.orchestrator.lmstudio_client import list_models


@dataclass
class ContextState:
    router_warmed_up: bool = False
    brain_warmed_up: bool = False
    server_fingerprint: Optional[str] = None


_STATE = ContextState()


def reset_warmup() -> None:
    _STATE.router_warmed_up = False
    _STATE.brain_warmed_up = False


def mark_router_warmed() -> None:
    _STATE.router_warmed_up = True


def mark_brain_warmed() -> None:
    _STATE.brain_warmed_up = True


def router_warmed() -> bool:
    return _STATE.router_warmed_up


def brain_warmed() -> bool:
    return _STATE.brain_warmed_up


def check_server_fingerprint(base_url: str) -> bool:
    """
    Returns True if server fingerprint changed (likely restart), and resets warm flags.
    """
    try:
        models = list_models(base_url, timeout_sec=5)
        first_id = models[0].get("id") if models else ""
        fingerprint = f"{len(models)}:{first_id}"
        if _STATE.server_fingerprint is None:
            _STATE.server_fingerprint = fingerprint
            return False
        if _STATE.server_fingerprint != fingerprint:
            _STATE.server_fingerprint = fingerprint
            reset_warmup()
            return True
    except Exception:
        # If server temporarily unreachable, don't reset warmup here.
        return False
    return False
