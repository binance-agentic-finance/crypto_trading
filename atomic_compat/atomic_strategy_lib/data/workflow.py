"""shim — atomic.data.workflow (re-export workflow helpers)"""
from cyqnt_trd.data_cli import (  # noqa: F401
    workflow_leaderboard, workflow_token, workflow_analysis,
)


def workflow_arb_scan(*args, **kwargs):
    """Stub — atomic-only workflow arb scan."""
    return []


def workflow_briefing(*args, **kwargs):
    """Stub — atomic-only briefing."""
    return {"summary": "", "items": []}


def workflow_top_strategies(*args, **kwargs):
    """Stub — atomic-only top strategies."""
    return []


def workflow_smart_money(*args, **kwargs):
    """Stub — atomic-only smart money."""
    return {"flow": [], "summary": ""}
