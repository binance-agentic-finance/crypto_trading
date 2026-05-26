"""shim — atomic.data.pro (re-export AI signal helpers)"""
from cyqnt_trd.data_cli import (  # noqa: F401
    pro_indicators_fetch, pro_trade_signal_query, pro_trade_signal_rank,
)


def pro_arb_scan(*args, **kwargs):
    """Stub — atomic-only arb scanner."""
    return []
