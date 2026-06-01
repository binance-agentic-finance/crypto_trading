"""shim — atomic.data.pro (re-export AI signal helpers)"""
from cyqnt_trd.data_cli import (  # noqa: F401
    pro_indicators_fetch, pro_trade_signal_query, pro_trade_signal_rank,
)
from cyqnt_trd.data_cli.pro import pro_arb_scan as _cy_pro_arb_scan


def pro_arb_scan(symbols=None, profile=None, **_legacy):
    """atomic-style — workflow arb-scan across base symbols.

    Forwards to ``cyqnt_trd.data_cli.pro.pro_arb_scan`` and guarantees
    a dict shape (``{"symbols": [...]}``) so callers like
    square-buzz-screener's ``fetch_shared_arb`` can do
    ``raw.get("symbols")`` safely. ``profile`` is accepted for legacy
    compatibility and ignored (cyqnt_trd handles auth profile selection
    out-of-band).
    """
    syms = list(symbols) if symbols else []
    if not syms:
        return {"symbols": []}
    try:
        result = _cy_pro_arb_scan(syms)
    except Exception:
        return {"symbols": []}
    if isinstance(result, dict):
        return result
    if isinstance(result, list):
        return {"symbols": result}
    return {"symbols": []}
