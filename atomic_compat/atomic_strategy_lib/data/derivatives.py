"""shim — atomic.data.derivatives"""
from cyqnt_trd.data_cli import fetch_long_short_ratio  # noqa: F401


def long_short_ratio_fetch(symbol, period="5m", limit=30, ratio_type="topPosition",
                           profile=None, binary="binance-cli"):
    """atomic-style — returns list of dicts with timestamp + ratio.

    ratio_type: 'topPosition' (top traders by position) or 'topAccount'
    (top traders by account count) or 'global'.
    """
    df = fetch_long_short_ratio(symbol, ratio_type=ratio_type, period=period, limit=limit)
    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")


def top_trader_ls_ratio_fetch(symbol, by="position", period="5m", limit=30,
                              profile=None, binary="binance-cli"):
    """atomic-style — top traders LS ratio."""
    ratio_type = "topPosition" if by == "position" else "topAccount"
    return long_short_ratio_fetch(symbol, period=period, limit=limit, ratio_type=ratio_type)


def basis_fetch(symbol, period="5m", limit=30, profile=None, binary="binance-cli"):
    """atomic-style basis (futures price - spot price). Returns list of dicts."""
    try:
        from cyqnt_trd.data_cli.ratios import fetch_basis as _impl
        df = _impl(symbol, period=period, limit=limit)
        return df.to_dict(orient="records") if df is not None and not df.empty else []
    except (ImportError, AttributeError):
        return []


def taker_volume_fetch(symbol, period="5m", limit=30, profile=None, binary="binance-cli"):
    """atomic-style taker buy/sell volume. Returns list of dicts."""
    try:
        from cyqnt_trd.data_cli.ratios import fetch_taker_volume as _impl
        df = _impl(symbol, period=period, limit=limit)
        return df.to_dict(orient="records") if df is not None and not df.empty else []
    except (ImportError, AttributeError):
        return []


def leverage_brackets_fetch(symbol, profile=None, binary="binance-cli"):
    """atomic-style leverage brackets. Returns list of dicts."""
    try:
        from cyqnt_trd.data_cli.ratios import fetch_leverage_brackets as _impl
        df = _impl(symbol)
        return df.to_dict(orient="records") if df is not None and not df.empty else []
    except (ImportError, AttributeError):
        return []
