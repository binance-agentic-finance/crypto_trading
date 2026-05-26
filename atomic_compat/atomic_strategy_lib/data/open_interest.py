"""shim — atomic.data.open_interest"""
from cyqnt_trd.data_cli import fetch_open_interest, fetch_oi_history  # noqa: F401
from cyqnt_trd.compat.types import OpenInterest, OIHistoryPoint


def open_interest_fetch(symbol, profile=None, binary="binance-cli"):
    """atomic-style — returns OpenInterest dataclass."""
    df = fetch_open_interest(symbol)
    if df is None or df.empty:
        return OpenInterest(symbol=symbol, oi_base=0.0, oi_value=0.0)
    row = df.iloc[0]
    return OpenInterest(
        symbol=symbol,
        oi_base=float(row.get("oi_base", row.get("openInterest", 0))),
        oi_value=float(row.get("oi_value", row.get("openInterestValue", 0))),
        timestamp=int(row.get("timestamp", 0) or 0),
    )


def oi_history_fetch(symbol, period="5m", limit=30, profile=None, binary="binance-cli"):
    """atomic-style — returns list[OIHistoryPoint]."""
    df = fetch_oi_history(symbol, interval=period, limit=limit)
    if df is None or df.empty:
        return []
    out = []
    for _, row in df.iterrows():
        out.append(OIHistoryPoint(
            timestamp=int(row.get("timestamp", 0)),
            oi_value=float(row.get("oi_value", row.get("sumOpenInterestValue", 0))),
            oi_base=float(row.get("oi_base", row.get("sumOpenInterest", 0))),
        ))
    return out


def oi_delta_pct(history):
    """Compute percentage change between oldest and newest OI points.

    Sourced verbatim from atomic_strategy_lib.data.open_interest.oi_delta_pct.

    Parameters
    ----------
    history : list[OIHistoryPoint]
        OI history points ordered oldest → newest.

    Returns
    -------
    Optional[float]
        Percentage change, or None if fewer than 2 points or oldest is zero.
    """
    if len(history) < 2:
        return None
    oldest = history[0].oi_value
    newest = history[-1].oi_value
    if oldest == 0:
        return None
    return ((newest - oldest) / oldest) * 100
