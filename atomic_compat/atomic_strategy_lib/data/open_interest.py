"""shim — atomic.data.open_interest

Compatibility adapter from atomic_strategy_lib.data.open_interest onto
cyqnt_trd.data_cli.

Key translation rules:
- cyqnt_trd.data_cli.fetch_open_interest(symbol) -> dict snapshot
- cyqnt_trd.data_cli.fetch_oi_history(symbol, period=..., limit=...) -> DataFrame
- atomic callers expect:
  * open_interest_fetch(symbol) -> OpenInterest dataclass
  * oi_history_fetch(symbol, period, limit) -> list[OIHistoryPoint]
  * oi_delta_pct(history) or oi_delta_pct(symbol, lookback_hours=...) -> pct/dict
"""
from cyqnt_trd.data_cli import fetch_open_interest, fetch_oi_history  # noqa: F401
from cyqnt_trd.compat.types import OpenInterest, OIHistoryPoint


def open_interest_fetch(symbol, profile=None, binary="binance-cli"):
    """atomic-style — returns OpenInterest dataclass."""
    raw = fetch_open_interest(symbol)
    if raw is None or not isinstance(raw, dict):
        return OpenInterest(symbol=symbol, oi_base=0.0, oi_value=0.0)

    oi_base = float(raw.get("oi_base", raw.get("openInterest", 0)) or 0)
    ts = int(raw.get("timestamp", raw.get("time", 0)) or 0) or None

    # Best effort for oi_value (quote currency): use latest history row if present.
    oi_value = 0.0
    try:
        hist = fetch_oi_history(symbol, period="5m", limit=1)
        if hist is not None and not hist.empty:
            oi_value = float(hist.iloc[-1].get("oi_value", 0) or 0)
    except Exception:
        pass
    if oi_value == 0.0:
        # Fallback: leave as base if quote value unavailable.
        oi_value = oi_base

    return OpenInterest(
        symbol=str(raw.get("symbol", symbol)),
        oi_base=oi_base,
        oi_value=oi_value,
        timestamp=ts,
    )


def oi_history_fetch(symbol, period="5m", limit=30, profile=None, binary="binance-cli"):
    """atomic-style — returns list[OIHistoryPoint]."""
    df = fetch_oi_history(symbol, period=period, limit=limit)
    if df is None or getattr(df, "empty", True):
        return []
    out = []
    for _, row in df.iterrows():
        out.append(OIHistoryPoint(
            timestamp=int(row.get("timestamp", 0) or 0),
            oi_value=float(row.get("oi_value", row.get("sumOpenInterestValue", 0)) or 0),
            oi_base=float(row.get("oi_base", row.get("sumOpenInterest", 0)) or 0),
        ))
    return out


def oi_delta_pct(history_or_symbol, lookback_hours=24, profile=None, binary="binance-cli"):
    """Compute percentage change in OI.

    Supports two call forms for atomic compatibility:
    1) oi_delta_pct(history_list[OIHistoryPoint]) -> float | None
    2) oi_delta_pct(symbol: str, lookback_hours=24, ...) -> {"delta_pct": ...}
    """
    # Form 2: symbol string -> fetch history internally
    if isinstance(history_or_symbol, str):
        period = "1h"
        limit = max(int(lookback_hours), 2)
        history = oi_history_fetch(history_or_symbol, period=period, limit=limit, profile=profile, binary=binary)
        delta = oi_delta_pct(history)
        return {"delta_pct": delta if delta is not None else 0}

    # Form 1: list[OIHistoryPoint]
    history = history_or_symbol or []
    if len(history) < 2:
        return None
    oldest = history[0].oi_value
    newest = history[-1].oi_value
    if oldest == 0:
        return None
    return ((newest - oldest) / oldest) * 100
