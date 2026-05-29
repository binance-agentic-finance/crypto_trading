"""shim — atomic.data.funding

Compatibility adapter from atomic_strategy_lib.data.funding onto cyqnt_trd.data_cli.

Key translation rules:
- cyqnt_trd.data_cli.fetch_funding_rate(symbol) → float (current rate)
- cyqnt_trd.data_cli.fetch_funding_history(symbol, limit=N) → DataFrame history
- atomic callers expect:
  * funding_rate_current(symbol) -> FundingRate dataclass
  * funding_rate_fetch(symbol, limit=N) -> list[FundingRate]
"""
from cyqnt_trd.data_cli import fetch_funding_rate, fetch_funding_history  # noqa: F401
from cyqnt_trd.compat.types import FundingRate


def _df_last(df):
    if df is None or getattr(df, "empty", True):
        return None
    try:
        return df.iloc[-1]
    except Exception:
        return None


def funding_rate_current(symbol, profile=None, binary="binance-cli"):
    """atomic-style — returns single FundingRate dataclass (latest).

    Compatibility notes:
    - `profile` and `binary` are accepted for atomic signature parity but
      ignored; cyqnt_trd.data_cli uses binance-cli plus internal cache.
    - Current rate comes from fetch_funding_rate() (float); timestamp and
      mark_price are best-effort from the latest funding history row.
    """
    rate = fetch_funding_rate(symbol)
    try:
        rate = float(rate)
    except Exception:
        rate = 0.0

    # Best-effort metadata from latest history row
    df = fetch_funding_history(symbol, limit=1)
    last = _df_last(df)
    return FundingRate(
        symbol=symbol,
        rate=rate,
        timestamp=int(last.get("timestamp", 0)) if last is not None and "timestamp" in last else None,
        mark_price=float(last.get("mark_price", 0) or 0) if last is not None and "mark_price" in last else None,
        index_price=None,
        next_funding_time=None,
    )


def funding_rate_fetch(symbol, limit=100, profile=None, binary="binance-cli"):
    """atomic-style funding history → list[FundingRate]."""
    df = fetch_funding_history(symbol, limit=limit)
    if df is None or getattr(df, "empty", True):
        return []
    out = []
    for _, row in df.iterrows():
        out.append(FundingRate(
            symbol=str(row.get("symbol", symbol)),
            rate=float(row.get("rate", 0) or 0),
            timestamp=int(row.get("timestamp", 0) or 0),
            mark_price=float(row.get("mark_price", 0) or 0),
            index_price=None,
            next_funding_time=None,
        ))
    return out


def funding_rate_info(symbol, profile=None, binary="binance-cli"):
    """atomic alias — returns dict with rate + metadata."""
    fr = funding_rate_current(symbol, profile=profile, binary=binary)
    return {
        "symbol": fr.symbol,
        "rate": fr.rate,
        "timestamp": fr.timestamp,
        "mark_price": fr.mark_price,
        "index_price": fr.index_price,
        "next_funding_time": fr.next_funding_time,
    }
