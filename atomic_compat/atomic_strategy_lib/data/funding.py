"""shim — atomic.data.funding"""
from cyqnt_trd.data_cli import fetch_funding_rate, fetch_funding_history  # noqa: F401
from cyqnt_trd.compat.types import FundingRate


def funding_rate_current(symbol, profile=None, binary="binance-cli"):
    """atomic-style — returns single FundingRate dataclass (latest)."""
    df = fetch_funding_rate(symbol)
    if df is None or df.empty:
        return FundingRate(symbol=symbol, rate=0.0)
    last = df.iloc[-1]
    return FundingRate(
        symbol=symbol,
        rate=float(last.get("rate", last.get("funding_rate", 0))),
        timestamp=int(last.get("timestamp", 0)) if "timestamp" in last else None,
        mark_price=float(last.get("mark_price", 0) or 0) if "mark_price" in last else None,
        index_price=float(last.get("index_price", 0) or 0) if "index_price" in last else None,
        next_funding_time=int(last.get("next_funding_time", 0) or 0) if "next_funding_time" in last else None,
    )


def funding_rate_fetch(symbol, profile=None, binary="binance-cli"):
    """atomic alias for funding_rate_current."""
    return funding_rate_current(symbol)


def funding_rate_info(symbol, profile=None, binary="binance-cli"):
    """atomic alias — returns dict with rate + metadata."""
    fr = funding_rate_current(symbol)
    return {
        "symbol": fr.symbol,
        "rate": fr.rate,
        "timestamp": fr.timestamp,
        "mark_price": fr.mark_price,
        "index_price": fr.index_price,
        "next_funding_time": fr.next_funding_time,
    }
