"""shim — atomic.data.kline (atomic-named wrappers around cyqnt_trd.data_cli)"""
from cyqnt_trd.data_cli import fetch_klines, fetch_klines_multi_tf  # noqa: F401
from cyqnt_trd.data_cli.kline import df_to_candles as _df_to_candles


def kline_fetch(symbol, interval="1h", limit=100, market="spot",
                profile=None, binary="binance-cli"):
    """atomic-style kline_fetch — returns list[Candle].

    Internally delegates to fetch_klines (returns DataFrame) then converts.
    """
    df = fetch_klines(symbol, interval=interval, limit=limit, market=market)
    candles = []
    if df.empty:
        return candles
    from cyqnt_trd.compat.types import Candle
    for _, row in df.iterrows():
        candles.append(Candle(
            timestamp=int(row.get("open_time", 0)),
            open=float(row.get("open", 0)),
            high=float(row.get("high", 0)),
            low=float(row.get("low", 0)),
            close=float(row.get("close", 0)),
            volume=float(row.get("volume", 0)),
            quote_volume=float(row.get("quote_volume", 0)),
            trades=int(row.get("trades", 0)),
        ))
    return candles


def kline_fetch_multi_tf(symbol, timeframes, limits=None, limit=None,
                        market="spot", profile=None, binary="binance-cli"):
    """atomic-style — returns dict[tf -> list[Candle]]."""
    out = {}
    for tf in timeframes:
        lim = (limits or {}).get(tf, limit or 100)
        out[tf] = kline_fetch(symbol, interval=tf, limit=lim, market=market)
    return out


def listing_age_fetch(symbol, profile=None, binary="binance-cli"):
    """atomic-style listing_age_fetch — re-export from data_cli."""
    from cyqnt_trd.data_cli.kline import listing_age_fetch as _impl
    return _impl(symbol)
