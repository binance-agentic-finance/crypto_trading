"""shim — atomic.data.ticker"""
from cyqnt_trd.data_cli import fetch_24h_ticker, fetch_ticker_price as _fetch_ticker_price, fetch_price as _fetch_price  # noqa: F401
from cyqnt_trd.compat.types import Ticker


def _df_row_to_ticker(df, symbol_fallback=""):
    if df is None or df.empty:
        return Ticker(symbol=symbol_fallback, price=0.0)
    row = df.iloc[0]
    return Ticker(
        symbol=str(row.get("symbol", symbol_fallback)),
        price=float(row.get("price", 0)),
        change_pct_24h=float(row.get("change_pct", 0) or 0),
        high_24h=float(row.get("high_24h", 0) or 0),
        low_24h=float(row.get("low_24h", 0) or 0),
        volume_base_24h=float(row.get("volume_base", 0) or 0),
        volume_quote_24h=float(row.get("volume_quote", 0) or 0),
        trades_24h=int(row.get("trades", 0) or 0),
        weighted_avg_price=float(row.get("weighted_avg_price", 0) or 0),
    )


def ticker_24h_fetch(symbol, market="spot", profile=None, binary="binance-cli", refresh=False):
    """atomic-style — returns Ticker dataclass."""
    df = fetch_24h_ticker(symbol, market=market, refresh=refresh)
    return _df_row_to_ticker(df, symbol)


def ticker_price_fetch(symbol, market="spot", profile=None, binary="binance-cli", refresh=False):
    """atomic-style — returns Ticker dataclass."""
    df = _fetch_ticker_price(symbol, market=market, refresh=refresh)
    return _df_row_to_ticker(df, symbol)


def gainers_list_fetch(market="spot", min_quote_volume=0.0, top_n=10, profile=None,
                      binary="binance-cli"):
    """atomic-style — returns list[Ticker] sorted by 24h change."""
    from cyqnt_trd.data_cli import full_market_scan
    df = full_market_scan(market=market, min_quote_volume=min_quote_volume)
    if df is None or df.empty:
        return []
    df_sorted = df.sort_values("change_pct", ascending=False).head(top_n)
    return [_df_row_to_ticker(df_sorted.iloc[[i]]) for i in range(len(df_sorted))]
