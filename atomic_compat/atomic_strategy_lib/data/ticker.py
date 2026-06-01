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
                      binary="binance-cli", min_change_pct=None, **_legacy):
    """atomic-style — returns list[dict] sorted by 24h change.

    Each dict carries: ``symbol``, ``price``, ``change_pct``,
    ``volume_quote``, ``volume_base``, ``high_24h``, ``low_24h``,
    ``trades``, ``weighted_avg_price``. Case scripts consume this with
    dict access (``g["symbol"]``, ``g.get("change_pct")``), so we keep
    the dict shape rather than the Ticker dataclass.

    cyqnt_trd's ``full_market_scan`` only returns a list of symbol
    strings — it does not carry change_pct / volume_quote and does not
    accept ``min_quote_volume``. We therefore call the 24h-ticker
    endpoint directly, filter by quote volume / change %, and sort by
    24h change.
    """
    from cyqnt_trd.data_cli._subprocess import run_binance_cli

    if market == "futures":
        args = ["futures-usds", "ticker24hr-price-change-statistics"]
    else:
        args = ["spot", "ticker24hr"]

    raw = run_binance_cli(args)
    items = raw if isinstance(raw, list) else []

    def _f(v, default=0.0):
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    rows = []
    for item in items:
        sym = item.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        rows.append({
            "symbol":             sym,
            "price":              _f(item.get("lastPrice", item.get("price"))),
            "change_pct":         _f(item.get("priceChangePercent")),
            "high_24h":           _f(item.get("highPrice")),
            "low_24h":            _f(item.get("lowPrice")),
            "volume_base":        _f(item.get("volume")),
            "volume_quote":       _f(item.get("quoteVolume")),
            "trades":             int(item.get("count", 0) or 0),
            "weighted_avg_price": _f(item.get("weightedAvgPrice")),
        })
    if not rows:
        return []

    if min_quote_volume and min_quote_volume > 0:
        rows = [r for r in rows if r["volume_quote"] >= float(min_quote_volume)]
    if min_change_pct is not None:
        rows = [r for r in rows if r["change_pct"] >= float(min_change_pct)]

    rows.sort(key=lambda r: r["change_pct"], reverse=True)
    return rows[:top_n]
