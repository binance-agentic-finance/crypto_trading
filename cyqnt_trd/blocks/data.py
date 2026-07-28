"""Data conversion + Binance public-API fetchers.

This module bridges the world of the user-facing pandas DataFrames with
the internal ``standard_bot`` types (``Bar``, ``MarketBundle``,
``DataSnapshot``).

It also exposes thin wrappers around Binance's **public** REST endpoints
for the futures data series that the user dataset most often asks for:
klines, open interest, funding rate, top long/short ratio, taker
buy/sell volume, and the 24-hour ticker (for universe scanning).

These do *not* require an API key — they call the public CoinTime
endpoints. ``requests`` is imported lazily so this module can be
imported in restricted environments.

Examples
--------
>>> from cyqnt_trd.blocks import data
>>> df = data.fetch_klines("BTCUSDT", "15m", limit=500)
>>> oi = data.fetch_oi("BTCUSDT", period="5m", limit=288)
>>> fr = data.fetch_funding_rate("BTCUSDT", limit=200)
>>> tickers = data.fetch_24h_tickers()
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ._utils import ensure_df

if TYPE_CHECKING:  # pragma: no cover
    from ..standard_bot.core import Bar, MarketBundle, DataSnapshot

__all__ = [
    # Type conversion
    "df_to_bars",
    "bars_to_df",
    "snapshot_to_df",
    # Binance public REST
    "fetch_klines",
    "fetch_oi",
    "fetch_funding_rate",
    "fetch_long_short_ratio",
    "fetch_taker_buy_sell_ratio",
    "fetch_24h_tickers",
    "fetch_exchange_info",
    "fetch_premium_index",
]


# ---------------------------------------------------------------------------
# DataFrame <-> Bar conversions
# ---------------------------------------------------------------------------


def df_to_bars(
    df: pd.DataFrame,
    instrument_id: str,
    timeframe: str,
    *,
    timestamp_col: str = "close_time",
    confirmed: bool = True,
) -> List["Bar"]:
    """Convert a DataFrame into a list of :class:`Bar` objects.

    The DataFrame must contain ``open / high / low / close / volume`` and
    a millisecond-timestamp column (default ``close_time``).
    """
    from ..standard_bot.core import Bar  # local import to avoid heavy deps at module import

    df = ensure_df(df)
    if timestamp_col not in df.columns:
        raise ValueError(f"timestamp column {timestamp_col!r} not found in DataFrame")
    bars: List["Bar"] = []
    qv_col = "quote_volume" if "quote_volume" in df.columns else None
    # Columns already represented as first-class Bar fields / standard extras.
    _std_cols = {
        "open", "high", "low", "close", "volume", "quote_volume",
        "open_time", "close_time", "trades", "instrument_id", "timeframe",
        timestamp_col,
    }
    _extra_numeric_cols = [c for c in df.columns if c not in _std_cols]
    for _, row in df.iterrows():
        extras = {}
        if "open_time" in df.columns:
            extras["open_time"] = int(row["open_time"])
        if "close_time" in df.columns:
            extras["close_time"] = int(row["close_time"])
        if "trades" in df.columns:
            extras["trades"] = int(row["trades"])
        # Carry any additional numeric columns (funding_rate, open_interest,
        # funding_rate_bps, oi_change_bps, ...) into extras so bars_to_df can
        # surface them again — keeps the df<->bars round-trip lossless for the
        # derivative features that T3-style make_signals strategies consume.
        for col in _extra_numeric_cols:
            val = row[col]
            if isinstance(val, (int, float, np.integer, np.floating)) and pd.notna(val):
                extras[col] = float(val)
        bars.append(
            Bar(
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                timestamp=int(row[timestamp_col]),
                instrument_id=instrument_id.upper(),
                timeframe=timeframe,
                confirmed=confirmed,
                quote_volume=(float(row[qv_col]) if qv_col else None),
                extras=extras,
            )
        )
    return bars


def bars_to_df(bars: List["Bar"]) -> pd.DataFrame:
    """Convert a list of ``Bar`` objects into a DataFrame indexed by ``close_time``.

    Columns: ``open, high, low, close, volume, quote_volume, open_time,
    close_time, trades, instrument_id, timeframe``.
    """
    if not bars:
        return pd.DataFrame(
            columns=[
                "open", "high", "low", "close", "volume", "quote_volume",
                "open_time", "close_time", "trades", "instrument_id", "timeframe",
            ]
        )
    rows = []
    for b in bars:
        row = {
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
            "quote_volume": b.quote_volume,
            "open_time": b.extras.get("open_time"),
            "close_time": b.extras.get("close_time", b.timestamp),
            "trades": b.extras.get("trades"),
            "instrument_id": b.instrument_id,
            "timeframe": b.timeframe,
        }
        # Spill any additional Bar.extras (funding_rate, open_interest,
        # funding_rate_bps, oi_change_bps, mark_price, liq notionals, ...)
        # into columns additively so make_signals(df) can consume derivative
        # features through the same snapshot->bars_to_df path as OHLCV.
        # setdefault protects the canonical open_time/close_time/trades above.
        for k, v in b.extras.items():
            row.setdefault(k, v)
        rows.append(row)
    return pd.DataFrame(rows)


def snapshot_to_df(
    snapshot: "DataSnapshot",
    instrument_id: str,
    timeframe: str,
) -> pd.DataFrame:
    """Extract the bars for ``(instrument_id, timeframe)`` from a snapshot."""
    from ..standard_bot.core import MarketBundle

    market = snapshot.require_market()
    key = MarketBundle.key(instrument_id, timeframe)
    bars = market.bars.get(key, [])
    return bars_to_df(bars)


# ---------------------------------------------------------------------------
# Binance public REST helpers
# ---------------------------------------------------------------------------

# Endpoint base URLs for Binance USDM (USDT-margined) Futures
_FAPI_BASE = "https://fapi.binance.com"
_FAPI_KLINES = f"{_FAPI_BASE}/fapi/v1/klines"
_FAPI_24H = f"{_FAPI_BASE}/fapi/v1/ticker/24hr"
_FAPI_FUNDING = f"{_FAPI_BASE}/fapi/v1/fundingRate"
_FAPI_PREMIUM = f"{_FAPI_BASE}/fapi/v1/premiumIndex"
_FAPI_EXINFO = f"{_FAPI_BASE}/fapi/v1/exchangeInfo"
_FAPI_OI = f"{_FAPI_BASE}/futures/data/openInterestHist"
_FAPI_TOPLS_ACC = f"{_FAPI_BASE}/futures/data/topLongShortAccountRatio"
_FAPI_TOPLS_POS = f"{_FAPI_BASE}/futures/data/topLongShortPositionRatio"
_FAPI_GLOBALLS = f"{_FAPI_BASE}/futures/data/globalLongShortAccountRatio"
_FAPI_TAKER = f"{_FAPI_BASE}/futures/data/takerlongshortRatio"

_DEFAULT_TIMEOUT = 15.0
_DEFAULT_RETRIES = 3
_DEFAULT_USER_AGENT = "cyqnt-trd-blocks/0.1 (+https://pypi.org/project/cyqnt-trd/)"


def _request_json(url: str, params: Dict[str, Any]) -> Any:
    """GET *url* with *params*, retrying on transient errors."""
    import requests  # lazy import

    last_exc: Optional[BaseException] = None
    for attempt in range(_DEFAULT_RETRIES):
        try:
            resp = requests.get(
                url,
                params={k: v for k, v in params.items() if v is not None},
                timeout=_DEFAULT_TIMEOUT,
                headers={"User-Agent": _DEFAULT_USER_AGENT},
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:  # type: ignore[attr-defined]
            last_exc = exc
            if attempt < _DEFAULT_RETRIES - 1:
                time.sleep(0.5 * (2 ** attempt))
    raise RuntimeError(f"GET {url} failed after {_DEFAULT_RETRIES} attempts: {last_exc}")


# -- klines --


def fetch_klines(
    symbol: str,
    interval: str,
    *,
    limit: int = 500,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    market_type: str = "futures",
) -> pd.DataFrame:
    """Fetch klines from Binance public REST.

    Returns a DataFrame indexed sequentially with columns
    ``open_time, open, high, low, close, volume, close_time,
    quote_volume, trades, taker_buy_base, taker_buy_quote``.

    *market_type*: ``"futures"`` (USDM) or ``"spot"``.
    """
    if market_type not in ("futures", "spot"):
        raise ValueError(f"market_type must be 'futures' or 'spot', got {market_type!r}")
    if market_type == "futures":
        url = _FAPI_KLINES
    else:
        url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": min(int(limit), 1500),
        "startTime": start_ms,
        "endTime": end_ms,
    }
    raw = _request_json(url, params)
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    if df.empty:
        return df.drop(columns=["ignore"])
    numeric = ["open", "high", "low", "close", "volume", "quote_volume",
               "taker_buy_base", "taker_buy_quote"]
    df[numeric] = df[numeric].astype(float)
    df[["open_time", "close_time", "trades"]] = df[
        ["open_time", "close_time", "trades"]
    ].astype("int64")
    return df.drop(columns=["ignore"])


# -- open interest --


def fetch_oi(
    symbol: str,
    period: str = "5m",
    *,
    limit: int = 500,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch historical open-interest series for a USDM-futures symbol.

    Period is one of ``5m | 15m | 30m | 1h | 2h | 4h | 6h | 12h | 1d``.

    Returns a DataFrame with columns ``timestamp, oi, oi_value``.
    """
    raw = _request_json(
        _FAPI_OI,
        {
            "symbol": symbol.upper(),
            "period": period,
            "limit": min(int(limit), 500),
            "startTime": start_ms,
            "endTime": end_ms,
        },
    )
    rows: List[Dict[str, Any]] = []
    for r in raw:
        rows.append(
            {
                "timestamp": int(r["timestamp"]),
                "oi": float(r["sumOpenInterest"]),
                "oi_value": float(r["sumOpenInterestValue"]),
            }
        )
    return pd.DataFrame(rows)


# -- funding rate --


def fetch_funding_rate(
    symbol: str,
    *,
    limit: int = 1000,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch the historical funding-rate series.

    Returns a DataFrame with columns ``timestamp, funding_rate``.
    Funding rate is a fraction (e.g. ``0.0001`` = 1 bp).
    """
    raw = _request_json(
        _FAPI_FUNDING,
        {
            "symbol": symbol.upper(),
            "limit": min(int(limit), 1000),
            "startTime": start_ms,
            "endTime": end_ms,
        },
    )
    rows = []
    for r in raw:
        rows.append(
            {
                "timestamp": int(r["fundingTime"]),
                "funding_rate": float(r["fundingRate"]),
            }
        )
    return pd.DataFrame(rows)


# -- long/short ratio --


def fetch_long_short_ratio(
    symbol: str,
    period: str = "5m",
    mode: str = "top_account",
    *,
    limit: int = 500,
) -> pd.DataFrame:
    """Long/short ratio of top traders (account or position) or global retail.

    *mode*:

    * ``"top_account"`` — top traders' long/short *account* ratio
    * ``"top_position"`` — top traders' long/short *position* ratio
    * ``"global"``     — global retail account ratio

    Returns a DataFrame with columns ``timestamp, long_short_ratio,
    long_account, short_account``.
    """
    if mode == "top_account":
        url = _FAPI_TOPLS_ACC
    elif mode == "top_position":
        url = _FAPI_TOPLS_POS
    elif mode == "global":
        url = _FAPI_GLOBALLS
    else:
        raise ValueError(f"mode must be one of top_account / top_position / global, got {mode!r}")
    raw = _request_json(
        url,
        {"symbol": symbol.upper(), "period": period, "limit": min(int(limit), 500)},
    )
    rows = []
    for r in raw:
        rows.append(
            {
                "timestamp": int(r["timestamp"]),
                "long_short_ratio": float(r["longShortRatio"]),
                "long_account": float(r.get("longAccount", r.get("longPosition", "nan"))),
                "short_account": float(r.get("shortAccount", r.get("shortPosition", "nan"))),
            }
        )
    return pd.DataFrame(rows)


# -- taker buy/sell volume --


def fetch_taker_buy_sell_ratio(
    symbol: str,
    period: str = "5m",
    *,
    limit: int = 500,
) -> pd.DataFrame:
    """Taker buy/sell volume ratio.

    Returns a DataFrame with columns ``timestamp, buy_sell_ratio,
    buy_volume, sell_volume``.
    """
    raw = _request_json(
        _FAPI_TAKER,
        {"symbol": symbol.upper(), "period": period, "limit": min(int(limit), 500)},
    )
    rows = []
    for r in raw:
        rows.append(
            {
                "timestamp": int(r["timestamp"]),
                "buy_sell_ratio": float(r["buySellRatio"]),
                "buy_volume": float(r["buyVol"]),
                "sell_volume": float(r["sellVol"]),
            }
        )
    return pd.DataFrame(rows)


# -- 24h tickers (for universe scanning) --


def fetch_24h_tickers(market_type: str = "futures") -> pd.DataFrame:
    """Fetch the full 24h ticker snapshot for all symbols.

    Useful for building dynamic universes (top gainers, volume filter, etc.)
    """
    if market_type == "futures":
        url = _FAPI_24H
    elif market_type == "spot":
        url = "https://api.binance.com/api/v3/ticker/24hr"
    else:
        raise ValueError(f"market_type must be 'futures' or 'spot', got {market_type!r}")
    raw = _request_json(url, {})
    if not isinstance(raw, list):
        raise RuntimeError(f"unexpected response from {url}: {type(raw).__name__}")
    df = pd.DataFrame(raw)
    if df.empty:
        return df
    numeric = ["priceChange", "priceChangePercent", "weightedAvgPrice",
               "lastPrice", "openPrice", "highPrice", "lowPrice",
               "volume", "quoteVolume"]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_premium_index(symbol: Optional[str] = None) -> pd.DataFrame:
    """Mark-price / index-price / current funding-rate snapshot.

    If *symbol* is None, returns the snapshot for *all* USDM symbols.
    """
    raw = _request_json(_FAPI_PREMIUM, {"symbol": symbol.upper()} if symbol else {})
    if not isinstance(raw, list):
        raw = [raw]
    df = pd.DataFrame(raw)
    if df.empty:
        return df
    numeric = ["markPrice", "indexPrice", "lastFundingRate", "interestRate", "estimatedSettlePrice"]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_exchange_info(market_type: str = "futures") -> Dict[str, Any]:
    """Symbol metadata (lot size, min notional, …)."""
    if market_type == "futures":
        url = _FAPI_EXINFO
    elif market_type == "spot":
        url = "https://api.binance.com/api/v3/exchangeInfo"
    else:
        raise ValueError(f"market_type must be 'futures' or 'spot', got {market_type!r}")
    return _request_json(url, {})
