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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

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
    "fetch_contract_meta",
    "fetch_premium_index",
    "fetch_funding_info",
    "fetch_book_ticker_cross_section",
    "fetch_open_interest_cross_section",
    "fetch_oi_history_cross_section",
    "fetch_long_short_ratio_cross_section",
    "fetch_klines_cross_section",
    "kline_request_weight",
    "FAN_OUT_MAX_SYMBOLS",
    "KLINE_FAN_OUT_MAX_WEIGHT",
]

#: Columns :func:`fetch_contract_meta` always returns, in vendor spelling.
#:
#: Declared rather than inferred from the response so an endpoint that stops
#: sending one of them is a loud ``KeyError``-shaped failure at the fetcher,
#: not a NaN column that a downstream sector filter reads as "this coin has no
#: sector" and silently drops.
_CONTRACT_META_FIELDS = (
    "symbol", "contractType", "underlyingType", "underlyingSubType",
    "baseAsset", "quoteAsset", "status",
)

#: Most instruments one ``fetch_*_cross_section`` call may fan out over.
#:
#: These three fields have **no all-market endpoint**: omitting ``symbol`` from
#: ``/fapi/v1/openInterest`` or ``/futures/data/globalLongShortAccountRatio``
#: answers HTTP 400 (measured), so a cross-section can only be built one request
#: per instrument. That makes the roster size a cost, and an unbounded roster a
#: way to spend a minute's rate budget on one screen.
#:
#: Why 200. Binance meters the two paths separately: ``/fapi/v1/*`` against
#: 2400 request-weight per minute per IP, and the ``/futures/data/*`` group
#: against 1000 requests per 5 minutes. Two of the three fan-outs here live in
#: the second group, so a roster of N costs ``2N`` of that 1000 — 400 at the cap,
#: 40 %, which leaves the same box room to run a second screen inside the same
#: window. The ``/fapi/v1/openInterest`` leg is weight 1, so N=200 is 8 % of the
#: minute's weight. Measured: a 127-symbol roster ran 176 calls in 9 s.
#:
#: Exceeding it **raises**; it is never truncated. Truncation would silently
#: change which instruments can be selected — the ones cut are the tail of
#: whatever order the roster happened to arrive in — and no field of the output
#: would record that the screen saw a smaller market than it says it did. The
#: fix is upstream and free: narrow the cross-section first (``filter_sub_type``,
#: ``filter_crypto_only``, ``filter_quote_volume`` are all pure column filters
#: over a table already in hand) and fan out over what survives.
FAN_OUT_MAX_SYMBOLS = 200


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
# The CURRENT reading, one symbol at a time. Not the same endpoint as the line
# above and not interchangeable with it: this one is live and exact but
# coin-denominated only, while openInterestHist is bucketed (5m at the finest)
# and already carries the venue's own dollar conversion.
_FAPI_OI_NOW = f"{_FAPI_BASE}/fapi/v1/openInterest"
_FAPI_BOOK_TICKER = f"{_FAPI_BASE}/fapi/v1/ticker/bookTicker"
_FAPI_FUNDING_INFO = f"{_FAPI_BASE}/fapi/v1/fundingInfo"
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


def fetch_contract_meta(market_type: str = "futures") -> pd.DataFrame:
    """One row per listed contract: *what the instrument is*, not how it traded.

    ``exchangeInfo`` already answers a question no other source here can — is
    this perpetual a crypto coin, a tokenised US equity, a synthetic index?
    which sector tags does the venue give it? — and the repo was already fetching
    it and throwing the answer away: ``data_cli.scanner.full_market_scan`` keeps
    only ``symbol``.

    Columns are :data:`_CONTRACT_META_FIELDS`, in Binance's own spelling:

    ``symbol``             the instrument (``BTCUSDT``)
    ``contractType``       ``PERPETUAL`` / ``TRADIFI_PERPETUAL`` / ``CURRENT_QUARTER`` …
    ``underlyingType``     ``COIN`` / ``EQUITY`` / ``COMMODITY`` / ``INDEX`` / ``HK_EQUITY`` …
    ``underlyingSubType``  **a list** of sector tags (``['Alpha', 'DeFi']``)
    ``baseAsset`` / ``quoteAsset``   the two legs
    ``status``             ``TRADING`` / ``SETTLING`` / ``PENDING_TRADING``

    **Every symbol is returned, unfiltered.** That is a deliberate difference
    from ``full_market_scan``, which keeps only ``status == "TRADING" and
    quoteAsset == "USDT"``. Copying that filter here would hand back a table
    missing every USDC-quoted pair, and the join in
    ``universe.augment_with_contract_meta`` would then give those pairs a NaN
    ``underlying_type`` — so ``filter_crypto_only`` would drop them for *being
    unknown* while appearing to drop them for *not being crypto*. Right answer,
    wrong reason, and it hides whether a by-name exclusion list is still needed.
    ``status`` and ``quoteAsset`` are handed over as columns instead, so a YAML
    spec does the filtering where it can be read.

    ``underlyingSubType`` is left as the list the API sends. Flattening happens
    at the join boundary (:func:`universe.augment_with_contract_meta`), which is
    the only place that knows it is about to become a scalar cell in a
    cross-section.
    """
    raw = fetch_exchange_info(market_type=market_type)
    symbols = raw.get("symbols") if isinstance(raw, dict) else None
    if not isinstance(symbols, list):
        raise RuntimeError(
            "exchangeInfo(%s) has no 'symbols' list (got %s); contract metadata "
            "cannot be built from this response"
            % (market_type, type(symbols).__name__))
    if not symbols:
        # An exchange with no listed contracts is not a market condition, it is a
        # broken response — and returning an empty frame here would surface two
        # steps later as "every symbol has unknown metadata".
        raise RuntimeError(
            "exchangeInfo(%s) listed no symbols; refusing to return an empty "
            "contract-metadata table, which downstream reads as 'no symbol has "
            "sector metadata'" % market_type)

    frame = pd.DataFrame(symbols)
    missing = [name for name in _CONTRACT_META_FIELDS if name not in frame.columns]
    if missing:
        raise RuntimeError(
            "exchangeInfo(%s) response is missing field(s) %s; it carries %s. "
            "Binance changed the schema — update _CONTRACT_META_FIELDS and the "
            "universe.augment_with_contract_meta alias table together."
            % (market_type, missing, sorted(frame.columns)))
    return frame[list(_CONTRACT_META_FIELDS)].copy()


#: Columns :func:`fetch_book_ticker_cross_section` returns, in vendor spelling.
#:
#: Declared rather than inferred, for the reason :data:`_CONTRACT_META_FIELDS`
#: gives: an endpoint that stops sending ``bidQty`` should fail here, loudly,
#: rather than one join later where the missing depth reads as "this instrument
#: has no size at the touch" — which is a reason to skip it.
_BOOK_TICKER_FIELDS = ("symbol", "bidPrice", "bidQty", "askPrice", "askQty", "time")

#: Columns :func:`fetch_funding_info` returns, in vendor spelling.
_FUNDING_INFO_FIELDS = ("symbol", "fundingIntervalHours",
                        "adjustedFundingRateCap", "adjustedFundingRateFloor")


def fetch_book_ticker_cross_section(market_type: str = "futures") -> pd.DataFrame:
    """Best bid/ask **for every symbol**, in ONE request. The real liquidity read.

    Columns are :data:`_BOOK_TICKER_FIELDS`, in Binance's own spelling:

    ``symbol``     the instrument
    ``bidPrice`` / ``bidQty``   best bid and the size resting on it
    ``askPrice`` / ``askQty``   best ask and the size resting on it
    ``time``       the venue's timestamp on the book reading

    Why this is not another turnover column
    ---------------------------------------
    ``quoteVolume`` is what a screen reaches for when the request is "drop the
    illiquid air coins", and it answers a different question: how much traded
    over 24 h, not whether an order can be filled now. The two disagree by orders
    of magnitude on the same market (measured 2026-08-02, all figures from one
    snapshot):

    ======================  ==============  ==========  ===================
    instrument              24h turnover    spread      size at the touch
    ======================  ==============  ==========  ===================
    BTCUSDT                 $3.36b           0.016 bps  $88,027
    SNXXUSDT                $111m           11.00 bps   $16,905
    TAKEUSDT                $24.5m           7.78 bps   $0.03
    ======================  ==============  ==========  ===================

    SNXXUSDT clears the ``min_quote_volume: 100000000`` floor that
    ``example_selection.yaml`` ships with, and its spread is 683x BTCUSDT's.
    TAKEUSDT turned over $24.5m and offers **three cents** at the best ask — a
    basket built on turnover alone puts both in front of an operator with nothing
    saying they cannot be entered. The median instrument on this venue sits at
    5.5 bps and 395 of 727 are above 5 bps, so this is the ordinary case and not
    a tail.

    Unlike open interest and the long/short ratio, this one costs nothing to take
    across the whole market: omitting ``symbol`` returns every instrument (727,
    measured) for a single call of weight 5, so there is **no fan-out** and no
    roster to plan. That is why it has no ``symbols`` parameter — a per-symbol
    variant would invite a 727-request version of a free read.

    Futures only, deliberately. The spot response
    (``api.binance.com/api/v3/ticker/bookTicker``) carries the same four
    price/quantity fields and **no** ``time``: a spread with no timestamp cannot
    be PIT-gated by the bundle assembler and cannot be checked for staleness, and
    stamping it with "now" would be inventing the one fact that makes it safe to
    replay. Refused rather than back-filled.
    """
    caller = "fetch_book_ticker_cross_section"
    if market_type != "futures":
        raise ValueError(
            "%s: market_type=%r is not supported. The spot venue does publish a "
            "bookTicker, but its response omits the `time` field the futures one "
            "sends, so a spot spread reading arrives with no timestamp — it could "
            "not be PIT-gated or checked for staleness, and stamping it with "
            "'now' would fabricate exactly the fact that makes a snapshot "
            "replayable. Screen the futures venue (market_type='futures'), or add "
            "the spot shape as its own catalog node with its own returns spec."
            % (caller, market_type))

    raw = _request_json(_FAPI_BOOK_TICKER, {})
    if not isinstance(raw, list):
        raise RuntimeError(
            "%s: %s answered %s, not the all-market list of book tops. Omitting "
            "`symbol` is what makes this one request instead of 727, so a "
            "non-list here means the endpoint's contract changed rather than that "
            "one instrument is missing."
            % (caller, _FAPI_BOOK_TICKER, type(raw).__name__))
    if not raw:
        # An exchange where no instrument has a quote is not a market condition,
        # it is a broken response — and an empty frame surfaces one join later as
        # "every instrument has an unknown spread", i.e. as a strict screen.
        raise RuntimeError(
            "%s: %s returned no rows. Refusing to hand back an empty book "
            "cross-section: downstream that reads as 'no instrument is quoted', "
            "which a max_spread_bps filter turns into an empty basket. Retry — a "
            "public Binance endpoint answering nothing is usually transient."
            % (caller, _FAPI_BOOK_TICKER))

    frame = pd.DataFrame(raw)
    missing = [name for name in _BOOK_TICKER_FIELDS if name not in frame.columns]
    if missing:
        raise RuntimeError(
            "%s: the bookTicker response is missing field(s) %s; it carries %s. "
            "Binance changed the schema — update _BOOK_TICKER_FIELDS and "
            "universe.augment_with_spread's alias table together."
            % (caller, missing, sorted(frame.columns)))
    # ``lastUpdateId`` is deliberately dropped: it is a per-symbol book revision
    # counter, monotonic within one instrument and meaningless across them, so it
    # would ride into every bundle as a large integer column that looks like data
    # and cannot be compared. ``time`` already answers the only question a reader
    # has about it (how old is this reading).
    out = frame[list(_BOOK_TICKER_FIELDS)].copy()
    for column in ("bidPrice", "bidQty", "askPrice", "askQty"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    return out


def fetch_funding_info(market_type: str = "futures") -> pd.DataFrame:
    """**How often** each perpetual settles funding, plus its rate caps.

    Columns are :data:`_FUNDING_INFO_FIELDS`, in Binance's own spelling:

    ``symbol``                  the instrument
    ``fundingIntervalHours``    hours between settlements — 8, 4 or 1 (measured)
    ``adjustedFundingRateCap`` / ``adjustedFundingRateFloor``
                                the venue's clamp on the rate, as a ratio

    Why the interval is not a detail
    -------------------------------
    ``lastFundingRate`` is *per settlement*, so two instruments showing the same
    0.01% are not paying the same carry: an 8-hourly one annualises to 10.95%
    and an hourly one to 87.6%, eight times more. Ranking the raw rate therefore
    orders the column by a number whose unit differs per row, and the basket that
    comes out looks entirely healthy.

    That is the ordinary case on this venue, not a corner: of 743 contracts
    (measured 2026-08-02) **443 settle 4-hourly**, 296 8-hourly and 4 hourly. So
    the majority of the market is mis-annualised by the 8h assumption that
    reading the raw rate implies, and the assumption is not even the plurality.

    One request for the whole market, weight 1, no fan-out. 743 rows covers 723
    of the 727 instruments in a futures 24h ticker snapshot; the four absentees
    are the dated delivery contracts (``BTCUSDT_260925`` and friends), which pay
    no funding at all. See :func:`cyqnt_trd.blocks.universe.augment_with_funding`
    for what a missing row becomes there — NaN, never an assumed 8.

    ``disclaimer`` and ``updateTime`` are dropped. ``updateTime`` in particular
    must not be promoted to the frame's event time: it is ``null`` for BTCUSDT
    and ETHUSDT (measured), so a bundle keyed on it would gate the two most
    important rows of the cross-section out on a PIT rule they never violated.
    """
    caller = "fetch_funding_info"
    if market_type != "futures":
        raise ValueError(
            "%s: funding is a futures-only mechanism and market_type=%r was "
            "requested. A spot market settles no funding, so there is no interval "
            "to read — either screen the futures venue "
            "(market_type='futures') or drop this step."
            % (caller, market_type))

    raw = _request_json(_FAPI_FUNDING_INFO, {})
    if not isinstance(raw, list):
        raise RuntimeError(
            "%s: %s answered %s instead of the all-market list of funding "
            "schedules." % (caller, _FAPI_FUNDING_INFO, type(raw).__name__))
    if not raw:
        raise RuntimeError(
            "%s: %s returned no rows. Refusing to hand back an empty schedule "
            "table: every downstream annualisation would become NaN, and a "
            "funding-carry screen over an all-NaN column returns an empty basket "
            "for a reason nothing in the output records. Retry."
            % (caller, _FAPI_FUNDING_INFO))

    frame = pd.DataFrame(raw)
    missing = [name for name in _FUNDING_INFO_FIELDS if name not in frame.columns]
    if missing:
        raise RuntimeError(
            "%s: the fundingInfo response is missing field(s) %s; it carries %s. "
            "Binance changed the schema — update _FUNDING_INFO_FIELDS and "
            "universe.augment_with_funding's alias table together. Do NOT paper "
            "over a missing fundingIntervalHours with a default of 8: that is the "
            "silent mis-ranking this node exists to remove."
            % (caller, missing, sorted(frame.columns)))
    out = frame[list(_FUNDING_INFO_FIELDS)].copy()
    for column in _FUNDING_INFO_FIELDS[1:]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


# ---------------------------------------------------------------------------
# Derivatives fan-out: one request per instrument, because there is no
# all-market endpoint for open interest or for long/short ratio
# ---------------------------------------------------------------------------


def _fan_out_roster(symbols: Any, *, caller: str) -> List[str]:
    """Normalise a fan-out roster, refusing an empty one and an oversized one.

    Order is preserved and duplicates are collapsed: a roster built by
    concatenating two filters can name the same instrument twice, and paying for
    the same request twice inside a capped budget is the one cost that buys
    nothing at all.

    An empty roster raises rather than returning an empty frame. The frame would
    then reach a universe block that joins it, produce an all-NaN column, and a
    ``min_notional_usd`` filter over an all-NaN column returns an empty basket —
    the same output a correctly-working strict screen gives, with nothing to tell
    the two apart. See :data:`FAN_OUT_MAX_SYMBOLS` for why the ceiling raises
    instead of truncating.
    """
    if isinstance(symbols, str):
        raise ValueError(
            "%s takes a list of instruments (['BTCUSDT', 'ETHUSDT']); a bare "
            "string would be fanned out one character at a time" % caller)
    if symbols is None:
        raise ValueError(
            "%s needs an explicit roster of instruments. There is no all-market "
            "endpoint for this field, so it cannot default to 'the whole "
            "universe' — that would be one request per listed contract (727 on a "
            "recent snapshot)." % caller)
    roster: List[str] = []
    for value in symbols:
        name = str(value).strip().upper()
        if not name:
            continue
        if name not in roster:
            roster.append(name)
    if not roster:
        raise ValueError(
            "%s was given no instruments to fan out over. An empty cross-section "
            "is not a market state here: it becomes an all-NaN column after the "
            "join, and a filter over that returns an empty basket with nothing to "
            "point at. Narrow the universe first, then pass the survivors."
            % caller)
    if len(roster) > FAN_OUT_MAX_SYMBOLS:
        raise ValueError(
            "%s was asked to fan out over %d instruments, above the %d ceiling. "
            "This field has no all-market endpoint, so the roster size IS the "
            "request count, and the roster is NOT truncated — cutting it would "
            "silently decide which instruments can be selected, and no field of "
            "the output would say so. Narrow the cross-section first with the "
            "free filters (universe.filter_sub_type / filter_crypto_only / "
            "filter_quote_volume all run on a table already in hand) and fan out "
            "over what survives; raise blocks.data.FAN_OUT_MAX_SYMBOLS only "
            "after re-checking the rate budget documented beside it."
            % (caller, len(roster), FAN_OUT_MAX_SYMBOLS))
    return roster


def _require_usdm(market_type: str, field: str, caller: str) -> None:
    """These three fields exist only on the USDⓈ-M futures venue.

    Raising rather than ignoring the argument: ``market_type="spot"`` reaching a
    futures endpoint would answer with real futures numbers for a spot screen,
    and "spot has no open interest" is a fact about the market that the caller
    has to see, not a parameter to drop on the floor.
    """
    if market_type != "futures":
        raise ValueError(
            "%s: %s is a futures-only field and market_type=%r was requested. "
            "A spot market has no perpetual position inventory and no perpetual "
            "long/short ratio, so there is nothing to fetch — either screen the "
            "futures venue (market_type='futures') or drop this step."
            % (caller, field, market_type))


def _mark_prices(roster: Sequence[str], *, caller: str) -> Dict[str, float]:
    """Mark price for every instrument in *roster*, in ONE all-market request.

    This is the one leg of the open-interest fan-out that does not have to fan
    out: ``/fapi/v1/premiumIndex`` returns every symbol for a single call, so the
    conversion price costs one request no matter how long the roster is.

    A roster entry that the venue does not price raises. It cannot be left NaN:
    the notional is ``oi_base * mark_price``, so one missing price becomes one
    missing dollar figure, which a ``min_notional_usd`` floor then drops for
    looking small rather than for being unknown.
    """
    frame = fetch_premium_index()
    if frame.empty or "symbol" not in frame.columns or "markPrice" not in frame.columns:
        raise RuntimeError(
            "%s: premiumIndex returned no usable mark-price snapshot (columns=%s). "
            "Open interest is quoted in COINS, so without a price there is no "
            "dollar figure to screen on."
            % (caller, list(frame.columns)))
    prices = {
        str(row.symbol).upper(): float(row.markPrice)
        for row in frame.itertuples()
        if pd.notna(row.markPrice)
    }
    absent = [symbol for symbol in roster if symbol not in prices]
    if absent:
        raise RuntimeError(
            "%s: the venue prices %d instruments and %d of the requested ones are "
            "not among them (e.g. %s). premiumIndex covers every listed perpetual, "
            "so this roster is stale — rebuild it from the same universe snapshot "
            "the fan-out is for."
            % (caller, len(prices), len(absent), absent[:5]))
    return prices


def fetch_open_interest_cross_section(
    symbols: Sequence[str],
    *,
    market_type: str = "futures",
) -> pd.DataFrame:
    """CURRENT open interest for each named instrument — in coins and in dollars.

    Columns, in Binance's own spelling plus the conversion inputs:

    ``symbol``         the instrument
    ``openInterest``   open positions **denominated in the base coin**
    ``markPrice``      the price used to convert it, read in the same collection
    ``time``           the venue's timestamp on the open-interest reading

    ``openInterest`` alone is not comparable across instruments and reading it
    as if it were is the mistake this signature exists to prevent: 2.79e9
    DOGEUSDT and 1.09e5 BTCUSDT are $193m and $6.8b. So the mark price travels
    with it and :func:`cyqnt_trd.blocks.universe.augment_with_open_interest`
    multiplies them into ``oi_notional_usd``.

    Why the price comes from here rather than from the funding frame
    ---------------------------------------------------------------
    ``markPrice`` is also present on the cross-sectional funding snapshot, so a
    spec *could* say ``with: [open_interest_snapshot, funding]``. It is fetched
    here instead, for two reasons. It survives in the funding frame only as an
    undeclared pass-through column — ``funding_snapshot`` declares
    ``value_columns=("funding_rate",)`` — so the moment anyone melts it into a
    metric row or normalisation stops carrying undeclared columns,
    ``oi_notional_usd`` becomes NaN and a ``>= $5m`` screen silently empties the
    basket. And it costs nothing: one call for the whole market, whatever the
    roster length. A dollar threshold should not depend on another node's
    accident.

    Cross-checked against the venue's own arithmetic: ``openInterest *
    markPrice`` reproduced ``openInterestHist``'s ``sumOpenInterestValue`` to
    within 0.25 % on BTCUSDT / ETHUSDT / DOGEUSDT (2026-08-02). The residual is
    the statistics endpoint's 5-minute bucketing, not a different definition.

    One request per instrument — see :data:`FAN_OUT_MAX_SYMBOLS`. A symbol that
    still fails after the transport's retries raises rather than being skipped:
    a hole here reappears as "this instrument had no open interest", which is a
    reason to *not select* it.
    """
    caller = "fetch_open_interest_cross_section"
    roster = _fan_out_roster(symbols, caller=caller)
    _require_usdm(market_type, "open interest", caller)

    prices = _mark_prices(roster, caller=caller)
    rows: List[Dict[str, Any]] = []
    for symbol in roster:
        payload = _request_json(_FAPI_OI_NOW, {"symbol": symbol})
        if not isinstance(payload, dict) or "openInterest" not in payload:
            raise RuntimeError(
                "%s: openInterest(%s) answered %r, which carries no "
                "'openInterest' field. Every other instrument in the roster is "
                "discarded with it — a partial cross-section would screen a "
                "market it does not name." % (caller, symbol, payload))
        rows.append({
            "symbol": symbol,
            "openInterest": float(payload["openInterest"]),
            "markPrice": prices[symbol],
            "time": int(payload.get("time") or 0),
        })
    return pd.DataFrame(rows, columns=["symbol", "openInterest", "markPrice", "time"])


#: Fields kept from an ``openInterestHist`` row, declared rather than inferred.
#:
#: ``CMCCirculatingSupply`` is deliberately dropped. It is a third-party token
#: supply figure riding along on a market-data endpoint: it is not open interest,
#: no block here reads it, and because the response is one row per period it
#: would be repeated on every row of a multi-day series for every instrument —
#: paid for in fixture bytes and in reader attention. A supply number belongs to
#: a fundamentals node that can state where it came from.
_OI_HIST_FIELDS = ("symbol", "sumOpenInterest", "sumOpenInterestValue", "timestamp")


def fetch_oi_history_cross_section(
    symbols: Sequence[str],
    *,
    period: str = "1d",
    limit: int = 8,
    market_type: str = "futures",
) -> pd.DataFrame:
    """Recent open-interest series for each named instrument, oldest first.

    Columns are :data:`_OI_HIST_FIELDS`: ``symbol``, ``sumOpenInterest`` (base
    coins), ``sumOpenInterestValue`` (USD, computed by the venue) and
    ``timestamp``. Both magnitudes are returned because they answer different
    questions and the difference is not small — see
    :func:`cyqnt_trd.blocks.universe.augment_with_oi_change`, which measures a
    change on each of them and hands both to the caller.

    ``limit`` must cover the baseline the change is measured against: a 7-day
    lookback needs 8 daily readings (the latest plus seven before it). The
    default pair (``period="1d"``, ``limit=8``) is exactly that.

    **The public endpoint keeps roughly 30 days.** A longer lookback than the
    window silently starts mid-series, so ``augment_with_oi_change`` refuses to
    compute a change for an instrument with too few readings rather than
    averaging whatever arrived.

    An instrument the venue returns *no* rows for is kept out of the frame
    without failing: a freshly listed perpetual genuinely has no week of
    history, and that is a market state ("read it, it was empty"), unlike a
    transport error, which raises. One request per instrument — see
    :data:`FAN_OUT_MAX_SYMBOLS`.
    """
    caller = "fetch_oi_history_cross_section"
    roster = _fan_out_roster(symbols, caller=caller)
    _require_usdm(market_type, "open-interest history", caller)

    rows: List[Dict[str, Any]] = []
    for symbol in roster:
        payload = _request_json(
            _FAPI_OI,
            {"symbol": symbol, "period": period, "limit": min(int(limit), 500)},
        )
        if not isinstance(payload, list):
            raise RuntimeError(
                "%s: openInterestHist(%s, period=%s) answered %r instead of a "
                "list of readings." % (caller, symbol, period, payload))
        for entry in payload:
            missing = [name for name in _OI_HIST_FIELDS if name not in entry]
            if missing:
                raise RuntimeError(
                    "%s: an openInterestHist(%s) row is missing %s; it carries "
                    "%s. Binance changed the schema — update _OI_HIST_FIELDS and "
                    "the universe.augment_with_oi_change alias table together."
                    % (caller, symbol, missing, sorted(entry)))
            rows.append({
                # The response echoes the symbol, but the request is the
                # authority on what was asked for, and normalising the case here
                # keeps the join key in one vocabulary.
                "symbol": symbol,
                "sumOpenInterest": float(entry["sumOpenInterest"]),
                "sumOpenInterestValue": float(entry["sumOpenInterestValue"]),
                "timestamp": int(entry["timestamp"]),
            })
    return pd.DataFrame(rows, columns=list(_OI_HIST_FIELDS))


#: ``mode`` -> (endpoint, what the numbers are about). Same three modes as the
#: per-symbol :func:`fetch_long_short_ratio`, so one vocabulary covers both.
_LS_MODES = {
    "global": (_FAPI_GLOBALLS, "every account on the venue (retail positioning)"),
    "top_account": (_FAPI_TOPLS_ACC, "the top traders, counted by account"),
    "top_position": (_FAPI_TOPLS_POS, "the top traders, weighted by position"),
}


def fetch_long_short_ratio_cross_section(
    symbols: Sequence[str],
    *,
    period: str = "1h",
    mode: str = "global",
    market_type: str = "futures",
) -> pd.DataFrame:
    """The latest long/short account ratio for each named instrument.

    Columns: ``symbol``, ``longAccount``, ``shortAccount``, ``longShortRatio``,
    ``timestamp``. ``longAccount`` / ``shortAccount`` are **fractions of 1** as
    the venue sends them (0.6728 = 67.28 %); the conversion to percentage points
    happens in :func:`cyqnt_trd.blocks.universe.augment_with_long_short_ratio`,
    which is also where the fraction is verified rather than assumed.

    *mode* selects whose positioning is measured — see :data:`_LS_MODES`. The
    default is ``global``, the whole venue, because "retail is leaning long" is
    the condition that actually turns up in selection requests; the top-trader
    variants measure a different and much smaller population and are not
    interchangeable with it.

    Exactly ONE reading per instrument is fetched and there is no ``limit``
    knob, unlike the per-symbol :func:`fetch_long_short_ratio`. A cross-section
    is one row per instrument at one as-of; several readings per instrument under
    that shape would be a claim about the frame's grain that is not true, and the
    ratio's own history is what the per-symbol node is for.

    One request per instrument — see :data:`FAN_OUT_MAX_SYMBOLS`.
    """
    caller = "fetch_long_short_ratio_cross_section"
    roster = _fan_out_roster(symbols, caller=caller)
    _require_usdm(market_type, "the long/short account ratio", caller)
    if mode not in _LS_MODES:
        raise ValueError(
            "%s: mode must be one of %s, got %r — they measure different "
            "populations (%s)"
            % (caller, sorted(_LS_MODES), mode,
               "; ".join("%s=%s" % (name, note)
                         for name, (_url, note) in sorted(_LS_MODES.items()))))
    url = _LS_MODES[mode][0]

    rows: List[Dict[str, Any]] = []
    for symbol in roster:
        payload = _request_json(url, {"symbol": symbol, "period": period, "limit": 1})
        if not isinstance(payload, list):
            raise RuntimeError(
                "%s: %s(%s) answered %r instead of a list of readings."
                % (caller, mode, symbol, payload))
        if not payload:
            # No reading for this instrument is a market state, not a failure:
            # the statistics series starts a while after a perpetual is listed.
            continue
        entry = payload[-1]
        rows.append({
            "symbol": symbol,
            "longAccount": float(entry["longAccount"]),
            "shortAccount": float(entry["shortAccount"]),
            "longShortRatio": float(entry["longShortRatio"]),
            "timestamp": int(entry["timestamp"]),
        })
    return pd.DataFrame(rows, columns=["symbol", "longAccount", "shortAccount",
                                       "longShortRatio", "timestamp"])


#: Request weight of ONE ``/fapi/v1/klines`` call, per ``limit`` band.
#:
#: ``(inclusive upper bound on limit, weight)``, first match wins. MEASURED on
#: 2026-08-02 by reading the ``X-MBX-USED-WEIGHT-1M`` response header before and
#: after bursts of six identical calls, because the cost of the plan below is the
#: whole reason a bars fan-out has to be ordered last and a guessed table would
#: put the ceiling in the wrong place:
#:
#: ===========  ======  =========================
#: limit        weight  measured deltas
#: ===========  ======  =========================
#: <= 100            1  1,1,1,1  (limit 90 / 100)
#: 101 - 500         2  2,2,2,2  (limit 200 / 500)
#: 501 - 1000        5  5,5,5,5  (limit 600)
#: 1001 - 1500      10  10,10,10 (limit 1200)
#: ===========  ======  =========================
_KLINE_WEIGHT_BANDS = ((100, 1), (500, 2), (1000, 5), (1500, 10))

#: Most request WEIGHT one bars fan-out may spend, across every (symbol,
#: timeframe) pair it is asked for.
#:
#: Weight and not a request count, because unlike the three derivative fan-outs
#: the per-call price here depends on ``limit`` — a 90-bar daily history and a
#: 500-bar 15-minute history are the same request and cost 1 and 2. Counting
#: requests would leave the expensive shape unbounded.
#:
#: Why 600. ``/fapi/v1/*`` is metered at 2400 request-weight per minute per IP,
#: and this is a quarter of it. That is enough for the plan a real screen runs —
#: measured, 5 survivors x 3 timeframes at limit 200 is 30 weight, 1.25 % of the
#: minute — while leaving the rest of the collection (a whole-market 24h ticker is
#: weight 40, ``bookTicker`` 5, ``premiumIndex`` 10) and a second screen in the
#: same window room to run.
#:
#: It is also the number that makes the ordering argument enforceable rather than
#: advisory. Fanning out over the venue BEFORE narrowing is 727 pairs per
#: timeframe: at limit 200 that is 1454 weight for one timeframe (61 % of the
#: minute) and 4362 for the three a resonance screen asks for — 182 %, which
#: cannot run at all. Narrowing first is free (the filters read a table already in
#: hand), so the cheap plan is always available and the expensive one raises.
#:
#: Exceeding it **raises and never truncates**, for the same reason as
#: :data:`FAN_OUT_MAX_SYMBOLS`: a truncated roster silently decides which
#: instruments can be selected, and a truncated TIMEFRAME set silently answers a
#: three-timeframe resonance question with two.
KLINE_FAN_OUT_MAX_WEIGHT = 600


def kline_request_weight(limit: int) -> int:
    """Binance's request weight for one ``klines`` call of *limit* bars."""
    limit = int(limit)
    if limit < 1:
        raise ValueError("limit must be >= 1, got %d" % limit)
    for bound, weight in _KLINE_WEIGHT_BANDS:
        if limit <= bound:
            return weight
    raise ValueError(
        "limit=%d is above the %d the endpoint serves; ask for fewer bars, or "
        "page — a request the venue rejects costs weight and returns nothing"
        % (limit, _KLINE_WEIGHT_BANDS[-1][0]))


def _bar_timeframes(timeframes: Any, *, caller: str) -> List[str]:
    """Normalise a timeframe list, refusing the two shapes that fan out wrongly.

    A bare string is refused rather than wrapped: ``timeframes="4h"`` would be
    iterated one CHARACTER at a time and the venue would reject ``"4"`` — four
    requests spent to learn that. Duplicates are collapsed because paying twice
    for the same series is the one cost that buys nothing.
    """
    if isinstance(timeframes, str):
        raise ValueError(
            "%s takes a LIST of timeframes (['4h', '1h', '15m']); a bare string "
            "would be fanned out one character at a time" % caller)
    if timeframes is None:
        raise ValueError(
            "%s needs an explicit list of timeframes. There is no default: a bars "
            "capture exists to answer a spec's own indicator steps, and guessing "
            "one would either miss the timeframe the spec asks for (every "
            "indicator then NaN) or pay for ones it never reads." % caller)
    out: List[str] = []
    for value in timeframes:
        name = str(value).strip()
        if not name:
            continue
        if name not in out:
            out.append(name)
    if not out:
        raise ValueError(
            "%s was given an EMPTY timeframe list. That is not 'collect nothing': "
            "it would land a bar frame with no rows, which the joining block reads "
            "as a failed capture and refuses." % caller)
    return out


def fetch_klines_cross_section(
    symbols: Sequence[str],
    *,
    timeframes: Sequence[str],
    limit: int = 200,
    end_ms: Optional[int] = None,
    market_type: str = "futures",
) -> pd.DataFrame:
    """OHLCV bars for every (instrument, timeframe) pair, in ONE long frame.

    Columns are :func:`fetch_klines`'s, plus the two that make the frame a
    cross-section: ``symbol`` and ``timeframe``. Rows are oldest-first within
    each pair.

    This is the frame a SELECTION strategy needs and had no way to get: the
    universe cross-section has one row per instrument and no bars, so "Supertrend
    bearish on 4h AND 1h AND 15m" or "up 100% from its 3-month low" could not be
    stated at all — the shipped substitution was ``top_losers(n=30)``, a 24h
    percentage wearing the words of an indicator.

    ``end_ms`` is why this goes through :func:`fetch_klines` rather than the
    ``klines`` catalog node
    ----------------------------------------------------------------------
    The ``klines`` node's fetcher is a ``binance-cli`` subprocess with no
    ``endTime`` parameter, so it can only answer "the last N bars as of now". A
    cross-section captured for a decision at time T must ask for bars as of T, or
    the frame silently mixes a universe snapshot from T with prices from now —
    which is not a small error for a screen whose whole subject is price history.
    So this calls the REST fetcher, which takes ``end_ms``, and a capture passes
    the bundle's own ``decision_time``.

    ``endTime`` is INCLUSIVE of the bar that contains it, so the last row of each
    series is normally an UNFINISHED candle whose high/low/close are still moving.
    That row is not dropped here: it carries its real ``close_time``, which is in
    the future relative to the decision, and the bundle's point-in-time gate
    (``live_bundle._regate`` / ``input_bundle._pit``) drops it for that reason.
    Dropping it here as well would hide the property in this function instead of
    leaving it visible in the artifact — and the artifact is what a reviewer can
    check.

    Cost, and why the roster has to be narrowed first: see
    :data:`KLINE_FAN_OUT_MAX_WEIGHT`. One request per pair, refused rather than
    truncated when the plan is too expensive.

    A pair that answers with no bars **raises**. A hole cannot be left out: the
    joining block would leave that instrument's indicator NaN, and NaN read
    downstream as "the condition did not hold" is a one-sided bias — the coin
    silently leaves a "bearish on three timeframes" screen but would also leave a
    "bullish on three timeframes" one.
    """
    caller = "fetch_klines_cross_section"
    roster = _fan_out_roster(symbols, caller=caller)
    frames = _bar_timeframes(timeframes, caller=caller)
    weight = len(roster) * len(frames) * kline_request_weight(limit)
    if weight > KLINE_FAN_OUT_MAX_WEIGHT:
        raise ValueError(
            "%s: %d instrument(s) x %d timeframe(s) at limit=%d costs %d request "
            "weight, above the %d ceiling (a quarter of the 2400/min the "
            "/fapi/v1 group allows one IP). The roster is NOT truncated — cutting "
            "it would silently decide which instruments can be selected, and "
            "cutting the timeframes would answer a three-timeframe resonance "
            "question with two. Narrow the cross-section FIRST with the free "
            "filters (universe.filter_sub_type / filter_crypto_only / "
            "filter_quote_volume all read a table already in hand), or ask for "
            "fewer bars: limit<=100 is weight 1 and 101-500 is weight 2."
            % (caller, len(roster), len(frames), int(limit), weight,
               KLINE_FAN_OUT_MAX_WEIGHT))

    collected: List[pd.DataFrame] = []
    for symbol in roster:
        for timeframe in frames:
            bars = fetch_klines(symbol, timeframe, limit=limit, end_ms=end_ms,
                                market_type=market_type)
            if bars is None or bars.empty:
                raise RuntimeError(
                    "%s: klines(%s, %s) came back with no bars. Every other pair "
                    "in the plan is discarded with it: an instrument missing from "
                    "the frame leaves its indicator NaN after the join, and a NaN "
                    "read as 'the condition did not hold' quietly removes that "
                    "coin from a bearish screen AND from a bullish one. Check the "
                    "symbol is listed on the %s venue at this end_ms (%s)."
                    % (caller, symbol, timeframe, market_type, end_ms))
            bars = bars.copy()
            bars.insert(0, "symbol", symbol)
            bars.insert(1, "timeframe", timeframe)
            collected.append(bars)
    return pd.concat(collected, ignore_index=True)
