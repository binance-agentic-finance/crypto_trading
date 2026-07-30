"""Ready-to-use PUBLIC data source specs (real endpoints, backtestable).

These are :class:`RestSourceSpec` definitions for the **publicly reachable**
data sources that the analysed user bots most often relied on — expressed
declaratively on top of :mod:`rest_source`. Every URL here is a public endpoint
(Binance ``fapi`` USDⓈ-M futures-data + ``alternative.me``); registering them
gives an out-of-the-box, backtestable fetch with no credentials and no internal
network.

Coverage vs. observed user demand
---------------------------------
Mapped to the demand ranking from the 507 explicit data-need conversations:

    funding_rate   38.5% of data chats   ✅ deep public history
    open_interest  23.7%                  🟡 ~30d public window
    ls_account     11.6%                  🟡 ~30d
    ls_top_pos     (crowding view)        🟡 ~30d
    taker_ratio     6.5%                  🟡 ~30d
    fear_greed      (regime switch)       ✅ deep daily

Backtestability tiers mirror the project's data discipline:

* ``BACKTESTABLE`` — deep replayable history (funding, fear&greed).
* ``SEMI`` — windowed (~30 days) public history; fine for gates / limited
  backtests, not long-horizon claims (OI, long-short, taker).

What is deliberately NOT here
-----------------------------
Sources that depend on non-public / deployment-specific endpoints (ETF flow,
whale movement, sector rotation, social rank, screener, next-funding, on-chain,
etc.) are **not defined in this repo**. Point ``fetch_rest`` at them through a
private, git-ignored config loaded with :func:`rest_source.load_specs`
(``$CYQNT_SOURCES_CONFIG``) — see ``docs/CONFIGURABLE_REST_SOURCES.md``.

Premium/basis is array-shaped (same as klines) and is better served by the
existing kline fetcher, so it is intentionally not a dict-row spec here.

Usage
-----
>>> from cyqnt_trd.data_cli import register_public_sources, fetch_rest
>>> register_public_sources()                       # register all of the below
>>> df = fetch_rest("funding_rate", params={"symbol": "ETHUSDT", "limit": 1000})
>>> fg = fetch_rest("fear_greed", params={"limit": 0})   # 0 = full history
"""

from __future__ import annotations

from typing import Dict, List

from .rest_source import RestSourceSpec, FieldSpec, register_spec

__all__ = [
    "PUBLIC_SOURCES",
    "register_public_sources",
    "BACKTESTABLE",
    "SEMI",
]

#: Backtestability tier tags (documentation; also surfaced on each spec name).
BACKTESTABLE = "BACKTESTABLE"
SEMI = "SEMI"

_FAPI = "https://fapi.binance.com"
_ALT = "https://api.alternative.me"


#: Registered, ready-to-fetch public specs. Override ``symbol`` / ``period`` /
#: ``limit`` per call via ``fetch_rest(..., params={...})``.
PUBLIC_SOURCES: Dict[str, RestSourceSpec] = {
    # --- BACKTESTABLE (deep public history) --------------------------------
    "funding_rate": RestSourceSpec(
        name="funding_rate",
        base_url=_FAPI,
        path="fapi/v1/fundingRate",
        params={"symbol": "BTCUSDT", "limit": 1000},
        data_path="",  # top-level JSON array
        fields={
            "symbol": FieldSpec("symbol", "str"),
            "timestamp": FieldSpec("fundingTime", "ms"),
            "funding_rate": FieldSpec("fundingRate", "float"),
            "mark_price": FieldSpec("markPrice", "float"),
        },
        sort_by="timestamp",
        ttl=300,
    ),
    "fear_greed": RestSourceSpec(
        name="fear_greed",
        base_url=_ALT,
        path="fng/",
        params={"limit": 30, "format": "json"},
        data_path="data",
        fields={
            "timestamp": FieldSpec("timestamp", "ms"),
            "value": FieldSpec("value", "int"),
            "value_classification": FieldSpec("value_classification", "str"),
        },
        sort_by="timestamp",
        ttl=3600,
    ),

    # --- SEMI (~30d public window) -----------------------------------------
    "open_interest": RestSourceSpec(
        name="open_interest",
        base_url=_FAPI,
        path="futures/data/openInterestHist",
        params={"symbol": "BTCUSDT", "period": "1h", "limit": 500},
        data_path="",
        fields={
            "symbol": FieldSpec("symbol", "str"),
            "timestamp": FieldSpec("timestamp", "ms"),
            "oi_base": FieldSpec("sumOpenInterest", "float"),
            "oi_value": FieldSpec("sumOpenInterestValue", "float"),
        },
        sort_by="timestamp",
        ttl=300,
    ),
    "ls_account": RestSourceSpec(
        name="ls_account",
        base_url=_FAPI,
        path="futures/data/globalLongShortAccountRatio",
        params={"symbol": "BTCUSDT", "period": "1h", "limit": 500},
        data_path="",
        fields={
            "symbol": FieldSpec("symbol", "str"),
            "timestamp": FieldSpec("timestamp", "ms"),
            "long_account": FieldSpec("longAccount", "float"),
            "short_account": FieldSpec("shortAccount", "float"),
            "long_short_ratio": FieldSpec("longShortRatio", "float"),
        },
        sort_by="timestamp",
        ttl=300,
    ),
    "ls_top_position": RestSourceSpec(
        name="ls_top_position",
        base_url=_FAPI,
        path="futures/data/topLongShortPositionRatio",
        params={"symbol": "BTCUSDT", "period": "1h", "limit": 500},
        data_path="",
        fields={
            "symbol": FieldSpec("symbol", "str"),
            "timestamp": FieldSpec("timestamp", "ms"),
            "long_account": FieldSpec("longAccount", "float"),
            "short_account": FieldSpec("shortAccount", "float"),
            "long_short_ratio": FieldSpec("longShortRatio", "float"),
        },
        sort_by="timestamp",
        ttl=300,
    ),
    "taker_ratio": RestSourceSpec(
        name="taker_ratio",
        base_url=_FAPI,
        path="futures/data/takerlongshortRatio",
        params={"symbol": "BTCUSDT", "period": "1h", "limit": 500},
        data_path="",
        fields={
            "timestamp": FieldSpec("timestamp", "ms"),
            "buy_sell_ratio": FieldSpec("buySellRatio", "float"),
            "buy_vol": FieldSpec("buyVol", "float"),
            "sell_vol": FieldSpec("sellVol", "float"),
        },
        sort_by="timestamp",
        ttl=300,
    ),
}

#: Backtestability tier per source (documentation / gate wiring).
SOURCE_TIERS: Dict[str, str] = {
    "funding_rate": BACKTESTABLE,
    "fear_greed": BACKTESTABLE,
    "open_interest": SEMI,
    "ls_account": SEMI,
    "ls_top_position": SEMI,
    "taker_ratio": SEMI,
}


def register_public_sources() -> List[str]:
    """Register every spec in :data:`PUBLIC_SOURCES`; return the names.

    Idempotent — safe to call at import time or on each process start.
    """
    for spec in PUBLIC_SOURCES.values():
        register_spec(spec)
    return sorted(PUBLIC_SOURCES)
