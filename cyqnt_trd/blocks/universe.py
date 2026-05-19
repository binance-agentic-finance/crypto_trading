"""Universe / target-pool management.

Helpers for building a dynamic list of symbols to scan, with the filter
chain that the user dataset most often asks for:

* perpetual-futures only
* 24h volume above N USDT
* 24h change within ±X%
* funding rate within ±Y%
* top-N gainers / losers
* explicit blacklist / whitelist

Examples
--------
>>> from cyqnt_trd.blocks import universe
>>> tickers = universe.fetch_perpetual_universe()
>>> selected = (
...     universe.UniverseFilter(tickers)
...     .filter_quote_volume(min_quote_volume=100_000_000)
...     .filter_change_pct(max_abs_pct=1.0)
...     .top_gainers(n=10)
...     .symbols()
... )
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import pandas as pd

from . import data as _data

__all__ = [
    "UniverseFilter",
    "fetch_perpetual_universe",
    "filter_quote_volume",
    "filter_change_pct",
    "filter_funding_rate",
    "top_gainers",
    "top_losers",
    "exclude_symbols",
    "only_symbols",
]


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------


def fetch_perpetual_universe(market_type: str = "futures") -> pd.DataFrame:
    """Return the 24h ticker snapshot for all symbols, plus a derived
    column ``symbol`` (the canonical key)."""
    df = _data.fetch_24h_tickers(market_type=market_type)
    if df.empty:
        return df
    if "symbol" not in df.columns:
        raise RuntimeError("Binance ticker response missing 'symbol' column")
    return df.copy()


def filter_quote_volume(
    tickers: pd.DataFrame, min_quote_volume: float = 100_000_000.0
) -> pd.DataFrame:
    """Keep symbols with 24h quote volume >= *min_quote_volume* (USDT)."""
    if "quoteVolume" not in tickers.columns:
        raise ValueError("DataFrame missing 'quoteVolume' column")
    return tickers[tickers["quoteVolume"] >= float(min_quote_volume)].copy()


def filter_change_pct(
    tickers: pd.DataFrame, max_abs_pct: float = 100.0, min_pct: Optional[float] = None
) -> pd.DataFrame:
    """Keep symbols with 24h pct-change within ``[min_pct, max_abs_pct]`` and ``|change| <= max_abs_pct``."""
    if "priceChangePercent" not in tickers.columns:
        raise ValueError("DataFrame missing 'priceChangePercent' column")
    out = tickers[tickers["priceChangePercent"].abs() <= float(max_abs_pct)]
    if min_pct is not None:
        out = out[out["priceChangePercent"] >= float(min_pct)]
    return out.copy()


def filter_funding_rate(
    tickers: pd.DataFrame, max_abs_pct: float = 0.5
) -> pd.DataFrame:
    """Keep symbols whose funding-rate (%) absolute value is <= *max_abs_pct*.

    Requires the DataFrame to be augmented with a ``fundingRatePct``
    column (use :func:`augment_with_funding`).
    """
    if "fundingRatePct" not in tickers.columns:
        raise ValueError("DataFrame missing 'fundingRatePct' column — call augment_with_funding first")
    return tickers[tickers["fundingRatePct"].abs() <= float(max_abs_pct)].copy()


def top_gainers(tickers: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top *n* symbols by 24h pct-change descending."""
    if "priceChangePercent" not in tickers.columns:
        raise ValueError("DataFrame missing 'priceChangePercent' column")
    return tickers.nlargest(int(n), "priceChangePercent").copy()


def top_losers(tickers: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top *n* symbols by 24h pct-change ascending (biggest losers)."""
    if "priceChangePercent" not in tickers.columns:
        raise ValueError("DataFrame missing 'priceChangePercent' column")
    return tickers.nsmallest(int(n), "priceChangePercent").copy()


def exclude_symbols(tickers: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    """Drop the given symbols from the universe."""
    if "symbol" not in tickers.columns:
        raise ValueError("DataFrame missing 'symbol' column")
    drop_set = {s.upper() for s in symbols}
    return tickers[~tickers["symbol"].str.upper().isin(drop_set)].copy()


def only_symbols(tickers: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    """Keep only the given symbols."""
    if "symbol" not in tickers.columns:
        raise ValueError("DataFrame missing 'symbol' column")
    keep_set = {s.upper() for s in symbols}
    return tickers[tickers["symbol"].str.upper().isin(keep_set)].copy()


def augment_with_funding(tickers: pd.DataFrame) -> pd.DataFrame:
    """Augment the universe with the latest funding rate as ``fundingRatePct``."""
    premium = _data.fetch_premium_index()
    if premium.empty or "symbol" not in premium.columns:
        tickers = tickers.copy()
        tickers["fundingRatePct"] = float("nan")
        return tickers
    fr = premium[["symbol", "lastFundingRate"]].copy()
    fr["fundingRatePct"] = fr["lastFundingRate"].astype(float) * 100.0
    return tickers.merge(fr[["symbol", "fundingRatePct"]], on="symbol", how="left")


# ---------------------------------------------------------------------------
# Fluent filter builder
# ---------------------------------------------------------------------------


class UniverseFilter:
    """Fluent builder for chained universe filters.

    All methods return ``self`` so users can chain calls without
    re-assigning variables.
    """

    def __init__(self, tickers: pd.DataFrame) -> None:
        self.df = tickers.copy()

    def filter_quote_volume(self, min_quote_volume: float = 100_000_000.0) -> "UniverseFilter":
        self.df = filter_quote_volume(self.df, min_quote_volume)
        return self

    def filter_change_pct(
        self, max_abs_pct: float = 100.0, min_pct: Optional[float] = None
    ) -> "UniverseFilter":
        self.df = filter_change_pct(self.df, max_abs_pct, min_pct)
        return self

    def with_funding(self) -> "UniverseFilter":
        self.df = augment_with_funding(self.df)
        return self

    def filter_funding_rate(self, max_abs_pct: float = 0.5) -> "UniverseFilter":
        self.df = filter_funding_rate(self.df, max_abs_pct)
        return self

    def top_gainers(self, n: int = 10) -> "UniverseFilter":
        self.df = top_gainers(self.df, n)
        return self

    def top_losers(self, n: int = 10) -> "UniverseFilter":
        self.df = top_losers(self.df, n)
        return self

    def exclude(self, symbols: Sequence[str]) -> "UniverseFilter":
        self.df = exclude_symbols(self.df, symbols)
        return self

    def only(self, symbols: Sequence[str]) -> "UniverseFilter":
        self.df = only_symbols(self.df, symbols)
        return self

    def filter_quote_suffix(self, suffix: str = "USDT") -> "UniverseFilter":
        """Keep symbols whose name ends with *suffix* (e.g. only USDT pairs)."""
        if "symbol" not in self.df.columns:
            raise ValueError("DataFrame missing 'symbol' column")
        self.df = self.df[self.df["symbol"].str.upper().str.endswith(suffix.upper())].copy()
        return self

    def symbols(self) -> List[str]:
        if "symbol" not in self.df.columns:
            raise ValueError("DataFrame missing 'symbol' column")
        return [str(s).upper() for s in self.df["symbol"].tolist()]

    def to_frame(self) -> pd.DataFrame:
        return self.df.copy()
