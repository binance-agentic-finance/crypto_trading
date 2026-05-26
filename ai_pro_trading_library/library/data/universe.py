"""Universe / target-pool management — L1 parity vs `cyqnt_trd.blocks.universe`.

The pure DataFrame filters and `UniverseFilter` builder are migrated
verbatim. Network-dependent helpers (`fetch_perpetual_universe`,
`augment_with_funding`) accept a `tickers_provider` / `funding_provider`
callable so the library stays free of concrete REST adapters; the legacy
no-arg form is unsupported in the new repo (use a real data adapter).
"""

from __future__ import annotations

from typing import Callable, Sequence

import pandas as pd


def filter_quote_volume(
    tickers: pd.DataFrame,
    min_quote_volume: float = 100_000_000.0,
) -> pd.DataFrame:
    if "quoteVolume" not in tickers.columns:
        raise ValueError("DataFrame missing 'quoteVolume' column")
    return tickers[tickers["quoteVolume"] >= float(min_quote_volume)].copy()


def filter_change_pct(
    tickers: pd.DataFrame,
    max_abs_pct: float = 100.0,
    min_pct: float | None = None,
) -> pd.DataFrame:
    if "priceChangePercent" not in tickers.columns:
        raise ValueError("DataFrame missing 'priceChangePercent' column")
    out = tickers[tickers["priceChangePercent"].abs() <= float(max_abs_pct)]
    if min_pct is not None:
        out = out[out["priceChangePercent"] >= float(min_pct)]
    return out.copy()


def filter_funding_rate(tickers: pd.DataFrame, max_abs_pct: float = 0.5) -> pd.DataFrame:
    """Requires the DataFrame to have a `fundingRatePct` column (use `augment_with_funding`)."""
    if "fundingRatePct" not in tickers.columns:
        raise ValueError(
            "DataFrame missing 'fundingRatePct' column — call augment_with_funding first"
        )
    return tickers[tickers["fundingRatePct"].abs() <= float(max_abs_pct)].copy()


def top_gainers(tickers: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if "priceChangePercent" not in tickers.columns:
        raise ValueError("DataFrame missing 'priceChangePercent' column")
    return tickers.nlargest(int(n), "priceChangePercent").copy()


def top_losers(tickers: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if "priceChangePercent" not in tickers.columns:
        raise ValueError("DataFrame missing 'priceChangePercent' column")
    return tickers.nsmallest(int(n), "priceChangePercent").copy()


def exclude_symbols(tickers: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    if "symbol" not in tickers.columns:
        raise ValueError("DataFrame missing 'symbol' column")
    drop_set = {s.upper() for s in symbols}
    return tickers[~tickers["symbol"].str.upper().isin(drop_set)].copy()


def only_symbols(tickers: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    if "symbol" not in tickers.columns:
        raise ValueError("DataFrame missing 'symbol' column")
    keep_set = {s.upper() for s in symbols}
    return tickers[tickers["symbol"].str.upper().isin(keep_set)].copy()


def fetch_perpetual_universe(
    tickers_provider: Callable[[str], pd.DataFrame],
    market_type: str = "futures",
) -> pd.DataFrame:
    """Get the 24h ticker snapshot through a caller-supplied provider.

    Intentional divergence from blocks: the new repo does not ship a
    network-bound default. Pass `tickers_provider=binance_adapter.fetch_24h`
    or any callable returning a DataFrame with a `symbol` column.
    """
    df = tickers_provider(market_type)
    if df.empty:
        return df
    if "symbol" not in df.columns:
        raise RuntimeError("ticker provider missing 'symbol' column")
    return df.copy()


def augment_with_funding(
    tickers: pd.DataFrame,
    funding_provider: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """Add `fundingRatePct` column via caller-supplied provider."""
    premium = funding_provider()
    if premium.empty or "symbol" not in premium.columns:
        out = tickers.copy()
        out["fundingRatePct"] = float("nan")
        return out
    fr = premium[["symbol", "lastFundingRate"]].copy()
    fr["fundingRatePct"] = fr["lastFundingRate"].astype(float) * 100.0
    return tickers.merge(fr[["symbol", "fundingRatePct"]], on="symbol", how="left")


class UniverseFilter:
    """Fluent builder for chained universe filters."""

    def __init__(self, tickers: pd.DataFrame) -> None:
        self.df = tickers.copy()

    def filter_quote_volume(self, min_quote_volume: float = 100_000_000.0) -> "UniverseFilter":
        self.df = filter_quote_volume(self.df, min_quote_volume)
        return self

    def filter_change_pct(
        self,
        max_abs_pct: float = 100.0,
        min_pct: float | None = None,
    ) -> "UniverseFilter":
        self.df = filter_change_pct(self.df, max_abs_pct, min_pct)
        return self

    def with_funding(self, funding_provider: Callable[[], pd.DataFrame]) -> "UniverseFilter":
        self.df = augment_with_funding(self.df, funding_provider)
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
        if "symbol" not in self.df.columns:
            raise ValueError("DataFrame missing 'symbol' column")
        self.df = self.df[self.df["symbol"].str.upper().str.endswith(suffix.upper())].copy()
        return self

    def symbols(self) -> list[str]:
        if "symbol" not in self.df.columns:
            raise ValueError("DataFrame missing 'symbol' column")
        return [str(s).upper() for s in self.df["symbol"].tolist()]

    def to_frame(self) -> pd.DataFrame:
        return self.df.copy()


__all__ = [
    "UniverseFilter",
    "augment_with_funding",
    "exclude_symbols",
    "fetch_perpetual_universe",
    "filter_change_pct",
    "filter_funding_rate",
    "filter_quote_volume",
    "only_symbols",
    "top_gainers",
    "top_losers",
]
