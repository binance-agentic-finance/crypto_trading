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
    "augment_with_news",
    "top_mentioned",
    "top_bullish",
    "filter_sentiment",
]

# Columns added by :func:`augment_with_news`.
_NEWS_COLS = [
    "news_mention_rank", "news_mention_count", "news_unique_authors",
    "news_bullish_count", "news_bearish_count", "news_neutral_count",
    "news_bull_ratio",
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


def _with_symbol_column(tickers):
    """Return a copy that definitely has a ``symbol`` column.

    Universe tables reach these blocks two ways: straight off the Binance 24h
    ticker endpoint (``symbol``) or out of an input bundle, where
    ``RankFrame@1.0`` names it ``instrument_id``. Accepting only the former meant
    the canonical shape could not be fed to its own blocks.

    ``ticker`` is deliberately NOT accepted. In Square's frames that column is a
    BASE TOKEN (``BTC``), not an instrument, and every consumer downstream treats
    ``symbol`` as something a venue can fill. Aliasing it produced candidates
    named ``BTC`` with a fully-populated news_bull_ratio — no error anywhere,
    because the base-token join succeeds precisely *because* the wrong column was
    accepted — and a basket naming an instrument no exchange lists. It also makes
    de-duplication by base asset a silent no-op, since ``_base_token('BTC')`` is
    already ``'BTC'``.
    """
    tickers = tickers.copy()
    if "symbol" in tickers.columns:
        return tickers
    if "instrument_id" in tickers.columns:
        tickers["symbol"] = tickers["instrument_id"]
        return tickers
    if "ticker" in tickers.columns:
        raise ValueError(
            "this frame is keyed on 'ticker', which in a Square frame is a BASE "
            "TOKEN (BTC), not a tradable instrument (BTCUSDT). Join it onto a "
            "universe first — universe.augment_with_news does exactly that — "
            "rather than renaming it to 'symbol', which would put an unfillable "
            "instrument into the basket.")
    raise ValueError(
        "DataFrame missing 'symbol' / 'instrument_id' column; got %s"
        % list(tickers.columns))


def filter_quote_volume(
    tickers: pd.DataFrame, min_quote_volume: float = 100_000_000.0
) -> pd.DataFrame:
    """Keep symbols with 24h quote volume >= *min_quote_volume* (USDT).

    Accepts either the raw Binance ticker column ``quoteVolume`` or the
    canonical ``quote_volume`` used by ``RankFrame@1.0``. A universe that came
    through an input bundle is normalised to snake_case, and requiring only the
    camelCase name meant the canonical shape could not be fed to its own blocks.
    """
    for column in ("quoteVolume", "quote_volume"):
        if column in tickers.columns:
            return tickers[tickers[column] >= float(min_quote_volume)].copy()
    raise ValueError(
        "DataFrame missing 'quoteVolume' / 'quote_volume' column; got %s"
        % list(tickers.columns))


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
# News / social selection helpers
# ---------------------------------------------------------------------------


def _news_base_token(symbol: str) -> str:
    # Reuse the exact base-token normalisation used by the feature layer so a
    # universe join and an attached feature always agree on "BTCUSDT" -> "BTC".
    from .news_feed import base_token
    return base_token(symbol)


def augment_with_news(
    tickers: pd.DataFrame,
    ticker_rank_df: Optional[pd.DataFrame] = None,
    *,
    window: str = "24h",
    limit: int = 50,
    env: str = "prod",
) -> pd.DataFrame:
    """Augment the universe with Square ticker-mention stats.

    Joins ``getTickerRank`` output (keyed on the base token) onto the universe
    (keyed on ``<BASE>USDT``-style symbols), adding the :data:`_NEWS_COLS`
    columns. If *ticker_rank_df* is not supplied it is fetched live via
    :func:`cyqnt_trd.data_cli.fetch_ticker_rank`. A cache-miss / empty rank
    frame yields NaN columns rather than an error.
    """
    tickers = _with_symbol_column(tickers)

    if ticker_rank_df is None:
        from ..data_cli.news import fetch_ticker_rank
        ticker_rank_df = fetch_ticker_rank(window=window, limit=limit, env=env)

    if ticker_rank_df is None or ticker_rank_df.empty:
        for col in _NEWS_COLS:
            tickers[col] = float("nan")
        return tickers

    rank = ticker_rank_df.copy()
    # Square's raw frame keys on ``ticker`` (a base token, "BTC"); a bundle's
    # RankFrame@1.0 keys on ``instrument_id`` (a full symbol, "BTCUSDT"). Accept
    # either and normalise to a base token so the join works from both sources.
    if "ticker" not in rank.columns:
        for alias in ("instrument_id", "symbol"):
            if alias in rank.columns:
                rank["ticker"] = rank[alias].map(
                    lambda value: _news_base_token(str(value)))
                break
        else:
            raise ValueError(
                "ticker_rank_df missing 'ticker' / 'instrument_id' column; got %s"
                % list(rank.columns))
    rank["ticker"] = rank["ticker"].astype(str).str.upper()
    for column in ("bullish_count", "bearish_count", "neutral_count",
                   "mention_count", "unique_authors", "rank"):
        if column not in rank.columns:
            rank[column] = float("nan")
    # Sentiment arrives two ways and only one of them was understood. Square's
    # raw frame carries the counts; a bundle's RankFrame@1.0 — which is what
    # build_input_bundle and news_feed's PIT frame emit — carries the ratio
    # already computed, and no counts at all. Deriving solely from counts turned
    # every canonical row into NaN, and a NaN ratio reads downstream as "not
    # bullish", so a whole basket silently came back short.
    supplied = next(
        (column for column in ("news_bull_ratio", "bull_ratio") if column in rank.columns),
        None,
    )
    if supplied is not None:
        rank["news_bull_ratio"] = rank[supplied].astype(float)
    else:
        bull = rank["bullish_count"].astype(float)
        bear = rank["bearish_count"].astype(float)
        denom = bull + bear
        rank["news_bull_ratio"] = (bull / denom).where(denom > 0, other=float("nan"))
    rank = rank.rename(columns={
        "rank": "news_mention_rank",
        "mention_count": "news_mention_count",
        "unique_authors": "news_unique_authors",
        "bullish_count": "news_bullish_count",
        "bearish_count": "news_bearish_count",
        "neutral_count": "news_neutral_count",
    })
    join = rank[["ticker", *_NEWS_COLS]].drop_duplicates("ticker", keep="first")

    tickers["_base"] = tickers["symbol"].map(_news_base_token)
    out = tickers.merge(join, left_on="_base", right_on="ticker", how="left")
    out = out.drop(columns=[c for c in ("ticker", "_base") if c in out.columns])
    return out


def top_mentioned(tickers: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top *n* symbols by Square mention count (needs :func:`augment_with_news`)."""
    if "news_mention_count" not in tickers.columns:
        raise ValueError("DataFrame missing 'news_mention_count' — call augment_with_news first")
    ranked = tickers.dropna(subset=["news_mention_count"])
    return ranked.nlargest(int(n), "news_mention_count").copy()


def top_bullish(tickers: pd.DataFrame, n: int = 10, min_mentions: int = 0) -> pd.DataFrame:
    """Top *n* symbols by bullish ratio (needs :func:`augment_with_news`).

    Symbols with fewer than *min_mentions* Square mentions are excluded first so
    a single bullish post can't rank a thinly-covered token to the top.
    """
    if "news_bull_ratio" not in tickers.columns:
        raise ValueError("DataFrame missing 'news_bull_ratio' — call augment_with_news first")
    ranked = tickers.dropna(subset=["news_bull_ratio"])
    if min_mentions > 0 and "news_mention_count" in ranked.columns:
        ranked = ranked[ranked["news_mention_count"].fillna(0) >= float(min_mentions)]
    return ranked.nlargest(int(n), "news_bull_ratio").copy()


def filter_sentiment(tickers: pd.DataFrame, min_bull_ratio: float = 0.5) -> pd.DataFrame:
    """Keep symbols whose bullish ratio is >= *min_bull_ratio*.

    Requires :func:`augment_with_news`. Symbols with no sentiment data (NaN
    ratio) are dropped.
    """
    if "news_bull_ratio" not in tickers.columns:
        raise ValueError("DataFrame missing 'news_bull_ratio' — call augment_with_news first")
    return tickers[tickers["news_bull_ratio"] >= float(min_bull_ratio)].copy()


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

    def with_news(
        self, ticker_rank_df: Optional[pd.DataFrame] = None, **kwargs
    ) -> "UniverseFilter":
        self.df = augment_with_news(self.df, ticker_rank_df, **kwargs)
        return self

    def top_mentioned(self, n: int = 10) -> "UniverseFilter":
        self.df = top_mentioned(self.df, n)
        return self

    def top_bullish(self, n: int = 10, min_mentions: int = 0) -> "UniverseFilter":
        self.df = top_bullish(self.df, n, min_mentions)
        return self

    def filter_sentiment(self, min_bull_ratio: float = 0.5) -> "UniverseFilter":
        self.df = filter_sentiment(self.df, min_bull_ratio)
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
