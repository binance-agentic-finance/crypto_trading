"""Universe / target-pool management.

Helpers for building a dynamic list of symbols to scan, with the filter
chain that the user dataset most often asks for:

* perpetual-futures only
* 24h volume above N USDT
* 24h change within ±X%
* funding rate within ±Y%
* top-N gainers / losers
* explicit blacklist / whitelist
* quote currency (only USDT pairs / drop the USDC ones)

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

from typing import List, Optional, Sequence, Tuple, Union

import pandas as pd

from . import data as _data

__all__ = [
    "UniverseFilter",
    "fetch_perpetual_universe",
    "filter_quote_volume",
    "filter_change_pct",
    "filter_funding_rate",
    "augment_with_funding",
    "top_gainers",
    "top_losers",
    "exclude_symbols",
    "only_symbols",
    "filter_quote_suffix",
    "augment_with_news",
    "top_mentioned",
    "top_bullish",
    "filter_sentiment",
    "augment_with_contract_meta",
    "filter_underlying_type",
    "filter_sub_type",
    "filter_crypto_only",
    "augment_with_open_interest",
    "augment_with_oi_change",
    "augment_with_long_short_ratio",
    "filter_open_interest",
    "filter_oi_change",
    "filter_long_short_ratio",
    "augment_with_spread",
    "filter_spread",
    "filter_top_of_book",
    "augment_with_indicator",
]

# Columns added by :func:`augment_with_news`.
_NEWS_COLS = [
    "news_mention_rank", "news_mention_count", "news_unique_authors",
    "news_bullish_count", "news_bearish_count", "news_neutral_count",
    "news_bull_ratio",
]

#: Column :func:`augment_with_contract_meta` adds -> the source spellings it
#: accepts for it, most preferred first.
#:
#: Two vocabularies reach the join and both are real: Binance's own camelCase (a
#: direct :func:`cyqnt_trd.blocks.data.fetch_contract_meta` call, and a
#: ``cyqnt.input/v1`` frame, which keeps camelCase end to end exactly as the 24h
#: ticker does) and the snake_case a hand-built frame naturally writes.
#:
#: ``status`` becomes ``contract_status`` on the way out. A bare ``status``
#: column sitting on a cross-section beside ``news_bull_ratio`` and
#: ``fundingRatePct`` does not say WHAT is TRADING, and the bundle envelope
#: already uses ``status`` for "could we read this source at all" — two
#: different facts under one name in one artifact.
_CONTRACT_META_COLUMNS = {
    "contract_type": ("contractType", "contract_type"),
    "underlying_type": ("underlyingType", "underlying_type"),
    "underlying_sub_type": ("underlyingSubType", "underlying_sub_type"),
    "base_asset": ("baseAsset", "base_asset"),
    "quote_asset": ("quoteAsset", "quote_asset"),
    "contract_status": ("status", "contractStatus", "contract_status"),
}

#: How :func:`augment_with_contract_meta` encodes a multi-tag ``underlyingSubType``
#: as one scalar cell, and how :func:`filter_sub_type` reads it back.
_SUB_TYPE_SEPARATOR = ","

#: Minimum share of the universe the contract-metadata join must cover.
#:
#: Not a market fact but an identity: ``exchangeInfo`` is the listing registry of
#: the very venue the ticker table came from, so every symbol that can be quoted
#: must be in it (measured: 727 of 727). Anything meaningfully short of total
#: therefore means the source is *wrong* — a spot ``exchangeInfo`` joined onto a
#: futures universe, a truncated response, a stale capture from before a batch of
#: listings — and every uncovered row becomes NaN metadata, which makes
#: ``filter_crypto_only`` return a basket that is short for a reason no field of
#: the output records. The small slack is for the honest race: a symbol can enter
#: the 24h ticker between the two fetches.
_CONTRACT_META_MIN_COVERAGE = 0.95


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


def _without_derived_symbol(out: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Hand *out* back in the key vocabulary *source* arrived in.

    A row filter narrows rows; it has no business widening columns. But
    :func:`_with_symbol_column` has to derive ``symbol`` for a ``RankFrame@1.0``
    (keyed on ``instrument_id``) to be filterable at all, and leaving that column
    behind changed what every LATER step saw:

    * a step that had raised "missing 'symbol'" succeeded once some earlier step
      had quietly injected the column, so the SAME steps in a different order gave
      different answers — and both orders validated green;
    * it decided which name the interpreter treats as the instrument and which
      falls through into a candidate's ``features`` as a duplicate of it.

    The ``augment_*`` functions deliberately do NOT do this: widening the frame is
    their contract, and callers read ``symbol`` straight off their output (see
    ``strategies/news_buzz_selector.py``).
    """
    if "symbol" in out.columns and "symbol" not in source.columns:
        return out.drop(columns=["symbol"])
    return out


def _warn_matched_nothing(message: str) -> None:
    """Report a filter whose own vocabulary matched no row in the universe.

    A name the universe does not contain makes the filter indistinguishable from
    a filter that is not there: ``exclude`` drops nothing, the basket comes back
    identical to the unfiltered one, and no field of the output admits that the
    author's condition was never applied. :func:`_quote_suffixes` already refuses
    a *blank* suffix on exactly this reasoning; a non-blank name that matches
    nothing has the same effect and was still silent.

    A warning and not a raise, because matching nothing is a property of the DATA,
    not of the spec: a step earlier in the chain may legitimately leave a frame
    with no USDC pair in it, and aborting a live selection run over that would be
    an alarm the author cannot act on. The spelling mistake — the case that IS the
    author's to fix — is caught one step earlier instead: ``validate`` surfaces
    this warning out of its dry-run, where the stand-in universe carries the same
    quote currencies and instrument shapes a real one does.
    """
    import warnings

    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _warn_unknown_metadata(caller: str, column: str, dropped: int,
                           total: int,
                           remedy: str = "universe.augment_with_contract_meta") -> None:
    """Report rows dropped because their metadata was absent, not because it failed.

    "This coin is not TradFi" and "we do not know what this coin is" are
    different answers, and only one of them is a reason to trade it. The metadata
    filters therefore drop an unknown row whichever direction they are asked in
    — an unknown cannot satisfy ``include`` and cannot be cleared by ``exclude``
    — and say how many, because the alternative readings are both wrong:
    treating NaN as "no tags" lets an un-joined equity perpetual through
    ``exclude=['TradFi']``, and raising would abort a live basket over one symbol
    listed in the seconds between two fetches.

    A warning and not a raise for the same reason as
    :func:`_warn_matched_nothing`: this is a property of the data. The size of
    the hole is a property of the *source*, and that is what
    :data:`_CONTRACT_META_MIN_COVERAGE` refuses in
    :func:`augment_with_contract_meta`.
    """
    import warnings

    warnings.warn(
        "%s dropped %d of %d row(s) whose %s is missing — they were neither "
        "kept nor excluded, because an instrument with no reading cannot be "
        "shown to match or to fail the condition. Run %s first, and check its "
        "coverage warning if you already did."
        % (caller, dropped, total, column, remedy),
        RuntimeWarning, stacklevel=3)


def _warn_absent_meta_values(caller: str, argument: str,
                             absent: Sequence[str], present: Sequence[str]) -> None:
    """Report a named category that no row in the frame carries.

    :func:`_warn_matched_nothing` only fires when a filter matched *nothing at
    all*, so ``include=['COIN', 'EQUTIY']`` stayed silent: ``COIN`` matched, the
    typo did not, and the basket was simply missing a category the author asked
    for. These vocabularies are small and closed (seven underlying types, ~20
    sector tags), so the frame's own values can be listed in the message — which
    turns "why is there no equity in my basket" into a one-line fix.
    """
    import warnings

    warnings.warn(
        "%s was given %s=%s, and no row in this frame carries %s. Values present "
        "here: %s."
        % (caller, argument, list(absent), "them" if len(absent) > 1 else "it",
           list(present) or "(none)"),
        RuntimeWarning, stacklevel=3)


def _universe_sample(keyed: pd.DataFrame, limit: int = 3) -> str:
    """A few of the universe's own keys, to show the shape that WOULD match.

    Takes a frame that already went through :func:`_with_symbol_column`.
    """
    return ", ".join(str(value) for value in keyed["symbol"].head(limit).tolist())


def filter_quote_volume(
    tickers: pd.DataFrame, min_quote_volume: float = 100_000_000.0
) -> pd.DataFrame:
    """Keep symbols with 24h quote volume >= *min_quote_volume* (USDT).

    Accepts ``quoteVolume`` or ``quote_volume``, in that order of preference, so a
    hand-built snake_case cross-section can be fed to the same block.

    ``quoteVolume`` is the one a real universe has — the 24h ticker response is
    camelCase and stays camelCase through a ``cyqnt.input/v1`` bundle, as the
    frozen fixture in ``tests/standard_bot/fixtures`` shows. This tolerance is
    local to this function: a YAML ``selection.score`` is a plain column lookup, so
    ``score: quote_volume`` raises ``cannot resolve reference`` on any real
    universe. Name the column the frame actually has.
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


def _named_symbols(symbols: Sequence[str], caller: str) -> set:
    """Normalise an explicit roster to upper-case names, refusing an empty one.

    An empty roster is refused for the same reason a blank quote suffix is
    (:func:`_quote_suffixes`): ``only_symbols([])`` empties the universe and
    ``exclude_symbols([])`` is a no-op, and both look like a working filter. Unlike
    a roster that merely matches nothing — which can be an honest property of an
    already-narrowed frame — an empty one is wrong against every possible dataset,
    so it is the spec's bug and raising names it.
    """
    if isinstance(symbols, str):
        raise ValueError(
            "%s takes a list of instruments (symbols: [BTCUSDT, ETHUSDT]); a bare "
            "string would be read one character at a time" % caller)
    named = {str(value).strip().upper() for value in symbols}
    named.discard("")
    if not named:
        raise ValueError(
            "%s was given no symbols; name at least one instrument "
            "(symbols: [BTCUSDT]) or drop the step. An empty roster cannot express "
            "a filter: only_symbols([]) empties the universe and "
            "exclude_symbols([]) drops nothing." % caller)
    return named


def exclude_symbols(tickers: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    """Drop the given symbols from the universe.

    Reads either key vocabulary — see :func:`_with_symbol_column` — and hands the
    frame back in the one it arrived in.
    """
    keyed = _with_symbol_column(tickers)
    drop_set = _named_symbols(symbols, "exclude_symbols")
    matches = keyed["symbol"].astype(str).str.upper().isin(drop_set)
    if len(keyed) and not matches.any():
        _warn_matched_nothing(
            "exclude_symbols named %s and this universe of %d contains none of "
            "them (e.g. %s), so nothing was dropped. Names are matched whole, so "
            "pass full instruments, not base tokens (BTC)."
            % (sorted(drop_set), len(keyed), _universe_sample(keyed)))
    return _without_derived_symbol(keyed[~matches], tickers).copy()


def only_symbols(tickers: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    """Keep only the given symbols.

    Reads either key vocabulary — see :func:`_with_symbol_column` — and hands the
    frame back in the one it arrived in.
    """
    keyed = _with_symbol_column(tickers)
    keep_set = _named_symbols(symbols, "only_symbols")
    matches = keyed["symbol"].astype(str).str.upper().isin(keep_set)
    if len(keyed) and not matches.any():
        _warn_matched_nothing(
            "only_symbols named %s and this universe of %d contains none of them "
            "(e.g. %s), so the universe is now EMPTY. Names are matched whole, so "
            "pass full instruments, not base tokens (BTC)."
            % (sorted(keep_set), len(keyed), _universe_sample(keyed)))
    return _without_derived_symbol(keyed[matches], tickers).copy()


def _quote_suffixes(suffix: Union[str, Sequence[str]]) -> Tuple[str, ...]:
    """Normalise the *suffix* argument to a tuple of upper-case quote names.

    A blank suffix is refused rather than accepted: ``"".endswith("")`` is True
    for every symbol, so a blank one would make ``exclude=True`` empty the whole
    universe and ``exclude=False`` a no-op. Both look like a working filter, and
    neither is what the author asked for.
    """
    if isinstance(suffix, str):
        values: List[object] = [suffix]
    elif isinstance(suffix, (list, tuple, set, frozenset)):
        values = list(suffix)
    else:
        raise ValueError(
            "suffix must be a quote currency name ('USDT') or a list of them "
            "(['USDT', 'USDC']); got %s" % type(suffix).__name__)

    cleaned: List[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError(
                "every suffix must be a quote currency name like 'USDT'; got %r"
                % (value,))
        name = value.strip().upper()
        if not name:
            raise ValueError(
                "suffix is blank; name the quote currency to match, e.g. "
                "suffix='USDT' (or drop this filter entirely)")
        cleaned.append(name)
    if not cleaned:
        raise ValueError(
            "suffix list is empty; name at least one quote currency, e.g. "
            "suffix=['USDT', 'USDC'] (or drop this filter entirely)")
    # Order-preserving de-duplication: ('USDT', 'usdt') is one criterion, and a
    # repeated name in str.endswith would only cost work.
    return tuple(dict.fromkeys(cleaned))


def filter_quote_suffix(
    tickers: pd.DataFrame,
    suffix: Union[str, Sequence[str]] = "USDT",
    exclude: bool = False,
) -> pd.DataFrame:
    """Keep (or with *exclude* drop) symbols by quote currency.

    ``exclude=False`` keeps only the pairs quoted in *suffix*; ``exclude=True``
    drops them and keeps the rest. Matching is case-insensitive.

    *suffix* takes one name (``"USDT"``) or several (``["USDT", "USDC"]``),
    because "only USDT and USDC pairs" is as common a request as a single quote,
    and a list is what a YAML spec naturally writes.

    This is the filter that lets a spec say "no USDC-quoted pairs" instead of
    hand-listing ``BTCUSDC, ETHUSDC, SOLUSDC, …`` through
    :func:`exclude_symbols` — a roster that silently goes stale the next time the
    venue lists another USDC pair.

    The match is on the *end* of the symbol, so it is the quote and not the base
    that is tested: excluding ``USDC`` keeps ``USDCUSDT`` (a USDT-quoted pair
    whose base happens to be USDC).

    A named quote that matches no row is reported — see
    :func:`_warn_matched_nothing`. That is the ``exclude=True`` half of what
    :func:`_quote_suffixes` refuses for a blank suffix: ``exclude="USDCC"`` (a
    typo) dropped nothing and returned a basket identical to the unfiltered one.
    """
    keyed = _with_symbol_column(tickers)
    suffixes = _quote_suffixes(suffix)
    matches = keyed["symbol"].astype(str).str.upper().str.endswith(suffixes)
    if len(keyed) and not matches.any():
        _warn_matched_nothing(
            "quote suffix %s is not the quote of any of the %d symbols in this "
            "universe (e.g. %s), so this step %s. Check the spelling against the "
            "venue's quote names (USDT / USDC / FDUSD / BTC …) — a quote that "
            "matches nothing leaves the same basket as no filter at all."
            % (list(suffixes), len(keyed), _universe_sample(keyed),
               "dropped nothing" if exclude else "kept nothing"))
    return _without_derived_symbol(
        keyed[~matches if exclude else matches], tickers).copy()


def augment_with_funding(
    tickers: pd.DataFrame,
    funding_df: Optional[pd.DataFrame] = None,
    funding_info_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Join a cross-sectional funding snapshot, per settlement **and annualised**.

    Adds three columns:

    ``fundingRatePct``        the rate for ONE settlement, in percentage points.
                              Unchanged semantics — existing specs rank on it.
    ``fundingIntervalHours``  hours between this contract's settlements
    ``fundingRateApr``        ``fundingRatePct * (24 / hours) * 365`` — the carry
                              a position actually earns or pays over a year

    Why the annualised column had to exist
    -------------------------------------
    ``lastFundingRate`` is quoted PER SETTLEMENT and Binance settles different
    perpetuals at different frequencies, so the raw column's unit differs from row
    to row. Of 743 contracts (measured 2026-08-02) **443 settle 4-hourly**, 296
    8-hourly and 4 hourly — the majority is not the 8h everyone assumes, and the
    assumption is not even the plurality. Two instruments both showing 0.01%
    therefore pay 10.95%/yr and 87.6%/yr, eight times apart, and
    ``score: fundingRatePct`` ranks them as equal. Every "highest funding" and
    "most negative funding" basket built that way was mis-ordered, and nothing in
    the output said so: five symbols, five plausible rates.

    ``fundingRatePct`` is deliberately **left alone** rather than corrected in
    place. Existing specs, the demo templates and the frozen golden baskets all
    pin it, and silently changing what a column名 means would move every one of
    them for a reason a reader could not attribute. The fix is a new column with
    the unit in its name.

    The multiplier is always positive, so the SIGN is preserved: ``order: asc``
    still means "most negative", and only the ordering WITHIN a sign changes.

    ``funding_info_df`` absent is a warning and NaN, never an assumed 8 hours
    ----------------------------------------------------------------------------
    A default of 8 would keep the mis-ranking in place while making the output
    look repaired — the worst of the three outcomes, because it is the only one
    nobody would investigate. So an absent schedule frame leaves both new columns
    NaN and warns; ``validate`` relays that warning into the spec report. A spec
    that ranks on ``fundingRateApr`` must pass ``with: [funding, funding_info]``.

    ``funding_df`` is the replay-safe/YAML path: it is supplied by the unified
    input bundle and this function performs no I/O.  Direct Python callers may
    omit it to retain the original convenience behaviour of fetching Binance's
    current all-symbol premium-index snapshot. ``funding_info_df`` has no such
    fallback on purpose: a live fetch here would fire during ``validate``, and
    unlike the funding rate this frame is cheap to declare and always available
    from the same bundle.

    Accepted source vocabularies are Binance's raw
    ``symbol,lastFundingRate`` response, a wide canonical frame carrying
    ``instrument_id,funding_rate``, and a long ``MetricFrame@1.0`` carrying
    ``instrument_id,metric,value``.  Ratio values are converted to percentage
    points; ``*_bps`` values are converted from basis points.
    """
    tickers = _with_symbol_column(tickers)
    supplied = funding_df is not None
    source = funding_df if supplied else _data.fetch_premium_index()

    if source is None or not isinstance(source, pd.DataFrame):
        raise ValueError(
            "funding source must be a pandas DataFrame, got %s"
            % type(source).__name__)
    if source.empty:
        if supplied:
            raise ValueError(
                "funding source is empty; a selection needs a cross-sectional snapshot"
            )
        out = tickers.copy()
        out["fundingRatePct"] = float("nan")
        # Still annualised, even though every rate is NaN: the COLUMNS have to
        # exist whichever branch ran, or `score: fundingRateApr` resolves in one
        # and raises "cannot resolve reference" in the other.
        return _with_funding_apr(out, funding_info_df,
                                 caller="augment_with_funding")

    fr = source.copy()
    symbol_col = next(
        (column for column in ("symbol", "instrument_id") if column in fr.columns),
        None,
    )
    if symbol_col is None:
        raise ValueError(
            "funding source missing 'symbol' / 'instrument_id' column; got %s"
            % list(fr.columns))
    fr = fr[fr[symbol_col].notna()].copy()
    fr["symbol"] = fr[symbol_col].astype(str).str.upper()
    fr = fr[(fr["symbol"] != "") & (fr["symbol"] != "NAN")]

    # Canonical MetricFrame is long.  Keep funding rows only, then take the
    # latest value per instrument.  The bundle assembler already applied the
    # available_time <= decision_time gate; sorting here chooses among the
    # already-safe rows and deliberately does not invent a second PIT rule.
    if {"metric", "value"} <= set(fr.columns):
        metric = fr["metric"].astype(str).str.lower()
        aliases = {
            "rate", "funding_rate", "funding_rate_8h",
            "funding_rate_pct", "fundingratepct", "funding_rate_bps",
        }
        fr = fr[metric.isin(aliases)].copy()
        if fr.empty:
            raise ValueError(
                "funding MetricFrame has no funding metric; expected one of %s"
                % sorted(aliases))
        fr["__metric"] = fr["metric"].astype(str).str.lower()
        fr["__funding_value"] = pd.to_numeric(fr["value"], errors="coerce")
    else:
        value_col = next(
            (column for column in (
                "fundingRatePct", "funding_rate_pct", "funding_rate_bps",
                "funding_rate", "rate", "lastFundingRate",
            ) if column in fr.columns),
            None,
        )
        if value_col is None:
            raise ValueError(
                "funding source has no recognised rate column; got %s"
                % list(fr.columns))
        fr["__metric"] = value_col
        fr["__funding_value"] = pd.to_numeric(fr[value_col], errors="coerce")

    time_cols = [column for column in
                 ("available_time", "event_time", "time", "timestamp")
                 if column in fr.columns]
    if time_cols:
        fr = fr.sort_values(time_cols, kind="stable")
    fr = fr.dropna(subset=["symbol", "__funding_value"])
    fr = fr.drop_duplicates(subset=["symbol"], keep="last")

    universe_symbols = set(tickers["symbol"].astype(str).str.upper())
    source_symbols = set(fr["symbol"]) & universe_symbols
    if len(universe_symbols) > 1 and len(source_symbols) < 2:
        raise ValueError(
            "funding source covers only %d of %d universe instruments; this looks "
            "like single-symbol funding history, not the required cross-sectional "
            "snapshot" % (len(source_symbols), len(universe_symbols)))

    def _to_pct(row) -> float:
        value = float(row["__funding_value"])
        metric_name = str(row["__metric"]).lower()
        unit = str(row.get("unit", "")).lower()
        if metric_name.endswith("_bps") or unit in {"bp", "bps", "basis_points"}:
            return value / 100.0
        if metric_name in {"fundingratepct", "funding_rate_pct"} or unit in {
            "pct", "percent", "percentage",
        }:
            return value
        return value * 100.0

    fr["fundingRatePct"] = fr.apply(_to_pct, axis=1)
    base = tickers.drop(columns=["fundingRatePct"], errors="ignore")
    joined = base.merge(fr[["symbol", "fundingRatePct"]], on="symbol", how="left")
    return _with_funding_apr(joined, funding_info_df,
                             caller="augment_with_funding")


#: The one column :func:`augment_with_funding` needs from the schedule frame ->
#: accepted source spellings, most preferred first.
#:
#: Binance's camelCase (a direct :func:`cyqnt_trd.blocks.data.fetch_funding_info`
#: call) and the canonical snake_case a ``cyqnt.input/v1`` ``RankFrame@1.0``
#: carries after the ``funding_info`` node's ``column_map`` has run. The caps
#: (``adjustedFundingRateCap`` / ``Floor``) travel in the bundle but are NOT read
#: here: clamping is the venue's own behaviour, already reflected in the rate it
#: reports, so applying it again would be double-counting.
_FUNDING_INFO_COLUMNS = {
    "fundingIntervalHours": ("fundingIntervalHours", "funding_interval_hours"),
}

#: Settlements per day x days per year — the factor that turns ONE settlement's
#: rate into an annual one.
#:
#: 365 and not 365.25: a funding APR is a quoted convention rather than a
#: calendar computation, and 365 is what the venue's own UI, every funding
#: dashboard and the carry literature use. Using 365.25 here would put this
#: repo's numbers 0.07 % away from every external number they get compared to,
#: for no gain.
_HOURS_PER_DAY = 24.0
_DAYS_PER_YEAR = 365.0

#: Minimum share of the frame the funding-schedule join must cover.
#:
#: Looser than the book join's 95 % because this source has ONE legitimate hole
#: and it is a known set: the dated delivery contracts, which have no funding
#: schedule because they pay no funding (4 of 727 measured = 0.55 %). Still tight
#: enough to catch the case that matters — a stale or single-symbol schedule
#: frame, which would leave most of the cross-section un-annualised and let a
#: carry screen quietly rank the handful that survived.
_FUNDING_INFO_MIN_COVERAGE = 0.90

_FUNDING_INFO_COVERAGE_CAUSE = (
    "fundingInfo is a whole-market read — 743 rows covering 723 of the 727 "
    "instruments in a futures 24h ticker (measured) — so it cannot come back "
    "partial. Either the frame is stale / was captured for another venue, or "
    "this universe has been narrowed to DATED DELIVERY contracts "
    "(BTCUSDT_260925 and friends): those pay no funding at all, so there is no "
    "interval to annualise with and no divisor would make a carry screen mean "
    "anything on them."
)


def _with_funding_apr(frame: pd.DataFrame,
                      funding_info_df: Optional[pd.DataFrame], *,
                      caller: str) -> pd.DataFrame:
    """Add ``fundingIntervalHours`` and ``fundingRateApr`` to a funded frame.

    Split out of :func:`augment_with_funding` so both of its exits produce the
    same column set — see the note at the empty-source branch.
    """
    node = "funding_info"
    added = ["fundingIntervalHours", "fundingRateApr"]
    if funding_info_df is None:
        _warn_missing_funding_schedule(caller)
        out = frame.drop(columns=added, errors="ignore").copy()
        for column in added:
            out[column] = float("nan")
        return out

    source = _derivative_source(
        funding_info_df, supplied=True, caller=caller, node=node,
        empty_cause=_WHOLE_MARKET_EMPTY_CAUSE,
        misreading="no perpetual on this venue settles funding")
    source = _source_symbol_key(source, caller=caller, node=node)
    resolved = _resolve_source_columns(source, _FUNDING_INFO_COLUMNS,
                                       caller=caller, node=node)

    join = pd.DataFrame({"symbol": source["symbol"].values})
    # Positional, not index-aligned: a frame read out of a bundle's rows can
    # carry a duplicated index.
    join["fundingIntervalHours"] = pd.to_numeric(
        source[resolved["fundingIntervalHours"]], errors="coerce").values
    join = join.drop_duplicates(subset=["symbol"], keep="last")

    positive = join["fundingIntervalHours"] > 0
    if not bool(positive.all()):
        _warn_unusable_funding_interval(caller, join, ~positive)
        join["fundingIntervalHours"] = join["fundingIntervalHours"].where(positive)

    _require_join_coverage(frame, join, caller=caller, node=node,
                           floor=_FUNDING_INFO_MIN_COVERAGE,
                           cause=_FUNDING_INFO_COVERAGE_CAUSE)

    out = frame.drop(columns=added, errors="ignore").merge(
        join[["symbol", "fundingIntervalHours"]], on="symbol", how="left")
    out["fundingRateApr"] = (
        out["fundingRatePct"]
        * (_HOURS_PER_DAY / out["fundingIntervalHours"])
        * _DAYS_PER_YEAR
    )
    return out


def _warn_missing_funding_schedule(caller: str) -> None:
    """Report that the annualised column could not be computed, and why not 8h.

    A warning rather than a raise because ``fundingRatePct`` — what every
    existing spec ranks on — is unaffected and correct, so refusing here would
    break specs that never asked for the new column. The one thing that must not
    happen is a DEFAULT: assuming 8 hours would produce numbers for the 447 of
    743 contracts that do not settle 8-hourly, and being wrong in a way that
    looks repaired is worse than being NaN, because nobody investigates it.
    """
    import warnings

    warnings.warn(
        "%s was given no funding_info frame, so fundingIntervalHours and "
        "fundingRateApr are NaN for every instrument. They are NOT defaulted to "
        "8 hours: on this venue 443 of 743 contracts settle 4-hourly and 4 "
        "hourly, so an 8h assumption would annualise the majority of the market "
        "wrongly while looking correct. Declare the source — in YAML that is "
        "`with: [funding, funding_info]` on the universe.augment_with_funding "
        "step — or rank on fundingRatePct, which is per settlement and "
        "unaffected." % caller,
        RuntimeWarning, stacklevel=4)


def _warn_unusable_funding_interval(caller: str, join, bad) -> None:
    """Report schedule rows whose interval cannot divide 24 hours."""
    import warnings

    warnings.warn(
        "%s: %d of %d funding_info row(s) carry a settlement interval that is "
        "zero, negative or missing (e.g. %s), so their annualised rate is NaN. "
        "Dividing by it would give an infinite APR that sorts first in every "
        "'highest funding' basket. The venue publishes 8 / 4 / 1 — a value "
        "outside that means the response changed shape, so re-check "
        "blocks.data.fetch_funding_info."
        % (caller, int(bad.sum()), len(join),
           join.loc[bad, "symbol"].head(3).tolist()),
        RuntimeWarning, stacklevel=4)


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
# Contract metadata: what the instrument IS (sector / asset class)
# ---------------------------------------------------------------------------


def _flatten_sub_type(value):
    """One ``underlyingSubType`` cell -> a scalar, comma-separated tag string.

    This has to happen at the join boundary, and it is not cosmetic.
    ``underlyingSubType`` is a JSON **array** (``['Alpha', 'DeFi']``), and a
    cell holding a 2-element array reaches
    ``interpreter.selection_fn``'s ``if pd.notna(value)`` when it assembles a
    candidate's ``features`` — where numpy raises ``ValueError: The truth value
    of an array with more than one element is ambiguous``. A ONE-element array
    does not raise, which is why this was invisible: 722 of 727 symbols carry a
    single tag, so it appears only for a multi-tag coin, only in production, and
    only after the filters have already run.

    An empty array becomes ``""`` — the venue assigning no sector IS a value —
    while an absent row stays NaN, meaning "we could not look this up". The
    filters below treat those two differently, so collapsing them here would
    destroy the distinction.
    """
    # isinstance BEFORE any pd.isna: pd.isna on a multi-element list raises the
    # very ambiguity error this function exists to prevent.
    if isinstance(value, (list, tuple, set, frozenset)):
        tags = [str(item).strip() for item in value]
        tags = [tag for tag in tags if tag]
        for tag in tags:
            if _SUB_TYPE_SEPARATOR in tag:
                raise ValueError(
                    "sector tag %r contains %r, which is the separator this "
                    "column is encoded with, so the tag set cannot be read back "
                    "unambiguously. The venue's tag vocabulary changed — pick "
                    "another separator in universe._SUB_TYPE_SEPARATOR and "
                    "update filter_sub_type with it."
                    % (tag, _SUB_TYPE_SEPARATOR))
        return _SUB_TYPE_SEPARATOR.join(tags)
    if value is None:
        return float("nan")
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return float("nan")
    return str(value)


def augment_with_contract_meta(
    tickers: pd.DataFrame,
    contract_meta_df: Optional[pd.DataFrame] = None,
    *,
    market_type: str = "futures",
) -> pd.DataFrame:
    """Join the venue's listing registry onto the universe.

    Adds :data:`_CONTRACT_META_COLUMNS`' keys — ``contract_type``,
    ``underlying_type``, ``underlying_sub_type``, ``base_asset``,
    ``quote_asset``, ``contract_status`` — so a selection spec can express what
    an instrument *is* rather than only how it traded.

    ``contract_meta_df`` is the replay-safe / YAML path: it is supplied by the
    unified input bundle (``with: [contract_meta]``) and this function performs
    no I/O. Direct Python callers may omit it to fetch Binance's current
    ``exchangeInfo`` for *market_type*.

    Why this exists at all: a 24h ticker cross-section cannot distinguish
    ``BTCUSDT`` from ``SNDKUSDT`` (a tokenised SanDisk perpetual). Ranked by
    turnover, the tokenised-equity perpetuals dominate — on one frozen
    cross-section the five biggest 24h losers above a $2m floor were five TradFi
    perpetuals — so "scan Binance futures for short candidates, no stocks" could
    not be stated, only approximated by hand-listing names that go stale.

    ``underlying_sub_type`` is a comma-separated scalar, never a list; see
    :func:`_flatten_sub_type` for why that conversion belongs here.
    """
    tickers = _with_symbol_column(tickers)
    supplied = contract_meta_df is not None
    source = contract_meta_df if supplied else _data.fetch_contract_meta(
        market_type=market_type)

    if source is None or not isinstance(source, pd.DataFrame):
        raise ValueError(
            "contract metadata source must be a pandas DataFrame, got %s"
            % type(source).__name__)
    if source.empty:
        # Never NaN-fill: "the registry answered with nothing" is a broken source,
        # and a frame of NaN metadata makes every filter below return an empty
        # basket with nothing to point at. Compare augment_with_news, which DOES
        # NaN-fill — buzz is genuinely absent for most coins, whereas every listed
        # contract has a contract type by construction.
        raise ValueError(
            "contract metadata source is empty; a listed contract always has a "
            "contract type, so an empty registry is a failed fetch and not a "
            "market state. Check source_status for the 'contract_meta' node.")

    meta = source.copy()
    symbol_col = next(
        (column for column in ("symbol", "instrument_id") if column in meta.columns),
        None,
    )
    if symbol_col is None:
        raise ValueError(
            "contract metadata source missing 'symbol' / 'instrument_id' column; "
            "got %s" % list(meta.columns))

    resolved = {}
    for out_column, aliases in _CONTRACT_META_COLUMNS.items():
        found = next((alias for alias in aliases if alias in meta.columns), None)
        if found is None:
            raise ValueError(
                "contract metadata source has no column for %r (accepted: %s); "
                "it carries %s. A partial registry would hand the filters a NaN "
                "column, which reads as 'this instrument has no sector' for every "
                "symbol." % (out_column, list(aliases), list(meta.columns)))
        resolved[out_column] = found

    join = pd.DataFrame({
        "symbol": meta[symbol_col].astype(str).str.upper(),
    })
    for out_column, source_column in resolved.items():
        values = meta[source_column]
        if out_column == "underlying_sub_type":
            values = values.map(_flatten_sub_type)
        # Positionally, not index-aligned: every column here comes out of the same
        # ``meta``, and a registry frame read straight from a bundle's rows can
        # carry a duplicated index — which index alignment would refuse.
        join[out_column] = values.values
    join = join[(join["symbol"] != "") & (join["symbol"] != "NAN")]
    # Last row wins, matching augment_with_funding: the bundle assembler has
    # already applied the PIT gate, so any duplicate here is the registry listing
    # the same instrument twice and the later row is the current one.
    join = join.drop_duplicates(subset=["symbol"], keep="last")

    universe_symbols = set(tickers["symbol"].astype(str).str.upper())
    if universe_symbols:
        covered = set(join["symbol"]) & universe_symbols
        coverage = len(covered) / len(universe_symbols)
        if coverage < _CONTRACT_META_MIN_COVERAGE:
            missing = sorted(universe_symbols - covered)
            raise ValueError(
                "contract metadata covers only %d of %d universe instruments "
                "(%.1f%%, floor %.0f%%); e.g. %s are absent. exchangeInfo is the "
                "listing registry of the same venue the universe came from, so a "
                "hole this size means the wrong source, not a market fact — the "
                "usual causes are a spot exchangeInfo joined onto a futures "
                "universe (check market_type) and a truncated or stale capture. "
                "Left alone, these rows would carry NaN metadata and every "
                "sector filter would silently return a short basket."
                % (len(covered), len(universe_symbols), coverage * 100.0,
                   _CONTRACT_META_MIN_COVERAGE * 100.0, missing[:5]))

    base = tickers.drop(columns=list(_CONTRACT_META_COLUMNS), errors="ignore")
    return base.merge(join, on="symbol", how="left")


def _meta_criterion(values, *, caller: str, argument: str) -> Tuple[str, ...]:
    """Normalise an ``include``/``exclude`` argument to upper-case category names.

    Accepts one name (``"COIN"``) or several (``["COIN", "INDEX"]``), like
    :func:`_quote_suffixes` and unlike :func:`_named_symbols`: a category is a
    value from a small closed vocabulary, so a bare scalar is the natural way to
    write it in YAML and there is no ambiguity to protect against. An empty list
    is still refused — ``include=[]`` empties the universe and ``exclude=[]`` is
    a no-op, and both look like a filter that is doing something.
    """
    if isinstance(values, str):
        items: List[object] = [values]
    elif isinstance(values, (list, tuple, set, frozenset)):
        items = list(values)
    else:
        raise ValueError(
            "%s: %s must be a category name ('COIN') or a list of them "
            "(['COIN', 'INDEX']); got %s"
            % (caller, argument, type(values).__name__))

    cleaned: List[str] = []
    for item in items:
        if not isinstance(item, str):
            raise ValueError(
                "%s: every %s entry must be a category name like 'COIN'; got %r"
                % (caller, argument, item))
        name = item.strip().upper()
        if not name:
            raise ValueError(
                "%s: %s contains a blank name; name the category to match, e.g. "
                "%s=['COIN'] (or drop this filter)" % (caller, argument, argument))
        cleaned.append(name)
    if not cleaned:
        raise ValueError(
            "%s: %s is empty. An empty list cannot express a filter — include=[] "
            "empties the universe and exclude=[] drops nothing — so name at least "
            "one category or drop the step." % (caller, argument))
    return tuple(dict.fromkeys(cleaned))


def _require_meta_column(tickers: pd.DataFrame, column: str, caller: str) -> None:
    if column not in tickers.columns:
        raise ValueError(
            "%s needs the %r column, which this frame does not have (it has %s). "
            "Add a `universe.augment_with_contract_meta` step first — in YAML "
            "that is `- block: universe.augment_with_contract_meta` with "
            "`with: [contract_meta]`."
            % (caller, column, list(tickers.columns)))


def filter_underlying_type(
    tickers: pd.DataFrame,
    include: Optional[Union[str, Sequence[str]]] = None,
    exclude: Optional[Union[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """Keep or drop instruments by ``underlying_type`` (a single scalar category).

    Needs :func:`augment_with_contract_meta`. The venue's values are
    ``COIN`` / ``EQUITY`` / ``COMMODITY`` / ``HK_EQUITY`` / ``KR_EQUITY`` /
    ``INDEX`` / ``PREMARKET``; matching is exact and case-insensitive, never a
    substring, so ``EQUITY`` does not match ``HK_EQUITY``.

    Exactly one of *include* / *exclude* must be given. Both together are refused
    rather than composed: this column holds one value per row, so ``include``
    alone already determines the surviving set and adding ``exclude`` can only
    subtract from a set the author has already enumerated — a way to write a
    contradiction and get an empty basket with no error. (Contrast
    :func:`filter_sub_type`, where a row carries several tags and "AI but not
    TradFi" is a real condition neither list can state alone.)

    Rows whose ``underlying_type`` is missing are dropped in *both* directions
    and reported — see :func:`_warn_unknown_metadata`.
    """
    caller = "filter_underlying_type"
    _require_meta_column(tickers, "underlying_type", caller)
    if include is None and exclude is None:
        raise ValueError(
            "%s needs include= or exclude=; with neither it returns the frame "
            "unchanged, which is indistinguishable from the step not being there. "
            "For \"crypto only\" prefer universe.filter_crypto_only, which names "
            "the intent once." % caller)
    if include is not None and exclude is not None:
        raise ValueError(
            "%s takes include= OR exclude=, not both: underlying_type holds one "
            "value per row, so include= already names the whole surviving set and "
            "exclude= can only contradict it. Use include= for a whitelist "
            "(recommended — it cannot be widened by a newly added category) or "
            "exclude= for a blacklist." % caller)

    values = tickers["underlying_type"]
    known = values.notna()
    if not bool(known.all()):
        _warn_unknown_metadata(caller, "underlying_type",
                               int((~known).sum()), len(tickers))
    normalised = values.where(known, other="").astype(str).str.strip().str.upper()
    present = sorted({value for value in normalised[known].tolist() if value})

    if include is not None:
        wanted = _meta_criterion(include, caller=caller, argument="include")
        absent = [name for name in wanted if name not in present]
        if len(tickers) and absent:
            _warn_absent_meta_values(caller, "include", absent, present)
        matches = known & normalised.isin(wanted)
    else:
        unwanted = _meta_criterion(exclude, caller=caller, argument="exclude")
        absent = [name for name in unwanted if name not in present]
        if len(tickers) and absent:
            _warn_absent_meta_values(caller, "exclude", absent, present)
        matches = known & ~normalised.isin(unwanted)

    return tickers[matches].copy()


def _sub_type_sets(tickers: pd.DataFrame):
    """``underlying_sub_type`` -> one upper-case tag *set* per row, plus a known mask."""
    values = tickers["underlying_sub_type"]
    known = values.notna()
    text = values.where(known, other="").astype(str).str.upper()
    # A set intersection, deliberately not a substring test: `"AI" in "Alpha-AI"`
    # is True and `{"AI"} & {"ALPHA-AI"}` is not, and the tags really do share
    # prefixes ("Layer-1"/"Layer-2", "AI"/"Alpha"). A substring match would put
    # coins in a sector they were never tagged with, and nothing downstream would
    # contradict it.
    tag_sets = text.str.split(_SUB_TYPE_SEPARATOR).map(
        lambda parts: {part.strip() for part in parts if part.strip()})
    return tag_sets, known


def filter_sub_type(
    tickers: pd.DataFrame,
    include: Optional[Union[str, Sequence[str]]] = None,
    exclude: Optional[Union[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """Keep or drop instruments by sector tag — a **multi-valued** membership test.

    Needs :func:`augment_with_contract_meta`. The venue tags each contract with
    zero or more of ``DeFi`` / ``TradFi`` / ``Alpha`` / ``Infrastructure`` / ``AI``
    / ``Layer-1`` / ``Meme`` / ``Gaming`` / ``ETF`` / ``Pre-IPO`` / … and a coin
    can carry several (``FOLKSUSDT`` is ``Alpha,DeFi``).

    ``include`` keeps a row carrying **any** of the named tags; ``exclude`` drops
    a row carrying **any** of them. Both may be given, and unlike
    :func:`filter_underlying_type` that is meaningful here precisely because a row
    has several tags: ``include=[AI], exclude=[TradFi]`` keeps the AI coins while
    dropping something tagged ``AI,TradFi``. ``include`` is applied first.

    A row tagged with nothing (the venue sends ``[]``, which arrives as ``""``)
    is a *known* answer: it fails every ``include`` and passes every ``exclude``.
    A row whose tags are *missing* (NaN — no registry row joined) is dropped in
    both directions and reported; see :func:`_warn_unknown_metadata`.

    For "exclude TradFi" prefer :func:`filter_crypto_only`, which is stricter and
    says why — see its docstring.
    """
    caller = "filter_sub_type"
    _require_meta_column(tickers, "underlying_sub_type", caller)
    if include is None and exclude is None:
        raise ValueError(
            "%s needs include= or exclude=; with neither it returns the frame "
            "unchanged, which is indistinguishable from the step not being there."
            % caller)

    tag_sets, known = _sub_type_sets(tickers)
    if not bool(known.all()):
        _warn_unknown_metadata(caller, "underlying_sub_type",
                               int((~known).sum()), len(tickers))
    present = sorted({tag for tags in tag_sets[known] for tag in tags})

    matches = known.copy()
    if include is not None:
        wanted = set(_meta_criterion(include, caller=caller, argument="include"))
        absent = [name for name in sorted(wanted) if name not in present]
        if len(tickers) and absent:
            _warn_absent_meta_values(caller, "include", absent, present)
        matches &= tag_sets.map(lambda tags: bool(tags & wanted))
    if exclude is not None:
        unwanted = set(_meta_criterion(exclude, caller=caller, argument="exclude"))
        absent = [name for name in sorted(unwanted) if name not in present]
        if len(tickers) and absent:
            _warn_absent_meta_values(caller, "exclude", absent, present)
        matches &= ~tag_sets.map(lambda tags: bool(tags & unwanted))

    return tickers[matches].copy()


def filter_crypto_only(tickers: pd.DataFrame) -> pd.DataFrame:
    """Keep only genuinely crypto-native instruments: ``underlying_type == COIN``.

    Needs :func:`augment_with_contract_meta`. This exists as its own named block,
    rather than as a spelling of :func:`filter_underlying_type`, because "exclude
    the TradFi stuff" has three plausible encodings which give three different
    answers, and nothing in a basket reveals which one was used:

    ==========================================  =======
    ``contract_type != TRADIFI_PERPETUAL``          577
    ``'TradFi' not in underlying_sub_type``         577
    ``underlying_type == 'COIN'``                   575
    ==========================================  =======

    (One venue snapshot, 727 TRADING contracts.) The two extra names the
    blacklists let through are ``ALLUSDT`` and ``BTCDOMUSDT`` — both
    ``underlying_type == INDEX``, synthetic baskets of many coins rather than any
    one coin. They are not TradFi, so both blacklists are *correct* about what
    they were asked; they are also not a coin you can reason about with a
    per-asset thesis, which is what an author asking for "crypto only" means.

    So the definition here is the whitelist, the tightest of the three: a row
    survives only if the venue positively calls it a ``COIN``. A whitelist is
    also the safer half of the choice over time — the next asset class the venue
    lists (tokenised bonds, prediction markets, whatever ``PREMARKET`` becomes)
    is excluded by default instead of arriving inside a basket that used to be
    all crypto. Name the intent once and nobody has to re-derive which of the
    three they wrote.

    Use :func:`filter_underlying_type` directly when the answer really is one of
    the looser sets (``include=[COIN, INDEX]`` to keep the crypto indices) or
    when the categories wanted are the non-crypto ones.
    """
    return filter_underlying_type(tickers, include=("COIN",))


# ---------------------------------------------------------------------------
# Derivatives cross-section: position inventory and crowd positioning
#
# These three joins are fed by a FAN-OUT capture — one request per instrument,
# because Binance publishes no all-market open-interest or long/short endpoint
# (both answer HTTP 400 with the symbol omitted). Nothing below fetches: the
# roster is chosen and paid for at capture time, and a block's only job is to
# join what the bundle already carries.
#
# The consequence a spec author feels: **these steps must come after the steps
# that narrow the universe.** The capture fanned out over a narrowed roster, so
# joining it onto the full 727-row cross-section covers 17% of it — which the
# coverage guards below refuse by name rather than letting it become a column of
# mostly-NaN that empties the basket. See
# ``standard_bot/data/live_snapshot.py`` for the cost table that makes the
# ordering a requirement rather than a style.
# ---------------------------------------------------------------------------


#: Column :func:`augment_with_open_interest` adds -> source spellings accepted
#: for it, most preferred first.
#:
#: Two vocabularies reach the join, exactly as with contract metadata: Binance's
#: camelCase (a direct :func:`cyqnt_trd.blocks.data.fetch_open_interest_cross_section`
#: call) and the canonical snake_case a ``cyqnt.input/v1`` ``RankFrame@1.0``
#: carries after the node's ``column_map`` has run.
_OPEN_INTEREST_COLUMNS = {
    "oi_base": ("openInterest", "oi_base"),
    # Named for its role, not for the field it came from. A universe frame can
    # also carry a funding-sourced ``markPrice``, and the two are read at
    # different instants; ``oi_mark_price`` says which reading produced
    # ``oi_notional_usd`` so the dollar figure can be re-derived from the row.
    "oi_mark_price": ("markPrice", "mark_price", "oi_mark_price"),
}

#: Column :func:`augment_with_long_short_ratio` adds -> accepted source spellings.
_LONG_SHORT_COLUMNS = {
    "long_short_ratio": ("longShortRatio", "long_short_ratio"),
    "__long_account": ("longAccount", "long_account"),
    "__short_account": ("shortAccount", "short_account"),
}

#: Metric names :func:`augment_with_oi_change` accepts for the two magnitudes.
#:
#: A long ``MetricFrame@1.0`` names them ``oi_base`` / ``oi_value`` (the catalog
#: node's ``column_map``); a raw response names them ``sumOpenInterest`` /
#: ``sumOpenInterestValue``.
_OI_HIST_BASE_METRICS = ("oi_base", "sumopeninterest", "sum_open_interest")
_OI_HIST_VALUE_METRICS = ("oi_value", "sumopeninterestvalue",
                          "sum_open_interest_value", "oi_notional_usd")

#: The same two magnitudes as WIDE column names, for a vendor-shaped frame.
_OI_HIST_BASE_COLUMNS = ("sumOpenInterest", "oi_base", "sum_open_interest")
_OI_HIST_VALUE_COLUMNS = ("sumOpenInterestValue", "oi_value",
                          "sum_open_interest_value")

#: Minimum share of the frame the open-interest join must cover before it raises.
#:
#: Every listed perpetual has a current open-interest reading — the fetcher
#: raises per instrument rather than skipping one — so a hole is never a market
#: fact here. It is one of two mistakes, and both are named in the message: the
#: augment step was placed before the steps that narrow the universe (the
#: signature case: 127 of 727 = 17 %), or the roster was built from a different
#: snapshot than the one being screened. The slack is for the honest race, a
#: symbol entering the ticker between the two collections.
_OPEN_INTEREST_MIN_COVERAGE = 0.95

#: The same floor for the two *statistics* sources, and it is deliberately loose.
#:
#: ``openInterestHist`` and ``globalLongShortAccountRatio`` legitimately return
#: nothing for a recently listed perpetual — the aggregated series starts some
#: time after the contract does — so a hole here IS sometimes a market fact, and
#: refusing at 95 % would abort a live screen over a handful of new listings.
#: Every hole is still reported with its size (see :func:`_warn_unknown_metadata`)
#: and every uncovered row is dropped rather than defaulted. What this floor
#: catches is the categorically different case: a frame captured for another
#: roster entirely, where the mis-ordered-step example lands at 17 %.
_OI_STATISTICS_MIN_COVERAGE = 0.50

#: How far the observed cadence of an open-interest history may sit from one day
#: before :func:`augment_with_oi_change` refuses to call the result a *daily*
#: change. See its docstring for why this is checked rather than trusted.
_ONE_DAY_MS = 86_400_000
_CADENCE_TOLERANCE = 0.25


#: Why an empty cross-section is refused, per the shape of the collection.
#:
#: The two call for different next steps, so the message says which one failed:
#: a fan-out that collected nothing is a roster/rate-budget question, while a
#: whole-market read that came back empty is one request to retry.
_FAN_OUT_EMPTY_CAUSE = (
    "This field has no all-market endpoint, so an empty frame means the fan-out "
    "collected nothing"
)
_WHOLE_MARKET_EMPTY_CAUSE = (
    "This field is read for the WHOLE market in ONE request, so an empty frame "
    "means that single request failed or answered with an empty list — no roster "
    "and no per-instrument hole can be at fault"
)


def _derivative_source(source, *, supplied: bool, caller: str, node: str,
                       empty_cause: str = _FAN_OUT_EMPTY_CAUSE,
                       misreading: str = "no instrument has open interest"):
    """Common front door for the cross-sectional joins.

    Refuses a non-frame and an empty frame. Empty is a refusal and never a
    NaN-fill: unlike Square buzz — genuinely absent for most coins, which is why
    :func:`augment_with_news` fills NaN — a cross-section of open interest,
    positioning or quotes is either captured or it is not. A frame of NaN would
    make every threshold below return an empty basket, i.e. present a failed
    capture as a strict screen.

    ``empty_cause`` / ``misreading`` name the collection that failed and the
    false conclusion the caller must not draw from it, because both differ
    between a fan-out and a whole-market read.
    """
    import pandas as pd

    if source is None or not isinstance(source, pd.DataFrame):
        raise ValueError(
            "%s: the %s source must be a pandas DataFrame, got %s"
            % (caller, node, type(source).__name__))
    if source.empty:
        raise ValueError(
            "%s: the %s source is empty. %s — check "
            "source_status for the %r node rather than reading this as "
            "'%s'.%s"
            % (caller, node, empty_cause, node, misreading,
               "" if supplied else
               " (No source was passed, so this came from a live fetch.)"))
    return source.copy()


def _source_symbol_key(frame, *, caller: str, node: str):
    """Add an upper-case ``symbol`` join key to a source frame, whichever
    vocabulary it arrived in, and drop the rows that have no key at all."""
    column = next((name for name in ("symbol", "instrument_id")
                   if name in frame.columns), None)
    if column is None:
        raise ValueError(
            "%s: the %s source has no 'symbol' / 'instrument_id' column; it "
            "carries %s" % (caller, node, list(frame.columns)))
    frame = frame[frame[column].notna()].copy()
    frame["symbol"] = frame[column].astype(str).str.upper()
    return frame[(frame["symbol"] != "") & (frame["symbol"] != "NAN")]


def _resolve_source_columns(frame, wanted, *, caller: str, node: str):
    """``{out_column: (alias, ...)}`` -> ``{out_column: the alias present}``.

    Missing one is a refusal rather than a NaN column, for the same reason
    :func:`augment_with_contract_meta` refuses a partial registry: a NaN column
    reads downstream as "this instrument has a small position", which is a
    reason to trade it.
    """
    resolved = {}
    for out_column, aliases in wanted.items():
        found = next((alias for alias in aliases if alias in frame.columns), None)
        if found is None:
            raise ValueError(
                "%s: the %s source has no column for %r (accepted: %s); it "
                "carries %s. A partial frame would become a NaN column, and a "
                "NaN threshold silently drops every instrument."
                % (caller, node, out_column.lstrip("_"), list(aliases),
                   list(frame.columns)))
        resolved[out_column] = found
    return resolved


def _require_join_coverage(keyed, join, *, caller: str, node: str,
                           floor: float, cause: str) -> None:
    """Refuse a join that covers too little of the frame it is joined onto."""
    universe_symbols = set(keyed["symbol"].astype(str).str.upper())
    if not universe_symbols:
        return
    covered = set(join["symbol"]) & universe_symbols
    coverage = len(covered) / len(universe_symbols)
    if coverage >= floor:
        return
    missing = sorted(universe_symbols - covered)
    raise ValueError(
        "%s: the %s cross-section covers only %d of %d instruments in this frame "
        "(%.1f%%, floor %.0f%%); e.g. %s are absent. %s Left alone, those rows "
        "would carry a NaN reading and every threshold below would return a "
        "basket that is short for a reason nothing in the output records."
        % (caller, node, len(covered), len(universe_symbols), coverage * 100.0,
           floor * 100.0, missing[:5], cause))


_STEP_ORDER_CAUSE = (
    "This field has no all-market endpoint, so the capture fanned out over a "
    "NARROWED roster — put this step AFTER the steps that narrow the universe "
    "(filter_sub_type / filter_crypto_only / filter_quote_volume), not before "
    "them; joining a 127-instrument roster onto the full 727-row cross-section "
    "lands at 17%. The other cause is a roster built from a different snapshot "
    "than the one being screened."
)


def augment_with_open_interest(
    tickers: pd.DataFrame,
    oi_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Join current open interest, in coins **and** converted to dollars.

    Adds three columns:

    ``oi_base``           open positions denominated in the base coin
    ``oi_mark_price``     the price the conversion used
    ``oi_notional_usd``   ``oi_base * oi_mark_price`` — the comparable magnitude

    ``oi_df`` is the replay-safe / YAML path (``with: [open_interest_snapshot]``)
    and this function performs no I/O. A direct Python caller may omit it, in
    which case the roster is *this frame's own* instruments — which means the
    fan-out ceiling applies to the frame, and a 727-row cross-section raises. That
    is the intended failure: narrow first.

    Why the dollar column is not optional
    -------------------------------------
    ``oi_base`` alone is not comparable across instruments and looks as if it
    were. On one snapshot DOGEUSDT carried 2.79e9 and BTCUSDT 1.09e5 — $193m and
    $6.8b. A screen written as "open interest above 5 million" against the base
    column keeps every sub-dollar coin on the venue and drops BTC, and every
    number in the basket looks plausible. So both magnitudes are joined, the
    dollar one is named in the unit it is in, and
    :func:`filter_open_interest` takes ``min_notional_usd`` — a keyword that
    cannot be read as coins.

    The mark price is carried as a column, not consumed and discarded, so a
    reader can re-derive the notional from the emitted candidate. It comes from
    the open-interest capture rather than from the funding cross-section; see
    :func:`cyqnt_trd.blocks.data.fetch_open_interest_cross_section` for that
    trade-off.
    """
    caller = "augment_with_open_interest"
    node = "open_interest_snapshot"
    keyed = _with_symbol_column(tickers)
    supplied = oi_df is not None
    if supplied:
        source = oi_df
    else:
        source = _data.fetch_open_interest_cross_section(
            keyed["symbol"].astype(str).str.upper().tolist())
    source = _derivative_source(source, supplied=supplied, caller=caller, node=node)
    source = _source_symbol_key(source, caller=caller, node=node)
    resolved = _resolve_source_columns(source, _OPEN_INTEREST_COLUMNS,
                                       caller=caller, node=node)

    join = pd.DataFrame({"symbol": source["symbol"].values})
    for out_column, source_column in resolved.items():
        # Positional, not index-aligned: a frame read out of a bundle's rows can
        # carry a duplicated index, which alignment would refuse.
        join[out_column] = pd.to_numeric(source[source_column], errors="coerce").values
    # Last row wins, matching augment_with_funding and augment_with_contract_meta:
    # the bundle assembler has already applied the PIT gate, so a duplicate here
    # is the same instrument read twice and the later read is the current one.
    join = join.drop_duplicates(subset=["symbol"], keep="last")

    # NaN fails this too, and deliberately: an unpriced instrument is exactly as
    # unconvertible as a zero-priced one.
    priced = join["oi_mark_price"] > 0
    if not bool(priced.all()):
        raise ValueError(
            "%s: %d instrument(s) came with a mark price that is zero, negative "
            "or missing (e.g. %s), so their dollar open interest cannot be "
            "computed. Left alone it would be zero or NaN, which a "
            "min_notional_usd floor then drops for being SMALL rather than for "
            "being unknown — recapture the %s frame."
            % (caller, int((~priced).sum()),
               join.loc[~priced, "symbol"].head(5).tolist(), node))
    join["oi_notional_usd"] = join["oi_base"] * join["oi_mark_price"]

    _require_join_coverage(keyed, join, caller=caller, node=node,
                           floor=_OPEN_INTEREST_MIN_COVERAGE,
                           cause=_STEP_ORDER_CAUSE)

    added = ["oi_base", "oi_mark_price", "oi_notional_usd"]
    base = keyed.drop(columns=added, errors="ignore")
    return base.merge(join[["symbol", *added]], on="symbol", how="left")


def _oi_history_series(source, *, caller: str, node: str):
    """The history source, in either vocabulary, as ``symbol, time, metric, value``.

    A long ``MetricFrame@1.0`` (what a bundle carries) and a wide vendor response
    (what the fetcher returns) are both accepted, exactly as
    :func:`augment_with_funding` accepts both — a block that only understood the
    bundle shape could not be called from Python, and one that only understood
    the vendor shape could not be replayed.
    """
    frame = _source_symbol_key(source, caller=caller, node=node)
    time_column = next((name for name in ("event_time", "timestamp", "time")
                        if name in frame.columns), None)
    if time_column is None:
        raise ValueError(
            "%s: the %s source has no time column ('event_time' / 'timestamp'); "
            "it carries %s. Without one there is no way to tell the latest "
            "reading from the baseline it is compared against."
            % (caller, node, list(frame.columns)))

    if {"metric", "value"} <= set(frame.columns):
        long = pd.DataFrame({
            "symbol": frame["symbol"].values,
            "time": pd.to_numeric(frame[time_column], errors="coerce").values,
            "metric": frame["metric"].astype(str).str.lower().str.replace(
                " ", "", regex=False).values,
            "value": pd.to_numeric(frame["value"], errors="coerce").values,
        })
        return long.dropna(subset=["time", "value"])

    wide = _resolve_source_columns(
        frame,
        {"oi_base": _OI_HIST_BASE_COLUMNS, "oi_value": _OI_HIST_VALUE_COLUMNS},
        caller=caller, node=node)
    parts = []
    for metric, column in wide.items():
        parts.append(pd.DataFrame({
            "symbol": frame["symbol"].values,
            "time": pd.to_numeric(frame[time_column], errors="coerce").values,
            "metric": metric,
            "value": pd.to_numeric(frame[column], errors="coerce").values,
        }))
    return pd.concat(parts, ignore_index=True).dropna(subset=["time", "value"])


def _require_daily_cadence(long, *, caller: str, lookback_days: int) -> None:
    """Refuse to call a change over N readings a change over N *days*.

    ``lookback_days`` is a promise about wall-clock time, but the readings come
    from a capture that chose ``period=`` — and ``openInterestHist`` serves
    ``5m`` just as happily as ``1d``. Trusting the argument would turn
    "open interest moved 20 % this week" into "…in the last 35 minutes" with the
    same column name, the same YAML, and no error anywhere. So the cadence is
    measured from the timestamps that actually arrived, which also covers a
    hand-built frame that carries no ``timeframe`` column to check.
    """
    gaps = (long.sort_values(["symbol", "time"])
                .groupby("symbol", sort=False)["time"].diff().dropna())
    gaps = gaps[gaps > 0]
    if gaps.empty:
        return
    observed = float(gaps.median())
    if abs(observed - _ONE_DAY_MS) <= _ONE_DAY_MS * _CADENCE_TOLERANCE:
        return
    raise ValueError(
        "%s: lookback_days=%d says the baseline is %d DAYS of open interest, but "
        "the readings in this frame are spaced %.4g hours apart (median), so the "
        "same computation would measure %.4g hours and report it under a column "
        "named for days. Recapture the open-interest history with period='1d' "
        "(the oi_change_snapshot node's default), or change the lookback to "
        "match the cadence you have."
        % (caller, lookback_days, lookback_days, observed / 3.6e6,
           observed * lookback_days / 3.6e6))


def _change_vs_baseline(long, metric_names, *, lookback_days: int):
    """``{symbol: pct change of the latest reading vs the mean of the previous N}``.

    Returns ``(changes, short_history, undefined)``. The two rejected sets are
    kept apart because they are different facts and one of them is reported:
    ``short_history`` is "not enough readings yet", ``undefined`` is "the baseline
    was zero, so a percentage is not a number". Both come back as NaN, which is
    the only honest value, and neither is silently absent from the join — a symbol
    the source DID cover must count towards coverage, or a market full of new
    listings would read as a mismatched roster.
    """
    subset = long[long["metric"].isin(metric_names)]
    changes = {}
    short_history = set()
    undefined = set()
    for symbol, rows in subset.groupby("symbol", sort=False):
        values = rows.sort_values("time")["value"].tolist()
        # The latest reading plus the whole baseline, or nothing. Averaging three
        # days and calling it a week is the failure mode this refuses: a
        # perpetual listed on Tuesday has open interest growing from zero, so a
        # short baseline manufactures a huge "change" for exactly the newest and
        # thinnest instruments — the ones a screen like this surfaces.
        if len(values) < lookback_days + 1:
            short_history.add(symbol)
            continue
        baseline_values = values[-(lookback_days + 1):-1]
        baseline = sum(baseline_values) / float(len(baseline_values))
        if baseline == 0:
            # Zero open interest a week ago is not a percentage change, it is an
            # undefined one. NaN keeps it out of the basket; a division would put
            # an infinity at rank 1 under ``order: desc``.
            undefined.add(symbol)
            continue
        changes[symbol] = (values[-1] - baseline) / baseline * 100.0
    return changes, short_history, undefined


def augment_with_oi_change(
    tickers: pd.DataFrame,
    oi_hist_df: Optional[pd.DataFrame] = None,
    *,
    lookback_days: int = 7,
) -> pd.DataFrame:
    """Join the recent change in open interest, measured both ways.

    Adds two columns, both in percentage points:

    ``oi_change_pct``        change in the DOLLAR position inventory
    ``oi_base_change_pct``   change in the COIN count

    Each is ``(latest - baseline) / baseline * 100`` where the baseline is the
    arithmetic mean of the ``lookback_days`` readings *before* the latest one — a
    mean rather than a single point, so one quiet day does not decide the answer.

    Both are joined because the two disagree, often and by a lot, and the
    disagreement is the whole point. Measured on 2026-08-02 over a 7-day
    lookback::

        USUSDT     base  -3.2%   notional  +22.1%
        ALLOUSDT   base  -3.5%   notional  -26.5%
        TAGUSDT    base  +7.0%   notional  +28.2%
        UBUSDT     base  +1.5%   notional  +22.2%
        UAIUSDT    base +33.3%   notional  +39.8%

    A ``|change| >= 20%`` screen keeps all five on the dollar column and one on
    the coin column. Neither is wrong: the notional moves with price, so it
    answers "how much money is parked here now versus last week" — which is what
    a request for 持倉異動 means and what every venue's own OI chart plots — while
    the coin count answers "did positions actually open or close", free of price.
    Publishing one and hiding the other would make a 5-vs-1 difference invisible,
    so :func:`filter_oi_change` takes an explicit ``basis`` and BOTH numbers land
    in the emitted candidate's ``features`` whichever one filtered.

    An instrument with fewer than ``lookback_days + 1`` readings gets NaN, not a
    change computed over the days it does have: a perpetual listed three days ago
    has open interest growing from zero, so a short baseline manufactures a
    spectacular change for precisely the newest instruments. The count is
    reported.

    ``oi_hist_df`` is the replay-safe / YAML path (``with: [oi_change_snapshot]``).
    """
    caller = "augment_with_oi_change"
    node = "oi_change_snapshot"
    if int(lookback_days) < 1:
        raise ValueError(
            "%s: lookback_days must be at least 1, got %r — a baseline of zero "
            "readings has nothing to compare the latest one against"
            % (caller, lookback_days))
    lookback_days = int(lookback_days)

    keyed = _with_symbol_column(tickers)
    supplied = oi_hist_df is not None
    if supplied:
        source = oi_hist_df
    else:
        source = _data.fetch_oi_history_cross_section(
            keyed["symbol"].astype(str).str.upper().tolist(),
            period="1d", limit=lookback_days + 1)
    source = _derivative_source(source, supplied=supplied, caller=caller, node=node)

    long = _oi_history_series(source, caller=caller, node=node)
    if long.empty:
        raise ValueError(
            "%s: the %s source carried rows but none of them had both a "
            "timestamp and a numeric value, so no change can be measured."
            % (caller, node))
    _require_daily_cadence(long, caller=caller, lookback_days=lookback_days)

    # Metric PRESENCE is checked on the frame, not on the computed changes.
    # Deriving it from an empty result conflated two different failures: a source
    # whose metric names we do not understand, and a source we understand
    # perfectly in which no instrument had enough history yet (or every baseline
    # was zero). The second is a market state with its own warning, and reporting
    # it as "this frame carries no open interest" sent the reader to the wrong
    # place entirely.
    present = set(long["metric"].unique())
    absent = {
        "oi_change_pct": (list(_OI_HIST_VALUE_METRICS)
                          if not (present & set(_OI_HIST_VALUE_METRICS)) else None),
        "oi_base_change_pct": (list(_OI_HIST_BASE_METRICS)
                               if not (present & set(_OI_HIST_BASE_METRICS)) else None),
    }
    missing = {column: names for column, names in absent.items() if names}
    if missing:
        # Refused rather than left NaN, for the same reason a partial wide frame
        # is refused in _resolve_source_columns: this block promises both columns,
        # and an all-NaN one makes filter_oi_change(basis=...) return an empty
        # basket that looks exactly like a strict screen working correctly.
        raise ValueError(
            "%s: the %s source carries no metric for %s (accepted: %s); it has "
            "%s. Both magnitudes are required — the dollar change and the coin "
            "change select different instruments, so a NaN column would make "
            "filter_oi_change(basis=...) silently empty the basket."
            % (caller, node, sorted(missing),
               sorted({name for names in missing.values() for name in names}),
               sorted(present)[:8]))

    notional, short_notional, undefined_notional = _change_vs_baseline(
        long, _OI_HIST_VALUE_METRICS, lookback_days=lookback_days)
    base_change, short_base, undefined_base = _change_vs_baseline(
        long, _OI_HIST_BASE_METRICS, lookback_days=lookback_days)

    short_history = short_notional | short_base
    covered = (set(notional) | set(base_change) | short_history
               | undefined_notional | undefined_base)
    join = pd.DataFrame({"symbol": sorted(covered)})
    join["oi_change_pct"] = join["symbol"].map(notional)
    join["oi_base_change_pct"] = join["symbol"].map(base_change)

    _require_join_coverage(keyed, join, caller=caller, node=node,
                           floor=_OI_STATISTICS_MIN_COVERAGE,
                           cause=_STEP_ORDER_CAUSE)
    in_frame = set(keyed["symbol"].astype(str).str.upper())
    if short_history & in_frame:
        _warn_absent_history(caller, node, sorted(short_history & in_frame),
                            lookback_days)
    undefined = (undefined_notional | undefined_base) & in_frame
    if undefined:
        _warn_undefined_change(caller, node, sorted(undefined), lookback_days)

    added = ["oi_change_pct", "oi_base_change_pct"]
    base = keyed.drop(columns=added, errors="ignore")
    return base.merge(join[["symbol", *added]], on="symbol", how="left")


def _warn_absent_history(caller: str, node: str, symbols: Sequence[str],
                         lookback_days: int) -> None:
    """Report instruments whose series was too short for the stated lookback.

    A warning and not a raise, because it is a property of the DATA — a perpetual
    listed this week cannot have a week of open interest, and aborting a live
    screen over that would be an alarm nobody can act on. What must not happen is
    the silent version: these rows carry NaN and are dropped by the filters, so
    without this the basket is simply short and the reason is nowhere.
    """
    import warnings

    warnings.warn(
        "%s: %d instrument(s) have fewer than %d+1 open-interest readings, so "
        "their change is NaN and every threshold below drops them: %s. A "
        "recently listed perpetual genuinely has no %d-day history — the "
        "alternative would be averaging the days it does have and reporting that "
        "under the same column name, which manufactures a large change for the "
        "newest instruments. Widen the %s capture's limit if you expected them."
        % (caller, len(symbols), lookback_days, symbols[:8], lookback_days, node),
        RuntimeWarning, stacklevel=3)


def _warn_undefined_change(caller: str, node: str, symbols: Sequence[str],
                           lookback_days: int) -> None:
    """Report instruments whose baseline open interest was zero.

    Reported separately from :func:`_warn_absent_history` because it is a
    different fact with a different remedy: the readings ARE there, they are all
    zero, so the instrument had no position inventory at all over the window.
    Silently NaN would be indistinguishable from a source that skipped it.
    """
    import warnings

    warnings.warn(
        "%s: %d instrument(s) had zero open interest across the whole %d-day "
        "baseline, so a percentage change is undefined and theirs is NaN: %s. "
        "This is not a change of 0%% and not an infinite one — dividing by the "
        "zero baseline would rank them first."
        % (caller, len(symbols), lookback_days, symbols[:8]),
        RuntimeWarning, stacklevel=3)


def augment_with_long_short_ratio(
    tickers: pd.DataFrame,
    ls_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Join the crowd's positioning skew onto the cross-section.

    Adds:

    ``long_short_ratio``   long accounts / short accounts, as the venue reports it
    ``long_account_pct``   the long share in PERCENTAGE POINTS (67.28, not 0.6728)

    The unit conversion is the reason this is a block and not a rename. The venue
    sends fractions of 1, and "long/short ratio above 60" versus "above 0.6"
    versus "above 1.5" are three different conditions that all look like
    plausible YAML. So the column carrying a share is named ``_pct`` and is in
    the same unit as ``fundingRatePct`` and ``priceChangePercent`` — percentage
    points, the unit everything else in a universe frame uses — while the ratio
    keeps its own scale, where 1.0 is the balanced point.

    The fraction is **verified, not assumed**: ``longAccount + shortAccount``
    must be 1 within a rounding tolerance, so a source that switched to
    percentages raises here instead of yielding a ``long_account_pct`` of 6728.

    ``ls_df`` is the replay-safe / YAML path
    (``with: [long_short_ratio_snapshot]``).
    """
    caller = "augment_with_long_short_ratio"
    node = "long_short_ratio_snapshot"
    keyed = _with_symbol_column(tickers)
    supplied = ls_df is not None
    if supplied:
        source = ls_df
    else:
        source = _data.fetch_long_short_ratio_cross_section(
            keyed["symbol"].astype(str).str.upper().tolist())
    source = _derivative_source(source, supplied=supplied, caller=caller, node=node)
    source = _source_symbol_key(source, caller=caller, node=node)
    resolved = _resolve_source_columns(source, _LONG_SHORT_COLUMNS,
                                       caller=caller, node=node)

    join = pd.DataFrame({"symbol": source["symbol"].values})
    for out_column, source_column in resolved.items():
        join[out_column] = pd.to_numeric(source[source_column], errors="coerce").values
    join = join.drop_duplicates(subset=["symbol"], keep="last")

    total = join["__long_account"] + join["__short_account"]
    off = total.notna() & ((total - 1.0).abs() > 0.02)
    if bool(off.any()):
        sample = join.loc[off, ["symbol", "__long_account", "__short_account"]]
        raise ValueError(
            "%s: longAccount + shortAccount should be 1 (they are shares of the "
            "accounts holding a position) and %d row(s) sum to something else, "
            "e.g. %s. The source's unit changed — if it now sends percentages, "
            "multiplying by 100 here would produce a long_account_pct of ~6700 "
            "and every 'retail is above 60%%' screen would match everything."
            % (caller, int(off.sum()), sample.head(3).to_dict("records")))
    join["long_account_pct"] = join["__long_account"] * 100.0

    _require_join_coverage(keyed, join, caller=caller, node=node,
                           floor=_OI_STATISTICS_MIN_COVERAGE,
                           cause=_STEP_ORDER_CAUSE)

    added = ["long_short_ratio", "long_account_pct"]
    base = keyed.drop(columns=added, errors="ignore")
    return base.merge(join[["symbol", *added]], on="symbol", how="left")


# ---------------------------------------------------------------------------
# Liquidity: the top of the book, which is what "illiquid" actually means
# ---------------------------------------------------------------------------


#: Column :func:`augment_with_spread` reads -> the source spellings accepted for
#: it, most preferred first.
#:
#: Two vocabularies again: Binance's camelCase (a direct
#: :func:`cyqnt_trd.blocks.data.fetch_book_ticker_cross_section` call) and the
#: canonical snake_case a ``cyqnt.input/v1`` ``RankFrame@1.0`` carries after the
#: ``book_ticker`` node's ``column_map`` has run.
_BOOK_TICKER_COLUMNS = {
    "__bid_price": ("bidPrice", "bid_price"),
    "__bid_qty": ("bidQty", "bid_qty"),
    "__ask_price": ("askPrice", "ask_price"),
    "__ask_qty": ("askQty", "ask_qty"),
}

#: Minimum share of the frame the book join must cover before it raises.
#:
#: Higher than the fan-out floors, and for a different reason: this endpoint is
#: whole-market in ONE request, so it returns every instrument the venue quotes
#: (measured: 727 of 727 in a futures 24h ticker). A hole is therefore never a
#: market fact and never a rate-budget casualty — it means the two frames are
#: from different markets or different snapshots. The slack is for the honest
#: race, a symbol entering the ticker between the two reads.
_BOOK_TICKER_MIN_COVERAGE = 0.95

#: What a book-coverage hole means, given that the read cannot be partial.
#:
#: Deliberately NOT :data:`_STEP_ORDER_CAUSE`: this join has no roster and no
#: fan-out, so "put the step after the narrowing steps" would be wrong advice —
#: it works in any position.
_BOOK_COVERAGE_CAUSE = (
    "This read is whole-market in one request, so it cannot come back partial: "
    "either the frame was captured for a DIFFERENT venue (a spot bookTicker "
    "joined onto a futures universe — the symbol sets overlap, which is what "
    "makes this hard to see) or for a different snapshot than the universe being "
    "screened. Collect both in the same pass."
)


def augment_with_spread(
    tickers: pd.DataFrame,
    book_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Join the touch: how WIDE the book is and how MUCH is resting on it.

    Adds two columns:

    ``spread_bps``        ``(ask - bid) / mid * 10000`` — the cost of a round trip
                          at the touch, in basis points
    ``top_of_book_usd``   ``min(bidQty * bid, askQty * ask)`` — the smaller of the
                          two sides in dollars, i.e. how much can be done at the
                          touch in the harder direction

    Why turnover is not enough
    --------------------------
    "剔除低流動性的空氣幣" is normally written as ``filter_quote_volume``, and
    ``quoteVolume`` is a proxy that fails in the direction that costs money: it
    reports what traded over 24 h, not whether an order can be filled now. From
    one snapshot (2026-08-02):

    * ``SNXXUSDT`` turned over **$111m** — it clears the ``min_quote_volume:
      100000000`` floor ``example_selection.yaml`` ships — at an **11.0 bps**
      spread, 683x BTCUSDT's 0.016 bps, with $16.9k at the touch.
    * ``TAKEUSDT`` turned over **$24.5m** and offers **$0.03** at the best ask.

    Neither is exotic: the median instrument sits at 5.5 bps and 395 of 727 are
    above 5 bps. So a turnover-only screen hands an operator a basket in which
    nothing records that the fill will not happen, and ``top_of_book_usd`` is the
    direct answer to "能不能進出" that no volume column can give.

    ``min`` and not the average of the two sides, because a basket is entered AND
    exited: a book with $500k of bids and $30 of asks is a $30 book for anything
    that has to get out, and averaging it to $250k would describe a position
    nobody can close.

    ``book_df`` is the replay-safe / YAML path (``with: [book_ticker]``) and this
    function then performs no I/O. A direct Python caller may omit it, in which
    case ONE whole-market request is made whatever the frame's length — unlike
    the three derivative joins, whose live fallback fans out per instrument, so
    there is no roster ceiling and no step-order requirement here.

    A quote that cannot be a quote
    -----------------------------
    A one-sided book (bid or ask absent / non-positive) and a crossed or locked
    one (``bid >= ask``) both get ``spread_bps = NaN`` and are counted in a
    warning, rather than being reported as a number. This is the case the column
    exists for and it is the one a plain subtraction gets exactly backwards: an
    unpriced side makes ``mid`` zero and the spread infinite or undefined, while a
    crossed book yields a NEGATIVE spread and a locked one yields 0.0 — and every
    one of those **passes** ``max_spread_bps``. So the tightest-looking rows in
    the basket would be the ones whose book could not be read. NaN is dropped by
    :func:`filter_spread` and counted, which is the honest answer: "we could not
    price this instrument" is not "this instrument is cheap to trade".
    """
    caller = "augment_with_spread"
    node = "book_ticker"
    keyed = _with_symbol_column(tickers)
    supplied = book_df is not None
    source = book_df if supplied else _data.fetch_book_ticker_cross_section()
    source = _derivative_source(
        source, supplied=supplied, caller=caller, node=node,
        empty_cause=_WHOLE_MARKET_EMPTY_CAUSE,
        misreading="no instrument on this venue is quoted")
    source = _source_symbol_key(source, caller=caller, node=node)
    resolved = _resolve_source_columns(source, _BOOK_TICKER_COLUMNS,
                                       caller=caller, node=node)

    join = pd.DataFrame({"symbol": source["symbol"].values})
    for out_column, source_column in resolved.items():
        # Positional, not index-aligned: a frame read out of a bundle's rows can
        # carry a duplicated index, which alignment would refuse.
        join[out_column] = pd.to_numeric(source[source_column], errors="coerce").values
    # Last row wins, matching every other join here: the bundle assembler has
    # already applied the PIT gate, so a duplicate is the same instrument read
    # twice and the later read is the current one.
    join = join.drop_duplicates(subset=["symbol"], keep="last")

    bid, ask = join["__bid_price"], join["__ask_price"]
    unpriced = ~((bid > 0) & (ask > 0))
    # ``>=`` and not ``>``: a locked book (bid == ask) is 0.0 bps, which is not a
    # tight market but two sides read at different instants, and 0.0 clears every
    # ceiling a spec can write.
    crossed = (~unpriced) & (bid >= ask)
    usable = ~(unpriced | crossed)
    if bool(unpriced.any()) or bool(crossed.any()):
        _warn_unquotable_book(caller, join, unpriced=unpriced, crossed=crossed)

    mid = (bid + ask) / 2.0
    join["spread_bps"] = ((ask - bid) / mid * 10_000.0).where(usable)
    # Quantities are NOT held to the same test. A resting size of exactly zero is
    # a true statement about the book — nothing at the touch — and a
    # min_top_of_book_usd floor is right to drop it for being small. Only the
    # PRICE has to be usable, because that is what the ratio divides by.
    join["top_of_book_usd"] = pd.concat([
        join["__bid_qty"] * bid, join["__ask_qty"] * ask,
    ], axis=1).min(axis=1).where(usable)

    _require_join_coverage(keyed, join, caller=caller, node=node,
                           floor=_BOOK_TICKER_MIN_COVERAGE,
                           cause=_BOOK_COVERAGE_CAUSE)

    added = ["spread_bps", "top_of_book_usd"]
    base = keyed.drop(columns=added, errors="ignore")
    return base.merge(join[["symbol", *added]], on="symbol", how="left")


def _warn_unquotable_book(caller: str, join, *, unpriced, crossed) -> None:
    """Report the rows whose touch could not be turned into a spread.

    A warning and not a raise, for the reason :func:`_warn_matched_nothing`
    gives: a halted or newly-listed contract with one side of the book empty is a
    property of the market, and aborting a live basket over it would be an alarm
    the spec author cannot act on. What must not happen is the row keeping a
    NUMBER — see :func:`augment_with_spread` — so the values are NaN and the
    count is stated here.
    """
    import warnings

    parts = []
    if bool(unpriced.any()):
        parts.append(
            "%d with no usable bid or ask (e.g. %s)"
            % (int(unpriced.sum()), join.loc[unpriced, "symbol"].head(3).tolist()))
    if bool(crossed.any()):
        parts.append(
            "%d crossed or locked, bid >= ask (e.g. %s)"
            % (int(crossed.sum()), join.loc[crossed, "symbol"].head(3).tolist()))
    warnings.warn(
        "%s: %s of %d instrument(s) in the %s cross-section have no quotable "
        "touch — theirs are NaN, not numbers. Left as arithmetic they would be "
        "infinite, negative or exactly 0.0 bps, and all three CLEAR a "
        "max_spread_bps ceiling, so the widest books would sort as the tightest. "
        "universe.filter_spread drops them and says how many."
        % (caller, " and ".join(parts), len(join), "book_ticker"),
        RuntimeWarning, stacklevel=3)


# ---------------------------------------------------------------------------
# Derivative filters
#
# Why these are blocks rather than ``conditions.value_above`` in ``long_when``:
#
# * a ``long_when`` condition decides a candidate's SIDE, it does not narrow the
#   universe, so everything that fails it still competes for a top_k slot and
#   still gets ranked. "Open interest above $5m" is a screen, and a screen has to
#   be a universe step to be one.
# * NaN. ``conditions.value_above`` on a missing reading yields False, which is
#   indistinguishable from a small reading; these filters drop an unknown and say
#   how many (:func:`_warn_unknown_metadata`). "We could not read this
#   instrument's open interest" is not "this instrument has little open
#   interest", and only one of the two is a reason not to trade it.
# * the unit is in the keyword. ``min_notional_usd`` cannot be read as coins;
#   ``args: [oi_base, 5000000]`` can, and would silently screen a different
#   quantity — see :func:`augment_with_open_interest`.
# ---------------------------------------------------------------------------


#: augment block -> the bundle frame it must be given. Kept beside the message
#: that quotes it so the two cannot drift.
_AUGMENT_SOURCES = {
    "augment_with_open_interest": "open_interest_snapshot",
    "augment_with_oi_change": "oi_change_snapshot",
    "augment_with_long_short_ratio": "long_short_ratio_snapshot",
    "augment_with_spread": "book_ticker",
}


def _require_derived_column(tickers: pd.DataFrame, column: str, caller: str,
                            step: str) -> None:
    if column not in tickers.columns:
        raise ValueError(
            "%s needs the %r column, which this frame does not have (it has %s). "
            "Add a `universe.%s` step first — in YAML that is `- block: "
            "universe.%s` with `with: [%s]` — and put it AFTER the steps that "
            "narrow the universe."
            % (caller, column, list(tickers.columns), step, step,
               _AUGMENT_SOURCES[step]))


def _bounded_filter(tickers: pd.DataFrame, column: str, *, caller: str,
                    step: str, bounds: Sequence[Tuple[str, Optional[float], str]],
                    remedy: str) -> pd.DataFrame:
    """Apply inclusive/exclusive numeric bounds to one column, dropping unknowns.

    ``bounds`` is ``(keyword name, value, "min"|"max"|"absmin"|"absmax")``.
    """
    _require_derived_column(tickers, column, caller, step)
    named = [(name, value, kind) for name, value, kind in bounds if value is not None]
    if not named:
        raise ValueError(
            "%s needs at least one bound (%s); with none it returns the frame "
            "unchanged, which is indistinguishable from the step not being there."
            % (caller, ", ".join(name for name, _value, _kind in bounds)))

    values = pd.to_numeric(tickers[column], errors="coerce")
    known = values.notna()
    if len(tickers) and not bool(known.all()):
        _warn_unknown_metadata(caller, column, int((~known).sum()), len(tickers),
                               remedy=remedy)
    matches = known.copy()
    for _name, value, kind in named:
        threshold = float(value)
        if kind == "min":
            matches &= values >= threshold
        elif kind == "strict_min":
            matches &= values > threshold
        elif kind == "max":
            matches &= values <= threshold
        elif kind == "strict_max":
            matches &= values < threshold
        elif kind == "absmin":
            matches &= values.abs() >= threshold
        elif kind == "absmax":
            matches &= values.abs() <= threshold
        else:                                       # pragma: no cover - internal
            raise AssertionError("unknown bound kind %r" % (kind,))
    return tickers[matches.fillna(False)].copy()


def filter_open_interest(
    tickers: pd.DataFrame,
    min_notional_usd: Optional[float] = None,
    max_notional_usd: Optional[float] = None,
) -> pd.DataFrame:
    """Keep instruments whose DOLLAR open interest is within the given bounds.

    Needs :func:`augment_with_open_interest`. Both bounds are inclusive and
    absolute; either may be omitted.

    The keyword says ``usd`` because the underlying field does not: open interest
    arrives denominated in the base coin, where 5,000,000 is a rounding error for
    a meme coin and more than the venue holds for BTC. ``max_notional_usd`` is
    here for the symmetric screen — "big enough to trade, small enough that a
    position is not the whole float".

    Instruments whose open interest is unknown are dropped and counted; see
    :func:`_warn_unknown_metadata`.
    """
    return _bounded_filter(
        tickers, "oi_notional_usd", caller="filter_open_interest",
        step="augment_with_open_interest",
        bounds=(("min_notional_usd", min_notional_usd, "min"),
                ("max_notional_usd", max_notional_usd, "max")),
        remedy="universe.augment_with_open_interest")


#: ``basis`` -> the column :func:`filter_oi_change` screens on.
_OI_CHANGE_BASES = {
    "notional": "oi_change_pct",
    "base": "oi_base_change_pct",
}


def filter_oi_change(
    tickers: pd.DataFrame,
    min_abs_pct: Optional[float] = None,
    min_pct: Optional[float] = None,
    max_pct: Optional[float] = None,
    *,
    basis: str = "notional",
) -> pd.DataFrame:
    """Keep instruments whose open interest moved (or did not) over the lookback.

    Needs :func:`augment_with_oi_change`. All bounds are inclusive percentage
    points and may be combined:

    ``min_abs_pct``  magnitude in either direction — "positions moved a lot"
    ``min_pct``      signed floor — "positions were ADDED"
    ``max_pct``      signed ceiling — "positions were closed"

    ``min_abs_pct`` is separate rather than expressed as two ranges because "a
    week of |Δ| ≥ 20 %" is one condition and ``any_of`` over two signed bounds is
    not available to a universe step (a step narrows rows; it is not a combinator
    tree).

    ``basis`` names WHICH change, and it is explicit because the two answers
    differ materially — on the measurement in
    :func:`augment_with_oi_change`'s docstring, a ±20 % screen keeps five
    instruments on ``notional`` and one on ``base``. ``notional`` is the default
    because it is what a request for 持倉異動 / "OI up 20 %" means and what the
    venue's own chart plots; ``base`` is the price-free position count. Both
    columns stay on the frame either way, so the emitted candidate shows the
    number that was not used as well as the one that was.
    """
    caller = "filter_oi_change"
    if basis not in _OI_CHANGE_BASES:
        raise ValueError(
            "%s: basis must be one of %s, got %r. 'notional' screens the dollar "
            "position inventory (moves with price — this is what \"OI up 20%%\" "
            "usually means); 'base' screens the coin count (price-free). They "
            "disagree often enough that the choice cannot be implicit."
            % (caller, sorted(_OI_CHANGE_BASES), basis))
    return _bounded_filter(
        tickers, _OI_CHANGE_BASES[basis], caller=caller,
        step="augment_with_oi_change",
        bounds=(("min_abs_pct", min_abs_pct, "absmin"),
                ("min_pct", min_pct, "min"),
                ("max_pct", max_pct, "max")),
        remedy="universe.augment_with_oi_change")


def filter_long_short_ratio(
    tickers: pd.DataFrame,
    min_long_account_pct: Optional[float] = None,
    max_long_account_pct: Optional[float] = None,
    min_long_account_pct_exclusive: Optional[float] = None,
) -> pd.DataFrame:
    """Keep instruments by how long-leaning the crowd is, in percentage points.

    Needs :func:`augment_with_long_short_ratio`. ``min_long_account_pct=60``
    means "at least 60 % of accounts holding this perpetual are long"; use
    ``min_long_account_pct_exclusive=60`` for the strict ``> 60 %`` wording.
    The two lower bounds are mutually exclusive so a generated YAML cannot
    quietly choose between inclusive and strict semantics.

    The bounds are on ``long_account_pct`` and not on ``long_short_ratio``
    because the share has a fixed scale: 50 is balanced and the value cannot
    leave 0..100, whereas the ratio is unbounded above and its balanced point
    (1.0) is a number a reader has to know. Both columns remain available for a
    spec that really wants the ratio.
    """
    if min_long_account_pct is not None and min_long_account_pct_exclusive is not None:
        raise ValueError(
            "filter_long_short_ratio accepts either min_long_account_pct (>=) or "
            "min_long_account_pct_exclusive (>) but not both"
        )
    return _bounded_filter(
        tickers, "long_account_pct", caller="filter_long_short_ratio",
        step="augment_with_long_short_ratio",
        bounds=(("min_long_account_pct", min_long_account_pct, "min"),
                ("min_long_account_pct_exclusive", min_long_account_pct_exclusive,
                 "strict_min"),
                ("max_long_account_pct", max_long_account_pct, "max")),
        remedy="universe.augment_with_long_short_ratio")


def filter_spread(
    tickers: pd.DataFrame,
    max_spread_bps: Optional[float] = None,
    min_spread_bps: Optional[float] = None,
) -> pd.DataFrame:
    """Keep instruments whose touch is tight enough to trade.

    Needs :func:`augment_with_spread`. ``max_spread_bps=5`` is "the round trip at
    the touch costs at most 5 basis points" — the screen "剔除低流動性的空氣幣"
    actually asks for, and one that ``filter_quote_volume`` cannot express: on a
    measured snapshot SNXXUSDT clears a $100m turnover floor at 11.0 bps.

    The bound is in basis points and the keyword says so, because the same
    threshold is written three ways in the wild (0.0005, 0.05 %, 5 bps) and two of
    them are silently 100x wrong. bps is the unit an execution desk quotes and the
    one the column is in.

    ``min_spread_bps`` is here for the opposite screen, and it is not symmetric
    padding: a *market-making* or mean-reversion selection wants the instruments
    where the spread is wide enough to be worth capturing, and without it that
    request has to be written as a ``long_when`` condition, which does not narrow
    the universe (see the note above :func:`filter_open_interest`).

    Instruments whose spread is unknown — no quotable touch, see
    :func:`augment_with_spread` — are dropped and counted, never kept. Their
    arithmetic value would be infinite, negative or 0.0 bps, and all three clear
    a ceiling.
    """
    return _bounded_filter(
        tickers, "spread_bps", caller="filter_spread",
        step="augment_with_spread",
        bounds=(("max_spread_bps", max_spread_bps, "max"),
                ("min_spread_bps", min_spread_bps, "min")),
        remedy="universe.augment_with_spread")


def filter_top_of_book(
    tickers: pd.DataFrame,
    min_top_of_book_usd: Optional[float] = None,
    max_top_of_book_usd: Optional[float] = None,
) -> pd.DataFrame:
    """Keep instruments with enough SIZE at the touch, in dollars.

    Needs :func:`augment_with_spread`. This is the second half of the liquidity
    question and it is genuinely separate from :func:`filter_spread`: a spread can
    be one tick wide with three cents behind it. ``TAKEUSDT`` on a measured
    snapshot quoted 7.8 bps — respectable — with **$0.03** at the best ask, so a
    spread-only screen keeps it and an operator discovers the depth at fill time.

    ``min_top_of_book_usd`` is the bound that matters; ``max_top_of_book_usd``
    exists for the symmetric case a size-limited strategy needs ("small enough
    that my order is not the whole book").

    The keyword says ``usd`` for the same reason ``filter_open_interest`` does:
    the underlying fields are quantities in the BASE coin, where 1,000 is a
    rounding error for a meme coin and more than the venue holds for BTC.
    """
    return _bounded_filter(
        tickers, "top_of_book_usd", caller="filter_top_of_book",
        step="augment_with_spread",
        bounds=(("min_top_of_book_usd", min_top_of_book_usd, "min"),
                ("max_top_of_book_usd", max_top_of_book_usd, "max")),
        remedy="universe.augment_with_spread")


# ---------------------------------------------------------------------------
# Per-instrument technical indicators on the cross-section
# ---------------------------------------------------------------------------
#
# ★ THE FRAME HAS ROWS PER INSTRUMENT; AN INDICATOR NEEDS ROWS PER BAR.
#
# That mismatch is why "Supertrend(10,3) bearish on H4 and H1 and M15" — the
# single most common shape in the selection corpus — was not expressible here at
# all, and why the substitution that shipped was ``top_losers(n=30)``: a 24-hour
# percentage standing in for an indicator, losing its period, its timeframe, its
# state and the difference between "downtrend" and "one bad hour". The spec
# validated, the run succeeded, and nothing in the output named the proxy.
#
# :func:`augment_with_indicator` closes it by taking a SECOND frame — the bars,
# one row per (instrument, timeframe, bar) — computing the indicator per
# instrument, and folding the result back onto the cross-section as one column.
#
# ★ THE INDICATOR IS COMPUTED HERE AND ITS PARAMETERS COME FROM THE SPEC.
#
# The alternative — have the capture pre-compute indicator VALUES and ship those
# in the bundle — was rejected. ``params: {period: 10}`` against a bundle that
# baked in 14 makes the spec a lie: it validates, it runs, it reports
# "Supertrend(10,3)" and it screened on something else. It also welds every
# bundle to one set of hyper-parameters, so "same market state, different spec"
# stops being a question anyone can ask.
#
# Carrying BARS instead has a second, sharper benefit: point-in-time correctness
# becomes a property of the DATA rather than a promise made by capture code. An
# unfinished candle has a ``close_time`` in the future and the bundle's own PIT
# gate drops it; a pre-computed indicator value carries no evidence of which bars
# went into it.


#: How :func:`augment_with_indicator` collapses the window into one number.
#:
#: ``window_bars: 1`` + ``agg: last`` is "right now". Everything else exists
#: because a screen says "in the last two hours", which on 15-minute bars is EIGHT
#: bars — and reading only the final bar answers a different question. On the
#: measured 15m series a Supertrend that flipped bearish two bars ago and back is
#: ``last=+1``, ``min=-1``: "is it bearish now" and "has it been bearish at all"
#: are both legitimate screens and they are not the same set of coins.
#:
#: ``any_negative`` / ``all_negative`` are the boolean forms, emitted as 1.0 / 0.0
#: so the ordinary comparators read them (``conditions.value_above(col, 0.5)``).
#: They are not redundant with ``min`` / ``max``: those return the indicator's own
#: units, which for a LEVEL indicator (a Supertrend line, a moving average) is a
#: price and never negative, so a spec that meant "was it ever below zero" and
#: wrote ``min`` would get a plausible price back.
_INDICATOR_AGGS = ("last", "min", "max", "mean", "any_negative", "all_negative")

#: Default multiple of the indicator's own period a series must carry.
#:
#: There are TWO different short-history failures and this handles the second one:
#:
#: 1. **The indicator cannot produce a value at all.** Every rolling indicator is
#:    NaN for its first ``period - 1`` bars. This one needs no heuristic and gets
#:    none — :func:`augment_with_indicator` refuses any instrument whose
#:    aggregated value comes out NaN, which is exact for every indicator and
#:    cannot be switched off.
#: 2. **The indicator produces a value that has not settled.** The RMA/EMA-seeded
#:    family (``supertrend``, ``adx``, ``rsi``, anything built on
#:    ``indicators.rma``) returns a number as soon as its window fills, but that
#:    number still carries the seed for a few more periods. No NaN appears, so
#:    only a length requirement catches it. Three periods is where the seed's
#:    influence falls below a rounding error.
#:
#: Why it is a settable parameter and not a constant: for a PURE rolling-window
#: indicator (``highest``, ``lowest``, ``donchian``, ``range_gain_pct``) there is
#: no seed and no settling, so ``period`` bars is not an approximation — it is the
#: exact answer, and demanding 3x would refuse a 3-month range screen on anything
#: listed less than nine months ago. Those are precisely the instruments such a
#: screen is looking for, so a constant here would have quietly inverted the
#: question. ``min_bars_multiple: 1`` is the correct value for that family and the
#: spec has to say so, which is what makes the choice auditable.
#:
#: Checked per ``(instrument, timeframe)`` and never as a frame-wide average: a
#: capture is usually complete for the majors and short for one new listing, and an
#: average hides exactly that row.
_INDICATOR_WARMUP_MULTIPLE = 3

#: Share of the cross-section the bars frame must cover before the join raises.
#:
#: 1.0 — total, unlike every other join in this module, and the reason is that a
#: bars capture's roster is DERIVED from the frame it will be joined onto (the
#: surviving prefix of the same pipeline), so a hole is never a market fact and
#: never a rate-budget casualty. It means the frame being screened is not the
#: frame the capture was planned from: a re-captured universe, a spec edited
#: between the capture and the run, or an indicator step moved earlier in the
#: pipeline than the one the roster was planned at.
_BARS_MIN_COVERAGE = 1.0


def _resolve_indicator(name: str):
    """Resolve an indicator NAME to a callable in :mod:`cyqnt_trd.blocks.indicators`.

    Deliberately not a dotted ``"<module>.<fn>"`` reference. Accepting one would
    make this block a second dispatch surface into the whole blocks package — one
    that is not behind the YAML interpreter's denylist, so ``indicator:
    data.fetch_klines`` would fetch during ``validate`` and
    ``indicator: strategy.register`` would mutate the process-wide plugin registry.
    It would also invert the layering: ``blocks.universe`` would have to import
    ``standard_bot.yaml_pipeline.interpreter.resolve_block``, i.e. a block
    importing the interpreter that calls it.

    One namespace, and it is the right one: an indicator is by definition
    something in ``blocks.indicators``.
    """
    from . import indicators as _indicators

    available = ", ".join(sorted(name for name in _indicators.__all__)[:12])
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            "augment_with_indicator: `indicator:` must be the NAME of a function "
            "in cyqnt_trd.blocks.indicators (e.g. 'supertrend'), got %r. Some "
            "available: %s, ..." % (name, available))
    name = name.strip()
    if "." in name or name.startswith("_"):
        raise ValueError(
            "augment_with_indicator: `indicator: %r` — give a BARE name, not a "
            "dotted reference and not a private one. This block resolves only "
            "inside cyqnt_trd.blocks.indicators on purpose: a dotted ref would "
            "open a second way into every block module, including the fetchers "
            "the YAML interpreter refuses by name. Some available: %s, ..."
            % (name, available))
    fn = getattr(_indicators, name, None)
    if fn is None or not callable(fn):
        raise ValueError(
            "augment_with_indicator: cyqnt_trd.blocks.indicators has no indicator "
            "%r. Some available: %s, ..." % (name, available))
    origin = getattr(fn, "__module__", "")
    if origin != "cyqnt_trd.blocks.indicators":
        raise ValueError(
            "augment_with_indicator: %r resolves to %s.%s, which is imported into "
            "the indicators module rather than defined there — typing aliases and "
            "helpers are callable but are not indicators."
            % (name, origin or "?", name))
    return fn


def _indicator_warmup_period(fn, params) -> int:
    """The longest integer window this indicator call will look back over.

    Read from the EFFECTIVE parameters — the caller's values merged over the
    signature's defaults — because the default is what actually runs when the spec
    is silent, and a warm-up assertion derived from the spec alone would pass a
    frame that is far too short for ``ichimoku`` (``senkou_b=52``) merely because
    the spec named no period.

    Integers only, and that is the discriminator rather than a name list:
    every window/period argument in this module's indicators is an int and every
    multiplier / threshold / deviation is a float, so ``supertrend(period=10,
    multiplier=3.0)`` yields 10 and not 3. Booleans are excluded because
    ``bool`` is an ``int`` in Python and a flag is not a lookback.
    """
    import inspect

    effective = {}
    try:
        signature = inspect.signature(fn)
    except (ValueError, TypeError):                  # pragma: no cover - builtins
        signature = None
    if signature is not None:
        for name, parameter in signature.parameters.items():
            if parameter.default is not inspect.Parameter.empty:
                effective[name] = parameter.default
    effective.update(params)
    periods = [int(value) for value in effective.values()
               if isinstance(value, int) and not isinstance(value, bool) and value >= 1]
    return max(periods) if periods else 1


def _bars_for_timeframe(bars, timeframe: str, *, caller: str):
    """The bars of ONE timeframe, keyed and sorted, or a refusal naming what IS here.

    An absent timeframe is a refusal and not an empty frame: the timeframe set a
    capture collects is the union of the spec's own indicator steps, so asking for
    one that is not there means the bundle and the spec disagree — and the
    alternative outcome, an all-NaN column, reads downstream as "no coin met the
    condition".
    """
    if bars is None or not isinstance(bars, pd.DataFrame):
        raise ValueError(
            "%s: the universe_bars source must be a pandas DataFrame, got %s"
            % (caller, type(bars).__name__))
    if bars.empty:
        raise ValueError(
            "%s: the universe_bars source is empty. Bars are fetched one request "
            "per (instrument, timeframe), so an empty frame means that fan-out "
            "collected nothing — check source_status for the 'universe_bars' node "
            "rather than reading this as 'no instrument has price history'."
            % caller)

    frame = bars.copy()
    key = next((name for name in ("instrument_id", "symbol")
                if name in frame.columns), None)
    if key is None:
        raise ValueError(
            "%s: the universe_bars source has no 'instrument_id' / 'symbol' "
            "column; it carries %s" % (caller, list(frame.columns)))
    if "timeframe" not in frame.columns:
        raise ValueError(
            "%s: the universe_bars source has no 'timeframe' column, so bars of "
            "different intervals cannot be told apart — every timeframe would be "
            "concatenated into one series and the indicator would be computed over "
            "a mixture. It carries %s." % (caller, list(frame.columns)))
    frame["symbol"] = frame[key].astype(str).str.upper()
    wanted = str(timeframe).strip()
    selected = frame[frame["timeframe"].astype(str).str.strip() == wanted]
    if selected.empty:
        present = sorted({str(value).strip()
                          for value in frame["timeframe"].dropna().unique()})
        raise ValueError(
            "%s: the universe_bars source carries no %r bars; it carries %s. The "
            "capture collects the UNION of the timeframes named by this spec's "
            "indicator steps, so this means the bundle was captured for a "
            "different spec — re-capture, or name a timeframe that is here."
            % (caller, wanted, present))

    # Chronological within each instrument, and by open_time rather than by the
    # frame's order: a bundle's rows are ordered by the PIT gate's sort key, which
    # is available_time, and a series handed to a rolling indicator out of order
    # produces numbers with no meaning and no error.
    time_column = next((name for name in ("open_time", "close_time", "event_time")
                        if name in selected.columns), None)
    if time_column is None:
        raise ValueError(
            "%s: the universe_bars source has no 'open_time' / 'close_time' "
            "column, so its bars cannot be put in order; it carries %s. A rolling "
            "indicator over unordered bars returns numbers rather than an error."
            % (caller, list(selected.columns)))
    return selected.sort_values(["symbol", time_column], kind="stable")


def _aggregate_indicator_window(values, agg: str, window_bars: int) -> float:
    """Collapse the last ``window_bars`` values of one instrument into one number.

    A NaN anywhere in the window makes the result NaN, in EVERY mode. That is not
    pandas' default (``Series.min()`` skips NaN, and ``(NaN < 0)`` is False, so
    ``any_negative`` would answer "no" for a series it could not read) and the
    default is the dangerous one: a NaN read as "the condition did not hold"
    removes the instrument from a bearish screen and from a bullish one alike,
    which is a bias no field of the output records. Unknown stays unknown, and the
    ranker drops unknown rows explicitly.
    """
    window = values.iloc[-int(window_bars):]
    if window.isna().any() or window.empty:
        return float("nan")
    if agg == "last":
        return float(window.iloc[-1])
    if agg == "min":
        return float(window.min())
    if agg == "max":
        return float(window.max())
    if agg == "mean":
        return float(window.mean())
    if agg == "any_negative":
        return 1.0 if bool((window < 0).any()) else 0.0
    if agg == "all_negative":
        return 1.0 if bool((window < 0).all()) else 0.0
    raise AssertionError("unknown agg %r" % (agg,))    # pragma: no cover - guarded


def augment_with_indicator(
    tickers: pd.DataFrame,
    bars_df: Optional[pd.DataFrame] = None,
    *,
    indicator: str,
    timeframe: str,
    agg: str = "last",
    window_bars: int = 1,
    input: str = "auto",
    output: Optional[int] = None,
    column: Optional[str] = None,
    min_bars_multiple: int = _INDICATOR_WARMUP_MULTIPLE,
    limit: int = 200,
    market_type: str = "futures",
    **indicator_params,
) -> pd.DataFrame:
    """Compute one technical indicator per instrument and join it as one column.

    This is the block that lets a SELECTION spec say what a chart says::

        - block: universe.augment_with_indicator
          with: [universe_bars]
          params:
            indicator: supertrend      # a name in cyqnt_trd.blocks.indicators
            timeframe: "4h"            # which bars, out of the captured set
            output: 1                  # supertrend returns (level, direction)
            as: st_dir_4h              # the column to write
            agg: min                   # over the window, not just the last bar
            window_bars: 1
            period: 10                 # everything else goes to the indicator
            multiplier: 3.0

    Three such steps plus ``all_of([...value_below 0 x3])`` is "bearish on 4h and
    1h and 15m at the same time" — no new combinator, because the columns are just
    columns once they are on the frame.

    Parameters
    ----------
    tickers
        The running cross-section, one row per instrument.
    bars_df
        ``BarFrame@1.0`` rows for the instruments in *tickers*: one row per
        (instrument, timeframe, bar). This is the replay-safe / YAML path
        (``with: [universe_bars]``) and no I/O happens on it. A direct Python
        caller may omit it, in which case the roster is *this frame's own*
        instruments and the bars are fetched live — which means the fan-out weight
        ceiling applies to the frame, so a 727-row cross-section raises. That is
        the intended failure: narrow first.
    indicator, timeframe
        Required. ``indicator`` resolves inside ``blocks.indicators`` only — see
        :func:`_resolve_indicator`.
    agg, window_bars
        How the last ``window_bars`` bars collapse to the one number a
        cross-sectional row can hold. See :data:`_INDICATOR_AGGS`.
    input
        ``"auto"`` (default) hands the indicator the whole OHLCV frame when its
        first parameter is named ``df`` and the ``close`` series otherwise —
        the SAME detection the YAML interpreter uses for ``signals.indicators``,
        shared as :func:`blocks._utils.first_param_is_df` so the two cannot
        disagree. Name a column (``high``, ``low``, ``volume``) to override; that
        is what ``indicators.lowest`` needs, since "the 3-month low" is a low and
        not a close.
    output, column
        Which component of a tuple-returning indicator (``supertrend`` ->
        ``(level, direction)``, ``macd`` -> three lines) or which column of a
        frame-returning one (``ichimoku``). Same semantics and the same code as
        ``output:`` / ``column:`` in ``signals.indicators``.
    min_bars_multiple
        Settling margin, in multiples of the indicator's own longest period. See
        :data:`_INDICATOR_WARMUP_MULTIPLE`: leave it at 3 for the RMA/EMA-seeded
        family, set it to 1 for a pure rolling window such as
        ``indicators.range_gain_pct``.
    as
        The output column name, passed through ``**indicator_params`` because
        ``as`` is a Python keyword and cannot be a parameter name. Optional;
        defaults to ``"<indicator>_<timeframe>"``. An existing column is never
        overwritten — see below.
    limit, market_type
        Only used on the live-fetch path (``bars_df=None``).

    Everything else is forwarded to the indicator, so ``period`` / ``multiplier``
    / ``tenkan`` mean exactly what they mean in ``blocks.indicators``.

    Refusals, and what each one stops
    ---------------------------------
    * **Writing over an existing column.** Two steps that both land in
      ``st_dir_4h`` — the same indicator and timeframe at two periods, which is a
      normal thing to want — would leave the frame holding the second and the spec
      claiming both. Raises, and says to give ``as:`` a distinct name.
    * **An instrument with no bars.** :data:`_BARS_MIN_COVERAGE` is total; see
      that note.
    * **A series shorter than the indicator's warm-up**, and separately **an
      instrument whose value comes out NaN**. See
      :data:`_INDICATOR_WARMUP_MULTIPLE` for why those are two different checks.
      This pair matters most, because the failure mode is invisible: NaN reads as
      "condition not met" for long and short alike.
    """
    caller = "augment_with_indicator"
    keyed = _with_symbol_column(tickers)
    output_column = str(indicator_params.pop(
        "as", "%s_%s" % (str(indicator).strip(), str(timeframe).strip())))
    if not output_column:
        raise ValueError("%s: `as:` must be a non-empty column name" % caller)
    if output_column in keyed.columns:
        raise ValueError(
            "%s: the frame already has a %r column, and this step would overwrite "
            "it. Two indicator steps writing one name leaves the frame holding the "
            "second while the spec claims both — give this one a distinct `as:`."
            % (caller, output_column))
    if agg not in _INDICATOR_AGGS:
        raise ValueError(
            "%s: agg must be one of %s, got %r. 'last' is now, 'min' over a window "
            "is 'at any point during it' — they select different instruments."
            % (caller, list(_INDICATOR_AGGS), agg))
    window_bars = int(window_bars)
    if window_bars < 1:
        raise ValueError(
            "%s: window_bars must be >= 1, got %d — a zero-bar window has nothing "
            "to aggregate" % (caller, window_bars))

    min_bars_multiple = int(min_bars_multiple)
    if min_bars_multiple < 1:
        raise ValueError(
            "%s: min_bars_multiple must be >= 1, got %d — an indicator cannot "
            "report a value over fewer bars than its own period"
            % (caller, min_bars_multiple))

    fn = _resolve_indicator(indicator)
    if bars_df is None:
        # Same convenience-with-a-ceiling contract as augment_with_open_interest:
        # a direct Python caller gets a live fan-out over this frame's own symbols,
        # and the weight ceiling turns "I forgot to narrow first" into a refusal
        # rather than a rate-limit ban. The YAML path never reaches here — the
        # interpreter refuses a step that omits `with: [universe_bars]`.
        bars_df = _data.fetch_klines_cross_section(
            keyed["symbol"].astype(str).str.upper().tolist(),
            timeframes=[timeframe], limit=limit, market_type=market_type)
    selected = _bars_for_timeframe(bars_df, timeframe, caller=caller)

    warmup_period = _indicator_warmup_period(fn, indicator_params)
    required_bars = min_bars_multiple * warmup_period + window_bars - 1

    universe_symbols = set(keyed["symbol"].astype(str).str.upper())
    values: dict = {}
    short: list = []
    unresolved: list = []
    for symbol, group in selected.groupby("symbol", sort=False):
        if symbol not in universe_symbols:
            # A bars capture may legitimately cover more than the frame it is
            # joined to (the roster was planned before a later filter ran); paying
            # to compute an indicator for a row that will not be joined is not.
            continue
        if len(group) < required_bars:
            short.append((symbol, len(group)))
            continue
        series = _indicator_input(group, fn, input, caller=caller)
        try:
            raw = fn(series, **indicator_params)
        except TypeError as exc:
            raise ValueError(
                "%s: indicators.%s rejected the call: %s. Every param other than "
                "%s is forwarded to the indicator, so check the spelling against "
                "its signature."
                % (caller, indicator, exc,
                   "indicator/timeframe/agg/window_bars/input/output/column/as")
            ) from exc
        from ._utils import IndicatorShapeError, select_indicator_component

        try:
            reduced = select_indicator_component(
                raw, ref="indicators.%s" % indicator, output=output, column=column)
        except IndicatorShapeError as exc:
            raise ValueError("%s: %s" % (caller, exc)) from exc
        value = _aggregate_indicator_window(reduced, agg, window_bars)
        if value != value:                             # NaN, whatever produced it
            unresolved.append((symbol, len(group)))
            continue
        values[symbol] = value

    if short:
        raise ValueError(
            "%s: %d instrument(s) have fewer than %d %s bars, which indicators.%s "
            "needs before its value settles (min_bars_multiple=%d x its longest "
            "period %d, plus the %d-bar window): e.g. %s. Those rows would carry "
            "NaN, and a NaN is read downstream as 'the condition did not hold' — "
            "for a BEARISH screen and a BULLISH one alike, so the freshest "
            "listings would silently leave both. Ask the capture for more bars "
            "(limit=), shorten the period, or — for a pure rolling window with no "
            "seed to settle, such as indicators.range_gain_pct — set "
            "min_bars_multiple: 1."
            % (caller, len(short), required_bars, timeframe, indicator,
               min_bars_multiple, warmup_period, window_bars,
               ["%s=%d" % item for item in sorted(short)[:5]]))
    if unresolved:
        raise ValueError(
            "%s: indicators.%s came out NaN over the last %d %s bar(s) for %d "
            "instrument(s) even though each has at least %d bars: e.g. %s. The "
            "length check passed, so this is the indicator itself declining to "
            "produce a value — a gap in the captured series, a non-positive price, "
            "or a period longer than the aggregation window can see. It is refused "
            "rather than joined: NaN in this column reads as 'the condition did "
            "not hold' for both directions, so those instruments would leave a "
            "bearish screen and a bullish one without appearing in either."
            % (caller, indicator, window_bars, timeframe, len(unresolved),
               required_bars, ["%s=%d" % item for item in sorted(unresolved)[:5]]))

    join = pd.DataFrame({"symbol": list(values), output_column: list(values.values())})
    _require_join_coverage(keyed, join, caller=caller, node="universe_bars",
                           floor=_BARS_MIN_COVERAGE, cause=_BARS_COVERAGE_CAUSE)
    return keyed.merge(join, on="symbol", how="left")


_BARS_COVERAGE_CAUSE = (
    "A bars capture's roster is DERIVED from the surviving prefix of this very "
    "pipeline, so a hole is not a market fact and not a rate-budget casualty — it "
    "means the frame being screened is not the frame the capture was planned "
    "from. Either the universe was re-captured without the bars, or an indicator "
    "step moved to a point in the pipeline where more instruments are still "
    "alive than when the roster was planned."
)


def _indicator_input(group, fn, input: str, *, caller: str):
    """What to pass as the indicator's first argument for one instrument's bars."""
    from ._utils import ensure_df, first_param_is_df

    if input == "auto":
        if first_param_is_df(fn):
            return ensure_df(group)
        input = "close"
    frame = ensure_df(group, required=())
    name = str(input).strip().lower()
    if name == "df":
        return ensure_df(group)
    if name not in frame.columns:
        raise ValueError(
            "%s: input=%r is not a column of the bars frame (it has %s). Use 'df' "
            "for a whole-frame indicator, or a bar column — 'low' is what "
            "indicators.lowest needs, because a three-month LOW is a low and not "
            "a close." % (caller, input, sorted(frame.columns)[:12]))
    return pd.to_numeric(frame[name], errors="coerce").astype(float)


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

    def filter_quote_suffix(
        self, suffix: Union[str, Sequence[str]] = "USDT", exclude: bool = False
    ) -> "UniverseFilter":
        """Keep symbols quoted in *suffix* — see :func:`filter_quote_suffix`."""
        self.df = filter_quote_suffix(self.df, suffix, exclude)
        return self

    def with_contract_meta(
        self, contract_meta_df: Optional[pd.DataFrame] = None, **kwargs
    ) -> "UniverseFilter":
        """Join the listing registry — see :func:`augment_with_contract_meta`."""
        self.df = augment_with_contract_meta(self.df, contract_meta_df, **kwargs)
        return self

    def filter_underlying_type(
        self,
        include: Optional[Union[str, Sequence[str]]] = None,
        exclude: Optional[Union[str, Sequence[str]]] = None,
    ) -> "UniverseFilter":
        """Keep/drop by asset class — see :func:`filter_underlying_type`."""
        self.df = filter_underlying_type(self.df, include, exclude)
        return self

    def filter_sub_type(
        self,
        include: Optional[Union[str, Sequence[str]]] = None,
        exclude: Optional[Union[str, Sequence[str]]] = None,
    ) -> "UniverseFilter":
        """Keep/drop by sector tag — see :func:`filter_sub_type`."""
        self.df = filter_sub_type(self.df, include, exclude)
        return self

    def filter_crypto_only(self) -> "UniverseFilter":
        """Keep only ``underlying_type == COIN`` — see :func:`filter_crypto_only`."""
        self.df = filter_crypto_only(self.df)
        return self

    def with_open_interest(
        self, oi_df: Optional[pd.DataFrame] = None
    ) -> "UniverseFilter":
        """Join current open interest — see :func:`augment_with_open_interest`.

        Omitting *oi_df* fans out over the CURRENT frame, so narrow before
        chaining this: see :data:`cyqnt_trd.blocks.data.FAN_OUT_MAX_SYMBOLS`.
        """
        self.df = augment_with_open_interest(self.df, oi_df)
        return self

    def with_oi_change(
        self, oi_hist_df: Optional[pd.DataFrame] = None, **kwargs
    ) -> "UniverseFilter":
        """Join the recent open-interest change — see :func:`augment_with_oi_change`."""
        self.df = augment_with_oi_change(self.df, oi_hist_df, **kwargs)
        return self

    def with_long_short_ratio(
        self, ls_df: Optional[pd.DataFrame] = None
    ) -> "UniverseFilter":
        """Join crowd positioning — see :func:`augment_with_long_short_ratio`."""
        self.df = augment_with_long_short_ratio(self.df, ls_df)
        return self

    def filter_open_interest(
        self,
        min_notional_usd: Optional[float] = None,
        max_notional_usd: Optional[float] = None,
    ) -> "UniverseFilter":
        """Bound dollar open interest — see :func:`filter_open_interest`."""
        self.df = filter_open_interest(self.df, min_notional_usd, max_notional_usd)
        return self

    def filter_oi_change(
        self,
        min_abs_pct: Optional[float] = None,
        min_pct: Optional[float] = None,
        max_pct: Optional[float] = None,
        *,
        basis: str = "notional",
    ) -> "UniverseFilter":
        """Bound the open-interest change — see :func:`filter_oi_change`."""
        self.df = filter_oi_change(self.df, min_abs_pct, min_pct, max_pct,
                                   basis=basis)
        return self

    def filter_long_short_ratio(
        self,
        min_long_account_pct: Optional[float] = None,
        max_long_account_pct: Optional[float] = None,
        min_long_account_pct_exclusive: Optional[float] = None,
    ) -> "UniverseFilter":
        """Bound crowd skew — see :func:`filter_long_short_ratio`."""
        self.df = filter_long_short_ratio(
            self.df,
            min_long_account_pct=min_long_account_pct,
            max_long_account_pct=max_long_account_pct,
            min_long_account_pct_exclusive=min_long_account_pct_exclusive,
        )
        return self

    def with_spread(
        self, book_df: Optional[pd.DataFrame] = None
    ) -> "UniverseFilter":
        """Join the top of the book — see :func:`augment_with_spread`.

        Omitting *book_df* costs ONE whole-market request whatever the frame's
        length, so unlike :meth:`with_open_interest` this may be chained anywhere.
        """
        self.df = augment_with_spread(self.df, book_df)
        return self

    def filter_spread(
        self,
        max_spread_bps: Optional[float] = None,
        min_spread_bps: Optional[float] = None,
    ) -> "UniverseFilter":
        """Bound the quoted spread in bps — see :func:`filter_spread`."""
        self.df = filter_spread(self.df, max_spread_bps, min_spread_bps)
        return self

    def filter_top_of_book(
        self,
        min_top_of_book_usd: Optional[float] = None,
        max_top_of_book_usd: Optional[float] = None,
    ) -> "UniverseFilter":
        """Bound the dollar size at the touch — see :func:`filter_top_of_book`."""
        self.df = filter_top_of_book(self.df, min_top_of_book_usd,
                                     max_top_of_book_usd)
        return self

    def symbols(self) -> List[str]:
        # Either key vocabulary, like every other symbol-keyed entry point here:
        # a chain that starts from a bundle universe (``instrument_id``) and never
        # happens to pass through a step that injects ``symbol`` used to filter
        # correctly and then refuse to tell anyone what it had selected.
        return [str(s).upper()
                for s in _with_symbol_column(self.df)["symbol"].tolist()]

    def to_frame(self) -> pd.DataFrame:
        return self.df.copy()
