"""Load, validate, and register a YAML strategy spec.

The validator does two things:

1. **Static checks** — required keys present, every referenced block resolves
   to a real whitelisted callable, entry declares at least one direction, and
   live mode carries its safety guards.
2. **Dry-run** — compiles the spec into ``make_signals`` and executes it on a
   synthetic OHLCV frame. This is the important one: it catches the *exact*
   class of signature / type / arity bugs (wrong arg count, tuple-vs-series
   mixups, unknown params) at ``validate`` time, before a single real order.

``validate`` therefore gives the frontend / bdp-ai-trading-bot a hard gate:
a spec that validates is structurally runnable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from .interpreter import (
    COLUMN_REQUIRES_SOURCE,
    INDICATOR_KEYS,
    SELECTION_KEYS,
    UNIVERSE_STEP_KEYS,
    SpecError,
    build_make_signals,
    build_selection_fn,
    eval_indicator,
    eval_node,
    resolve_block,
)
from .vocabulary import DATA_SECTIONS, synthetic_columns

VALID_MODES = {"backtest", "paper", "live"}

#: Keys allowed under ``data:``. Closed on purpose — a typo'd optional key used
#: to be accepted in silence, so ``data: {derivative: {...}}`` looked like it had
#: turned on funding data and had no effect whatsoever.
DATA_KEYS = frozenset(
    {"symbol", "market_type", "primary", "source", "htf"} | set(DATA_SECTIONS)
)
SIZING_KEYS = frozenset({"size"})
#: Exactly the keys the three engines read out of ``exit_cfg`` — verified by
#: grepping ``cfg.get("...")`` in blocks/strategy.py, vectorized_backtest.py,
#: runner.py and python_live_paper_session.py. Anything else was a typo that the
#: engine silently defaulted past (``stop_pctt`` cost a real stop-loss).
EXIT_KEYS = frozenset({
    "type", "max_bars", "stop_pct", "tp_pct", "atr_period", "stop_mult",
    "tp_mult", "trail_mult", "period", "ma_type",
})
VALID_EXIT_TYPES = {
    "time_only",
    "pct_stop_tp",
    "atr_stop_tp",
    "atr_trailing_stop",
    "ma_cross_exit",
    "opposite_signal",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_spec(path: str) -> Dict[str, Any]:
    import yaml

    text = Path(path).read_text(encoding="utf-8")
    spec = yaml.safe_load(text)
    if not isinstance(spec, dict):
        raise SpecError(f"{path}: top-level YAML must be a mapping")
    return spec


# ---------------------------------------------------------------------------
# Synthetic data for the dry-run
# ---------------------------------------------------------------------------


def _collect_period_hints(obj: Any, acc: List[int]) -> None:
    """Walk the spec collecting int values of period/window/lookback params."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (int, float)) and any(
                tok in str(key).lower() for tok in ("period", "window", "lookback", "bars")
            ):
                try:
                    acc.append(int(value))
                except (TypeError, ValueError):
                    pass
            else:
                _collect_period_hints(value, acc)
    elif isinstance(obj, list):
        for item in obj:
            _collect_period_hints(item, acc)


def _synthetic_df(spec: Dict[str, Any]):
    import numpy as np
    import pandas as pd

    hints: List[int] = []
    _collect_period_hints(spec, hints)
    max_period = max(hints) if hints else 50
    n = max(300, max_period * 3 + 60)

    # Deterministic gentle wave + drift so crossover/threshold conditions
    # actually trigger during the dry-run (exercises the real branches).
    idx = np.arange(n)
    base = 100.0 + idx * 0.05 + 8.0 * np.sin(idx / 15.0) + 3.0 * np.sin(idx / 4.0)
    close = base
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    volume = 1000.0 + 50.0 * np.abs(np.sin(idx / 7.0))
    step_ms = 3_600_000  # 1h bars; only relative spacing matters
    open_time = 1_700_000_000_000 + idx * step_ms
    close_time = open_time + step_ms - 1

    df = pd.DataFrame(
        {
            "open_time": open_time,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "close_time": close_time,
            "trades": (volume / 10).astype(int),
        }
    )
    df["timestamp"] = df["close_time"]

    # Stand-in HTF columns so conditions referencing them don't KeyError.
    for htf in (spec.get("data", {}) or {}).get("htf", []) or []:
        period = int(htf.get("sma_period", 200))
        tf = htf.get("interval", "4h")
        col = f"_htf_{tf}_sma_{period}"
        df[col] = df["close"].rolling(window=min(period, n), min_periods=1).mean()

    # Columns from the data sources this spec DECLARED. Only the declared ones:
    # fabricating everything would let a spec dry-run green and then meet an
    # absent column at runtime, which is the failure mode worth preventing.
    for column in synthetic_columns(spec):
        df[column] = _synthetic_series(column, idx, close, volume, np)
    return df


def _synthetic_series(column: str, idx, close, volume, np):
    """A plausible stand-in for one derived column.

    Plausible matters: a funding-rate column of zeros makes every
    ``funding_rate_state`` branch dead during the dry-run, so the dry-run stops
    proving anything. These oscillate through their real sign and magnitude
    ranges so the conditions built on them actually fire.
    """
    if column == "funding_rate":
        return 0.0001 * np.sin(idx / 11.0)
    if column == "funding_rate_bps":
        return 1.0 * np.sin(idx / 11.0)
    if column == "mark_price":
        return close * (1.0 + 0.0002 * np.sin(idx / 6.0))
    if column == "open_interest":
        return 50_000.0 + 5_000.0 * np.sin(idx / 19.0) + idx * 2.0
    if column == "open_interest_value":
        return (50_000.0 + 5_000.0 * np.sin(idx / 19.0) + idx * 2.0) * close
    if column == "oi_change_bps":
        return 40.0 * np.sin(idx / 9.0)
    if column.endswith("_count"):
        return np.abs(np.round(8.0 + 6.0 * np.sin(idx / 5.0)))
    if column.endswith("_qty"):
        return np.abs(2.0 + 1.5 * np.sin(idx / 7.0))
    if column == "net_liq_notional_usd":
        return 250_000.0 * np.sin(idx / 8.0)
    if column == "liq_imbalance_ratio":
        return 0.5 + 0.4 * np.sin(idx / 13.0)
    if column.endswith("_notional_usd"):
        return np.abs(300_000.0 + 250_000.0 * np.sin(idx / 8.0))
    # Unknown-but-declared: a bounded positive series is the least surprising
    # thing a block can be handed, and it will not divide by zero.
    return 1.0 + 0.5 * np.abs(np.sin(idx / 10.0))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


#: Real Binance USDⓈ-M instrument -> its real ``exchangeInfo`` classification,
#: for the non-crypto part of the stand-in cross-section.
#:
#: Real names and real classifications, not invented ones, so a reader can check
#: the stand-in against the venue. ``ALLUSDT`` is here specifically because it is
#: an ``INDEX``: it is one of the two contracts that "exclude TradFi" spelled as a
#: blacklist keeps and ``filter_crypto_only``'s whitelist drops, i.e. the case
#: that makes the two spellings different answers rather than styles.
_SYNTHETIC_NON_CRYPTO = {
    "SNDKUSDT": ("EQUITY", ["TradFi"]),
    "SOXLUSDT": ("EQUITY", ["TradFi", "ETF"]),
    "ALLUSDT": ("INDEX", []),
}

#: The one base token in the stand-in that carries SEVERAL sector tags.
#:
#: A real one (``FOLKSUSDT`` is tagged ``['Alpha', 'DeFi']`` on the live venue),
#: and load-bearing: a 2-element tag array is what raises "truth value of an array
#: with more than one element is ambiguous" in the candidate builder, while a
#: 1-element array does not. Only 5 of 727 live contracts are multi-tagged, so
#: without one here the dry-run would miss the only shape that breaks.
_SYNTHETIC_MULTI_TAG_BASE = "FOLKS"

#: Real Alpha / AI instruments, appended because no major carries either tag.
#:
#: Every base token above is a Layer-1 / PoW / Meme / Payment name — the venue
#: tags none of them ``Alpha`` or ``AI``, and those two are the sectors a
#: derivatives screen actually narrows to (Alpha 71 + AI 57 of 727 on the frozen
#: snapshot, and the roster the committed fan-out fixture was captured over). A
#: stand-in without them made ``filter_sub_type(include=[Alpha, AI])`` empty the
#: universe during the dry-run, so every later step — the whole open-interest
#: chain — was never reached and ``validate`` answered "no candidates", which
#: proves nothing.
#:
#: Real names with the venue's real tags, like :data:`_SYNTHETIC_NON_CRYPTO`.
#: ``CLANKERUSDT`` carries BOTH tags on the live venue and is the one instrument
#: an ``include=[Alpha, AI]`` set intersection must not double-count.
_SYNTHETIC_SECTOR_BASES = {
    "US": ["Alpha"], "UB": ["Alpha"], "ALLO": ["AI"], "TAG": ["AI"],
    "CLANKER": ["Alpha", "AI"],
}

#: Crypto base token -> its real sector tags, for the crypto part of the stand-in.
_SYNTHETIC_CRYPTO_TAGS = {
    "BTC": ["PoW"], "ETH": ["Layer-1"], "SOL": ["Layer-1"], "BNB": ["Layer-1"],
    "XRP": ["Payment"], "DOGE": ["Meme"], "ADA": ["Layer-1"], "AVAX": ["Layer-1"],
    "LINK": ["Infrastructure"], "DOT": ["Layer-1"], "TON": ["Layer-1"],
    "TRX": ["Layer-1"], _SYNTHETIC_MULTI_TAG_BASE: ["Alpha", "DeFi"],
    **_SYNTHETIC_SECTOR_BASES,
}


def _synthetic_universe(n: int = 12):
    """A stand-in universe frame: rows are symbols, not bars.

    Its column names are exactly a real cross-section's, no more and no less.
    ``tests/standard_bot/test_selection_fixture_replay.py`` asserts that set
    equality against the frozen Binance 24h-ticker frame, and that assertion is
    what this function is for: one extra name here makes ``validate`` strictly
    more permissive than every real run. It used to offer ``symbol`` and
    ``quote_volume``, which a ``cyqnt.input/v1`` universe (a ``RankFrame@1.0``,
    keyed on ``instrument_id`` and carrying Binance's camelCase) does not have —
    so ``score: quote_volume`` and ``universe.exclude_symbols`` both dry-ran green
    and then raised on every bundle, live or replayed.

    The VALUES are chosen for the same reason, because a dry-run over rows nothing
    can match proves as little as one over the wrong columns:

    * ``priceChangePercent`` straddles zero, so ``universe.top_gainers`` /
      ``top_losers`` / ``filter_change_pct`` all have rows to select. (It is here
      at all because a real 24h ticker always has it: without it every spec using
      one of those three failed validation with ``DataFrame missing
      'priceChangePercent'``, which reads as the author's bug and was ours.)
    * two rows are USDC-quoted, because a venue lists several quotes per token —
      the situation ``dedupe_by: base_asset`` and ``universe.filter_quote_suffix``
      both exist for. With one quote per token neither was exercised, and "exclude
      the USDC pairs" dry-ran as a filter that matched no row, which is
      indistinguishable from the typo ``USDCC``.
    * :data:`_SYNTHETIC_NON_CRYPTO` puts three NON-crypto instruments in the
      frame, because a real Binance futures cross-section is 20% tokenised
      equities and indices (131 EQUITY + 8 COMMODITY + 2 INDEX of 727 on one
      snapshot). A crypto-only stand-in made ``universe.filter_crypto_only`` and
      ``filter_underlying_type`` dry-run as steps that drop nothing — the same
      output a missing step gives, so validate could not tell the two apart.

    Deliberately NOT here: the Square mention / sentiment columns, and every
    contract-metadata column. Those belong to the ``ticker_rank`` and
    ``contract_meta`` sources and reach a spec only through
    ``universe.augment_with_news`` / ``augment_with_contract_meta`` — see
    :func:`_synthetic_ticker_rank` and :func:`_synthetic_contract_meta`.
    """
    import numpy as np
    import pandas as pd

    bases = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX",
             "LINK", "DOT", "TON", "TRX"][:n]
    instruments = ["%sUSDT" % base for base in bases]
    instruments += ["%sUSDC" % base for base in bases[:2]]
    # The multi-tag coin (``Alpha,DeFi`` on the live venue) and the non-crypto
    # rows are appended unconditionally: they are the two situations the metadata
    # filters exist for, and ``n`` must not be able to slice them away.
    instruments += ["%sUSDT" % _SYNTHETIC_MULTI_TAG_BASE]
    instruments += list(_SYNTHETIC_NON_CRYPTO)
    # LAST, and contiguously. The derivative stand-ins cycle their values with
    # position in the fan-out roster (see :func:`_synthetic_open_interest`), and
    # six consecutive positions is what puts rows on BOTH sides of the $5m floor
    # and of the ±20% change *within the sector subset a screen narrows to* —
    # which is the only place those filters can be observed at all.
    # ``test_universe_derivatives.py`` asserts that property rather than trusting
    # this comment.
    instruments += ["%sUSDT" % base for base in _SYNTHETIC_SECTOR_BASES]
    idx = np.arange(len(instruments))

    # -11%..+11%, alternating sign: exercises gainers and losers in one frame.
    change_pct = np.round(11.0 * np.cos(idx / 1.7) * np.where(idx % 2, -1.0, 1.0), 3)
    last = 100.0 + idx * 3.0
    open_ = last / (1.0 + change_pct / 100.0)
    turnover = 1e9 / (idx + 1)
    close_time = 1_700_000_000_000
    return pd.DataFrame({
        "instrument_id": instruments,
        "priceChange": last - open_,
        "priceChangePercent": change_pct,
        "weightedAvgPrice": (open_ + last) / 2.0,
        "lastPrice": last,
        "openPrice": open_,
        "highPrice": np.maximum(open_, last) * 1.01,
        "lowPrice": np.minimum(open_, last) * 0.99,
        "volume": turnover / last,
        "quoteVolume": turnover,
        # A string, because ``blocks.data.fetch_24h_tickers`` coerces nine named
        # columns and leaves the rest as JSON delivered them. A spec that ranks on
        # this one should meet a string here, not only at runtime.
        "lastQty": [str(2 + int(value)) for value in idx],
        "count": 90_000 - idx * 1_500,
        "firstId": 1_000_000 + idx * 5_000,
        "lastId": 1_090_000 + idx * 5_000,
        "openTime": close_time - 86_400_000,
        "closeTime": close_time,
        "event_time": close_time,
        "available_time": close_time,
    })


def _synthetic_ticker_rank(universe):
    """The stand-in ``ticker_rank`` source, keyed the way Square keys it.

    Modelled on the frozen fixture's frame: ``instrument_id`` here is a BASE TOKEN
    (``BTC``, not ``BTCUSDT``), one row per token, and sentiment arrives as raw
    counts with no ratio. The counts matter — ``augment_with_news`` derives
    ``news_bull_ratio`` from them, and a stand-in that pre-computed the ratio
    skipped that derivation, i.e. skipped the exact path a real bundle takes.

    Ratios straddle 0.45 / 0.55 so the ``long_when`` / ``short_when`` rules a
    selection spec typically writes actually fire during the dry-run.
    """
    import numpy as np
    import pandas as pd

    from cyqnt_trd.blocks.news_feed import base_token

    tokens = list(dict.fromkeys(
        base_token(str(value)) for value in universe["instrument_id"]))
    idx = np.arange(len(tokens))
    bull = 0.5 + 0.45 * np.sin(idx / 2.0)
    mentions = (500 - idx * 35).astype(float)
    return pd.DataFrame({
        "instrument_id": tokens,
        "rank": idx + 1,
        "mention_count": mentions,
        "unique_authors": mentions / 4.0,
        "total_engagement": mentions * 3.0,
        "bullish_count": np.round(mentions * bull),
        "bearish_count": np.round(mentions * (1.0 - bull)),
        "neutral_count": np.round(mentions * 0.1),
        "source_id": "synthetic.ticker_rank",
        "event_time": 1_700_000_000_000,
        "available_time": 1_700_000_000_000,
    })


def _synthetic_funding(universe):
    """Canonical multi-symbol MetricFrame used by the selection dry-run.

    Supplying this explicitly is a safety property: validation must exercise
    ``with: [funding]`` without letting ``augment_with_funding`` fall back to a
    live Binance request.
    """
    import numpy as np
    import pandas as pd

    symbols = universe["instrument_id"].astype(str).tolist()
    idx = np.arange(len(symbols), dtype=float)
    return pd.DataFrame({
        "instrument_id": symbols,
        "metric": "funding_rate",
        "value": 0.00005 + idx * 0.00001,
        "unit": "ratio",
        "source_id": "synthetic.funding",
        "event_time": 1_700_000_000_000,
        "available_time": 1_700_000_000_000,
    })


#: The three settlement intervals Binance actually publishes, and the share of
#: the market on each (measured 2026-08-02: 443 / 296 / 4 of 743 contracts).
#:
#: The stand-in cycles all three because ONE interval makes the annualisation a
#: constant multiplier — the ranking would then be identical to ``fundingRatePct``
#: and a dry-run could not tell ``score: fundingRateApr`` from ``score:
#: fundingRatePct``, which is the whole distinction the column exists to draw.
_SYNTHETIC_FUNDING_INTERVALS = (8, 4, 1)


def _synthetic_funding_info(universe):
    """The stand-in ``funding_info`` source: each contract's settlement schedule.

    Supplied explicitly for the same safety reason as :func:`_synthetic_funding`,
    plus a sharper one: ``augment_with_funding`` has no live fallback for this
    argument at all, so a spec reaching validate without it does not fetch — it
    silently gets NaN, and the point of the stand-in is that the dry-run computes
    real numbers.

    Binance's own column spelling, because a bundle delivers the canonical one
    (the node's ``column_map``) — a vendor-shaped stand-in exercises the block's
    alias table from the Python side while the frozen fixture covers the other.

    Two properties are load-bearing:

    * the intervals CYCLE (see :data:`_SYNTHETIC_FUNDING_INTERVALS`), so a
      dry-run of ``score: fundingRateApr`` produces a genuinely different order
      from ``score: fundingRatePct``. A stand-in on one interval would let a spec
      that meant to annualise, and a spec that forgot to, validate identically.
    * coverage is TOTAL, and that is not laziness. The real frame's only holes are
      the dated delivery contracts, and :func:`_synthetic_universe` contains no
      dated contract — so a hole here would model a row the stand-in universe does
      not have, and would drop a live instrument from the dry-run's ranking for a
      reason no real capture reproduces. The 90 % floor and the NaN-per-row path
      are pinned by unit tests against hand-built frames instead.
    """
    import pandas as pd

    instruments = [str(value) for value in universe["instrument_id"]]
    rows = []
    for position, instrument in enumerate(instruments):
        hours = _SYNTHETIC_FUNDING_INTERVALS[
            position % len(_SYNTHETIC_FUNDING_INTERVALS)]
        rows.append({
            "symbol": instrument,
            "fundingIntervalHours": hours,
            # The venue clamps tightly on the majors and loosely elsewhere; the
            # block reads neither (the reported rate is already clamped), so these
            # are here only because the real frame carries them and a stand-in must
            # not offer a narrower column set than the source it stands in for.
            "adjustedFundingRateCap": 0.003 if hours == 8 else 0.02,
            "adjustedFundingRateFloor": -0.003 if hours == 8 else -0.02,
        })
    return pd.DataFrame(rows)


def _synthetic_book_ticker(universe):
    """The stand-in ``book_ticker`` source: the top of each instrument's book.

    Whole-market in one request on the real venue, so unlike the fan-out
    stand-ins this one covers EVERY row of the synthetic universe — a partial
    book frame is not a state the real source can be in, and pretending otherwise
    would make specs fail validation for a reason production cannot produce.

    The values are chosen so each filter's outcome is observable, which is the
    only way a dry-run proves more than "it did not raise":

    * spreads straddle a 5 bps ceiling, the threshold a real request lands on
      (measured: the venue's median is 5.5 bps and 395 of 727 are above 5), so
      ``filter_spread`` narrows the frame AND leaves rows.
    * one instrument in four is deep-but-wide and another tight-but-empty,
      because ``spread_bps`` and ``top_of_book_usd`` are two independent questions
      and a stand-in where they agreed would let a spec screening the wrong one
      validate as if it had screened the right one. On the live venue that is
      TAKEUSDT: 7.8 bps — respectable — with three cents at the best ask.

    Every book here is well-formed, and that is deliberate: no crossed, locked or
    one-sided row. Baking a defective quote into the stand-in would make
    ``augment_with_spread`` emit its unquotable-book warning on EVERY validate of
    EVERY correct spec — a warning about the stand-in, dressed as a warning about
    the user's spec, on a channel whose value is that it is rare. Those three
    shapes are pinned by ``tests/standard_bot/test_universe_spread.py`` against
    hand-built frames, where the assertion can name which one it is testing.
    """
    import numpy as np
    import pandas as pd

    instruments = [str(value) for value in universe["instrument_id"]]
    mid = np.asarray(universe["lastPrice"], dtype=float)
    idx = np.arange(len(instruments))
    # 1 / 4 / 9 / 16 bps, cycling: rows on both sides of a 5 bps ceiling.
    spread_bps = np.array([1.0, 4.0, 9.0, 16.0])[idx % 4]
    half = mid * spread_bps / 2.0 / 10_000.0
    bid, ask = mid - half, mid + half
    # Dollar depth that does NOT track the spread, so the two screens disagree:
    # every 4th instrument is tight (4 bps) and holds $30 at the touch.
    depth_usd = np.where(idx % 4 == 1, 30.0, 250_000.0 - idx * 1_000.0)
    return pd.DataFrame({
        "symbol": instruments,
        "bidPrice": bid,
        "bidQty": depth_usd / bid,
        "askPrice": ask,
        "askQty": depth_usd / ask,
        "time": 1_700_000_000_000,
    })


def _synthetic_contract_meta(universe):
    """The stand-in ``contract_meta`` source: the venue's listing registry.

    Supplied explicitly for the same safety reason as :func:`_synthetic_funding`:
    validation must exercise ``with: [contract_meta]`` without letting
    ``augment_with_contract_meta`` fall back to a live ``exchangeInfo`` request.

    Its column names are Binance's own, because that is what reaches the block
    from a ``cyqnt.input/v1`` frame — and note what is deliberately NOT done here:
    these columns are added to a SEPARATE frame, never to
    :func:`_synthetic_universe`. A 24h ticker carries no ``underlyingType``, and a
    stand-in universe that offered one would let ``score: underlying_type`` and a
    spec that forgot the augment step both validate green and then fail on every
    real bundle.

    The VALUES have to make every filter's outcome observable, or the dry-run
    proves nothing beyond "it did not raise". Every classification below is the
    live venue's own answer for that instrument (see :data:`_SYNTHETIC_NON_CRYPTO`
    and :data:`_SYNTHETIC_CRYPTO_TAGS`), which means:

    * both sides of ``universe.filter_crypto_only`` are populated, so it and
      ``filter_underlying_type`` each narrow the frame *and* leave rows. With a
      crypto-only stand-in, "exclude the stocks" dry-ran as a filter matching no
      row — the same result the typo ``include: [COINS]`` gives.
    * ``INDEX`` (``ALLUSDT``) is present because it is the whole reason
      ``filter_crypto_only`` is a whitelist: it is what "exclude TradFi" spelled
      as a blacklist keeps.
    * one row is MULTI-tagged (``FOLKSUSDT`` → ``['Alpha', 'DeFi']``) and one is
      tagged with nothing (``ALLUSDT`` → ``[]``). Those are the two shapes naive
      code gets wrong: a 2-element list raises "truth value of an array … is
      ambiguous" in the candidate builder, and an empty list must read as "no
      sector", never as "unknown sector".
    * ``underlyingSubType`` is a LIST here, not a pre-flattened string, so the
      dry-run exercises ``_flatten_sub_type`` — the conversion a real bundle needs.
    * one row (``DELISTEDUSDT``) is in the registry and NOT in the universe, and it
      is ``SETTLING``: the node filters nothing, so both facts have to be able to
      arrive, and a left join must not widen the universe with them.
    * ``status`` / ``quoteAsset`` are real per-row values rather than constants,
      because the node does no filtering and a spec may screen on either.
    """
    import pandas as pd

    from cyqnt_trd.blocks.news_feed import base_token

    instruments = [str(value) for value in universe["instrument_id"]]
    # A registry covers the venue, not only the rows a universe happens to carry.
    instruments.append("DELISTEDUSDT")

    rows = []
    for instrument in instruments:
        quote = "USDC" if instrument.endswith("USDC") else "USDT"
        base = base_token(instrument)
        if instrument in _SYNTHETIC_NON_CRYPTO:
            underlying, tags = _SYNTHETIC_NON_CRYPTO[instrument]
        else:
            underlying = "COIN"
            # An unlisted base token gets no sector tag rather than a made-up one:
            # "the venue tagged nothing" is a real state (33 of 727 live
            # contracts), and inventing a tag would hide it from the dry-run.
            tags = _SYNTHETIC_CRYPTO_TAGS.get(base, [])
        rows.append({
            "instrument_id": instrument,
            # TRADIFI_PERPETUAL tracks the *underlying being off-chain*, not the
            # sector tag: on the live venue every EQUITY / COMMODITY / *_EQUITY /
            # PREMARKET contract is one and every COIN and INDEX contract is not.
            "contractType": ("PERPETUAL" if underlying in ("COIN", "INDEX")
                             else "TRADIFI_PERPETUAL"),
            "underlyingType": underlying,
            "underlyingSubType": list(tags),
            "baseAsset": base,
            "quoteAsset": quote,
            "status": ("SETTLING" if instrument == "DELISTEDUSDT" else "TRADING"),
            "source_id": "synthetic.contract_meta",
            "event_time": 1_700_000_000_000,
            "available_time": 1_700_000_000_000,
        })
    return pd.DataFrame(rows)


#: The stand-in roster the three fan-out sources cover — deliberately NOT the
#: whole synthetic universe.
#:
#: A real capture fans out over a NARROWED roster (127 of 727 on the measured
#: snapshot), and the joining blocks refuse a frame that does not cover the frame
#: it is joined onto. If the stand-in covered every synthetic instrument, a spec
#: that put ``augment_with_open_interest`` BEFORE its narrowing steps would
#: dry-run green and then fail on the first real bundle with a coverage error —
#: the exact "validate green, real run explodes" asymmetry the stand-in universe's
#: own column set exists to prevent.
#:
#: So the roster is the crypto rows: the non-crypto ones (:data:`_SYNTHETIC_NON_CRYPTO`)
#: are absent, which is what a ``filter_crypto_only``-then-fan-out capture
#: produces, and 3 of 17 missing is below the open-interest join's 95 % floor.
def _synthetic_fan_out_roster(universe) -> List[str]:
    return [str(value) for value in universe["instrument_id"]
            if str(value) not in _SYNTHETIC_NON_CRYPTO]


def _synthetic_open_interest(universe):
    """The stand-in ``open_interest_snapshot`` source.

    Supplied explicitly for the same safety reason as :func:`_synthetic_funding`,
    and one that is sharper here: the live fallback fans out over the frame, so a
    dry-run that reached it would fire one request per synthetic instrument.

    Binance's own column names, because that is NOT what a bundle delivers — the
    node's ``column_map`` renames them — so a stand-in in vendor spelling proves
    the block's alias table works from the Python side while the frozen-fixture
    test proves the canonical side. The VALUES straddle a $5m notional floor and
    include a sub-dollar coin priced at 0.07, so the coins-versus-dollars mistake
    (a base-quantity screen keeps the meme coin and drops BTC) shows up as a
    different basket during the dry-run instead of in production.
    """
    import numpy as np
    import pandas as pd

    roster = _synthetic_fan_out_roster(universe)
    idx = np.arange(len(roster), dtype=float)
    # Prices spanning five orders of magnitude, like a real venue: the whole
    # point of oi_notional_usd is that oi_base is not comparable across them.
    price = np.where(idx % 3 == 0, 60_000.0, np.where(idx % 3 == 1, 1_800.0, 0.07))
    # Notionals that straddle 5e6: every third instrument is below it.
    notional = np.where(idx % 3 == 2, 1.2e6 + idx * 1e5, 8.0e6 + idx * 5e6)
    return pd.DataFrame({
        "symbol": roster,
        "openInterest": notional / price,
        "markPrice": price,
        "time": 1_700_000_000_000,
    })


def _synthetic_oi_history(universe, *, days: int = 8):
    """The stand-in ``oi_change_snapshot`` source: a DAILY series per instrument.

    Three properties are load-bearing and each was chosen against a way the
    dry-run could otherwise prove nothing:

    * the readings are spaced exactly one day apart, because
      ``augment_with_oi_change`` measures the cadence and refuses to call a
      change over eight 5-minute buckets a change over eight days;
    * ``days`` is ``lookback_days + 1`` for the default 7-day lookback, so at
      least some instruments HAVE a computable change. A spec asking for a longer
      lookback than the stand-in carries gets the short-history warning relayed
      into its validate report, which is the correct answer;
    * the changes straddle ±20 % in BOTH the dollar and the coin column, and
      they disagree on which instruments clear it — that disagreement is the
      reason ``filter_oi_change`` has an explicit ``basis``, and a stand-in where
      the two agreed would let a spec with the wrong basis validate as if it were
      the right one.
    """
    import pandas as pd

    roster = _synthetic_fan_out_roster(universe)
    latest_day = 1_700_000_000_000
    rows = []
    for position, symbol in enumerate(roster):
        # A flat baseline and one step, so each instrument's change is EXACTLY
        # the declared number — a plausible-looking ramp would leave the dry-run's
        # thresholds depending on arithmetic nobody reading this can check.
        # The two columns clear a ±20 % screen on DISJOINT sets of instruments,
        # which is what makes `basis` observable during validate.
        value_change = (0.34, 0.02, -0.28)[position % 3]
        base_change = (0.02, -0.30, 0.05)[position % 3]
        for age in range(days - 1, -1, -1):
            latest = age == 0
            rows.append({
                "symbol": symbol,
                "timestamp": latest_day - age * 86_400_000,
                "sumOpenInterest": 1_000_000.0 * (1.0 + (base_change if latest else 0.0)),
                "sumOpenInterestValue":
                    5_000_000.0 * (1.0 + (value_change if latest else 0.0)),
            })
    return pd.DataFrame(rows)


def _synthetic_long_short_ratio(universe):
    """The stand-in ``long_short_ratio_snapshot`` source.

    ``longAccount`` / ``shortAccount`` are SHARES OF 1, as the venue sends them,
    and they sum to exactly 1 — the block verifies that rather than trusting it,
    so a stand-in that pre-converted to percentage points would exercise the
    refusal path instead of the working one. Shares straddle 0.60 so a
    "retail is more than 60 % long" screen has rows on both sides.
    """
    import numpy as np
    import pandas as pd

    roster = _synthetic_fan_out_roster(universe)
    idx = np.arange(len(roster))
    long_share = np.round(0.55 + 0.12 * np.sin(idx / 1.9), 4)
    return pd.DataFrame({
        "symbol": roster,
        "longAccount": long_share,
        "shortAccount": np.round(1.0 - long_share, 4),
        "longShortRatio": np.round(long_share / (1.0 - long_share), 4),
        "timestamp": 1_700_000_000_000,
    })


#: Bar columns the stand-in ``universe_bars`` frame carries — exactly the ones a
#: real one does, no more.
#:
#: "No more" is the rule the stand-in universe learned the hard way: it used to
#: offer ``symbol`` and ``quote_volume``, which a ``cyqnt.input/v1`` universe does
#: not have, so two specs validated green and raised on every real bundle. The same
#: trap is live here — ``fetch_klines`` returns ``taker_buy_base`` /
#: ``taker_buy_quote`` and a bundle carries them through untouched, so they ARE
#: real and are included; ``confirmed`` is NOT, because the klines fan-out does not
#: emit it and a spec reading it must fail at validate rather than at run.
_SYNTHETIC_BAR_COLUMNS = (
    "instrument_id", "timeframe", "open_time", "open", "high", "low", "close",
    "volume", "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "event_time", "available_time",
)


#: Instrument -> compound per-bar drift applied at EVERY timeframe of its bars.
#:
#: These two names are the whole reason the bars stand-in proves anything, and the
#: reasoning is the same as the six-consecutive-positions trick in
#: :func:`_synthetic_universe`: a dry-run over rows that cannot disagree is worth
#: no more than one over the wrong columns.
#:
#: Both are ``Alpha``-tagged (see :data:`_SYNTHETIC_SECTOR_BASES`) and both survive
#: the derivative stand-ins' thresholds, so they are the rows that actually reach an
#: indicator step in a realistic pipeline. They are engineered to land on OPPOSITE
#: sides of the two screens this frame exists to exercise:
#:
#: ============  =========  ==========================  =========================
#: instrument    per bar    Supertrend, every timeframe  range_gain_pct over 90
#: ============  =========  ==========================  =========================
#: ``TAGUSDT``     -1.0 %   -1  (kept by all_of)         0.99^90 -> +148 % (kept)
#: ``USUSDT``      +0.6 %   +1  (dropped by all_of)      1.006^90 -> +71 % (dropped
#:                                                       by min_score: 100)
#: ============  =========  ==========================  =========================
#:
#: A monotone compound drift is what makes the Supertrend direction the SAME on
#: every timeframe for these two, which is what a resonance screen needs to have a
#: row it keeps. Every OTHER instrument gets no drift at all, so its direction is
#: decided by the per-timeframe phase offset and it DISAGREES across timeframes —
#: the case that separates ``all_of`` from ``any_of``. Without both classes present
#: the dry-run answers "produced no candidates", which is a warning nobody acts on.
#:
#: Compound and not linear, because a linear -1 %/bar crosses zero at bar 100 and a
#: negative price makes ``range_gain_pct`` meaningless.
_SYNTHETIC_BAR_DRIFT = {
    "TAGUSDT": -0.010,
    "USUSDT": +0.006,
}


def _synthetic_universe_bars(universe, spec: Dict[str, Any]):
    """The stand-in ``universe_bars`` source: OHLCV per instrument per timeframe.

    Returns ``None`` when the spec has no indicator step, so a spec that does not
    ask for bars is not handed a frame nothing reads.

    Four properties are load-bearing, and each is here because without it the
    dry-run proves nothing:

    * **The timeframes come from the SPEC**, via
      ``interpreter.bar_timeframes_for_spec``. Fabricating a fixed set would make
      ``timeframe: "3d"`` validate against bars that a real capture — which
      collects exactly the union the spec names — would never contain.
    * **Both sides of the screens have rows.** See :data:`_SYNTHETIC_BAR_DRIFT`.
    * **Coverage is TOTAL** — every instrument in the stand-in universe, unlike the
      three derivative fan-outs whose roster is deliberately partial. That is not
      laziness: a bars roster is DERIVED from the surviving prefix of the same
      pipeline (see ``bundle_runner.plan_bars_capture``), so it covers the frame by
      construction and a partial stand-in would fail specs for a reason no real
      capture can produce.
    * **The bar count follows the spec's own period hints**, so an indicator asking
      for 90 bars is given enough to clear its warm-up guard instead of tripping it
      on the stand-in.
    """
    import numpy as np
    import pandas as pd

    from .interpreter import bar_timeframes_for_spec

    timeframes = bar_timeframes_for_spec(spec)
    if not timeframes:
        return None

    hints: List[int] = []
    _collect_period_hints(spec, hints)
    # x3 mirrors augment_with_indicator's default settling margin, +30 of slack so
    # a spec that leaves min_bars_multiple alone still clears it.
    bars = max(120, (max(hints) if hints else 20) * 3 + 30)

    instruments = [str(value) for value in universe["instrument_id"]]
    price = dict(zip(instruments, np.asarray(universe["lastPrice"], dtype=float)))
    index = np.arange(bars)
    rows = []
    for position, instrument in enumerate(instruments):
        drift = _SYNTHETIC_BAR_DRIFT.get(instrument, 0.0)
        base = price[instrument]
        trend = base * (1.0 + drift) ** index
        volume = 1_000.0 + 25.0 * np.abs(np.sin(index / 5.0 + position))
        for timeframe_position, timeframe in enumerate(timeframes):
            # Each timeframe gets its own bar spacing AND its own oscillation
            # phase. Same spacing would make the three frames byte-identical
            # series, so three indicator steps reading three timeframes and three
            # reading ONE would produce the same columns — and a spec that named
            # the wrong timeframe three times would dry-run as if it were right.
            step_ms = 900_000 * (timeframe_position + 1)
            phase = position + 1.7 * timeframe_position
            close = trend * (1.0 + 0.02 * np.sin(index / 9.0 + phase))
            open_ = np.concatenate([[close[0]], close[:-1]])
            open_time = 1_700_000_000_000 + index * step_ms
            rows.append(pd.DataFrame({
                "instrument_id": instrument,
                "timeframe": timeframe,
                "open_time": open_time,
                "open": open_,
                "high": np.maximum(open_, close) * 1.004,
                "low": np.minimum(open_, close) * 0.996,
                "close": close,
                "volume": volume,
                "close_time": open_time + step_ms - 1,
                "quote_volume": volume * close,
                "trades": (volume / 10.0).astype(int),
                "taker_buy_base": volume * 0.5,
                "taker_buy_quote": volume * close * 0.5,
                "event_time": open_time + step_ms - 1,
                "available_time": open_time + step_ms - 1,
            }))
    return pd.concat(rows, ignore_index=True)[list(_SYNTHETIC_BAR_COLUMNS)]


def _relay_block_warnings(caught, warnings: List[str]) -> None:
    """Copy warnings the blocks raised during the dry-run into the spec report.

    A block that says "this filter matched nothing" (see
    ``blocks.universe._warn_matched_nothing``) was writing to stderr, where a
    frontend calling :func:`validate_spec` never sees it — so validate could know
    that the spec's quote currency matched no row in the stand-in universe and
    still answer ``errors=[] warnings=[]``, which is the whole complaint.

    Only ``RuntimeWarning``, and only from this package. A pandas / numpy
    Deprecation- or FutureWarning is about this repo's use of a library, not about
    the spec being validated; putting those in a spec report would train its
    reader to skim it.
    """
    for entry in caught:
        if not issubclass(entry.category, RuntimeWarning):
            continue
        if "cyqnt_trd" not in str(entry.filename):
            continue
        warnings.append("selection dry-run warned: %s" % entry.message)


def _dry_run_selection(spec: Dict[str, Any], errors: List[str],
                       warnings: List[str]) -> Tuple[List[str], List[str]]:
    """Execute the compiled selection against a synthetic universe.

    Same purpose as the trade dry-run: prove the spec is structurally runnable
    before any real data or money is involved.
    """
    import warnings as _warnings

    universe = _synthetic_universe()
    rank = _synthetic_ticker_rank(universe)
    # ``None`` when the spec has no indicator step. Passed anyway: the selection
    # runner refuses a ``with:`` name whose frame is None with a message naming the
    # source, which is the right answer for a spec asking for bars it did not
    # declare — and a nicer one than KeyError.
    bars = _synthetic_universe_bars(universe, spec)
    with _warnings.catch_warnings(record=True) as caught:
        # "always": the default filter shows one warning per code location per
        # process, so a second validate call in the same process — every frontend
        # request after the first — would have reported nothing.
        _warnings.simplefilter("always")
        try:
            candidates = build_selection_fn(spec)(
                universe,
                rank,
                frames={
                    "funding": _synthetic_funding(universe),
                    "funding_info": _synthetic_funding_info(universe),
                    "book_ticker": _synthetic_book_ticker(universe),
                    "contract_meta": _synthetic_contract_meta(universe),
                    "open_interest_snapshot": _synthetic_open_interest(universe),
                    "oi_change_snapshot": _synthetic_oi_history(universe),
                    "long_short_ratio_snapshot":
                        _synthetic_long_short_ratio(universe),
                    "universe_bars": bars,
                },
            )
            failure: Any = None
        except Exception as exc:
            failure = exc
    # Relayed before the failure branch returns: a warning explaining WHICH step
    # emptied the universe is most useful precisely when the run then failed.
    _relay_block_warnings(caught, warnings)
    if failure is not None:
        errors.append("selection dry-run failed: %s: %s"
                      % (type(failure).__name__, failure))
        return errors, warnings

    if not isinstance(candidates, list):
        errors.append("selection must produce a list of candidates, got %s"
                      % type(candidates).__name__)
        return errors, warnings
    if not candidates:
        warnings.append(
            "selection produced no candidates on synthetic data. That may be "
            "correct for a strict filter, but it also means the dry-run "
            "exercised none of the ranking or direction rules.")
        return errors, warnings

    required = {"symbol", "rank", "score", "side"}
    missing = required - set(candidates[0])
    if missing:
        errors.append("candidate is missing %s" % sorted(missing))
    return errors, warnings


#: Keys whose values name sources, callables or literals rather than frame
#: columns, and are therefore skipped by :func:`_selection_tokens`.
#:
#: ``params`` is the one that bites: a param value is an arbitrary string, so a
#: threshold, a roster or a description that happens to contain a column name
#: would be read as the spec USING that column — and
#: :func:`_refuse_column_without_its_source` would then demand a source for a
#: column nothing ranks on, which is an error its author cannot act on.
#:
#: ``block`` and ``with`` are skipped by the same rule rather than because of an
#: observed collision: they name callables and bundle sources, and no source name
#: currently equals a key of :data:`interpreter.COLUMN_REQUIRES_SOURCE`. The rule
#: is what keeps that true as the table grows — a column named after the source
#: that produces it is the obvious next entry, and it would otherwise be satisfied
#: by the very step that omitted the source.
_NON_COLUMN_SELECTION_KEYS = frozenset({"block", "with", "params"})


def _selection_tokens(selection: Any) -> List[str]:
    """Every string a selection section references as a COLUMN or feature name."""
    tokens: List[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            tokens.append(value)
        elif isinstance(value, dict):
            for key, item in value.items():
                if key in _NON_COLUMN_SELECTION_KEYS:
                    continue
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    # One recursive walk over the whole section, and no special case for
    # ``universe``: the skip list applies at every depth, so a step's ``block`` /
    # ``with`` / ``params`` are excluded wherever they appear. Special-casing the
    # top level is how a ``features`` entry's ``params`` used to slip through.
    walk(selection)
    return tokens


def _refuse_column_without_its_source(selection: Dict[str, Any],
                                      err) -> None:
    """Refuse a spec that ranks on a column its declared sources cannot produce.

    See :data:`interpreter.COLUMN_REQUIRES_SOURCE`. Without this, ``score:
    fundingRateApr`` with ``with: [funding]`` validates green — the block warns
    and NaN-fills, the dry-run reports the generic "no candidates" warning — and
    then returns an EMPTY basket on every real bundle. Checked statically, and by
    NAME, so the message can say which key to add.
    """
    referenced = set(_selection_tokens(selection))
    # ``isinstance`` on both, because this runs on UNVALIDATED yaml: a model that
    # emits ``with: 123`` must reach the malformed-``with`` error below, not a
    # TypeError from here. Iterating a bare int raised; iterating a bare STRING
    # would have been worse, silently reading "funding_info" one character at a
    # time and therefore never matching.
    steps = selection.get("universe")
    supplied = {
        str(name)
        for step in (steps if isinstance(steps, (list, tuple)) else ())
        if isinstance(step, dict)
        for name in (step.get("with") if isinstance(step.get("with"),
                                                    (list, tuple)) else ())
    }
    for column, (block, source) in COLUMN_REQUIRES_SOURCE.items():
        if column not in referenced or source in supplied:
            continue
        err(
            "selection references %r, which %s only produces when it is ALSO "
            "given %r — with `with: [funding]` alone the column exists but is NaN "
            "for every instrument (the block refuses to assume an 8-hour "
            "settlement interval), so the basket would come back empty and this "
            "spec would still validate. Write `with: [funding, %s]`, or rank on "
            "fundingRatePct, which is the rate PER SETTLEMENT and needs no "
            "schedule." % (column, block, source, source)
        )


def _condition_refs(node: Any, acc: List[str] | None = None) -> List[str]:
    """Every ``cond:`` block reference in a combinator tree, depth-first."""
    acc = [] if acc is None else acc
    if isinstance(node, dict):
        if isinstance(node.get("cond"), str):
            acc.append(node["cond"])
        for value in node.values():
            _condition_refs(value, acc)
    elif isinstance(node, list):
        for item in node:
            _condition_refs(item, acc)
    return acc


def validate_spec(spec: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return ``(errors, warnings)``. Empty ``errors`` ⇒ spec is runnable."""
    errors: List[str] = []
    warnings: List[str] = []

    def err(msg: str) -> None:
        errors.append(msg)

    # ---- structural ----
    if spec.get("target") not in (None, "standard_bot"):
        warnings.append(f"target={spec.get('target')!r}; this pipeline targets standard_bot")

    strategy = spec.get("strategy") or {}
    if not strategy.get("id"):
        err("strategy.id is required")

    run = spec.get("run") or {}
    mode = run.get("mode")
    if mode not in VALID_MODES:
        err(f"run.mode must be one of {sorted(VALID_MODES)}, got {mode!r}")

    data = spec.get("data") or {}
    if not data.get("symbol"):
        err("data.symbol is required")
    if not (data.get("primary") or {}).get("interval"):
        err("data.primary.interval is required")
    for key in sorted(set(data) - DATA_KEYS):
        err(
            "unknown data.%s; a misspelt optional section is accepted in silence "
            "and simply attaches nothing. Known sections: %s"
            % (key, sorted(DATA_KEYS))
        )
    for name, section in DATA_SECTIONS.items():
        declared = data.get(name)
        if declared is None:
            continue
        if not isinstance(declared, dict) or not declared.get("dir"):
            err("data.%s needs a 'dir'. %s" % (name, section.example))

    signals = spec.get("signals") or {}
    entry = signals.get("entry") or {}
    selection = spec.get("selection")
    if selection is not None and signals:
        err("a spec is either a trade strategy (signals:) or a selection "
            "strategy (selection:), not both — they emit different signal kinds")
    # ``isinstance``, matching every other selection check in this file. Using
    # ``is None`` here meant a non-None non-dict ``selection:`` — the classic
    # mis-indentation that makes it a YAML scalar — skipped this check AND the
    # isinstance branch in register_from_yaml, so a spec with no signals at all
    # validated clean and was registered as a TRADE strategy whose make_signals
    # returns all-False. It then backtested to a spotless trades=0.
    if not isinstance(selection, dict) and not entry.get("long") \
            and not entry.get("short"):
        err("signals.entry must define at least one of long / short")
    if selection is not None and not isinstance(selection, dict):
        err("selection: must be a mapping, got %s — a scalar here is usually a "
            "mis-indented block, and it would otherwise be registered as a trade "
            "strategy that can never fire" % type(selection).__name__)

    # ---- exit / risk ----
    exit_cfg = (spec.get("risk") or {}).get("exit")
    if exit_cfg is not None:
        etype = exit_cfg.get("type") if isinstance(exit_cfg, dict) else None
        if etype not in VALID_EXIT_TYPES:
            err(f"risk.exit.type must be one of {sorted(VALID_EXIT_TYPES)}, got {etype!r}")
        if isinstance(exit_cfg, dict):
            for key in sorted(set(exit_cfg) - EXIT_KEYS):
                err(
                    "unknown risk.exit.%s — the engines read only %s, so this key "
                    "would be dropped and the exit would fall back to its default"
                    % (key, sorted(EXIT_KEYS))
                )

    # ---- sizing ----
    sizing = spec.get("sizing") or {}
    for key in sorted(set(sizing) - SIZING_KEYS):
        err("unknown sizing.%s; allowed: %s" % (key, sorted(SIZING_KEYS)))
    size = sizing.get("size", 1.0)
    try:
        if not (0.0 < float(size) <= 1.0):
            err(f"sizing.size must be in (0, 1], got {size}")
    except (TypeError, ValueError):
        err(f"sizing.size must be numeric, got {size!r}")

    # ---- live guards ----
    if mode == "live":
        guards = (spec.get("risk") or {}).get("live_guards") or {}
        if not guards.get("max_notional"):
            err("run.mode=live requires risk.live_guards.max_notional (hard per-order cap)")
        if not run.get("duration_end_at"):
            err("run.mode=live requires run.duration_end_at (ISO8601; sessions must be time-bounded)")

    # ---- static block resolution ----
    ind_specs = (signals.get("indicators") or {})
    for name, ispec in ind_specs.items():
        if not isinstance(ispec, dict) or "block" not in ispec:
            err(f"indicator {name!r} must be a mapping with a 'block' field")
            continue
        try:
            resolve_block(ispec["block"])
        except SpecError as exc:
            err(f"indicator {name!r}: {exc}")
        unknown_fields = sorted(set(ispec) - INDICATOR_KEYS)
        if unknown_fields:
            err(
                "indicator %r has unknown field(s) %s; allowed: %s"
                % (name, unknown_fields, sorted(INDICATOR_KEYS))
            )

    # Conditions resolve statically too. The dry-run would catch these anyway,
    # but only the first one, and only as a dry-run traceback — listing them all
    # up front is the difference between one round trip and five.
    for label in ("long", "short"):
        for ref in _condition_refs(entry.get(label)):
            try:
                resolve_block(ref)
            except SpecError as exc:
                err(f"entry.{label}: {exc}")

    # ---- selection ----
    if isinstance(selection, dict):
        for key in sorted(set(selection) - SELECTION_KEYS):
            err("unknown selection.%s; allowed: %s" % (key, sorted(SELECTION_KEYS)))
        if not selection.get("score"):
            err("selection.score is required: name the column or feature to rank by")
        # ``min_score``/``max_score`` are absolute bounds (floor/ceiling), so an
        # inverted pair can never match anything. The dry-run does not catch it:
        # an empty basket is a legitimate outcome of a strict filter, so it comes
        # back as the generic "no candidates" warning with no hint that the
        # thresholds themselves are the reason. Say so statically instead.
        bounds = {}
        for key in ("min_score", "max_score"):
            if selection.get(key) is None:
                continue
            try:
                bounds[key] = float(selection[key])
            except (TypeError, ValueError):
                err("selection.%s must be numeric, got %r" % (key, selection[key]))
        if len(bounds) == 2 and bounds["min_score"] > bounds["max_score"]:
            err("selection.min_score (%g) is above selection.max_score (%g); they "
                "are absolute bounds regardless of selection.order, so no score "
                "can satisfy both and the basket is always empty. For a "
                "bottom-of-the-column screen use `order: asc` with `max_score:` "
                "as the ceiling." % (bounds["min_score"], bounds["max_score"]))
        for position, step in enumerate(selection.get("universe") or []):
            if not isinstance(step, dict) or "block" not in step:
                err("selection.universe[%d] must be a mapping with a 'block'" % position)
                continue
            for key in sorted(set(step) - UNIVERSE_STEP_KEYS):
                err("unknown selection.universe[%d].%s; allowed: %s"
                    % (position, key, sorted(UNIVERSE_STEP_KEYS)))
            try:
                resolve_block(step["block"])
            except SpecError as exc:
                err("selection.universe[%d]: %s" % (position, exc))
        for name, fspec in (selection.get("features") or {}).items():
            if not isinstance(fspec, dict) or "block" not in fspec:
                err("selection.features.%s must be a mapping with a 'block'" % name)
                continue
            try:
                resolve_block(fspec["block"])
            except SpecError as exc:
                err("selection.features.%s: %s" % (name, exc))
        for label in ("long_when", "short_when"):
            for ref in _condition_refs(selection.get(label)):
                try:
                    resolve_block(ref)
                except SpecError as exc:
                    err("selection.%s: %s" % (label, exc))
        _refuse_column_without_its_source(selection, err)

    # If structural errors already exist, skip the dry-run (it would just
    # re-raise the same problems less clearly).
    if errors:
        return errors, warnings

    if isinstance(selection, dict):
        return _dry_run_selection(spec, errors, warnings)

    # ---- dry-run on synthetic data ----
    try:
        df = _synthetic_df(spec)
    except Exception as exc:  # pragma: no cover - defensive
        err(f"could not build synthetic data for dry-run: {exc}")
        return errors, warnings

    try:
        make_signals = build_make_signals(spec)
        long_s, short_s = make_signals(df)
    except Exception as exc:
        err(f"dry-run failed: {type(exc).__name__}: {exc}")
        return errors, warnings

    import pandas as pd

    for label, series in (("long", long_s), ("short", short_s)):
        if series is None:
            continue
        if not isinstance(series, pd.Series):
            err(f"entry.{label} must evaluate to a boolean Series, got {type(series).__name__}")
        elif series.dtype != bool:
            warnings.append(
                f"entry.{label} evaluated to dtype {series.dtype} (expected bool); "
                "will be coerced by the runner"
            )

    return errors, warnings


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_from_yaml(path: str) -> Dict[str, Any]:
    """Validate a spec and register it as a block strategy in this process.

    After this returns, ``spec['strategy']['id']`` is a known block strategy
    that the standard_bot entrypoints (``mvp_backtest`` / ``mvp_paper_daemon``)
    can run via ``--strategy <id>`` with ``--engine python``.
    """
    spec = load_spec(path)
    errors, _warnings = validate_spec(spec)
    if errors:
        joined = "\n  - ".join(errors)
        raise SpecError(f"spec {path} is invalid:\n  - {joined}")

    from cyqnt_trd.blocks import strategy as _strategy

    if isinstance(spec.get("selection"), dict):
        # Same registry, same run_pipeline_step, different signal kind — the
        # selection path was already built and simply had no way in from YAML.
        _strategy.register_selection(
            spec["strategy"]["id"],
            build_selection_fn(spec),
            market_type=(spec.get("data") or {}).get("market_type", "futures"),
        )
        return spec

    make_signals = build_make_signals(spec)

    htf_specs = [
        (h["interval"], int(h["sma_period"]))
        for h in (spec.get("data", {}) or {}).get("htf", []) or []
        if isinstance(h, dict) and "sma_period" in h
    ] or None
    exit_cfg = (spec.get("risk") or {}).get("exit")
    size = float((spec.get("sizing") or {}).get("size", 1.0))

    _strategy.register(
        spec["strategy"]["id"],
        make_signals,
        htf_specs=htf_specs,
        exit_cfg=exit_cfg if isinstance(exit_cfg, dict) else None,
        size=size,
    )
    return spec
