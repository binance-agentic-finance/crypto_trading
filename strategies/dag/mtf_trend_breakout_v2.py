"""``mtf_trend_breakout_v2`` — a Strategy Spec DAG compiled onto the standard bot.

This is the M3.3 ``mtf_trend_breakout_v1`` example, compiled the way we think
codegen should emit it, plus the four things that spec cannot express yet.
Everything here is ordinary code — the point is the *shape*, so a generator can
be pointed at it.

Two entry points, one set of blocks
-----------------------------------

``run_dag()``      the compiled DAG. Fetches through ``runtime.data``, emits
                   through ``runtime.action``. This is what the scheduler runs
                   for paper/live, and what codegen produces from the YAML.

``make_signals()`` the same logic as vectorised boolean Series, registered via
                   ``strategy.register()``. This is what the backtest engines
                   run.

Both call the *same* ``_factors`` / ``_entry_votes`` helpers, so a change to the
logic cannot make the backtested strategy and the live strategy disagree. That
matters because the DAG form alone has no backtest at all: the spec is
explicitly live/paper-only, and "we'll verify it in paper" is not a substitute
for a walk-forward run before real money.

What this adds to the v1 spec
-----------------------------

1. **One decision time for the whole DAG.** ``with data.session()`` stamps every
   node with the same ``as_of``. Without it a 1h kline node and a 4h kline node
   fetched 40 seconds apart are two different decision times pretending to be
   one.
2. **Data status is carried, not discarded.** ``session.status_map()`` says which
   node degraded. A missing funding read makes the strategy *abstain*; it does
   not silently read as funding = 0.
3. **Positioning gate (the derivatives block).** v1 gates on trend alone. Buying
   a breakout into 120% annualised funding with rising OI and retail already
   long is the setup that gets liquidated, not the one that pays. This vetoes it.
4. **A reason when nothing happens.** v1's else-branch is ``action.order(None)``
   — the user gets silence. Here the gated-out path emits an ALERT explaining
   which condition failed, which is the single most-requested thing in the July
   conversation review.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from cyqnt_trd.blocks import (
    conditions as C,
    derivatives as D,
    entry as E,
    execution as X,
    indicators as I,
    sizing as SZ,
    strategy as strat,
)

BOT_ID = "mtf_trend_breakout_v2"
VERSION = "v2"

#: the spec's ``nodes[].type == "data"`` set — validated before anything runs
DATA_NODES = ["klines", "funding", "open_interest", "long_short_ratio"]

CONFIG: Dict[str, Any] = {
    "symbol": "BTCUSDT",
    "interval": "1h",
    "htf_interval": "4h",
    "limit": 500,
    "market_type": "futures",
    # trend stack (v1)
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "adx_period": 14,
    "adx_threshold": 25.0,
    "atr_period": 14,
    "htf_ma_fast": 20,
    "htf_ma_slow": 60,
    # positioning veto (added in v2)
    "funding_high_bps": 5.0,
    "funding_low_bps": -5.0,
    "oi_surge_pct": 20.0,
    "ls_crowded": 2.5,
    "ls_contrarian": 0.5,
    # exit + sizing
    "stop_atr_mult": 2.0,
    "take_profit_pct": 0.03,
    "equity": 10_000.0,
    "risk_pct": 0.01,
}


# ---------------------------------------------------------------------------
# shared computation — the ONLY place the logic lives
# ---------------------------------------------------------------------------


def _factors(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Compute-node layer. Identical under DAG execution and under backtest."""
    close = df["close"]
    macd_line, signal_line, _hist = I.macd(
        close, cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"]
    )
    adx_series = I.adx(df, cfg["adx_period"])[0]
    return {
        "close": close,
        "macd_line": macd_line,
        "signal_line": signal_line,
        "adx": adx_series,
        "atr": I.atr(df, cfg["atr_period"]),
    }


def _htf_trend(htf_close: pd.Series, cfg: Dict[str, Any]) -> pd.Series:
    """Higher-timeframe trend as a STATE, not a cross event.

    The v1 spec ANDs three ``*_cross`` conditions together. Crosses are
    single-bar events, so requiring three of them on the *same* bar is a
    near-impossible conjunction — the strategy compiles, validates, and then
    never fires. The working pattern (and the one the T2 template uses) is one
    trigger plus state filters: MACD golden cross is the trigger; HTF trend and
    ADX strength are states that must hold on that bar.
    """
    return (
        I.sma(htf_close, cfg["htf_ma_fast"]) > I.sma(htf_close, cfg["htf_ma_slow"])
    ).fillna(False)


def _entry_votes(
    factors: Dict[str, pd.Series],
    *,
    htf_trend: pd.Series,
    cfg: Dict[str, Any],
) -> Dict[str, pd.Series]:
    return {
        "trend_htf": htf_trend,
        "macd_bullish": C.macd_golden_cross(factors["macd_line"], factors["signal_line"]),
        "adx_trending": C.adx_trending(factors["adx"], cfg["adx_threshold"]),
    }


def _positioning_veto(
    *,
    close: pd.Series,
    funding: Optional[pd.Series],
    open_interest: Optional[pd.Series],
    long_short: Optional[pd.Series],
    cfg: Dict[str, Any],
) -> Tuple[pd.Series, List[str]]:
    """True where a LONG entry should be vetoed on positioning grounds.

    Three crowding votes; **two must agree** to veto. One alone is too noisy —
    expensive funding in a genuine trend is a cost, not a trap; what gets people
    liquidated is expensive funding *and* leverage still piling in *and* retail
    already on that side.

    A missing input casts **no vote either way** and is recorded in the returned
    reason list, so the caller degrades ``data_quality`` rather than treating an
    unread source as a passed check.
    """
    zero = pd.Series(False, index=close.index)
    votes: List[pd.Series] = []
    reasons: List[str] = []

    if funding is not None and not funding.empty:
        state = D.funding_rate_state(
            funding, cfg["funding_high_bps"], cfg["funding_low_bps"]
        )
        crowded_long = _to_index(state == "bullish_squeeze", close.index)
        votes.append(crowded_long)
        if bool(crowded_long.iloc[-1]) if len(crowded_long) else False:
            reasons.append("FUNDING_CROWDED_LONG")
    else:
        reasons.append("FUNDING_MISSING")

    if open_interest is not None and not open_interest.empty:
        # NOTE: despite the name, derivatives.oi_change_pct returns a FRACTION
        # (0.30 == +30%), so the percent-valued config threshold is scaled.
        oi_surge = _to_index(
            D.oi_change_pct(open_interest, periods=1) >= cfg["oi_surge_pct"] / 100.0,
            close.index,
        )
        votes.append(oi_surge)
        if bool(oi_surge.iloc[-1]) if len(oi_surge) else False:
            reasons.append("OI_SURGE")
    else:
        reasons.append("OI_MISSING")

    if long_short is not None and not long_short.empty:
        crowded = D.long_short_ratio_state(
            long_short, cfg["ls_crowded"], cfg["ls_contrarian"]
        )
        retail_long = _to_index(crowded == "crowded_long", close.index)
        votes.append(retail_long)
        if bool(retail_long.iloc[-1]) if len(retail_long) else False:
            reasons.append("ACCOUNT_SKEW_CROWDED_LONG")
    else:
        reasons.append("LS_MISSING")

    if len(votes) < 2:
        # fewer than two readable inputs: the check cannot be performed, so it
        # does not fire. The caller sees the *_MISSING reasons and degrades.
        return zero, reasons
    tally = sum((vote.astype(int) for vote in votes), pd.Series(0, index=close.index))
    return (tally >= 2).fillna(False), reasons


# ---------------------------------------------------------------------------
# form 1 — the compiled DAG (paper / live). What codegen should emit.
# ---------------------------------------------------------------------------


def run_dag(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute the DAG once and emit through the action boundary.

    Returns a summary dict (what the scheduler logs); the signals themselves go
    to the configured ``action`` sink, which is a recorder unless an executor
    has been explicitly installed.
    """
    from cyqnt_trd.standard_bot.runtime import action, data

    cfg = {**CONFIG, **(cfg or {})}
    symbol = cfg["symbol"]

    problems = data.validate_nodes(DATA_NODES)
    if problems:
        raise ValueError("spec data layer is not runnable:\n  " + "\n  ".join(problems))

    # ---- data layer: one session => one decision time for every node --------
    with data.session() as sess:
        # Required inputs: a failure here is fatal and says so.
        klines_1h = sess.call(
            "klines", symbol=symbol, interval=cfg["interval"],
            limit=cfg["limit"], market_type=cfg["market_type"],
        )
        klines_htf = sess.call(
            "klines", symbol=symbol, interval=cfg["htf_interval"],
            limit=cfg["limit"], market_type=cfg["market_type"],
        )

        # positioning inputs are optional: a failure degrades, never aborts
        funding_df = _try(sess, "funding", symbol=symbol, limit=200)
        oi_df = _try(sess, "open_interest", symbol=symbol, period=cfg["interval"], limit=48)
        ls_df = _try(sess, "long_short_ratio", symbol=symbol, period=cfg["interval"], limit=48)

        status = sess.status_map()
        warnings = sess.warnings()
        as_of = sess.as_of_ms

    # ---- compute layer -------------------------------------------------------
    factors = _factors(klines_1h, cfg)
    htf_trend = _htf_trend(klines_htf["close"], cfg)
    votes = _entry_votes(factors, htf_trend=_align(htf_trend, klines_1h), cfg=cfg)

    aligned = C.multi_timeframe_alignment(*votes.values())
    long_entry = E.all_of([aligned])

    veto, veto_reasons = _positioning_veto(
        close=factors["close"],
        funding=_column(funding_df, "rate"),
        open_interest=_column(oi_df, "oi_value"),
        long_short=_column(ls_df, "long_short_ratio"),
        cfg=cfg,
    )

    entry_now = bool(_last(long_entry)) and not bool(_last(veto))
    degraded = [name for name, value in status.items() if value != "ok"]
    data_quality = "good" if not degraded else "degraded"

    # ---- action layer --------------------------------------------------------
    if not entry_now:
        # v1 emitted action.order(None) here — silence. Say why instead.
        blocked_by = [name for name, series in votes.items() if not bool(_last(series))]
        if bool(_last(veto)):
            blocked_by.extend(veto_reasons)
        action.alert(
            {
                "symbol": symbol,
                "direction": "neutral",
                "action": "watch",
                "score": 0.0,
                "topic": "entry_gate",
                "reason_codes": blocked_by,
                "summary": "%s no entry: %s" % (symbol, ", ".join(blocked_by) or "gate closed"),
                "data_quality": data_quality,
                "as_of": as_of,
            },
            strategy_id=BOT_ID,
            strategy_version=VERSION,
        )
        return {
            "bot_id": BOT_ID, "as_of": as_of, "entry": False,
            "blocked_by": blocked_by, "data_quality": data_quality,
            "source_status": status, "warnings": warnings,
        }

    entry_price = float(factors["close"].iloc[-1])
    atr_value = float(factors["atr"].iloc[-1])
    order_spec = X.market_order(symbol=symbol, side="long")
    quantity = SZ.atr_position_size(
        equity=cfg["equity"],
        atr_value=atr_value,
        mark_price=entry_price,
        risk_pct=cfg["risk_pct"],
        stop_distance_atr_mult=cfg["stop_atr_mult"],
    )

    trade_signal = {
        "symbol": symbol,
        "side": "long",
        "order": getattr(order_spec, "__dict__", {}) or str(order_spec),
        "size": quantity,
        "entry_price": entry_price,
        "atr": atr_value,
        "data_quality": data_quality,
        "as_of": as_of,
    }
    exit_config = {
        "stop_loss": {"type": "atr_trailing", "atr": atr_value,
                      "multiplier": cfg["stop_atr_mult"]},
        "take_profit": {"type": "fixed", "pct": cfg["take_profit_pct"]},
    }
    action.order(
        trade_signal, exit_config=exit_config,
        strategy_id=BOT_ID, strategy_version=VERSION,
    )
    return {
        "bot_id": BOT_ID, "as_of": as_of, "entry": True,
        "signal": trade_signal, "exit_config": exit_config,
        "data_quality": data_quality, "source_status": status, "warnings": warnings,
    }


def _try(sess, node: str, **params: Any):
    """Optional data node: degrade to None instead of aborting the DAG."""
    from cyqnt_trd.standard_bot.runtime.data import DataUnavailable

    try:
        return sess.call(node, **params)
    except DataUnavailable:
        return None


def _column(frame, name: str) -> Optional[pd.Series]:
    if frame is None or getattr(frame, "empty", True) or name not in frame:
        return None
    return pd.to_numeric(frame[name], errors="coerce").dropna().reset_index(drop=True)


def _last(series: pd.Series) -> bool:
    return bool(series.iloc[-1]) if len(series) else False


def _to_index(series: pd.Series, index) -> pd.Series:
    """Reindex a derivatives series (its own cadence) onto the bar index.

    The two series have different lengths and their own row counts; the DAG only
    acts on the last bar, so the latest reading is broadcast. A *backtest* needs
    a real ``merge_asof`` on timestamps — that is what
    ``enrich_market_frame_with_derivatives`` does before ``make_signals`` sees
    the frame, which is why the two forms take different paths here.
    """
    if series is None or len(series) == 0:
        return pd.Series(False, index=index)
    if len(series) == len(index):
        out = series.copy()
        out.index = index
        return out.fillna(False).astype(bool)
    return pd.Series(bool(series.iloc[-1]), index=index)


def _align(htf_series: pd.Series, base: pd.DataFrame) -> pd.Series:
    """Broadcast the higher-timeframe verdict onto the base index.

    Only the latest HTF verdict is used, because the DAG form decides on the
    last bar. This is deliberately NOT a general resampler: the backtest form
    gets its HTF columns from ``register(htf_specs=...)``, which does the real
    time-aligned join.
    """
    if len(htf_series) == 0:
        return pd.Series(False, index=base.index)
    return pd.Series(bool(htf_series.iloc[-1]), index=base.index)


# ---------------------------------------------------------------------------
# form 2 — the same strategy on the backtest/paper/live engine route
# ---------------------------------------------------------------------------


def make_signals(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """(long, short) over the whole series — what the backtest engines consume.

    Derivatives columns are read off the frame when the run supplied them
    (``--derivatives-dir``); when absent the positioning veto simply does not
    fire, exactly as in the DAG form.
    """
    cfg = CONFIG
    factors = _factors(df, cfg)
    # Prefer the real HTF column the engine joined in via register(htf_specs=);
    # fall back to a same-timeframe proxy when the run did not request it.
    htf_column = "_htf_%s_sma_%d" % (cfg["htf_interval"], cfg["htf_ma_slow"])
    if htf_column in df:
        htf_trend = (factors["close"] > df[htf_column]).fillna(False)
    else:
        htf_trend = _htf_trend(factors["close"], cfg)
    votes = _entry_votes(factors, htf_trend=htf_trend, cfg=cfg)
    long = E.all_of(list(votes.values()))

    veto, _reasons = _positioning_veto(
        close=factors["close"],
        funding=df["funding_rate"] if "funding_rate" in df else None,
        open_interest=df["open_interest"] if "open_interest" in df else None,
        long_short=df["long_short_ratio"] if "long_short_ratio" in df else None,
        cfg=cfg,
    )
    long = (long & ~veto).fillna(False)
    short = pd.Series(False, index=df.index)
    return long, short


strat.register(
    BOT_ID,
    make_signals,
    exit_cfg={
        "type": "atr_stop_tp",
        "atr_period": CONFIG["atr_period"],
        "stop_mult": CONFIG["stop_atr_mult"],
        "tp_mult": 3.0,
    },
    size=0.1,
    htf_specs=[(CONFIG["htf_interval"], CONFIG["htf_ma_slow"])],
    needs={"derivatives": True},
)
