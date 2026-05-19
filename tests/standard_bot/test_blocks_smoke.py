"""Smoke tests for the cyqnt_trd.blocks composable strategy library.

These tests run on synthetic OHLCV data — no network required. They
verify that:

* every public block can be imported and called with sensible defaults,
* all indicator outputs have the right shape / dtype / index alignment,
* boolean conditions never silently return NaN-typed series,
* exit / risk / sizing helpers obey their declared invariants,
* a complete user-style strategy can be registered via
  ``blocks.strategy.register`` and discovered by
  ``entrypoints.common.make_registry``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.blocks import (
    conditions as cond,
    data,
    derivatives as deriv,
    entry,
    exit as ex,
    execution,
    indicators as ind,
    microstructure as micro,
    patterns as pat,
    regime,
    risk,
    scoring,
    sizing,
    strategy,
    universe,
)


# ---------------------------------------------------------------------------
# Synthetic OHLCV fixture
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.normal(0.0, 1.0, size=n).cumsum()
    close = np.clip(close, 50.0, None)
    high = close + np.abs(rng.normal(0.0, 0.5, size=n))
    low = close - np.abs(rng.normal(0.0, 0.5, size=n))
    open_ = close + rng.normal(0.0, 0.2, size=n)
    volume = np.abs(rng.normal(1000.0, 200.0, size=n))
    open_time = np.arange(n, dtype=np.int64) * 60_000
    close_time = open_time + 60_000 - 1
    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "close_time": close_time,
            "quote_volume": volume * close,
            "trades": (volume / 10).astype("int64"),
        }
    )


@pytest.fixture()
def df() -> pd.DataFrame:
    return _make_ohlcv()


# ---------------------------------------------------------------------------
# Indicators — output shape & basic correctness
# ---------------------------------------------------------------------------


def test_sma_ema_lengths(df):
    s = ind.sma(df["close"], 20)
    assert len(s) == len(df)
    assert s.iloc[:19].isna().all()
    assert not s.iloc[19:].isna().any()

    e = ind.ema(df["close"], 20)
    assert len(e) == len(df)


def test_macd_components(df):
    m, s, h = ind.macd(df["close"], 12, 26, 9)
    assert len(m) == len(s) == len(h) == len(df)
    # By definition: hist = m - s
    pd.testing.assert_series_equal(h, m - s, check_names=False)


def test_rsi_bounds(df):
    r = ind.rsi(df["close"], 14)
    valid = r.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_atr_positive(df):
    a = ind.atr(df, 14)
    valid = a.dropna()
    assert (valid >= 0).all()


def test_adx_bounds(df):
    adx, plus_di, minus_di = ind.adx(df, 14)
    for s in (adx, plus_di, minus_di):
        valid = s.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()


def test_bollinger_order(df):
    upper, mid, lower = ind.bollinger(df["close"], 20, 2.0)
    valid = (~upper.isna()) & (~lower.isna())
    assert (upper[valid] >= mid[valid]).all()
    assert (mid[valid] >= lower[valid]).all()


def test_donchian_order(df):
    up, lo, mid = ind.donchian(df, 20)
    valid = (~up.isna()) & (~lo.isna())
    assert (up[valid] >= lo[valid]).all()


def test_ma_alignment(df):
    fast = ind.sma(df["close"], 5)
    mid = ind.sma(df["close"], 20)
    slow = ind.sma(df["close"], 60)
    align = ind.ma_alignment(fast, mid, slow)
    assert set(align.dropna().unique()) <= {"bullish", "bearish", "mixed"}


def test_ma_direction_returns_string(df):
    ma = ind.sma(df["close"], 20)
    d = ind.ma_direction(ma)
    assert set(d.dropna().unique()) <= {"up", "down", "flat"}


def test_swing_high_low_match_rolling(df):
    h = ind.swing_high(df, 10)
    expected = df["high"].rolling(10, min_periods=10).max()
    pd.testing.assert_series_equal(h, expected, check_names=False)


def test_supertrend_runs(df):
    st, dirn = ind.supertrend(df, 10, 3.0)
    assert len(st) == len(df)
    valid = ~st.isna()
    assert valid.sum() > 0
    assert set(dirn.dropna().unique()) <= {-1, 0, 1}


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------


def test_patterns_return_bool_series(df):
    for fn in (
        pat.doji, pat.hammer, pat.shooting_star, pat.bullish_engulfing,
        pat.bearish_engulfing, pat.three_white_soldiers, pat.three_black_crows,
        pat.gap_up, pat.gap_down, pat.morning_star, pat.evening_star,
    ):
        out = fn(df)
        assert isinstance(out, pd.Series)
        assert out.dtype == bool
        assert len(out) == len(df)


def test_engulfing_self_consistency():
    df = pd.DataFrame(
        {
            "open":  [100, 95, 90,  88, 92],
            "high":  [101, 96, 92,  95, 96],
            "low":   [ 99, 89, 85,  87, 91],
            "close": [ 99, 90, 95,  93, 91],
            "volume":[1000]*5,
        }
    )
    bull = pat.bullish_engulfing(df)
    # bar index 2: prev bearish (95→90), now bullish (90→95) and engulfs
    assert bull.iloc[2]


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


def test_ma_crossover_truthiness():
    fast = pd.Series([1, 2, 3, 4, 5], dtype=float)
    slow = pd.Series([3, 3, 3, 3, 3], dtype=float)
    above = cond.ma_cross_above(fast, slow)
    # crossover happens at index 3 where fast goes from 3 to 4 (was equal then >)
    assert above.iloc[3]
    assert not above.iloc[0]


def test_breakout_high_simple():
    df = pd.DataFrame(
        {
            "open":  [10, 11, 12, 13, 20],
            "high":  [11, 12, 13, 14, 21],
            "low":   [ 9, 10, 11, 12, 19],
            "close": [10, 11, 12, 13, 20],
            "volume":[100]*5,
        }
    )
    out = cond.breakout_high(df, lookback=2)
    assert out.iloc[-1]


def test_volume_surge_threshold(df):
    vma = ind.volume_ma(df, 20)
    sig = cond.volume_surge(df, vma, multiplier=10.0)
    # threshold so high almost no bar should exceed it
    assert sig.sum() <= 5


def test_rsi_in_range_works(df):
    r = ind.rsi(df["close"], 14)
    sig = cond.rsi_in_range(r, 40, 60)
    assert sig.dtype == bool
    assert len(sig) == len(df)


def test_funding_window_safe_excludes_settlement():
    # Build timestamps anchored exactly at UTC 16:00 of 2026-01-01.
    base_ms = int(pd.Timestamp("2026-01-01 16:00:00", tz="UTC").timestamp() * 1000)
    ts = pd.Series([
        base_ms - 5 * 60_000,   # 15:55 UTC, within 15-min buffer → unsafe
        base_ms - 30 * 60_000,  # 15:30 UTC, outside buffer        → safe
        base_ms + 5 * 60_000,   # 16:05 UTC, within 15-min buffer  → unsafe
        base_ms + 60 * 60_000,  # 17:00 UTC, outside buffer        → safe
    ])
    safe = cond.funding_window_safe(ts, settle_hours_utc=(0, 8, 16), buffer_min=15)
    assert list(safe) == [False, True, False, True]


# ---------------------------------------------------------------------------
# Entry combinators
# ---------------------------------------------------------------------------


def test_all_of_any_of():
    a = pd.Series([True, True, False, False])
    b = pd.Series([True, False, True, False])
    assert list(entry.all_of([a, b])) == [True, False, False, False]
    assert list(entry.any_of([a, b])) == [True, True, True, False]


def test_score_entry_threshold():
    a = pd.Series([True, False, True])
    b = pd.Series([False, True, True])
    score = entry.score_entry({"a": (a, 1.0), "b": (b, 2.0)}, threshold=2.0)
    assert list(score) == [False, True, True]


def test_consecutive():
    s = pd.Series([True, True, True, False, True, True])
    assert list(entry.consecutive(s, 2)) == [False, True, True, False, False, True]


# ---------------------------------------------------------------------------
# Exit
# ---------------------------------------------------------------------------


def test_fixed_stop_tp_math():
    assert ex.fixed_stop_price(100.0, 0.02, "long") == pytest.approx(98.0)
    assert ex.fixed_stop_price(100.0, 0.02, "short") == pytest.approx(102.0)
    assert ex.fixed_tp_price(100.0, 0.05, "long") == pytest.approx(105.0)
    assert ex.risk_reward(100.0, 98.0, 105.0) == pytest.approx(2.5)
    assert ex.passes_min_rr(100.0, 98.0, 105.0, min_rr=2.0)


def test_fixed_stop_loss_rule(df):
    rule = ex.FixedStopLoss(pct=0.02)
    out = rule.evaluate(df, entry_price=df["close"].iloc[0], side="long")
    assert out.dtype == bool
    assert len(out) == len(df)


def test_atr_trailing_stop_rule(df):
    a = ind.atr(df, 14).fillna(1.0)
    rule = ex.AtrTrailingStop(atr=a, multiplier=2.0)
    out = rule.evaluate(df, entry_price=df["close"].iloc[0], side="long")
    assert out.dtype == bool


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


def test_risk_guard_pause_after_consecutive_losses():
    cfg = risk.RiskConfig(
        max_positions=3,
        consecutive_loss_pause=(3, 60_000),  # 1 minute pause
    )
    guard = risk.RiskGuard(cfg)
    guard.on_equity_update(now_ms=0, equity=10_000.0)
    for _ in range(3):
        guard.on_position_opened()
        guard.on_trade_closed("BTCUSDT", pnl=-100.0, now_ms=10_000)
    ok, reason = guard.can_open_new(now_ms=10_001, equity=9_700.0)
    assert not ok
    assert reason and reason.startswith("paused_until")
    # after pause expires
    ok, _ = guard.can_open_new(now_ms=10_000 + 70_000, equity=9_700.0)
    assert ok


def test_risk_guard_drawdown_halt():
    cfg = risk.RiskConfig(max_drawdown_halt_pct=0.5)
    guard = risk.RiskGuard(cfg)
    guard.on_equity_update(now_ms=0, equity=10_000.0)
    guard.on_equity_update(now_ms=1_000, equity=4_000.0)  # 60% drawdown
    ok, reason = guard.can_open_new(now_ms=2_000, equity=4_000.0)
    assert not ok and reason == "halted_max_drawdown"


def test_funding_window_block():
    cfg = risk.RiskConfig(
        funding_buffer_min=15,
        funding_settle_hours_utc=(0, 8, 16),
    )
    guard = risk.RiskGuard(cfg)
    guard.on_equity_update(now_ms=0, equity=10_000.0)
    # 15:55 UTC == 15h55m == 15*3600*1000 + 55*60*1000
    near = 15 * 3600 * 1000 + 55 * 60 * 1000
    ok, reason = guard.can_open_new(now_ms=near, equity=10_000.0)
    assert not ok and reason == "funding_window"
    # 14:00 UTC should be safe
    safe = 14 * 3600 * 1000
    ok, _ = guard.can_open_new(now_ms=safe, equity=10_000.0)
    assert ok


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def test_sizing_helpers():
    assert sizing.fixed_pct_of_equity(10_000.0, 0.15) == pytest.approx(1_500.0)
    assert sizing.kelly_fraction(0.6, 1.0, 1.0, fractional=1.0) == pytest.approx(0.2)
    assert sizing.round_step_size(0.123456, 0.001) == pytest.approx(0.123)
    levels = sizing.grid_levels(100.0, 0.01, 5, 200.0)
    assert len(levels) == 5
    assert levels[0][0] < levels[-1][0]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_scoring_system(df):
    rsi_v = ind.rsi(df["close"], 14)
    ma20 = ind.sma(df["close"], 20)
    sys = scoring.ScoringSystem()
    sys.add_rule("rsi_zone", cond.rsi_in_range(rsi_v, 50, 70), weight=2.0)
    sys.add_rule("trend", cond.price_above_ma(df, ma20), weight=1.0)
    score = sys.evaluate()
    assert len(score) == len(df)
    bd = sys.breakdown()
    assert set(bd.columns) == {"rsi_zone", "trend"}


# ---------------------------------------------------------------------------
# Regime
# ---------------------------------------------------------------------------


def test_regime_adx(df):
    adx, *_ = ind.adx(df, 14)
    out = regime.adx_regime(adx)
    assert set(out.dropna().unique()) <= {"trend", "range", "transition"}


# ---------------------------------------------------------------------------
# Microstructure
# ---------------------------------------------------------------------------


def test_microstructure_basic():
    s = pd.Series(np.arange(200, dtype=float))
    flag = micro.whale_buy_signal(s, rolling_period=50, threshold_quantile=0.95)
    assert flag.dtype == bool
    inflow = micro.smart_money_inflow(s, s * 0.5, period=10)
    assert len(inflow) == len(s)


# ---------------------------------------------------------------------------
# Universe (no network)
# ---------------------------------------------------------------------------


def test_universe_filter_offline():
    fake = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "PEPEUSDT"],
            "quoteVolume": [3e9, 1.5e9, 5e7, 2e8],
            "priceChangePercent": [2.0, -3.0, 50.0, 5.0],
        }
    )
    syms = (
        universe.UniverseFilter(fake)
        .filter_quote_volume(min_quote_volume=1e8)
        .filter_change_pct(max_abs_pct=20.0)
        .top_gainers(2)
        .symbols()
    )
    assert "PEPEUSDT" in syms or "BTCUSDT" in syms
    assert "DOGEUSDT" not in syms  # |50%| > 20%
    assert len(syms) <= 2


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def test_execution_specs():
    o = execution.market_order("BTCUSDT", "long", quantity=0.01)
    assert o.symbol == "BTCUSDT"
    assert o.order_type == "MARKET"
    with pytest.raises(ValueError):
        execution.limit_order("BTCUSDT", "long", price=None, quantity=0.01)  # type: ignore[arg-type]
    tp, stop = execution.oco_pair(
        "BTCUSDT", "long", take_profit_price=110.0, stop_price=90.0, quantity=0.01,
    )
    assert tp.order_type == "LIMIT"
    assert stop.order_type == "STOP_MARKET"


# ---------------------------------------------------------------------------
# Strategy registration end-to-end
# ---------------------------------------------------------------------------


def test_strategy_register_and_flush(monkeypatch):
    """User-style strategy registration + make_registry pickup."""
    # Reset state to keep test idempotent across reruns
    strategy._PENDING_REGISTRATIONS.clear()
    strategy._KNOWN_BLOCK_STRATEGY_IDS.discard("__test_smoke__")

    def make_signals(df_):
        ma_fast = ind.sma(df_["close"], 5)
        ma_slow = ind.sma(df_["close"], 20)
        long_s = cond.ma_cross_above(ma_fast, ma_slow)
        short_s = cond.ma_cross_below(ma_fast, ma_slow)
        return long_s, short_s

    strategy.register("__test_smoke__", make_signals)
    assert strategy.is_known_block_strategy("__test_smoke__")
    assert any(p.plugin_id == "__test_smoke__" for p, _ in strategy.registered_block_strategies())

    from cyqnt_trd.standard_bot.entrypoints.common import make_registry
    registry = make_registry()
    plugin = registry.get("__test_smoke__")
    assert plugin.plugin_id == "__test_smoke__"


def test_strategy_plugin_run_on_snapshot(df):
    """End-to-end: register a strategy and execute its run() through standard_bot."""
    strategy._PENDING_REGISTRATIONS.clear()
    strategy._KNOWN_BLOCK_STRATEGY_IDS.discard("__test_e2e__")

    def make_signals(d):
        ma5 = ind.sma(d["close"], 5)
        ma20 = ind.sma(d["close"], 20)
        return cond.ma_cross_above(ma5, ma20), cond.ma_cross_below(ma5, ma20)

    strategy.register("__test_e2e__", make_signals)

    from cyqnt_trd.standard_bot.core import (
        Bar, BundleMeta, MarketBundle, SnapshotMeta, DataSnapshot,
    )

    bars = [
        Bar(
            open=row["open"], high=row["high"], low=row["low"], close=row["close"],
            volume=row["volume"], timestamp=int(row["close_time"]),
            instrument_id="BTCUSDT", timeframe="1m", confirmed=True,
            quote_volume=row["quote_volume"],
            extras={"open_time": int(row["open_time"]), "close_time": int(row["close_time"])},
        )
        for _, row in df.iterrows()
    ]
    bundle = MarketBundle(
        bars={MarketBundle.key("BTCUSDT", "1m"): bars},
        meta=BundleMeta(data_source="test"),
    )
    snapshot = DataSnapshot(
        version="v1",
        market=bundle,
        social=None,
        onchain=None,
        meta=SnapshotMeta(snapshot_id="snap1", assembled_at=0),
    )

    from cyqnt_trd.standard_bot.entrypoints.common import make_registry
    registry = make_registry()
    plugin = registry.get("__test_e2e__")
    config = registry._config_factories["__test_e2e__"](
        {"instrument_id": "BTCUSDT", "timeframe": "1m"}
    )
    batch = plugin.run(snapshot, config)
    # MA cross strategy on random data should produce *some* signals
    assert isinstance(batch.signals, list)


# ---------------------------------------------------------------------------
# Data conversion
# ---------------------------------------------------------------------------


def test_df_bars_roundtrip(df):
    from cyqnt_trd.standard_bot.core import Bar  # noqa: F401

    bars = data.df_to_bars(df, instrument_id="BTCUSDT", timeframe="1m")
    assert len(bars) == len(df)
    df2 = data.bars_to_df(bars)
    pd.testing.assert_series_equal(
        df["close"].reset_index(drop=True), df2["close"].reset_index(drop=True),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------


def test_derivatives_helpers():
    n = 200
    base = 100.0 + np.cumsum(np.random.default_rng(0).normal(0, 1, n))
    oi = 1_000_000.0 * (1 + np.random.default_rng(1).normal(0, 0.05, n).cumsum() * 0.01)
    p = pd.Series(base)
    o = pd.Series(oi)
    div = deriv.oi_price_divergence(p, o)
    assert set(div.dropna().unique()) <= {
        "bullish_buildup", "bearish_buildup", "long_squeeze", "short_squeeze", "none"
    }

    cvd = deriv.cvd(p, p * 0.5)
    assert len(cvd) == n
    assert math.isfinite(float(cvd.iloc[-1]))

    li = deriv.liquidation_imbalance(
        pd.Series([1e6, 0, 5e5, 0]),
        pd.Series([0, 1e6, 5e5, 0]),
    )
    assert (li.abs() <= 1.0).all()


# ---------------------------------------------------------------------------
# Aliases & SAR (Round 2 補強 — 處理 LLM 常見命名錯誤)
# ---------------------------------------------------------------------------


def test_parabolic_sar(df):
    sar, dirn = ind.parabolic_sar(df)
    assert len(sar) == len(df)
    valid = sar.dropna()
    assert valid.size > 0
    # direction is always +1 or -1 after the first bar
    assert set(dirn.iloc[1:].unique()) <= {-1, 1}


def test_bollinger_bands_alias(df):
    a = ind.bollinger_bands(df["close"], 20, 2.0)
    b = ind.bollinger(df["close"], 20, 2.0)
    pd.testing.assert_series_equal(a[0], b[0], check_names=False)


def test_close_above_below_helpers(df):
    ma20 = ind.sma(df["close"], 20)
    above = cond.close_above(df, ma20)
    below = cond.close_below(df, ma20)
    assert above.dtype == bool
    assert below.dtype == bool
    # mutually exclusive (ignore equality which is rare)
    assert ((above & below).sum()) == 0
    # scalar form
    above_scalar = cond.close_above(df, 50.0, bars=1)
    assert above_scalar.dtype == bool


def test_price_touch_or_cross():
    df = pd.DataFrame({
        "open": [10, 11, 12, 13, 12],
        "high": [11, 12, 13, 14, 13],
        "low":  [9.5, 10, 11, 12.5, 11],
        "close":[10.5, 11.5, 12.5, 13.5, 11.5],
        "volume":[100]*5,
    })
    touched = cond.price_touch_or_cross(df, 11.5, "any")
    # bar 1's range [10, 12] contains 11.5 → True
    assert touched.iloc[1]
    # direction filter
    up = cond.price_touch_or_cross(df, 11.5, "up")
    down = cond.price_touch_or_cross(df, 11.5, "down")
    assert up.dtype == bool and down.dtype == bool


def test_conditions_re_exports(df):
    """conditions.* re-exports for ma_alignment / consecutive / candle_*_shadow."""
    fast = ind.sma(df["close"], 5)
    mid = ind.sma(df["close"], 20)
    slow = ind.sma(df["close"], 60)
    align = cond.ma_alignment(fast, mid, slow)
    assert set(align.dropna().unique()) <= {"bullish", "bearish", "mixed"}

    sig = cond.consecutive(pd.Series([True, True, False, True, True, True]), 2)
    assert sig.dtype == bool

    # Just verify the re-exports are callable and return a Series of the right length.
    # (The synthetic fixture has random open/close so shadow values can be negative;
    # what we're testing here is that the re-export plumbing works, not the math.)
    lower = cond.candle_lower_shadow(df)
    upper = cond.candle_upper_shadow(df)
    assert isinstance(lower, pd.Series) and len(lower) == len(df)
    assert isinstance(upper, pd.Series) and len(upper) == len(df)
