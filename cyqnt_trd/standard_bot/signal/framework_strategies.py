"""The built-in strategies as ``make_signals(df) -> (long, short)``.

These reproduce the Numba kernel strategies (``signal/numba_kernels.py``) as vectorized,
df-based block strategies so they run on the unified ``cyqnt_trd.eval`` backtest instead of
the compiled event engine. Kernel semantics reproduced faithfully:

* the target is stateful — ``TARGET_KEEP`` means "hold the current position" — so each
  strategy emits entry/flat/flip *events* and :func:`_hold_forward` forward-fills them into a
  position, then returns ``(pos > 0, pos < 0)``;
* RSI uses the kernel's **simple** gain/loss mean (not Wilder);
* Donchian bands exclude the current bar (``shift(1)`` before the rolling window).

Numbers differ from the compiled engine (vectorized net-change vs single-position event loop),
which is the intended, framework-unified口径. The multi-timeframe strategy uses a
``secondary_span`` proxy for the higher-timeframe MA (a single df carries one timeframe).

Every strategy is split into three stages, chained by :func:`_analyze`, and the stages only
hand each other a ``dict``::

    ① <sid>_factors(df, p)       原始量(均線、通道、RSI、衍生品欄位),不做判斷
    ② <sid>_forecast(factors, p) 因子 → 事件 + verdict / score / bias
    ③ _sizing(forecast)          事件 → 持倉(+1 / 0 / -1)、止損(內置策略沒有止損)

This is the single-symbol form of the ``cyqnt_trd.eval`` stage split: ``_factors`` is the
factor stage, ``_forecast`` is stage 4 (``eval.forecast.make_forecast``: factors → mu) and
``_sizing`` is stage 5 (``eval.forecast.positions_from_forecast`` / ``eval.strategy
.target_weights``: forecast → position). A forecast never reads ``df`` and sizing never reads
factors, so a stage can be swapped or scored on its own. ``strategies/_square/`` carries the submission spec + code built from these
same stage functions (``scripts/build_square_payloads.py``).
"""
from __future__ import annotations

import inspect
from typing import Callable

import numpy as np
import pandas as pd

__all__ = ["FRAMEWORK_STRATEGIES", "STRATEGY_STAGES", "make_signals_for", "analyze",
           "strategy_defaults", "register_builtin_block_plugin",
           "moving_average_cross", "price_moving_average", "rsi_reversion",
           "donchian_breakout", "multi_timeframe_ma_spread",
           "oi_funding_breakout", "liquidation_reversal"]


def _hold_forward(entry_long: pd.Series, entry_short: pd.Series, go_flat: pd.Series,
                  index, held: float = 0.0) -> tuple:
    """Kernel TARGET_{LONG,SHORT,FLAT,KEEP} → forward-filled position → (long, short).

    Precedence on a bar: long > short > flat (kernel events are mutually exclusive, so this
    only matters if a caller passes overlapping masks). Bars with no event keep the position;
    bars before the first event hold ``held`` (0 in a backtest, the live holding in a
    windowed live run — otherwise a window with no event would silently flatten it).
    """
    target = pd.Series(np.nan, index=index)
    target[go_flat.fillna(False)] = 0.0
    target[entry_short.fillna(False)] = -1.0
    target[entry_long.fillna(False)] = 1.0
    pos = target.ffill().fillna(held)
    return pos > 0, pos < 0


def _col(df, name: str) -> pd.Series:
    """A df column if present, else an all-zero series (derivative feeds are optional).

    Zero-filling keeps the backtest contract, but it is a silent no-op: without the feed
    ``oi_funding_breakout`` degenerates into ``donchian_breakout`` and ``liquidation_reversal``
    never trades. The factor stages therefore report absent feeds under ``"missing"``
    (see :func:`_missing`), and the square submission code refuses to trade on them.
    """
    if name in df.columns:
        return df[name]
    return pd.Series(0.0, index=df.index)


def _missing(df, *names: str) -> list:
    return [name for name in names if name not in df.columns]


def _events(factors: dict, score: pd.Series, *, long=None, short=None, flat=None) -> dict:
    """Event masks → forecast dict. ``verdict`` is the bar's target event, with the same
    long > short > flat precedence as :func:`_hold_forward`; ``KEEP`` means no event.
    ``price`` (the close) is passed through so sizing can place a stop."""
    index = factors["index"]
    none = pd.Series(False, index=index)
    long = none if long is None else long
    short = none if short is None else short
    flat = none if flat is None else flat
    verdict = pd.Series("KEEP", index=index, dtype=object)
    verdict[flat.fillna(False)] = "FLAT"
    verdict[short.fillna(False)] = "SHORT"
    verdict[long.fillna(False)] = "LONG"
    bias = verdict.map({"LONG": "long", "SHORT": "short"}).fillna("neutral")
    return {"index": index, "price": factors["close"], "entry_long": long, "entry_short": short,
            "go_flat": flat, "verdict": verdict, "score": score, "bias": bias}


def _channel_position(close: pd.Series, upper: pd.Series, lower: pd.Series) -> pd.Series:
    """Where close sits in the channel: -1 at the lower band, +1 at the upper (unbounded)."""
    width = (upper - lower).replace(0.0, np.nan)
    return 2.0 * (close - (upper + lower) / 2.0) / width


# ------------------------------------------------------------------ ③ sizing
#: Protective stop distance used by sizing. The built-in backtest does not simulate stops, so
#: this only reaches the live/submission code; positions (and so backtest parity) ignore it.
DEFAULT_STOP_PCT = 0.03


def _sizing(fc: dict, held: float = 0.0, stop_pct: float = DEFAULT_STOP_PCT) -> dict:
    """③ forecast → position + protective stop price.

    The target is stateful (no event = keep the position), so sizing needs the position
    held before ``fc`` starts: 0 in a backtest, the exchange holding live. ``stop`` is
    ``price × (1 ∓ stop_pct)`` on the held side and NaN when flat.
    """
    long, short = _hold_forward(fc["entry_long"], fc["entry_short"], fc["go_flat"], fc["index"],
                                held)
    position = long.astype(float) - short.astype(float)
    price = fc["price"]
    stop = (price * (1.0 - stop_pct)).where(long, (price * (1.0 + stop_pct)).where(short))
    return {"long": long, "short": short, "position": position, "stop": stop}


# ------------------------------------------------------- per-strategy ① + ②
def _moving_average_cross_factors(df, p: dict) -> dict:
    fast = df["close"].rolling(p["fast_window"]).mean()
    slow = df["close"].rolling(p["slow_window"]).mean()
    return {"index": df.index, "close": df["close"], "fast_ma": fast, "slow_ma": slow, "spread": (fast - slow) / slow}


def _moving_average_cross_forecast(factors: dict, p: dict) -> dict:
    spread = factors["spread"]
    buy = spread > p["entry_threshold"]
    sell = spread < -p["entry_threshold"]                  # SELL -> FLAT (long-only)
    return _events(factors, spread, long=buy, flat=sell)


def _price_moving_average_factors(df, p: dict) -> dict:
    ma = df["close"].rolling(p["period"]).mean()
    return {"index": df.index, "close": df["close"], "ma": ma,
            "prev_close": df["close"].shift(1), "prev_ma": ma.shift(1),
            "spread": (df["close"] - ma) / ma}


def _price_moving_average_forecast(factors: dict, p: dict) -> dict:
    close, ma = factors["close"], factors["ma"]
    prev_c, prev_ma = factors["prev_close"], factors["prev_ma"]
    spread = ((close - ma) / ma).abs()
    up = (prev_c <= prev_ma) & (close > ma) & (spread > p["entry_threshold"])
    dn = (prev_c >= prev_ma) & (close < ma) & (spread > p["entry_threshold"])
    return _events(factors, factors["spread"], long=up, flat=dn)


def _rsi_reversion_factors(df, p: dict) -> dict:
    period = p["period"]
    delta = df["close"].diff()
    avg_gain = delta.clip(lower=0.0).rolling(period).sum() / period
    avg_loss = (-delta.clip(upper=0.0)).rolling(period).sum() / period
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = (100.0 - 100.0 / (1.0 + rs)).where(avg_loss != 0.0, 100.0)
    return {"index": df.index, "close": df["close"], "rsi": rsi}


def _rsi_reversion_forecast(factors: dict, p: dict) -> dict:
    rsi = factors["rsi"]
    buy = rsi < p["oversold"]
    sell = rsi > p["overbought"]                            # SELL -> FLAT
    return _events(factors, (50.0 - rsi) / 50.0, long=buy, flat=sell)


def _donchian_breakout_factors(df, p: dict) -> dict:
    upper = df["high"].shift(1).rolling(p["lookback_window"]).max()
    lower = df["low"].shift(1).rolling(p["lookback_window"]).min()
    return {"index": df.index, "close": df["close"], "upper": upper, "lower": lower}


def _donchian_breakout_forecast(factors: dict, p: dict) -> dict:
    close, upper, lower = factors["close"], factors["upper"], factors["lower"]
    buy = close > upper * (1.0 + p["breakout_buffer_bps"] / 1e4)
    sell = close < lower * (1.0 - p["breakout_buffer_bps"] / 1e4)   # SELL -> SHORT
    return _events(factors, _channel_position(close, upper, lower),
                   long=buy, short=sell)


def _multi_timeframe_ma_spread_factors(df, p: dict) -> dict:
    primary_ma = df["close"].rolling(p["primary_period"]).mean()
    secondary_ma = df["close"].rolling(max(2, p["secondary_period"] * p["secondary_factor"])).mean()
    return {"index": df.index, "close": df["close"], "primary_ma": primary_ma, "secondary_ma": secondary_ma,
            "spread": (primary_ma - secondary_ma) / secondary_ma}


def _multi_timeframe_ma_spread_forecast(factors: dict, p: dict) -> dict:
    spread = factors["spread"]
    thr = p["threshold_bps"] / 1e4
    buy = spread > thr
    sell = spread < -thr                                    # SELL -> SHORT
    return _events(factors, spread, long=buy, short=sell)


def _oi_funding_breakout_factors(df, p: dict) -> dict:
    upper = df["high"].shift(1).rolling(p["lookback_window"]).max()
    lower = df["low"].shift(1).rolling(p["lookback_window"]).min()
    return {"index": df.index, "close": df["close"], "upper": upper, "lower": lower,
            "oi_change_bps": _col(df, "oi_change_bps"),
            "funding_rate_bps": _col(df, "funding_rate_bps"),
            "missing": _missing(df, "oi_change_bps", "funding_rate_bps")}


def _oi_funding_breakout_forecast(factors: dict, p: dict) -> dict:
    close, upper, lower = factors["close"], factors["upper"], factors["lower"]
    oi, funding = factors["oi_change_bps"], factors["funding_rate_bps"]
    oi_ok = oi >= p["oi_threshold_bps"]
    buy = (oi_ok & (close > upper * (1.0 + p["breakout_buffer_bps"] / 1e4))
           & (funding <= p["max_funding_rate_bps"]))
    sell = (oi_ok & (close < lower * (1.0 - p["breakout_buffer_bps"] / 1e4))
            & (funding >= -p["max_funding_rate_bps"]))
    return _events(factors, _channel_position(close, upper, lower),
                   long=buy, short=sell)


def _liquidation_reversal_factors(df, p: dict) -> dict:
    long_liq = _col(df, "long_liq_notional_usd")
    short_liq = _col(df, "short_liq_notional_usd")
    total = long_liq + short_liq
    safe_total = total.replace(0.0, np.nan)
    return {"index": df.index, "close": df["close"], "long_liq": long_liq, "short_liq": short_liq, "total": total,
            "long_ratio": long_liq / safe_total, "short_ratio": short_liq / safe_total,
            "missing": _missing(df, "long_liq_notional_usd", "short_liq_notional_usd")}


def _liquidation_reversal_forecast(factors: dict, p: dict) -> dict:
    long_liq, short_liq, total = factors["long_liq"], factors["short_liq"], factors["total"]
    long_ratio, short_ratio = factors["long_ratio"], factors["short_ratio"]
    buy = ((total > 0.0) & (long_liq >= p["long_liquidation_threshold_usd"])
           & (long_ratio >= p["liquidation_imbalance_ratio"]))
    sell = ((total > 0.0) & (short_liq >= p["short_liquidation_threshold_usd"])
            & (short_ratio >= p["liquidation_imbalance_ratio"]))
    return _events(factors, (long_ratio - short_ratio).fillna(0.0),
                   long=buy, short=sell)


#: Built-in strategy id -> (① factors, ② forecast). ③ sizing is shared.
STRATEGY_STAGES: dict[str, tuple] = {
    "moving_average_cross": (_moving_average_cross_factors, _moving_average_cross_forecast),
    "price_moving_average": (_price_moving_average_factors, _price_moving_average_forecast),
    "rsi_reversion": (_rsi_reversion_factors, _rsi_reversion_forecast),
    "donchian_breakout": (_donchian_breakout_factors, _donchian_breakout_forecast),
    "multi_timeframe_ma_spread": (_multi_timeframe_ma_spread_factors,
                                  _multi_timeframe_ma_spread_forecast),
    "oi_funding_breakout": (_oi_funding_breakout_factors, _oi_funding_breakout_forecast),
    "liquidation_reversal": (_liquidation_reversal_factors, _liquidation_reversal_forecast),
}


def _analyze(df, strategy_id: str, p: dict) -> dict:
    """单标的三段式:factors → forecast → sizing。三段之间只传 dict。"""
    factors_fn, forecast_fn = STRATEGY_STAGES[strategy_id]
    factors = factors_fn(df, p)
    fc = forecast_fn(factors, p)
    size = _sizing(fc)
    return {"verdict": fc["verdict"], "score": fc["score"], "bias": fc["bias"],
            "long": size["long"], "short": size["short"], "position": size["position"],
            "stop": size["stop"], "missing": factors.get("missing", []), "factors": factors}


# ------------------------------------------------------- public make_signals
def moving_average_cross(df, fast_window: int = 5, slow_window: int = 20,
                         entry_threshold: float = 0.0):
    """Fast/slow SMA spread: long while fast>slow (by threshold), flat when fast<slow."""
    out = _analyze(df, "moving_average_cross", {
        "fast_window": fast_window, "slow_window": slow_window,
        "entry_threshold": entry_threshold})
    return out["long"], out["short"]


def price_moving_average(df, period: int = 20, entry_threshold: float = 0.0):
    """Price crossing its SMA: long on upward cross, flat on downward cross."""
    out = _analyze(df, "price_moving_average",
                   {"period": period, "entry_threshold": entry_threshold})
    return out["long"], out["short"]


def rsi_reversion(df, period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
    """RSI (simple mean, matching the kernel): long below oversold, flat above overbought."""
    out = _analyze(df, "rsi_reversion",
                   {"period": period, "oversold": oversold, "overbought": overbought})
    return out["long"], out["short"]


def donchian_breakout(df, lookback_window: int = 20, breakout_buffer_bps: float = 0.0):
    """Donchian channel breakout (bands exclude the current bar): long/short on breakout."""
    out = _analyze(df, "donchian_breakout", {
        "lookback_window": lookback_window, "breakout_buffer_bps": breakout_buffer_bps})
    return out["long"], out["short"]


def multi_timeframe_ma_spread(df, primary_period: int = 20, secondary_period: int = 20,
                              threshold_bps: float = 0.0, secondary_factor: int = 4):
    """Primary SMA vs a slower higher-timeframe SMA; long/short on the signed spread.

    ``secondary_factor`` approximates the higher timeframe (bars per HTF bar): the secondary
    MA is a rolling mean over ``secondary_period * secondary_factor`` primary bars.
    """
    out = _analyze(df, "multi_timeframe_ma_spread", {
        "primary_period": primary_period, "secondary_period": secondary_period,
        "threshold_bps": threshold_bps, "secondary_factor": secondary_factor})
    return out["long"], out["short"]


def oi_funding_breakout(df, lookback_window: int = 20, breakout_buffer_bps: float = 0.0,
                        oi_threshold_bps: float = 0.0, max_funding_rate_bps: float = 100.0):
    """OI-confirmed Donchian breakout gated by funding (needs ``oi_change_bps`` /
    ``funding_rate_bps`` columns): long on an up-breakout when OI rises and funding
    is not extreme; short on the symmetric down-breakout."""
    out = _analyze(df, "oi_funding_breakout", {
        "lookback_window": lookback_window, "breakout_buffer_bps": breakout_buffer_bps,
        "oi_threshold_bps": oi_threshold_bps, "max_funding_rate_bps": max_funding_rate_bps})
    return out["long"], out["short"]


def liquidation_reversal(df, long_liquidation_threshold_usd: float = 100_000.0,
                         short_liquidation_threshold_usd: float = 100_000.0,
                         liquidation_imbalance_ratio: float = 0.60):
    """Fade a one-sided liquidation cascade (needs ``long_liq_notional_usd`` /
    ``short_liq_notional_usd`` columns): a long-liquidation spike -> go long, a
    short-liquidation spike -> go short."""
    out = _analyze(df, "liquidation_reversal", {
        "long_liquidation_threshold_usd": long_liquidation_threshold_usd,
        "short_liquidation_threshold_usd": short_liquidation_threshold_usd,
        "liquidation_imbalance_ratio": liquidation_imbalance_ratio})
    return out["long"], out["short"]


#: Built-in strategy id -> make_signals. The unified backtest looks these up by --strategy.
FRAMEWORK_STRATEGIES: dict[str, Callable] = {
    "moving_average_cross": moving_average_cross,
    "price_moving_average": price_moving_average,
    "rsi_reversion": rsi_reversion,
    "donchian_breakout": donchian_breakout,
    "multi_timeframe_ma_spread": multi_timeframe_ma_spread,
    "oi_funding_breakout": oi_funding_breakout,
    "liquidation_reversal": liquidation_reversal,
}


def strategy_defaults(strategy_id: str) -> dict:
    """Default params of a built-in, read off its ``make_signals`` signature."""
    sig = inspect.signature(FRAMEWORK_STRATEGIES[strategy_id])
    return {k: v.default for k, v in sig.parameters.items() if k != "df"}


def analyze(strategy_id: str, df, **params) -> dict:
    """Run a built-in's three stages and return every stage's output (not just long/short).

    Unknown params raise, as they would through the ``make_signals`` signature.
    """
    if strategy_id not in FRAMEWORK_STRATEGIES:
        raise KeyError(f"no framework strategy {strategy_id!r}; "
                       f"have {sorted(FRAMEWORK_STRATEGIES)}")
    defaults = strategy_defaults(strategy_id)
    unknown = sorted(set(params) - set(defaults))
    if unknown:
        raise TypeError(f"{strategy_id} got unexpected params {unknown}")
    return _analyze(df, strategy_id, {**defaults, **params})


def make_signals_for(strategy_id: str, **params) -> Callable:
    """Bind params to a built-in strategy, returning a plain ``make_signals(df)``."""
    if strategy_id not in FRAMEWORK_STRATEGIES:
        raise KeyError(f"no framework strategy {strategy_id!r}; "
                       f"have {sorted(FRAMEWORK_STRATEGIES)}")
    fn = FRAMEWORK_STRATEGIES[strategy_id]
    return lambda df: fn(df, **params)


#: Built-ins whose ``make_signals`` reads a derivative feed (extra df columns).
_DERIVATIVE_BUILTINS = frozenset({"oi_funding_breakout", "liquidation_reversal"})


def register_builtin_block_plugin(strategy_id: str, **params) -> None:
    """Register a built-in framework strategy as a ``blocks`` strategy plugin.

    This lets the live/paper :class:`PythonLivePaperSession` (which resolves a
    :class:`BlockStrategyPlugin` by id) drive the built-in strategies through the
    unified ``make_signals`` contract — no Numba kernels involved. Idempotent: a
    strategy id already registered is left untouched.
    """
    from ...blocks.strategy import is_known_block_strategy, register  # type: ignore

    if is_known_block_strategy(strategy_id):
        return
    needs = {"derivatives": True} if strategy_id in _DERIVATIVE_BUILTINS else None
    register(strategy_id, make_signals_for(strategy_id, **params), needs=needs)
