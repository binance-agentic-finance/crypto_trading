"""Test fixture: frozen copy of ``signal/framework_strategies.py`` at e11527e (pre three-stage).

The refactor into ``_factors -> _forecast -> _sizing`` must not change any signal; the
equivalence tests compare against this snapshot. Do not edit.
"""
from __future__ import annotations
# Original module docstring:
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
"""

from typing import Callable

import numpy as np
import pandas as pd

__all__ = ["FRAMEWORK_STRATEGIES", "make_signals_for", "register_builtin_block_plugin",
           "moving_average_cross", "price_moving_average", "rsi_reversion",
           "donchian_breakout", "multi_timeframe_ma_spread",
           "oi_funding_breakout", "liquidation_reversal"]


def _hold_forward(entry_long: pd.Series, entry_short: pd.Series, go_flat: pd.Series,
                  index) -> tuple:
    """Kernel TARGET_{LONG,SHORT,FLAT,KEEP} → forward-filled position → (long, short).

    Precedence on a bar: long > short > flat (kernel events are mutually exclusive, so this
    only matters if a caller passes overlapping masks). Bars with no event keep the position.
    """
    target = pd.Series(np.nan, index=index)
    target[go_flat.fillna(False)] = 0.0
    target[entry_short.fillna(False)] = -1.0
    target[entry_long.fillna(False)] = 1.0
    pos = target.ffill().fillna(0.0)
    return pos > 0, pos < 0


def moving_average_cross(df, fast_window: int = 5, slow_window: int = 20,
                         entry_threshold: float = 0.0):
    """Fast/slow SMA spread: long while fast>slow (by threshold), flat when fast<slow."""
    fast = df["close"].rolling(fast_window).mean()
    slow = df["close"].rolling(slow_window).mean()
    spread = (fast - slow) / slow
    buy = spread > entry_threshold
    sell = spread < -entry_threshold                       # SELL -> FLAT (long-only)
    return _hold_forward(buy, pd.Series(False, index=df.index), sell, df.index)


def price_moving_average(df, period: int = 20, entry_threshold: float = 0.0):
    """Price crossing its SMA: long on upward cross, flat on downward cross."""
    ma = df["close"].rolling(period).mean()
    prev_c, prev_ma = df["close"].shift(1), ma.shift(1)
    spread = ((df["close"] - ma) / ma).abs()
    up = (prev_c <= prev_ma) & (df["close"] > ma) & (spread > entry_threshold)
    dn = (prev_c >= prev_ma) & (df["close"] < ma) & (spread > entry_threshold)
    return _hold_forward(up, pd.Series(False, index=df.index), dn, df.index)


def rsi_reversion(df, period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
    """RSI (simple mean, matching the kernel): long below oversold, flat above overbought."""
    delta = df["close"].diff()
    avg_gain = delta.clip(lower=0.0).rolling(period).sum() / period
    avg_loss = (-delta.clip(upper=0.0)).rolling(period).sum() / period
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = (100.0 - 100.0 / (1.0 + rs)).where(avg_loss != 0.0, 100.0)
    buy = rsi < oversold
    sell = rsi > overbought                                 # SELL -> FLAT
    return _hold_forward(buy, pd.Series(False, index=df.index), sell, df.index)


def donchian_breakout(df, lookback_window: int = 20, breakout_buffer_bps: float = 0.0):
    """Donchian channel breakout (bands exclude the current bar): long/short on breakout."""
    upper = df["high"].shift(1).rolling(lookback_window).max()
    lower = df["low"].shift(1).rolling(lookback_window).min()
    buy = df["close"] > upper * (1.0 + breakout_buffer_bps / 1e4)
    sell = df["close"] < lower * (1.0 - breakout_buffer_bps / 1e4)   # SELL -> SHORT
    return _hold_forward(buy, sell, pd.Series(False, index=df.index), df.index)


def multi_timeframe_ma_spread(df, primary_period: int = 20, secondary_period: int = 20,
                              threshold_bps: float = 0.0, secondary_factor: int = 4):
    """Primary SMA vs a slower higher-timeframe SMA; long/short on the signed spread.

    ``secondary_factor`` approximates the higher timeframe (bars per HTF bar): the secondary
    MA is a rolling mean over ``secondary_period * secondary_factor`` primary bars.
    """
    primary_ma = df["close"].rolling(primary_period).mean()
    secondary_ma = df["close"].rolling(max(2, secondary_period * secondary_factor)).mean()
    spread = (primary_ma - secondary_ma) / secondary_ma
    thr = threshold_bps / 1e4
    buy = spread > thr
    sell = spread < -thr                                    # SELL -> SHORT
    return _hold_forward(buy, sell, pd.Series(False, index=df.index), df.index)


def _col(df, name: str) -> pd.Series:
    """A df column if present, else an all-zero series (derivative feeds are optional)."""
    if name in df.columns:
        return df[name]
    return pd.Series(0.0, index=df.index)


def oi_funding_breakout(df, lookback_window: int = 20, breakout_buffer_bps: float = 0.0,
                        oi_threshold_bps: float = 0.0, max_funding_rate_bps: float = 100.0):
    """OI-confirmed Donchian breakout gated by funding (needs ``oi_change_bps`` /
    ``funding_rate_bps`` columns): long on an up-breakout when OI rises and funding
    is not extreme; short on the symmetric down-breakout."""
    upper = df["high"].shift(1).rolling(lookback_window).max()
    lower = df["low"].shift(1).rolling(lookback_window).min()
    oi = _col(df, "oi_change_bps")
    funding = _col(df, "funding_rate_bps")
    oi_ok = oi >= oi_threshold_bps
    buy = oi_ok & (df["close"] > upper * (1.0 + breakout_buffer_bps / 1e4)) & (funding <= max_funding_rate_bps)
    sell = oi_ok & (df["close"] < lower * (1.0 - breakout_buffer_bps / 1e4)) & (funding >= -max_funding_rate_bps)
    return _hold_forward(buy, sell, pd.Series(False, index=df.index), df.index)


def liquidation_reversal(df, long_liquidation_threshold_usd: float = 100_000.0,
                         short_liquidation_threshold_usd: float = 100_000.0,
                         liquidation_imbalance_ratio: float = 0.60):
    """Fade a one-sided liquidation cascade (needs ``long_liq_notional_usd`` /
    ``short_liq_notional_usd`` columns): a long-liquidation spike -> go long, a
    short-liquidation spike -> go short."""
    long_liq = _col(df, "long_liq_notional_usd")
    short_liq = _col(df, "short_liq_notional_usd")
    total = long_liq + short_liq
    safe_total = total.replace(0.0, np.nan)
    long_ratio = long_liq / safe_total
    short_ratio = short_liq / safe_total
    buy = (total > 0.0) & (long_liq >= long_liquidation_threshold_usd) & (long_ratio >= liquidation_imbalance_ratio)
    sell = (total > 0.0) & (short_liq >= short_liquidation_threshold_usd) & (short_ratio >= liquidation_imbalance_ratio)
    return _hold_forward(buy, sell, pd.Series(False, index=df.index), df.index)


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
