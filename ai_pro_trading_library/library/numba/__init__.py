"""Numba-accelerated kernels for hot-path indicator math.

These kernels are functionally equivalent to the pandas implementations
in `library.features.indicators` but use numba's njit for ~50-200x
speedup on >100k bars. Without numba installed, the decorator falls
through to plain Python (no functional change, just slower).

Available helpers (all operate on plain `np.ndarray` inputs):
- `ema_series(values, period)` → EMA
- `rsi_series(values, period)` → Wilder's RSI
- `true_range_series(highs, lows, closes)` → ATR's TR component
- `atr_series(highs, lows, closes, period)` → ATR
- `macd_series(values, fast, slow, signal)` → (macd, signal, hist)

Pandas factories opt in via `library.features.indicators.use_numba()`
(future hook); today they always use pure pandas. This module exists
so users who want raw speed can call the kernels directly.
"""

from ai_pro_trading_library.library.numba.kernels import (
    NUMBA_AVAILABLE,
    atr_series,
    ema_series,
    macd_series,
    rsi_series,
    true_range_series,
)

__all__ = [
    "NUMBA_AVAILABLE",
    "atr_series",
    "ema_series",
    "macd_series",
    "rsi_series",
    "true_range_series",
]
