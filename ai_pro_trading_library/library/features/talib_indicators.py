"""TA-Lib indicator wrappers (optional `[talib]` extra).

Per `docs/architecture/dependency-strategy.md`:

- Default canonical indicators live in `library.features.indicators` (pandas
  rolling, no C dependency, 1e-9 parity vs `cyqnt_trd.blocks`).
- These TA-Lib wrappers are **opt-in via `pip install
  ai-pro-trading-library[talib]`**, and exist for two reasons:
  1. Expand the catalogue cheaply (KAMA / MAMA / ULTOSC / NATR / MFI / ROC /
     AROON / ADOSC / BOP / TRIX / DPO / CMO / WILLR / etc. — TA-Lib has 150+
     indicators we haven't reimplemented).
  2. Performance: TA-Lib's C kernels are 2-5× faster than pandas rolling on
     some indicators.

TA-Lib's RSI / MACD / ATR have **slightly different first-bar semantics**
than blocks-style Wilder smoothing — values diverge in the first ~30 bars
and converge afterward. Tests use a relaxed tolerance (~1e-3 after warm-up).

Usage:

    pip install 'ai-pro-trading-library[talib]'

    from ai_pro_trading_library.library.features.talib_indicators import rsi_talib, kama
    out = rsi_talib(close_series, 14)
"""

from __future__ import annotations

import pandas as pd


def _talib():
    """Lazy-import talib with a friendly error message."""
    try:
        import talib  # type: ignore[import-not-found]
        return talib
    except ImportError as e:
        raise ImportError(
            "TA-Lib is not installed. To use library.features.talib_indicators:\n"
            "  macOS:   brew install ta-lib\n"
            "  Linux:   apt install libta-lib0-dev (or build from source)\n"
            "  Windows: install prebuilt wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/\n"
            "Then:    pip install 'ai-pro-trading-library[talib]'\n"
            "See docs/architecture/dependency-strategy.md for details."
        ) from e


def is_available() -> bool:
    """Quick check — returns True if `import talib` works."""
    try:
        import talib  # noqa: F401
        return True
    except ImportError:
        return False


def _series(values, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.astype(float)
    return pd.Series(values, name=name).astype(float)


# ---------------------------------------------------------------------------
# 5 example wrappers covering the cheap-catalog-expansion case
# ---------------------------------------------------------------------------


def rsi_talib(close, period: int = 14) -> pd.Series:
    """TA-Lib RSI. Note: TA-Lib uses SMA-then-Wilder for the first period; the
    canonical `library.features.indicators.rsi` uses pure Wilder. Values
    diverge in the first ~30 bars."""
    s = _series(close, "close")
    out = _talib().RSI(s.values, timeperiod=period)
    return pd.Series(out, index=s.index)


def macd_talib(
    close,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """TA-Lib MACD. Returns `(macd_line, signal_line, histogram)`."""
    s = _series(close, "close")
    line, sig, hist = _talib().MACD(
        s.values, fastperiod=fast, slowperiod=slow, signalperiod=signal
    )
    return (
        pd.Series(line, index=s.index),
        pd.Series(sig, index=s.index),
        pd.Series(hist, index=s.index),
    )


def kama(close, period: int = 30) -> pd.Series:
    """Kaufman Adaptive Moving Average — not in the canonical indicator set."""
    s = _series(close, "close")
    return pd.Series(_talib().KAMA(s.values, timeperiod=period), index=s.index)


def ultosc(
    high,
    low,
    close,
    timeperiod1: int = 7,
    timeperiod2: int = 14,
    timeperiod3: int = 28,
) -> pd.Series:
    """Ultimate Oscillator — three-timeframe momentum normalised to 0-100."""
    h = _series(high, "high")
    l = _series(low, "low")
    c = _series(close, "close")
    out = _talib().ULTOSC(
        h.values,
        l.values,
        c.values,
        timeperiod1=timeperiod1,
        timeperiod2=timeperiod2,
        timeperiod3=timeperiod3,
    )
    return pd.Series(out, index=c.index)


def natr(high, low, close, period: int = 14) -> pd.Series:
    """Normalised ATR (ATR as % of price) — useful for cross-asset comparison."""
    h = _series(high, "high")
    l = _series(low, "low")
    c = _series(close, "close")
    out = _talib().NATR(h.values, l.values, c.values, timeperiod=period)
    return pd.Series(out, index=c.index)


__all__ = [
    "is_available",
    "kama",
    "macd_talib",
    "natr",
    "rsi_talib",
    "ultosc",
]
