"""Adapters: convert between pandas DataFrames and atomic-style dataclass lists.

These helpers let you move freely between the two worlds:

* **DataFrame style** — preferred for backtesting, bulk computation, and all
  ``cyqnt_trd.blocks`` functions.
* **Dataclass style** — preferred by ``atomic_strategy_lib`` usage cases
  (individual Candle / Signal / TradePlan objects).

All functions are pure Python and have no runtime dependency beyond
``pandas`` and ``numpy``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .types import Candle, Signal, TradePlan

__all__ = [
    "df_to_candles",
    "candles_to_df",
    "signal_series_to_signals",
    "signals_to_series",
    "trade_plan_to_dict",
    "dict_to_trade_plan",
]


# ---------------------------------------------------------------------------
# Candle ↔ DataFrame
# ---------------------------------------------------------------------------

def df_to_candles(df: pd.DataFrame) -> List[Candle]:
    """Convert an OHLCV DataFrame into a list of atomic-style :class:`Candle` objects.

    The DataFrame must contain columns (case-insensitive):
    ``open``, ``high``, ``low``, ``close``, ``volume``.

    Optional columns: ``timestamp`` / ``close_time`` (used for
    ``Candle.timestamp`` in milliseconds), ``quote_volume``, ``trades``.

    If neither ``timestamp`` nor ``close_time`` is present the row's integer
    index is used as the timestamp.

    Parameters
    ----------
    df:
        OHLCV DataFrame.

    Returns
    -------
    List[Candle]
        One :class:`Candle` per row, preserving row order.

    Examples
    --------
    >>> from cyqnt_trd.compat.adapters import df_to_candles, candles_to_df
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "open": [100.0], "high": [110.0], "low": [95.0],
    ...     "close": [105.0], "volume": [1000.0], "close_time": [1_700_000_000_000],
    ... })
    >>> candles = df_to_candles(df)
    >>> len(candles), candles[0].close
    (1, 105.0)
    """
    df = df.copy()
    # Normalise column names to lower-case
    df.columns = [c.lower() for c in df.columns]

    _require_cols(df, ["open", "high", "low", "close", "volume"])

    # Resolve timestamp column
    if "timestamp" in df.columns:
        ts_col: Optional[str] = "timestamp"
    elif "close_time" in df.columns:
        ts_col = "close_time"
    elif "open_time" in df.columns:
        ts_col = "open_time"
    else:
        ts_col = None

    candles: List[Candle] = []
    for idx, row in df.iterrows():
        ts = int(row[ts_col]) if ts_col else int(idx)
        candles.append(
            Candle(
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                quote_volume=float(row.get("quote_volume", 0.0) or 0.0),
                trades=int(row.get("trades", 0) or 0),
            )
        )
    return candles


def candles_to_df(candles: List[Candle]) -> pd.DataFrame:
    """Convert a list of atomic-style :class:`Candle` objects into an OHLCV DataFrame.

    The returned DataFrame has columns:
    ``timestamp``, ``open``, ``high``, ``low``, ``close``, ``volume``,
    ``quote_volume``, ``trades``.

    ``timestamp`` values are milliseconds since epoch.  The DataFrame index
    is a default :class:`~pandas.RangeIndex`.

    Parameters
    ----------
    candles:
        List of :class:`Candle` instances (must be non-empty).

    Returns
    -------
    pd.DataFrame

    Examples
    --------
    >>> candles = [Candle(1_700_000_000_000, 100.0, 110.0, 95.0, 105.0, 1000.0)]
    >>> df = candles_to_df(candles)
    >>> list(df.columns)
    ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']
    """
    if not candles:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades"]
        )
    records = [c.to_dict() for c in candles]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Signal ↔ pandas Series
# ---------------------------------------------------------------------------

def signal_series_to_signals(
    series: pd.Series,
    signal_type: str,
    *,
    direction: str = "NEUTRAL",
    strength: float = 1.0,
) -> List[Signal]:
    """Convert a boolean pandas Series into a list of :class:`Signal` objects.

    One :class:`Signal` is created for every ``True`` (or truthy) value in
    ``series``.  The signal's ``value`` is the index label of the bar where
    the condition fired; ``name`` is set to ``signal_type``.

    Parameters
    ----------
    series:
        Boolean (or 0/1) Series aligned to a DataFrame index.
    signal_type:
        Name of the signal (e.g. ``"ma_cross"``, ``"rsi_oversold"``).
    direction:
        ``"BULLISH"``, ``"BEARISH"``, or ``"NEUTRAL"`` (default).
    strength:
        Strength value to assign to each fired signal (default 1.0).

    Returns
    -------
    List[Signal]

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series([False, True, False, True])
    >>> sigs = signal_series_to_signals(s, "test", direction="BULLISH")
    >>> len(sigs)
    2
    """
    signals: List[Signal] = []
    for idx, val in series.items():
        if val:
            signals.append(
                Signal(
                    name=signal_type,
                    value=float(idx) if isinstance(idx, (int, float, np.integer, np.floating)) else 0.0,
                    direction=direction,
                    strength=strength,
                    metadata={"bar_index": idx},
                )
            )
    return signals


def signals_to_series(
    signals: List[Signal],
    df_index: pd.Index,
) -> pd.Series:
    """Convert a list of :class:`Signal` objects back into a boolean Series.

    The resulting Series is aligned to ``df_index`` — bars at positions
    matching a signal's ``metadata["bar_index"]`` (or, if that key is absent,
    nearest bar whose integer position equals ``int(signal.value)``) are set
    to ``True``; all others are ``False``.

    Parameters
    ----------
    signals:
        List of :class:`Signal` instances.
    df_index:
        The DataFrame index to align against (e.g. ``df.index``).

    Returns
    -------
    pd.Series[bool]
        Boolean Series with the same index as ``df_index``.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.RangeIndex(5)
    >>> sigs = [Signal("x", 1.0, "BULLISH", metadata={"bar_index": 1}),
    ...         Signal("x", 3.0, "BULLISH", metadata={"bar_index": 3})]
    >>> ser = signals_to_series(sigs, idx)
    >>> list(ser)
    [False, True, False, True, False]
    """
    result = pd.Series(False, index=df_index, dtype=bool)
    index_set = set(df_index)
    for sig in signals:
        bar_idx = sig.metadata.get("bar_index", None)
        if bar_idx is not None and bar_idx in index_set:
            result.loc[bar_idx] = True
        else:
            # Fall back to integer position lookup
            try:
                pos = int(sig.value)
                if 0 <= pos < len(df_index):
                    result.iloc[pos] = True
            except (ValueError, TypeError):
                pass
    return result


# ---------------------------------------------------------------------------
# TradePlan serialization
# ---------------------------------------------------------------------------

def trade_plan_to_dict(plan: TradePlan) -> Dict[str, Any]:
    """Serialize a :class:`TradePlan` to a JSON-safe dict (for state.json compat).

    Parameters
    ----------
    plan:
        :class:`TradePlan` instance to serialize.

    Returns
    -------
    Dict[str, Any]
        Flat dict with all fields.  All values are JSON-primitives.

    Examples
    --------
    >>> from cyqnt_trd.compat.types import TradePlan, Verdict
    >>> plan = TradePlan("BTCUSDT", "LONG", Verdict.CANDIDATE, 0.75, 50000.0,
    ...                  48000.0, 0.04, 5.0, 250.0, 10.0, ["rsi_ok", "macd_ok"])
    >>> d = trade_plan_to_dict(plan)
    >>> d["symbol"]
    'BTCUSDT'
    """
    return plan.to_dict()


def dict_to_trade_plan(d: Dict[str, Any]) -> TradePlan:
    """Reconstruct a :class:`TradePlan` from a previously serialized dict.

    Parameters
    ----------
    d:
        Dict as produced by :func:`trade_plan_to_dict`.

    Returns
    -------
    TradePlan

    Examples
    --------
    >>> d = {"symbol": "BTCUSDT", "direction": "LONG", "verdict": "CANDIDATE",
    ...      "score": 0.75, "entry_price": 50000.0, "stop_price": 48000.0,
    ...      "stop_pct": 0.04, "leverage": 5.0, "notional": 250.0,
    ...      "max_loss": 10.0, "reasons": []}
    >>> plan = dict_to_trade_plan(d)
    >>> plan.symbol
    'BTCUSDT'
    """
    return TradePlan.from_dict(d)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )
