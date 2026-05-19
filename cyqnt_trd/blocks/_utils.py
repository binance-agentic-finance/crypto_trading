"""Internal helpers for blocks.

User code should import from the public sub-modules instead of this one.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Union

import numpy as np
import pandas as pd

OHLCV_COLS = ("open", "high", "low", "close", "volume")
"""Canonical OHLCV column names (lower-case)."""

SeriesLike = Union[pd.Series, np.ndarray, Sequence[float]]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with column names lower-cased and stripped.

    Accepts both ``"Close"`` and ``"close"`` to be friendly to data from
    different sources without forcing every caller to rename columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected pandas.DataFrame, got {type(df).__name__}")
    return df.rename(columns=lambda c: str(c).strip().lower())


def ensure_df(df: pd.DataFrame, *, required: Iterable[str] = OHLCV_COLS) -> pd.DataFrame:
    """Validate that *df* contains the required columns and normalise names.

    Parameters
    ----------
    df:
        Input DataFrame. Must contain at minimum the required columns
        (case-insensitive).
    required:
        Required column names. Defaults to standard OHLCV.

    Returns
    -------
    pd.DataFrame
        A view (or copy) of *df* with columns lower-cased.

    Raises
    ------
    TypeError
        If *df* is not a DataFrame.
    ValueError
        If required columns are missing.
    """
    df = normalize_columns(df)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "DataFrame is missing required columns: {missing}. Got: {got}".format(
                missing=list(missing), got=list(df.columns)
            )
        )
    return df


def ensure_series(x: SeriesLike, name: str = "value") -> pd.Series:
    """Coerce *x* into a ``pandas.Series``."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, np.ndarray):
        if x.ndim != 1:
            raise ValueError(f"expected 1-D array, got shape {x.shape}")
        return pd.Series(x, name=name)
    if isinstance(x, (list, tuple)):
        return pd.Series(list(x), name=name)
    raise TypeError(f"cannot convert {type(x).__name__} to Series")


def safe_divide(
    numerator: SeriesLike, denominator: SeriesLike, fill: float = 0.0
) -> pd.Series:
    """Divide two series, replacing inf / NaN with *fill*."""
    a = ensure_series(numerator)
    b = ensure_series(denominator)
    out = a / b.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


def boolean_series(x: SeriesLike, *, length_hint: int = 0) -> pd.Series:
    """Coerce *x* into a boolean ``pandas.Series``."""
    s = ensure_series(x).fillna(False)
    if s.dtype != bool:
        s = s.astype(bool)
    if length_hint and len(s) != length_hint:
        raise ValueError(f"expected length {length_hint}, got {len(s)}")
    return s


def align_index(*series: pd.Series) -> Sequence[pd.Series]:
    """Reindex all *series* to the union of their indices, forward-filled.

    Used internally to combine series of different rolling-window lengths.
    """
    if not series:
        return ()
    union = series[0].index
    for s in series[1:]:
        union = union.union(s.index)
    return [s.reindex(union) for s in series]


def crossover(left: pd.Series, right: pd.Series) -> pd.Series:
    """Return a boolean series that is True on the bar *left* crosses above *right*."""
    left = ensure_series(left)
    right = ensure_series(right)
    prev = (left.shift(1) <= right.shift(1))
    now = (left > right)
    return (prev & now).fillna(False)


def crossunder(left: pd.Series, right: pd.Series) -> pd.Series:
    """Return a boolean series that is True on the bar *left* crosses below *right*."""
    left = ensure_series(left)
    right = ensure_series(right)
    prev = (left.shift(1) >= right.shift(1))
    now = (left < right)
    return (prev & now).fillna(False)


def rolling_max(s: SeriesLike, period: int) -> pd.Series:
    """Rolling maximum over *period* bars."""
    return ensure_series(s).rolling(window=period, min_periods=period).max()


def rolling_min(s: SeriesLike, period: int) -> pd.Series:
    """Rolling minimum over *period* bars."""
    return ensure_series(s).rolling(window=period, min_periods=period).min()


def pct_change_safe(s: SeriesLike, periods: int = 1) -> pd.Series:
    """Pct-change that fills the first *periods* with 0 instead of NaN."""
    return ensure_series(s).pct_change(periods=periods).fillna(0.0)


def positive_int(value: int, name: str) -> int:
    """Validate that *value* is a positive integer."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def non_negative_float(value: float, name: str) -> float:
    """Validate that *value* is a finite, non-negative float."""
    fv = float(value)
    if not np.isfinite(fv):
        raise ValueError(f"{name} must be finite, got {value}")
    if fv < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return fv
