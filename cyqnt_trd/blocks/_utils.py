"""Internal helpers for blocks.

User code should import from the public sub-modules instead of this one.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

OHLCV_COLS = ("open", "high", "low", "close", "volume")
"""Canonical OHLCV column names (lower-case)."""

SeriesLike = Union[pd.Series, np.ndarray, Sequence[float]]


class IndicatorShapeError(ValueError):
    """A block's return value cannot be reduced to the one Series asked for."""


def first_param_is_df(fn: Callable[..., Any]) -> bool:
    """True if the block's first positional parameter is named ``df``.

    The blocks package has two calling conventions — ``indicators.atr(df, ...)``
    and ``indicators.ema(series, ...)`` — and a caller that guesses wrong gets a
    ``KeyError`` from inside the indicator rather than an explanation. Detecting
    it from the signature is what lets both be reached by one name.

    Lives here, and not in the YAML interpreter where it was written, because
    :func:`cyqnt_trd.blocks.universe.augment_with_indicator` needs exactly the
    same answer. Two copies would be two answers: an indicator handed ``close``
    by one caller and the whole frame by the other computes different numbers
    under the same spec, and nothing in either output would say which happened.
    """
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (ValueError, TypeError):
        return False
    if not params:
        return False
    return params[0].name == "df"


def select_indicator_component(
    out: Any,
    *,
    ref: str = "?",
    output: Optional[int] = None,
    column: Optional[str] = None,
) -> pd.Series:
    """Reduce a block's return value to the single Series an indicator must be.

    Blocks return four shapes and only one of them is directly usable, so this
    is where a caller finds out — with the fix in the message, rather than as a
    ``NoneType has no attribute`` three frames deeper.

    ``output`` picks a component of a tuple return (``supertrend`` -> value,
    direction) and ``column`` picks a column of a DataFrame return (``ichimoku``,
    ``pivot_points``). Shared with the YAML interpreter on purpose: the index
    ``output: 1`` names is the difference between a Supertrend LEVEL and a
    Supertrend DIRECTION, and two implementations of "which one is 1" would show
    up as a spec that validates and then screens on the wrong column.
    """
    if isinstance(out, tuple):
        if output is None and len(out) > 1:
            raise IndicatorShapeError(
                f"{ref!r} returns a {len(out)}-tuple; add 'output: <0..{len(out) - 1}>' "
                f"to say which component you mean"
            )
        idx = int(0 if output is None else output)
        if idx < 0 or idx >= len(out):
            raise IndicatorShapeError(
                f"{ref!r} returns a {len(out)}-tuple; output index {idx} out of range"
            )
        out = out[idx]

    if isinstance(out, pd.DataFrame):
        if column is None:
            raise IndicatorShapeError(
                f"{ref!r} returns a DataFrame; add 'column: <name>' to pick one. "
                f"Available: {list(out.columns)[:12]}"
            )
        if column not in out.columns:
            raise IndicatorShapeError(
                f"{ref!r} returns a DataFrame without column {column!r}; "
                f"available: {list(out.columns)[:12]}"
            )
        out = out[column]

    if not isinstance(out, pd.Series):
        raise IndicatorShapeError(
            f"{ref!r} returned {type(out).__name__}, which cannot be used as an "
            f"indicator — an indicator must reduce to one pandas Series. Blocks "
            f"that return a plan, a score or a scalar belong in the section that "
            f"consumes them (risk.exit / sizing), not in signals.indicators"
        )
    return out


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
