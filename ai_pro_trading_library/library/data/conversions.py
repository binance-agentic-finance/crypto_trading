"""Conversions between canonical `Bar` lists, pandas frames, and atomic `Candle`.

Per `docs/architecture/data-layer-design.md` the canonical L1 shape is
`list[Bar]`. These helpers bridge to:

- pandas DataFrame (for `library.features.*` Series-in/Series-out math)
- atomic-style Candle list (for `library.features.atomic_signals.*`)

The atomic `Candle` is duck-typed (any object with the same attribute
names works), so this module does **not** import from atomic_strategy_lib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from ai_pro_trading_library.library.core.protocols import Bar


_BAR_NUMERIC_COLS = ("open", "high", "low", "close", "volume", "quote_volume",
                     "taker_buy_base", "taker_buy_quote")


def bars_to_frame(bars: Iterable[Bar]) -> pd.DataFrame:
    """Convert a list of `Bar` to a DataFrame indexed by `close_time` (timestamp ms).

    Required columns: `open, high, low, close, volume, close_time`.
    Optional: `quote_volume, trades, taker_buy_base, taker_buy_quote, open_time`.
    """
    rows: list[dict[str, Any]] = []
    for b in bars:
        rows.append(
            {
                "open_time": b.open_time,
                "close_time": b.timestamp,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "quote_volume": b.quote_volume,
                "trades": b.trades,
                "taker_buy_base": b.taker_buy_base,
                "taker_buy_quote": b.taker_buy_quote,
                "instrument_id": b.instrument_id,
                "timeframe": b.timeframe,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.set_index("close_time", drop=False)


def frame_to_bars(
    df: pd.DataFrame,
    instrument_id: str,
    timeframe: str,
    *,
    confirmed: bool = True,
) -> list[Bar]:
    """Convert a DataFrame to a list of `Bar`. Columns: open/high/low/close/volume,
    `close_time` (or use the index), optional `quote_volume / trades /
    taker_buy_base / taker_buy_quote / open_time`."""
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
    if "close_time" in df.columns:
        ts_col = df["close_time"]
    else:
        ts_col = df.index
    bars: list[Bar] = []
    for i, (_, row) in enumerate(df.iterrows()):
        bars.append(
            Bar(
                instrument_id=instrument_id.upper(),
                timeframe=timeframe,
                timestamp=int(ts_col.iloc[i] if hasattr(ts_col, "iloc") else ts_col[i]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                quote_volume=float(row["quote_volume"]) if "quote_volume" in df.columns and pd.notna(row["quote_volume"]) else None,
                trades=int(row["trades"]) if "trades" in df.columns and pd.notna(row["trades"]) else None,
                taker_buy_base=float(row["taker_buy_base"]) if "taker_buy_base" in df.columns and pd.notna(row["taker_buy_base"]) else None,
                taker_buy_quote=float(row["taker_buy_quote"]) if "taker_buy_quote" in df.columns and pd.notna(row["taker_buy_quote"]) else None,
                open_time=int(row["open_time"]) if "open_time" in df.columns and pd.notna(row["open_time"]) else None,
                confirmed=confirmed,
            )
        )
    return bars


@dataclass
class _AtomicCandle:
    """Local atomic-Candle-shape facade. Duck-types `atomic.core.types.Candle`."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0
    trades: int = 0


def bars_to_atomic_candles(bars: Iterable[Bar]) -> list:
    """Convert canonical `Bar` list to atomic-shape Candle list.

    Returns objects with `.timestamp/.open/.high/.low/.close/.volume/.quote_volume/.trades`,
    matching `atomic_strategy_lib.core.types.Candle`. The atomic library is not
    imported; the returned objects are plain dataclasses with the right shape.
    """
    out: list[_AtomicCandle] = []
    for b in bars:
        out.append(
            _AtomicCandle(
                timestamp=b.timestamp,
                open=b.open,
                high=b.high,
                low=b.low,
                close=b.close,
                volume=b.volume,
                quote_volume=b.quote_volume if b.quote_volume is not None else 0.0,
                trades=b.trades if b.trades is not None else 0,
            )
        )
    return out


def candles_to_bars(
    candles: Iterable[Any],
    instrument_id: str,
    timeframe: str,
    *,
    confirmed: bool = True,
) -> list[Bar]:
    """Convert atomic Candle-shape objects to canonical `Bar` list.

    Accepts any duck-typed Candle (`.timestamp/.open/.high/.low/.close/.volume`,
    optional `.quote_volume/.trades`).
    """
    bars: list[Bar] = []
    for c in candles:
        bars.append(
            Bar(
                instrument_id=instrument_id.upper(),
                timeframe=timeframe,
                timestamp=int(c.timestamp),
                open=float(c.open),
                high=float(c.high),
                low=float(c.low),
                close=float(c.close),
                volume=float(c.volume),
                quote_volume=float(getattr(c, "quote_volume", 0.0)) or None,
                trades=int(getattr(c, "trades", 0)) or None,
                confirmed=confirmed,
            )
        )
    return bars


__all__ = [
    "bars_to_atomic_candles",
    "bars_to_frame",
    "candles_to_bars",
    "frame_to_bars",
]
