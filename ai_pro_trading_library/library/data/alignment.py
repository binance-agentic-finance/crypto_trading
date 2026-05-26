"""Point-in-time alignment helpers.

Mirrors `cyqnt_trd.standard_bot.data.alignment` so cases that depend on
"decision_as_of" / last-closed-bar semantics behave identically. Two
shape differences vs the source:

1. `MarketBundle` here is `frames[symbol][timeframe] -> pandas.DataFrame`
   instead of `bars[(symbol, timeframe)] -> list[Bar]`. `filter_market_bundle`
   filters DataFrame rows whose `close_time` (or `timestamp`) is
   `<= decision_as_of`.
2. Social and on-chain bundles are not yet ported (Step 23 deferred), so
   only the market path is implemented here. `assemble_snapshot` is
   intentionally omitted until DataSnapshot/SocialFeedBundle/
   OnChainSignalBundle land.

`timeframe_to_ms`, `resolve_decision_as_of`, and `AlignmentPolicy` are
pure-math/dataclass parity copies — L1 1e-9.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from ai_pro_trading_library.library.core.protocols import MarketBundle


_TIMEFRAME_TO_MS: dict[str, int] = {
    "1s": 1000,
    "1m": 60 * 1000,
    "3m": 3 * 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "2h": 2 * 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "6h": 6 * 60 * 60 * 1000,
    "8h": 8 * 60 * 60 * 1000,
    "12h": 12 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
    "3d": 3 * 24 * 60 * 60 * 1000,
    "1w": 7 * 24 * 60 * 60 * 1000,
    "1M": 30 * 24 * 60 * 60 * 1000,
}


@dataclass(frozen=True)
class AlignmentPolicy:
    policy_id: str = "bar_close_v1"
    primary_timeframe: str = "1m"
    use_observable_at: bool = False
    require_confirmed_onchain: bool = False


def timeframe_to_ms(timeframe: str) -> int:
    if timeframe not in _TIMEFRAME_TO_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return _TIMEFRAME_TO_MS[timeframe]


def resolve_decision_as_of(now_ts_ms: int, timeframe: str) -> int:
    """Return the close timestamp of the last fully closed bar for `timeframe`."""
    interval_ms = timeframe_to_ms(timeframe)
    closed_bar_open = (now_ts_ms // interval_ms) * interval_ms - interval_ms
    return closed_bar_open + interval_ms - 1


def filter_market_bundle(
    bundle: MarketBundle | None,
    decision_as_of: int,
    *,
    timestamp_column: str = "close_time",
) -> MarketBundle | None:
    """Trim each DataFrame to rows with timestamp <= `decision_as_of`.

    Defaults to the `close_time` column (matches `BinanceRestAdapter`
    output). Falls back to `timestamp` if `close_time` is absent. Rows
    are filtered with `<=` to mirror `standard_bot.data.alignment`.
    """
    if bundle is None:
        return None
    new_frames: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol, by_tf in bundle.frames.items():
        new_frames[symbol] = {}
        for tf, frame in by_tf.items():
            col = timestamp_column if timestamp_column in frame.columns else "timestamp"
            if col not in frame.columns:
                # frame has neither — pass through unchanged
                new_frames[symbol][tf] = frame
                continue
            new_frames[symbol][tf] = frame[frame[col] <= decision_as_of].reset_index(drop=True)
    return MarketBundle(frames=new_frames, metadata=bundle.metadata)


__all__ = [
    "AlignmentPolicy",
    "filter_market_bundle",
    "resolve_decision_as_of",
    "timeframe_to_ms",
]
