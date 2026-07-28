"""Assemble a ``DataSnapshot`` carrying cross-sectional SELECTION inputs.

Bridges the live Binance Square news API (:mod:`cyqnt_trd.data_cli.news`) and
the 24h perpetual universe (:mod:`cyqnt_trd.blocks.universe`) into the standard
``DataSnapshot.universe`` slot, so ``register_selection()`` strategies (N1/N2)
run on the standard registry / ``SignalPlugin`` route with no manual glue —
the same way ``HistoricalSnapshotAssembler`` feeds the market/trade route.

Live use::

    from cyqnt_trd.standard_bot.data.selection_assembler import run_selection
    import strategies.news.news_catalyst_selector   # noqa: F401 (registers it)

    cands = run_selection("news_catalyst_selector")                       # auto-fetch
    cands = run_selection("social_heat_breakout",                          # + candidate klines
                          kline_top=10, kline_interval="1h")

Offline / test: pass ``universe_df`` / ``ticker_rank_df`` (and, for N2,
``ticker_rank_prev_df`` / ``klines``) yourself and no network call is made.

NOTE: this is the **live** assembler. Historical/PIT-gated Square backtest data
has limited depth and is handled separately (see the derivatives/market path
which already has ``HistoricalSnapshotAssembler`` + enrichment).
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

import pandas as pd

from ..core import BundleMeta, DataSnapshot, SnapshotMeta, UniverseBundle

__all__ = ["build_universe_bundle", "build_selection_snapshot", "run_selection"]

_SELECTION_NS = uuid.UUID("b6f2a4c1-3d5e-5a9b-8c7d-2e1f0a9b8c7d")


def build_universe_bundle(
    *,
    market_type: str = "futures",
    as_of_ms: Optional[int] = None,
    universe_df: Optional[pd.DataFrame] = None,
    ticker_rank_df: Optional[pd.DataFrame] = None,
    ticker_rank_prev_df: Optional[pd.DataFrame] = None,
    klines: Optional[Dict[str, pd.DataFrame]] = None,
    window: str = "24h",
    limit: int = 100,
    kline_top: int = 0,
    kline_interval: str = "1h",
    kline_limit: int = 200,
) -> UniverseBundle:
    """Build a :class:`UniverseBundle`.

    Frames left as ``None`` are fetched live from the Square API + the 24h
    ticker snapshot. If ``kline_top > 0`` also fetch klines for the top-N
    ``ticker_rank`` symbols (needed by breakout-confirming selectors like N2).
    """
    if as_of_ms is None:
        as_of_ms = int(time.time() * 1000)
    if universe_df is None:
        from ...blocks import universe as U
        universe_df = U.fetch_perpetual_universe(market_type=market_type)
    if ticker_rank_df is None:
        from ...data_cli import fetch_ticker_rank
        ticker_rank_df = fetch_ticker_rank(window=window, limit=limit)

    klines = dict(klines or {})
    if kline_top > 0 and ticker_rank_df is not None and not ticker_rank_df.empty:
        from ...blocks import data as _data
        top_tickers = [str(t).upper() for t in ticker_rank_df["ticker"].head(kline_top).tolist()]
        for tk in top_tickers:
            symbol = f"{tk}USDT"
            if symbol in klines:
                continue
            try:
                klines[symbol] = _data.fetch_klines(
                    symbol, kline_interval, limit=kline_limit, market_type=market_type
                )
            except Exception:
                # one symbol failing to fetch must not sink the whole bundle
                continue

    return UniverseBundle(
        as_of=int(as_of_ms),
        universe=universe_df,
        ticker_rank=ticker_rank_df,
        ticker_rank_prev=ticker_rank_prev_df,
        klines=klines,
        meta=BundleMeta(data_source="binance_square+24h_ticker", fetched_at=int(as_of_ms)),
    )


def build_selection_snapshot(
    *,
    market_type: str = "futures",
    as_of_ms: Optional[int] = None,
    version: str = "mvp/v1",
    **bundle_kwargs,
) -> DataSnapshot:
    """Wrap a :class:`UniverseBundle` in a ``DataSnapshot`` (the standard bot input)."""
    if as_of_ms is None:
        as_of_ms = int(time.time() * 1000)
    ub = build_universe_bundle(market_type=market_type, as_of_ms=as_of_ms, **bundle_kwargs)
    snapshot_id = str(uuid.uuid5(_SELECTION_NS, f"selection|{market_type}|{as_of_ms}"))
    return DataSnapshot(
        version=version,
        meta=SnapshotMeta(
            snapshot_id=snapshot_id,
            assembled_at=int(as_of_ms),
            decision_as_of=int(as_of_ms),
            partial_ok=True,
        ),
        universe=ub,
    )


def run_selection(
    strategy_id: str,
    *,
    market_type: str = "futures",
    config: Optional[dict] = None,
    **snapshot_kwargs,
) -> List[Dict]:
    """Build a selection snapshot and run the registered SELECTION strategy on
    the standard ``SignalPlugin`` route, returning the ranked candidate dicts.

    ``strategy_id`` must have been registered via
    :func:`cyqnt_trd.blocks.strategy.register_selection` (importing the strategy
    module fires the registration). Extra kwargs are forwarded to
    :func:`build_selection_snapshot` — pass ``universe_df=...``/``ticker_rank_df=...``
    for offline use, or ``kline_top=10`` to fetch candidate klines live.
    """
    from types import SimpleNamespace

    from ...blocks import strategy as S

    plugin = S.get_selection_plugin(strategy_id)
    if plugin is None:
        raise ValueError(
            f"no SELECTION strategy registered as {strategy_id!r}; import its "
            "module first (register_selection fires at import time)"
        )
    snapshot = build_selection_snapshot(market_type=market_type, **snapshot_kwargs)
    cfg = {"market_type": market_type, **(config or {})}
    batch = plugin.run(snapshot, SimpleNamespace(**cfg))
    sels = batch.selection_signals()
    return sels[0].payload["candidates"] if sels else []
