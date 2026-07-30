"""Assemble a ``DataSnapshot`` carrying named long-form ADVISORY frames.

Bridges the advisory data adapters
(:mod:`cyqnt_trd.standard_bot.advisory.data`) into the standard
``DataSnapshot.frames`` slot, so :class:`~..advisory.base.AdvisoryBot` plugins
run on the standard registry / ``SignalPlugin`` route with no manual glue — the
same way :mod:`.selection_assembler` feeds the SELECTION route and
:class:`.HistoricalSnapshotAssembler` feeds the market/trade route.

What it guarantees:

* every frame lands in ``DataSnapshot.frames[name]``;
* every fetch status lands in ``SnapshotMeta.source_status[name]``, so an empty
  required frame can be told apart from "the source really had nothing";
* rows whose ``available_time`` is after ``decision_as_of`` are dropped (PIT
  gate) and counted in ``SnapshotMeta.warnings`` — the bot's own
  ``FrameContract.validate`` then stays a second, redundant guard;
* all timestamps are floored to milliseconds, the resolution of
  ``SnapshotMeta.decision_as_of``.

Live use::

    from cyqnt_trd.standard_bot.data import run_advisory

    result = run_advisory("derivatives_positioning_monitor", symbols=["BTCUSDT"])
    result.batch.signals

Offline / test: pass ``frames={name: FetchFrame(...)}`` yourself and no network
call is made.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from ..core import DataSnapshot, SnapshotMeta, SourceStatus

__all__ = [
    "build_advisory_snapshot",
    "run_advisory",
]

_ADVISORY_NS = uuid.UUID("2f5a1d84-9c31-5f0a-9a2b-7c4d6e8f1a03")

#: column carrying the availability timestamp used by the PIT gate
_AVAILABLE_TIME_COLUMN = "available_time"

#: ``SnapshotMeta.decision_as_of`` is integer milliseconds, so every timestamp
#: the bots compare against it is floored to the same resolution. Without this a
#: sub-millisecond capture time reads as "after the cut-off" and a perfectly
#: good live frame gets rejected as future data.
_TIME_COLUMNS = ("available_time", "event_time", "published_at")
_TIME_RESOLUTION = "ms"


def build_advisory_snapshot(
    *,
    frames: Mapping[str, Any],
    decision_as_of: Optional[Any] = None,
    version: str = "advisory/v1",
    partial_ok: bool = True,
    trace_id: Optional[str] = None,
) -> DataSnapshot:
    """Assemble a PIT-gated :class:`DataSnapshot` for advisory bots.

    Parameters
    ----------
    frames:
        ``{frame_name: FetchFrame}``. Frame names must match the
        ``FrameContract.name`` the bot declares (``market_metrics`` /
        ``news_events``).
    decision_as_of:
        Decision cut-off. Defaults to the **latest ``available_time`` present
        across all frames** (falling back to wall clock when every frame is
        empty), which is the honest "as of the freshest thing we actually
        have" reading for a live monitor. Pass an explicit value to replay.
    """
    from ..advisory.contracts import epoch_ms, utc_datetime

    if not frames:
        raise ValueError("build_advisory_snapshot requires at least one frame")

    cutoff = (
        utc_datetime(decision_as_of)
        if decision_as_of is not None
        else _latest_available_time(frames)
    )
    cutoff_ts = pd.Timestamp(cutoff).floor(_TIME_RESOLUTION)
    cutoff = cutoff_ts.to_pydatetime()

    gated: Dict[str, Any] = {}
    source_status: Dict[str, SourceStatus] = {}
    warnings: List[str] = []

    for name, fetched in frames.items():
        frame = fetched.frame
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame %r must be a pandas DataFrame" % name)
        frame = frame.copy()
        if not frame.empty:
            for column in _TIME_COLUMNS:
                if column in frame.columns:
                    parsed = pd.to_datetime(frame[column], utc=True, errors="coerce")
                    if parsed.notna().any():
                        frame[column] = parsed.dt.floor(_TIME_RESOLUTION)
        if not frame.empty and _AVAILABLE_TIME_COLUMN in frame.columns:
            available = pd.to_datetime(
                frame[_AVAILABLE_TIME_COLUMN], utc=True, errors="coerce"
            )
            future = available.isna() | (available > cutoff_ts)
            dropped = int(future.sum())
            if dropped:
                warnings.append(
                    "%s: dropped %d row(s) with available_time after decision_as_of"
                    % (name, dropped)
                )
                frame = frame[~future].reset_index(drop=True)
        gated[name] = frame
        status = _coerce_status(fetched.status)
        # A frame that fetched "ok" but is empty AFTER gating is no longer a
        # trustworthy "nothing happened" — say so instead of implying no events.
        if status is SourceStatus.OK and frame.empty and not fetched.frame.empty:
            status = SourceStatus.DEGRADED
            warnings.append("%s: all rows were gated out by decision_as_of" % name)
        source_status[name] = status
        if fetched.warning:
            warnings.append("%s: %s" % (name, fetched.warning))

    snapshot_id = str(
        uuid.uuid5(
            _ADVISORY_NS,
            "advisory|%s|%s" % (",".join(sorted(frames)), epoch_ms(cutoff)),
        )
    )
    return DataSnapshot(
        version=version,
        meta=SnapshotMeta(
            snapshot_id=snapshot_id,
            assembled_at=int(time.time() * 1000),
            partial_ok=partial_ok,
            source_status=source_status,
            warnings=warnings,
            decision_as_of=epoch_ms(cutoff),
            trace_id=trace_id,
        ),
        frames=gated,
    )


def run_advisory(
    bot_id: str,
    *,
    frames: Optional[Mapping[str, Any]] = None,
    symbols: Optional[Sequence[str]] = None,
    config: Optional[Mapping[str, Any]] = None,
    venue: str = "binance",
    product: str = "usd_m_perpetual",
    market: str = "futures",
    decision_as_of: Optional[Any] = None,
    news_page_size: int = 50,
    registry: Optional[Any] = None,
    previous_states: Optional[Dict[str, Any]] = None,
):
    """Assemble a snapshot and run one advisory bot **through the standard registry**.

    Execution goes over ``SignalPluginRegistry.run_pipeline_step`` — the same
    call the trade and selection routes use — so advisory bots get the standard
    config factory, ``SignalState`` handling and incremental de-duplication
    instead of a private code path.

    Pass ``frames`` for offline/replay use. Otherwise the frames the bot
    declares in ``meta.required_frames`` / ``meta.optional_frames`` are fetched
    live: ``market_metrics`` from ``BinanceMarketMetricAdapter`` (needs
    ``symbols``), ``news_events`` from ``SquarePublicAdapter``.

    Returns a ``PipelineStepResult`` (``.batch`` / ``.states``); feed
    ``.states`` back in as ``previous_states`` on the next cycle to keep
    suppressing alerts that were already delivered.
    """
    from ..advisory import create_advisory_bot
    from ..entrypoints.common import build_advisory_pipeline, make_registry

    bot = create_advisory_bot(bot_id)
    if frames is None:
        frames = _fetch_declared_frames(
            bot,
            symbols=symbols,
            venue=venue,
            product=product,
            market=market,
            news_page_size=news_page_size,
        )
    snapshot = build_advisory_snapshot(frames=frames, decision_as_of=decision_as_of)
    registry = registry if registry is not None else make_registry()
    return registry.run_pipeline_step(
        snapshot,
        build_advisory_pipeline(bot=bot_id, config=config).plugin_chain,
        previous_states=previous_states,
    )


def _fetch_declared_frames(
    bot,
    *,
    symbols: Optional[Sequence[str]],
    venue: str,
    product: str,
    market: str,
    news_page_size: int,
) -> Dict[str, Any]:
    from ..advisory.contracts import MARKET_METRIC_FRAME, NEWS_EVENT_FRAME
    from ..advisory.data import BinanceMarketMetricAdapter, SquarePublicAdapter

    declared = list(bot.meta.required_frames) + list(bot.meta.optional_frames)
    fetched: Dict[str, Any] = {}
    for contract in declared:
        if contract.name == MARKET_METRIC_FRAME.name:
            if not symbols:
                raise ValueError(
                    "%s needs %s; pass symbols=[...] or supply frames yourself"
                    % (bot.plugin_id, MARKET_METRIC_FRAME.name)
                )
            adapter = BinanceMarketMetricAdapter(
                venue=venue, product=product, market=market
            )
            fetched[contract.name] = adapter.fetch_market_metrics(symbols)
        elif contract.name == NEWS_EVENT_FRAME.name:
            fetched[contract.name] = SquarePublicAdapter().fetch_news_events(
                page_size=news_page_size, venue=venue, product=product
            )
        else:
            raise ValueError(
                "no live adapter registered for frame %r; supply it via frames="
                % contract.name
            )
    return fetched


def _latest_available_time(frames: Mapping[str, Any]):
    from ..advisory.contracts import utc_datetime

    latest = None
    for fetched in frames.values():
        frame = fetched.frame
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        if _AVAILABLE_TIME_COLUMN not in frame.columns:
            continue
        available = pd.to_datetime(
            frame[_AVAILABLE_TIME_COLUMN], utc=True, errors="coerce"
        ).dropna()
        if available.empty:
            continue
        candidate = available.max()
        latest = candidate if latest is None or candidate > latest else latest
    if latest is None:
        return utc_datetime(pd.Timestamp.now(tz="UTC"))
    return utc_datetime(latest)


def _coerce_status(status: Any) -> SourceStatus:
    if isinstance(status, SourceStatus):
        return status
    try:
        return SourceStatus(str(status))
    except ValueError:
        return SourceStatus.ERROR
