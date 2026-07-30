"""Declared slots for internal-domain data — fields here, client elsewhere.

Why this file is in the public repo and the client is not
--------------------------------------------------------
The internal BigData nodes (indicators API, futuresRadar, hot events, calendars,
token unlocks, macro, ETF flows, sector rotation, large on-chain flow…) are real
inputs a strategy wants. Their **client** cannot ship here: it hardcodes
corporate-network hostnames, and this repository is public. Naming those hosts —
even in a comment explaining why they are excluded — publishes them just as
effectively as calling them, so they are not written down anywhere in this repo.

The *field contract*, though, is not secret and is exactly what everyone needs
in order to build against these inputs:

* a strategy author needs to know a slot exists, what shape it has and what
  ``metric`` names it emits, so the code can be written and tested with
  synthetic rows;
* an input bundle needs a stable key for the slot, so a bundle produced inside
  the network and a bundle produced outside differ only in ``source_status``,
  never in structure;
* whoever wires the private client needs to know exactly what to normalise to.

So: **slots and columns are declared here, and the fetcher is referenced by
dotted path as a string.** Nothing in this module imports
``cyqnt_trd.data_cli.internal*``; if that package is absent, every slot simply
reports ``unavailable`` and a bundle built without it is still structurally
identical.

Normalisation
-------------
Each slot declares the canonical shape it lands in, so a bot reads an internal
metric with the same accessor it uses for funding or open interest:

    latest_metric(ctx, "internal_etf_flow", "flow", "BTCUSDT")

``value_columns`` lists which raw columns become ``metric`` rows in a
``MetricFrame``; ``time_column`` is the source of both ``event_time`` and (unless
``available_column`` says otherwise) ``available_time``.

⚠️ Most of these are Redis-TTL snapshots with **no point-in-time history**. A
slot marked ``pit_safe=False`` may be collected forward but must not be replayed
in a walk-forward backtest: doing so reuses today's value at every bar. That flag
travels with the declaration precisely so the mistake is hard to make silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["InternalSlot", "INTERNAL_SLOTS", "slot_frame_shapes",
           "normalize_internal_frame", "internal_client_available"]


@dataclass(frozen=True)
class InternalSlot:
    """One internal data node: where it lands, what it carries, can you replay it."""

    key: str
    shape: str
    fetcher: str
    description: str
    columns: Tuple[str, ...] = ()
    #: raw columns that become MetricFrame ``metric`` rows
    value_columns: Tuple[str, ...] = ()
    time_column: str = "event_time"
    available_column: Optional[str] = None
    #: False = snapshot only, no point-in-time history. Do not replay.
    pit_safe: bool = False
    instrument_column: Optional[str] = None


#: Every internal node we know how to normalise. Adding one is a row here plus a
#: fetcher in the private package — no change to the bundle builder or readers.
INTERNAL_SLOTS: Dict[str, InternalSlot] = {
    slot.key: slot for slot in (
        InternalSlot(
            key="internal_futures_radar", shape="MetricFrame@1.0",
            fetcher="cyqnt_trd.data_cli.internal_frames.fetch_futures_radar",
            description="futuresRadar 指標看板:每個 symbol 一組指標值",
            columns=("symbol", "metric", "value", "event_time"),
            value_columns=("value",), time_column="event_time",
            instrument_column="symbol", pit_safe=False),
        InternalSlot(
            key="internal_etf_flow", shape="MetricFrame@1.0",
            fetcher="cyqnt_trd.data_cli.internal_frames.fetch_etf_flow",
            description="現貨 ETF 每日淨流入 / 淨資產 / 收盤價",
            columns=("token", "date", "flow", "net_assets", "close_price"),
            value_columns=("flow", "net_assets", "close_price"),
            time_column="date", instrument_column="token", pit_safe=True),
        InternalSlot(
            key="internal_sector_flow", shape="MetricFrame@1.0",
            fetcher="cyqnt_trd.data_cli.internal_frames.fetch_sector_flow",
            description="板塊資金流:淨流入 / 市值 / 漲跌 / 熱度",
            columns=("category", "net_inflow", "market_cap", "change_pct",
                     "heat", "members"),
            value_columns=("net_inflow", "market_cap", "change_pct", "heat"),
            instrument_column="category", pit_safe=False),
        InternalSlot(
            key="internal_large_flow", shape="MetricFrame@1.0",
            fetcher="cyqnt_trd.data_cli.internal_frames.fetch_large_flow",
            description="大額進出金:窗格內存提量與大額筆數",
            columns=("window_start_time", "window_duration", "total_deposit_qty",
                     "total_withdraw_qty", "total_deposit_amt", "total_withdraw_amt",
                     "large_deposit_qty", "large_withdraw_qty", "large_deposit_amt",
                     "large_withdraw_amt", "avg_amt", "total_signal"),
            value_columns=("total_deposit_amt", "total_withdraw_amt",
                           "large_deposit_amt", "large_withdraw_amt", "total_signal"),
            time_column="window_start_time", pit_safe=True),
        InternalSlot(
            key="internal_hot_event", shape="EventFrame@1.0",
            fetcher="cyqnt_trd.data_cli.internal_frames.fetch_hot_event",
            description="熱點事件:排名 + 情緒 + 摘要 + 關聯幣種",
            columns=("event_id", "publish_time", "news_rank", "sentiment",
                     "category", "summary", "summary_cn", "related_coins",
                     "related_categories"),
            time_column="publish_time", pit_safe=True),
        InternalSlot(
            key="internal_event_upcoming", shape="EventFrame@1.0",
            fetcher="cyqnt_trd.data_cli.internal_frames.fetch_event_upcoming",
            description="即將發生的訊號卡:幣種 / 類型 / 多空 / 有效期",
            columns=("card_coin", "signal_type", "card_context", "signal_time",
                     "signal_expired_time", "is_bullish"),
            time_column="signal_time", instrument_column="card_coin", pit_safe=False),
        InternalSlot(
            key="internal_calendar", shape="EventFrame@1.0",
            fetcher="cyqnt_trd.data_cli.internal_frames.fetch_calendar",
            description="幣種事件曆",
            columns=("event_id", "event_time", "event_type", "coin",
                     "event_title", "deeplink"),
            time_column="event_time", instrument_column="coin", pit_safe=True),
        InternalSlot(
            key="internal_macro_calendar", shape="EventFrame@1.0",
            fetcher="cyqnt_trd.data_cli.internal_frames.fetch_macro_calendar",
            description="總經事件曆:實際 / 預期 / 前值",
            columns=("event_time", "event_type", "actual", "forecast", "previous"),
            time_column="event_time", pit_safe=True),
        InternalSlot(
            key="internal_token_unlock", shape="MetricFrame@1.0",
            fetcher="cyqnt_trd.data_cli.internal_frames.fetch_token_unlock",
            description="代幣解鎖:已解鎖量與下次解鎖",
            columns=("token", "price", "total_unlocked", "next_unlock_time",
                     "next_unlock_amount"),
            value_columns=("price", "total_unlocked", "next_unlock_amount"),
            time_column="next_unlock_time", instrument_column="token", pit_safe=True),
        InternalSlot(
            key="internal_ai_signal", shape="MetricFrame@1.0",
            fetcher="cyqnt_trd.data_cli.internal_frames.fetch_ai_signal",
            description="ai-skill 訊號槽",
            instrument_column="symbol", pit_safe=False),
        InternalSlot(
            key="internal_chip_distribution", shape="MetricFrame@1.0",
            fetcher="cyqnt_trd.data_cli.internal_frames.fetch_chip_distribution",
            description="籌碼分佈", instrument_column="token", pit_safe=False),
        InternalSlot(
            key="internal_bdp_screen", shape="RankFrame@1.0",
            fetcher="cyqnt_trd.data_cli.internal_frames.fetch_bdp_screen",
            description="bdp 選股 / 篩選結果", pit_safe=False),
    )
}


def slot_frame_shapes() -> Dict[str, str]:
    """``{slot key: canonical shape}`` — merged into ``FRAME_SHAPES``."""
    return {slot.key: slot.shape for slot in INTERNAL_SLOTS.values()}


def internal_client_available() -> bool:
    """True only where the private internal client is installed."""
    try:
        import importlib

        importlib.import_module("cyqnt_trd.data_cli.internal_frames")
        return True
    except Exception:
        return False


def normalize_internal_frame(key: str, frame, *, decision_time: int):
    """Raw internal DataFrame -> rows in the slot's canonical shape.

    Kept here rather than in the private package so a bundle built inside the
    network and one built outside are byte-comparable apart from the rows.
    """
    import pandas as pd

    slot = INTERNAL_SLOTS.get(key)
    if slot is None or frame is None or not len(frame):
        return []
    rows: List[Dict] = []
    time_col = slot.time_column if slot.time_column in frame.columns else None
    avail_col = slot.available_column or time_col

    for record in frame.where(pd.notna(frame), None).to_dict(orient="records"):
        ts = record.get(time_col) if time_col else None
        ts = int(ts) if ts is not None else decision_time
        avail = record.get(avail_col) if avail_col else None
        avail = int(avail) if avail is not None else ts
        instrument = (str(record.get(slot.instrument_column))
                      if slot.instrument_column and record.get(slot.instrument_column)
                      else None)
        if slot.shape == "MetricFrame@1.0":
            for metric in (slot.value_columns or ()):
                if record.get(metric) is None:
                    continue
                try:
                    value = float(record[metric])
                except (TypeError, ValueError):
                    continue
                rows.append({"event_time": ts, "available_time": avail,
                             "instrument_id": instrument, "metric": metric,
                             "value": value})
        else:                                   # EventFrame / RankFrame
            out = dict(record)
            out.setdefault("event_time", ts)
            out.setdefault("available_time", avail)
            if instrument:
                out.setdefault("instrument_id", instrument)
            if slot.shape == "EventFrame@1.0":
                # An EventFrame must be identifiable and attributable: event_id
                # is what de-duplicates a story across polls, and source_id /
                # topic are what let a bot refuse a directional read from an
                # unverified source. Passing the vendor record straight through
                # left all three missing, so the frame claimed a shape it did not
                # have — and nothing noticed until the schema was made to check.
                out.setdefault("source_id", key)
                out.setdefault("topic", str(record.get("event_type") or "unclassified"))
                if not out.get("event_id"):
                    import hashlib

                    seed = "|".join(str(out.get(column, "")) for column in (
                        "source_id", "topic", "event_time", "instrument_id", "title"))
                    out["event_id"] = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
            rows.append(out)
    return rows
