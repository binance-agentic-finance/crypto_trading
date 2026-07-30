"""Typed contracts for monitor/advisory bots.

Input is carried in ``DataSnapshot.frames`` as normalized long-form DataFrames.
Output remains the canonical ``SignalBatch`` with ``SignalKind.ALERT``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from ..core import SignalBatch


class FrameValidationError(ValueError):
    pass


def utc_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        number = float(value)
        if abs(number) > 10_000_000_000:
            number /= 1000.0
        dt = datetime.fromtimestamp(number, tz=timezone.utc)
    else:
        dt = pd.Timestamp(value).to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def epoch_ms(value: Any) -> int:
    return int(utc_datetime(value).timestamp() * 1000)


@dataclass(frozen=True)
class FrameContract:
    name: str
    schema: str
    required_columns: Tuple[str, ...]
    event_time_column: str
    available_time_column: str = "available_time"
    scope: str = "multi_asset"

    def validate(self, frame: pd.DataFrame, decision_time: Any) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise FrameValidationError("%s must be a pandas DataFrame" % self.name)
        missing = sorted(set(self.required_columns) - set(frame.columns))
        if missing:
            raise FrameValidationError("%s missing columns: %s" % (self.name, ", ".join(missing)))
        if frame.empty:
            return
        event_time = pd.to_datetime(frame[self.event_time_column], utc=True, errors="coerce")
        if event_time.isna().any():
            raise FrameValidationError(
                "%s has invalid %s" % (self.name, self.event_time_column)
            )
        available = pd.to_datetime(frame[self.available_time_column], utc=True, errors="coerce")
        if available.isna().any():
            raise FrameValidationError("%s has invalid available_time" % self.name)
        if (event_time > available).any():
            raise FrameValidationError(
                "%s contains event_time later than available_time" % self.name
            )
        cutoff = pd.Timestamp(utc_datetime(decision_time))
        if (available > cutoff).any():
            raise FrameValidationError(
                "%s contains future data: available_time > decision_time" % self.name
            )

    def port(self, *, required: bool = True) -> Dict[str, Any]:
        return {
            "kind": "dataframe",
            "schema": self.schema,
            "required": required,
            "scope": self.scope,
            "rowCardinality": "many",
            "eventTime": self.event_time_column,
            "availableTime": self.available_time_column,
            "alignment": "asof",
            "pitRequired": True,
        }


MARKET_METRIC_FRAME = FrameContract(
    name="market_metrics",
    schema="MarketMetricFrame@1.0",
    required_columns=(
        "event_time",
        "available_time",
        "venue",
        "product",
        "symbol",
        "metric",
        "value",
    ),
    event_time_column="event_time",
)

NEWS_EVENT_FRAME = FrameContract(
    name="news_events",
    schema="NewsEventFrame@1.0",
    required_columns=(
        "event_id",
        "published_at",
        "available_time",
        "source_id",
        "topic",
        "urgency",
        "symbol",
        "title",
    ),
    event_time_column="published_at",
)


@dataclass(frozen=True)
class DataRequirement:
    field_key: str
    source: str
    availability: str
    required: bool = True
    note: str = ""


@dataclass(frozen=True)
class BotMeta:
    bot_id: str
    display_name: str
    version: str
    description: str
    required_frames: Tuple[FrameContract, ...]
    optional_frames: Tuple[FrameContract, ...] = ()
    data_requirements: Tuple[DataRequirement, ...] = ()
    supported_products: Tuple[str, ...] = ("spot", "usd_m_perpetual", "mixed")
    bot_class: str = "monitor_or_advisory"
    advisory_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["required_frames"] = [
            {"name": item.name, **item.port(required=True)} for item in self.required_frames
        ]
        value["optional_frames"] = [
            {"name": item.name, **item.port(required=False)} for item in self.optional_frames
        ]
        return value


@dataclass(frozen=True)
class Evidence:
    source: str
    source_id: str
    available_time: datetime
    observed: Mapping[str, Any] = field(default_factory=dict)
    url: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "available_time": utc_datetime(self.available_time).isoformat(),
            "observed": dict(self.observed),
            "url": self.url,
            "note": self.note,
        }


@dataclass(frozen=True)
class AdvisoryDecision:
    symbol: Optional[str]
    venue: str
    product: str
    direction: str
    action: str
    score: float
    confidence: float
    horizon_seconds: int
    valid_until: datetime
    topic: str
    urgency: str
    reason_codes: Tuple[str, ...]
    summary: str
    recommended_behavior: str
    evidence: Tuple[Evidence, ...]
    data_quality: str = "good"
    signal_family: str = ""
    market_scope: str = "asset"
    auto_trade_eligible: bool = False
    dedup_key: str = ""

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "neutral"):
            raise ValueError("direction must be long, short, or neutral")
        if self.action not in ("alert", "watch", "avoid", "investigate"):
            raise ValueError("invalid advisory action %r" % self.action)
        if not 0.0 <= float(self.score) <= 100.0:
            raise ValueError("score must be in [0, 100]")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be positive")
        if self.auto_trade_eligible:
            raise ValueError("advisory decisions cannot be auto-trade eligible")


OUTPUT_COLUMNS = [
    "bot_id",
    "bot_version",
    "signal_id",
    "decision_time",
    "symbol",
    "venue",
    "product",
    "direction",
    "action",
    "score",
    "confidence",
    "horizon_seconds",
    "valid_until",
    "topic",
    "signal_family",
    "urgency",
    "reason_codes",
    "summary",
    "recommended_behavior",
    "evidence",
    "data_quality",
    "market_scope",
    "auto_trade_eligible",
    "dedup_key",
]


def signals_to_frame(batch: SignalBatch) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for signal in batch.signals:
        payload = dict(signal.payload)
        rows.append(
            {
                "bot_id": signal.provenance.plugin_id,
                "bot_version": signal.provenance.plugin_version,
                "signal_id": signal.signal_id,
                "decision_time": pd.to_datetime(
                    payload.get("decision_time", batch.created_at),
                    unit=None if payload.get("decision_time") else "ms",
                    utc=True,
                ),
                "symbol": payload.get("symbol"),
                "venue": payload.get("venue"),
                "product": payload.get("product"),
                "direction": payload.get("direction"),
                "action": payload.get("action"),
                "score": payload.get("score"),
                "confidence": payload.get("confidence"),
                "horizon_seconds": payload.get("horizon_seconds"),
                "valid_until": pd.to_datetime(payload.get("valid_until"), utc=True),
                "topic": payload.get("topic"),
                "signal_family": payload.get("signal_family"),
                "urgency": payload.get("urgency"),
                "reason_codes": payload.get("reason_codes", []),
                "summary": payload.get("summary"),
                "recommended_behavior": payload.get("recommended_behavior"),
                "evidence": payload.get("evidence", []),
                "data_quality": payload.get("data_quality"),
                "market_scope": payload.get("market_scope"),
                "auto_trade_eligible": payload.get("auto_trade_eligible", False),
                "dedup_key": payload.get("dedup_key", ""),
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
