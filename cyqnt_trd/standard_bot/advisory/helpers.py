"""Small shared helpers for sample advisory bots."""

from __future__ import annotations

import re
from dataclasses import fields
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Type, TypeVar

import pandas as pd

from .contracts import Evidence, utc_datetime


T = TypeVar("T")


def config_from_dict(cls: Type[T], value: Any) -> T:
    if isinstance(value, cls):
        return value
    raw = dict(value or {})
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError("unknown config fields for %s: %s" % (cls.__name__, ", ".join(unknown)))
    return cls(**raw)


def instrument_keys(frame: pd.DataFrame):
    columns = ["venue", "product", "symbol"]
    values = frame[columns].dropna(subset=["symbol"]).drop_duplicates()
    return sorted(
        (str(row["venue"]), str(row["product"]), str(row["symbol"]).upper())
        for _, row in values.iterrows()
    )


def latest_metrics(
    frame: pd.DataFrame,
    symbol: str,
    venue: Optional[str] = None,
    product: Optional[str] = None,
) -> Dict[str, pd.Series]:
    mask = frame["symbol"].astype(str).str.upper() == symbol.upper()
    if venue is not None:
        mask &= frame["venue"].astype(str) == venue
    if product is not None:
        mask &= frame["product"].astype(str) == product
    selected = frame[mask].copy()
    if selected.empty:
        return {}
    selected["_available"] = pd.to_datetime(selected["available_time"], utc=True)
    selected = selected.sort_values("_available")
    return {str(row["metric"]): row for _, row in selected.groupby("metric").tail(1).iterrows()}


def metric_float(metrics: Mapping[str, pd.Series], name: str) -> Optional[float]:
    row = metrics.get(name)
    if row is None:
        return None
    try:
        value = float(row["value"])
    except (TypeError, ValueError):
        return None
    return value if pd.notna(value) else None


def metric_evidence(metrics: Mapping[str, pd.Series], names: Sequence[str]) -> List[Evidence]:
    result: List[Evidence] = []
    for name in names:
        row = metrics.get(name)
        if row is None:
            continue
        result.append(
            Evidence(
                source=str(row.get("source_id", "bn-data:%s" % name)),
                source_id=str(row.get("source_id", name)),
                available_time=utc_datetime(row["available_time"]),
                observed={
                    "metric": name,
                    "value": row["value"],
                    "unit": row.get("unit", ""),
                    "window": row.get("window", ""),
                },
                note=str(row.get("quality", "")),
            )
        )
    return result


def event_evidence(row: pd.Series) -> Evidence:
    return Evidence(
        source=str(row.get("source_id", "news")),
        source_id=str(row.get("event_id", "")),
        available_time=utc_datetime(row["available_time"]),
        observed={
            "title": str(row.get("title", "")),
            "event_type": str(row.get("event_type", "")),
            "topic": str(row.get("topic", "")),
            "published_at": utc_datetime(row.get("published_at")).isoformat(),
            "author_name": str(row.get("author_name", "")),
            "author_role": str(row.get("author_role", "")),
            "source_reliability": row_float(row, "source_reliability", 0.0),
            "corroboration_count": int(row_float(row, "corroboration_count", 0.0)),
            "body_hash": str(row.get("body_hash", "")),
            "quality_flags": row.get("quality_flags", []),
        },
        url=str(row.get("url", "")),
        note="Evidence is captured at availability time; headline bias is not an order.",
    )


def valid_until(decision_time: Any, seconds: int):
    return utc_datetime(decision_time) + timedelta(seconds=seconds)


def row_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return default if pd.isna(value) else value


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    lowered = str(text).lower()
    for keyword in keywords:
        value = str(keyword).lower()
        if value.isascii() and any(char.isalnum() for char in value):
            pattern = r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(value)
            if re.search(pattern, lowered):
                return True
        elif value in lowered:
            return True
    return False


def is_fresh(available_time: Any, decision_time: Any, max_age_seconds: int) -> bool:
    age = (utc_datetime(decision_time) - utc_datetime(available_time)).total_seconds()
    return 0 <= age <= max_age_seconds


def latest_available_time(metrics: Mapping[str, pd.Series]):
    return max(utc_datetime(row["available_time"]) for row in metrics.values())


def direction_from_bias(value: Any) -> str:
    lowered = str(value or "").lower()
    if lowered in ("positive", "bullish", "long", "buy", "up"):
        return "long"
    if lowered in ("negative", "bearish", "short", "sell", "down"):
        return "short"
    return "neutral"
