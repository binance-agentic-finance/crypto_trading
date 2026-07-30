"""Exchange listing/delisting/maintenance monitor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from ...core import DataSnapshot
from ..base import AdvisoryBot
from ..contracts import AdvisoryDecision, BotMeta, DataRequirement, NEWS_EVENT_FRAME
from ..helpers import (
    config_from_dict,
    contains_any,
    event_evidence,
    is_fresh,
    row_float,
    valid_until,
)


LISTING = ("listing", "list new", "上币", "上线", "launchpool")
DELISTING = ("delisting", "delist", "下币", "下架", "停止交易")
MAINTENANCE = ("maintenance", "suspend", "resume", "维护", "暂停充提", "恢复充提")


@dataclass(frozen=True)
class ExchangeEventConfig:
    min_source_reliability: float = 0.7
    horizon_seconds: int = 21600
    max_event_age_seconds: int = 86400

    def __post_init__(self) -> None:
        if not 0 <= self.min_source_reliability <= 1:
            raise ValueError("min_source_reliability must be in [0, 1]")
        if self.horizon_seconds <= 0 or self.max_event_age_seconds <= 0:
            raise ValueError("time windows must be positive")

    @classmethod
    def from_dict(cls, value: Any) -> "ExchangeEventConfig":
        return config_from_dict(cls, value)


class ExchangeEventMonitor(AdvisoryBot):
    meta = BotMeta(
        bot_id="exchange_event_monitor",
        display_name="交易所公告监控",
        version="1.0.0",
        description="识别上币、下币和维护公告，输出证据化提醒。",
        required_frames=(NEWS_EVENT_FRAME,),
        data_requirements=(
            DataRequirement("news_feed", "Square getNews", "LIVE_CALLABLE"),
            DataRequirement("hot_event", "bn-data", "REGISTERED_NOT_WIRED", required=False),
            DataRequirement("crypto_event_calendar", "bn-data", "NOT_REGISTERED", required=False),
        ),
    )

    def normalize_config(self, config: Any) -> ExchangeEventConfig:
        return ExchangeEventConfig.from_dict(config)

    def evaluate(
        self,
        snapshot: DataSnapshot,
        config: ExchangeEventConfig,
    ) -> Iterable[AdvisoryDecision]:
        frame = snapshot.frames[NEWS_EVENT_FRAME.name]
        decision_time = snapshot.meta.decision_as_of or snapshot.meta.assembled_at
        output = []
        for _, row in frame.iterrows():
            expires_at = valid_until(row["available_time"], config.horizon_seconds)
            if not is_fresh(row["available_time"], decision_time, config.max_event_age_seconds):
                continue
            if expires_at <= valid_until(decision_time, 0):
                continue
            event_type = _classify(row)
            if event_type is None:
                continue
            reliability = row_float(row, "source_reliability", 0.5)
            corroboration = int(row_float(row, "corroboration_count", 0))
            verified = (
                reliability >= config.min_source_reliability and bool(row.get("url"))
            ) or corroboration >= 2
            direction, action, score, reason = {
                "listing": ("long", "alert", 75.0, "EXCHANGE_LISTING"),
                "delisting": ("short", "avoid", 90.0, "EXCHANGE_DELISTING"),
                "maintenance": ("neutral", "watch", 55.0, "EXCHANGE_MAINTENANCE"),
            }[event_type]
            if not verified:
                direction = "neutral"
                action = "investigate"
                score = min(score, 50.0)
            symbol = _optional_symbol(row.get("symbol"))
            venue = str(row.get("venue", "binance"))
            product = str(row.get("product", "mixed"))
            output.append(
                AdvisoryDecision(
                    symbol=symbol,
                    venue=venue,
                    product=product,
                    direction=direction,
                    action=action,
                    score=score,
                    confidence=min(0.95, 0.35 + reliability * 0.45 + 0.08 * corroboration),
                    horizon_seconds=config.horizon_seconds,
                    valid_until=expires_at,
                    topic="exchange_announcement",
                    urgency="breaking" if event_type != "maintenance" else "scheduled",
                    reason_codes=(reason, "SOURCE_VERIFIED" if verified else "SOURCE_UNVERIFIED"),
                    summary=str(row.get("title", "交易所公告")),
                    recommended_behavior=(
                        "暂停新增风险并核对官方公告。" if event_type == "delisting"
                        else "核对官方公告、流动性与生效时间；不自动下单。"
                    ),
                    evidence=(event_evidence(row),),
                    data_quality="good" if verified else "degraded",
                    signal_family="exchange_event",
                    market_scope="asset" if symbol else "global",
                    dedup_key="%s:%s:%s:%s"
                    % (row.get("event_id", ""), venue, product, symbol or "GLOBAL"),
                )
            )
        return output


def _classify(row) -> Optional[str]:
    explicit = str(row.get("event_type", "")).lower()
    if explicit in ("listing", "delisting", "maintenance"):
        return explicit
    text = "%s %s" % (row.get("title", ""), row.get("summary", ""))
    if contains_any(text, DELISTING):
        return "delisting"
    if contains_any(text, LISTING):
        return "listing"
    if contains_any(text, MAINTENANCE):
        return "maintenance"
    return None


def _optional_symbol(value: Any) -> Optional[str]:
    if value is None or str(value).strip().lower() in ("", "none", "nan"):
        return None
    return str(value).upper()
