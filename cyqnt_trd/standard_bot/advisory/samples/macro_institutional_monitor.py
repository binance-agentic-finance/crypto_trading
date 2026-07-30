"""Macro calendar and ETF/institutional flow monitor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from ...core import DataSnapshot
from ..base import AdvisoryBot
from ..contracts import AdvisoryDecision, BotMeta, DataRequirement, NEWS_EVENT_FRAME
from ..helpers import (
    config_from_dict,
    contains_any,
    direction_from_bias,
    event_evidence,
    is_fresh,
    row_float,
    valid_until,
)


MACRO_WORDS = ("fomc", "cpi", "nfp", "interest rate", "fed", "央行", "利率", "非农", "通胀")
INSTITUTION_WORDS = ("etf", "institutional", "fund flow", "机构", "资金流", "净流入", "净流出")


@dataclass(frozen=True)
class MacroInstitutionalConfig:
    min_importance: float = 0.6
    horizon_seconds: int = 86400
    max_event_age_seconds: int = 604800

    def __post_init__(self) -> None:
        if not 0 <= self.min_importance <= 1:
            raise ValueError("min_importance must be in [0, 1]")
        if self.horizon_seconds <= 0 or self.max_event_age_seconds <= 0:
            raise ValueError("time windows must be positive")

    @classmethod
    def from_dict(cls, value: Any) -> "MacroInstitutionalConfig":
        return config_from_dict(cls, value)


class MacroInstitutionalMonitor(AdvisoryBot):
    meta = BotMeta(
        bot_id="macro_institutional_monitor",
        display_name="宏观与机构资金监控",
        version="1.0.0",
        description="监控宏观日历、政策和 ETF/机构资金事件。",
        required_frames=(NEWS_EVENT_FRAME,),
        data_requirements=(
            DataRequirement("news_feed", "Square getNews", "LIVE_CALLABLE"),
            DataRequirement("macro_calendar", "bn-data", "REGISTERED_NOT_WIRED", required=False),
            DataRequirement("etf_flow", "bn-data", "REGISTERED_NOT_WIRED", required=False),
        ),
    )

    def normalize_config(self, config: Any) -> MacroInstitutionalConfig:
        return MacroInstitutionalConfig.from_dict(config)

    def evaluate(
        self,
        snapshot: DataSnapshot,
        config: MacroInstitutionalConfig,
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
            family = _family(row)
            if family is None:
                continue
            reliability = row_float(row, "source_reliability", 0.5)
            corroboration = int(row_float(row, "corroboration_count", 0))
            verified = (reliability >= 0.7 and bool(row.get("url"))) or corroboration >= 2
            default_importance = 0.7 if verified else 0.5
            importance = row_float(
                row,
                "importance",
                row_float(row, "impact_score", default_importance),
            )
            if importance < config.min_importance:
                continue
            flow = row_float(row, "flow_usd", 0.0)
            direction = direction_from_bias(row.get("bias"))
            # A validated numeric flow is stronger evidence than headline bias.
            if family == "institutional_flow" and flow != 0:
                direction = "long" if flow > 0 else "short"
            if not verified:
                direction = "neutral"
            quality = "good" if verified else "degraded"
            symbol = _optional_symbol(row.get("symbol"))
            venue = str(row.get("venue", "binance"))
            product = str(row.get("product", "mixed"))
            output.append(
                AdvisoryDecision(
                    symbol=symbol,
                    venue=venue,
                    product=product,
                    direction=direction,
                    action="watch" if quality == "good" else "investigate",
                    score=min(100.0, 45.0 + 50.0 * importance),
                    confidence=min(0.9, 0.35 + 0.5 * reliability),
                    horizon_seconds=config.horizon_seconds,
                    valid_until=expires_at,
                    topic=family,
                    urgency=str(row.get("urgency", "scheduled")),
                    reason_codes=(
                        "MACRO_EVENT" if family == "macro_policy" else "INSTITUTIONAL_FLOW",
                        "NUMERIC_FLOW_AVAILABLE" if flow != 0 else "NEWS_ONLY",
                    ),
                    summary=str(row.get("title", "宏观/机构事件")),
                    recommended_behavior="按公布时间和市场范围调整观察级别；新闻金额不可替代正式 Flow 数据。",
                    evidence=(event_evidence(row),),
                    data_quality=quality,
                    signal_family="macro_institutional",
                    market_scope="asset" if symbol else "global",
                    dedup_key="%s:%s:%s:%s"
                    % (row.get("event_id", ""), venue, product, symbol or "GLOBAL"),
                )
            )
        return output


def _family(row) -> Optional[str]:
    topic = str(row.get("topic", "")).lower()
    if topic in ("macro_policy", "institutional_flow"):
        return topic
    text = "%s %s" % (row.get("title", ""), row.get("summary", ""))
    if contains_any(text, MACRO_WORDS):
        return "macro_policy"
    if contains_any(text, INSTITUTION_WORDS):
        return "institutional_flow"
    return None


def _optional_symbol(value: Any) -> Optional[str]:
    if value is None or str(value).strip().lower() in ("", "none", "nan"):
        return None
    return str(value).upper()
