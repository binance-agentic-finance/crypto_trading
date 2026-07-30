"""Project catalyst, tokenomics and security risk monitor."""

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


SECURITY = ("hack", "exploit", "breach", "attack", "漏洞", "攻击", "被盗", "安全事件")
UNLOCK = ("token unlock", "unlock", "解锁", "释放")
UPGRADE = ("mainnet", "upgrade", "hard fork", "主网", "升级", "硬分叉")
GOVERNANCE = ("governance", "proposal", "vote", "治理", "提案", "投票")


@dataclass(frozen=True)
class CatalystRiskConfig:
    min_source_reliability: float = 0.7
    horizon_seconds: int = 86400
    max_event_age_seconds: int = 604800

    def __post_init__(self) -> None:
        if not 0 <= self.min_source_reliability <= 1:
            raise ValueError("min_source_reliability must be in [0, 1]")
        if self.horizon_seconds <= 0 or self.max_event_age_seconds <= 0:
            raise ValueError("time windows must be positive")

    @classmethod
    def from_dict(cls, value: Any) -> "CatalystRiskConfig":
        return config_from_dict(cls, value)


class CatalystRiskMonitor(AdvisoryBot):
    meta = BotMeta(
        bot_id="catalyst_risk_monitor",
        display_name="项目 Catalyst 与安全风险监控",
        version="1.0.0",
        description="识别升级、治理、解锁和安全事故。",
        required_frames=(NEWS_EVENT_FRAME,),
        data_requirements=(
            DataRequirement("news_feed", "Square getNews/getSearch", "LIVE_CALLABLE"),
            DataRequirement("hot_event", "bn-data", "REGISTERED_NOT_WIRED", required=False),
            DataRequirement("token_unlock", "bn-data", "NOT_REGISTERED", required=False),
            DataRequirement("onchain_signal", "bn-data", "FORWARD_ONLY", required=False),
        ),
    )

    def normalize_config(self, config: Any) -> CatalystRiskConfig:
        return CatalystRiskConfig.from_dict(config)

    def evaluate(
        self,
        snapshot: DataSnapshot,
        config: CatalystRiskConfig,
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
            direction, action, base_score, reason = {
                "security": ("short", "avoid", 95.0, "SECURITY_INCIDENT"),
                "unlock": ("short", "watch", 70.0, "TOKEN_UNLOCK"),
                "upgrade": ("long", "watch", 65.0, "PROJECT_UPGRADE"),
                "governance": ("neutral", "watch", 50.0, "GOVERNANCE_EVENT"),
            }[event_type]
            if not verified:
                direction = "neutral"
                action = "investigate"
                base_score = min(base_score, 55.0)
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
                    score=base_score,
                    confidence=min(0.95, 0.3 + 0.5 * reliability + 0.08 * corroboration),
                    horizon_seconds=config.horizon_seconds,
                    valid_until=expires_at,
                    topic="security_incident" if event_type == "security" else "project_catalyst",
                    urgency="breaking" if event_type == "security" else str(row.get("urgency", "scheduled")),
                    reason_codes=(reason, "SOURCE_VERIFIED" if verified else "UNVERIFIED"),
                    summary=str(row.get("title", "项目事件")),
                    recommended_behavior=(
                        "停止新增风险并核对官方处置进展。" if event_type == "security"
                        else "核对官方来源、生效时间和代币供给影响；不自动下单。"
                    ),
                    evidence=(event_evidence(row),),
                    data_quality="good" if verified else "degraded",
                    signal_family="project_catalyst_risk",
                    market_scope="asset" if symbol else "global",
                    dedup_key="%s:%s:%s:%s"
                    % (row.get("event_id", ""), venue, product, symbol or "GLOBAL"),
                )
            )
        return output


def _classify(row) -> Optional[str]:
    explicit = str(row.get("event_type", "")).lower()
    if explicit in ("security", "unlock", "upgrade", "governance"):
        return explicit
    text = "%s %s" % (row.get("title", ""), row.get("summary", ""))
    if contains_any(text, SECURITY):
        return "security"
    if contains_any(text, UNLOCK):
        return "unlock"
    if contains_any(text, UPGRADE):
        return "upgrade"
    if contains_any(text, GOVERNANCE):
        return "governance"
    return None


def _optional_symbol(value: Any) -> Optional[str]:
    if value is None or str(value).strip().lower() in ("", "none", "nan"):
        return None
    return str(value).upper()
