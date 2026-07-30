"""Square attention and sentiment advisory bot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List

from ...core import DataSnapshot
from ..base import AdvisoryBot
from ..contracts import AdvisoryDecision, BotMeta, DataRequirement, MARKET_METRIC_FRAME
from ..helpers import (
    config_from_dict,
    instrument_keys,
    is_fresh,
    latest_available_time,
    latest_metrics,
    metric_evidence,
    metric_float,
    valid_until,
)


@dataclass(frozen=True)
class SocialSentimentConfig:
    rank_max: int = 10
    mention_count_min: int = 100
    bull_ratio_high: float = 0.65
    bull_ratio_low: float = 0.35
    horizon_seconds: int = 7200
    max_age_seconds: int = 3600

    def __post_init__(self) -> None:
        if self.rank_max <= 0 or self.mention_count_min <= 0:
            raise ValueError("rank_max and mention_count_min must be positive")
        if not 0 <= self.bull_ratio_low < self.bull_ratio_high <= 1:
            raise ValueError("bull ratio thresholds must satisfy 0 <= low < high <= 1")
        if self.horizon_seconds <= 0 or self.max_age_seconds <= 0:
            raise ValueError("time windows must be positive")

    @classmethod
    def from_dict(cls, value: Any) -> "SocialSentimentConfig":
        return config_from_dict(cls, value)


class SocialSentimentMonitor(AdvisoryBot):
    meta = BotMeta(
        bot_id="social_sentiment_monitor",
        display_name="Square 社交热度与情绪监控",
        version="1.0.0",
        description="按排行、提及量和多空情绪提示热点币。",
        required_frames=(MARKET_METRIC_FRAME,),
        data_requirements=(
            DataRequirement("social_rank", "Square getTickerRank", "FORWARD_ONLY"),
            DataRequirement("social_sentiment", "Square getSentiment", "FORWARD_ONLY", required=False),
            DataRequirement("social_topic_trending", "Square getTopicTrending", "FORWARD_ONLY", required=False),
            DataRequirement("fear_greed", "bn-data/public REST", "BACKTESTABLE", required=False),
        ),
    )

    def normalize_config(self, config: Any) -> SocialSentimentConfig:
        return SocialSentimentConfig.from_dict(config)

    def evaluate(
        self,
        snapshot: DataSnapshot,
        config: SocialSentimentConfig,
    ) -> Iterable[AdvisoryDecision]:
        frame = snapshot.frames[MARKET_METRIC_FRAME.name]
        decision_time = snapshot.meta.decision_as_of or snapshot.meta.assembled_at
        output: List[AdvisoryDecision] = []
        for venue, product, symbol in instrument_keys(frame):
            metrics = latest_metrics(frame, symbol, venue, product)
            source_time = latest_available_time(metrics)
            expires_at = valid_until(source_time, config.horizon_seconds)
            if not is_fresh(source_time, decision_time, config.max_age_seconds):
                continue
            if expires_at <= valid_until(decision_time, 0):
                continue
            rank = metric_float(metrics, "rank")
            mentions = metric_float(metrics, "mention_count")
            bull = metric_float(metrics, "social_bull_ratio")
            is_hot = (rank is not None and rank <= config.rank_max) or (
                mentions is not None and mentions >= config.mention_count_min
            )
            if not is_hot:
                continue
            if bull is None:
                direction = "neutral"
                reason = "SOCIAL_ATTENTION"
                quality = "degraded"
            elif bull >= config.bull_ratio_high:
                direction = "long"
                reason = "BULLISH_SOCIAL_SKEW"
                quality = "good"
            elif bull <= config.bull_ratio_low:
                direction = "short"
                reason = "BEARISH_SOCIAL_SKEW"
                quality = "good"
            else:
                direction = "neutral"
                reason = "MIXED_SOCIAL_SENTIMENT"
                quality = "good"
            rank_score = 0.0 if rank is None else max(0.0, 45.0 - 3.0 * (rank - 1))
            mention_score = 0.0 if mentions is None else min(30.0, 30.0 * mentions / config.mention_count_min)
            sentiment_score = 0.0 if bull is None else min(25.0, abs(bull - 0.5) * 100.0)
            score = min(100.0, rank_score + mention_score + sentiment_score)
            output.append(
                AdvisoryDecision(
                    symbol=symbol,
                    venue=venue,
                    product=product,
                    direction=direction,
                    action="watch" if bull is None else "alert",
                    score=round(score, 2),
                    confidence=0.55 if bull is None else 0.72,
                    horizon_seconds=config.horizon_seconds,
                    valid_until=expires_at,
                    topic="social_attention",
                    urgency="breaking" if rank is not None and rank <= 3 else "background",
                    reason_codes=("TOP_SOCIAL_RANK", reason),
                    summary="%s 社交排名=%s，提及=%s，多头占比=%s" % (
                        symbol,
                        _show(rank),
                        _show(mentions),
                        _show(bull),
                    ),
                    recommended_behavior="核对消息来源与量价确认；情绪只作提醒，不直接交易。",
                    evidence=tuple(
                        metric_evidence(metrics, ("rank", "mention_count", "social_bull_ratio"))
                    ),
                    data_quality=quality,
                    signal_family="social_attention_sentiment",
                    dedup_key="%s:%s:%s:social:%s"
                    % (venue, product, symbol, int(source_time.timestamp())),
                )
            )
        return output


def _show(value):
    return "NA" if value is None else round(value, 3)
