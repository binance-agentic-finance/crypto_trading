"""Price/volume/OI anomaly monitor. Threshold orchestration only."""

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
class PriceVolumeConfig:
    price_move_24h_pct: float = 5.0
    volume_ratio_24h: float = 2.0
    oi_growth_24h_pct: float = 20.0
    horizon_seconds: int = 3600
    max_age_seconds: int = 3600

    def __post_init__(self) -> None:
        values = (
            self.price_move_24h_pct,
            self.volume_ratio_24h,
            self.oi_growth_24h_pct,
            self.horizon_seconds,
            self.max_age_seconds,
        )
        if any(value <= 0 for value in values):
            raise ValueError("all price-volume thresholds and time windows must be positive")

    @classmethod
    def from_dict(cls, value: Any) -> "PriceVolumeConfig":
        return config_from_dict(cls, value)


class PriceVolumeMonitor(AdvisoryBot):
    meta = BotMeta(
        bot_id="price_volume_monitor",
        display_name="量价与 OI 异常监控",
        version="1.0.0",
        description="监控 24h 涨跌、量比和 OI 增长，仅输出提醒。",
        required_frames=(MARKET_METRIC_FRAME,),
        data_requirements=(
            DataRequirement("kline", "bn-data", "BACKTESTABLE"),
            DataRequirement("open_interest", "bn-data/public REST", "SEMI", required=False),
        ),
    )

    def normalize_config(self, config: Any) -> PriceVolumeConfig:
        return PriceVolumeConfig.from_dict(config)

    def evaluate(
        self,
        snapshot: DataSnapshot,
        config: PriceVolumeConfig,
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
            price = metric_float(metrics, "price_change_24h_pct")
            volume = metric_float(metrics, "volume_ratio_24h")
            oi = metric_float(metrics, "open_interest_change_24h_pct")
            reasons = []
            severity = []
            if price is not None and abs(price) >= config.price_move_24h_pct:
                reasons.append("PRICE_MOVE_24H")
                severity.append(abs(price) / config.price_move_24h_pct)
            if volume is not None and volume >= config.volume_ratio_24h:
                reasons.append("VOLUME_SURGE_24H")
                severity.append(volume / config.volume_ratio_24h)
            if oi is not None and oi >= config.oi_growth_24h_pct:
                reasons.append("OI_GROWTH_24H")
                severity.append(oi / config.oi_growth_24h_pct)
            if not reasons:
                continue

            direction = "neutral" if price is None or price == 0 else ("long" if price > 0 else "short")
            score = min(100.0, 50.0 + 25.0 * (max(severity) - 1.0) + 8.0 * (len(reasons) - 1))
            confidence = min(0.92, 0.55 + 0.12 * len(reasons))
            output.append(
                AdvisoryDecision(
                    symbol=symbol,
                    venue=venue,
                    product=product,
                    direction=direction,
                    action="alert",
                    score=round(score, 2),
                    confidence=round(confidence, 2),
                    horizon_seconds=config.horizon_seconds,
                    valid_until=expires_at,
                    topic="market_anomaly",
                    urgency="breaking" if score >= 80 else "background",
                    reason_codes=tuple(reasons),
                    summary=(
                        "%s 异常：24h涨跌=%s%%，量比=%s，OI变化=%s%%"
                        % (symbol, _show(price), _show(volume), _show(oi))
                    ),
                    recommended_behavior="检查盘口与事件来源，等待二次确认；不自动下单。",
                    evidence=tuple(
                        metric_evidence(
                            metrics,
                            (
                                "price_change_24h_pct",
                                "volume_ratio_24h",
                                "open_interest_change_24h_pct",
                            ),
                        )
                    ),
                    data_quality="good" if len(reasons) >= 2 else "degraded",
                    signal_family="price_volume_oi",
                    dedup_key="%s:%s:%s:market_anomaly:%s"
                    % (venue, product, symbol, int(source_time.timestamp())),
                )
            )
        return output


def _show(value):
    return "NA" if value is None else round(value, 3)
