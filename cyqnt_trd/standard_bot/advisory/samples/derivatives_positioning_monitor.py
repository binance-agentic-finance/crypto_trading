"""Perpetual positioning / crowding advisory bot.

Covers the derivatives block of the July user-need survey — funding (195
chats), OI (120), long-short ratio (59), basis (41) — which the other five
sample bots do not touch. It reads only ``MarketMetricFrame@1.0`` rows and
emits alerts; it never fetches and never trades.

Reading convention (positioning, not prediction):

* funding is the **cost of the crowded side**. Deeply positive annualized
  funding means longs are paying — the crowd is long, so the *risk* is a long
  squeeze; the informational direction is therefore ``short``, and vice versa.
* OI + price together label *who* is being added:
  OI up + price up = new longs; OI up + price down = new shorts;
  OI down + price up = short covering; OI down + price down = long unwind.
  Only the two "new money" quadrants get a direction; the two unwind quadrants
  are explicitly ``neutral`` because they are exits, not fresh conviction.
* long/short account ratio is retail account skew, used only as a
  corroborating crowding vote, never on its own as a direction.
* basis is a dislocation/carry note: ``neutral`` by construction.

When funding, OI growth and account skew all point at the same crowded side,
the action escalates to ``avoid`` — that is the liquidation-cascade setup, and
the right advice is to stand aside, not to take the other side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

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

_METRICS = (
    "funding_rate_8h",
    "funding_rate_annualized_pct",
    "open_interest_change_24h_pct",
    "long_short_account_ratio",
    "basis_rate_pct",
    "price_change_24h_pct",
)


@dataclass(frozen=True)
class DerivativesPositioningConfig:
    funding_annualized_high_pct: float = 30.0
    funding_annualized_extreme_pct: float = 75.0
    oi_growth_24h_pct: float = 20.0
    oi_drop_24h_pct: float = 15.0
    price_move_24h_pct: float = 2.0
    long_short_ratio_high: float = 3.0
    long_short_ratio_low: float = 0.5
    basis_dislocation_pct: float = 0.5
    horizon_seconds: int = 14400
    max_age_seconds: int = 3600

    def __post_init__(self) -> None:
        positive = (
            self.funding_annualized_high_pct,
            self.funding_annualized_extreme_pct,
            self.oi_growth_24h_pct,
            self.oi_drop_24h_pct,
            self.price_move_24h_pct,
            self.basis_dislocation_pct,
            self.horizon_seconds,
            self.max_age_seconds,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("all derivatives thresholds and time windows must be positive")
        if self.funding_annualized_extreme_pct <= self.funding_annualized_high_pct:
            raise ValueError(
                "funding_annualized_extreme_pct must exceed funding_annualized_high_pct"
            )
        if not 0 < self.long_short_ratio_low < 1 < self.long_short_ratio_high:
            raise ValueError(
                "long_short_ratio thresholds must satisfy 0 < low < 1 < high"
            )

    @classmethod
    def from_dict(cls, value: Any) -> "DerivativesPositioningConfig":
        return config_from_dict(cls, value)


class DerivativesPositioningMonitor(AdvisoryBot):
    meta = BotMeta(
        bot_id="derivatives_positioning_monitor",
        display_name="永续持仓拥挤度监控",
        version="1.0.0",
        description="按资金费率、OI 变化、多空账户比与基差判断拥挤方向和挤压风险，仅输出提醒。",
        required_frames=(MARKET_METRIC_FRAME,),
        data_requirements=(
            DataRequirement("funding_rate", "bn-data/binance-cli futures-usds", "BACKTESTABLE"),
            DataRequirement("open_interest", "bn-data/binance-cli futures-usds", "SEMI"),
            DataRequirement(
                "long_short_ratio", "binance-cli futures-usds", "SEMI", required=False,
                note="未接入 enrich_market_frame_with_derivatives；advisory 侧由适配器直接读取",
            ),
            DataRequirement("basis", "binance-cli futures-usds", "SEMI", required=False),
            DataRequirement("kline", "bn-data", "BACKTESTABLE", required=False),
        ),
        supported_products=("usd_m_perpetual", "mixed"),
    )

    def normalize_config(self, config: Any) -> DerivativesPositioningConfig:
        return DerivativesPositioningConfig.from_dict(config)

    def evaluate(
        self,
        snapshot: DataSnapshot,
        config: DerivativesPositioningConfig,
    ) -> Iterable[AdvisoryDecision]:
        frame = snapshot.frames[MARKET_METRIC_FRAME.name]
        decision_time = snapshot.meta.decision_as_of or snapshot.meta.assembled_at
        output: List[AdvisoryDecision] = []
        for venue, product, symbol in instrument_keys(frame):
            if product == "spot":
                # Perp positioning metrics must not be attributed to a spot leg.
                continue
            metrics = latest_metrics(frame, symbol, venue, product)
            if not metrics:
                continue
            source_time = latest_available_time(metrics)
            expires_at = valid_until(source_time, config.horizon_seconds)
            if not is_fresh(source_time, decision_time, config.max_age_seconds):
                continue
            if expires_at <= valid_until(decision_time, 0):
                continue

            funding_apr = metric_float(metrics, "funding_rate_annualized_pct")
            if funding_apr is None:
                rate = metric_float(metrics, "funding_rate_8h")
                funding_apr = None if rate is None else rate * 3 * 365 * 100.0
            oi_change = metric_float(metrics, "open_interest_change_24h_pct")
            price_change = metric_float(metrics, "price_change_24h_pct")
            ls_ratio = metric_float(metrics, "long_short_account_ratio")
            basis = metric_float(metrics, "basis_rate_pct")

            votes: List[Tuple[str, str, float]] = []  # (direction, reason_code, severity)

            funding_vote = self._funding_vote(funding_apr, config)
            if funding_vote:
                votes.append(funding_vote)
            flow_vote = self._oi_flow_vote(oi_change, price_change, config)
            if flow_vote:
                votes.append(flow_vote)
            skew_vote = self._account_skew_vote(ls_ratio, config)
            if skew_vote:
                votes.append(skew_vote)
            basis_vote = self._basis_vote(basis, config)
            if basis_vote:
                votes.append(basis_vote)
            if not votes:
                continue

            direction, conflicted = self._resolve_direction(votes)
            reason_codes = tuple(reason for _, reason, _ in votes)
            if conflicted:
                reason_codes += ("CONFLICTING_POSITIONING",)
            severity = max(sev for _, _, sev in votes)
            directional = [vote for vote in votes if vote[0] != "neutral"]
            agreeing = [vote for vote in directional if vote[0] == direction]

            squeeze = self._is_squeeze_setup(
                direction, funding_apr, oi_change, ls_ratio, config
            )
            if squeeze:
                action = "avoid"
            elif len(agreeing) >= 2:
                action = "alert"
            else:
                action = "watch"

            score = min(100.0, 40.0 + 20.0 * (severity - 1.0) + 12.0 * (len(agreeing)))
            confidence = min(0.9, 0.45 + 0.13 * len(agreeing) + (0.08 if squeeze else 0.0))
            present = [name for name in ("funding_rate_annualized_pct",
                                         "open_interest_change_24h_pct",
                                         "long_short_account_ratio") if name in metrics]
            quality = "good" if len(present) >= 2 else "degraded"

            output.append(
                AdvisoryDecision(
                    symbol=symbol,
                    venue=venue,
                    product=product,
                    direction=direction,
                    action=action,
                    score=round(score, 2),
                    confidence=round(confidence, 2),
                    horizon_seconds=config.horizon_seconds,
                    valid_until=expires_at,
                    topic="derivatives_positioning",
                    urgency="breaking" if squeeze else "background",
                    reason_codes=reason_codes + (("SQUEEZE_RISK",) if squeeze else ()),
                    summary=(
                        "%s 持仓结构：资金费率年化=%s%%，OI变化=%s%%，多空账户比=%s，基差=%s%%，24h涨跌=%s%%"
                        % (
                            symbol,
                            _show(funding_apr),
                            _show(oi_change),
                            _show(ls_ratio),
                            _show(basis),
                            _show(price_change),
                        )
                    ),
                    recommended_behavior=(
                        "拥挤+杠杆堆积，优先规避而不是反向追单；确认强平密集区与资金费率结算时间。"
                        if squeeze
                        else "把拥挤方向当作风险提示而非入场信号；与量价、事件面交叉确认后再决定。"
                    ),
                    evidence=tuple(metric_evidence(metrics, _METRICS)),
                    data_quality=quality,
                    signal_family="derivatives_positioning",
                    dedup_key="%s:%s:%s:derivatives_positioning:%s"
                    % (venue, product, symbol, int(source_time.timestamp())),
                )
            )
        return output

    # ---- individual votes; each returns (direction, reason_code, severity) ----

    @staticmethod
    def _funding_vote(funding_apr, config) -> Optional[Tuple[str, str, float]]:
        if funding_apr is None:
            return None
        magnitude = abs(funding_apr)
        if magnitude < config.funding_annualized_high_pct:
            return None
        severity = magnitude / config.funding_annualized_high_pct
        if funding_apr > 0:
            return ("short", "FUNDING_CROWDED_LONG", severity)
        return ("long", "FUNDING_CROWDED_SHORT", severity)

    @staticmethod
    def _oi_flow_vote(oi_change, price_change, config) -> Optional[Tuple[str, str, float]]:
        if oi_change is None or price_change is None:
            return None
        if abs(price_change) < config.price_move_24h_pct:
            return None
        if oi_change >= config.oi_growth_24h_pct:
            severity = oi_change / config.oi_growth_24h_pct
            if price_change > 0:
                return ("long", "OI_UP_PRICE_UP_NEW_LONGS", severity)
            return ("short", "OI_UP_PRICE_DOWN_NEW_SHORTS", severity)
        if oi_change <= -config.oi_drop_24h_pct:
            severity = abs(oi_change) / config.oi_drop_24h_pct
            reason = (
                "OI_DOWN_PRICE_UP_SHORT_COVERING"
                if price_change > 0
                else "OI_DOWN_PRICE_DOWN_LONG_UNWIND"
            )
            # Unwind is an exit, not fresh conviction — no direction is claimed.
            return ("neutral", reason, severity)
        return None

    @staticmethod
    def _account_skew_vote(ls_ratio, config) -> Optional[Tuple[str, str, float]]:
        if ls_ratio is None:
            return None
        if ls_ratio >= config.long_short_ratio_high:
            return ("short", "ACCOUNT_SKEW_CROWDED_LONG", ls_ratio / config.long_short_ratio_high)
        if ls_ratio <= config.long_short_ratio_low:
            return ("long", "ACCOUNT_SKEW_CROWDED_SHORT", config.long_short_ratio_low / max(ls_ratio, 1e-9))
        return None

    @staticmethod
    def _basis_vote(basis, config) -> Optional[Tuple[str, str, float]]:
        if basis is None or abs(basis) < config.basis_dislocation_pct:
            return None
        return ("neutral", "BASIS_DISLOCATION", abs(basis) / config.basis_dislocation_pct)

    #: a side must win by this share of the dominant weight to be claimed;
    #: below it the two readings are treated as genuinely conflicting.
    DIRECTION_MARGIN = 0.2

    @classmethod
    def _resolve_direction(cls, votes: List[Tuple[str, str, float]]) -> Tuple[str, bool]:
        """Severity-weighted side, or ``neutral`` when the sides roughly cancel.

        Returns ``(direction, conflicted)``. Funding saying "crowded long" while
        OI+price says "new longs are being added" is a real disagreement about
        positioning; asserting either side off a near-tie would be false
        precision, so it is reported as neutral and flagged.
        """
        weights = {"long": 0.0, "short": 0.0}
        for direction, _, severity in votes:
            if direction in weights:
                weights[direction] += severity
        dominant = max(weights.values())
        if dominant <= 0:
            return "neutral", False
        gap = abs(weights["long"] - weights["short"])
        if gap < cls.DIRECTION_MARGIN * dominant:
            # both sides voted and neither wins clearly
            return "neutral", min(weights.values()) > 0
        return ("long" if weights["long"] > weights["short"] else "short"), False

    @staticmethod
    def _is_squeeze_setup(direction, funding_apr, oi_change, ls_ratio, config) -> bool:
        """Crowded side + leverage still being added + retail on the same side."""
        if direction == "neutral" or funding_apr is None or oi_change is None:
            return False
        if abs(funding_apr) < config.funding_annualized_extreme_pct:
            return False
        if oi_change < config.oi_growth_24h_pct:
            return False
        if ls_ratio is None:
            return False
        if direction == "short":
            return funding_apr > 0 and ls_ratio >= config.long_short_ratio_high
        return funding_apr < 0 and ls_ratio <= config.long_short_ratio_low


def _show(value):
    return "NA" if value is None else round(value, 4)
