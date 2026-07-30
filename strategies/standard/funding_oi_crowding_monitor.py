"""D001 — funding + OI crowding, as an advisory monitor.

Ported from ``策略开发/70_D_derivatives/strategies/D001_funding_oi_crowding.py``.

Why advisory and not a trade bot
--------------------------------
D001 measured a **real** neutral edge in the local work — Sharpe 0.70, beta ≈ 0
— but that is below the 1.0 deploy gate that N003 and X001 cleared. Shipping it
as a trade bot would quietly put size behind a book that did not pass its own
screen. Shipping it as a monitor keeps the information and drops the claim: the
capability declaration (``BotKind.ADVISORY``) makes that structural rather than
a note someone can ignore.

Thesis
------
Funding tells you the *price* of the crowd's positioning; OI change tells you
the *size* behind it. The fragile setup is positive funding **and** rising OI:
longs are both paying up and piling in. Symmetrically for negative funding.

Coverage honesty (carried over verbatim from the original)
----------------------------------------------------------
Public OI history is ~30 days and the local deep panels only exist for a few
majors, so the OI overlay is coverage-limited. Rather than let that silently
degrade into "OI = 0", a symbol with no readable OI is reported with
``OI_UNAVAILABLE`` in its reason codes and the alert is downgraded — and the
bot accepts an optional **user-registered** OI panel so a team with its own
deeper history can plug it in without touching this file::

    from cyqnt_trd.standard_bot.data import custom_sources as cs
    cs.register_file_node(name="oi_panel", path="~/research/oi_daily.parquet",
                          availability="SEMI", time_column="ts",
                          pit_hazard="research capture, starts 2024-01")

    FundingOiCrowdingMonitor(oi_panel_node="custom_oi_panel")
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from cyqnt_trd.standard_bot.bot import BotContext, BotKind, BotSpec, DataRequest, StandardBot
from cyqnt_trd.standard_bot.core import (
    AdvisoryAction,
    DataQuality,
    Direction,
    Evidence,
    PositionIntent,
    Provenance,
    StandardSignal,
    TimeHorizon,
)

BOT_ID = "funding_oi_crowding_monitor"

DEFAULTS: Dict[str, Any] = {
    "universe": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "funding_prints": 200,
    "oi_period": "1h",
    "oi_bars": 48,
    "funding_crowded_bps": 5.0,      # per-8h settlement
    "funding_extreme_bps": 15.0,
    "oi_confirm_pct": 10.0,          # OI growth over the lookback that confirms build-up
    "horizon_seconds": 14_400,
    #: optional user-registered node supplying a deeper OI panel
    "oi_panel_node": "",
}


class FundingOiCrowdingMonitor(StandardBot):
    spec = BotSpec(
        bot_id=BOT_ID,
        kind=BotKind.ADVISORY,
        display_name="资金费率 + OI 拥挤度监控",
        description="资金费率给拥挤的价格，OI 变化给拥挤的规模；两者同向即为挤压风险。",
        products=("usd_m_perpetual",),
        default_horizon_seconds=DEFAULTS["horizon_seconds"],
    )

    def __init__(self, **config: Any) -> None:
        self.config = {**DEFAULTS, **config}

    def required_data(self) -> Sequence[DataRequest]:
        requests: List[DataRequest] = []
        for symbol in self.config["universe"]:
            requests.append(
                DataRequest("funding",
                            {"symbol": symbol, "limit": self.config["funding_prints"]},
                            alias="funding_%s" % symbol)
            )
            requests.append(
                DataRequest("open_interest",
                            {"symbol": symbol, "period": self.config["oi_period"],
                             "limit": self.config["oi_bars"]},
                            alias="oi_%s" % symbol, required=False)
            )
        if self.config["oi_panel_node"]:
            requests.append(
                DataRequest(self.config["oi_panel_node"], {}, alias="oi_panel",
                            required=False)
            )
        return requests

    # ---- decision ----

    def decide(self, ctx: BotContext) -> Iterable[StandardSignal]:
        for symbol in self.config["universe"]:
            reading = self._read(ctx, symbol)
            if reading is None:
                continue
            signal = self._assess(ctx, symbol, reading)
            if signal is not None:
                yield signal

    def _read(self, ctx: BotContext, symbol: str) -> Optional[Dict[str, Any]]:
        funding = ctx.frame("funding_%s" % symbol)
        if funding is None or getattr(funding, "empty", True) or "rate" not in funding:
            return None
        rates = pd.to_numeric(funding["rate"], errors="coerce").dropna()
        if rates.empty:
            return None

        oi_change_pct: Optional[float] = self._oi_change(ctx, symbol)
        return {
            "funding_bps": float(rates.iloc[-1]) * 10_000.0,
            "funding_mean_bps": float(rates.tail(9).mean()) * 10_000.0,
            "oi_change_pct": oi_change_pct,
        }

    def _oi_change(self, ctx: BotContext, symbol: str) -> Optional[float]:
        """Prefer the user's deeper panel; fall back to the ~30d public series."""
        panel = ctx.frame("oi_panel")
        if panel is not None and not getattr(panel, "empty", True) and symbol in panel:
            series = pd.to_numeric(panel[symbol], errors="coerce").dropna()
            if len(series) >= 2 and series.iloc[0] > 0:
                return float(series.iloc[-1] / series.iloc[0] - 1.0) * 100.0

        frame = ctx.frame("oi_%s" % symbol)
        if frame is None or getattr(frame, "empty", True) or "oi_value" not in frame:
            return None
        values = pd.to_numeric(frame["oi_value"], errors="coerce").dropna()
        if len(values) < 2 or float(values.iloc[0]) <= 0:
            return None
        return float(values.iloc[-1] / values.iloc[0] - 1.0) * 100.0

    def _assess(
        self, ctx: BotContext, symbol: str, reading: Dict[str, Any]
    ) -> Optional[StandardSignal]:
        funding_bps = reading["funding_bps"]
        oi_change = reading["oi_change_pct"]
        crowded = abs(funding_bps) >= self.config["funding_crowded_bps"]
        if not crowded:
            return None

        crowded_side = "long" if funding_bps > 0 else "short"
        # informational direction points at the side under threat, i.e. the
        # opposite of the crowd
        direction = Direction.SHORT if crowded_side == "long" else Direction.LONG
        reasons: List[str] = [
            "FUNDING_CROWDED_LONG" if crowded_side == "long" else "FUNDING_CROWDED_SHORT"
        ]

        confirmed = False
        if oi_change is None:
            reasons.append("OI_UNAVAILABLE")
            quality = DataQuality.DEGRADED
        else:
            quality = ctx.data_quality()
            if oi_change >= self.config["oi_confirm_pct"]:
                reasons.append("OI_BUILDUP_CONFIRMS")
                confirmed = True
            elif oi_change <= -self.config["oi_confirm_pct"]:
                reasons.append("OI_UNWIND_DECROWDING")

        extreme = abs(funding_bps) >= self.config["funding_extreme_bps"]
        if extreme and confirmed:
            action, score, confidence = AdvisoryAction.AVOID, 90.0, 0.8
        elif confirmed or extreme:
            action, score, confidence = AdvisoryAction.ALERT, 70.0, 0.7
        else:
            action, score, confidence = AdvisoryAction.WATCH, 55.0, 0.55
        if oi_change is None:
            # a single unconfirmed reading is a watch, never an avoid
            action = AdvisoryAction.WATCH
            score = min(score, 55.0)
            confidence = min(confidence, 0.5)

        return StandardSignal(
            bot_id=BOT_ID,
            decision_time=ctx.decision_time,
            provenance=Provenance(strategy_id=BOT_ID),
            symbol=symbol,
            intent=PositionIntent.HOLD,          # advisory: never an instruction
            direction=direction,
            advisory_action=action,
            score=score,
            confidence=confidence,
            time_horizon=TimeHorizon.INTRADAY,
            horizon_seconds=self.config["horizon_seconds"],
            topic="derivatives_positioning",
            reason_codes=tuple(reasons) + (("SQUEEZE_RISK",) if extreme and confirmed else ()),
            summary="%s 拥挤%s：资金费率 %.2f bps（9期均值 %.2f），OI 变化 %s" % (
                symbol, "多" if crowded_side == "long" else "空",
                funding_bps, reading["funding_mean_bps"],
                "NA" if oi_change is None else "%.1f%%" % oi_change,
            ),
            recommended_behavior=(
                "拥挤 + 杠杆仍在堆积，优先规避而非反向追单；确认强平密集区与结算时间。"
                if extreme and confirmed
                else "把拥挤方向当风险提示，与量价、事件面交叉确认后再决定。"
            ),
            evidence=(
                Evidence(
                    source="data.funding",
                    observed={"funding_bps": round(funding_bps, 4),
                              "mean_9_bps": round(reading["funding_mean_bps"], 4)},
                    available_time=ctx.decision_time,
                ),
                Evidence(
                    source="data.open_interest",
                    observed={"change_pct": oi_change},
                    available_time=ctx.decision_time,
                    note="public OI history is ~30d; a deeper user panel can be "
                         "registered via custom_sources and passed as oi_panel_node",
                ),
            ),
            data_quality=quality,
        )


# 导入即注册，和 blocks.strategy.register() 的习惯一致
from cyqnt_trd.standard_bot.registry import register_bot  # noqa: E402

register_bot(FundingOiCrowdingMonitor)
