"""社交热度 × 交易所资金流 背离监控 —— 完全不看 K 线。

一个只用**非量价**数据的读法：把"谁在说"和"钱在往哪走"放在一起看。

    热度在涨 + 资金在净流入交易所  →  有人在借热度出货（派发）
    热度在涨 + 资金在净流出交易所  →  买了往外搬，是持有不是套现（吸筹）
    热度在跌 + 资金在净流出         →  安静地被吸走，最容易被忽略的一种

单看任何一边都没有信息量：热度高可能只是行情大，净流入可能只是做市商调仓。
**背离**才是信号 —— 说的和做的不一致。

激活的数据
----------
``ticker_rank``   RankFrame   Square 提及量 / 独立作者数 / 多空计数（社交热度）
``large_flow``    MetricFrame 大额充提窗口（钱进交易所 = 准备卖）
``funding``       MetricFrame 只作佐证，判断这波是不是已经拥挤

为什么是 ADVISORY 而不是 TRADE
------------------------------
两个主输入都是 ``FORWARD_ONLY``：Square 的热度是 60 秒缓存快照，充提是 1 小时窗口
且内部源。**没有任何 PIT 历史，这个逻辑今天证伪不了**。有真实 edge 也可能没有，
现在无从判断，所以只出提醒不下单——`BotKind.ADVISORY` 让这件事是结构性的，
而不是一条可以被忽略的注释。
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

BOT_ID = "social_flow_divergence"

DEFAULTS: Dict[str, Any] = {
    "top_n": 10,                  # 只看社交榜前 N
    "window": "24h",
    "min_mentions": 100,
    "hot_rank": 5,                # 排进前 5 算「热」
    "net_inflow_usd": 1_000_000,  # 净流入交易所超过这个数算「在准备卖」
    "net_outflow_usd": 1_000_000,
    "funding_crowded_bps": 10.0,
    "horizon_seconds": 14_400,
}


class SocialFlowDivergence(StandardBot):
    spec = BotSpec(
        bot_id=BOT_ID,
        kind=BotKind.ADVISORY,
        display_name="社交热度 × 资金流背离",
        description="热度和交易所净流向不一致时提醒：说的和做的对不上。",
        products=("usd_m_perpetual",),
        default_horizon_seconds=DEFAULTS["horizon_seconds"],
    )

    def __init__(self, **config: Any) -> None:
        self.config = {**DEFAULTS, **config}

    def required_data(self) -> Sequence[DataRequest]:
        return [
            DataRequest("ticker_rank",
                        {"window": self.config["window"], "limit": self.config["top_n"]}),
            DataRequest("large_flow", {"token": "BTC", "interval": "1h"},
                        alias="flow_BTC", required=False),
            DataRequest("large_flow", {"token": "ETH", "interval": "1h"},
                        alias="flow_ETH", required=False),
            DataRequest("large_flow", {"token": "SOL", "interval": "1h"},
                        alias="flow_SOL", required=False),
            DataRequest("funding", {"symbol": "BTCUSDT", "limit": 3},
                        alias="funding_BTCUSDT", required=False),
            DataRequest("funding", {"symbol": "ETHUSDT", "limit": 3},
                        alias="funding_ETHUSDT", required=False),
            DataRequest("funding", {"symbol": "SOLUSDT", "limit": 3},
                        alias="funding_SOLUSDT", required=False),
        ]

    # ---- 决策 ----

    def decide(self, ctx: BotContext) -> Iterable[StandardSignal]:
        ranks = ctx.view("ticker_rank")
        if ranks is None or ranks.empty:
            return

        for _, row in ranks.frame.iterrows():
            instrument = str(row.get("instrument_id") or "").upper()
            if not instrument:
                continue
            token = _base_token(instrument)
            symbol = instrument if instrument.endswith("USDT") else token + "USDT"

            heat = self._heat(row)
            if heat is None:
                continue
            flow = self._net_flow(ctx, token)
            if flow is None:
                # 只有热度没有资金流 —— 说了但不知道做了什么，这不是背离
                continue

            signal = self._assess(ctx, symbol, heat, flow)
            if signal is not None:
                yield signal

    def _assess(self, ctx: BotContext, symbol: str, heat: Dict[str, Any],
                flow: float) -> Optional[StandardSignal]:
        hot = heat["rank"] is not None and heat["rank"] <= self.config["hot_rank"]
        loud = heat["mentions"] is not None and heat["mentions"] >= self.config["min_mentions"]
        if not (hot or loud):
            return None

        inflow = flow >= self.config["net_inflow_usd"]
        outflow = flow <= -self.config["net_outflow_usd"]
        if not (inflow or outflow):
            return None                     # 资金没动，热度自己说了不算

        if inflow:
            # 有人一边喊单一边往交易所搬币 —— 派发的形状
            direction, action = Direction.SHORT, AdvisoryAction.ALERT
            reasons = ["SOCIAL_HOT", "EXCHANGE_NET_INFLOW", "DISTRIBUTION_SHAPE"]
            summary_tail = "净流入交易所 %.2fM，热度与资金流背离（派发）" % (flow / 1e6)
            behavior = "把热度当出货窗口而不是入场理由；若持有考虑减仓。"
        else:
            direction, action = Direction.LONG, AdvisoryAction.WATCH
            reasons = ["SOCIAL_HOT", "EXCHANGE_NET_OUTFLOW", "ACCUMULATION_SHAPE"]
            summary_tail = "净流出交易所 %.2fM，买了往外搬（吸筹）" % (abs(flow) / 1e6)
            behavior = "吸筹形状但非入场信号；等量价确认。"

        funding_bps = ctx.metric("funding_%s" % symbol, "rate")
        funding_bps = None if funding_bps is None else funding_bps * 10_000.0
        if funding_bps is not None and abs(funding_bps) >= self.config["funding_crowded_bps"]:
            reasons.append("FUNDING_ALREADY_CROWDED")
            # 已经拥挤 → 背离的解释力下降，不升级
            action = AdvisoryAction.WATCH

        quality = ctx.data_quality(required=["ticker_rank"])
        if funding_bps is None:
            quality = DataQuality.DEGRADED

        score = 55.0
        if heat["rank"] is not None:
            score += max(0.0, 25.0 - 4.0 * (heat["rank"] - 1))
        score = float(np.clip(score + min(15.0, abs(flow) / 1e6), 0.0, 100.0))

        return StandardSignal(
            bot_id=BOT_ID,
            decision_time=ctx.decision_time,
            provenance=Provenance(strategy_id=BOT_ID),
            symbol=symbol,
            intent=PositionIntent.HOLD,     # advisory：永远不是指令
            direction=direction,
            advisory_action=action,
            score=round(score, 2),
            confidence=0.55 if funding_bps is None else 0.65,
            time_horizon=TimeHorizon.INTRADAY,
            horizon_seconds=self.config["horizon_seconds"],
            topic="social_flow_divergence",
            reason_codes=tuple(reasons),
            summary="%s 社交排名 %s / 提及 %s；%s" % (
                symbol,
                "NA" if heat["rank"] is None else int(heat["rank"]),
                "NA" if heat["mentions"] is None else int(heat["mentions"]),
                summary_tail,
            ),
            recommended_behavior=behavior,
            evidence=(
                Evidence(source="data.ticker_rank",
                         observed={"rank": heat["rank"], "mentions": heat["mentions"],
                                   "bull_ratio": heat["bull_ratio"]},
                         available_time=ctx.decision_time,
                         note="Square 快照，60 秒缓存，无历史"),
                Evidence(source="data.large_flow",
                         observed={"net_exchange_flow_usd": flow},
                         available_time=ctx.decision_time,
                         note="正数 = 净流入交易所 = 准备卖"),
            ),
            data_quality=quality,
        )

    # ---- 读数 ----

    @staticmethod
    def _heat(row: pd.Series) -> Optional[Dict[str, Any]]:
        def _num(key):
            value = pd.to_numeric(row.get(key), errors="coerce")
            return None if pd.isna(value) else float(value)

        rank = _num("rank")
        mentions = _num("mention_count")
        if rank is None and mentions is None:
            return None
        bullish, bearish = _num("bullish_count"), _num("bearish_count")
        total = (bullish or 0) + (bearish or 0)
        return {
            "rank": rank, "mentions": mentions,
            "bull_ratio": None if total <= 0 else round((bullish or 0) / total, 3),
        }

    def _net_flow(self, ctx: BotContext, token: str) -> Optional[float]:
        """净流入交易所（USD）。正数 = 币在往交易所搬 = 准备卖。"""
        view = ctx.view("flow_%s" % token)
        if view is None or view.empty:
            return None
        deposit = view.metric("large_deposit_amt")
        withdraw = view.metric("large_withdraw_amt")
        if deposit is None and withdraw is None:
            return None
        return float((deposit or 0.0) - (withdraw or 0.0))


def _base_token(symbol: str) -> str:
    value = str(symbol).upper()
    for quote in ("USDT", "USDC", "FDUSD", "TUSD", "USD"):
        if value.endswith(quote) and len(value) > len(quote):
            return value[: -len(quote)]
    return value


# 导入即注册，和 blocks.strategy.register() 的习惯一致
from cyqnt_trd.standard_bot.registry import register_bot  # noqa: E402

register_bot(SocialFlowDivergence)
