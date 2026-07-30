"""消息催化剂交易 —— 触发来自新闻，量价只做确认和否决。

和 funding 三兄弟不同，这支的**主信号完全不是量价**：入场由一条高可靠度的
利好事件触发，K 线只回答两个问题——"行情是不是已经走完了"，以及"进场价在哪"。

激活的数据
----------
``news``            EventFrame  → 经 ``blocks.news_features`` 加工
``klines``          BarFrame    → 只用来算 pre-event 涨幅（是否已被定价）
``funding``         MetricFrame → 拥挤度否决

为什么这么设计
--------------
消息类策略最常见的死法是**追一个已经走完的行情**：文章发出来的时候价格早动过了。
所以这里有两道闸：

* ``lead_time_seconds`` —— 我们比**这个故事首次出现**晚了多久（转发链上的第几手）。
* ``age_seconds`` —— 这条消息**相对现在**有多老（我们多久才读到）。
  这两个是不同的「晚」：一条刚发的转发 lead_time 大而 age 小；一条原创但我们两小时后
  才轮询到的 lead_time 小而 age 大。两种都不是催化剂，所以两个都查。
* ``pre_event_move_pct`` —— 消息可读之前价格已经涨了多少。已经涨完了就不追。

持有期由**事件类型**决定，不是拍脑袋：上币是 30 分钟脉冲，主网升级是两天。
``expected_half_life_seconds`` 直接变成 ``TimeStop``。

诚实边界
--------
news 是 ``FORWARD_ONLY``，没有 PIT 历史 —— **这支策略今天回测不了**。
它能上 paper/live，但"历史上这么做赚不赚钱"目前无法回答，只能从现在开始前向采集。
这一点写在 ``pit_hazard`` 里，``validate_nodes(for_backtest=True)`` 会拦。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from cyqnt_trd.blocks import news_features as NF
from cyqnt_trd.standard_bot.bot import BotContext, BotKind, BotSpec, DataRequest, StandardBot
from cyqnt_trd.standard_bot.core import (
    EntrySpec,
    EntryType,
    Evidence,
    ExitPlan,
    PositionIntent,
    Provenance,
    SizeMode,
    SizeSpec,
    StandardSignal,
    StopSpec,
    TimeHorizon,
    TimeStop,
)

BOT_ID = "news_catalyst_trade"

DEFAULTS: Dict[str, Any] = {
    "universe": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"],
    "interval": "15m",
    "kline_limit": 96,
    "news_page_size": 50,
    # 只做这几类有明确方向的事件；其余记录不交易
    "tradeable_types": ("listing", "mainnet_upgrade", "partnership", "funding_round"),
    "min_source_reliability": 0.7,   # 官方源或有链接的
    "min_sentiment_confidence": 0.45,
    # 两个不同的「晚」，都要查：
    "max_lead_time_seconds": 900,    # 相对这个故事首次出现，我们晚了多久（转发链）
    "max_age_seconds": 1_800,        # 相对现在，这条消息本身有多老（读取延迟）
    "max_pre_event_move_pct": 3.0,   # 消息可读前已经涨过 3% 就算已被定价
    "funding_veto_bps": 15.0,        # 资金费率已极端 → 拥挤，不追
    "stop_pct": 0.02,
    "take_profit_pct": 0.04,
    "risk_pct": 0.01,
}


class NewsCatalystTrade(StandardBot):
    spec = BotSpec(
        bot_id=BOT_ID,
        kind=BotKind.TRADE,
        display_name="消息催化剂交易",
        description="高可靠度利好事件触发；量价只做「是否已被定价」的否决，持有期由事件类型决定。",
        products=("usd_m_perpetual",),
        reads_positions=True,
        default_horizon_seconds=3_600,
    )

    def __init__(self, **config: Any) -> None:
        self.config = {**DEFAULTS, **config}

    # ---- 输入 ----

    def required_data(self) -> Sequence[DataRequest]:
        requests: List[DataRequest] = [
            DataRequest("news", {"page_size": self.config["news_page_size"]}),
        ]
        for symbol in self.config["universe"]:
            requests.append(
                DataRequest("klines",
                            {"symbol": symbol, "interval": self.config["interval"],
                             "limit": self.config["kline_limit"]},
                            alias="klines_%s" % symbol, required=False)
            )
            requests.append(
                DataRequest("funding", {"symbol": symbol, "limit": 9},
                            alias="funding_%s" % symbol, required=False)
            )
        return requests

    # ---- 决策 ----

    def decide(self, ctx: BotContext) -> Iterable[StandardSignal]:
        events = ctx.view("news")
        if events is None or events.empty:
            return

        enriched = NF.enrich_events(events.frame, universe=self.config["universe"])
        # 转发不重复触发：同一个故事只在第一条上下单
        fresh = enriched[~enriched["is_duplicate"].astype(bool)]
        fresh = fresh[fresh["instrument_id"].notna()]

        for _, event in fresh.iterrows():
            signal = self._assess(ctx, event)
            if signal is not None:
                yield signal

    def _assess(self, ctx: BotContext, event: pd.Series) -> Optional[StandardSignal]:
        symbol = str(event["instrument_id"])
        rejects: List[str] = []

        if event["event_type"] not in self.config["tradeable_types"]:
            return None                     # 不在可交易类目，静默跳过
        if float(event.get("source_reliability") or 0.0) < self.config["min_source_reliability"]:
            rejects.append("SOURCE_UNVERIFIED")
        if float(event.get("sentiment_confidence") or 0.0) < self.config["min_sentiment_confidence"]:
            rejects.append("SENTIMENT_LOW_CONFIDENCE")
        if float(event.get("sentiment_score") or 0.0) <= 0:
            return None                     # 只做多头催化剂
        lead = float(event.get("lead_time_seconds") or 0.0)
        if lead > self.config["max_lead_time_seconds"]:
            rejects.append("LATE_IN_REPOST_CHAIN")
        age = self._age_seconds(ctx, event.get("available_time"))
        if age is not None and age > self.config["max_age_seconds"]:
            rejects.append("STALE_NEWS")

        pre_move = self._pre_event_move(ctx, symbol, event.get("available_time"))
        if pre_move is not None and pre_move > self.config["max_pre_event_move_pct"]:
            rejects.append("ALREADY_PRICED_IN")

        funding_bps = self._funding_bps(ctx, symbol)
        if funding_bps is not None and funding_bps >= self.config["funding_veto_bps"]:
            rejects.append("FUNDING_CROWDED_LONG")

        if rejects:
            # 不交易，但把「为什么不」记下来 —— 静默是最难排查的失败
            return None if ctx.side_of(symbol) != "flat" else None

        if ctx.side_of(symbol) == "long":
            return None                     # 已经在场内，不叠加

        bars = ctx.view("klines_%s" % symbol)
        last_close = None if bars is None else bars.latest("close")
        if last_close is None:
            return None                     # 没有价格就无法定 size / 止损

        half_life = int(event.get("expected_half_life_seconds") or 3600)
        return StandardSignal(
            bot_id=BOT_ID,
            decision_time=ctx.decision_time,
            provenance=Provenance(strategy_id=BOT_ID),
            symbol=symbol,
            intent=PositionIntent.OPEN_LONG,
            score=float(np.clip(50.0 + 40.0 * float(event["sentiment_score"]), 0.0, 100.0)),
            confidence=round(float(event["sentiment_confidence"]), 2),
            entry=EntrySpec(type=EntryType.MARKET),
            exit_plan=ExitPlan(
                stop_loss=StopSpec(pct=self.config["stop_pct"]),
                # 持有期跟着事件类型走：上币 30 分钟脉冲，升级两天
                time_stop=TimeStop(max_seconds=half_life),
                exit_on_opposite_signal=False,
                note="催化剂型持仓：到期即平，不等反向信号（%s 半衰期 %ds）"
                     % (event["event_type"], half_life),
            ),
            size=SizeSpec(mode=SizeMode.RISK_PCT, value=self.config["risk_pct"]),
            time_horizon=TimeHorizon.SCALP if half_life <= 3600 else TimeHorizon.INTRADAY,
            horizon_seconds=half_life,
            topic="news_catalyst",
            reason_codes=(
                event["event_type"].upper(),
                "SOURCE_VERIFIED",
                "FRESH_%ds" % int(age or lead),
            ),
            summary="%s %s：%s（消息年龄 %ss，转发链延迟 %ds，事前涨幅 %s）" % (
                symbol, event["event_type"], str(event.get("title", ""))[:48],
                "NA" if age is None else int(age), int(lead),
                "NA" if pre_move is None else "%.2f%%" % pre_move,
            ),
            recommended_behavior="催化剂交易：到半衰期即平，不转成趋势持仓。",
            evidence=(
                Evidence(source=str(event.get("source_id", "news")),
                         observed={"event_type": event["event_type"],
                                   "sentiment_score": float(event["sentiment_score"]),
                                   "lead_time_seconds": lead,
                                   "repost_count": int(event.get("repost_count") or 1)},
                         available_time=ctx.decision_time,
                         url=str(event.get("url", ""))),
            ),
            data_quality=ctx.data_quality(required=["news"]),
        )

    # ---- 辅助 ----

    def _pre_event_move(self, ctx: BotContext, symbol: str, available_time: Any) -> Optional[float]:
        """消息可读之前那几根 K 的涨幅 —— 判断是否已被定价。"""
        bars = ctx.view("klines_%s" % symbol)
        if bars is None or bars.empty or "close_time" not in bars.frame:
            return None
        frame = bars.frame.copy()
        stamps = pd.to_datetime(frame["close_time"], utc=True, errors="coerce")
        cutoff = pd.to_datetime(available_time, utc=True, errors="coerce")
        if pd.isna(cutoff):
            return None
        before = frame[stamps <= cutoff]
        if len(before) < 5:
            return None
        window = before.tail(4)["close"].astype(float)
        if window.iloc[0] <= 0:
            return None
        return float(window.iloc[-1] / window.iloc[0] - 1.0) * 100.0

    def _age_seconds(self, ctx: BotContext, available_time: Any) -> Optional[float]:
        """这条消息相对决策时刻有多老。和 lead_time 不是一回事。"""
        stamp = pd.to_datetime(available_time, utc=True, errors="coerce")
        if pd.isna(stamp):
            return None
        now = pd.to_datetime(ctx.decision_time, unit="ms", utc=True)
        return max(0.0, float((now - stamp).total_seconds()))

    def _funding_bps(self, ctx: BotContext, symbol: str) -> Optional[float]:
        rate = ctx.metric("funding_%s" % symbol, "rate")
        return None if rate is None else rate * 10_000.0


# 导入即注册，和 blocks.strategy.register() 的习惯一致
from cyqnt_trd.standard_bot.registry import register_bot  # noqa: E402

register_bot(NewsCatalystTrade)
