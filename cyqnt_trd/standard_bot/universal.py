"""``UniversalBot`` —— 那个「什么数据都能处理」的标准 bot 本体。

模型
----
不是「一个抽象基类，每支策略去继承」，而是**一个具体的 bot，什么都会**；
一支实际的交易 bot = 这个 bot **激活其中一部分能力**。

    UniversalBot                       全部 66 个数据节点 + 全部加工
        │
        ├── activate(...)              打开需要的那几个
        └── rule(ctx, f) -> signals    这支策略自己的判断
                    ▲
                    └── f 是 Features：不管激活了哪些，读法都一样

策略作者要做的只有两件事：**开哪些数据**、**怎么判断**。
取数、PIT 闸门、规范化、状态记录、溯源盖章、能力校验，全在这一层做完了。

写一支 bot
----------
::

    from cyqnt_trd.standard_bot import UniversalBot, BotKind

    def rule(ctx, f):
        funding = f.funding("BTCUSDT")          # bps；没开这个数据源就是 None
        if funding is not None and funding > 20:
            yield f.signal("BTCUSDT", "open_short",
                           reason=["FUNDING_CROWDED_LONG"], stop_pct=0.02)

    bot = (UniversalBot("my_bot", kind=BotKind.TRADE)
           .activate_funding(["BTCUSDT"])
           .activate_klines(["BTCUSDT"], interval="1h")
           .with_rule(rule))

    result = bot.run_once()                      # 取数 → 判断 → StandardSignal

没激活的数据源，``Features`` 上对应的读法返回 ``None``——**不是 0，也不报错**。
策略据此弃权，和"读到了但值是 0"分得开。

为什么读法要统一
----------------
funding 给 ``rate``/``timestamp``，OI 给 ``oi_value``/``timestamp``，ticker_rank 给
``ticker``/``mention_count``。一支读三个源的策略本来要记三套列名。规范化成 7 种形状
之后再包一层 ``Features``，策略看到的就只有 ``f.funding(sym)`` / ``f.oi_change(sym)``
/ ``f.social(sym)``——**加数据源不改策略代码的读法**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .bot import BotContext, BotKind, BotSpec, DataRequest, StandardBot
from .core import (
    AdvisoryAction,
    Direction,
    EntrySpec,
    EntryType,
    Evidence,
    ExitPlan,
    FrameKind,
    MarketScope,
    PositionIntent,
    Provenance,
    SelectionCandidate,
    SizeMode,
    SizeSpec,
    StandardSignal,
    StopSpec,
    TimeHorizon,
    TimeStop,
)

__all__ = ["UniversalBot", "Features", "Activation", "CAPABILITIES"]


# ---------------------------------------------------------------------------
# 能力清单 —— 「全面」具体指什么
# ---------------------------------------------------------------------------

#: 能力名 -> (数据节点, 说明)。激活一个能力就是打开对应的数据节点，
#: 并让 ``Features`` 上的相应读法开始返回值。
CAPABILITIES: Dict[str, Tuple[str, str]] = {
    # 量价
    "klines":            ("klines", "K 线 OHLCV"),
    "indicators":        ("indicator_charts", "K 线 + 14 项 ta4j 指标"),
    "ticker":            ("ticker_24h", "24h 涨跌 / 量"),
    # 衍生品
    "funding":           ("funding", "资金费率历史"),
    "open_interest":     ("open_interest", "持仓量历史"),
    "long_short":        ("long_short_ratio", "多空账户比"),
    "top_trader":        ("top_trader_ratio", "顶级交易者持仓比"),
    "taker":             ("taker_volume", "主动买卖量比"),
    "basis":             ("basis", "期现基差"),
    "liquidations":      ("liquidations", "强平流"),
    "radar":             ("futures_radar", "futuresRadar 全市场衍生品快照"),
    # 消息 / 事件
    "news":              ("news", "Square 新闻"),
    "news_search":       ("news_search", "Square 关键词检索"),
    "announcements":     ("announcements", "官方公告"),
    "hot_event":         ("hot_event", "热点事件"),
    "calendar":          ("calendar", "上币 / 活动日历"),
    "token_unlock":      ("token_unlock", "代币解锁"),
    "macro_calendar":    ("macro_calendar", "宏观日历（actual vs forecast）"),
    # 社交
    "social_rank":       ("ticker_rank", "Square 币种热度榜"),
    "sentiment":         ("sentiment", "Square 多空投票"),
    "topic_trending":    ("topic_trending", "话题热度"),
    # 截面
    "universe":          ("universe", "24h 全市场 ticker"),
    "bdp_screen":        ("bdp_screen", "bdp 截面筛选字段"),
    "coin_metrics":      ("coin_metrics", "市值 / 流动性 / 流通量"),
    "sector":            ("sector_flow", "板块热度与净流入"),
    # 资金流
    "large_flow":        ("large_flow", "大额充提（交易所进出）"),
    "whale":             ("whale_trades", "鲸鱼成交"),
    "chip":              ("chip_distribution", "筹码分布"),
    "top_player":        ("top_player_movement", "顶级交易者动向"),
    # 宏观 / 情绪
    "fear_greed":        ("fear_greed", "恐贪指数"),
    "ahr999":            ("ahr999", "AHR999 估值"),
    "etf_flow":          ("etf_flow", "现货 ETF 净流"),
    # 账户
    "positions":         ("contract_positions", "当前持仓"),
    "balance":           ("account_balance", "账户余额"),
    # 微观结构
    "orderbook":         ("orderbook_depth", "盘口深度"),
    # 平台既有信号
    "ai_signal":         ("ai_signal", "平台既有多因子信号"),
}


@dataclass(frozen=True)
class Activation:
    """一次激活：哪个能力、按什么参数、针对哪些标的。"""

    capability: str
    params: Dict[str, Any] = field(default_factory=dict)
    symbols: Tuple[str, ...] = ()
    required: bool = False
    #: 自定义节点直接给 node 名（``custom_*``），不走 CAPABILITIES 表
    node: str = ""

    def resolve_node(self) -> str:
        if self.node:
            return self.node
        if self.capability not in CAPABILITIES:
            raise KeyError(
                "未知能力 %r；已知：%s。自定义数据源用 node= 直接给节点名。"
                % (self.capability, ", ".join(sorted(CAPABILITIES)))
            )
        return CAPABILITIES[self.capability][0]

    def to_requests(self) -> List[DataRequest]:
        node = self.resolve_node()
        if not self.symbols:
            return [DataRequest(node, dict(self.params),
                                alias=self.capability, required=self.required)]
        requests = []
        for symbol in self.symbols:
            params = dict(self.params)
            # 节点自己声明必填 symbol 还是 token，这里两个都试着填
            params.setdefault("symbol", symbol)
            requests.append(DataRequest(
                node, params, alias="%s:%s" % (self.capability, symbol),
                required=self.required,
            ))
        return requests


# ---------------------------------------------------------------------------
# Features —— 统一读法
# ---------------------------------------------------------------------------


class Features:
    """把激活了的数据包成一套统一读法。

    **没激活的能力返回 ``None``，不是 0，也不抛异常。** 策略据此弃权。
    这一条是整个设计里最重要的约定：缺失和零是两个事实。
    """

    def __init__(self, bot: "UniversalBot", ctx: BotContext) -> None:
        self._bot = bot
        self.ctx = ctx

    # ---- 通用 ----

    def has(self, capability: str) -> bool:
        return self._bot.is_active(capability)

    def view(self, capability: str, symbol: Optional[str] = None):
        key = capability if symbol is None else "%s:%s" % (capability, symbol)
        return self.ctx.view(key)

    def _metric(self, capability: str, symbol: Optional[str], name: str) -> Optional[float]:
        view = self.view(capability, symbol)
        return None if view is None else view.metric(name)

    # ---- 量价 ----

    def close(self, symbol: str) -> Optional[float]:
        view = self.view("klines", symbol) or self.view("indicators", symbol)
        return None if view is None else view.latest("close")

    def bars(self, symbol: str):
        view = self.view("klines", symbol) or self.view("indicators", symbol)
        return None if view is None else view.frame

    def change_24h(self, symbol: str) -> Optional[float]:
        return self._metric("ticker", symbol, "change_pct")

    # ---- 衍生品 ----

    def funding(self, symbol: str) -> Optional[float]:
        """最新资金费率，**单位 bps**（统一过，避免小数/百分比混用）。"""
        rate = self._metric("funding", symbol, "rate")
        return None if rate is None else rate * 10_000.0

    def funding_series(self, symbol: str):
        view = self.view("funding", symbol)
        return None if view is None else view.series("value")

    def oi_change_pct(self, symbol: str) -> Optional[float]:
        """区间内 OI 变化百分比。取不到返回 None。"""
        view = self.view("open_interest", symbol)
        if view is None or view.empty:
            return None
        series = view.frame[view.frame["metric"] == "oi_value"]["value"]
        if len(series) < 2 or float(series.iloc[0]) <= 0:
            return None
        return float(series.iloc[-1] / series.iloc[0] - 1.0) * 100.0

    def long_short_ratio(self, symbol: str) -> Optional[float]:
        return self._metric("long_short", symbol, "long_short_ratio")

    def taker_ratio(self, symbol: str) -> Optional[float]:
        return self._metric("taker", symbol, "buy_sell_ratio")

    def basis_pct(self, symbol: str) -> Optional[float]:
        rate = self._metric("basis", symbol, "basis_rate")
        return None if rate is None else rate * 100.0

    # ---- 消息 / 事件 ----

    def events(self, *, enrich: bool = True, universe: Sequence[str] = ()):
        """所有激活了的事件源合并成一张 EventFrame；``enrich`` 会跑 news_features。"""
        import pandas as pd

        frames = [
            view.frame for key, view in self.ctx.typed.items()
            if view.kind is FrameKind.EVENT and not view.empty
        ]
        if not frames:
            return None
        merged = pd.concat(frames, ignore_index=True, sort=False)
        if not enrich:
            return merged
        from ..blocks import news_features as NF

        return NF.enrich_events(merged, universe=list(universe or self._bot.symbols))

    def news_metrics(self, *, windows: Sequence[str] = ("1h", "24h")):
        """事件流聚合成 MetricFrame —— 新闻从此和 funding 一样能读。"""
        from ..blocks import news_features as NF

        enriched = self.events(enrich=True)
        if enriched is None:
            return None
        return NF.aggregate_to_metrics(
            enriched, as_of=self.ctx.decision_time, windows=windows)

    # ---- 社交 ----

    def social(self, symbol: str) -> Optional[Dict[str, Any]]:
        """{rank, mentions, bull_ratio}；没激活或榜上无名返回 None。"""
        view = self.view("social_rank")
        if view is None or view.empty:
            return None
        token = _base_token(symbol)
        rows = view.frame[view.frame["instrument_id"].astype(str).str.upper().isin(
            {token, symbol.upper()})]
        if rows.empty:
            return None
        row = rows.iloc[0]
        bullish = _num(row.get("bullish_count"))
        bearish = _num(row.get("bearish_count"))
        total = (bullish or 0) + (bearish or 0)
        return {
            "rank": _num(row.get("rank")),
            "mentions": _num(row.get("mention_count")),
            "bull_ratio": None if total <= 0 else round((bullish or 0) / total, 3),
        }

    # ---- 资金流 ----

    def net_exchange_flow(self, symbol: str) -> Optional[float]:
        """净流入交易所（USD）。正数 = 币在往交易所搬 = 准备卖。"""
        view = self.view("large_flow", symbol)
        if view is None or view.empty:
            return None
        deposit = view.metric("large_deposit_amt")
        withdraw = view.metric("large_withdraw_amt")
        if deposit is None and withdraw is None:
            return None
        return float((deposit or 0.0) - (withdraw or 0.0))

    # ---- 宏观 / 情绪 ----

    def fear_greed(self) -> Optional[float]:
        return self._metric("fear_greed", None, "fear_greed")

    def ahr999(self) -> Optional[float]:
        return self._metric("ahr999", None, "ahr999_value")

    # ---- 截面 ----

    def universe_frame(self):
        view = self.view("universe")
        return None if view is None else view.frame

    def sectors(self):
        view = self.view("sector")
        return None if view is None else view.frame

    # ---- 账户 ----

    def side_of(self, symbol: str) -> str:
        return self.ctx.side_of(symbol)

    def position(self, symbol: str) -> float:
        return self.ctx.position(symbol)

    @property
    def equity(self) -> Optional[float]:
        return self.ctx.equity

    # ---- 质量 ----

    @property
    def quality(self):
        return self.ctx.data_quality()

    def degraded(self) -> List[str]:
        return self.ctx.degraded_inputs

    # ---- 出信号（省掉样板） ----

    def signal(
        self,
        symbol: Optional[str],
        intent: str,
        *,
        reason: Sequence[str] = (),
        summary: str = "",
        score: float = 60.0,
        confidence: float = 0.6,
        stop_pct: Optional[float] = None,
        stop_atr: Optional[Tuple[float, float]] = None,
        take_profit_pct: Optional[float] = None,
        max_seconds: Optional[int] = None,
        size_pct: float = 0.1,
        size_mode: str = "equity_pct",
        entry_price: Optional[float] = None,
        advisory: Optional[str] = None,
        behavior: str = "",
        evidence: Sequence[Evidence] = (),
        horizon_seconds: Optional[int] = None,
        exit_on_opposite: bool = True,
        **extra: Any,
    ) -> StandardSignal:
        """按 ``UniversalBot`` 的默认口径造一条完整信号。

        契约要求进场必须带出场计划；这里 ``stop_pct`` / ``stop_atr`` /
        ``max_seconds`` 一个都不给时，落到 ``exit_on_opposite_signal`` 并把
        「靠反向信号出场」明说出来，而不是留一个没有出场的仓位。
        """
        resolved = PositionIntent(intent) if isinstance(intent, str) else intent
        stop = None
        if stop_pct is not None:
            stop = StopSpec(pct=stop_pct)
        elif stop_atr is not None:
            stop = StopSpec(atr_mult=stop_atr[0], atr_value=stop_atr[1])

        plan = None
        if resolved.is_entry:
            plan = ExitPlan(
                stop_loss=stop,
                take_profit=(
                    (TakeProfitLegShim(take_profit_pct),) if take_profit_pct else ()
                ),
                time_stop=TimeStop(max_seconds=max_seconds) if max_seconds else None,
                exit_on_opposite_signal=exit_on_opposite,
                note="" if (stop or max_seconds) else "无价格止损：靠反向信号出场（显式声明）",
            )

        return StandardSignal(
            bot_id=self._bot.spec.bot_id,
            decision_time=self.ctx.decision_time,
            provenance=Provenance(strategy_id=self._bot.spec.bot_id),
            symbol=symbol,
            product=self._bot.spec.products[0],
            intent=resolved,
            advisory_action=AdvisoryAction(advisory) if advisory else None,
            score=score,
            confidence=confidence,
            entry=EntrySpec(type=EntryType.LIMIT, price=entry_price)
            if entry_price else EntrySpec(),
            exit_plan=plan,
            size=SizeSpec(mode=SizeMode(size_mode), value=size_pct),
            horizon_seconds=horizon_seconds or self._bot.spec.default_horizon_seconds,
            topic=self._bot.topic,
            reason_codes=tuple(reason),
            summary=summary,
            recommended_behavior=behavior,
            evidence=tuple(evidence),
            data_quality=self.ctx.data_quality(),
            **extra,
        )


def TakeProfitLegShim(pct: float):
    from .core import TakeProfitLeg

    return TakeProfitLeg(close_pct=1.0, pct=pct)


def _num(value) -> Optional[float]:
    import pandas as pd

    parsed = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(parsed) else float(parsed)


def _base_token(symbol: str) -> str:
    value = str(symbol).upper()
    for quote in ("USDT", "USDC", "FDUSD", "TUSD", "USD"):
        if value.endswith(quote) and len(value) > len(quote):
            return value[: -len(quote)]
    return value


# ---------------------------------------------------------------------------
# UniversalBot
# ---------------------------------------------------------------------------


class UniversalBot(StandardBot):
    """具体的、可直接实例化的标准 bot。

    它本身什么数据都会处理；一支实际策略是它**激活一部分**加上一条判断规则。
    """

    def __init__(
        self,
        bot_id: str,
        *,
        kind: BotKind = BotKind.TRADE,
        display_name: str = "",
        description: str = "",
        products: Sequence[str] = ("usd_m_perpetual",),
        symbols: Sequence[str] = (),
        market_scope: MarketScope = MarketScope.SINGLE,
        horizon_seconds: int = 3_600,
        topic: str = "",
        rule: Optional[Callable[[BotContext, Features], Iterable[StandardSignal]]] = None,
    ) -> None:
        self.symbols: Tuple[str, ...] = tuple(s.upper() for s in symbols)
        self.topic = topic or bot_id
        self._activations: List[Activation] = []
        self._rule = rule
        self.spec = BotSpec(
            bot_id=bot_id,
            kind=kind,
            display_name=display_name or bot_id,
            description=description,
            products=tuple(products),
            market_scope=market_scope,
            default_horizon_seconds=horizon_seconds,
            reads_positions=False,      # 激活 positions 时自动改 True
        )

    # ---- 激活 ----

    def activate(self, capability: str, *, symbols: Sequence[str] = (),
                 required: bool = False, node: str = "", **params: Any) -> "UniversalBot":
        """打开一个能力。``symbols`` 不给就用 bot 级的 symbols。"""
        targets = tuple(s.upper() for s in symbols) if symbols else ()
        activation = Activation(
            capability=capability, params=params,
            symbols=targets, required=required, node=node,
        )
        activation.resolve_node()          # 名字打错在这里就炸，不是取数时
        self._activations.append(activation)

        # 激活持仓 = 这支 bot 真的观察持仓 → 解锁 close/reduce/flip
        if capability == "positions":
            self.spec = BotSpec(**{**self.spec.__dict__, "reads_positions": True})
        return self

    # 常用的几个给个短名，读起来像话
    def activate_klines(self, symbols: Sequence[str], *, interval: str = "1h",
                        limit: int = 500) -> "UniversalBot":
        return self.activate("klines", symbols=symbols, required=True,
                             interval=interval, limit=limit)

    def activate_funding(self, symbols: Sequence[str], *, limit: int = 400) -> "UniversalBot":
        return self.activate("funding", symbols=symbols, limit=limit)

    def activate_open_interest(self, symbols: Sequence[str], *, period: str = "1h",
                               limit: int = 48) -> "UniversalBot":
        return self.activate("open_interest", symbols=symbols, period=period, limit=limit)

    def activate_news(self, *, page_size: int = 50) -> "UniversalBot":
        return self.activate("news", page_size=page_size)

    def activate_social(self, *, window: str = "24h", limit: int = 20) -> "UniversalBot":
        return self.activate("social_rank", window=window, limit=limit)

    def activate_positions(self) -> "UniversalBot":
        return self.activate("positions")

    def activate_custom(self, node: str, *, alias: str = "", required: bool = False,
                        **params: Any) -> "UniversalBot":
        """激活一个用户自注册的数据节点（``custom_*``）。"""
        return self.activate(alias or node, node=node, required=required, **params)

    def with_rule(
        self, rule: Callable[[BotContext, Features], Iterable[StandardSignal]]
    ) -> "UniversalBot":
        self._rule = rule
        return self

    def is_active(self, capability: str) -> bool:
        return any(item.capability == capability for item in self._activations)

    @property
    def activations(self) -> List[Activation]:
        return list(self._activations)

    # ---- StandardBot 接口 ----

    def required_data(self) -> Sequence[DataRequest]:
        requests: List[DataRequest] = []
        for activation in self._activations:
            targets = activation.symbols or (
                self.symbols if _needs_symbol(activation) else ()
            )
            resolved = Activation(
                capability=activation.capability, params=activation.params,
                symbols=targets, required=activation.required, node=activation.node,
            )
            requests.extend(resolved.to_requests())
        return requests

    def decide(self, ctx: BotContext) -> Iterable[StandardSignal]:
        if self._rule is None:
            raise ValueError(
                "%s 没有判断规则：用 .with_rule(fn) 给一条，或继承并覆盖 decide()。"
                % self.spec.bot_id
            )
        return self._rule(ctx, Features(self, ctx)) or []

    # ---- 一步跑通 ----

    def run_once(self, **kwargs: Any):
        """取数 → 判断 → 校验 → 盖章，返回 ``List[StandardSignal]``。"""
        ctx = self.fetch_context(**kwargs)
        return self.decide_checked(ctx)

    def describe(self) -> Dict[str, Any]:
        """这支 bot 激活了什么、能发什么 —— 前端和评审读这个。"""
        from .registry import describe_bot  # noqa: F401  (保持描述口径一致)

        return {
            "bot": self.spec.to_dict(),
            "activated": [
                {"capability": item.capability, "node": item.resolve_node(),
                 "symbols": list(item.symbols or self.symbols),
                 "required": item.required,
                 "note": CAPABILITIES.get(item.capability, ("", "自定义节点"))[1]}
                for item in self._activations
            ],
            "inactive": sorted(
                name for name in CAPABILITIES if not self.is_active(name)
            ),
            "inputs": self.typed_inputs(),
        }


#: 这些能力是按标的取的，其余是全市场/账户级的
_PER_SYMBOL = {
    "klines", "indicators", "ticker", "funding", "open_interest", "long_short",
    "top_trader", "taker", "basis", "liquidations", "large_flow", "whale",
    "chip", "orderbook", "ai_signal", "sentiment",
}


def _needs_symbol(activation: Activation) -> bool:
    return activation.capability in _PER_SYMBOL
