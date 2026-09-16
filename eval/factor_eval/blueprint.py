"""一张策略蓝图：把规则型 / 事件驱动型 / 量价截面型写成同一种东西。

读了一批现成策略之后，它们的骨架其实是同一副：

    触发 → 硬门 → 分层打分 → 裁决 → 方向 → 仓位 → 出场 → 复评

三类策略的差别不在骨架，只在**填进去的内容**：

=============== ==================== ==================== ====================
                规则型                事件驱动型             量价截面型
=============== ==================== ==================== ====================
触发 schedule    每根 bar             事件发生时             固定周期调仓
硬门 gates       指标处于某区间         事件强度达标           成交额/上市天数达标
打分 tiers       EMA/RSI/MACD 分档     事件类型与烈度分档       动量/资金流/衍生品分档
裁决 verdict     分数 → 多/空/观望      分数 → 是否介入         分数 → 入选名单
仓位 sizing      止损距离 → 名义        固定风险预算           截面排序 → 权重
出场 exits       止损/止盈/超时         事件结束或超时          下次调仓
=============== ==================== ==================== ====================

**这就是"必须要有的几个部分"。** 少一块就有一类策略写不出来：没有硬门，筛选型写不了；
没有出场规则，规则型写不了；没有事件触发，事件型写不了；没有截面排序，量价型写不了。

产出统一是一张 ``日期 × 标的`` 的**目标权重**表，交给 :func:`factor_eval.portfolio.simulate`
去算钱 —— 因此不管哪类策略，损益口径、成本、资金费、换手都是同一套，可比。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .baselines import cross_sectional_rank
from .portfolio import Risk, simulate
from .strategy import realised_vol

__all__ = ["Gate", "Tier", "Exits", "Blueprint", "compile_targets", "blueprint_targets",
           "run_blueprint"]

Signal = Callable[[object], pd.DataFrame]     # Panel -> 日期 × 标的


# --------------------------------------------------------------------------- 硬门
@dataclass
class Gate:
    """硬门：不过就出局，**不是**扣分。

    筛选型策略的"必须出现在热榜""24h 成交额 ≥ 50 万"属于这一类。把它写成扣分项是常见
    错误：分数可以被别的维度补回来，硬门不行。
    """

    name: str
    signal: Signal
    op: str                                   # ">=", ">", "<=", "<", "between", "abs>="
    value: float | tuple[float, float]

    def mask(self, panel) -> pd.DataFrame:
        x = self.signal(panel)
        if self.op == ">=":
            out = x >= self.value
        elif self.op == ">":
            out = x > self.value
        elif self.op == "<=":
            out = x <= self.value
        elif self.op == "<":
            out = x < self.value
        elif self.op == "abs>=":
            out = x.abs() >= self.value
        elif self.op == "between":
            lo, hi = self.value
            out = (x >= lo) & (x <= hi)
        else:
            raise ValueError(f"gate {self.name}: unsupported op {self.op!r}")
        # 缺失一律**不放行**。把 NaN 当成通过,等于让一个还没有数据的标的进场。
        return out.where(x.notna(), False).fillna(False).astype(bool)


# --------------------------------------------------------------------------- 打分
@dataclass
class Tier:
    """一层打分：把一个信号按档位映射成分数。

    ``bands`` 是 ``(下界, 上界, 分数)`` 的列表，左闭右开，命中第一个就停；都不命中记 0。
    分档而不是线性映射，是因为现成策略基本都这么写（"RSI 在 40~60 给 +2"），而且分档
    对异常值天然稳健。
    """

    name: str
    signal: Signal
    bands: Sequence[tuple[float, float, float]]
    weight: float = 1.0

    def score(self, panel) -> pd.DataFrame:
        x = self.signal(panel)
        out = pd.DataFrame(0.0, index=x.index, columns=x.columns)
        assigned = pd.DataFrame(False, index=x.index, columns=x.columns)
        for lo, hi, points in self.bands:
            hit = (x >= lo) & (x < hi) & ~assigned & x.notna()
            out = out.where(~hit, float(points))
            assigned |= hit
        return (out * self.weight).where(x.notna())


# --------------------------------------------------------------------------- 出场
@dataclass
class Exits:
    """出场规则。全部按**入场价**计算，用日内高低价判定是否触发。

    没有建模的：盘中触发顺序（同一根 bar 里止损和止盈都够到时，这里按**止损优先**，
    这是保守的一侧）、滑点、部分成交。
    """

    stop_pct: float | None = None             # 止损距离，例如 0.05
    take_profit_pct: float | None = None      # 止盈距离
    max_holding_days: int | None = None       # 超时平仓
    trailing_pct: float | None = None         # 移动止损（按持仓期内最优价回撤）


# --------------------------------------------------------------------------- 蓝图
@dataclass
class Blueprint:
    """一张完整策略蓝图。"""

    name: str
    tiers: Sequence[Tier]
    gates: Sequence[Gate] = ()
    trigger: Signal | None = None             # 事件驱动：为真的那天才评估
    schedule: int = 5                         # 周期型：每 N 天评估一次
    entry_score: float = 1.0                  # 达到这个分才进场
    top_k: int | None = None                  # 截面型：只取分数最高的 k 个
    direction: str = "score"                  # "score" | "long_only" | "short_only"
    sizing: str = "equal"                     # "equal" | "score" | "inverse_vol" | "risk_budget"
    gross: float = 1.0
    cap: float = 0.35
    neutral: bool = False                     # 截面中性（多空对冲）
    vol_window: int = 20
    risk_per_trade: float = 0.02              # sizing="risk_budget" 时每笔风险预算
    exits: Exits = field(default_factory=Exits)

    def score(self, panel) -> pd.DataFrame:
        """总分 = 各层加权分数之和，硬门不过的格子为缺失。"""
        if not self.tiers:
            raise ValueError(f"blueprint {self.name}: needs at least one tier")
        total = None
        for tier in self.tiers:
            s = tier.score(panel)
            total = s if total is None else total.add(s, fill_value=0.0)
        allowed = panel.mask.copy()
        for gate in self.gates:
            allowed &= gate.mask(panel)
        return total.where(allowed)


def _size(score: pd.DataFrame, panel, bp: Blueprint) -> pd.DataFrame:
    """分数 → 权重。四种口径,对应现成策略里能见到的几种写法。"""
    picked = score.notna()
    if bp.direction == "long_only":
        side = picked.astype(float)
    elif bp.direction == "short_only":
        side = -picked.astype(float)
    else:
        side = np.sign(score).where(picked)

    if bp.sizing == "equal":
        raw = side
    elif bp.sizing == "score":
        raw = side * score.abs()
    elif bp.sizing == "inverse_vol":
        raw = side / realised_vol(panel, bp.vol_window).clip(lower=1e-4)
    elif bp.sizing == "risk_budget":
        # 现成策略最常见的写法:先定"这笔最多亏多少",再由止损距离反推名义。
        stop = bp.exits.stop_pct or 0.05
        raw = side * (bp.risk_per_trade / stop)
    else:
        raise ValueError(f"unknown sizing {bp.sizing!r}")

    w = raw.where(picked)
    if bp.neutral:
        w = w.sub(w.mean(axis=1), axis=0)
    w = w.clip(lower=-bp.cap, upper=bp.cap)
    if bp.sizing == "risk_budget":
        # 风险预算是**绝对**口径,不归一到 gross;只在总敞口超限时整体缩放。
        over = w.abs().sum(axis=1) / bp.gross
        w = w.div(over.clip(lower=1.0), axis=0)
    else:
        scale = w.abs().sum(axis=1).replace(0, np.nan)
        w = w.div(scale, axis=0) * bp.gross
    return w.fillna(0.0)


def compile_targets(bp: Blueprint, panel) -> pd.DataFrame:
    """蓝图 → 目标权重表（非评估日留空，交给模拟器继续持有）。"""
    score = bp.score(panel)
    if bp.top_k is not None:
        rank = score.rank(axis=1, ascending=False)
        score = score.where(rank <= bp.top_k)
    score = score.where(score >= bp.entry_score)
    weights = _size(score, panel, bp)

    when = pd.Series(False, index=panel.index)
    if bp.trigger is not None:
        fired = bp.trigger(panel)
        when |= (fired.any(axis=1) if isinstance(fired, pd.DataFrame) else fired).astype(bool)
    if bp.schedule:
        periodic = np.zeros(len(panel.index), dtype=bool)
        periodic[::bp.schedule] = True
        when |= pd.Series(periodic, index=panel.index)
    return weights.where(when, np.nan)


def apply_exits(targets: pd.DataFrame, panel, exits: Exits) -> pd.DataFrame:
    """在目标权重上叠加止损 / 止盈 / 超时：触发当天把该标的的目标置 0。

    用**开盘价**记入场、用当日高低价判触发，与模拟器的成交约定一致。同一天两侧都够到
    时按**止损优先**，是保守的一侧。
    """
    if not any((exits.stop_pct, exits.take_profit_pct, exits.max_holding_days, exits.trailing_pct)):
        return targets
    opens, high, low = panel.open, panel.high, panel.low
    out = targets.copy()
    for symbol in panel.symbols:
        want = targets[symbol]
        entry_price, entry_i, side, best = np.nan, -1, 0.0, np.nan
        values = want.to_numpy(dtype=float)
        for i in range(len(want)):
            w = values[i]
            if np.isfinite(w) and w != 0 and side == 0:
                entry_price, entry_i, side = opens[symbol].iloc[i], i, np.sign(w)
                best = entry_price
                continue
            if side == 0 or not np.isfinite(entry_price):
                continue
            hi, lo = high[symbol].iloc[i], low[symbol].iloc[i]
            best = max(best, hi) if side > 0 else min(best, lo)
            move_bad = (lo / entry_price - 1) if side > 0 else (1 - hi / entry_price)
            move_good = (hi / entry_price - 1) if side > 0 else (1 - lo / entry_price)
            drop = (best - lo) / best if side > 0 else (hi - best) / best
            hit = ((exits.stop_pct is not None and move_bad <= -exits.stop_pct)
                   or (exits.trailing_pct is not None and np.isfinite(drop) and drop >= exits.trailing_pct)
                   or (exits.take_profit_pct is not None and move_good >= exits.take_profit_pct)
                   or (exits.max_holding_days is not None and i - entry_i >= exits.max_holding_days))
            if hit:
                out.iloc[i, out.columns.get_loc(symbol)] = 0.0
                side, entry_price, entry_i = 0.0, np.nan, -1
            elif np.isfinite(values[i]) and values[i] == 0:
                side, entry_price, entry_i = 0.0, np.nan, -1
    return out


def blueprint_targets(bp: Blueprint, panel, *, start="2022-04-01") -> pd.DataFrame:
    """编译 → 截掉预热期 → 叠加出场。**不含模拟**，因此可以被组合后再一次性记账。"""
    targets = compile_targets(bp, panel)
    if start is not None:
        bound = pd.Timestamp(start, tz=panel.index.tz)
        targets = targets.copy()
        targets.loc[targets.index < bound] = np.nan   # 预热期不交易
    return apply_exits(targets, panel, bp.exits)


def run_blueprint(bp: Blueprint, panel, *, entry_lag: int = 2, cost_bps: float = 6.5,
                  risk: Risk | None = None, start="2022-04-01") -> dict:
    """编译 → 叠加出场 → 模拟。返回 ``{"targets", "score", "book"}``。"""
    targets = blueprint_targets(bp, panel, start=start)
    return {"targets": targets, "score": bp.score(panel),
            "book": simulate(targets, panel, entry_lag=entry_lag, cost_bps=cost_bps, risk=risk)}
