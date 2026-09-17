"""完整链路：任意因子 → 评测矩阵 → 组合 → 仓位 → **调仓买卖信号**。

前面几个模块各管一段，这里把它们接起来，并且**让评测的结论真的参与决策**，而不是评完
放在一边、构建时另起炉灶：

    factors            用户丢进来的一批因子（callable 或 DataFrame，量纲随意）
      ↓  ① screen      对每个因子跑 evaluate()，拿回完整评估矩阵
    cards              裁决 / 阻塞闸门 / 冻结方向 / 各段 RankIC
      ↓  ② admit       用评估结论决定谁能进组合、方向朝哪边
    score              合成分数（截面排名后等权，方向由 dev 段冻结）
      ↓  ③ construct   截面标准化 → 定仓 → 中性化/上限/归一
    weights            日期 × 标的 的目标权重
      ↓  ④ simulate    净额调仓、成本、资金费
    book + orders      权益曲线 **和一张可执行的调仓清单**

**评测到底给了构建什么。** 三件事，按重要性排：

1. **方向**（``sign``）。一个因子是"越大越好"还是"越小越好"由 dev 段的 RankIC 符号说了算，
   不由写因子的人顺手定的正负号说了算。不冻结方向直接交易，等于有一半因子在反着做。
   这是评测给构建的最硬的一条信息。
2. **准入**（``verdict`` / ``blocking``）。G0 数据闸门没过的因子**不进组合** —— 覆盖率不足
   或含未来信息，这种因子进来不是贡献分散度，是污染。其余裁决默认只记录不拦截：
   ``HOLD_*`` 表示证据不足以宣称有 alpha，但作为组合的一个分量仍然可以试。
3. **权重**。默认**等权**。按 dev 段 IC 大小加权（``scheme="ic"``）看着更聪明，但本仓库
   反复测到的同一件事是：用样本内统计量做选择/加权不带出样本（PBO 实测 0.66~0.71）。
   所以聪明的那个不是默认值，想用要自己开，并且 ``tune()`` 会把它和等权放在一起比。

**产出的调仓信号来自模拟器自己那本账**（``book.fills``），不是另算一遍 —— 另算一遍就会出现
"报表上的单子"和"回测真交的单子"对不上，而这种不一致只会在上线之后才被发现。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .baselines import cross_sectional_rank
from .portfolio import Book, Risk, simulate
from .strategy import target_weights

__all__ = ["FactorCard", "screen", "cards_frame", "admit", "combine_scores",
           "build_strategy", "orders", "latest_orders"]


# --------------------------------------------------------------------- 评估卡
@dataclass
class FactorCard:
    """一个因子从评测矩阵里带出来的、**构建阶段用得上的**那几项。

    完整矩阵留在 ``scorecard`` 里（14 维证据、诊断、分段指标都在），这里只把驱动决策的
    几个字段提到台面上，免得构建代码到处去翻 scorecard 的内部结构。
    """

    name: str
    verdict: str
    blocking: list[str]
    direction: float                      # dev 段冻结的方向，+1 / -1
    ic_dev: float                         # dev 段平均 RankIC（已按冻结方向取号）
    ic_oot: float
    t_hac_dev: float                      # dev 段 IC 的 HAC t 值
    net_bp_dev: float                     # dev 段每周期净收益（bp，已扣成本与资金费）
    admitted: bool = True
    reason: str = ""
    unverified: bool = False              # 闸门为 N/A：没验证过，不等于验证不过
    scorecard: object = None

    def to_row(self) -> dict:
        return {"factor": self.name, "verdict": self.verdict, "direction": self.direction,
                "ic_dev": self.ic_dev, "ic_oot": self.ic_oot, "t_hac_dev": self.t_hac_dev,
                "net_bp_dev": self.net_bp_dev, "admitted": self.admitted,
                "unverified": self.unverified, "blocking": ",".join(self.blocking),
                "reason": self.reason}

    def gate_status(self, key: str) -> str:
        """闸门状态：``PASS`` / ``FAIL`` / ``N/A``。**N/A 与 FAIL 不是一回事。**"""
        gates = getattr(self.scorecard, "gates", None) or {}
        gate = gates.get(key)
        return getattr(gate, "status", "N/A") if gate is not None else "N/A"


def _cell(metrics: pd.DataFrame, split: str, primary_h: int, column: str) -> float:
    """取某一段、某个期限的一格。``ic_mean`` 已经按冻结方向取过号。"""
    row = metrics[(metrics.split == split) & (metrics.h == primary_h)]
    if row.empty or column not in row:
        return np.nan
    return float(row.iloc[0][column])


def _direction(metrics: pd.DataFrame, primary_h: int) -> float:
    """方向只看 dev 段。val/oot 参与就是用未来信息挑方向。"""
    row = metrics[(metrics.split == "dev") & (metrics.h == primary_h)]
    if row.empty or "sign" not in row:
        return 1.0
    sign = float(row.iloc[0]["sign"])
    return sign if np.isfinite(sign) and sign != 0 else 1.0


def screen(factors: dict, panel, *, primary_h: int = 3, fast: bool = True,
           trials_seen: int | None = None,
           **evaluate_kwargs) -> dict[str, FactorCard]:
    """对每个因子跑一遍评测矩阵，回收构建需要的结论。

    ``fast=True`` 关掉诊断图与增量闸门（每个因子 ~0.03s 对 ~3s），闸门裁决与冻结方向不受
    影响；要完整的 14 维证据就关掉它。

    ``trials_seen`` 默认**按因子个数**如实上报：一次丢进来 20 个因子就是 20 个候选，
    评测那侧的搜索校正得知道这件事。

    候选是从一个更大的池子里挑出来的时候必须显式传：从 101 个里抽 8 个跑，看过的是
    101 个而不是 8 个，按 8 上报会把搜索宽度说小一个数量级。
    """
    from . import evaluate                                   # 延迟导入，避免循环依赖

    n = len(factors) if trials_seen is None else int(trials_seen)
    if n < len(factors):
        raise ValueError(f"trials_seen={n} is smaller than the {len(factors)} factors "
                         f"being screened; the pool cannot be smaller than the sample")
    out: dict[str, FactorCard] = {}
    for name, factor in factors.items():
        card = evaluate(factor, panel, name=name, primary_h=primary_h,
                        trials_seen=max(n, 1),
                        with_diagnostics=not fast, with_incremental=not fast,
                        **evaluate_kwargs)
        out[name] = FactorCard(
            name=name, verdict=card.verdict, blocking=list(card.blocking),
            direction=_direction(card.metrics, primary_h),
            ic_dev=_cell(card.metrics, "dev", primary_h, "ic_mean"),
            ic_oot=_cell(card.metrics, "oot", primary_h, "ic_mean"),
            t_hac_dev=_cell(card.metrics, "dev", primary_h, "ic_t_hac"),
            net_bp_dev=_cell(card.metrics, "dev", primary_h, "net_bp"),
            scorecard=card)
    return out


def cards_frame(cards: dict[str, FactorCard]) -> pd.DataFrame:
    """把一批评估卡摊成一张表，人看的。"""
    return pd.DataFrame([c.to_row() for c in cards.values()]).set_index("factor")


def admit(cards: dict[str, FactorCard], *, block_on=("G0_data",),
          require_verdict: tuple[str, ...] | None = None,
          min_ic_dev: float | None = None) -> dict[str, FactorCard]:
    """用评估结论决定谁进组合。**就地**在卡上写 ``admitted`` 与理由。

    默认只拦 ``G0_data`` 且**只在它真的 FAIL 时拦**：覆盖率不足、或因果抽查证明因子读了
    未来，这种进组合不是贡献分散度，是污染。

    ``FAIL`` 与 ``N/A`` 必须分开对待，这是本函数最容易写错的一处。传进来的因子如果是一张
    **DataFrame**，评测无法在前缀上重跑它，因果性就只能记 ``N/A`` —— 那是"没验证过"，
    不是"验证不过"。按 ``blocking`` 列表拦截会把两者混为一谈，结果是所有 DataFrame 因子
    全被拒之门外（实测：三个正常因子一个都进不来）。这里改为按**闸门状态**判，N/A 放行
    但在卡上标 ``unverified``，让"我们没检查过"这件事留在台面上而不是被静默吞掉。

    其余裁决默认放行 —— ``HOLD_*`` 的含义是"证据不足以宣称有 alpha"，不是"确定没用"，
    而组合本来就不需要每个分量单独成立。要更严就传 ``require_verdict=("PASS",...)``。
    """
    for card in cards.values():
        failed = [g for g in block_on if card.gate_status(g) == "FAIL"]
        card.unverified = any(card.gate_status(g) == "N/A" for g in block_on)
        if failed:
            card.admitted, card.reason = False, f"闸门 {','.join(failed)} FAIL"
        elif require_verdict is not None and card.verdict not in require_verdict:
            card.admitted, card.reason = False, f"裁决 {card.verdict} 不在准入集合"
        elif min_ic_dev is not None and not (abs(card.ic_dev) >= min_ic_dev):
            # 标定文件里的噪声地板是现成的参照（h=3 时 p95 ≈ 0.043）。用它当门槛，
            # 意思是"连随机信号都达得到的 IC 不算证据"。缺 IC 一律不放行。
            card.admitted, card.reason = False, f"dev IC {card.ic_dev:.3f} < {min_ic_dev}"
        else:
            card.admitted = True
            card.reason = "闸门未验证（因子以 DataFrame 传入，无法重跑前缀）" if card.unverified else ""
    return cards


def combine_scores(factors: dict, cards: dict[str, FactorCard], panel, *,
                   scheme: str = "equal") -> pd.DataFrame:
    """合成分数：各因子先截面排名（消掉量纲），乘评测冻结的方向，再加权相加。

    ``scheme``：``"equal"`` 等权（默认，见模块文档为什么）；``"ic"`` 按 dev 段 |IC| 加权。
    后者是**用样本内统计量加权**，本仓库实测这类动作不带出样本，用之前先量。

    注意不存在"按 IC 符号筛选"这种 scheme：方向已经在 dev 段冻结过，dev 段的 IC 取号
    之后按构造必然非负，拿它筛等于什么都没筛。要筛强弱请用 ``admit(min_ic_dev=...)``，
    那里比的是**幅度**与标定出来的噪声地板。

    加权求和本身**转调** :func:`factor_eval.strategy.combine` —— 这里原来另有一份实现，
    而且两份的缺失值处理不一致：那一份把缺失当 0 计入分子、却按**总**权重做分母，正是
    本函数注释声称要避免的"替没数据的标的投一票中性"。合成只留一份实现。
    """
    from .strategy import combine                     # 延迟导入，避免循环依赖

    usable = {n: c for n, c in cards.items() if c.admitted}
    if not usable:
        raise ValueError("没有因子通过准入，无法组合")
    if scheme not in ("equal", "ic"):
        raise ValueError(f"unknown scheme {scheme!r}")

    signed, weights = {}, {}
    for name, card in usable.items():
        raw = factors[name]
        frame = raw(panel) if callable(raw) else raw
        w = 1.0 if scheme == "equal" else (
            abs(card.ic_dev) if np.isfinite(card.ic_dev) else 0.0)
        if w <= 0:
            continue
        # 方向在 dev 段冻结，合成前乘上去；量纲由 combine 内部的截面排名消掉。
        signed[name] = frame * card.direction
        weights[name] = w
    if not signed:
        raise ValueError("加权之后没有有效分量")
    # 末位取整**不是**美观处理，是必需的。合成分数是若干条离散排名之和，精确并列非常
    # 常见；而下游还要再排一次名，1e-16 的浮点差异就会把一个并列打破、排名跳一档，
    # 在十个标的的截面上放大成 0.25 的权重差异。实测：把每个因子取负号（数学上完全
    # 等价，因为方向已由评测冻结）本应得到同一组权重，不取整时有 581 格对不上。
    return combine(signed, panel.mask, weights=weights).round(12)


# ------------------------------------------------------------------- 调仓信号
def orders(book: Book, panel, *, equity: float = 1.0, min_weight: float = 1e-4,
           since=None) -> pd.DataFrame:
    """把模拟器真正执行的成交摊成一张**调仓清单**。

    每一行是一笔要下的单：哪天、哪个标的、买还是卖、目标权重是多少、这一笔动多少。
    ``delta_weight`` 取自 ``book.fills`` —— 也就是回测里实际成交的那一笔，不是另算的。

    ``date`` 是**成交日**（决策日之后 ``entry_lag`` 天的开盘），``ref_price`` 是那一笔的
    参考成交价，因此这张表可以直接对着盘口核对。``min_weight`` 以下的碎单丢掉：净额调仓
    每天都会产生一堆几个 bp 的噪声单，真下单时它们只贡献手续费。
    """
    fills = book.fills
    if since is not None:
        fills = fills.loc[fills.index >= pd.Timestamp(since, tz=panel.index.tz)]
    rows = []
    for date, row in fills.iterrows():
        moved = row[row.abs() >= min_weight]
        if moved.empty:
            continue
        for symbol, delta in moved.items():
            rows.append({
                "date": date, "symbol": symbol,
                "side": "BUY" if delta > 0 else "SELL",
                "delta_weight": float(delta),
                "target_weight": float(book.weights.loc[date, symbol])
                if date in book.weights.index else np.nan,
                "notional": float(delta) * equity * float(book.equity.loc[date]),
                "ref_price": float(panel.open.loc[date, symbol]),
            })
    out = pd.DataFrame(rows)
    return out if out.empty else out.sort_values(["date", "symbol"]).reset_index(drop=True)


def latest_orders(book: Book, panel, **kwargs) -> pd.DataFrame:
    """最近一次调仓的单子。上线时要发出去的就是这一张。"""
    table = orders(book, panel, **kwargs)
    return table if table.empty else table[table["date"] == table["date"].max()]


# --------------------------------------------------------------------- 全链路
@dataclass
class Strategy:
    """一条链路跑完的全部产物。"""

    cards: dict[str, FactorCard]
    score: pd.DataFrame
    weights: pd.DataFrame
    book: Book
    orders: pd.DataFrame
    config: dict = field(default_factory=dict)

    def summary(self) -> str:
        used = [n for n, c in self.cards.items() if c.admitted]
        return (f"因子 {len(self.cards)} 个，准入 {len(used)} 个；"
                f"{self.book.summary()}；调仓单 {len(self.orders)} 笔")


def build_strategy(factors: dict, panel, *, primary_h: int = 3, scheme: str = "equal",
                   sizing: str = "equal_notional", rebalance: int = 3, cap: float = 0.35,
                   gross: float = 1.0, neutral: bool = True, vol_window: int = 20,
                   entry_lag: int = 2, cost_bps: float = 6.5, start="2022-04-01",
                   risk: Risk | None = None, block_on=("G0_data",),
                   require_verdict: tuple[str, ...] | None = None,
                   min_ic_dev: float | None = None,
                   fast: bool = True, min_weight: float = 1e-4) -> Strategy:
    """一次调用跑完 **评测 → 准入 → 合成 → 定仓 → 模拟 → 调仓信号**。

    ``factors`` 是 ``{名字: callable 或 DataFrame}``，量纲随意、正负号随意 —— 方向由评测
    的 dev 段冻结，量纲由截面排名消掉。这就是"不管来的是什么因子"这句话在代码里的兑现。
    """
    cards = admit(screen(factors, panel, primary_h=primary_h, fast=fast),
                  block_on=block_on, require_verdict=require_verdict, min_ic_dev=min_ic_dev)
    score = combine_scores(factors, cards, panel, scheme=scheme)

    # 定仓、中性化、上限、调仓节奏、预热期全部复用构建层的既有口径，不在这里重写一遍 ——
    # 两处实现迟早会漂，而漂掉的那部分正好是最难看出来的。
    targets = target_weights(score, panel, sizing=sizing, vol_window=vol_window, cap=cap,
                             gross=gross, neutral=neutral, rebalance=rebalance, start=start)

    book = simulate(targets, panel, entry_lag=entry_lag, cost_bps=cost_bps, risk=risk)
    return Strategy(
        cards=cards, score=score, weights=targets, book=book,
        orders=orders(book, panel, min_weight=min_weight, since=start),
        config={"primary_h": primary_h, "scheme": scheme, "sizing": sizing,
                "rebalance": rebalance, "cap": cap, "gross": gross, "neutral": neutral,
                "entry_lag": entry_lag, "cost_bps": cost_bps, "start": start,
                "block_on": list(block_on), "require_verdict": require_verdict,
                "min_ic_dev": min_ic_dev,
                "admitted": [n for n, c in cards.items() if c.admitted]})
