"""从因子到仓位：合成 → 定仓 → 风控 → 目标权重。

矩阵回答"这个因子有没有信息"。这一层回答下一个问题：**手上有一个（或几个）因子，
怎么变成一个组合。**好坏的判据是可检验的：同样的标的、成本、调仓节奏、模拟器，只有
权重算法不同，去比 :func:`naive_weights`。

**已测到的结论，先写在这里免得被误用**：在随库的前十永续面板上，等风险仓位
（``sizing="risk_scaled"``）、波动目标、回撤阈值这几项**都没有稳定跑赢等名义金额**的
裸权重；回撤阈值那条更是退化成了恒定半仓（这类组合只有约 4% 的交易日创新高）。真正
稳定的只有换手下降。换个标的池或换个时期结论可能不同，所以这些口径都留着、并且默认
关闭 —— 但不要默认打开它们就会更好。

三步（与 14 号设计文档一致）::

    s_i,t                                  因子分数
      ↓  ① 截面标准化
    z_i,t = 秩去均值 / 截面标准差
      ↓  ② 除以风险
    w̃_i,t = z_i,t / σ_i,t                  σ 为事前已实现波动
      ↓  ③ 中性化 + 上限 + 归一
    w_i,t,  Σw = 0,  |w_i| ≤ cap,  Σ|w| = gross

第二步为什么是 ``z/σ`` 而不是 ``z/σ²``：``z/σ²`` 对应"分数预测的是**绝对**收益"，
``z/σ`` 对应"分数预测的是**风险调整后**收益"。RankIC 衡量的是序，序表达的是相对强弱
而不是绝对 bp，所以这里取 ``z/σ`` —— 等风险贡献。加密截面里 σ 跨标的能差 3~5 倍，这个
选择决定组合是被 BTC 主导还是被 memecoin 主导，不是细节。

杠杆与尾部保护在 :class:`factor_eval.portfolio.Risk` 里，属于组合层，不改变持仓的相对
结构。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import cross_sectional_rank
from .engine import DEFAULT_SPLITS, _split_bounds
from .portfolio import Book, Risk, simulate
from .statistics import daily_cross_sectional
from .targets import build_targets

__all__ = ["combine", "realised_vol", "target_weights", "naive_weights", "construct",
           "frozen_sign", "factor_ic_series", "walk_forward"]


def frozen_sign(score: pd.DataFrame, panel, *, h: int = 3, entry_lag: int = 2,
                splits=None, min_assets: int = 5) -> float:
    """按开发段平均 RankIC 的符号确定多空方向，之后不再翻转。

    这是构建层要从评测拿走的**第一条信息**：一个因子是"越大越好"还是"越小越好",由
    数据说了算,而不是由写因子的人顺手定的正负号。不冻结方向就直接交易,等于有一半
    因子在反着做 —— 这条在评测里是硬约定,策略层必须沿用同一条,否则两边说的不是同
    一个因子。

    方向只在 dev 段确定，val/oot 不再看 —— 否则就是用未来信息选方向。
    """
    bounds = _split_bounds(DEFAULT_SPLITS if splits is None else splits)
    start, end = bounds["dev"]
    target = build_targets(panel, h, entry_lag)["targets"]["raw_rtf"]
    ic = daily_cross_sectional(score, target, panel.mask, min_assets=min_assets)["rank_ic"]
    dev = ic.loc[(ic.index >= start) & (ic.index <= end)]
    mean = float(dev.mean())
    return -1.0 if np.isfinite(mean) and mean < 0 else 1.0


def combine(factors: dict[str, pd.DataFrame], mask: pd.DataFrame, *,
            weights: dict[str, float] | None = None) -> pd.DataFrame:
    """把多个因子合成一个分数：逐期截面标准化后加权求和。

    默认等权。**不按 IC 加权** —— IC 是在同一段样本上估出来的，拿它当权重等于把估计
    误差再放大一次；等权在因子数不多时几乎总是更稳。要按信念加权就显式传 ``weights``。

    缺失处理：某个因子当日缺某个标的时，该因子在这个格子上不投票，其余因子按**实际
    参与的权重**重新归一，而不是把缺失当成 0 分（0 在标准化之后是"中位"，等于替它投了
    一票中性票）。
    """
    if not factors:
        raise ValueError("combine needs at least one factor")
    weights = {k: float(weights.get(k, 0.0)) for k in factors} if weights else {k: 1.0 for k in factors}
    if sum(abs(v) for v in weights.values()) == 0:
        raise ValueError("combine weights cannot all be zero")
    total = None
    used = None
    for name, frame in factors.items():
        w = weights[name]
        if w == 0:
            continue
        z = cross_sectional_rank(frame, mask) * w
        present = z.notna()
        contribution = z.fillna(0.0)
        total = contribution if total is None else total.add(contribution, fill_value=0.0)
        share = present.astype(float) * abs(w)
        used = share if used is None else used.add(share, fill_value=0.0)
    score = total / used.replace(0, np.nan)
    return score.where(mask)


def realised_vol(panel, window: int = 20, min_periods: int | None = None) -> pd.DataFrame:
    """事前已实现波动（日频），用于把分数换算成仓位。

    收益取 ``close`` 的对数收益，窗口内样本标准差，**再滞后一日** —— 决策日当天的收盘
    收益在决策那一刻还没完全落定，直接用会把当日信息混进仓位。
    """
    returns = np.log(panel.close / panel.close.shift(1))
    floor = max(5, window // 2) if min_periods is None else int(min_periods)
    return returns.rolling(window, min_periods=floor).std().shift(1)


def _finalise(raw: pd.DataFrame, mask: pd.DataFrame, *, neutral: bool, cap: float,
              gross: float, min_assets: int, passes: int = 8) -> pd.DataFrame:
    """原始分数 → 可交易权重：中性化 → 归一到 gross → 截上限 → 再归一。

    **顺序很要紧。** ``cap`` 说的是"单个标的最多占权益多少"，所以必须在归一化**之后**
    才有意义。先截再归一是另一回事：原始分数是标准差单位（±1.5 量级），拿 0.35 去截
    等于把信号压平成几乎等权的两堆，收益凭空少掉一大截，而权重表面上看还很正常。

    截断会破坏中性和 gross，重新归一又可能把别的标的顶过上限，所以迭代几轮；截面只有
    约 8 个名字时通常 2~3 轮就收敛。轮数用完仍越界的，宁可保留一点越界也不放弃中性 ——
    净额敞口是更难承受的那个。
    """
    w = raw.where(mask)
    enough = w.notna().sum(axis=1) >= min_assets
    for _ in range(passes):
        if neutral:
            w = w.sub(w.mean(axis=1), axis=0)
        scale = w.abs().sum(axis=1).replace(0, np.nan)
        w = w.div(scale, axis=0) * gross
        if float(w.abs().max().max() or 0) <= cap * gross + 1e-12:
            break
        w = w.clip(lower=-cap * gross, upper=cap * gross)
    keep = enough.reindex(w.index, fill_value=False)
    # 名字不够的调仓日**清仓**(全 0),不是留空。留空在模拟器里等于"继续持有",
    # 那会让一个早期仓位在截面塌掉之后仍然挂着,原样扛过后面的跳空 ——
    # 实测这正是把一条曲线打到归零的原因。
    return w.where(keep, 0.0)


SIZINGS = ("equal_notional", "risk_scaled")


def target_weights(score: pd.DataFrame, panel, *, sizing: str = "equal_notional",
                   vol_window: int = 20, cap: float = 0.35,
                   gross: float = 1.0, neutral: bool = True, min_assets: int = 5,
                   rebalance: int = 3, vol_floor: float = 1e-4,
                   start=None) -> pd.DataFrame:
    """分数 → 目标权重，并按 ``rebalance`` 天输出调仓行。

    ``sizing``：

    * ``"equal_notional"``（默认）—— ``w ∝ z``，等名义金额。
    * ``"risk_scaled"`` —— ``w ∝ z/σ``，等风险贡献。

    **默认为什么不是 risk_scaled**：理论上等风险贡献更漂亮，但在本仓库这个前十永续
    面板上，开发段就测出它跑输等名义金额（四因子 rebalance=5：年化 +4.5% vs +8.8%）。
    原因是 ``/σ`` 会系统性地把仓位从高波动名字上挪走，而这个样本里涨得最多的恰恰是
    高波动的那几个。**这是数据在开发段给出的结论，不是事后挑的**；换个标的池或换个
    年份，结论可能反过来，所以两种都留着、并且都写清楚。

    非调仓日整行留空，交给模拟器"继续持有"。调仓频率本身就是一个风控旋钮：拉长它换手
    下降、成本下降，但跟不上信号；矩阵里的秩自相关可以用来判断拉多长还划算。
    """
    if sizing not in SIZINGS:
        raise ValueError(f"sizing must be one of {SIZINGS}")
    if not 0 < cap <= 1:
        raise ValueError("cap must be in (0, 1]")
    if gross <= 0:
        raise ValueError("gross must be positive")
    if rebalance < 1:
        raise ValueError("rebalance must be >= 1 day")
    z = cross_sectional_rank(score, panel.mask)
    if sizing == "risk_scaled":
        raw = z / realised_vol(panel, vol_window).clip(lower=vol_floor)
    else:
        raw = z
    w = _finalise(raw, panel.mask, neutral=neutral, cap=cap, gross=gross, min_assets=min_assets)
    on = pd.Series(False, index=w.index)
    on.iloc[::rebalance] = True
    if start is not None:
        on &= w.index >= pd.Timestamp(start, tz=w.index.tz)
    return w.where(on, np.nan)


def naive_weights(score: pd.DataFrame, panel, *, gross: float = 1.0, neutral: bool = True,
                  min_assets: int = 5, rebalance: int = 3, start=None) -> pd.DataFrame:
    """基准：**裸用因子**。等名义金额的秩权重，不看波动、不设上限、不做杠杆调节。

    这正是评测引擎内部的权重口径，也是"随便直接拿因子去买卖"的意思。构建层要赢的就是
    它 —— 同样的标的、成本、调仓节奏，只有权重算法不同。
    """
    z = cross_sectional_rank(score, panel.mask)
    w = _finalise(z, panel.mask, neutral=neutral, cap=1.0, gross=gross, min_assets=min_assets)
    on = pd.Series(False, index=w.index)
    on.iloc[::rebalance] = True
    if start is not None:
        on &= w.index >= pd.Timestamp(start, tz=w.index.tz)
    return w.where(on, np.nan)


def construct(factors: dict[str, pd.DataFrame] | pd.DataFrame, panel, *,
              factor_weights: dict[str, float] | None = None,
              sizing: str = "equal_notional",
              vol_window: int = 20, cap: float = 0.35, gross: float = 1.0,
              neutral: bool = True, rebalance: int = 3, entry_lag: int = 2,
              cost_bps: float = 6.5, risk: Risk | None = None,
              freeze_direction: bool = True, start="2022-04-01") -> dict:
    """一次跑完：合成 → 构建组合 → 同时跑基准 → 返回两本账。

    返回 ``{"score", "constructed": Book, "naive": Book, "config"}``。两本账走的是同一个
    模拟器、同一份面板、同一套成本与调仓节奏，因此可以直接比。
    """
    if isinstance(factors, pd.DataFrame):
        factors = {"factor": factors}
    signs = {}
    if freeze_direction:
        # 每个因子**各自**在 dev 上定向后再合成:先合成再定向会让一正一反的两个因子
        # 互相抵消,合出来的分数既不是这个也不是那个。
        signs = {name: frozen_sign(frame, panel, h=rebalance, entry_lag=entry_lag)
                 for name, frame in factors.items()}
        factors = {name: frame * signs[name] for name, frame in factors.items()}
    score = combine(factors, panel.mask, weights=factor_weights)
    built = target_weights(score, panel, sizing=sizing, vol_window=vol_window, cap=cap,
                           gross=gross, neutral=neutral, rebalance=rebalance, start=start)
    base = naive_weights(score, panel, gross=gross, neutral=neutral, rebalance=rebalance,
                         start=start)
    return {
        "score": score,
        "constructed": simulate(built, panel, entry_lag=entry_lag, cost_bps=cost_bps, risk=risk),
        "naive": simulate(base, panel, entry_lag=entry_lag, cost_bps=cost_bps),
        "config": {"factors": list(factors), "factor_weights": factor_weights,
                   "frozen_signs": signs, "sizing": sizing, "start": str(start),
                   "vol_window": vol_window, "cap": cap, "gross": gross, "neutral": neutral,
                   "rebalance": rebalance, "entry_lag": entry_lag, "cost_bps": cost_bps,
                   "risk": None if risk is None else risk.__dict__},
    }


def factor_ic_series(factors: dict[str, pd.DataFrame], panel, *, h: int = 5,
                     entry_lag: int = 2) -> pd.DataFrame:
    """每个因子的**逐日** RankIC，用于在线判断"这个因子最近还行不行"。

    一行的 IC 要到 ``entry_lag + h`` 天之后才算得出来（前瞻收益那时才走完），所以
    调用方必须按这个天数做禁运，否则就是拿还没发生的结果挑因子。禁运由
    :func:`walk_forward` 负责施加，这里只负责算。
    """
    target = build_targets(panel, h, entry_lag)["targets"]["raw_rtf"]
    return pd.DataFrame({
        name: daily_cross_sectional(frame, target, panel.mask)["rank_ic"]
        for name, frame in factors.items()})


def walk_forward(factors: dict[str, pd.DataFrame], panel, *, lookback: int = 252,
                 step: int = 21, top_k: int | None = None, min_ic: float = 0.0,
                 resign: bool = True, select: bool = True, start="2022-04-01",
                 rebalance: int = 5, entry_lag: int = 2, cost_bps: float = 6.5,
                 sizing: str = "equal_notional", cap: float = 0.35, gross: float = 1.0,
                 neutral: bool = True, risk: Risk | None = None,
                 vol_window: int = 20) -> dict:
    """滚动向前：每 ``step`` 天，只用**过去** ``lookback`` 天的评估结果重选因子。

    这是"先找能通过矩阵的因子，再构建"的在线版本。每个决策点做三件事，全部只看过去：

    1. 取过去 ``lookback`` 天里**已经可以核算**的逐日 RankIC（按 ``entry_lag + h``
       禁运，尚未走完前瞻期的日期一律不算）；
    2. 按平均 IC 定方向，并筛掉 ``|IC|`` 达不到 ``min_ic`` 的因子（或只留 ``top_k`` 个）；
    3. 用选中的因子合成、定仓，交易接下来的 ``step`` 天。

    与一次性在开发段选参数的关键差别：**没有任何一个决策用到了它交易的那段时间的
    信息**，所以得到的不是一条曲线，而是几十个互不重叠的向前区间 —— 可以直接问
    "有多大比例的区间达到了要求"，而不是"这一条曲线好不好看"。

    没有任何因子入选的区间**空仓**，不硬凑一个。这在评估里是诚实的默认：宁可不交易，
    也不拿一个当期没有证据支持的因子上场。
    """
    if lookback < 60 or step < 1:
        raise ValueError("lookback must be >= 60 days and step >= 1 day")
    ic = factor_ic_series(factors, panel, h=rebalance, entry_lag=entry_lag)
    embargo = entry_lag + rebalance          # 这么多天之内的 IC 还没算完
    # ``resign=False``:方向只在第一个回看窗口定一次,之后永不翻转。滚动重定方向看着
    # 更"自适应",实测却是净损失 —— 过去 IC 的符号本身不稳定,于是仓位被反复打脸。
    # ``select=False``:全部因子都用,只保留构建层的仓位处理。
    first_sign = None
    index = panel.index
    blocks, selections = [], []
    weights = pd.DataFrame(np.nan, index=index, columns=panel.symbols)

    first = lookback
    if start is not None:
        # 预热期的截面只有两三个币,在那里定出来的方向和入选名单没有意义;
        # 若又恰好把方向冻住,这个错误会一直带到后面。第一个决策点必须落在评测期内。
        bound = pd.Timestamp(start, tz=index.tz)
        first = max(lookback, int(np.searchsorted(index.values, bound.to_datetime64())))
    for start_i in range(first, len(index), step):
        decide_at = index[start_i]
        usable = ic.loc[:index[start_i - 1]].iloc[:-embargo] if embargo else ic.loc[:index[start_i - 1]]
        window = usable.iloc[-lookback:]
        means = window.mean()
        if first_sign is None:
            first_sign = {n: (-1.0 if np.isfinite(v) and v < 0 else 1.0) for n, v in means.items()}
        chosen = {}
        for name, value in means.items():
            if select and (not np.isfinite(value) or abs(value) < min_ic):
                continue
            chosen[name] = (-1.0 if value < 0 else 1.0) if resign else first_sign[name]
        if top_k is not None and select and chosen:
            ranked = sorted(chosen, key=lambda n: -abs(means[n]))[:top_k]
            chosen = {n: chosen[n] for n in ranked}
        stop = min(start_i + step, len(index))
        selections.append({"decide_at": decide_at, "n_selected": len(chosen),
                           "selected": sorted(chosen),
                           "mean_ic": {n: float(means[n]) for n in chosen}})
        if not chosen:
            continue                      # 没有证据支持的区间就空仓
        score = combine({n: factors[n] * s for n, s in chosen.items()}, panel.mask)
        block = target_weights(score, panel, sizing=sizing, vol_window=vol_window, cap=cap,
                               gross=gross, neutral=neutral, rebalance=rebalance)
        weights.iloc[start_i:stop] = block.iloc[start_i:stop]
        blocks.append((index[start_i], index[stop - 1]))

    return {"weights": weights, "selections": selections, "blocks": blocks,
            "book": simulate(weights, panel, entry_lag=entry_lag, cost_bps=cost_bps, risk=risk),
            "config": {"lookback": lookback, "step": step, "top_k": top_k, "min_ic": min_ic,
                       "rebalance": rebalance, "sizing": sizing, "cap": cap, "gross": gross,
                       "embargo_days": embargo, "resign": resign, "select": select,
                       "start": str(start)}}
