"""参数与决策逻辑的选择：把"调参"变成一件能被统计检验的事。

调参是这套框架里**最容易把钱调没**的一步。同一批因子、同一副骨架，只要在全样本上
扫一遍网格取最高的那格，年化可以从 25% "提升"到 80%，而这 80% 有很大一部分是选择
偏差本身 —— 在 N 次无技能的试验里取最大值，期望值随 ``sqrt(2 ln N)`` 增长，与有没有
alpha 无关。所以这个模块的目标不是"找到最好的参数"，是**在承认自己搜过多少次的前提下
挑一个能带出样本的参数**。

三件事，缺一不可：

1. **滚动前推（walk-forward）**：在训练窗上选参数，只在其后未见过的窗上记账，向前滑动，
   把所有 OOS 段拼成一条曲线。这条拼出来的曲线才是"如果我一直这么干"的答案。
2. **选平台而不是选尖峰**：不取原始网格的 argmax，取**邻域平滑后**曲面的 argmax。
   尖峰是"一次幸运对齐"，平台才是结构。注意平滑曲面的几何必须**锚在平滑曲面自己的
   最优点**上评估 —— 锚在原始 argmax 上会得到相反的结论。
3. **记账（trial ledger）**：每一个试过的参数点、它的逐日收益，全部留档。没有这本账，
   DSR 和 PBO 都算不出来 —— :mod:`cyqnt_trd.eval.diagnostics` 里那句
   "no DSR/PBO without a complete trial ledger" 说的就是这件事。有了账，
   :func:`effective_trials` / :func:`deflated_sharpe` / :func:`pbo` 才有输入。

**为什么可以先全量跑一遍再切片。** 蓝图内部没有拟合状态：``compile_targets`` 是面板的
纯函数，参数由外部给定。因此一个候选的逐日收益序列与"它被拿去哪个窗口评分"无关，
全量跑一次、按窗口切片评分，与在每个窗口上重跑等价（差别仅在切片首日的持仓继承，
这恰好也是真实情形）。这让 N×W 次回测塌缩成 N 次。

**这个模块不保证赚钱。** 它保证的是：你报出来的那个数字，已经为"你搜了多少次"付过账。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .blueprint import Blueprint, run_blueprint
from .portfolio import Risk, simulate

__all__ = ["Ledger", "WalkForward", "build_ledger", "grid_points", "smooth_scores",
           "select_trial", "select_trials", "ensemble_targets", "walk_forward",
           "effective_trials", "expected_max_sharpe", "deflated_sharpe", "pbo",
           "sensitivity", "score_returns", "tune"]

_DAYS = 365.0
_MIN_DAYS = 30            # 少于这么多天不给分：短样本上的 Sharpe 基本是噪声
_EULER = 0.5772156649015329


# --------------------------------------------------------------------- 目标函数
def _equity(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0.0)).cumprod()


def score_returns(r: pd.Series, objective: str = "sharpe") -> float:
    """在一段收益序列上打分。**只用这段**，不看别处。

    ``sharpe`` 是默认：研究文献一致的建议是在窗口内优化风险调整后收益而不是 PnL，
    因为按 PnL 选出来的参数是在挑一段特定的行情序列。``calmar`` 直接对应"回撤要低"，
    但它由单一极值决定，样本内极不稳定，当目标时务必配合平台选择使用。
    """
    r = r.dropna()
    if len(r) < _MIN_DAYS:
        return np.nan
    sd = float(r.std())
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    if objective == "sharpe":
        return float(r.mean() / sd * np.sqrt(_DAYS))
    equity = _equity(r)
    total = float(equity.iloc[-1])
    if total <= 0:
        return np.nan
    ann = total ** (_DAYS / len(r)) - 1.0
    if objective == "ann_return":
        return float(ann)
    if objective == "calmar":
        dd = float((equity / equity.cummax() - 1).min())
        return float(ann / abs(dd)) if dd < 0 else np.nan
    raise ValueError(f"unknown objective {objective!r}")


# ----------------------------------------------------------------------- 网格
def grid_points(axes: Mapping[str, Sequence]) -> list[dict]:
    """参数轴 → 笛卡尔积。顺序固定，下游用下标当 trial id。"""
    names = list(axes)
    return [dict(zip(names, combo)) for combo in itertools.product(*(axes[n] for n in names))]


@dataclass
class Ledger:
    """**试验账本**：搜过的每一个参数点，以及它的逐日净收益。

    这是本模块所有统计量的唯一输入。报告一个回测数字却不交出这本账，等于报告
    "最好的一次"而隐瞒试了多少次。
    """

    axes: dict[str, tuple]
    points: list[dict]
    returns: pd.DataFrame                 # 日期 × trial id
    failures: dict[int, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.points)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(v) for v in self.axes.values())

    def scores(self, objective: str = "sharpe", lo=None, hi=None) -> pd.Series:
        """在 ``[lo, hi)`` 这段日期上给每个候选打分。"""
        window = self.returns
        if lo is not None:
            window = window.loc[window.index >= lo]
        if hi is not None:
            window = window.loc[window.index < hi]
        return pd.Series({i: score_returns(window[i], objective) for i in window.columns},
                         dtype=float)

    def frame(self, objective: str = "sharpe") -> pd.DataFrame:
        """参数 + 全样本得分，便于人眼看曲面。**不要拿这张表直接选参数。**"""
        out = pd.DataFrame(self.points)
        out["score"] = self.scores(objective).to_numpy()
        return out


def build_ledger(build: Callable[..., Blueprint], axes: Mapping[str, Sequence], panel, *,
                 start="2022-04-01", entry_lag: int = 2, cost_bps: float = 6.5,
                 risk: Risk | None = None) -> Ledger:
    """把整个网格跑一遍，留下账本。

    ``build`` 接收一组关键字参数（轴名 → 取值），返回**一张 Blueprint 或一张目标权重表**。
    后者让整条链路（评测 → 准入 → 合成 → 定仓）也能被同一套调参机制检验，而不是只有
    蓝图那一侧可调 —— 用 :mod:`cyqnt_trd.eval.pipeline` 时把 ``build`` 写成返回
    ``strategy.weights`` 即可。

    某个参数点跑崩了不会中断整体搜索，但**会被记进 ``failures``** —— 悄悄跳过失败的
    候选会让试验计数偏小，DSR 就被高估了。
    """
    axes = {k: tuple(v) for k, v in axes.items()}
    points = grid_points(axes)
    if not points:
        raise ValueError("empty grid")
    columns, failures = {}, {}
    for i, point in enumerate(points):
        try:
            made = build(**point)
            if isinstance(made, pd.DataFrame):          # 已经是目标权重表
                columns[i] = simulate(made, panel, entry_lag=entry_lag,
                                      cost_bps=cost_bps, risk=risk).returns
            else:
                columns[i] = run_blueprint(made, panel, entry_lag=entry_lag,
                                           cost_bps=cost_bps, risk=risk,
                                           start=start)["book"].returns
        except Exception as exc:                                   # noqa: BLE001
            failures[i] = f"{type(exc).__name__}: {exc}"
            columns[i] = pd.Series(np.nan, index=panel.index)
    returns = pd.DataFrame(columns).reindex(columns=range(len(points)))
    if start is not None:
        # 预热期不交易，那段收益恒为 0。留着会让"训练窗"里塞满空仓的日子，选参数时
        # 等于用一段没发生过的历史投票。
        returns = returns.loc[returns.index >= pd.Timestamp(start, tz=panel.index.tz)]
    return Ledger(axes=dict(axes), points=points, returns=returns, failures=failures)


# ------------------------------------------------------------------- 平台选择
def _neighbours(shape: tuple[int, ...]) -> list[list[int]]:
    """网格上每个点的邻居（每根轴上 ±1，含自身）。

    只取轴向邻居而不含对角，是因为维度一高对角邻居数量爆炸，会把邻域均值摊平成全局均值，
    平台和尖峰就分不出来了。
    """
    n = int(np.prod(shape)) if shape else 0
    out = []
    for flat in range(n):
        idx = np.unravel_index(flat, shape)
        group = [flat]
        for axis, pos in enumerate(idx):
            for step in (-1, 1):
                nxt = pos + step
                if 0 <= nxt < shape[axis]:
                    moved = list(idx)
                    moved[axis] = nxt
                    group.append(int(np.ravel_multi_index(tuple(moved), shape)))
        out.append(group)
    return out


def smooth_scores(scores: pd.Series, ledger: Ledger, *, min_fraction: float = 0.5) -> pd.Series:
    """邻域平均，得到去噪曲面。

    孤立的高分格子会被邻居拉下来，成片的高分区域则保持。邻居里有效值不足
    ``min_fraction`` 的格子记为缺失 —— 网格边角本来就没几个邻居，硬给它一个分数等于
    偏袒边界。
    """
    values = scores.reindex(range(len(ledger))).to_numpy(dtype=float)
    out = np.full(len(ledger), np.nan)
    for flat, group in enumerate(_neighbours(ledger.shape)):
        vals = values[group]
        good = np.isfinite(vals)
        if good.sum() >= max(1, int(np.ceil(min_fraction * len(group)))):
            out[flat] = vals[good].mean()
    return pd.Series(out, index=range(len(ledger)), dtype=float)


def select_trial(scores: pd.Series, ledger: Ledger, *, rule: str = "plateau") -> int | None:
    """按规则挑一个候选。``plateau`` 选去噪曲面的最优点，``peak`` 选原始 argmax。

    受控实验的结论很明确：作为**选择规则**，平台优于尖峰（二维网格上 OOS Sharpe 平均
    +0.31），而且参数越多优势越大；但平台的几何形状**不是**一个合格的过拟合检验 ——
    该由 DSR / PBO 来判。所以这里只把它当选择器用。
    """
    if rule not in ("plateau", "peak"):
        raise ValueError(f"unknown rule {rule!r}")
    surface = (smooth_scores(scores, ledger) if rule == "plateau" else scores).dropna()
    if surface.empty:
        return None
    return int(surface.idxmax())


def select_trials(scores: pd.Series, ledger: Ledger, *, rule: str = "plateau",
                  top: int = 1) -> list[int]:
    """选前 ``top`` 个候选。``top>1`` 时下游不是"选一个参数"，而是**集成一片参数**。

    集成是这套流程里唯一一个不用押注"哪一格是对的"的动作：把整片高分区域一起持有，
    选择偏差随之消失（没有选择就没有偏差），代价是拿不到最好那格的收益。实测在本面板上
    这个代价是负的 —— 集成反而更好，因为"最好那格"本来就选不准。
    """
    if rule not in ("plateau", "peak"):
        raise ValueError(f"unknown rule {rule!r}")
    surface = (smooth_scores(scores, ledger) if rule == "plateau" else scores).dropna()
    if surface.empty:
        return []
    return [int(i) for i in surface.sort_values(ascending=False).index[:max(1, top)]]


def ensemble_targets(build: Callable[..., Blueprint], points: Sequence[Mapping], panel, *,
                     start="2022-04-01") -> pd.DataFrame:
    """把一组参数点的目标权重平均成**一张**目标表。

    这是集成的**可部署**写法：先平均目标、再模拟一次，子策略之间互相抵消的单子自然轧掉，
    只交净额。如果改成平均各自的收益曲线，等于给每个子策略单独开一个账户各交各的 ——
    结论方向一样，但白付一笔换手成本。

    平均之前每个候选先 ``ffill``：蓝图里"非评估日留空"表示继续持有，摊平成显式权重之后
    才能和别的候选对齐相加。
    """
    from .blueprint import blueprint_targets           # 局部导入避免循环依赖

    def one(point):
        made = build(**dict(point))
        return (made if isinstance(made, pd.DataFrame)
                else blueprint_targets(made, panel, start=start)).ffill()

    stack = [one(p) for p in points]
    if not stack:
        raise ValueError("empty ensemble")
    values = np.stack([t.to_numpy(dtype=float) for t in stack])
    good = np.isfinite(values)
    count = good.sum(axis=0)
    total = np.where(good, values, 0.0).sum(axis=0)
    mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return pd.DataFrame(mean, index=stack[0].index, columns=stack[0].columns)


def sensitivity(scores: pd.Series, ledger: Ledger, trial: int) -> dict:
    """一个候选点的稳健性体检：点分 vs 邻域均值 vs 邻域离散。

    点分远高于邻域均值 = 尖峰，扔掉。邻域均值高且离散小 = 平台。邻域均值低且离散小是
    "平的坏区"，不是平台 —— 这两者最容易混。
    """
    values = scores.reindex(range(len(ledger))).to_numpy(dtype=float)
    group = _neighbours(ledger.shape)[trial]
    vals = values[group][np.isfinite(values[group])]
    point = values[trial]
    mean = float(vals.mean()) if len(vals) else np.nan
    return {"params": ledger.points[trial], "point": float(point), "neighbour_mean": mean,
            "neighbour_std": float(vals.std()) if len(vals) > 1 else np.nan,
            "neighbours": int(len(vals)),
            "gap": float(point - mean) if np.isfinite(point) and np.isfinite(mean) else np.nan}


# ------------------------------------------------------------------- 滚动前推
@dataclass
class WalkForward:
    """滚动前推的结果。``oos`` 是拼出来的那条**唯一可信**的曲线。"""

    windows: pd.DataFrame
    oos: pd.Series
    rule: str
    objective: str

    @property
    def wfer(self) -> float:
        """OOS 年化 / IS 年化。>0.5 可接受，<0.3 基本是拟合了噪声，<0 就是反的。"""
        is_ann = self.windows["is_ann_return"].mean()
        oos_ann = self.windows["oos_ann_return"].mean()
        if not np.isfinite(is_ann) or is_ann <= 0:
            return np.nan
        return float(oos_ann / is_ann)

    @property
    def stability(self) -> float:
        """选出来的参数有多不稳：窗口间参数取值的平均变动比例。0 = 每次都选同一组。"""
        picks = self.windows["params"].tolist()
        if len(picks) < 2:
            return np.nan
        changes = [sum(a[k] != b[k] for k in a) / len(a) for a, b in zip(picks, picks[1:])]
        return float(np.mean(changes))

    def summary(self) -> str:
        return (f"[{self.rule}] 窗口 {len(self.windows)}  "
                f"OOS 年化 {score_returns(self.oos, 'ann_return'):+.1%}  "
                f"Sharpe {score_returns(self.oos, 'sharpe'):.2f}  "
                f"Calmar {score_returns(self.oos, 'calmar'):.2f}  "
                f"WFER {self.wfer:.2f}  参数变动 {self.stability:.2f}")


def walk_forward(ledger: Ledger, *, train: int = 504, test: int = 126, embargo: int = 10,
                 rule: str = "plateau", objective: str = "sharpe", top: int = 1) -> WalkForward:
    """滚动前推：在训练窗选参数，在其后的测试窗记账，向前滑。

    ``embargo`` 是训练窗末尾与测试窗开头之间扔掉的天数。信号用的是滚动窗口、成交又滞后
    ``entry_lag`` 天，紧挨着的那几天两边都沾，不隔开就是把答案从缝里递过去。

    训练窗默认 504 天（约两年）、测试窗 126 天（约四个月）—— 测试约占训练的 25%，落在
    常见的 20%~33% 区间；加密市场换挡快，用滚动窗而非锚定窗（不把最早的行情一直背着）。
    """
    index = ledger.returns.index
    rows, pieces = [], []
    start = 0
    while start + train + embargo + test <= len(index):
        train_lo, train_hi = index[start], index[start + train]
        test_lo = index[start + train + embargo]
        test_hi = index[start + train + embargo + test]

        is_scores = ledger.scores(objective, train_lo, train_hi)
        picks = select_trials(is_scores, ledger, rule=rule, top=top)
        if picks:
            chosen = ledger.returns[picks].mean(axis=1)
            oos = chosen.loc[test_lo:test_hi].iloc[:-1]
            pieces.append(oos)
            rows.append({
                "train_start": train_lo, "train_end": train_hi,
                "test_start": test_lo, "test_end": test_hi,
                "trial": picks[0], "n_picked": len(picks),
                "params": ledger.points[picks[0]],
                "is_score": float(is_scores.get(picks[0], np.nan)),
                "is_ann_return": score_returns(chosen.loc[train_lo:train_hi], "ann_return"),
                "oos_score": score_returns(oos, objective),
                "oos_ann_return": score_returns(oos, "ann_return"),
            })
        start += test
    if not rows:
        raise ValueError("样本长度不足以切出一个完整的训练+测试窗")
    stitched = pd.concat(pieces)
    return WalkForward(windows=pd.DataFrame(rows), oos=stitched[~stitched.index.duplicated()],
                       rule=rule, objective=objective)


# ------------------------------------------------------------- 选择偏差的统计
def tune(build: Callable[..., Blueprint], axes: Mapping[str, Sequence], panel, *,
         objective: str = "sharpe", train: int = 504, test: int = 126, embargo: int = 10,
         start="2022-04-01", entry_lag: int = 2, cost_bps: float = 6.5,
         risk: Risk | None = None, n_splits: int = 10) -> dict:
    """完整的"调参"流程，一次调用。

    它跑三条路线并放在一起比：

    ``pick``
        滚动前推 + 平台选择，每个窗口挑**一组**参数。这是"调参"的常规做法。
    ``ensemble``
        整片网格净额集成，**不做任何选择**。没有选择就没有选择偏差。
    ``stats``
        有效试验数、DSR（对着"无技能搜这么多次能搜出多少"这个基准）、PBO（"挑最好的"
        这个动作本身泛化吗）。

    返回的 ``verdict`` 只陈述事实：谁在 OOS 段上更好、PBO 有没有超过 0.5。它**不**声称
    集成一定更好 —— 在别的标的池上可能不是。它声称的是：你现在有证据分辨这两条路。
    """
    ledger = build_ledger(build, axes, panel, start=start, entry_lag=entry_lag,
                          cost_bps=cost_bps, risk=risk)
    picked = walk_forward(ledger, train=train, test=test, embargo=embargo,
                          rule="plateau", objective=objective, top=1)
    book = simulate(ensemble_targets(build, ledger.points, panel, start=start), panel,
                    entry_lag=entry_lag, cost_bps=cost_bps, risk=risk)

    span = picked.oos.index                       # 两条路线在**同一段**日期上比
    ens_oos = book.returns.reindex(span)
    n_eff = effective_trials(ledger.returns)
    sigma = ledger.scores(objective).dropna().std() / np.sqrt(_DAYS)
    stats_ = {
        "n_trials": len(ledger), "n_failed": len(ledger.failures),
        "effective_trials": n_eff,
        "dsr_ensemble": deflated_sharpe(ens_oos, n_trials=n_eff, sigma_trials=sigma)["dsr"],
        "dsr_picked": deflated_sharpe(picked.oos, n_trials=n_eff, sigma_trials=sigma)["dsr"],
        "pbo_peak": pbo(ledger, n_splits=n_splits, objective=objective, rule="peak")["pbo"],
        "pbo_plateau": pbo(ledger, n_splits=n_splits, objective=objective, rule="plateau")["pbo"],
    }
    ens_score, pick_score = score_returns(ens_oos, objective), score_returns(picked.oos, objective)
    verdict = {
        "oos_start": span[0], "oos_end": span[-1],
        "ensemble_beats_picking": bool(np.isfinite(ens_score) and np.isfinite(pick_score)
                                       and ens_score > pick_score),
        "selection_generalises": bool(np.isfinite(stats_["pbo_peak"])
                                      and stats_["pbo_peak"] < 0.5),
    }
    return {"ledger": ledger, "picked": picked, "book": book, "ensemble_oos": ens_oos,
            "stats": stats_, "verdict": verdict}


def effective_trials(returns: pd.DataFrame) -> float:
    """**有效**试验数：相关矩阵特征值的参与率 ``(Σλ)² / Σλ²``。

    网格上相邻的两个参数几乎产出同一条收益曲线，把它们当两次独立试验会把搜索宽度说得
    太大。参与率给出"这堆试验里其实有几注独立的赌"。实测名义与有效常差一到两个数量级。
    """
    clean = returns.dropna(axis=1, how="all").fillna(0.0)
    if clean.shape[1] <= 1:
        return float(clean.shape[1])
    corr = np.corrcoef(clean.to_numpy().T)
    corr = np.nan_to_num(corr, nan=0.0)
    lam = np.linalg.eigvalsh(corr)
    lam = lam[lam > 0]
    if lam.size == 0:
        return 1.0
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def expected_max_sharpe(n_trials: float, sigma: float) -> float:
    """N 次**无技能**试验里最大 Sharpe 的期望（同一时间口径下的 ``sigma``）。

    这是 DSR 的比较基准：不是和 0 比，是和"什么都不会的人搜这么多次能搜出多少"比。
    """
    n = max(float(n_trials), 2.0)
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(sigma * ((1.0 - _EULER) * z1 + _EULER * z2))


def deflated_sharpe(returns: pd.Series, *, n_trials: float, sigma_trials: float) -> dict:
    """紧缩 Sharpe：扣掉试验次数、偏度、峰度和样本长度之后，真实 Sharpe > 基准的概率。

    返回的是**概率**不是 Sharpe。常用门槛 0.95；0.6 这种数字读作"接近抛硬币"。
    ``sigma_trials`` 是账本里各候选 Sharpe 的离散度，与 ``returns`` 用同一时间口径
    （本函数内部一律用日频）。
    """
    r = returns.dropna()
    if len(r) < _MIN_DAYS or r.std() <= 0:
        return {"dsr": np.nan, "sharpe": np.nan, "benchmark": np.nan, "n_trials": n_trials}
    sr = float(r.mean() / r.std())                       # 日频口径
    sr0 = expected_max_sharpe(n_trials, sigma_trials)
    skew, kurt = float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 0:
        return {"dsr": np.nan, "sharpe": sr * np.sqrt(_DAYS), "benchmark": sr0 * np.sqrt(_DAYS),
                "n_trials": n_trials}
    z = (sr - sr0) * np.sqrt(len(r) - 1) / np.sqrt(denom)
    return {"dsr": float(stats.norm.cdf(z)), "sharpe": sr * np.sqrt(_DAYS),
            "benchmark": sr0 * np.sqrt(_DAYS), "n_trials": float(n_trials)}


def pbo(ledger: Ledger, *, n_splits: int = 10, objective: str = "sharpe",
        rule: str = "peak") -> dict:
    """过拟合概率（CSCV）：样本内选出来的冠军，在样本外落到中位数以下的频率。

    把时间切成 ``n_splits`` 块，穷举一半作训练、另一半作测试的所有组合，每次在训练集
    上按 ``rule`` 选一个候选，看它在测试集里排第几。<0.25 健康，>0.5 意味着"挑最好的"
    这个动作本身不泛化。

    它检验的是**选择程序**，与 DSR 检验"这个数字本身"互补，两个都要看。
    """
    if n_splits % 2:
        raise ValueError("n_splits must be even")
    blocks = np.array_split(np.arange(len(ledger.returns)), n_splits)
    ranks, logits = [], []
    for combo in itertools.combinations(range(n_splits), n_splits // 2):
        is_rows = np.concatenate([blocks[b] for b in combo])
        oos_rows = np.concatenate([blocks[b] for b in range(n_splits) if b not in combo])
        is_scores = pd.Series(
            {i: score_returns(ledger.returns[i].iloc[is_rows], objective)
             for i in ledger.returns.columns}, dtype=float)
        pick = select_trial(is_scores, ledger, rule=rule)
        if pick is None:
            continue
        oos_scores = pd.Series(
            {i: score_returns(ledger.returns[i].iloc[oos_rows], objective)
             for i in ledger.returns.columns}, dtype=float).dropna()
        if pick not in oos_scores.index or len(oos_scores) < 2:
            continue
        omega = float(oos_scores.rank().loc[pick] / (len(oos_scores) + 1))
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        ranks.append(omega)
        logits.append(np.log(omega / (1 - omega)))
    if not logits:
        return {"pbo": np.nan, "n_combinations": 0, "median_rank": np.nan}
    return {"pbo": float(np.mean(np.asarray(logits) <= 0)),
            "n_combinations": len(logits), "median_rank": float(np.median(ranks))}
