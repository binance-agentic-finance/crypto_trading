"""组合模拟器：目标权重 + 现金，变了就交易差额。

和评测引擎的核算**刻意不同**，两者回答的不是一个问题：

===================== ================================== ==================================
                      评测引擎 ``engine.evaluate_factor``   本模块 ``simulate``
===================== ================================== ==================================
持仓模型               非重叠完整周期，每周期**全平全开**      **连续持仓**，调仓只交易差额
换手                   每周期 = 入场名义 + 出场名义 ≈ 2.0    Σ\\|Δ名义\\| / 权益
资金                   每周期固定一单位，不复利               权益复利
用途                   因子有没有信息、扣成本还剩什么           这个策略实际赚不赚钱
===================== ================================== ==================================

评测那套对**因子**是对的：固定一单位、全平全开，才能让不同因子在同一条件下比较。但拿它
当策略损益会系统性高估成本 —— 真实策略持仓是连着的，相邻两期重叠的部分根本不用动。

所以策略这一层自己做净额核算。**基准（裸因子直接买卖）也走同一个模拟器**，否则比较不
公平；两边唯一的差别只有权重怎么算出来。

权重语义（对应"现金 + 标的"的理解）：

* ``w`` 是**占权益的比例**，现金 = ``1 - Σw``；
* 截面中性组合 ``Σw = 0``，多空对冲，现金托底；
* 单标的方向性组合 ``w ∈ [-1, 1]``，就是现金和该标的的二元组合；
* 两者走同一条代码，区别只在权重向量本身。

没有建模的东西，别当它不存在：盘口冲击、部分成交、点差（成本是一个固定 bps）、保证金
与借币成本。权益跌到 0 即停止交易（见 :func:`simulate`），这只是防止出现负权益这种无
意义读数，**不是**一个强平模型 —— 真实杠杆账户会在权益归零**之前**就被强平。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["simulate", "Book", "Risk", "performance"]

_DAYS = 365.0


@dataclass
class Risk:
    """组合层风控。三项都只看**过去**，因此可以在模拟推进时在线判断。

    ``target_vol`` 把整体杠杆按"过去实现波动"反向缩放：波动低就放大、高就收缩，目标
    是让曲线的波动稳定在一个水平。它**不改变持仓的相对结构**，只动总杠杆。

    ``drawdown_limit`` 是一个粗糙但有效的尾部保护：回撤超过阈值就把杠杆乘上
    ``drawdown_scale``，等回撤修复再恢复。它会在反弹初期少赚 —— 这是它的代价，不是
    bug。
    """

    target_vol: float | None = None      # 年化目标波动，例如 0.20
    vol_window: int = 60                 # 用过去 N 日组合收益估计已实现波动
    min_days: int = 20                   # 不足这么多天不缩放（没有估计就不假装有）
    max_leverage: float = 2.0            # 缩放上限，防止低波期把杠杆放飞
    drawdown_limit: float | None = None  # 例如 -0.15
    drawdown_scale: float = 0.5          # 触发后乘这个系数

    # 基于**自身近期表现**的不对称杠杆:赚钱时加、亏钱时减。
    #
    # 与 drawdown_limit 的区别很要紧。距历史峰值的回撤在加密因子组合上几乎恒为真
    # (实测这类组合只有约 4% 的交易日创新高、中位回撤 -29%),所以那条规则会退化成一个
    # 静态的半仓,不是保护。用"过去 N 日自己赚没赚"做状态判断,才是可切换的。
    #
    # 它成立的前提是组合损益本身有惯性,这一点必须先量:实测 20 日窗口上
    # corr(过去, 未来) ≈ +0.28,过去为正后未来 20 日均值 +0.24%、为负后 -1.90%。
    # 换个标的池或换个时期,这个前提可能不成立 —— 那时这套杠杆就只是加噪声。
    # 状态判断一律读**未加杠杆的影子组合**,不读加了杠杆之后的自己。这不是细节:
    # 若用自身收益判断,而杠杆又可以降到 0,组合就进入吸收态 —— 空仓之后自身收益恒为 0,
    # "我在赚钱吗"永远为假,再也回不来。影子组合始终满仓运行,只用于产生状态信号。
    perf_window: int | None = None       # 用过去 N 日影子组合收益判断盈亏状态
    perf_up: float = 1.0                 # 盈利状态下的杠杆
    perf_down: float = 1.0               # 亏损状态下的杠杆


@dataclass
class Book:
    """一次模拟的完整结果。"""

    equity: pd.Series                    # 逐日权益（以执行价计）
    returns: pd.Series                   # 逐日净收益率
    weights: pd.DataFrame                # 逐日**实际**持仓权重（随价格漂移）
    fills: pd.DataFrame                  # 逐日逐标的成交额 / 权益（正买负卖）
    trades: pd.Series                    # 逐日成交名义 / 权益
    costs: pd.Series                     # 逐日交易成本（占权益）
    funding: pd.Series                   # 逐日资金费（占权益，正为支出）
    leverage: pd.Series | None = None    # 风控给出的缩放系数（无风控时恒为 1）
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        return (f"年化 {m['ann_return']:+.1%}  波动 {m['ann_vol']:.1%}  "
                f"Sharpe {m['sharpe']:.2f}  最大回撤 {m['max_drawdown']:.1%}  "
                f"Calmar {m['calmar']:.2f}  日均换手 {m['daily_turnover']:.3f}")


def performance(returns: pd.Series, equity: pd.Series, trades: pd.Series) -> dict:
    """标准绩效口径。收益是**净**的（已扣交易成本与资金费）。"""
    r = returns.dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("ann_return", "ann_vol", "sharpe", "max_drawdown",
                                    "calmar", "daily_turnover", "hit_rate", "n_days")}
    ann_vol = float(r.std() * np.sqrt(_DAYS))
    # 几何年化：用权益首尾，避免算术平均在高波动下虚高
    years = len(r) / _DAYS
    total = float(equity.dropna().iloc[-1] / equity.dropna().iloc[0])
    ann_return = total ** (1 / years) - 1 if years > 0 and total > 0 else np.nan
    peak = equity.cummax()
    drawdown = float((equity / peak - 1).min())
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": float(r.mean() / r.std() * np.sqrt(_DAYS)) if r.std() > 0 else np.nan,
        "max_drawdown": drawdown,
        "calmar": ann_return / abs(drawdown) if drawdown < 0 else np.nan,
        "daily_turnover": float(trades.reindex(r.index).fillna(0).mean()),
        "hit_rate": float((r > 0).mean()),
        "n_days": int(len(r)),
    }


def simulate(targets: pd.DataFrame, panel, *, entry_lag: int = 2, cost_bps: float = 6.5,
             initial_equity: float = 1.0, risk: "Risk | None" = None) -> Book:
    """按目标权重逐日模拟，只交易差额。

    参数
    ----
    targets
        ``日期 × 标的`` 的目标权重，占权益比例。**整行全是 NaN 表示当日不调仓**
        （继续持有），否则按该行调仓；NaN 单元按 0 处理（不持有该标的）。
        行的日期是**决策日**，实际成交在 ``open[决策日 + entry_lag]``。
    panel
        提供 ``open`` 与 ``funding``。资金费按每日、每单位基础币计，正为支出。

    risk
        组合层风控，见 :class:`Risk`。只读过去的权益路径，因此是时点安全的。

    与评测引擎共用的约定：决策日收盘出信号 → 等满 ``entry_lag`` 天 → 在开盘价成交。
    """
    if not isinstance(targets, pd.DataFrame):
        raise TypeError("targets must be a date × symbol DataFrame of portfolio weights")
    targets = targets.reindex(index=panel.index, columns=panel.symbols)
    opens = panel.open.to_numpy(dtype=float)
    funding = panel.funding.to_numpy(dtype=float)
    target_values = targets.to_numpy(dtype=float)
    rebalance = ~np.isnan(target_values).all(axis=1)
    n_days, n_assets = opens.shape

    units = np.zeros(n_assets)
    last_price = np.full(n_assets, np.nan)   # 停牌/缺价时的估值依据
    cash = float(initial_equity)
    equity = np.full(n_days, np.nan)
    daily_return = np.full(n_days, np.nan)
    ref_units = np.zeros(n_assets)       # 影子组合:恒定 scale=1,只为产生状态信号
    ref_cash = float(initial_equity)
    ref_return = np.full(n_days, np.nan)
    ref_previous = np.nan
    daily_trade = np.zeros(n_days)
    daily_cost = np.zeros(n_days)
    daily_funding = np.zeros(n_days)
    weights_out = np.full((n_days, n_assets), np.nan)
    # 逐标的成交额 / 权益。订单清单从这里出，与损益共用同一笔账 —— 另算一遍
    # 就会出现"报表上的单子"和"回测真交的单子"对不上。
    fills_out = np.zeros((n_days, n_assets))
    leverage_out = np.ones(n_days)
    previous_equity = np.nan
    peak_equity = -np.inf

    for i in range(n_days):
        price = opens[i]
        tradable = np.isfinite(price) & (price > 0)
        last_price = np.where(tradable, price, last_price)
        # 持有但当日无价:按最后一个有效价估值。直接记 0 等于让仓位凭空消失,
        # 那会把一次停牌伪装成一笔盈利或亏损。
        mark = np.where(np.isfinite(last_price), last_price, 0.0)
        value = cash + float((units * mark).sum())

        # 资金费按当日持有的单位计
        pay = float(np.nansum(np.where(np.isfinite(funding[i]), units * funding[i], 0.0)))
        cash -= pay
        value -= pay
        daily_funding[i] = pay / value if value > 0 else np.nan

        # 影子组合先走一步:同样的目标权重、恒定满仓,提供不受杠杆污染的状态信号
        ref_pay = float(np.nansum(np.where(np.isfinite(funding[i]), ref_units * funding[i], 0.0)))
        ref_cash -= ref_pay
        ref_value = ref_cash + float((ref_units * mark).sum())
        if i >= entry_lag and rebalance[i - entry_lag] and ref_value > 0:
            ref_want = np.where(tradable, np.nan_to_num(target_values[i - entry_lag], nan=0.0), 0.0)
            ref_target = ref_want * ref_value
            ref_delta = ref_target - np.where(tradable, ref_units * price, 0.0)
            ref_cost = float(np.abs(ref_delta).sum()) * cost_bps / 1e4
            ref_new = np.where(tradable, ref_target / np.where(tradable, price, 1.0), ref_units)
            ref_cash -= float((ref_new - ref_units)[tradable] @ price[tradable]) + ref_cost
            ref_units = ref_new
            ref_value -= ref_cost
        if np.isfinite(ref_previous) and ref_previous > 0:
            ref_return[i] = ref_value / ref_previous - 1.0
        ref_previous = ref_value

        # 风控:只用截至前一日的收益与权益路径,当日决策看不到当日结果
        scale = 1.0
        if risk is not None and i > 0:
            past = ref_return[:i]
            past = past[np.isfinite(past)]
            if risk.target_vol is not None and len(past) >= risk.min_days:
                window = past[-risk.vol_window:]
                realised = float(window.std(ddof=1) * np.sqrt(_DAYS)) if len(window) > 1 else np.nan
                if np.isfinite(realised) and realised > 0:
                    scale = min(risk.target_vol / realised, risk.max_leverage)
            if risk.drawdown_limit is not None and np.isfinite(peak_equity) and peak_equity > 0:
                if value / peak_equity - 1.0 <= risk.drawdown_limit:
                    scale *= risk.drawdown_scale
            if risk.perf_window is not None and len(past) >= risk.perf_window:
                # 只看截至前一日的自身收益,当日的结果还没发生
                recent = float(past[-risk.perf_window:].sum())
                scale *= risk.perf_up if recent > 0 else risk.perf_down
        leverage_out[i] = scale

        traded = 0.0
        filled = None
        if i >= entry_lag and rebalance[i - entry_lag] and value > 0:
            want = np.nan_to_num(target_values[i - entry_lag], nan=0.0) * scale
            want = np.where(tradable, want, 0.0)
            target_notional = want * value
            current_notional = np.where(tradable, units * price, 0.0)  # 不可交易的仓位动不了
            delta = target_notional - current_notional
            traded = float(np.abs(delta).sum())
            filled = delta.copy()          # 分母要和 daily_trade 一致,见下方
            cost = traded * cost_bps / 1e4
            new_units = np.where(tradable, target_notional / np.where(tradable, price, 1.0), units)
            cash -= float((new_units - units)[tradable] @ price[tradable]) + cost
            units = new_units
            daily_cost[i] = cost / value
            value -= cost

        if value <= 0:
            # 权益归零即出局。没有保证金与强平模型,但"亏穿之后继续按比例交易"会产生
            # 负权益和 -100% 以上的回撤这种无意义读数,比缺少强平模型更误导。
            units[:] = 0.0
            cash = 0.0
            value = 0.0
        daily_trade[i] = traded / value if value > 0 else 0.0
        if filled is not None and value > 0:
            # 与 daily_trade 用**同一个**扣费后权益做分母。分母差一点,订单清单的
            # 合计就对不上换手率,而这种对不上只会在事后核账时才被发现。
            fills_out[i] = filled / value
        equity[i] = value
        peak_equity = max(peak_equity, value)
        weights_out[i] = (units * mark) / value if value > 0 else np.nan
        if np.isfinite(previous_equity) and previous_equity > 0:
            daily_return[i] = max(value / previous_equity - 1.0, -1.0)
        previous_equity = value

    index = panel.index
    book = Book(
        equity=pd.Series(equity, index=index),
        returns=pd.Series(daily_return, index=index),
        weights=pd.DataFrame(weights_out, index=index, columns=panel.symbols),
        fills=pd.DataFrame(fills_out, index=index, columns=panel.symbols),
        trades=pd.Series(daily_trade, index=index),
        costs=pd.Series(daily_cost, index=index),
        funding=pd.Series(daily_funding, index=index),
        leverage=pd.Series(leverage_out, index=index),
    )
    book.metrics = performance(book.returns, book.equity, book.trades)
    return book
