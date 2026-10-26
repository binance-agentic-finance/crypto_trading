"""把单标的 ``make_signals`` 策略接到新框架的向量化回测上。

`standard_bot` 的 block 策略是 ``make_signals(df) -> (long_signal, short_signal)``（单标的、
布尔信号）。本模块把它桥接成新框架的**目标权重 + 面板回测**：

    make_signals(df) -> (long, short)
        │  signals_to_weights            → W(单列 {+1,-1,0})
        │  build_panel({symbol: df})     → 1 列的 Panel（regular cells）
        ▼  portfolio.simulate(W, panel)  → Book（净额调仓、换手成本、资金费）

于是**所有** block/yaml 策略都用同一条框架回测路径，不必逐个改写。止盈止损/加减仓这类有
状态逻辑走 :mod:`cyqnt_trd.eval.blueprint` 的 ``apply_exits``；本桥接给的是"跟随信号持有"
的基线，执行口径与旧的事件引擎不同（下一根开盘成交、净额调仓），**回测数字会与旧引擎不同**。
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from . import bars, framework, portfolio

__all__ = ["signals_to_weights", "make_panel_from_df", "backtest_signals", "backtest_weights_from_df"]

_OHLCV = ("open", "high", "low", "close", "volume", "quote_volume")


def _boolean(sig, index) -> pd.Series:
    if sig is None:
        return pd.Series(False, index=index)
    s = pd.Series(sig, index=index) if not isinstance(sig, pd.Series) else sig.reindex(index)
    return s.fillna(False).astype(bool)


def signals_to_weights(long_signal, short_signal, index) -> pd.Series:
    """``(long, short)`` 布尔信号 → 单标的目标权重 ∈ {+1, 0, -1}（多空冲突时取多头）。"""
    long = _boolean(long_signal, index)
    short = _boolean(short_signal, index)
    w = pd.Series(0.0, index=index)
    w[short] = -1.0
    w[long] = 1.0                       # 冲突时多头优先，等价于旧引擎的"信号即持有"
    return w


def make_panel_from_df(df: pd.DataFrame, *, symbol: str = "ASSET",
                       funding: pd.Series | None = None, min_history: int = 20):
    """把一个单标的 OHLCV DataFrame 建成 1 列的 :class:`~cyqnt_trd.eval.bundle.Panel`。

    缺 ``quote_volume`` 时用 ``close * volume`` 估算。``cell_scheme='regular'`` —— 任意
    固定 bar 宽都行（1h/4h…），只要等间隔无缺口。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("make_signals df must have a DatetimeIndex")
    idx = df.index if df.index.tz is not None else df.index.tz_localize("UTC")
    frame = df.copy()
    frame.index = idx
    if "quote_volume" not in frame:
        frame["quote_volume"] = frame["close"] * frame["volume"]
    one = frame[list(_OHLCV)]
    fund = None
    if funding is not None:
        fund = pd.DataFrame({symbol: pd.Series(funding).reindex(idx)})
    return bars.build_panel({symbol: one}, funding=fund,
                            min_history=min_history, cell_scheme="regular")


def backtest_signals(make_signals: Callable[[pd.DataFrame], tuple], df: pd.DataFrame, *,
                     symbol: str = "ASSET", entry_lag: int = 2, cost_bps: float = 6.5,
                     funding: pd.Series | None = None, min_history: int = 20,
                     risk=None) -> portfolio.Book:
    """回测一个 ``make_signals`` 策略，走新框架的 :func:`portfolio.simulate`（净额调仓）。

    返回 :class:`~cyqnt_trd.eval.portfolio.Book`（``equity`` / ``fills`` / 指标）。这是研究/
    实盘同源的那条路：``Book`` 的最新目标仓位即实盘目标仓位。
    """
    result = make_signals(df)
    long, short = (result if isinstance(result, tuple) else (result, None))
    panel = make_panel_from_df(df, symbol=symbol, funding=funding, min_history=min_history)
    w = signals_to_weights(long, short, panel.index)
    W = pd.DataFrame({symbol: w}).reindex(panel.index).fillna(0.0).where(panel.mask, 0.0)
    return portfolio.simulate(W, panel, entry_lag=entry_lag, cost_bps=cost_bps, risk=risk)


def backtest_weights_from_df(make_signals: Callable[[pd.DataFrame], tuple], df: pd.DataFrame, *,
                             symbol: str = "ASSET", entry_lag: int = 2, h: int = 1,
                             cost_bps: float = 6.5, funding: pd.Series | None = None,
                             min_history: int = 20):
    """同上，但走 :func:`framework.backtest_weights`（非重叠 ΣW·R + 分段指标）。"""
    result = make_signals(df)
    long, short = (result if isinstance(result, tuple) else (result, None))
    panel = make_panel_from_df(df, symbol=symbol, funding=funding, min_history=min_history)
    w = signals_to_weights(long, short, panel.index)
    W = pd.DataFrame({symbol: w}).reindex(panel.index).fillna(0.0).where(panel.mask, 0.0)
    return framework.backtest_weights(W, panel, entry_lag=entry_lag, h=h, cost_bps=cost_bps)
