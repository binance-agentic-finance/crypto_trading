"""快慢均线交叉(只做多) —— BTC 1h 快慢 SMA 价差:快线高于慢线持多,低于慢线平仓,只做多。

strategyId: st_moving_average_cross  version: r1
三段式:_factors(只算因子)→ _forecast(因子 → verdict/score/bias)→ _sizing(forecast → 仓位/止损),
由 _analyze 串起来,三段之间只传 dict。

由 scripts/build_square_payloads.py 从 cyqnt_trd/standard_bot/signal/framework_strategies.py
(moving_average_cross)生成;信号与仓库内置策略逐根一致(tests/standard_bot/test_square_submit.py)。不要手改。
"""
import asyncio
import time
from decimal import ROUND_DOWN, ROUND_UP, Decimal

import numpy as np
import pandas as pd

from binance.strategy.node.capabilities.data import klines
from binance.strategy.node.capabilities.execution import (
    futures_account_config, futures_close_position, futures_open_position, notify)
from binance.strategy.runtime import ctx, node, workflow

# ── 交易所规则 ────────────────────────────────────────────────
SYMBOL = "BTCUSDT"
STEP = Decimal("0.001")            # 数量步长
TICK = Decimal("0.1")              # 价格步长
MIN_NOTIONAL = Decimal("100")      # 最小名义金额 (USDT)
MAX_LEV = 125

# ── STRATEGY PARAMS ──────────────────────────────────────────
INTERVAL = "1h"
MARKET_TYPE = "futures"
VENUE_CLASS = "um"                 # U 本位永续
INTERVAL_SEC = 3600
PANDAS_FREQ = "1h"
KLINE_LIMIT = 1000                  # 持仓是事件驱动的(无事件 = 继续持有),窗口要够长
ORDER_NOTIONAL_USDT = Decimal("100")
LEVERAGE = 1
PARAMS = {'fast_window': 5, 'slow_window': 20, 'entry_threshold': 0.0}


def _hold_forward(entry_long: pd.Series, entry_short: pd.Series, go_flat: pd.Series,
                  index, held: float = 0.0) -> tuple:
    """Kernel TARGET_{LONG,SHORT,FLAT,KEEP} → forward-filled position → (long, short).

    Precedence on a bar: long > short > flat (kernel events are mutually exclusive, so this
    only matters if a caller passes overlapping masks). Bars with no event keep the position;
    bars before the first event hold ``held`` (0 in a backtest, the live holding in a
    windowed live run — otherwise a window with no event would silently flatten it).
    """
    target = pd.Series(np.nan, index=index)
    target[go_flat.fillna(False)] = 0.0
    target[entry_short.fillna(False)] = -1.0
    target[entry_long.fillna(False)] = 1.0
    pos = target.ffill().fillna(held)
    return pos > 0, pos < 0


def _events(index, score: pd.Series, *, long=None, short=None, flat=None) -> dict:
    """Event masks → forecast dict. ``verdict`` is the bar's target event, with the same
    long > short > flat precedence as :func:`_hold_forward`; ``KEEP`` means no event."""
    none = pd.Series(False, index=index)
    long = none if long is None else long
    short = none if short is None else short
    flat = none if flat is None else flat
    verdict = pd.Series("KEEP", index=index, dtype=object)
    verdict[flat.fillna(False)] = "FLAT"
    verdict[short.fillna(False)] = "SHORT"
    verdict[long.fillna(False)] = "LONG"
    bias = verdict.map({"LONG": "long", "SHORT": "short"}).fillna("neutral")
    return {"index": index, "entry_long": long, "entry_short": short, "go_flat": flat,
            "verdict": verdict, "score": score, "bias": bias}


# ① 因子(只算原始量,不做判断)
def _factors(df, p: dict) -> dict:
    fast = df["close"].rolling(p["fast_window"]).mean()
    slow = df["close"].rolling(p["slow_window"]).mean()
    return {"index": df.index, "fast_ma": fast, "slow_ma": slow, "spread": (fast - slow) / slow}


# ② forecast:因子 → verdict / score / bias
def _forecast(factors: dict, p: dict) -> dict:
    spread = factors["spread"]
    buy = spread > p["entry_threshold"]
    sell = spread < -p["entry_threshold"]                  # SELL -> FLAT (long-only)
    return _events(factors["index"], spread, long=buy, flat=sell)


# ③ 仓位:forecast → 持仓 / 止损
def _sizing(fc: dict, held: float = 0.0) -> dict:
    """③ forecast → position. Built-ins trade a unit position and carry no stop.

    The target is stateful (no event = keep the position), so sizing needs the position
    held before ``fc`` starts: 0 in a backtest, the account/``ctx.state`` holding live.
    """
    long, short = _hold_forward(fc["entry_long"], fc["entry_short"], fc["go_flat"], fc["index"],
                                held)
    position = long.astype(float) - short.astype(float)
    return {"long": long, "short": short, "position": position, "stop": None}


def _last(series):
    value = series.iloc[-1] if len(series) else np.nan
    return float(value) if pd.notna(value) else None


def _analyze(symbol: str, df: pd.DataFrame, p: dict = PARAMS, held: float = 0.0) -> dict:
    """单标的三段式分析:factors → forecast → sizing,取最后一根已收盘 K 线的结论。

    ``held`` 是当前实际持仓:窗口内没有任何事件时沿用它(无事件 = 保持仓位),
    这样按窗口算出的目标仓位与回测的逐根持有语义一致。
    """
    factors = _factors(df, p)
    fc = _forecast(factors, p)
    size = _sizing(fc, held)
    return {"symbol": symbol, "verdict": fc["verdict"].iloc[-1], "score": _last(fc["score"]),
            "bias": fc["bias"].iloc[-1], "target_position": int(size["position"].iloc[-1]),
            "stop": size["stop"], "missing": list(factors.get("missing", [])),
            "factors": {k: _last(v) for k, v in factors.items() if isinstance(v, pd.Series)}}


_KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                  "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def _rows_frame(rows, columns) -> pd.DataFrame:
    """capability 返回的 list[list] 或 list[dict] → DataFrame。"""
    if rows and isinstance(rows[0], dict):
        return pd.DataFrame(rows)
    return pd.DataFrame(rows, columns=columns[:len(rows[0])] if rows else columns)


def _bar_time(ms) -> pd.DatetimeIndex:
    """毫秒时间戳 → 所在 K 线的开盘时间(UTC)。"""
    return pd.to_datetime(pd.Series(ms).astype("int64"), unit="ms", utc=True).dt.floor(PANDAS_FREQ)


def _klines_frame(result) -> pd.DataFrame:
    """klines 能力返回 {"close": [...], "rows": [...]};也兼容直接给 rows 的写法。"""
    rows = result["rows"] if isinstance(result, dict) else result
    df = _rows_frame(rows, _KLINE_COLUMNS).rename(
        columns={"openTime": "open_time", "closeTime": "close_time", "quoteVolume": "quote_volume"})
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df.index = pd.DatetimeIndex(_bar_time(df["open_time"]))
    # 只用已收盘的 K 线:最后一根还没走完就丢掉,和回测逐根决策的口径一致。
    if "close_time" in df.columns and len(df) and int(df["close_time"].iloc[-1]) > time.time() * 1000:
        df = df.iloc[:-1]
    return df


@node("std:fetch", retries=2)
async def fetch_klines():
    return await klines(symbol=SYMBOL, timeframe=INTERVAL, limit=KLINE_LIMIT,
                        market_type=MARKET_TYPE, closed_only=True)


@node("std:signal")
async def signal_engine(df: pd.DataFrame) -> dict:
    # ctx.state["position"] 是跨轮次的持仓状态(不是 spec 节点),由 rebalance 维护。
    held = int(ctx.state.get("position", 0))
    out = _analyze(SYMBOL, df, PARAMS, held=float(held))
    if out["missing"]:
        # 缺衍生品数据时 _factors 会补 0,信号就退化成别的策略(或永不触发);宁可这一轮不交易。
        ctx.log("WARN", "missing_feeds", {"missing": out["missing"]})
        raise RuntimeError(f"missing feeds {out['missing']}; skip this round")
    out["held_position"] = held
    out["rebalance_needed"] = out["target_position"] != held
    return out


def _qty(price: float) -> Decimal:
    px = Decimal(str(price))
    qty = (ORDER_NOTIONAL_USDT / px).quantize(STEP, rounding=ROUND_DOWN)
    floor = (MIN_NOTIONAL / px).quantize(STEP, rounding=ROUND_UP)
    return max(qty, floor)


@node("exec:entry")
async def rebalance(signal: dict, price: float) -> dict:
    """目标仓位 ≠ 当前仓位:先平旧仓,再按 ORDER_NOTIONAL_USDT 开新仓(+1 多 / -1 空 / 0 空仓)。"""
    current = int(ctx.state.get("position", 0))
    target = int(signal["target_position"])
    if target == current:
        return {"changed": False, "from": current, "to": target}
    if current != 0:
        # 立即市价平仓(样例里 close_at_trigger=True + STOP_MARKET 是挂止损,这里不是)
        await futures_close_position(venue_class=VENUE_CLASS, instrument=SYMBOL,
                                     close_at_trigger=False, order_type="MARKET")
    if target != 0:
        await futures_open_position(venue_class=VENUE_CLASS, instrument=SYMBOL,
                                    size=str(_qty(price)),
                                    side="BUY" if target > 0 else "SELL", order_type="MARKET")
    ctx.state["position"] = target
    return {"changed": True, "from": current, "to": target}


@node("exec:notify")
async def notify_signal(signal: dict, fill: dict) -> dict:
    message = (f"{signal['symbol']} {signal['verdict']} bias={signal['bias']} "
               f"score={signal['score']} position {fill['from']} -> {fill['to']}")
    await notify(message=message, channel="app")
    return {"message": message, "channel": "app"}


@workflow
async def execute_strategy():
    klines_raw = await fetch_klines()
    ctx.state["fetch_klines"] = klines_raw
    df = _klines_frame(klines_raw)
    signal = await signal_engine(df)
    ctx.state["signal_engine"] = signal
    if signal["rebalance_needed"]:
        fill = await rebalance(signal, float(df["close"].iloc[-1]))
        ctx.state["rebalance"] = fill
        if fill["changed"]:
            ctx.state["notify_signal"] = await notify_signal(signal, fill)


async def main():
    configured = False
    while True:
        try:
            if not configured:
                await futures_account_config(instrument=SYMBOL, leverage=LEVERAGE,
                                             margin_type="ISOLATED")
                configured = True
            await execute_strategy()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 —— 单轮失败只记日志,不让循环崩掉
            ctx.log("ERROR", "strategy_round_error", {"error": str(exc)})
        await asyncio.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    asyncio.run(main())
