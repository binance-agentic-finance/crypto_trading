"""唐奇安通道突破(多空) —— BTC 1h 唐奇安通道(不含当根)向上突破做多、向下突破做空,多空翻转。

strategyId: st_donchian_breakout  version: r1
三段式:_factors(只算因子)→ _forecast(因子 → verdict/score/bias)→ _sizing(forecast → 仓位/止损),
由 _analyze 串起来,三段之间只传 dict。

由 scripts/build_square_payloads.py 从 cyqnt_trd/standard_bot/signal/framework_strategies.py
(donchian_breakout)生成;信号与仓库内置策略逐根一致(tests/standard_bot/test_square_submit.py)。不要手改。
"""
import asyncio
import time
from decimal import ROUND_DOWN, ROUND_UP, Decimal

import numpy as np
import pandas as pd

from binance.strategy.node.capabilities.data import klines, account_balances, futures_position_risk
from binance.strategy.node.capabilities.execution import (
    futures_account_config, futures_close_position, futures_open_position, notify)
from binance.strategy.runtime import ctx, node, workflow

# ── 交易所规则 ────────────────────────────────────────────────
SYMBOL = "BTCUSDT"
BASE, QUOTE = "BTC", "USDT"
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
LEVERAGE = 1
PARAMS = {'lookback_window': 20,
 'breakout_buffer_bps': 0.0,
 'target_fraction': 0.2,
 'stop_pct': 0.03}


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


def _events(factors: dict, score: pd.Series, *, long=None, short=None, flat=None) -> dict:
    """Event masks → forecast dict. ``verdict`` is the bar's target event, with the same
    long > short > flat precedence as :func:`_hold_forward`; ``KEEP`` means no event.
    ``price`` (the close) is passed through so sizing can place a stop."""
    index = factors["index"]
    none = pd.Series(False, index=index)
    long = none if long is None else long
    short = none if short is None else short
    flat = none if flat is None else flat
    verdict = pd.Series("KEEP", index=index, dtype=object)
    verdict[flat.fillna(False)] = "FLAT"
    verdict[short.fillna(False)] = "SHORT"
    verdict[long.fillna(False)] = "LONG"
    bias = verdict.map({"LONG": "long", "SHORT": "short"}).fillna("neutral")
    return {"index": index, "price": factors["close"], "entry_long": long, "entry_short": short,
            "go_flat": flat, "verdict": verdict, "score": score, "bias": bias}


def _channel_position(close: pd.Series, upper: pd.Series, lower: pd.Series) -> pd.Series:
    """Where close sits in the channel: -1 at the lower band, +1 at the upper (unbounded)."""
    width = (upper - lower).replace(0.0, np.nan)
    return 2.0 * (close - (upper + lower) / 2.0) / width


# ① 因子(只算原始量,不做判断)
def _factors(df, p: dict) -> dict:
    upper = df["high"].shift(1).rolling(p["lookback_window"]).max()
    lower = df["low"].shift(1).rolling(p["lookback_window"]).min()
    return {"index": df.index, "close": df["close"], "upper": upper, "lower": lower}


# ② forecast:因子 → verdict / score / bias
def _forecast(factors: dict, p: dict) -> dict:
    close, upper, lower = factors["close"], factors["upper"], factors["lower"]
    buy = close > upper * (1.0 + p["breakout_buffer_bps"] / 1e4)
    sell = close < lower * (1.0 - p["breakout_buffer_bps"] / 1e4)   # SELL -> SHORT
    return _events(factors, _channel_position(close, upper, lower),
                   long=buy, short=sell)


# ③ 仓位:forecast → 持仓 / 止损
DEFAULT_STOP_PCT = 0.03


def _sizing(fc: dict, held: float = 0.0, stop_pct: float = DEFAULT_STOP_PCT) -> dict:
    """③ forecast → position + protective stop price.

    The target is stateful (no event = keep the position), so sizing needs the position
    held before ``fc`` starts: 0 in a backtest, the exchange holding live. ``stop`` is
    ``price × (1 ∓ stop_pct)`` on the held side and NaN when flat.
    """
    long, short = _hold_forward(fc["entry_long"], fc["entry_short"], fc["go_flat"], fc["index"],
                                held)
    position = long.astype(float) - short.astype(float)
    price = fc["price"]
    stop = (price * (1.0 - stop_pct)).where(long, (price * (1.0 + stop_pct)).where(short))
    return {"long": long, "short": short, "position": position, "stop": stop}


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
    size = _sizing(fc, held, p["stop_pct"])
    return {"symbol": symbol, "verdict": fc["verdict"].iloc[-1], "score": _last(fc["score"]),
            "bias": fc["bias"].iloc[-1], "target_position": int(size["position"].iloc[-1]),
            "stop": _last(size["stop"]), "missing": list(factors.get("missing", [])),
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


def _records(out) -> list:
    return [r for r in ((out or {}).get("records") or []) if isinstance(r, dict)]


def _num(rec: dict, *keys) -> float:
    for key in keys:
        if rec.get(key) not in (None, ""):
            try:
                return float(rec[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def _asset_qty(records, asset: str) -> float:
    """可用 + 冻结(挂着的保护止损单会冻结基础币,持仓判断要算上)。"""
    for r in records:
        if (r.get("asset") or r.get("coin")) == asset:
            return _num(r, "free", "available", "balance", "wallet_balance") + _num(r, "locked")
    return 0.0


def _equity(records) -> float:
    """合约账户权益(钱包余额)。"""
    for r in records:
        if (r.get("asset") or r.get("coin") or "USDT") == "USDT":
            value = _num(r, "wallet_balance", "walletBalance", "marginBalance", "balance", "total")
            if value:
                return value
    return 0.0


def _position_amt(records, symbol: str) -> float:
    """带符号的持仓数量:多 > 0,空 < 0。"""
    for r in records:
        if (r.get("symbol") or r.get("instrument")) == symbol:
            amt = _num(r, "positionAmt", "position_amt", "size")
            side = str(r.get("positionSide") or r.get("position_side") or "").upper()
            return -abs(amt) if side == "SHORT" else amt
    return 0.0


@node("std:fetch", retries=2)
async def fetch_klines():
    return klines(symbol=SYMBOL, timeframe=INTERVAL, limit=KLINE_LIMIT,
                  market_type=MARKET_TYPE, closed_only=True)


@node("std:fetch", retries=2)
async def fetch_position() -> dict:
    """持仓账本 = 交易所:合约读 futures_position_risk。"""
    recs = _records(futures_position_risk(risk_type="positions"))
    equity = _equity(_records(account_balances(balance_type="futures")))
    return {"records": recs, "position_amt": _position_amt(recs, SYMBOL), "equity": equity}


def _equity_usdt(position: dict, price: float) -> float:
    return position["equity"]


def _held(position: dict, price: float) -> int:
    amt = position["position_amt"]
    if abs(amt) * price < float(MIN_NOTIONAL):
        return 0
    return 1 if amt > 0 else -1


@node("std:signal")
async def signal_engine(df: pd.DataFrame, position: dict) -> dict:
    held = _held(position, float(df["close"].iloc[-1]))
    out = _analyze(SYMBOL, df, PARAMS, held=float(held))
    if out["missing"]:
        # 缺衍生品数据时 _factors 会补 0,信号就退化成别的策略(或永不触发);宁可这一轮不交易。
        ctx.log("WARN", "missing_feeds", {"missing": out["missing"]})
        raise RuntimeError(f"missing feeds {out['missing']}; skip this round")
    out["held_position"] = held
    out["rebalance_needed"] = out["target_position"] != held
    return out


def _notional(position: dict, price: float) -> float:
    """目标名义金额 = target_fraction × 真实权益(随盈亏复利,不写死金额)。"""
    return PARAMS["target_fraction"] * _equity_usdt(position, price)


def _qty(notional: float, price: float) -> float:
    return float((Decimal(str(notional)) / Decimal(str(price))).quantize(STEP, rounding=ROUND_DOWN))


def _round_qty(qty: float) -> float:
    return float(Decimal(str(qty)).quantize(STEP, rounding=ROUND_DOWN))


def _round_px(price: float) -> float:
    return float(Decimal(str(price)).quantize(TICK))


@node("exec:entry")
async def rebalance(signal: dict, position: dict, price: float) -> dict:
    """目标仓位 ≠ 当前仓位:先市价平旧仓,再按新方向开仓(+1 多 / -1 空 / 0 空仓)。"""
    held, target = signal["held_position"], signal["target_position"]
    if target == held:
        return {"changed": False, "from": held, "to": held, "action": "hold"}
    qty = _qty(_notional(position, price) * LEVERAGE, price) if target else 0.0
    if target and qty * price < float(MIN_NOTIONAL) and held == 0:
        return {"changed": False, "from": held, "to": held, "action": "skip",
                "reason": "equity_too_small"}
    if held != 0:
        # 立即市价平仓(样例里 close_at_trigger=True + STOP_MARKET 是挂止损,这里不是)
        futures_close_position(venue_class=VENUE_CLASS, instrument=SYMBOL,
                               close_at_trigger=False, order_type="MARKET")
    order = None
    if target != 0 and qty * price >= float(MIN_NOTIONAL):
        order = futures_open_position(venue_class=VENUE_CLASS, instrument=SYMBOL,
                                      side="BUY" if target > 0 else "SELL",
                                      position_side="LONG" if target > 0 else "SHORT",
                                      size=qty, order_type="MARKET")
        # 保护止损:开仓后立即挂 STOP_MARKET 平仓单(价格来自 _sizing 的 stop)
        futures_close_position(venue_class=VENUE_CLASS, instrument=SYMBOL, close_at_trigger=True,
                               order_type="STOP_MARKET", trigger_price=_round_px(signal["stop"]))
    else:
        target = 0                    # 已平仓,但权益不够开新仓
    return {"changed": True, "from": held, "to": target, "qty": qty,
            "stop_price": _round_px(signal["stop"]) if target else None,
            "action": "flip" if held and target else ("enter" if target else "exit"),
            "order": order}


@node("exec:notify")
async def notify_signal(signal: dict, fill: dict) -> dict:
    message = (f"{signal['symbol']} {signal['verdict']} bias={signal['bias']} "
               f"score={signal['score']} position {fill['from']} -> {fill['to']}")
    notify(message=message, channel="app")
    return {"message": message, "channel": "app"}


@workflow
async def execute_strategy():
    klines_raw, position = await asyncio.gather(
        fetch_klines(), fetch_position())
    ctx.state["fetch_klines"] = klines_raw
    ctx.state["fetch_position"] = position
    df = _klines_frame(klines_raw)
    signal = await signal_engine(df, position)
    ctx.state["signal_engine"] = signal
    if signal["rebalance_needed"]:
        fill = await rebalance(signal, position, float(df["close"].iloc[-1]))
        ctx.state["rebalance"] = fill
        if fill["changed"]:
            ctx.state["notify_signal"] = await notify_signal(signal, fill)


async def main():
    configured = False
    while True:
        try:
            if not configured:
                futures_account_config(instrument=SYMBOL, leverage=LEVERAGE,
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
