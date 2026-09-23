"""多周期均线价差(多空) —— BTC 1h 主周期 SMA 对高周期 SMA 的价差多空;高周期均线用 secondary_period×secondary_factor 根 1h K 线近似。

strategyId: st_multi_timeframe_ma_spread  version: r1
三段式:_factors(只算因子)→ _forecast(因子 → verdict/score/bias)→ _sizing(forecast → 仓位/止损),
由 _analyze 串起来,三段之间只传 dict。

由 scripts/build_square_payloads.py 从 cyqnt_trd/standard_bot/signal/framework_live.py
(multi_timeframe_ma_spread)生成;信号与仓库内置策略的回测逐根一致(tests/standard_bot/test_square_submit.py)。不要手改。
"""
import asyncio
import time
from decimal import ROUND_DOWN, Decimal

from binance.strategy.node.capabilities.data import klines, account_balances, futures_position_risk
from binance.strategy.node.capabilities.analysis import factor_evaluate
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
KLINE_LIMIT = 500                   # 覆盖最长窗口 + 1(持仓从交易所读,不需要更长历史)
LEVERAGE = 1
PARAMS = {'primary_period': 20,
 'secondary_period': 20,
 'threshold_bps': 0.0,
 'secondary_factor': 4,
 'target_fraction': 0.2,
 'stop_pct': 0.03}


def _sma(xs: list, n: int, end: int | None = None):
    """Mean of the ``n`` values ending before ``end`` (pandas ``rolling(n).mean()``), or None."""
    end = len(xs) if end is None else end
    if n <= 0 or end < n:
        return None
    return sum(xs[end - n:end]) / n


def _gt(a, b) -> bool:
    return a is not None and b is not None and a > b


def _lt(a, b) -> bool:
    return a is not None and b is not None and a < b


def _event(factors: dict, score, *, long=False, short=False, flat=False) -> dict:
    """Last-bar event → forecast dict; precedence long > short > flat, no event = KEEP."""
    verdict = "LONG" if long else "SHORT" if short else "FLAT" if flat else "KEEP"
    bias = {"LONG": "long", "SHORT": "short"}.get(verdict, "neutral")
    return {"price": factors["close"], "verdict": verdict, "score": score, "bias": bias}


# ① 因子(只算原始量,不做判断)
def _factors(bars: dict, p: dict) -> dict:
    c = bars["close"]
    primary = _sma(c, p["primary_period"])
    secondary = _sma(c, max(2, p["secondary_period"] * p["secondary_factor"]))
    spread = (primary - secondary) / secondary if primary is not None and secondary else None
    return {"close": c[-1], "primary_ma": primary, "secondary_ma": secondary, "spread": spread}


# ② forecast:因子 → verdict / score / bias
def _forecast(factors: dict, p: dict) -> dict:
    spread, thr = factors["spread"], p["threshold_bps"] / 1e4
    return _event(factors, spread, long=_gt(spread, thr), short=_lt(spread, -thr))


# ③ 仓位:forecast → 目标仓位 / 止损
def _sizing(fc: dict, held: float = 0.0, stop_pct: float = 0.03) -> dict:
    """③ forecast → target position + protective stop. No event keeps ``held``."""
    position = {"LONG": 1, "SHORT": -1, "FLAT": 0}.get(fc["verdict"], int(held))
    price = fc["price"]
    stop = (price * (1.0 - stop_pct) if position > 0 else
            price * (1.0 + stop_pct) if position < 0 else None)
    return {"position": position, "stop": stop}


def _analyze(symbol: str, bars: dict, p: dict = PARAMS, held: float = 0.0) -> dict:
    """单标的三段式分析:factors → forecast → sizing,只判断最后一根已收盘 K 线。

    ``held`` 是交易所上的实际持仓:最后一根没有事件就沿用它(无事件 = 保持仓位),
    与回测的逐根持有语义一致。
    """
    factors = _factors(bars, p)
    fc = _forecast(factors, p)
    size = _sizing(fc, held, p["stop_pct"])
    return {"symbol": symbol, "verdict": fc["verdict"], "score": fc["score"], "bias": fc["bias"],
            "target_position": size["position"], "stop": size["stop"],
            "missing": list(factors.get("missing", [])),
            "factors": {k: v for k, v in factors.items() if isinstance(v, (int, float))}}


_KLINE_FIELDS = ("open_time", "open", "high", "low", "close", "volume", "close_time")


def _bars(result) -> dict:
    """klines 返回 {"close": [...], "rows": [...]} → 各列 float 列表(行可以是 list 或 dict)。"""
    rows = result["rows"] if isinstance(result, dict) else result
    rows = [r if isinstance(r, dict) else dict(zip(_KLINE_FIELDS, r)) for r in (rows or [])]
    # closed_only=True 已经只给收盘 K 线;再防一手:最后一根没走完就丢掉。
    if rows and rows[-1].get("close_time") is not None \
            and int(rows[-1]["close_time"]) > time.time() * 1000:
        rows = rows[:-1]
    if not rows:
        raise RuntimeError(f"no klines for {SYMBOL}")
    return {k: [float(r[k]) for r in rows] for k in ("open", "high", "low", "close", "volume")}


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


_FACTOR_SPEC = {'operator': {'mode': 'emit',
              'function_name': 'mtf_ma_spread',
              'impl_source': 'def mtf_ma_spread(*, close, primary_period=20, '
                             'secondary_period=20, secondary_factor=4):\n'
                             '    c = [float(v) for v in close if v is not None]\n'
                             '    span = max(2, secondary_period * secondary_factor)\n'
                             '    if len(c) < max(primary_period, span):\n'
                             "        return {'value': None}\n"
                             '    primary = sum(c[-primary_period:]) / primary_period\n'
                             '    secondary = sum(c[-span:]) / span\n'
                             "    return {'value': (primary - secondary) / secondary if "
                             'secondary else None}\n'},
 'binding': {'inputs': {'close': 'close'},
             'params': {'primary_period': 20,
                        'secondary_period': 20,
                        'secondary_factor': 4},
             'window': 80,
             'output': 'value'},
 'name': 'mtf_ma_spread'}
_TRADEABLE = ('PASS', 'PASS_CONDITIONAL', 'HOLD_INFO')


@node("std:gate")
async def gate() -> dict:
    """研究闸门:factor_evaluate 给出的 verdict 决定这个因子能不能交易。"""
    res = factor_evaluate(factor=_FACTOR_SPEC)
    verdict = res.get("verdict") if res.get("status") == "ok" else None
    return {"verdict": verdict, "tradeable": verdict in _TRADEABLE}


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
async def signal_engine(bars: dict, position: dict) -> dict:
    held = _held(position, bars["close"][-1])
    out = _analyze(SYMBOL, bars, PARAMS, held=float(held))
    if out["missing"]:
        # 缺衍生品数据时信号会退化成别的策略(或永不触发);宁可这一轮不交易。
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
    if "gate" not in ctx.state:                  # 首轮评一次并缓存(检查和写入用同一个 key)
        ctx.state["gate"] = await gate()
    if not ctx.state["gate"]["tradeable"]:
        ctx.log("WARN", "gate_blocked", {"verdict": ctx.state["gate"]["verdict"]})
        return {"action": "skip", "reason": "gate", "verdict": ctx.state["gate"]["verdict"]}
    klines_raw, position = await asyncio.gather(
        fetch_klines(), fetch_position())
    ctx.state["fetch_klines"] = klines_raw
    ctx.state["fetch_position"] = position
    bars = _bars(klines_raw)
    signal = await signal_engine(bars, position)
    ctx.state["signal_engine"] = signal
    if signal["rebalance_needed"]:
        fill = await rebalance(signal, position, bars["close"][-1])
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
