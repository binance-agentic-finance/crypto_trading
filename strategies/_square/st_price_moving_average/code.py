"""价格穿越均线(只做多) —— BTC 1h 收盘价上穿 SMA 开多、下穿卖出,只做多现货。

strategyId: st_price_moving_average  version: r1
三段式:_factors(只算因子)→ _forecast(因子 → verdict/score/bias)→ _sizing(forecast → 仓位/止损),
由 _analyze 串起来,三段之间只传 dict。

由 scripts/build_square_payloads.py 从 cyqnt_trd/standard_bot/signal/framework_strategies.py
(price_moving_average)生成;信号与仓库内置策略逐根一致(tests/standard_bot/test_square_submit.py)。不要手改。
"""
import asyncio
import time
from decimal import ROUND_DOWN, Decimal

from binance.strategy.node.capabilities.data import klines, account_balances
from binance.strategy.node.capabilities.analysis import factor_evaluate
from binance.strategy.node.capabilities.execution import notify, place_order
from binance.strategy.runtime import ctx, node, workflow

# ── 交易所规则 ────────────────────────────────────────────────
SYMBOL = "BTCUSDT"
BASE, QUOTE = "BTC", "USDT"
STEP = Decimal("0.00001")          # 数量步长
TICK = Decimal("0.01")             # 价格步长
MIN_NOTIONAL = Decimal("5")        # 最小名义金额 (USDT)

# ── STRATEGY PARAMS ──────────────────────────────────────────
INTERVAL = "1h"
MARKET_TYPE = "spot"
INTERVAL_SEC = 3600
KLINE_LIMIT = 500                   # 覆盖最长窗口 + 1(持仓从交易所读,不需要更长历史)
PARAMS = {'period': 20, 'entry_threshold': 0.0, 'target_fraction': 0.2, 'stop_pct': 0.03}


def _sma(xs: list, n: int, end: int | None = None):
    """Mean of the ``n`` values ending before ``end`` (pandas ``rolling(n).mean()``), or None."""
    end = len(xs) if end is None else end
    if n <= 0 or end < n:
        return None
    return sum(xs[end - n:end]) / n


def _event(factors: dict, score, *, long=False, short=False, flat=False) -> dict:
    """Last-bar event → forecast dict; precedence long > short > flat, no event = KEEP."""
    verdict = "LONG" if long else "SHORT" if short else "FLAT" if flat else "KEEP"
    bias = {"LONG": "long", "SHORT": "short"}.get(verdict, "neutral")
    return {"price": factors["close"], "verdict": verdict, "score": score, "bias": bias}


# ① 因子(只算原始量,不做判断)
def _factors(bars: dict, p: dict) -> dict:
    c = bars["close"]
    ma, prev_ma = _sma(c, p["period"]), _sma(c, p["period"], len(c) - 1)
    return {"close": c[-1], "ma": ma, "prev_close": c[-2] if len(c) > 1 else None,
            "prev_ma": prev_ma, "spread": (c[-1] - ma) / ma if ma else None}


# ② forecast:因子 → verdict / score / bias
def _forecast(factors: dict, p: dict) -> dict:
    close, ma = factors["close"], factors["ma"]
    prev_c, prev_ma = factors["prev_close"], factors["prev_ma"]
    crossed = prev_c is not None and prev_ma is not None and ma is not None
    far = crossed and abs((close - ma) / ma) > p["entry_threshold"]
    up = far and prev_c <= prev_ma and close > ma
    dn = far and prev_c >= prev_ma and close < ma
    return _event(factors, factors["spread"], long=up, flat=dn)


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
              'function_name': 'price_ma_spread',
              'impl_source': 'def price_ma_spread(*, close, period=20):\n'
                             '    c = [float(v) for v in close if v is not None]\n'
                             '    if len(c) < period:\n'
                             "        return {'value': None}\n"
                             '    ma = sum(c[-period:]) / period\n'
                             "    return {'value': (c[-1] - ma) / ma if ma else None}\n"},
 'binding': {'inputs': {'close': 'close'},
             'params': {'period': 20},
             'window': 20,
             'output': 'value'},
 'name': 'price_ma_spread'}
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
    """持仓账本 = 交易所:现货读基础币 / 计价币余额。"""
    recs = _records(account_balances(balance_type="spot"))
    return {"records": recs, "base_qty": _asset_qty(recs, BASE), "quote_free": _asset_qty(recs, QUOTE)}


def _equity_usdt(position: dict, price: float) -> float:
    """现货权益 = 可用 USDT + 基础币市值。"""
    return position["quote_free"] + position["base_qty"] * price


def _held(position: dict, price: float) -> int:
    """持有的基础币市值达到最小名义金额才算持仓(碎币不算)。"""
    return 1 if position["base_qty"] * price >= float(MIN_NOTIONAL) else 0


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
    """只做多现货:目标 1 且空仓 → 市价买入;目标 0 且持仓 → 卖出全部基础币(完整往返)。"""
    held, target = signal["held_position"], signal["target_position"]
    if target > 0 and held == 0:
        notional = _notional(position, price)
        if notional < float(MIN_NOTIONAL):
            return {"changed": False, "from": held, "to": held, "action": "skip",
                    "reason": "equity_too_small", "notional": notional}
        order = place_order(instrument=SYMBOL, side="BUY", quote_size=round(notional, 2),
                            order_type="MARKET")
        qty = _num(order or {}, "executed_qty", "executedQty", "filled_qty") or notional / price
        # 保护止损:买入后立即挂 STOP_LOSS(价格来自 _sizing 的 stop)
        stop = place_order(instrument=SYMBOL, side="SELL", size=_round_qty(qty),
                           order_type="STOP_LOSS", stop_price=_round_px(signal["stop"]))
        return {"changed": True, "from": held, "to": 1, "action": "enter", "order": order,
                "notional": round(notional, 2), "stop_price": _round_px(signal["stop"]),
                "stop_order": stop}
    if target == 0 and held > 0:
        order = place_order(instrument=SYMBOL, side="SELL", size=position["base_qty"],
                            order_type="MARKET")
        return {"changed": True, "from": held, "to": 0, "action": "exit", "order": order}
    return {"changed": False, "from": held, "to": held, "action": "hold"}


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
    while True:
        try:
            await execute_strategy()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 —— 单轮失败只记日志,不让循环崩掉
            ctx.log("ERROR", "strategy_round_error", {"error": str(exc)})
        await asyncio.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    asyncio.run(main())
