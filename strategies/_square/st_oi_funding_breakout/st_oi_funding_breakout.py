"""持仓量确认突破(多空) —— BTC 1h 唐奇安突破需持仓量同步上升、资金费率不过热才入场,多空对称;缺持仓量/资金费率数据时不交易。

strategyId: st_oi_funding_breakout  version: r1
三段式:_factors(只算因子)→ _forecast(因子 → verdict/score/bias)→ _sizing(forecast → 仓位/止损),
由 _analyze 串起来,三段之间只传 dict。

由 scripts/build_square_payloads.py 从 cyqnt_trd/standard_bot/signal/framework_live.py
(oi_funding_breakout)生成;信号与仓库内置策略的回测逐根一致(tests/standard_bot/test_square_submit.py)。不要手改。
"""
import asyncio
import time
from decimal import ROUND_DOWN, Decimal

from binance.strategy.node.capabilities.data import klines, account_balances, derivatives_market_metrics, futures_position_risk
from binance.strategy.node.capabilities.analysis import rolling_extreme
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
PARAMS = {'lookback_window': 20,
 'breakout_buffer_bps': 0.0,
 'oi_threshold_bps': 0.0,
 'max_funding_rate_bps': 100.0,
 'target_fraction': 0.2,
 'stop_pct': 0.03}


def _channel(high: list, low: list, n: int):
    """唐奇安通道:不含当前 K 线的前 n 根最高 / 最低价(平台 rolling_extreme)。"""
    if len(high) < n + 1 or len(low) < n + 1:
        return None, None
    upper = rolling_extreme(series=high[:-1], op="max", period=n)["value"]
    lower = rolling_extreme(series=low[:-1], op="min", period=n)["value"]
    return upper, lower


def _channel_position(close, upper, lower):
    if upper is None or lower is None or upper == lower:
        return None
    return 2.0 * (close - (upper + lower) / 2.0) / (upper - lower)


def _event(factors: dict, score, *, long=False, short=False, flat=False) -> dict:
    """Last-bar event → forecast dict; precedence long > short > flat, no event = KEEP."""
    verdict = "LONG" if long else "SHORT" if short else "FLAT" if flat else "KEEP"
    bias = {"LONG": "long", "SHORT": "short"}.get(verdict, "neutral")
    return {"price": factors["close"], "verdict": verdict, "score": score, "bias": bias}


# ① 因子(只算原始量,不做判断)
def _factors(bars: dict, p: dict) -> dict:
    upper, lower = _channel(bars["high"], bars["low"], p["lookback_window"])
    oi, funding = bars.get("oi_change_bps"), bars.get("funding_rate_bps")
    return {"close": bars["close"][-1], "upper": upper, "lower": lower,
            "oi_change_bps": oi, "funding_rate_bps": funding,
            "missing": [k for k, v in (("oi_change_bps", oi), ("funding_rate_bps", funding))
                        if v is None]}


# ② forecast:因子 → verdict / score / bias
def _forecast(factors: dict, p: dict) -> dict:
    close, upper, lower = factors["close"], factors["upper"], factors["lower"]
    oi, funding = factors["oi_change_bps"], factors["funding_rate_bps"]
    buf = p["breakout_buffer_bps"] / 1e4
    oi_ok = oi is not None and oi >= p["oi_threshold_bps"]
    buy = (oi_ok and upper is not None and close > upper * (1.0 + buf)
           and funding is not None and funding <= p["max_funding_rate_bps"])
    sell = (oi_ok and lower is not None and close < lower * (1.0 - buf)
            and funding is not None and funding >= -p["max_funding_rate_bps"])
    return _event(factors, _channel_position(close, upper, lower), long=buy, short=sell)


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


def _attach_open_interest(bars: dict, out) -> dict:
    """持仓量历史 → 最后一根 K 线的 oi_change_bps(相邻两期变化,bps)。

    记录按时间升序、最后一条对应最新一期(字段名待 SDK 确认);取不到就不填,由 missing 拦下。
    """
    values = [_num(r, "open_interest", "sumOpenInterest") for r in _records(out)]
    values = [v for v in values if v > 0]
    if len(values) >= 2:
        bars = {**bars, "oi_change_bps": (values[-1] / values[-2] - 1.0) * 10_000.0}
    return bars


def _attach_funding(bars: dict, out) -> dict:
    """当前资金费率 → funding_rate_bps(字段名待 SDK 确认)。"""
    recs = _records(out)
    rate = recs[-1].get("funding_rate", recs[-1].get("fundingRate")) if recs else None
    if rate not in (None, ""):
        bars = {**bars, "funding_rate_bps": float(rate) * 10_000.0}
    return bars


@node("std:gate")
async def gate() -> dict:
    """这个策略的因子依赖 OI / 资金费率确认,不能写成只吃价格的单函数 operator,
    factor_evaluate 评不了 —— 固定 HOLD_INFO(可交易,但没有研究证据)。"""
    return {"verdict": "HOLD_INFO", "tradeable": True, "reason": "factor_not_expressible"}


@node("std:fetch", retries=2)
async def fetch_klines():
    return klines(symbol=SYMBOL, timeframe=INTERVAL, limit=KLINE_LIMIT,
                  market_type=MARKET_TYPE, closed_only=True)


@node("std:fetch", retries=2)
async def fetch_open_interest():
    return derivatives_market_metrics(metric_type='open_interest_history', symbol=SYMBOL, period=INTERVAL, limit=50)


@node("std:fetch", retries=2)
async def fetch_funding_rate():
    return derivatives_market_metrics(metric_type='funding_rate_info', symbol=SYMBOL)


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
    klines_raw, position, fetch_open_interest_raw, fetch_funding_rate_raw = await asyncio.gather(
        fetch_klines(), fetch_position(), fetch_open_interest(), fetch_funding_rate())
    ctx.state["fetch_klines"] = klines_raw
    ctx.state["fetch_position"] = position
    ctx.state["fetch_open_interest"] = fetch_open_interest_raw
    ctx.state["fetch_funding_rate"] = fetch_funding_rate_raw
    bars = _bars(klines_raw)
    bars = _attach_open_interest(bars, fetch_open_interest_raw)
    bars = _attach_funding(bars, fetch_funding_rate_raw)
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
