"""Build the square submission artefacts for the built-in strategies.

For every submittable entry in ``strategies/_square/registry.json`` this writes

* ``<out-dir>/<strategyId>/<strategyId>.py``   — runnable submission code (三段式);
* ``<out-dir>/<strategyId>/<strategyId>.yaml`` — node/edge spec, one node per ``@node``;
* ``<out-dir>/<strategyId>/basic_info.json``   — the full submit payload (strategyId /
  version / spec / code / description / tags / shareLevel / freeFork / icon; spec and code as
  strings), i.e. the body for ``POST /v1/square/strategies/submit``;
* ``<out-dir>/<strategyId>/requirement.md``    — the strategy in plain Chinese;
* ``dist/square_payloads/<strategyId>.json``            — the same payload, with ``--strategy-id``
  overrides applied (not committed).

The stage functions are copied out of ``cyqnt_trd/standard_bot/signal/framework_live.py`` (the
list-only, last-bar form of ``framework_strategies.py``) with :func:`inspect.getsource`; a test
checks bar by bar that the submitted code matches the backtest signals.
Nothing is sent anywhere: this only writes files.

``<out-dir>`` defaults to ``dist/square`` (git-ignored). The published packages live in
binance-ai-platform ``examples/strategy-case-corpus/three_stage/``:

    python scripts/build_square_payloads.py --out-dir <platform>/examples/strategy-case-corpus/three_stage
    python scripts/build_square_payloads.py --out-dir <...>/three_stage --check   # fail if stale
"""
from __future__ import annotations

import argparse
import inspect
import json
import pprint
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cyqnt_trd.standard_bot.signal import framework_live as fl  # noqa: E402
from cyqnt_trd.standard_bot.signal import framework_strategies as fs  # noqa: E402

REGISTRY = REPO / "strategies" / "_square" / "registry.json"
#: Generated packages are NOT committed here any more: the published copy lives in
#: binance-ai-platform ``examples/strategy-case-corpus/three_stage/``. Pass ``--out-dir`` pointing
#: at that checkout to regenerate (or ``--check``) it; the default is a git-ignored scratch dir.
SQUARE_DIR = REPO / "dist" / "square"
PAYLOAD_DIR = REPO / "dist" / "square_payloads"
SHARE_LEVELS = ("READ_ONLY", "FULL")
KLINE_LIMIT = 500
#: Live-only knobs (the built-in backtest has neither): exposed as signal_engine custom params.
LIVE_PARAMS = {
    "target_fraction": (0.2, {"label": "目标仓位占权益比例", "widget": "number",
                              "min": 0.01, "max": 1.0, "step": 0.01}),
    "stop_pct": (0.03, {"label": "止损比例", "widget": "number",
                        "min": 0.005, "max": 0.2, "step": 0.005}),
}
INTERVAL_SEC = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}

#: Derivative feeds: builtin -> [(node id, capability, call kwargs (code), readable name, emoji)].
#: Both go through ``derivatives_market_metrics``; the record field names are read defensively
#: and still need confirming against the SDK. Liquidations have no platform capability, so
#: ``liquidation_reversal`` is marked ``submittable: false`` in the registry.
DERIVATIVE_FEEDS = {
    "oi_funding_breakout": [
        ("fetch_open_interest", "derivatives_market_metrics",
         {"metric_type": "open_interest_history", "symbol": "SYMBOL", "period": "INTERVAL",
          "limit": 50}, "持仓量历史", "📦"),
        ("fetch_funding_rate", "derivatives_market_metrics",
         {"metric_type": "funding_rate_info", "symbol": "SYMBOL"}, "资金费率", "💸"),
    ],
}


def load_registry(path: Path = REGISTRY) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def entries(registry: dict) -> list[dict]:
    """Registry rows with ``defaults`` merged in."""
    base = registry.get("defaults", {})
    return [{**base, **row} for row in registry["strategies"]]


def submittable(registry: dict) -> list[dict]:
    """Rows that can be built and submitted (``submittable: false`` rows carry a reason)."""
    return [e for e in entries(registry) if e.get("submittable", True)]


#: Every platform capability the generated code may call. They are **synchronous** on the
#: platform (``out = klines(...)``); only the ``@node`` / ``@workflow`` functions are async.
CAPABILITIES = ("klines", "rolling_extreme", "account_balances", "futures_position_risk", "derivatives_market_metrics",
                "factor_evaluate", "place_order", "futures_open_position",
                "futures_close_position", "futures_account_config", "notify",
                "open_interest_hist", "funding_rate_history", "liquidation_orders")


# ------------------------------------------------------------------ code
def _source(fn, rename: str | None = None) -> str:
    src = inspect.getsource(fn)
    if rename:
        src = re.sub(rf"^def {fn.__name__}\(", f"def {rename}(", src, count=1, flags=re.M)
    return src.rstrip() + "\n"


def stage_source(builtin: str) -> str:
    """Helpers + ``_factors`` / ``_forecast`` / ``_sizing`` for one built-in, list-only.

    Taken from :mod:`framework_live` (the last-bar, no-pandas form of the strategies), so the
    submitted code needs nothing beyond the standard library.
    """
    factors_fn, forecast_fn = fl.LIVE_STAGES[builtin]
    stages = _source(factors_fn, "_factors") + "\n\n" + _source(forecast_fn, "_forecast")
    helpers = [h for h in (fl._sma, fl._gt, fl._lt, fl._channel, fl._channel_position)
               if f"{h.__name__}(" in stages] + [fl._event]
    parts = [CHANNEL_SRC if h is fl._channel else _source(h) for h in helpers]
    parts += ["# ① 因子(只算原始量,不做判断)\n" + _source(factors_fn, "_factors"),
              "# ② forecast:因子 → verdict / score / bias\n" + _source(forecast_fn, "_forecast"),
              "# ③ 仓位:forecast → 目标仓位 / 止损\n" + _source(fl._sizing)]
    return "\n\n".join(parts)


#: Donchian bands through the platform's rolling_extreme. ``highs[:-1]`` drops the current bar,
#: exactly like the backtest's ``shift(1)``; max/min carry no rounding, so the bands are the
#: same numbers as ``framework_live._channel`` (a test runs both on every bar).
CHANNEL_SRC = """\
def _channel(high: list, low: list, n: int):
    \"\"\"唐奇安通道:不含当前 K 线的前 n 根最高 / 最低价(平台 rolling_extreme)。\"\"\"
    if len(high) < n + 1 or len(low) < n + 1:
        return None, None
    upper = rolling_extreme(series=high[:-1], op="max", period=n)["value"]
    lower = rolling_extreme(series=low[:-1], op="min", period=n)["value"]
    return upper, lower
"""

ANALYZE_SRC = """\
def _analyze(symbol: str, bars: dict, p: dict = PARAMS, held: float = 0.0) -> dict:
    \"\"\"单标的三段式分析:factors → forecast → sizing,只判断最后一根已收盘 K 线。

    ``held`` 是交易所上的实际持仓:最后一根没有事件就沿用它(无事件 = 保持仓位),
    与回测的逐根持有语义一致。
    \"\"\"
    factors = _factors(bars, p)
    fc = _forecast(factors, p)
    size = _sizing(fc, held, p["stop_pct"])
    return {"symbol": symbol, "verdict": fc["verdict"], "score": fc["score"], "bias": fc["bias"],
            "target_position": size["position"], "stop": size["stop"],
            "missing": list(factors.get("missing", [])),
            "factors": {k: v for k, v in factors.items() if isinstance(v, (int, float))}}
"""

KLINES_SRC = """\
_KLINE_FIELDS = ("open_time", "open", "high", "low", "close", "volume", "close_time")


def _bars(result) -> dict:
    \"\"\"klines 返回 {"close": [...], "rows": [...]} → 各列 float 列表(行可以是 list 或 dict)。\"\"\"
    rows = result["rows"] if isinstance(result, dict) else result
    rows = [r if isinstance(r, dict) else dict(zip(_KLINE_FIELDS, r)) for r in (rows or [])]
    # closed_only=True 已经只给收盘 K 线;再防一手:最后一根没走完就丢掉。
    if rows and rows[-1].get("close_time") is not None \\
            and int(rows[-1]["close_time"]) > time.time() * 1000:
        rows = rows[:-1]
    if not rows:
        raise RuntimeError(f"no klines for {SYMBOL}")
    return {k: [float(r[k]) for r in rows] for k in ("open", "high", "low", "close", "volume")}
"""

FEED_SRC = {
    "fetch_open_interest": """\
def _attach_open_interest(bars: dict, out) -> dict:
    \"\"\"持仓量历史 → 最后一根 K 线的 oi_change_bps(相邻两期变化,bps)。

    记录按时间升序、最后一条对应最新一期(字段名待 SDK 确认);取不到就不填,由 missing 拦下。
    \"\"\"
    values = [_num(r, "open_interest", "sumOpenInterest") for r in _records(out)]
    values = [v for v in values if v > 0]
    if len(values) >= 2:
        bars = {**bars, "oi_change_bps": (values[-1] / values[-2] - 1.0) * 10_000.0}
    return bars
""",
    "fetch_funding_rate": """\
def _attach_funding(bars: dict, out) -> dict:
    \"\"\"当前资金费率 → funding_rate_bps(字段名待 SDK 确认)。\"\"\"
    recs = _records(out)
    rate = recs[-1].get("funding_rate", recs[-1].get("fundingRate")) if recs else None
    if rate not in (None, ""):
        bars = {**bars, "funding_rate_bps": float(rate) * 10_000.0}
    return bars
""",
}


#: Gate factors: builtin -> (function name, impl_source, inputs, window from params).
#: Each is the strategy's core factor (the forecast ``score``) as one list-only function that
#: ``factor_evaluate`` can score; tests check it equals the repo's score on the last bar.
#: oi_funding_breakout's edge is the OI / funding confirmation, which is not in a price-only
#: binding, so its gate is a fixed HOLD_INFO (see README).
GATE_FACTORS = {
    "moving_average_cross": ("ma_spread", '''\
def ma_spread(*, close, fast_window=5, slow_window=20):
    c = [float(v) for v in close if v is not None]
    if len(c) < max(fast_window, slow_window):
        return {'value': None}
    fast = sum(c[-fast_window:]) / fast_window
    slow = sum(c[-slow_window:]) / slow_window
    return {'value': (fast - slow) / slow if slow else None}
''', ("close",), lambda p: max(p["fast_window"], p["slow_window"])),
    "price_moving_average": ("price_ma_spread", '''\
def price_ma_spread(*, close, period=20):
    c = [float(v) for v in close if v is not None]
    if len(c) < period:
        return {'value': None}
    ma = sum(c[-period:]) / period
    return {'value': (c[-1] - ma) / ma if ma else None}
''', ("close",), lambda p: p["period"]),
    "rsi_reversion": ("rsi_reversion_score", '''\
def rsi_reversion_score(*, close, period=14):
    c = [float(v) for v in close if v is not None]
    if len(c) < period + 1:
        return {'value': None}
    d = [c[i] - c[i - 1] for i in range(len(c) - period, len(c))]
    gain = sum(x for x in d if x > 0) / period
    loss = sum(-x for x in d if x < 0) / period
    rsi = 100.0 if loss == 0 else 100.0 - 100.0 / (1.0 + gain / loss)
    return {'value': (50.0 - rsi) / 50.0}
''', ("close",), lambda p: p["period"] + 1),
    "donchian_breakout": ("donchian_position", '''\
def donchian_position(*, high, low, close, lookback_window=20):
    h = [float(v) for v in high if v is not None]
    lo = [float(v) for v in low if v is not None]
    c = [float(v) for v in close if v is not None]
    if min(len(h), len(lo), len(c)) < lookback_window + 1:
        return {'value': None}
    upper = max(h[-lookback_window - 1:-1])
    lower = min(lo[-lookback_window - 1:-1])
    if upper == lower:
        return {'value': None}
    return {'value': 2.0 * (c[-1] - (upper + lower) / 2.0) / (upper - lower)}
''', ("high", "low", "close"), lambda p: p["lookback_window"] + 1),
    "multi_timeframe_ma_spread": ("mtf_ma_spread", '''\
def mtf_ma_spread(*, close, primary_period=20, secondary_period=20, secondary_factor=4):
    c = [float(v) for v in close if v is not None]
    span = max(2, secondary_period * secondary_factor)
    if len(c) < max(primary_period, span):
        return {'value': None}
    primary = sum(c[-primary_period:]) / primary_period
    secondary = sum(c[-span:]) / span
    return {'value': (primary - secondary) / secondary if secondary else None}
''', ("close",), lambda p: max(p["primary_period"],
                                  max(2, p["secondary_period"] * p["secondary_factor"]))),
}
TRADEABLE_VERDICTS = ("PASS", "PASS_CONDITIONAL", "HOLD_INFO")


def gate_spec(builtin: str) -> dict | None:
    """The operator spec ``factor_evaluate`` scores, or None (fixed HOLD_INFO)."""
    if builtin not in GATE_FACTORS:
        return None
    name, src, inputs, window = GATE_FACTORS[builtin]
    params = {k: v for k, v in fs.strategy_defaults(builtin).items()
              if re.search(rf"\b{k}\b", src.split(":", 1)[0])}
    return {"operator": {"mode": "emit", "function_name": name, "impl_source": src},
            "binding": {"inputs": {k: k for k in inputs}, "params": params,
                        "window": window(fs.strategy_defaults(builtin)), "output": "value"},
            "name": name}


#: Account helpers. Record field names are read defensively (pending SDK confirmation).
ACCOUNT_SRC = '''\
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
'''

ATTACH_FN = {"fetch_open_interest": "_attach_open_interest",
             "fetch_funding_rate": "_attach_funding"}


def _call_src(kwargs: dict) -> str:
    """Call kwargs → source; the names SYMBOL / INTERVAL stay symbolic (module constants)."""
    return ", ".join(f"{k}={v if v in ('SYMBOL', 'INTERVAL') else repr(v)}"
                     for k, v in kwargs.items())


FEED_WIDGETS = {"metric_type": {"label": "指标类型"},
                "symbol": {"label": "交易对", "widget": "text"},
                "period": {"label": "统计周期", "widget": "text"},
                "limit": {"label": "条数", "widget": "number", "min": 2, "max": 500}}


def build_code(entry: dict) -> str:
    builtin = entry["builtin"]
    params = {**fs.strategy_defaults(builtin), **{k: v for k, (v, _) in LIVE_PARAMS.items()}}
    feeds = DERIVATIVE_FEEDS.get(builtin, [])
    market = entry["market"]
    spot = market == "spot"
    ledger = "account_balances" if spot else "futures_position_risk"
    data_imports = ", ".join(["klines"] + sorted({ledger, "account_balances"} | {f[1] for f in feeds}))
    analysis_names = (["factor_evaluate"] if gate_spec(builtin) else []) + \
        (["rolling_extreme"] if "_channel(" in stage_source(builtin) else [])
    analysis_import = (f"from binance.strategy.node.capabilities.analysis import "
                       f"{', '.join(analysis_names)}\n" if analysis_names else "")
    exec_imports = ("notify, place_order" if spot else
                    "(\n    futures_account_config, futures_close_position, futures_open_position, notify)")
    quote = "USDT"
    base = entry["symbol"][:-len(quote)]
    rules = ('STEP = Decimal("0.00001")          # 数量步长\n'
             'TICK = Decimal("0.01")             # 价格步长\n'
             'MIN_NOTIONAL = Decimal("5")        # 最小名义金额 (USDT)\n') if spot else \
            ('STEP = Decimal("0.001")            # 数量步长\n'
             'TICK = Decimal("0.1")              # 价格步长\n'
             'MIN_NOTIONAL = Decimal("100")      # 最小名义金额 (USDT)\n'
             'MAX_LEV = 125\n')
    venue = "" if spot else 'VENUE_CLASS = "um"                 # U 本位永续\n'
    leverage = "" if spot else "LEVERAGE = 1\n"
    out = [f'''"""{entry["name"]} —— {entry["description"]}

strategyId: {entry["strategyId"]}  version: {entry["version"]}
三段式:_factors(只算因子)→ _forecast(因子 → verdict/score/bias)→ _sizing(forecast → 仓位/止损),
由 _analyze 串起来,三段之间只传 dict。

由 scripts/build_square_payloads.py 从 cyqnt_trd/standard_bot/signal/framework_live.py
({builtin})生成;信号与仓库内置策略的回测逐根一致(tests/standard_bot/test_square_submit.py)。不要手改。
"""
import asyncio
import time
from decimal import ROUND_DOWN, Decimal

from binance.strategy.node.capabilities.data import {data_imports}
{analysis_import}from binance.strategy.node.capabilities.execution import {exec_imports}
from binance.strategy.runtime import ctx, node, workflow

# ── 交易所规则 ────────────────────────────────────────────────
SYMBOL = "{entry["symbol"]}"
BASE, QUOTE = "{base}", "{quote}"
{rules}
# ── STRATEGY PARAMS ──────────────────────────────────────────
INTERVAL = "{entry["interval"]}"
MARKET_TYPE = "{market}"
{venue}INTERVAL_SEC = {INTERVAL_SEC[entry["interval"]]}
KLINE_LIMIT = {KLINE_LIMIT}                   # 覆盖最长窗口 + 1(持仓从交易所读,不需要更长历史)
{leverage}PARAMS = {pprint.pformat(params, sort_dicts=False, width=90)}
''']
    out.append(stage_source(builtin))
    out.append(ANALYZE_SRC)
    out.append(KLINES_SRC)
    out.append(ACCOUNT_SRC)
    out += [FEED_SRC[f[0]] for f in feeds]

    spec = gate_spec(builtin)
    if spec:
        gate_src = (f"_FACTOR_SPEC = {pprint.pformat(spec, sort_dicts=False, width=90)}\n"
                    f"_TRADEABLE = {TRADEABLE_VERDICTS!r}\n\n\n"
                    '@node("std:gate")\n'
                    "async def gate() -> dict:\n"
                    '    """研究闸门:factor_evaluate 给出的 verdict 决定这个因子能不能交易。"""\n'
                    "    res = factor_evaluate(factor=_FACTOR_SPEC)\n"
                    '    verdict = res.get("verdict") if res.get("status") == "ok" else None\n'
                    '    return {"verdict": verdict, "tradeable": verdict in _TRADEABLE}\n')
    else:
        gate_src = ('@node("std:gate")\n'
                    "async def gate() -> dict:\n"
                    '    """这个策略的因子依赖 OI / 资金费率确认,不能写成只吃价格的单函数 operator,\n'
                    '    factor_evaluate 评不了 —— 固定 HOLD_INFO(可交易,但没有研究证据)。"""\n'
                    '    return {"verdict": "HOLD_INFO", "tradeable": True, "reason": "factor_not_expressible"}\n')
    nodes = [gate_src, '''\
@node("std:fetch", retries=2)
async def fetch_klines():
    return klines(symbol=SYMBOL, timeframe=INTERVAL, limit=KLINE_LIMIT,
                  market_type=MARKET_TYPE, closed_only=True)
''']
    for node_id, capability, call, _, _ in feeds:
        nodes.append(f'''\
@node("std:fetch", retries=2)
async def {node_id}():
    return {capability}({_call_src(call)})
''')
    if spot:
        nodes.append('''\
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
''')
    else:
        nodes.append('''\
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
''')
    nodes.append('''\
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
''')
    if spot:
        nodes.append('''\
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
''')
    else:
        nodes.append('''\
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
''')
    nodes.append('''\
@node("exec:notify")
async def notify_signal(signal: dict, fill: dict) -> dict:
    message = (f"{signal['symbol']} {signal['verdict']} bias={signal['bias']} "
               f"score={signal['score']} position {fill['from']} -> {fill['to']}")
    notify(message=message, channel="app")
    return {"message": message, "channel": "app"}
''')
    out += nodes

    fetch_ids = ["fetch_klines", "fetch_position"] + [f[0] for f in feeds]
    names = ", ".join(["klines_raw", "position"] + [f"{f[0]}_raw" for f in feeds])
    calls = ", ".join(f"{n}()" for n in fetch_ids)
    fetch = f"    {names} = await asyncio.gather(\n        {calls})\n"
    fetch += '    ctx.state["fetch_klines"] = klines_raw\n'
    fetch += '    ctx.state["fetch_position"] = position\n'
    fetch += "".join(f'    ctx.state["{f[0]}"] = {f[0]}_raw\n' for f in feeds)
    fetch += "    bars = _bars(klines_raw)\n"
    fetch += "".join(f"    bars = {ATTACH_FN[f[0]]}(bars, {f[0]}_raw)\n" for f in feeds)
    out.append(f'''\
@workflow
async def execute_strategy():
    if "gate" not in ctx.state:                  # 首轮评一次并缓存(检查和写入用同一个 key)
        ctx.state["gate"] = await gate()
    if not ctx.state["gate"]["tradeable"]:
        ctx.log("WARN", "gate_blocked", {{"verdict": ctx.state["gate"]["verdict"]}})
        return {{"action": "skip", "reason": "gate", "verdict": ctx.state["gate"]["verdict"]}}
{fetch}    signal = await signal_engine(bars, position)
    ctx.state["signal_engine"] = signal
    if signal["rebalance_needed"]:
        fill = await rebalance(signal, position, bars["close"][-1])
        ctx.state["rebalance"] = fill
        if fill["changed"]:
            ctx.state["notify_signal"] = await notify_signal(signal, fill)


async def main():
{{configure}}    while True:
        try:
{{configure_once}}            await execute_strategy()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 —— 单轮失败只记日志,不让循环崩掉
            ctx.log("ERROR", "strategy_round_error", {{"error": str(exc)}})
        await asyncio.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    asyncio.run(main())
''')
    text = "\n\n".join(part.rstrip() + "\n" for part in out)
    if spot:
        return text.replace("{configure}", "").replace("{configure_once}", "")
    return (text.replace("{configure}", "    configured = False\n")
                .replace("{configure_once}",
                         "            if not configured:\n"
                         "                futures_account_config(instrument=SYMBOL, leverage=LEVERAGE,\n"
                         "                                       margin_type=\"ISOLATED\")\n"
                         "                configured = True\n"))


# ------------------------------------------------------------------ spec
class _Dumper(yaml.SafeDumper):
    pass


def _str(dumper, value: str):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


class _Quoted(str):
    """A scalar dumped with double quotes (spec ``strategy.version`` is ``"1.0"`` in the samples)."""


_Dumper.add_representer(str, _str)
_Dumper.add_representer(_Quoted, lambda d, v: d.represent_scalar("tag:yaml.org,2002:str", str(v),
                                                                 style='"'))


def build_spec(entry: dict, strategy_id: str | None = None) -> dict:
    builtin = entry["builtin"]
    defaults = fs.strategy_defaults(builtin)
    feeds = DERIVATIVE_FEEDS.get(builtin, [])
    symbol, interval = entry["symbol"], entry["interval"]
    gate = gate_spec(builtin)
    nodes = [{"id": "gate", "type": "analysis", "function": "factor_evaluate",
              "name": "研究闸门(factor_evaluate)", "emoji": "🚦",
              "params": [{"key": "factor", "value": gate},
                         {"key": "tradeable_verdicts", "value": list(TRADEABLE_VERDICTS)}]}
             if gate else
             {"id": "gate", "type": "custom", "name": "研究闸门(固定 HOLD_INFO)", "emoji": "🚦",
              "params": [{"key": "verdict", "value": "HOLD_INFO"}],
              "code": "return {'verdict': 'HOLD_INFO', 'tradeable': True}\n"}]
    nodes += [{"id": "fetch_klines", "type": "data", "function": "klines",
              "name": f"{symbol} {interval} K线", "emoji": "🕯️",
              "params": [{"key": "symbol", "value": symbol, "label": "交易对", "widget": "text"},
                         {"key": "timeframe", "value": interval, "label": "K线周期",
                          "widget": "text"},
                         {"key": "limit", "value": KLINE_LIMIT, "label": "K线根数",
                          "widget": "number", "min": 100, "max": 1500},
                         {"key": "market_type", "value": entry["market"], "label": "市场"},
                         {"key": "closed_only", "value": True, "label": "只用已收盘K线"}]}]
    spot = entry["market"] == "spot"
    nodes.append({"id": "fetch_position", "type": "data",
                  "function": "account_balances" if spot else "futures_position_risk",
                  "name": "现货余额(持仓账本)" if spot else "合约持仓(持仓账本)", "emoji": "💰",
                  "params": [{"key": "balance_type", "value": "spot"}] if spot else
                            [{"key": "risk_type", "value": "positions"}]})
    for node_id, capability, call, name, emoji in feeds:
        params = [{"key": k, "value": {"SYMBOL": symbol, "INTERVAL": interval}.get(v, v),
                   **FEED_WIDGETS[k]} for k, v in call.items()]
        nodes.append({"id": node_id, "type": "data", "function": capability,
                      "name": f"{symbol} {name}", "emoji": emoji, "params": params})
    widgets = entry["params"]
    nodes.append({
        "id": "signal_engine", "type": "custom", "name": "三段式信号:因子 → forecast → 仓位",
        "emoji": "🧠",
        "params": [{"key": k, "value": v, **widgets[k]} for k, v in defaults.items()]
                  + [{"key": k, "value": v, **w} for k, (v, w) in LIVE_PARAMS.items()],
        "code": stage_source(builtin) + "\n\n" + ANALYZE_SRC})
    notional = {"key": "target_notional",
                "value": "{{ state.signal_engine.params.target_fraction }} × 权益(USDT + 持仓市值)"}
    if spot:
        rebalance_params = [{"key": "instrument", "value": symbol},
                            {"key": "side", "value": "BUY 进场 / SELL 离场(卖出全部基础币)"},
                            {"key": "quote_size", "value": "target_fraction × 权益(BUY)"},
                            {"key": "size", "value": "{{ state.fetch_position.output.base_qty }}(SELL)"},
                            {"key": "order_type", "value": "MARKET"}, notional,
                            {"key": "stop_order", "value": "place_order(SELL, order_type=STOP_LOSS, "
                                                           "stop_price={{ state.signal_engine.output.stop }})"}]
    else:
        rebalance_params = [{"key": "venue_class", "value": "um"},
                            {"key": "instrument", "value": symbol},
                            {"key": "size", "value": "target_fraction × 权益 × LEVERAGE / close,按 STEP 取整"},
                            {"key": "side", "value": "BUY / SELL(target_position > 0 → BUY 开多,< 0 → SELL 开空)"},
                            {"key": "position_side", "value": "LONG / SHORT"},
                            {"key": "order_type", "value": "MARKET"}, notional,
                            {"key": "stop_order", "value": "futures_close_position(close_at_trigger=True, "
                                                           "order_type=STOP_MARKET, "
                                                           "trigger_price={{ state.signal_engine.output.stop }})"}]
    nodes.append({
        "id": "rebalance", "type": "execution",
        "function": "place_order" if spot else "futures_open_position",
        "name": "调仓到目标仓位", "emoji": "⚖️",
        "condition": "{{ state.signal_engine.output.rebalance_needed }}",
        "params": rebalance_params})
    nodes.append({
        "id": "notify_signal", "type": "execution", "function": "notify",
        "name": "推送调仓通知", "emoji": "🔔",
        "condition": "{{ state.rebalance.output.changed }}",
        "params": [{"key": "message",
                    "value": "{{ state.signal_engine.output.symbol }} "
                             "{{ state.signal_engine.output.verdict }} "
                             "{{ state.rebalance.output.from }} -> {{ state.rebalance.output.to }}",
                    "label": "通知内容", "widget": "text"},
                   {"key": "channel", "value": "app"}]})
    edges = [{"from": "gate", "to": "signal_engine",
              "label": "verdict ∈ PASS / PASS_CONDITIONAL / HOLD_INFO 才交易"}]
    edges += [{"from": n["id"], "to": "signal_engine"} for n in nodes if n["type"] == "data"]
    edges += [{"from": "signal_engine", "to": "rebalance", "label": "目标仓位变化"},
              {"from": "rebalance", "to": "notify_signal", "label": "已调仓"}]
    return {"strategy": {"id": strategy_id or entry["strategyId"],
                         "version": _Quoted(entry["specVersion"]),
                         "name": entry["name"], "description": entry["description"],
                         "source": f"cyqnt_trd.standard_bot.signal.framework_strategies:{builtin}"},
            "trigger": {"type": "schedule", "config": {"interval": interval}},
            "nodes": nodes, "edges": edges}


def dump_spec(spec: dict) -> str:
    return yaml.dump(spec, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=100)


# ------------------------------------------------------------------ payload
def validate_entry(entry: dict) -> None:
    sid = entry["strategyId"]
    if not re.fullmatch(r"st_[a-z0-9]+(_[a-z0-9]+)*", sid):
        raise ValueError(f"{sid}: strategyId must be st_ + snake_case")
    if entry["builtin"] not in fs.FRAMEWORK_STRATEGIES:
        raise ValueError(f"{sid}: unknown builtin {entry['builtin']!r}")
    if not entry.get("requirement"):
        raise ValueError(f"{sid}: requirement (plain-language description) is required")
    if not 3 <= len(entry["tags"]) <= 5:
        raise ValueError(f"{sid}: needs 3-5 tags")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", entry["icon"]):
        raise ValueError(f"{sid}: icon must be a short lowercase identifier, got {entry['icon']!r}")
    if entry["shareLevel"] not in SHARE_LEVELS:
        raise ValueError(f"{sid}: shareLevel must be one of {SHARE_LEVELS}")
    if set(entry["params"]) != set(fs.strategy_defaults(entry["builtin"])):
        raise ValueError(f"{sid}: params {sorted(entry['params'])} != make_signals signature "
                         f"{sorted(fs.strategy_defaults(entry['builtin']))}")


def platform_strategy_id(entry: dict, overrides: dict | None = None) -> str:
    """The id the submit API looks up (x-user-id + strategyId + version).

    It must name a strategy that already exists on the platform and has been deployed at least
    once (paper or live); our ``st_*`` id is only the spec's ``strategy.id``. Falls back to it
    when neither an override nor ``platformStrategyId`` is set, so the payload is still built.
    """
    return ((overrides or {}).get(entry["strategyId"]) or entry.get("platformStrategyId")
            or entry["strategyId"])


def build_payload(entry: dict, overrides: dict | None = None) -> dict:
    """The submit body. The spec's ``strategy.id`` equals the payload ``strategyId``."""
    sid = platform_strategy_id(entry, overrides)
    return {"strategyId": sid, "version": entry["version"],
            "spec": dump_spec(build_spec(entry, sid)), "code": build_code(entry),
            "description": entry["description"], "tags": list(entry["tags"]),
            "shareLevel": entry["shareLevel"], "freeFork": bool(entry["freeFork"]),
            "icon": entry["icon"]}


def build_requirement(entry: dict) -> str:
    """requirement.md: what the strategy does, its parameters and its risk rules, in Chinese."""
    defaults = fs.strategy_defaults(entry["builtin"])
    widgets = entry["params"]
    lines = [f"# {entry['name']}", "", entry["requirement"], "", "## 参数", ""]
    lines += [f"- {widgets[k]['label']}(`{k}`):默认 {v}" for k, v in defaults.items()]
    lines += [f"- {w['label']}(`{k}`):默认 {v}" for k, (v, w) in LIVE_PARAMS.items()]
    venue = "现货" if entry["market"] == "spot" else "U 本位永续合约(1 倍杠杆、逐仓)"
    lines += ["", "## 仓位与风控", "",
              f"- 交易标的:{entry['symbol']} {venue},{entry['interval']} K 线收盘后决策。",
              "- 持仓以交易所为准:每轮从账户读当前持仓,没有新信号就保持。",
              "- 仓位 = 目标仓位占权益比例 × 真实权益("
              + ("可用 USDT + 基础币市值" if entry["market"] == "spot" else "合约账户钱包余额")
              + ");不足最小下单金额时本轮不下单。",
              "- 每次开仓后立即挂保护止损单,止损价 = 开仓参考价 ×(1 ∓ 止损比例)。",
              "- 首轮先用 factor_evaluate 评估核心因子,verdict 为 PASS / PASS_CONDITIONAL / HOLD_INFO 才交易。"
              if entry["builtin"] in GATE_FACTORS else
              "- 核心因子依赖持仓量 / 资金费率确认,无法交给 factor_evaluate 评估,研究闸门固定为 HOLD_INFO。",
              "- 取数或下单出错时本轮跳过并记录日志,下一轮继续。", ""]
    return "\n".join(lines)


def artefacts(registry: dict | None = None) -> dict[Path, str]:
    """Every file under ``SQUARE_DIR/<strategyId>/`` → its expected text."""
    files = {}
    for entry in submittable(registry or load_registry()):
        validate_entry(entry)
        folder = SQUARE_DIR / entry["strategyId"]
        payload = build_payload(entry)
        files[folder / f"{entry['strategyId']}.py"] = payload["code"]
        files[folder / f"{entry['strategyId']}.yaml"] = payload["spec"]
        files[folder / "basic_info.json"] = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        files[folder / "requirement.md"] = build_requirement(entry)
    return files


def extra_files(files: dict) -> list[Path]:
    """Files in a package folder the generator does not produce (e.g. a leftover code.py)."""
    folders = {p.parent for p in files}
    return sorted(p for d in folders if d.exists() for p in d.iterdir()
                  if p not in files and p.name != "__pycache__")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="only compare committed code/spec against the generator")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where the <strategyId>/ package folders go (default dist/square); "
                         "point at binance-ai-platform examples/strategy-case-corpus/three_stage")
    ap.add_argument("--payload-dir", type=Path, default=PAYLOAD_DIR)
    ap.add_argument("--no-payloads", action="store_true")
    ap.add_argument("--strategy-id", action="append", default=[], metavar="ST_ID=PLATFORM_ID",
                    help="payload strategyId override: the platform id of an already-deployed "
                         "strategy (repeatable); defaults to registry platformStrategyId")
    args = ap.parse_args(argv)
    overrides = dict(item.split("=", 1) for item in args.strategy_id)
    global SQUARE_DIR
    if args.out_dir is not None:
        SQUARE_DIR = args.out_dir.resolve()

    registry = load_registry()
    files = artefacts(registry)
    if args.check:
        stale = [p for p, text in files.items()
                 if not p.exists() or p.read_text(encoding="utf-8") != text]
        stale += extra_files(files)
        for p in stale:
            print(f"stale: {p}")
        return 1 if stale else 0
    for entry in entries(registry):
        if not entry.get("submittable", True):
            print(f"skip {entry['strategyId']}: not submittable ({entry.get('notSubmittableReason')})")
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")
    if not args.no_payloads:
        args.payload_dir.mkdir(parents=True, exist_ok=True)
        unknown = sorted(set(overrides) - {e["strategyId"] for e in submittable(registry)})
        if unknown:
            raise SystemExit(f"--strategy-id for unknown strategies: {unknown}")
        for entry in submittable(registry):
            out = args.payload_dir / f"{entry['strategyId']}.json"
            payload = build_payload(entry, overrides)
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
            note = "" if payload["strategyId"] != entry["strategyId"] else \
                "  (strategyId not mapped to a deployed platform strategy yet)"
            print(f"wrote {out}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
