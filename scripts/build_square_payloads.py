"""Build the square submission artefacts for the built-in strategies.

For every entry in ``strategies/_square/registry.json`` this writes

* ``strategies/_square/<strategyId>/code.py``   — runnable submission code (三段式);
* ``strategies/_square/<strategyId>/spec.yaml`` — node/edge spec, one node per ``@node``;
* ``dist/square_payloads/<strategyId>.json``    — the body for ``POST /v1/square/strategies/submit``
  (strategyId / version / spec / code / description / tags / shareLevel / freeFork / icon).

The stage functions are copied out of ``cyqnt_trd/standard_bot/signal/framework_strategies.py``
with :func:`inspect.getsource`, so the submitted code cannot drift from the repo's signals.
Nothing is sent anywhere: this only writes files.

    python scripts/build_square_payloads.py            # regenerate code/spec + payloads
    python scripts/build_square_payloads.py --check    # fail if committed code/spec are stale
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

from cyqnt_trd.standard_bot.signal import framework_strategies as fs  # noqa: E402

SQUARE_DIR = REPO / "strategies" / "_square"
REGISTRY = SQUARE_DIR / "registry.json"
PAYLOAD_DIR = REPO / "dist" / "square_payloads"
SHARE_LEVELS = ("READ_ONLY", "FULL")
KLINE_LIMIT = 1000
INTERVAL_SEC = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}

#: Derivative feeds: builtin -> [(node id, capability, spec params, emoji, readable name)].
DERIVATIVE_FEEDS = {
    "oi_funding_breakout": [
        ("fetch_open_interest", "open_interest_hist", "持仓量历史", "📦"),
        ("fetch_funding_rate", "funding_rate_history", "资金费率历史", "💸"),
    ],
    "liquidation_reversal": [
        ("fetch_liquidations", "liquidation_orders", "强平订单", "💥"),
    ],
}


def load_registry(path: Path = REGISTRY) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def entries(registry: dict) -> list[dict]:
    """Registry rows with ``defaults`` merged in."""
    base = registry.get("defaults", {})
    return [{**base, **row} for row in registry["strategies"]]


# ------------------------------------------------------------------ code
def _source(fn, rename: str | None = None) -> str:
    src = inspect.getsource(fn)
    if rename:
        src = re.sub(rf"^def {fn.__name__}\(", f"def {rename}(", src, count=1, flags=re.M)
    return src.rstrip() + "\n"


def stage_source(builtin: str) -> str:
    """Helpers + ``_factors`` / ``_forecast`` / ``_sizing`` for one built-in."""
    factors_fn, forecast_fn = fs.STRATEGY_STAGES[builtin]
    stages = _source(factors_fn, "_factors") + "\n\n" + _source(forecast_fn, "_forecast")
    helpers = [fs._hold_forward, fs._events]
    helpers += [h for h in (fs._col, fs._missing, fs._channel_position)
                if f"{h.__name__}(" in stages]
    parts = [_source(h) for h in helpers]
    parts += ["# ① 因子(只算原始量,不做判断)\n" + _source(factors_fn, "_factors"),
              "# ② forecast:因子 → verdict / score / bias\n" + _source(forecast_fn, "_forecast"),
              "# ③ 仓位:forecast → 持仓 / 止损\n" + _source(fs._sizing)]
    return "\n\n".join(parts)


ANALYZE_SRC = '''\
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
'''

KLINES_SRC = '''\
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


def _klines_frame(rows) -> pd.DataFrame:
    df = _rows_frame(rows, _KLINE_COLUMNS).rename(
        columns={"openTime": "open_time", "closeTime": "close_time", "quoteVolume": "quote_volume"})
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df.index = pd.DatetimeIndex(_bar_time(df["open_time"]))
    # 只用已收盘的 K 线:最后一根还没走完就丢掉,和回测逐根决策的口径一致。
    if "close_time" in df.columns and len(df) and int(df["close_time"].iloc[-1]) > time.time() * 1000:
        df = df.iloc[:-1]
    return df
'''

FEED_SRC = {
    "fetch_open_interest": '''\
def _attach_open_interest(df: pd.DataFrame, rows) -> pd.DataFrame:
    """持仓量 → oi_change_bps(逐根变化,bps),与 data/derivatives.py 同口径。"""
    oi = _rows_frame(rows, ["timestamp", "sumOpenInterest"])
    if oi.empty:
        return df
    oi.index = pd.DatetimeIndex(_bar_time(oi["timestamp"]))
    value = oi["sumOpenInterest"].astype(float).groupby(level=0).last()
    df = df.copy()
    # 没被持仓量数据覆盖的 K 线留 NaN(比较结果为 False = 不产生事件),不补 0。
    df["oi_change_bps"] = (value.pct_change() * 10_000.0).reindex(df.index)
    return df
''',
    "fetch_funding_rate": '''\
def _attach_funding(df: pd.DataFrame, rows) -> pd.DataFrame:
    """资金费率 → funding_rate_bps,按结算时间向后填充到每根 K 线。"""
    fr = _rows_frame(rows, ["fundingTime", "fundingRate"])
    if fr.empty:
        return df
    fr.index = pd.DatetimeIndex(_bar_time(fr["fundingTime"]))
    rate = (fr["fundingRate"].astype(float) * 10_000.0).groupby(level=0).last()
    df = df.copy()
    df["funding_rate_bps"] = rate.reindex(df.index.union(rate.index)).ffill().reindex(df.index)
    return df
''',
    "fetch_liquidations": '''\
def _attach_liquidations(df: pd.DataFrame, rows) -> pd.DataFrame:
    """强平订单 → 每根 K 线的多/空爆仓名义额(SELL 单 = 多头被强平),与 data/liquidations.py 同口径。"""
    liq = _rows_frame(rows, ["time", "side", "price", "origQty"])
    if liq.empty:
        return df
    notional = liq["price"].astype(float) * liq["origQty"].astype(float)
    side = liq["side"].astype(str).str.upper()
    bars = pd.DatetimeIndex(_bar_time(liq["time"]))
    df = df.copy()
    df["long_liq_notional_usd"] = notional.where(side.values == "SELL", 0.0).groupby(bars).sum().reindex(df.index).fillna(0.0)
    df["short_liq_notional_usd"] = notional.where(side.values == "BUY", 0.0).groupby(bars).sum().reindex(df.index).fillna(0.0)
    return df
''',
}
ATTACH_FN = {"fetch_open_interest": "_attach_open_interest",
             "fetch_funding_rate": "_attach_funding",
             "fetch_liquidations": "_attach_liquidations"}
FEED_CALL = {"open_interest_hist": "symbol=SYMBOL, period=INTERVAL, limit=500",
             "funding_rate_history": "symbol=SYMBOL, limit=100",
             "liquidation_orders": "symbol=SYMBOL, limit=1000"}


def _pandas_freq(interval: str) -> str:
    return interval[:-1] + {"m": "min", "h": "h", "d": "D"}[interval[-1]]


def build_code(entry: dict) -> str:
    builtin = entry["builtin"]
    params = fs.strategy_defaults(builtin)
    feeds = DERIVATIVE_FEEDS.get(builtin, [])
    data_imports = ", ".join(["klines"] + sorted(f[1] for f in feeds))
    out = [f'''"""{entry["name"]} —— {entry["description"]}

strategyId: {entry["strategyId"]}  version: {entry["version"]}
三段式:_factors(只算因子)→ _forecast(因子 → verdict/score/bias)→ _sizing(forecast → 仓位/止损),
由 _analyze 串起来,三段之间只传 dict。

由 scripts/build_square_payloads.py 从 cyqnt_trd/standard_bot/signal/framework_strategies.py
({builtin})生成;信号与仓库内置策略逐根一致(tests/standard_bot/test_square_submit.py)。不要手改。
"""
import asyncio
import time
from decimal import ROUND_DOWN, ROUND_UP, Decimal

import numpy as np
import pandas as pd

from binance.strategy.node.capabilities.data import {data_imports}
from binance.strategy.node.capabilities.execution import (
    futures_account_config, futures_close_position, futures_open_position, notify)
from binance.strategy.runtime import ctx, node, workflow

# ── 交易所规则 ────────────────────────────────────────────────
SYMBOL = "{entry["symbol"]}"
STEP = Decimal("0.001")            # 数量步长
TICK = Decimal("0.1")              # 价格步长
MIN_NOTIONAL = Decimal("100")      # 最小名义金额 (USDT)
MAX_LEV = 125

# ── STRATEGY PARAMS ──────────────────────────────────────────
INTERVAL = "{entry["interval"]}"
INTERVAL_SEC = {INTERVAL_SEC[entry["interval"]]}
PANDAS_FREQ = "{_pandas_freq(entry["interval"])}"
KLINE_LIMIT = {KLINE_LIMIT}                  # 持仓是事件驱动的(无事件 = 继续持有),窗口要够长
ORDER_NOTIONAL_USDT = Decimal("100")
LEVERAGE = 1
PARAMS = {pprint.pformat(params, sort_dicts=False, width=90)}
''']
    out.append(stage_source(builtin))
    out.append(ANALYZE_SRC)
    out.append(KLINES_SRC)
    out += [FEED_SRC[f[0]] for f in feeds]

    nodes = ['''\
@node("std:fetch", retries=2)
async def fetch_klines() -> pd.DataFrame:
    return _klines_frame(await klines(symbol=SYMBOL, interval=INTERVAL, limit=KLINE_LIMIT))
''']
    for node_id, capability, _, _ in feeds:
        nodes.append(f'''\
@node("std:fetch", retries=2)
async def {node_id}():
    return await {capability}({FEED_CALL[capability]})
''')
    nodes.append('''\
@node("std:signal")
async def signal_engine(df: pd.DataFrame) -> dict:
    out = _analyze(SYMBOL, df, PARAMS, held=float(ctx.state.get("position", 0)))
    if out["missing"]:
        # 缺衍生品数据时 _factors 会补 0,信号就退化成别的策略(或永不触发);宁可这一轮不交易。
        raise RuntimeError(f"missing feeds {out['missing']}; skip this round")
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
        await futures_close_position(symbol=SYMBOL)
    if target != 0:
        await futures_open_position(symbol=SYMBOL, side="LONG" if target > 0 else "SHORT",
                                    quantity=str(_qty(price)), leverage=LEVERAGE)
    ctx.state["position"] = target
    return {"changed": True, "from": current, "to": target}


@node("exec:notify")
async def notify_signal(signal: dict, fill: dict) -> None:
    await notify(message=f"{signal['symbol']} {signal['verdict']} bias={signal['bias']} "
                         f"score={signal['score']} position {fill['from']} -> {fill['to']}")
''')
    out += nodes

    if feeds:
        names = ", ".join(["kl"] + [f"raw_{i}" for i in range(len(feeds))])
        calls = ", ".join(["fetch_klines()"] + [f"{f[0]}()" for f in feeds])
        joins = "".join(f"    df = {ATTACH_FN[f[0]]}(df, raw_{i})\n" for i, f in enumerate(feeds))
        fetch = f"    {names} = await asyncio.gather({calls})\n    df = kl\n{joins}"
    else:
        fetch = "    df = await fetch_klines()\n"
    out.append(f'''\
@workflow
async def execute_strategy():
{fetch}    signal = await signal_engine(df)
    ctx.state["signal_engine"] = {{"output": signal}}
    if signal["target_position"] != int(ctx.state.get("position", 0)):
        fill = await rebalance(signal, float(df["close"].iloc[-1]))
        ctx.state["rebalance"] = {{"output": fill}}
        if fill["changed"]:
            await notify_signal(signal, fill)


async def main():
    configured = False
    while True:
        try:
            if not configured:
                await futures_account_config(symbol=SYMBOL, leverage=LEVERAGE)
                configured = True
            await execute_strategy()
        except Exception as exc:  # noqa: BLE001 —— 单轮失败只记日志,不让循环崩掉
            ctx.log(f"execute_strategy failed: {{exc!r}}")
        await asyncio.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    asyncio.run(main())
''')
    return "\n\n".join(part.rstrip() + "\n" for part in out)


# ------------------------------------------------------------------ spec
class _Dumper(yaml.SafeDumper):
    pass


def _str(dumper, value: str):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_Dumper.add_representer(str, _str)


def build_spec(entry: dict) -> dict:
    builtin = entry["builtin"]
    defaults = fs.strategy_defaults(builtin)
    feeds = DERIVATIVE_FEEDS.get(builtin, [])
    symbol, interval = entry["symbol"], entry["interval"]
    nodes = [{"id": "fetch_klines", "type": "data", "function": "klines",
              "name": f"{symbol} {interval} K线", "emoji": "🕯️",
              "params": [{"key": "symbol", "value": symbol}, {"key": "interval", "value": interval},
                         {"key": "limit", "value": KLINE_LIMIT}]}]
    for node_id, capability, name, emoji in feeds:
        call = dict(kv.split("=") for kv in FEED_CALL[capability].split(", "))
        params = [{"key": k, "value": {"SYMBOL": symbol, "INTERVAL": interval}.get(v, v)}
                  for k, v in call.items()]
        params = [{**p, "value": int(p["value"]) if str(p["value"]).isdigit() else p["value"]}
                  for p in params]
        nodes.append({"id": node_id, "type": "data", "function": capability,
                      "name": f"{symbol} {name}", "emoji": emoji, "params": params})
    widgets = entry["params"]
    nodes.append({
        "id": "signal_engine", "type": "custom", "name": "三段式信号:因子 → forecast → 仓位",
        "emoji": "🧠",
        "params": [{"key": k, "value": v, **widgets[k]} for k, v in defaults.items()],
        "code": stage_source(builtin) + "\n\n" + ANALYZE_SRC})
    nodes.append({
        "id": "rebalance", "type": "execution", "function": "futures_open_position",
        "name": "调仓到目标仓位", "emoji": "⚖️",
        "condition": "{{ state.signal_engine.output.target_position }} != {{ state.position }}",
        "params": [{"key": "symbol", "value": symbol},
                   {"key": "notional_usdt", "value": 100.0},
                   {"key": "leverage", "value": 1}]})
    nodes.append({
        "id": "notify_signal", "type": "execution", "function": "notify",
        "name": "推送调仓通知", "emoji": "🔔",
        "condition": "{{ state.rebalance.output.changed }} == true",
        "params": [{"key": "message",
                    "value": "{{ state.signal_engine.output.symbol }} "
                             "{{ state.signal_engine.output.verdict }}"}]})
    edges = [{"from": n["id"], "to": "signal_engine"} for n in nodes if n["type"] == "data"]
    edges += [{"from": "signal_engine", "to": "rebalance", "label": "目标仓位变化"},
              {"from": "rebalance", "to": "notify_signal", "label": "已调仓"}]
    return {"strategy": {"id": entry["strategyId"], "version": entry["version"],
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
    if not 3 <= len(entry["tags"]) <= 5:
        raise ValueError(f"{sid}: needs 3-5 tags")
    if entry["shareLevel"] not in SHARE_LEVELS:
        raise ValueError(f"{sid}: shareLevel must be one of {SHARE_LEVELS}")
    if set(entry["params"]) != set(fs.strategy_defaults(entry["builtin"])):
        raise ValueError(f"{sid}: params {sorted(entry['params'])} != make_signals signature "
                         f"{sorted(fs.strategy_defaults(entry['builtin']))}")


def build_payload(entry: dict) -> dict:
    return {"strategyId": entry["strategyId"], "version": entry["version"],
            "spec": dump_spec(build_spec(entry)), "code": build_code(entry),
            "description": entry["description"], "tags": list(entry["tags"]),
            "shareLevel": entry["shareLevel"], "freeFork": bool(entry["freeFork"]),
            "icon": entry["icon"]}


def artefacts(registry: dict | None = None) -> dict[Path, str]:
    """Every committed file under ``strategies/_square/<strategyId>/`` → its expected text."""
    files = {}
    for entry in entries(registry or load_registry()):
        validate_entry(entry)
        folder = SQUARE_DIR / entry["strategyId"]
        files[folder / "code.py"] = build_code(entry)
        files[folder / "spec.yaml"] = dump_spec(build_spec(entry))
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="only compare committed code/spec against the generator")
    ap.add_argument("--payload-dir", type=Path, default=PAYLOAD_DIR)
    ap.add_argument("--no-payloads", action="store_true")
    args = ap.parse_args(argv)

    registry = load_registry()
    files = artefacts(registry)
    if args.check:
        stale = [p for p, text in files.items()
                 if not p.exists() or p.read_text(encoding="utf-8") != text]
        for p in stale:
            print(f"stale: {p.relative_to(REPO)}")
        return 1 if stale else 0
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(REPO)}")
    if not args.no_payloads:
        args.payload_dir.mkdir(parents=True, exist_ok=True)
        for entry in entries(registry):
            out = args.payload_dir / f"{entry['strategyId']}.json"
            out.write_text(json.dumps(build_payload(entry), ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
            print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
