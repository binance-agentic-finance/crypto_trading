"""内置策略的三段式重构 + 广场提交物(spec / code / payload)。

1. 重构不改信号:三段式版本与重构前(``_fixture_legacy_framework_strategies``)在同一份数据上
   long/short 逐根相同,框架回测结果也相同。
2. 三段式契约:factors → forecast → sizing 之间只传 dict,缺衍生品数据会被标出来。
3. 提交物:``strategies/_square`` 里提交的 code/spec 与生成器同步;生成的 code 在桩运行时下
   跑出的信号与仓库一致;spec 的节点/边与 code 的 ``@node`` 一一对应。
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import _fixture_legacy_framework_strategies as legacy
from cyqnt_trd.standard_bot.signal import framework_strategies as fs
from cyqnt_trd.standard_bot.simulation import FrameworkBacktestRunner

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("build_square_payloads",
                                               REPO / "scripts" / "build_square_payloads.py")
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)

SIDS = sorted(fs.FRAMEWORK_STRATEGIES)
ENTRIES = {e["builtin"]: e for e in builder.entries(builder.load_registry())}


def _df(n=600, seed=7, derivatives=True):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = np.exp(np.cumsum(rng.normal(0, 0.012, n)) + 4)
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    vol = np.abs(rng.normal(1e3, 1e2, n))
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": vol, "quote_volume": vol * close}, index=idx)
    if derivatives:
        df["oi_change_bps"] = rng.normal(0, 30, n)
        df["funding_rate_bps"] = rng.normal(0, 80, n)
        spike = rng.random(n) < 0.25
        df["long_liq_notional_usd"] = np.where(spike, np.abs(rng.normal(0, 3e5, n)), 0.0)
        df["short_liq_notional_usd"] = np.where(rng.random(n) < 0.25,
                                                np.abs(rng.normal(0, 3e5, n)), 0.0)
    return df


PARAM_SETS = {
    "moving_average_cross": [{}, {"fast_window": 3, "slow_window": 40, "entry_threshold": 0.002}],
    "price_moving_average": [{}, {"period": 7, "entry_threshold": 0.001}],
    "rsi_reversion": [{}, {"period": 6, "oversold": 25.0, "overbought": 60.0}],
    "donchian_breakout": [{}, {"lookback_window": 8, "breakout_buffer_bps": 5.0}],
    "multi_timeframe_ma_spread": [{}, {"primary_period": 5, "secondary_period": 6,
                                       "threshold_bps": 20.0, "secondary_factor": 3}],
    "oi_funding_breakout": [{}, {"lookback_window": 10, "oi_threshold_bps": 10.0,
                                 "max_funding_rate_bps": 50.0}],
    "liquidation_reversal": [{}, {"long_liquidation_threshold_usd": 50_000.0,
                                  "liquidation_imbalance_ratio": 0.7}],
}
CASES = [(sid, i, seed, deriv) for sid in SIDS for i in range(len(PARAM_SETS[sid]))
         for seed in (1, 7, 42) for deriv in (True, False)]


# ---------------------------------------------------------------- 1. 信号不变
@pytest.mark.parametrize("sid,pi,seed,deriv", CASES)
def test_three_stage_matches_legacy_signals(sid, pi, seed, deriv):
    df = _df(seed=seed, derivatives=deriv)
    params = PARAM_SETS[sid][pi]
    old_long, old_short = legacy.FRAMEWORK_STRATEGIES[sid](df, **params)
    new_long, new_short = fs.FRAMEWORK_STRATEGIES[sid](df, **params)
    pd.testing.assert_series_equal(new_long, old_long)
    pd.testing.assert_series_equal(new_short, old_short)


@pytest.mark.parametrize("sid", SIDS)
def test_three_stage_matches_legacy_backtest(sid):
    df = _df(n=400, seed=3)
    run = FrameworkBacktestRunner().run
    old = run(lambda d: legacy.FRAMEWORK_STRATEGIES[sid](d), df, instrument_id="BTCUSDT",
              min_history=40)
    new = run(fs.make_signals_for(sid), df, instrument_id="BTCUSDT", min_history=40)
    assert new.total_return == old.total_return
    assert new.metrics == old.metrics


def test_legacy_fixture_is_a_real_counterfactual():
    """The equivalence test is only meaningful if the strategies actually trade here."""
    df = _df(seed=1)
    for sid in SIDS:
        long, short = legacy.FRAMEWORK_STRATEGIES[sid](df)
        assert long.any() or short.any(), sid


# ---------------------------------------------------------------- 2. 三段式契约
@pytest.mark.parametrize("sid", SIDS)
def test_stages_pass_dicts_and_forecast_never_sees_df(sid):
    df = _df()
    p = fs.strategy_defaults(sid)
    factors_fn, forecast_fn = fs.STRATEGY_STAGES[sid]
    factors = factors_fn(df, p)
    assert isinstance(factors, dict)
    fc = forecast_fn(dict(factors), p)          # a plain dict, no df anywhere
    assert {"verdict", "score", "bias", "entry_long", "entry_short", "go_flat"} <= set(fc)
    size = fs._sizing(fc)
    assert set(size) == {"long", "short", "position", "stop"}
    out = fs.analyze(sid, df)
    assert set(out["verdict"].unique()) <= {"LONG", "SHORT", "FLAT", "KEEP"}
    assert set(out["bias"].unique()) <= {"long", "short", "neutral"}
    # verdict is the event, position is its held consequence
    assert (out["position"][out["verdict"] == "LONG"] == 1.0).all()
    assert (out["position"][out["verdict"] == "SHORT"] == -1.0).all()
    assert (out["position"][out["verdict"] == "FLAT"] == 0.0).all()
    pd.testing.assert_series_equal(out["long"], fs.FRAMEWORK_STRATEGIES[sid](df)[0])


def test_analyze_rejects_unknown_params():
    with pytest.raises(TypeError):
        fs.analyze("rsi_reversion", _df(), perod=7)


@pytest.mark.parametrize("sid", ["oi_funding_breakout", "liquidation_reversal"])
def test_missing_derivative_feed_is_reported(sid):
    assert fs.analyze(sid, _df())["missing"] == []
    assert fs.analyze(sid, _df(derivatives=False))["missing"]
    assert fs.analyze("donchian_breakout", _df(derivatives=False))["missing"] == []


def test_sizing_held_only_fills_bars_before_the_first_event():
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    none = pd.Series(False, index=idx)
    fc = fs._events(idx, pd.Series(0.0, index=idx))
    assert (fs._sizing(fc, held=-1.0)["position"] == -1.0).all()   # no event: keep holding
    long = none.copy()
    long.iloc[2] = True
    fc = fs._events(idx, pd.Series(0.0, index=idx), long=long)
    assert fs._sizing(fc, held=-1.0)["position"].tolist() == [-1.0, -1.0, 1.0, 1.0, 1.0]
    assert fs._sizing(fc)["position"].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


# ---------------------------------------------------------------- 3. 提交物
def test_registry_covers_every_builtin_and_is_valid():
    assert set(ENTRIES) == set(fs.FRAMEWORK_STRATEGIES)
    ids = [e["strategyId"] for e in ENTRIES.values()]
    assert len(set(ids)) == len(ids)
    for entry in ENTRIES.values():
        builder.validate_entry(entry)
        assert entry["version"] == "r1"
        assert entry["description"] and "\n" not in entry["description"]


def test_long_only_builtins_are_tagged_as_such():
    df = _df()
    for sid, entry in ENTRIES.items():
        shorts = fs.FRAMEWORK_STRATEGIES[sid](df)[1].any()
        assert ("只做多" in entry["tags"]) == (not shorts), sid
        assert ("只做多" in entry["name"]) == (not shorts), sid


def test_committed_artefacts_are_in_sync_with_the_generator():
    stale = [str(p.relative_to(REPO)) for p, text in builder.artefacts().items()
             if not p.exists() or p.read_text(encoding="utf-8") != text]
    assert not stale, f"regenerate with `python scripts/build_square_payloads.py`: {stale}"


class _Ctx:
    def __init__(self):
        self.state, self.logs = {}, []

    def log(self, level, event, data):
        assert level in {"INFO", "WARN", "ERROR"} and isinstance(data, dict)
        self.logs.append((level, event, data))


def _load_code(entry, calls, feeds):
    """Exec a generated code.py with the platform runtime replaced by recording stubs."""
    ctx = _Ctx()

    def cap(name):
        def fn(**kwargs):                     # platform capabilities are synchronous
            calls.append((name, kwargs))
            return feeds.get(name)
        return fn

    runtime = types.ModuleType("binance.strategy.runtime")
    runtime.ctx = ctx
    runtime.node = lambda *a, **k: (lambda f: f)
    runtime.workflow = lambda f: f
    data = types.ModuleType("binance.strategy.node.capabilities.data")
    execution = types.ModuleType("binance.strategy.node.capabilities.execution")
    for name in ("klines", "open_interest_hist", "funding_rate_history", "liquidation_orders"):
        setattr(data, name, cap(name))
    for name in ("futures_account_config", "futures_close_position",
                 "futures_open_position", "notify"):
        setattr(execution, name, cap(name))
    mods = {"binance": types.ModuleType("binance"),
            "binance.strategy": types.ModuleType("binance.strategy"),
            "binance.strategy.node": types.ModuleType("binance.strategy.node"),
            "binance.strategy.node.capabilities": types.ModuleType("binance.strategy.node.capabilities"),
            "binance.strategy.node.capabilities.data": data,
            "binance.strategy.node.capabilities.execution": execution,
            "binance.strategy.runtime": runtime}
    saved = {k: sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    try:
        ns = {"__name__": f"square_{entry['strategyId']}"}
        path = builder.SQUARE_DIR / entry["strategyId"] / "code.py"
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return ns, ctx


@pytest.mark.parametrize("sid", SIDS)
def test_generated_code_reproduces_repo_signals(sid):
    ns, _ = _load_code(ENTRIES[sid], [], {})
    df = _df(seed=11)
    p = ns["PARAMS"]
    assert p == fs.strategy_defaults(sid)
    size = ns["_sizing"](ns["_forecast"](ns["_factors"](df, p), p))
    long, short = fs.FRAMEWORK_STRATEGIES[sid](df)
    pd.testing.assert_series_equal(size["long"], long)
    pd.testing.assert_series_equal(size["short"], short)
    for n in (120, 333, len(df)):            # _analyze = last bar of the same three stages
        out = ns["_analyze"]("BTCUSDT", df.iloc[:n], p)
        l, s = fs.FRAMEWORK_STRATEGIES[sid](df.iloc[:n])
        assert out["target_position"] == int(l.iloc[-1]) - int(s.iloc[-1])


def _kline_rows(df):
    """Shape of the platform ``klines`` return: {"close": [...], "rows": [...]}."""
    ms = df.index.asi8 // 1_000_000
    rows = [[int(t), str(o), str(h), str(l), str(c), str(v), int(t) + 3_599_999]
            for t, o, h, l, c, v in zip(ms, df["open"], df["high"], df["low"], df["close"],
                                        df["volume"])]
    return {"close": [float(c) for c in df["close"]], "rows": rows}


@pytest.mark.parametrize("sid", SIDS)
def test_generated_workflow_runs_and_trades_on_a_target_change(sid):
    df = _df(n=300, seed=5)
    ms = df.index.asi8 // 1_000_000
    feeds = {"klines": _kline_rows(df),
             "open_interest_hist": [{"timestamp": int(t), "sumOpenInterest": str(1e4 + i)}
                                    for i, t in enumerate(ms)],
             "funding_rate_history": [{"fundingTime": int(t), "fundingRate": "0.0001"}
                                      for t in ms[::8]],
             "liquidation_orders": [{"time": int(ms[-1]) + 60_000, "side": "SELL",
                                     "price": "50000", "origQty": "10"}]}
    calls = []
    ns, ctx = _load_code(ENTRIES[sid], calls, feeds)
    asyncio.run(ns["execute_strategy"]())
    signal = ctx.state["signal_engine"]
    assert ctx.state["fetch_klines"] is feeds["klines"]          # raw return, keyed by node id
    assert signal["rebalance_needed"] == (signal["target_position"] != 0)
    if signal["target_position"] != 0:
        assert ctx.state["rebalance"]["changed"] and "message" in ctx.state["notify_signal"]
    assert signal["missing"] == []
    assert signal["verdict"] in {"LONG", "SHORT", "FLAT", "KEEP"}
    kl = next(kw for name, kw in calls if name == "klines")
    assert kl == {"symbol": "BTCUSDT", "timeframe": "1h", "limit": builder.KLINE_LIMIT,
                  "market_type": "futures", "closed_only": True}
    opened = [kw for name, kw in calls if name == "futures_open_position"]
    if signal["target_position"] != 0:
        assert opened and opened[0]["side"] in {"BUY", "SELL"}
        assert opened[0]["side"] == ("BUY" if signal["target_position"] > 0 else "SELL")
        assert set(opened[0]) == {"venue_class", "instrument", "size", "side", "order_type"}
        assert opened[0]["venue_class"] == "um" and opened[0]["order_type"] == "MARKET"
        assert ctx.state["position"] == signal["target_position"]
    else:
        assert not opened
    if sid == "liquidation_reversal":
        assert signal["target_position"] == 1       # a 500k long-liquidation spike on the last bar


@pytest.mark.parametrize("sid", ["oi_funding_breakout", "liquidation_reversal"])
def test_generated_code_refuses_to_trade_without_its_feed(sid):
    df = _df(n=200, seed=5)
    calls = []
    ns, ctx = _load_code(ENTRIES[sid], calls, {"klines": _kline_rows(df)})
    with pytest.raises(RuntimeError, match="missing feeds"):
        asyncio.run(ns["execute_strategy"]())
    assert not [c for c in calls if c[0].startswith("futures_")]
    assert ctx.logs and ctx.logs[0][:2] == ("WARN", "missing_feeds")


@pytest.mark.parametrize("sid", SIDS)
def test_spec_nodes_and_edges_match_code_workflow(sid):
    entry = ENTRIES[sid]
    folder = builder.SQUARE_DIR / entry["strategyId"]
    spec = yaml.safe_load((folder / "spec.yaml").read_text(encoding="utf-8"))
    code = (folder / "code.py").read_text(encoding="utf-8")
    assert spec["strategy"]["id"] == entry["strategyId"]
    assert spec["strategy"]["version"] == entry["specVersion"] == "1.0"
    assert 'version: "1.0"' in (folder / "spec.yaml").read_text(encoding="utf-8")
    assert spec["trigger"] == {"type": "schedule", "config": {"interval": entry["interval"]}}
    node_fns = re.findall(r'^@node\([^)]*\)\nasync def (\w+)\(', code, flags=re.M)
    ids = [n["id"] for n in spec["nodes"]]
    assert sorted(ids) == sorted(node_fns)
    workflow = code.split("@workflow", 1)[1].split("\n\n\n", 1)[0]
    for node_id in ids:
        assert f"{node_id}(" in workflow, f"{node_id} is not called by the workflow"
        assert f'ctx.state["{node_id}"] = ' in workflow, f"{node_id} is not written to ctx.state"
    assert '{"output":' not in code
    for n in spec["nodes"]:
        if n["type"] == "data":
            by_key = {p["key"]: p for p in n["params"]}
            for p in n["params"]:
                if p["key"] in ("symbol", "timeframe", "period", "limit"):
                    assert p.get("label") and p.get("widget") in {"text", "number"}, p
            assert "min" in by_key["limit"] and "max" in by_key["limit"]
    for cond in [n["condition"] for n in spec["nodes"] if "condition" in n]:
        for ref in re.findall(r"state\.(\w+)", cond):
            assert ref in ids, f"condition refers to non-node state {ref!r}"
    for n in spec["nodes"]:
        assert n["type"] in {"data", "analysis", "custom", "execution"}
        if n["type"] in ("data", "execution"):
            assert re.search(rf"\b{n['function']}\b", code)
    for e in spec["edges"]:
        assert e["from"] in ids and e["to"] in ids
    custom = next(n for n in spec["nodes"] if n["id"] == "signal_engine")
    assert {p["key"]: p["value"] for p in custom["params"]} == fs.strategy_defaults(sid)
    for fn in ("def _factors(", "def _forecast(", "def _sizing(", "def _analyze("):
        assert fn in custom["code"] and fn in code


def test_icons_are_short_identifiers():
    for entry in ENTRIES.values():
        assert re.fullmatch(r"[a-z][a-z0-9_]*", entry["icon"]), entry["icon"]


def test_payload_strategy_id_override(tmp_path):
    assert builder.main(["--payload-dir", str(tmp_path),
                         "--strategy-id", "st_rsi_reversion=st_btc_rsi_1h_0123abcd"]) == 0
    rsi = json.loads((tmp_path / "st_rsi_reversion.json").read_text(encoding="utf-8"))
    assert rsi["strategyId"] == "st_btc_rsi_1h_0123abcd"
    assert yaml.safe_load(rsi["spec"])["strategy"]["id"] == "st_rsi_reversion"
    other = json.loads((tmp_path / "st_donchian_breakout.json").read_text(encoding="utf-8"))
    assert other["strategyId"] == "st_donchian_breakout"
    with pytest.raises(SystemExit):
        builder.main(["--payload-dir", str(tmp_path), "--strategy-id", "st_nope=x"])


def test_payloads_carry_every_submit_field(tmp_path):
    assert builder.main(["--payload-dir", str(tmp_path)]) == 0
    files = sorted(tmp_path.glob("*.json"))
    assert len(files) == len(SIDS)
    for f in files:
        payload = json.loads(f.read_text(encoding="utf-8"))
        assert set(payload) == {"strategyId", "version", "spec", "code", "description", "tags",
                                "shareLevel", "freeFork", "icon"}
        assert f.stem == payload["strategyId"]
        assert isinstance(payload["spec"], str) and isinstance(payload["code"], str)
        assert payload["version"] == "r1"
        assert re.fullmatch(r"[a-z][a-z0-9_]*", payload["icon"])
        assert yaml.safe_load(payload["spec"])["strategy"]["id"] == payload["strategyId"]
        compile(payload["code"], f.name, "exec")


@pytest.mark.parametrize("sid", ["donchian_breakout", "multi_timeframe_ma_spread"])
def test_generated_code_flips_by_closing_then_opening_with_buy_sell(sid):
    df = _df(n=300, seed=5)
    probe, _ = _load_code(ENTRIES[sid], [], {})
    target = probe["_analyze"]("BTCUSDT", df, probe["PARAMS"])["target_position"]
    assert target != 0
    calls = []
    ns, ctx = _load_code(ENTRIES[sid], calls, {"klines": _kline_rows(df)})
    ctx.state["position"] = -target                  # hold the opposite side
    asyncio.run(ns["execute_strategy"]())
    trades = [(name, kw) for name, kw in calls if name.startswith("futures_")]
    assert [name for name, _ in trades] == ["futures_close_position", "futures_open_position"]
    assert trades[1][1]["side"] in {"BUY", "SELL"}
    assert trades[1][1]["side"] == ("BUY" if target > 0 else "SELL")
    assert ctx.state["position"] == target


@pytest.mark.parametrize("sid", SIDS)
def test_generated_code_logs_with_level_event_payload(sid):
    import ast
    code = (builder.SQUARE_DIR / ENTRIES[sid]["strategyId"] / "code.py").read_text(encoding="utf-8")
    logs = [n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == "log"
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "ctx"]
    assert logs
    for call in logs:
        assert len(call.args) == 3 and not call.keywords
        assert call.args[0].value in {"INFO", "WARN", "ERROR"}
        assert isinstance(call.args[2], ast.Dict)


def test_main_loop_logs_a_failed_round_and_keeps_going():
    calls = []
    ns, ctx = _load_code(ENTRIES["rsi_reversion"], calls, {"klines": None})

    async def once():
        task = asyncio.ensure_future(ns["main"]())
        for _ in range(20):
            await asyncio.sleep(0)
            if ctx.logs:
                break
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    ns["INTERVAL_SEC"] = 0
    asyncio.run(once())
    assert ctx.logs[0][:2] == ("ERROR", "strategy_round_error")


@pytest.mark.parametrize("sid", SIDS)
def test_generated_code_never_awaits_a_capability(sid):
    import ast
    code = (builder.SQUARE_DIR / ENTRIES[sid]["strategyId"] / "code.py").read_text(encoding="utf-8")
    awaited = [n.value.func.id for n in ast.walk(ast.parse(code))
               if isinstance(n, ast.Await) and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Name)]
    assert not set(awaited) & set(builder.CAPABILITIES), awaited
    called = {n.func.id for n in ast.walk(ast.parse(code))
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "klines" in called
