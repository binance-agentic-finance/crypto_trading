"""Demo server: 自然語言 → (LLM/LiteLLM) → YAML → 回測 → PnL vs BTC 基準.

這是一個「單純用語言模型把自然語言轉成策略 YAML 再跑回測」的展示,不是 agent。
瀏覽器 → 本後端 → LiteLLM(OpenAI 相容 /chat/completions)。API key 只留在本機。

啟動:
    PYTHONPATH=<repo_root> <venv>/bin/python docs/strategy_yaml_spec/demo/server.py
    # 然後開 http://127.0.0.1:8799
"""

from __future__ import annotations

import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# --- 讓後端找得到 cyqnt_trd(repo root = 這個檔往上三層)---
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
SPEC_DIR = HERE.parents[1]                      # docs/strategy_yaml_spec
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import requests
import yaml

from cyqnt_trd.standard_bot.yaml_pipeline import build_make_signals, validate_spec
from cyqnt_trd.standard_bot.simulation.vectorized_backtest import run_vectorized_backtest

PORT = 8799
SCHEMA_PATH = SPEC_DIR / "strategy.schema.yaml"
FIXTURE_DIR = REPO_ROOT / "tests" / "blocks" / "fixtures"


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

BLOCKS_CHEATSHEET = """\
可用 indicators(input=close 或自動 df):
  indicators.ema(series,period) indicators.sma(series,period) indicators.rsi(series,period)
  indicators.atr(df,period) indicators.adx(df,period)->tuple(取 output:0=adx)
  indicators.macd(series,fast_period,slow_period,signal_period)->tuple(0=macd線,1=signal線,2=hist)
可用 conditions(回傳 bool):
  conditions.ma_cross_above(fast,slow) conditions.ma_cross_below(fast,slow)
  conditions.rsi_in_range(rsi,low,high) conditions.rsi_overbought(rsi,threshold) conditions.rsi_oversold(rsi,threshold)
  conditions.adx_trending(adx,threshold)
  conditions.breakout_high(df,lookback) conditions.breakout_low(df,lookback)
  conditions.macd_golden_cross(macd_line,signal_line) conditions.macd_death_cross(macd_line,signal_line)
組合器(可任意巢狀):{all_of:[...]} {any_of:[...]} {not: <node>}  葉節點:{cond:"conditions.xxx", args:[...], params:{...}}
出場 risk.exit.type:pct_stop_tp{stop_pct,tp_pct,max_bars} / atr_stop_tp{atr_period,stop_mult,tp_mult,max_bars} / time_only{max_bars} / opposite_signal{max_bars}
"""

EXAMPLE_YAML = """\
spec_version: "1.0"
target: standard_bot
strategy:
  id: btc_ema_rsi_1h
  description: "EMA 交叉 + RSI 過濾"
run:
  mode: backtest
data:
  symbol: BTCUSDT
  market_type: futures
  primary: { interval: "1h", poll_interval: 3570 }
  source: { type: binance_rest }
signals:
  indicators:
    ema_fast: { block: indicators.ema, input: close, params: { period: 12 } }
    ema_slow: { block: indicators.ema, input: close, params: { period: 26 } }
    rsi14:    { block: indicators.rsi, input: close, params: { period: 14 } }
  entry:
    long:
      all_of:
        - { cond: conditions.ma_cross_above, args: [ema_fast, ema_slow] }
        - not: { cond: conditions.rsi_overbought, args: [rsi14], params: { threshold: 75 } }
    short:
      all_of:
        - { cond: conditions.ma_cross_below, args: [ema_fast, ema_slow] }
        - not: { cond: conditions.rsi_oversold, args: [rsi14], params: { threshold: 25 } }
sizing: { size: 0.95 }
risk:
  exit: { type: pct_stop_tp, stop_pct: 0.02, tp_pct: 0.04, max_bars: 96 }
  fees: { commission_bps: 4.0, slippage_bps: 2.0 }
backtest: { initial_capital: 10000, execution_model: next_bar_open }
"""


def build_system_prompt() -> str:
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    return (
        "你是一個把「自然語言交易策略描述」轉成 YAML 規格的轉換器。"
        "只輸出一份合法 YAML,不要有 markdown 圍欄(```)、不要解釋、不要多餘文字。\n\n"
        "必須嚴格遵守下面的 schema 與可用 block 清單;只能使用清單內的 block,參數名要精確。\n"
        "不要使用 data.htf(單一時間框即可)。若使用者沒指定,採保守預設:"
        "market_type=futures、interval=1h、fees commission_bps=4 slippage_bps=2、size=0.95、"
        "出場預設 pct_stop_tp{stop_pct:0.02,tp_pct:0.04,max_bars:96}。\n"
        "若使用者語意含『下穿做空 / 空方』就同時給 entry.long 與 entry.short;否則 long-only(只給 long)。\n\n"
        "=== SCHEMA ===\n" + schema + "\n\n"
        "=== 可用 BLOCKS ===\n" + BLOCKS_CHEATSHEET + "\n\n"
        "=== 範例輸出 ===\n" + EXAMPLE_YAML
    )


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        # drop first fence line (``` or ```yaml) and trailing fence
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def call_llm(api_base: str, api_key: str, model: str, nl: str) -> str:
    base = api_base.rstrip("/")
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": nl},
        ],
    }
    resp = requests.post(url, headers=headers, json=body, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return _strip_fences(content)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_KLINES_HOST = {"spot": "https://api.binance.com/api/v3/klines",
                "futures": "https://fapi.binance.com/fapi/v1/klines"}
_FIXTURES = {
    ("BTCUSDT", "1h"): "BTCUSDT_1h_500bars.parquet",
    ("BTCUSDT", "4h"): "BTCUSDT_4h_300bars.parquet",
    ("BTCUSDT", "15m"): "BTCUSDT_15m_500bars.parquet",
    ("ETHUSDT", "1h"): "ETHUSDT_1h_500bars.parquet",
}


def _fixture_df(symbol: str, interval: str):
    name = _FIXTURES.get((symbol.upper(), interval))
    if not name:
        return None
    p = FIXTURE_DIR / name
    if not p.exists():
        return None
    return pd.read_parquet(p).reset_index(drop=True)


def fetch_klines(symbol: str, interval: str, market_type: str, limit: int = 1000):
    """Return an OHLCV DataFrame. Try Binance public REST, fall back to fixture."""
    host = _KLINES_HOST.get(market_type, _KLINES_HOST["futures"])
    try:
        r = requests.get(host, params={"symbol": symbol.upper(), "interval": interval,
                                       "limit": min(limit, 1000)}, timeout=20)
        r.raise_for_status()
        rows = r.json()
        df = pd.DataFrame({
            "open_time": [int(k[0]) for k in rows],
            "open": [float(k[1]) for k in rows],
            "high": [float(k[2]) for k in rows],
            "low": [float(k[3]) for k in rows],
            "close": [float(k[4]) for k in rows],
            "volume": [float(k[5]) for k in rows],
            "close_time": [int(k[6]) for k in rows],
            "quote_volume": [float(k[7]) for k in rows],
        })
        df["timestamp"] = df["close_time"]
        return df, "binance_%s" % market_type
    except Exception as exc:
        fx = _fixture_df(symbol, interval)
        if fx is not None:
            fx["timestamp"] = fx["close_time"]
            return fx, "fixture(offline: %s)" % type(exc).__name__
        raise RuntimeError(f"無法取得 {symbol} {interval} 行情:{exc}") from exc


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


def run_backtest(yaml_text: str) -> dict:
    spec = yaml.safe_load(yaml_text)
    if not isinstance(spec, dict):
        return {"ok": False, "error": "YAML 不是有效的 mapping"}
    errors, warnings = validate_spec(spec)
    if errors:
        return {"ok": False, "error": "spec 驗證失敗", "errors": errors, "warnings": warnings}

    data = spec.get("data") or {}
    symbol = data["symbol"].upper()
    interval = (data.get("primary") or {})["interval"]
    market_type = data.get("market_type", "futures")
    entry = (spec.get("signals") or {}).get("entry") or {}
    exit_cfg = (spec.get("risk") or {}).get("exit")
    fees = (spec.get("risk") or {}).get("fees") or {}
    size = float((spec.get("sizing") or {}).get("size", 0.95))
    initial_capital = float((spec.get("backtest") or {}).get("initial_capital", 10000.0))
    long_only = (market_type == "spot") or (not entry.get("short"))

    df, source = fetch_klines(symbol, interval, market_type)
    if df is None or len(df) < 50:
        return {"ok": False, "error": "行情資料不足(需 ≥ 50 根)"}

    make_signals = build_make_signals(spec)
    result = run_vectorized_backtest(
        df=df, signal_fn=make_signals, exit_cfg=exit_cfg, timeframe=interval,
        size=size, fee_bps=float(fees.get("commission_bps", 4.0)),
        slippage_bps=float(fees.get("slippage_bps", 2.0)),
        initial_capital=initial_capital, long_only=long_only,
    )

    equity = result.equity_curve
    if equity is None:
        equity = np.full(len(df), initial_capital, dtype=float)
    equity = np.asarray(equity, dtype=float)

    # --- BTC buy-and-hold 基準(同區間,initial_capital 全押)---
    if symbol == "BTCUSDT":
        btc_close = df["close"].to_numpy(dtype=float)
    else:
        btc_df, _ = fetch_klines("BTCUSDT", interval, market_type, limit=len(df))
        btc_close = btc_df["close"].to_numpy(dtype=float)
    m = min(len(equity), len(btc_close), len(df))
    equity = equity[-m:]
    btc_close = btc_close[-m:]
    baseline = initial_capital * (btc_close / btc_close[0])
    ts = df["close_time"].to_numpy()[-m:].astype("int64").tolist()

    # 下採樣避免傳太大(圖用)
    def _ds(arr, k=600):
        arr = list(arr)
        if len(arr) <= k:
            return arr
        step = len(arr) / k
        return [arr[int(i * step)] for i in range(k)]

    bh_return = float(baseline[-1] / baseline[0] - 1.0)

    # --- 交易加上實際時間(entry_idx/exit_idx → close_time → 可讀時間)---
    from datetime import datetime, timezone

    def _iso(ms):
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

    close_times = df["close_time"].to_numpy()
    enriched_trades = []
    for t in (result.trades[-8:] if result.trades else []):
        nt = {}
        ei, xi = t.get("entry_idx"), t.get("exit_idx")
        if ei is not None and 0 <= ei < len(close_times):
            nt["entry_time"] = _iso(close_times[ei])
        if xi is not None and 0 <= xi < len(close_times):
            nt["exit_time"] = _iso(close_times[xi])
        nt.update(t)
        enriched_trades.append(nt)

    return {
        "ok": True,
        "symbol": symbol, "interval": interval, "market_type": market_type,
        "bars": int(m), "data_source": source,
        "period": {"start": _iso(ts[0]), "end": _iso(ts[-1])} if ts else None,
        "metrics": {
            "total_return": result.total_return,
            "total_pnl": result.total_pnl,
            "final_equity": result.final_equity if result.final_equity != 1.0 else float(equity[-1]),
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "trade_count": result.trade_count,
            "avg_trade_pnl": result.avg_trade_pnl,
            "exposure": result.exposure,
        },
        "baseline": {
            "label": "BTC buy & hold",
            "total_return": bh_return,
            "total_pnl": float(baseline[-1] - baseline[0]),
            "final_equity": float(baseline[-1]),
        },
        "chart": {
            "timestamps": _ds(ts),
            "strategy": _ds(equity),
            "baseline": _ds(baseline),
            "initial_capital": initial_capital,
        },
        "trades_sample": enriched_trades,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 馬上抓資料 / 產生訊號 / 選幣
# ---------------------------------------------------------------------------


def fetch_live_bundle(symbol: str, interval: str, limit: int = 300) -> dict:
    """Fetch every catalog node live and return one ``cyqnt.input/v1`` dict.

    This is the input half of the demo. It matters that it is ONE call producing
    ONE artifact: the same ``build_live_bundle`` that paper/live uses, gated once
    on ``available_time``, with a status for every source it was asked for —
    including the ones that came back empty.
    """
    import time

    from cyqnt_trd.standard_bot.data.live_bundle import build_live_bundle

    started = time.time()
    bundle = build_live_bundle(symbol=symbol.upper(), interval=interval, limit=limit)
    elapsed = time.time() - started

    raw = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    frames = bundle.get("frames") or {}

    # Skeleton = everything except the rows, so the shape is readable at a glance.
    skeleton = {k: v for k, v in bundle.items() if k != "frames"}
    skeleton["frames"] = {
        key: "{ shape: %s, rows: [ … %d 列 … ] }" % (spec.get("shape"), len(spec.get("rows") or []))
        for key, spec in frames.items()
    }

    table = [
        {"node": key,
         "shape": spec.get("shape"),
         "rows": len(spec.get("rows") or []),
         "status": str((bundle.get("source_status") or {}).get(key, "")),
         # one real row — "here is the format" says nothing without a value in it
         "sample": (spec.get("rows") or [None])[-1]}
        for key, spec in frames.items()
    ]
    # Sources that were asked for and produced no frame at all still belong in the
    # report: "not fetched" and "fetched and empty" are different failures.
    for key, status in (bundle.get("source_status") or {}).items():
        if key not in frames:
            table.append({"node": key, "shape": "-", "rows": 0,
                          "status": str(status), "sample": None})

    return {
        "ok": True,
        "schema": bundle.get("schema"),
        "decision_time": bundle.get("decision_time"),
        "decision_time_basis": bundle.get("decision_time_basis"),
        "symbol": symbol.upper(), "interval": interval,
        "elapsed_sec": round(elapsed, 2),
        "bytes": len(raw.encode("utf-8")),
        "node_count": len(bundle.get("source_status") or {}),
        "row_total": sum(len(s.get("rows") or []) for s in frames.values()),
        "warnings": bundle.get("warnings") or [],
        "skeleton": skeleton,
        "table": sorted(table, key=lambda r: -r["rows"]),
    }


def _spec_from_yaml(yaml_text: str):
    """Parse + validate, returning ``(spec, error_payload)``."""
    spec = yaml.safe_load(yaml_text)
    if not isinstance(spec, dict):
        return None, {"ok": False, "error": "YAML 不是有效的 mapping"}
    errors, warnings = validate_spec(spec)
    if errors:
        return None, {"ok": False, "error": "spec 驗證失敗",
                      "errors": errors, "warnings": warnings}
    return spec, None


def _row_count(frame) -> int:
    """Rows in a frame that may be None. ``len()`` only, never truthiness."""
    return 0 if frame is None else int(len(frame))


def _snapshot(*, market=None, universe=None, as_of: int):
    from cyqnt_trd.standard_bot.core import DataSnapshot, SnapshotMeta

    return DataSnapshot(
        version="demo/v1", market=market, universe=universe,
        meta=SnapshotMeta(snapshot_id="demo-%d" % as_of, assembled_at=as_of,
                          decision_as_of=as_of, partial_ok=True),
    )


def make_signal(yaml_text: str) -> dict:
    """YAML → the latest ``cyqnt.signal/v2`` signal.

    The backtest answers "would this have made money". This answers "what does
    the strategy say RIGHT NOW, in the format a consumer receives" — which is the
    part of the contract a downstream team actually has to implement against.
    """
    spec, bad = _spec_from_yaml(yaml_text)
    if bad:
        return bad
    if isinstance(spec.get("selection"), dict):
        return {"ok": False, "error": "這是選幣 spec,請用下面的『執行選幣』"}

    from cyqnt_trd.blocks import data as blocks_data
    from cyqnt_trd.blocks import strategy as blocks_strategy
    from cyqnt_trd.standard_bot.adapter import envelope_to_signal
    from cyqnt_trd.standard_bot.core import MarketBundle

    data = spec.get("data") or {}
    symbol = str(data["symbol"]).upper()
    interval = str((data.get("primary") or {})["interval"])
    market_type = data.get("market_type", "futures")

    df, source = fetch_klines(symbol, interval, market_type)
    bars = blocks_data.df_to_bars(df, instrument_id=symbol, timeframe=interval)
    as_of = int(bars[-1].timestamp)

    # build_plugin, not register: a demo click must not mutate the process-wide
    # registry, and repeated clicks would pile up registrations.
    plugin = blocks_strategy.build_plugin(
        spec["strategy"]["id"], build_make_signals(spec),
        exit_cfg=(spec.get("risk") or {}).get("exit"),
        size=float((spec.get("sizing") or {}).get("size", 1.0)),
    )
    market = MarketBundle(bars={MarketBundle.key(symbol, interval): bars})
    result = plugin.step(_snapshot(market=market, as_of=as_of), {},
                         {"instrument_id": symbol, "timeframe": interval})
    envelopes = getattr(result, "signals", result) or []
    if not envelopes:
        return {"ok": True, "signal": None, "source": source, "bars": len(df),
                "as_of": as_of,
                "note": "最後一根沒有觸發訊號 —— 這是正常結果,不是錯誤。"
                        "策略大多數時間應該是不動作的。"}
    signal = envelope_to_signal(envelopes[-1])
    return {"ok": True, "source": source, "bars": len(df), "as_of": as_of,
            "envelope_version": envelopes[-1].version,
            "signal": signal.to_dict(), "key_count": len(signal.to_dict())}


def run_selection(yaml_text: str) -> dict:
    """Selection YAML → live universe → ranked basket → one v2 signal.

    Deliberately the same output schema as :func:`make_signal`: a consumer parses
    one contract and branches on ``kind``.
    """
    spec, bad = _spec_from_yaml(yaml_text)
    if bad:
        return bad
    if not isinstance(spec.get("selection"), dict):
        return {"ok": False, "error": "這不是選幣 spec(缺 selection:),請用上面的『產生訊號』"}

    import time

    from cyqnt_trd.blocks import strategy as blocks_strategy
    from cyqnt_trd.standard_bot.adapter import envelope_to_signal
    from cyqnt_trd.standard_bot.core import UniverseBundle
    from cyqnt_trd.standard_bot.runtime import data as data_runtime
    from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import build_selection_fn

    market_type = (spec.get("data") or {}).get("market_type", "futures")
    started = time.time()
    status, frames = {}, {}
    with data_runtime.session() as session:
        as_of = session.as_of_ms
        for node, params in (("universe", {"market_type": market_type}),
                             ("ticker_rank", {"window": "24h", "limit": 50})):
            try:
                frames[node] = session.call(node, **params)
                status[node] = "ok" if len(frames[node]) else "empty"
            except data_runtime.DataUnavailable as exc:
                frames[node] = None
                status[node] = "error: %s" % exc.reason
    elapsed = time.time() - started

    plugin = blocks_strategy.build_selection_plugin(
        spec["strategy"]["id"], build_selection_fn(spec), market_type=market_type)
    universe = UniverseBundle(as_of=as_of, universe=frames.get("universe"),
                              ticker_rank=frames.get("ticker_rank"))
    result = plugin.step(_snapshot(universe=universe, as_of=as_of), {},
                         {"market_type": market_type})
    envelope = (getattr(result, "signals", result) or [None])[0]
    if envelope is None:
        return {"ok": False, "error": "選幣沒有產出 envelope", "status": status}

    payload = envelope.payload or {}
    candidates = payload.get("candidates") or []
    out = {"ok": True, "status": status, "as_of": as_of,
           "elapsed_sec": round(elapsed, 2),
           "universe_size": payload.get("universe_size"),
           # `df or []` asks a DataFrame for its truth value, which raises.
           "universe_rows": _row_count(frames.get("universe")),
           "rank_rows": _row_count(frames.get("ticker_rank")),
           "candidates": candidates,
           "envelope_version": envelope.version}
    if not candidates:
        out["note"] = ("篩選後沒有候選。常見原因:ticker_rank 回空(Square 快取冷)"
                       "或門檻設太嚴。這是真實結果,不是程式錯誤。")
        return out
    signal = envelope_to_signal(envelope)
    out["signal"] = signal.to_dict()
    out["key_count"] = len(signal.to_dict())
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload, ctype="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if "json" in ctype or "html" in ctype else ""))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def log_message(self, fmt, *args):
        sys.stderr.write("[demo] " + (fmt % args) + "\n")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = (HERE.parent / "index.html").read_text(encoding="utf-8")
            return self._send(200, html.encode("utf-8"), ctype="text/html")
        if self.path == "/api/schema":
            return self._send(200, {"schema": SCHEMA_PATH.read_text(encoding="utf-8")})
        if self.path.startswith("/api/example"):
            name = "example_selection.yaml"
            if "kind=trade" in self.path:
                name = "example_multi_source.yaml"
            path = SPEC_DIR / name
            if not path.exists():
                return self._send(404, {"ok": False, "error": "找不到 %s" % name})
            return self._send(200, {"ok": True, "name": name,
                                    "yaml": path.read_text(encoding="utf-8")})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            if self.path == "/api/convert":
                b = self._read_json()
                if not b.get("nl", "").strip():
                    return self._send(400, {"ok": False, "error": "請輸入自然語言描述"})
                if not b.get("api_base") or not b.get("model"):
                    return self._send(400, {"ok": False, "error": "請填 LLM API Base URL 與 model"})
                yaml_text = call_llm(b["api_base"], b.get("api_key", ""), b["model"], b["nl"])
                # 順手驗證(讓前端知道能不能直接跑)
                try:
                    spec = yaml.safe_load(yaml_text)
                    errors, warnings = validate_spec(spec) if isinstance(spec, dict) else (["非 mapping"], [])
                except Exception as exc:
                    errors, warnings = [f"YAML 解析失敗:{exc}"], []
                return self._send(200, {"ok": True, "yaml": yaml_text,
                                        "valid": not errors, "errors": errors, "warnings": warnings})
            if self.path == "/api/backtest":
                b = self._read_json()
                return self._send(200, run_backtest(b.get("yaml", "")))
            if self.path == "/api/fetch":
                b = self._read_json()
                return self._send(200, fetch_live_bundle(
                    b.get("symbol", "BTCUSDT"), b.get("interval", "1h"),
                    int(b.get("limit", 300))))
            if self.path == "/api/signal":
                b = self._read_json()
                return self._send(200, make_signal(b.get("yaml", "")))
            if self.path == "/api/selection":
                b = self._read_json()
                return self._send(200, run_selection(b.get("yaml", "")))
            return self._send(404, {"ok": False, "error": "not found"})
        except Exception as exc:
            sys.stderr.write(traceback.format_exc())
            return self._send(200, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[demo] serving on http://127.0.0.1:{PORT}  (Ctrl+C to stop)")
    print(f"[demo] repo_root={REPO_ROOT}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[demo] bye")


if __name__ == "__main__":
    main()
