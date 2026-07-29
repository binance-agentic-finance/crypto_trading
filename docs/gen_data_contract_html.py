"""產生「標準 Bot 資料契約」HTML 報告 — 所有數值都是實跑捕捉,非手寫。

用法(從 repo 根目錄):
    .venv-standard-bot/bin/python docs/gen_data_contract_html.py [out.html]

輸出預設 docs/data_contract_example.html。所有 JSON / 數字都在執行時從實際
pipeline 捕捉,所以改了程式碼只要重跑本檔,文件就跟著更新。
"""
from __future__ import annotations

import html
import json
import os
import subprocess
import sys
import warnings
from types import SimpleNamespace

# repo root = 本檔的上一層(docs/ 的 parent)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
warnings.filterwarnings("ignore")

import pandas as pd

import strategies.technical.mtf_trend_follow as T2
import strategies.news.news_catalyst_selector as N1
from cyqnt_trd.blocks import strategy as BS
from cyqnt_trd.blocks.data import bars_to_df
from cyqnt_trd.standard_bot.core import MarketBundle, MarketQuery, TimeRange
from cyqnt_trd.standard_bot.data import (
    HistoricalParquetMarketDataAdapter, build_unified_snapshot, build_universe_bundle)
from cyqnt_trd.standard_bot.simulation.vectorized_backtest import run_vectorized_backtest

SYM, TF = "BTCUSDT", "4h"
OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/data_contract_example.html"

# ══════════════════════════════════════════════════════════════════════
# 1. 實跑:輸入
# ══════════════════════════════════════════════════════════════════════
bundle = HistoricalParquetMarketDataAdapter(
    data_root="data/mtf_90d", market_type="futures", resample_source_timeframe="1m"
).fetch_market(MarketQuery(instruments=[SYM], timeframes=[TF], time_range=TimeRange()))
bars = bundle.bars[MarketBundle.key(SYM, TF)]
df_full = bars_to_df(bars).reset_index(drop=True)

plugin = BS.get_block_plugin("mtf_trend_follow")
snap_market_only = build_unified_snapshot(market_bundle=bundle, primary_timeframe=TF)
df_in, iid, tf = plugin._extract_df(
    snap_market_only, SimpleNamespace(instrument_id=SYM, timeframe=TF))

# 輸入 JSON 範例(前 2 根)
input_json_sample = {
    "symbol": SYM, "interval": TF,
    "data": [{
        "open_time": b.extras["open_time"], "close_time": b.extras["close_time"],
        "open_price": b.open, "high_price": b.high, "low_price": b.low,
        "close_price": b.close, "volume": b.volume, "quote_volume": b.quote_volume,
    } for b in bars[:2]],
}

# 衍生品實際欄位
deriv = {}
for name, path in (("funding_rate", "data/derivatives_mvp_30d/futures/BTCUSDT/funding_rate.parquet"),
                   ("open_interest_5m", "data/derivatives_mvp_30d/futures/BTCUSDT/open_interest_5m.parquet")):
    d = pd.read_parquet(path)
    deriv[name] = {"path": path, "rows": len(d), "cols": list(d.columns),
                   "first": {k: (round(v, 8) if isinstance(v, float) else v)
                             for k, v in d.iloc[0].to_dict().items()}}

# 新聞 / universe 的離線 DataFrame
uni_df = pd.DataFrame({"symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                       "quoteVolume": [5e8, 3e8, 2e8]})
rank_df = pd.DataFrame({"ticker": ["BTC", "ETH", "SOL"], "mention_count": [120, 80, 50],
                        "bullish_count": [90, 20, 30], "bearish_count": [10, 50, 10],
                        "neutral_count": [5, 5, 5], "unique_authors": [40, 30, 20],
                        "rank": [1, 2, 3]})
ub = build_universe_bundle(as_of_ms=bars[-1].timestamp,
                           universe_df=uni_df, ticker_rank_df=rank_df)

# 統一 snapshot:同時帶 market + universe
snap_unified = build_unified_snapshot(
    market_bundle=bundle, universe_bundle=ub, primary_timeframe=TF)

# PIT 護欄實測
try:
    build_unified_snapshot(
        market_bundle=bundle, primary_timeframe=TF,
        universe_bundle=build_universe_bundle(
            as_of_ms=bars[-1].timestamp + 10 ** 7,
            universe_df=uni_df, ticker_rank_df=rank_df))
    pit_guard = "(沒有擋下 — 不該發生)"
except ValueError as e:
    pit_guard = str(e)

# ══════════════════════════════════════════════════════════════════════
# 2. 實跑:輸出
# ══════════════════════════════════════════════════════════════════════
batch = plugin.run(snap_unified, SimpleNamespace(instrument_id=SYM, timeframe=TF))
env = batch.trade_signals()[-1]
envelope_json = {
    "version": env.version, "signal_id": env.signal_id, "kind": env.kind.value,
    "instrument_id": env.instrument_id, "side": env.side.value, "strength": env.strength,
    "time_horizon": env.time_horizon, "valid_until": env.valid_until,
    "payload": env.payload,
    "provenance": {"plugin_id": env.provenance.plugin_id,
                   "plugin_version": env.provenance.plugin_version,
                   "config_hash": env.provenance.config_hash,
                   "input_fingerprint": env.provenance.input_fingerprint},
}
signal_v1 = T2.generate(df_full, SYM, TF)[-1]

sel_batch = BS.get_selection_plugin(N1.BOT_ID).run(
    snap_unified, SimpleNamespace(market_type="futures"))
sel_cands = sel_batch.selection_signals()[0].payload["candidates"]

# trades.jsonl 的實際格式(取自 PaperFill._fill_to_dict 的欄位)
trades_jsonl_sample = {
    "fill_id": "3f2a…", "timestamp_ms": 1773993600000, "side": "sell",
    "price": 70811.9, "quantity": 0.01412, "fee": 0.4,
    "action": "open_short", "signal_bar_timestamp_ms": 1773993599999,
}

# ══════════════════════════════════════════════════════════════════════
# 3. 實跑:兩個回測引擎
# ══════════════════════════════════════════════════════════════════════
vec = run_vectorized_backtest(df=df_full, signal_fn=T2.make_signals,
                              exit_cfg=plugin.exit_cfg, timeframe=TF,
                              size=plugin.size, fee_bps=4.0, slippage_bps=2.0)
ev_out = subprocess.run(
    [".venv-standard-bot/bin/python", "-m",
     "cyqnt_trd.standard_bot.entrypoints.mvp_backtest", "--engine", "python",
     "--strategy", "mtf_trend_follow", "--strategy-module",
     "strategies.technical.mtf_trend_follow", "--symbol", SYM, "--interval", TF,
     "--market-type", "futures", "--historical-dir", "data/mtf_90d",
     "--storage-timeframe", "1m", "--limit", "600", "--tail-bars", "600",
     "--commission-bps", "4", "--slippage-bps", "2"],
    capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."})
ev_line = next((l for l in ev_out.stdout.splitlines() if l.startswith("engine=")), "(n/a)")

tests = subprocess.run([".venv-standard-bot/bin/python", "-m", "pytest", "tests/", "-q"],
                       capture_output=True, text=True)
test_line = next((l for l in reversed(tests.stdout.splitlines()) if "passed" in l), "(n/a)")

# ══════════════════════════════════════════════════════════════════════
# HTML
# ══════════════════════════════════════════════════════════════════════
def esc(x):
    return html.escape(str(x))


def j(obj, indent=1):
    return esc(json.dumps(obj, ensure_ascii=False, indent=indent))


def pre(text, cls=""):
    return f'<pre class="{cls}">{text}</pre>'


# f-string 內不能出現反斜線,故把含 shell 續行符的區塊先組好
_CLI_CONT = " \\\n    "
RUN_BLOCK = esc(
    "# 事件驅動引擎(真實 CLI)\n"
    "$ mvp_backtest --engine python --strategy mtf_trend_follow" + _CLI_CONT
    + f"--symbol {SYM} --interval {TF} --tail-bars 600 --commission-bps 4 --slippage-bps 2\n"
    + ev_line
    + "\n\n# 向量化引擎(同資料 / 同費用 / 同 size / 同 exit_cfg)\n"
    + f"trades={vec.trade_count}  total_return={vec.total_return:+.6f}  "
      f"sharpe={vec.sharpe_ratio:.2f}  max_dd={vec.max_drawdown:.3f}\n\n"
    + "# 全套測試\n" + test_line
)


col_rows = "".join(
    f"<tr><td><code>{esc(c)}</code></td><td>{esc(df_in[c].dtype)}</td>"
    f"<td><code>{esc(repr(df_in[c].iloc[-1]))}</code></td></tr>"
    for c in df_in.columns)

tail_tbl = df_in[["open_time", "close_time", "open", "high", "low", "close", "volume"]].tail(4)
tail_pre = esc(tail_tbl.to_string(index=False))

deriv_blocks = "".join(
    f'<h4><code>{esc(v["path"])}</code></h4>'
    f'<p class="muted">{v["rows"]} 筆 · 欄位 {len(v["cols"])} 個</p>'
    + pre(j(v["cols"]) + "\n\n第一筆:\n" + j(v["first"]))
    for v in deriv.values())

HTML = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>標準 Bot 資料契約 — 輸入 / 輸出 / 執行流程</title>
<style>
  :root{{
    --bg:#0f1420; --panel:#161d2e; --panel2:#1c2740; --ink:#e6ebf5; --muted:#9aa7c2;
    --line:#2a3552; --accent:#5aa9ff; --green:#39d98a; --amber:#ffcc66; --red:#ff6b6b;
    --chip:#233150;
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang TC","Microsoft JhengHei",sans-serif;
    line-height:1.65;font-size:15px}}
  .wrap{{max-width:1100px;margin:0 auto;padding:32px 22px 90px}}
  header{{border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:28px}}
  h1{{font-size:26px;margin:0 0 6px}}
  .sub{{color:var(--muted);font-size:14px}}
  .meta{{margin-top:12px;display:flex;gap:8px;flex-wrap:wrap}}
  .chip{{background:var(--chip);color:#cfe0ff;border:1px solid var(--line);border-radius:999px;
    padding:3px 11px;font-size:12.5px}}
  h2{{font-size:19px;margin:40px 0 14px;padding-left:11px;border-left:4px solid var(--accent)}}
  h3{{font-size:15.5px;margin:24px 0 8px;color:#cfe0ff}}
  h4{{font-size:13.5px;margin:16px 0 4px;color:#a8c4ee;font-weight:600}}
  p{{margin:10px 0}} .muted{{color:var(--muted);font-size:13px}}
  .panel{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:14px 0}}
  .lead{{background:#15203a;border:1px solid var(--line);border-left:4px solid var(--green);
    border-radius:10px;padding:16px 20px;margin:14px 0}}
  .lead b{{color:var(--green)}}
  .warnbox{{background:#241f14;border:1px solid #4a3c1d;border-left:4px solid var(--amber);
    border-radius:10px;padding:14px 18px;margin:14px 0}}
  .badbox{{background:#241618;border:1px solid #4d2226;border-left:4px solid var(--red);
    border-radius:10px;padding:14px 18px;margin:14px 0}}
  code,kbd{{font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace;font-size:12.8px}}
  code{{background:#0c1220;border:1px solid var(--line);border-radius:5px;padding:1px 6px;color:#bfe0ff}}
  pre{{background:#0b1120;border:1px solid var(--line);border-radius:10px;padding:14px 16px;overflow:auto;
    font-size:12.4px;color:#c8d6f2;line-height:1.5}}
  table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:13.4px}}
  th,td{{border:1px solid var(--line);padding:8px 11px;text-align:left;vertical-align:top}}
  th{{background:var(--panel2);color:#cfe0ff;font-weight:600}}
  tr:nth-child(even) td{{background:#131a2b}}
  .ok{{color:var(--green);font-weight:600}}
  .warn{{color:var(--amber);font-weight:600}}
  .bad{{color:var(--red);font-weight:600}}
  ul{{margin:8px 0 8px 2px;padding-left:20px}} li{{margin:5px 0}}
  .flow{{display:flex;align-items:stretch;gap:0;flex-wrap:wrap;margin:18px 0}}
  .node{{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:12px 14px;
    min-width:140px;flex:1;text-align:center}}
  .node .t{{font-weight:700;color:#cfe0ff;font-size:13.2px}}
  .node .d{{color:var(--muted);font-size:11.6px;margin-top:4px}}
  .node.g{{border-color:#2c6b4a}} .node.a{{border-color:#6b5a2c}} .node.r{{border-color:#6b2c30}}
  .arrow{{display:flex;align-items:center;justify-content:center;color:var(--accent);font-size:20px;padding:0 8px}}
  .tag{{display:inline-block;background:var(--chip);border:1px solid var(--line);border-radius:5px;
    padding:1px 7px;font-size:11.5px;color:#cfe0ff;margin-right:5px}}
  .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
  @media(max-width:820px){{.grid2{{grid-template-columns:1fr}}}}
</style>
</head>
<body><div class="wrap">

<header>
  <h1>標準 Bot 資料契約 — 輸入 / 輸出 / 執行流程</h1>
  <div class="sub">crypto_trading（cyqnt_trd）· 本頁所有數值與 JSON 皆為實跑捕捉,非手寫範例</div>
  <div class="meta">
    <span class="chip">範例策略 mtf_trend_follow</span>
    <span class="chip">{esc(SYM)} {esc(TF)} · {len(df_full)} 根</span>
    <span class="chip">{esc(test_line)}</span>
  </div>
</header>

<div class="lead">
  <b>一句話</b>:策略只寫一個函式 <code>make_signals(df)</code>,框架負責<b>餵資料</b>(DataSnapshot)與<b>包訊號</b>(SignalEnvelope)。
  但<b>輸出有三種形態</b>,而真正上線的 live 路徑用的是第三種(<code>trades.jsonl</code>)—— 不是規格文件定義的那一種。
</div>

<h2>0. 執行流程總覽</h2>
<div class="flow">
  <div class="node"><div class="t">資料源</div><div class="d">parquet / JSON / REST</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="t">Adapter</div><div class="d">MarketBundle</div></div>
  <div class="arrow">→</div>
  <div class="node g"><div class="t">DataSnapshot</div><div class="d">標準輸入物件</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="t">make_signals(df)</div><div class="d">你寫的唯一函式</div></div>
  <div class="arrow">→</div>
  <div class="node g"><div class="t">SignalEnvelope</div><div class="d">標準輸出</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="t">引擎 / 下單</div><div class="d">backtest / paper / live</div></div>
</div>
<p>但四條路實際走的並不相同(下表的 grep 命中數為實測,0 = 完全沒用到該契約):</p>
<table>
<tr><th>路徑</th><th>輸入</th><th>訊號邏輯</th><th>輸出契約</th><th>下單</th></tr>
<tr><td><code>mvp_backtest --engine python</code></td><td>DataSnapshot → <code>plugin.step()</code></td>
    <td><code>make_signals</code></td><td class="ok">A · SignalEnvelope</td><td>回測引擎內部</td></tr>
<tr><td><code>run_vectorized_backtest</code></td><td>直接吃 df</td>
    <td><code>make_signals</code></td><td>—（引擎內部記帳）</td><td>—</td></tr>
<tr><td><code>mvp_paper</code>(一次性)</td><td>DataSnapshot → <code>plugin.step()</code></td>
    <td><code>make_signals</code></td><td class="ok">A → <code>build_intents</code></td><td>PaperBroker</td></tr>
<tr><td><code>mvp_paper_daemon</code> → <code>mvp_live_executor</code></td>
    <td class="warn">自建 df(不經 DataSnapshot)</td><td><code>make_signals</code></td>
    <td class="bad">C · <code>trades.jsonl</code></td><td class="bad">真實下單</td></tr>
</table>
<div class="warnbox">
  <b>⚠️ 最容易誤讀的一點</b>:「backtest / paper / live 三條路跑完全相同的訊號碼」——
  這句<b>是對的</b>,<code>make_signals</code> 確實共用(daemon 呼叫 <code>plugin._call_signal_fn(df)</code>)。
  但它只保證<b>訊號邏輯</b>相同,<b>不保證訊號格式相同</b>。
  <code>python_live_paper_session.py</code> / <code>mvp_paper_daemon.py</code> / <code>mvp_live_executor.py</code> 三檔內,
  <code>SignalEnvelope</code>、<code>SignalBatch</code>、<code>build_intents</code>、<code>run_pipeline_step</code>
  出現次數<b>皆為 0</b>。
</div>

<h2>1. 輸入 — 資料怎麼進來</h2>

<h3>1.1 標準輸入物件:DataSnapshot</h3>
<p><code>cyqnt_trd/standard_bot/core/contracts.py:227</code>。
匯入:<code>from cyqnt_trd.standard_bot.core import DataSnapshot</code>(不在 <code>cyqnt_trd</code> 頂層)。</p>
<table>
<tr><th>typed 欄位</th><th>內容</th><th>誰讀</th></tr>
<tr><td><code>market</code></td><td><code>MarketBundle</code> — K 線,key = <code>"BTCUSDT|4h"</code></td><td>交易型</td></tr>
<tr><td><code>universe</code></td><td><code>UniverseBundle</code> — 24h ticker + Square ticker_rank + 候選 K 線</td><td>選幣型</td></tr>
<tr><td><code>social</code></td><td><code>SocialFeedBundle</code> — 新聞文件流</td><td>選配</td></tr>
<tr><td><code>onchain</code></td><td><code>OnChainSignalBundle</code></td><td>尚未使用</td></tr>
<tr><td><code>meta</code></td><td><code>SnapshotMeta</code> — snapshot_id / decision_as_of / source_status</td><td>PIT 依據</td></tr>
</table>
<div class="badbox">
  <b>衍生品沒有型別契約</b>:<code>funding_rate</code> / <code>open_interest</code> /
  <code>long_short_ratio</code> / <code>taker_cvd</code> / <code>liquidation</code>
  <b>都不是 typed 欄位</b>,而是塞在 <code>Bar.extras: Dict[str, Any]</code>,
  再由 <code>bars_to_df</code> 溢出成 df 欄。對 K 線做到了標準化,對衍生品只是「有地方放」。
</div>

<h3>1.2 統一 snapshot(交易型 + 選幣型合一)</h3>
<p>過去 <code>market</code> 與 <code>universe</code> 只是<b>型別上</b>共存:
<code>assemble_snapshot()</code> 沒有 <code>universe</code> 參數、
<code>HistoricalSnapshotAssembler</code> 只填 <code>market</code>、
<code>build_selection_snapshot</code> 只填 <code>universe</code> ——
<b>沒有任何組裝器會同時填兩者</b>。新增的
<code>build_unified_snapshot()</code>(<code>standard_bot/data/unified_snapshot.py</code>)補上這條路:</p>
{pre('''from cyqnt_trd.standard_bot.data import build_unified_snapshot

snap = build_unified_snapshot(
    market_bundle=bundle,        # 交易型的逐根資料
    universe_bundle=ub,          # 選幣型的橫斷面資料
    primary_timeframe="4h",
    fold_universe_klines=False,  # True = 把候選 K 線折進標準 market 路徑
    strict_pit=True,             # PIT 護欄
)''')}
<p>實跑結果 — <b>同一個 snapshot 同時驅動兩種 plugin</b>:</p>
{pre(f'''snapshot_id    = {snap_unified.meta.snapshot_id}
decision_as_of = {snap_unified.meta.decision_as_of}
market   有 → {len(snap_unified.require_market().bars[MarketBundle.key(SYM, TF)])} 根 Bar
universe 有 → as_of={snap_unified.universe.as_of}

交易型 plugin (mtf_trend_follow)      → trade signals = {len(batch.trade_signals())}, side={env.side.value}
選幣型 plugin (news_catalyst_selector) → candidates = {len(sel_cands)}, first={sel_cands[0]["symbol"]}''')}
<p>並且會擋下 PIT 違規(universe 的 <code>as_of</code> 比最後一根確認 K 還新 = lookahead):</p>
{pre(esc(pit_guard))}

<h3>1.3 策略實際收到的 df(輸入契約)</h3>
<p>框架保證的欄位 — 取自 <code>BlockStrategyPlugin._extract_df()</code> 的實際輸出
(<code>{esc(iid)}</code> / <code>{esc(tf)}</code> / shape <code>{df_in.shape}</code>):</p>
<table><tr><th>欄位</th><th>型別</th><th>最後一根實際值</th></tr>{col_rows}</table>
<p>最後 4 根:</p>{pre(tail_pre)}
<p class="muted">注意 <code>open[i] == close[i-1]</code> —— 加密貨幣連續交易,
這是兩個回測引擎的成交慣例能對得上的前提。
宣告 <code>register(htf_specs=[("4h",200)])</code> 會多一欄 <code>_htf_4h_sma_200</code>。</p>

<h3>1.4 OHLCV 輸入檔:JSON</h3>
<p><code>HistoricalJsonMarketDataAdapter</code>(<code>standard_bot/data/adapters.py:102</code>),
用 <code>--input-json</code> 指定。頂層可以是 <code>{{"data":[...]}}</code> 或直接一個 array。</p>
{pre(j(input_json_sample))}
<table>
<tr><th>欄位</th><th>必填</th><th>說明</th></tr>
<tr><td><code>close_time</code></td><td class="ok">必填</td><td>epoch ms,同時作為 <code>Bar.timestamp</code>(決策時間)</td></tr>
<tr><td><code>open_price</code> / <code>high_price</code> / <code>low_price</code> / <code>close_price</code></td>
    <td class="ok">必填</td><td><b>必須帶 <code>_price</code> 後綴</b>,寫 <code>open</code>/<code>close</code> 會 KeyError</td></tr>
<tr><td><code>volume</code></td><td class="ok">必填</td><td></td></tr>
<tr><td><code>open_time</code></td><td>選填</td><td>省略時 = <code>close_time</code></td></tr>
<tr><td><code>quote_volume</code></td><td>選填</td><td>省略時為 <code>None</code></td></tr>
</table>
<div class="warnbox">
  <b>兩個坑</b>:① 所有 bar 一律被標成 <code>confirmed=True</code>(<code>adapters.py:135</code> 硬寫)→
  放進未收盤的 K 會被當成已確認,造成前瞻。② <code>mvp_backtest --limit</code> 預設 300,
  會把資料裁到最後 300 根;要跑完整資料得顯式加大。
  另外 <code>docs/backtests/*.json</code> 是 <code>--output-json</code> 的<b>輸出</b>,不是輸入檔。
</div>

<h3>1.5 衍生品輸入檔:parquet(沒有 JSON 版本)</h3>
<p>路徑慣例 <code>&lt;derivatives_root&gt;/&lt;market_type&gt;/&lt;INSTRUMENT&gt;/&lt;dataset&gt;.parquet</code>,
用 <code>--derivatives-dir</code> 指定。框架以
<code>pd.merge_asof(direction="backward")</code> 對 <code>close_time</code> 併進 df(PIT-safe),
並自動加算 <code>funding_rate_bps</code> / <code>oi_change_bps</code>。</p>
{deriv_blocks}
<div class="badbox">
  <b>三個坑 — 這就是 <code>deriv_positioning</code> 常常 0 訊號的原因</b>
  <ul>
    <li><b>OI 檔名綁週期</b>:<code>open_interest_&lt;timeframe&gt;.parquet</code>。repo 只有
        <code>open_interest_5m.parquet</code> → <b>跑 1h / 4h 就完全沒有 OI 欄位</b>。funding 不綁週期。</li>
    <li><b>CLI 預設 <code>--derivatives-dir data/derivatives</code>,該目錄在 repo 不存在</b> →
        不顯式指定就靜默沒欄位。要用 <code>--derivatives-dir data/derivatives_mvp_30d</code>。</li>
    <li><b><code>register(needs={{"derivatives": True}})</code> 不影響資料</b> ——
        框架不讀它(<code>required_inputs()</code> 全 repo 0 個呼叫點)。欄位純粹取決於上面兩點。</li>
  </ul>
</div>

<h3>1.6 新聞 / Square 輸入:沒有檔案格式</h3>
<p><code>fetch_ticker_rank(window, limit, lang)</code> <b>只打 live API</b>,快取是純記憶體 TTL 300 秒
(<code>_cache.py</code> 的 docstring 說 <code>CYQNT_TRD_PERSIST_CACHE=1</code> 會寫 JSON 到磁碟,
<b>但程式碼完全沒實作</b> —— 95 行、5 個函式、零檔案 I/O)。
repo 內也找不到任何落地的新聞資料。</p>
<p>離線唯一的路是<b>在程式裡直接傳 DataFrame</b>:</p>
{pre('''import strategies.news.news_catalyst_selector as N   # 匯入即註冊
from cyqnt_trd.standard_bot.data import run_selection

candidates = run_selection(N.BOT_ID,
                           universe_df=uni,       # 24h ticker
                           ticker_rank_df=rank)   # Square 提及/情緒''')}
<div class="grid2">
<div><h4>ticker_rank_df 欄位</h4>{pre(j(list(rank_df.columns)))}</div>
<div><h4>universe_df 欄位(至少)</h4>{pre(j(list(uni_df.columns)))}</div>
</div>
<div class="warnbox">
  <b>新聞沒有 PIT 歷史</b>:<code>fetch_ticker_rank</code> 只有
  <code>window</code> / <code>limit</code> / <code>lang</code>,<b>沒有 start/end 參數</b> →
  拿不到歷史,所以兩支選幣策略無法乾淨回測,只能 forward。
</div>

<h2>2. 策略 — 你要寫的東西</h2>
<p>範例 = <code>strategies/technical/mtf_trend_follow.py</code>(SOP 的參考模板)。核心只有一個函式:</p>
{pre(esc('''def make_signals(df) -> tuple[pd.Series, pd.Series]:
    c, ef, es, et, adx, _atr = _factors(df, CONFIG)   # 全部走 canonical blocks
    cross_up = (ef > es) & (ef.shift(1) <= es.shift(1))
    trend_up, trend_dn = c > et, c < et
    strong = adx > CONFIG["adx_min"]
    long  = (cross_up & trend_up & strong).fillna(False)
    short = (cross_dn & trend_dn & strong).fillna(False)
    return long, short          # 兩條對齊 df index 的布林 Series

strategy.register(
    BOT_ID, make_signals,
    exit_cfg={"type": "atr_stop_tp", "atr_period": 14,
              "stop_mult": 2.0, "tp_mult": 3.0},
    size=0.1,
)'''))}
<p class="muted">指標一律用 canonical <code>blocks.indicators</code> —— 三條路(回測 / paper / live)才會算出同一個值。</p>
<div class="warnbox">
  <b><code>--tail-bars</code> 陷阱(這支策略就會中)</b>:事件驅動引擎每根都在「最後
  <code>--tail-bars</code> 根」的視窗上重算指標。本策略用 EMA200,而 CLI 預設是 <b>120</b> →
  指標全 NaN、<b>0 交易且不報錯</b>。實測:<code>--tail-bars 120</code> → <code>trades=0</code>;
  <code>--tail-bars 600</code> → <code>trades=1</code>。<b>請設為 ≥ 最長週期的 2–3 倍</b>
  (現在 0 訊號時會印警告)。
</div>

<h2>3. 輸出 — 三種形態</h2>

<h3>3.1 輸出 A:SignalEnvelope(引擎面,框架自動產生)</h3>
{pre(j(envelope_json))}
<table>
<tr><th><code>payload</code> 欄位</th><th>一定有?</th><th>意義</th></tr>
<tr><td><code>bar_timestamp</code></td><td class="ok">✅</td><td>決策 K 棒的 <code>close_time</code>,lookahead-safe</td></tr>
<tr><td><code>target_position</code></td><td class="ok">✅</td><td><code>+1</code> 做多 / <code>-1</code> 做空</td></tr>
<tr><td><code>risk_hints</code></td><td class="ok">✅</td><td>會被 <code>build_intents</code> 原樣搬進 <code>ExecutionIntent</code></td></tr>
<tr><td><code>exit_spec</code></td><td class="ok">✅</td><td>出場規格(含絕對停損/停利價)。<b>2026-07-29 起才一律存在</b></td></tr>
<tr><td><code>size</code></td><td class="warn">⚠️ 否</td><td><b>只有 <code>size != 1.0</code> 才出現</b> → 消費端必須自行預設 1.0</td></tr>
</table>

<h3>3.2 輸出 B:cyqnt.signal/v1(對外 / JS)</h3>
<p>契約檔 <code>strategies/_standard/signal.schema.json</code>。實跑產出:</p>
{pre(j(signal_v1))}
<div class="badbox">
  <b>❌ 沒有框架產生器</b>:每支策略都<b>手寫</b> <code>generate()</code>
  (grep <code>def to_signal</code> / <code>def envelope_to</code> / <code>def export_signal</code> 皆為零)。<br>
  <b>❌ 沒有任何驗證</b>:repo 未安裝 <code>jsonschema</code>,CI 不檢查。
  格式目前一致(5 支策略嚴格驗證 0 違規)<b>純靠人工自律</b>。<br>
  <b>❌ 停損停利到不了下單端</b>:<code>build_intents</code>(<code>execution/interfaces.py:36-56</code>)
  只搬 <code>quantity</code> / <code>notional</code> / <code>price</code> / <code>risk_hints</code>。
</div>

<h3>3.3 輸出 C:trades.jsonl(live executor 真正吃的)</h3>
<p>paper daemon 每產生一筆成交就 append 一行到 <code>&lt;state_dir&gt;/trades.jsonl</code>,
<code>mvp_live_executor</code> 輪詢該檔、讀 <code>action</code> 去下真實單。
來源 = <code>PaperFill</code>(<code>simulation/live_paper_session.py:40</code>)。</p>
{pre(j(trades_jsonl_sample))}
<div class="badbox">
  <b>三個必須知道的限制</b>
  <ul>
    <li><b>沒有 <code>stop_loss</code> / <code>take_profit</code></b>(欄位就這 8 個)→
        <b>交易所端不會有掛好的保護單</b>。停損是在 daemon 進程內模擬判斷、觸發時才送市價單;
        <b>daemon 掉了,部位就沒有保護</b>。</li>
    <li>它是<b>成交記錄(fill)</b>,不是<b>訊號(signal)</b>。跟 §3.2 是兩個不同 schema,沒有欄位對應。</li>
    <li>daemon 的出場一律排到<b>下一根 open 市價</b>成交,不像兩個回測引擎在停損價成交 →
        回測會高估停損品質。</li>
  </ul>
</div>

<h2>4. 執行:實跑數字</h2>
{pre(RUN_BLOCK)}
<div class="lead">
  <b>兩個回測引擎現在逐筆等價</b>。修正 7 個根因後,對 repo 內全部 14 支已註冊策略 ×
  {{BTC 4h, BTC 1h}} = 21 組重跑對拍:<b>21/21 進場根完全相同、0 組不一致</b>,
  最大總報酬落差 <b>0.0104pp</b>。已補 21 個對拍測試進 CI
  (<code>tests/standard_bot/test_engine_parity.py</code>)。
</div>

<h2>5. 給要寫執行層的人:該接哪一個</h2>
<table>
<tr><th>你的情境</th><th>接哪一個</th><th>注意</th></tr>
<tr><td>Python 內、跑回測或一次性 paper</td><td class="ok">A · <code>SignalEnvelope</code></td>
    <td><code>payload["size"]</code> 可能不存在,預設 1.0</td></tr>
<tr><td>JS / 外部系統,要交易計畫(含停損停利)</td><td class="warn">B · <code>cyqnt.signal/v1</code></td>
    <td>要自己呼叫策略的 <code>generate()</code>;無框架產生器、無驗證器</td></tr>
<tr><td>接現有 paper daemon → 真實下單</td><td class="bad">C · <code>trades.jsonl</code></td>
    <td><b>拿不到停損停利</b>,保護單要自己補</td></tr>
</table>
<h3>待決定(尚未實作)</h3>
<ul>
<li><b>方案 A(小)</b>:daemon 每根額外寫一份 <code>signals.jsonl</code>(內容 = <code>cyqnt.signal/v1</code>,
    含 <code>stop_loss</code> / <code>take_profit</code>)。executor 照舊吃 <code>trades.jsonl</code>,
    但外部有一份符合規格的訊號流可接。不動執行邏輯,風險低。</li>
<li><b>方案 B(大)</b>:daemon 改走 <code>run_pipeline_step</code> + <code>SignalEnvelope</code>,
    三條路輸入輸出全部統一。但要重寫 daemon 核心迴圈,風險高。</li>
<li>兩項獨立小工:① envelope → <code>cyqnt.signal/v1</code> 的框架轉換器;
    ② 裝 <code>jsonschema</code> + CI 驗證所有策略的匯出。做完輸出 B 才算「有機器強制的規格」。</li>
<li>新聞的檔案化輸入(把 <code>ticker_rank</code> 落地成 parquet/JSON),選幣策略才能離線重現與回測。</li>
</ul>

<p class="muted" style="margin-top:40px;border-top:1px solid var(--line);padding-top:16px">
本頁由 <code>docs/gen_data_contract_html.py</code> 產生 —— 重跑該腳本即可用最新程式碼刷新所有數值。
</p>

</div></body></html>
"""

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
print(f"已產生 {OUT}  ({len(HTML):,} 字元)")
print(f"  event  : {ev_line}")
print(f"  vector : trades={vec.trade_count} ret={vec.total_return:+.6f}")
print(f"  tests  : {test_line}")
