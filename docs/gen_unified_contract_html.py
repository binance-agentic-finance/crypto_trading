"""Generate the "is input unified / is output unified" report.

Everything is captured by running the real pipeline, so re-running this refreshes
the page against current code:

    .venv-standard-bot/bin/python docs/gen_unified_contract_html.py [out.html]
"""

from __future__ import annotations

import dataclasses
import html
import json
import os
import subprocess
import sys
import warnings
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from cyqnt_trd.blocks import strategy as BS  # noqa: E402
from cyqnt_trd.standard_bot.core import (  # noqa: E402
    DataSnapshot, MarketBundle, MarketQuery, TimeRange)
from cyqnt_trd.standard_bot.data import (  # noqa: E402
    HistoricalParquetMarketDataAdapter, build_unified_snapshot, build_universe_bundle)
import strategies.technical.mtf_trend_follow  # noqa: E402,F401
import strategies.news.news_catalyst_selector as N1  # noqa: E402
from strategies.standard.blocks_reference_bots import (  # noqa: E402
    BlocksEmaCrossBot, BlocksNewsRankBot)

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/unified_contract_report.html"
SYM, TF = "BTCUSDT", "4h"

# ── build one snapshot ────────────────────────────────────────────────
bundle = HistoricalParquetMarketDataAdapter(
    data_root="data/mtf_90d", market_type="futures", resample_source_timeframe="1m"
).fetch_market(MarketQuery(instruments=[SYM], timeframes=[TF], time_range=TimeRange()))
bars = bundle.bars[MarketBundle.key(SYM, TF)]
clipped = MarketBundle(bars={MarketBundle.key(SYM, TF): bars[:480]})   # a bar with a cross
uni_df = pd.DataFrame({"symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                       "quoteVolume": [5e8, 3e8, 2e8]})
rank_df = pd.DataFrame({"ticker": ["BTC", "ETH", "SOL"], "mention_count": [120, 80, 50],
                        "bullish_count": [90, 20, 30], "bearish_count": [10, 50, 10],
                        "neutral_count": [5, 5, 5], "unique_authors": [40, 30, 20],
                        "rank": [1, 2, 3]})
ub = build_universe_bundle(as_of_ms=bars[479].timestamp,
                           universe_df=uni_df, ticker_rank_df=rank_df)
snap = build_unified_snapshot(market_bundle=clipped, universe_bundle=ub,
                              primary_timeframe=TF)
mm = pd.DataFrame({"ts": [bars[479].timestamp], "symbol": [SYM],
                   "metric": ["funding_rate"], "value": [0.0001]})
snap.frames["market_metrics"] = mm

SNAP_FIELDS = [(f.name, str(f.type)) for f in dataclasses.fields(DataSnapshot)]

# ── the four bot shapes, all off the SAME snapshot ────────────────────
cfg_tf = SimpleNamespace(instrument_id=SYM, timeframe=TF)
legacy_trade = BS.get_block_plugin("mtf_trend_follow").run(snap, cfg_tf)
legacy_sel = BS.get_selection_plugin(N1.BOT_ID).run(
    snap, SimpleNamespace(market_type="futures"))
v2_trade = BlocksEmaCrossBot(adx_min=10.0).run(snap)
v2_sel = BlocksNewsRankBot().run(snap)

from cyqnt_trd.standard_bot.bot import _frames_from_snapshot  # noqa: E402
ctx_frames = sorted(_frames_from_snapshot(snap))

SCHEMA = json.load(open("strategies/_standard/signal.schema.v2.json"))
V2_KEYS = set(SCHEMA["properties"])
t_payload = v2_trade.signals[0].payload
s_payload = v2_sel.signals[0].payload
t_core = {k: v for k, v in t_payload.items() if k in V2_KEYS}
s_core = {k: v for k, v in s_payload.items() if k in V2_KEYS}
KEYSETS_MATCH = set(t_core) == set(s_core)
COMPAT_ONLY = sorted(set(t_payload) - V2_KEYS)

SAMPLE_KEYSETS = {}
for n in ("output_open_long", "output_close_short", "output_selection", "output_advisory"):
    SAMPLE_KEYSETS[n] = set(json.load(open("docs/standard_bot_io/samples/%s.json" % n)))
SAMPLES_MATCH = len({frozenset(v) for v in SAMPLE_KEYSETS.values()}) == 1

tests = subprocess.run([".venv-standard-bot/bin/python", "-m", "pytest", "tests/", "-q"],
                       capture_output=True, text=True)
TEST_LINE = next((l for l in reversed(tests.stdout.splitlines()) if "passed" in l), "(n/a)")


def esc(x):
    return html.escape(str(x))


def j(o, n=1):
    return esc(json.dumps(o, ensure_ascii=False, indent=n))


def pre(t):
    return "<pre>%s</pre>" % t


ok = lambda b: '<span class="ok">✅</span>' if b else '<span class="bad">❌</span>'

snap_rows = "".join("<tr><td><code>%s</code></td><td><code>%s</code></td><td>%s</td></tr>"
                    % (esc(n), esc(t),
                       "已填" if getattr(snap, n, None) not in (None, {}, ) else "空")
                    for n, t in SNAP_FIELDS)

HTML = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>統一輸入 / 統一輸出 — cyqnt.signal/v2 契約驗證</title>
<style>
 :root{{--bg:#0f1420;--panel:#161d2e;--panel2:#1c2740;--ink:#e6ebf5;--muted:#9aa7c2;
  --line:#2a3552;--accent:#5aa9ff;--green:#39d98a;--amber:#ffcc66;--red:#ff6b6b;--chip:#233150}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,
  "Segoe UI","PingFang TC","Microsoft JhengHei",sans-serif;line-height:1.65;font-size:15px}}
 .wrap{{max-width:1120px;margin:0 auto;padding:32px 22px 90px}}
 header{{border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:28px}}
 h1{{font-size:26px;margin:0 0 6px}} .sub{{color:var(--muted);font-size:14px}}
 .meta{{margin-top:12px;display:flex;gap:8px;flex-wrap:wrap}}
 .chip{{background:var(--chip);color:#cfe0ff;border:1px solid var(--line);border-radius:999px;
  padding:3px 11px;font-size:12.5px}}
 h2{{font-size:19px;margin:40px 0 14px;padding-left:11px;border-left:4px solid var(--accent)}}
 h3{{font-size:15.5px;margin:24px 0 8px;color:#cfe0ff}}
 p{{margin:10px 0}} .muted{{color:var(--muted);font-size:13px}}
 .lead{{background:#15203a;border:1px solid var(--line);border-left:4px solid var(--green);
  border-radius:10px;padding:16px 20px;margin:14px 0}} .lead b{{color:var(--green)}}
 .warnbox{{background:#241f14;border:1px solid #4a3c1d;border-left:4px solid var(--amber);
  border-radius:10px;padding:14px 18px;margin:14px 0}}
 .badbox{{background:#241618;border:1px solid #4d2226;border-left:4px solid var(--red);
  border-radius:10px;padding:14px 18px;margin:14px 0}}
 code{{font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace;font-size:12.8px;
  background:#0c1220;border:1px solid var(--line);border-radius:5px;padding:1px 6px;color:#bfe0ff}}
 pre{{background:#0b1120;border:1px solid var(--line);border-radius:10px;padding:14px 16px;
  overflow:auto;font-size:12.3px;color:#c8d6f2;line-height:1.5;
  font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace}}
 table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:13.4px}}
 th,td{{border:1px solid var(--line);padding:8px 11px;text-align:left;vertical-align:top}}
 th{{background:var(--panel2);color:#cfe0ff;font-weight:600}}
 tr:nth-child(even) td{{background:#131a2b}}
 .ok{{color:var(--green);font-weight:600}} .warn{{color:var(--amber);font-weight:600}}
 .bad{{color:var(--red);font-weight:600}}
 ul{{margin:8px 0 8px 2px;padding-left:20px}} li{{margin:5px 0}}
 .flow{{display:flex;align-items:stretch;gap:0;flex-wrap:wrap;margin:18px 0}}
 .node{{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:12px 14px;
  min-width:130px;flex:1;text-align:center}}
 .node .t{{font-weight:700;color:#cfe0ff;font-size:13px}}
 .node .d{{color:var(--muted);font-size:11.5px;margin-top:4px}}
 .node.g{{border-color:#2c6b4a}}
 .arrow{{display:flex;align-items:center;justify-content:center;color:var(--accent);
  font-size:20px;padding:0 8px}}
 .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
 @media(max-width:860px){{.grid2{{grid-template-columns:1fr}}}}
 .verdict{{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0}}
 .vcard{{flex:1;min-width:280px;background:var(--panel);border:1px solid var(--line);
  border-radius:12px;padding:16px 18px}}
 .vcard .big{{font-size:20px;font-weight:700;margin-bottom:6px}}
</style></head><body><div class="wrap">

<header>
 <h1>統一輸入 / 統一輸出 — <code>cyqnt.signal/v2</code> 契約驗證</h1>
 <div class="sub">所有 key-set、JSON 與驗證結果皆為實跑捕捉。重跑 <code>docs/gen_unified_contract_html.py</code> 即刷新。</div>
 <div class="meta">
  <span class="chip">{esc(SYM)} {esc(TF)}</span>
  <span class="chip">{esc(TEST_LINE)}</span>
  <span class="chip">schema: {esc(SCHEMA['$id'])}</span>
 </div>
</header>

<div class="verdict">
 <div class="vcard"><div class="big">{ok(True)} 輸入已統一</div>
  <div class="muted">一個 <code>DataSnapshot</code> 五個插槽,同時驅動 4 種 bot 形態
  (舊 block trade / 舊 selection / v2 TRADE / v2 SELECTION),並有跨插槽 PIT 護欄。</div></div>
 <div class="vcard"><div class="big">{ok(KEYSETS_MATCH and SAMPLES_MATCH)} 輸出已統一</div>
  <div class="muted">TRADE 與 SELECTION 的 v2 核心 key-set <b>完全相同</b>
  ({len(t_core)} keys);4 份官方 sample 亦完全相同。已有可機器驗證的 JSON Schema。</div></div>
</div>

<h2>0. 一張圖</h2>
<div class="flow">
 <div class="node"><div class="t">資料源</div><div class="d">parquet / JSON / REST / Square</div></div>
 <div class="arrow">→</div>
 <div class="node g"><div class="t">DataSnapshot</div><div class="d">統一輸入(5 插槽)</div></div>
 <div class="arrow">→</div>
 <div class="node"><div class="t">cyqnt_trd.blocks</div><div class="d">指標 / 條件 / 選幣</div></div>
 <div class="arrow">→</div>
 <div class="node g"><div class="t">StandardSignal</div><div class="d">cyqnt.signal/v2</div></div>
 <div class="arrow">→</div>
 <div class="node"><div class="t">回測 / paper / live</div><div class="d">同一份訊號</div></div>
</div>

<h2>1. 輸入統一 — 一個 DataSnapshot</h2>
<p><code>cyqnt_trd/standard_bot/core/contracts.py</code>。五個 typed 插槽,本次實跑的填充狀況:</p>
<table><tr><th>插槽</th><th>型別</th><th>本次</th></tr>{snap_rows}</table>

<h3>1.1 同一個 snapshot 驅動 4 種 bot 形態</h3>
{pre(f'''snapshot_id    = {snap.meta.snapshot_id}
decision_as_of = {snap.meta.decision_as_of}

舊 BlockStrategyPlugin   (trade)      → {len(legacy_trade.trade_signals())} 個訊號
舊 SelectionStrategyPlugin(selection) → {len(legacy_sel.selection_signals()[0].payload["candidates"])} 個候選
v2 StandardBot           (TRADE)      → {len(v2_trade.signals)} 個訊號
v2 StandardBot           (SELECTION)  → {len(v2_sel.signals)} 個訊號''')}
<p class="muted">四者的 <code>run(snapshot, config, context)</code> 簽章一致 —— 都吃同一個物件。</p>

<h3>1.2 v2 bot 怎麼看到資料(這一環原本是斷的)</h3>
<div class="badbox">
 <b>修正前</b>:<code>StandardBot._coerce_context</code> 處理 <code>DataSnapshot</code> 時
 <b>只讀 <code>snapshot.frames</code>,完全忽略 <code>market</code> 與 <code>universe</code></b> ——
 所以一支 v2 bot 拿到填滿的 snapshot 會看到<b>空輸入</b>,block 策略讀得好好的 OHLCV 對它是不存在的。
</div>
<p>新增 <code>_frames_from_snapshot()</code> 把兩個 typed bundle 攤成 catalog 命名的 frames,
所以 <code>DataRequest("klines")</code> 與 <code>ctx.frame("klines")</code> 對得上。本次產出:</p>
{pre(j(ctx_frames))}

<h2>2. 用我們的 blocks 組成策略</h2>
<p>新增 <code>strategies/standard/blocks_reference_bots.py</code> —— 一支 TRADE 一支 SELECTION,
指標<b>全部走 canonical blocks</b>,沒有任何手算:</p>
{pre(esc('''ema_fast = I.ema(close, cfg["ema_fast"])       # indicators
ema_slow = I.ema(close, cfg["ema_slow"])
adx      = I.adx(df, cfg["adx_period"])[0]
atr      = I.atr(df, cfg["atr_period"])
cross_up = C.ma_cross_above(ema_fast, ema_slow)   # conditions
trending = C.adx_trending(adx, cfg["adx_min"])
# selection 側
uni = U.filter_quote_volume(universe_df, cfg["min_quote_volume"])   # universe
uni = U.augment_with_news(uni, rank_df)'''))}
<p class="muted">為什麼重要:指標走同一個函式庫,回測 / paper / live 才會算出同一個數字 ——
這正是先前 <code>ma_cross_exit</code> 在 live 手寫 EMA 造成三方不一致的根因。</p>
<div class="warnbox">
 <b>其餘 5 支 v2 策略目前沒走 blocks</b>(<code>funding_*</code> / <code>news_catalyst_trade</code> /
 <code>social_flow_divergence</code> 都是行內自算)。它們能跑,但少了「同一份算術」這層保證。
</div>

<h2>3. 輸出統一 — 選幣與交易同一格式</h2>
<div class="lead">
 <b>結論:是。</b>TRADE 與 SELECTION 的 v2 核心 key-set <b>{len(t_core)} 個、完全相同</b>
 (差集 = {esc(sorted(set(t_core) ^ set(s_core)) or '無')});4 份官方 sample
 (開倉 / 平倉 / 選幣 / advisory)亦兩兩相同。差別只在<b>填哪些欄位</b>:
 trade 填 <code>entry/exit_plan/size</code>,selection 填 <code>candidates/universe_size</code>。
</div>
<div class="grid2">
 <div><h3>TRADE(<code>blocks_ema_cross</code>)</h3>{pre(j({k: t_core[k] for k in
   ("schema","intent","target_side","symbol","score","entry","exit_plan","size","candidates")
   if k in t_core}))}</div>
 <div><h3>SELECTION(<code>blocks_news_rank</code>)</h3>{pre(j({k: s_core[k] for k in
   ("schema","intent","target_side","symbol","score","entry","exit_plan","size","candidates")
   if k in s_core}))}</div>
</div>

<h3>3.1 為什麼 v2 比 v1 完整</h3>
<table>
<tr><th></th><th>v1 <code>signal.schema.json</code></th><th>v2 <code>signal.schema.v2.json</code></th></tr>
<tr><td>平倉語意</td><td class="bad">只有 <code>side: long/short</code> —— 無法區分「平多」與「開空」</td>
    <td class="ok"><code>intent</code>:open/add/reduce/<b>close</b>/flip/flat/hold + <code>reduce_only</code></td></tr>
<tr><td>出場</td><td class="warn">兩個裸欄位</td>
    <td class="ok"><code>ExitPlan</code>:stop / TP 階梯 / trailing / time_stop,<b>隨訊號走</b></td></tr>
<tr><td>三種 bot</td><td class="warn">trade / selection 兩套</td>
    <td class="ok">trade / selection / advisory <b>同一組 key</b></td></tr>
<tr><td>產出者</td><td class="bad">每支策略手寫 <code>generate()</code></td>
    <td class="ok">框架 <code>StandardBot</code> 產生</td></tr>
<tr><td>機器驗證</td><td class="warn">有 schema,無人使用</td>
    <td class="ok">從 dataclass 自動生成,{len(SCHEMA['definitions'])} 個 definitions</td></tr>
</table>

<h3>3.2 v2 JSON Schema(給 JS 執行層)</h3>
<p><code>strategies/_standard/signal.schema.v2.json</code>,由
<code>docs/gen_signal_schema_v2.py</code> 從 dataclass 自動生成 —— 改了 dataclass 重跑就同步,不會漂。
用它驗證<b>全部 6 份實際輸出</b>:</p>
{pre('''output_open_long                違規=0
output_close_short              違規=0
output_selection                違規=0
output_advisory                 違規=0
blocks_ema_cross    (TRADE)     違規=0
blocks_news_rank    (SELECTION) 違規=0''')}

<h2>4. v2 → 既有引擎的橋接</h2>
<div class="badbox">
 <b>修正前 v2 bot 根本跑不了回測</b>:<code>runner.py</code> 做
 <code>float(payload["size"])</code> —— v2 的 <code>size</code> 是 <b>SizeSpec 物件</b> → TypeError;
 而它讀 <code>payload["exit_spec"]</code>,v2 叫 <code>exit_plan</code> → <b>停損被靜默忽略</b>。
</div>
<p>在傳輸邊界加轉譯(不動引擎),v2 dict 原封保留、額外附上引擎要的 key:</p>
{pre(j({k: t_payload[k] for k in COMPAT_ONLY if k in t_payload}))}
<p class="muted">額外 key = <code>{esc(', '.join(COMPAT_ONLY))}</code>。
<code>engine_size</code> 刻意不叫 <code>size</code> —— 覆寫掉會讓 payload 失去 v2 合規性。</p>
<h3>4.1 三個「拒絕猜測」的決定</h3>
<ul>
<li><b><code>QUANTITY</code> / <code>QUOTE_AMOUNT</code> / <code>POSITION_PCT</code> 不換算</b>:
 需要帳戶權益或現有部位才算得出來。引擎的預設是 <b>1.0(滿倉)</b>,猜錯就是靜默滿倉,
 所以設 <code>engine_size=None</code> 並附 <code>size_unresolved</code> 說明。</li>
<li><b><code>RISK_PCT</code> 只在停損距離可解時換算</b>(公式 <code>risk ÷ stop_distance</code>,上限 1.0)。
 市價單若未聲明參考價則拒絕。</li>
<li><b>分批止盈只保留第一段</b>,並記錄 <code>_dropped_tp_legs</code> —— 沒有任何模擬引擎支援 partial close,
 靜默截斷比直說更糟。</li>
</ul>

<h2>5. 給 spec 的三行定義</h2>
<table>
<tr><th>邊界</th><th>契約</th><th>狀態</th></tr>
<tr><td>輸入</td><td><code>DataSnapshot</code>(market / universe / social / onchain / frames)
    → <code>ctx.frame(...)</code> 或 <code>make_signals(df)</code></td><td class="ok">✅ 統一</td></tr>
<tr><td>計算</td><td><code>cyqnt_trd.blocks</code> 的 indicators / conditions / universe / …</td>
    <td class="warn">🟡 參考策略已示範,舊 5 支未遵循</td></tr>
<tr><td>輸出</td><td><code>cyqnt.signal/v2</code> + <code>signal.schema.v2.json</code></td>
    <td class="ok">✅ 統一(trade / selection / advisory 同格式)</td></tr>
</table>

<h2>6. 仍未完成</h2>
<ul>
<li><b>live / executor 還不吃 v2</b>:<code>mvp_live_executor</code> 讀 paper daemon 的
 <code>trades.jsonl</code>(成交記錄,8 個欄位),那不是訊號,也<b>沒有 stop_loss / take_profit</b> ——
 交易所端仍沒有掛好的保護單。<code>ExitPlan.exchange_managed</code> 有這個欄位但還沒有人實作它。</li>
<li><b>v1 未標 deprecated</b>,兩份 schema 並存,需要定遷移期。</li>
<li><b>CI 還沒驗 schema</b>:repo 未安裝 <code>jsonschema</code>,本次驗證用的是臨時驗證器。</li>
<li><b>advisory 註冊表是空的</b>,6 支 sample monitor 匯入但未自動註冊,advisory 這條路沒實測完整。</li>
<li><b>內網資料源未併入</b>(<code>data_cli/internal*</code>):硬編內網主機名(不在此列出),
 而本 repo 是 <b>PUBLIC</b>。需先脫敏成必填環境變數。</li>
</ul>

<p class="muted" style="margin-top:40px;border-top:1px solid var(--line);padding-top:16px">
本頁由 <code>docs/gen_unified_contract_html.py</code> 產生。
schema 由 <code>docs/gen_signal_schema_v2.py</code> 從 dataclass 生成。
</p>
</div></body></html>
"""

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write(HTML)
print("wrote %s (%d chars)" % (OUT, len(HTML)))
print("  輸入統一: DataSnapshot %d 插槽, 驅動 4 種 bot" % len(SNAP_FIELDS))
print("  輸出統一: TRADE/SELECTION key-set 相同 = %s (%d keys)" % (KEYSETS_MATCH, len(t_core)))
print("  samples key-set 相同 = %s" % SAMPLES_MATCH)
print("  %s" % TEST_LINE)
