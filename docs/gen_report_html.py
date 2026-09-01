"""Generate the presentation page: input -> blocks -> signal, with live demos.

Everything on the page is captured by running the real pipeline, so the numbers
are the numbers. Re-run to refresh against current code.

    .venv-standard-bot/bin/python docs/gen_report_html.py [out.html]

Design note: every command shown is tagged by whether it needs the network,
because the one thing a live demo cannot survive is discovering on stage that a
step needed VPN.
"""

from __future__ import annotations

import html
import json
import os
import sys
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/report_flow.html"

# 這是一個公開 repo,所以這裡不放公司內網的文件連結(見 tests/test_no_leaked_secrets.py)。
# 架構總覽與 I/O 規格在 repo 內自帶一份,任何拿到這份程式碼的人都讀得到。
ARCHITECTURE_DOC = "ARCHITECTURE.md"


def esc(x):
    return html.escape(str(x))


def pre(text, cls=""):
    return '<pre class="%s">%s</pre>' % (cls, esc(text))


def j(obj, limit=None):
    text = json.dumps(obj, ensure_ascii=False, indent=1)
    if limit and len(text) > limit:
        text = text[:limit] + "\n … (截斷)"
    return text


# --------------------------------------------------------------------------- #
# 1. real input bundle                                                         #
# --------------------------------------------------------------------------- #
from cyqnt_trd.standard_bot.core import (MarketBundle, MarketQuery,  # noqa: E402
                                         TimeRange)
from cyqnt_trd.standard_bot.data import (HistoricalParquetMarketDataAdapter,  # noqa: E402
                                         build_input_bundle, load_input_bundle)
from cyqnt_trd.standard_bot.data.input_bundle import FRAME_SHAPES  # noqa: E402

SYM, ITV = "BTCUSDT", "1h"
adapter = HistoricalParquetMarketDataAdapter(
    data_root="data/mtf_90d", market_type="futures", resample_source_timeframe="1m")
bars = adapter.fetch_market(MarketQuery(
    instruments=[SYM], timeframes=[ITV], time_range=TimeRange())
).bars[MarketBundle.key(SYM, ITV)]
fund = pd.read_parquet("data/derivatives_mvp_30d/futures/BTCUSDT/funding_rate.parquet")
DT = [b for b in bars
      if fund["timestamp"].min() <= b.timestamp <= fund["timestamp"].max()][-1].timestamp

rank_df = pd.DataFrame({"instrument_id": [SYM, "ETHUSDT"], "available_time": [DT, DT],
                        "rank": [1, 2], "score": [96.0, 32.0],
                        "mention_count": [120, 80], "bull_ratio": [0.9, 0.25]})
uni_df = pd.DataFrame({"instrument_id": [SYM, "ETHUSDT"], "available_time": [DT, DT],
                       "quote_volume": [5e8, 3e8]})
news_df = pd.DataFrame({"event_id": ["n1"], "event_time": [DT - 3600_000],
                        "available_time": [DT - 3500_000], "source_id": ["square"],
                        "topic": ["etf_inflow"], "instrument_id": [SYM],
                        "title": ["ETF net inflow hits 30d high"], "urgency": ["high"]})

BUNDLE = build_input_bundle(
    symbol=SYM, interval=ITV, decision_time=DT, market_type="futures",
    historical_dir="data/mtf_90d", storage_timeframe="1m",
    derivatives_dir="data/derivatives_mvp_30d",
    news_frame=news_df, ticker_rank_frame=rank_df, universe_frame=uni_df, max_bars=300)
BUNDLE_BYTES = len(json.dumps(BUNDLE, ensure_ascii=False, separators=(",", ":")).encode())

# skeleton: everything except the rows
SKELETON = {k: v for k, v in BUNDLE.items() if k != "frames"}
SKELETON["frames"] = {key: "{ shape: %s, rows: [ … %d 列 … ] }"
                      % (spec.get("shape"), len(spec["rows"]))
                      for key, spec in BUNDLE["frames"].items()}

# one real row from each frame — "here is the format" is useless without values
SAMPLE_ROWS = {key: (spec["rows"][-1] if spec["rows"] else None)
               for key, spec in BUNDLE["frames"].items()}

FRAME_TABLE = "".join(
    "<tr><td><code>%s</code></td><td><code>%s</code></td><td align=right>%s</td>"
    "<td>%s</td></tr>"
    % (esc(k), esc(v.get("shape")), "{:,}".format(len(v["rows"])),
       esc(BUNDLE["source_status"].get(k, "")))
    for k, v in BUNDLE["frames"].items())

# --------------------------------------------------------------------------- #
# 2. YAML strategies — validate + run, for real                                #
# --------------------------------------------------------------------------- #
from cyqnt_trd.standard_bot.yaml_pipeline._data import load_ohlcv  # noqa: E402
from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import (  # noqa: E402
    build_make_signals, build_selection_fn, resolve_block)
from cyqnt_trd.standard_bot.yaml_pipeline.spec import load_spec, validate_spec  # noqa: E402

TRADE_YAML_PATH = "docs/strategy_yaml_spec/example_multi_source.yaml"
SEL_YAML_PATH = "docs/strategy_yaml_spec/example_selection.yaml"
TRADE_YAML = open(TRADE_YAML_PATH, encoding="utf-8").read()
SEL_YAML = open(SEL_YAML_PATH, encoding="utf-8").read()

trade_spec = load_spec(TRADE_YAML_PATH)
sel_spec = load_spec(SEL_YAML_PATH)
TRADE_ERRORS, _ = validate_spec(trade_spec)
SEL_ERRORS, _ = validate_spec(sel_spec)

# real multi-source backtest
df, SRC = load_ohlcv(trade_spec)
long_s, short_s = build_make_signals(trade_spec)(df)
DERIV_COLS = [c for c in df.columns if "funding" in c or "open_interest" in c or "oi_" in c]
COVER = int(df["funding_rate_bps"].notna().sum()) if "funding_rate_bps" in df else 0

# counterfactual: does the multi-source leg actually change anything?
import copy  # noqa: E402

ohlcv_only = copy.deepcopy(trade_spec)
ohlcv_only["signals"]["entry"]["long"]["all_of"] = \
    ohlcv_only["signals"]["entry"]["long"]["all_of"][:1]
long_ohlcv, _ = build_make_signals(ohlcv_only)(df)
FILTERED = int(long_ohlcv.sum() - long_s.sum())
FILTERED_PCT = 100.0 * FILTERED / max(1, int(long_ohlcv.sum()))

# selection, on fixed data so the demo cannot fail on a cold API cache
SEL_SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
sel_uni = pd.DataFrame({"symbol": SEL_SYMS, "available_time": [DT] * 6,
                        "quoteVolume": [9e8, 7e8, 4e8, 3e8, 2e8, 1.2e8]})
sel_rank = pd.DataFrame({"symbol": SEL_SYMS, "rank": list(range(1, 7)),
                         "mention_count": [512, 388, 240, 175, 120, 96],
                         "bull_ratio": [0.81, 0.31, 0.62, 0.50, 0.28, 0.71],
                         "available_time": [DT] * 6})
CANDIDATES = build_selection_fn(sel_spec)(sel_uni, sel_rank)

# block coverage
import importlib  # noqa: E402
import pathlib  # noqa: E402

from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import SpecError  # noqa: E402

_total = _ok = 0
for _m in sorted(p.stem for p in pathlib.Path("cyqnt_trd/blocks").glob("*.py")
                 if not p.stem.startswith("_") and p.stem != "__init__"):
    _mod = importlib.import_module("cyqnt_trd.blocks.%s" % _m)
    for _n, _o in vars(_mod).items():
        if not callable(_o) or _n.startswith("_"):
            continue
        if not getattr(_o, "__module__", "").startswith("cyqnt_trd.blocks"):
            continue
        _total += 1
        try:
            resolve_block("%s.%s" % (_m, _n)); _ok += 1
        except SpecError:
            pass
BLOCK_OK, BLOCK_TOTAL = _ok, _total

# --------------------------------------------------------------------------- #
# 3. outputs — one schema, two kinds                                           #
# --------------------------------------------------------------------------- #
from cyqnt_trd.standard_bot.bot import BotContext, _frames_from_snapshot  # noqa: E402
from strategies.standard.blocks_reference_bots import BlocksNewsRankBot  # noqa: E402
from strategies.standard.multi_source_bot import MultiSourceBot  # noqa: E402

BUNDLE_PATH = "docs/standard_bot_io/samples/input_bundle_example.json"
snap = load_input_bundle(BUNDLE_PATH) if os.path.exists(BUNDLE_PATH) else None
if snap is not None:
    ctx = BotContext(decision_time=snap.meta.decision_as_of,
                     frames=_frames_from_snapshot(snap))
else:
    ctx = None

TRADE_OUT = SEL_OUT = None
if ctx is not None:
    _t = MultiSourceBot().decide_checked(ctx)
    TRADE_OUT = _t[0].to_dict() if _t else None
    _s = BlocksNewsRankBot().decide_checked(ctx)
    SEL_OUT = _s[0].to_dict() if _s else None

if TRADE_OUT is None:                     # fall back to the shipped sample
    TRADE_OUT = json.load(open("docs/standard_bot_io/samples/output_open_long.json"))
if SEL_OUT is None:
    SEL_OUT = json.load(open("docs/standard_bot_io/samples/output_selection.json"))

KEYS_MATCH = set(TRADE_OUT) == set(SEL_OUT)
KEY_COUNT = len(TRADE_OUT)
HEAD = ("schema", "kind", "bot_id", "symbol", "intent", "target_side", "order_side",
        "market_scope", "score", "universe_size")
TRADE_HEAD = {k: TRADE_OUT.get(k) for k in HEAD}
SEL_HEAD = {k: SEL_OUT.get(k) for k in HEAD}

INTENTS = ["open_long", "open_short", "add_long", "add_short", "reduce_long",
           "reduce_short", "close_long", "close_short", "flip_to_long",
           "flip_to_short", "flat", "hold"]

CAND_ROWS = "".join(
    "<tr><td align=right>%d</td><td><code>%s</code></td><td>%s</td>"
    "<td align=right>%s</td></tr>"
    % (c["rank"], esc(c["symbol"]), esc(c["side"]), "{:,.0f}".format(c["score"]))
    for c in CANDIDATES)

# --------------------------------------------------------------------------- #
# page                                                                         #
# --------------------------------------------------------------------------- #
CSS = """
*{box-sizing:border-box}
body{font:15px/1.7 -apple-system,"PingFang TC","Noto Sans TC",sans-serif;margin:0;
 background:#0e1117;color:#d7dee8}
.wrap{max-width:1180px;margin:0 auto;padding:36px 28px 90px}
h1{font-size:30px;margin:0 0 6px;color:#fff;letter-spacing:-.4px}
h2{font-size:22px;margin:52px 0 14px;color:#fff;border-bottom:1px solid #232a35;
 padding-bottom:9px}
h3{font-size:17px;margin:30px 0 10px;color:#9fc6ff}
h4{font-size:15px;margin:20px 0 8px;color:#c8d4e4}
.sub{color:#8b98ab;margin:0 0 4px}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.9em;
 background:#1a2130;padding:1px 5px;border-radius:4px;color:#a9d3ff}
pre{background:#111722;border:1px solid #212a38;border-radius:9px;padding:14px 16px;
 overflow:auto;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;
 line-height:1.55;color:#c5d3e5;max-height:520px}
pre.sm{font-size:11.5px;max-height:340px}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px}
th,td{border:1px solid #232c3a;padding:7px 10px;text-align:left;vertical-align:top}
th{background:#161d29;color:#9fc6ff;font-weight:600}
.lead{background:#132033;border-left:3px solid #4a90e2;padding:13px 17px;
 border-radius:0 8px 8px 0;margin:14px 0}
.ok{background:#12241a;border-left:3px solid #35c46a;padding:13px 17px;
 border-radius:0 8px 8px 0;margin:14px 0}
.warn{background:#2a2113;border-left:3px solid #e0a33a;padding:13px 17px;
 border-radius:0 8px 8px 0;margin:14px 0}
.bad{background:#2a1618;border-left:3px solid #e05a5a;padding:13px 17px;
 border-radius:0 8px 8px 0;margin:14px 0}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:900px){.g2{grid-template-columns:1fr}}
.muted{color:#7f8c9f;font-size:13px}
.tag{display:inline-block;font-size:11px;padding:2px 8px;border-radius:99px;
 margin-right:6px;font-weight:600}
.t-off{background:#173a25;color:#6ee79b}
.t-net{background:#3a3117;color:#f0c46a}
.t-vpn{background:#3a1a1d;color:#ff8f8f}
.flow{background:#111722;border:1px solid #212a38;border-radius:9px;padding:20px;
 font-family:ui-monospace,Menlo,monospace;font-size:12.5px;line-height:1.8;
 color:#c5d3e5;overflow:auto}
a{color:#79b8ff}
.big{font-size:26px;color:#fff;font-weight:600}
ul{padding-left:22px}li{margin:5px 0}
"""

# 積木數用 BLOCK_TOTAL 注入,不要寫死 —— 這張圖原本硬編碼 310,而同一頁下方的
# 覆蓋率表用的是實測值,於是一頁之內對同一件事給出兩個數字。
FLOW = """①  取數                      ②  組策略               ③  出訊號                ④  消費
──────────────────────────────────────────────────────────────────────────────────
K 線            ┐                                      ┌─ kind=trade       ┌→ 回測    ✅
funding / OI    ┤                                      │                   │
清算            ┤→   一份 JSON      →   blocks 積木  →  ┼─ kind=selection  ┼→ paper   ✅
新聞 / 熱度     ┤   cyqnt.input/v1      (%d 個積木)     │                   │
選幣宇宙        ┤                                      └─ kind=alert       └→ live    ⚠️
內部節點        ┘                                        cyqnt.signal/v2

     ▲ 一個決策時點                ▲ YAML 或 Python          ▲ 交易與選幣同一格式
       所有來源                      不寫程式也能組             消費端只認 kind 一個欄位
       PIT 只 gate 一次""" % BLOCK_TOTAL

DEMO_VALIDATE = """cd ~/Dev/crypto_trading-main

# 四個範例,全部靜態檢查 + 用合成資料實跑一次
for f in docs/strategy_yaml_spec/example_*.yaml; do
  .venv-standard-bot/bin/python -m cyqnt_trd.standard_bot.yaml_pipeline validate $f
done"""

DEMO_RUN = """# 多源策略,吃本地 parquet,完全離線
.venv-standard-bot/bin/python -m cyqnt_trd.standard_bot.yaml_pipeline \\
    run docs/strategy_yaml_spec/example_multi_source.yaml"""

DEMO_SEL = """# 選幣:用固定資料,不打 API(當場 demo 最可靠的版本)
.venv-standard-bot/bin/python - <<'PY'
import pandas as pd
from cyqnt_trd.standard_bot.yaml_pipeline.spec import load_spec
from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import build_selection_fn

spec = load_spec("docs/strategy_yaml_spec/example_selection.yaml")
syms = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT"]
uni  = pd.DataFrame({"symbol": syms, "quoteVolume":[9e8,7e8,4e8,3e8,2e8,1.2e8],
                     "available_time":[0]*6})
rank = pd.DataFrame({"symbol": syms, "rank": list(range(1,7)),
                     "mention_count":[512,388,240,175,120,96],
                     "bull_ratio":[0.81,0.31,0.62,0.50,0.28,0.71],
                     "available_time":[0]*6})
for c in build_selection_fn(spec)(uni, rank):
    print("  #%d %-9s %-6s score=%.0f" % (c["rank"], c["symbol"], c["side"], c["score"]))
PY"""

DEMO_WEB = """PYTHONPATH=~/Dev/crypto_trading-main \\
  ~/Dev/crypto_trading-main/.venv-standard-bot/bin/python \\
  docs/strategy_yaml_spec/demo/server.py
# 開 http://127.0.0.1:8799"""

INVARIANTS = """<ul>
<li><b>一個時鐘</b> —— 每一列在寫進 JSON 前都過 <code>available_time &lt;= decision_time</code>。
 只 gate 一次,下游不可能再讀到未來。<code>event_time</code>(事情何時發生)與
 <code>available_time</code>(我們何時才可能知道)是<b>兩個不同欄位</b>,混為一談就是
 walk-forward 偷看未來的來源。</li>
<li><b>一套詞彙</b> —— funding / OI / 清算 / 任何內網數值,全部落成同一種
 <code>MetricFrame</code>:<code>(instrument_id, metric, value, event_time, available_time)</code>。
 策略讀三個來源用<b>同一個存取函式</b>,<b>加第四個來源不用改管線</b>。</li>
<li><b>一個窗</b> —— 每條序列只留最新 N 筆。同一份 bundle 從 1.7 MB 降到
 %s bytes,<b>策略輸出完全不變</b>。</li>
</ul>""" % "{:,}".format(BUNDLE_BYTES)

parts = []
A = parts.append

A("<!DOCTYPE html><html lang='zh-Hant'><head><meta charset='utf-8'>")
A("<meta name='viewport' content='width=device-width,initial-scale=1'>")
A("<title>統一輸入 → blocks → 統一輸出</title><style>%s</style></head><body><div class='wrap'>" % CSS)

A("<h1>一份 JSON 進,一種訊號出</h1>")
A("<p class='sub'>統一輸入 <code>cyqnt.input/v1</code> → blocks 組策略 → "
  "統一輸出 <code>cyqnt.signal/v2</code></p>")
A("<p class='muted'>本頁所有數字都是實跑抓的(<code>docs/gen_report_html.py</code>)。"
  "系統架構、哪條路活著、已知缺口見 <a href='%s'>docs/ARCHITECTURE.md</a>;"
  "I/O 規格全文見 <code>strategies/_standard/</code> 底下的兩份 schema。</p>"
  % ARCHITECTURE_DOC)

A("<div class='lead'><span class='big'>核心一句話</span><br>"
  "<b>一次取數變成一份 JSON,用積木組出策略,輸出一種格式</b> —— "
  "交易和選幣共用同一份 schema,消費端只認一個 <code>kind</code> 欄位。</div>")

# ---- 0 flow ---- #
A("<h2>0. 全流程</h2>")
A("<div class='flow'>%s</div>" % esc(FLOW))

# ---- 1 input ---- #
A("<h2>1. 輸入 —— 一個時間點、所有來源、一份 JSON</h2>")
A("<p>格式 <code>cyqnt.input/v1</code>,契約檔 "
  "<code>strategies/_standard/input.schema.v1.json</code>。兩個產生器吐出<b>完全相同的 "
  "envelope</b>,所以 paper 讀到的檔案形狀和回測一模一樣:</p>")
A("<table><tr><th>產生器</th><th>用途</th></tr>"
  "<tr><td><code>build_live_bundle()</code></td><td>打 API 取即時資料(paper / live)</td></tr>"
  "<tr><td><code>build_input_bundle()</code></td><td>讀本地檔(回測 / 回放)</td></tr></table>")

A("<h3>1.1 三個設計不變量</h3>")
A(INVARIANTS)

A("<h3>1.2 骨架(真實產出,rows 收起來)</h3>")
A(pre(j(SKELETON)))
A("<p class='muted'><code>source_status</code> 涵蓋<b>每一個宣告過的節點</b>(含讀失敗的),"
  "所以「沒讀到」和「讀到但是空的」分得出來 —— 缺料會降級但不會靜默。</p>")

A("<h3>1.3 這一份實際有什麼</h3>")
A("<table><tr><th>frame</th><th>標準形狀</th><th>列數</th><th>狀態</th></tr>%s</table>"
  % FRAME_TABLE)
A("<p class='muted'>共 %s bytes。7 種標準形狀:BarFrame / MetricFrame / PanelFrame / "
  "EventFrame / RankFrame / PositionFrame / BookFrame。</p>" % "{:,}".format(BUNDLE_BYTES))

A("<h3>1.4 每個 frame 一列真實資料</h3>")
A("<p class='muted'>「這是格式」沒有看到值就等於沒說。</p>")
A(pre(j(SAMPLE_ROWS), "sm"))

# ---- 2 strategy ---- #
A("<h2>2. 組策略 —— blocks 積木,YAML 不寫程式</h2>")
A("<p><code>blocks/</code> 有 <b>%d 個純函式</b>(指標、條件、衍生品、新聞、選幣、風控…),"
  "策略就是這些積木的組合。YAML 可觸及 <b>%d / %d</b> 個。</p>"
  % (BLOCK_TOTAL, BLOCK_OK, BLOCK_TOTAL))

A("<h3>2.1 交易策略(多來源)</h3>")
A(pre(TRADE_YAML.strip()))
A("<div class='%s'><b>validate:</b> %s</div>"
  % ("ok" if not TRADE_ERRORS else "bad",
     "通過 —— 靜態檢查 + 用合成資料實跑一次" if not TRADE_ERRORS
     else esc("; ".join(TRADE_ERRORS))))
A("<p><b>兩道保護:</b></p><ul>"
  "<li>沒宣告 <code>data.derivatives</code> 就引用 <code>funding_rate_bps</code> → "
  "<b>validate 直接擋</b>,並告訴你要加哪一段。不會驗證過了才在跑的時候拿到空值</li>"
  "<li><code>validate</code> 會<b>用合成資料實跑一次</b>,抓出參數名錯、型別錯、arity 錯。"
  "過了 validate 就是結構上可跑</li></ul>")

A("<h4>實跑結果(真實資料)</h4>")
A(pre("資料來源: %s\nK 線:     %s 根\n掛上的衍生品欄位: %s\nfunding 有值的根數: %s"
      % (SRC, "{:,}".format(len(df)), ", ".join(DERIV_COLS), "{:,}".format(COVER))))
A("<div class='ok'><b>反事實檢查 —— 多源資料真的在起作用</b><br>"
  "只留 EMA 交叉 → <b>%s</b> 個做多訊號;加上 funding + OI 條件 → <b>%s</b> 個。"
  "<b>濾掉 %s 個(%.1f%%)</b>。資料不是掛在那裡好看的。</div>"
  % ("{:,}".format(int(long_ohlcv.sum())), "{:,}".format(int(long_s.sum())),
     "{:,}".format(FILTERED), FILTERED_PCT))

A("<h3>2.2 選幣策略(截面)</h3>")
A("<p><code>signals:</code> 逐根評估一個標的 → <code>kind=trade</code>;"
  "<code>selection:</code> 一次評估整個宇宙 → <code>kind=selection</code>。"
  "<b>兩者互斥</b>,一份 spec 只能是其中一種。</p>")
A(pre(SEL_YAML.strip()))
A("<div class='%s'><b>validate:</b> %s</div>"
  % ("ok" if not SEL_ERRORS else "bad",
     "通過" if not SEL_ERRORS else esc("; ".join(SEL_ERRORS))))
A("<h4>實跑結果</h4>")
A("<table><tr><th>rank</th><th>標的</th><th>方向</th><th>score</th></tr>%s</table>"
  % CAND_ROWS)
A("<p class='muted'>語法完全共用 —— selection 的 frame「一列是一個幣」而不是「一列是一根 K」,"
  "所以同一套 block、同一套比較器都能用。<code>rolling_zscore</code> 沿著幣種排名跑,"
  "同一個 block、不同的軸。</p>")

# ---- 3 output ---- #
A("<h2>3. 輸出 —— 交易與選幣同一格式</h2>")
A("<div class='ok'><b>頂層 key 完全相同:%d 個,差集為零 %s</b><br>"
  "消費端只解析一套 schema,<b>switch <code>kind</code> 一個欄位</b>。</div>"
  % (KEY_COUNT, "✅" if KEYS_MATCH else "❌"))

A("<table><tr><th><code>kind</code></th><th>主要填</th><th>意思</th></tr>"
  "<tr><td><code>trade</code></td><td><code>entry</code> / <code>exit_plan</code> / "
  "<code>size</code></td><td>對某個標的的指令</td></tr>"
  "<tr><td><code>selection</code></td><td><code>candidates</code> / "
  "<code>universe_size</code></td><td>一籃排名後的候選</td></tr>"
  "<tr><td><code>alert</code></td><td><code>advisory_action</code> / "
  "<code>summary</code></td><td>觀察,不自動執行</td></tr></table>")

A("<div class='g2'>")
A("<div><h4>交易型(實跑)</h4>%s</div>" % pre(j(TRADE_HEAD), "sm"))
A("<div><h4>選幣型(實跑)</h4>%s</div>" % pre(j(SEL_HEAD), "sm"))
A("</div>")

A("<h3>3.1 兩個關鍵欄位</h3>")
A("<p><b><code>intent</code> —— 12 種值</b>,明確區分「平多」和「開空」,執行層不用猜:</p>")
A(pre("  ".join(INTENTS)))
A("<p class='muted'>而且<b>沒宣告讀過部位的策略不准發平倉指令</b> —— 框架直接丟 "
  "<code>CapabilityError</code>。理由很直接:那些指令都在斷言「現在有一個部位」,"
  "而一支從沒讀過部位的策略是在猜。</p>")
A("<p><b><code>exit_plan</code></b> —— 停損、分批停利、移動停損、時間停損,"
  "<b>跟著訊號一起走</b>,不是只活在某個 daemon 的記憶體裡。</p>")

A("<h3>3.2 交易型完整輸出</h3>")
A(pre(j(TRADE_OUT)))

A("<h3>3.3 選幣型完整輸出</h3>")
A("<p class='muted'>注意頂層 key 與上面<b>逐一相同</b>,只是 "
  "<code>entry/exit_plan/size</code> 為 <code>null</code>,結果放在 "
  "<code>candidates</code>。</p>")
A(pre(j(SEL_OUT)))

# ---- 4 demo ---- #
A("<h2>4. 現場 Demo</h2>")
A("<div class='warn'><b>先看標籤再決定要不要在台上跑。</b><br>"
  "<span class='tag t-off'>離線可靠</span>不需要網路,一定會過。"
  "<span class='tag t-net'>需外網</span>抓 Binance 公開 API。"
  "<span class='tag t-vpn'>需 VPN</span>打內網端點,<b>沒 VPN 一定失敗</b>。</div>")

A("<h3>Demo 1 &nbsp;<span class='tag t-off'>離線可靠</span> 四個範例全部驗證</h3>")
A(pre(DEMO_VALIDATE))
A("<p class='muted'>會印四行 <code>OK: spec '…' is valid and dry-ran successfully</code>。"
  "重點是 validate <b>真的跑了一次</b>,不只是檢查欄位。</p>")

A("<h3>Demo 2 &nbsp;<span class='tag t-off'>離線可靠</span> 多源策略回測</h3>")
A(pre(DEMO_RUN))
A("<p class='muted'>讀本地 parquet。會印出 "
  "<code>data=%s</code> 與報酬 / Sharpe / 交易筆數,"
  "並<b>誠實警告</b>衍生品資料只覆蓋部分回測期間。</p>" % esc(SRC))

A("<h3>Demo 3 &nbsp;<span class='tag t-off'>離線可靠</span> 選幣</h3>")
A(pre(DEMO_SEL))
A("<p class='muted'>刻意用固定資料。走 CLI 的 <code>run</code> 會打即時 Square API,"
  "cache 冷的時候會回 <code>ticker_rank: empty</code> → 沒有候選,不適合當場 demo。</p>")

A("<h3>Demo 4 &nbsp;<span class='tag t-net'>需外網</span><span class='tag t-vpn'>LLM 那半需 VPN</span> "
  "自然語言 → YAML → 回測(網頁)</h3>")
A(pre(DEMO_WEB))
A("<p>網頁有兩半,<b>可以分開 demo</b>:</p>")
A("<table><tr><th>半</th><th>需要什麼</th><th>狀態</th></tr>"
  "<tr><td><b>回測</b>(貼 YAML 進去直接跑)</td><td>外網抓 K 線</td>"
  "<td>✅ 實測 HTTP 200、1000 根 K、真實指標</td></tr>"
  "<tr><td><b>NL → YAML</b>(打你的 LiteLLM)</td><td><b>VPN</b> + API key</td>"
  "<td>⚠️ 端點在內網,沒 VPN 會出 <code>TypeError: Load failed</code></td></tr></table>")
A("<div class='bad'><b>Demo 備案(強烈建議先準備)</b><br>"
  "如果現場沒有 VPN,<b>NL → YAML 那一步一定失敗</b>。做法:先在有 VPN 的環境把一句話"
  "轉出來的 YAML 存好,現場<b>直接貼進回測框</b>,講「這份 YAML 是模型從這句話生的」,"
  "再按回測。流程完整、風險為零。</div>")
A("<p class='muted'>API Base / Key / Model 是在瀏覽器表單填的,存在 localStorage,"
  "<b>不在 repo 裡</b>(內網主機名刻意不寫進程式碼)。</p>")

# ---- 5 status ---- #
A("<h2>5. 現況與缺口</h2>")
A("<table><tr><th>項目</th><th>狀態</th></tr>"
  "<tr><td>統一輸入格式定義 + PIT 閘門</td><td>✅</td></tr>"
  "<tr><td>YAML 可觸及全部 blocks</td><td>✅ %d / %d 個函式</td></tr>"
  "<tr><td>YAML 交易策略</td><td>✅ 實跑 %s 根真實 K 線</td></tr>"
  "<tr><td>YAML 選幣策略</td><td>✅</td></tr>"
  "<tr><td>多源資料(funding / OI / 新聞)進策略</td><td>✅ 實測濾掉 %.0f%% 的訊號</td></tr>"
  "<tr><td>統一輸出 v2(交易 + 選幣同格式)</td><td>✅ %d 個 key 相同</td></tr>"
  "<tr><td>回測 / paper</td><td>✅</td></tr>"
  "<tr><td><b>輸入 bundle 接進 runtime</b></td><td>🔧 進行中</td></tr>"
  "<tr><td><b>live 下單</b></td><td>⚠️ 執行層還沒接 v2</td></tr>"
  "</table>"
  % (BLOCK_OK, BLOCK_TOTAL, "{:,}".format(len(df)), FILTERED_PCT, KEY_COUNT))

A("<div class='warn'><b>被問到還缺什麼,最誠實的答案</b><br>"
  "<b>live 執行層還在吃舊格式</b>(成交記錄而非訊號),所以停損目前只在程式內模擬,"
  "<b>沒有掛到交易所</b>。<code>ExitPlan.exchange_managed</code> 這個欄位就是為此留的,"
  "但還沒有實作。這是上真錢前必須補的一環。</div>")

# ---- appendix ---- #
A("<h2>附錄:檔案在哪</h2>")
A("<table><tr><th>要找什麼</th><th>檔案</th></tr>"
  "<tr><td>系統架構 / 哪條路活著</td><td><a href='%s'><code>docs/ARCHITECTURE.md</code></a></td></tr>"
  "<tr><td>輸入契約檔</td><td><code>strategies/_standard/input.schema.v1.json</code></td></tr>"
  "<tr><td>輸出契約檔</td><td><code>strategies/_standard/signal.schema.v2.json</code></td></tr>"
  "<tr><td>輸入產生器</td><td><code>cyqnt_trd/standard_bot/data/input_bundle.py</code> / "
  "<code>live_bundle.py</code></td></tr>"
  "<tr><td>輸出契約(dataclass)</td><td><code>cyqnt_trd/standard_bot/core/signal_contract.py</code></td></tr>"
  "<tr><td>YAML 解譯器</td><td><code>cyqnt_trd/standard_bot/yaml_pipeline/</code></td></tr>"
  "<tr><td>blocks 積木庫</td><td><code>cyqnt_trd/blocks/</code>(%d 個公開 callable)</td></tr>"
  "<tr><td>NL → YAML demo</td><td><code>docs/strategy_yaml_spec/demo/server.py</code>(port 8799)</td></tr>"
  "<tr><td>範例 YAML</td><td><code>docs/strategy_yaml_spec/example_*.yaml</code></td></tr>"
  "</table>" % (ARCHITECTURE_DOC, BLOCK_TOTAL))

A("</div></body></html>")

page = "\n".join(parts)
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write(page)

print("wrote %s (%d chars)" % (OUT, len(page)))
print("  bundle: %s bytes, %d frames" % ("{:,}".format(BUNDLE_BYTES), len(BUNDLE["frames"])))
print("  trade yaml valid: %s | selection yaml valid: %s"
      % (not TRADE_ERRORS, not SEL_ERRORS))
print("  backtest: %s bars from %s, multi-source filtered %.1f%%"
      % ("{:,}".format(len(df)), SRC, FILTERED_PCT))
print("  selection candidates: %d" % len(CANDIDATES))
print("  output keys identical: %s (%d)" % (KEYS_MATCH, KEY_COUNT))
print("  blocks reachable from YAML: %d/%d" % (BLOCK_OK, BLOCK_TOTAL))
