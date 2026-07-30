"""Generate the full-flow spec page: one input bundle -> blocks -> one signal.

Everything is captured by running the real pipeline (including building a real
input bundle from the repo's own data), so re-running refreshes the page against
current code.

    .venv-standard-bot/bin/python docs/gen_full_flow_html.py [out.html]
"""

from __future__ import annotations

import collections
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


def esc(x):
    return html.escape(str(x))

from cyqnt_trd.blocks import strategy as BS  # noqa: E402
from cyqnt_trd.standard_bot.core import (  # noqa: E402
    MarketBundle, MarketQuery, TimeRange, input_contract as IC)
from cyqnt_trd.standard_bot.data import (  # noqa: E402
    HistoricalParquetMarketDataAdapter, build_input_bundle, load_input_bundle,
    write_input_bundle)
from cyqnt_trd.standard_bot.data.catalog import list_nodes  # noqa: E402
from cyqnt_trd.standard_bot.data.input_bundle import FRAME_SHAPES  # noqa: E402
from cyqnt_trd.standard_bot.data.internal_slots import (  # noqa: E402
    INTERNAL_SLOTS, internal_client_available)
import strategies.technical.mtf_trend_follow  # noqa: E402,F401
import strategies.news.news_catalyst_selector as N1  # noqa: E402
from strategies.standard.blocks_reference_bots import (  # noqa: E402
    BlocksEmaCrossBot, BlocksNewsRankBot)
from strategies.standard.multi_source_bot import MultiSourceBot  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/full_flow_spec.html"
BUNDLE_PATH = "docs/standard_bot_io/samples/input_bundle_example.json"
SYM, ITV = "BTCUSDT", "1h"

# ── 1. build a real bundle from the repo's own data ───────────────────
adapter = HistoricalParquetMarketDataAdapter(
    data_root="data/mtf_90d", market_type="futures", resample_source_timeframe="1m")
bars = adapter.fetch_market(MarketQuery(
    instruments=[SYM], timeframes=[ITV], time_range=TimeRange())
).bars[MarketBundle.key(SYM, ITV)]
fund = pd.read_parquet("data/derivatives_mvp_30d/futures/BTCUSDT/funding_rate.parquet")
DT = [b for b in bars
      if fund["timestamp"].min() <= b.timestamp <= fund["timestamp"].max()][-1].timestamp

rank_df = pd.DataFrame({"instrument_id": ["BTCUSDT", "ETHUSDT"],
                        "available_time": [DT, DT], "rank": [1, 2],
                        "score": [96.0, 32.0], "mention_count": [120, 80],
                        "bull_ratio": [0.9, 0.25]})
uni_df = pd.DataFrame({"instrument_id": ["BTCUSDT", "ETHUSDT"],
                       "available_time": [DT, DT], "quote_volume": [5e8, 3e8]})
news_df = pd.DataFrame({"event_id": ["n1"], "event_time": [DT - 3600_000],
                        "available_time": [DT - 3500_000], "source_id": ["square"],
                        "topic": ["etf_inflow"], "instrument_id": ["BTCUSDT"],
                        "title": ["ETF net inflow hits 30d high"], "urgency": ["high"]})
internal_df = pd.DataFrame({"event_time": [DT], "available_time": [DT],
                            "instrument_id": ["BTCUSDT"],
                            "metric": ["etf_net_flow_usd"], "value": [3.1e8]})

bundle = build_input_bundle(
    symbol=SYM, interval=ITV, decision_time=DT, market_type="futures",
    historical_dir="data/mtf_90d", storage_timeframe="1m",
    derivatives_dir="data/derivatives_mvp_30d",
    liquidations_dir="data/liquidations_mvp_live",
    news_frame=news_df, ticker_rank_frame=rank_df, universe_frame=uni_df,
    extra_frames={"internal_metrics": internal_df}, max_bars=300,
    declare_internal=list(INTERNAL_SLOTS),
    internal_frames={
        "internal_etf_flow": pd.DataFrame({
            "token": ["BTC"], "date": [DT], "flow": [3.1e8],
            "net_assets": [9.2e10], "close_price": [74985.9]}),
        "internal_futures_radar": pd.DataFrame({
            "symbol": ["BTCUSDT"], "metric": ["oi_change_24h"],
            "value": [-0.051], "event_time": [DT]}),
        "internal_macro_calendar": pd.DataFrame({
            "event_time": [DT - 7200_000], "event_type": ["CPI"],
            "actual": [3.1], "forecast": [3.2], "previous": [3.4]}),
    })
UNBOUNDED = build_input_bundle(
    symbol=SYM, interval=ITV, decision_time=DT, market_type="futures",
    historical_dir="data/mtf_90d", storage_timeframe="1m",
    derivatives_dir="data/derivatives_mvp_30d",
    liquidations_dir="data/liquidations_mvp_live",
    news_frame=news_df, ticker_rank_frame=rank_df, universe_frame=uni_df,
    extra_frames={"internal_metrics": internal_df}, max_bars=300,
    metric_lookback=None, max_event_rows=None)
UNBOUNDED_BYTES = len(json.dumps(UNBOUNDED, ensure_ascii=False,
                                 separators=(",", ":")).encode())
SIZE_ROWS = "".join(
    "<tr><td><code>%s</code></td><td align=right>%s</td><td align=right>%s</td>"
    "<td align=right>%s</td></tr>" % (
        esc(k),
        "{:,}".format(len(UNBOUNDED["frames"].get(k, {}).get("rows", []))),
        "{:,}".format(len(v["rows"])),
        "{:,}".format(len(json.dumps(v, ensure_ascii=False,
                                     separators=(",", ":")).encode())))
    for k, v in sorted(bundle["frames"].items(),
                       key=lambda kv: -len(json.dumps(kv[1], ensure_ascii=False,
                                                      separators=(",", ":")).encode())))

PUBLIC_BUNDLE = build_input_bundle(
    symbol=SYM, interval=ITV, decision_time=DT, market_type="futures",
    historical_dir="data/mtf_90d", storage_timeframe="1m",
    derivatives_dir="data/derivatives_mvp_30d",
    news_frame=news_df, ticker_rank_frame=rank_df, universe_frame=uni_df,
    max_bars=300, declare_internal=list(INTERNAL_SLOTS))
INTERNAL_KEYS = sorted(k for k in bundle["source_status"] if k.startswith("internal_"))
SAME_STATUS_KEYS = set(PUBLIC_BUNDLE["source_status"]) == set(bundle["source_status"])
write_input_bundle(bundle, BUNDLE_PATH)
BUNDLE_BYTES = os.path.getsize(BUNDLE_PATH)

# A human-readable twin of the bundle: identical structure, a few real rows per
# frame. The full artifact is 1.7 MB — nobody can read that, and "here is the
# format" is useless without seeing actual values in it.
PREVIEW_PATH = "docs/standard_bot_io/samples/input_bundle_preview.json"
_KEEP = {"klines": 3, "funding": 2, "open_interest": 2, "news": 1,
         "ticker_rank": 2, "universe": 2, "internal_metrics": 1,
         "internal_etf_flow": 1, "internal_futures_radar": 1,
         "internal_macro_calendar": 1}
preview = {k: v for k, v in bundle.items() if k != "frames"}
preview["_note"] = ("Same structure as input_bundle_example.json; rows truncated "
                    "for reading. Row counts of the full artifact are in "
                    "frames[*].full_row_count.")
preview["frames"] = {}
for key, spec in bundle["frames"].items():
    keep = _KEEP.get(key, 1)
    preview["frames"][key] = {**{k: v for k, v in spec.items() if k != "rows"},
                              "full_row_count": len(spec["rows"]),
                              "rows": spec["rows"][:keep]}
with open(PREVIEW_PATH, "w", encoding="utf-8") as _fh:
    json.dump(preview, _fh, ensure_ascii=False, indent=1)
PREVIEW_BYTES = os.path.getsize(PREVIEW_PATH)
PREVIEW_JSON = json.dumps(preview, ensure_ascii=False, indent=1)

# ── 2. load it back and run everything off it ─────────────────────────
snap = load_input_bundle(BUNDLE_PATH)
from cyqnt_trd.standard_bot.bot import _frames_from_snapshot, StandardBot, BotSpec, BotKind  # noqa: E402


class _Probe(StandardBot):
    spec = BotSpec(bot_id="_probe", kind=BotKind.ADVISORY)
    seen: dict = {}

    def required_data(self):
        return []

    def decide(self, ctx):
        _Probe.seen = {k: getattr(v, "shape", None) for k, v in ctx.frames.items()}
        _Probe.status = dict(ctx.source_status)
        return []


_Probe().run(snap)
CTX_FRAMES = dict(sorted(_Probe.seen.items()))

legacy_trade = BS.get_block_plugin("mtf_trend_follow").run(
    snap, SimpleNamespace(instrument_id=SYM, timeframe=ITV))
legacy_sel = BS.get_selection_plugin(N1.BOT_ID).run(
    snap, SimpleNamespace(market_type="futures"))
multi = MultiSourceBot(symbol=SYM, interval=ITV).run(snap)
v2_sel = BlocksNewsRankBot().run(snap)

m_payload = multi.signals[0].payload
IN_SCHEMA = json.load(open("strategies/_standard/input.schema.v1.json"))
OUT_SCHEMA = json.load(open("strategies/_standard/signal.schema.v2.json"))
V2_KEYS = set(OUT_SCHEMA["properties"])
m_core = {k: v for k, v in m_payload.items() if k in V2_KEYS}
s_core = {k: v for k, v in v2_sel.signals[0].payload.items() if k in V2_KEYS}
COMPAT = sorted(set(m_payload) - V2_KEYS)

nodes = list(list_nodes())
AVAIL = dict(collections.Counter(n.availability.value for n in nodes))

tests = subprocess.run([".venv-standard-bot/bin/python", "-m", "pytest", "tests/", "-q"],
                       capture_output=True, text=True)
TEST_LINE = next((l for l in reversed(tests.stdout.splitlines()) if "passed" in l), "(n/a)")


def j(o, n=1):
    return esc(json.dumps(o, ensure_ascii=False, indent=n, default=str))


# --------------------------------------------------------------------------- #
# The LIVE half: the same envelope, filled by calling the APIs.                #
#                                                                             #
# Read from disk rather than fetched here, so regenerating this page does not  #
# need the network (and does not quietly produce different numbers each run).  #
# Refresh it with:                                                            #
#     python -c "from cyqnt_trd.standard_bot.data import build_live_bundle;\   #
#                from cyqnt_trd.standard_bot.data.input_bundle import \        #
#                write_input_bundle as w;\                                     #
#                w(build_live_bundle(symbol='BTCUSDT', interval='1h',          #
#                  limit=300), 'docs/standard_bot_io/samples/'                 #
#                  'live_bundle_example.json')"                                #
# --------------------------------------------------------------------------- #
LIVE_PATH = "docs/standard_bot_io/samples/live_bundle_example.json"
LIVE = None
LIVE_BYTES = 0
PANEL = None
PANEL_CALLS = []
if os.path.exists(LIVE_PATH):
    with open(LIVE_PATH, encoding="utf-8") as fh:
        LIVE = json.load(fh)
    LIVE_BYTES = os.path.getsize(LIVE_PATH)
    from cyqnt_trd.standard_bot.data import to_panel                # noqa: E402

    PANEL = to_panel(LIVE)
    from cyqnt_trd.blocks import conditions as _C                   # noqa: E402
    from cyqnt_trd.blocks import derivatives as _DV                 # noqa: E402
    from cyqnt_trd.blocks import entry as _E                        # noqa: E402
    from cyqnt_trd.blocks import indicators as _I                   # noqa: E402

    _ema20, _ema50 = _I.ema(PANEL["close"], 20), _I.ema(PANEL["close"], 50)
    _adx = _I.adx(PANEL, 14)[0]
    _bools = {"trend": (_C.ma_cross_above(_ema20, _ema50), 1.0),
              "adx": (_C.adx_trending(_adx, 20.0), 0.8)}
    PANEL_CALLS = [
        ("indicators.ema(panel[\"close\"], 20)", "%.1f" % _ema20.iloc[-1]),
        ("indicators.adx(panel, 14)[0]", "%.1f" % _adx.iloc[-1]),
        ("indicators.atr(panel, 14)", "%.1f" % _I.atr(PANEL, 14).iloc[-1]),
        ("entry.weighted_score({trend, adx})", "%.2f" % _E.weighted_score(_bools).iloc[-1]),
    ]
    for _name, _series in [
        ("derivatives.oi_price_divergence(panel[\"close\"], panel[\"oi_value\"])",
         _DV.oi_price_divergence(PANEL["close"], PANEL["oi_value"])
         if "oi_value" in PANEL else None),
        ("derivatives.funding_rate_state(panel[\"rate\"] * 1e4)",
         _DV.funding_rate_state(PANEL["rate"] * 1e4) if "rate" in PANEL else None),
        ("derivatives.taker_buy_sell_state(panel[\"buy_vol\"], panel[\"sell_vol\"])",
         _DV.taker_buy_sell_state(PANEL["buy_vol"], PANEL["sell_vol"])
         if "buy_vol" in PANEL else None),
    ]:
        if _series is not None:
            PANEL_CALLS.append((_name, str(_series.iloc[-1])))

#: does the live artifact satisfy the contract it announces?
LIVE_SCHEMA_ERRORS = "n/a"
if LIVE is not None:
    try:
        import jsonschema

        LIVE_SCHEMA_ERRORS = str(len(list(
            jsonschema.Draft7Validator(IN_SCHEMA).iter_errors(LIVE))))
    except ImportError:
        LIVE_SCHEMA_ERRORS = "jsonschema not installed"


def pre(t):
    return "<pre>%s</pre>" % t


ok, warn, bad = ('<span class="ok">✅</span>', '<span class="warn">🟡</span>',
                 '<span class="bad">❌</span>')


def _live_section() -> str:
    """Section 1b, from the live artifact on disk."""
    if LIVE is None:
        return ('<div class="warnbox">尚未產生 <code>%s</code> —— '
                '本節需要一次 live 取數。</div>' % esc(LIVE_PATH))
    rows = []
    for key, entry in LIVE["frames"].items():
        status = entry.get("status", "")
        mark = ok if status == "ok" else (warn if status == "empty" else bad)
        note = entry.get("source_fallback") or entry.get("reason") or ""
        if entry.get("source_fallback"):
            note = "改由 public HTTPS 後備來源提供"
        rows.append(
            "<tr><td><code>%s</code></td><td><code>%s</code></td>"
            "<td align=right>%d</td><td>%s %s</td><td class='muted'>%s</td></tr>"
            % (esc(key), esc(entry.get("shape", "")), len(entry.get("rows") or ()),
               mark, esc(status), esc(note[:90])))
    served = sum(1 for e in LIVE["frames"].values() if e.get("status") == "ok")
    fell_back = [k for k, e in LIVE["frames"].items() if e.get("source_fallback")]
    return f"""
<div class="badbox">
 <b>先前的問題</b>:catalog 宣告的參數名和實際 fetcher 收的參數名<b>不一致</b>,
 而 <code>runtime/data.py</code> 會把宣告的預設值自動塞進去 ——
 所以 <code>data.klines(symbol=…, interval=…, limit=…)</code>
 <b>不管怎麼呼叫都會 TypeError</b>。25 個可呼叫節點裡 8 個是這樣,
 包含 <code>klines</code> 這個最基礎的節點。既有測試全綠,因為它只驗
 fetcher <b>import 得到</b>,沒驗<b>呼叫得動</b>。
</div>
<p>修法是在節點上宣告 <code>param_aliases</code> 做邊界翻譯 —— 對外保留
 <code>market_type</code>(那是 <code>DATA_API.md</code> 與 spec 的契約),
 對內轉成 fetcher 要的 <code>market</code>。護欄在
 <code>tests/standard_bot/test_node_params_match_fetchers.py</code>:
 <b>宣告的參數必須是 fetcher 簽名的子集</b>。</p>
<table><tr><th>frame key</th><th>shape</th><th>rows</th><th>status</th><th>備註</th></tr>
{"".join(rows)}</table>
{pre(f'''檔案     {LIVE_PATH}
大小     {LIVE_BYTES:,} bytes
節點     {served} / {len(LIVE["frames"])} 有資料
schema   cyqnt.input/v1 違規 {LIVE_SCHEMA_ERRORS}
時間軸   decision_time = {LIVE["decision_time"]}''')}
<div class="warnbox">
 <b>衍生品統計原本全滅。</b><code>open_interest</code> / <code>long_short_ratio</code> /
 <code>top_trader_ratio</code> / <code>taker_volume</code> 走本機
 <code>binance-cli</code> 子程序,而它回的是純文字 <code>ok</code> 不是 JSON。
 這正是為什麼未平倉量只能讀一個月前的 5 分鐘 parquet。
 <b>同樣的數字在 public 端點上拿得到,而且欄位名本來就對得上</b> ——
 <code>public_fallback</code> 接上之後這 {len(fell_back)} 個節點恢復,
 而且是 1h 顆粒度、跟 K 線逐根對齊。替換來源會寫進
 <code>source_fallback</code>,不靜默替換。
</div>"""


def _panel_section() -> str:
    """Section 2b: the shape blocks actually consume."""
    if PANEL is None:
        return '<div class="warnbox">需要 live artifact 才能產生本節。</div>'
    calls = "".join(
        "<tr><td><code>%s</code></td><td><code>%s</code></td></tr>"
        % (esc(name), esc(value)) for name, value in PANEL_CALLS)
    cols = ", ".join("<code>%s</code>" % esc(c) for c in PANEL.columns)
    return f"""
<div class="badbox">
 <b>bundle 本身餵不進 blocks。</b><code>cyqnt_trd.blocks</code> 每個函式要的是
 <code>pd.Series</code>,divergence 類的還要<b>兩個同 index 的 Series</b>。
 <code>BarFrame</code> 剛好符合(<code>bars["close"]</code> 就是 Series),
 但 <code>MetricFrame</code> 是 long form,交過去會有兩種下場:<br>
 · <code>oi_change_pct(frame)</code> → <code>TypeError</code>;<br>
 · <code>oi_change_pct(frame["value"])</code> → <b>不報錯,但值是錯的</b> ——
 那一欄把 <code>oi_base</code> 和 <code>oi_value</code> 交錯放在一起,
 算出來是「跨兩個不同指標的變化率」。<b>第二種是危險的那種</b>,
 而且呼叫端再小心也防不了。
</div>
<p><code>cyqnt_trd/standard_bot/data/panel.py</code> 的 <code>to_panel()</code>
 把 bundle 攤平到 <b>bar 時鐘</b>上,一個 metric 一欄:</p>
{pre(f'''from cyqnt_trd.standard_bot.data import build_live_bundle, to_panel

bundle = build_live_bundle(symbol="BTCUSDT", interval="1h", limit=300)
panel  = to_panel(bundle)          # {PANEL.shape[0]} 列 x {PANEL.shape[1]} 欄,index = bar close_time

panel["close"]                                            # 量價
derivatives.oi_price_divergence(panel["close"], panel["oi_value"])   # 衍生品
derivatives.funding_rate_state(panel["rate"] * 1e4)
panel["news_count_24h"]                                   # 事件聚合成每根 bar 的計數''')}
<p><b>欄位({PANEL.shape[1]} 個):</b> {cols}</p>
<h3>2b.1 實跑:blocks 零轉換直接吃</h3>
<table><tr><th>呼叫</th><th>最後一根 bar 的值</th></tr>{calls}</table>
<h3>2b.2 三個關鍵設計決定</h3>
<ol>
<li><b>對齊用 <code>available_time</code>,不用 <code>event_time</code>。</b>
 一根 bar 只能看到它收盤時<b>已經可知</b>的值。<code>event_time</code> 是事情
 發生的時間,可能早於可發布時間 —— 用它對齊就是前視。</li>
<li><b>最後一根 bar 的截止點是 <code>decision_time</code>,不是 bar 收盤。</b>
 那一列就是「現在要下的決策」:實盤 bot 在 bar 收完後決策時,確實知道當下的
 24h ticker / 盤口 / 社群熱度,而這些的時間戳都晚於收盤。若最後一列也用收盤截,
 所有快照類來源會全部是 NaN —— 看起來嚴謹,但唯一有用的那一列變成空的。
 <b>前面每一根仍用自己的收盤截,所以回放是誠實的。</b></li>
<li><b>撞名帶節點名。</b><code>long_short_ratio</code> 和
 <code>top_trader_ratio</code> 都產出一個叫 <code>long_short_ratio</code> 的
 metric,但量的是不同族群(全部帳戶 vs 頂級交易者)。所以第二個叫
 <code>top_trader_ratio.long_short_ratio</code>,而不是
 <code>long_short_ratio_2</code>。</li>
</ol>
<p class="muted">多標的:<code>to_panel(bundle, symbols=[...])</code> 回
 <code>(symbol, 欄位)</code> 的 MultiIndex,供選幣 / 截面策略用;
 <code>panel["BTCUSDT"]</code> 切回單標的就是 blocks 吃的那張表。
 防前視的測試在 <code>tests/standard_bot/test_panel.py</code>。</p>"""

LIVE_SECTION = _live_section()
PANEL_SECTION = _panel_section()


def _schema_check() -> tuple:
    """Validate every cyqnt.input/v1 artifact, and prove the schema can fail."""
    import copy

    names = ("live_bundle_example.json", "input_bundle_example.json",
             "input_bundle_preview.json", "bot_context.json")
    try:
        import jsonschema
    except ImportError:
        return ("jsonschema 未安裝 —— 無法在本頁驗證",
                "jsonschema 未安裝 —— 污染測試無法執行")

    validator = jsonschema.Draft7Validator(IN_SCHEMA)
    lines = []
    for name in names:
        path = os.path.join("docs/standard_bot_io/samples", name)
        if not os.path.exists(path):
            lines.append("%-32s (不存在)" % name)
            continue
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        count = len(list(validator.iter_errors(doc)))
        lines.append("%-32s 違規 %d   %s" % (name, count, "OK" if count == 0 else "FAIL"))

    live_path = "docs/standard_bot_io/samples/live_bundle_example.json"
    if not os.path.exists(live_path):
        return ("\n".join(lines), "需要 live artifact 才能執行污染測試")
    with open(live_path, encoding="utf-8") as fh:
        base = json.load(fh)
    cases = [
        ("數值欄塞字串",
         lambda d: d["frames"]["klines"]["rows"][0].update({"close": "not-a-number"})),
        ("刪掉必填欄位",
         lambda d: d["frames"]["klines"]["rows"][0].pop("instrument_id")),
        ("shape 與 rows 不符",
         lambda d: d["frames"]["klines"].update({"shape": "MetricFrame@1.0"})),
        ("status 用未定義的值",
         lambda d: d["frames"]["klines"].update({"status": "probably_fine"})),
        ("frame 沒有 shape", lambda d: d["frames"]["klines"].pop("shape")),
        ("rows 不是陣列",
         lambda d: d["frames"]["klines"].update({"rows": {"close": 1.0}})),
        ("decision_time 用 ISO 字串",
         lambda d: d.update({"decision_time": "2026-08-06T07:00:00Z"})),
        ("schema id 錯", lambda d: d.update({"schema": "cyqnt.input/v2"})),
    ]
    checks = []
    for label, mutate in cases:
        polluted = copy.deepcopy(base)
        mutate(polluted)
        caught = len(list(validator.iter_errors(polluted)))
        checks.append("%-24s %s" % (label, "reject ✓" if caught else "*** 沒抓到 ***"))
    return "\n".join(lines), "\n".join(checks)


SCHEMA_TABLE, POLLUTION_BLOCK = _schema_check()

_DESC = {
    "klines": "OHLCV K 線", "funding": "資金費率 + mark price",
    "open_interest": "未平倉量 + OI 名目價值", "liquidations": "清算",
    "news": "新聞事件", "ticker_rank": "Square 提及量 / 情緒排名",
    "universe": "選幣宇宙(24h ticker)", "internal_metrics": "自訂 metric(擴充點示範)",
}
frame_rows = "".join(
    "<tr><td><code>%s</code></td><td><code>%s</code></td><td align=right>%s</td>"
    "<td>%s</td></tr>" % (
        esc(k), esc(v["shape"]), len(v["rows"]),
        esc(_DESC.get(k) or (INTERNAL_SLOTS[k].description
                             if k in INTERNAL_SLOTS else "")))
    for k, v in bundle["frames"].items())

internal_rows = "".join(
    "<tr><td><code>%s</code></td><td><code>%s</code></td><td>%s</td>"
    "<td><code>%s</code></td></tr>" % (
        esc(s.key), esc(s.shape),
        '<span class="ok">可回測</span>' if s.pit_safe
        else '<span class="warn">僅 forward</span>',
        esc(bundle["source_status"].get(s.key, "-")))
    for s in INTERNAL_SLOTS.values())

shape_rows = "".join(
    "<tr><td><code>%s</code></td><td>%s</td><td><code>%s</code></td></tr>"
    % (esc(fs.name), esc(fs.row_grain), esc(", ".join(fs.required)))
    for fs in IC.SCHEMAS.values())

ctx_rows = "".join("<tr><td><code>%s</code></td><td><code>%s</code></td></tr>"
                   % (esc(k), esc(v)) for k, v in CTX_FRAMES.items())

STATUS_BLOCK = ("source_status = " + j(bundle["source_status"])
                + "\n\nwarnings = " + j(bundle["warnings"]))

ev_rows = "".join(
    "<tr><td><code>%s</code></td><td><code>%s</code></td></tr>"
    % (esc(e["source"]), esc(json.dumps(e["observed"], ensure_ascii=False)[:110]))
    for e in m_payload["evidence"])

BOUNDED_KB = "{:,}".format(BUNDLE_BYTES // 1024)
PRE_SIZE = pre(
    "未裁切:  %s bytes  (%s 列)\n"
    "  裁切:  %s bytes  (%s 列)   ← 縮小 %.1f%%" % (
        "{:,}".format(UNBOUNDED_BYTES),
        "{:,}".format(sum(len(v["rows"]) for v in UNBOUNDED["frames"].values())),
        "{:,}".format(BUNDLE_BYTES),
        "{:,}".format(sum(len(v["rows"]) for v in bundle["frames"].values())),
        (1 - BUNDLE_BYTES / UNBOUNDED_BYTES) * 100))
PRE_LOOKBACK = pre(esc('''build_input_bundle(
    ...,
    max_bars=300,            # K 線:最多 300 根
    metric_lookback=240,     # 每個 (instrument, metric) 最新 240 筆(預設)
    max_event_rows=200,      # 事件 / 排名類 frame 上限(預設)
)
# 要全量歷史就傳 None —— 但那是明確的選擇,不是預設'''))

PRE_AB = pre(
    "公開環境(無內網 client):  frames=%d  internal 插槽全部 status='unavailable'\n"
    "內網環境(client 有資料):  frames=%d  多了 %s\n"
    "source_status 的 key 集合完全相同: %s" % (
        len(PUBLIC_BUNDLE["frames"]), len(bundle["frames"]),
        esc(sorted(set(bundle["frames"]) - set(PUBLIC_BUNDLE["frames"]))),
        SAME_STATUS_KEYS))
PRE_USAGE = pre(esc('''bundle = build_input_bundle(
    symbol="BTCUSDT", interval="1h", decision_time=DT,
    historical_dir="data/mtf_90d",                    # K線
    derivatives_dir="data/derivatives_mvp_30d",       # funding / OI
    news_frame=news_df, ticker_rank_frame=rank_df,    # Square
    declare_internal=list(INTERNAL_SLOTS),            # 申報全部內網插槽(有欄位)
    internal_frames={"internal_etf_flow": etf_df},    # 內網環境才填得出來
)'''))
PRE_PITWARN = pre(esc([w for w in bundle["warnings"] if "point-in-time" in w][:1] or ["(無)"]))

HTML = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>標準 Bot 完整流程 — 一份輸入 JSON,一份輸出訊號</title>
<style>
 :root{{--bg:#0f1420;--panel:#161d2e;--panel2:#1c2740;--ink:#e6ebf5;--muted:#9aa7c2;
  --line:#2a3552;--accent:#5aa9ff;--green:#39d98a;--amber:#ffcc66;--red:#ff6b6b;--chip:#233150}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,
  "Segoe UI","PingFang TC","Microsoft JhengHei",sans-serif;line-height:1.66;font-size:15px}}
 .wrap{{max-width:1140px;margin:0 auto;padding:32px 22px 90px}}
 header{{border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:26px}}
 h1{{font-size:26px;margin:0 0 6px}} .sub{{color:var(--muted);font-size:14px}}
 .meta{{margin-top:12px;display:flex;gap:8px;flex-wrap:wrap}}
 .chip{{background:var(--chip);color:#cfe0ff;border:1px solid var(--line);border-radius:999px;
  padding:3px 11px;font-size:12.5px}}
 h2{{font-size:19px;margin:42px 0 14px;padding-left:11px;border-left:4px solid var(--accent)}}
 h3{{font-size:15.5px;margin:24px 0 8px;color:#cfe0ff}}
 h4{{font-size:13.5px;margin:16px 0 4px;color:#a8c4ee}}
 p{{margin:10px 0}} .muted{{color:var(--muted);font-size:13px}}
 .lead{{background:#15203a;border:1px solid var(--line);border-left:4px solid var(--green);
  border-radius:10px;padding:16px 20px;margin:14px 0}} .lead b{{color:var(--green)}}
 .warnbox{{background:#241f14;border:1px solid #4a3c1d;border-left:4px solid var(--amber);
  border-radius:10px;padding:14px 18px;margin:14px 0}}
 .badbox{{background:#241618;border:1px solid #4d2226;border-left:4px solid var(--red);
  border-radius:10px;padding:14px 18px;margin:14px 0}}
 code{{font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace;font-size:12.7px;
  background:#0c1220;border:1px solid var(--line);border-radius:5px;padding:1px 6px;color:#bfe0ff}}
 pre{{background:#0b1120;border:1px solid var(--line);border-radius:10px;padding:14px 16px;
  overflow:auto;font-size:12.2px;color:#c8d6f2;line-height:1.5;
  font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace}}
 table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:13.3px}}
 th,td{{border:1px solid var(--line);padding:8px 11px;text-align:left;vertical-align:top}}
 th{{background:var(--panel2);color:#cfe0ff;font-weight:600}}
 tr:nth-child(even) td{{background:#131a2b}}
 .ok{{color:var(--green);font-weight:600}} .warn{{color:var(--amber);font-weight:600}}
 .bad{{color:var(--red);font-weight:600}}
 ul,ol{{margin:8px 0 8px 2px;padding-left:22px}} li{{margin:5px 0}}
 .flow{{display:flex;align-items:stretch;gap:0;flex-wrap:wrap;margin:18px 0}}
 .node{{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:12px 14px;
  min-width:118px;flex:1;text-align:center}}
 .node .t{{font-weight:700;color:#cfe0ff;font-size:13px}}
 .node .d{{color:var(--muted);font-size:11.4px;margin-top:4px}}
 .node.g{{border-color:#2c6b4a}} .node.r{{border-color:#6b2c30}}
 .arrow{{display:flex;align-items:center;justify-content:center;color:var(--accent);
  font-size:20px;padding:0 7px}}
 .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
 @media(max-width:880px){{.grid2{{grid-template-columns:1fr}}}}
 .verdict{{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0}}
 .vcard{{flex:1;min-width:250px;background:var(--panel);border:1px solid var(--line);
  border-radius:12px;padding:15px 17px}}
 .vcard .big{{font-size:18px;font-weight:700;margin-bottom:5px}}
 .toc{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 20px}}
 .toc a{{color:#bfe0ff;text-decoration:none}} .toc a:hover{{text-decoration:underline}}
</style></head><body><div class="wrap">

<header>
 <h1>標準 Bot 完整流程 — 一份輸入 JSON,一份輸出訊號</h1>
 <div class="sub">全部資料源(K線 / funding / OI / 新聞 / 選幣 / 內網)合成同一份 JSON,
  同一個 timestamp,運算後輸出統一格式訊號。所有數字皆為實跑捕捉。</div>
 <div class="meta">
  <span class="chip">輸入 {esc(IN_SCHEMA['$id'])}</span>
  <span class="chip">輸出 {esc(OUT_SCHEMA['$id'])}</span>
  <span class="chip">bundle {BUNDLE_BYTES:,} bytes</span>
  <span class="chip">{esc(TEST_LINE)}</span>
 </div>
</header>

<div class="verdict">
 <div class="vcard"><div class="big">{ok} 輸入統一</div><div class="muted">
  <b>{len(bundle['frames'])} 個資料源、一份 JSON、一個 decision_time</b>。
  funding / OI / 內網 metric 共用同一套欄位。</div></div>
 <div class="vcard"><div class="big">{ok} 輸出統一</div><div class="muted">
  交易與選幣 <b>{len(m_core)} 個 key 完全相同</b>。
  兩份 JSON Schema 都從 dataclass 自動生成。</div></div>
 <div class="vcard"><div class="big">{warn} live 未接</div><div class="muted">
  executor 仍讀 <code>trades.jsonl</code>(成交記錄,無停損),見 §6</div></div>
</div>

<div class="toc"><b>目錄</b> &nbsp;
 <a href="#s1">1 統一輸入 bundle</a> · <a href="#slive">1b live 取數</a> ·
 <a href="#s2">2 七種 frame shape</a> · <a href="#spanel">2b blocks 輸入格式</a> ·
 <a href="#s3">3 多源策略</a> · <a href="#s4">4 統一輸出</a> ·
 <a href="#s5">5 兩份 Schema</a> · <a href="#s6">6 未完成</a>
</div>

<h2>0. 全流程</h2>
<div class="flow">
 <div class="node"><div class="t">K線</div><div class="d">parquet</div></div>
 <div class="node"><div class="t">funding/OI</div><div class="d">parquet</div></div>
 <div class="node"><div class="t">新聞/Square</div><div class="d">API</div></div>
 <div class="node"><div class="t">內網 BigData</div><div class="d">HTTP</div></div>
 <div class="arrow">→</div>
 <div class="node g"><div class="t">① 一份 JSON</div><div class="d">cyqnt.input/v1</div></div>
 <div class="arrow">→</div>
 <div class="node"><div class="t">② blocks 運算</div><div class="d">多源共同決策</div></div>
 <div class="arrow">→</div>
 <div class="node g"><div class="t">③ 一份訊號</div><div class="d">cyqnt.signal/v2</div></div>
 <div class="arrow">→</div>
 <div class="node r"><div class="t">④ 回測/live</div><div class="d">live 未接</div></div>
</div>
{pre('''build_input_bundle(...)  →  input_bundle_example.json   (%s bytes)
load_input_bundle(path)  →  DataSnapshot
                         →  BotContext.frames  (%d 個源全部到達策略)
                         →  MultiSourceBot     →  cyqnt.signal/v2''' % (
    "{:,}".format(BUNDLE_BYTES), len(CTX_FRAMES)))}

<h2 id="s1">1. 統一輸入 bundle</h2>
<div class="badbox">
 <b>先前的問題</b>:「輸入」在每個資料源都是不同格式 —— OHLCV 有 <code>--input-json</code>,
 funding/OI 是 <code>--derivatives-dir</code> 底下的 parquet,Square 新聞<b>根本沒有檔案格式</b>
 (只有 live API),內網 BigData 又是自己的 client。
 想同時吃價格 + funding + 新聞的策略<b>無法用一個檔案餵</b>,
 也就意味著一次執行無法重現、無法版控、無法 diff、無法交給別人。
</div>
<p><code>cyqnt_trd/standard_bot/data/input_bundle.py</code> 就是那個單一檔案。
本頁的 bundle 是<b>用 repo 自己的資料實跑產生的</b>:</p>

<h3>1.1 一份 JSON 裡有什麼</h3>
{pre(f'''decision_time = {DT}      ← 一個時間戳,全部資料對齊到它
snapshot_id   = {bundle["snapshot_id"]}''')}
<table><tr><th>frame key</th><th>canonical shape</th><th>rows</th><th>內容</th></tr>{frame_rows}</table>
{pre(STATUS_BLOCK)}
<p class="muted">完整檔案:<code>{esc(BUNDLE_PATH)}</code>({BUNDLE_BYTES:,} bytes)。
 讀寫用 <code>build_input_bundle()</code> / <code>load_input_bundle()</code>。</p>

<h3>1.2 實際的 JSON 長什麼樣</h3>
<p>下面是<b>完整結構</b>,每個 frame 保留真實資料列(完整檔 1.7 MB,列數見
 <code>full_row_count</code>)。可讀版同步寫到
 <code>{esc(PREVIEW_PATH)}</code>({PREVIEW_BYTES:,} bytes):</p>
{pre(esc(PREVIEW_JSON))}
<p class="muted">注意 <code>klines</code> 的一列就是一根 K 棒的開高低收量;
 <code>funding</code> / <code>open_interest</code> / <code>internal_etf_flow</code> 三個不同來源
 <b>用的是同一組欄位</b>(<code>instrument_id</code> / <code>metric</code> / <code>value</code> /
 <code>event_time</code> / <code>available_time</code>),只有 <code>metric</code> 的值不同 ——
 這就是「一套詞彙」在實際資料上的樣子。</p>

<h3>1.3 回看窗格 — 為什麼一個決策不是 1.7 MB</h3>
<div class="badbox"><b>第一版做錯了</b>:K 線有 <code>max_bars</code> 上限,
 <b>metric frame 卻完全沒有上限</b>。結果一個 <b>1 小時</b>的決策把 30 天的
 <b>5 分鐘</b>未平倉量整段拖進來 —— <code>open_interest</code> 一個 frame
 12,144 列,佔掉 1.7 MB 檔案的 <b>94%</b>,而那些資料沒有任何策略會讀。</div>
<p>bundle 是<b>一個決策時點</b>的輸入,所以每個序列只需要一個<b>回看窗格</b>,不是全部歷史。
 <code>_tail_per_series()</code> 對每個 <code>(instrument_id, metric)</code> 只留最新 N 筆:</p>
<table><tr><th>frame</th><th>未裁切列數</th><th>裁切後</th><th>bytes</th></tr>{SIZE_ROWS}</table>
{PRE_SIZE}
<p><b>策略輸出完全沒變</b> —— 同樣 <code>{esc(m_core["intent"])}</code>、
 同樣 <code>score={esc(m_core["score"])}</code>、同樣 4 個 reason_codes、evidence 數值一模一樣。</p>
{PRE_LOOKBACK}
<p class="muted"><b>沒修的第二個成本</b>:長表每列都重複欄位名 ——
 一列 MetricFrame 約 136 bytes,其中 <b>63 bytes(46%)是欄位名</b>。
 這是「一套詞彙」換來的代價:因為每列自帶完整標籤,funding / OI / 內網 metric 才能用
 同一個 accessor 讀。改欄式編碼可再省約 46%,但會失去「一列就是一筆、可讀」的特性,
 現在 {BOUNDED_KB} KB 還不值得付這個代價。</p>

<h3>1.4 兩個讓它值得存在的不變量</h3>
<div class="lead">
 <b>① 一個時鐘。</b>每一列在寫出前都過 <code>available_time &lt;= decision_time</code>。
 <code>available_time</code> 是「我們最早何時可能<em>知道</em>這一列」,不是「事情何時<em>發生</em>」——
 把兩者混為一談,就是 walk-forward 悄悄讀到未來的方式。<b>只 gate 一次、在建 bundle 時</b>,
 下游任何讀取者都不可能弄錯。
</div>
<div class="lead">
 <b>② 一套詞彙。</b>funding、OI、清算、<b>任何內網 metric</b> 全部落成 <code>MetricFrame</code>
 (<code>instrument_id</code> / <code>metric</code> / <code>value</code> /
 <code>event_time</code> / <code>available_time</code>)。
 一支讀三種的策略用<b>同一組欄位名</b>,而<b>新增一個資料源不需要任何新管線</b> ——
 它就是多幾列 MetricFrame。
</div>

<h3>1.5 內網資料:欄位在公開 repo,client 不在</h3>
<div class="badbox"><b>約束</b>:內網 BigData 節點(indicators API / futuresRadar / 熱點事件 /
 事件曆 / 代幣解鎖 / 總經 / ETF 流向 / 板塊輪動 / 大額進出金…)是策略真的要用的輸入,
 但它們的 <b>client 不能進這個 repo</b> —— 它硬編內網主機名,而本 repo 是
 <b>PUBLIC</b>。那些主機名連在註解裡都不寫,寫下來就等於發布。</div>
<p><code>data/internal_slots.py</code> 把兩者拆開:<b>欄位契約留在公開 repo,fetcher 只用字串路徑引用</b>。
 這個模組<b>不 import</b> 任何內網套件;沒裝時每個插槽回報 <code>unavailable</code>,
 而 bundle 的結構完全不變。已宣告 <b>{len(INTERNAL_SLOTS)} 個插槽</b>:</p>
<table><tr><th>插槽 key</th><th>canonical shape</th><th>可回測性</th><th>本次 status</th></tr>
{internal_rows}</table>
<div class="lead"><b>實測:公開環境 vs 內網環境,結構完全相同。</b>
{PRE_AB}</div>
<p>擴充點的用法:</p>
{PRE_USAGE}
<div class="warnbox"><b>PIT 旗標會跟著宣告走。</b>多數內網節點是 Redis-TTL 快照、
 <b>沒有時點歷史</b>,標為<code>僅 forward</code>。bundle 建立時若填入這類資料會自動加警告:
 {PRE_PITWARN}</div>
<div class="warnbox">
 <b>OI 檔名綁週期的坑改成 fail-soft</b>:找不到 <code>open_interest_1h.parquet</code> 時
 退用 <code>open_interest_5m.parquet</code>,<b>並把這件事寫進 <code>warnings</code></b>。
 原本是靜默完全沒有 OI 欄位。
</div>

<h3>1.6 策略實際看到的(讀回來之後)</h3>
<table><tr><th>ctx.frames key</th><th>shape</th></tr>{ctx_rows}</table>
<p class="muted">{len(CTX_FRAMES)} 個 frame 全部到達 <code>BotContext</code>,
 <code>source_status</code> 帶 {len(_Probe.status)} 個節點(含讀失敗的)。
 <code>klines</code> 同時以 <code>klines:BTCUSDT:1h</code> 提供,供多標的策略指名取用。</p>

<h3>1.7 catalog:哪些資料能拿來回測</h3>
<p><code>data/catalog.py</code> 收錄 <b>{len(nodes)} 個</b>資料節點,每個都標明可回測性:</p>
{pre(j(AVAIL))}
<div class="warnbox">
 <b>只有 {AVAIL.get('BACKTESTABLE', 0)} 個節點是 BACKTESTABLE。</b>
 {AVAIL.get('FORWARD_ONLY', 0)} 個是 <code>FORWARD_ONLY</code> —— Redis-TTL 快照,
 <b>沒有時點歷史</b>,拿去 walk-forward 會在每一根重複使用「今天的值」。
 catalog 用 <code>pit_hazard</code> 欄位逐一寫明它「會怎麼騙你」。
 <b>「有這個資料」不等於「能拿來回測」。</b>
</div>

<h2 id="slive">1b. 同一個 envelope,live 取數版</h2>
<p>上面那份 bundle 是<b>讀本機檔案</b>組出來的。實盤要的是<b>直接打 API</b> ——
 <code>cyqnt_trd/standard_bot/data/live_bundle.py</code> 的
 <code>build_live_bundle()</code> 做這件事,吐出<b>完全相同的 envelope</b>,
 所以 paper / live 產訊號讀到的檔案形狀跟回放一模一樣。</p>
{LIVE_SECTION}

<h2 id="s2">2. 七種 canonical frame shape</h2>
<p>bundle 裡每個 frame 都是這七種之一。正規化後所有資料共用同一組欄位詞彙:</p>
<table><tr><th>Shape</th><th>row grain</th><th>必填欄位</th></tr>{shape_rows}</table>
<p class="muted">對映表在 <code>input_bundle.FRAME_SHAPES</code>
 ({len(FRAME_SHAPES)} 個 key)。加資料源 = 加一行對映,不是加一個容器。</p>

<h2 id="spanel">2b. blocks 的輸入格式 — <code>to_panel()</code></h2>
{PANEL_SECTION}

<h2 id="s3">3. 多源策略 — 合併後運算</h2>
<p>參考實作 <code>strategies/standard/multi_source_bot.py</code>。
 讀四個源,共同決定一個訊號:</p>
{pre(esc('''# 價格定方向(blocks)
ema_fast = I.ema(close, cfg["ema_fast"]);  ema_slow = I.ema(close, cfg["ema_slow"])
atr      = I.atr(df, cfg["atr_period"])
long_side = ema_fast.iloc[i] > ema_slow.iloc[i]

# funding 否決擁擠方向 —— 付錢持有共識部位,是 edge 的反面
funding = latest_metric(ctx, "funding", "funding_rate", symbol)
if crowded: return [hold(...)]          # 直接不交易

# OI 確認:新錢進場 vs 部位平倉的擠壓
oi_chg = metric_change(ctx, "open_interest", "open_interest", lookback=24)
score = score + 25 if oi_chg >= 0.01 else score * 0.5

# 新聞加分
news = ctx.frame("news")   # EventFrame'''))}
<div class="lead"><b>重點不是這個 alpha,而是 funding / OI / 內網 metric
 用<em>同樣三行程式碼</em>讀取</b> —— 因為它們都是 MetricFrame:
{pre(esc('''funding = latest_metric(ctx, "funding",          "funding_rate",     symbol)
oi_chg  = metric_change(ctx, "open_interest",   "open_interest", lookback=24)
etf     = latest_metric(ctx, "internal_metrics","etf_net_flow_usd", symbol)'''))}
 加第五個 metric 來源,這裡<b>一行都不用改</b>。</div>

<h3>3.1 實跑輸出</h3>
{pre(j({k: m_core[k] for k in ("schema","bot_id","intent","target_side","symbol",
    "score","confidence","summary","reason_codes","data_quality") if k in m_core}))}
<table><tr><th>evidence 來源</th><th>observed</th></tr>{ev_rows}</table>
<p class="muted">四個源各自留痕。<code>data_quality =
 {esc(m_core.get("data_quality"))}</code> 是對的 —— <code>liquidations</code> 是
 <code>empty</code>(repo 只有 1 筆且日期落在 K 線之外)。<b>缺料會降級但不會崩,而且降級看得見。</b></p>

<h3>3.2 同一份 JSON 也餵得動其他策略</h3>
{pre(f'''舊 block 策略 (make_signals)        → {len(legacy_trade.trade_signals())} 訊號
舊 selection 策略                   → {len(legacy_sel.selection_signals()[0].payload["candidates"])} 候選
v2 MultiSourceBot   (TRADE)        → {len(multi.signals)} 訊號
v2 BlocksNewsRankBot(SELECTION)    → {len(v2_sel.signals)} 訊號''')}

<h2 id="s4">4. 統一輸出 — <code>cyqnt.signal/v2</code></h2>
<div class="lead"><b>交易與選幣同一格式。</b>核心 key-set <b>{len(m_core)} 個完全相同</b>
 (差集 = {esc(sorted(set(m_core) ^ set(s_core)) or '無')})。差別只在<b>填哪些欄位</b>:
 trade 填 <code>entry / exit_plan / size</code>,selection 填
 <code>candidates / universe_size</code>,advisory 填 <code>advisory_action / summary</code>。
 消費端<b>解析一套 schema,靠 <code>kind</code> 這一個欄位分流</b>
 (<code>trade</code> / <code>selection</code> / <code>alert</code>)。</div>
<div class="badbox"><b><code>kind</code> 是這次補回來的。</b>v2 第一版沒有它,要消費端自己從
 <code>market_scope == cross_section</code> 加上 <code>candidates</code> 非空去推。
 那個推論不成立:<code>BotSpec.market_scope</code> 預設是 <code>single</code>,
 一支沒覆寫它的選幣 bot 會發出<b>標成 single 的截面籃子</b>
 (<code>BlocksNewsRankBot</code> 實測就是這樣)。
 <code>cyqnt.signal/v1</code> 本來就有 <code>kind</code>,弄丟它是退步。
 現在 <code>kind</code> <b>由 payload 推導而非由生產者宣告</b> ——
 貼錯標籤會在 <code>__post_init__</code> 直接丟 <code>ValueError</code>,
 而 <code>market_scope</code> 也會被一併校正。</div>
<div class="grid2">
 <div><h4>TRADE(多源策略)</h4>{pre(j({k: m_core[k] for k in
   ("kind","intent","target_side","symbol","entry","exit_plan","size","candidates")
   if k in m_core}))}</div>
 <div><h4>SELECTION(選幣策略)</h4>{pre(j({k: s_core[k] for k in
   ("kind","intent","target_side","symbol","entry","exit_plan","size","candidates")
   if k in s_core}))}</div>
</div>

<h3>4.1 為什麼是 v2 而不是 v1</h3>
<table><tr><th></th><th>v1(<code>signal.schema.json</code>)</th><th>v2</th></tr>
<tr><td>部位指令</td>
 <td class="bad"><code>side ∈ long|short|flat</code> × <code>action ∈ entry|exit|adjust</code>
  —— <b>兩個欄位的組合,而且沒有任何驗證</b>。<code>side=flat, action=entry</code>
  這種無意義組合照樣通過;<b>加倉與開倉都是 <code>long/entry</code></b>、
  <b>減倉與全平都是 <code>long/exit</code></b>,分不出來;反手要拆成兩筆</td>
 <td class="ok"><b>一個</b>封閉列舉 <code>intent</code>(12 值)+ 框架自動導出
  <code>target_side</code> / <code>closes_side</code> / <code>order_side</code> /
  <code>reduce_only</code>,消費端不需要自己組合推理</td></tr>
<tr><td>誰能發平倉</td><td class="bad">無限制</td>
 <td class="ok">沒宣告 <code>reads_positions=True</code> 的 bot 發 close/reduce/flip 會被
  <code>decide_checked()</code> 丟 <code>CapabilityError</code> —— 沒看過部位就不能斷言部位存在</td></tr>
<tr><td>出場</td><td class="warn"><code>stop_loss</code> + <code>take_profit[]</code> 兩個裸欄位,
  沒有 trailing / time_stop</td>
 <td class="ok"><code>ExitPlan</code>:stop / TP 階梯 / trailing / time_stop,<b>隨訊號走</b></td></tr>
<tr><td>型別判斷</td><td class="ok"><code>kind ∈ trade|selection|alert|noop</code></td>
 <td class="ok">同一個 <code>kind</code>(v2 第一版漏了,已補回;而且改為<b>推導</b>而非宣告)</td></tr>
<tr><td>三種 bot</td><td class="warn">trade / selection 各自一組欄位,以註解分段</td>
 <td class="ok">三者同一組 key,靠 <code>kind</code> 分流</td></tr>
<tr><td>產出者</td><td class="bad">每支策略手寫 <code>generate()</code></td><td class="ok">框架產生</td></tr>
</table>
<p class="muted">v1 唯一勝過 v2 第一版的地方就是 <code>kind</code>。把交易與選幣放進同一個
 envelope,前提是有一個欄位說得出「這是哪一種」—— 那個欄位不該靠推論。</p>

<h3>4.2 v2 → 既有引擎的橋接</h3>
<div class="badbox"><b>原本 v2 bot 跑不了回測</b>:<code>runner.py</code> 做
 <code>float(payload["size"])</code> —— v2 的 <code>size</code> 是 <b>SizeSpec 物件</b> → TypeError;
 它讀 <code>payload["exit_spec"]</code>,v2 叫 <code>exit_plan</code> → <b>停損被靜默忽略</b>。</div>
<p>在傳輸邊界加轉譯(不動引擎),v2 dict 原封保留,額外附上引擎要的 key:</p>
{pre(j({k: m_payload[k] for k in COMPAT if k in m_payload}))}
<p class="muted"><code>engine_size</code> 刻意<b>不叫</b> <code>size</code> ——
 覆寫掉會讓 payload 失去自身 schema 合規性。此處 <code>RISK_PCT 1%</code> ÷ 停損距離
 = <code>{esc(m_payload.get("engine_size"))}</code>。</p>
<h4>三個「拒絕猜測」的決定</h4>
<ol>
<li><code>QUANTITY</code> / <code>QUOTE_AMOUNT</code> / <code>POSITION_PCT</code> <b>不換算</b>
 —— 需要帳戶權益或現有部位。引擎預設是 <b>1.0(滿倉)</b>,猜錯就是靜默滿倉,
 所以設 <code>engine_size=null</code> + <code>size_unresolved</code> 說明。</li>
<li><code>RISK_PCT</code> 只在停損距離可解時換算(上限 1.0)。</li>
<li><b>分批止盈只保留第一段</b>並記錄 <code>_dropped_tp_legs</code> ——
 沒有任何模擬引擎支援 partial close,靜默截斷比直說更糟。</li>
</ol>

<h2 id="s5">5. 兩份 JSON Schema</h2>
<table><tr><th>檔案</th><th>契約</th><th>規模</th><th>產生器</th></tr>
<tr><td><code>strategies/_standard/input.schema.v1.json</code></td><td>{esc(IN_SCHEMA['$id'])}</td>
 <td>{len(IN_SCHEMA['definitions'])} 個 frame shape</td>
 <td><code>docs/gen_input_schema_v1.py</code></td></tr>
<tr><td><code>strategies/_standard/signal.schema.v2.json</code></td><td>{esc(OUT_SCHEMA['$id'])}</td>
 <td>{len(OUT_SCHEMA['definitions'])} definitions / {len(OUT_SCHEMA['properties'])} properties</td>
 <td><code>docs/gen_signal_schema_v2.py</code></td></tr>
</table>
<p>兩份都<b>從 dataclass 自動生成</b> —— 改了程式碼重跑就同步,不會漂。
 用輸出 schema 驗證 6 份實際輸出:全部 <b>違規 0</b>。</p>

<h3>5.1 輸入 schema 原本驗不到任何東西</h3>
<div class="badbox">
 <b>同一個 <code>cyqnt.input/v1</code> 有三種方言。</b>
 <code>input_bundle_example.json</code> 用 <code>frames[node] = {{shape, rows}}</code>、
 整數 ms;<code>bot_context.json</code> 把 rows 包在另一種 key 的外殼裡、用 ISO 字串;
 而 schema 自己描述的是<b>第三種</b>(<code>typed</code> 底下的裸陣列)。
 三者都宣告自己是 <code>cyqnt.input/v1</code>。<br><br>
 更糟的是 <code>frames</code> 當時是 <code>additionalProperties: true</code>,
 而每個真實產出者寫的都是 <code>frames</code> 不是 <code>typed</code> ——
 <b>七個 shape 定義從來沒有被執行過</b>。把 K 線那格塞成
 <code>[{{"close": "not-a-number"}}]</code> 仍然驗過。
</div>
<p>現在 schema 描述<b>外殼</b>,並用 <code>if shape == "BarFrame@1.0" then rows: [...]</code>
 把 rows 綁到它宣告的 shape 上。外殼不是包裝:<code>status</code> / <code>reason</code>
 是「沒讀到」和「讀了是空的」唯一分得開的地方,<code>source_fallback</code>
 讓「這次是換來源拿的」看得見。四份 artifact 現在<b>同一個形狀</b>:</p>
{pre(SCHEMA_TABLE)}
<p>而且它現在<b>真的會拒絕</b> ——
 <code>tests/standard_bot/test_input_schema_v1.py</code> 對每一種污染都斷言必須被抓到
 (之前這 8 種全部靜默通過):</p>
{pre(POLLUTION_BLOCK)}
<p class="muted">schema 一改好就<b>立刻抓到一個真缺陷</b>:
 <code>internal_macro_calendar</code> 宣告 <code>EventFrame@1.0</code> 卻缺必填的
 <code>event_id</code> / <code>source_id</code> / <code>topic</code> —— 已修
 (<code>event_id</code> 由內容 hash 推導,可跨輪去重)。</p>
<p class="muted">現成 sample:<code>docs/standard_bot_io/samples/</code> ——
 <b><code>live_bundle_example.json</code>(live 打 API 產生的那份)</b>、
 <code>input_bundle_example.json</code>(本頁這份離線輸入)、
 <code>input_bundle_preview.json</code>(可讀版)、
 <code>input_shapes.json</code>、<code>bot_context.json</code>、
 <code>node_shape_table.json</code>、4 份輸出實例。</p>

<h2 id="s6">6. 仍未完成</h2>
<ul>
<li>{bad} <b>live / executor 還不吃 v2</b>:<code>mvp_live_executor</code> 讀 paper daemon 的
 <code>trades.jsonl</code>(成交記錄,8 欄),那不是訊號,<b>也沒有 stop_loss / take_profit</b> ——
 交易所端仍沒有掛好的保護單。<code>ExitPlan.exchange_managed</code> 有欄位但沒人實作。</li>
<li>{ok} <b><code>input_bundle.py</code> 已有測試</b>(12 個),
 <code>live_bundle.py</code> 8 個、<code>panel.py</code> 12 個、
 節點參數護欄 4 個、輸入 schema 18 個。全套 <b>{esc(TEST_LINE)}</b>。</li>
<li>{warn} <b>v1 未標 deprecated</b>,兩份輸出 schema 並存,需定遷移期。</li>
<li>{ok} <b>輸入 schema 現在真的會驗</b>:<code>jsonschema</code> 已列入
 <code>requirements.txt</code>,4 份 artifact 逐列驗證 + 8 種污染必須被 reject。</li>
<li>{bad} <b><code>basis</code> 仍取不到</b>:走 <code>binance-cli</code> 而它回非 JSON,
 而 <code>public_sources.py</code> 沒有對應的 public spec。其餘 4 個衍生品節點
 已由 <code>public_fallback</code> 救回。</li>
<li>{warn} <b>多標的 panel 已實作並測過,但還沒有實際選幣策略用它跑過</b> ——
 <code>to_panel(bundle, symbols=[...])</code> 的 MultiIndex 目前只有測試在用。</li>
<li>{warn} <b>advisory 註冊表是空的</b>,6 支 sample monitor 匯入但未自動註冊。</li>
<li>{warn} 舊 5 支 v2 策略未走 blocks。</li>
<li>{bad} <b>內網 client 未併入</b>(<code>data_cli/internal*</code>):硬編
 內網主機名(不在此列出),而本 repo 是
 <b>PUBLIC</b>。<b>格式已統一(走 <code>extra_frames</code>)</b>,缺的只是那個 HTTP client,
 需先脫敏成必填環境變數。</li>
</ul>

<p class="muted" style="margin-top:40px;border-top:1px solid var(--line);padding-top:16px">
本頁由 <code>docs/gen_full_flow_html.py</code> 產生(含實際建立 bundle)。
Schema 由 <code>docs/gen_input_schema_v1.py</code> 與
<code>docs/gen_signal_schema_v2.py</code> 從 dataclass 生成。
</p>
</div></body></html>
"""

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write(HTML)
print("wrote %s (%d chars)" % (OUT, len(HTML)))
print("  bundle: %s (%s bytes) — %d frames, %d ctx frames" % (
    BUNDLE_PATH, "{:,}".format(BUNDLE_BYTES), len(bundle["frames"]), len(CTX_FRAMES)))
print("  multi-source signal: intent=%s score=%s reasons=%s" % (
    m_core["intent"], m_core["score"], m_core["reason_codes"]))
print("  TRADE/SELECTION key-set identical: %s (%d keys)" % (
    set(m_core) == set(s_core), len(m_core)))
print("  %s" % TEST_LINE)
