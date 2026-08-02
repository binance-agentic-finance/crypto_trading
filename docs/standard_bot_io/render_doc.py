"""从 README.template.md + samples/ 渲染出 README.md。

改了契约就重跑，文档不会和代码脱节：

    python docs/standard_bot_io/gen_samples.py    # 先刷 samples
    python docs/standard_bot_io/render_doc.py     # 再渲染文档
"""
import json, pathlib, sys
from collections import Counter
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cyqnt_trd.standard_bot.data.catalog import get_node, list_nodes
from cyqnt_trd.standard_bot.universal import CAPABILITIES

S = ROOT / "docs/standard_bot_io/samples"
load = lambda n: json.load(open(S / n))
shapes, ctx = load("input_shapes.json"), load("bot_context.json")
out_long, out_exec = load("output_open_long.json"), load("output_open_long.execution_request.json")
out_close, out_sel, out_adv = load("output_close_short.json"), load("output_selection.json"), load("output_advisory.json")
nodes = sorted(list_nodes(), key=lambda n: (n.emits.value, n.name))
j = lambda o: json.dumps(o, ensure_ascii=False, indent=2)

LABEL = {"bar":"BarFrame@1.0","metric":"MetricFrame@1.0","event":"EventFrame@1.0",
         "rank":"RankFrame@1.0","position":"PositionFrame@1.0","book":"BookFrame@1.0"}

GROUPS = [
    ("量价", ("klines","indicators","ticker")),
    ("衍生品 / 拥挤度", ("funding","open_interest","long_short","top_trader","taker",
                    "basis","liquidations","radar")),
    ("消息 / 事件", ("news","news_search","announcements","hot_event","calendar",
                 "token_unlock","macro_calendar")),
    ("社交", ("social_rank","sentiment","topic_trending")),
    ("截面 / 选币", ("universe","bdp_screen","coin_metrics","sector")),
    ("资金流 / 链上", ("large_flow","whale","chip","top_player")),
    ("宏观 / 情绪", ("fear_greed","ahr999","etf_flow")),
    ("账户", ("positions","balance")),
    ("微观结构", ("orderbook",)),
    ("平台既有信号", ("ai_signal",)),
]


def capability_table():
    lines = ["`UniversalBot` 覆盖下面全部，一支策略 `.activate(...)` 打开需要的那几项。\n"]
    for title, caps in GROUPS:
        lines += ["### %s\n" % title,
                  "| 能力 | 数据节点 | 形状 | 可回放 | 说明 |", "|---|---|---|---|---|"]
        for cap in caps:
            node_name, note = CAPABILITIES[cap]
            n = get_node(node_name)
            lines.append("| `%s` | `data.%s` | %s | `%s` | %s |" % (
                cap, node_name, n.emits.value, n.availability.value, note))
        lines.append("")
    return "\n".join(lines)


def node_table():
    by = {}
    for n in nodes: by.setdefault(n.emits.value, []).append(n)
    lines = []
    for kind in ("bar","metric","event","rank","position","book"):
        items = by.get(kind, [])
        lines += ["#### `%s` / %s -- %d 个\n" % (kind, LABEL[kind], len(items)),
                  "| 节点 | 通道 | 可回放 | 已接线 |", "|---|---|---|:-:|"]
        for n in items:
            lines.append("| `data.%s` | %s | `%s` | %s |" % (
                n.name, n.source_path.value, n.availability.value, "yes" if n.fetcher else "--"))
        lines.append("")
    return "\n".join(lines)

def shape_sections():
    cnt = Counter(n.emits.value for n in nodes)
    order = [("BarFrame@1.0","bar"),("MetricFrame@1.0","metric"),("EventFrame@1.0","event"),
             ("RankFrame@1.0","rank"),("PositionFrame@1.0","position"),("BookFrame@1.0","book"),
             ("PanelFrame@1.0","panel")]
    out = []
    for name, kind in order:
        sc = shapes[name]
        out.append("### `%s` - %d 个节点\n\n**一行 = %s**\n\n%s\n\n```json\n%s\n```\n"
                   % (name, cnt.get(kind,0), sc["row_grain"], sc["description"], j(sc["sample_row"])))
    return "\n".join(out)

tpl = (pathlib.Path(__file__).resolve().parent / "README.template.md").read_text(encoding="utf-8")
doc = (tpl
  .replace("@@CAPABILITY_TABLE@@", capability_table())
  .replace("@@SHAPES@@", shape_sections())
  .replace("@@NODE_TABLE@@", node_table())
  .replace("@@CTX@@", j({k: ctx[k] for k in ("schema","bot_id","decision_time","equity","positions",
                                             "source_status","warnings","inferred_availability")}))
  .replace("@@CTX_TYPED@@", j({k: {kk: vv for kk, vv in v.items()
                                   if kk in ("kind","schema","rows","status","availability")}
                               for k, v in list(ctx["frames"].items())[:2]}))
  .replace("@@EXEC@@", j(out_exec))
  .replace("@@OPEN@@", j(out_long))
  .replace("@@CLOSE@@", j({k: out_close[k] for k in ("intent","target_side","closes_side","order_side",
                                                      "reduce_only","symbol","size","reason_codes","summary")}))
  .replace("@@SEL@@", j({k: out_sel[k] for k in ("market_scope","symbol","intent","universe_size",
                                                  "candidates","summary")}))
  .replace("@@ADV@@", j({k: out_adv[k] for k in ("symbol","intent","direction","advisory_action",
                                                  "score","reason_codes","summary","auto_trade_eligible")})))
assert "@@" not in doc
(ROOT / "docs/standard_bot_io/README.md").write_text(doc, encoding="utf-8")
print("wrote README.md -- %d lines" % doc.count("\n"))
