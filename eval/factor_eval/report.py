"""Rendering a scorecard so the *first failure* is the first thing you read."""
from __future__ import annotations

import numpy as np

MARK = {"PASS": "✓", "FAIL": "✗", "WARN": "!", "N/A": "–"}


def _fmt(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def scorecard_text(card) -> str:
    lines = [f"{card.name}: {card.verdict}",
             f"  {card.gates and ''}{_verdict_line(card)}",
             f"  panel: {card.panel_description}",
             f"  h={card.primary_h}d, cost={card.cost_bps} bp one-way"]
    for key, gate in card.gates.items():
        lines.append(f"  {MARK[gate.status]} {key:16s} {gate.status:4s} {gate.title}")
        for c in gate.checks:
            thr = "" if c.threshold is None else f" {c.rule} {_fmt(c.threshold)}"
            lines.append(f"      {MARK[c.status]} {c.name:28s} {_fmt(c.value):>10s}{thr}")
    return "\n".join(lines)


def _verdict_line(card) -> str:
    from .matrix import VERDICTS
    tail = f" — first failure: {', '.join(card.blocking)}" if card.blocking else ""
    return VERDICTS[card.verdict] + tail


def scorecard_markdown(card) -> str:
    out = [f"# 因子评分卡 / factor scorecard — `{card.name}`", "",
           f"**裁决 / verdict: `{card.verdict}`** — {_verdict_line(card)}", "",
           "| 项 | 内容 |", "|---|---|",
           f"| 标的池 / panel | {card.panel_description} |",
           f"| 主期限 / primary horizon | {card.primary_h} 天，非重叠周期 |",
           f"| 成本 / cost | {card.cost_bps} bp 单边 + 实际逐次资金费 |",
           f"| 执行 / execution | {card.config.get('execution', 'open[t+2] → open[t+2+h]')} |",
           f"| 方向 / direction | 在 dev 段冻结，后两段不翻转 |", ""]

    out += ["## 闸门 / gates", "",
            "| 闸门 | 状态 | 问题 |", "|---|---|---|"]
    for key, gate in card.gates.items():
        out.append(f"| `{key}` | {MARK[gate.status]} {gate.status} | {gate.question} |")
    out.append("")

    for key, gate in card.gates.items():
        out += [f"### {MARK[gate.status]} {key} — {gate.title}", "",
                "| 检查 | 实测 | 门槛 | 判定 | 门槛来自哪里 |", "|---|---|---|---|---|"]
        for c in gate.checks:
            thr = "—" if c.threshold is None else f"`{c.rule} {_fmt(c.threshold)}`"
            note = f"<br>_{c.note}_" if c.note else ""
            out.append(f"| {c.name} | {_fmt(c.value)} | {thr} | {MARK[c.status]} {c.status} "
                       f"| {c.basis}{note} |")
        out.append("")

    if card.structure.get("bins"):
        out += ["## 分箱 / bins（冻结方向，入场对齐）", "",
                "| 箱 | 资产·日 | 平均前瞻收益 bp |", "|---|---|---|"]
        for row in card.structure["bins"]:
            out.append(f"| {row['bin']} | {row['n']} | {_fmt(row['mean_bp'])} |")
        out += ["", f"首尾 spread {_fmt(card.structure.get('spread_bp'))} bp，"
                    f"Spearman(箱号, 收益) {_fmt(card.structure.get('spearman'))}", ""]

    if card.yearly_ic is not None and len(card.yearly_ic):
        out += ["## 分年 IC / yearly IC", "", "| 年 | IC |", "|---|---|"]
        for year, value in card.yearly_ic.items():
            out.append(f"| {year} | {_fmt(float(value))} |")
        out.append("")

    if card.incremental.get("rank_corr"):
        out += ["## 与公开基线的关系 / vs public baselines", "",
                "| 基线 | 横截面 rank 相关 |", "|---|---|"]
        for name, value in card.incremental["rank_corr"].items():
            out.append(f"| {name} | {_fmt(value)} |")
        out += ["", f"投影掉全部五个基线后，dev IC 保留 "
                    f"{_fmt(card.incremental.get('retention'))}；"
                    f"残差在 val 的净收益 {_fmt(card.incremental.get('residual_net_val'))} bp", ""]

    out += ["## 怎么读这张卡 / how to read this", "",
            "- **第一个 ✗ 就是结论**：后面的闸门只是补充说明，不能用来抵消它。",
            "- `G3_cost` 是硬否决：信息再强，收不上来就不是可交易因子。",
            "- 标了 *null p95* 的门槛，是随机信号走同一条管线得到的分布，不是拍的。",
            "- 标了 *convention* 的门槛可以争论；标了 *required* / *hard veto* 的不建议放松。",
            "- 这张卡评的是**一个**因子、**一个**冻结方向。扫参数属于搜索，届时这些门槛不再成立。"]
    return "\n".join(out)
