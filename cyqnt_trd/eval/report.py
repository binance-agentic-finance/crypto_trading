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
             f"  h={card.primary_h}d, cost={card.cost_bps} bp one-way",
             "  scope: single-factor research; strategy combination NOT_EVALUATED"]
    for key, gate in card.gates.items():
        lines.append(f"  {MARK[gate.status]} {key:16s} {gate.status:4s} {gate.title}")
        for c in gate.checks:
            thr = "" if c.threshold is None else f" {c.rule} {_fmt(c.threshold)}"
            lines.append(f"      {MARK[c.status]} {c.name:28s} {_fmt(c.value):>10s}{thr}")
    return "\n".join(lines)


def _verdict_line(card) -> str:
    from .matrix import VERDICTS
    tail = f" — blocking or missing evidence: {', '.join(card.blocking)}" if card.blocking else ""
    return VERDICTS[card.verdict] + tail


def scorecard_markdown(card) -> str:
    out = [f"# 因子评分卡 / factor scorecard — `{card.name}`", "",
           f"**裁决 / verdict: `{card.verdict}`** — {_verdict_line(card)}", "",
           "| 项 | 内容 |", "|---|---|",
           f"| 标的池 / panel | {card.panel_description} |",
           f"| 主期限 / primary horizon | {card.primary_h} 天，非重叠周期 |",
           f"| 成本 / cost | {card.cost_bps} bp 单边 + 实际逐次资金费 |",
           f"| 执行 / execution | {card.config.get('execution', 'open[t+2] → open[t+2+h]')} |",
           f"| 方向 / direction | 在 dev 段冻结，后两段不翻转 |",
           f"| 标的池限制 | {card.config.get('selection_caveat', 'unknown')} |",
           f"| 样本限制 | {card.config.get('holdout_status', 'unknown')} |",
           f"| 资金费估值 | {card.config.get('funding_price_basis', 'unknown')} |",
           "| 组合策略 / strategy combination | NOT_EVALUATED；G5 只检查公开基线重复性 |",
           f"| 已查看候选数 | {card.config.get('trials_seen', 1)}；多候选结果不能按单候选标定通过 |", ""]

    out += ["## 分段核心指标 / split metrics", "",
            "净收益采用完整周期口径；partial net 仅作缺数诊断，不能使成本闸门通过。", "",
            "| h / split | RankIC | HAC t | IC日期数 | 完整/计划周期 | 毛bp | 净bp | partial净bp | 换手 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for row in card.metrics.to_dict("records"):
        out.append(f"| {row['h']} / {row['split']} | {_fmt(row['ic_mean'])} | {_fmt(row['ic_t_hac'])} | "
                   f"{row['n_ic']} | {row['n_complete']}/{row['n_periods']} | {_fmt(row['gross_bp'])} | "
                   f"{_fmt(row['net_bp'])} | {_fmt(row['net_available_bp'])} | {_fmt(row['turnover'])} |")
    out.append("")

    if card.diagnostics:
        out += ["## 全面评估矩阵 / full evaluation matrix", "",
                "| 维度 | 状态 | 证据与边界 |", "|---|---|---|"]
        for row in card.diagnostics.get("matrix", []):
            out.append(f"| {row['dimension']} | {row['status']} | {row['evidence']} |")
        out += ["", "`MEASURED` 表示已计算诊断，不代表该维度证明了盈利。`NOT_EVALUATED` 表示还缺输入或验证。", "",
                "完整四目标、分箱、条件切片、期限/延迟、极端案例和成本压力保存在 JSON；`--report-dir` 生成离线图表报告。", ""]

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
        out += ["## 分箱 / bins（val，冻结方向，入场对齐）", "",
                "| 箱 | 资产·日 | 平均前瞻收益 bp |", "|---|---|---|"]
        for row in card.structure["bins"]:
            out.append(f"| {row['bin']} | {row['n']} | {_fmt(row['mean_bp'])} |")
        out += ["", card.structure.get("bins_note", ""), "",
                f"首尾 spread {_fmt(card.structure.get('spread_bp'))} bp，"
                    f"Spearman(箱号, 收益) {_fmt(card.structure.get('spearman'))}", ""]

    if card.yearly_ic is not None and len(card.yearly_ic):
        out += ["## 分年 IC / yearly IC", "", "| 年 | IC |", "|---|---|"]
        for year, value in card.yearly_ic.items():
            out.append(f"| {year} | {_fmt(float(value))} |")
        out.append("")

    if card.incremental.get("rank_corr"):
        out += ["## 与公开基线的关系 / vs public baselines（dev）", "",
                "| 基线 | 横截面 rank 相关 |", "|---|---|"]
        for name, value in card.incremental["rank_corr"].items():
            out.append(f"| {name} | {_fmt(value)} |")
        out += ["", f"投影掉全部五个基线后，dev IC 保留 "
                    f"{_fmt(card.incremental.get('retention'))}；"
                    f"残差在 val 的净收益 {_fmt(card.incremental.get('residual_net_val'))} bp", ""]

    out += ["## 怎么读这张卡 / how to read this", "",
            "- 必需检查失败或缺失都阻止完整通过；后面的指标不能抵消它。",
            "- `G3_cost` 是硬否决：信息再强，收不上来就不是可交易因子。",
            "- *null p95* 是指定 AR(1) 随机信号的参考分布，不是所有候选的普适显著性保证。",
            "- 标了 *convention* 的门槛可以争论；标了 *required* / *hard veto* 的不建议放松。",
            "- 前缀一致性是抽查，不能证明上游数据、外部状态或闭包中的数据都按时可得。",
            "- 这张卡评的是**一个**因子。扫参数属于搜索，需完整试验账本及新样本验证。",
            "- PASS 是当前数据和假设下的研究筛选通过；封存样本、容量、组合收益仍需另行验证。", ""]
    contract = card.config.get("calibration_contract", {})
    out += ["## 复现标识 / provenance", "",
            f"- 面板 SHA256：`{contract.get('panel_sha256', 'unknown')}`",
            f"- 引擎：`{contract.get('engine_protocol', 'unknown')}` / `{contract.get('engine_sha256', 'unknown')}`",
            "- JSON 同时保存全部分段指标、切分边界、执行配置及标定契约；非有限量编码为 null。"]
    return "\n".join(out)
