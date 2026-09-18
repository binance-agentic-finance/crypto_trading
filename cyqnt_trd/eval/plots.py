"""Offline diagnostic figures and a portable, escaped HTML research report.

Plot only the statistics supplied by the diagnostic layer. Missing measurements
remain visible as N/A panels; rendering never substitutes synthetic observations
or recalculates a more favourable sample. Every image is a local PNG beside the
HTML file, so the report remains usable without JavaScript or a network.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from html import escape
import json
from pathlib import Path

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


GROUPS = (
    ("distribution", "Signal distribution and coverage", "分布与覆盖"),
    ("bins", "Four-target bin diagnostics", "四目标分箱"),
    ("past_return_slices", "Conditioning on past returns", "过去收益条件切片"),
    ("ftr_delay", "Horizon and execution-delay sensitivity", "期限与延迟"),
    ("stability", "Time and market-state stability", "时间与状态稳定性"),
    ("ic_vs_n", "Information versus volatility / spread", "IC 与交易难度 N"),
    ("extreme_cases", "Extreme correct and incorrect cases", "极端对错案例"),
    ("cost_stress", "Trading-cost stress", "成本压力"),
)
COLORS = ("#176b87", "#d17b38", "#468b67", "#8962a4")
TARGET_LABELS = {
    "raw_rtf": "Raw return",
    "normed_rtf": "Volatility-normalized return",
    "res_rtf": "Benchmark-residual return",
    "normed_res_rtf": "Volatility-normalized residual",
}


def _unavailable(ax, reason):
    ax.set_axis_off()
    ax.text(0.5, 0.60, "N/A", ha="center", va="center", fontsize=24,
            fontweight="bold", color="#526174", transform=ax.transAxes)
    ax.text(0.5, 0.40, str(reason), ha="center", va="center", fontsize=10,
            color="#526174", wrap=True, transform=ax.transAxes)


def _reason(section):
    if not isinstance(section, Mapping):
        return "Diagnostic data were not supplied."
    return str(section.get("reason") or section.get("note") or "No usable observations were supplied.")


def _style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dce3eb", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    for name in ("bottom", "left"):
        ax.spines[name].set_color("#b4c0cd")
    ax.tick_params(colors="#526174", labelsize=8)


def _number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _numbers(rows, key):
    return np.array([_number(row.get(key)) for row in rows], dtype=float)


def _format(value):
    number = _number(value)
    return f"{number:.4g}" if np.isfinite(number) else "N/A"


def _rows(section):
    return section.get("rows", []) if isinstance(section, Mapping) else []


def _intervals(ax, x, low, high, color):
    valid = np.isfinite(x) & np.isfinite(low) & np.isfinite(high)
    if (low[valid] > high[valid]).any():
        raise ValueError("confidence interval lower bound exceeds upper bound")
    # Draw intervals independently of the estimate: a percentile interval can
    # legitimately exclude the point estimate, so negative yerr is unsuitable.
    ax.vlines(x[valid], low[valid], high[valid], color=color, alpha=0.65, linewidth=1.5)
    ax.scatter(x[valid], low[valid], marker="_", color=color, s=22)
    ax.scatter(x[valid], high[valid], marker="_", color=color, s=22)


def _relation_curve(ax, rows, xkey, *, ykey="rank_ic", color=None, label=None):
    rows = sorted(rows, key=lambda row: _number(row.get(xkey)))
    x, y = _numbers(rows, xkey), _numbers(rows, ykey)
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return False
    color = color or COLORS[0]
    # Keep unavailable y values as gaps instead of connecting across them.
    ax.plot(x, y, marker="o", markersize=3, linewidth=1.4, color=color, label=label)
    _intervals(ax, x, _numbers(rows, "ci_low"), _numbers(rows, "ci_high"), color)
    _style(ax)
    return True


def _distribution(fig, section):
    axes = fig.subplots(1, 2)
    histogram, cdf = section.get("histogram", {}), section.get("cdf", {})
    edges = np.array([_number(v) for v in histogram.get("edges", [])])
    counts = np.array([_number(v) for v in histogram.get("counts", [])])
    if len(counts) and len(edges) == len(counts) + 1 and np.isfinite(counts).any():
        axes[0].bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=COLORS[0], alpha=0.85)
        axes[0].set(xlabel="Signal value", ylabel="Eligible observations", title="Histogram")
        _style(axes[0])
    else:
        _unavailable(axes[0], _reason(section))
    x, y = np.array([_number(v) for v in cdf.get("x", [])]), np.array([_number(v) for v in cdf.get("y", [])])
    if len(x) == len(y) and len(x) and (np.isfinite(x) & np.isfinite(y)).any():
        axes[1].plot(x, y, color=COLORS[1], linewidth=1.6)
        axes[1].set(xlabel="Signal value", ylabel="Cumulative fraction", ylim=(0, 1), title="Empirical CDF")
        _style(axes[1])
    else:
        _unavailable(axes[1], _reason(section))
    stats = "  |  ".join(f"{key}: {_format(section.get(key))}"
                          for key in ("total", "eligible", "finite", "zero", "nan", "posinf", "neginf"))
    fig.supxlabel(stats, fontsize=8, color="#526174")


def _bins(fig, section):
    axes = fig.subplots(2, 2).ravel()
    for ax, (target, label), color in zip(axes, TARGET_LABELS.items(), COLORS):
        data = section.get(target, {})
        rows = _rows(data)
        x, y = _numbers(rows, "bin"), _numbers(rows, "mean")
        ax.set_title(label)
        if not (np.isfinite(x) & np.isfinite(y)).any():
            _unavailable(ax, _reason(data))
            continue
        ax.bar(x, y, color=color, alpha=0.75, width=0.7)
        _intervals(ax, x, _numbers(rows, "ci_low"), _numbers(rows, "ci_high"), color)
        ax.axhline(0, color="#526174", linewidth=0.7)
        ax.set(xlabel="Frozen signal bin", ylabel=str(data.get("unit") or "Mean target"))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        _style(ax)


def _past_slices(fig, section):
    axes = fig.subplots(1, 2)
    rows = _rows(section)
    labels = [str(row.get("label", index + 1)) for index, row in enumerate(rows)]
    x = np.arange(len(rows), dtype=float)
    for ax, field, label, color in zip(axes, ("corr", "slope"), ("Correlation", "Slope"), COLORS):
        y = _numbers(rows, field)
        if not np.isfinite(y).any():
            _unavailable(ax, _reason(section))
            continue
        ax.plot(x, y, marker="o", markersize=3, color=color)
        if field == "corr":
            _intervals(ax, x, _numbers(rows, "ci_low"), _numbers(rows, "ci_high"), color)
        ax.axhline(0, color="#526174", linewidth=0.7)
        stride = max(1, int(np.ceil(len(rows) / 8)))
        ax.set_xticks(x[::stride], labels[::stride], rotation=25, ha="right")
        ax.set(xlabel="Previously realized return slice", ylabel=label)
        _style(ax)


def _horizon_delay(fig, section):
    axes = fig.subplots(1, 2)
    for ax, key, xkey, xlabel in zip(axes, ("ftr", "delay"), ("h", "lag"),
                                    ("Holding horizon (days)", "Entry lag (days)")):
        data = section.get(key, {})
        rows = _rows(data)
        targets = list(dict.fromkeys(row.get("target", "raw_rtf") for row in rows))
        plotted = False
        for index, target in enumerate(targets):
            selected = [row for row in rows if row.get("target", "raw_rtf") == target]
            plotted |= _relation_curve(ax, selected, xkey, color=COLORS[index % len(COLORS)],
                                        label=TARGET_LABELS.get(target, str(target)))
        if not plotted:
            _unavailable(ax, _reason(data))
            continue
        ax.axhline(0, color="#526174", linewidth=0.7)
        ax.set(xlabel=xlabel, ylabel="Rank IC")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.set_title("Cumulative target horizon" if key == "ftr"
                     else f"Fixed holding horizon: {_format(data.get('holding_h'))} days")
        ax.legend(fontsize=7, loc="best", frameon=False)


def _stability(fig, section):
    ax = fig.subplots()
    rows = _rows(section)
    y = _numbers(rows, "rank_ic")
    if not np.isfinite(y).any():
        _unavailable(ax, _reason(section))
        return
    short = {"leave_one_asset_out": "omit", "ex_ante_volatility": "vol",
             "year": "year", "quarter": "quarter", "weekday": "day"}
    labels = [f"{short.get(row.get('dimension'), row.get('dimension', ''))}: {row.get('label', index)}"
              for index, row in enumerate(rows)]
    x = np.arange(len(rows), dtype=float)
    dimensions = list(dict.fromkeys(row.get("dimension", "") for row in rows))
    colors = [COLORS[dimensions.index(row.get("dimension", "")) % len(COLORS)] for row in rows]
    ax.bar(x, y, color=colors, alpha=0.8)
    _intervals(ax, x, _numbers(rows, "ci_low"), _numbers(rows, "ci_high"), "#34485d")
    ax.axhline(0, color="#526174", linewidth=0.7)
    ax.set_xticks(x, labels, rotation=55, ha="right")
    ax.set_ylabel("Rank IC")
    _style(ax)


def _ic_vs_n(fig, section):
    ax = fig.subplots()
    rows = _rows(section)
    if not (np.isfinite(_numbers(rows, "n_median")) & np.isfinite(_numbers(rows, "rank_ic"))).any():
        _unavailable(ax, _reason(section))
        return
    groups = list(dict.fromkeys(row.get("group", "") for row in rows))
    for index, group in enumerate(groups):
        selected = [row for row in rows if row.get("group", "") == group]
        color = COLORS[index % len(COLORS)]
        _relation_curve(ax, selected, "n_median", color=color, label=str(group))
        for row in selected:
            x, y = _number(row.get("n_median")), _number(row.get("rank_ic"))
            lo, hi = _number(row.get("n_q25")), _number(row.get("n_q75"))
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if np.isfinite(lo) and np.isfinite(hi):
                ax.hlines(y, lo, hi, color=color, alpha=0.4)
            ax.annotate(f"h={row.get('h', '?')}", (x, y), xytext=(3, 5),
                        textcoords="offset points", fontsize=7, color=color)
    ax.axhline(0, color="#526174", linewidth=0.7)
    ax.set(xlabel="N: prior volatility / full bid-ask spread", ylabel="Rank IC")
    ax.legend(frameon=False, fontsize=8)


def _timestamp(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _extremes(fig, section):
    cases = [*[("Correct", case) for case in section.get("correct", [])[:6]],
             *[("Wrong", case) for case in section.get("wrong", [])[:6]]]
    if not cases:
        _unavailable(fig.subplots(), _reason(section))
        return
    nrows = int(np.ceil(len(cases) / 2))
    fig.set_size_inches(12, max(5.2, nrows * 2.8))
    axes = np.array(fig.subplots(nrows, 2, squeeze=False)).ravel()
    for ax, (kind, case) in zip(axes, cases):
        times = [_timestamp(value) for value in case.get("times", [])]
        prices = np.array([_number(v) for v in case.get("prices", [])])
        ax.set_title(f"{kind}: {case.get('symbol', '?')} | target {_format(case.get('target'))}", fontsize=10)
        if len(times) != len(prices) or not len(times) or not np.isfinite(prices).any():
            _unavailable(ax, "No price trajectory was supplied for this case.")
            continue
        ax.plot(times, prices, color=COLORS[0], linewidth=1.4, label="Price")
        for field, label, color in (("signal_time", "Signal", "#526174"),
                                    ("entry_time", "Entry", COLORS[2]), ("exit_time", "Exit", COLORS[1])):
            if case.get(field):
                ax.axvline(_timestamp(case[field]), color=color, linestyle="--", linewidth=0.9, label=label)
        signal = np.array([_number(v) for v in case.get("signal", [])])
        if len(signal) == len(times) and np.isfinite(signal).any():
            twin = ax.twinx()
            twin.plot(times, signal, color=COLORS[3], linewidth=0.9, alpha=0.7)
            twin.set_ylabel("Signal", fontsize=7, color=COLORS[3])
            twin.tick_params(axis="y", labelsize=7, colors=COLORS[3])
            twin.spines[["top", "left"]].set_visible(False)
        locator = AutoDateLocator(minticks=2, maxticks=4)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        ax.set_ylabel("Price", fontsize=8)
        ax.legend(fontsize=6, frameon=False, loc="best", ncol=2)
        _style(ax)
    for ax in axes[len(cases):]:
        ax.set_axis_off()
    fig.supxlabel("Post-hoc diagnostic cases; they do not estimate a win rate.", fontsize=9)


def _cost(fig, section):
    ax = fig.subplots()
    rows = _rows(section)
    splits = list(dict.fromkeys(row.get("split", "") for row in rows))
    plotted = False
    for index, split in enumerate(splits):
        selected = sorted((row for row in rows if row.get("split", "") == split),
                          key=lambda row: _number(row.get("cost_bps")))
        x, net = _numbers(selected, "cost_bps"), _numbers(selected, "net_bp")
        available = _numbers(selected, "net_available_bp")
        color = COLORS[index % len(COLORS)]
        if (np.isfinite(x) & np.isfinite(net)).any():
            ax.plot(x, net, marker="o", color=color, label=f"{split}: complete net")
            plotted = True
        missing = ~np.isfinite(net) & np.isfinite(available)
        if missing.any():
            ax.plot(x, np.where(missing, available, np.nan), marker="x", linestyle="--",
                    color=color, alpha=0.65, label=f"{split}: available only (incomplete)")
            plotted = True
    if not plotted:
        _unavailable(ax, _reason(section))
        return
    ax.axhline(0, color="#526174", linewidth=0.7)
    ax.set(xlabel="One-way trading cost (bp)", ylabel="Net return per cycle (bp)")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)


def _draw_group(fig, key, section):
    if not isinstance(section, Mapping):
        raise TypeError(f"diagnostic group '{key}' must be a mapping")
    functions = {"distribution": _distribution, "bins": _bins, "past_return_slices": _past_slices,
                 "ftr_delay": _horizon_delay, "stability": _stability, "ic_vs_n": _ic_vs_n,
                 "extreme_cases": _extremes, "cost_stress": _cost}
    functions[key](fig, section)


def _section(diagnostics, key):
    if key == "ftr_delay":
        return {"ftr": diagnostics.get("ftr", {}), "delay": diagnostics.get("delay", {})}
    return diagnostics.get(key, {})


def _table(rows, fields):
    if not rows:
        return ""
    header = "".join(f"<th>{escape(str(field))}</th>" for field in fields)
    body = []
    for row in rows:
        cells = []
        for field in fields:
            value = row.get(field)
            if value is None:
                value = "N/A"
            elif isinstance(value, (float, np.floating)):
                value = _format(value)
            cells.append(f"<td>{escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _relation_rows(rows):
    """Name each interval by its statistic instead of reusing ambiguous CI columns."""
    result = []
    for row in rows:
        named = dict(row)
        for metric in ("rank_ic", "corr_pooled", "slope"):
            interval = row.get("ci", {}).get(metric, [None, None])
            named[f"{metric}_ci_low"], named[f"{metric}_ci_high"] = interval
        result.append(named)
    return result


def _html_report(card, diagnostics, images):
    title = escape(str(getattr(card, "name", "Factor diagnostics")))
    verdict = escape(str(getattr(card, "verdict", "NOT_EVALUATED")))
    sections = []
    for key, _, label in GROUPS:
        section = _section(diagnostics, key)
        reason = _reason(section) if not section else str(section.get("reason") or section.get("note") or "")
        note = f"<p class=note>{escape(reason)}</p>" if reason else ""
        filename = escape(Path(images[key]).name, quote=True)
        evidence_rows = _rows(section)
        fields = ("label", "dimension", "group", "target", "h", "lag", "rank_ic", "rank_ic_ci_low", "rank_ic_ci_high",
                  "corr", "slope", "n_dates", "n_dates_pooled", "n_pairs", "block_length", "ci_reasons")
        if key == "ftr_delay":
            evidence_rows = [{"curve": curve, **row} for curve in ("ftr", "delay")
                             for row in _rows(section.get(curve, {}))]
            fields = ("curve", *fields)
        elif key == "bins":
            evidence_rows = [{"target": target, **row} for target in TARGET_LABELS
                             for row in _rows(section.get(target, {}))]
            fields = ("target", "bin", "mean", "ci_low", "ci_high", "n_dates", "n_pairs", "block_length", "reason")
            note += "<p class=note>dev 分位点冻结；每箱内日期等权，各箱日期可能不同。这是描述性关系，不能读作可交易分组收益。</p>"
        elif key == "cost_stress":
            fields = ("split", "multiplier", "cost_bps", "net_bp", "net_available_bp", "n_complete", "n_periods")
        elif key == "past_return_slices":
            fields = ("label", "corr", "corr_pooled_ci_low", "corr_pooled_ci_high", "slope", "slope_ci_low", "slope_ci_high",
                      "n_dates_pooled", "n_pairs", "block_length", "ci_reasons")
        elif key == "ic_vs_n":
            fields = ("group", "h", "lag", "n_q25", "n_median", "n_q75", "rank_ic", "rank_ic_ci_low", "rank_ic_ci_high", "n_dates", "n_ic_pairs")
        elif key == "extreme_cases":
            evidence_rows = [{"outcome": outcome, **row} for outcome in ("correct", "wrong")
                             for row in section.get(outcome, [])]
            fields = ("outcome", "symbol", "signal_time", "entry_time", "exit_time", "score", "target")
        elif key == "distribution":
            evidence_rows = [section]
            fields = ("total", "eligible", "finite", "zero", "nan", "posinf", "neginf")
        evidence = _table(_relation_rows(evidence_rows), fields)
        details = f"<details><summary>指标、样本和区间明细</summary>{evidence}</details>" if evidence else ""
        sections.append(f'<section id="{key}"><h2>{escape(label)}</h2>{note}'
                        f'<img src="{filename}" alt="{escape(label, quote=True)}">{details}</section>')
    matrix = _table(diagnostics.get("matrix", []), ("dimension", "status", "evidence"))
    comparison = diagnostics.get("target_comparison", {})
    target_table = _table(_relation_rows(comparison.get("rows", [])),
                          ("target", "split", "sample", "rank_ic", "rank_ic_ci_low", "rank_ic_ci_high", "pearson_ic", "corr", "slope", "n_dates", "n_dates_pooled", "n_pairs", "block_length", "ci_reasons"))
    coverage = _table([{"target": target, **values, **comparison.get("availability", {}).get(target, {})}
                       for target, values in comparison.get("coverage", {}).items()],
                      ("target", "status", "reason", "eligible_cells", "valid_cells", "valid_dates", "fraction"))
    target_metadata = escape(json.dumps(comparison.get("metadata", {}), ensure_ascii=False, indent=2))
    gate_rows = []
    for gate_id, gate in getattr(card, "gates", {}).items():
        gate_rows.extend({"gate": gate_id, **check.to_dict()} for check in gate.checks)
    gate_table = _table(gate_rows, ("gate", "name", "status", "value", "rule", "threshold", "basis", "required", "note"))
    artifacts = ('<p><a href="scorecard.md">完整研究卡 Markdown</a> · <a href="scorecard.json">完整机器数据 JSON</a></p>'
                 if hasattr(card, "to_dict") and hasattr(card, "to_markdown") else "")
    config = diagnostics.get("config", {})
    context = escape(str(config))
    return f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — factor diagnostics</title>
<style>
body{{margin:0;background:#f2f5f8;color:#17283a;font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1160px;margin:auto;padding:32px 20px}}
header,section{{background:white;border:1px solid #dce3eb;border-radius:10px;padding:24px;margin-bottom:20px}}
h1{{font-size:28px;line-height:1.25;margin:0 0 12px;overflow-wrap:anywhere}}
h2{{font-size:20px;margin:0 0 12px}} p{{margin:8px 0}} .note{{color:#526174;white-space:pre-wrap}}
.verdict{{display:inline-block;background:#e8eff5;border-radius:5px;padding:3px 10px;font-weight:650}}
img{{width:100%;height:auto;display:block}} footer{{color:#526174;font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;border-bottom:1px solid #dce3eb;padding:8px}}
th{{background:#eef3f7}}.table-wrap{{overflow:auto}}details{{margin:12px 0}}summary{{cursor:pointer}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
@media print{{body{{background:white}}main{{padding:0}}section{{break-inside:avoid}}}}
</style>
<main><header><h1>{title}</h1><p class="verdict">{verdict}</p>
<p>单因子研究诊断。图表展示当前输入与约定下的证据；N/A 表示缺证，不能按通过读取。</p>
<p class="note">组合收益、封存样本和可交易容量需要另行验证。全部图片保存在报告旁，报告无需联网。</p>
{artifacts}</header><section><h2>Factor evaluation matrix</h2>{matrix}<p class=note>{context}</p>
<details><summary>六道闸门：检查、门槛与证据</summary>{gate_table}</details></section>
<section><h2>四目标与共同样本</h2>{coverage}{target_table}
<details><summary>目标定义与时间口径</summary><pre>{target_metadata}</pre></details></section>{''.join(sections)}
<footer>极端案例用于事后诊断；置信区间须结合有效样本和重采样约定解读。</footer></main></html>"""


def render_diagnostics(card, output_dir) -> dict:
    """Write eight PNG panels plus ``index.html`` and return their file paths.

``card.diagnostics`` supplies the precomputed measurements. Missing groups still
produce explicit N/A panels, so an absent spread or insufficient history cannot
quietly remove a requested diagnostic from the report.
"""
    diagnostics = getattr(card, "diagnostics", None) or {}
    if not isinstance(diagnostics, Mapping):
        raise TypeError("card.diagnostics must be a mapping")
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    images = {}
    settings = {"font.family": "DejaVu Sans", "font.size": 10, "text.parse_math": False,
                "axes.titlesize": 12, "axes.labelsize": 10, "figure.facecolor": "white"}
    with rc_context(settings):
        for key, title, _ in GROUPS:
            figure = Figure(figsize=(10, 5.2), layout="constrained")
            FigureCanvasAgg(figure)
            figure.suptitle(title, fontsize=15, fontweight="bold", color="#17283a")
            section = _section(diagnostics, key)
            _draw_group(figure, key, section)
            config = diagnostics.get("config", {})
            timing = diagnostics.get("target_comparison", {}).get("metadata", {})
            scope = {"distribution": "all eligible observations", "cost_stress": "dev / val / oot; frozen positions",
                     "extreme_cases": "validation; post-hoc cases, not a win-rate estimate"}.get(key, "validation; transforms frozen on dev")
            context = (f"Daily UTC | {scope}"
                       f" | primary h={timing.get('horizon_days', '?')}, lag={timing.get('entry_lag', '?')}")
            if key not in ("distribution", "cost_stress", "extreme_cases"):
                context += f" | pointwise 95% block intervals, B={config.get('n_bootstrap', '?')}"
            if key == "distribution":
                context = " | ".join(f"{k}={section.get(k, 'N/A')}" for k in ("finite", "zero", "nan", "posinf", "neginf")) + "\n" + context
            figure.supxlabel(context, fontsize=7, color="#526174")
            path = directory / f"{key}.png"
            figure.savefig(path, dpi=150, facecolor="white")
            figure.clear()
            images[key] = str(path)
    html = directory / "index.html"
    html.write_text(_html_report(card, diagnostics, images), encoding="utf-8")
    if hasattr(card, "to_dict"):
        (directory / "scorecard.json").write_text(json.dumps(card.to_dict(), ensure_ascii=False,
                                                           indent=2, allow_nan=False), encoding="utf-8")
    if hasattr(card, "to_markdown"):
        (directory / "scorecard.md").write_text(card.to_markdown(), encoding="utf-8")
    return {"html": str(html), "images": images, "output_dir": str(directory)}
