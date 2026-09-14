"""The evaluation matrix: six gates, in the order that fails cheapest first.

The matrix answers one question — *is this factor good?* — by refusing to answer
it with a single number. Each gate answers a narrower question, and a factor is
only as good as its first failure:

| gate | question | a failure means |
|---|---|---|
| G0 data | did we measure anything at all? | the score is an artifact, not a result |
| G1 information | does the ranking beat a random signal? | no information |
| G2 structure | is the relationship shaped like a usable one? | information, but not monotone/stable across horizons |
| G3 cost | does it survive fees and funding? | **hard veto** — information that cannot be collected |
| G4 robustness | does it hold across splits and years? | one regime, not an effect |
| G5 incremental | is it more than a public baseline? | a re-parameterisation of size / low-vol / reversal |

**Thresholds are not opinions where it matters.** Every information and money
threshold is the 95th percentile of a *random* signal put through this exact
pipeline on this exact universe (`calibration/`). "Better than noise" is measured,
not asserted. Thresholds that are conventions are labelled as conventions, so a
reviewer can argue with the right ones.

The two verdicts people find surprising, and why they exist:

* ``HOLD_INFO`` — real ranking information, no money after costs. This is the
  single most common honest outcome, and the reference study's best factor lands
  here. It is not a pass.
* ``HOLD_WEAK`` — money, but no measurable information. Usually a handful of
  lucky cycles; treat as unexplained until it has an information story.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CALIBRATION = HERE / "calibration" / "null_top10_h3.json"

PASS, FAIL, WARN, NA = "PASS", "FAIL", "WARN", "N/A"


@dataclass
class Check:
    name: str
    value: float
    threshold: float | None
    rule: str
    status: str
    basis: str
    note: str = ""

    def to_dict(self):
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                for k, v in self.__dict__.items()}


@dataclass
class Gate:
    key: str
    title: str
    question: str
    checks: list[Check] = field(default_factory=list)

    @property
    def status(self) -> str:
        states = [c.status for c in self.checks]
        if not states or all(s == NA for s in states):
            return NA
        if FAIL in states:
            return FAIL
        if WARN in states:
            return WARN
        return PASS


def _val(metrics, split, column, h):
    row = metrics[(metrics.split == split) & (metrics.h == h)]
    if row.empty or column not in row:
        return np.nan
    return float(row.iloc[0][column])


def _net(metrics, split, h):
    """Strict net when every cycle is accountable, otherwise the available-cycle
    diagnostic with a flag. Never silently mixes the two."""
    strict = _val(metrics, split, "net_bp", h)
    if np.isfinite(strict):
        return strict, False
    return _val(metrics, split, "net_available_bp", h), True


def _check(name, value, threshold, rule, basis, note="", warn_only=False):
    if not np.isfinite(value) or (threshold is not None and not np.isfinite(threshold)):
        return Check(name, value, threshold, rule, NA, basis, note or "not computable")
    ok = {">": value > threshold, ">=": value >= threshold,
          "<": value < threshold, "<=": value <= threshold}[rule]
    status = PASS if ok else (WARN if warn_only else FAIL)
    return Check(name, float(value), float(threshold), rule, status, basis, note)


def load_calibration(path: Path | None = None) -> dict:
    path = Path(path) if path else CALIBRATION
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- G0
def gate_data(metrics, h, min_coverage=0.5, min_ic_days=100, min_activation=0.5):
    g = Gate("G0_data", "数据 / data health", "did we measure anything at all?")
    g.checks += [
        _check("coverage_dev", _val(metrics, "dev", "coverage", h), min_coverage, ">=",
               "convention: below half the eligible cells the cross-section is not the universe"),
        _check("ic_days_dev", _val(metrics, "dev", "n_ic", h), min_ic_days, ">=",
               "convention: fewer than 100 usable dates makes every later number noise"),
    ]
    n_periods = _val(metrics, "dev", "n_periods", h)
    n_active = _val(metrics, "dev", "n_active", h)
    activation = n_active / n_periods if n_periods else np.nan
    g.checks.append(_check("activation_dev", activation, min_activation, ">=",
                           "convention: a factor flat most cycles cannot be held to its own thesis",
                           note=f"{n_active:.0f}/{n_periods:.0f} cycles held a position"))
    frozen = metrics[(metrics.split == "dev") & (metrics.h == h)]
    ok = bool(frozen.iloc[0]["direction_frozen"]) if not frozen.empty else False
    g.checks.append(Check("direction_frozen", float(ok), 1.0, ">=",
                          PASS if ok else FAIL,
                          "required: the sign is frozen on dev and never re-chosen later",
                          "" if ok else "no dev direction could be frozen"))
    return g


# --------------------------------------------------------------------------- G1
def gate_information(metrics, h, cal):
    g = Gate("G1_information", "信息 / information",
             "does the ranking beat a random signal on this universe?")
    for split in ("dev", "val", "oot"):
        g.checks.append(_check(
            f"ic_{split}", _val(metrics, split, "ic_mean", h), cal["ic_mean"][split]["p95"], ">",
            f"null p95: {cal['trials']} AR(1) random signals through this same pipeline",
            warn_only=(split == "oot")))
    g.checks.append(_check("ic_t_hac_dev", _val(metrics, "dev", "ic_t_hac", h),
                           cal["ic_t_hac"]["dev"]["p95"], ">", "null p95 (HAC t of the same random signals)"))
    g.checks.append(_check("ic_win_dev", _val(metrics, "dev", "ic_win", h), 0.5, ">",
                           "convention: a signed IC should be positive more often than not",
                           warn_only=True))
    return g


# --------------------------------------------------------------------------- G2
def gate_structure(structure, metrics, h, horizons):
    g = Gate("G2_structure", "结构 / structure",
             "is the relationship monotone, and does it survive a change of horizon?")
    per_bin = structure.get("names_per_bin", np.nan)
    if np.isfinite(per_bin) and per_bin < 2.0:
        g.checks.append(Check("bin_monotonicity", structure.get("spearman", np.nan), 0.6, ">=",
                              NA, "not read: fewer than 2 names per bin per date",
                              structure.get("bins_note", "")))
    else:
        g.checks.append(_check("bin_monotonicity", structure.get("spearman", np.nan), 0.6, ">=",
                               "convention: Spearman(bin index, per-date-then-averaged forward "
                               "return); bin count follows the width of the cross-section",
                               note=structure.get("bins_note", ""), warn_only=True))
    g.checks.append(_check(
        "topbottom_spread_bp", structure.get("spread_bp", np.nan), 0.0, ">",
        "required: the top bin must out-return the bottom bin in the frozen direction",
        note=("a positive IC with a negative spread is not a bug: IC ranks, the spread "
              "averages, so a factor can be right more often while the money sits in the "
              "fat tail it is shorting")))
    signs = [np.sign(_val(metrics, "dev", "ic_mean", hh)) for hh in horizons]
    signs = [s for s in signs if np.isfinite(s) and s != 0]
    agree = (len(set(signs)) == 1) if signs else False
    g.checks.append(Check("horizon_sign_agreement", float(len(signs) and len(set(signs)) == 1),
                          1.0, ">=", PASS if agree else WARN,
                          f"convention: dev IC keeps one sign across h={list(horizons)}",
                          "" if agree else "the effect flips sign with holding period"))
    return g


# --------------------------------------------------------------------------- G3
def gate_cost(metrics, h, cost_bps, cal, safety=2.0):
    g = Gate("G3_cost", "成本 / cost (hard veto)",
             "does anything survive fees and actual funding?")
    for split in ("val", "oot"):
        net, partial = _net(metrics, split, h)
        g.checks.append(_check(
            f"net_bp_{split}", net, 0.0, ">",
            "hard veto: net of trading cost and per-settlement funding, after the frozen direction",
            note="available-cycle diagnostic; some cycles have unknown funding" if partial else ""))
    be_val = _val(metrics, "val", "breakeven_cost_bps", h)
    g.checks.append(_check("breakeven_cost_bps_val", be_val, safety * cost_bps, ">",
                           f"cost-derived: needs {safety:g}× headroom over the assumed {cost_bps} bp one-way",
                           note="breakeven = (gross − funding) / turnover"))
    g.checks.append(_check("net_bp_val_vs_null", _net(metrics, "val", h)[0],
                           cal["net_available_bp"]["val"]["p95"], ">",
                           "null p95: a random signal's net over the same cycles", warn_only=True))
    turnover = _val(metrics, "dev", "turnover", h)
    g.checks.append(Check("turnover_per_cycle", turnover, None, ">", NA,
                          "reported, not gated: 2.0 means a full close-and-reopen every cycle",
                          f"cost drag ≈ {turnover * cost_bps:.1f} bp/cycle at {cost_bps} bp one-way"))
    return g


# --------------------------------------------------------------------------- G4
def gate_robustness(metrics, yearly, h, cal):
    g = Gate("G4_robustness", "稳健 / robustness",
             "does it hold across splits and years, or is it one regime?")
    ics = [_val(metrics, s, "ic_mean", h) for s in ("dev", "val", "oot")]
    ics = [x for x in ics if np.isfinite(x)]
    same = all(x > 0 for x in ics) if ics else False
    g.checks.append(Check("split_sign_consistency", float(sum(x > 0 for x in ics)), float(len(ics)),
                          ">=", PASS if same else FAIL,
                          "required: the frozen direction keeps its sign in every split",
                          f"{sum(x > 0 for x in ics)}/{len(ics)} splits positive"))
    if yearly is not None and len(yearly):
        pos = int((yearly > 0).sum())
        g.checks.append(Check("yearly_sign_consistency", float(pos), float(len(yearly) - 1), ">=",
                              PASS if pos >= len(yearly) - 1 else WARN,
                              "convention: at most one calendar year may disagree",
                              f"{pos}/{len(yearly)} years positive"))
    g.checks.append(_check("sharpe_net_val", _val(metrics, "val", "sharpe_net", h),
                           cal["sharpe_net"]["val"]["p95"], ">",
                           "null p95 (annualised on the same cycle count)", warn_only=True))
    return g


# --------------------------------------------------------------------------- G5
def gate_incremental(incremental):
    g = Gate("G5_incremental", "增量 / incremental",
             "is this more than a public baseline?")
    worst = incremental.get("max_abs_rank_corr", np.nan)
    g.checks.append(_check("max_abs_rank_corr_vs_baselines", worst, 0.7, "<",
                           "convention: |ρ| ≥ 0.7 against any single baseline is the same bet",
                           note=f"closest: {incremental.get('closest_baseline', '?')}"))
    g.checks.append(_check("residual_ic_retention", incremental.get("retention", np.nan), 0.5, ">=",
                           "convention: after projecting out all five baselines, keep at least half the IC"))
    g.checks.append(_check("residual_net_bp_val", incremental.get("residual_net_val", np.nan), 0.0, ">",
                           "required: the part that is not a baseline must still pay",
                           warn_only=True))
    return g


# --------------------------------------------------------------------------- verdict
VERDICTS = {
    "PASS": "clears every gate: information, money, robustness and incremental value",
    "PASS_CONDITIONAL": "information and money, but it is one regime or mostly a known baseline",
    "HOLD_INFO": "real ranking information, no money after cost — the most common honest outcome",
    "HOLD_WEAK": "money without measurable information — usually a few lucky cycles",
    "REJECT": "no information and no money",
    "REJECT_DATA": "not measurable on this panel; the other gates were not read",
}


def verdict_from(gates: dict[str, Gate]) -> tuple[str, list[str]]:
    def ok(key, *, missing=True):
        """A gate that was not run cannot pass; ``missing`` says whether that
        absence is allowed to block. G5 is skippable (it is the slow one), but a
        skipped G5 must not be reported as a clean PASS."""
        gate = gates.get(key)
        return missing if gate is None else gate.status in (PASS, WARN)

    if gates["G0_data"].status == FAIL:
        return "REJECT_DATA", ["G0_data"]
    info, money = ok("G1_information"), ok("G3_cost")
    blocking = [k for k, g in gates.items() if g.status == FAIL]
    if info and money:
        checked_all = "G5_incremental" in gates and ok("G5_incremental")
        if ok("G4_robustness") and checked_all:
            return "PASS", blocking
        return "PASS_CONDITIONAL", blocking
    if info:
        return "HOLD_INFO", blocking
    if money:
        return "HOLD_WEAK", blocking
    return "REJECT", blocking
