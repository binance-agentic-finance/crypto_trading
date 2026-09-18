"""cyqnt_trd.eval — score a cross-sectional crypto factor against a fixed matrix.

    from cyqnt_trd.eval import evaluate

    def my_factor(p):                      # p is a Panel of aligned daily frames
        return -(p.close / p.close.shift(5) - 1)      # 5-day reversal

    card = evaluate(my_factor, name="rev5")
    print(card.to_markdown())
    card.verdict        # 'HOLD_INFO'
    card.blocking       # ['G3_cost']

What it does, in one paragraph: ranks your factor inside the eligible top-ten
universe each day, freezes the sign on the development split and never re-picks
it, enters at ``open[t+2]`` (one full day after the bar that produced the signal),
holds ``h`` days in non-overlapping cycles, charges both sides of every
close-and-reopen plus the *actual* per-settlement funding, and then reports where
the factor fails — data, information, structure, cost, robustness, or incremental
value. Thresholds for "is this better than nothing?" come from random signals put
through the identical pipeline, not from taste.

What it deliberately does not do: search. One factor, one frozen direction, one
primary horizon. If you scan variants, that is a search and the thresholds here
no longer hold — the reference study in `eval/alpha101_crypto/` shows how far a
101-formula scan drifts from its own single-factor statistics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import inspect
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import baseline_signals, cross_sectional_rank, residualise
from .capability import capability_factor, read_port
from .bundle import DEFAULT_BUNDLE, Panel, load_bundle, to_long
from .causality import check_prefix_invariance
from .engine import DAY, DEFAULT_SPLITS, _split_bounds, evaluate_factor
from .diagnostics import build_diagnostics, signal_persistence
from .matrix import (Gate, VERDICTS, gate_cost, gate_data, gate_incremental,
                     gate_information, gate_robustness, gate_structure,
                     load_calibration, verdict_from)
from .preprocess import neutralize, winsorize_mad, winsorize_quantile, zscore
from .report import scorecard_markdown, scorecard_text
from .provenance import calibration_contract, validate_calibration

__all__ = ["evaluate", "Scorecard", "Panel", "load_bundle", "to_long",
           "baseline_signals", "DEFAULT_BUNDLE", "DEFAULT_SPLITS", "VERDICTS",
           # helpers for writing a factor, same role as jqfactor_analyzer.preprocess
           "winsorize_mad", "winsorize_quantile", "zscore", "neutralize",
           "signal_persistence",
           # adapter: capability node output -> panel the matrix can score
           "capability_factor", "read_port"]

PRIMARY_H = 3
HORIZONS = (1, 3, 5)
COST_BPS = 6.5


@dataclass
class Scorecard:
    name: str
    verdict: str
    blocking: list[str]
    gates: dict[str, Gate]
    metrics: pd.DataFrame
    primary_h: int
    cost_bps: float
    panel_description: str
    structure: dict = field(default_factory=dict)
    incremental: dict = field(default_factory=dict)
    yearly_ic: pd.Series | None = None
    config: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def headline(self) -> str:
        return f"{self.name}: {self.verdict} — {VERDICTS[self.verdict]}"

    def to_markdown(self) -> str:
        return scorecard_markdown(self)

    def __str__(self) -> str:
        return scorecard_text(self)

    def to_dict(self) -> dict:
        return _json_clean({
            "name": self.name, "verdict": self.verdict, "blocking": self.blocking,
            "primary_h": self.primary_h, "cost_bps": self.cost_bps,
            "panel": self.panel_description,
            "config": self.config,
            "metrics": self.metrics.to_dict("records"),
            "diagnostics": self.diagnostics,
            "assessments": {"information": self.gates["G1_information"].status,
                            "economic": self.gates["G3_cost"].status,
                            "baseline_novelty": self.gates["G5_incremental"].status
                                if "G5_incremental" in self.gates else "NOT_EVALUATED",
                            "strategy_combination": "NOT_EVALUATED"},
            "gates": {k: {"title": g.title, "status": g.status,
                          "checks": [c.to_dict() for c in g.checks]}
                      for k, g in self.gates.items()},
            "structure": self.structure, "incremental": self.incremental,
            "yearly_ic": None if self.yearly_ic is None else
                         {str(k): float(v) for k, v in self.yearly_ic.items()},
        })


def _json_clean(value):
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _reproduction_identity(factor, signal, calibration):
    values = signal.to_numpy(dtype="<f8", copy=True)
    values[np.isnan(values)] = np.nan
    values[values == 0] = 0.0
    source_hash = None
    if callable(factor):
        try:
            source_hash = hashlib.sha256(inspect.getsource(factor).encode()).hexdigest()
        except (OSError, TypeError):
            pass  # Interactive callables can lack source; signal hash still identifies the output.
    digest = hashlib.sha256()
    for name in ("__init__.py", "engine.py", "matrix.py", "baselines.py", "bundle.py", "causality.py", "provenance.py", "diagnostics.py", "targets.py", "statistics.py"):
        digest.update(name.encode())
        digest.update((Path(__file__).parent / name).read_bytes())
    return {"signal_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            "factor_source_sha256": source_hash, "evaluation_code_sha256": digest.hexdigest(),
            "calibration_sha256": hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest(),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}


def _as_signal(factor, panel: Panel) -> pd.DataFrame:
    sig = factor(panel) if callable(factor) else factor
    if isinstance(sig, pd.Series):
        raise TypeError("a factor must produce a date × symbol DataFrame, not a Series")
    if not isinstance(sig, pd.DataFrame):
        raise TypeError(f"a factor must produce a DataFrame, got {type(sig).__name__}")
    if not sig.index.is_unique or not sig.columns.is_unique:
        raise ValueError("factor signal must have unique date and symbol labels")
    sig = sig.reindex(index=panel.index, columns=panel.symbols).astype(float)
    if not np.isfinite(sig.to_numpy()).any():
        raise ValueError("the factor produced no finite values on this panel")
    return sig.astype(float)


def _structure(signal, panel, h, entry_lag, sign, bins=None, splits=None):
    """Bin the frozen-direction signal and read the forward return per bin.

    ``sign`` is the direction the engine froze on dev; the bins must be built on
    ``sign * signal``, not the raw signal, or a negative-direction factor reads as
    perfectly *inverted* here while its IC and P&L are positive.

    Uses the same entry the engine trades on (open[t+entry_lag] → open[t+entry_lag+h]),
    so the picture and the P&L cannot disagree about what was tradable.
    """
    signal = signal * float(sign)
    opens = panel.open
    fwd = np.log(opens.shift(-(entry_lag + h)) / opens.shift(-entry_lag))
    usable = panel.mask & np.isfinite(signal)
    # Rank using decision-time eligibility only. A missing future label must
    # not move the surviving names into different bins.
    s, f = signal.where(usable), fwd.where(usable)
    # Bin count follows the width of the cross-section. Ten names split five ways
    # leaves ~1.5 per bin, i.e. the extreme bins are single names and the "shape"
    # is one coin's tail. Aim for >=2.5 names per bin, 3..5 bins.
    bounds = _split_bounds(DEFAULT_SPLITS if splits is None else splits)
    dev_start, dev_end = bounds["dev"]
    dev_dates = (panel.index >= dev_start) & (panel.index + (entry_lag + h) * DAY <= dev_end)
    width = float(usable.loc[dev_dates].sum(axis=1).replace(0, np.nan).mean())
    if bins is None:
        bins = int(np.clip(np.floor(width / 2.5), 3, 5)) if np.isfinite(width) else 3
    q = s.rank(axis=1, pct=True)
    selections = [(q > i / bins) & (q <= (i + 1) / bins) for i in range(bins)]
    counts = pd.DataFrame({i: sel.sum(axis=1) for i, sel in enumerate(selections)})
    # Ties stay together; a thin/empty bin invalidates the entire date. Compare
    # every bin on the same dates rather than averaging different regimes.
    common_dates = counts.ge(2).all(axis=1) & (~usable | np.isfinite(fwd)).all(axis=1)
    by_split = {}
    for split, (start, end) in bounds.items():
        dates = common_dates & (panel.index >= start) & (panel.index + (entry_lag + h) * DAY <= end)
        rows = []
        for i, sel in enumerate(selections):
            per_date = f.where(sel).mean(axis=1).loc[dates]
            rows.append({"bin": i + 1, "n": int(counts.loc[dates, i].sum()),
                         "mean_bp": float(per_date.mean() * 1e4)})
        table = pd.DataFrame(rows)
        enough = int(dates.sum()) >= 30
        spearman = table["bin"].corr(table.mean_bp, method="spearman") if enough else np.nan
        spread = float(table.mean_bp.iloc[-1] - table.mean_bp.iloc[0]) if enough else np.nan
        per_bin = float(counts.loc[dates].to_numpy().mean()) if dates.any() else np.nan
        by_split[split] = {"bins": rows, "spearman": float(spearman), "spread_bp": spread,
                           "n_bins": int(bins), "n_dates": int(dates.sum()),
                           "names_per_bin": per_bin,
                           "bins_note": f"{split}: {bins} bins; {int(dates.sum())} common dates; >=2 names/bin; >=30 dates required"}
    return {**by_split["val"], "gate_split": "val", "by_split": by_split,
            "bin_fit_split": "dev", "weighting": "equal date then equal asset; common dates across bins"}


def _yearly_ic(ic_table, h):
    x = ic_table[(ic_table.h == h)].dropna(subset=["ic"])
    if x.empty:
        return None
    return x.groupby(x.signal_time.dt.year)["ic"].mean()


def _incremental(signal, panel, h, cost_bps, splits, entry_lag, metrics):
    base = baseline_signals(panel)
    sig_rank = cross_sectional_rank(signal, panel.mask)
    dev_start, dev_end = _split_bounds(splits)["dev"]
    dev_dates = (panel.index >= dev_start) & (panel.index <= dev_end)
    corrs = {name: float(sig_rank.corrwith(cross_sectional_rank(b, panel.mask), axis=1).loc[dev_dates].mean())
             for name, b in base.items()}
    finite_corrs = {k: v for k, v in corrs.items() if np.isfinite(v)}
    closest = max(finite_corrs, key=lambda k: abs(finite_corrs[k])) if finite_corrs else None
    resid = residualise(signal, base, panel.mask)
    out = {"rank_corr": corrs, "closest_baseline": closest,
           "max_abs_rank_corr": max(abs(v) for v in finite_corrs.values()) if finite_corrs else np.nan,
           "correlation_split": "dev", "scope": "public-baseline novelty, not strategy combination"}
    try:
        r = evaluate_factor(resid, panel.open, panel.close, panel.mask, panel.funding,
                            horizons=(h,), cost_bps=cost_bps, splits=splits, entry_lag=entry_lag)
        rm = r["metrics"]
        base_ic = abs(float(metrics[(metrics.split == "dev") & (metrics.h == h)].iloc[0]["ic_mean"]))
        res_ic = abs(float(rm[(rm.split == "dev") & (rm.h == h)].iloc[0]["ic_mean"]))
        net = rm[(rm.split == "val") & (rm.h == h)]
        net_val = float(net.iloc[0]["net_bp"])
        out["residual_ic_dev"] = res_ic
        out["retention"] = res_ic / base_ic if base_ic > 0 else np.nan
        out["residual_net_val"] = net_val
    except Exception as exc:                        # noqa: BLE001 - diagnostic only
        out["residual_error"] = f"{type(exc).__name__}: {exc}"
        out["retention"] = np.nan
        out["residual_net_val"] = np.nan
    return out


def evaluate(factor, panel: Panel | None = None, *, name: str = "factor",
             horizons=HORIZONS, primary_h: int = PRIMARY_H, cost_bps: float = COST_BPS,
             splits=None, entry_lag: int = 2, calibration=None,
             with_incremental: bool = True, trials_seen: int = 1,
             with_diagnostics: bool = True, n_bootstrap: int = 100,
             diagnostic_horizons=tuple(range(1, 61)), benchmark_symbol="BTCUSDT",
             spread: pd.DataFrame | None = None) -> Scorecard:
    """Score one factor. ``factor`` is a callable ``Panel -> DataFrame`` or a DataFrame."""
    panel = panel if panel is not None else load_bundle()
    panel.validate()
    horizons = tuple(horizons)
    if isinstance(trials_seen, bool) or not isinstance(trials_seen, (int, np.integer)) or trials_seen < 1:
        raise ValueError("trials_seen must be a positive integer counting all examined variants")
    if primary_h not in horizons:
        horizons = tuple(sorted({*horizons, primary_h}))
    signal = _as_signal(factor, panel)
    splits = DEFAULT_SPLITS if splits is None else splits
    if set(splits) != {"dev", "val", "oot"}:
        raise ValueError("the scorecard requires exactly dev, val and oot splits")
    cal = calibration if isinstance(calibration, dict) else load_calibration(calibration)
    validate_calibration(cal, panel, primary_h=primary_h, cost_bps=cost_bps,
                         splits=splits, entry_lag=entry_lag)
    causality = check_prefix_invariance(factor, panel, signal)

    result = evaluate_factor(signal, panel.open, panel.close, panel.mask, panel.funding,
                             horizons=tuple(horizons), cost_bps=cost_bps, splits=splits,
                             entry_lag=entry_lag)
    metrics, ic_table = result["metrics"], result["ic"]
    dev_row = metrics[(metrics.split == "dev") & (metrics.h == primary_h)]
    frozen_sign = float(dev_row.iloc[0]["sign"]) if not dev_row.empty else 1.0
    structure = _structure(signal, panel, primary_h, entry_lag, frozen_sign, splits=splits)
    yearly = _yearly_ic(ic_table, primary_h)
    incremental = (_incremental(signal, panel, primary_h, cost_bps, splits, entry_lag, metrics)
                   if with_incremental else {})

    # Computed once and consumed twice: the cost gate reads it, the diagnostics
    # export it. Persistence is sign-invariant, so the frozen direction is irrelevant.
    persistence = signal_persistence(signal, panel.mask, (1, primary_h, 5))

    gates = {
        "G0_data": gate_data(metrics, primary_h),
        "G1_information": gate_information(metrics, primary_h, cal),
        "G2_structure": gate_structure(structure, metrics, primary_h, tuple(horizons)),
        "G3_cost": gate_cost(metrics, primary_h, cost_bps, cal, persistence=persistence),
        "G4_robustness": gate_robustness(metrics, yearly, primary_h, cal),
    }
    if with_incremental:
        gates["G5_incremental"] = gate_incremental(incremental)
    gates["G0_data"].checks.append(causality)
    verdict, blocking = verdict_from(gates)
    if trials_seen > 1 and verdict.startswith("PASS"):
        verdict, blocking = "HOLD_SEARCH", [*blocking, "search_selection"]
    diagnostics = (build_diagnostics(signal, panel, primary_h=primary_h, entry_lag=entry_lag,
                                    splits=splits, sign=frozen_sign, metrics=metrics, periods=result["periods"],
                                    cost_bps=cost_bps, gates=gates, trials_seen=trials_seen,
                                    n_bootstrap=n_bootstrap, ftr_horizons=diagnostic_horizons,
                                    benchmark_symbol=benchmark_symbol, spread=spread,
                                    persistence=persistence)
                   if with_diagnostics else {})
    return Scorecard(name=name, verdict=verdict, blocking=blocking, gates=gates,
                     metrics=metrics, primary_h=primary_h, cost_bps=cost_bps,
                     panel_description=panel.describe(), structure=structure,
                     incremental=incremental, yearly_ic=yearly,
                     diagnostics=diagnostics,
                     config={**result["config"], "calibration": cal.get("source", "custom calibration"),
                             **_reproduction_identity(factor, signal, cal),
                             "calibration_contract": calibration_contract(
                                 panel, primary_h=primary_h, cost_bps=cost_bps, splits=splits, entry_lag=entry_lag),
                             "trials_seen": int(trials_seen),
                             "evaluation_scope": "single_cyqnt_trd.evaluation_matrix" if with_diagnostics else "six_gate_research_screen",
                             "diagnostics_enabled": bool(with_diagnostics),
                             "strategy_combination": "NOT_EVALUATED",
                             "selection_caveat": panel.meta.get("selection_caveat", "universe provenance not supplied"),
                             "holdout_status": "oot is a time split, not evidence of a sealed holdout",
                             "funding_price_basis": "provided settlement reference; may include historical mark-open proxies",
                             "capacity": "NOT_EVALUATED: no order-book depth or market-impact model"})
