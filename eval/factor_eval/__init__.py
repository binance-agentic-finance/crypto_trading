"""factor_eval — score a cross-sectional crypto factor against a fixed matrix.

    from factor_eval import evaluate

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
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import baseline_signals, cross_sectional_rank, residualise
from .bundle import DEFAULT_BUNDLE, Panel, load_bundle, to_long
from .engine import DEFAULT_SPLITS, evaluate_factor
from .matrix import (Gate, VERDICTS, gate_cost, gate_data, gate_incremental,
                     gate_information, gate_robustness, gate_structure,
                     load_calibration, verdict_from)
from .report import scorecard_markdown, scorecard_text

__all__ = ["evaluate", "Scorecard", "Panel", "load_bundle", "to_long",
           "baseline_signals", "DEFAULT_BUNDLE", "DEFAULT_SPLITS", "VERDICTS"]

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

    @property
    def headline(self) -> str:
        return f"{self.name}: {self.verdict} — {VERDICTS[self.verdict]}"

    def to_markdown(self) -> str:
        return scorecard_markdown(self)

    def __str__(self) -> str:
        return scorecard_text(self)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "verdict": self.verdict, "blocking": self.blocking,
            "primary_h": self.primary_h, "cost_bps": self.cost_bps,
            "panel": self.panel_description,
            "gates": {k: {"title": g.title, "status": g.status,
                          "checks": [c.to_dict() for c in g.checks]}
                      for k, g in self.gates.items()},
            "structure": self.structure, "incremental": self.incremental,
            "yearly_ic": None if self.yearly_ic is None else
                         {str(k): float(v) for k, v in self.yearly_ic.items()},
        }


def _as_signal(factor, panel: Panel) -> pd.DataFrame:
    sig = factor(panel) if callable(factor) else factor
    if isinstance(sig, pd.Series):
        raise TypeError("a factor must produce a date × symbol DataFrame, not a Series")
    if not isinstance(sig, pd.DataFrame):
        raise TypeError(f"a factor must produce a DataFrame, got {type(sig).__name__}")
    sig = sig.reindex(index=panel.index, columns=panel.symbols)
    if not sig.notna().to_numpy().any():
        raise ValueError("the factor produced no finite values on this panel")
    return sig.astype(float)


def _structure(signal, panel, h, entry_lag, sign, bins=None):
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
    usable = panel.mask & signal.notna() & fwd.notna()
    s, f = signal.where(usable), fwd.where(usable)
    # Bin count follows the width of the cross-section. Ten names split five ways
    # leaves ~1.5 per bin, i.e. the extreme bins are single names and the "shape"
    # is one coin's tail. Aim for >=2.5 names per bin, 3..5 bins.
    width = float(usable.sum(axis=1).replace(0, np.nan).mean())
    if bins is None:
        bins = int(np.clip(np.floor(width / 2.5), 3, 5)) if np.isfinite(width) else 3
    q = s.rank(axis=1, pct=True)
    rows = []
    for i in range(bins):
        sel = (q > i / bins) & (q <= (i + 1) / bins)
        per_date = f.where(sel).mean(axis=1)      # equal weight per decision date …
        counts = sel.sum(axis=1)
        rows.append({"bin": i + 1, "n": int(counts.sum()),
                     # … then average across dates, the same weighting the IC uses.
                     # Pooling asset-days instead lets a handful of fat-tailed moves
                     # in the high-volatility names invert the picture while the
                     # per-date IC is positive — the two gates would then disagree
                     # for a purely mechanical reason.
                     "mean_bp": float(per_date.mean() * 1e4) if counts.sum() else np.nan})
    table = pd.DataFrame(rows)
    ok = table.dropna(subset=["mean_bp"])
    spearman = (ok["bin"].corr(ok["mean_bp"], method="spearman")
                if len(ok) >= 3 else np.nan)
    spread = (table.mean_bp.iloc[-1] - table.mean_bp.iloc[0]
              if table.mean_bp.notna().all() else np.nan)
    per_bin = width / bins if np.isfinite(width) else np.nan
    return {"bins": table.to_dict("records"), "spearman": float(spearman),
            "spread_bp": float(spread), "n_bins": int(bins),
            "names_per_bin": float(per_bin),
            "bins_note": (f"{bins} bins, ~{per_bin:.1f} names per bin per date, "
                          f"{int(table.n.sum())} asset-days")}


def _yearly_ic(ic_table, h):
    x = ic_table[(ic_table.h == h)].dropna(subset=["ic"])
    if x.empty:
        return None
    return x.groupby(x.signal_time.dt.year)["ic"].mean()


def _incremental(signal, panel, h, cost_bps, splits, entry_lag, metrics):
    base = baseline_signals(panel)
    sig_rank = cross_sectional_rank(signal, panel.mask)
    corrs = {name: float(sig_rank.corrwith(cross_sectional_rank(b, panel.mask), axis=1).mean())
             for name, b in base.items()}
    closest = max(corrs, key=lambda k: abs(corrs[k])) if corrs else None
    resid = residualise(signal, base, panel.mask)
    out = {"rank_corr": corrs, "closest_baseline": closest,
           "max_abs_rank_corr": float(max(abs(v) for v in corrs.values())) if corrs else np.nan}
    try:
        r = evaluate_factor(resid, panel.open, panel.close, panel.mask, panel.funding,
                            horizons=(h,), cost_bps=cost_bps, splits=splits, entry_lag=entry_lag)
        rm = r["metrics"]
        base_ic = abs(float(metrics[(metrics.split == "dev") & (metrics.h == h)].iloc[0]["ic_mean"]))
        res_ic = abs(float(rm[(rm.split == "dev") & (rm.h == h)].iloc[0]["ic_mean"]))
        net = rm[(rm.split == "val") & (rm.h == h)]
        net_val = float(net.iloc[0]["net_bp"])
        if not np.isfinite(net_val):
            net_val = float(net.iloc[0]["net_available_bp"])
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
             with_incremental: bool = True) -> Scorecard:
    """Score one factor. ``factor`` is a callable ``Panel -> DataFrame`` or a DataFrame."""
    panel = panel if panel is not None else load_bundle()
    panel.validate()
    if primary_h not in tuple(horizons):
        horizons = tuple(sorted({*horizons, primary_h}))
    signal = _as_signal(factor, panel)
    splits = splits or DEFAULT_SPLITS
    cal = calibration if isinstance(calibration, dict) else load_calibration(calibration)
    if int(cal.get("primary_h", PRIMARY_H)) != int(primary_h) or abs(
            float(cal.get("cost_bps", COST_BPS)) - float(cost_bps)) > 1e-9:
        raise ValueError(
            f"calibration is for h={cal.get('primary_h')} and cost={cal.get('cost_bps')} bp; "
            f"you asked for h={primary_h} and cost={cost_bps} bp. Re-run "
            "`python -m factor_eval calibrate` for this setting instead of comparing "
            "against a noise floor that was measured elsewhere.")

    result = evaluate_factor(signal, panel.open, panel.close, panel.mask, panel.funding,
                             horizons=tuple(horizons), cost_bps=cost_bps, splits=splits,
                             entry_lag=entry_lag)
    metrics, ic_table = result["metrics"], result["ic"]
    dev_row = metrics[(metrics.split == "dev") & (metrics.h == primary_h)]
    frozen_sign = float(dev_row.iloc[0]["sign"]) if not dev_row.empty else 1.0
    structure = _structure(signal, panel, primary_h, entry_lag, frozen_sign)
    yearly = _yearly_ic(ic_table, primary_h)
    incremental = (_incremental(signal, panel, primary_h, cost_bps, splits, entry_lag, metrics)
                   if with_incremental else {})

    gates = {
        "G0_data": gate_data(metrics, primary_h),
        "G1_information": gate_information(metrics, primary_h, cal),
        "G2_structure": gate_structure(structure, metrics, primary_h, tuple(horizons)),
        "G3_cost": gate_cost(metrics, primary_h, cost_bps, cal),
        "G4_robustness": gate_robustness(metrics, yearly, primary_h, cal),
    }
    if with_incremental:
        gates["G5_incremental"] = gate_incremental(incremental)
    verdict, blocking = verdict_from(gates)
    return Scorecard(name=name, verdict=verdict, blocking=blocking, gates=gates,
                     metrics=metrics, primary_h=primary_h, cost_bps=cost_bps,
                     panel_description=panel.describe(), structure=structure,
                     incremental=incremental, yearly_ic=yearly,
                     config={**result["config"], "calibration": cal.get("source", str(calibration))})
