"""The pipeline, as a fixed list of stages with declared input and output formats.

Everything else in this package is one stage's implementation. This module says
what the stages *are*, what each one takes, what each one must return, and checks
the handover at every boundary so a shape error surfaces at the stage that caused
it rather than three stages later as an empty result.

    #  stage         input                                  output
    0  sample        raw fine-grained bars per symbol       Bars    dict[symbol, DataFrame]
    1  panel         Bars (+ extras, + eligibility)         Panel   aligned cells x symbols
    2  factors       Panel                                  Factors dict[name, DataFrame]
    3  preprocess    Factors                                Factors (standardised)
    4  forecast      Factors, Panel                         Mu      DataFrame cells x symbols
    5  sizing        Mu, mask                               W       DataFrame cells x symbols
    6  backtest      W, Panel                               BacktestResult
    7  compare       BacktestResult, benchmark books        DataFrame per split

Stages 2-5 are **stateless maps**: `value[t] = h(inputs up to t, cross-section at t)`.
No cross-cell mutable state, no "have I already entered", no future labels written
back. That is what makes the backtest a matrix product rather than an event loop,
and what makes `W`'s last row a live order without running a second code path.
Stop-losses and scaling paths need state and belong to an execution layer that is
deliberately not part of this chain.

`check_stateless()` verifies the claim rather than trusting it: truncate the panel,
recompute, and require the surviving rows to be bit-identical.

The engine here is **not** `engine.evaluate_factor`. That one scores a *factor* by
constructing its own rank portfolio; this one takes an arbitrary `W` that a caller
already decided on. Both charge the same way -- open-to-open after `entry_lag`,
non-overlapping `h`-cell periods, both sides of turnover, real per-settlement
funding, missing data propagated as NaN rather than zero. `engine.py` is not
extended in place because its source hash is part of the calibration contract
(`provenance.calibration_contract`): editing it invalidates the shipped null
distribution and makes every `evaluate()` call raise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .bundle import PRICE_FIELDS, Panel
from .engine import DEFAULT_SPLITS, _split_bounds

__all__ = ["STAGES", "Factors", "Mu", "Weights", "BacktestSpec", "BacktestResult",
           "backtest_weights", "run", "check_stateless", "describe_stages",
           "validate_factors", "validate_mu", "validate_weights"]

# Stage 2-5 all carry the same cells x symbols shape; the aliases exist so a
# signature says which stage's output it means.
Factors = Mapping[str, pd.DataFrame]
Mu = pd.DataFrame
Weights = pd.DataFrame

STAGES = (
    {"n": 0, "name": "sample", "input": "raw bars per symbol",
     "output": "dict[symbol, DataFrame] on one grid", "impl": "cyqnt_trd.eval.bars.sample_bars"},
    {"n": 1, "name": "panel", "input": "bars + extras + eligibility",
     "output": "Panel (aligned cells x symbols)", "impl": "cyqnt_trd.eval.bars.build_panel"},
    {"n": 2, "name": "factors", "input": "Panel",
     "output": "dict[name, DataFrame]", "impl": "cyqnt_trd.eval.capability_registry / library"},
    {"n": 3, "name": "preprocess", "input": "dict[name, DataFrame]",
     "output": "dict[name, DataFrame]", "impl": "cyqnt_trd.eval.preprocess"},
    {"n": 4, "name": "forecast", "input": "factors + Panel",
     "output": "DataFrame (expected return score)", "impl": "cyqnt_trd.eval.forecast"},
    {"n": 5, "name": "sizing", "input": "mu + mask",
     "output": "DataFrame (target weights)", "impl": "cyqnt_trd.eval.forecast.positions_from_forecast"},
    {"n": 6, "name": "backtest", "input": "W + Panel",
     "output": "BacktestResult", "impl": "cyqnt_trd.eval.framework.backtest_weights"},
    {"n": 7, "name": "compare", "input": "BacktestResult + benchmarks",
     "output": "DataFrame per split", "impl": "cyqnt_trd.eval.benchmarks.compare"},
)


def describe_stages() -> str:
    head = f"{'#':<3}{'stage':<12}{'input':<38}{'output':<36}impl"
    rows = [f"{s['n']:<3}{s['name']:<12}{s['input']:<38}{s['output']:<36}{s['impl']}"
            for s in STAGES]
    return "\n".join([head, "-" * len(head), *rows])


# --------------------------------------------------------------- boundary checks
def _check_frame(frame, panel: Panel, what: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{what} must be a DataFrame (cells x symbols), got "
                        f"{type(frame).__name__}")
    if not frame.index.equals(panel.index):
        raise ValueError(f"{what} is not on the panel's cell index "
                         f"({len(frame.index)} rows vs {len(panel.index)})")
    if list(frame.columns) != panel.symbols:
        raise ValueError(f"{what} columns must be the panel's symbols in order")
    values = frame.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError(f"{what} contains infinite values")
    return frame


def validate_factors(factors: Factors, panel: Panel) -> dict[str, pd.DataFrame]:
    """Stage 2/3 output: a named set of cells x symbols frames, all on the grid."""
    if not isinstance(factors, Mapping) or not factors:
        raise TypeError("stage 2/3 must return a non-empty mapping of name -> DataFrame")
    out = {}
    for name, frame in factors.items():
        out[name] = _check_frame(frame, panel, f"factor {name!r}")
        if not np.isfinite(out[name].where(panel.mask).to_numpy(dtype=float)).any():
            raise ValueError(f"factor {name!r} has no finite value inside the mask")
    return out


def validate_mu(mu, panel: Panel) -> Mu:
    """Stage 4 output: one expected-return score per cell per symbol."""
    return _check_frame(mu, panel, "forecast mu")


def validate_weights(weights, panel: Panel, *, gross_tol: float = 1e-6) -> Weights:
    """Stage 5 output: target weights as a fraction of equity.

    NaN is rejected here. A weight is a decision, and "unknown" is not one -- an
    unknown *input* must have already become an explicit 0 (do not hold) by this
    point. Letting NaN through makes it ambiguous downstream whether a cell means
    flat or means carry the previous position.
    """
    w = _check_frame(weights, panel, "weights W")
    values = w.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("weights contain NaN; an undecidable cell must be an explicit 0")
    held_outside = (np.abs(values) > gross_tol) & ~panel.mask.to_numpy(dtype=bool)
    if held_outside.any():
        raise ValueError(f"weights hold {int(held_outside.sum())} cells outside the "
                         f"eligibility mask")
    return w


# ------------------------------------------------------------------- the engine
@dataclass
class BacktestResult:
    """Stage 6 output."""

    weights: Weights                     # cells x symbols, as handed in
    pnl: pd.DataFrame                    # one row per traded period
    equity: pd.Series                    # compounded net equity, indexed by decision cell
    metrics: pd.DataFrame                # one row per split
    config: dict = field(default_factory=dict)

    def summary(self) -> str:
        rows = [f"{r.split}: net {r.net_bp:+.1f}bp/period over {int(r.n_periods)} periods, "
                f"Sharpe {r.sharpe_net:.2f}, turnover {r.turnover:.3f}"
                for r in self.metrics.itertuples() if np.isfinite(r.net_bp)]
        return " | ".join(rows) if rows else "no complete period"


def backtest_weights(weights: Weights, panel: Panel, *, entry_lag: int = 2,
                     h: int = 3, cost_bps: float = 6.5, splits=None,
                     annualization: float = 365.0) -> BacktestResult:
    """`W (.) R` with turnover cost and real funding, on non-overlapping periods.

    Periods start at the first cell that can complete `entry_lag + h` and step by
    `h`, so no cell's return is counted twice. For a period decided at cell `t`:

        e            = t + entry_lag                      execution cell
        R[t, s]      = open[e+h, s] / open[e, s] - 1
        funding[t,s] = sum(funding[e:e+h, s]) / open[e, s]   per unit of notional
        gross[t]     = sum_s W[t, s] * R[t, s]
        turnover[t]  = sum_s |W[t, s] - W[previous period, s]|
        net[t]       = gross[t] - turnover[t]*cost_bps/1e4 - sum_s W[t,s]*funding[t,s]

    A period where any held symbol is missing a price or a funding observation is
    `NaN`, not zero -- an unknown cost is not a zero cost. Those periods are
    excluded from the reported means and counted separately.
    """
    w = validate_weights(weights, panel)
    if entry_lag < 0 or h < 1:
        raise ValueError("entry_lag must be >= 0 and h >= 1 cell")
    opens = panel.open.to_numpy(dtype=float)
    funding = panel.funding.to_numpy(dtype=float)
    values = w.to_numpy(dtype=float)
    n_cells = len(panel.index)

    starts = [t for t in range(0, n_cells) if t + entry_lag + h < n_cells]
    starts = starts[::h]
    rows, previous = [], np.zeros(values.shape[1])
    for t in starts:
        e = t + entry_lag
        entry, exit_ = opens[e], opens[e + h]
        held = np.abs(values[t]) > 0
        forward = np.where((entry > 0) & np.isfinite(entry) & np.isfinite(exit_),
                           exit_ / np.where(entry > 0, entry, np.nan) - 1.0, np.nan)
        carried = funding[e:e + h]
        carry = np.where(np.isfinite(entry) & (entry > 0),
                         carried.sum(axis=0) / np.where(entry > 0, entry, np.nan), np.nan)
        incomplete = bool(held.any() and (
            ~np.isfinite(forward[held]) | ~np.isfinite(carry[held])
            | ~np.isfinite(carried[:, held]).all(axis=0)).any())
        turnover = float(np.abs(values[t] - previous).sum())
        cost = turnover * cost_bps / 1e4
        if incomplete:
            gross = fund = net = np.nan
        else:
            gross = float((values[t] * np.nan_to_num(forward, nan=0.0)).sum())
            fund = float((values[t] * np.nan_to_num(carry, nan=0.0)).sum())
            net = gross - cost - fund
        rows.append({"cell": panel.index[t], "entry": panel.index[e],
                     "exit": panel.index[e + h], "gross": gross, "trading_cost": cost,
                     "funding": fund, "net": net, "turnover": turnover,
                     "n_held": int(held.sum()), "complete": not incomplete})
        previous = values[t]

    pnl = pd.DataFrame(rows)
    if pnl.empty:
        raise ValueError("the panel is too short for one complete entry_lag + h period")
    equity = pd.Series((1.0 + pnl["net"].fillna(0.0)).cumprod().to_numpy(),
                       index=pd.DatetimeIndex(pnl["cell"]), name="equity")

    bounds = _split_bounds(DEFAULT_SPLITS if splits is None else splits)
    per_period = annualization / h
    metrics = []
    for name, (lo, hi) in bounds.items():
        cells = pd.DatetimeIndex(pnl["cell"])
        window = pnl[(cells >= lo) & (pd.DatetimeIndex(pnl["exit"]) <= hi)]
        complete = window[window["complete"]]
        net = complete["net"]
        metrics.append({
            "split": name, "n_periods": len(window), "n_complete": len(complete),
            "n_incomplete": int((~window["complete"]).sum()),
            "gross_bp": float(complete["gross"].mean() * 1e4) if len(complete) else np.nan,
            "cost_bp": float(complete["trading_cost"].mean() * 1e4) if len(complete) else np.nan,
            "funding_bp": float(complete["funding"].mean() * 1e4) if len(complete) else np.nan,
            "net_bp": float(net.mean() * 1e4) if len(complete) else np.nan,
            "sharpe_net": (float(net.mean() / net.std() * np.sqrt(per_period))
                           if len(complete) > 1 and net.std() > 0 else np.nan),
            "turnover": float(window["turnover"].mean()) if len(window) else np.nan,
            "breakeven_cost_bps": (
                float((complete["gross"] - complete["funding"]).mean()
                      / complete["turnover"].replace(0, np.nan).mean() * 1e4)
                if len(complete) else np.nan),
        })
    config = {"engine": "factor-eval.framework/weights-v1", "entry_lag": entry_lag,
              "h": h, "cost_bps": cost_bps, "annualization": annualization,
              "periods": len(pnl), "overlap": "non-overlapping, step = h",
              "execution": "signal at cell t, fill at open[t+entry_lag], exit at open[t+entry_lag+h]",
              "missing": "a period holding a symbol with an unknown price or funding is NaN",
              "splits": {k: [str(v[0]), str(v[1])] for k, v in bounds.items()}}
    return BacktestResult(weights=w, pnl=pnl, equity=equity,
                          metrics=pd.DataFrame(metrics), config=config)


# ------------------------------------------------------------------ the pipeline
@dataclass
class BacktestSpec:
    """Stage 2-6 inputs. Stages 0-1 produce the `panel` this refers to."""

    panel: Panel
    factors: Mapping[str, Callable[[Panel], pd.DataFrame] | pd.DataFrame]
    forecast: Callable[[dict, Panel], pd.DataFrame]
    sizing: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
    preprocess: Callable[[dict, Panel], dict] | None = None
    entry_lag: int = 2
    horizon: int = 3
    cost_bps: float = 6.5
    splits: dict | None = None
    grid: dict = field(default_factory=dict)      # how stage 0 sampled, for the record

    def describe(self) -> str:
        return (f"{len(self.factors)} factors, h={self.horizon}, entry_lag={self.entry_lag}, "
                f"cost={self.cost_bps}bp, grid={self.grid or 'daily (bundled)'}")


def run(spec: BacktestSpec) -> BacktestResult:
    """Run stages 2-6, validating every handover.

    Each stage's output is checked against the panel grid before the next stage
    sees it, so a stage that returns the wrong shape fails with its own name in
    the message.
    """
    panel = spec.panel
    panel.validate()

    raw = {name: (f(panel) if callable(f) else f) for name, f in spec.factors.items()}
    stage2 = validate_factors(raw, panel)

    stage3 = stage2 if spec.preprocess is None else validate_factors(
        spec.preprocess(stage2, panel), panel)

    mu = validate_mu(spec.forecast(stage3, panel), panel)
    w = validate_weights(spec.sizing(mu, panel.mask), panel)

    result = backtest_weights(w, panel, entry_lag=spec.entry_lag, h=spec.horizon,
                              cost_bps=spec.cost_bps, splits=spec.splits)
    result.config.update({"spec": spec.describe(), "factors": sorted(stage2),
                          "grid": dict(spec.grid),
                          "stages": [s["name"] for s in STAGES[2:7]]})
    return result


# ------------------------------------------------------- the stateless contract
def _truncate(panel: Panel, rows: int) -> Panel:
    cut = {f: getattr(panel, f).iloc[:rows] for f in PRICE_FIELDS}
    return Panel(**cut, funding=panel.funding.iloc[:rows], mask=panel.mask.iloc[:rows],
                 meta=dict(panel.meta), cell_scheme=panel.cell_scheme,
                 extras={k: v.iloc[:rows] for k, v in panel.extras.items()})


def check_stateless(build: Callable[[Panel], pd.DataFrame], panel: Panel, *,
                    fractions: Sequence[float] = (0.6, 0.8), atol: float = 0.0) -> dict:
    """Recompute on a truncated panel; surviving rows must be unchanged.

    Catches the usual ways a future leaks in: centred windows, whole-sample
    standardisation, a stray `shift(-1)`, coefficients fitted on everything. If
    this passes, the last row of the full-panel result is the same number a live
    run would have produced at that moment -- which is what lets research and
    production share one code path.

    Three ways a run can disagree, and all three are failures:

    `mismatched`
        Both runs produced a number and the numbers differ.
    `known_only_on_full`
        The full panel produced a value where the prefix could not. The extra
        information came from rows after `t`, which is the definition of a leak.
        **This is the one that matters and the one easiest to miss**: a
        `shift(-k)` agrees everywhere except the last `k` rows of the prefix, and
        those rows are exactly the ones a "compare where both are finite" rule
        throws away. Checked against a deliberate `shift(-3)`, which passes without
        it.
    `known_only_on_truncated`
        The prefix produced a value the full run did not. Cannot come from causal
        code either.
    """
    full = build(panel)
    findings = []
    for fraction in fractions:
        rows = int(len(panel.index) * fraction)
        if rows < 30:
            continue
        short = build(_truncate(panel, rows))
        a = full.iloc[:rows].to_numpy(dtype=float)
        b = short.to_numpy(dtype=float)
        finite_a, finite_b = np.isfinite(a), np.isfinite(b)
        both = finite_a & finite_b
        differ = both & ~np.isclose(a, b, rtol=0.0, atol=atol, equal_nan=True)
        findings.append({"rows": rows, "compared": int(both.sum()),
                         "mismatched": int(differ.sum()),
                         "known_only_on_full": int((finite_a & ~finite_b).sum()),
                         "known_only_on_truncated": int((finite_b & ~finite_a).sum())})
    passed = bool(findings) and all(
        f["mismatched"] == 0 and f["known_only_on_full"] == 0
        and f["known_only_on_truncated"] == 0 for f in findings)
    return {"stateless": passed, "checks": findings,
            "note": "a value the full panel knows and its own prefix does not came "
                    "from the future"}
