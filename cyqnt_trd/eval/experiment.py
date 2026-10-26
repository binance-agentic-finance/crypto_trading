"""The end-to-end run: candidates -> matrix -> construction -> comparison.

    candidates      capability nodes (production path) or Alpha101 (smoke test)
      | screen      cheap pass: six gates, frozen direction, per-split RankIC
      | admit       G0 failures out; dev |IC| below the calibrated noise floor out
      | confirm     full matrix on the survivors, including G5
      | dedupe      near-duplicates out, measured on dev only
    admitted        the factors construction is allowed to use
      | construct   combine -> size -> cap -> rebalance -> simulate
    strategy        equity curve, turnover, orders
      | compare     against buy-and-hold, equal weight, cash and the raw factor

Everything that selects is computed on **dev**. `val` and `oot` are only ever
read to report. That is the whole point: a comparison where the winner was chosen
using the period it is measured on answers nothing.

The `confirm` pass is not optional padding. Screening runs with `fast=True`, which
skips G3 (cost) and G5 (baseline novelty), and admitting on dev |IC| alone then
lets through anything correlated with a known style. On the capability catalogue
that failure is spectacular: `ema[raw]`, `sma[raw]`, `pivot_points.pp[raw]` and a
dozen others emit a **price level**, so ranking the cross-section by them ranks
coins by unit price -- BTC at 110k above DOGE at 0.2, every single day. That scores
a stable RankIC around 0.12 without containing any alpha, and G5's size baseline is
precisely the check that says so.

The candidate count reported to the matrix as `trials_seen` is the size of the pool
the sample was drawn from, not the size of the sample. Picking 8 of 101 is a
101-candidate search and the matrix's search-selection dimension has to say so.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmarks import (compare, equal_weight, run_benchmarks, single_asset_timing,
                         single_factor_naive)
from .library import alpha101_factors, deduplicate, sample_factors
from .matrix import VERDICTS, load_calibration
from .pipeline import admit, build_strategy, cards_frame, screen
from .portfolio import simulate

__all__ = ["ExperimentResult", "candidate_pool", "noise_floor", "run_comparison",
           "write_report"]

_START = "2022-04-01"


def noise_floor(primary_h: int = 3, calibration=None) -> float:
    """The dev |RankIC| a random signal reaches 95% of the time.

    Used as the admission threshold: an IC a coin flip also achieves is not
    evidence. Taken from the shipped calibration so the number moves when the
    calibration is regenerated, instead of being frozen in code.
    """
    cal = calibration if isinstance(calibration, dict) else load_calibration(calibration)
    value = (cal.get("ic_mean", {}).get("dev", {}) or {}).get("p95")
    if value is None or not np.isfinite(float(value)):
        raise ValueError("calibration has no dev ic_mean p95; regenerate it with "
                         "`python -m cyqnt_trd.eval calibrate`")
    return float(value)


def candidate_pool(panel, source: str = "capability", *, index=None,
                   rejected: dict | None = None) -> dict:
    """Build every candidate a source can offer on this panel.

    Candidates with nothing finite inside the mask are dropped here rather than
    left for `evaluate()` to raise on. Both sources produce some: capabilities that
    need data the panel lacks, Alpha101 formulas that need a market-cap or industry
    input the crypto panel has no point-in-time equivalent for.
    """
    rejected = rejected if rejected is not None else {}
    if source == "capability":
        from .capability_registry import capability_factors
        pool = capability_factors(index, panel=panel, rejected=rejected)
    elif source == "alpha101":
        pool = alpha101_factors(panel)
    else:
        raise ValueError(f"unknown source {source!r}; use 'capability' or 'alpha101'")

    out = {}
    for name, factor in pool.items():
        frame = factor(panel) if callable(factor) else factor
        if np.isfinite(frame.where(panel.mask).to_numpy(dtype=float)).any():
            out[name] = factor
        else:
            rejected[name] = "no finite values inside the eligibility mask"
    return out


@dataclass
class ExperimentResult:
    source: str
    seed: int
    pool_size: int
    sampled: list[str]
    cards: dict
    admitted: list[str]
    dropped: dict
    strategies: dict
    books: dict
    table: pd.DataFrame
    config: dict = field(default_factory=dict)
    rejected_candidates: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"{self.source}: pool {self.pool_size}, sampled {len(self.sampled)}, "
                f"admitted {len(self.admitted)}, dropped as duplicate {len(self.dropped)}")

    def oot(self) -> pd.DataFrame:
        return self.table.loc["oot"] if "oot" in self.table.index.get_level_values(0) \
            else self.table.iloc[:0]

    def to_dict(self) -> dict:
        return {
            "source": self.source, "seed": self.seed, "pool_size": self.pool_size,
            "sampled": self.sampled, "admitted": self.admitted,
            "dropped_as_duplicate": self.dropped,
            "rejected_candidates": self.rejected_candidates,
            "config": self.config,
            "cards": cards_frame(self.cards).reset_index().to_dict("records")
                     if self.cards else [],
            "comparison": self.table.reset_index().to_dict("records"),
        }


def run_comparison(panel, *, source: str = "capability", k: int | None = None,
                   seed: int = 20260917, primary_h: int = 3, cost_bps: float = 6.5,
                   entry_lag: int = 2, rebalance: int = 3, cap: float = 0.35,
                   gross: float = 1.0, benchmark_symbol: str = "BTCUSDT",
                   dedupe: bool = True, max_corr: float = 0.8,
                   admit_on_noise_floor: bool = True, confirm: bool = True,
                   reject_verdicts=("REJECT", "REJECT_DATA"), long_tilt: float = 0.5,
                   variants=("neutral", "long_biased"),
                   start=_START, splits=None, index=None,
                   calibration=None) -> ExperimentResult:
    """Run the whole chain once and return the comparison.

    `k=None` uses the entire pool. `dedupe`, `admit_on_noise_floor`, `confirm` and
    `cap` are the levers this experiment claims are doing work; each can be switched
    off to check that claim rather than assert it.
    """
    rejected: dict = {}
    pool = candidate_pool(panel, source, index=index, rejected=rejected)
    if not pool:
        raise ValueError(f"source {source!r} produced no runnable candidates")
    chosen = pool if k is None else sample_factors(pool, k, seed)

    # Honest search accounting: the pool is what was examined, not the sample.
    screen_kwargs = dict(primary_h=primary_h, trials_seen=len(pool), cost_bps=cost_bps,
                         entry_lag=entry_lag, splits=splits, calibration=calibration)
    cards = screen(chosen, panel, fast=True, **screen_kwargs)
    floor = noise_floor(primary_h, calibration) if admit_on_noise_floor else None
    cards = admit(cards, min_ic_dev=floor)
    shortlist = {n: chosen[n] for n, c in cards.items() if c.admitted}
    if not shortlist:
        raise ValueError(f"no candidate cleared the cheap screen (dev |IC| >= {floor})")

    rejected_by_gate: dict = {}
    if confirm:
        # Second pass with the full matrix so G3 and G5 actually run. Without it a
        # price-level proxy sails through on dev |IC| alone.
        cards = {**cards, **screen(shortlist, panel, fast=False, **screen_kwargs)}
        cards = admit(cards, min_ic_dev=floor,
                      require_verdict=tuple(v for v in VERDICTS if v not in reject_verdicts))
        for name in list(shortlist):
            if not cards[name].admitted:
                rejected_by_gate[name] = f"{cards[name].verdict}: {cards[name].reason}"
                shortlist.pop(name)
        if not shortlist:
            raise ValueError("no candidate survived the full matrix; "
                             f"verdicts: {rejected_by_gate}")
    admitted = shortlist

    dropped: dict = {}
    if dedupe:
        strength = {n: abs(cards[n].ic_dev) for n in admitted}
        admitted, dropped = deduplicate(admitted, panel, max_corr=max_corr,
                                        strength=strength, splits=splits)
    for card in cards.values():
        card.scorecard = None            # drop the heavy object; the fields we use are lifted

    strategies, books = {}, {}
    base = build_strategy(admitted, panel, primary_h=primary_h, rebalance=rebalance,
                          cap=cap, gross=gross, neutral=True, entry_lag=entry_lag,
                          cost_bps=cost_bps, start=start, fast=True)
    for variant in variants:
        if variant == "neutral":
            strategies[variant] = base
            books["constructed_neutral"] = base.book
            continue
        if variant != "long_biased":
            raise ValueError(f"unknown variant {variant!r}")
        # A genuinely long-biased book needs an explicit market sleeve. Merely
        # passing neutral=False does nothing: the combined score is already a
        # demeaned cross-sectional rank, so skipping the demean step leaves the
        # weights essentially unchanged -- measured, not assumed.
        blend = (base.weights.fillna(0.0) * (1.0 - long_tilt)
                 + equal_weight(panel, gross=gross, rebalance=rebalance,
                                start=start).ffill().fillna(0.0) * long_tilt)
        blend = blend.where(base.weights.notna().any(axis=1), np.nan)
        strategies[variant] = base
        books["constructed_long_biased"] = simulate(blend, panel, entry_lag=entry_lag,
                                                    cost_bps=cost_bps)

    # The raw-factor baseline is the single strongest admitted factor on dev, traded
    # unmanaged. Strongest-on-dev, not strongest overall -- picking it on oot would
    # hand the baseline the answer and make the comparison meaningless in our favour.
    best = max(admitted, key=lambda n: abs(cards[n].ic_dev))
    books["single_factor_naive"] = simulate(
        single_factor_naive(admitted[best], panel, h=primary_h, entry_lag=entry_lag,
                            rebalance=rebalance, gross=gross, splits=splits, start=start),
        panel, entry_lag=entry_lag, cost_bps=cost_bps)

    score = strategies[variants[0]].score
    books["btc_timing"] = simulate(
        single_asset_timing(score, panel, symbol=benchmark_symbol, start=start,
                            rebalance=rebalance),
        panel, entry_lag=entry_lag, cost_bps=cost_bps)
    books.update(run_benchmarks(panel, symbol=benchmark_symbol, entry_lag=entry_lag,
                                cost_bps=cost_bps, rebalance=5, gross=gross, start=start))

    config = {"source": source, "seed": seed, "k": k, "primary_h": primary_h,
              "cost_bps": cost_bps, "entry_lag": entry_lag, "rebalance": rebalance,
              "cap": cap, "gross": gross, "dedupe": dedupe, "max_corr": max_corr,
              "admit_on_noise_floor": admit_on_noise_floor, "noise_floor": floor,
              "confirm": confirm, "long_tilt": long_tilt,
              "rejected_by_gate": rejected_by_gate,
              "variants": list(variants), "start": str(start),
              "benchmark_symbol": benchmark_symbol,
              "naive_baseline_factor": best,
              "panel": panel.describe()}
    return ExperimentResult(
        source=source, seed=seed, pool_size=len(pool), sampled=list(chosen),
        cards=cards, admitted=list(admitted), dropped=dropped,
        strategies=strategies, books=books, table=compare(books, splits=splits),
        config=config, rejected_candidates=rejected)


def _md_table(frame: pd.DataFrame) -> str:
    """Markdown table without pulling in `tabulate`, which is not a declared dep."""
    frame = frame.reset_index()
    header = [str(c) for c in frame.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in row] for row in frame.to_numpy()]
    width = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
             for i in range(len(header))]
    line = lambda cells: "| " + " | ".join(c.ljust(width[i]) for i, c in enumerate(cells)) + " |"
    return "\n".join([line(header), "|" + "|".join("-" * (w + 2) for w in width) + "|",
                      *(line(r) for r in rows)])


def _fmt(table: pd.DataFrame) -> str:
    out = table.copy()
    for column in ("ann_return", "max_drawdown"):
        if column in out:
            out[column] = out[column].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}")
    for column in ("sharpe", "calmar"):
        if column in out:
            out[column] = out[column].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    if "daily_turnover" in out:
        out["daily_turnover"] = out["daily_turnover"].map(
            lambda v: "" if pd.isna(v) else f"{v:.3f}")
    return _md_table(out)


def write_report(result: ExperimentResult, out_dir: str | Path) -> dict:
    """Write `comparison.md` and `comparison.json` into `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = result.config
    lines = [
        f"# Strategy comparison -- {result.source}", "",
        f"- panel: {cfg['panel']}",
        f"- candidate pool: {result.pool_size}"
        + (f", sampled {len(result.sampled)} with seed {result.seed}"
           if cfg.get("k") else " (all used)"),
        f"- admitted: {len(result.admitted)}"
        + (f" (dev |IC| >= {cfg['noise_floor']:.4f})" if cfg.get("noise_floor") else ""),
        f"- dropped as duplicates: {len(result.dropped)}",
        f"- raw-factor baseline: `{cfg['naive_baseline_factor']}` (strongest on dev)",
        "",
        "Selection uses dev only; val and oot are reported, never consulted.", "",
        "## Comparison", "", _fmt(result.table), "",
        "## Factor cards", "",
    ]
    if result.cards:
        lines += [_md_table(cards_frame(result.cards).round(4)), ""]
    if result.dropped:
        lines += ["## Dropped as duplicate", ""]
        lines += [f"- `{k}` -- {v}" for k, v in result.dropped.items()] + [""]
    (out_dir / "comparison.md").write_text("\n".join(lines))
    (out_dir / "comparison.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str))
    return {"markdown": str(out_dir / "comparison.md"),
            "json": str(out_dir / "comparison.json")}
