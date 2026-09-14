"""Full single-factor evidence matrix on a frozen daily research contract.

Diagnostics never select a new direction, horizon, or trading rule. Every slice
and curve is exported, including unavailable evidence and unfavourable results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import DAY, _split_bounds
from .statistics import daily_cross_sectional, summarize_relation, summarize_time_mean
from .targets import build_targets


def _dates(index, bounds, split, h=0, lag=0):
    start, end = bounds[split]
    return (index >= start) & (index + (h + lag) * DAY <= end)


def _weighted_quantiles(frame, mask, quantiles):
    usable = mask & np.isfinite(frame)
    weights = usable.div(usable.sum(axis=1).replace(0, np.nan), axis=0)
    values = frame.to_numpy()[usable.to_numpy()]
    weight = weights.to_numpy()[usable.to_numpy()]
    if not len(values):
        return np.full(len(quantiles), np.nan)
    order = np.argsort(values, kind="stable")
    values, weight = values[order], weight[order]
    cumulative = np.cumsum(weight) / weight.sum()
    return values[np.minimum(np.searchsorted(cumulative, quantiles), len(values) - 1)]


def _edges(frame, mask, count):
    """Dev quantiles with total weight one per date; tied values stay together."""
    cuts = _weighted_quantiles(frame, mask, np.arange(1, count) / count)
    return np.r_[-np.inf, np.unique(cuts[np.isfinite(cuts)]), np.inf]


def _bucket(frame, edges):
    result = np.searchsorted(edges, frame.to_numpy(), side="right") - 1
    return pd.DataFrame(result, index=frame.index, columns=frame.columns).where(np.isfinite(frame))


def _relation(signal, target, mask, dates, *, h, n_bootstrap, seed):
    # Keep the contiguous split grid, including missing dates within it.
    result = summarize_relation(signal.loc[dates], target.loc[dates], mask.loc[dates],
                                block_length=max(10, 2 * h), n_bootstrap=n_bootstrap, seed=seed)
    ci = result.get("ci", {})
    rank_ci = ci.get("rank_ic", [np.nan, np.nan])
    return {**result, "corr": result.get("corr_pooled", np.nan),
            "ci_low": rank_ci[0], "ci_high": rank_ci[1]}


def _distribution(signal, mask):
    x = signal.to_numpy()[mask.to_numpy()]
    finite = x[np.isfinite(x)]
    if len(finite):
        counts, edges = np.histogram(finite, bins=50)
        sample = np.quantile(finite, np.linspace(0, 1, min(501, len(finite))))
    else:
        counts, edges, sample = np.array([]), np.array([]), np.array([])
    return {"total": int(signal.size), "eligible": int(len(x)), "finite": int(len(finite)),
            "zero": int((x == 0).sum()), "nan": int(np.isnan(x).sum()),
            "posinf": int(np.isposinf(x).sum()), "neginf": int(np.isneginf(x).sum()),
            "histogram": {"edges": edges.tolist(), "counts": counts.tolist()},
            "cdf": {"x": sample.tolist(), "y": np.linspace(0, 1, len(sample)).tolist()},
            "scope": "all eligible cells; zeros retained; CDF denominator is finite observations"}


def _extremes(signal, raw, panel, bounds, h, lag):
    dev = _dates(panel.index, bounds, "dev", h, lag)
    val = _dates(panel.index, bounds, "val", h, lag)
    finite = panel.mask & np.isfinite(signal)
    values = signal.where(finite).loc[dev].to_numpy().ravel()
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.std(values) <= 0:
        return {"correct": [], "wrong": [], "reason": "insufficient dev signal dispersion"}
    dev_mask = finite.copy()
    dev_mask.loc[~dev] = False
    center = float(signal.where(dev_mask).mean(axis=1).mean())
    scale = float(np.sqrt(((signal - center) ** 2).where(dev_mask).mean(axis=1).mean()))
    score = (signal - center) / scale
    abs_score = score.abs()
    threshold = float(_weighted_quantiles(abs_score, dev_mask, [.995])[0])
    target_threshold = float(_weighted_quantiles(raw.abs(), dev_mask & np.isfinite(raw), [.99])[0])
    eligible = finite & np.isfinite(raw) & (abs_score > threshold) & (raw.abs() > target_threshold)
    eligible.loc[~val] = False
    rows = []
    for t, a in np.argwhere(eligible.to_numpy()):
        rows.append((float(abs_score.iloc[t, a] * abs(raw.iloc[t, a])), int(t), int(a)))
    rows.sort(reverse=True)
    output = {"correct": [], "wrong": []}
    selected = []
    for _, t, a in rows:
        key = "correct" if score.iloc[t, a] * raw.iloc[t, a] > 0 else "wrong"
        # One market episode contributes at most one case, across all assets.
        if len(output[key]) >= 6 or any(abs(t - old) <= h + lag for old in selected):
            continue
        selected.append(t)
        lo, hi = max(0, t - 10), min(len(panel.index), t + lag + h + 11)
        output[key].append({"symbol": panel.symbols[a], "bar_time": panel.index[t].isoformat(),
                            "signal_time": (panel.index[t] + DAY).isoformat(),
                            "score": float(score.iloc[t, a]), "target": float(raw.iloc[t, a]),
                            "times": [(stamp + DAY).isoformat() for stamp in panel.index[lo:hi]],
                            "prices": panel.close.iloc[lo:hi, a].tolist(),
                            "signal": score.iloc[lo:hi, a].tolist(),
                            "entry_time": panel.index[t + lag].isoformat(),
                            "exit_time": panel.index[t + lag + h].isoformat()})
    all_extreme = finite & (abs_score > threshold) & np.isfinite(raw)
    all_extreme.loc[~val] = False
    prediction = (score * raw).where(all_extreme)
    n = int(all_extreme.to_numpy().sum())
    return {**output, "signal_threshold": float(threshold), "target_threshold": target_threshold,
            "signal_center_dev": center, "signal_scale_dev": scale,
            "all_extreme_signals": n,
            "all_extreme_mean_raw_return": float(raw.where(all_extreme).mean(axis=1).mean()),
            "all_extreme_directional_accuracy": float((prediction > 0).to_numpy().sum() / n) if n else np.nan,
            "reason": "val post-hoc cases selected using future outcomes; not an estimate of live win rate; thresholds frozen on dev; no padding to six cases"}


def build_diagnostics(signal, panel, *, primary_h, entry_lag, splits, sign, metrics, periods, cost_bps,
                      gates, trials_seen=1, n_bootstrap=100, seed=20260914,
                      ftr_horizons=tuple(range(1, 61)), benchmark_symbol="BTCUSDT", spread=None):
    """Return JSON-ready evidence for every first-stage matrix dimension.

    ``spread`` is optional decision-time full log(ask/bid), aligned to the panel.
    Missing quotes never become a spread proxy or an implied capacity estimate.
    """
    bounds = _split_bounds(splits)
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 0:
        raise ValueError("n_bootstrap must be a nonnegative integer")
    horizons = tuple(sorted(set(ftr_horizons)))
    if not horizons or any(isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in horizons):
        raise ValueError("ftr_horizons must contain positive integers")
    signed = signal * sign
    observed = panel.mask & np.isfinite(signed)
    dev = _dates(panel.index, bounds, "dev", primary_h, entry_lag)
    val_grid = _dates(panel.index, bounds, "val")
    dev_mask = observed.copy()
    dev_mask.loc[~dev] = False
    primary = build_targets(panel, primary_h, entry_lag, benchmark_symbol=benchmark_symbol)
    target_rows, bins = [], {}
    signal_edges = _edges(signed, dev_mask, 25)
    bucket = _bucket(signed, signal_edges)
    common = primary["common_mask"] & observed
    for name, target in primary["targets"].items():
        for split in ("dev", "val", "oot"):
            dates = _dates(panel.index, bounds, split)
            valid = _dates(panel.index, bounds, split, primary_h, entry_lag)
            mask = common.copy()
            mask.loc[~valid] = False
            summary = _relation(signed, target, mask, dates, h=primary_h,
                                n_bootstrap=n_bootstrap, seed=seed)
            target_rows.append({"target": name, "split": split, "sample": "four_target_common", **summary})
            # Keep useful raw measurements when another target cannot be built.
            own = observed & np.isfinite(target)
            own.loc[~valid] = False
            if not own.equals(mask):
                target_rows.append({"target": name, "split": split, "sample": "target_available",
                                    **_relation(signed, target, own, dates, h=primary_h,
                                                n_bootstrap=0, seed=seed)})
        rows = []
        for b in range(len(signal_edges) - 1):
            mask = common & (bucket == b)
            mask.loc[~_dates(panel.index, bounds, "val", primary_h, entry_lag)] = False
            per_date = target.where(mask).mean(axis=1).loc[val_grid]
            result = summarize_time_mean(per_date, block_length=max(10, 2 * primary_h),
                                         n_bootstrap=n_bootstrap, seed=seed)
            rows.append({"bin": b + 1, "n_pairs": int(mask.to_numpy().sum()), **result})
        bins[name] = {"rows": rows, "unit": primary["metadata"]["units"][name],
                      "status": "MEASURED" if any(r["n_dates"] for r in rows) else "NOT_EVALUATED",
                      "reason": "dev weighted quantiles; ties stay together; equal date means within each bin, dates can differ between bins; four-target common cells; descriptive, not portfolio returns",
                      "edges": signal_edges.tolist()}

    past = np.log(panel.close / panel.close.shift(1))
    past_edges = _edges(past, dev_mask & np.isfinite(past), 20)
    past_bucket = _bucket(past, past_edges)
    raw = primary["targets"]["raw_rtf"]
    past_rows = []
    valid_val = _dates(panel.index, bounds, "val", primary_h, entry_lag)
    for b in range(len(past_edges) - 1):
        mask = observed & (past_bucket == b) & np.isfinite(raw)
        mask.loc[~valid_val] = False
        row = _relation(signed, raw, mask, val_grid, h=primary_h, n_bootstrap=n_bootstrap, seed=seed)
        ci = row["ci"].get("corr_pooled", [np.nan, np.nan])
        past_rows.append({**row, "label": str(b + 1), "ci_low": ci[0], "ci_high": ci[1]})

    # Freeze one common set of cells for the entire horizon curve. Record the
    # loss of sample relative to every available-horizon diagnostic explicitly.
    target_cache = {h: build_targets(panel, h, entry_lag, benchmark_symbol=benchmark_symbol) for h in horizons}
    curve_mask = observed.copy()
    curve_mask.loc[~_dates(panel.index, bounds, "val", max(horizons), entry_lag)] = False
    residual_available = primary["availability"]["res_rtf"]["status"] == "AVAILABLE"
    curve_targets = ("raw_rtf", "res_rtf") if residual_available else ("raw_rtf",)
    for item in target_cache.values():
        for name in curve_targets:
            curve_mask &= np.isfinite(item["targets"][name])
    ftr = []
    for h, item in target_cache.items():
        for name in curve_targets:
            own = observed & np.isfinite(item["targets"][name])
            own.loc[~_dates(panel.index, bounds, "val", h, entry_lag)] = False
            ftr.append({"h": h, "target": name, "available_cells": int(own.to_numpy().sum()),
                        "common_cells": int(curve_mask.to_numpy().sum()),
                        **_relation(signed, item["targets"][name], curve_mask, val_grid, h=h,
                                    n_bootstrap=n_bootstrap, seed=seed)})
    delay_cache = {lag: build_targets(panel, primary_h, lag, benchmark_symbol=benchmark_symbol)
                   for lag in sorted({1, entry_lag, entry_lag + 1, entry_lag + 2, entry_lag + 5})}
    delay_mask = observed.copy()
    delay_mask.loc[~_dates(panel.index, bounds, "val", primary_h, max(delay_cache))] = False
    for item in delay_cache.values():
        for name in curve_targets:
            delay_mask &= np.isfinite(item["targets"][name])
    delay = [{"lag": lag, "target": name,
              **_relation(signed, item["targets"][name], delay_mask, val_grid, h=primary_h,
                          n_bootstrap=n_bootstrap, seed=seed)}
             for lag, item in delay_cache.items() for name in curve_targets]

    stability = []
    # Calendar slices are predeclared; market-state boundaries are fitted on dev.
    calendar = {"year": panel.index.strftime("%Y"), "quarter": panel.index.strftime("%Y-Q") + ((panel.index.month - 1) // 3 + 1).astype(str),
                "weekday": panel.index.day_name()}
    for dimension, labels in calendar.items():
        for label in sorted(set(labels[val_grid])):
            mask = observed & np.isfinite(raw)
            mask.loc[~(valid_val & (labels == label))] = False
            stability.append({"dimension": dimension, "label": str(label),
                              **_relation(signed, raw, mask, val_grid, h=primary_h,
                                          n_bootstrap=n_bootstrap, seed=seed)})
    vol = primary["estimates"]["sigma_raw"]
    vol_edges = _edges(vol, dev_mask & np.isfinite(vol), 3)
    vol_bucket = _bucket(vol, vol_edges)
    for b in range(len(vol_edges) - 1):
        mask = observed & (vol_bucket == b) & np.isfinite(raw)
        mask.loc[~valid_val] = False
        stability.append({"dimension": "ex_ante_volatility", "label": str(b + 1),
                          **_relation(signed, raw, mask, val_grid, h=primary_h,
                                      n_bootstrap=n_bootstrap, seed=seed)})
    for symbol in panel.symbols:
        mask = observed.copy()
        mask.loc[:, symbol] = False
        mask.loc[~valid_val] = False
        stability.append({"dimension": "leave_one_asset_out", "label": symbol,
                          **_relation(signed, raw, mask, val_grid, h=primary_h,
                                      n_bootstrap=n_bootstrap, seed=seed)})

    cost_rows = []
    for split in ("dev", "val", "oot"):
        cycles = periods.loc[(periods.h == primary_h) & (periods.split == split) & periods.label_valid]
        for multiple in (0, .5, 1, 1.5, 2, 3):
            cost = multiple * cost_bps
            net = (cycles.gross - cycles.funding) * 1e4 - cost * cycles.turnover
            complete = np.isfinite(net).all() and len(net) > 0
            cost_rows.append({"split": split, "multiplier": multiple, "cost_bps": cost,
                              "net_bp": float(net.mean()) if complete else np.nan,
                              "net_available_bp": float(net.mean()),
                              "n_complete": int(np.isfinite(net).sum()), "n_periods": len(net)})

    ic_vs_n = {"status": "NOT_EVALUATED", "rows": [],
               "reason": "No decision-time full bid/ask spread supplied; daily high-low range is not a spread proxy"}
    if spread is not None:
        if not isinstance(spread, pd.DataFrame) or not spread.index.equals(panel.index) or list(spread.columns) != panel.symbols:
            raise ValueError("spread must align with the panel and contain decision-time log(ask/bid)")
        if (spread <= 0).any().any() or np.isinf(spread.to_numpy()).any():
            raise ValueError("spread must be positive or missing")
        daily = build_targets(panel, 1, entry_lag, benchmark_symbol=benchmark_symbol)
        tradability = daily["estimates"]["sigma_raw"] / spread
        groups = _bucket(tradability, _edges(tradability, dev_mask & np.isfinite(tradability), 3))
        rows = []
        for h in horizons:
            item = target_cache[h]
            n = item["estimates"]["sigma_raw"] / spread
            for group in sorted(groups.stack().unique()):
                mask = curve_mask & (groups == group) & np.isfinite(n)
                relation = _relation(signed, item["targets"]["raw_rtf"], mask, val_grid, h=h,
                                     n_bootstrap=n_bootstrap, seed=seed)
                daily = daily_cross_sectional(signed.loc[val_grid], item["targets"]["raw_rtf"].loc[val_grid], mask.loc[val_grid])
                n_mask = mask.copy()
                n_mask.loc[daily.index[daily.rank_ic.isna()]] = False
                quantiles = _weighted_quantiles(n, n_mask, [.25, .5, .75])
                rows.append({"group": str(int(group) + 1), "h": h, "lag": entry_lag,
                             "n_median": quantiles[1], "n_q25": quantiles[0], "n_q75": quantiles[2], **relation})
        ic_vs_n = {"status": "MEASURED" if any(np.isfinite(r["rank_ic"]) for r in rows) else "INSUFFICIENT", "rows": rows,
                   "reason": "N=same-horizon ex-ante volatility/full log spread; no sqrt scaling; not profitability or capacity"}

    dimensions = [
        ("data_integrity", "G0_data", "finite values, coverage, sample sizes, daily time contract"),
        ("information", "G1_information", "signed RankIC, HAC and calibrated noise reference"),
        ("structure", "G2_structure", "frozen direction, bins, primary-sign horizon comparison"),
        ("net_economics", "G3_cost", "complete held-position cash flows and cost sensitivity"),
        ("robustness", "G4_robustness", "split/year consistency plus all predeclared slices"),
        ("baseline_novelty", "G5_incremental", "public baseline correlation and unsaturated residual projection"),
    ]
    matrix = [{"dimension": dim, "status": gates[key].status if key in gates else "NOT_EVALUATED", "evidence": evidence}
              for dim, key, evidence in dimensions]
    matrix += [
        {"dimension": "four_target_comparison", "status": "MEASURED" if all(r["n_dates"] > 0 for r in target_rows if r["split"] == "val" and r["sample"] == "four_target_common") else "NOT_EVALUATED", "evidence": "same entry/exit and common cells; individual coverage and unavailable reasons retained"},
        {"dimension": "sampling_uncertainty", "status": "MEASURED" if any(np.isfinite(r["ci_low"]) for r in target_rows) else "NOT_EVALUATED", "evidence": "moving time blocks keep all assets together; pointwise 95% intervals, not simultaneous confidence bands"},
        {"dimension": "horizon_and_delay", "status": "MEASURED" if any(np.isfinite(r["rank_ic"]) for r in ftr + delay) else "NOT_EVALUATED", "evidence": "fixed primary sign and original signal; common-sample horizon and delay curves"},
        {"dimension": "tradability", "status": ic_vs_n["status"], "evidence": ic_vs_n["reason"]},
        {"dimension": "capacity", "status": "NOT_EVALUATED", "evidence": "requires size-dependent fills, order-book depth and market impact; no such model supplied"},
        {"dimension": "search_selection", "status": "UNADJUSTED" if trials_seen > 1 else "SINGLE_CANDIDATE_DECLARED", "evidence": f"{trials_seen} examined candidates declared; no DSR/PBO without a complete trial ledger"},
        {"dimension": "sealed_holdout", "status": "NOT_EVALUATED", "evidence": "split name oot does not establish that researchers had not inspected it"},
        {"dimension": "strategy_combination", "status": "NOT_EVALUATED", "evidence": "second-stage factor weights, portfolio risk and combination PnL are outside this matrix"},
    ]
    return {"version": "factor-eval.diagnostics/v1", "matrix": matrix,
            "distribution": _distribution(signal, panel.mask),
            "target_comparison": {"rows": target_rows, "metadata": primary["metadata"],
                                  "coverage": primary["coverage"], "availability": primary["availability"]},
            "bins": bins,
            "past_return_slices": {"rows": past_rows, "edges": past_edges.tolist(), "reason": "val; dev-fitted past-return slices; corr is date-weighted pooled Pearson, not cross-sectional IC"},
            "ftr": {"rows": ftr, "common_cells": int(curve_mask.to_numpy().sum()), "reason": "val common sample across all horizons and available raw/res targets; primary direction frozen; pointwise intervals"},
            "delay": {"rows": delay, "holding_h": primary_h, "reason": "fixed holding length and unchanged signal; lag=1 is next open after the signal bar closes"},
            "stability": {"rows": stability, "volatility_edges": vol_edges.tolist(), "reason": "val predeclared slices and leave-one-asset-out; narrow slices may lack CS IC"},
            "ic_vs_n": ic_vs_n,
            "extreme_cases": _extremes(signed, raw, panel, bounds, primary_h, entry_lag),
            "cost_stress": {"rows": cost_rows, "reason": "fixed-position cost sensitivity; partial net is diagnostic only; no market-impact simulation"},
            "config": {"n_bootstrap": n_bootstrap, "seed": seed, "confidence": .95,
                       "block_length": "max(10,2*h) dates", "horizons": list(horizons), "benchmark_symbol": benchmark_symbol,
                       "direction": sign, "scope": "single-factor diagnostics; no post-hoc rule selection"}}
