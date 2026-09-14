"""Reproducible descriptive figures for the repaired Alpha101 experiment.

Run: python diagnostics.py --tag current_normalized
Only dev h=3 metrics choose candidates, including a >=50% active-cycle gate.
No OOT statistic selects a candidate,
horizon, direction, or bin boundary. Full-sample figures are descriptive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import sys as _sys
_EVAL_ROOT = str(Path(__file__).resolve().parents[1])  # eval/ holds the shared measurement core
if _EVAL_ROOT not in _sys.path: _sys.path.insert(0, _EVAL_ROOT)
from factor_eval.engine import _spearman_rows, _split_bounds


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SPLITS = {"dev": ("2022-04-01", "2024-06-30"),
          "val": ("2024-07-01", "2025-09-30"),
          "oot": ("2025-10-01", "2026-09-09")}
COLORS = {"dev": "#2563eb", "val": "#d97706", "oot": "#16856b"}
TARGET_NAMES = {"raw": "Raw log return", "normed_raw": "Vol-normalized raw return",
                "res": "BTC-beta residual log return", "normed_res": "Vol-normalized residual"}
SEED = 20260910
H = 3
LAG = 2


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [clean(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, Path)):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def dump(path, obj):
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False))


def ascii_label(value):
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


def finite_mean(x, strict=False):
    a = np.asarray(x, float)
    if not len(a) or (strict and not np.isfinite(a).all()):
        return np.nan
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else np.nan


def weighted_quantile(values, weights, probabilities):
    x, w = np.asarray(values, float).ravel(), np.asarray(weights, float).ravel()
    good = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not good.any():
        return np.full(len(probabilities), np.nan)
    x, w = x[good], w[good]
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    cdf = np.cumsum(w) / w.sum()
    return x[np.minimum(np.searchsorted(cdf, probabilities), len(x) - 1)]


def frozen_bins(values, common, dev_dates, weights):
    valid_dev = common & dev_dates[:, None]
    cuts = weighted_quantile(np.where(valid_dev, values, np.nan), weights, [0.2, 0.4, 0.6, 0.8])
    dev_values = values[valid_dev]
    if not len(dev_values):
        return np.full(values.shape, -1), [], 0
    cuts = np.unique(cuts[np.isfinite(cuts)])
    cuts = cuts[cuts < np.nanmax(dev_values)]
    bins = np.searchsorted(cuts, values, side="left").astype(int)
    bins[~common] = -1
    return bins, cuts.tolist(), len(cuts) + 1


def bootstrap_indices(length, count, block, seed):
    if length < 1:
        return np.empty((count, 0), dtype=int)
    rng = np.random.default_rng(seed)
    block = min(block, length)
    starts = rng.integers(0, length - block + 1, size=(count, int(np.ceil(length / block))))
    return (starts[:, :, None] + np.arange(block)).reshape(count, -1)[:, :length]


def save(fig, directory, name):
    fig.savefig(directory / name, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return name


def setup_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.16,
                         "figure.titlesize": 15, "savefig.dpi": 165})


def four_targets(raw, index, columns):
    opens = raw["open"].reindex(index=index, columns=columns)
    close = raw["close"].reindex(index=index, columns=columns)
    daily = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    forward = np.log(opens.shift(-(LAG + H)) / opens.shift(-LAG)).replace([np.inf, -np.inf], np.nan)
    variance20 = daily.rolling(20, min_periods=20).var(ddof=1).shift(1)
    sigma = np.sqrt(variance20 * H)
    if "BTCUSDT" not in columns:
        empty = forward * np.nan
        return {"raw": forward, "normed_raw": forward / sigma.where(sigma > 1e-12),
                "res": empty, "normed_res": empty}, {"status": "BTC benchmark missing"}
    btc = daily["BTCUSDT"]
    beta = daily.rolling(90, min_periods=90).cov(btc).shift(1).div(
        btc.rolling(90, min_periods=90).var(ddof=1).shift(1), axis=0)
    beta["BTCUSDT"] = beta["BTCUSDT"].where(beta["BTCUSDT"].isna(), 1.0)
    btc_forward = forward["BTCUSDT"]
    residual = forward - beta.mul(btc_forward, axis=0)
    cov20 = daily.rolling(20, min_periods=20).cov(btc).shift(1)
    btc_variance20 = btc.rolling(20, min_periods=20).var(ddof=1).shift(1)
    # Apply beta known before the signal bar to the same prior 20-day window.
    residual_var = variance20 + beta.pow(2).mul(btc_variance20, axis=0) - 2 * beta * cov20
    residual_var["BTCUSDT"] = 0.0
    residual_sigma = np.sqrt(residual_var.clip(lower=0) * H)
    targets = {"raw": forward, "normed_raw": forward / sigma.where(sigma > 1e-12),
               "res": residual, "normed_res": residual / residual_sigma.where(residual_sigma > 1e-12)}
    return targets, {
        "status": "computed", "h": H, "entry_lag": LAG,
        "raw": "log(open[t+2+3]/open[t+2]); diagnostic, not portfolio PnL",
        "sigma_raw": "sqrt(3) * sd(log daily close returns over prior 20 bars), shifted 1 bar",
        "beta": "cov(asset,BTC)/var(BTC), full prior 90 daily returns, shifted 1 bar",
        "residual": "raw_forward - beta_asof * BTC_forward on identical entry/exit timestamps",
        "sigma_residual": "sqrt(3 * (prior20 var(asset) + beta_asof^2 var(BTC) - 2 beta_asof cov(asset,BTC)))",
        "zero_volatility": "undefined normalized targets stay NA; BTC self-residual volatility is zero",
        "common_sample": "all four targets, signal and eligibility simultaneously valid; at least 5 assets per date",
    }


def distribution_figure(signal, mask, valid_dates, out, leader):
    eligible = mask.to_numpy(bool) & valid_dates[:, None]
    values = signal.to_numpy(float)
    finite = eligible & np.isfinite(values)
    n = finite.sum(axis=1)
    weights = np.divide(finite, np.maximum(n, 1)[:, None])
    x, w = values[finite], weights[finite]
    summary = {"eligible_cells": int(eligible.sum()), "finite_cells": int(finite.sum()),
               "zero_cells": int((eligible & (values == 0)).sum()),
               "nan_cells": int((eligible & np.isnan(values)).sum()),
               "infinite_cells": int((eligible & np.isinf(values)).sum()),
               "dates_with_finite_signal": int((n > 0).sum()),
               "sign": "dev-frozen orientation", "weight": "equal date, equal finite asset within date"}
    if not len(x):
        return {**summary, "status": "N/A: no finite signal"}
    scale = float(weighted_quantile(np.abs(x), w, [0.5])[0])
    scale = max(scale, float(np.max(np.abs(x))) * 1e-6, 1e-12)
    lo, hi = np.arcsinh(x.min() / scale), np.arcsinh(x.max() / scale)
    edges = np.linspace(x.min() - 0.5, x.max() + 0.5, 3) if lo == hi else scale * np.sinh(np.linspace(lo, hi, 55))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    ax = axes[0]
    ax.hist(x, bins=edges, weights=w / w.sum(), color="#477fb8", alpha=0.75)
    ax.set_xscale("symlog", linthresh=scale)
    ax.set_xlabel("Signed factor value (symlog; all finite values retained)")
    ax.set_ylabel("Weighted probability per bin")
    twin = ax.twinx()
    order = np.argsort(x)
    twin.plot(x[order], np.cumsum(w[order]) / w.sum(), color="#b45309", lw=1.6)
    twin.set_ylabel("CDF"); twin.set_ylim(0, 1.02); twin.grid(False)
    ax.set_title("Signal distribution and CDF")
    expected = eligible.sum(axis=1)
    cover = np.divide(n, expected, out=np.full(len(n), np.nan), where=expected > 0)
    monthly = pd.Series(cover, index=signal.index).resample("MS").mean()
    axes[1].plot(monthly.index, monthly.values, color="#16856b", lw=1.8)
    axes[1].set_ylim(-0.03, 1.03); axes[1].set_ylabel("Mean daily finite / eligible")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].set_title("Coverage through time")
    axes[1].text(0.02, 0.05,
                 f"Eligible cells: {summary['eligible_cells']:,}\nFinite: {summary['finite_cells']:,}  |  Zero: {summary['zero_cells']:,}\n"
                 f"NaN: {summary['nan_cells']:,}  |  Inf: {summary['infinite_cells']:,}",
                 transform=axes[1].transAxes, fontsize=9,
                 bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#dddddd"})
    fig.suptitle(f"{leader} | full-sample description | h=3, entry lag=2")
    summary.update(status="ok", figure=save(fig, out, "01_distribution.png"))
    return summary


def binplot_figure(score, targets, common, dev_dates, weights, boot, out, leader):
    bins, cuts, count = frozen_bins(score, common, dev_dates, weights)
    if count == 0:
        return {"status": "N/A: no common dev sample"}
    numerator = np.zeros((len(score), count, len(targets)))
    denominator = np.zeros((len(score), count))
    rows = []
    for b in range(count):
        chosen = bins == b
        w = weights * chosen
        denominator[:, b] = w.sum(axis=1)
        rows.append({"bin": b + 1, "raw_rows": int(chosen.sum()),
                     "dates": int(chosen.any(axis=1).sum()), "weight": float(w.sum())})
        for k, arr in enumerate(targets.values()):
            numerator[:, b, k] = np.where(chosen, w * arr, 0.0).sum(axis=1)
    means = np.divide(numerator.sum(axis=0), denominator.sum(axis=0)[:, None],
                      out=np.full((count, len(targets)), np.nan), where=denominator.sum(axis=0)[:, None] > 0)
    draws = []
    for sample in boot:
        num, den = numerator[sample].sum(axis=0), denominator[sample].sum(axis=0)
        draws.append(np.divide(num, den[:, None], out=np.full_like(num, np.nan), where=den[:, None] > 0))
    draws = np.asarray(draws)
    ci = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    spreads = {name: {"mean": means[-1, k] - means[0, k],
                      "ci95": np.nanpercentile(draws[:, -1, k] - draws[:, 0, k], [2.5, 97.5]).tolist()}
               for k, name in enumerate(targets)}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), constrained_layout=True)
    for k, (name, arr) in enumerate(targets.items()):
        scale = 1e4 if name in ("raw", "res") else 1.0
        ax = axes.flat[k]; x = np.arange(1, count + 1)
        ax.axhline(0, color="#666666", lw=0.8)
        ax.plot(x, means[:, k] * scale, "o-", color="#2563eb", lw=1.5)
        ax.vlines(x, ci[0, :, k] * scale, ci[1, :, k] * scale, color="#2563eb", lw=2)
        ax.set_xticks(x); ax.set_xlabel("Signal-strength bin (dev cutpoints)")
        ax.set_title(TARGET_NAMES[name]); ax.set_ylabel("Mean forward log return (bp)" if scale == 1e4 else "Mean return / prior volatility")
        for b in range(count):
            rows[b][name] = {"mean": means[b, k], "ci95": ci[:, b, k].tolist()}
    fig.suptitle(f"{leader} | full-sample descriptive bins; 95% time-block intervals")
    fig.supxlabel("Equal date / equal asset weights; 200 resamples of 21-day blocks. Not independent validation.")
    return {"status": "ok", "figure": save(fig, out, "02_four_target_bins.png"),
            "requested_bins": 5, "effective_bins": count, "dev_rank_score_cutpoints": cuts,
            "ties": "equal signal scores are not randomly split; duplicate boundaries merge",
            "top_minus_bottom": spreads,
            "sample_rows": int(common.sum()), "sample_dates": int(common.any(axis=1).sum()), "bins": rows}


def moment_statistics(moment):
    # Final axis: W, W*x, W*y, W*x*x, W*y*y, W*x*y.
    w = moment[..., 0]
    mean_x = np.divide(moment[..., 1], w, out=np.full_like(w, np.nan), where=w > 0)
    mean_y = np.divide(moment[..., 2], w, out=np.full_like(w, np.nan), where=w > 0)
    exx = np.divide(moment[..., 3], w, out=np.full_like(w, np.nan), where=w > 0)
    eyy = np.divide(moment[..., 4], w, out=np.full_like(w, np.nan), where=w > 0)
    exy = np.divide(moment[..., 5], w, out=np.full_like(w, np.nan), where=w > 0)
    var_x, var_y, cov = exx - mean_x**2, eyy - mean_y**2, exy - mean_x * mean_y
    scale = np.sqrt(np.maximum(var_x * var_y, 0))
    corr = np.divide(cov, scale, out=np.full_like(cov, np.nan), where=(var_x > 1e-15) & (var_y > 1e-15))
    slope = np.divide(cov, var_x, out=np.full_like(cov, np.nan), where=var_x > 1e-15)
    return corr, slope


def conditional_figure(score, past, targets, common, dev_dates, weights, boot, out, leader):
    complete = common & np.isfinite(past)
    n = complete.sum(axis=1)
    w_base = np.divide(complete, np.maximum(n, 1)[:, None])
    bins, cuts, count = frozen_bins(past, complete, dev_dates, w_base)
    if count == 0:
        return {"status": "N/A: no common dev sample"}
    xmean = np.where(complete, score, 0).sum(axis=1) / np.maximum(n, 1)
    x = score - xmean[:, None]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    records = []
    for name, color in (("raw", "#2563eb"), ("res", "#d97706")):
        y = targets[name]
        ymean = np.where(complete, y, 0).sum(axis=1) / np.maximum(n, 1)
        y = y - ymean[:, None]
        moments = np.zeros((len(score), count, 6))
        for b in range(count):
            chosen = bins == b
            w = w_base * chosen
            for j, field in enumerate((np.ones_like(x), x, y, x * x, y * y, x * y)):
                moments[:, b, j] = np.where(chosen, w * field, 0.0).sum(axis=1)
        corr, slope = moment_statistics(moments.sum(axis=0))
        bc, bs = [], []
        for sample in boot:
            a, b = moment_statistics(moments[sample].sum(axis=0)); bc.append(a); bs.append(b)
        cic = np.nanpercentile(bc, [2.5, 97.5], axis=0)
        cis = np.nanpercentile(bs, [2.5, 97.5], axis=0)
        pos = np.arange(1, count + 1) + (-0.035 if name == "raw" else 0.035)
        for ax, mean, ci, multiplier in ((axes[0], corr, cic, 1), (axes[1], slope, cis, 1e4)):
            ax.plot(pos, mean * multiplier, "o-", color=color, label=name)
            ax.vlines(pos, ci[0] * multiplier, ci[1] * multiplier, color=color, lw=1.5)
        for b in range(count):
            chosen = bins == b
            records.append({"target": name, "past_return_bin": b + 1,
                            "n_rows": int(chosen.sum()), "n_dates": int(chosen.any(axis=1).sum()),
                            "weight": float(w_base[chosen].sum()),
                            "past_return_mean_bp": float(np.average(past[chosen], weights=w_base[chosen]) * 1e4) if chosen.any() else np.nan,
                            "pooled_centered_pearson": corr[b], "pearson_ci95": cic[:, b].tolist(),
                            "slope_bp_per_rank_score": slope[b] * 1e4, "slope_ci95_bp": (cis[:, b] * 1e4).tolist()})
    for ax in axes:
        ax.axhline(0, color="#666666", lw=0.8); ax.set_xticks(np.arange(1, count + 1))
        ax.set_xlabel("Past one-day return bin (dev cutpoints)"); ax.legend()
    axes[0].set_ylabel("Pooled Pearson correlation"); axes[0].set_title("Correlation after within-date centering")
    axes[1].set_ylabel("Forward log bp / rank-score unit"); axes[1].set_title("Conditional slope after within-date centering")
    fig.suptitle(f"{leader} | full-sample conditional description, h=3")
    fig.supxlabel("Not per-date two-asset IC. 95% intervals: 200 resamples, 21-day blocks; not a causal estimate.")
    return {"status": "ok", "figure": save(fig, out, "03_past_return_slices.png"),
            "dev_past_return_cutpoints": cuts, "effective_bins": count,
            "centering": "rank score and target demeaned within each date across common assets before slicing",
            "weights": "1/common_asset_count per date, then pooled within each frozen past-return bucket",
            "rows": records}


def delay_and_horizon_figure(leader, sign, signal, raw, mask, metrics, bounds, out):
    index = signal.index
    log_targets = {lag: np.log(raw["open"].shift(-(lag + H)) / raw["open"].shift(-lag)).to_numpy()
                   for lag in (1, 2, 3, 6)}
    values = signal.to_numpy(float)
    common = mask.to_numpy(bool) & np.isfinite(values)
    for target in log_targets.values():
        common &= np.isfinite(target)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    records = []
    for split, (start, end) in bounds.items():
        part = metrics[(metrics.factor == leader) & (metrics.split == split)].sort_values("h")
        axes[0].plot(part.h, part.ic_mean, "o-", label=split, color=COLORS.get(split))
        common_dates = np.asarray((index >= start) & (index <= end) & (index + pd.Timedelta(days=6 + H) <= end))
        points = []
        for lag, target in log_targets.items():
            ic, n = _spearman_rows(values, target, common, 5)
            ic[~common_dates] = np.nan
            mean = finite_mean(ic * sign)
            points.append(mean)
            records.append({"split": split, "entry_lag": lag, "additional_wait_days": lag - 1,
                            "h": H, "sign_frozen_from_h3_lag2_dev": sign,
                            "ic_mean": mean, "n_ic": int(np.isfinite(ic).sum()),
                            "common_assets_mean": finite_mean(n[common_dates])})
        axes[1].plot([0, 1, 2, 5], points, "o-", label=split, color=COLORS.get(split))
    for ax in axes:
        ax.axhline(0, color="#666666", lw=0.8); ax.set_ylabel("Mean per-date RankIC"); ax.legend()
    axes[0].set_xticks([1, 3, 5]); axes[0].set_xlabel("Holding horizon (days)")
    axes[0].set_title("Horizon scan: each h has its own dev sign")
    axes[1].set_xticks([0, 1, 2, 5]); axes[1].set_xlabel("Additional wait after signal-bar close (days)")
    axes[1].set_title("Fixed H=3; frozen signal and direction")
    fig.suptitle(f"{leader} | horizon exploration and actual execution delay")
    return {"status": "ok", "figure": save(fig, out, "04_horizon_and_delay.png"),
            "horizons": metrics[(metrics.factor == leader)][["h", "split", "sign", "ic_mean", "n_ic"]].to_dict("records"),
            "delay_rows": records,
            "delay_common_sample": "same asset-date pairs with finite targets for all four lags; purge each split for the longest delay",
            "boundary_note": "lag1 is the adjacent-bar boundary, not one day's latency; lag2 waits one complete extra day"}


def heatmap(ax, values, labels, years, title, fmt):
    finite = np.abs(values[np.isfinite(values)])
    bound = max(float(finite.max()) if len(finite) else 1, 1e-6)
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#e5e7eb")
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(len(years)), years); ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title); ax.grid(False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            ax.text(j, i, format(v, fmt) if np.isfinite(v) else "NA", ha="center", va="center",
                    color="white" if np.isfinite(v) and abs(v) > 0.62 * bound else "#111827", fontsize=10)
    return image


def annual_figure(candidates, ic, periods, out):
    select_ic = ic[(ic.h == H) & ic.factor.isin(candidates) & ic.label_valid].copy()
    select_periods = periods[(periods.h == H) & periods.factor.isin(candidates) & periods.label_valid].copy()
    for frame in (select_ic, select_periods):
        frame["year"] = frame.signal_time.dt.year
    years = sorted(select_ic.year.unique())
    if not years:
        return {"status": "N/A: no annual data"}
    a = np.full((len(candidates), len(years)), np.nan); b = a.copy(); records = []
    for i, factor in enumerate(candidates):
        for j, year in enumerate(years):
            x = select_ic[(select_ic.factor == factor) & (select_ic.year == year)
                          & (select_ic.exit_time.dt.year == year)]
            y = select_periods[(select_periods.factor == factor) & (select_periods.year == year)
                              & (select_periods.exit_time.dt.year == year)]
            a[i, j] = finite_mean(x.ic)
            b[i, j] = finite_mean(y.net_bp, strict=True)
            records.append({"factor": factor, "year": int(year), "ic_mean": a[i, j],
                            "n_ic": int(np.isfinite(x.ic).sum()), "net_bp": b[i, j],
                            "n_periods": len(y), "n_known_net_periods": int(np.isfinite(y.net_bp).sum()),
                            "net_available_bp": finite_mean(y.net_bp)})
    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.6, len(candidates) * 0.8 + 1.8)), constrained_layout=True)
    heatmap(axes[0], a, candidates, years, "Annual mean RankIC (h=3)", ".3f")
    heatmap(axes[1], b, candidates, years, "Annual mean complete-cycle net PnL (bp)", ".1f")
    fig.suptitle("Candidates selected only on dev IC | annual description")
    fig.supxlabel("Exit must remain within its calendar year. Annual net is NA if any scheduled cycle has unknown PnL.")
    return {"status": "ok", "figure": save(fig, out, "05_annual_candidates.png"),
            "rows": records, "year_boundary": "cross-year holding labels removed; direction remains h3 dev-frozen",
            "funding_price_note": "funding-response marks; where absent, historical time-aligned 8h/4h/1h mark-kline open is a settlement-price reference"}


def cases_figure(signal, rankscore, targets, common, dev_dates, weights, raw, out, leader):
    signed = signal.to_numpy(float)
    available = np.isfinite(signed)
    n = available.sum(axis=1)
    means = np.where(available, signed, 0).sum(axis=1) / np.maximum(n, 1)
    centered = signed - means[:, None]
    sd = np.sqrt(np.where(available, centered**2, 0).sum(axis=1) / np.maximum(n, 1))
    z = np.divide(centered, sd[:, None], out=np.full_like(centered, np.nan), where=sd[:, None] > 1e-12)
    valid = common & np.isfinite(z)
    threshold = weighted_quantile(np.where(valid & dev_dates[:, None], np.abs(z), np.nan), weights, [0.99])[0]
    if not np.isfinite(threshold):
        return {"status": "N/A: no finite dev standardized extreme-score threshold"}
    t, c = np.where(valid & (np.abs(z) >= threshold) & (np.abs(z) > 0))
    events = []
    for i, j in zip(t, c):
        direction = 1 if z[i, j] > 0 else -1
        signed_residual = direction * targets["res"][i, j]
        if not np.isfinite(signed_residual) or signed_residual == 0:
            continue
        events.append({"i": int(i), "j": int(j), "symbol": str(signal.columns[j]),
                       "signal_date": signal.index[i].isoformat(), "z_score": float(z[i, j]),
                       "direction": direction, "raw_forward_bp": targets["raw"][i, j] * 1e4,
                       "residual_forward_bp": targets["res"][i, j] * 1e4,
                       "outcome": "correct" if signed_residual > 0 else "incorrect"})
    chosen = {"correct": [], "incorrect": []}
    for kind in chosen:
        used_dates, used_asset_dates = set(), []
        for e in sorted((x for x in events if x["outcome"] == kind), key=lambda x: (-abs(x["z_score"]), x["signal_date"], x["symbol"])):
            if e["i"] in used_dates or any(s == e["symbol"] and abs(i - e["i"]) < 7 for s, i in used_asset_dates):
                continue
            chosen[kind].append(e); used_dates.add(e["i"]); used_asset_dates.append((e["symbol"], e["i"]))
            if len(chosen[kind]) == 6:
                break
    fig, axes = plt.subplots(4, 3, figsize=(14, 12.5), constrained_layout=True)
    opens = raw["open"].to_numpy(float)
    for category_index, kind in enumerate(("correct", "incorrect")):
        for slot in range(6):
            ax = axes.flat[category_index * 6 + slot]
            if slot >= len(chosen[kind]):
                ax.text(0.5, 0.5, "No qualifying case", ha="center", va="center", transform=ax.transAxes); ax.axis("off"); continue
            e = chosen[kind][slot]; i, j = e["i"], e["j"]
            locations = np.arange(max(0, i - 4), min(len(signal), i + LAG + H + 5))
            entry_price = opens[i + LAG, j]
            path = (opens[locations, j] / entry_price - 1) * 100
            ax.plot(locations - (i + 1), path, color="#16856b" if kind == "correct" else "#b45309", lw=1.5)
            ax.axhline(0, color="#888888", lw=0.7)
            ax.axvline(LAG - 1, color="#2563eb", ls="--", lw=1)
            ax.axvline(LAG + H - 1, color="#c2410c", ls="--", lw=1)
            arrow = "long" if e["direction"] > 0 else "short"
            ax.set_title(f"{kind.upper()}: {ascii_label(e['symbol'])}\n{e['signal_date'][:10]} | {arrow} | z={e['z_score']:.2f}", fontsize=9)
            ax.set_xlabel("Days from signal availability", fontsize=8); ax.set_ylabel("Open / entry - 1 (%)", fontsize=8)
            ax.tick_params(labelsize=8)
            ax.text(0.03, 0.04, f"Raw {e['raw_forward_bp']:+.0f} bp\nBTC-res {e['residual_forward_bp']:+.0f} bp",
                    transform=ax.transAxes, fontsize=8, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
            e["entry_time"] = signal.index[i + LAG].isoformat(); e["exit_time"] = signal.index[i + LAG + H].isoformat()
            e["entry_open"] = entry_price
    fig.suptitle(f"{leader} | extreme-score cases, descriptive only")
    fig.supxlabel("Correctness uses predicted sign x BTC-beta residual return. Blue = entry, orange = exit. Price paths are not net PnL.")
    return {"status": "ok", "figure": save(fig, out, "06_extreme_cases.png"),
            "threshold": threshold, "threshold_rule": "dev weighted 99th percentile of absolute within-date z-score of dev-signed factor",
            "ranking": "largest absolute score first, not largest profit; at most one case per date and seven-day separation per asset within each outcome",
            "outcome_rule": "sign(z_score) * forward BTC-beta residual >0 is correct, <0 incorrect; zero excluded",
            "sample": "full common sample after dev-only threshold; post-outcome case illustration, not validation",
            "count": {k: len(v) for k, v in chosen.items()}, "cases": chosen}


def scan_csv(metrics, out):
    fields = ["ic_mean", "ic_t_hac", "icir", "coverage", "n_ic", "n_active", "n_periods", "net_bp", "gross_bp",
              "trading_cost_bp", "funding_bp", "turnover", "sharpe_net", "n_invalid_price", "n_missing_funding", "sign"]
    frame = metrics[(metrics.h == H) & metrics.factor.str.startswith("alpha")]
    wide = frame.pivot(index="factor", columns="split", values=fields)
    wide.columns = [f"{field}_{split}" for field, split in wide.columns]
    wide["active_fraction_dev"] = wide.n_active_dev / wide.n_periods_dev.replace(0, np.nan)
    wide["qualified_dev"] = ((wide.coverage_dev >= 0.5) & (wide.n_ic_dev >= 100)
                              & (wide.active_fraction_dev >= 0.5) & np.isfinite(wide.ic_mean_dev))
    wide = wide.sort_values(["qualified_dev", "ic_mean_dev"], ascending=[False, False], kind="stable")
    wide.to_csv(out / "alpha101_h3_scan.csv", index=True)
    return {"file": "alpha101_h3_scan.csv", "rows": len(wide),
            "sort": "qualified dev first, then dev signed mean IC descending; never OOT sort"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, choices=[f"{p}_{m}" for p in ("current", "historical") for m in ("normalized", "paper")])
    args = parser.parse_args(); tag = args.tag
    required = {"factors": RESULTS / f"factors_{tag}.pkl.gz",
                "metrics": RESULTS / f"metrics_{tag}.parquet",
                "ic": RESULTS / f"ic_{tag}.parquet",
                "periods": RESULTS / f"periods_{tag}.parquet"}
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise SystemExit("Awaiting experiment outputs: " + ", ".join(missing))
    setup_style(); out = RESULTS / "figures" / tag; out.mkdir(parents=True, exist_ok=True)
    obj = pd.read_pickle(required["factors"])
    metrics = pd.read_parquet(required["metrics"]); ic = pd.read_parquet(required["ic"]); periods = pd.read_parquet(required["periods"])
    config_path = RESULTS / f"evaluation_config_{tag}.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    bounds = _split_bounds(config.get("splits", SPLITS))
    active_fraction = metrics.n_active / metrics.n_periods.replace(0, np.nan)
    dev = metrics[(metrics.h == H) & (metrics.split == "dev") & metrics.factor.str.startswith("alpha")
                  & (metrics.coverage >= 0.5) & (metrics.n_ic >= 100) & (active_fraction >= 0.5)
                  & np.isfinite(metrics.ic_mean)]
    dev = dev.sort_values(["ic_mean", "factor"], ascending=[False, True], kind="stable")
    candidates = dev.head(3).factor.tolist()
    report = {"tag": tag, "sources": {k: str(v.relative_to(HERE)) for k, v in required.items()},
              "selection": {"rule": "h3 dev coverage >=0.5, n_ic >=100, active_cycles/scheduled_cycles >=0.5; descending dev signed IC, factor ID breaks ties",
                            "qualifying_factors": len(dev), "candidates": candidates,
                            "candidate_rows": dev.head(3).to_dict("records"), "uses_oot_for_selection": False},
              "bootstrap": {"count": 200, "block_days": 21, "seed": SEED, "type": "moving blocks with replacement; all assets of each date kept together"},
              "interpretation": "All bins, slices, annual summaries and cases are full-sample descriptions, not new independent tests.",
              "design": obj.get("design"), "formula_version": obj["metadata"].get("version"),
              "formula_sha256": obj["metadata"].get("implementation_sha256"),
              "funding_price_note": "Historical missing funding marks may be filled with time-aligned 8h/4h/1h mark-kline open; these are reference prices, not verified exact settlement marks.",
              "tradability": {"status": "N/A", "reason": "Historical spread/depth series unavailable; no IC-vs-N figure fabricated."},
              "scan": scan_csv(metrics, out)}
    if not candidates:
        report["status"] = "N/A: no candidate meets dev-only data thresholds"
        dump(out / "diagnostics.json", report); print(report["status"]); return
    leader = candidates[0]; sign = int(dev.iloc[0].sign)
    raw, mask = obj["raw"], obj["mask"].astype(bool)
    signal = (obj["factors"][leader] * sign).where(mask)
    index = signal.index
    raw_signal = obj["factors"][leader].where(mask)
    targets_df, target_definitions = four_targets(raw, index, signal.columns)
    target_values = {k: v.to_numpy(float) for k, v in targets_df.items()}
    leader_ic = ic[(ic.factor == leader) & (ic.h == H)].set_index("signal_time")
    valid_dates = leader_ic.label_valid.reindex(index).eq(True).to_numpy(bool)
    dev_dates = np.asarray((index >= bounds["dev"][0]) & (index <= bounds["dev"][1]))
    common = mask.to_numpy(bool) & np.isfinite(signal.to_numpy(float)) & valid_dates[:, None]
    for target in target_values.values():
        common &= np.isfinite(target)
    common[common.sum(axis=1) < 5] = False
    counts = common.sum(axis=1)
    weights = np.divide(common, np.maximum(counts, 1)[:, None])
    score = signal.rank(axis=1, method="average", pct=True).to_numpy(float)
    # Retain actual calendar-day gaps, including split-boundary purges.
    analysis_dates = np.asarray((index >= min(v[0] for v in bounds.values())) & (index <= max(v[1] for v in bounds.values())))
    pos = np.flatnonzero(analysis_dates)
    sample = slice(pos[0], pos[-1] + 1)
    boot = bootstrap_indices(len(pos), 200, 21, SEED)
    report["leader"] = leader; report["leader_sign"] = sign
    report["targets"] = target_definitions
    report["common_sample"] = {"rows": int(common.sum()), "dates": int(common.any(axis=1).sum()),
                               "common_count_min": int(counts[counts > 0].min()) if (counts > 0).any() else None,
                               "by_asset": {str(c): int(common[:, j].sum()) for j, c in enumerate(signal.columns)}}
    report["distribution"] = distribution_figure(signal, mask, valid_dates, out, leader)
    report["binplot"] = binplot_figure(score[sample], {k: v[sample] for k, v in target_values.items()},
                                      common[sample], dev_dates[sample], weights[sample], boot, out, leader)
    past = np.log(raw["close"] / raw["close"].shift(1)).to_numpy(float)
    report["conditional"] = conditional_figure(score[sample], past[sample], {k: v[sample] for k, v in target_values.items()},
                                              common[sample], dev_dates[sample], weights[sample], boot, out, leader)
    report["horizon_delay"] = delay_and_horizon_figure(leader, sign, raw_signal, raw, mask, metrics, bounds, out)
    report["annual"] = annual_figure(candidates, ic, periods, out)
    report["cases"] = cases_figure(signal, score, target_values, common, dev_dates, weights, raw, out, leader)
    report["status"] = "complete"
    dump(out / "diagnostics.json", report)
    print(json.dumps({"tag": tag, "leader": leader, "candidates": candidates,
                      "common_rows": report["common_sample"]["rows"],
                      "figures": [v.get("figure") for v in report.values() if isinstance(v, dict) and v.get("figure")],
                      "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
