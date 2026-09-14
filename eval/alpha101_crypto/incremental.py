"""Post-hoc, common-signal-sample projection diagnostics; no portfolio optimization.

Scope is deliberately fixed: current cohort, v2 normalized inputs, h=3,
alpha088/alpha040 versus the five existing public baselines. Projection uses
only dev score ranks, never returns. Funding/outcome exclusions are reported
separately and never used to create the signal mask or fit the projection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
_EVAL_ROOT = str(Path(__file__).resolve().parents[1])  # eval/ holds the shared measurement core
if _EVAL_ROOT not in _sys.path: _sys.path.insert(0, _EVAL_ROOT)
from factor_eval.engine import evaluate_factor
from run_pipeline import DATA, OUT, SPLITS, baseline_signals, funding_panel

FOCUS = ("alpha088", "alpha040")
BASELINES = ("B_size", "B_lowvol10", "B_rev5", "B_mom20", "B_funding7")
HORIZON = 3
ENTRY_LAG = 2
MIN_ASSETS = 5
TAG = "current_normalized"
DESTINATION = OUT / "incremental_current_normalized_h3"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, (pd.Timestamp, Path)):
            return str(value)
        return value
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def fit_projection(score, baseline_ranks, common_mask, training_dates):
    """Each training date has total weight 1, divided among its valid assets."""
    x_grid = np.stack([baseline_ranks[b].to_numpy() for b in BASELINES], axis=2)
    training_cells = common_mask.to_numpy() & np.asarray(training_dates)[:, None]
    n_per_date = common_mask.sum(axis=1).to_numpy()
    if np.sum(np.asarray(training_dates) & (n_per_date >= MIN_ASSETS)) < 10:
        raise ValueError("fewer than ten common dev dates; projection is not estimable")
    x = np.column_stack([np.ones(int(training_cells.sum())), x_grid[training_cells]])
    y = score.to_numpy()[training_cells]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("projection training data must be finite on the common mask")
    weight_grid = np.broadcast_to(1.0 / np.maximum(n_per_date, 1)[:, None], common_mask.shape)
    weights = weight_grid[training_cells]
    root_weights = np.sqrt(weights)
    beta, _, rank, singular = np.linalg.lstsq(x * root_weights[:, None], y * root_weights, rcond=None)
    predictions = beta[0] + np.sum(x_grid * beta[1:], axis=2)
    residual = pd.DataFrame(score.to_numpy() - predictions, index=score.index,
                            columns=score.columns).where(common_mask)
    errors = y - np.sum(x * beta, axis=1)
    y_mean = np.average(y, weights=weights)
    sst = np.sum(weights * (y - y_mean) ** 2)
    stats = {
        "intercept": float(beta[0]), "coefficients": dict(zip(BASELINES, beta[1:].tolist())),
        "training_dates": int(np.sum(np.asarray(training_dates) & (n_per_date >= MIN_ASSETS))),
        "training_cells": int(training_cells.sum()), "design_rank": int(rank),
        "design_columns_including_intercept": len(BASELINES) + 1,
        "condition_number": float(singular[0] / singular[-1]) if singular[-1] > 0 else None,
        "weighted_score_rmse": float(np.sqrt(np.average(errors ** 2, weights=weights))),
        "weighted_score_r2": float(1 - np.sum(weights * errors ** 2) / sst) if sst > 0 else None,
        "score_r2_is_not_return_prediction_r2": True,
    }
    return residual, stats


def common_cycle_diagnostics(periods):
    """Ex-post accounting subset, explicitly separate from the causal score mask."""
    by_signal = periods.set_index(["signal_time", "factor"])
    net = by_signal.net.unstack("factor")
    label_valid = by_signal.label_valid.unstack("factor").all(axis=1)
    complete = label_valid & np.isfinite(net).all(axis=1)
    split = periods.drop_duplicates("signal_time").set_index("signal_time").split
    rows = []
    for factor, group in periods.groupby("factor", sort=False):
        for part in SPLITS:
            dates = complete.index[complete & split.eq(part)]
            selected = group.set_index("signal_time").reindex(dates)
            count = len(selected)
            std = selected.net.std(ddof=1)
            row = {"factor": factor, "split": part, "h": HORIZON,
                   "n_common_complete_cycles": count,
                   "n_common_scheduled_cycles": int((label_valid & split.eq(part)).sum()),
                   "net_sharpe": float(selected.net.mean() / std * np.sqrt(365 / HORIZON))
                                  if count > 1 and std > 0 else np.nan,
                   "turnover": selected.turnover.mean()}
            for field in ("gross", "trading_cost", "funding", "net_ex_funding", "net"):
                row[field + "_bp"] = selected[field].mean() * 1e4
            rows.append(row)
    dates_table = pd.DataFrame({"signal_time": complete.index, "split": split.reindex(complete.index).values,
                               "label_valid": label_valid.values,
                               "all_diagnostic_signals_accounting_complete": complete.values})
    return pd.DataFrame(rows), dates_table


def run():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    cache = OUT / f"factors_{TAG}.pkl.gz"
    primary_metrics_path = OUT / f"metrics_{TAG}.parquet"
    obj = pd.read_pickle(cache)
    raw, eligible = obj["raw"], obj["mask"]
    funding, funding_provenance = funding_panel(raw["open"].index, raw["open"].columns)
    baselines = baseline_signals(raw, eligible, funding)
    sources = {**{f: obj["factors"][f] for f in FOCUS}, **baselines}
    primary = pd.read_parquet(primary_metrics_path)
    primary = primary[(primary.h == HORIZON) & (primary.split == "dev")].set_index("factor")
    if any(not bool(primary.loc[name, "direction_frozen"]) for name in sources):
        raise ValueError("at least one source has no frozen primary dev direction")
    source_signs = {name: int(primary.loc[name, "sign"]) for name in sources}

    # No forward returns, future price availability, or future funding enter this mask.
    common = eligible.copy()
    for signal in sources.values():
        common &= np.isfinite(signal)
    usable_dates = common.sum(axis=1).ge(MIN_ASSETS)
    for signal in sources.values():
        usable_dates &= signal.where(common).nunique(axis=1).gt(1)
    common = common.where(usable_dates, False)
    ranks = {name: (signal * source_signs[name]).where(common).rank(axis=1, pct=True, method="average")
             for name, signal in sources.items()}

    evaluated = {}
    def evaluate(name, score):
        result = evaluate_factor(score, raw["open"], raw["close"], common, funding,
                                 horizons=(HORIZON,), splits=SPLITS, entry_lag=ENTRY_LAG,
                                 min_assets=MIN_ASSETS)
        evaluated[name] = result
        return result

    for name, score in ranks.items():
        evaluate(name, score)
    # This maturity condition depends only on timestamps, not realized outcomes.
    date_info = evaluated[FOCUS[0]]["ic"].set_index("signal_time")
    training_dates = pd.Series(False, index=common.index)
    dev_dates = date_info.index[date_info.split.eq("dev") & date_info.label_valid]
    training_dates.loc[dev_dates] = True

    projection = {}
    for name in FOCUS:
        residual, stats = fit_projection(ranks[name], ranks, common, training_dates)
        projection[name] = stats
        ranks[name + "_residual"] = residual
        evaluate(name + "_residual", residual)

    metrics = pd.concat([r["metrics"].assign(factor=name) for name, r in evaluated.items()], ignore_index=True)
    metrics["ic_at_primary_or_projection_orientation"] = metrics.ic_mean_raw
    metrics["ic_after_diagnostic_dev_sign"] = metrics.ic_mean
    metrics["direction_stage"] = "engine dev sign on common diagnostic sample"
    periods = pd.concat([r["periods"].assign(factor=name) for name, r in evaluated.items()], ignore_index=True)
    ic = pd.concat([r["ic"].assign(factor=name) for name, r in evaluated.items()], ignore_index=True)
    metrics.to_csv(DESTINATION / "metrics_common_signal_sample.csv", index=False)
    metrics.to_parquet(DESTINATION / "metrics_common_signal_sample.parquet", index=False)
    periods.to_parquet(DESTINATION / "periods_common_signal_sample.parquet", index=False)
    ic.to_parquet(DESTINATION / "ic_common_signal_sample.parquet", index=False)

    correlations = []
    for name in FOCUS:
        for baseline in BASELINES:
            daily = ranks[name].corrwith(ranks[baseline], axis=1)
            for part in SPLITS:
                dates = date_info.index[date_info.split.eq(part) & date_info.label_valid]
                values = daily.reindex(dates).dropna()
                correlations.append({"factor": name, "baseline": baseline, "split": part,
                                     "mean_daily_rank_correlation": values.mean(), "n_dates": len(values)})
    pd.DataFrame(correlations).to_csv(DESTINATION / "daily_rank_correlations.csv", index=False)
    comparison, comparable_dates = common_cycle_diagnostics(periods)
    comparison.to_csv(DESTINATION / "metrics_common_complete_cycles.csv", index=False)
    comparable_dates.to_csv(DESTINATION / "accounting_comparison_dates.csv", index=False)
    common.rename_axis("signal_time").to_csv(DESTINATION / "common_signal_mask.csv.gz")
    pd.to_pickle({"scores": ranks, "common_mask": common, "projection": projection},
                 DESTINATION / "scores_and_projection.pkl.gz")
    coefficient_rows = []
    for name, stats in projection.items():
        for term, value in {"intercept": stats["intercept"], **stats["coefficients"]}.items():
            coefficient_rows.append({"factor": name, "term": term, "coefficient": value})
    pd.DataFrame(coefficient_rows).to_csv(DESTINATION / "projection_coefficients.csv", index=False)

    sample_rows = []
    for part in SPLITS:
        dates = date_info.index[date_info.split.eq(part) & date_info.label_valid]
        counts = common.reindex(dates).sum(axis=1)
        active_counts = counts[counts >= MIN_ASSETS]
        sample_rows.append({"split": part, "mature_signal_dates": len(dates),
                            "common_signal_dates": len(active_counts),
                            "common_asset_cells": int(counts.sum()),
                            "mean_assets_on_common_dates": active_counts.mean(),
                            "first_common_signal": str(active_counts.index.min()) if len(active_counts) else None,
                            "last_common_signal": str(active_counts.index.max()) if len(active_counts) else None})
    source_files = {"factor_cache": {"path": str(cache), "sha256": sha(cache)},
                    "primary_metrics": {"path": str(primary_metrics_path), "sha256": sha(primary_metrics_path)},
                    "script": {"path": str(Path(__file__)), "sha256": sha(__file__)}}
    for name in ("run_pipeline.py", "engine.py"):
        path = Path(__file__).with_name(name)
        source_files[name] = {"path": str(path), "sha256": sha(path)}
    for source in funding_provenance:
        symbol = source["symbol"]
        path = DATA / "funding" / f"{symbol}.parquet"
        if path.exists():
            source_files[f"funding_{symbol}"] = {"path": str(path), "sha256": sha(path)}
            stamp = path.with_suffix(".json")
            if stamp.exists():
                source_files[f"funding_metadata_{symbol}"] = {"path": str(stamp), "sha256": sha(stamp)}
    meta = {"analysis": "post-hoc score-projection diagnostic, not tradable combination incremental value",
            "pool": "current", "mode": "normalized", "h": HORIZON, "entry_lag": ENTRY_LAG,
            "focus": list(FOCUS), "baselines": list(BASELINES), "min_assets": MIN_ASSETS,
            "primary_dev_directions_used_for_correlations_and_projection": source_signs,
            "projection": projection, "sample": sample_rows,
            "signal_mask": "eligible AND finite alpha088, alpha040 and all 5 baselines; at least 5 names; all seven source scores nonconstant on each retained date; no future outcomes",
            "projection_fit": "dev mature timestamps only; intercept; each date total OLS weight=1; no returns as regressors or target",
            "frozen_coefficients": True,
            "direction_warning": "Source rank directions come from primary h=3 dev. Engine selects another dev sign on this common diagnostic sample, including residuals. Metrics retain unflipped IC explicitly. This is additional post-hoc selection.",
            "accounting_subset": "separate ex-post intersection of finite net outcomes for all 9 diagnostic signals; not used in signal mask or fit; partial-sample descriptive comparison only",
            "differs_from_primary_sample": True,
            "funding_provenance": funding_provenance, "sources": source_files}
    dump(DESTINATION / "method_and_projection.json", meta)
    method = """# 共同样本投影诊断：current / normalized / h=3

这是事后信息重叠诊断，与主报告样本不同，不代表可交易组合增量，也不进行组合权重优化。

共同信号池要求 alpha088、alpha040 与五个基线在当时均可计算，每时点至少五个币，并且七个原始分数在当天共同资产中均有横截面差异。信号池不使用未来收益、未来成交价或未来资金费是否可得。全部比较使用同一资产/日期掩码；条件不满足的计划周期保持空仓，仍按评估器合同计入。

相关性与投影使用主实验 h=3 开发段冻结方向后的横截面百分位秩。OLS 带截距，以五个基线排名解释因子排名，仅在开发段到期标签对应的日期拟合；每日期总权重为一，再在该日期资产之间平分。收益不参与投影回归。系数固定后对全部三段生成残差。

评估器会在本次共同样本开发段再次决定输入信号的方向，残差同样如此。该步骤属于新增选择，JSON 和 CSV 保留方向以及未翻转残差 IC。投影的 score R² 仅描述分数拟合，不是收益预测 R²，也不是“可复制的信息百分比”。

`metrics_common_signal_sample.csv` 保留严格指标：任一持仓周期资金费或价格未知，整段相应净指标仍为 NA。`metrics_common_complete_cycles.csv` 另按全部九个诊断信号都能核算净值的周期取交集，是事后完整周期子集，仅用于同周期描述性比较；不得替代全样本成绩。计数与日期见配套 CSV。

此分析未给出投影系数或增量的区块置信区间；横截面小、信号持续性、资金费代理以及共同样本选择均限制结论。请与未参与本轮选择的新样本验证分开。
"""
    (DESTINATION / "METHOD.md").write_text(method)
    print("Wrote", DESTINATION)
    print(metrics[["factor", "split", "ic_at_primary_or_projection_orientation", "ic_after_diagnostic_dev_sign", "net_bp"]].to_string(index=False))
    print("Common signal sample:", sample_rows)


if __name__ == "__main__":
    run()
