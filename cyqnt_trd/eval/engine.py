"""Daily factor evaluation with explicit, fully closed holding periods.

Rows are UTC daily bar *open* timestamps. A signal on row t is available at
open[t+1]; entry_lag=2 waits one complete additional day before open[t+2].
Each scheduled cycle starts with NAV/notional budget 1, holds fixed base units,
and closes every position at its exit. Consecutive cycles do not net orders.
This conservative implementation deliberately pays both close and reopen fees.

``funding`` contains quote currency paid per long base unit during each daily
[open[d], open[d+1]) interval: sum(rate * supplied settlement reference price).
The caller must preserve actual-mark versus historical mark-open proxy provenance.
Positive values are paid by longs; negative values are received by longs.
Only confirmed complete funding intervals may contain explicit zeroes.

Return value: {metrics: DataFrame, ic: DataFrame, periods: DataFrame, config: dict}.
The first three are long tables with an ``h`` column. Metrics contain one row
per horizon/split. PnL columns in periods are fractions of the fixed cycle
budget, with corresponding *_bp fields. Unknown held-position outcomes are
never summed away. Official split PnL averages require complete scheduled
cycles; *_available_bp columns are explicitly partial-sample diagnostics.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
import pandas as pd


DEFAULT_SPLITS = {
    "dev": ("2022-04-01", "2024-06-30"),
    "val": ("2024-07-01", "2025-09-30"),
    "oot": ("2025-10-01", "2026-09-08"),
}
DAY = pd.Timedelta(days=1)
# Change this whenever execution, weighting, funding, or missing-data semantics
# change, so a calibration cannot silently cross incompatible engine contracts.
ENGINE_PROTOCOL = "factor-eval/daily-closed-cycles-v3"
PNL_FIELDS = (
    "gross", "long_gross", "short_gross", "trading_cost", "funding",
    "net_ex_funding", "net",
)


def _utc(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Split boundaries must be finite timestamps")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _split_bounds(splits):
    if not isinstance(splits, Mapping) or "dev" not in splits:
        raise ValueError("splits must be a mapping containing 'dev'")
    result = {}
    for name, bounds in splits.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Split names must be nonempty strings")
        if (isinstance(bounds, (str, bytes, Mapping)) or not hasattr(bounds, "__len__")
                or not hasattr(bounds, "__getitem__") or len(bounds) != 2):
            raise ValueError("Each split requires (start, end)")
        start, end = _utc(bounds[0]), _utc(bounds[1])
        # Old-style YYYY-MM-DD ends include that calendar day, but not the
        # following midnight/open. Explicit timestamps are exact cutoffs.
        if isinstance(bounds[1], str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", bounds[1]):
            end += DAY - pd.Timedelta(nanoseconds=1)
        if start > end:
            raise ValueError("Split start must not follow split end")
        result[name] = (start, end)
    ordered = sorted(result.values())
    if any(right[0] <= left[1] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Splits must not overlap")
    if result["dev"][0] != ordered[0][0]:
        raise ValueError("The dev split must precede every evaluation split")
    if "val" in result and "oot" in result and result["val"][0] > result["oot"][0]:
        raise ValueError("The val split must precede the oot split")
    return result


def _panel(frame, index, columns, name, *, is_mask=False):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name} requires a DatetimeIndex")
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError(f"{name} has duplicate labels")
    result = frame.copy()
    result.index = result.index.tz_localize("UTC") if result.index.tz is None else result.index.tz_convert("UTC")
    result = result.reindex(index=index, columns=columns)
    if is_mask:
        if result.isna().to_numpy().any() or not result.isin([False, True]).to_numpy().all():
            raise ValueError(f"{name} must explicitly contain only booleans or 0/1 on the signal grid")
        return result.astype(bool)
    return result.astype(float)


def _mean(values, *, strict=False):
    x = np.asarray(values, dtype=float)
    if not len(x) or (strict and not np.isfinite(x).all()):
        return np.nan
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def _std(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1)) if len(x) > 1 else np.nan


def _automatic_lags(n, minimum=0):
    return min(max(0, n - 1), max(minimum, int(4 * (n / 100.0) ** (2 / 9))))


def _hac_mean_t(values, lags):
    """Bartlett HAC for a regular time grid, retaining missing-time gaps.

    Missing observations have zero influence rather than being concatenated
    with their neighbours. This is an estimate, not a conservative bound.
    """
    x = np.asarray(values, dtype=float)
    valid = np.isfinite(x)
    n = int(valid.sum())
    if n < 3:
        return np.nan
    mean = float(x[valid].mean())
    influence = np.where(valid, x - mean, 0.0)
    lags = min(max(int(lags), 0), len(x) - 1)
    variance_sum = float(influence @ influence)
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        variance_sum += 2 * weight * float(influence[lag:] @ influence[:-lag])
    if variance_sum <= 0:
        return np.nan
    return mean / (np.sqrt(variance_sum) / n)


def _spearman_rows(signal, target, mask, min_assets):
    both = mask & np.isfinite(signal) & np.isfinite(target)
    n = both.sum(axis=1)
    a = pd.DataFrame(np.where(both, signal, np.nan)).rank(axis=1).to_numpy()
    b = pd.DataFrame(np.where(both, target, np.nan)).rank(axis=1).to_numpy()
    denominator = np.maximum(n, 1)[:, None]
    a = np.where(both, a - np.nansum(a, axis=1)[:, None] / denominator, 0.0)
    b = np.where(both, b - np.nansum(b, axis=1)[:, None] / denominator, 0.0)
    scale = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    ic = np.full(len(signal), np.nan)
    np.divide((a * b).sum(axis=1), scale, out=ic, where=(scale > 0) & (n >= min_assets))
    return ic, n


def _target_weights(signal, mask, min_assets):
    observed = mask & np.isfinite(signal)
    n = observed.sum(axis=1)
    ranks = pd.DataFrame(np.where(observed, signal, np.nan)).rank(axis=1).to_numpy()
    mean_rank = np.nansum(ranks, axis=1) / np.maximum(n, 1)
    centered = np.where(observed, ranks - mean_rank[:, None], 0.0)
    gross = np.abs(centered).sum(axis=1)
    weights = np.zeros_like(centered)
    np.divide(centered, gross[:, None], out=weights,
              where=((gross > 0) & (n >= min_assets))[:, None])
    return weights, n


def _sharpe(values, periods_per_year):
    x = np.asarray(values, dtype=float)
    if len(x) < 2 or not np.isfinite(x).all():
        return np.nan
    std = x.std(ddof=1)
    return float(x.mean() / std * np.sqrt(periods_per_year)) if std > 0 else np.nan


def evaluate_factor(signal, opens, closes, mask, funding=None, horizons=(1, 3, 5),
                    cost_bps=6.5, splits=None, entry_lag=2, *, min_assets=5,
                    hac_lags=None, annualization=365.0):
    """Evaluate one factor; select and freeze a separate dev sign for each h.

    Horizon-specific sign selection is exploratory and must count in the
    experiment ledger. ``closes`` is aligned/validated for the daily signal
    contract; it is never silently substituted for a funding settlement mark.
    Schedules are anchored at the first panel row with signal offsets 0, h, 2h.
    Splits use signal dates and require the complete exit timestamp <= cutoff.
    A YYYY-MM-DD end includes that date; an explicit timestamp is an exact end.
    """
    if not isinstance(signal, pd.DataFrame) or not isinstance(signal.index, pd.DatetimeIndex):
        raise TypeError("signal must be a DataFrame with a DatetimeIndex")
    index = signal.index.tz_localize("UTC") if signal.index.tz is None else signal.index.tz_convert("UTC")
    if not len(index) or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("signal index must be nonempty, unique, and increasing")
    if not index.equals(index.normalize()) or (len(index) > 1 and not (np.diff(index.asi8) == DAY.value).all()):
        raise ValueError("The engine requires a complete daily UTC midnight grid")
    if not signal.columns.is_unique or not len(signal.columns):
        raise ValueError("signal requires unique, nonempty asset columns")
    if isinstance(entry_lag, (bool, np.bool_)) or not isinstance(entry_lag, (int, np.integer)) or entry_lag < 1:
        raise ValueError("entry_lag must be an integer >= 1")
    horizons = tuple(horizons)
    if not horizons or any(isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer)) or h < 1 for h in horizons):
        raise ValueError("horizons must contain positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be unique")
    horizons = tuple(int(h) for h in horizons)
    if isinstance(cost_bps, (bool, np.bool_)) or not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("cost_bps must be finite and nonnegative")
    if (isinstance(min_assets, (bool, np.bool_)) or not isinstance(min_assets, (int, np.integer))
            or min_assets < 2):
        raise ValueError("min_assets must be an integer >= 2")
    if isinstance(annualization, (bool, np.bool_)) or not np.isfinite(annualization) or annualization <= 0:
        raise ValueError("annualization must be finite and positive")
    if hac_lags is not None and (isinstance(hac_lags, (bool, np.bool_))
                               or not isinstance(hac_lags, (int, np.integer)) or hac_lags < 0):
        raise ValueError("hac_lags must be a nonnegative integer")

    bounds = _split_bounds(DEFAULT_SPLITS if splits is None else splits)
    columns = signal.columns
    sig = _panel(signal, index, columns, "signal").to_numpy()
    px = _panel(opens, index, columns, "opens").to_numpy()
    _panel(closes, index, columns, "closes")
    eligible = _panel(mask, index, columns, "mask", is_mask=True).to_numpy()
    fund = None if funding is None else _panel(funding, index, columns, "funding").to_numpy()
    raw_weights, n_signal = _target_weights(sig, eligible, min_assets)
    signal_complete = n_signal >= min_assets
    n_expected = eligible.sum(axis=1)
    sample_size, asset_count = sig.shape
    if fund is not None:
        fund_good = np.isfinite(fund)
        fund_sum = np.vstack([np.zeros(asset_count), np.cumsum(np.where(fund_good, fund, 0.0), axis=0)])
        fund_bad = np.vstack([np.zeros(asset_count, dtype=int), np.cumsum(~fund_good, axis=0)])

    metric_rows, ic_tables, period_tables = [], [], []
    for h in horizons:
        entry_index = np.arange(sample_size) + entry_lag
        exit_index = entry_index + h
        in_panel = exit_index < sample_size
        entry_times = index + entry_lag * DAY
        exit_times = index + (entry_lag + h) * DAY
        entry_prices = np.full_like(px, np.nan)
        exit_prices = np.full_like(px, np.nan)
        entry_exists = entry_index < sample_size
        entry_prices[entry_exists] = px[entry_index[entry_exists]]
        exit_prices[in_panel] = px[exit_index[in_panel]]
        good_price = (np.isfinite(entry_prices) & (entry_prices > 0)
                      & np.isfinite(exit_prices) & (exit_prices > 0))
        ratio = np.full_like(px, np.nan)
        np.divide(exit_prices, entry_prices, out=ratio, where=good_price)
        with np.errstate(invalid="ignore", divide="ignore"):
            log_target = np.log(ratio)
        raw_ic, n_label = _spearman_rows(sig, log_target, eligible, min_assets)
        split_names = np.full(sample_size, "", dtype=object)
        label_valid = np.zeros(sample_size, dtype=bool)
        for name, (start, end) in bounds.items():
            in_split = np.asarray((index >= start) & (index <= end))
            split_names[in_split] = name
            label_valid[in_split] = (in_panel & np.asarray((entry_times >= start) & (exit_times <= end)))[in_split]
        raw_ic[~label_valid] = np.nan
        dev_mean = _mean(raw_ic[split_names == "dev"])
        direction_frozen = bool(np.isfinite(dev_mean) and dev_mean != 0)
        sign = -1 if direction_frozen and dev_mean < 0 else 1
        signed_ic = sign * raw_ic
        weights = sign * raw_weights
        held = np.abs(weights) > 0
        active = held.any(axis=1)
        price_complete = (~held | good_price).all(axis=1)

        simple_return = ratio - 1.0
        long_gross = np.where(weights > 0, weights * simple_return, 0.0).sum(axis=1)
        short_gross = np.where(weights < 0, weights * simple_return, 0.0).sum(axis=1)
        gross = long_gross + short_gross
        entry_notional = np.abs(weights).sum(axis=1)
        exit_notional = np.where(held, np.abs(weights) * ratio, 0.0).sum(axis=1)
        turnover = entry_notional + exit_notional
        trading_cost = turnover * cost_bps / 1e4
        units = np.zeros_like(weights)
        np.divide(weights, entry_prices, out=units, where=held & np.isfinite(entry_prices) & (entry_prices > 0))
        if fund is None:
            funding_complete = ~active
            funding_paid = np.where(active, np.nan, 0.0)
        else:
            accumulated = np.full_like(fund, np.nan)
            missing = np.ones_like(fund, dtype=bool)
            accumulated[in_panel] = fund_sum[exit_index[in_panel]] - fund_sum[entry_index[in_panel]]
            missing[in_panel] = (fund_bad[exit_index[in_panel]] - fund_bad[entry_index[in_panel]]) > 0
            funding_complete = (~held | ~missing).all(axis=1)
            funding_paid = np.where(held, units * accumulated, 0.0).sum(axis=1)
            funding_paid[~funding_complete] = np.nan
        for values in (gross, long_gross, short_gross, exit_notional, turnover, trading_cost, funding_paid):
            values[~price_complete | ~label_valid | ~signal_complete] = np.nan
        net_ex_funding = gross - trading_cost
        net = net_ex_funding - funding_paid
        status = np.full(sample_size, "ok", dtype=object)
        status[~active] = "flat"
        status[active & ~funding_complete] = "missing_funding"
        status[active & ~price_complete] = "missing_price"
        status[~signal_complete] = "missing_signal"
        status[~label_valid] = "purged_label"
        coverage = np.full(sample_size, np.nan)
        target_coverage = np.full(sample_size, np.nan)
        np.divide(n_signal, n_expected, out=coverage, where=n_expected > 0)
        np.divide(n_label, n_signal, out=target_coverage, where=n_signal > 0)

        ic_table = pd.DataFrame({
            "h": h, "split": split_names, "signal_time": index,
            "available_at": index + DAY, "entry_time": entry_times,
            "exit_time": exit_times, "sign": sign, "direction_frozen": direction_frozen,
            "ic_raw": raw_ic, "ic": signed_ic, "n_assets": n_label,
            "n_signal_assets": n_signal, "n_expected_assets": n_expected,
            "coverage": coverage, "target_coverage": target_coverage,
            "label_valid": label_valid,
            "signal_complete": signal_complete,
        })
        ic_tables.append(ic_table.loc[ic_table.split != ""].reset_index(drop=True))
        period_table = pd.DataFrame({
            "h": h, "split": split_names, "signal_time": index,
            "available_at": index + DAY, "entry_time": entry_times,
            "exit_time": exit_times, "sign": sign, "status": status,
            "label_valid": label_valid, "active": active,
            "signal_complete": signal_complete,
            "price_complete": price_complete, "funding_complete": funding_complete,
            "n_positions": held.sum(axis=1), "entry_notional": entry_notional,
            "exit_notional": exit_notional, "turnover": turnover,
            "gross": gross, "long_gross": long_gross, "short_gross": short_gross,
            "trading_cost": trading_cost, "funding": funding_paid,
            "net_ex_funding": net_ex_funding, "net": net,
        }).iloc[::h].copy()
        for field in PNL_FIELDS:
            period_table[f"{field}_bp"] = period_table[field] * 1e4
        period_table["funding_paid_quote_per_nav"] = period_table.funding
        period_table = period_table.loc[period_table.split != ""].reset_index(drop=True)
        period_tables.append(period_table)

        for name in bounds:
            dates = (split_names == name)
            valid_dates = dates & label_valid
            values = signed_ic[dates]  # Keep daily missing-time gaps for HAC.
            lag = int(hac_lags) if hac_lags is not None else _automatic_lags(len(values), minimum=h - 1)
            lag = min(lag, max(0, len(values) - 1))
            cycles_all = period_table.loc[period_table.split == name]
            cycles = cycles_all.loc[cycles_all.label_valid]
            ic_mean, ic_std = _mean(values), _std(values)
            m = {
                "h": h, "split": name, "sign": sign,
                "direction_frozen": direction_frozen, "dev_ic_raw_mean": dev_mean,
                "ic_mean": ic_mean, "ic_mean_raw": _mean(raw_ic[dates]),
                "icir": ic_mean / ic_std if ic_std > 0 else np.nan,
                "ic_win": _mean((values[np.isfinite(values)] > 0).astype(float)),
                "ic_t_hac": _hac_mean_t(values, lag), "ic_hac_lags": lag,
                "n_ic": int(np.isfinite(values).sum()),
                "n_signal_dates": int(valid_dates.sum()),
                "coverage": _mean(coverage[valid_dates]),
                "target_coverage": _mean(target_coverage[valid_dates]),
                "mean_ic_assets": _mean(n_label[valid_dates]),
                "n_periods": len(cycles), "n_purged": len(cycles_all) - len(cycles),
                "n_active": int(cycles.active.sum()),
                "n_flat": int((~cycles.active & cycles.signal_complete).sum()),
                "n_missing_signal": int((~cycles.signal_complete).sum()),
                "n_invalid_price": int((~cycles.price_complete).sum()),
                "n_missing_funding": int((cycles.active & ~cycles.funding_complete).sum()),
                "n_complete": int(np.isfinite(cycles.net).sum()),
                "turnover": _mean(cycles.turnover, strict=True),
                "turnover_available": _mean(cycles.turnover),
                "entry_notional": _mean(cycles.entry_notional, strict=True),
                "exit_notional": _mean(cycles.exit_notional, strict=True),
                "sharpe_net": _sharpe(cycles.net, annualization / h),
                "sharpe_net_ex_funding": _sharpe(cycles.net_ex_funding, annualization / h),
                "funding_supplied": fund is not None,
            }
            for field in PNL_FIELDS:
                m[f"{field}_bp"] = _mean(cycles[field], strict=True) * 1e4
                m[f"{field}_available_bp"] = _mean(cycles[field]) * 1e4
            turnover_mean = m["turnover"]
            m["breakeven_cost_bps"] = ((m["gross_bp"] - m["funding_bp"]) / turnover_mean
                                         if turnover_mean > 0 else np.nan)
            m["breakeven_ex_funding_bps"] = (m["gross_bp"] / turnover_mean
                                              if turnover_mean > 0 else np.nan)
            net_lag = _automatic_lags(len(cycles))
            m["net_t_hac"] = (_hac_mean_t(cycles.net, net_lag)
                               if len(cycles) and np.isfinite(cycles.net).all() else np.nan)
            m["net_hac_lags_cycles"] = net_lag
            metric_rows.append(m)

    return {
        "metrics": pd.DataFrame(metric_rows),
        "ic": pd.concat(ic_tables, ignore_index=True),
        "periods": pd.concat(period_tables, ignore_index=True),
        "config": {
            "version": "v3_closed_cycles",
            "engine_protocol": ENGINE_PROTOCOL,
            "bar": "1d", "timezone": "UTC", "entry_lag": int(entry_lag),
            "additional_wait_days_after_bar_close": int(entry_lag - 1),
            "horizons": list(horizons), "cost_bps_one_way": float(cost_bps),
            "min_assets": int(min_assets), "annualization": float(annualization),
            "signal_available_at": "bar_open + 1 day",
            "execution": "open[t+entry_lag] to open[t+entry_lag+h]",
            "cycle_capital": "fixed 1 unit per cycle; fixed base units; no compounding",
            "weights": "demeaned cross-sectional ranks; dollar net 0; gross 1 when active",
            "schedule": "signal row offsets 0,h,2h; anchored at first panel row",
            "cost_method": "complete close and reopen; adjacent cycles do not net orders; exit notional drifts with price",
            "funding_method": "sum(units * sum(rate * supplied settlement price reference)) over [entry, exit); positive is expense; actual/proxy price provenance is caller-supplied",
            "funding_supplied": funding is not None,
            "incomplete_pnl": "whole held-position cycle invalid; official split means require all cycles; available fields are partial diagnostics",
            "flat_cycles": "included with zero PnL and zero current-cycle cost; prior cycle already pays its full exit",
            "missing_signal": "fewer than min_assets finite eligible signals is unknown PnL, never a flat zero-return cycle",
            "direction": "sign of dev mean raw RankIC separately for each horizon; exploratory horizon-specific selection",
            "missing_dev_direction": "missing or zero dev mean gives sign +1 placeholder with direction_frozen=False; no validated direction",
            "ic": "per-date Spearman versus forward log returns; report valid-pair coverage",
            "hac": "Bartlett on regular time grids; missing date gaps retained; not a guaranteed conservative bound",
            "splits": {name: [start.isoformat(), end.isoformat()] for name, (start, end) in bounds.items()},
        },
    }
