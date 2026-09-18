"""Comparable raw, risk-scaled, and benchmark-residual return diagnostics.

All four targets use the same future open-to-open interval. Their beta and
volatility estimates use only completed close observations available at the
signal date; future target returns never enter those estimates. These are
diagnostic labels, not net trading returns or strategy-combination evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bundle import Panel


TARGET_NAMES = ("raw_rtf", "normed_rtf", "res_rtf", "normed_res_rtf")


def _integer(name, value, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def build_targets(panel: Panel, h: int, entry_lag: int = 2, lookback: int = 60,
                  benchmark_symbol: str | None = "BTCUSDT", min_samples: int = 40) -> dict:
    """Build four aligned targets and their strictly historical risk estimates.

    A row at ``t`` represents a signal formed after that day's close. Its entry
    and exit are ``open[t + entry_lag]`` and ``open[t + entry_lag + h]``.
    ``lookback`` counts daily observations of trailing *h-day* close returns,
    which may overlap. No square-root-of-time scaling is used. ``min_samples``
    is the minimum number of finite historical observations in that window.

    Return keys are ``targets`` (four DataFrames), ``common_mask`` (all four
    finite and eligible), ``coverage`` (individual eligible-cell counts and
    fractions), ``availability`` (status/reason per target), ``estimates``
    (historical beta, scales and sample counts), and JSON-safe ``metadata``.
    An absent/constant benchmark leaves its dependent targets unavailable.
    """
    panel.validate()
    h = _integer("h", h, 1)
    entry_lag = _integer("entry_lag", entry_lag, 1)
    lookback = _integer("lookback", lookback, 2)
    min_samples = _integer("min_samples", min_samples, 2)
    if min_samples > lookback:
        raise ValueError("min_samples cannot exceed lookback")
    if benchmark_symbol is not None and (not isinstance(benchmark_symbol, str) or not benchmark_symbol.strip()):
        raise ValueError("benchmark_symbol must be a nonempty symbol or None")

    log_open, log_close = np.log(panel.open), np.log(panel.close)
    raw = log_open.shift(-(entry_lag + h)) - log_open.shift(-entry_lag)
    historical = log_close - log_close.shift(h)
    rolling = historical.rolling(lookback, min_periods=min_samples)
    sigma_raw = rolling.std(ddof=1)
    sigma_raw = sigma_raw.where(np.isfinite(sigma_raw) & (sigma_raw > 0))
    raw_count = historical.rolling(lookback, min_periods=1).count()

    blank = pd.DataFrame(np.nan, index=panel.index, columns=panel.symbols)
    beta, sigma_res, paired_count = blank.copy(), blank.copy(), blank.copy()
    benchmark_forward = pd.Series(np.nan, index=panel.index, name="benchmark_forward")
    benchmark_present = benchmark_symbol in panel.symbols if benchmark_symbol is not None else False
    if benchmark_present:
        market = historical[benchmark_symbol]
        benchmark_forward = raw[benchmark_symbol].rename("benchmark_forward")
        for symbol in panel.symbols:
            # Every covariance and variance uses the exact same finite pairs.
            # Different missing-value patterns must not imply negative residual
            # variance or a beta computed from a different estimation sample.
            y = historical[symbol]
            paired = np.isfinite(y) & np.isfinite(market)
            y_pair, m_pair = y.where(paired), market.where(paired)
            y_window = y_pair.rolling(lookback, min_periods=min_samples)
            m_window = m_pair.rolling(lookback, min_periods=min_samples)
            covariance = y_window.cov(m_pair, ddof=1)
            variance_y = y_window.var(ddof=1)
            variance_m = m_window.var(ddof=1)
            valid = np.isfinite(variance_m) & (variance_m > 0)
            slope = (covariance / variance_m).where(valid)
            beta[symbol] = slope.where(np.isfinite(slope))
            residual_variance = variance_y - covariance * slope
            # Exact replicas have zero residual scale; roundoff must not create
            # an enormous normalized score by dividing by a spurious tiny scale.
            tolerance = 32 * np.finfo(float).eps * variance_y.abs()
            residual_variance = residual_variance.where(valid & (residual_variance > tolerance))
            sigma_res[symbol] = np.sqrt(residual_variance)
            paired_count[symbol] = paired.astype(int).rolling(lookback, min_periods=1).sum()

    residual = raw - beta.mul(benchmark_forward, axis=0)
    values = {
        "raw_rtf": raw,
        "normed_rtf": raw / sigma_raw,
        "res_rtf": residual,
        "normed_res_rtf": residual / sigma_res,
    }
    targets = {name: value.where(panel.mask & np.isfinite(value)) for name, value in values.items()}
    common_mask = panel.mask.copy()
    for target in targets.values():
        common_mask &= np.isfinite(target)

    eligible_cells = int(panel.mask.to_numpy().sum())
    coverage, availability = {}, {}
    for name, target in targets.items():
        valid_mask = panel.mask & np.isfinite(target)
        valid_cells = int(valid_mask.to_numpy().sum())
        coverage[name] = {
            "eligible_cells": eligible_cells, "valid_cells": valid_cells,
            "fraction": valid_cells / eligible_cells if eligible_cells else None,
            "valid_dates": int(valid_mask.any(axis=1).sum()),
        }
        if valid_cells:
            status, reason = "AVAILABLE", "finite labels exist; inspect coverage before comparing targets"
        elif name in ("res_rtf", "normed_res_rtf") and not benchmark_present:
            status, reason = "UNAVAILABLE", "requested benchmark is absent; no substitute was selected"
        elif name in ("res_rtf", "normed_res_rtf") and not np.isfinite(beta.to_numpy()).any():
            status, reason = "UNAVAILABLE", "insufficient paired history or zero benchmark variance"
        elif name == "normed_res_rtf":
            status, reason = "UNAVAILABLE", "no positive residual scale with a finite forward label"
        elif name == "normed_rtf":
            status, reason = "UNAVAILABLE", "insufficient history or no positive raw scale with a finite forward label"
        else:
            status, reason = "UNAVAILABLE", "no eligible observation with a finite forward label"
        availability[name] = {"status": status, "reason": reason}

    return {
        "targets": targets,
        "common_mask": common_mask,
        "coverage": coverage,
        "availability": availability,
        "estimates": {
            "sigma_raw": sigma_raw, "beta": beta, "sigma_res": sigma_res,
            "raw_observations": raw_count, "paired_observations": paired_count,
            "benchmark_forward": benchmark_forward,
        },
        "metadata": {
            "version": "factor-eval.targets/v1", "horizon_days": h, "entry_lag": entry_lag,
            "lookback_observations": lookback, "min_samples": min_samples,
            "benchmark_symbol": benchmark_symbol, "benchmark_present": benchmark_present,
            "price_basis": "daily open; not mid or an assumed executable fill",
            "estimate_price_basis": "completed daily close observations through signal date t",
            "target_interval": "open[t+entry_lag] to open[t+entry_lag+h] for every target",
            "historical_return": "R_h(t)=log(close[t]/close[t-h]); no sqrt(h) scaling",
            "formulas": {
                "raw_rtf": "log(open[t+entry_lag+h]/open[t+entry_lag])",
                "normed_rtf": "raw_rtf / sigma_raw(t)",
                "res_rtf": "raw_rtf - beta(t) * benchmark_raw_rtf",
                "normed_res_rtf": "res_rtf / sigma_res(t)",
                "sigma_raw": "rolling sample std of historical R_h; ddof=1",
                "beta": "rolling cov(R_h,benchmark_R_h) / var(benchmark_R_h); same finite pairs",
                "sigma_res": "sqrt(var(R_h)-cov(R_h,benchmark_R_h)^2/var(benchmark_R_h)); same finite pairs; ddof=1",
            },
            "units": {"raw_rtf": "log return", "normed_rtf": "dimensionless",
                      "res_rtf": "log return", "normed_res_rtf": "dimensionless"},
            "residual_scale_definition": "historical OLS residual sample dispersion with an intercept, not an unbiased n-2 regression variance",
            "coverage_denominator": "all decision-time eligible cells, including unavailable forward-label tails and estimation warmup",
            "common_cells": int(common_mask.to_numpy().sum()),
            "common_dates": int(common_mask.any(axis=1).sum()),
            "scope": "single-factor target diagnostics; excludes fees, funding, impact and strategy combinations",
        },
    }
