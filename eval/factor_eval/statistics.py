"""Date-weighted factor relations with calendar-preserving uncertainty estimates.

Cross-sectional IC is an equal-date average, with equal asset weights within
each date. Pooled relations also give each observed date total weight one, but
are not IC: they can include time-series level effects and narrow cross-sections.
Bootstrap draws move whole date blocks, never individual asset observations.
Callers can reuse ``moving_block_indices`` across targets on the same date grid.
"""

from statistics import NormalDist

import numpy as np
import pandas as pd


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _confidence(value):
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("confidence must be between zero and one")
    return float(value)


def _regular_series(values):
    """Restore omitted calendar dates; an array must already contain its gaps."""
    series = values.astype(float).copy() if isinstance(values, pd.Series) else pd.Series(values, dtype=float)
    if not series.index.is_unique or not series.index.is_monotonic_increasing:
        raise ValueError("date observations must have a unique, increasing index")
    if isinstance(series.index, pd.DatetimeIndex) and len(series):
        if series.index.hasnans:
            raise ValueError("date observations cannot contain NaT")
        grid = pd.date_range(series.index[0], series.index[-1], freq="D")
        if not series.index.isin(grid).all():
            raise ValueError("date observations must lie on a daily time grid")
        series = series.reindex(grid)
    return series.replace([np.inf, -np.inf], np.nan)


def _mean(values, axis=None):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    total = np.where(valid, values, 0.0).sum(axis=axis)
    result = np.full(np.shape(total), np.nan, dtype=float)
    np.divide(total, count, out=result, where=count > 0)
    return float(result) if result.ndim == 0 else result


def _block_length(n_dates, block_length):
    return (max(1, int(np.ceil(max(n_dates, 1) ** (1 / 3)))) if block_length is None
            else _integer(block_length, "block_length", 1))


def moving_block_indices(n_dates, block_length=None, n_bootstrap=200, seed=0):
    """Draw overlapping, non-circular calendar blocks as a (draw, date) array.

    At least two full blocks must fit in the observed calendar span. Missing
    dates stay in the grid and therefore retain their positions inside blocks.
    One returned row must be used for every asset/target in that bootstrap draw.
    """
    n_dates = _integer(n_dates, "n_dates", 1)
    n_bootstrap = _integer(n_bootstrap, "n_bootstrap", 1)
    length = _block_length(n_dates, block_length)
    if n_dates < 2 * length:
        raise ValueError("bootstrap requires at least two complete calendar blocks")
    n_blocks = int(np.ceil(n_dates / length))
    starts = np.random.default_rng(seed).integers(0, n_dates - length + 1,
                                                size=(n_bootstrap, n_blocks))
    return (starts[..., None] + np.arange(length)).reshape(n_bootstrap, -1)[:, :n_dates]


def hac_mean_ci(values, lags=None, confidence=0.95):
    """Normal-approximation Bartlett HAC interval, retaining missing-date gaps.

    The variance estimates the sample mean using observed-count normalization;
    missing dates have zero influence rather than joining their neighbours.
    These intervals are estimates, not distribution-free coverage guarantees.
    """
    confidence = _confidence(confidence)
    x = _regular_series(values).to_numpy()
    valid = np.isfinite(x)
    n = int(valid.sum())
    mean = _mean(x)
    lag = (int(4 * (len(x) / 100) ** (2 / 9)) if lags is None
           else _integer(lags, "lags", 0))
    lag = min(lag, max(0, len(x) - 1))
    result = {"mean": mean, "ci_low": np.nan, "ci_high": np.nan,
              "std_error": np.nan, "t": np.nan, "n_dates": n,
              "n_dates_total": len(x), "lags": lag, "confidence": confidence,
              "method": "Bartlett HAC mean; normal approximation; calendar gaps retained"}
    if n < 3:
        result["reason"] = "fewer than three finite date observations"
        return result
    influence = np.where(valid, x - mean, 0.0)
    variance_sum = float(np.sum(influence * influence))
    for offset in range(1, lag + 1):
        variance_sum += 2 * (1 - offset / (lag + 1)) * float(np.sum(influence[offset:] * influence[:-offset]))
    if variance_sum < 0:
        result["reason"] = "numerically invalid HAC variance"
        return result
    se = np.sqrt(variance_sum) / n
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    result.update(ci_low=float(mean - z * se), ci_high=float(mean + z * se),
                  std_error=float(se), t=float(mean / se) if se > 0 else np.nan)
    return result


def _draws(n_dates, length, n_bootstrap, seed, supplied=None):
    if n_bootstrap == 0 and supplied is None:
        return None, "bootstrap disabled"
    if n_dates < 2 * length:
        return None, "fewer than two complete calendar blocks"
    if supplied is not None:
        indices = np.asarray(supplied)
        if (indices.ndim != 2 or indices.shape[1] != n_dates or not len(indices)
                or not np.issubdtype(indices.dtype, np.integer)
                or (indices < 0).any() or (indices >= n_dates).any()):
            raise ValueError("bootstrap_indices must be an integer draw-by-date matrix on this grid")
        return indices, ""
    return moving_block_indices(n_dates, length, n_bootstrap, seed), ""


def _interval(draws, confidence, finite_dates, minimum_dates=3):
    finite = np.asarray(draws, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite_dates < minimum_dates or len(finite) < 2:
        return [np.nan, np.nan], len(finite)
    alpha = (1 - confidence) / 2
    return np.quantile(finite, [alpha, 1 - alpha]).tolist(), len(finite)


def summarize_time_mean(values, block_length=None, n_bootstrap=200, seed=0,
                        *, confidence=0.95, bootstrap_indices=None):
    """Equal-date mean and block-bootstrap interval, including explicit zeroes.

    NaN means unobserved; zero activation/zero return must be supplied as zero.
    Pass ``n_bootstrap=0`` for point estimates plus HAC without resampling.
    """
    n_bootstrap = _integer(n_bootstrap, "n_bootstrap", 0)
    confidence = _confidence(confidence)
    series = _regular_series(values)
    x = series.to_numpy()
    n = int(np.isfinite(x).sum())
    length = _block_length(len(x), block_length)
    indices, reason = _draws(len(x), length, n_bootstrap, seed, bootstrap_indices)
    ci, valid_draws = [np.nan, np.nan], 0
    if indices is not None:
        ci, valid_draws = _interval(_mean(x[indices], axis=1), confidence, n, max(3, 2 * length))
        if n < max(3, 2 * length):
            reason = "fewer than two blocks worth of finite date observations"
    return {"mean": _mean(x), "ci_low": ci[0], "ci_high": ci[1],
            "n_dates": n, "n_dates_total": len(x), "block_length": length,
            "observed_dates_per_block_length": n / length,
            "confidence": confidence, "n_bootstrap": len(indices) if indices is not None else 0,
            "n_bootstrap_valid": valid_draws, "reason": reason,
            "method": "equal-date mean; moving calendar-block percentile bootstrap; missing gaps retained",
            "hac": hac_mean_ci(series, lags=length - 1, confidence=confidence)}


def _paired(signal, target, mask, sign):
    if sign not in (-1, 1) or isinstance(sign, (bool, np.bool_)):
        raise ValueError("sign must be +1 or -1")
    for name, frame in (("signal", signal), ("target", target), ("mask", mask)):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a DataFrame")
        if not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError(f"{name} labels must be unique")
        if not frame.index.equals(signal.index) or not frame.columns.equals(signal.columns):
            raise ValueError("signal, target and mask must have identical date and asset labels")
    if mask.isna().to_numpy().any() or not mask.isin([False, True]).to_numpy().all():
        raise ValueError("mask must explicitly contain booleans or 0/1")
    x, y = signal.to_numpy(dtype=float) * sign, target.to_numpy(dtype=float)
    valid = mask.to_numpy(dtype=bool) & np.isfinite(x) & np.isfinite(y)
    return x, y, valid


def _correlation_rows(x, y, valid, min_assets):
    count = valid.sum(axis=1)
    divisor = np.maximum(count, 1)[:, None]
    dx = np.where(valid, x - np.where(valid, x, 0).sum(axis=1)[:, None] / divisor, 0.0)
    dy = np.where(valid, y - np.where(valid, y, 0).sum(axis=1)[:, None] / divisor, 0.0)
    scale = np.sqrt((dx * dx).sum(axis=1) * (dy * dy).sum(axis=1))
    result = np.full(len(x), np.nan)
    np.divide((dx * dy).sum(axis=1), scale, out=result,
              where=(scale > 0) & (count >= min_assets))
    return np.clip(result, -1, 1)


def daily_cross_sectional(signal, target, mask, sign=1, min_assets=5):
    """Daily Pearson/Spearman IC; thin or constant cross-sections remain NA."""
    min_assets = _integer(min_assets, "min_assets", 2)
    x, y, valid = _paired(signal, target, mask, sign)
    ranks_x = pd.DataFrame(np.where(valid, x, np.nan)).rank(axis=1).to_numpy()
    ranks_y = pd.DataFrame(np.where(valid, y, np.nan)).rank(axis=1).to_numpy()
    daily = pd.DataFrame({"rank_ic": _correlation_rows(ranks_x, ranks_y, valid, min_assets),
                          "pearson_ic": _correlation_rows(x, y, valid, min_assets),
                          "n_assets": valid.sum(axis=1)}, index=signal.index)
    grid = _regular_series(daily.rank_ic).index
    daily = daily.reindex(grid)
    daily["n_assets"] = daily.n_assets.fillna(0).astype(int)
    return daily


def _daily_moments(signal, target, mask, sign):
    x, y, valid = _paired(signal, target, mask, sign)
    n = valid.sum(axis=1)
    # A fixed origin avoids catastrophic cancellation for large signal levels.
    x = np.where(valid, x - (x[valid][0] if valid.any() else 0), 0.0)
    y = np.where(valid, y - (y[valid][0] if valid.any() else 0), 0.0)
    moments = pd.DataFrame({key: np.divide(value.sum(axis=1), n,
                                         out=np.full(len(n), np.nan), where=n > 0)
                            for key, value in {"x": x, "y": y, "xx": x * x,
                                               "yy": y * y, "xy": x * y}.items()},
                           index=signal.index)
    return moments.reindex(_regular_series(moments.x).index), int(n.sum())


def _pooled_from_moments(m):
    cov = m["xy"] - m["x"] * m["y"]
    vx, vy = m["xx"] - m["x"] ** 2, m["yy"] - m["y"] ** 2
    shape = np.shape(cov)
    corr, slope = np.full(shape, np.nan), np.full(shape, np.nan)
    np.divide(cov, np.sqrt(np.maximum(vx, 0) * np.maximum(vy, 0)),
              out=corr, where=(vx > 0) & (vy > 0))
    np.divide(cov, vx, out=slope, where=vx > 0)
    return np.clip(corr, -1, 1), slope


def weighted_panel_relation(signal, target, mask, sign=1):
    """Pooled weighted correlation/OLS slope; explicitly not cross-sectional IC.

    Each nonempty date has total weight one, split equally among finite pairs.
    Slope is in target units per signed signal unit; no basis-point conversion
    is implicit. One-asset dates may contribute and are reported explicitly.
    """
    moments, n_pairs = _daily_moments(signal, target, mask, sign)
    corr, slope = _pooled_from_moments({key: _mean(moments[key]) for key in moments})
    return {"corr_pooled": float(corr), "slope": float(slope),
            "n_dates": int(moments.x.notna().sum()), "n_dates_total": len(moments),
            "n_pairs": n_pairs,
            "method": "pooled relation, not IC; equal date then equal asset; includes time-level effects"}


def summarize_relation(signal, target, mask, sign=1, min_assets=5,
                       block_length=None, n_bootstrap=200, seed=0, *,
                       confidence=0.95, bootstrap_indices=None):
    """Summarize one target/slice on its full date grid with shared date draws.

    ``ci`` maps rank_ic/pearson_ic/corr_pooled/slope to [low, high]. A narrow
    slice can have pooled estimates while IC is NA; their sample counts differ.
    Reuse ``bootstrap_indices`` across related calls for paired comparisons.
    """
    n_bootstrap = _integer(n_bootstrap, "n_bootstrap", 0)
    confidence = _confidence(confidence)
    daily = daily_cross_sectional(signal, target, mask, sign, min_assets)
    moments, n_pairs = _daily_moments(signal, target, mask, sign)
    pooled = _pooled_from_moments({key: _mean(moments[key]) for key in moments})
    n_rank, n_pearson = int(daily.rank_ic.notna().sum()), int(daily.pearson_ic.notna().sum())
    n_pooled = int(moments.x.notna().sum())
    length = _block_length(len(daily), block_length)
    indices, reason = _draws(len(daily), length, n_bootstrap, seed, bootstrap_indices)
    fields = ("rank_ic", "pearson_ic", "corr_pooled", "slope")
    ci = {name: [np.nan, np.nan] for name in fields}
    valid_draws = {name: 0 for name in fields}
    if indices is not None:
        draws = {name: _mean(daily[name].to_numpy()[indices], axis=1)
                 for name in ("rank_ic", "pearson_ic")}
        pooled_draws = _pooled_from_moments({key: _mean(moments[key].to_numpy()[indices], axis=1)
                                           for key in moments})
        draws.update(corr_pooled=pooled_draws[0], slope=pooled_draws[1])
        for name, count in zip(fields, (n_rank, n_pearson, n_pooled, n_pooled)):
            ci[name], valid_draws[name] = _interval(draws[name], confidence, count, max(3, 2 * length))
    return {"rank_ic": _mean(daily.rank_ic), "pearson_ic": _mean(daily.pearson_ic),
            "corr_pooled": float(pooled[0]), "slope": float(pooled[1]), "ci": ci,
            "hac": {name: hac_mean_ci(daily[name], lags=length - 1, confidence=confidence)
                    for name in ("rank_ic", "pearson_ic")},
            "n_dates": n_rank, "n_dates_pearson": n_pearson,
            "n_dates_pooled": n_pooled, "n_dates_total": len(daily),
            "ic_reason": "" if n_rank else "no dates have enough finite pairs and nonconstant cross-sections",
            "n_pairs": n_pairs,
            "n_ic_pairs": int(daily.loc[daily.rank_ic.notna(), "n_assets"].sum()),
            "n_assets_mean": _mean(daily.loc[daily.rank_ic.notna(), "n_assets"]),
            "n_assets_pooled_mean": n_pairs / n_pooled if n_pooled else np.nan,
            "min_assets": int(min_assets), "block_length": length,
            "observed_dates_per_block_length": n_rank / length,
            "ci_reasons": {name: "" if count >= max(3, 2 * length) else "fewer than two blocks worth of finite dates"
                           for name, count in zip(fields, (n_rank, n_pearson, n_pooled, n_pooled))},
            "n_bootstrap": len(indices) if indices is not None else 0,
            "n_bootstrap_valid": valid_draws, "confidence": confidence, "reason": reason,
            "method": "equal-date CS IC; pooled corr/slope are not IC; moving calendar-block percentile bootstrap",
            "pooled_method": "equal date then equal finite asset pair; includes time-level effects",
            "slope_units": "target units per signed signal unit"}
