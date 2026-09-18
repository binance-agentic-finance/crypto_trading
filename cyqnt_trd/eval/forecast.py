"""Stage 4 and 5: factors -> expected return (mu) -> target weights (W).

These are two separate stages on purpose. Collapsing them -- going from ranked
factors straight to weights -- hides the one quantity a strategy is actually
claiming: how much each name is expected to earn, relative to the others. Keeping
`mu` explicit means the claim can be scored on its own before any position sizing
decides what to do about it.

Both stages are stateless maps. `mu[t]` uses factor values at `t` and the
cross-section at `t`; `W[t]` uses `mu[t]` and the mask at `t`. Nothing carries
across cells, so `W`'s last row is a live order and `check_stateless` can verify
the claim by truncation.

Coefficients are **fitted on dev and frozen**. Fitting on the whole sample is a
future function even when the model looks innocent: standardising by a full-sample
mean already leaks. `fit_betas` only ever reads dev, and returns numbers that the
forecast then applies to every split unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .baselines import cross_sectional_rank
from .engine import DEFAULT_SPLITS, _split_bounds
from .statistics import daily_cross_sectional
from .targets import build_targets

__all__ = ["Betas", "fit_betas", "make_forecast", "equal_forecast",
           "positions_from_forecast", "sign_positions", "quantile_positions"]


@dataclass(frozen=True)
class Betas:
    """Frozen forecast coefficients plus the evidence they were fitted on."""

    weights: dict[str, float]
    method: str
    split: str = "dev"
    diagnostics: dict = field(default_factory=dict)

    def describe(self) -> str:
        parts = ", ".join(f"{k}={v:+.3f}" for k, v in sorted(self.weights.items()))
        return f"{self.method} on {self.split}: {parts}"


def _dev_window(panel, splits, split: str = "dev"):
    lo, hi = _split_bounds(DEFAULT_SPLITS if splits is None else splits)[split]
    return (panel.index >= lo) & (panel.index <= hi)


def fit_betas(factors: Mapping[str, pd.DataFrame], panel, *, method: str = "ic",
              h: int = 3, entry_lag: int = 2, splits=None, split: str = "dev",
              shrink: float = 0.0, min_assets: int = 5) -> Betas:
    """Fit forecast coefficients on one split and freeze them.

    `ic`
        beta_k = mean RankIC of factor k on the split. Carries the sign (which is
        what direction freezing means) and the magnitude. Simple and stable.
    `equal`
        beta_k = sign of the mean RankIC. Magnitude discarded -- the safer default
        when the ICs are all within noise of each other, because an in-sample IC
        ratio is mostly estimation error.
    `ridge`
        Cross-sectionally standardised factors regressed on the forward return with
        an L2 penalty, pooled over the split. Uses more of the covariance structure
        and needs more data to be worth it.

    `shrink` pulls the fitted weights toward equal weighting: `(1-s)*fitted +
    s*equal`. Shrinkage is the cheap defence against fitting the noise in an IC
    estimate, and 0 means "trust the fit completely", which is rarely right.
    """
    if method not in ("ic", "equal", "ridge"):
        raise ValueError(f"unknown method {method!r}")
    if not 0.0 <= shrink <= 1.0:
        raise ValueError("shrink must be in [0, 1]")
    window = _dev_window(panel, splits, split)
    target = build_targets(panel, h, entry_lag)["targets"]["raw_rtf"]

    ics = {}
    for name, frame in factors.items():
        series = daily_cross_sectional(frame, target, panel.mask,
                                       min_assets=min_assets)["rank_ic"]
        ics[name] = float(series.loc[window].mean())

    if method == "ridge":
        fitted = _ridge(factors, target, panel, window, min_assets)
    elif method == "ic":
        fitted = {k: (v if np.isfinite(v) else 0.0) for k, v in ics.items()}
    else:
        fitted = {k: (1.0 if (np.isfinite(v) and v >= 0) else -1.0) for k, v in ics.items()}

    scale = sum(abs(v) for v in fitted.values())
    if scale <= 0:
        raise ValueError(f"every fitted coefficient is zero on {split}; "
                         f"mean RankIC was {ics}")
    fitted = {k: v / scale for k, v in fitted.items()}
    if shrink > 0:
        equal = {k: (1.0 if fitted[k] >= 0 else -1.0) / len(fitted) for k in fitted}
        fitted = {k: (1 - shrink) * fitted[k] + shrink * equal[k] for k in fitted}
    return Betas(weights=fitted, method=method, split=split,
                 diagnostics={"mean_rank_ic": ics, "shrink": shrink, "h": h,
                              "entry_lag": entry_lag})


def _ridge(factors, target, panel, window, min_assets, alpha: float = 1.0) -> dict:
    """Pooled cross-sectional ridge on the split, standardised per cell."""
    names = list(factors)
    stacked, labels = [], []
    mask = panel.mask & window[:, None] if isinstance(window, np.ndarray) else panel.mask
    for t in panel.index[window]:
        row_mask = panel.mask.loc[t]
        y = target.loc[t].where(row_mask)
        cols = []
        for name in names:
            x = factors[name].loc[t].where(row_mask)
            x = (x - x.mean()) / (x.std() or np.nan)
            cols.append(x)
        frame = pd.concat([*cols, y], axis=1).dropna()
        if len(frame) < min_assets:
            continue
        stacked.append(frame.iloc[:, :-1].to_numpy(float))
        labels.append(frame.iloc[:, -1].to_numpy(float))
    if not stacked:
        raise ValueError("ridge found no cell with enough paired observations")
    X = np.vstack(stacked)
    y = np.concatenate(labels)
    beta = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)
    return dict(zip(names, (float(b) for b in beta)))


def make_forecast(betas: Betas, *, standardise: str = "rank") -> Callable:
    """Build the stage-4 map `factors, panel -> mu` from frozen coefficients.

    `standardise="rank"` puts every factor on a common cross-sectional scale
    before weighting; without it a factor measured in basis points and one measured
    in standard deviations would be combined by their units rather than by beta.
    """
    if standardise not in ("rank", "zscore", "none"):
        raise ValueError(f"unknown standardise {standardise!r}")

    def forecast(factors: Mapping[str, pd.DataFrame], panel) -> pd.DataFrame:
        missing = sorted(set(betas.weights) - set(factors))
        if missing:
            raise KeyError(f"forecast was fitted with {missing} but they are not in "
                           f"this factor set")
        mu, used = None, None
        for name, beta in betas.weights.items():
            if beta == 0:
                continue
            x = factors[name]
            if standardise == "rank":
                x = cross_sectional_rank(x, panel.mask)
            elif standardise == "zscore":
                x = x.where(panel.mask)
                x = x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1).replace(0, np.nan), axis=0)
            present = x.notna()
            term = (x * beta).fillna(0.0)
            mu = term if mu is None else mu.add(term, fill_value=0.0)
            share = present.astype(float) * abs(beta)
            used = share if used is None else used.add(share, fill_value=0.0)
        if mu is None:
            raise ValueError("every coefficient is zero; there is no forecast")
        # A factor missing a cell does not vote there, and the others are
        # renormalised by the weight that actually participated. Treating the gap
        # as 0 would cast a "neutral" vote on that factor's behalf.
        return (mu / used.replace(0, np.nan)).where(panel.mask).round(12)

    forecast.betas = betas
    forecast.__doc__ = f"frozen forecast: {betas.describe()}"
    return forecast


def equal_forecast(factors: Mapping[str, pd.DataFrame], panel, *, h: int = 3,
                   entry_lag: int = 2, splits=None) -> pd.DataFrame:
    """Stage 4 with no fitting beyond the dev-frozen direction of each factor."""
    betas = fit_betas(factors, panel, method="equal", h=h, entry_lag=entry_lag,
                      splits=splits)
    return make_forecast(betas)(factors, panel)


# ------------------------------------------------------------------- stage 5
def positions_from_forecast(mu: pd.DataFrame, mask: pd.DataFrame, *, gross: float = 1.0,
                            market_neutral: bool = True, max_weight: float | None = 0.35,
                            min_assets: int = 5, passes: int = 8) -> pd.DataFrame:
    """mu -> target weights: mask, demean, normalise to gross, cap, renormalise.

    The order is the part that goes wrong. `max_weight` means "at most this share
    of equity in one name", which is only meaningful **after** normalisation.
    Capping the raw scores first squashes the signal into two near-equal blocks and
    quietly gives away most of the spread while the weights still look reasonable.

    Capping breaks neutrality and the gross target, renormalising can push another
    name over the cap, so it iterates. If it has not converged, the residual cap
    breach is kept rather than the neutrality -- a net exposure you did not ask for
    is the worse of the two.

    **`gross` is a target, not a guarantee.** With `n` eligible names the most the
    book can hold is `n * max_weight`, so a cell with 6 names and a 0.15 cap tops
    out at 0.90 however high `gross` is set. The cap wins, quietly, and the result
    is an under-invested book -- check `w.abs().sum(axis=1)` rather than assuming
    it equals `gross`.

    Cells with fewer than `min_assets` eligible names are flat, explicitly 0. Left
    as NaN they would read as "hold whatever you had", which is how a stale
    position survives the cross-section collapsing underneath it.
    """
    if gross <= 0:
        raise ValueError("gross must be positive")
    if max_weight is not None and not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1]")
    frame = mu.where(mask)
    enough = frame.notna().sum(axis=1) >= min_assets
    cap = None if max_weight is None else max_weight * gross
    w = np.nan_to_num(frame.to_numpy(dtype=float), nan=0.0)
    live = mask.to_numpy(dtype=bool)
    w = np.where(live, w, 0.0)

    if market_neutral:
        count = live.sum(axis=1, keepdims=True)
        mean = np.where(count > 0, w.sum(axis=1, keepdims=True) / np.maximum(count, 1), 0.0)
        w = np.where(live, w - mean, 0.0)
        # Both sides are projected to the same total, so `sum(w) == 0` holds by
        # construction however hard the cap bites. Scaling the whole vector and
        # then capping cannot do that: the clip removes more from whichever side
        # holds the larger names, and the book ends up with a net exposure nobody
        # asked for.
        #
        # That shared total is whatever the *weaker* side can actually carry. A
        # demeaned cross-section splits unevenly -- two longs against four shorts is
        # normal on a panel this narrow -- and two names capped at 0.15 cannot hold
        # gross/2 = 0.5 whatever the other side does. So each side is projected on
        # its own and the stronger one is then scaled down to match; shrinking never
        # breaches the cap, so the result satisfies all three constraints and is
        # simply under-invested when the cap makes the target unreachable. The
        # caller sees that in `w.abs().sum(axis=1)` instead of getting a supposedly
        # market-neutral book with a standing directional bet in it.
        #
        # Deliberately not estimated from the count of names per side: a weight of
        # -1e-17 counts as a short and does not carry one, which sized the long side
        # 50% too large and left a 0.15 net exposure.
        half = np.full((w.shape[0], 1), gross / 2.0)
        longs = _project_side(w, np.maximum(w, 0.0), half, cap, live, passes)
        shorts = _project_side(w, np.maximum(-w, 0.0), half, cap, live, passes)
        achieved = np.minimum(longs.sum(axis=1, keepdims=True),
                              shorts.sum(axis=1, keepdims=True))
        w = _rescale(longs, achieved) - _rescale(shorts, achieved)
    else:
        w = _project_side(w, np.abs(w), np.full((w.shape[0], 1), gross), cap,
                          live, passes) * np.sign(w)

    out = pd.DataFrame(w, index=mu.index, columns=mu.columns)
    return out.where(enough, 0.0).fillna(0.0)


def _rescale(magnitude: np.ndarray, target: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Scale each row down to `target`. Only ever shrinks, so a cap still holds."""
    total = magnitude.sum(axis=1, keepdims=True)
    factor = np.divide(target, total, out=np.ones_like(total), where=total > tol)
    return magnitude * np.minimum(factor, 1.0)


def _project_side(_w, magnitude: np.ndarray, target, cap: float | None,
                  live: np.ndarray, passes: int, tol: float = 1e-12) -> np.ndarray:
    """Scale non-negative magnitudes to sum to `target`, respecting `cap`.

    Repeatedly normalises and then pushes the shortfall onto the entries still
    strictly below the cap. Without that redistribution the loop would exit on a
    clip and leave the book under-invested -- measured at 0.65 against a target of
    1.0 on a seven-name cross-section with a 0.15 cap, a third of the intended
    exposure missing while every individual weight still looked reasonable.
    """
    m = np.where(live, magnitude, 0.0)
    target = np.broadcast_to(np.asarray(target, dtype=float).reshape(-1, 1),
                             (m.shape[0], 1))
    for _ in range(passes):
        total = m.sum(axis=1, keepdims=True)
        m = np.divide(m, total, out=np.zeros_like(m), where=total > tol) * target
        if cap is None or m.max(initial=0.0) <= cap + tol:
            break
        m = np.minimum(m, cap)
        deficit = target - m.sum(axis=1, keepdims=True)
        free = live & (m < cap - tol) & (m > tol)
        headroom = (m * free).sum(axis=1, keepdims=True)
        grow = np.divide(deficit, headroom, out=np.zeros_like(deficit),
                         where=(headroom > tol) & (deficit > tol))
        m = np.minimum(np.where(free, m * (1.0 + grow), m), cap)
    return m


def sign_positions(mu: pd.DataFrame, mask: pd.DataFrame, *, gross: float = 1.0,
                   min_assets: int = 5) -> pd.DataFrame:
    """Equal-weight long/short on the sign of mu."""
    side = np.sign(mu.where(mask))
    return positions_from_forecast(side, mask, gross=gross, market_neutral=True,
                                   max_weight=None, min_assets=min_assets)


def quantile_positions(mu: pd.DataFrame, mask: pd.DataFrame, *, quantile: float = 0.3,
                       gross: float = 1.0, min_assets: int = 5) -> pd.DataFrame:
    """Long the top quantile, short the bottom, equal weight inside each leg.

    On a cross-section this narrow a 0.3 quantile is two or three names per leg, so
    the result is closer to a concentrated pair trade than to a diversified
    portfolio. That is a property of the universe, not of the rule.
    """
    if not 0 < quantile < 0.5:
        raise ValueError("quantile must be in (0, 0.5)")
    ranks = mu.where(mask).rank(axis=1, pct=True)
    side = pd.DataFrame(0.0, index=mu.index, columns=mu.columns)
    side = side.where(ranks.isna() | (ranks > quantile), -1.0)
    side = side.where(ranks.isna() | (ranks < 1 - quantile), 1.0)
    return positions_from_forecast(side.where(mask), mask, gross=gross,
                                   market_neutral=True, max_weight=None,
                                   min_assets=min_assets)
