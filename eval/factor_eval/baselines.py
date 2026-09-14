"""The five things a new factor has to beat before it is interesting.

None of these is clever. That is the point: each is one line of public data, and
in the reference study (`eval/alpha101_crypto/`) several of them out-scored most
of the 101 published formulas out of sample. A factor that cannot separate itself
from ``B_size`` is not a finding, it is a re-parameterisation of "hold the big
names".

Definitions are shared with the reference study so the two are comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def baseline_signals(panel) -> dict[str, pd.DataFrame]:
    close = panel.close
    ret = np.log(close / close.shift(1))
    return {
        # size: 30-day dollar volume, lagged one day (a near-static big/small tilt)
        "B_size": panel.quote_volume.rolling(30, min_periods=30).mean().shift(1),
        # low volatility: the crypto cross-section's most persistent style
        "B_lowvol10": -ret.rolling(10, min_periods=10).std(),
        # short-term reversal
        "B_rev5": -np.log(close / close.shift(5)),
        # medium-term momentum
        "B_mom20": np.log(close / close.shift(20)),
        # carry: who has been paying to hold the position
        "B_funding7": -(panel.funding / close).rolling(7, min_periods=7).sum(),
    }


def cross_sectional_rank(frame: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank inside the eligible set, demeaned and unit-scaled per date."""
    ranked = frame.replace([np.inf, -np.inf], np.nan).where(mask).rank(axis=1, pct=True)
    centred = ranked.sub(ranked.mean(axis=1), axis=0)
    return centred.div(centred.std(axis=1).replace(0, np.nan), axis=0)


def residualise(signal: pd.DataFrame, others: dict[str, pd.DataFrame],
                mask: pd.DataFrame, min_assets: int = 5) -> pd.DataFrame:
    """Per-date cross-sectional OLS of the signal on the baselines; keep the residual.

    Fitted per date rather than with frozen coefficients: this answers "on the day
    itself, how much of the ranking is not already the baselines?", which is the
    question the incremental gate asks. It is deliberately the *harsher* of the two
    readings — a frozen-coefficient projection leaves more residual.

    A saturated fit has no residual degrees of freedom. An exactly replicated
    signal also has no residual information. Both remain unavailable instead of
    allowing later ranking to magnify floating-point roundoff into a new signal.
    """
    y = cross_sectional_rank(signal, mask)
    xs = [cross_sectional_rank(v, mask) for v in others.values()]
    resid = pd.DataFrame(np.nan, index=y.index, columns=y.columns)
    for t in y.index:
        yt = y.loc[t]
        cols = [x.loc[t] for x in xs]
        ok = yt.notna()
        for c in cols:
            ok &= c.notna()
        if int(ok.sum()) < min_assets:
            continue
        A = np.column_stack([np.ones(int(ok.sum()))] + [c[ok].to_numpy() for c in cols])
        target = yt[ok].to_numpy()
        beta, _, rank, _ = np.linalg.lstsq(A, target, rcond=None)
        if len(target) <= rank:
            continue
        error = target - A @ beta
        # A backward-error scale accounts for both the target and the fitted
        # projection, including an ill-conditioned design's larger coefficients.
        scale = max(1.0, float(np.linalg.norm(target, ord=np.inf)),
                    float(np.linalg.norm(A, ord=np.inf) * np.linalg.norm(beta, ord=np.inf)))
        tolerance = 32 * np.finfo(float).eps * max(A.shape) * scale
        if float(np.linalg.norm(error, ord=np.inf)) <= tolerance:
            continue
        resid.loc[t, ok[ok].index] = error
    return resid
