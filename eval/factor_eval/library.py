"""Factor library: where candidates come from, and which of them are duplicates.

Two sources feed the matrix.

`capability_factors` (see `factor_eval.capability_registry`)
    The production path -- operators declared by the capability SDK.

`alpha101_factors`
    The reference study in `eval/alpha101_crypto/`. A hundred and one published
    formulas make a fast, adversarial smoke test of the whole chain: they are not
    tuned for this panel, most of them are weak, and several are near-copies of
    each other, which is exactly the situation de-duplication has to survive.

De-duplication is the part that matters for construction. Averaging five copies of
one signal buys no diversification while looking like a five-factor book, and both
sources contain near-copies (alpha101 by construction; capabilities because `raw`
and `ts_zscore` variants of a slow indicator often rank almost identically). The
correlation is measured on **dev only**, so dropping a candidate never consults
data the strategy is later scored on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import cross_sectional_rank
from .engine import DEFAULT_SPLITS, _split_bounds

__all__ = ["alpha101_factors", "sample_factors", "rank_corr", "deduplicate"]


def alpha101_factors(panel, *, mode: str = "paper", names=None) -> dict[str, pd.DataFrame]:
    """Compute the Alpha101 reference formulas on a panel.

    Shape adaptation only -- the formulas stay in `eval/alpha101_crypto/factors.py`
    and are not copied here. All 101 are computed in one pass because that module
    shares operator state across the set; `names` then selects from the result.
    """
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]      # eval/
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from alpha101_crypto.factors import compute_factors

    raw = {f: getattr(panel, f) for f in
           ("open", "high", "low", "close", "volume", "quote_volume")}
    frames, _meta = compute_factors(raw, panel.mask, mode=mode)
    if names is not None:
        missing = sorted(set(names) - set(frames))
        if missing:
            raise KeyError(f"unknown alpha101 factors: {missing}")
        frames = {n: frames[n] for n in names}
    return frames


def sample_factors(factors: dict, k: int, seed: int) -> dict:
    """Reproducible random subset, drawn without replacement.

    Sorted keys before drawing so the sample depends on the seed and not on dict
    insertion order.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    keys = sorted(factors)
    if k > len(keys):
        raise ValueError(f"asked for {k} factors, only {len(keys)} available")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(keys), size=k, replace=False)
    return {keys[int(i)]: factors[keys[int(i)]] for i in sorted(chosen)}


def _as_frame(factor, panel) -> pd.DataFrame:
    return factor(panel) if callable(factor) else factor


def rank_corr(factors: dict, panel, *, split: str = "dev", splits=None) -> pd.DataFrame:
    """Average cross-sectional rank correlation between candidates, on one split.

    Cross-sectional rank rather than raw level, because that is what the matrix and
    the combination step both score -- two factors with wildly different units can
    still produce the same ordering every day, and that is what makes them
    duplicates.
    """
    lo, hi = _split_bounds(DEFAULT_SPLITS if splits is None else splits)[split]
    window = (panel.index >= lo) & (panel.index <= hi)
    ranks = {name: cross_sectional_rank(_as_frame(f, panel), panel.mask).loc[window]
             for name, f in factors.items()}
    names = list(ranks)
    out = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        out.loc[a, a] = 1.0
        for b in names[i + 1:]:
            value = float(ranks[a].corrwith(ranks[b], axis=1).mean())
            out.loc[a, b] = out.loc[b, a] = value
    return out


def deduplicate(factors: dict, panel, *, max_corr: float = 0.8,
                strength: dict | None = None, split: str = "dev", splits=None
                ) -> tuple[dict, dict]:
    """Drop near-duplicate candidates; return `(kept, dropped -> reason)`.

    Candidates are considered strongest-first (`strength`, normally `|dev IC|` from
    the scorecards) and a candidate is dropped when its rank correlation with an
    already-kept candidate exceeds `max_corr`. Without `strength` the input order
    decides, which is stable but arbitrary -- pass the scorecards when you have them.

    `max_corr=0.8` is a convention, not a measurement. Above it two factors move
    together on nearly every date and contribute one bet, not two.
    """
    if not 0 < max_corr <= 1:
        raise ValueError("max_corr must be in (0, 1]")
    if len(factors) < 2:
        return dict(factors), {}
    corr = rank_corr(factors, panel, split=split, splits=splits)
    strength = strength or {}
    order = sorted(factors, key=lambda n: (-abs(strength.get(n, 0.0))
                                           if np.isfinite(strength.get(n, 0.0)) else 0.0,
                                           list(factors).index(n)))
    kept, dropped = {}, {}
    for name in order:
        clash = next((k for k in kept
                      if np.isfinite(corr.loc[name, k]) and abs(corr.loc[name, k]) > max_corr),
                     None)
        if clash is None:
            kept[name] = factors[name]
        else:
            dropped[name] = (f"rank corr {corr.loc[name, clash]:+.2f} with {clash} "
                             f"on {split} (> {max_corr})")
    # Preserve the caller's ordering in the kept set; the strength ordering was
    # only needed to decide who survives a clash.
    return {n: factors[n] for n in factors if n in kept}, dropped
