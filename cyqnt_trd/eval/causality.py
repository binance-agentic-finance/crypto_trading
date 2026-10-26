"""Sampled prefix invariance: future panel rows must not change past signals.

This catches common shifts, centred windows, and full-sample normalisation.
It cannot prove that upstream data or external state was point-in-time correct.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from .bundle import PRICE_FIELDS
from .matrix import Check, FAIL, NA, PASS


def check_prefix_invariance(factor, panel, signal):
    basis = "required: sampled past signals agree after truncating future input rows"
    if not callable(factor):
        return Check("causality_prefix", np.nan, 1.0, ">=", NA, basis,
                     "precomputed DataFrame has no callable to audit; causal provenance is unverified")
    cutoffs = sorted(set(int(v) for v in np.linspace(max(2, len(panel.index) // 4),
                                                   len(panel.index) - 2, 9)))
    compared = 0
    for end in cutoffs:
        if end < 1 or end >= len(panel.index):
            continue
        prefix = replace(panel, **{name: getattr(panel, name).iloc[:end + 1].copy()
                                   for name in (*PRICE_FIELDS, "funding", "mask")},
                         meta=dict(panel.meta))
        try:
            actual = factor(prefix)
            if not isinstance(actual, pd.DataFrame):
                raise TypeError("factor must return a DataFrame")
            if not actual.index.isin(prefix.index).all():
                return Check("causality_prefix", 0.0, 1.0, ">=", FAIL, basis,
                             "factor returned timestamps outside the supplied prefix")
            expected = signal.iloc[:end + 1]
            actual = actual.reindex(index=expected.index, columns=expected.columns).astype(float)
            eligible = prefix.mask.to_numpy()
            left, right = expected.to_numpy()[eligible], actual.to_numpy()[eligible]
            compared += int(np.isfinite(left).sum())
            # Raw relative tolerance is unsafe: adding 1e12 to a score must
            # not hide future-dependent changes of order 1 in that score.
            scale = expected.where(prefix.mask).std(axis=1).fillna(0).to_numpy()[:, None]
            tolerance = np.broadcast_to(1e-12 + 1e-9 * scale, expected.shape)[eligible]
            with np.errstate(invalid="ignore"):
                same = ((left == right) | (np.isnan(left) & np.isnan(right))
                        | (np.isfinite(left) & np.isfinite(right) & (np.abs(left - right) <= tolerance)))
            if not same.all():
                return Check("causality_prefix", 0.0, 1.0, ">=", FAIL, basis,
                             f"past signal changes when input ends at {prefix.index[-1].isoformat()}")
        except Exception as exc:
            return Check("causality_prefix", np.nan, 1.0, ">=", NA, basis,
                         f"prefix evaluation failed ({type(exc).__name__}); causal provenance is unverified")
    return Check("causality_prefix", 1.0 if compared else np.nan, 1.0, ">=",
                 PASS if compared else NA, basis,
                 f"{len(cutoffs)} cutoffs; sampled check only, upstream availability still requires audit")
