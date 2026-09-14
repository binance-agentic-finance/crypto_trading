"""Bind a measured noise floor to its data, timing, and numerical protocol.

A universe name is not a data identity: prices, missing observations, or the
eligibility mask can change while that name stays the same. Hash the numerical
panel itself, then store the hash together with every engine setting that affects
the primary-horizon null distribution. File locations and descriptive metadata
are deliberately excluded so a copied bundle remains reproducible.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from . import engine
from .bundle import PRICE_FIELDS, Panel

CONTRACT_SCHEMA = "factor-eval.calibration/v1"


def panel_fingerprint(panel: Panel) -> str:
    """SHA256 of aligned axes and all values, independent of parquet encoding.

Float representations are normalized to little-endian float64, with canonical
NaNs and signed zeroes. Dtype-only changes therefore do not invalidate a floor;
changing even one price, funding observation, missing value, or mask cell does.
"""
    panel.validate()
    digest = hashlib.sha256()
    header = {"schema": "factor-eval.panel/v1", "symbols": panel.symbols,
              "shape": list(panel.close.shape), "fields": [*PRICE_FIELDS, "funding", "mask"]}
    digest.update(json.dumps(header, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8"))
    timestamps = panel.index.tz_convert("UTC").to_numpy(dtype="datetime64[ns]")
    digest.update(timestamps.astype("<M8[ns]").view("<i8").tobytes(order="C"))
    for name in (*PRICE_FIELDS, "funding"):
        values = getattr(panel, name).to_numpy(dtype="<f8", na_value=np.nan, copy=True)
        values[np.isnan(values)] = np.nan
        values[values == 0] = 0.0
        digest.update(values.tobytes(order="C"))
    digest.update(panel.mask.to_numpy(dtype="uint8").tobytes(order="C"))
    return digest.hexdigest()


def calibration_contract(panel: Panel, *, primary_h: int, cost_bps: float,
                         splits=None, entry_lag: int = 2, min_assets: int = 5,
                         hac_lags=None, annualization: float = 365.0,
                         engine_protocol: str | None = None) -> dict:
    """Return the JSON-safe identity that a calibration must carry.

Split boundaries are normalized by the engine itself, preserving the distinction
between an inclusive date-only end and an exact timestamp cutoff. The engine
source hash provides an additional fail-closed guard if a protocol bump is missed.
"""
    for name, value, minimum in (("primary_h", primary_h, 1), ("entry_lag", entry_lag, 1),
                                 ("min_assets", min_assets, 2)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("cost_bps must be finite and nonnegative")
    if not np.isfinite(annualization) or annualization <= 0:
        raise ValueError("annualization must be finite and positive")
    if hac_lags is not None and (isinstance(hac_lags, (bool, np.bool_))
                               or not isinstance(hac_lags, (int, np.integer)) or hac_lags < 0):
        raise ValueError("hac_lags must be a nonnegative integer or None")
    bounds = engine._split_bounds(engine.DEFAULT_SPLITS if splits is None else splits)
    protocol = engine.ENGINE_PROTOCOL if engine_protocol is None else engine_protocol
    if not isinstance(protocol, str) or not protocol:
        raise ValueError("engine_protocol must be a nonempty string")
    return {
        "schema": CONTRACT_SCHEMA,
        "panel_sha256": panel_fingerprint(panel),
        "engine_protocol": protocol,
        "engine_sha256": hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
        "primary_h": int(primary_h), "cost_bps": float(cost_bps),
        "entry_lag": int(entry_lag), "min_assets": int(min_assets),
        "hac_lags": int(hac_lags) if hac_lags is not None else None,
        "annualization": float(annualization),
        "splits": {name: [start.isoformat(), end.isoformat()]
                   for name, (start, end) in sorted(bounds.items())},
    }


def validate_calibration(calibration: Mapping, panel: Panel, **settings) -> None:
    """Reject stale or unverifiable noise floors before computing a scorecard.

``settings`` are the keyword arguments of :func:`calibration_contract`. Legacy
files cannot be made valid by assigning a universe label: rerun the null trials
using the matching data and engine, and store the resulting ``contract``.
"""
    instruction = "Re-run `python -m factor_eval calibrate` for this panel and setting."
    if not isinstance(calibration, Mapping) or not isinstance(calibration.get("contract"), Mapping):
        raise ValueError(f"calibration lacks a verifiable contract. {instruction}")
    expected = calibration_contract(panel, **settings)
    actual = calibration["contract"]
    mismatches = sorted(key for key, value in expected.items() if actual.get(key) != value)
    # Retain compatibility with the human-readable top-level metadata, while
    # refusing contradictory headers that could misidentify the measured floor.
    for key in ("primary_h", "cost_bps", "entry_lag"):
        if key in calibration and calibration[key] != expected[key] and key not in mismatches:
            mismatches.append(key)
    if mismatches:
        raise ValueError(f"calibration contract mismatch: {', '.join(sorted(mismatches))}. {instruction}")
