"""The capability adapter has to refuse to run a capability it cannot run faithfully."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from factor_eval import load_bundle                                      # noqa: E402
from factor_eval.capability import capability_factor, panel_rows         # noqa: E402
from factor_eval.capability_registry import (capability_factors, load_index,  # noqa: E402
                                             runtime_shim, ts_zscore,
                                             upstream_factor)


@pytest.fixture(scope="module")
def panel():
    return load_bundle()


@pytest.fixture(scope="module")
def index():
    return load_index()


def test_index_carries_its_provenance(index):
    assert len(index) > 0
    assert index.source["repo"] and index.source["commit"]
    assert index.describe().startswith(str(len(index)))


def test_series_and_scalar_ports_are_not_two_factors(index):
    """`value` and `series` are the same output read two ways."""
    spec = index["cci"]
    assert "series" in spec.series_outputs
    assert spec.ports == ("value",)


def test_unmapped_manifest_params_raise(index):
    """Dropping a param silently would score a different factor than requested."""
    with pytest.raises(ValueError, match="no counterpart"):
        upstream_factor(index["vwap"], port="vwap")


def test_documented_param_drops_are_allowed(index):
    """`rsi` declares wilder/simple; upstream implements Wilder, so that value drops."""
    assert upstream_factor(index["rsi"], port="rsi") is not None
    with pytest.raises(ValueError, match="no counterpart"):
        upstream_factor(index["rsi"], port="rsi", params={"method": "simple"})


def test_port_count_mismatch_raises_rather_than_guessing(index, panel):
    """`bollinger` declares five ports, upstream returns three."""
    factor = upstream_factor(index["bollinger"], port="middle", skip_failures=False)
    with pytest.raises(RuntimeError, match="cannot match positionally"):
        factor(panel)


def test_rows_input_reaches_the_operator(panel):
    """A `rows`-shaped capability gets bar dicts, not a bare float series."""
    seen = {}

    def operator(*, rows, period):
        seen["n"] = len(rows)
        seen["keys"] = set(rows[-1])
        return {"value": float(period)}

    frame = capability_factor(operator, rows="rows", params={"period": 7},
                              window=5)(panel)
    assert seen["n"] == 5
    assert {"open", "high", "low", "close", "volume"} <= seen["keys"]
    assert np.nanmax(frame.to_numpy()) == 7.0


def test_panel_rows_keeps_missing_values_missing(panel):
    rows = panel_rows(panel, panel.symbols[0])
    assert len(rows) == len(panel.index)
    assert all(v is None or np.isfinite(v) for v in rows[0].values())


def test_capability_factor_rejects_a_param_clash(panel):
    with pytest.raises(ValueError, match="rows input and a fixed param"):
        capability_factor(lambda **_: 1.0, rows="rows", params={"rows": 1})


def test_ts_zscore_is_lagged(panel):
    """Scaling constants must not see the day they scale."""
    frame = panel.close
    z = ts_zscore(frame, window=30)
    shifted = ts_zscore(frame.shift(1), window=30)
    assert not np.allclose(z.dropna(how="all").to_numpy()[-1],
                           shifted.dropna(how="all").to_numpy()[-1])
    assert z.iloc[:10].isna().all().all()


def test_factors_that_cannot_run_are_rejected_with_a_reason(index, panel):
    rejected = {}
    factors = capability_factors(index, panel=panel, rejected=rejected)
    assert factors and rejected
    assert all(isinstance(v, str) and v for v in rejected.values())
    # Composition nodes take other signals, not market data, and must not silently
    # become candidates.
    assert not any(k.startswith("additive_combine") for k in factors)


def test_every_returned_factor_actually_runs(index, panel):
    factors = capability_factors(index, panel=panel)
    name = sorted(factors)[0]
    frame = factors[name](panel)
    assert frame.shape == (len(panel.index), len(panel.symbols))
    assert np.isfinite(frame.where(panel.mask).to_numpy(dtype=float)).any()


def test_runtime_shim_is_importable_and_inert():
    runtime_shim()
    from binance.strategy.runtime import ctx                             # noqa: E402
    from binance.strategy.node.tools.blocks_adapter import (last_valid,  # noqa: E402
                                                            series_from, to_json_list)
    assert ctx.log("INFO", "X", {}) is None
    assert last_valid([np.nan, 1.0, np.nan]) == 1.0
    assert to_json_list(pd.Series([1.0, np.nan])) == [1.0, None]
    assert series_from([1, None, 3]).isna().sum() == 1
