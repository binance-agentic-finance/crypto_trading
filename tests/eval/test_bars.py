"""Cell sampling decides what "one period" means; every later number inherits it."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from factor_eval.bars import (SCHEMES, align_bars, build_panel,          # noqa: E402
                              sample_bars)


@pytest.fixture(scope="module")
def raw():
    rng = np.random.default_rng(11)
    n = 60 * 24 * 20
    index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.0004, n)))
    return pd.DataFrame({"open": price, "high": price * 1.001, "low": price * 0.999,
                         "close": price, "volume": np.abs(rng.normal(10, 3, n)),
                         "quote_volume": np.abs(rng.normal(1000, 300, n))}, index=index)


def test_every_scheme_produces_ordered_non_empty_cells(raw):
    params = {"time": {"interval": "1h"}, "volume": {"threshold": 5000},
              "dollar": {"threshold": 5e5}, "vol": {"target_vol": 0.02}}
    for scheme in SCHEMES:
        bars = sample_bars(raw, scheme, **params[scheme])
        assert len(bars) > 10
        assert bars.index.is_monotonic_increasing and bars.index.is_unique
        assert (bars["n_raw"] > 0).all()


def test_ohlc_is_aggregated_not_sampled(raw):
    bars = sample_bars(raw, "time", interval="1h")
    first = raw.loc[bars.index[0]:bars.index[1]].iloc[:-1]
    assert bars["open"].iloc[0] == first["open"].iloc[0]
    assert bars["close"].iloc[0] == first["close"].iloc[-1]
    assert bars["high"].iloc[0] == first["high"].max()
    assert bars["low"].iloc[0] == first["low"].min()
    assert np.isclose(bars["volume"].iloc[0], first["volume"].sum())


def test_a_dollar_cell_carries_at_least_its_threshold(raw):
    threshold = 5e5
    bars = sample_bars(raw, "dollar", threshold=threshold)
    # Every cell but a dropped final partial must have crossed the threshold.
    assert (bars["quote_volume"] >= threshold).all()


def test_the_final_partial_cell_is_dropped(raw):
    """Keeping it would make the last period of a backtest a different object."""
    full = sample_bars(raw, "dollar", threshold=5e5)
    assert full["close_time"].iloc[-1] < raw.index[-1]


def test_unknown_scheme_and_stray_params_are_refused(raw):
    with pytest.raises(ValueError, match="scheme must be"):
        sample_bars(raw, "fibonacci")
    with pytest.raises(TypeError, match="unexpected params"):
        sample_bars(raw, "time", interval="1h", threshold=5)


def test_raw_input_is_validated(raw):
    with pytest.raises(ValueError, match="missing columns"):
        sample_bars(raw.drop(columns=["quote_volume"]), "time")
    naive = raw.copy()
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(ValueError, match="tz-aware"):
        sample_bars(naive, "time")


def test_alignment_marks_absent_cells_rather_than_filling(raw):
    """A forward-filled price is one the market never printed."""
    a = sample_bars(raw, "time", interval="1h")
    b = a.iloc[::2]                                    # half the cells missing
    grid = a.index
    frames, covered = align_bars({"A": a, "B": b}, grid)
    assert covered["A"].all()
    assert not covered["B"].all()
    assert frames["close"]["B"].isna().sum() == len(grid) - len(b)


def test_build_panel_holds_back_new_listings(raw):
    bars = {s: sample_bars(raw, "time", interval="1h") for s in ("A", "B", "C",
                                                                "D", "E", "F")}
    panel = build_panel(bars, min_history=60)
    assert not panel.mask.iloc[:60].to_numpy().any()
    assert panel.mask.iloc[60:].to_numpy().any()
    assert panel.cell_scheme == "regular"


def test_a_non_daily_panel_needs_its_scheme_declared(raw):
    """The daily contract stays the default so the shipped calibration is untouched."""
    bars = {s: sample_bars(raw, "time", interval="1h") for s in ("A", "B")}
    with pytest.raises(ValueError, match="daily UTC"):
        build_panel(bars, min_history=5, cell_scheme="daily_utc")
    assert build_panel(bars, min_history=5, cell_scheme="regular") is not None


def test_an_irregular_panel_is_allowed_but_a_gapped_regular_one_is_not(raw):
    bars = {s: sample_bars(raw, "dollar", threshold=5e5) for s in ("A", "B")}
    assert build_panel(bars, min_history=5, cell_scheme="irregular") is not None
    with pytest.raises(ValueError, match="evenly spaced"):
        build_panel(bars, min_history=5, cell_scheme="regular")


def test_extras_ride_the_same_grid(raw):
    bars = {s: sample_bars(raw, "time", interval="1h") for s in ("A", "B")}
    panel = build_panel(bars, min_history=5)
    with_extra = panel.with_extras(taker_buy=panel.volume * 0.4)
    with_extra.validate()
    assert with_extra.field("taker_buy").shape == panel.close.shape
    with pytest.raises(KeyError, match="no field"):
        panel.field("taker_buy")
