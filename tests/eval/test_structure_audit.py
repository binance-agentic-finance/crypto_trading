"""Structural comparisons must use known ranks and isolated, shared samples."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cyqnt_trd.eval import Panel, _structure  # noqa: E402


@pytest.fixture
def case():
    index = pd.date_range("2020-01-01", periods=630, tz="UTC")
    scores = np.tile(np.arange(8), (len(index), 1))
    signal = pd.DataFrame(scores, index=index)
    price = pd.DataFrame(100 * np.exp(np.arange(len(index))[:, None]
                                     * (0.0001 + np.arange(8)[None, :] * 0.0002)), index=index)
    panel = Panel(open=price, high=price * 1.01, low=price * .99, close=price,
                  volume=price * 0 + 1, quote_volume=price,
                  funding=price * 0, mask=price.notna())
    splits = {name: (index[lo], index[hi]) for name, lo, hi in
              (("dev", 0, 199), ("val", 200, 399), ("oot", 400, 629))}
    return panel, signal, splits


def test_oot_price_shock_cannot_change_validation_structure(case):
    panel, signal, splits = case
    expected = _structure(signal, panel, 3, 2, 1, splits=splits)
    changed = panel.open.copy()
    changed.iloc[400:] *= np.exp(np.arange(230)[:, None] * np.arange(8)[None, :] * -.001)
    actual = _structure(signal, replace(panel, open=changed), 3, 2, 1, splits=splits)
    assert expected["spread_bp"] > 0
    assert actual["by_split"]["val"] == expected["by_split"]["val"]
    assert actual["by_split"]["oot"]["spread_bp"] < 0


def test_missing_future_price_drops_whole_date_without_reassigning_bins(case):
    panel, signal, splits = case
    expected = _structure(signal, panel, 3, 2, 1, splits=splits)
    changed = panel.open.copy()
    changed.iloc[300, 0] = np.nan
    actual = _structure(signal, replace(panel, open=changed), 3, 2, 1, splits=splits)
    assert actual["n_dates"] == expected["n_dates"] - 2  # one entry and one exit label
    for old, new in zip(expected["bins"], actual["bins"]):
        assert old["n"] / expected["n_dates"] == new["n"] / actual["n_dates"]
        assert new["mean_bp"] == pytest.approx(old["mean_bp"])


def test_tied_signal_values_are_not_split_to_manufacture_three_bins(case):
    panel, signal, splits = case
    signal = (signal > 3).astype(float)
    actual = _structure(signal, panel, 3, 2, 1, splits=splits)
    assert actual["n_dates"] == 0
    assert np.isnan(actual["spread_bp"])