"""Benchmarks must be mechanically correct before any comparison means anything."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from factor_eval import load_bundle                                     # noqa: E402
from factor_eval.benchmarks import (buy_and_hold, cash, compare, equal_weight,  # noqa: E402
                                    run_benchmarks, single_asset_timing,
                                    single_factor_naive, split_metrics)
from factor_eval.examples.example_factors import low_volatility          # noqa: E402
from factor_eval.portfolio import simulate                              # noqa: E402


@pytest.fixture(scope="module")
def panel():
    return load_bundle()


def test_cash_never_moves(panel):
    book = simulate(cash(panel), panel)
    assert book.equity.nunique() == 1
    assert float(book.trades.sum()) == 0.0
    assert float(book.costs.sum()) == 0.0


def test_buy_and_hold_tracks_the_underlying(panel):
    """A fully invested single-name book must follow that name's price path.

    Allows for entry cost and funding, but the shape has to match: if this drifts,
    the simulator and not the strategy is producing the comparison.
    """
    book = simulate(buy_and_hold(panel, "BTCUSDT"), panel, cost_bps=0.0)
    traded = book.equity.dropna()
    traded = traded[traded.index >= pd.Timestamp("2022-05-01", tz=panel.index.tz)]
    price = panel.open["BTCUSDT"].reindex(traded.index)
    equity_return = traded / traded.iloc[0]
    price_return = price / price.iloc[0]
    assert equity_return.corr(price_return) > 0.99


def test_buy_and_hold_rejects_an_unknown_symbol(panel):
    with pytest.raises(ValueError, match="not in the panel"):
        buy_and_hold(panel, "NOTASYMBOL")


def test_equal_weight_is_fully_invested_and_equal(panel):
    w = equal_weight(panel).dropna(how="all")
    row = w.loc[w.abs().sum(axis=1) > 0].iloc[-1]
    held = row[row != 0]
    assert np.allclose(float(row.abs().sum()), 1.0)
    assert np.allclose(held.to_numpy(), held.iloc[0])      # every holding the same size


def test_equal_weight_refuses_to_pretend_to_be_long_short(panel):
    with pytest.raises(ValueError, match="long-only by construction"):
        equal_weight(panel, long_only=False)


def test_equal_weight_holds_only_eligible_names(panel):
    w = equal_weight(panel).dropna(how="all")
    assert not ((w != 0) & ~panel.mask.reindex(w.index)).to_numpy().any()


def test_single_factor_naive_is_sign_invariant(panel):
    """Direction is frozen on dev, so negating the factor must change nothing."""
    frame = low_volatility(panel)
    a = single_factor_naive(frame, panel).fillna(0.0)
    b = single_factor_naive(-frame, panel).fillna(0.0)
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-12)


def test_single_asset_timing_only_ever_holds_its_symbol(panel):
    score = low_volatility(panel)
    w = single_asset_timing(score, panel, symbol="BTCUSDT").dropna(how="all")
    others = [c for c in w.columns if c != "BTCUSDT"]
    assert (w[others].to_numpy() == 0).all()
    assert w["BTCUSDT"].isin([0.0, 1.0]).all()


def test_split_metrics_measure_drawdown_inside_the_window(panel):
    """Rebasing matters: a peak set in dev must not count as an oot drawdown."""
    book = simulate(buy_and_hold(panel, "BTCUSDT"), panel)
    per_split = split_metrics(book)
    assert set(per_split) == {"dev", "val", "oot"}
    full = book.equity / book.equity.cummax() - 1
    assert per_split["oot"]["max_drawdown"] >= float(full.min())


def test_compare_reports_every_book_on_every_split(panel):
    books = run_benchmarks(panel)
    table = compare(books)
    assert set(table.index.get_level_values("book")) == set(books)
    assert list(dict.fromkeys(table.index.get_level_values("split"))) == ["dev", "val", "oot"]
