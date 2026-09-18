"""Benchmarks: what a constructed strategy has to beat.

The construction layer already ships one comparison, `strategy.naive_weights` --
the factor used raw, as rank weights. That answers "did the weighting scheme help",
but not "was any of this worth doing at all". The baselines here answer the second
question:

    buy_and_hold   one symbol plus cash. The thing everyone actually holds.
    equal_weight   an equal-weight basket of the eligible names, rebalanced.
    cash           flat. The floor; a strategy below it lost money trading.

Every baseline is a `date x symbol` target-weight frame and goes through the same
`portfolio.simulate` as the strategy, so costs, funding, turnover and equity
accounting are identical and the only difference is the weights.

One caveat that decides how the numbers read. A cross-sectional long/short book is
`sum(w) = 0` and holds cash; a buy-and-hold book is `w = 1` on one name. Comparing
their raw returns during a bull market compares market exposure, not skill. Read
Sharpe, Calmar and max drawdown across both, and raw return only against a
benchmark with comparable exposure -- `compare()` prints all of them side by side
rather than picking one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import DEFAULT_SPLITS, _split_bounds
from .portfolio import Book, Risk, performance, simulate
from .strategy import frozen_sign, naive_weights

__all__ = ["buy_and_hold", "equal_weight", "cash", "single_factor_naive",
           "single_asset_timing", "run_benchmarks", "compare", "split_metrics"]

_START = "2022-04-01"


def _rebalance_rows(index, rebalance: int, start) -> pd.Series:
    on = pd.Series(False, index=index)
    on.iloc[::max(1, int(rebalance))] = True
    if start is not None:
        on &= index >= pd.Timestamp(start, tz=index.tz)
    return on


def buy_and_hold(panel, symbol: str = "BTCUSDT", *, weight: float = 1.0,
                 rebalance: int = 21, start=_START) -> pd.DataFrame:
    """Hold `weight` of equity in one symbol, the rest in cash.

    Rebalancing back to the target weight is what makes this a *strategy* rather
    than a drifting position, and it pays the same costs the factor book pays.
    With `weight=1.0` the drift is bounded anyway, so `rebalance` is mostly about
    keeping the comparison mechanically identical.
    """
    if symbol not in panel.symbols:
        raise ValueError(f"{symbol!r} is not in the panel: {panel.symbols}")
    if not 0 <= weight <= 1:
        raise ValueError("weight must be in [0, 1]")
    w = pd.DataFrame(0.0, index=panel.index, columns=panel.symbols)
    w[symbol] = float(weight)
    # Only hold on days the name is actually eligible; ranking or holding outside
    # the mask is the leak the bundle's mask exists to prevent.
    w = w.where(panel.mask, 0.0)
    return w.where(_rebalance_rows(panel.index, rebalance, start), np.nan)


def equal_weight(panel, *, gross: float = 1.0, rebalance: int = 5,
                 long_only: bool = True, min_assets: int = 5,
                 start=_START) -> pd.DataFrame:
    """Equal weight across the eligible names, `sum(|w|) = gross`.

    `long_only=False` is not meaningful for an equal-weight basket (every name gets
    the same sign) and is rejected rather than silently producing a long book.
    """
    if not long_only:
        raise ValueError("an equal-weight basket is long-only by construction; "
                         "use naive_weights for a long/short benchmark")
    if gross <= 0:
        raise ValueError("gross must be positive")
    live = panel.mask.astype(float)
    n = live.sum(axis=1)
    w = live.div(n.replace(0, np.nan), axis=0) * gross
    w = w.where(n >= min_assets, 0.0)
    return w.where(_rebalance_rows(panel.index, rebalance, start), np.nan)


def cash(panel, *, start=_START) -> pd.DataFrame:
    """Hold nothing. Equity stays at 1.0; the floor any strategy must clear."""
    w = pd.DataFrame(0.0, index=panel.index, columns=panel.symbols)
    return w.where(_rebalance_rows(panel.index, 1, start), np.nan)


def single_factor_naive(factor, panel, *, h: int = 3, entry_lag: int = 2,
                        rebalance: int = 3, gross: float = 1.0, neutral: bool = True,
                        splits=None, start=_START) -> pd.DataFrame:
    """One factor, traded raw: rank weights, equal notional, no cap, no sizing.

    The direction is still frozen on dev -- trading a factor with the sign its
    author happened to write is not a baseline, it is a coin flip on half the
    candidates. Everything else is deliberately unmanaged, because this is the
    "just buy the factor" comparison the construction layer has to beat.
    """
    frame = factor(panel) if callable(factor) else factor
    sign = frozen_sign(frame, panel, h=h, entry_lag=entry_lag, splits=splits)
    return naive_weights(frame * sign, panel, gross=gross, neutral=neutral,
                         rebalance=rebalance, start=start)


def single_asset_timing(score: pd.DataFrame, panel, *, symbol: str = "BTCUSDT",
                        weight: float = 1.0, threshold: float = 0.0,
                        rebalance: int = 3, start=_START) -> pd.DataFrame:
    """Hold one symbol only while the combined score likes it; otherwise cash.

    This is the like-for-like answer to "is the strategy better than just holding
    BTC": same single asset, same costs, same simulator, and the only difference is
    that entry is conditional on the score.

    The score is a demeaned cross-sectional rank, so `> 0` reads as "this name ranks
    above the median of the cross-section today". That still needs the other symbols
    to exist -- it is a timing rule derived from a cross-section, not a standalone
    single-asset signal, and it degenerates on a one-symbol panel.
    """
    if symbol not in panel.symbols:
        raise ValueError(f"{symbol!r} is not in the panel: {panel.symbols}")
    signal = score[symbol].where(panel.mask[symbol], np.nan)
    w = pd.DataFrame(0.0, index=panel.index, columns=panel.symbols)
    w[symbol] = np.where(signal.to_numpy(dtype=float) > threshold, float(weight), 0.0)
    return w.where(_rebalance_rows(panel.index, rebalance, start), np.nan)


def run_benchmarks(panel, *, symbol: str = "BTCUSDT", entry_lag: int = 2,
                   cost_bps: float = 6.5, rebalance: int = 5, gross: float = 1.0,
                   risk: Risk | None = None, start=_START,
                   extra: dict | None = None) -> dict[str, Book]:
    """Simulate the standard baseline set on one panel."""
    targets = {
        f"buy_hold_{symbol}": buy_and_hold(panel, symbol, rebalance=21, start=start),
        "equal_weight": equal_weight(panel, gross=gross, rebalance=rebalance, start=start),
        "cash": cash(panel, start=start),
        **(extra or {}),
    }
    return {name: simulate(frame, panel, entry_lag=entry_lag, cost_bps=cost_bps, risk=risk)
            for name, frame in targets.items()}


def split_metrics(book: Book, splits=None) -> dict[str, dict]:
    """Performance recomputed inside each split window.

    Not a slice of the full-sample numbers: dev runs long and, on this panel,
    contains most of the move, so a full-sample figure quietly reports dev.
    """
    bounds = _split_bounds(DEFAULT_SPLITS if splits is None else splits)
    out = {}
    for name, (lo, hi) in bounds.items():
        window = (book.returns.index >= lo) & (book.returns.index <= hi)
        equity = book.equity[window].dropna()
        if len(equity) < 2:
            out[name] = {k: np.nan for k in ("ann_return", "ann_vol", "sharpe",
                                             "max_drawdown", "calmar",
                                             "daily_turnover", "hit_rate", "n_days")}
            continue
        # Re-base equity so drawdown is measured from inside the window, not from a
        # peak set before it.
        out[name] = performance(book.returns[window],
                                equity / equity.iloc[0],
                                book.trades[window])
    return out


def compare(books: dict[str, Book], *, splits=None,
            metrics: tuple[str, ...] = ("ann_return", "sharpe", "max_drawdown",
                                        "calmar", "daily_turnover")) -> pd.DataFrame:
    """Side-by-side table: one row per (book, split).

    Sorted by split in dev/val/oot order, then by the order the books were passed
    in, so the strategy and its baselines stay adjacent.
    """
    order = {"dev": 0, "val": 1, "oot": 2}
    rows = []
    for position, (name, book) in enumerate(books.items()):
        for split, m in split_metrics(book, splits).items():
            rows.append({"split": split, "book": name, "_o": order.get(split, 9),
                         "_p": position, **{k: m.get(k, np.nan) for k in metrics},
                         "n_days": m.get("n_days", np.nan)})
    table = pd.DataFrame(rows).sort_values(["_o", "_p"]).drop(columns=["_o", "_p"])
    return table.set_index(["split", "book"])
