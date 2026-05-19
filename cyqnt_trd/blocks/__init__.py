"""Composable building blocks for crypto trading strategies.

This package provides a curated library of pure Python / pandas / numpy
building blocks that can be ``import``-ed directly by user-written or
LLM-generated strategy scripts and combined to form complete trading
strategies for ``standard_bot``.

Quick start
-----------

A minimal long-only strategy that uses MA20/MA60 crossover plus MACD
confirmation::

    # save as my_strategy.py
    from cyqnt_trd.blocks import indicators as ind, conditions as cond, strategy

    def make_signals(df):
        ma20 = ind.sma(df["close"], 20)
        ma60 = ind.sma(df["close"], 60)
        macd_line, signal_line, _ = ind.macd(df["close"])
        long = cond.ma_cross_above(ma20, ma60) & cond.macd_above_zero(macd_line)
        short = cond.ma_cross_below(ma20, ma60)
        return long, short

    strategy.register("my_ma_macd", make_signals)

Then run a backtest::

    python -m cyqnt_trd.standard_bot.entrypoints.mvp_backtest \
        --engine python --strategy my_ma_macd \
        --strategy-module my_strategy \
        --symbol BTCUSDT --interval 15m --limit 500

Module layout
-------------

* :mod:`cyqnt_trd.blocks.indicators` — technical indicators (SMA/EMA/RSI/MACD/ADX/ATR/...)
* :mod:`cyqnt_trd.blocks.patterns` — candlestick pattern detectors
* :mod:`cyqnt_trd.blocks.derivatives` — futures-specific blocks (OI, funding, long/short, CVD, liquidation)
* :mod:`cyqnt_trd.blocks.conditions` — atomic boolean entry conditions
* :mod:`cyqnt_trd.blocks.entry` — combinators for conditions (all_of, any_of, scoring)
* :mod:`cyqnt_trd.blocks.exit` — stop-loss / take-profit / trailing / partial-close helpers
* :mod:`cyqnt_trd.blocks.risk` — portfolio-level risk control state machine
* :mod:`cyqnt_trd.blocks.sizing` — position-size calculators
* :mod:`cyqnt_trd.blocks.universe` — symbol-pool filters & scanners
* :mod:`cyqnt_trd.blocks.execution` — order spec helpers
* :mod:`cyqnt_trd.blocks.scoring` — multi-factor scoring system
* :mod:`cyqnt_trd.blocks.regime` — market regime classifier (trend/range/vol)
* :mod:`cyqnt_trd.blocks.microstructure` — whale / order flow detection
* :mod:`cyqnt_trd.blocks.data` — pandas <-> Bar conversion + Binance public data fetchers
* :mod:`cyqnt_trd.blocks.strategy` — register a user-defined strategy as a SignalPlugin

Design principles
-----------------

1. **DataFrame-first** — most blocks accept a ``pandas.DataFrame`` with
   lower-case OHLCV columns or a ``pandas.Series``. The same blocks work in
   notebooks, jobs, and ``--strategy-module`` plugins.
2. **No silent failures** — invalid input raises ``ValueError`` / ``TypeError``
   with a clear message, never returns wrong-shape data.
3. **Look-ahead-safe** — every rolling/EMA computation uses the standard
   trailing window. Crossovers are detected on the bar where they
   actually become true (so simulators must shift one bar before
   execution; the framework already does this for ``next_open`` mode).
4. **No hard external API dependency at import time** — fetcher functions in
   ``data`` lazy-import ``requests`` so that the package is usable in
   restricted environments.
"""

from __future__ import annotations

# Re-export the sub-modules so user code can do
# ``from cyqnt_trd.blocks import indicators as ind`` instead of
# ``from cyqnt_trd.blocks.indicators import ...``.
from . import (
    conditions,
    data,
    derivatives,
    entry,
    execution,
    exit,  # noqa: A004 — name shadows builtin on purpose; users normally write `from blocks import exit as ex`
    indicators,
    microstructure,
    patterns,
    regime,
    risk,
    scoring,
    sizing,
    strategy,
    universe,
)

# ---------------------------------------------------------------------------
# Top-level convenience re-exports
# ---------------------------------------------------------------------------
# Cherry-pick the most-commonly used functions so that LLM-generated code
# such as ``from cyqnt_trd.blocks import sma, register, all_of`` works.
from .indicators import (  # noqa: E402
    sma, ema, rsi, macd, atr, adx, bollinger, donchian, stochastic,
    vwap, obv, volume_ma, swing_high, swing_low,
)
from .entry import all_of, any_of, score_entry, weighted_score, consecutive  # noqa: E402
from .strategy import register  # noqa: E402
from .risk import RiskConfig, RiskGuard, RiskState  # noqa: E402
from .scoring import ScoringSystem  # noqa: E402

__all__ = [
    # sub-modules
    "conditions",
    "data",
    "derivatives",
    "entry",
    "execution",
    "exit",
    "indicators",
    "microstructure",
    "patterns",
    "regime",
    "risk",
    "scoring",
    "sizing",
    "strategy",
    "universe",
    # top-level convenience
    "sma",
    "ema",
    "rsi",
    "macd",
    "atr",
    "adx",
    "bollinger",
    "donchian",
    "stochastic",
    "vwap",
    "obv",
    "volume_ma",
    "swing_high",
    "swing_low",
    "all_of",
    "any_of",
    "score_entry",
    "weighted_score",
    "consecutive",
    "register",
    "RiskConfig",
    "RiskGuard",
    "RiskState",
    "ScoringSystem",
]

__version__ = "0.1.0"
