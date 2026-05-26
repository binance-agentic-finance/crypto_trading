"""Unified trading skill — executable helpers."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ai_pro_trading_library.cases import load_catalog
from ai_pro_trading_library.cases.loader import load_case_strategy
from ai_pro_trading_library.library.core.protocols import StrategySpec
from ai_pro_trading_library.runtime.backtest import BacktestRunner
from ai_pro_trading_library.runtime.daemon import PaperDaemon
from ai_pro_trading_library.runtime.watcher import project_from_path


_KEYWORDS: dict[str, tuple[str, ...]] = {
    "rsi-mean-reversion": ("rsi", "oversold", "mean reversion", "mean-reversion"),
    "bb-squeeze-momentum": ("bollinger", "squeeze", "breakout"),
    "btc-multi-factor-trend": ("trend", "ema", "multi factor", "btc trend"),
    "structure-breakout-priceaction": ("structure", "price action", "breakout high"),
    "stochrsi-mdi-trend-v1": ("stochrsi", "mdi", "stochastic rsi"),
    "funding-rate-scanner": ("funding", "perpetual", "scanner"),
}


def _tokenize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())


def select_case_by_intent(text: str) -> str:
    """Match a free-form intent string to a case_id from the catalog.

    Deterministic keyword scoring; raises if nothing matches.
    """
    if not text or not text.strip():
        raise ValueError("intent text must be non-empty")
    catalog_ids = {c["case_id"] for c in load_catalog()["cases"]}
    text_norm = _tokenize(text)
    scores: dict[str, int] = {}
    for case_id, keywords in _KEYWORDS.items():
        if case_id not in catalog_ids:
            continue
        scores[case_id] = sum(1 for kw in keywords if kw in text_norm)
    if not scores or max(scores.values()) == 0:
        raise LookupError(f"no case matched intent: {text!r}")
    return max(scores, key=scores.get)


def build_strategy_spec(
    case_id: str,
    overrides: Mapping[str, Any] | None = None,
) -> StrategySpec:
    """Resolve a case_id to its `StrategySpec`, with optional parameter overrides."""
    spec = load_case_strategy(case_id)
    if not overrides:
        return spec
    merged = dict(spec.parameters)
    merged.update(overrides)
    return replace(spec, parameters=merged)


def dispatch_runtime(
    spec: StrategySpec,
    mode: str = "backtest",
    output_dir: str | Path | None = None,
    *,
    data: pd.DataFrame | None = None,
    max_iterations: int = 1,
) -> dict[str, Any]:
    """Dispatch a `StrategySpec` to backtest or paper runtime.

    - `backtest` returns a result dict with sharpe / max_drawdown / trade_count.
    - `paper` runs the daemon and returns the run directory and snapshot.
    - `live` is intentionally not enabled here.
    """
    if mode == "backtest":
        df = data if data is not None else _synthetic_ohlcv()
        result = BacktestRunner().run(spec, df)
        return {
            "mode": "backtest",
            "strategy_id": spec.strategy_id,
            "bars": result.bars,
            "trades": result.trades,
            "trade_count": result.trade_count,
            "total_return": result.total_return,
            "sharpe": result.sharpe,
            "max_drawdown": result.max_drawdown,
        }
    if mode == "paper":
        if output_dir is None:
            raise ValueError("output_dir is required for paper mode")
        daemon = PaperDaemon(output_dir=output_dir)
        state = daemon.start(spec, max_iterations=max_iterations)
        return {
            "mode": "paper",
            "strategy_id": spec.strategy_id,
            "run_id": state.run_id,
            "run_dir": str(daemon.run_dir),
            "status": state.status,
            "equity": state.equity,
            "trade_count": len(state.trades),
            "event_count": len(state.events),
        }
    if mode == "live":
        raise NotImplementedError("live mode requires explicit confirmation; not enabled in this build")
    raise ValueError(f"unsupported mode: {mode}")


def summarize_run_status(run_dir: str | Path) -> dict[str, Any]:
    """Read `<run_dir>/state.json` and return a watcher-style snapshot dict."""
    state_path = Path(run_dir) / "state.json"
    snap = project_from_path(state_path)
    return {
        "run_id": snap.run_id,
        "status": snap.status,
        "trade_count": snap.trade_count,
        "event_count": snap.event_count,
        "equity": snap.equity,
    }


def _synthetic_ohlcv(rows: int = 200) -> pd.DataFrame:
    close = pd.Series([100.0 + i * 0.3 + ((i % 7) - 3) * 0.4 for i in range(rows)])
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0,
        }
    )
