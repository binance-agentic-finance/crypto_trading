"""Deterministic local smoke data and runners."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai_pro_trading_library.cases.loader import load_case_strategy
from ai_pro_trading_library.runtime.backtest import BacktestRunner


def synthetic_ohlcv(rows: int = 120) -> pd.DataFrame:
    close = pd.Series([100 + i * 0.4 + ((i % 9) - 4) * 0.7 for i in range(rows)], dtype=float)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 1.5,
            "low": close - 1.5,
            "close": close,
            "volume": pd.Series([1000 + (i % 13) * 25 for i in range(rows)], dtype=float),
        }
    )


def run_case_backtest_smoke(case_id: str, rows: int = 120) -> dict[str, object]:
    spec = load_case_strategy(case_id)
    result = BacktestRunner().run(spec, synthetic_ohlcv(rows))
    return {
        "case_id": case_id,
        "strategy_id": spec.strategy_id,
        "bars": result.bars,
        "trade_count": result.trade_count,
        "total_return": result.total_return,
        "max_drawdown": result.max_drawdown,
    }


def run_all_active_case_smokes(case_ids: list[str], rows: int = 120) -> list[dict[str, object]]:
    return [run_case_backtest_smoke(case_id, rows=rows) for case_id in case_ids]

