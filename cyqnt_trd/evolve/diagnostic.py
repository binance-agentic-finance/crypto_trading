"""Phase A — diagnostic adapter that wraps auto-optimize's segment / diagnose
pipeline so we can apply it to an evolve genome.

Pipeline:
    genome (evolve) + df (cyqnt OHLCV)
        ↓ build cyqnt make_signals + run vectorized_backtest with record_trades=True
        ↓ convert candles + cyqnt trades → auto_opt BacktestResult
        ↓ auto_opt.segments.build_diagnostic_report()
        ↓ auto_opt.diagnose.diagnose()
    DiagnosticResult (regime / volatility breakdown + failure_modes)

The point: the LLM (you, in CodeMax) can call this on each top/bottom genome
and see *why* it's failing rather than just `fitness=2.687`.

Requires the auto-optimize ``src`` directory to be importable. Point at it by
setting the ``AUTO_OPT_SRC`` environment variable, or pass ``auto_opt_path=...``
explicitly.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .bridge import make_signal_fn
from .genome import StrategyGenome


# ── Auto-optimize path bootstrap ──────────────────────────────────────────

_AUTO_OPT_ENV = os.environ.get("AUTO_OPT_SRC")
_DEFAULT_AUTO_OPT = Path(_AUTO_OPT_ENV) if _AUTO_OPT_ENV else None


def _ensure_auto_opt_on_path(path: Optional[str | Path] = None) -> None:
    """Ensure the auto_opt package is importable.

    Resolution order: explicit ``path`` argument, then the ``AUTO_OPT_SRC``
    environment variable. Raises if neither is set / valid.
    """
    target = Path(path) if path else _DEFAULT_AUTO_OPT
    if target is None:
        raise FileNotFoundError(
            "auto-optimize source path is not configured. Set the AUTO_OPT_SRC "
            "environment variable, or pass auto_opt_path= to its 'src' directory."
        )
    if not target.exists():
        raise FileNotFoundError(
            f"auto-optimize source not found at {target}. "
            "Set AUTO_OPT_SRC or pass auto_opt_path= to a valid 'src' directory."
        )
    target_str = str(target)
    if target_str not in sys.path:
        sys.path.insert(0, target_str)


# ── DataFrame → auto_opt candle dicts ─────────────────────────────────────

def _df_to_candles(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert an evolve-style OHLCV DataFrame (DatetimeIndex + open/high/low/
    close/volume) to the auto_opt candle dict format expected by segments.py.

    auto_opt format per row:
        {open_ts, open, high, low, close, volume, close_ts}

    open_ts / close_ts are ms epoch.
    """
    candles: List[Dict[str, Any]] = []

    # Derive open_ts / close_ts. Prefer existing close_time column; fall back
    # to DatetimeIndex.
    if "close_time" in df.columns:
        # close_time is ms; open_ts is the bar's open time (close_time - interval)
        # Use index for open_ts when available
        if isinstance(df.index, pd.DatetimeIndex):
            open_ts_arr = (df.index.astype("int64") // 1_000_000).to_numpy()
        else:
            open_ts_arr = df["close_time"].astype("int64").to_numpy()
        close_ts_arr = df["close_time"].astype("int64").to_numpy()
    elif isinstance(df.index, pd.DatetimeIndex):
        open_ts_arr = (df.index.astype("int64") // 1_000_000).to_numpy()
        # Approximate close_ts = next open_ts (last bar uses same as open)
        close_ts_arr = open_ts_arr.copy()
        if len(close_ts_arr) > 1:
            close_ts_arr[:-1] = open_ts_arr[1:]
    else:
        # Fallback: synthetic timestamps at 1ms intervals
        open_ts_arr = list(range(len(df)))
        close_ts_arr = open_ts_arr

    for i, row in enumerate(df.itertuples(index=False)):
        candles.append({
            "open_ts": int(open_ts_arr[i]),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
            "close_ts": int(close_ts_arr[i]),
        })
    return candles


# ── Public API ─────────────────────────────────────────────────────────────

def diagnose_genome(
    genome: StrategyGenome,
    df: pd.DataFrame,
    *,
    auto_opt_path: Optional[str | Path] = None,
    fee_bps: float = 4.0,
    slippage_bps: float = 1.0,
    initial_capital: float = 10_000.0,
) -> Dict[str, Any]:
    """Run full Phase A diagnostic on a single genome.

    Parameters
    ----------
    genome : StrategyGenome
        The evolve genome to diagnose.
    df : pd.DataFrame
        OHLCV DataFrame (typically the IS slice).
    auto_opt_path : str | Path | None
        Path to ``auto-optimize/src``. Defaults to local install.
    fee_bps, slippage_bps, initial_capital : float
        Backtest config (matches evolve defaults).

    Returns
    -------
    dict with keys:
        - genome_id : str
        - overall_metrics : {total_return, sharpe, max_drawdown, ...}
        - regime_breakdown : per-regime SegmentReport (trend_up/trend_dn/range)
        - volatility_breakdown : per-vol bucket (low/mid/high)
        - time_breakdown : 4 equal time blocks
        - good_segments : list of segment_id where strategy excels
        - bad_segments : list of segment_id where strategy fails (sorted by severity)
        - regime_timeline : run-length encoded regime sequence
        - failure_modes : list of {mode_id, root_cause, severity, ...}
        - overall_pattern : human-readable strategy character
        - primary_weakness : the most actionable problem
        - good_pattern : where the edge concentrates
    """
    _ensure_auto_opt_on_path(auto_opt_path)

    # Lazy imports — keep evolve usable without auto-optimize installed
    from auto_opt._cyqnt_bridge import run_cyqnt_backtest  # type: ignore
    from auto_opt.diagnose import diagnose                  # type: ignore
    from auto_opt.segments import build_diagnostic_report   # type: ignore

    # 1. Convert df → candles, build signal fn from genome
    candles = _df_to_candles(df)
    signal_fn = make_signal_fn(genome)

    # 2. Construct exit_cfg from genome (mirrors bridge.backtest_genome)
    exit_cfg = {"type": genome.exit_type, **dict(genome.exit_params)}

    # 3. Run cyqnt backtest with record_trades=True via auto_opt's bridge.
    #    This gives us an auto_opt BacktestResult (with Trade dataclass list).
    bt = run_cyqnt_backtest(
        candles=candles,
        signal_fn=signal_fn,
        exit_cfg=exit_cfg,
        interval=genome.preferred_interval,
        initial_capital=initial_capital,
        commission_bps=fee_bps,
        slippage_bps=slippage_bps,
        size=genome.size,
        spec_dict=genome.to_dict(),
    )

    # 4. Build segmented diagnostic report
    full_report = build_diagnostic_report(bt)

    # 5. Apply rule-based diagnosis (auto_opt's MVP — this is the LLM-hookable point)
    diagnosis = diagnose(full_report)

    # 6. Pack into a JSON-friendly dict
    return {
        "genome_id": genome.genome_id,
        "species": genome.species,
        "preferred_interval": genome.preferred_interval,
        "overall_metrics": dict(full_report.overall_metrics),
        "regime_breakdown": [_segment_to_dict(s) for s in full_report.regime_segments],
        "volatility_breakdown": [_segment_to_dict(s) for s in full_report.volatility_segments],
        "time_breakdown": [_segment_to_dict(s) for s in full_report.time_segments],
        "good_segments": [s.segment_id for s in full_report.good_segments],
        "bad_segments": [s.segment_id for s in full_report.bad_segments],
        "regime_timeline_summary": _summarize_timeline(full_report.regime_timeline),
        "failure_modes": [asdict(fm) for fm in diagnosis.failure_modes],
        "overall_pattern": diagnosis.overall_pattern,
        "primary_weakness": diagnosis.primary_weakness,
        "good_pattern": diagnosis.good_pattern,
    }


def _segment_to_dict(s) -> Dict[str, Any]:
    """SegmentReport → flat dict (skip large fields)."""
    return {
        "segment_id": s.segment_id,
        "label": s.label,
        "trade_count": s.trade_count,
        "total_return": round(s.total_return, 6),
        "win_rate": round(s.win_rate, 4),
        "avg_return_per_trade": round(s.avg_return_per_trade, 6),
        "avg_holding_bars": round(s.avg_holding_bars, 2),
        "bars_in_segment": s.bars_in_segment,
        "contribution_pct": round(s.contribution_pct, 4),
        "grade": s.grade,
        "severity": round(s.severity, 4),
    }


def _summarize_timeline(timeline: List[Dict]) -> Dict[str, int]:
    """Compress full timeline (bar-by-bar) into a regime-count summary."""
    from collections import Counter

    counts: Counter = Counter()
    spans: Counter = Counter()
    for entry in timeline:
        regime = entry["regime"]
        span = entry["end_idx"] - entry["start_idx"] + 1
        counts[regime] += 1
        spans[regime] += span
    return {
        "regime_run_counts": dict(counts),
        "regime_total_bars": dict(spans),
    }


# ── Pretty printer for terminal use ────────────────────────────────────────

def format_diagnosis(diag: Dict[str, Any]) -> str:
    """Render a diagnosis dict as a human-readable string (for CLI output)."""
    lines: List[str] = []
    gid = diag["genome_id"]
    sp = diag["species"]
    m = diag["overall_metrics"]
    lines.append(f"=== Diagnosis for `{gid}` ({sp}) ===")
    lines.append(
        f"  Overall: ret={m.get('total_return', 0):+.2%}  "
        f"sharpe={m.get('sharpe', 0):+.2f}  "
        f"trades={int(m.get('trade_count', 0))}  "
        f"win={m.get('win_rate', 0):.2f}  "
        f"PF={m.get('profit_factor', 0):.2f}"
    )

    lines.append("")
    lines.append("  Regime breakdown:")
    for r in diag["regime_breakdown"]:
        flag = {"good": "✓", "bad": "✗", "neutral": "·"}.get(r["grade"], "?")
        lines.append(
            f"    {flag} {r['segment_id']:<18} "
            f"trades={r['trade_count']:>3}  "
            f"return={r['total_return']:+.2%}  "
            f"win={r['win_rate']:.2f}  "
            f"contribution={r['contribution_pct']:+.1%}  "
            f"bars={r['bars_in_segment']}"
        )

    lines.append("")
    lines.append("  Volatility breakdown:")
    for v in diag["volatility_breakdown"]:
        flag = {"good": "✓", "bad": "✗", "neutral": "·"}.get(v["grade"], "?")
        lines.append(
            f"    {flag} {v['segment_id']:<18} "
            f"trades={v['trade_count']:>3}  "
            f"return={v['total_return']:+.2%}  "
            f"win={v['win_rate']:.2f}  "
            f"contribution={v['contribution_pct']:+.1%}"
        )

    lines.append("")
    lines.append("  Time blocks:")
    for t in diag["time_breakdown"]:
        flag = {"good": "✓", "bad": "✗", "neutral": "·"}.get(t["grade"], "?")
        lines.append(
            f"    {flag} {t['segment_id']:<18} "
            f"trades={t['trade_count']:>3}  "
            f"return={t['total_return']:+.2%}  "
            f"win={t['win_rate']:.2f}"
        )

    lines.append("")
    lines.append(f"  Pattern: {diag['overall_pattern']}")
    lines.append(f"  Weakness: {diag['primary_weakness']}")
    if diag["good_pattern"]:
        lines.append(f"  Good pattern: {diag['good_pattern']}")
    if diag["failure_modes"]:
        lines.append("  Failure modes:")
        for fm in diag["failure_modes"]:
            lines.append(
                f"    - {fm['mode_id']:<28}  severity={fm['severity']:.2f}  "
                f"loss_contrib={fm['loss_contribution_pct']:.1%}"
            )
            lines.append(f"        {fm['description']}")
            lines.append(f"        root cause: {fm['root_cause']}")

    return "\n".join(lines)
