"""volume-surge-daily-report — scanner_report runner.

Scans multiple symbols for volume anomalies. Per symbol, computes:

  - Volume z-score: (last_volume - rolling_mean) / rolling_std over the
    `volume_window`.
  - Price alignment: EMA(short) vs EMA(long) and price-vs-EMA(short).
  - Breakout: last close vs rolling N-bar high.

Aggregates into a composite 0-100 score with weights from preset.scoring.weights
and assigns a verdict.

Adapter contract: ``fetch_klines(req) -> list[Bar]``. Fixture mode reads
pre-computed OHLCV from ``inputs.bars_by_symbol``.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ai_pro_trading_library.library.features.indicators import ema, rolling_zscore


def _bars_to_df(bars: list[Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "timestamp": b.timestamp,
            }
            for b in bars
        ]
    )


def _ohlcv_from_inputs(symbol: str, inputs: Mapping[str, Any]) -> pd.DataFrame | None:
    bars = inputs.get("bars_by_symbol", {}).get(symbol)
    if not bars:
        return None
    return pd.DataFrame(bars)


def _verdict(score: float) -> str:
    if score >= 75:
        return "STRONG_CANDIDATE"
    if score >= 60:
        return "CANDIDATE"
    if score >= 40:
        return "WATCHLIST"
    if score >= 25:
        return "SKIP"
    return "AVOID"


def _analyze_symbol(symbol: str, df: pd.DataFrame, scoring: dict) -> dict:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    volume = df["volume"].astype(float)

    vol_window = int(scoring["volume_window"])
    z_threshold = float(scoring["volume_surge_zscore_threshold"])

    # Volume z-score using the canonical rolling_zscore indicator.
    vol_z_series = rolling_zscore(volume, vol_window)
    vol_z = float(vol_z_series.iloc[-1]) if not vol_z_series.empty else 0.0

    if vol_z >= z_threshold * 1.5:
        volume_score = 100
    elif vol_z >= z_threshold:
        volume_score = 80
    elif vol_z >= 1.0:
        volume_score = 60
    elif vol_z >= 0:
        volume_score = 40
    else:
        volume_score = 20

    # Price alignment.
    ema_s = float(ema(close, scoring["ema_short"]).iloc[-1])
    ema_l = float(ema(close, scoring["ema_long"]).iloc[-1])
    last = float(close.iloc[-1])
    if last > ema_s > ema_l:
        alignment_score = 100
        alignment_label = "bullish"
    elif last < ema_s < ema_l:
        alignment_score = 100
        alignment_label = "bearish"
    elif last > ema_s:
        alignment_score = 60
        alignment_label = "above_short"
    else:
        alignment_score = 30
        alignment_label = "weak"

    # Breakout: last close vs rolling N-bar high (excluding current bar).
    bw = int(scoring["breakout_window"])
    rolling_high = float(high.iloc[-(bw + 1) : -1].max()) if len(high) > bw else float(high.max())
    breakout_pct = ((last - rolling_high) / rolling_high) * 100 if rolling_high else 0.0
    if breakout_pct >= 1.0:
        breakout_score = 100
        breakout_label = "fresh_breakout"
    elif breakout_pct >= 0.0:
        breakout_score = 70
        breakout_label = "near_high"
    elif breakout_pct >= -2.0:
        breakout_score = 40
        breakout_label = "consolidating"
    else:
        breakout_score = 15
        breakout_label = "below_range"

    weights = scoring["weights"]
    composite = (
        volume_score * weights["volume"]
        + alignment_score * weights["alignment"]
        + breakout_score * weights["breakout"]
    )

    return {
        "symbol": symbol,
        "score": round(composite, 2),
        "verdict": _verdict(composite),
        "volume_zscore": round(vol_z, 2),
        "volume_score": volume_score,
        "alignment": {"label": alignment_label, "score": alignment_score},
        "breakout": {
            "label": breakout_label,
            "score": breakout_score,
            "rolling_high": round(rolling_high, 4),
            "current": round(last, 4),
            "breakout_pct": round(breakout_pct, 3),
        },
    }


def build_report(
    *,
    preset: dict,
    adapter: Any | None = None,
    inputs: Mapping[str, Any] | None = None,
) -> dict:
    inputs = inputs or {}
    symbols = list(inputs.get("symbols") or preset.get("symbols", []))
    timeframe = preset.get("timeframe", "1h")
    lookback = int(preset.get("lookback_bars", 200))
    scoring = dict(preset.get("scoring", {}))
    max_candidates = int(scoring.get("max_candidates", 15))

    if not symbols:
        return {
            "signals": [],
            "summary": "No symbols supplied; pass preset.symbols or inputs.symbols.",
            "recommended_next_step": "Provide a candidate list (top-N by 24h volume etc).",
            "results": [],
        }

    rows: list[dict] = []
    for symbol in symbols:
        df = _ohlcv_from_inputs(symbol, inputs)
        if df is None:
            if adapter is None:
                continue
            from ai_pro_trading_library.library.data.interfaces import KlineRequest

            bars = adapter.fetch_klines(
                KlineRequest(instrument_id=symbol, timeframe=timeframe, limit=lookback)
            )
            df = _bars_to_df(bars)

        if len(df) < scoring.get("ema_long", 50):
            continue
        rows.append(_analyze_symbol(symbol, df, scoring))

    if not rows:
        return {
            "signals": [],
            "summary": "No symbol had sufficient history.",
            "recommended_next_step": "Increase lookback_bars or check adapter responses.",
            "results": [],
        }

    rows.sort(key=lambda r: r["score"], reverse=True)
    rows = rows[:max_candidates]

    signals = [
        {
            "symbol": r["symbol"],
            "side": "long" if r["alignment"]["label"] in ("bullish", "above_short") else "short",
            "score": r["score"],
            "reason": (
                f"verdict={r['verdict']}; vol_z={r['volume_zscore']}; "
                f"align={r['alignment']['label']}; breakout={r['breakout']['label']}"
            ),
        }
        for r in rows
        if r["verdict"] in ("STRONG_CANDIDATE", "CANDIDATE", "WATCHLIST")
    ]

    top_strong = [r for r in rows if r["verdict"] == "STRONG_CANDIDATE"]
    return {
        "signals": signals,
        "summary": (
            f"Scanned {len(symbols)} symbol(s); kept top {len(rows)}. "
            f"{len(top_strong)} STRONG_CANDIDATE(s)."
        ),
        "recommended_next_step": (
            "Promote STRONG_CANDIDATE rows into structured-trade-plan or "
            "three-tier-strategy for entry zone derivation."
        ),
        "results": rows,
    }
