"""kline-indicator-checkup — scanner_report runner.

Computes a 3-pillar (trend / momentum / volatility) 0-100 score per symbol
using the canonical `library.features.indicators` implementations.

Adapter contract: any object with a ``fetch_klines(req)`` method returning
``list[Bar]``. The case unwraps Bars to a pandas.DataFrame internally.

Fixture mode: when ``adapter`` is None, the case loads
``fixtures/input.json`` (created via the case's fixture-generation step)
and treats it as pre-computed OHLCV input keyed by symbol.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ai_pro_trading_library.library.features.indicators import (
    adx,
    atr,
    bollinger,
    ema,
    macd,
    rsi,
)


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


def _trend_score(close: pd.Series, params: dict) -> tuple[float, dict]:
    ema_short = ema(close, params["ema_short"]).iloc[-1]
    ema_mid = ema(close, params["ema_mid"]).iloc[-1]
    ema_long = ema(close, params["ema_long"]).iloc[-1]
    macd_line, signal_line, hist = macd(
        close, params["macd_fast"], params["macd_slow"], params["macd_signal"]
    )
    last = float(close.iloc[-1])

    above_short = last > ema_short
    above_mid = last > ema_mid
    above_long = last > ema_long
    aligned = ema_short > ema_mid > ema_long
    macd_bull = macd_line.iloc[-1] > signal_line.iloc[-1]

    score = 0
    score += 20 if above_short else 0
    score += 20 if above_mid else 0
    score += 20 if above_long else 0
    score += 20 if aligned else 0
    score += 20 if macd_bull else 0

    return float(score), {
        "ema_short": float(ema_short),
        "ema_mid": float(ema_mid),
        "ema_long": float(ema_long),
        "ema_aligned": bool(aligned),
        "macd_bullish": bool(macd_bull),
        "price_above_ema_long": bool(above_long),
    }


def _momentum_score(close: pd.Series, params: dict) -> tuple[float, dict]:
    rsi_value = float(rsi(close, params["rsi_period"]).iloc[-1])
    if rsi_value <= params["rsi_oversold"]:
        zone = "oversold"
        score = 90
    elif rsi_value >= params["rsi_overbought"]:
        zone = "overbought"
        score = 10
    else:
        # Map [oversold, overbought] linearly to [80, 20] (extremes are
        # high-confidence; mid is neutral).
        span = params["rsi_overbought"] - params["rsi_oversold"]
        offset = rsi_value - params["rsi_oversold"]
        score = 80 - (offset / span) * 60
        zone = "neutral"
    return float(score), {"rsi": rsi_value, "zone": zone}


def _volatility_score(df: pd.DataFrame, close: pd.Series, params: dict) -> tuple[float, dict]:
    atr_series = atr(df, params["atr_period"])
    last_atr = float(atr_series.iloc[-1])
    last_close = float(close.iloc[-1])
    atr_pct = (last_atr / last_close) * 100 if last_close else 0.0

    upper, middle, lower = bollinger(
        close, params["bollinger_period"], params["bollinger_std"]
    )
    bb_width = float((upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1]) if middle.iloc[-1] else 0.0

    # Higher ATR% / wider BB → lower score (more risk).
    if atr_pct < 1.0:
        atr_score = 80
    elif atr_pct < 2.0:
        atr_score = 60
    elif atr_pct < 4.0:
        atr_score = 40
    else:
        atr_score = 20
    if bb_width < 0.04:
        bb_score = 80
    elif bb_width < 0.08:
        bb_score = 60
    elif bb_width < 0.15:
        bb_score = 40
    else:
        bb_score = 20

    score = (atr_score + bb_score) / 2
    return float(score), {
        "atr": last_atr,
        "atr_pct": atr_pct,
        "bb_width": bb_width,
    }


def _verdict(score: float) -> str:
    if score >= 75:
        return "STRONG_TREND"
    if score >= 55:
        return "ACTIONABLE"
    if score >= 40:
        return "WATCHLIST"
    if score >= 25:
        return "WEAK"
    return "AVOID"


def _analyze_symbol(symbol: str, df: pd.DataFrame, params: dict, weights: dict) -> dict:
    close = df["close"].astype(float)

    trend, trend_detail = _trend_score(close, params)
    momentum, momentum_detail = _momentum_score(close, params)
    volatility, volatility_detail = _volatility_score(df, close, params)

    composite = (
        trend * weights["trend_weight"]
        + momentum * weights["momentum_weight"]
        + volatility * weights["volatility_weight"]
    )

    return {
        "symbol": symbol,
        "score": round(composite, 2),
        "verdict": _verdict(composite),
        "pillars": {
            "trend": {"score": trend, **trend_detail},
            "momentum": {"score": momentum, **momentum_detail},
            "volatility": {"score": volatility, **volatility_detail},
        },
    }


def build_report(
    *,
    preset: dict,
    adapter: Any | None = None,
    inputs: Mapping[str, Any] | None = None,
) -> dict:
    inputs = inputs or {}
    symbols = list(preset.get("symbols", []))
    if not symbols:
        symbols = list(inputs.get("symbols") or [])
    timeframe = preset.get("timeframe", "1h")
    lookback = int(preset.get("lookback_bars", 200))
    params = dict(preset.get("indicator_params", {}))
    weights = dict(preset.get("scoring", {})) or {
        "trend_weight": 0.4,
        "momentum_weight": 0.3,
        "volatility_weight": 0.3,
    }

    pillars: list[dict] = []
    signals: list[dict] = []

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

        if len(df) < params.get("ema_long", 200):
            continue
        result = _analyze_symbol(symbol, df, params, weights)
        pillars.append(result)
        signals.append({
            "symbol": symbol,
            "side": "long" if result["pillars"]["trend"]["score"] >= 60 else "neutral",
            "score": result["score"],
            "reason": f"verdict={result['verdict']}; trend={result['pillars']['trend']['score']:.0f}",
        })

    if not pillars:
        return {
            "signals": [],
            "summary": "No symbols had sufficient data for indicator computation.",
            "recommended_next_step": "Provide longer kline history or check adapter returned bars.",
            "results": [],
        }

    # Rank by composite score.
    pillars.sort(key=lambda r: r["score"], reverse=True)
    signals.sort(key=lambda s: s["score"], reverse=True)
    top = pillars[0]
    return {
        "signals": signals,
        "summary": (
            f"Analyzed {len(pillars)} symbol(s). "
            f"Top: {top['symbol']} (score={top['score']:.1f}, verdict={top['verdict']})."
        ),
        "recommended_next_step": (
            "Feed top symbols into structured-trade-plan or three-tier-strategy "
            "for entry/exit zone analysis."
        ),
        "results": pillars,
    }
