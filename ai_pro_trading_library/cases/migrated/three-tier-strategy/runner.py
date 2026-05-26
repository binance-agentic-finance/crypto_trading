"""three-tier-strategy — trade_plan_report runner.

Produces three regime-based scenarios for the target symbol. Each scenario
ships its own leverage cap, stop multiple, target multiple, and risk-per-trade
ceiling. The shared input is current trend score, momentum state, and ATR
volatility — derived once from `library.features.indicators`.

Scenario gating rule: a scenario is marked ``active`` when the current
trend_score >= scenario.min_trend_score. Inactive scenarios still appear in
the output (so the user can compare) but tagged ``hold``.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ai_pro_trading_library.library.features.indicators import atr, ema, rsi


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


def _compute_trend_score(close: pd.Series, periods: dict) -> tuple[float, str, dict]:
    """Returns (score 0-100, bias label, detail)."""
    ema_short = float(ema(close, periods["ema_short"]).iloc[-1])
    ema_mid = float(ema(close, periods["ema_mid"]).iloc[-1])
    ema_long = float(ema(close, periods["ema_long"]).iloc[-1])
    last = float(close.iloc[-1])

    long_aligned = last > ema_short > ema_mid > ema_long
    short_aligned = last < ema_short < ema_mid < ema_long

    score = 0
    if last > ema_short:
        score += 25
    if last > ema_mid:
        score += 25
    if last > ema_long:
        score += 25
    if long_aligned:
        score += 25
    if short_aligned:
        score = 100 - score  # score still positive but representing "short alignment strength"

    bias = "long" if long_aligned else "short" if short_aligned else "neutral"
    return float(score), bias, {
        "ema_short": ema_short,
        "ema_mid": ema_mid,
        "ema_long": ema_long,
        "last_price": last,
    }


def _build_scenario(
    *,
    name: str,
    bias: str,
    trend_score: float,
    last_price: float,
    last_atr: float,
    rsi_value: float,
    sr_low: float,
    sr_high: float,
    cfg: dict,
) -> dict:
    is_active = trend_score >= cfg["min_trend_score"] and bias != "neutral"

    if bias == "long" or (bias == "neutral" and is_active):
        entry = last_price
        stop = entry - cfg["stop_atr"] * last_atr
        target = entry + cfg["target_atr"] * last_atr
        side = "long"
    else:
        entry = last_price
        stop = entry + cfg["stop_atr"] * last_atr
        target = entry - cfg["target_atr"] * last_atr
        side = "short"

    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr = round(reward / risk, 2) if risk > 0 else None

    key_factors = []
    if trend_score >= 70:
        key_factors.append(f"strong trend (score={trend_score:.0f})")
    elif trend_score >= 50:
        key_factors.append(f"moderate trend (score={trend_score:.0f})")
    else:
        key_factors.append(f"weak/no trend (score={trend_score:.0f})")
    if rsi_value < 30:
        key_factors.append("RSI oversold")
    elif rsi_value > 70:
        key_factors.append("RSI overbought")
    else:
        key_factors.append(f"RSI neutral ({rsi_value:.1f})")
    if last_price >= sr_high:
        key_factors.append("at/above resistance")
    elif last_price <= sr_low:
        key_factors.append("at/below support")
    else:
        key_factors.append("inside structural range")

    return {
        "scenario": name,
        "active": is_active,
        "side": side if is_active else "hold",
        "leverage": cfg["leverage"],
        "risk_pct": cfg["risk_pct"],
        "entry_price": round(entry, 4),
        "stop_price": round(stop, 4),
        "target_price": round(target, 4),
        "risk_reward": rr,
        "atr_used": round(last_atr, 4),
        "key_factors": key_factors,
        "rationale": (
            f"{name.title()}: leverage={cfg['leverage']}x, "
            f"stop={cfg['stop_atr']}xATR, target={cfg['target_atr']}xATR, "
            f"risk/trade={cfg['risk_pct']}% NAV"
        ),
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
        symbols = list(inputs.get("symbols") or ["BTCUSDT"])
    timeframe = preset.get("timeframe", "1h")
    lookback = int(preset.get("lookback_bars", 250))
    params = dict(preset.get("indicator_params", {}))
    scenarios_cfg = dict(preset.get("scenarios", {}))

    symbol = symbols[0]
    df = _ohlcv_from_inputs(symbol, inputs)
    if df is None:
        if adapter is None:
            return {
                "signals": [],
                "summary": f"No fixture and no adapter — cannot fetch {symbol}.",
                "recommended_next_step": "Pass adapter=... or supply bars_by_symbol fixture.",
                "scenarios": [],
            }
        from ai_pro_trading_library.library.data.interfaces import KlineRequest

        bars = adapter.fetch_klines(
            KlineRequest(instrument_id=symbol, timeframe=timeframe, limit=lookback)
        )
        df = _bars_to_df(bars)

    if len(df) < params.get("ema_long", 200):
        return {
            "signals": [],
            "summary": f"Insufficient history ({len(df)} bars).",
            "recommended_next_step": "Increase lookback_bars or use longer timeframe.",
            "scenarios": [],
        }

    close = df["close"].astype(float)
    trend_score, bias, ema_detail = _compute_trend_score(close, params)
    rsi_value = float(rsi(close, params["rsi_period"]).iloc[-1])
    last_atr = float(atr(df, params["atr_period"]).iloc[-1])

    sw = int(params.get("structure_window", 50))
    sr_high = float(df["high"].astype(float).iloc[-sw:].max())
    sr_low = float(df["low"].astype(float).iloc[-sw:].min())

    tier_filter = inputs.get("tier") or preset.get("tier")
    scenarios_out: list[dict] = []
    for name, cfg in scenarios_cfg.items():
        if tier_filter and tier_filter != name:
            continue
        scenarios_out.append(_build_scenario(
            name=name,
            bias=bias,
            trend_score=trend_score,
            last_price=ema_detail["last_price"],
            last_atr=last_atr,
            rsi_value=rsi_value,
            sr_low=sr_low,
            sr_high=sr_high,
            cfg=cfg,
        ))

    active_scenarios = [s for s in scenarios_out if s["active"]]
    signals = [
        {
            "symbol": symbol,
            "side": s["side"],
            "score": s["risk_reward"] or 0.0,
            "reason": s["rationale"],
            "scenario": s["scenario"],
        }
        for s in active_scenarios
    ]

    return {
        "signals": signals,
        "summary": (
            f"{symbol} — bias={bias}; trend_score={trend_score:.0f}; "
            f"RSI={rsi_value:.1f}; ATR={last_atr:.4f}; "
            f"{len(active_scenarios)}/{len(scenarios_out)} scenario(s) active."
        ),
        "recommended_next_step": (
            "Compare scenarios; pick one matching your risk profile and "
            "validate it through paper-stage before live execution."
        ),
        "symbol": symbol,
        "bias": bias,
        "trend_score": round(trend_score, 2),
        "rsi": round(rsi_value, 2),
        "atr": round(last_atr, 4),
        "support_resistance": {"support": sr_low, "resistance": sr_high, "window": sw},
        "ema_detail": ema_detail,
        "scenarios": scenarios_out,
    }
