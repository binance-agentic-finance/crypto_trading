"""structured-trade-plan — trade_plan_report runner.

For one symbol, builds three risk-tier trade plans (conservative / balanced /
aggressive). Each tier carries entry zone, stop, target, and risk:reward ratio
derived from current price, EMA-based trend bias, and ATR.

Output is advisory only. Execution (paper or live) belongs in a runtime call,
not in this case.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ai_pro_trading_library.library.features.indicators import atr, ema


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


def _trend_bias(close: pd.Series, periods: dict) -> tuple[str, dict]:
    ema_short = float(ema(close, periods["short"]).iloc[-1])
    ema_mid = float(ema(close, periods["mid"]).iloc[-1])
    ema_long = float(ema(close, periods["long"]).iloc[-1])
    last = float(close.iloc[-1])

    if last > ema_short > ema_mid > ema_long:
        bias = "long"
    elif last < ema_short < ema_mid < ema_long:
        bias = "short"
    else:
        bias = "neutral"

    return bias, {
        "ema_short": ema_short,
        "ema_mid": ema_mid,
        "ema_long": ema_long,
        "last_price": last,
    }


def _support_resistance(df: pd.DataFrame, window: int) -> dict:
    high_win = df["high"].astype(float).iloc[-window:]
    low_win = df["low"].astype(float).iloc[-window:]
    return {
        "support": float(low_win.min()),
        "resistance": float(high_win.max()),
        "window": int(len(low_win)),
    }


def _build_tier(
    *,
    name: str,
    bias: str,
    last_price: float,
    last_atr: float,
    support: float,
    resistance: float,
    entry_atr_distance: float,
    stop_atr: float,
    target_atr: float,
) -> dict:
    if bias == "long":
        # Pull entry slightly below the last price for fills.
        entry_low = last_price - entry_atr_distance * last_atr
        entry_high = last_price
        stop = max(entry_low - stop_atr * last_atr, support - 0.1 * last_atr)
        target = entry_high + target_atr * last_atr
    elif bias == "short":
        entry_low = last_price
        entry_high = last_price + entry_atr_distance * last_atr
        stop = min(entry_high + stop_atr * last_atr, resistance + 0.1 * last_atr)
        target = entry_low - target_atr * last_atr
    else:
        # Neutral bias — quote both sides; pick longs by default but mark hold.
        entry_low = last_price - entry_atr_distance * last_atr
        entry_high = last_price
        stop = entry_low - stop_atr * last_atr
        target = entry_high + target_atr * last_atr

    midpoint = (entry_low + entry_high) / 2
    risk = abs(midpoint - stop)
    reward = abs(target - midpoint)
    rr = round(reward / risk, 2) if risk > 0 else None

    return {
        "tier": name,
        "side": bias if bias != "neutral" else "hold",
        "entry_low": round(entry_low, 4),
        "entry_high": round(entry_high, 4),
        "stop_price": round(stop, 4),
        "target_price": round(target, 4),
        "risk_reward": rr,
        "atr_used": round(last_atr, 4),
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
    if not symbols:
        return {
            "signals": [],
            "summary": "No symbol provided for trade-plan generation.",
            "recommended_next_step": "Pass --inputs '{\"symbols\":[\"ETHUSDT\"]}' or set preset.symbols.",
            "tiers": [],
        }

    timeframe = preset.get("timeframe", "1h")
    lookback = int(preset.get("lookback_bars", 200))
    ema_periods = dict(preset.get("ema_periods", {"short": 20, "mid": 50, "long": 200}))
    atr_period = int(preset.get("atr_period", 14))
    sr_window = int(preset.get("support_resistance", {}).get("lookback_window", 50))
    tier_specs = dict(preset.get("tiers", {}))

    symbol = symbols[0]
    df = _ohlcv_from_inputs(symbol, inputs)
    if df is None:
        if adapter is None:
            return {
                "signals": [],
                "summary": f"No fixture and no adapter — cannot fetch {symbol}.",
                "recommended_next_step": "Pass adapter=... or supply bars_by_symbol fixture.",
                "tiers": [],
            }
        from ai_pro_trading_library.library.data.interfaces import KlineRequest

        bars = adapter.fetch_klines(
            KlineRequest(instrument_id=symbol, timeframe=timeframe, limit=lookback)
        )
        df = _bars_to_df(bars)

    if len(df) < ema_periods["long"]:
        return {
            "signals": [],
            "summary": f"Insufficient history ({len(df)} bars) — need at least {ema_periods['long']}.",
            "recommended_next_step": "Increase lookback_bars or use a longer timeframe.",
            "tiers": [],
        }

    close = df["close"].astype(float)
    bias, ema_detail = _trend_bias(close, ema_periods)
    sr = _support_resistance(df, sr_window)
    last_atr = float(atr(df, atr_period).iloc[-1])
    last_price = ema_detail["last_price"]

    tier_filter = inputs.get("tier") or preset.get("tier")
    tiers_out: list[dict] = []
    for tier_name, tier_cfg in tier_specs.items():
        if tier_filter and tier_filter != tier_name:
            continue
        tiers_out.append(_build_tier(
            name=tier_name,
            bias=bias,
            last_price=last_price,
            last_atr=last_atr,
            support=sr["support"],
            resistance=sr["resistance"],
            entry_atr_distance=float(tier_cfg["entry_atr_distance"]),
            stop_atr=float(tier_cfg["stop_atr"]),
            target_atr=float(tier_cfg["target_atr"]),
        ))

    signals = [
        {
            "symbol": symbol,
            "side": tier["side"],
            "score": tier["risk_reward"] or 0.0,
            "reason": f"{tier['tier']} tier (bias={bias}, atr={tier['atr_used']})",
            "tier": tier["tier"],
        }
        for tier in tiers_out
    ]

    return {
        "signals": signals,
        "summary": (
            f"{symbol} — bias={bias}; last={last_price:.2f}; ATR={last_atr:.4f}; "
            f"support={sr['support']:.2f}; resistance={sr['resistance']:.2f}; "
            f"{len(tiers_out)} tier(s) generated."
        ),
        "recommended_next_step": (
            "Pick one tier and run paper-stage to validate before live execution."
        ),
        "symbol": symbol,
        "bias": bias,
        "ema_detail": ema_detail,
        "support_resistance": sr,
        "atr": last_atr,
        "tiers": tiers_out,
    }
