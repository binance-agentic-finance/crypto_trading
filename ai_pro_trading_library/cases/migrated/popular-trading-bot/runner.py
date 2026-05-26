"""popular-trading-bot — scanner_report runner.

Recommends Binance's 7 trading-bot archetypes given a market regime + capital.
Produces ready-to-use parameter templates per bot. **No simulator runs here**
— the case is a planning step; running the bots themselves belongs to a
future ``runtime.bots`` package.

Input shape (all optional except where noted)::

    {
      "market_regime": "ranging" | "trending_up" | "trending_down" | "high_volatility",
      "capital_usdc": 1000.0,
      "symbol": "BTCUSDT",
      "bars": [{open, high, low, close, volume, timestamp}, ...]   # used to auto-detect regime if no market_regime supplied
    }
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ai_pro_trading_library.library.features.indicators import atr, ema


VALID_REGIMES = ("ranging", "trending_up", "trending_down", "high_volatility")


def _detect_regime(bars: list[dict]) -> tuple[str, dict]:
    """Auto-detect from EMA slope + ATR/price ratio."""
    df = pd.DataFrame(bars)
    if len(df) < 60 or "close" not in df.columns:
        return "ranging", {"reason": "insufficient_history"}

    close = df["close"].astype(float)
    ema_20 = ema(close, 20).iloc[-1]
    ema_50 = ema(close, 50).iloc[-1]
    last = float(close.iloc[-1])
    atr_pct = float(atr(df, 14).iloc[-1]) / last * 100 if last else 0.0

    if atr_pct >= 4.0:
        regime = "high_volatility"
    elif last > ema_20 > ema_50 and (last / ema_50 - 1) > 0.02:
        regime = "trending_up"
    elif last < ema_20 < ema_50 and (1 - last / ema_50) > 0.02:
        regime = "trending_down"
    else:
        regime = "ranging"

    return regime, {
        "ema_20": float(ema_20),
        "ema_50": float(ema_50),
        "atr_pct": round(atr_pct, 3),
        "last_price": last,
    }


def build_report(
    *,
    preset: dict,
    adapter: Any | None = None,
    inputs: Mapping[str, Any] | None = None,
) -> dict:
    inputs = inputs or {}
    regime = inputs.get("market_regime")
    detection_detail: dict | None = None
    if not regime:
        bars = list(inputs.get("bars") or [])
        if bars:
            regime, detection_detail = _detect_regime(bars)
        else:
            regime = "ranging"
    if regime not in VALID_REGIMES:
        return {
            "signals": [],
            "summary": f"Invalid market_regime={regime!r}; allowed: {VALID_REGIMES}.",
            "recommended_next_step": "Pass inputs.market_regime or supply bars for auto-detection.",
            "candidates": [],
        }

    capital = float(inputs.get("capital_usdc") or preset.get("default_capital_usdc", 1000.0))
    symbol = inputs.get("symbol") or preset.get("default_symbol", "BTCUSDT")

    fitness_table = preset["regime_fitness"]
    templates = preset["templates"]
    risk_caps = preset["risk_caps"]

    rows: list[dict] = []
    for bot_id, fitness_by_regime in fitness_table.items():
        score = fitness_by_regime.get(regime, 0)
        template = dict(templates.get(bot_id, {}))
        # Cap leverage by risk_caps.
        if "leverage" in template and template["leverage"] > risk_caps["max_leverage"]:
            template["leverage_capped_from"] = template["leverage"]
            template["leverage"] = risk_caps["max_leverage"]
        # Inject capital into template.
        template["capital_usdc"] = capital
        template["symbol"] = symbol
        template["account_drawdown_pct"] = risk_caps["account_drawdown_pct"]
        template["require_confirm"] = risk_caps["require_confirm"]

        rows.append({
            "bot_id": bot_id,
            "regime_score": score,
            "regime_fit": "excellent" if score >= 85 else "good" if score >= 65 else "moderate" if score >= 45 else "poor",
            "template": template,
        })

    rows.sort(key=lambda r: r["regime_score"], reverse=True)
    top3 = rows[:3]

    signals = [
        {
            "symbol": symbol,
            "side": "config",
            "score": r["regime_score"],
            "reason": f"{r['bot_id']} fits {regime} regime ({r['regime_fit']})",
            "bot_id": r["bot_id"],
        }
        for r in top3
    ]

    return {
        "signals": signals,
        "summary": (
            f"Regime: {regime}. "
            f"Top bot: {top3[0]['bot_id']} (score {top3[0]['regime_score']}). "
            f"7 bot templates ready for {symbol} @ ${capital:.0f}."
        ),
        "recommended_next_step": (
            "Pick one of the top-3 templates and run it through runtime.bots simulator "
            "(forthcoming); paper-stage before live deployment."
        ),
        "regime": regime,
        "regime_detection_detail": detection_detail,
        "capital_usdc": capital,
        "symbol": symbol,
        "risk_caps": risk_caps,
        "top3": top3,
        "all_bots": rows,
    }
