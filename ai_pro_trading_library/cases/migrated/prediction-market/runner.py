"""prediction-market — scanner_report runner.

Replaces the old `baw prediction` subprocess pipeline with a JSON-input
contract. The agent (or a future BAW adapter) supplies the markets snapshot
plus OHLCV history; this case computes 7-factor P_true, Edge, Kelly size.

Input shape::

    {
      "capital_usdc": 1000.0,
      "prediction_markets": [
        {
          "market_id": "BTC-15min-77924",
          "symbol": "BTCUSDT",
          "category": "crypto_price",
          "interval": "15min",
          "strike_price": 77924.0,
          "current_spot": 78050.0,
          "end_time_ms": 1718000000000,
          "yes_price": 0.55,
          "no_price": 0.45,
          "liquidity_usdc": 25000.0,
          "funding_rate": 0.00012,
          "long_short_ratio": 1.05,
          "orderbook_imbalance": 0.6,
          "volume_zscore": 1.8
        }
      ],
      "bars_by_symbol": {"BTCUSDT": [...]}
    }
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ai_pro_trading_library.library.features.indicators import ema, macd, rsi


def _signal_rsi(close: pd.Series, scoring: dict) -> float:
    """Returns -1..+1 (bullish positive)."""
    period = scoring["rsi_period"]
    rsi_value = float(rsi(close, period).iloc[-1])
    if rsi_value < scoring["rsi_oversold"]:
        return 1.0
    if rsi_value > scoring["rsi_overbought"]:
        return -1.0
    # Linear interpolation in between.
    span = scoring["rsi_overbought"] - scoring["rsi_oversold"]
    midpoint = (scoring["rsi_overbought"] + scoring["rsi_oversold"]) / 2
    offset = midpoint - rsi_value  # > 0 below midpoint = mild bullish
    return offset / (span / 2)


def _signal_macd(close: pd.Series, scoring: dict) -> float:
    macd_line, signal_line, _hist = macd(
        close, scoring["macd_fast"], scoring["macd_slow"], scoring["macd_signal"]
    )
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        return 1.0
    if macd_line.iloc[-1] < signal_line.iloc[-1]:
        return -1.0
    return 0.0


def _signal_ema_cross(close: pd.Series, scoring: dict) -> float:
    s = float(ema(close, scoring["ema_short"]).iloc[-1])
    l = float(ema(close, scoring["ema_long"]).iloc[-1])
    if s > l:
        return 1.0
    if s < l:
        return -1.0
    return 0.0


def _signal_funding(funding_rate: float | None) -> float:
    """Negative funding → contrarian long; positive → contrarian short."""
    if funding_rate is None:
        return 0.0
    if funding_rate <= -0.0005:
        return 1.0
    if funding_rate >= 0.0005:
        return -1.0
    # Linear in between.
    return -funding_rate / 0.0005


def _signal_long_short(ratio: float | None) -> float:
    if ratio is None:
        return 0.0
    # ratio > 1.5 → crowded long → contrarian short.
    if ratio >= 1.5:
        return -1.0
    if ratio <= 0.7:
        return 1.0
    return 0.0


def _signal_orderbook_imbalance(imbalance: float | None) -> float:
    """imbalance > 0.5 means bid pressure heavier → bullish."""
    if imbalance is None:
        return 0.0
    return max(-1.0, min(1.0, (imbalance - 0.5) * 2.0))


def _signal_volume_breakout(zscore: float | None) -> float:
    if zscore is None:
        return 0.0
    if zscore >= 2.0:
        return 1.0
    if zscore >= 1.0:
        return 0.5
    return 0.0


def _kelly_fraction(p_true: float, market_price: float) -> float:
    """Full Kelly for a binary outcome at decimal odds = 1/market_price."""
    if market_price <= 0 or market_price >= 1:
        return 0.0
    # b = (1 / market_price) - 1; p = p_true; q = 1 - p_true
    b = (1.0 / market_price) - 1.0
    p = p_true
    q = 1.0 - p_true
    if b == 0:
        return 0.0
    f = (b * p - q) / b
    return max(0.0, f)


def _analyze_market(market: dict, bars_by_symbol: dict, scoring: dict) -> dict:
    weights = scoring["signal_weights"]
    weight_total = sum(weights.values())
    sym = market.get("symbol", "")
    bars = bars_by_symbol.get(sym, [])
    df = pd.DataFrame(bars) if bars else pd.DataFrame()
    have_klines = (
        not df.empty
        and {"close"} <= set(df.columns)
        and len(df) >= max(scoring["ema_long"], scoring["macd_slow"]) + 5
    )

    if have_klines:
        close = df["close"].astype(float)
        rsi_s = _signal_rsi(close, scoring)
        macd_s = _signal_macd(close, scoring)
        ema_s = _signal_ema_cross(close, scoring)
    else:
        rsi_s = 0.0
        macd_s = 0.0
        ema_s = 0.0

    funding_s = _signal_funding(market.get("funding_rate"))
    ls_s = _signal_long_short(market.get("long_short_ratio"))
    ob_s = _signal_orderbook_imbalance(market.get("orderbook_imbalance"))
    vb_s = _signal_volume_breakout(market.get("volume_zscore"))

    weighted = (
        rsi_s * weights["rsi"]
        + macd_s * weights["macd"]
        + ema_s * weights["ema_cross"]
        + funding_s * weights["funding_rate"]
        + ls_s * weights["long_short_ratio"]
        + ob_s * weights["orderbook_imbalance"]
        + vb_s * weights["volume_breakout"]
    )
    # Map [-weight_total, +weight_total] → [0, 1] for P_true.
    p_true_yes = 0.5 + (weighted / (2 * weight_total))
    p_true_yes = max(0.01, min(0.99, p_true_yes))
    return {
        "p_true_yes": round(p_true_yes, 4),
        "have_klines": have_klines,
        "signal_components": {
            "rsi": round(rsi_s, 3),
            "macd": round(macd_s, 3),
            "ema_cross": round(ema_s, 3),
            "funding_rate": round(funding_s, 3),
            "long_short_ratio": round(ls_s, 3),
            "orderbook_imbalance": round(ob_s, 3),
            "volume_breakout": round(vb_s, 3),
        },
    }


def build_report(
    *,
    preset: dict,
    adapter: Any | None = None,
    inputs: Mapping[str, Any] | None = None,
) -> dict:
    inputs = inputs or {}
    markets = list(inputs.get("prediction_markets") or [])
    if not markets:
        return {
            "signals": [],
            "summary": "No prediction_markets in inputs.",
            "recommended_next_step": "Provide inputs.prediction_markets snapshot.",
            "candidates": [],
        }

    capital = float(inputs.get("capital_usdc") or 1000.0)
    bars_by_symbol = inputs.get("bars_by_symbol") or {}
    scoring = preset["scoring"]
    execution = preset["execution"]

    rows: list[dict] = []
    for market in markets:
        if market.get("liquidity_usdc", 0) < execution["min_liquidity_usdc"]:
            rows.append({**market, "decision": "SKIP_LIQUIDITY", "edge_yes": None,
                         "edge_no": None, "kelly_fraction": 0.0, "size_usdc": 0.0})
            continue

        analysis = _analyze_market(market, bars_by_symbol, scoring)
        p_yes = analysis["p_true_yes"]
        p_no = 1.0 - p_yes
        yes_price = float(market.get("yes_price", 0.5))
        no_price = float(market.get("no_price", 1.0 - yes_price))

        edge_yes = p_yes - yes_price
        edge_no = p_no - no_price

        if edge_yes >= execution["edge_threshold"]:
            side = "yes"
            edge = edge_yes
            full_kelly = _kelly_fraction(p_yes, yes_price)
        elif edge_no >= execution["edge_threshold"]:
            side = "no"
            edge = edge_no
            full_kelly = _kelly_fraction(p_no, no_price)
        else:
            side = None
            edge = max(edge_yes, edge_no)
            full_kelly = 0.0

        scaled_kelly = full_kelly * execution["kelly_fraction"]
        capped_kelly = min(scaled_kelly, execution["single_market_max_pct"])
        size_usdc = round(capital * capped_kelly, 2)

        rows.append({
            "market_id": market.get("market_id"),
            "symbol": market.get("symbol"),
            "interval": market.get("interval"),
            "strike_price": market.get("strike_price"),
            "current_spot": market.get("current_spot"),
            "yes_price": yes_price,
            "no_price": no_price,
            "p_true_yes": p_yes,
            "edge_yes": round(edge_yes, 4),
            "edge_no": round(edge_no, 4),
            "decision": "BUY" if side else "SKIP_EDGE",
            "side": side,
            "kelly_full": round(full_kelly, 4),
            "kelly_scaled": round(capped_kelly, 4),
            "size_usdc": size_usdc,
            "signal_components": analysis["signal_components"],
            "have_klines": analysis["have_klines"],
        })

    # Apply portfolio gates.
    rows.sort(key=lambda r: max(r.get("edge_yes") or 0, r.get("edge_no") or 0), reverse=True)
    actionable = [r for r in rows if r["decision"] == "BUY"]
    capped_count = actionable[: execution["daily_market_max_count"]]

    portfolio_pct = sum(r["size_usdc"] for r in capped_count) / capital if capital else 0
    if portfolio_pct > execution["portfolio_max_pct"]:
        scale = execution["portfolio_max_pct"] / portfolio_pct
        for r in capped_count:
            r["size_usdc"] = round(r["size_usdc"] * scale, 2)
            r["kelly_scaled"] = round(r["kelly_scaled"] * scale, 4)

    signals = [
        {
            "symbol": r["market_id"],
            "side": r["side"],
            "score": r.get("edge_yes") if r["side"] == "yes" else r.get("edge_no"),
            "reason": (
                f"edge={r['edge_yes'] if r['side']=='yes' else r['edge_no']:.4f}; "
                f"P_yes={r['p_true_yes']:.3f}; size={r['size_usdc']} USDC"
            ),
        }
        for r in capped_count
    ]

    return {
        "signals": signals,
        "summary": (
            f"Scanned {len(markets)} market(s); {len(actionable)} positive Edge, "
            f"{len(capped_count)} actionable after daily cap. "
            f"Portfolio allocation: ${sum(r['size_usdc'] for r in capped_count):.2f}/${capital:.2f}"
        ),
        "recommended_next_step": (
            "Place orders via BAW wallet for each BUY decision; monitor till market end_time."
        ),
        "capital_usdc": capital,
        "actionable_count": len(capped_count),
        "actionable": capped_count,
        "all_markets": rows,
    }
