"""The built-in strategies' three stages for **one live bar**, in plain Python lists.

:mod:`framework_strategies` is the vectorized (pandas) form used by the backtest. Live, the
position is read from the exchange every round, so a strategy only has to decide the *last*
bar: an event there sets the target, no event keeps the held position. That needs no history
beyond the longest window and no pandas — which matters because the submission runtime is not
guaranteed to ship pandas/numpy. ``scripts/build_square_payloads.py`` copies these functions
into the submitted code.

Same stage split and dict contract as the pandas version::

    ① <sid>_factors(bars, p)       bars = {"open","high","low","close","volume": [...], + feeds}
    ② <sid>_forecast(factors, p)   → verdict / score / bias for the last bar
    ③ _sizing(fc, held, stop_pct)  → target position (+1 / 0 / -1) + stop price

Parity is tested bar by bar: with ``held`` = the backtest position on the previous bar, the
verdict and target here equal the pandas strategy's on every bar
(``tests/standard_bot/test_square_submit.py``). Missing values are ``None`` and compare as
False, like NaN does in pandas.
"""
from __future__ import annotations

__all__ = ["LIVE_STAGES", "analyze_last"]


def _sma(xs: list, n: int, end: int | None = None):
    """Mean of the ``n`` values ending before ``end`` (pandas ``rolling(n).mean()``), or None."""
    end = len(xs) if end is None else end
    if n <= 0 or end < n:
        return None
    return sum(xs[end - n:end]) / n


def _gt(a, b) -> bool:
    return a is not None and b is not None and a > b


def _lt(a, b) -> bool:
    return a is not None and b is not None and a < b


def _event(factors: dict, score, *, long=False, short=False, flat=False) -> dict:
    """Last-bar event → forecast dict; precedence long > short > flat, no event = KEEP."""
    verdict = "LONG" if long else "SHORT" if short else "FLAT" if flat else "KEEP"
    bias = {"LONG": "long", "SHORT": "short"}.get(verdict, "neutral")
    return {"price": factors["close"], "verdict": verdict, "score": score, "bias": bias}


def _channel(high: list, low: list, n: int):
    """Donchian bands over the ``n`` bars before the last one (the current bar is excluded)."""
    if len(high) < n + 1 or len(low) < n + 1:
        return None, None
    return max(high[-n - 1:-1]), min(low[-n - 1:-1])


def _channel_position(close, upper, lower):
    if upper is None or lower is None or upper == lower:
        return None
    return 2.0 * (close - (upper + lower) / 2.0) / (upper - lower)


# ------------------------------------------------------------------ ③ sizing
def _sizing(fc: dict, held: float = 0.0, stop_pct: float = 0.03) -> dict:
    """③ forecast → target position + protective stop. No event keeps ``held``."""
    position = {"LONG": 1, "SHORT": -1, "FLAT": 0}.get(fc["verdict"], int(held))
    price = fc["price"]
    stop = (price * (1.0 - stop_pct) if position > 0 else
            price * (1.0 + stop_pct) if position < 0 else None)
    return {"position": position, "stop": stop}


# ------------------------------------------------------- per-strategy ① + ②
def _moving_average_cross_factors(bars: dict, p: dict) -> dict:
    c = bars["close"]
    fast, slow = _sma(c, p["fast_window"]), _sma(c, p["slow_window"])
    spread = (fast - slow) / slow if fast is not None and slow else None
    return {"close": c[-1], "fast_ma": fast, "slow_ma": slow, "spread": spread}


def _moving_average_cross_forecast(factors: dict, p: dict) -> dict:
    spread = factors["spread"]
    return _event(factors, spread, long=_gt(spread, p["entry_threshold"]),
                  flat=_lt(spread, -p["entry_threshold"]))          # SELL -> FLAT (long-only)


def _price_moving_average_factors(bars: dict, p: dict) -> dict:
    c = bars["close"]
    ma, prev_ma = _sma(c, p["period"]), _sma(c, p["period"], len(c) - 1)
    return {"close": c[-1], "ma": ma, "prev_close": c[-2] if len(c) > 1 else None,
            "prev_ma": prev_ma, "spread": (c[-1] - ma) / ma if ma else None}


def _price_moving_average_forecast(factors: dict, p: dict) -> dict:
    close, ma = factors["close"], factors["ma"]
    prev_c, prev_ma = factors["prev_close"], factors["prev_ma"]
    crossed = prev_c is not None and prev_ma is not None and ma is not None
    far = crossed and abs((close - ma) / ma) > p["entry_threshold"]
    up = far and prev_c <= prev_ma and close > ma
    dn = far and prev_c >= prev_ma and close < ma
    return _event(factors, factors["spread"], long=up, flat=dn)


def _rsi_reversion_factors(bars: dict, p: dict) -> dict:
    """RSI with the kernel's **simple** gain/loss mean (not Wilder)."""
    c, period = bars["close"], p["period"]
    rsi = None
    if len(c) >= period + 1:
        d = [c[i] - c[i - 1] for i in range(len(c) - period, len(c))]
        gain = sum(max(x, 0.0) for x in d) / period
        loss = sum(max(-x, 0.0) for x in d) / period
        rsi = 100.0 if loss == 0 else 100.0 - 100.0 / (1.0 + gain / loss)
    return {"close": c[-1], "rsi": rsi}


def _rsi_reversion_forecast(factors: dict, p: dict) -> dict:
    rsi = factors["rsi"]
    return _event(factors, None if rsi is None else (50.0 - rsi) / 50.0,
                  long=_lt(rsi, p["oversold"]), flat=_gt(rsi, p["overbought"]))   # SELL -> FLAT


def _donchian_breakout_factors(bars: dict, p: dict) -> dict:
    upper, lower = _channel(bars["high"], bars["low"], p["lookback_window"])
    return {"close": bars["close"][-1], "upper": upper, "lower": lower}


def _donchian_breakout_forecast(factors: dict, p: dict) -> dict:
    close, upper, lower = factors["close"], factors["upper"], factors["lower"]
    buf = p["breakout_buffer_bps"] / 1e4
    return _event(factors, _channel_position(close, upper, lower),
                  long=upper is not None and close > upper * (1.0 + buf),
                  short=lower is not None and close < lower * (1.0 - buf))   # SELL -> SHORT


def _multi_timeframe_ma_spread_factors(bars: dict, p: dict) -> dict:
    c = bars["close"]
    primary = _sma(c, p["primary_period"])
    secondary = _sma(c, max(2, p["secondary_period"] * p["secondary_factor"]))
    spread = (primary - secondary) / secondary if primary is not None and secondary else None
    return {"close": c[-1], "primary_ma": primary, "secondary_ma": secondary, "spread": spread}


def _multi_timeframe_ma_spread_forecast(factors: dict, p: dict) -> dict:
    spread, thr = factors["spread"], p["threshold_bps"] / 1e4
    return _event(factors, spread, long=_gt(spread, thr), short=_lt(spread, -thr))


def _oi_funding_breakout_factors(bars: dict, p: dict) -> dict:
    upper, lower = _channel(bars["high"], bars["low"], p["lookback_window"])
    oi, funding = bars.get("oi_change_bps"), bars.get("funding_rate_bps")
    return {"close": bars["close"][-1], "upper": upper, "lower": lower,
            "oi_change_bps": oi, "funding_rate_bps": funding,
            "missing": [k for k, v in (("oi_change_bps", oi), ("funding_rate_bps", funding))
                        if v is None]}


def _oi_funding_breakout_forecast(factors: dict, p: dict) -> dict:
    close, upper, lower = factors["close"], factors["upper"], factors["lower"]
    oi, funding = factors["oi_change_bps"], factors["funding_rate_bps"]
    buf = p["breakout_buffer_bps"] / 1e4
    oi_ok = oi is not None and oi >= p["oi_threshold_bps"]
    buy = (oi_ok and upper is not None and close > upper * (1.0 + buf)
           and funding is not None and funding <= p["max_funding_rate_bps"])
    sell = (oi_ok and lower is not None and close < lower * (1.0 - buf)
            and funding is not None and funding >= -p["max_funding_rate_bps"])
    return _event(factors, _channel_position(close, upper, lower), long=buy, short=sell)


#: Built-in strategy id -> (① factors, ② forecast) for the live bar. liquidation_reversal has
#: no platform feed and is not submitted, so it has no live form.
LIVE_STAGES: dict[str, tuple] = {
    "moving_average_cross": (_moving_average_cross_factors, _moving_average_cross_forecast),
    "price_moving_average": (_price_moving_average_factors, _price_moving_average_forecast),
    "rsi_reversion": (_rsi_reversion_factors, _rsi_reversion_forecast),
    "donchian_breakout": (_donchian_breakout_factors, _donchian_breakout_forecast),
    "multi_timeframe_ma_spread": (_multi_timeframe_ma_spread_factors,
                                  _multi_timeframe_ma_spread_forecast),
    "oi_funding_breakout": (_oi_funding_breakout_factors, _oi_funding_breakout_forecast),
}


def analyze_last(strategy_id: str, bars: dict, p: dict, held: float = 0.0,
                 stop_pct: float = 0.03) -> dict:
    """Run the three live stages on the last bar of ``bars``."""
    factors_fn, forecast_fn = LIVE_STAGES[strategy_id]
    factors = factors_fn(bars, p)
    fc = forecast_fn(factors, p)
    size = _sizing(fc, held, stop_pct)
    return {"verdict": fc["verdict"], "score": fc["score"], "bias": fc["bias"],
            "target_position": size["position"], "stop": size["stop"],
            "missing": factors.get("missing", []), "factors": factors}
