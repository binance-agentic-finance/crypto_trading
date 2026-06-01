"""shim — atomic.signals.momentum

Forwards momentum helpers to ``cyqnt_trd.compat.atomic_signals``.

Compatibility note:
    The native ``tf_velocity_compute`` / ``multi_tf_velocity`` expect
    ``list[Candle]`` (they read ``.high``, ``.low``, ``.open``, ``.close``,
    ``.volume`` / ``.quote_volume``). Some atomic-era case scripts pass
    ``list[float]`` (closes only). To keep both signatures working, this
    shim wraps both functions and synthesises a minimal Candle proxy from
    a closes list when needed — degraded volume/range fields fall back
    to neutral defaults so downstream pillars don't crash.
"""
from dataclasses import dataclass

from cyqnt_trd.compat.atomic_signals import (  # noqa: F401
    rsi_compute,
    rsi_current,
    rsi_zone_detect,
    macd_compute,
    stochrsi_compute,
    VelocityMetrics,
)
from cyqnt_trd.compat.atomic_signals import (
    tf_velocity_compute as _cy_tf_velocity_compute,
    multi_tf_velocity as _cy_multi_tf_velocity,
)


@dataclass
class _CloseOnlyCandleProxy:
    """Minimal Candle stand-in built from a single close price.

    The native compute reads .open/.high/.low/.close/.volume — all set to
    the same close so volume_ratio / range_ratio degrade to 1.0 (neutral)
    rather than crashing the caller.
    """
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0


def _looks_like_floats(seq) -> bool:
    if not seq:
        return False
    first = seq[0]
    if hasattr(first, "close"):
        return False
    return isinstance(first, (int, float))


def _floats_to_candles(closes):
    return [
        _CloseOnlyCandleProxy(
            open=float(c), high=float(c), low=float(c),
            close=float(c), volume=0.0,
        )
        for c in closes
    ]


def tf_velocity_compute(candles):
    """Per-timeframe velocity. Accepts list[Candle] or list[float]."""
    if _looks_like_floats(candles):
        candles = _floats_to_candles(candles)
    return _cy_tf_velocity_compute(candles)


def multi_tf_velocity(tf_candles: dict) -> dict:
    """Multi-timeframe velocity dict. Each entry may be list[Candle] or list[float]."""
    coerced = {}
    for tf, c in (tf_candles or {}).items():
        coerced[tf] = _floats_to_candles(c) if _looks_like_floats(c) else c
    return _cy_multi_tf_velocity(coerced)
