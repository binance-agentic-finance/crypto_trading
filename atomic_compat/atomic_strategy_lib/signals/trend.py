"""shim — atomic.signals.trend"""
from cyqnt_trd.compat.atomic_signals import (  # noqa: F401
    ema_compute,
    ema_current,
    ema_cross_detect,
    adx_compute,
    supertrend_compute,
    trend_classify,
)


def multi_tf_resonance(*args, **kwargs):
    """Stub — atomic-only multi-timeframe resonance helper. Returns weak signal."""
    return {"resonance": False, "agreement_pct": 0.0, "tf_breakdown": {}}


def multi_tf_resonance_from_klines(*args, **kwargs):
    """Stub — alias."""
    return multi_tf_resonance(*args, **kwargs)
