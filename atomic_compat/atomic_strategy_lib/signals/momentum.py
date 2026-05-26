"""shim — atomic.signals.momentum"""
from cyqnt_trd.compat.atomic_signals import (  # noqa: F401
    rsi_compute,
    rsi_current,
    rsi_zone_detect,
    macd_compute,
    stochrsi_compute,
)


def multi_tf_velocity(*args, **kwargs):
    """Stub — atomic-only helper not yet ported. Returns flat metrics."""
    return {
        "vol_vs_avg": None,
        "vol_accel_3v3": None,
        "range_vs_avg": None,
        "green_streak": 0,
        "red_streak": 0,
        "price_change_pct": 0.0,
        "last3_change_pct": 0.0,
    }


def tf_velocity_compute(*args, **kwargs):
    """Stub — alias for multi_tf_velocity."""
    return multi_tf_velocity(*args, **kwargs)
