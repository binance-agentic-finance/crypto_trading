"""shim — atomic.signals.volume"""
from cyqnt_trd.compat.atomic_signals import (  # noqa: F401
    volume_surge_detect,
    volume_trend,
)


def volume_surge_normalize(*args, **kwargs):
    """Stub — alias to volume_surge_detect for callers expecting 'normalize' name."""
    return volume_surge_detect(*args, **kwargs)
