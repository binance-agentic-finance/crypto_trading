"""shim — atomic.risk.stop_loss"""
from cyqnt_trd.blocks.stop_loss import (  # noqa: F401
    atr_dynamic_stop,
    fixed_pct_stop,
    trailing_stop,
    nbar_trailing_tp,
)
