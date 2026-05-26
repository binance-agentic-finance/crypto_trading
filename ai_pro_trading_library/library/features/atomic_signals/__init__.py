"""Atomic-shape signal detectors (dict-returning).

Per `docs/migration/overlap-policy.md` Class B: these helpers exist
parallel to the canonical pandas surface in `library.features.*`.
They preserve the atomic dict-return shape for consumers that expect
that schema; numerical correctness is L2 parity vs
`atomic_strategy_lib.signals.*`.
"""

from ai_pro_trading_library.library.features.atomic_signals.computes import (
    VelocityMetrics,
    adx_compute,
    atr_compute,
    atr_current,
    atr_ratio,
    bb_squeeze_detect,
    bollinger_compute,
    dual_speed_atr,
    ema_compute,
    ema_cross_detect,
    ema_current,
    macd_compute,
    multi_tf_resonance,
    multi_tf_resonance_from_klines,
    multi_tf_velocity,
    rsi_compute,
    rsi_current,
    rsi_zone_detect,
    stochrsi_compute,
    supertrend_compute,
    tf_velocity_compute,
    trend_classify,
    trend_strength,
)
from ai_pro_trading_library.library.features.atomic_signals.derivatives_detectors import (
    crowding_detect,
    funding_extreme_detect,
    oi_anomaly_detect,
)
from ai_pro_trading_library.library.features.atomic_signals.structure import (
    box_range_construct,
    candlestick_pattern_detect,
    fibonacci_levels,
    structural_pivot_identify,
    structural_quality_score,
    symmetry_structure_detect,
    top_bottom_count,
)
from ai_pro_trading_library.library.features.atomic_signals.volume_detectors import (
    volume_surge_detect,
    volume_surge_normalize,
    volume_trend,
)

__all__ = [
    "VelocityMetrics",
    "adx_compute",
    "atr_compute",
    "atr_current",
    "atr_ratio",
    "bb_squeeze_detect",
    "bollinger_compute",
    "box_range_construct",
    "candlestick_pattern_detect",
    "crowding_detect",
    "dual_speed_atr",
    "ema_compute",
    "ema_cross_detect",
    "ema_current",
    "fibonacci_levels",
    "funding_extreme_detect",
    "macd_compute",
    "multi_tf_resonance",
    "multi_tf_resonance_from_klines",
    "multi_tf_velocity",
    "oi_anomaly_detect",
    "rsi_compute",
    "rsi_current",
    "rsi_zone_detect",
    "stochrsi_compute",
    "structural_pivot_identify",
    "structural_quality_score",
    "supertrend_compute",
    "symmetry_structure_detect",
    "tf_velocity_compute",
    "top_bottom_count",
    "trend_classify",
    "trend_strength",
    "volume_surge_detect",
    "volume_surge_normalize",
    "volume_trend",
]
