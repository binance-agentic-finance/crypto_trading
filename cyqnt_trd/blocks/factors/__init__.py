"""JoinQuant 风格的按标的统计因子（``cyqnt_trd.blocks.factors``）。

迁移自 ``trading_signal/factor`` 的四类因子 —— **风险 / 技术 / 动量 / 情绪(成交量)**。
签名 ``factor(data_slice, **params) -> float``（方向票 ``1.0/-1.0/0.0``），输入列名用
``open_price/high_price/low_price/close_price/volume/quote_volume``（沿用原实现，未改逻辑）。

与 :mod:`cyqnt_trd.blocks.indicators` 的**向量化整段 Series** 不同，这些是**逐点**因子；
要拿整段、无未来函数的 Series，用 :func:`cyqnt_trd.blocks.alphas.to_series`。经典 TA 的
``{-1,0,1}`` 方向票另见 :mod:`cyqnt_trd.blocks.factor_votes`。
"""
from __future__ import annotations

from . import momentum, risk, sentiment, technical
from .momentum import *  # noqa: F401,F403
from .risk import *  # noqa: F401,F403
from .sentiment import *  # noqa: F401,F403
from .technical import *  # noqa: F401,F403

# Classic point-wise TA factors (migrated from trading_signal/factor). The votes
# themselves are vectorized in cyqnt_trd.blocks.factor_votes; these are the per-symbol
# f(data_slice) -> float wrappers that delegate to it, plus the self-contained TSI factor.
from .adx_factor import adx_factor  # noqa: E402
from .ao_factor import ao_factor  # noqa: E402
from .bbp_factor import bbp_factor  # noqa: E402
from .cci_factor import cci_factor  # noqa: E402
from .ema_factor import ema_cross_factor, ema_factor  # noqa: E402
from .ma_factor import ma_cross_factor, ma_factor  # noqa: E402
from .macd_factor import macd_level_factor  # noqa: E402
from .momentum_factor import momentum_factor  # noqa: E402
from .rsi_factor import rsi_factor  # noqa: E402
from .stochastic_factor import stochastic_k_factor  # noqa: E402
from .stochastic_tsi_factor import stochastic_tsi_fast_factor  # noqa: E402
from .uo_factor import uo_factor  # noqa: E402
from .williams_r_factor import williams_r_factor  # noqa: E402

_CLASSIC = ["ma_factor", "ma_cross_factor", "rsi_factor", "stochastic_k_factor",
            "cci_factor", "adx_factor", "ao_factor", "momentum_factor", "macd_level_factor",
            "stochastic_tsi_fast_factor", "williams_r_factor", "bbp_factor", "uo_factor",
            "ema_factor", "ema_cross_factor"]

__all__ = (_CLASSIC + list(risk.__all__) + list(technical.__all__)
           + list(momentum.__all__) + list(sentiment.__all__))
