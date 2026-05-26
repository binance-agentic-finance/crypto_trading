"""Concrete `MarketDataAdapter` implementations.

Per `docs/architecture/data-layer-design.md`:

- `binance_rest.BinanceRestAdapter` — direct REST to fapi/api.binance.com
  (replaces atomic CLI subprocess wrappers).
- `market_cache.CachedAdapter` — sqlite-backed decorator over any adapter.
- `synthetic.SyntheticAdapter` — deterministic OHLCV for tests / smoke.
- `local_parquet.LocalParquetAdapter` — offline backtest from parquet.
"""

from ai_pro_trading_library.library.data.adapters.binance_pro import (
    BIGDATA_ENDPOINTS,
    WEB3_ENDPOINTS,
    BinanceProAdapter,
)
from ai_pro_trading_library.library.data.adapters.binance_rest import (
    BinanceRestAdapter,
)
from ai_pro_trading_library.library.data.adapters.local_parquet import (
    LocalParquetAdapter,
)
from ai_pro_trading_library.library.data.adapters.market_cache import CachedAdapter
from ai_pro_trading_library.library.data.adapters.synthetic import SyntheticAdapter
from ai_pro_trading_library.library.data.adapters.tradfi import YahooFinanceAdapter

__all__ = [
    "BIGDATA_ENDPOINTS",
    "BinanceProAdapter",
    "BinanceRestAdapter",
    "CachedAdapter",
    "LocalParquetAdapter",
    "SyntheticAdapter",
    "WEB3_ENDPOINTS",
    "YahooFinanceAdapter",
]
