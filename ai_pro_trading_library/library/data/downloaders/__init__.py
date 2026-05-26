"""Historical data downloaders.

Real REST → parquet ingestion. Output files are compatible with
`LocalParquetAdapter` so the read path stays identical regardless of
whether data was downloaded via these helpers or copied in from
elsewhere.

`requests` + `pyarrow` are required (the `[data]` optional extra).
"""

from ai_pro_trading_library.library.data.downloaders.binance_derivatives import (
    BinanceDerivativesDownloader,
    DerivativesDownloadResult,
    enrich_market_frame_with_derivatives,
)
from ai_pro_trading_library.library.data.downloaders.binance_klines import (
    BinanceKlineDownloader,
    KlineDownloadResult,
)

__all__ = [
    "BinanceDerivativesDownloader",
    "BinanceKlineDownloader",
    "DerivativesDownloadResult",
    "KlineDownloadResult",
    "enrich_market_frame_with_derivatives",
]
