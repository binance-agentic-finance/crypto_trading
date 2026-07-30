"""
Data layer exports for the standard bot architecture.
"""

from .alignment import AlignmentPolicy, assemble_snapshot, resolve_decision_as_of, timeframe_to_ms
from .adapters import BinanceRestMarketDataAdapter, HistoricalJsonMarketDataAdapter
from .derivatives import (
    build_derivatives_path,
    DerivativesDownloadResult,
    HistoricalBinanceDerivativesDownloader,
)
from .liquidations import (
    aggregate_force_order_records,
    build_liquidation_path,
    build_liquidation_raw_path,
    HistoricalBinanceLiquidationRecorder,
    LiquidationCaptureResult,
)
from .downloader import HistoricalBinanceDownloader
from .historical import (
    HistoricalParquetMarketDataAdapter,
    build_history_path,
    determine_download_timeframes,
    ensure_pyarrow_available,
    LocalFirstMarketDataAdapter,
    read_parquet_frame,
    resample_frame_from_1m,
)
from .interfaces import MarketDataAdapter, OnChainDataAdapter, SnapshotAssembler, SocialDataAdapter
from .snapshot import HistoricalSnapshotAssembler
from .selection_assembler import build_selection_snapshot, build_universe_bundle, run_selection
from .unified_snapshot import build_unified_snapshot, universe_klines_to_market_bundle
from .input_bundle import (build_input_bundle, load_input_bundle,
                           write_input_bundle, read_input_bundle)
from .advisory_assembler import build_advisory_snapshot, run_advisory

__all__ = [
    "build_live_bundle",
    "to_panel",
    "AlignmentPolicy",
    "aggregate_force_order_records",
    "BinanceRestMarketDataAdapter",
    "build_derivatives_path",
    "build_liquidation_path",
    "build_liquidation_raw_path",
    "DerivativesDownloadResult",
    "HistoricalBinanceDownloader",
    "HistoricalBinanceDerivativesDownloader",
    "HistoricalBinanceLiquidationRecorder",
    "HistoricalJsonMarketDataAdapter",
    "LocalFirstMarketDataAdapter",
    "LiquidationCaptureResult",
    "HistoricalParquetMarketDataAdapter",
    "HistoricalSnapshotAssembler",
    "MarketDataAdapter",
    "OnChainDataAdapter",
    "SnapshotAssembler",
    "SocialDataAdapter",
    "build_selection_snapshot",
    "build_universe_bundle",
    "run_selection",
    "build_unified_snapshot",
    "build_input_bundle",
    "load_input_bundle",
    "write_input_bundle",
    "read_input_bundle",
    "build_advisory_snapshot",
    "run_advisory",
    "universe_klines_to_market_bundle",
    "assemble_snapshot",
    "build_history_path",
    "determine_download_timeframes",
    "ensure_pyarrow_available",
    "read_parquet_frame",
    "resolve_decision_as_of",
    "resample_frame_from_1m",
    "timeframe_to_ms",
]

# Live fetch -> one cyqnt.input/v1 bundle, and the bar-aligned wide frame
# cyqnt_trd.blocks consumes. Imported lazily-friendly at the bottom so the
# package keeps importing without pandas present.
from .live_bundle import build_live_bundle  # noqa: E402
from .panel import to_panel  # noqa: E402
