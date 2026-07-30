"""Standard monitor/advisory bot authoring layer.

Bots here are ordinary ``SignalPlugin`` implementations that emit
``SignalKind.ALERT``. They are seeded into the standard registry by
``entrypoints.common.make_registry()`` and executed through
``SignalPluginRegistry.run_pipeline_step`` like every other plugin.

Snapshot assembly lives with the other assemblers, in
:mod:`cyqnt_trd.standard_bot.data.advisory_assembler`
(``build_advisory_snapshot`` / ``run_advisory``), next to
``selection_assembler``. It is not re-exported here so that importing a bot
stays free of the REST/execution dependencies.
"""

from .base import AdvisoryBot
from .contracts import (
    AdvisoryDecision,
    BotMeta,
    DataRequirement,
    Evidence,
    FrameContract,
    FrameValidationError,
    MARKET_METRIC_FRAME,
    NEWS_EVENT_FRAME,
    signals_to_frame,
)
from .data import (
    BinanceMarketMetricAdapter,
    BnDataCatalogAdapter,
    FetchFrame,
    SquarePublicAdapter,
)
from .registry import (
    canvas_definition,
    create_advisory_bot,
    list_advisory_bots,
    register_advisory_bots,
)

__all__ = [
    "AdvisoryBot",
    "AdvisoryDecision",
    "BinanceMarketMetricAdapter",
    "BnDataCatalogAdapter",
    "BotMeta",
    "DataRequirement",
    "Evidence",
    "FetchFrame",
    "FrameContract",
    "FrameValidationError",
    "MARKET_METRIC_FRAME",
    "NEWS_EVENT_FRAME",
    "SquarePublicAdapter",
    "canvas_definition",
    "create_advisory_bot",
    "list_advisory_bots",
    "register_advisory_bots",
    "signals_to_frame",
]
