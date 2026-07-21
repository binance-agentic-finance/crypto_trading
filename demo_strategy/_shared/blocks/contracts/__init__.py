"""Contract dataclasses — one per file so a UI card can render/edit
each schema in isolation.

Import surface unchanged: `from demo_strategy._shared.blocks.contracts
import StrategyContext, StrategyDecision, …` still works.
"""
from .market_data       import MarketData
from .account_data      import AccountData
from .strategy_context  import StrategyContext
from .strategy_decision import StrategyDecision, flat_decision
from .selection_context import SelectionContext
from .selection_decision import SelectionDecision
from .template_meta     import TemplateMeta

__all__ = [
    "MarketData", "AccountData",
    "StrategyContext", "StrategyDecision", "flat_decision",
    "SelectionContext", "SelectionDecision",
    "TemplateMeta",
]
