"""Case strategy template.

Each migrated case should expose `build_spec()` and return a StrategySpec.
Runtime modules consume the spec; case files do not run backtests or orders.
"""

from ai_pro_trading_library.library.core.schema import StrategySpec


def build_spec() -> StrategySpec:
    raise NotImplementedError("case strategy templates must define StrategySpec")

