"""Legacy blocks adapter.

This module keeps the old import shape out of the canonical library surface.
New strategies should import from `ai_pro_trading_library.library.*`.
"""

from ai_pro_trading_library.library import features as indicators
from ai_pro_trading_library.library import conditions, scoring

__all__ = ["conditions", "indicators", "scoring"]

