"""Case signal template.

Each migrated case should expose `build_signal(spec, data) -> pd.Series`.
The signal is registered with `default_registry` automatically when
`ai_pro_trading_library.cases.migrated` is imported.
"""

from __future__ import annotations

import pandas as pd

from ai_pro_trading_library.library.core.protocols import StrategySpec


def build_signal(spec: StrategySpec, data: pd.DataFrame) -> pd.Series:
    raise NotImplementedError("case signal templates must implement build_signal")
