"""Bridge continuous feature values into discrete Signal objects."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from ai_pro_trading_library.library.core.protocols import Signal
from ai_pro_trading_library.library.scoring.gates import Gate


def signalize(
    feature_name: str,
    value: float,
    rules: Iterable[Gate],
) -> Signal:
    """Convert one continuous feature value through gates into a discrete Signal.

    The first passing gate determines the label and weight. If no gate passes,
    the result is an explicit failed signal, which keeps scoring audits complete.
    """
    for gate in rules:
        if gate.evaluate(float(value)):
            return Signal(
                feature_name=feature_name,
                label=gate.label,
                value=float(value),
                weight=gate.weight,
                passed=True,
                metadata={"operator": gate.operator, "threshold": gate.threshold},
            )
    return Signal(
        feature_name=feature_name,
        label="no_gate_passed",
        value=float(value),
        passed=False,
    )


def signalize_series(
    feature_name: str,
    values: pd.Series,
    rules: Iterable[Gate],
) -> pd.Series:
    """Vectorize `signalize` over a Series.

    Returns a Series of Signal objects aligned with `values.index`. NaN values
    yield a failed Signal with label `nan`.
    """
    rules_list = list(rules)
    out: list[Signal] = []
    for v in values.tolist():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            out.append(Signal(feature_name=feature_name, label="nan", value=float("nan"), passed=False))
        else:
            out.append(signalize(feature_name, v, rules_list))
    return pd.Series(out, index=values.index, dtype=object)
