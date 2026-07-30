"""Which DataFrame column comes from which declared data source.

A YAML spec says *what data it wants* under ``data:``; the loader attaches the
corresponding columns; the interpreter feeds those columns to blocks. Those
three places have to agree on the column names, so the names live here once.

They are not invented: each tuple is exactly what the runtime enrichment
produces, so a spec that dry-runs against the synthetic frame sees the same
column names it will see against real data.

* ``derivatives`` — :func:`standard_bot.data.derivatives.enrich_market_frame_with_derivatives`
* ``liquidations`` — :func:`standard_bot.data.liquidations.enrich_market_frame_with_liquidations`

Why this matters more than it looks: without it, a spec referencing
``funding_rate_bps`` either fails validation against an OHLCV-only synthetic
frame (so the whole ``derivatives`` block family was unusable), or — worse, if
the synthetic frame were simply widened — validates and then meets an absent
column at runtime. Declaring the source is what makes the dry-run honest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

__all__ = [
    "DataSection",
    "DATA_SECTIONS",
    "PRICE_COLUMNS",
    "column_owner",
    "declared_sections",
    "synthetic_columns",
]

#: Columns every OHLCV frame carries, from any source.
PRICE_COLUMNS: Tuple[str, ...] = (
    "open_time", "open", "high", "low", "close", "volume", "quote_volume",
    "close_time", "timestamp", "trades",
)


@dataclass(frozen=True)
class DataSection:
    """One optional ``data.<key>:`` section and the columns it attaches."""

    key: str
    #: columns the enrichment merges onto the bar frame
    columns: Tuple[str, ...]
    #: mvp_backtest flag that points the event engine at the same files
    cli_flag: str
    #: what a spec author writes to turn it on
    example: str
    description: str


DATA_SECTIONS: Dict[str, DataSection] = {
    "derivatives": DataSection(
        key="derivatives",
        columns=(
            "open_interest", "open_interest_value", "oi_change_bps",
            "funding_rate", "funding_rate_bps", "mark_price",
        ),
        cli_flag="--derivatives-dir",
        example="data:\n  derivatives: { dir: data/derivatives_mvp_30d }",
        description="funding rate + open interest, merged as-of each bar close",
    ),
    "liquidations": DataSection(
        key="liquidations",
        columns=(
            "long_liq_count", "short_liq_count", "long_liq_qty", "short_liq_qty",
            "long_liq_notional_usd", "short_liq_notional_usd",
            "total_liq_notional_usd", "net_liq_notional_usd", "liq_imbalance_ratio",
        ),
        cli_flag="--liquidations-dir",
        example="data:\n  liquidations: { dir: data/liquidations_mvp_live }",
        description="forced-liquidation counts and notionals per bar",
    ),
}

#: column -> the ``data:`` section that provides it
_OWNER: Dict[str, str] = {
    column: section.key
    for section in DATA_SECTIONS.values()
    for column in section.columns
}


def column_owner(column: str) -> Optional[str]:
    """Return the ``data:`` section key that provides *column*, or ``None``.

    ``None`` means the column is either OHLCV (always present) or unknown to
    this vocabulary — the caller decides which.
    """
    return _OWNER.get(column)


def declared_sections(spec: Dict[str, object]) -> Tuple[str, ...]:
    """Section keys the spec actually turned on, in declaration order."""
    data = spec.get("data") or {}
    if not isinstance(data, dict):
        return ()
    return tuple(key for key in DATA_SECTIONS if data.get(key))


def synthetic_columns(spec: Dict[str, object]) -> Tuple[str, ...]:
    """Columns the dry-run frame must carry for this spec's declared sources."""
    out: list[str] = []
    for key in declared_sections(spec):
        out.extend(DATA_SECTIONS[key].columns)
    return tuple(out)
