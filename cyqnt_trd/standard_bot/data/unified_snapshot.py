"""Unified DataSnapshot assembly — one snapshot that feeds BOTH strategy kinds.

Why this exists
---------------
``DataSnapshot`` has always had room for both worlds:

* ``DataSnapshot.market``   — per-bar, single-instrument  → read by ``BlockStrategyPlugin`` (trade)
* ``DataSnapshot.universe`` — cross-sectional, many symbols → read by ``SelectionStrategyPlugin`` (selection)

…but **no assembler ever populated both at once**, so the unification was only
type-level, never runtime:

* ``HistoricalSnapshotAssembler`` (trade path) filled ``market`` only; and
  ``assemble_snapshot()`` did not even accept a ``universe`` argument.
* ``build_selection_snapshot`` (selection path) filled ``universe`` only.

Consequence: a hybrid strategy ("rank the universe by news, then confirm the
top-K with klines") could not be expressed on the standard route. N2
(``social_heat_breakout``) works around it by stuffing per-candidate klines into
``UniverseBundle.klines`` — a *third* place bars can live, with a different type
(raw DataFrame) and different keying (bare symbol) from ``MarketBundle``
(``Bar`` objects keyed ``"SYMBOL|TIMEFRAME"``).

This module closes both gaps:

* :func:`build_unified_snapshot` produces ONE ``DataSnapshot`` with ``market``
  and ``universe`` both populated, and **guards the point-in-time invariant**
  between them (a universe snapshot from after the last confirmed bar is
  lookahead — nothing checked this before, because the two never coexisted).
* :func:`universe_klines_to_market_bundle` folds ``UniverseBundle.klines`` into
  a real ``MarketBundle``, so per-candidate bars become reachable through the
  standard ``market`` path instead of a bespoke dict.

Nothing here changes existing behaviour: both legacy assemblers keep working
unchanged, and ``universe=`` on ``assemble_snapshot`` is an optional additive
parameter.
"""

from __future__ import annotations

import time
import uuid
import warnings
from typing import Any, Dict, Optional

from ..core import DataSnapshot, MarketBundle, UniverseBundle
from .alignment import AlignmentPolicy, assemble_snapshot

__all__ = [
    "build_unified_snapshot",
    "universe_klines_to_market_bundle",
    "UNIFIED_SNAPSHOT_NAMESPACE",
]

UNIFIED_SNAPSHOT_NAMESPACE = uuid.UUID("9d1f7c4e-3b52-5a8d-9f16-2c7ab4e05d31")


def _last_confirmed_bar_ts(market: Optional[MarketBundle], primary_timeframe: str) -> Optional[int]:
    """Latest confirmed bar timestamp across the primary-timeframe series."""
    if market is None:
        return None
    stamps = [
        bar.timestamp
        for key, bars in market.bars.items()
        if key.endswith("|%s" % primary_timeframe)
        for bar in bars
        if bar.confirmed
    ]
    return max(stamps) if stamps else None


def universe_klines_to_market_bundle(
    universe_bundle: UniverseBundle,
    *,
    timeframe: str,
    base: Optional[MarketBundle] = None,
    confirmed: bool = True,
) -> MarketBundle:
    """Fold ``universe_bundle.klines`` into a :class:`MarketBundle`.

    ``UniverseBundle.klines`` is a ``{symbol: DataFrame}`` dict (what
    ``build_universe_bundle(kline_top=N)`` fetches for breakout-confirming
    selectors). This converts each frame to ``Bar`` objects and keys them the
    standard way (``"SYMBOL|TIMEFRAME"``), optionally merging into *base*.

    Symbols already present in *base* under the same key are left untouched —
    the richer, PIT-clipped market series wins over the convenience frames.
    """
    from ...blocks.data import df_to_bars

    merged: Dict[str, Any] = dict(base.bars) if base is not None else {}
    for symbol, frame in (universe_bundle.klines or {}).items():
        if frame is None or getattr(frame, "empty", True):
            continue
        sym = str(symbol).upper()
        key = MarketBundle.key(sym, timeframe)
        if key in merged:
            continue
        merged[key] = df_to_bars(frame, sym, timeframe, confirmed=confirmed)
    meta = base.meta if base is not None else universe_bundle.meta
    return MarketBundle(bars=merged, meta=meta)


def build_unified_snapshot(
    *,
    market_bundle: Optional[MarketBundle] = None,
    universe_bundle: Optional[UniverseBundle] = None,
    primary_timeframe: str = "1h",
    decision_as_of: Optional[int] = None,
    version: str = "mvp/v1",
    policy: Optional[AlignmentPolicy] = None,
    partial_ok: bool = True,
    fold_universe_klines: bool = False,
    strict_pit: bool = True,
    **universe_kwargs: Any,
) -> DataSnapshot:
    """Build ONE ``DataSnapshot`` carrying market *and* universe inputs.

    Parameters
    ----------
    market_bundle
        Per-bar inputs for trade strategies. ``None`` → snapshot has no market
        (selection-only, same as ``build_selection_snapshot``).
    universe_bundle
        Pre-built cross-sectional inputs. If ``None`` **and** any
        ``universe_kwargs`` are supplied (``universe_df=`` / ``ticker_rank_df=``
        / ``kline_top=`` / …), one is built via ``build_universe_bundle``.
        Passing neither leaves ``universe`` unset (trade-only, same as
        ``HistoricalSnapshotAssembler``).
    primary_timeframe
        Used to locate the latest confirmed bar and to build the default policy.
    decision_as_of
        PIT cutoff. Defaults to the latest confirmed primary-timeframe bar, or
        ``now`` when there is no market bundle.
    fold_universe_klines
        When True, per-candidate klines from ``UniverseBundle.klines`` are also
        exposed through ``DataSnapshot.market`` via
        :func:`universe_klines_to_market_bundle`.
    strict_pit
        When True (default) a ``universe.as_of`` later than ``decision_as_of``
        raises, because ranking built after the last confirmed bar is lookahead.
        Set False to downgrade to a warning.

    Returns
    -------
    DataSnapshot
        Suitable for ``BlockStrategyPlugin`` (needs ``market``),
        ``SelectionStrategyPlugin`` (needs ``universe``), or both.
    """
    if universe_bundle is None and universe_kwargs:
        from .selection_assembler import build_universe_bundle

        universe_bundle = build_universe_bundle(
            as_of_ms=decision_as_of, **universe_kwargs
        )

    if decision_as_of is None:
        decision_as_of = _last_confirmed_bar_ts(market_bundle, primary_timeframe)
    if decision_as_of is None:
        decision_as_of = (
            int(universe_bundle.as_of)
            if universe_bundle is not None and universe_bundle.as_of
            else int(time.time() * 1000)
        )
    decision_as_of = int(decision_as_of)

    # ---- PIT invariant between the two halves --------------------------------
    # Nothing checked this before because market and universe never coexisted in
    # one snapshot. A universe ranked at T2 combined with bars up to T1 < T2
    # lets a selection strategy see the future.
    if universe_bundle is not None and universe_bundle.as_of:
        u_as_of = int(universe_bundle.as_of)
        if u_as_of > decision_as_of:
            msg = (
                "universe.as_of (%d) is AFTER decision_as_of (%d): the "
                "cross-sectional ranking was built from data the bars have not "
                "reached yet, which is lookahead. Rebuild the UniverseBundle "
                "with as_of_ms=%d, or pass strict_pit=False to override."
                % (u_as_of, decision_as_of, decision_as_of)
            )
            if strict_pit:
                raise ValueError(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

    if fold_universe_klines and universe_bundle is not None:
        market_bundle = universe_klines_to_market_bundle(
            universe_bundle, timeframe=primary_timeframe, base=market_bundle
        )

    if policy is None:
        policy = AlignmentPolicy(
            policy_id="unified_v1", primary_timeframe=primary_timeframe
        )

    snapshot_id = str(
        uuid.uuid5(
            UNIFIED_SNAPSHOT_NAMESPACE,
            "unified|%s|%s|%s|%s"
            % (
                policy.policy_id,
                primary_timeframe,
                decision_as_of,
                "u" if universe_bundle is not None else "-",
            ),
        )
    )

    return assemble_snapshot(
        version=version,
        snapshot_id=snapshot_id,
        assembled_at=decision_as_of,
        policy=policy,
        market=market_bundle,
        decision_as_of=decision_as_of,
        partial_ok=partial_ok,
        universe=universe_bundle,
    )
