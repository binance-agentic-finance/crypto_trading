"""
Point-in-time alignment helpers for the standard bot data layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..core import (
    BundleMeta,
    DataSnapshot,
    MarketBundle,
    OnChainSignalBundle,
    SnapshotMeta,
    SocialFeedBundle,
    UniverseBundle,
)


_TIMEFRAME_TO_MS = {
    "1s": 1000,
    "1m": 60 * 1000,
    "3m": 3 * 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "2h": 2 * 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "6h": 6 * 60 * 60 * 1000,
    "8h": 8 * 60 * 60 * 1000,
    "12h": 12 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
    "3d": 3 * 24 * 60 * 60 * 1000,
    "1w": 7 * 24 * 60 * 60 * 1000,
    "1M": 30 * 24 * 60 * 60 * 1000,
}


@dataclass
class AlignmentPolicy:
    policy_id: str = "bar_close_v1"
    primary_timeframe: str = "1m"
    use_observable_at: bool = False
    require_confirmed_onchain: bool = False


def timeframe_to_ms(timeframe: str) -> int:
    if timeframe not in _TIMEFRAME_TO_MS:
        raise ValueError("Unsupported timeframe: %s" % timeframe)
    return _TIMEFRAME_TO_MS[timeframe]


def resolve_decision_as_of(now_ts_ms: int, timeframe: str) -> int:
    """
    Return the close timestamp of the last fully closed bar for ``timeframe``.
    """
    interval_ms = timeframe_to_ms(timeframe)
    closed_bar_open = (now_ts_ms // interval_ms) * interval_ms - interval_ms
    return closed_bar_open + interval_ms - 1


def filter_market_bundle(bundle: Optional[MarketBundle], decision_as_of: int) -> Optional[MarketBundle]:
    if bundle is None:
        return None

    filtered = {}
    for key, bars in bundle.bars.items():
        filtered[key] = [bar for bar in bars if bar.confirmed and bar.timestamp <= decision_as_of]
    return MarketBundle(bars=filtered, meta=bundle.meta)


def filter_social_bundle(
    bundle: Optional[SocialFeedBundle],
    decision_as_of: int,
    policy: AlignmentPolicy,
) -> Optional[SocialFeedBundle]:
    if bundle is None:
        return None

    if policy.use_observable_at:
        items = [
            item for item in bundle.items
            if (item.observable_at or item.published_at) <= decision_as_of
        ]
    else:
        items = [item for item in bundle.items if item.published_at <= decision_as_of]

    return SocialFeedBundle(items=items, query=bundle.query, meta=bundle.meta)


def filter_onchain_bundle(
    bundle: Optional[OnChainSignalBundle],
    decision_as_of: int,
    policy: AlignmentPolicy,
) -> Optional[OnChainSignalBundle]:
    if bundle is None:
        return None

    if policy.require_confirmed_onchain:
        observations = [
            obs for obs in bundle.observations
            if (obs.confirmed_at or obs.timestamp) <= decision_as_of
        ]
    else:
        observations = [
            obs for obs in bundle.observations
            if obs.timestamp <= decision_as_of
        ]

    return OnChainSignalBundle(
        observations=observations,
        watchlist_ref=bundle.watchlist_ref,
        meta=bundle.meta,
    )


def assemble_snapshot(
    version: str,
    snapshot_id: str,
    assembled_at: int,
    policy: AlignmentPolicy,
    market: Optional[MarketBundle] = None,
    social: Optional[SocialFeedBundle] = None,
    onchain: Optional[OnChainSignalBundle] = None,
    decision_as_of: Optional[int] = None,
    partial_ok: bool = True,
    universe: Optional["UniverseBundle"] = None,
) -> DataSnapshot:
    """
    Apply the PIT filtering policy and assemble a ``DataSnapshot``.

    ``universe`` is passed through unfiltered: its frames are cross-sectional
    (one row per symbol at a single ``as_of``) rather than per-bar, so the
    bar-level PIT filters here do not apply. Whoever builds the
    :class:`UniverseBundle` is responsible for PIT-gating it — see
    ``build_universe_bundle``. Accepting it here is what lets ONE snapshot feed
    both a trade strategy (``DataSnapshot.market``) and a selection strategy
    (``DataSnapshot.universe``); before this parameter existed the two lived on
    separate, mutually exclusive assembler paths.
    """
    if decision_as_of is None:
        decision_as_of = assembled_at

    filtered_market = filter_market_bundle(market, decision_as_of)
    filtered_social = filter_social_bundle(social, decision_as_of, policy)
    filtered_onchain = filter_onchain_bundle(onchain, decision_as_of, policy)

    meta = SnapshotMeta(
        snapshot_id=snapshot_id,
        assembled_at=assembled_at,
        partial_ok=partial_ok,
        source_status={},
        decision_as_of=decision_as_of,
        primary_timeframe=policy.primary_timeframe,
        alignment_policy=policy.policy_id,
    )

    return DataSnapshot(
        version=version,
        market=filtered_market,
        social=filtered_social,
        onchain=filtered_onchain,
        meta=meta,
        universe=universe,
    )
