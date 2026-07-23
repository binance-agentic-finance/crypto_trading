"""
Snapshot assembly helpers for the standard bot MVP.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Union

from ..core import Bar, DataSnapshot, MarketBundle
from .alignment import AlignmentPolicy, assemble_snapshot

SNAPSHOT_NAMESPACE = uuid.UUID("56a92c97-71c1-5197-9042-c8e4208e9ec8")


@dataclass
class HistoricalSnapshotAssembler:
    policy: AlignmentPolicy
    snapshot_version: str = "mvp/v1"
    tail_bars: Union[int, Dict[str, int]] = 200
    # When tail_bars is a single int and a non-primary timeframe is present,
    # apply this larger cap so that HTF rolling-window indicators (e.g. SMA-200
    # on 4h while primary is 1h) have enough history. Set to <=0 to disable
    # the HTF override and use ``tail_bars`` for every timeframe (legacy
    # behavior).
    htf_tail_bars: int = 1000

    def build(self, market_bundle: MarketBundle) -> list[DataSnapshot]:
        anchors = self._collect_anchor_points(market_bundle)
        snapshots: list[DataSnapshot] = []
        for decision_as_of in anchors:
            clipped_bundle = self._clip_market_bundle(market_bundle, decision_as_of)
            snapshot_id = str(
                uuid.uuid5(
                    SNAPSHOT_NAMESPACE,
                    "%s|%s|%s"
                    % (self.policy.policy_id, self.policy.primary_timeframe, decision_as_of),
                )
            )
            snapshots.append(
                assemble_snapshot(
                    version=self.snapshot_version,
                    snapshot_id=snapshot_id,
                    assembled_at=int(time.time() * 1000),
                    policy=self.policy,
                    market=clipped_bundle,
                    decision_as_of=decision_as_of,
                    partial_ok=False,
                )
            )
        return snapshots

    def _collect_anchor_points(self, market_bundle: MarketBundle) -> list[int]:
        anchors: list[int] = []
        for key, bars in market_bundle.bars.items():
            if not key.endswith("|%s" % self.policy.primary_timeframe):
                continue
            anchors.extend(bar.timestamp for bar in bars if bar.confirmed)
        return sorted(set(anchors))

    def _clip_market_bundle(self, market_bundle: MarketBundle, decision_as_of: int) -> MarketBundle:
        clipped: Dict[str, List[Bar]] = {}
        primary_tf = self.policy.primary_timeframe
        for key, bars in market_bundle.bars.items():
            visible = [bar for bar in bars if bar.confirmed and bar.timestamp <= decision_as_of]
            tf = key.rsplit("|", 1)[-1] if "|" in key else key
            tail = self._tail_for(tf, primary_tf)
            if tail > 0:
                visible = visible[-tail:]
            clipped[key] = visible
        return MarketBundle(bars=clipped, meta=market_bundle.meta)

    def _tail_for(self, tf: str, primary_tf: str) -> int:
        """Return per-timeframe tail-bars cap.

        - If ``tail_bars`` is a dict, look up by tf with fallback "default".
        - Else for the primary tf use ``tail_bars`` directly; for HTFs use
          ``max(tail_bars, htf_tail_bars)`` so non-primary timeframes are
          not over-clipped.
        """
        if isinstance(self.tail_bars, dict):
            if tf in self.tail_bars:
                return int(self.tail_bars[tf])
            return int(self.tail_bars.get("default", 0))
        base = int(self.tail_bars)
        if tf == primary_tf:
            return base
        if self.htf_tail_bars and self.htf_tail_bars > 0:
            return max(base, int(self.htf_tail_bars))
        return base
