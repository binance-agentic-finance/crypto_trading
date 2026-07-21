"""DirectionVoteBlock — majority vote across signal-derived direction hints."""
from __future__ import annotations

from collections import Counter
from typing import ClassVar

from demo_strategy._shared.blocks.base import Block


class DirectionVoteBlock(Block):
    NAME  = "direction_vote"
    LAYER = "decision"
    DEFAULT_PARAMS: ClassVar[dict] = {"tie_breaker": "flat"}

    def compute(self, inputs, **kwargs) -> dict:
        """`inputs` = iterable of strings ∈ {'long','short','flat'} (case-insensitive)."""
        p = self._merge(kwargs)
        votes = [str(v).lower() for v in (inputs or [])
                 if v is not None and str(v).strip()]
        if not votes:
            return {"side": p["tie_breaker"], "votes": {}}
        c = Counter(votes)
        long_n  = c.get("long", 0)
        short_n = c.get("short", 0)
        if long_n > short_n:
            side = "long"
        elif short_n > long_n:
            side = "short"
        else:
            side = p["tie_breaker"]
        return {"side": side, "votes": dict(c)}
