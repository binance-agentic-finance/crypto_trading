"""One live collection -> one ``DataSnapshot`` a block strategy can actually read.

The missing connector
---------------------
Both halves of this path already existed and were never joined:

* :func:`build_live_bundle` calls every catalog node, normalises through the
  node's own vocabulary, gates once on ``available_time``, and reports the status
  of each source — producing a ``cyqnt.input/v1`` dict.
* :func:`load_input_bundle` turns that dict back into a ``DataSnapshot``:
  ``klines`` → ``market``, ``universe`` / ``ticker_rank`` → ``universe``, and
  **every other frame → ``DataSnapshot.frames``**.

But nothing in the runtime called either one. ``assemble_snapshot`` — the builder
the paper and backtest entrypoints use — has no ``frames`` parameter at all, so
the snapshot it produces carries bars and nothing else. A strategy declaring
``needs={"derivatives": True}`` ran anyway, against price alone.

So this module is one function, and its value is entirely in being *called*::

    snapshot = build_live_snapshot(symbol="BTCUSDT", interval="1h")
    batch = plugin.run(snapshot, config)     # make_signals() sees 35 columns, not 13

Why go through the bundle instead of assembling directly
--------------------------------------------------------
The bundle is not an extra hop, it is the artifact that makes the run
reproducible. Collecting straight into a ``DataSnapshot`` would lose three
things, each of which has already caused a bug here:

* **One PIT gate, applied once.** Gating per-source mid-loop dropped every row of
  a source that was fetched 22 seconds after the session clock started.
* **``source_status`` for every declared node.** "I could not read it" and "I read
  it and it was empty" are different facts, and a strategy that abstains needs to
  know which one it is looking at.
* **Replay.** ``write_bundle=`` hands back the exact bytes; feeding them to
  ``load_input_bundle`` reproduces the decision offline, with no network.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Sequence, Tuple

from .input_bundle import load_input_bundle
from .live_bundle import LiveRequest, build_live_bundle, default_live_requests

__all__ = ["build_live_snapshot", "requests_for_sections", "SECTION_NODES"]

#: ``data.<section>:`` in a YAML spec -> the live nodes that carry those columns.
#:
#: The backtest reads these sections from parquet directories (``--derivatives-dir``);
#: paper and live had no equivalent, so the same spec backtested against real
#: funding and then ran against nothing. Mapping the sections onto catalog nodes
#: is what lets one spec mean one thing in both modes.
#:
#: ``{}`` params are filled from the symbol/interval at call time.
SECTION_NODES: Dict[str, Tuple[str, ...]] = {
    "derivatives": ("funding", "open_interest", "taker_volume", "long_short_ratio",
                    "top_trader_ratio"),
    "orderbook": ("orderbook_depth",),
    "news": ("news", "hot_post", "topic_trending", "sentiment"),
    "universe": ("universe", "ticker_rank"),
}


def requests_for_sections(
    sections: Sequence[str],
    *,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 500,
    market_type: str = "futures",
) -> list:
    """Live requests for ``klines`` plus the nodes the declared sections need.

    A spec that declares nothing still gets its bars — the primary series is not
    optional. Unknown section names are ignored rather than raising: the spec
    vocabulary owns validation, and refusing here would turn a naming drift into
    a crash in the data layer.
    """
    plan = default_live_requests(symbol=symbol, interval=interval, limit=limit,
                                 market_type=market_type)
    wanted = {"klines"}
    for section in sections or ():
        wanted.update(SECTION_NODES.get(str(section), ()))
    return [req for req in plan if req[0] in wanted]


def build_live_snapshot(
    *,
    requests: Optional[Sequence[LiveRequest]] = None,
    sections: Optional[Sequence[str]] = None,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 500,
    market_type: str = "futures",
    decision_time: Optional[int] = None,
    positions: Optional[Dict[str, float]] = None,
    equity: Optional[float] = None,
    include_account: bool = False,
    write_bundle: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Collect live, and return ``(snapshot, bundle)``.

    Both are returned on purpose. The snapshot is what the strategy consumes; the
    bundle is what makes the run auditable — it carries ``source_status`` per node
    and ``warnings``, neither of which survives into ``DataSnapshot``. Returning
    only the snapshot would mean a caller that wants to print "open_interest:
    error" has to re-fetch to find out.

    ``sections`` narrows the plan to what a spec declared; ``requests`` overrides
    the plan entirely. With neither, every node a single-instrument decision can
    use is fetched — a failed node then appears in ``source_status`` rather than
    as an absent key.
    """
    if requests is None and sections is not None:
        requests = requests_for_sections(
            sections, symbol=symbol, interval=interval, limit=limit,
            market_type=market_type)

    bundle = build_live_bundle(
        requests=requests, symbol=symbol, interval=interval, limit=limit,
        market_type=market_type, decision_time=decision_time,
        positions=positions, equity=equity, include_account=include_account)

    if write_bundle:
        directory = os.path.dirname(os.path.abspath(write_bundle))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(write_bundle, "w", encoding="utf-8") as handle:
            json.dump(bundle, handle, ensure_ascii=False, indent=2, default=str)

    return load_input_bundle(bundle), bundle
