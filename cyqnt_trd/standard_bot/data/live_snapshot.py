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

__all__ = ["build_live_snapshot", "requests_for_sections", "SECTION_NODES",
           "FAN_OUT_SECTIONS", "BARS_SECTION"]

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
    "liquidations": ("liquidations",),
    "orderbook": ("orderbook_depth",),
    "news": ("news", "hot_post", "topic_trending", "sentiment", "ticker_rank"),
    "universe": ("universe",),
    # Selection needs one current value for every symbol.  This is deliberately
    # not the historical per-symbol ``funding`` node used by trade strategies.
    "selection_funding": ("funding_snapshot",),
    # What each listed contract IS (asset class + sector tags), joined onto the
    # universe by ``universe.augment_with_contract_meta``.
    "contract_meta": ("contract_meta",),
    # Best bid/ask for every instrument, joined by ``universe.augment_with_spread``.
    # Whole-market in ONE request, so despite being a cross-section it is NOT a
    # fan-out section and needs no roster.
    "selection_book_ticker": ("book_ticker",),
    # How often each perpetual settles funding. Pointless alone and required
    # alongside ``selection_funding``: it is the divisor that turns a
    # per-settlement rate into a comparable annual one.
    "selection_funding_info": ("funding_info",),
    # The three FAN-OUT sections. Each costs one request per instrument — see
    # FAN_OUT_SECTIONS and the ordering table above it.
    "selection_open_interest": ("open_interest_snapshot",),
    "selection_oi_change": ("oi_change_snapshot",),
    "selection_long_short_ratio": ("long_short_ratio_snapshot",),
    # Bars for every candidate at every timeframe the spec's indicator steps name,
    # joined by ``universe.augment_with_indicator``. A fan-out like the three above
    # — see :data:`BARS_SECTION` for the two things that make it different.
    "selection_bars": ("universe_bars",),
}

#: Cross-sectional nodes that are NOT part of ``default_live_requests``.
#:
#: That plan is "every node a single-instrument decision can use", so a
#: whole-market table is not in it and cannot be selected out of it by name. Each
#: entry is appended to the plan instead: ``section -> (node, params, bundle key)``
#: where ``params`` is filled from the call's ``market_type``.
#:
#: ``selection_funding`` is not here because it needs to *displace* the plan's
#: per-symbol ``funding`` request under the same bundle key — see
#: :func:`requests_for_sections`.
#: All three are whole-market reads of ONE request each, which is what makes
#: them appendable rather than fan-out sections: cost does not scale with the
#: roster, so there is no roster to plan and no ordering requirement.
_APPENDED_SECTION_REQUESTS: Dict[str, Tuple[str, str]] = {
    "contract_meta": ("contract_meta", "contract_meta"),
    "selection_book_ticker": ("book_ticker", "book_ticker"),
    "selection_funding_info": ("funding_info", "funding_info"),
}

#: ★ COLLECTION ORDER IS A COST DECISION, AND THE SPREAD IS AN ORDER OF MAGNITUDE.
#:
#: The sections below are the ones with **no all-market endpoint**: Binance
#: answers HTTP 400 when ``symbol`` is omitted from ``/fapi/v1/openInterest`` or
#: ``/futures/data/globalLongShortAccountRatio`` (measured), so a cross-section
#: of them can only be assembled one request per instrument. That makes the plan
#: order the price of the screen:
#:
#: ======================================  ==============  ================
#: plan                                    fan-out calls   measured
#: ======================================  ==============  ================
#: narrow FIRST, then fan out:             127 x 3 = 381   127 s, committed
#: 3 whole-market reads (free) →                           fixture
#: 727 → 127 → fan out all three
#:
#: narrow BETWEEN fan-outs as well:        127 + 41 + 5    9 s, hand-run
#: 127 for open interest → 41 clear the      = 176
#: $5m floor → 5 clear the change
#:
#: fan out first, narrow after             727 x 3 = 2181  cannot run
#: ======================================  ==============  ================
#:
#: The bottom plan does not merely cost more, it *cannot run*: at 1000 requests
#: per 5 min on the whole ``/futures/data/*`` group, the 1454 calls two of the
#: three fan-outs put into that group are 145 % of the window. And the narrowing
#: that makes the top plan affordable is free — it reads columns of a table
#: already in hand (``filter_sub_type``, ``filter_crypto_only``,
#: ``filter_quote_volume``).
#:
#: The middle plan is the one this module does NOT yet reach, and the gap is
#: worth naming precisely: narrowing before the FIRST fan-out is enforced (the
#: joining blocks refuse a roster that does not cover the frame), but narrowing
#: BETWEEN fan-outs needs the plan to be re-derived after each frame lands, and
#: a plan here is a flat list evaluated once. Two thirds of the committed
#: capture's 381 calls are for instruments that the $5m floor discards
#: immediately afterwards.
#:
#: So a caller MUST pass ``fan_out_symbols`` and there is deliberately no
#: default. "The whole universe" would be the expensive plan, silently, on the
#: first spec that asked for open interest.
#:
#: ⚠️ **The runtime does not yet derive that roster by itself.** A live selection
#: run has to collect the cheap sections, apply the spec's own narrowing steps,
#: and only then plan the fan-out — two collection passes with a data dependency
#: between them, which ``build_live_snapshot`` (one plan, one pass) has no shape
#: for. ``scripts/freeze_selection_fixture.py::plan_fan_out_roster`` does exactly
#: this for the committed fixture and is the worked example to generalise from.
#: Until that lands, the roster is the caller's to supply.
#:
#: ``section -> (node, bundle key, extra params beyond symbols/market_type)``
FAN_OUT_SECTIONS: Dict[str, Tuple[str, str, Dict[str, Any]]] = {
    "selection_open_interest": ("open_interest_snapshot",
                                "open_interest_snapshot", {}),
    # A 7-day lookback needs 8 daily readings; the block refuses to average
    # fewer rather than reporting a shorter window under the same column name.
    "selection_oi_change": ("oi_change_snapshot", "oi_change_snapshot",
                            {"period": "1d", "limit": 8}),
    "selection_long_short_ratio": ("long_short_ratio_snapshot",
                                   "long_short_ratio_snapshot",
                                   {"period": "1h", "mode": "global"}),
}

#: The BARS fan-out: ``(section, node, bundle key)``.
#:
#: Kept out of :data:`FAN_OUT_SECTIONS` because it needs two things none of those
#: three do, and folding it in would have to give both a silent default:
#:
#: * a **timeframe set**, which is the union of the timeframes the spec's own
#:   indicator steps name. There is no default: guessing one either misses the
#:   timeframe the spec reads (every indicator column then NaN, which downstream
#:   reads as "no coin matched") or pays for series nothing reads.
#: * an **end instant**. This is the only cross-sectional node whose endpoint is
#:   addressable by time, so a capture for a decision at T must ask for bars as of
#:   T. Omitting it silently pairs a past universe with present prices — for a
#:   screen whose entire subject is price history.
#:
#: Its roster is also a DIFFERENT roster from theirs: the derivative fan-outs are
#: charged for the survivors of the free cross-sectional filters, while the bars
#: are charged only for what survives the derivative filters too — 127 versus 5 on
#: the measured plan. Sharing one ``fan_out_symbols`` would pay 25x.
BARS_SECTION: Tuple[str, str, str] = ("selection_bars", "universe_bars",
                                      "universe_bars")


def requests_for_sections(
    sections: Sequence[str],
    *,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 500,
    market_type: str = "futures",
    fan_out_symbols: Optional[Sequence[str]] = None,
    bar_symbols: Optional[Sequence[str]] = None,
    bar_timeframes: Optional[Sequence[str]] = None,
    bar_limit: int = 200,
    bars_end_ms: Optional[int] = None,
) -> list:
    """Live requests for ``klines`` plus the nodes the declared sections need.

    A spec that declares nothing still gets its bars — the primary series is not
    optional. Unknown section names are ignored rather than raising: the spec
    vocabulary owns validation, and refusing here would turn a naming drift into
    a crash in the data layer.

    ``fan_out_symbols`` is **required** by every section in
    :data:`FAN_OUT_SECTIONS` and has no default — see the ordering table there
    for why "all of them" is not an acceptable fallback.

    ``bar_symbols`` / ``bar_timeframes`` / ``bars_end_ms`` serve
    :data:`BARS_SECTION` and are required by it, for the reasons written there.
    ``bar_symbols`` is a separate roster from ``fan_out_symbols`` on purpose and
    does not fall back to it: the bars are charged for what survives the
    DERIVATIVE filters as well, which on the measured plan is 5 instruments rather
    than 127.
    """
    requested_sections = {str(section) for section in (sections or ())}
    plan = default_live_requests(symbol=symbol, interval=interval, limit=limit,
                                 market_type=market_type)
    wanted = {"klines"}
    for section in requested_sections:
        wanted.update(SECTION_NODES.get(section, ()))
    selected = [req for req in plan if req[0] in wanted]
    if "selection_funding" in requested_sections:
        # The default plan's ``funding`` request is a BTCUSDT history.  Keeping
        # it here would either pass a single-symbol frame to a cross-sectional
        # block or overwrite the same logical key, depending on request order.
        selected = [req for req in selected
                    if not (req[0] == "funding" and req[2] == "funding")]
        selected.append(("funding_snapshot", {}, "funding"))
    for section, (node, key) in _APPENDED_SECTION_REQUESTS.items():
        if section in requested_sections:
            selected.append((node, {"market_type": market_type}, key))

    fan_out = sorted(requested_sections & set(FAN_OUT_SECTIONS))
    if fan_out:
        roster = _fan_out_roster(fan_out, fan_out_symbols)
        for section in fan_out:
            node, key, extra = FAN_OUT_SECTIONS[section]
            selected.append((node, {"symbols": list(roster),
                                    "market_type": market_type, **extra}, key))
    bars_section, bars_node, bars_key = BARS_SECTION
    if bars_section in requested_sections:
        selected.append((bars_node, {
            "symbols": list(_fan_out_roster([bars_section], bar_symbols)),
            "timeframes": list(_bar_timeframes(bar_timeframes)),
            "limit": int(bar_limit),
            "end_ms": _bars_end_ms(bars_end_ms),
            "market_type": market_type,
        }, bars_key))
    _refuse_duplicate_keys(selected)
    return selected


def _bar_timeframes(timeframes: Optional[Sequence[str]]) -> list:
    """Validate the timeframe set the bars fan-out will be charged for."""
    if timeframes is None:
        raise ValueError(
            "section %r collects bars for the timeframes a spec's indicator steps "
            "name, so it needs an explicit bar_timeframes list. There is no "
            "default: a guessed timeframe either misses the one the spec reads — "
            "every indicator column then NaN, which downstream reads as 'no "
            "instrument matched' rather than as a short capture — or pays one "
            "request per instrument for a series nothing reads. Derive it with "
            "yaml_pipeline.interpreter.bar_timeframes_for_spec."
            % BARS_SECTION[0])
    names = [str(value).strip() for value in timeframes]
    names = [name for name in names if name]
    if not names:
        raise ValueError(
            "section %r was given an EMPTY bar_timeframes list. That is not "
            "'collect nothing': it would land a bar frame with no rows, which the "
            "joining block reads as a failed capture and refuses. If the spec has "
            "no indicator step, do not request this section at all."
            % BARS_SECTION[0])
    return names


def _bars_end_ms(end_ms: Optional[int]) -> Optional[int]:
    """The instant the bar histories are cut at, or ``None`` for "now".

    ``None`` is legitimate and is what a LIVE collection passes: the decision is
    being made now, so "the last N bars as of now" is the honest request, and the
    unfinished final candle is dropped by the bundle's own point-in-time gate.
    A REPLAY must pass the bundle's ``decision_time`` instead — see
    :data:`BARS_SECTION`. :func:`build_live_snapshot` does that automatically, so
    this exists to keep the coercion in one place rather than to decide anything.
    """
    return None if end_ms is None else int(end_ms)


def _fan_out_roster(sections: Sequence[str],
                    fan_out_symbols: Optional[Sequence[str]]) -> list:
    """Validate the roster the fan-out sections will be charged for.

    Refuses an absent or empty roster rather than defaulting to the universe:
    that default is the plan the table beside :data:`FAN_OUT_SECTIONS` shows
    cannot run, and it would be chosen by a spec author who never typed it. The
    per-request ceiling is enforced one layer down, by
    :func:`cyqnt_trd.blocks.data._fan_out_roster`, so there is exactly one
    number to change.
    """
    if fan_out_symbols is None:
        raise ValueError(
            "section(s) %s fan out over one request PER INSTRUMENT — Binance "
            "publishes no all-market endpoint for these fields — so they need an "
            "explicit fan_out_symbols roster. There is no default on purpose: "
            "'the whole universe' is 727 x N requests, which does not fit the "
            "rate budget (see FAN_OUT_SECTIONS). Collect the free cross-sections "
            "first, narrow, then pass the survivors."
            % list(sections))
    roster = [str(value).strip().upper() for value in fan_out_symbols]
    roster = [value for value in roster if value]
    if not roster:
        raise ValueError(
            "section(s) %s were given an EMPTY fan_out_symbols roster. That is "
            "not 'collect nothing': it would land an empty frame in the bundle, "
            "which the joining block reads as a failed capture and refuses. If "
            "the narrowing steps legitimately left no instrument, do not request "
            "these sections at all." % list(sections))
    return roster


def _refuse_duplicate_keys(plan: Sequence[LiveRequest]) -> None:
    """No two requests may land under the same bundle key.

    ``build_live_bundle`` writes ``frames[key]`` as it goes, so a repeated key
    means the later request silently overwrites the earlier one and the strategy
    reads a frame it did not ask for — with a ``source_status`` entry that looks
    fine. The live case this guards is a plan combining ``derivatives`` (whose
    per-symbol ``open_interest`` history is a BTCUSDT series) with a
    cross-sectional fan-out section; the two are different shapes under names one
    character apart, which is exactly when a collision is least visible.
    """
    seen: Dict[str, str] = {}
    for node, _params, key in plan:
        if key in seen and seen[key] != node:
            raise ValueError(
                "the collection plan puts two different nodes under the same "
                "bundle key %r (%s and %s). One would overwrite the other and "
                "the strategy would read whichever answered last, so the plan is "
                "refused instead. Request the sections separately, or give one of "
                "them its own key." % (key, seen[key], node))
        seen[key] = node


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
    fan_out_symbols: Optional[Sequence[str]] = None,
    bar_symbols: Optional[Sequence[str]] = None,
    bar_timeframes: Optional[Sequence[str]] = None,
    bar_limit: int = 200,
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

    ``fan_out_symbols`` is only read when ``sections`` names one of
    :data:`FAN_OUT_SECTIONS`, and those sections require it: they cost one request
    per instrument and there is no all-market endpoint to fall back on.

    ``bar_symbols`` / ``bar_timeframes`` serve :data:`BARS_SECTION` and are
    likewise required by it. The bars' ``end_ms`` is NOT a parameter: it is
    ``decision_time`` when one is given and ``None`` (= now) otherwise, which is
    the same replay/live split every other source in this bundle follows. Leaving
    it to the caller would let a replay ask for today's prices beside a frozen
    universe, and the request would look right.
    """
    if requests is None and sections is not None:
        requests = requests_for_sections(
            sections, symbol=symbol, interval=interval, limit=limit,
            market_type=market_type, fan_out_symbols=fan_out_symbols,
            bar_symbols=bar_symbols, bar_timeframes=bar_timeframes,
            bar_limit=bar_limit, bars_end_ms=decision_time)

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
