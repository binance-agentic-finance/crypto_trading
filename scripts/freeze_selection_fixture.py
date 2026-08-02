"""Freeze one live Binance cross-section into a replayable ``cyqnt.input/v1``.

Why this script exists
----------------------
A selection spec run through ``yaml_pipeline run`` fetches the universe live, so
its basket changes every minute. That makes the output untestable in the only way
that matters: when the basket changes you cannot tell whether the code broke or
the market moved. Nothing in ``standard_bot/core`` reads the wall clock during a
selection decision (the basket's ``signal_id`` is
``uuid5(snapshot_id|plugin_id|selection|as_of)``), so freezing the *input* is
enough to make the *output* a golden file.

    # capture live, trim, write the committed fixture
    python scripts/freeze_selection_fixture.py

    # re-cut an existing raw capture without touching the network
    python scripts/freeze_selection_fixture.py \
        --in tmp/raw_capture.json --out tests/standard_bot/fixtures/x.json

    # land ONE new registry frame in the committed fixture, leaving the rest of it
    # (and therefore every golden basket pinned against it) byte-identical
    python scripts/freeze_selection_fixture.py --add-frame contract_meta \
        --in tests/standard_bot/fixtures/universe_cross_section.json \
        --out tests/standard_bot/fixtures/universe_cross_section.json

    # the other two bundles, each its own capture and its own decision_time
    python scripts/freeze_selection_fixture.py --derivatives
    python scripts/freeze_selection_fixture.py --liquidity

Two details are load-bearing and were both found the hard way:

* the capture asks for sections **explicitly**. ``live_sections_for_spec`` on
  ``example_selection.yaml`` returns only ``universe,news`` — it derives
  ``selection_funding`` from a spec saying ``with: [funding]``, and the shipped
  examples do not. Driving the capture off a spec would therefore freeze a
  fixture with no cross-sectional funding frame in it, and the next spec that
  wants funding would have to re-capture.
* ``--sections news`` drags in ``news``/``hot_post``/``topic_trending``/
  ``sentiment`` as well (see ``live_snapshot.SECTION_NODES``) — 400 KB of prose
  no cross-sectional block reads. Only ``ticker_rank`` is kept. The frames are
  dropped *with* their ``source_status`` entries and warnings, because
  ``run_bundle`` copies ``source_status`` onto every emitted signal and a status
  line for a frame that is not in the bundle is a lie a reader cannot check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:                 # runnable as `python scripts/...`
    sys.path.insert(0, _REPO_ROOT)

from cyqnt_trd.standard_bot.core.input_contract import INPUT_SCHEMA_VERSION
from cyqnt_trd.standard_bot.data.input_bundle import write_input_bundle

#: Sections handed to the live collector. ``selection_funding`` is the
#: all-market ``premiumIndex`` snapshot; the per-symbol ``funding`` history is a
#: different node and cannot stand in for a cross-section.
CAPTURE_SECTIONS = ("universe", "news", "selection_funding", "contract_meta")

#: Frames a cross-sectional (``selection:``) spec can actually read. Everything
#: else the capture returns is dropped.
#:
#: ``ticker_rank`` stays because ``example_selection.yaml`` declares
#: ``with: [ticker_rank]``; ``funding`` (the ``funding_snapshot`` node, stored
#: under the logical key ``funding``) stays because a funding-ranked spec is the
#: obvious next one to pin. ``contract_meta`` stays because "no TradFi" / "AI
#: sector" cannot be expressed without it. ``klines`` is dropped: a selection
#: ranks a cross-section and never touches bars.
KEEP_FRAMES = ("universe", "ticker_rank", "funding", "contract_meta")

DEFAULT_OUT = os.path.join(
    "tests", "standard_bot", "fixtures", "universe_cross_section.json")

# ---------------------------------------------------------------------------
# The derivatives fixture: a SECOND frozen bundle, and why it is a second one
# ---------------------------------------------------------------------------
#
# Open interest, its weekly change and the long/short ratio have no all-market
# endpoint, so a cross-section of them is a FAN-OUT: one request per instrument,
# over a roster chosen from the universe. Two consequences shape this file.
#
# **It cannot be --add-frame'd into the existing fixture.** ``ADDABLE_NODES``
# admits a registry — rows that say what an instrument IS — precisely because
# such rows do not go stale between the frozen capture and now. These three
# MEASURE a market: today's open interest stamped with a decision_time from four
# days ago is a fabrication, and the guard above is right to refuse it.
#
# **And it cannot be a re-capture of the existing fixture either**, because that
# would move the golden baskets in ``test_selection_fixture_replay.py`` and
# ``test_universe_contract_meta.py`` — and every line of prose quoting them — for
# a reason unrelated to the change. So: a separate bundle, whose universe,
# registry and derivative frames all come from ONE collection, with its own
# decision_time. Nothing in ``universe_cross_section.json`` moves.
#
# The capture is two-phase because the plan has a data dependency: the roster is
# chosen FROM the universe, so a throwaway planning read comes first and the
# bundle is collected once afterwards. See ``plan_fan_out_roster``.

DERIVATIVES_SECTIONS = (
    "universe", "contract_meta",
    "selection_open_interest", "selection_oi_change",
    "selection_long_short_ratio",
)

#: Frames kept in the derivatives fixture. ``ticker_rank`` and ``funding`` are
#: NOT here: the screen this fixture exists to pin (position inventory and its
#: weekly change) reads neither, and each would add ~100 KB of rows no assertion
#: touches. The mark price that converts open interest to dollars travels inside
#: ``open_interest_snapshot`` itself, on purpose — see
#: ``blocks.data.fetch_open_interest_cross_section``.
DERIVATIVES_KEEP = (
    "universe", "contract_meta",
    "open_interest_snapshot", "oi_change_snapshot", "long_short_ratio_snapshot",
)

DERIVATIVES_OUT = os.path.join(
    "tests", "standard_bot", "fixtures", "universe_derivatives.json")

#: Sector tags the fan-out roster is narrowed to before anything is fetched.
#:
#: This is the "板塊" step of the request this fixture serves, and it is what
#: makes the capture affordable: on the frozen cross-section ``Alpha`` (71) plus
#: ``AI`` (57), minus the one contract carrying both, is exactly 127 of 727
#: instruments — the measured 176-request / 9-second plan. Fanning out first
#: instead would be 2181 requests and would not fit the rate budget at all.
DERIVATIVES_SUB_TYPES = ("Alpha", "AI")

# ---------------------------------------------------------------------------
# The liquidity + funding-APR fixture: a THIRD frozen bundle
# ---------------------------------------------------------------------------
#
# Two questions share one capture because they share one requirement: every frame
# has to be read in the SAME pass as the universe it is joined to.
#
# **The book.** ``bookTicker`` is whole-market in one request, so unlike the
# derivatives it is free — but it is the fastest-moving quantity in the catalog and
# ``ADDABLE_NODES`` is right to refuse it: stamping this second's spread with a
# four-day-old decision_time is a fabrication about the one field whose whole value
# is being current.
#
# **The funding schedule.** ``fundingInfo`` looks like a registry and is *almost*
# addable — but the venue rewrites it (443 of 743 contracts have been moved to
# 4-hourly settlement), so back-filling today's intervals onto an older funding
# snapshot would annualise past rates with a divisor those contracts may not have
# had. That is a WRONG number rather than a missing one, which is worse.
#
# So: one more bundle, one decision_time, one PIT gate. ``universe_cross_section``
# and ``universe_derivatives`` are both untouched, and their golden baskets with
# them.

LIQUIDITY_SECTIONS = (
    "universe", "selection_funding", "selection_funding_info",
    "selection_book_ticker",
)

#: Frames kept in the liquidity fixture.
#:
#: ``contract_meta`` and ``ticker_rank`` are NOT here: neither screen this fixture
#: pins reads them, and each would add ~100 KB of rows no assertion touches.
LIQUIDITY_KEEP = ("universe", "funding", "funding_info", "book_ticker")

LIQUIDITY_OUT = os.path.join(
    "tests", "standard_bot", "fixtures", "universe_liquidity.json")

#: Turnover floor the verification screens to before checking for vacuity.
#:
#: The interesting claims are all about instruments a real screen would keep: that
#: a $100m-turnover coin can still have a wide book, and that annualising reorders
#: a *tradable* basket. Checking them across all 727 rows would let the venue's
#: long tail of untraded listings satisfy every condition, which proves nothing
#: about the screens anyone writes.
LIQUIDITY_TURNOVER_FLOOR = 2e7


def capture_liquidity(*, symbol: str, interval: str, market_type: str,
                      sections: Sequence[str] = LIQUIDITY_SECTIONS,
                      ) -> Dict[str, Any]:
    """One-pass capture of the universe, its book, and its funding schedule.

    Single-phase, unlike :func:`capture_derivatives`: none of these nodes fans
    out, so there is no roster to plan and no data dependency between the reads.
    Four requests total.
    """
    from cyqnt_trd.standard_bot.data.live_snapshot import build_live_snapshot

    _snapshot, bundle = build_live_snapshot(
        sections=list(sections), symbol=symbol, interval=interval,
        market_type=market_type)
    return bundle


def _verify_liquidity(bundle: Dict[str, Any]) -> None:
    """Refuse a liquidity fixture that could not show its own screens working.

    Same discipline as :func:`_verify_derivatives`, run through the blocks that
    will read the frames so a fixture that passes here cannot fail a coverage
    guard inside a test.

    The third check is the one this fixture exists for. ``fundingRateApr`` is only
    worth a column if annualising actually REORDERS a real market; on a capture
    where every instrument happened to settle 8-hourly the multiplier would be a
    constant and every APR test would pass while proving nothing. So the capture
    is refused unless the settlement intervals genuinely differ AND the top of the
    two rankings genuinely disagrees.
    """
    import pandas as pd

    from cyqnt_trd.blocks import universe as universe_blocks

    frames = bundle.get("frames") or {}

    def frame(key: str):
        return pd.DataFrame((frames.get(key) or {}).get("rows") or [])

    universe = frame("universe")
    if universe.empty:
        raise FreezeError("the capture has no universe frame to verify against")

    try:
        with_book = universe_blocks.augment_with_spread(universe, frame("book_ticker"))
        joined = universe_blocks.augment_with_funding(
            with_book, frame("funding"), frame("funding_info"))
    except ValueError as exc:
        raise FreezeError(
            "the captured liquidity frames do not fit this universe, so the "
            "fixture would fail inside every test that replays it: %s" % exc)

    liquid = universe_blocks.filter_quote_volume(joined, LIQUIDITY_TURNOVER_FLOOR)
    if len(liquid) < 20:
        raise FreezeError(
            "only %d instrument(s) clear the $%.0fm turnover floor, which is too "
            "few to say anything about a ranking. Recapture, or lower "
            "LIQUIDITY_TURNOVER_FLOOR and re-justify it."
            % (len(liquid), LIQUIDITY_TURNOVER_FLOOR / 1e6))

    checks = {
        "spread_bps <= 5": (liquid["spread_bps"] <= 5.0),
        "top_of_book_usd >= 10000": (liquid["top_of_book_usd"] >= 10_000.0),
        "fundingRateApr < 0": (liquid["fundingRateApr"] < 0.0),
    }
    vacuous = [name for name, mask in checks.items()
               if not bool(mask.any()) or bool(mask.all())]
    if vacuous:
        raise FreezeError(
            "on this capture the condition(s) %s hold for every liquid instrument "
            "or for none, so a test using them could not fail. Do not commit it — "
            "recapture, or move the thresholds to where this market splits."
            % vacuous)

    intervals = sorted(liquid["fundingIntervalHours"].dropna().unique().tolist())
    if len(intervals) < 2:
        raise FreezeError(
            "every liquid instrument in this capture settles funding on the same "
            "%s-hour schedule, so annualising is a constant multiplier and cannot "
            "reorder anything. A test pinned to this fixture would pass whether or "
            "not fundingRateApr works. Recapture; the venue normally serves a mix "
            "of 8 / 4 / 1." % (intervals or ["?"]))

    top = 10
    by_raw = list(liquid.nsmallest(top, "fundingRatePct")["symbol"])
    by_apr = list(liquid.nsmallest(top, "fundingRateApr")["symbol"])
    if by_raw == by_apr:
        raise FreezeError(
            "the %d most negative instruments by fundingRatePct and by "
            "fundingRateApr are the same list in the same order on this capture, so "
            "it cannot demonstrate the mis-ranking the annualised column exists to "
            "fix (intervals present: %s). Recapture — this is a property of the "
            "hour, not of the code." % (top, intervals))


#: Nodes ``--add-frame`` may back-fill into an ALREADY FROZEN bundle, and why
#: each one is safe to stamp with that bundle's own ``decision_time``.
#:
#: The operation exists because the alternative is worse. When a new
#: cross-sectional node lands, recapturing the whole fixture also replaces the
#: universe, the ticker ranks and the funding snapshot with a different hour of
#: market — which invalidates every golden basket pinned against it, and every
#: line of prose quoting those baskets, for a reason unrelated to the change
#: being made. A reviewer then cannot tell the new capability from the market
#: having moved, which is the exact confusion the fixture exists to remove.
#:
#: It is an explicit allowlist and not an availability check, because
#: ``FORWARD_ONLY`` is not the property that matters. ``universe`` and
#: ``funding_snapshot`` are FORWARD_ONLY too, and back-filling either would be a
#: fabrication: they MEASURE a market that has since moved. These are different —
#: a REGISTRY, whose rows state what an instrument is rather than what it did.
#: The values are still not eternal (a symbol gets listed, delisted, re-tagged),
#: so ``--add-frame`` verifies the addition against the frozen universe it is
#: joining and refuses a hole rather than committing one.
ADDABLE_NODES = {
    "contract_meta": (
        "listing registry: contract type, underlying asset class and sector tags. "
        "States what an instrument IS, not what it did, so it does not go stale "
        "between the frozen capture and now the way a price or a funding rate does."
    ),
    "universe_bars": (
        "OHLCV history, addressable BY TIME. This is the second and different "
        "reason a frame may be back-filled: not that its values are timeless, but "
        "that the endpoint takes an `endTime`, so asking for `end_ms = "
        "decision_time` returns exactly the bars a capture at that instant would "
        "have seen. It is the only cross-sectional node in the catalog with "
        "availability BACKTESTABLE, and the property is CHECKED rather than "
        "trusted: `--bars` refuses the addition unless every bar closed at or "
        "before the target's decision_time. (`universe`, `funding_snapshot`, "
        "`book_ticker` and the three fan-outs serve only 'now' and stay refused.) "
        "What does NOT replay is the ROSTER — it is planned from a forward-only "
        "universe snapshot, so these bars answer for the instruments THAT bundle "
        "contained, which is correct for that bundle and is not a claim about any "
        "other instant."
    ),
}


class FreezeError(RuntimeError):
    """The capture cannot be frozen into a fixture worth committing."""


def capture_live(*, symbol: str, interval: str, market_type: str,
                 sections: Sequence[str] = CAPTURE_SECTIONS) -> Dict[str, Any]:
    """Collect the declared sections live and return the raw bundle dict."""
    from cyqnt_trd.standard_bot.data.live_snapshot import build_live_snapshot

    _snapshot, bundle = build_live_snapshot(
        sections=list(sections), symbol=symbol, interval=interval,
        market_type=market_type)
    return bundle


def plan_fan_out_roster(*, market_type: str = "futures",
                        sub_types: Sequence[str] = DERIVATIVES_SUB_TYPES) -> List[str]:
    """Choose the roster the fan-out will be charged for, before fetching it.

    Phase one of a two-phase capture, and the reason the whole plan fits in a
    rate budget: the two reads here are whole-market (one ticker table, one
    ``exchangeInfo``) and the narrowing is a column filter over a table already
    in hand, so cutting 727 instruments down to 127 costs nothing and removes
    1800 requests from phase two.

    The values read here are **thrown away**: only the surviving symbol NAMES are
    used, and the bundle's own universe frame — collected in phase two, inside the
    session that stamps the decision time — is the one that gets ranked. That is
    deliberate. Reusing this frame would put two collection instants in one
    artifact; discarding it means the only thing crossing the phase boundary is a
    list of names, and a name that has since been delisted shows up as a
    coverage hole in the joining block rather than as a silently short basket.

    This is also the worked example the runtime still needs (see the note beside
    ``live_snapshot.FAN_OUT_SECTIONS``): a live selection run has to derive this
    roster from the spec's own narrowing steps rather than from a hard-coded
    sector list.
    """
    import pandas as pd

    from cyqnt_trd.blocks import data as blocks_data
    from cyqnt_trd.blocks import universe as universe_blocks

    tickers = blocks_data.fetch_24h_tickers(market_type=market_type)
    registry = blocks_data.fetch_contract_meta(market_type=market_type)
    joined = universe_blocks.augment_with_contract_meta(tickers, registry)
    # Crypto only first: a tokenised-equity perpetual has open interest too, and
    # paying for it inside a capped roster is paying to screen something the
    # request excludes.
    joined = universe_blocks.filter_crypto_only(joined)
    narrowed = universe_blocks.filter_sub_type(joined, include=list(sub_types))
    roster = sorted({str(value).upper() for value in narrowed["symbol"]})
    if not roster:
        raise FreezeError(
            "the sector filter %s left no instrument, so there is nothing to fan "
            "out over and the fixture would be empty. Check the tag spelling "
            "against the venue's own vocabulary — universe.filter_sub_type warns "
            "with the tags actually present." % list(sub_types))
    if len(roster) > blocks_data.FAN_OUT_MAX_SYMBOLS:
        raise FreezeError(
            "the sector filter %s left %d instruments, above the %d fan-out "
            "ceiling. Narrow further (add a turnover floor) rather than raising "
            "the ceiling: the roster size IS the request count."
            % (list(sub_types), len(roster), blocks_data.FAN_OUT_MAX_SYMBOLS))
    return roster


def capture_derivatives(*, symbol: str, interval: str, market_type: str,
                        sub_types: Sequence[str] = DERIVATIVES_SUB_TYPES,
                        sections: Sequence[str] = DERIVATIVES_SECTIONS,
                        ) -> Tuple[Dict[str, Any], List[str]]:
    """Two-phase capture: plan the roster, then collect the whole bundle once.

    Returns ``(bundle, roster)``. Phase two is a single
    :func:`build_live_snapshot` call, so the bundle has one ``decision_time``
    ("collection_complete") and one PIT gate — the same guarantees a one-phase
    capture of seventeen nodes has, since that one also spans tens of seconds.
    """
    from cyqnt_trd.standard_bot.data.live_snapshot import build_live_snapshot

    roster = plan_fan_out_roster(market_type=market_type, sub_types=sub_types)
    _snapshot, bundle = build_live_snapshot(
        sections=list(sections), symbol=symbol, interval=interval,
        market_type=market_type, fan_out_symbols=roster)
    return bundle, roster


def _verify_derivatives(bundle: Dict[str, Any]) -> None:
    """Refuse a derivatives fixture that could not show its own filters working.

    Same discipline as :func:`_verify_addition`: the check runs the blocks that
    will read the frames, so a fixture that passes here cannot fail a coverage
    guard inside a test — where the finger would point at the test.

    The second half is about vacuity. A capture in which every instrument clears
    the $5m floor lets ``filter_open_interest`` pass its test by dropping
    nothing, which is indistinguishable from the filter not running.
    """
    import pandas as pd

    from cyqnt_trd.blocks import universe as universe_blocks

    frames = bundle.get("frames") or {}
    universe = pd.DataFrame((frames.get("universe") or {}).get("rows") or [])
    if universe.empty:
        raise FreezeError("the capture has no universe frame to verify against")

    joined = universe_blocks.augment_with_contract_meta(
        universe, pd.DataFrame((frames.get("contract_meta") or {}).get("rows") or []))
    narrowed = universe_blocks.filter_sub_type(
        universe_blocks.filter_crypto_only(joined),
        include=list(DERIVATIVES_SUB_TYPES))
    try:
        with_oi = universe_blocks.augment_with_open_interest(
            narrowed,
            pd.DataFrame((frames.get("open_interest_snapshot") or {}).get("rows") or []))
        with_change = universe_blocks.augment_with_oi_change(
            with_oi,
            pd.DataFrame((frames.get("oi_change_snapshot") or {}).get("rows") or []))
        with_ls = universe_blocks.augment_with_long_short_ratio(
            with_change,
            pd.DataFrame(
                (frames.get("long_short_ratio_snapshot") or {}).get("rows") or []))
    except ValueError as exc:
        raise FreezeError(
            "the captured derivative frames do not fit their own roster, so the "
            "fixture would fail inside every test that replays it: %s" % exc)

    checks = {
        "oi_notional_usd >= 5e6": (with_ls["oi_notional_usd"] >= 5e6),
        "|oi_change_pct| >= 20": (with_ls["oi_change_pct"].abs() >= 20.0),
        "long_account_pct >= 60": (with_ls["long_account_pct"] >= 60.0),
    }
    vacuous = [name for name, mask in checks.items()
               if not bool(mask.any()) or bool(mask.all())]
    if vacuous:
        raise FreezeError(
            "on this capture the condition(s) %s hold for every instrument or for "
            "none, so a test using them could not fail. Do not commit it — "
            "recapture, or move the thresholds to where this market splits."
            % vacuous)


#: Bars per (instrument, timeframe) in a ``--bars`` capture.
#:
#: 100 and not 500, for two reasons that point the same way. Weight: a klines call
#: is weight 1 at limit<=100 and 2 above it (measured), so the whole plan for the
#: E5 screen is 5 requests x 1 timeframe x 1 = 5 weight. Size: this fixture is
#: committed, and 400 extra bars per pair is ~80 KB of rows no assertion reads.
#:
#: It has to stay >= 91: the three-month screen asks for a 90-bar daily window and
#: the newest row of each series is the unfinished candle that the PIT gate drops.
BARS_LIMIT = 100


def capture_bars(bundle: Dict[str, Any], spec_paths: Sequence[str], *,
                 market_type: str = "futures",
                 limit: int = BARS_LIMIT) -> Tuple[Dict[str, Any], Any]:
    """Two-pass: plan the roster from *bundle* + the specs, then fetch the bars.

    Pass 1 touches no network at all — every frame it narrows is already in the
    frozen bundle — so the only cost is Pass 2, one request per
    (surviving instrument, timeframe). That asymmetry IS the plan: on the measured
    capture it is 5 requests instead of 727 per timeframe.

    Several specs may be planned together, and one bundle serving several specs is
    the normal case here (``universe_cross_section.json`` already serves four). The
    union is taken rather than one frame per spec, because two frames under two keys
    would need two ``with:`` names and a spec would have to know which capture it
    came from.

    Returns ``(frozen bundle, plan)``.
    """
    from cyqnt_trd.standard_bot.yaml_pipeline.bundle_runner import (
        BarsPlan, plan_bars_capture,
    )
    from cyqnt_trd.standard_bot.yaml_pipeline.spec import load_spec

    symbols: List[str] = []
    timeframes: List[str] = []
    prefix_steps = 0
    for spec_path in spec_paths:
        plan = plan_bars_capture(load_spec(spec_path), bundle)
        symbols.extend(name for name in plan.symbols if name not in symbols)
        timeframes.extend(name for name in plan.timeframes if name not in timeframes)
        prefix_steps = max(prefix_steps, plan.survivors_of)
    merged = BarsPlan(symbols=sorted(symbols), timeframes=timeframes,
                      end_ms=int(bundle["decision_time"]),
                      survivors_of=prefix_steps)
    frozen = add_frame(bundle, "universe_bars", market_type=market_type, params={
        "symbols": list(merged.symbols),
        "timeframes": list(merged.timeframes),
        "limit": int(limit),
        # PIT guard #1. Without it the request means "as of now" and the frame
        # would pair a frozen universe with today's prices — for a screen whose
        # whole subject is price history.
        "end_ms": merged.end_ms,
    })
    return frozen, merged


def add_frame(bundle: Dict[str, Any], node: str, *,
              market_type: str = "futures",
              params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Collect one registry node live and land it in an already-frozen *bundle*.

    Everything else in the bundle — ``decision_time``, ``snapshot_id``, and every
    existing frame, row for row — is returned untouched, which is what keeps the
    golden baskets valid.

    The collection goes through :func:`build_live_bundle` with the target's own
    ``decision_time``, so the new frame is normalised, gated and status-stamped by
    exactly the same code a real capture uses; nothing is hand-assembled here.
    Passing an explicit ``decision_time`` also puts that collector in replay mode,
    which is what stamps the rows ``available_time = decision_time`` instead of
    "now" — necessary, because rows stamped now would be dropped by the very PIT
    gate that makes the fixture honest.

    Refuses rather than degrades when: the node is not in :data:`ADDABLE_NODES`,
    the fetch failed, the frame came back empty, or the result does not cover the
    frozen universe it is going to be joined to.
    """
    if node not in ADDABLE_NODES:
        raise FreezeError(
            "--add-frame %r is not allowed. Only these nodes may be back-filled "
            "into a frozen bundle:\n%s\nEverything else MEASURES a market that "
            "has since moved, so stamping it with this bundle's decision_time "
            "would fabricate history — recapture the whole fixture instead."
            % (node, "\n".join("  %s: %s" % item for item in ADDABLE_NODES.items())))
    if node in (bundle.get("frames") or {}):
        raise FreezeError(
            "the bundle already carries a %r frame (%d rows). Re-adding it would "
            "mix two capture instants in one artifact; recapture the whole fixture "
            "if it needs refreshing."
            % (node, len(bundle["frames"][node].get("rows") or [])))

    from cyqnt_trd.standard_bot.data.live_bundle import build_live_bundle

    decision_time = int(bundle["decision_time"])
    side = build_live_bundle(
        requests=[(node, {"market_type": market_type, **(params or {})}, node)],
        decision_time=decision_time, market_type=market_type)
    entry = (side.get("frames") or {}).get(node)
    status = (side.get("source_status") or {}).get(node, "missing")
    if not isinstance(entry, dict) or not (entry.get("rows") or []):
        raise FreezeError(
            "collecting %r returned no rows (status=%s). Nothing was written; "
            "retry — a public Binance endpoint answering 'error' is usually "
            "transient." % (node, status))

    _verify_addition(bundle, node, entry["rows"])

    frozen = dict(bundle)
    frozen["frames"] = {**(bundle.get("frames") or {}), node: entry}
    frozen["source_status"] = {**(bundle.get("source_status") or {}), node: status}
    frozen["warnings"] = list(bundle.get("warnings") or []) + [
        warning for warning in (side.get("warnings") or [])
        if _warning_owner(warning) == node
    ]
    return frozen


def _verify_addition(bundle: Dict[str, Any], node: str,
                     rows: List[Dict[str, Any]]) -> None:
    """Check the new frame against the frozen universe, using its own consumer.

    Coverage is asserted by calling the block that will read it rather than by
    re-deriving a threshold here: the two would drift, and the one that matters is
    the one the runtime enforces. A fixture that passes this cannot fail its
    coverage guard at test time — which is the point, because a coverage failure
    discovered inside a test points at the test.

    The second assertion is what stops a *vacuous* fixture. A registry frame in
    which every row is a COIN would let ``filter_crypto_only`` pass its tests by
    dropping nothing, which is indistinguishable from the filter not running.
    """
    import pandas as pd

    if node == "universe_bars":
        _verify_bars_addition(bundle, rows)
        return
    if node != "contract_meta":                     # pragma: no cover - one node
        return
    from cyqnt_trd.blocks import universe as universe_blocks

    universe = pd.DataFrame(
        ((bundle.get("frames") or {}).get("universe") or {}).get("rows") or [])
    if universe.empty:
        raise FreezeError(
            "the bundle has no universe frame, so a contract_meta addition cannot "
            "be verified against anything")
    try:
        joined = universe_blocks.augment_with_contract_meta(
            universe, pd.DataFrame(rows))
    except ValueError as exc:
        raise FreezeError(
            "the captured contract_meta does not fit this universe, so the "
            "fixture would fail inside every test that replays it: %s" % exc)

    classes = joined["underlying_type"].dropna().unique().tolist()
    if "COIN" not in classes or len(classes) < 2:
        raise FreezeError(
            "the joined cross-section is all one asset class (%s), so a "
            "crypto-only filter could not be shown to do anything on it. Do not "
            "commit this fixture — a test that cannot fail is worse than none."
            % classes)


def _verify_bars_addition(bundle: Dict[str, Any],
                          rows: List[Dict[str, Any]]) -> None:
    """Refuse a bars addition that is not point-in-time, or is vacuous.

    :data:`ADDABLE_NODES` claims this frame may be back-filled because ``end_ms``
    makes it replayable. That claim is CHECKED here rather than trusted, and the
    check is the reason the whole ``--bars`` path is allowed to exist:

    * **Every bar closed at or before the target's decision_time.** ``endTime`` is
      inclusive of the candle CONTAINING it, so the newest row of each series is
      normally still moving — its high, low and close have not happened yet. The
      bundle's own PIT gate drops it; if any survivor is still in the future, the
      gate did not run and the fixture would let a spec read a bar from after the
      decision.
    * **Every series is long enough to be worth freezing**, per (instrument,
      timeframe), because that is the grain a warm-up failure has.
    * **The bars are not all one series.** A frame carrying one timeframe when the
      spec named three, or one instrument when the roster had five, is the shape
      that makes a resonance test pass by computing the same column three times.
    """
    import pandas as pd

    frame = pd.DataFrame(rows)
    if frame.empty:                                  # pragma: no cover - caller checks
        raise FreezeError("the bars addition carried no rows")
    decision_time = int(bundle["decision_time"])
    future = frame[pd.to_numeric(frame["close_time"], errors="coerce") > decision_time]
    if not future.empty:
        raise FreezeError(
            "%d of %d captured bars close AFTER the target's decision_time (%d), "
            "e.g. %s. endTime includes the candle containing it, so the newest bar "
            "of each series is normally unfinished and the bundle's PIT gate is "
            "what drops it — surviving rows mean the gate did not run, and the "
            "fixture would let a spec read a bar from after the decision it is "
            "replaying."
            % (len(future), len(frame), decision_time,
               future[["instrument_id", "timeframe", "close_time"]]
               .head(3).to_dict("records")))

    sizes = frame.groupby(["instrument_id", "timeframe"]).size()
    if len(sizes) < 2:
        raise FreezeError(
            "the capture covers only %d (instrument, timeframe) pair(s): %s. A "
            "single-series bar frame lets a multi-timeframe test pass by computing "
            "one column three times, which is indistinguishable from the resonance "
            "working." % (len(sizes), sizes.to_dict()))
    thin = sizes[sizes < 30]
    if not thin.empty:
        raise FreezeError(
            "these (instrument, timeframe) pairs carry fewer than 30 bars: %s. "
            "augment_with_indicator refuses a series shorter than its indicator's "
            "warm-up, so the fixture would fail inside every test that replays it "
            "— and the finger would point at the test." % thin.to_dict())


def _warning_owner(warning: str) -> str:
    """The frame key a warning is about, or ``""`` for a bundle-wide one.

    Warnings are emitted as ``"<key>: <note>"`` throughout ``live_bundle``, so
    the prefix is the attribution. Anything that does not name a frame we know
    about is kept — a warning with no owner is more likely to matter than a
    warning about ``hot_post``.
    """
    head = str(warning).split(":", 1)[0].strip()
    return head


def _symbols(rows: Iterable[Dict[str, Any]]) -> Set[str]:
    out = set()
    for row in rows:
        name = row.get("instrument_id") or row.get("symbol")
        if name:
            out.add(str(name).upper())
    return out


def trim(bundle: Dict[str, Any], keep: Sequence[str] = KEEP_FRAMES) -> Dict[str, Any]:
    """Drop every frame a cross-sectional decision cannot read, and verify it.

    Fails loudly rather than writing a fixture that "mostly" works: a frozen
    input whose funding frame covers one symbol looks fine in ``ls -la`` and then
    fails inside every test that replays it, which is the hardest kind of
    breakage to attribute.
    """
    if bundle.get("schema") != INPUT_SCHEMA_VERSION:
        raise FreezeError(
            "not a %s bundle (schema=%r). Capture it with "
            "`python -m cyqnt_trd.standard_bot.entrypoints.mvp_input_bundle "
            "--sections %s --out <file>`."
            % (INPUT_SCHEMA_VERSION, bundle.get("schema"),
               ",".join(CAPTURE_SECTIONS)))

    # This fixture is committed to a public repo. Market data is public; an
    # account snapshot is not, and --include-account would have put it here.
    if bundle.get("positions"):
        raise FreezeError(
            "bundle carries positions %r — recapture WITHOUT --include-account; "
            "this fixture is committed." % (bundle["positions"],))
    if bundle.get("equity") is not None:
        raise FreezeError(
            "bundle carries equity %r — recapture WITHOUT --include-account; "
            "this fixture is committed." % (bundle["equity"],))

    keep = tuple(keep)
    frames_in = dict(bundle.get("frames") or {})
    dropped = {key for key in frames_in if key not in keep}

    missing = [key for key in keep
               if not (frames_in.get(key) or {}).get("rows")]
    if missing:
        status = bundle.get("source_status") or {}
        raise FreezeError(
            "these frames are empty or absent, so the fixture would not exercise "
            "the spec that needs them: %s. Retry the capture — a Square/Binance "
            "node that answered 'error' is usually transient."
            % ", ".join(
                "%s=%s" % (key, "absent" if key not in frames_in
                           else status.get(key, "no status"))
                for key in missing))

    if "funding" in keep:
        universe_symbols = _symbols(frames_in["universe"]["rows"])
        covered = _symbols(frames_in["funding"]["rows"]) & universe_symbols
        # Mirror bundle_runner._assert_required_sources here so a bad capture is
        # rejected at freeze time. Discovering it at test time points the finger at
        # the test instead of at the capture.
        #
        # Conditional on ``keep``, because this check is about a frame this cut
        # asked for: the derivatives fixture keeps no funding frame (its mark
        # price travels with the open interest), and asserting the coverage of a
        # frame that was deliberately dropped would refuse a correct capture.
        if len(covered) < 2:
            raise FreezeError(
                "funding covers %d of %d universe symbols; the runtime rejects "
                "fewer than 2 as 'not a cross-section'. Recapture with the "
                "`selection_funding` section."
                % (len(covered), len(universe_symbols)))

    frozen = dict(bundle)
    frozen["frames"] = {key: frames_in[key] for key in keep}
    frozen["source_status"] = {
        key: value for key, value in (bundle.get("source_status") or {}).items()
        if key not in dropped
    }
    frozen["warnings"] = [
        warning for warning in (bundle.get("warnings") or [])
        if _warning_owner(warning) not in dropped
    ]
    return frozen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze a live cross-section into a replayable "
                    "cyqnt.input/v1 fixture")
    parser.add_argument(
        "--in", dest="source", default=None,
        help="re-cut this raw capture instead of fetching (no network)")
    parser.add_argument(
        "--add-frame", dest="add_frame", default=None, choices=sorted(ADDABLE_NODES),
        help="collect ONLY this node and land it in the --in bundle, leaving "
             "every existing frame and the decision_time untouched. Use when a "
             "new cross-sectional node lands and the golden baskets pinned "
             "against the fixture should not move.")
    parser.add_argument(
        "--derivatives", action="store_true",
        help="capture the FAN-OUT fixture instead: plan a roster from the sector "
             "tags %s, then collect universe + registry + open interest + its "
             "weekly change + long/short ratio in one pass. Writes %s and does "
             "not touch %s."
             % (list(DERIVATIVES_SUB_TYPES), DERIVATIVES_OUT, DEFAULT_OUT))
    parser.add_argument(
        "--liquidity", action="store_true",
        help="capture the LIQUIDITY fixture instead: universe + bookTicker + "
             "funding snapshot + funding schedule, four whole-market requests in "
             "one pass. Writes %s and touches neither of the other two. Refuses a "
             "capture on which annualising funding would not reorder anything."
             % LIQUIDITY_OUT)
    parser.add_argument(
        "--bars", default=None, metavar="SPEC",
        help="TWO-PASS (SPEC may be a comma-separated list; the roster and the "
             "timeframe set are the union): run SPEC's universe steps up to its "
             "first indicator step "
             "over the --in bundle (no network at all — those frames are already "
             "there), then fetch klines for the survivors with endTime = that "
             "bundle's decision_time and land them as `universe_bars`. Every "
             "existing frame and the decision_time are untouched, so golden "
             "baskets pinned against the bundle do not move. Needs --in.")
    parser.add_argument("--out", default=None,
                        help="fixture path (default: %s; %s with --derivatives; "
                             "%s with --liquidity)"
                             % (DEFAULT_OUT, DERIVATIVES_OUT, LIQUIDITY_OUT))
    parser.add_argument(
        "--keep", default=None,
        help="comma-separated frames to keep (default: %s; %s with "
             "--derivatives; %s with --liquidity). Extend this when a new "
             "cross-sectional node lands, then either recapture or use "
             "--add-frame if the node qualifies."
             % (",".join(KEEP_FRAMES), ",".join(DERIVATIVES_KEEP),
                ",".join(LIQUIDITY_KEEP)))
    parser.add_argument("--symbol", default="BTCUSDT",
                        help="representative instrument recorded in the bundle")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--market-type", default="futures",
                        choices=("spot", "futures"))
    parser.add_argument("--raw-out", default=None,
                        help="also write the untrimmed capture here")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.derivatives and args.liquidity:
        raise SystemExit(
            "--derivatives and --liquidity capture two different bundles with two "
            "different decision_times; asking for both would write one file with "
            "one of the two cuts. Run the script twice.")
    out_path = args.out or (
        DERIVATIVES_OUT if args.derivatives else
        LIQUIDITY_OUT if args.liquidity else DEFAULT_OUT)
    keep_arg = args.keep or ",".join(
        DERIVATIVES_KEEP if args.derivatives else
        LIQUIDITY_KEEP if args.liquidity else KEEP_FRAMES)
    roster: List[str] = []

    if args.bars and args.add_frame:
        raise SystemExit(
            "--bars IS an --add-frame of the `universe_bars` node, with the roster "
            "planned from the spec instead of typed by hand. Pass one or the other.")
    if args.source:
        with open(args.source, encoding="utf-8") as handle:
            bundle = json.load(handle)
        origin = args.source
    else:
        if args.add_frame or args.bars:
            raise SystemExit(
                "--add-frame / --bars need --in <existing fixture>: they add one "
                "frame to a bundle that is already frozen. Without --in there is "
                "nothing to add to, and nothing whose decision_time the bars would "
                "be cut at.")
        if args.derivatives:
            bundle, roster = capture_derivatives(
                symbol=args.symbol, interval=args.interval,
                market_type=args.market_type)
            origin = "live fan-out over %d %s instruments (%s)" % (
                len(roster), "/".join(DERIVATIVES_SUB_TYPES),
                ",".join(DERIVATIVES_SECTIONS))
        elif args.liquidity:
            bundle = capture_liquidity(symbol=args.symbol, interval=args.interval,
                                       market_type=args.market_type)
            origin = "live whole-market (%s)" % ",".join(LIQUIDITY_SECTIONS)
        else:
            bundle = capture_live(symbol=args.symbol, interval=args.interval,
                                  market_type=args.market_type)
            origin = "live (%s)" % ",".join(CAPTURE_SECTIONS)
    if args.raw_out:
        write_input_bundle(bundle, args.raw_out)

    keep = [name.strip() for name in keep_arg.split(",") if name.strip()]
    if args.bars:
        specs = [name.strip() for name in args.bars.split(",") if name.strip()]
        frozen, plan = capture_bars(bundle, specs,
                                    market_type=args.market_type)
        origin = ("%s + live klines for %d survivor(s) of %d prefix step(s) x %s "
                  "@ endTime=%d"
                  % (origin, len(plan.symbols), plan.survivors_of,
                     "/".join(plan.timeframes), plan.end_ms))
        print("  bars roster: %s" % ", ".join(plan.symbols))
    elif args.add_frame:
        # trim() is skipped on purpose: it is the cut that turns a raw capture
        # into a fixture, and re-cutting here would drop whatever the frozen
        # bundle already keeps that the CURRENT KEEP_FRAMES no longer lists.
        # "Leave everything else exactly as it is" is the whole contract.
        frozen = add_frame(bundle, args.add_frame, market_type=args.market_type)
        origin = "%s + live %s" % (origin, args.add_frame)
    else:
        frozen = trim(bundle, keep)
    if args.derivatives:
        _verify_derivatives(frozen)
    if args.liquidity:
        _verify_liquidity(frozen)
    path = write_input_bundle(frozen, out_path)

    print("[freeze] from %s" % origin)
    print("  decision_time=%s snapshot_id=%s"
          % (frozen["decision_time"], frozen["snapshot_id"]))
    for key, frame in frozen["frames"].items():
        print("  %-14s %-16s %5d rows" % (key, frame.get("shape"),
                                          len(frame.get("rows") or [])))
    print("  dropped: %s" % ("(none — --add-frame keeps everything)"
                             if (args.add_frame or args.bars) else
                             ", ".join(sorted(
                                 key for key in (bundle.get("frames") or {})
                                 if key not in keep)) or "(none)"))
    print("  wrote %s (%d bytes)" % (path, os.path.getsize(path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
