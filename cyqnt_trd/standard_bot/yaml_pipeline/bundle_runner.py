"""Canonical decision path: YAML + ``cyqnt.input/v1`` -> v2 signal batch.

The executor in this module deliberately does not fetch data and does not
execute orders.  :func:`collect_live_bundle_for_spec` is the one orchestration
wrapper around the data layer for callers that explicitly ask for a live
artifact; execution still receives only the finished input contract.  The
module joins the two contracts that already exist in the repo:

* the input bundle is the only data object;
* the YAML interpreter builds the existing Blocks plugin;
* every emitted item is normalised to the complete ``cyqnt.signal/v2`` shape;
* zero qualifying signals is represented by ``signals: []`` rather than an
  exception or an old-format envelope.

CLI, demo and future colleague-provided data adapters should all call this
function instead of growing another strategy execution path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, NamedTuple, Optional, Set

from cyqnt_trd.blocks import strategy as blocks_strategy

from ..adapter import batch_to_signals
from ..core import MarketBundle, StandardSignal
from ..data import load_input_bundle
from .interpreter import SpecError, build_make_signals, build_selection_fn
from .spec import assumption_warnings, load_spec, validate_spec

SIGNAL_BATCH_SCHEMA_VERSION = "cyqnt.signal-batch/v1"


class BundleRunError(ValueError):
    """The contracts are valid individually but cannot safely run together."""


class EmptySelectionPrefix(BundleRunError):
    """A valid screen eliminated every symbol before a paid fan-out source."""


_COLUMN_NODES = {
    "funding_rate": "funding", "funding_rate_bps": "funding",
    "mark_price": "funding",
    "open_interest": "open_interest", "open_interest_value": "open_interest",
    "oi_change_bps": "open_interest",
    "buy_vol": "taker_volume", "sell_vol": "taker_volume",
    "taker_buy_volume": "taker_volume", "taker_sell_volume": "taker_volume",
    "long_short_ratio": "long_short_ratio",
    "long_liq_count": "liquidations", "short_liq_count": "liquidations",
    "long_liq_qty": "liquidations", "short_liq_qty": "liquidations",
    "long_liq_notional_usd": "liquidations",
    "short_liq_notional_usd": "liquidations",
    "total_liq_notional_usd": "liquidations",
    "net_liq_notional_usd": "liquidations",
    "liq_imbalance_ratio": "liquidations",
}


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def required_bundle_nodes(spec: Mapping[str, Any]) -> Set[str]:
    """Infer only the sources the compiled strategy actually references."""
    selection = spec.get("selection")
    required = {"klines"} if not isinstance(selection, dict) else {"universe"}
    if isinstance(selection, Mapping):
        for step in selection.get("universe") or ():
            if not isinstance(step, Mapping):
                continue
            required.update(
                str(name) for name in (step.get("with") or ())
                if isinstance(name, str) and name
            )
    for token in _strings(spec.get("signals") or {}):
        node = _COLUMN_NODES.get(token)
        if node:
            required.add(node)
    return required


def live_sections_for_spec(spec: Mapping[str, Any]) -> list[str]:
    """The narrow live collection plan for a YAML strategy."""
    nodes = required_bundle_nodes(spec)
    sections = []
    selection = isinstance(spec.get("selection"), dict)
    if (not selection and nodes & {"funding", "open_interest", "taker_volume",
                                  "long_short_ratio", "top_trader_ratio"}):
        sections.append("derivatives")
    if "liquidations" in nodes:
        sections.append("liquidations")
    if selection:
        sections.append("universe")
        if "ticker_rank" in nodes:
            sections.append("news")
        if "funding" in nodes:
            sections.append("selection_funding")
        if "contract_meta" in nodes:
            sections.append("contract_meta")
        # Whole-market cross-sections whose bundle key IS their node name, so a
        # spec's ``with:`` name maps straight onto the section that collects it.
        # ``funding_info`` is listed here and NOT folded into
        # ``selection_funding``: it is the divisor that annualises the rate, and a
        # spec that ranks on ``fundingRatePct`` (per settlement) legitimately does
        # not need it — collecting it unasked would put a frame in the bundle that
        # nothing reads.
        if "book_ticker" in nodes:
            sections.append("selection_book_ticker")
        if "funding_info" in nodes:
            sections.append("selection_funding_info")
        # The fan-out sections. Their bundle key IS their node name — unlike
        # ``funding``, which the cross-sectional node has to displace — so a
        # spec's ``with:`` name maps straight onto the section that collects it.
        for node, section in _fan_out_sections_by_node().items():
            if node in nodes:
                sections.append(section)
        # Bars last, and that is the ordering claim of the whole two-pass capture:
        # its roster is what survives EVERY step above, including the derivative
        # filters, so it is planned after those frames have landed.
        from ..data.live_snapshot import BARS_SECTION

        bars_section, bars_node, _bars_key = BARS_SECTION
        if bars_node in nodes:
            sections.append(bars_section)
    return sections


def _fan_out_sections_by_node() -> Dict[str, str]:
    """Fan-out node -> the section that collects it.

    Read out of the data layer's own table instead of restated here, so adding a
    fourth fan-out field cannot leave a spec validating green (the block resolves,
    the ``with:`` name is accepted) and then running against a frame nobody
    collected. Imported lazily to keep this module's import graph unchanged.
    """
    from ..data.live_snapshot import FAN_OUT_SECTIONS

    return {node: section for section, (node, _key, _extra)
            in FAN_OUT_SECTIONS.items()}


def _validated_spec(spec_or_path: Any) -> Dict[str, Any]:
    spec = (load_spec(str(spec_or_path))
            if isinstance(spec_or_path, (str, bytes, Path))
            else dict(spec_or_path))
    errors, _warnings = validate_spec(spec)
    if errors:
        raise SpecError("invalid strategy spec:\n  - " + "\n  - ".join(errors))
    return spec


def _build_plugin(spec: Mapping[str, Any]):
    strategy_id = str((spec.get("strategy") or {})["id"])
    data = spec.get("data") or {}
    if isinstance(spec.get("selection"), dict):
        # Both construction sites pass assumptions. They are the same spec run
        # two ways, and a declared reading that survives one path and not the
        # other is worse than none: the reader cannot tell which they got.
        # test_declared_assumptions.py asserts parity so a third site cannot
        # quietly drop it.
        return blocks_strategy.build_selection_plugin(
            strategy_id,
            build_selection_fn(dict(spec)),
            market_type=str(data.get("market_type") or "futures"),
            assumptions=assumption_warnings(dict(spec)),
        )
    htf_specs = [
        (item["interval"], int(item["sma_period"]))
        for item in (data.get("htf") or ())
        if isinstance(item, dict) and "sma_period" in item
    ] or None
    exit_cfg = (spec.get("risk") or {}).get("exit")
    return blocks_strategy.build_plugin(
        strategy_id,
        build_make_signals(dict(spec)),
        htf_specs=htf_specs,
        exit_cfg=exit_cfg if isinstance(exit_cfg, dict) else None,
        size=float((spec.get("sizing") or {}).get("size", 1.0)),
    )


def _latest_bar_time(snapshot: Any, *, instrument_id: Optional[str] = None,
                     timeframe: Optional[str] = None) -> Optional[int]:
    """Return the latest bar of one declared series, or all bars for diagnostics.

    A serialized BarFrame can carry several instrument/timeframe series. A
    trade decision must be compared with the newest bar of *its own* declared
    series; taking the newest row of a foreign series can silently suppress a
    valid signal as an apparently ordinary zero-signal replay.
    """
    market = getattr(snapshot, "market", None)
    series = getattr(market, "bars", {}) or {}
    if instrument_id is not None or timeframe is not None:
        if not instrument_id or not timeframe:
            raise ValueError("latest bar lookup needs both instrument_id and timeframe")
        series = {MarketBundle.key(instrument_id, timeframe):
                  series.get(MarketBundle.key(instrument_id, timeframe), [])}
    values = [int(bar.timestamp) for bars in series.values() for bar in bars]
    return max(values) if values else None


def _source_error(status: Any) -> bool:
    return str(status).split(":", 1)[0].strip() == "error"


#: Selection sources for which "read it, and it was empty" is already a failure.
#:
#: The general rule for a selection run is the opposite — an empty frame is let
#: through, because ``ticker_rank`` legitimately comes back empty when Square has
#: nothing to say and ``universe.augment_with_news`` handles that by NaN-filling
#: the buzz columns. These two do not: both are cross-sections whose block raises
#: on an empty source (funding because one symbol's history cannot stand in for a
#: cross-section, contract_meta because every listed contract has a contract type
#: by construction). Catching it here names the source that could not be read
#: instead of surfacing it as a ValueError from inside a universe step.
#: The three fan-out nodes join this list for the same reason and one more of
#: their own: an empty frame there means the fan-out collected NOTHING, and since
#: there is no all-market endpoint to fall back on, that is always a failure and
#: never "no instrument has open interest".
#: ``universe_bars`` is here for the fan-out reason and one more: it is the only
#: frame in this list whose emptiness has a SECOND innocent-looking cause. Bars
#: are gated point-in-time like everything else, so a capture whose ``end_ms`` was
#: never set — asking for bars "as of now" against a past ``decision_time`` — has
#: every row dropped by the gate and lands an empty frame with ``status: ok`` from
#: the fetch. Read as "no instrument has price history" that would empty the
#: basket; named here it says which source could not be read.
_SELECTION_SOURCES_REQUIRING_ROWS = frozenset({
    "funding", "contract_meta",
    "open_interest_snapshot", "oi_change_snapshot", "long_short_ratio_snapshot",
    "universe_bars",
})


def _assert_required_sources(bundle: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    statuses = dict(bundle.get("source_status") or {})
    frames = dict(bundle.get("frames") or {})
    failed = []
    selection = isinstance(spec.get("selection"), dict)
    for node in sorted(required_bundle_nodes(spec)):
        frame = frames.get(node)
        if (
            selection
            and node in _SELECTION_SOURCES_REQUIRING_ROWS
            and _verified_empty_prefix_skip(bundle, spec, node)
        ):
            continue
        empty_required_frame = (
            isinstance(frame, dict)
            and not (frame.get("rows") or [])
            and (not selection or node in _SELECTION_SOURCES_REQUIRING_ROWS)
        )
        unavailable = (
            _source_error(statuses.get(node, "error"))
            or not isinstance(frame, dict)
            or empty_required_frame
        )
        if unavailable:
            failed.append("%s=%s" % (node, statuses.get(node, "missing")))
            continue
        if selection and node == "funding":
            universe_rows = ((frames.get("universe") or {}).get("rows") or [])
            funding_rows = frame.get("rows") or []

            def _symbols(rows):
                return {
                    str(row.get("instrument_id") or row.get("symbol") or "").upper()
                    for row in rows if row.get("instrument_id") or row.get("symbol")
                }

            universe_symbols = _symbols(universe_rows)
            covered = _symbols(funding_rows) & universe_symbols
            if len(universe_symbols) > 1 and len(covered) < 2:
                failed.append(
                    "funding=invalid cross-sectional coverage %d/%d"
                    % (len(covered), len(universe_symbols))
                )
    if failed:
        raise BundleRunError(
            "required input source unavailable; strategy was not run: "
            + ", ".join(failed)
        )


def _assert_trade_primary_series(snapshot: Any, *, symbol: str, interval: str) -> None:
    """Refuse a trade replay whose bars do not match the declared primary series.

    A ``klines`` frame being non-empty only proves *some* bars were captured.
    It does not prove that the Blocks plugin will receive the ``data.symbol`` and
    ``data.primary.interval`` series that the YAML declares. Returning an empty
    batch in that case is indistinguishable from a genuine no-signal decision,
    so keep the contract fail-closed at the canonical YAML + bundle boundary.
    """
    market = getattr(snapshot, "market", None)
    bars = getattr(market, "bars", None)
    key = MarketBundle.key(symbol, interval)
    if not isinstance(bars, Mapping) or not bars.get(key):
        raise BundleRunError(
            "declared primary kline series is unavailable; strategy was not run"
        )


class BarsPlan(NamedTuple):
    """What Pass 2 of a bars capture has to be asked for.

    ``symbols`` is what survived Pass 1, ``timeframes`` is the union the spec's
    indicator steps name, and ``end_ms`` is the instant the histories are cut at.
    ``survivors_of`` records how many steps of the pipeline were actually run, so a
    caller printing the plan can show the funnel rather than only its last number.
    """

    symbols: list
    timeframes: list
    end_ms: Optional[int]
    survivors_of: int


def plan_bars_capture(spec: Mapping[str, Any], bundle: Mapping[str, Any]) -> BarsPlan:
    """PASS 1 of a bars capture: run the cheap prefix, return the roster.

    A capture cannot know in advance which instruments a screen will keep, and a
    bars fan-out is charged per (instrument, timeframe) — so the roster has to be
    derived, and it can only be derived by running the spec's own narrowing steps
    over the frames that are already in hand.

    That is what this does, and the three properties that make it safe are worth
    naming because each has a wrong-looking alternative:

    * it runs **the spec's own steps**, through
      :func:`interpreter.run_universe_steps`, not a re-implementation of "apply
      the filters". A second implementation would produce a slightly different
      funnel from the one the run walks, and the difference surfaces as a coverage
      refusal inside ``augment_with_indicator`` — pointing at the join, the one
      place not at fault.
    * it runs only the steps **before** the first indicator step. Those are
      necessarily cross-sectional (an indicator step is the first thing that needs
      bars), so Pass 1 costs no requests at all: every frame it reads is already in
      the bundle.
    * ``end_ms`` is the bundle's **own** ``decision_time``. This is PIT guard #1:
      it makes the bars the ones a capture at that instant would have seen, rather
      than today's prices beside a past universe.

    Raises rather than returning an empty plan when the prefix keeps nothing:
    "collect no bars" and "this screen has no candidates" are different facts, and
    an empty roster would land an empty frame that the joining block reads as a
    failed capture.
    """
    import pandas as pd

    from .interpreter import (
        bar_timeframes_for_spec,
        run_universe_steps,
        universe_steps_before_bars,
    )

    timeframes = bar_timeframes_for_spec(dict(spec))
    if not timeframes:
        raise BundleRunError(
            "this spec has no %s step, so there are no bars to plan. Nothing was "
            "collected." % _bars_block())

    frames = dict(bundle.get("frames") or {})

    def table(key: str):
        return pd.DataFrame((frames.get(key) or {}).get("rows") or [])

    universe = table("universe")
    if universe.empty:
        raise BundleRunError(
            "the bundle carries no universe frame, so Pass 1 of a bars capture has "
            "nothing to narrow. Collect the `universe` section first — the bars "
            "roster is derived FROM it, which is why the two cannot be one pass.")
    extras = {key: table(key) for key in frames}
    extras["universe"] = universe
    extras.setdefault("ticker_rank", None)

    prefix = universe_steps_before_bars(dict(spec))
    survivors = run_universe_steps(prefix, universe, extras)
    column = next((name for name in ("instrument_id", "symbol")
                   if name in getattr(survivors, "columns", ())), None)
    if column is None or not len(survivors):
        raise EmptySelectionPrefix(
            "the %d step(s) before the first %s step left no instrument, so there "
            "is nothing to fetch bars for. That is not 'collect nothing': an empty "
            "bars frame is read by the joining block as a failed capture. Either "
            "the thresholds are too strict for this market, or a source the prefix "
            "needs is absent from this bundle."
            % (len(prefix), _bars_block()))
    roster = sorted({str(value).upper() for value in survivors[column]})
    return BarsPlan(symbols=roster, timeframes=timeframes,
                    end_ms=int(bundle["decision_time"]),
                    survivors_of=len(prefix))


def _bars_block() -> str:
    from .interpreter import BARS_BLOCK

    return BARS_BLOCK


def _frame_table(bundle: Mapping[str, Any], key: str):
    """Return one bundle frame as a DataFrame for a collection planner."""
    import pandas as pd

    return pd.DataFrame(
        (((bundle.get("frames") or {}).get(key) or {}).get("rows") or [])
    )


def _plan_fan_out_roster(
    spec: Mapping[str, Any], bundle: Mapping[str, Any], source: str,
) -> list[str]:
    """Run the spec prefix before *source* is first consumed.

    The source is a per-instrument fan-out, so its request roster is not a
    property of the endpoint.  It is the exact set of instruments that survives
    the preceding YAML steps.  Re-running that prefix through
    ``run_universe_steps`` keeps collection and execution on one implementation.
    """
    from .interpreter import run_universe_steps

    steps = list((spec.get("selection") or {}).get("universe") or ())
    position = next((
        index for index, step in enumerate(steps)
        if isinstance(step, Mapping) and source in (step.get("with") or ())
    ), None)
    if position is None:
        raise BundleRunError(
            "cannot plan fan-out source %r: no selection.universe step consumes "
            "it through `with:`" % source
        )

    universe = _frame_table(bundle, "universe")
    if universe.empty:
        raise BundleRunError(
            "cannot plan fan-out source %r: the cheap collection pass returned "
            "no universe rows" % source
        )
    extras = {
        key: _frame_table(bundle, key)
        for key in (bundle.get("frames") or {})
    }
    extras["universe"] = universe
    extras.setdefault("ticker_rank", None)
    survivors = run_universe_steps(steps[:position], universe, extras)
    column = next((
        name for name in ("instrument_id", "symbol")
        if name in getattr(survivors, "columns", ())
    ), None)
    if column is None or not len(survivors):
        raise EmptySelectionPrefix(
            "the %d step(s) before fan-out source %r left no instrument; an "
            "empty roster is a screen result, not a successful data capture"
            % (position, source)
        )
    return sorted({str(value).strip().upper() for value in survivors[column]
                   if str(value).strip()})


_EMPTY_PREFIX_STATUS = "skipped: empty selection prefix"


def _verified_empty_prefix_skip(
    bundle: Mapping[str, Any], spec: Mapping[str, Any], source: str,
) -> bool:
    """Prove that a skipped source could not have been reached by execution."""
    frame = ((bundle.get("frames") or {}).get(source) or {})
    status = str((bundle.get("source_status") or {}).get(source) or "")
    if (
        status != _EMPTY_PREFIX_STATUS
        or not isinstance(frame, Mapping)
        or not frame.get("collection_skipped")
        or frame.get("rows")
    ):
        return False
    try:
        _plan_fan_out_roster(spec, bundle, source)
    except EmptySelectionPrefix:
        return True
    except (BundleRunError, SpecError, ValueError):
        return False
    return False


def _mark_empty_prefix_skips(
    bundle: Dict[str, Any], sources: Iterable[str], reason: str,
) -> None:
    """Land explicit empty frames for sources a valid empty screen never reads."""
    from ..data.catalog import get_node

    frames = bundle.setdefault("frames", {})
    statuses = bundle.setdefault("source_status", {})
    warnings = bundle.setdefault("warnings", [])
    for source in sorted(set(str(name) for name in sources)):
        node = get_node(source)
        shape = node.input_schema.name if node.input_schema else "RawFrame@1.0"
        frames[source] = {
            "shape": shape,
            "status": "empty",
            "rows": [],
            "reason": "collection skipped because the spec prefix had no survivors",
            "collection_skipped": True,
        }
        statuses[source] = _EMPTY_PREFIX_STATUS
        warning = "%s: %s; %s" % (source, _EMPTY_PREFIX_STATUS, reason)
        if warning not in warnings:
            warnings.append(warning)


def _merge_collection_pass(
    base: Optional[Dict[str, Any]], addition: Mapping[str, Any],
) -> Dict[str, Any]:
    """Merge one independently PIT-gated live pass into one replay artifact."""
    if addition.get("schema") != "cyqnt.input/v1":
        raise BundleRunError(
            "live collector returned %r instead of cyqnt.input/v1"
            % addition.get("schema")
        )
    if base is None:
        merged = copy.deepcopy(dict(addition))
        merged["collection_passes"] = []
    else:
        merged = base
        for field in ("market_type", "primary_timeframe"):
            old = merged.get(field)
            new = addition.get(field)
            if old is not None and new is not None and old != new:
                raise BundleRunError(
                    "live collection passes disagree on %s (%r != %r)"
                    % (field, old, new)
                )
        for field in ("run_id", "trace_id"):
            old = str(merged.get(field) or "")
            new = str(addition.get(field) or "")
            if old and new and old != new:
                raise BundleRunError(
                    "live collection passes disagree on %s (%r != %r)"
                    % (field, old, new)
                )
            if not old and new:
                merged[field] = new

        frames = merged.setdefault("frames", {})
        statuses = merged.setdefault("source_status", {})
        for key, frame in (addition.get("frames") or {}).items():
            if key in frames and frames[key] != frame:
                raise BundleRunError(
                    "live collection attempted to replace frame %r with a "
                    "different capture; one input bundle may contain only one "
                    "version of each source" % key
                )
            frames.setdefault(key, copy.deepcopy(frame))
        for key, status in (addition.get("source_status") or {}).items():
            if key in statuses and statuses[key] != status:
                raise BundleRunError(
                    "live collection passes disagree on source_status[%r] "
                    "(%r != %r)" % (key, statuses[key], status)
                )
            statuses.setdefault(key, str(status))

        warnings = list(merged.get("warnings") or ())
        for warning in addition.get("warnings") or ():
            if warning not in warnings:
                warnings.append(str(warning))
        merged["warnings"] = warnings
        merged["decision_time"] = max(
            int(merged["decision_time"]), int(addition["decision_time"])
        )
        instruments = list(merged.get("instruments") or ())
        for instrument in addition.get("instruments") or ():
            if instrument not in instruments:
                instruments.append(instrument)
        merged["instruments"] = instruments

    merged.setdefault("collection_passes", []).append({
        "snapshot_id": str(addition.get("snapshot_id") or ""),
        "decision_time": int(addition["decision_time"]),
        "decision_time_basis": str(
            addition.get("decision_time_basis") or "unknown"
        ),
        "sources": sorted(str(key)
                          for key in (addition.get("source_status") or {})),
    })
    bases = {
        str(item.get("decision_time_basis") or "unknown")
        for item in merged["collection_passes"]
    }
    merged["decision_time_basis"] = (
        "replay" if bases == {"replay"} else "collection_complete"
    )
    return merged


def _finish_collected_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the final PIT gate and bind ``snapshot_id`` to the artifact."""
    from ..data.input_bundle import BUNDLE_NAMESPACE
    from ..data.live_bundle import _regate

    cutoff = int(bundle["decision_time"])
    frames = bundle.setdefault("frames", {})
    _regate(frames, cutoff)
    statuses = bundle.setdefault("source_status", {})
    for key, frame in frames.items():
        current = str(statuses.get(key, ""))
        if current in ("", "ok", "empty"):
            statuses[key] = str(frame.get("status") or statuses.get(key) or "empty")

    # The single-pass collector identifies a capture by its clock.  A merged
    # capture also needs its source content in the identity; otherwise two
    # different fan-out rosters finishing in the same millisecond collide.
    identity = {
        key: bundle.get(key)
        for key in (
            "schema", "decision_time", "decision_time_basis", "market_type",
            "instruments", "primary_timeframe", "frames", "source_status",
            "warnings", "positions", "equity", "config", "bot_id", "run_id",
            "trace_id", "collection_passes",
        )
    }
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    bundle["snapshot_id"] = str(uuid.uuid5(
        BUNDLE_NAMESPACE, "live-spec-multipass|%s" % digest
    ))
    return bundle


def collect_live_bundle_for_spec(
    spec_or_path: Any,
    *,
    symbol: Optional[str] = None,
    interval: Optional[str] = None,
    market_type: Optional[str] = None,
    limit: int = 500,
    positions: Optional[Dict[str, float]] = None,
    equity: Optional[float] = None,
    run_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    include_account: bool = False,
    bar_limit: int = 200,
    write_bundle: Optional[Any] = None,
) -> Dict[str, Any]:
    """Collect the live sources a spec needs into one ``cyqnt.input/v1``.

    Selection sources are collected in dependency order:

    1. whole-market/cheap cross-sections;
    2. each per-symbol derivative source over the survivors of its own YAML
       prefix;
    3. candidate bars, planned by :func:`plan_bars_capture` after the preceding
       derivative filters have run.

    Each network pass is normalised and PIT-gated by ``build_live_snapshot``.
    The returned bundle then receives one final gate, unioned status/warnings,
    the final collection-complete decision time, and a content-bound snapshot id.
    Optional ``run_id`` / ``trace_id`` values are attached to every pass and
    retained on the merged replay boundary.
    Serialising this dict is therefore sufficient to replay the exact decision;
    execution never needs to fetch again.  This live collector deliberately has
    no historical ``decision_time`` override: selection sources such as the
    universe and funding snapshot are forward-only, so fetching them now and
    stamping an older clock would fabricate a replay.  Historical execution must
    load a previously saved ``cyqnt.input/v1`` instead.
    """
    from ..data.input_bundle import write_input_bundle
    from ..data.live_snapshot import (
        BARS_SECTION,
        FAN_OUT_SECTIONS,
        build_live_snapshot,
        requests_for_sections,
    )

    spec = _validated_spec(spec_or_path)
    data = spec.get("data") or {}
    resolved_symbol = str(symbol or data.get("symbol") or "BTCUSDT").upper()
    resolved_interval = str(
        interval or (data.get("primary") or {}).get("interval") or "1h"
    )
    resolved_market = str(market_type or data.get("market_type") or "futures")
    sections = live_sections_for_spec(spec)

    def collect(requests, pass_sections):
        _snapshot, collected = build_live_snapshot(
            requests=requests,
            # Kept alongside the explicit request plan for observability and for
            # simple test adapters. ``requests`` remains authoritative.
            sections=pass_sections,
            symbol=resolved_symbol,
            interval=resolved_interval,
            limit=int(limit),
            market_type=resolved_market,
            positions=positions,
            equity=equity,
            include_account=include_account,
            bar_limit=int(bar_limit),
        )
        if run_id is not None:
            collected["run_id"] = str(run_id)
        if trace_id is not None:
            collected["trace_id"] = str(trace_id)
        return collected

    if not isinstance(spec.get("selection"), dict):
        bundle = collect(None, sections)
        if write_bundle is not None:
            write_input_bundle(bundle, str(write_bundle))
        return bundle

    required = required_bundle_nodes(spec)
    bars_section, bars_node, _bars_key = BARS_SECTION
    fan_out_by_node = {
        node: section
        for section, (node, _key, _extra) in FAN_OUT_SECTIONS.items()
    }
    dynamic_sections = set(FAN_OUT_SECTIONS) | {bars_section}
    cheap_sections = [section for section in sections
                      if section not in dynamic_sections]
    cheap_requests = [
        request for request in requests_for_sections(
            cheap_sections,
            symbol=resolved_symbol,
            interval=resolved_interval,
            limit=int(limit),
            market_type=resolved_market,
        )
        if request[2] in required
    ]
    if not cheap_requests:
        raise BundleRunError(
            "selection live collection has no cheap first-pass requests; every "
            "selection needs at least the universe cross-section"
        )
    bundle: Optional[Dict[str, Any]] = _merge_collection_pass(
        None, collect(cheap_requests, cheap_sections)
    )

    empty_prefix = False
    for step in (spec.get("selection") or {}).get("universe") or ():
        if not isinstance(step, Mapping):
            continue
        for source in step.get("with") or ():
            source = str(source)
            if source in (bundle.get("frames") or {}):
                continue
            if source == bars_node:
                try:
                    plan = plan_bars_capture(spec, bundle)
                except EmptySelectionPrefix as exc:
                    remaining = (
                        ((set(fan_out_by_node) | {bars_node}) & required)
                        - set(bundle.get("frames") or {})
                    )
                    _mark_empty_prefix_skips(bundle, remaining, str(exc))
                    empty_prefix = True
                    break
                requests = [
                    request for request in requests_for_sections(
                        [bars_section],
                        symbol=resolved_symbol,
                        interval=resolved_interval,
                        limit=int(limit),
                        market_type=resolved_market,
                        bar_symbols=plan.symbols,
                        bar_timeframes=plan.timeframes,
                        bar_limit=int(bar_limit),
                        bars_end_ms=plan.end_ms,
                    )
                    if request[2] == bars_node
                ]
                bundle = _merge_collection_pass(
                    bundle, collect(requests, [bars_section])
                )
                continue
            section = fan_out_by_node.get(source)
            if section is None:
                # A cheap source that failed is still present as an error frame.
                # Truly absent names are left for run_bundle's required-source
                # error, which reports the public bundle key and status.
                continue
            try:
                roster = _plan_fan_out_roster(spec, bundle, source)
            except EmptySelectionPrefix as exc:
                remaining = (
                    ((set(fan_out_by_node) | {bars_node}) & required)
                    - set(bundle.get("frames") or {})
                )
                _mark_empty_prefix_skips(bundle, remaining, str(exc))
                empty_prefix = True
                break
            requests = [
                request for request in requests_for_sections(
                    [section],
                    symbol=resolved_symbol,
                    interval=resolved_interval,
                    limit=int(limit),
                    market_type=resolved_market,
                    fan_out_symbols=roster,
                )
                if request[2] == source
            ]
            bundle = _merge_collection_pass(
                bundle, collect(requests, [section])
            )
        if empty_prefix:
            break

    missing_dynamic = sorted(
        ((set(fan_out_by_node) | {bars_node}) & required)
        - set((bundle.get("frames") or {}))
    )
    if missing_dynamic:
        raise BundleRunError(
            "spec requires live fan-out source(s) %s, but no universe step "
            "consumed them through `with:`" % missing_dynamic
        )
    finished = _finish_collected_bundle(bundle)
    if write_bundle is not None:
        write_input_bundle(finished, str(write_bundle))
    return finished


def run_bundle(spec_or_path: Any, bundle_or_path: Any) -> Dict[str, Any]:
    """Run one YAML decision against one input bundle and return one contract."""
    spec = _validated_spec(spec_or_path)
    if isinstance(bundle_or_path, (str, bytes, Path)):
        bundle = json.loads(Path(bundle_or_path).read_text(encoding="utf-8"))
    else:
        bundle = dict(bundle_or_path)
    _assert_required_sources(bundle, spec)
    snapshot = load_input_bundle(bundle)
    data = spec.get("data") or {}
    symbol = str(data.get("symbol") or "BTCUSDT").upper()
    interval = str((data.get("primary") or {}).get("interval")
                   or bundle.get("primary_timeframe") or "1h")
    market_type = str(data.get("market_type") or "futures")
    if not isinstance(spec.get("selection"), dict):
        _assert_trade_primary_series(snapshot, symbol=symbol, interval=interval)
    plugin = _build_plugin(spec)
    batch = plugin.run(snapshot, SimpleNamespace(
        instrument_id=symbol, symbol=symbol, timeframe=interval,
        interval=interval, market_type=market_type,
    ))

    envelopes = list(getattr(batch, "signals", ()) or ())
    if not isinstance(spec.get("selection"), dict):
        latest = _latest_bar_time(snapshot, instrument_id=symbol, timeframe=interval)
        # plugin.run evaluates the full warm-up window.  A decision output must
        # not republish old historical entries as if they fired now.
        envelopes = [env for env in envelopes
                     if int((env.payload or {}).get("bar_timestamp") or -1) == latest]

    decision_time = int(bundle["decision_time"])
    product = "spot" if market_type == "spot" else "usd_m_perpetual"
    signals = batch_to_signals(
        envelopes, decision_time=decision_time, product=product)
    statuses = {key: str(value)
                for key, value in (bundle.get("source_status") or {}).items()}
    warnings = tuple(bundle.get("warnings") or ())
    run_id = str(bundle.get("run_id") or "")
    trace_id = str(bundle.get("trace_id") or "")
    complete = []
    for signal in signals:
        provenance = replace(
            signal.provenance,
            run_id=signal.provenance.run_id or run_id,
            trace_id=signal.provenance.trace_id or trace_id,
        )
        candidates = []
        for candidate in signal.candidates:
            trade = candidate.trade
            if trade is not None:
                trade_provenance = replace(
                    trade.provenance,
                    run_id=trade.provenance.run_id or run_id,
                    trace_id=trade.provenance.trace_id or trace_id,
                )
                trade = replace(
                    trade,
                    provenance=trade_provenance,
                    source_status=trade.source_status or statuses,
                    warnings=trade.warnings or warnings,
                )
            candidates.append(replace(candidate, trade=trade))
        complete.append(replace(
            signal,
            provenance=provenance,
            source_status=signal.source_status or statuses,
            warnings=signal.warnings or warnings,
            candidates=tuple(candidates),
        ))

    return {
        "schema": SIGNAL_BATCH_SCHEMA_VERSION,
        "strategy_id": str((spec.get("strategy") or {})["id"]),
        "decision_time": decision_time,
        "snapshot_id": str(bundle.get("snapshot_id") or ""),
        "run_id": run_id,
        "trace_id": trace_id,
        "source_status": statuses,
        "warnings": list(warnings),
        "signal_count": len(complete),
        "signals": [signal.to_dict() for signal in complete],
    }


def write_signal_batch(batch: Mapping[str, Any], path: Any) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(batch), ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return str(target)


__all__ = [
    "SIGNAL_BATCH_SCHEMA_VERSION", "BundleRunError", "required_bundle_nodes",
    "live_sections_for_spec", "collect_live_bundle_for_spec", "run_bundle",
    "write_signal_batch",
    "BarsPlan", "plan_bars_capture",
]
