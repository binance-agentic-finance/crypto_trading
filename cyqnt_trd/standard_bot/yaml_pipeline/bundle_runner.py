"""Canonical decision path: YAML + ``cyqnt.input/v1`` -> v2 signal batch.

This module deliberately does not fetch data and does not execute orders.  It
joins the two contracts that already exist in the repo:

* the input bundle is the only data object;
* the YAML interpreter builds the existing Blocks plugin;
* every emitted item is normalised to the complete ``cyqnt.signal/v2`` shape;
* zero qualifying signals is represented by ``signals: []`` rather than an
  exception or an old-format envelope.

CLI, demo and future colleague-provided data adapters should all call this
function instead of growing another strategy execution path.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, NamedTuple, Optional, Set

from cyqnt_trd.blocks import strategy as blocks_strategy

from ..adapter import batch_to_signals
from ..core import StandardSignal
from ..data import load_input_bundle
from .interpreter import SpecError, build_make_signals, build_selection_fn
from .spec import assumption_warnings, load_spec, validate_spec

SIGNAL_BATCH_SCHEMA_VERSION = "cyqnt.signal-batch/v1"


class BundleRunError(ValueError):
    """The contracts are valid individually but cannot safely run together."""


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


def _latest_bar_time(snapshot: Any) -> Optional[int]:
    market = getattr(snapshot, "market", None)
    values = [int(bar.timestamp) for bars in (getattr(market, "bars", {}) or {}).values()
              for bar in bars]
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
        raise BundleRunError(
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
    plugin = _build_plugin(spec)
    batch = plugin.run(snapshot, SimpleNamespace(
        instrument_id=symbol, symbol=symbol, timeframe=interval,
        interval=interval, market_type=market_type,
    ))

    envelopes = list(getattr(batch, "signals", ()) or ())
    if not isinstance(spec.get("selection"), dict):
        latest = _latest_bar_time(snapshot)
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
        complete.append(replace(
            signal,
            provenance=provenance,
            source_status=signal.source_status or statuses,
            warnings=signal.warnings or warnings,
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
    "live_sections_for_spec", "run_bundle", "write_signal_batch",
    "BarsPlan", "plan_bars_capture",
]
