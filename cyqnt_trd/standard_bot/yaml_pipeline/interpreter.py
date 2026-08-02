"""YAML → ``make_signals(df)`` interpreter.

Turns a declarative strategy spec into a callable ``make_signals(df)`` that
composes ``cyqnt_trd.blocks.*`` primitives into ``(long_signal, short_signal)``
boolean Series — the exact contract that ``cyqnt_trd.blocks.strategy.register``
expects, and that runs identically across backtest / paper / live
(see ``standard_bot`` docs).

Design goals
------------
* **Every block composable**: any callable in a whitelisted ``cyqnt_trd.blocks``
  submodule can be referenced by ``"<module>.<fn>"``. First-argument shape
  (``df`` vs a price Series) is auto-detected from the function signature, so
  ``indicators.atr`` (df-first) and ``indicators.ema`` (series-first) both work
  without the author knowing the internal convention.
* **Arbitrary nesting**: entry rules are combinator trees
  (``all_of`` / ``any_of`` / ``not`` / ``exclude_when``) over condition leaves,
  nested to any depth.
* **Tuple outputs**: indicators returning tuples (``macd``, ``adx``,
  ``donchian``, ``bollinger``) expose a component via ``output: <index>``.

This module is import-light on purpose: it only imports pandas + the blocks
package lazily so validation can run without the heavy standard_bot stack.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

# The boundary is behavioural, not alphabetical. It used to be an allowlist of
# eight submodules, which had two problems: it hid 16 of the 24 block modules
# from YAML — including every funding / open-interest / liquidation / news /
# regime / microstructure block, i.e. everything that makes a strategy
# multi-source — while simultaneously letting `verdicts.dataclass`,
# `scoring.field` and 26 other *imported* symbols through, because it checked
# the module and never the function.
#
# So: any function DEFINED in a ``cyqnt_trd.blocks`` submodule is addressable,
# and the two things that actually matter are named and refused.

#: Submodules a spec may never dispatch into, and why.
DENIED_NAMESPACES = {
    "strategy": (
        "registration API, not a computation: register() mutates the "
        "process-wide plugin registry, so merely validating a spec would "
        "register a strategy as a side effect"
    ),
}

#: Individual blocks refused because calling them performs network I/O. A spec
#: is evaluated once per validate and once per backtest run, so a fetching block
#: would turn a dry-run into a burst of REST calls and make a backtest depend on
#: live data. Declare what you want under ``data:`` instead — that fetches once.
#:
#: Keyed on the bare function name, not on ``data.``: the fetchers are not all in
#: that module. ``universe.fetch_perpetual_universe`` is a live REST call and was
#: reachable, so ``validate`` on a frontend-supplied spec fired outbound requests
#: — verbatim the harm this list exists to prevent.
DENIED_FUNCTION_NAMES = {
    "fetch_klines": "performs a REST call; declare the source under `data:` instead",
    "fetch_24h_tickers": "performs a REST call; declare the source under `data:` instead",
    "fetch_perpetual_universe":
        "performs a REST call; the universe arrives via DataSnapshot.universe",
    "load_pit_index": "reads a caller-supplied path off disk",
}

#: Blocks that fetch ONLY when their optional source argument is absent. They are
#: allowed, but the spec has to supply the source with ``with: [...]`` — see
#: :func:`_refuse_implicit_fetch`.
#: The three derivative joins are here for a second reason on top of the network
#: one: their live fallback FANS OUT over the frame they are handed, one request
#: per instrument, so a spec that forgot ``with:`` would not just touch the
#: network during validate — it would touch it 727 times.
FETCHES_WITHOUT_SOURCE = {
    "universe.augment_with_news": "ticker_rank",
    "universe.augment_with_funding": "funding",
    "universe.augment_with_contract_meta": "contract_meta",
    "universe.augment_with_open_interest": "open_interest_snapshot",
    "universe.augment_with_oi_change": "oi_change_snapshot",
    "universe.augment_with_long_short_ratio": "long_short_ratio_snapshot",
    "universe.augment_with_spread": "book_ticker",
    "universe.augment_with_indicator": "universe_bars",
}

#: The one universe step that needs BARS, named once because three layers key off
#: it: the dry-run has to fabricate the timeframes it asks for, the capture has to
#: plan a roster from the steps BEFORE it, and the collection plan has to know the
#: timeframe set is the union of every occurrence.
BARS_BLOCK = "universe.augment_with_indicator"

#: Columns that only exist once a SECOND source is supplied to the step that
#: produces them: ``column -> (block, the ``with:`` name that produces it)``.
#:
#: ``universe.augment_with_funding`` emits ``fundingRatePct`` from ``funding``
#: alone and ``fundingRateApr`` only when ``funding_info`` is there too — and
#: when it is not, the block warns and leaves the column NaN (it must not assume
#: an 8-hour settlement interval; see its docstring). NaN is the right value and
#: the wrong end state for a spec: ``score: fundingRateApr`` over an all-NaN
#: column drops every row, so the run emits an EMPTY basket and ``validate``
#: returns the generic "produced no candidates" warning with errors=[]. That is
#: the green-validate / wrong-run asymmetry, so it is refused statically instead.
#:
#: :data:`FETCHES_WITHOUT_SOURCE` cannot express this: it is one required name per
#: block, checked whether or not the spec uses the column.
COLUMN_REQUIRES_SOURCE = {
    "fundingRateApr": ("universe.augment_with_funding", "funding_info"),
    "fundingIntervalHours": ("universe.augment_with_funding", "funding_info"),
}

DENIED_PREFIXES = {
    "data.fetch_": "performs a REST call; declare the source under `data:` instead",
}

PRICE_COLUMNS = {"open", "high", "low", "close", "volume", "quote_volume"}


class SpecError(ValueError):
    """Raised when a spec references an unknown block or malformed node."""


def block_namespaces() -> List[str]:
    """Every ``cyqnt_trd.blocks`` submodule a spec may dispatch into."""
    import pkgutil

    import cyqnt_trd.blocks as _blocks

    return sorted(
        info.name
        for info in pkgutil.iter_modules(_blocks.__path__)
        if not info.name.startswith("_") and info.name not in DENIED_NAMESPACES
    )


def resolve_block(ref: str) -> Callable[..., Any]:
    """Resolve ``"<namespace>.<fn>"`` to the real blocks callable.

    Raises :class:`SpecError` when the namespace is denied or unknown, when the
    name does not exist, or when it exists only because the module imported it
    from somewhere else (``Optional``, ``dataclass``, ``field`` …) — those are
    callable and would resolve, but they are not blocks.
    """
    if not isinstance(ref, str) or "." not in ref:
        raise SpecError(f"block reference must be '<module>.<fn>', got {ref!r}")
    namespace, _, fn_name = ref.partition(".")

    if namespace in DENIED_NAMESPACES:
        raise SpecError(
            f"namespace {namespace!r} is not available to a spec: "
            f"{DENIED_NAMESPACES[namespace]}"
        )
    if namespace.startswith("_"):
        # ``__init__`` re-exports register(), whose __module__ is
        # cyqnt_trd.blocks.strategy — so it passed every later check and merely
        # validating a spec could mutate the process-wide plugin registry, which
        # is exactly what denying the `strategy` namespace exists to stop.
        raise SpecError(
            f"namespace {namespace!r} is not a block module; a spec dispatches "
            f"into {block_namespaces()}"
        )
    if fn_name in DENIED_FUNCTION_NAMES:
        raise SpecError(
            f"block {ref!r} is not available to a spec: "
            f"{DENIED_FUNCTION_NAMES[fn_name]}"
        )
    for prefix, reason in DENIED_PREFIXES.items():
        if ref.startswith(prefix):
            raise SpecError(f"block {ref!r} is not available to a spec: {reason}")

    import importlib

    try:
        module = importlib.import_module(f"cyqnt_trd.blocks.{namespace}")
    except ImportError as exc:
        raise SpecError(
            f"no block module {namespace!r}; available: {block_namespaces()}"
        ) from exc

    fn = getattr(module, fn_name, None)
    if fn is None or not callable(fn):
        raise SpecError(f"{ref!r} is not a callable block in {namespace}")
    origin = getattr(fn, "__module__", "")
    if not origin.startswith("cyqnt_trd.blocks"):
        raise SpecError(
            f"{ref!r} resolves to {origin or '?'}.{fn_name}, which is imported "
            f"into {namespace} rather than defined in the blocks package — "
            f"typing aliases and dataclass helpers are callable but are not blocks"
        )
    # Anywhere inside the package is fine: `conditions` deliberately re-exports
    # a few names from `indicators` / `entry` / `patterns` via __getattr__
    # because that is where an author looks for them first.
    return fn


def _first_param_is_df(fn: Callable[..., Any]) -> bool:
    """True if the block's first positional parameter is named ``df``.

    Delegates to :func:`cyqnt_trd.blocks._utils.first_param_is_df`; see that
    docstring for why the detection does not live here any more.
    """
    from cyqnt_trd.blocks._utils import first_param_is_df

    return first_param_is_df(fn)


# ---------------------------------------------------------------------------
# Indicator evaluation
# ---------------------------------------------------------------------------


INDICATOR_KEYS = frozenset(
    {"block", "input", "inputs", "params", "output", "column"}
)


def eval_indicator(df, ispec: Dict[str, Any], env: Optional[Dict[str, Any]] = None):
    """Evaluate one named indicator spec against ``df`` → a pandas Series.

    Spec fields::

        block:  "indicators.ema"          # required, "<ns>.<fn>"
        input:  "close" | "df" | <ref>    # single first argument
        inputs: [<ref>, <ref>, ...]       # OR several, for multi-series blocks
        params: {period: 50}              # kwargs
        output: 0                         # component index when it returns a tuple
        column: "adx"                     # column name when it returns a DataFrame

    ``<ref>`` is a DataFrame column, ``"df"``, or **the name of an indicator
    defined earlier in the same spec** — that last one is what ``env`` is for.
    Without it ``regime.adx_regime`` could never be fed ``indicators.adx``, and
    the whole regime / derivatives family takes a Series rather than a frame, so
    chaining is not a convenience here: it is the only way to reach them.

    ``inputs:`` exists for the same reason. ``derivatives.cvd(buy_volume,
    sell_volume)``, ``derivatives.basis(spot_close, perp_close)`` and
    ``derivatives.liquidation_imbalance(long_liq_usd, short_liq_usd)`` all take
    two series; passing one positional argument could never call them.
    """
    env = env if env is not None else {}
    unknown = sorted(set(ispec) - INDICATOR_KEYS)
    if unknown:
        raise SpecError(
            f"unknown indicator field(s) {unknown}; allowed: {sorted(INDICATOR_KEYS)}"
        )
    if "input" in ispec and "inputs" in ispec:
        raise SpecError("give either 'input' or 'inputs', not both")

    fn = resolve_block(ispec["block"])
    params = dict(ispec.get("params", {}) or {})

    if "inputs" in ispec:
        refs = ispec["inputs"]
        if not isinstance(refs, (list, tuple)) or not refs:
            raise SpecError("'inputs' must be a non-empty list of references")
        args: List[Any] = [_series_from_token(df, env, ref) for ref in refs]
    elif "input" in ispec:
        # Explicit wins. Auto-detection is a default, not an override — silently
        # replacing a stated `input: close` with the whole frame is how you get a
        # block computing something other than what the spec says.
        args = [df if ispec["input"] == "df" else _series_from_token(df, env, ispec["input"])]
    else:
        args = [df if _first_param_is_df(fn) else _series_from_token(df, env, "close")]

    try:
        out = fn(*args, **params)
    except TypeError as exc:
        raise SpecError(
            f"{ispec['block']!r} rejected the call: {exc}. Its signature is "
            f"{_signature_text(fn)} — check 'input'/'inputs' arity and 'params' names"
        ) from exc

    return _select_component(out, ispec)


def _signature_text(fn: Callable[..., Any]) -> str:
    try:
        return f"{fn.__name__}{inspect.signature(fn)}"
    except (ValueError, TypeError):  # pragma: no cover - builtins
        return getattr(fn, "__name__", "?") + "(...)"


def _select_component(out: Any, ispec: Dict[str, Any]):
    """Reduce a block's return value to the single Series an indicator must be.

    The reduction itself lives in :func:`blocks._utils.select_indicator_component`
    so ``universe.augment_with_indicator`` resolves ``output:`` identically — see
    that docstring. This wrapper only re-labels the failure as a
    :class:`SpecError`, because a bad ``output:`` index is a fault in the SPEC and
    every caller of ``validate`` catches that type.
    """
    from cyqnt_trd.blocks._utils import (
        IndicatorShapeError,
        select_indicator_component,
    )

    try:
        return select_indicator_component(
            out,
            ref=ispec.get("block", "?"),
            output=(None if "output" not in ispec else ispec["output"]),
            column=ispec.get("column"),
        )
    except IndicatorShapeError as exc:
        raise SpecError(str(exc)) from exc


def _series_from_token(df, env: Dict[str, Any], token: Any):
    """Resolve a reference token to a value usable as a block argument.

    Order: literal ``"df"`` → the DataFrame; a name in ``env`` (a previously
    computed indicator) → that Series; a DataFrame column → that column.
    """
    if token == "df":
        return df
    if isinstance(token, (int, float, bool)):
        # A literal threshold, passed straight through: the comparators take
        # ``(series, float)``, so wrapping it in a Series would break the very
        # blocks it is meant to serve.
        return token
    if isinstance(token, str):
        if token in env:
            return env[token]
        if token in getattr(df, "columns", []):
            return df[token]
        from .vocabulary import column_owner

        owner = column_owner(token)
        if owner:
            raise SpecError(
                f"column {token!r} comes from the {owner!r} data source, which this "
                f"spec does not declare. Add:\n    data:\n      {owner}: "
                f"{{ dir: <path> }}"
            )
    raise SpecError(
        f"cannot resolve reference {token!r}; not 'df', a defined indicator, "
        f"a DataFrame column, or a number. A string in 'args' is always read as "
        f"a reference — there is no way to tell a column name from a string "
        f"value — so pass string constants as a keyword instead: "
        f"params: {{<name>: {token!r}}}"
    )


# ---------------------------------------------------------------------------
# Combinator tree evaluation
# ---------------------------------------------------------------------------


def eval_node(df, env: Dict[str, Any], node: Any):
    """Evaluate a combinator-tree node → a boolean pandas Series.

    Node shapes (checked in order)::

        {all_of: [node, ...]}
        {any_of: [node, ...]}
        {not: node}
        {exclude_when: {base: node, exclusions: [node, ...]}}
        {cond: "conditions.xxx", args: [ref, ...], params: {..}}   # leaf
    """
    from cyqnt_trd.blocks import entry as _entry

    if not isinstance(node, dict):
        raise SpecError(f"combinator node must be a mapping, got {type(node).__name__}")

    if "all_of" in node:
        return _entry.all_of([eval_node(df, env, n) for n in node["all_of"]])
    if "any_of" in node:
        return _entry.any_of([eval_node(df, env, n) for n in node["any_of"]])
    if "not" in node:
        inner = eval_node(df, env, node["not"])
        return ~inner.astype(bool)
    if "exclude_when" in node:
        block = node["exclude_when"]
        base = eval_node(df, env, block["base"])
        exclusions = [eval_node(df, env, n) for n in block.get("exclusions", [])]
        return _entry.exclude_when(base, exclusions)
    if "cond" in node:
        fn = resolve_block(node["cond"])
        args = [_series_from_token(df, env, a) for a in node.get("args", [])]
        params = dict(node.get("params", {}) or {})
        try:
            out = fn(*args, **params)
        except TypeError as exc:
            # Without the block name and its signature this surfaces as a bare
            # "cannot convert the series to float" from three frames down, which
            # says nothing about which condition in the tree is wrong.
            raise SpecError(
                f"condition {node['cond']!r} rejected the call: {exc}. Its "
                f"signature is {_signature_text(fn)} — check the order and count "
                f"of 'args' (a bare number stays a number; a name resolves to a "
                f"Series)"
            ) from exc
        import pandas as pd

        if not isinstance(out, pd.Series):
            raise SpecError(
                f"condition {node['cond']!r} returned {type(out).__name__}; a "
                f"condition must evaluate to a boolean Series"
            )
        return out

    raise SpecError(
        f"unknown combinator node keys {sorted(node)}; expected one of "
        "all_of / any_of / not / exclude_when / cond"
    )


# ---------------------------------------------------------------------------
# make_signals builder
# ---------------------------------------------------------------------------


SELECTION_KEYS = frozenset({"universe", "features", "score", "top_k", "long_when",
                            "short_when", "min_score", "max_score", "dedupe_by",
                            "order"})

#: how to collapse rows that are the same bet before taking the top K.
#:
#: ``base_asset`` (the default) counts BTCUSDT and BTCUSDC as one name. Square's
#: mention counts are keyed on the *base token*, so the same buzz score joins onto
#: every quote pair of that token: a "top 5" then came back as
#: ``BTCUSDC, BTCUSDT, SOLUSDT, SOLUSDC, BNBUSDT`` — three assets wearing five
#: slots, with double weight on two of them. Anyone sizing off that list doubles
#: their BTC exposure without asking for it.
#:
#: ``none`` keeps every row, for a strategy that really does treat quote pairs as
#: separate instruments (a USDC/USDT basis trade, say).
DEDUPE_MODES = ("base_asset", "none")

#: which end of the score column the basket is taken from.
#:
#: ``desc`` (the default) is the historical behaviour: biggest score first. It
#: suits "most mentioned", "highest turnover", "biggest gainer".
#:
#: ``asc`` exists because a whole class of screens is naturally expressed at the
#: *bottom* of a column — "the five most negative funding rates", "the biggest
#: losers" — and without it the only way to say so was to narrow the universe
#: with a step that happens to sort that way (``universe.top_losers``) and then
#: rank on a different, incidentally-descending column. That detour ranks by
#: something other than what the author asked for, and nothing in the output
#: admits it.
SORT_ORDERS = ("desc", "asc")
UNIVERSE_STEP_KEYS = frozenset({"block", "params", "with"})


def _refuse_implicit_fetch(step: Dict[str, Any]) -> None:
    """A universe step that would fetch its own source must be given one.

    ``augment_with_news(tickers, ticker_rank_df=None)`` falls back to a live
    Square call when the second argument is absent. That turns ``validate`` —
    which dry-runs the compiled selection — into outbound REST traffic on a
    frontend-supplied spec, and makes a backtest depend on today's data. The
    block stays available; the spec just has to say where the source comes from.
    """
    ref = step.get("block")
    needed = FETCHES_WITHOUT_SOURCE.get(ref)
    provided = set(step.get("with") or [])
    if needed and needed not in provided:
        raise SpecError(
            "universe step %r fetches %s itself when no source is given, which "
            "would make validate hit the network and a backtest read live data. "
            "Declare the source: `with: [%s]`." % (ref, needed, needed))


def run_universe_steps(steps: List[Dict[str, Any]], frame: Any,
                       extras: Dict[str, Any]) -> Any:
    """Run a ``selection.universe`` pipeline (or a prefix of one) over *frame*.

    Extracted so the two-pass capture's Pass 1 executes the SAME code the run
    does. A capture that re-implemented "apply the narrowing steps" would derive
    its bars roster from a slightly different funnel than the one the run walks,
    and the difference shows up as a coverage refusal inside
    ``augment_with_indicator`` — pointing at the join, which is the one place that
    is not at fault.

    Returns the narrowed frame, or an empty frame the moment a step empties it.
    """
    import pandas as pd

    for step in steps:
        _refuse_implicit_fetch(step)
        fn = resolve_block(step["block"])
        source_names = list(step.get("with") or [])
        missing = [name for name in source_names
                   if name not in extras or extras[name] is None]
        if missing:
            raise SpecError(
                "universe step %r requires source(s) %s from the input bundle, "
                "but they were not provided; `with:` never falls back to live "
                "network data" % (step["block"], missing))
        args = [frame] + [extras[name] for name in source_names]
        try:
            frame = fn(*args, **dict(step.get("params") or {}))
        except TypeError as exc:
            raise SpecError(
                f"universe step {step['block']!r} rejected the call: {exc}. Its "
                f"signature is {_signature_text(fn)} — the running frame is passed "
                f"first, then each name in 'with'"
            ) from exc
        if not isinstance(frame, pd.DataFrame):
            raise SpecError(
                f"universe step {step['block']!r} returned "
                f"{type(frame).__name__}; each step must return the narrowed "
                f"or widened universe frame"
            )
        if not len(frame):
            return frame
    return frame


def universe_steps_before_bars(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The universe steps that run BEFORE the first indicator step.

    That prefix is Pass 1 of a bars capture, and it is necessarily
    cross-sectional: every step in it reads a frame already in hand or a
    whole-market source, so running it costs nothing and its survivors are the
    roster the bars are fetched for.

    Steps AFTER the first indicator step are excluded even if they are cheap
    filters, because they may depend on the indicator column that does not exist
    yet — and a prefix that raised while being planned would be a capture failure
    for a spec that runs fine.
    """
    steps = (spec.get("selection") or {}).get("universe") or []
    prefix: List[Dict[str, Any]] = []
    for step in steps:
        if isinstance(step, dict) and step.get("block") == BARS_BLOCK:
            break
        prefix.append(step)
    return prefix


def bar_timeframes_for_spec(spec: Dict[str, Any]) -> List[str]:
    """Every timeframe this spec's indicator steps ask for, in first-seen order.

    The UNION, and not the first one: a resonance screen names three, and
    collecting one of them would leave two indicator columns NaN — which reads
    downstream as "no coin met the condition" rather than as a short capture.
    """
    out: List[str] = []
    for step in (spec.get("selection") or {}).get("universe") or []:
        if not isinstance(step, dict) or step.get("block") != BARS_BLOCK:
            continue
        timeframe = (step.get("params") or {}).get("timeframe")
        if timeframe is None:
            raise SpecError(
                "a %s step declares no `timeframe:`; the capture cannot know which "
                "bars to collect, and the block cannot know which to read" % BARS_BLOCK)
        name = str(timeframe).strip()
        if name and name not in out:
            out.append(name)
    return out


def build_selection_fn(spec: Dict[str, Any]) -> Callable[..., List[Dict[str, Any]]]:
    """Compile a ``selection:`` section into ``blocks.strategy``'s selection_fn.

    A trade spec evaluates over a frame whose rows are *bars*; a selection spec
    evaluates over a frame whose rows are *symbols*. That is the only
    difference, which is why this reuses ``eval_indicator`` / ``eval_node``
    unchanged rather than growing a second expression language: a column is a
    column, and ``conditions.value_above`` does not care whether the axis is
    time or instrument.

    Shape::

        selection:
          universe:                       # pipeline over the universe frame
            - { block: universe.filter_quote_volume, params: {...} }
            - { block: universe.augment_with_news, with: [ticker_rank] }
          features:                       # per-symbol Series, same syntax as indicators
            skew: { block: ..., input: news_bull_ratio }
          score: news_mention_count       # column or feature to rank by
          order: desc                     # desc (default) | asc
          top_k: 5
          min_score: 1.0                  # absolute floor  (>=), never flipped
          max_score: -0.01                # absolute ceiling (<=), never flipped
          long_when: { cond: conditions.value_above, args: [news_bull_ratio, 0.55] }

    ``order: asc`` plus ``max_score`` is how a bottom-of-the-column screen is
    stated directly: ``score: fundingRatePct, order: asc, max_score: -0.01``
    reads "the most negative funding rates, and only below -1bp".
    """
    section = spec.get("selection") or {}
    steps = list(section.get("universe") or [])
    feature_specs: Dict[str, Any] = dict(section.get("features") or {})
    score_ref = section.get("score")
    top_k = int(section.get("top_k", 10))
    min_score = section.get("min_score")
    max_score = section.get("max_score")
    long_node = section.get("long_when")
    short_node = section.get("short_when")
    dedupe_by = str(section.get("dedupe_by", "base_asset")).lower()
    if dedupe_by not in DEDUPE_MODES:
        raise SpecError(
            "selection.dedupe_by must be one of %s, got %r"
            % (", ".join(DEDUPE_MODES), dedupe_by))
    order = str(section.get("order", "desc")).lower()
    if order not in SORT_ORDERS:
        raise SpecError(
            "selection.order must be one of %s, got %r — %r is the default "
            "(biggest score first); use %r to rank from the bottom of the "
            "column (\"the most negative funding\")"
            % (", ".join(SORT_ORDERS), order, "desc", "asc"))
    score_ascending = order == "asc"

    def selection_fn(
        universe_df,
        ticker_rank_df=None,
        *,
        frames=None,
        **runtime_extras: Any,
    ) -> List[Dict[str, Any]]:
        import pandas as pd

        frame = universe_df
        if frame is None or not len(frame):
            return []
        # Every named non-market frame in cyqnt.input/v1 is available to a
        # selection step through ``with: [...]``.  This is the generic bridge
        # colleague-provided sources use; funding is the first consumer, not a
        # one-off field added to UniverseBundle.
        extras = dict(frames or {})
        extras.update(runtime_extras)
        extras.update({"ticker_rank": ticker_rank_df, "universe": universe_df})

        frame = run_universe_steps(steps, frame, extras)
        if not len(frame):
            return []

        env: Dict[str, Any] = {}
        for name, ispec in feature_specs.items():
            env[name] = eval_indicator(frame, ispec, env)

        if score_ref is None:
            raise SpecError("selection.score is required: name the column or feature to rank by")
        scores = _series_from_token(frame, env, score_ref)
        long_mask = eval_node(frame, env, long_node) if long_node else None
        short_mask = eval_node(frame, env, short_node) if short_node else None

        symbol_col = next((c for c in ("symbol", "instrument_id") if c in frame.columns), None)
        if symbol_col is None:
            raise SpecError(
                "the universe frame has no 'symbol' / 'instrument_id' column after "
                "the pipeline; a candidate with no instrument is not actionable")

        ranked = pd.DataFrame({
            "symbol": frame[symbol_col].astype(str).str.upper(),
            "score": pd.Series(scores).astype(float).reindex(frame.index),
        })
        # A symbol whose score could not be computed is dropped, never defaulted:
        # a NaN ranked against real numbers sorts arbitrarily and then reads as a
        # recommendation.
        ranked = ranked.dropna(subset=["score"])
        # Absolute bounds, inclusive, and deliberately NOT relative to ``order``:
        # ``min_score`` is always the floor and ``max_score`` always the ceiling,
        # whichever end the ranking is taken from. The tempting alternative — let
        # ``min_score`` mean "the cutoff on the winning side" so it flips to a
        # ceiling under ``order: asc`` — would make ``min_score: 1.0`` mean two
        # different things in two specs, and both example_selection.yaml and the
        # demo template already ship ``min_score: 1.0``. Someone later adding
        # ``order: asc`` to one of those would silently invert a filter they never
        # touched. So: floor is a floor.
        if min_score is not None:
            ranked = ranked[ranked["score"] >= float(min_score)]
        if max_score is not None:
            ranked = ranked[ranked["score"] <= float(max_score)]
        if dedupe_by == "base_asset":
            # Collapse BEFORE head(top_k), so the basket is top_k distinct bets
            # rather than top_k rows.
            #
            # Every quote pair of a token carries the SAME score (the buzz was
            # measured per token), so which pair survives is decided by the
            # tie-break, not the score. Order by turnover for it: keeping
            # ETHUSDC over ETHUSDT because it happened to appear first in the
            # response is an arbitrary choice about where the order will fill.
            # The SAME base-token function the news join uses. They diverged:
            # news_features._base_token strips only fiat quotes, so ETHBTC and
            # SOLBNB kept their full symbol while universe.augment_with_news had
            # already given them the per-token buzz score via news_feed.base_token
            # — the dedupe then failed to collapse exactly the pairs it exists
            # for, and a top-5 came back as three assets in five slots.
            from cyqnt_trd.blocks.news_feed import base_token as _base_token

            volume_column = next(
                (c for c in ("quoteVolume", "quote_volume", "volume_quote")
                 if c in frame.columns), None)
            if volume_column is not None:
                # Only the score key follows ``order``. ``__turnover__`` stays
                # descending in both directions because it is the liquidity
                # tie-break, not a ranking criterion: under ``order: asc`` a
                # turnover key that flipped with it would keep the *least*
                # tradable quote pair of every asset — and since the surviving
                # row carries the same score either way, nothing in the basket
                # or the reason string would reveal the swap.
                ranked = ranked.assign(
                    __turnover__=pd.to_numeric(
                        frame.loc[ranked.index, volume_column], errors="coerce"
                    ).fillna(0.0)
                ).sort_values(["score", "__turnover__"],
                              ascending=[score_ascending, False])
            else:
                ranked = ranked.sort_values("score", ascending=score_ascending)
            ranked = ranked[~ranked["symbol"].map(_base_token).duplicated(keep="first")]
            ranked = ranked.drop(columns=["__turnover__"], errors="ignore")
        else:
            ranked = ranked.sort_values("score", ascending=score_ascending)
        ranked = ranked.head(top_k)

        feature_columns = [c for c in frame.columns if c not in (symbol_col,)]
        kept: List[Dict[str, Any]] = []
        for index, row in ranked.iterrows():
            side = "neutral"
            if long_mask is not None and bool(long_mask.get(index, False)):
                side = "long"
            elif short_mask is not None and bool(short_mask.get(index, False)):
                side = "short"
            elif long_mask is not None or short_mask is not None:
                continue          # a direction was asked for and neither held
            features = {
                name: (float(value) if isinstance(value, (int, float)) else str(value))
                for name, value in frame.loc[index, feature_columns].items()
                if pd.notna(value)
            }
            # Mirror the frame-column branch above: a computed feature can be
            # categorical (derivatives.funding_rate_state yields
            # "bullish_squeeze"), and a bare float() turned that into
            # "ValueError: could not convert string to float" naming neither the
            # feature nor the block — for exactly the values conditions.state_equals
            # was added to compare.
            for name, series in env.items():
                value = series.get(index, None)
                if value is None or pd.isna(value):
                    continue
                features[name] = (float(value) if isinstance(value, (int, float))
                                  and not isinstance(value, bool) else str(value))
            kept.append({"symbol": row["symbol"], "score": round(float(row["score"]), 6),
                         "side": side, "features": features})

        # Rank AFTER the direction filter, so the numbers are contiguous and
        # "rank N of M" counts the basket that was actually returned. Numbering
        # during the loop left gaps where a symbol was skipped, and reported a
        # total that did not match the list beside it.
        #
        # The direction is spelled out because "rank 1 of 5" reads identically
        # for both orders: an operator holding a basket could not tell whether
        # rank 1 was the highest funding or the lowest, which is the difference
        # between a short screen and a long one. ``order=`` names the spec key
        # so the log can be matched back to the YAML that produced it.
        order_note = "order=%s (%s first)" % (
            order, "lowest" if score_ascending else "highest")
        out: List[Dict[str, Any]] = []
        for position, candidate in enumerate(kept, start=1):
            out.append({**candidate, "rank": position,
                        "reason": "%s=%.4g, rank %d of %d, %s"
                                  % (score_ref, candidate["score"], position,
                                     len(kept), order_note)})
        return out

    return selection_fn


def build_make_signals(spec: Dict[str, Any]) -> Callable[[Any], Tuple[Any, Optional[Any]]]:
    """Compile a validated spec into a ``make_signals(df) -> (long, short)``."""
    signals = spec.get("signals", {}) or {}
    ind_specs: Dict[str, Any] = signals.get("indicators", {}) or {}
    entry_spec: Dict[str, Any] = signals.get("entry", {}) or {}
    long_node = entry_spec.get("long")
    short_node = entry_spec.get("short")

    def make_signals(df):
        import pandas as pd

        env: Dict[str, Any] = {}
        for name, ispec in ind_specs.items():
            # env is threaded in so a later indicator can consume an earlier one
            # by name. Declaration order is the dependency order.
            env[name] = eval_indicator(df, ispec, env)
        long_s = eval_node(df, env, long_node) if long_node else None
        short_s = eval_node(df, env, short_node) if short_node else None
        # Long-only (no short node) / short-only: return an all-False boolean
        # Series for the missing side rather than None, so every consumer —
        # the event-driven BlockStrategyPlugin AND the vectorized backtester
        # (which does short_sig.reindex(...)) — handles it uniformly.
        if long_s is None:
            long_s = pd.Series(False, index=df.index)
        if short_s is None:
            short_s = pd.Series(False, index=df.index)
        return long_s, short_s

    return make_signals
