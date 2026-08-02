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
FETCHES_WITHOUT_SOURCE = {
    "universe.augment_with_news": "ticker_rank",
    "universe.augment_with_funding": "funding",
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
    """True if the block's first positional parameter is named ``df``."""
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (ValueError, TypeError):
        return False
    if not params:
        return False
    return params[0].name == "df"


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

    Blocks return four shapes and only one of them is directly usable, so this
    is where a spec finds out — with the fix in the message, rather than as a
    ``NoneType has no attribute`` three frames deeper.
    """
    import pandas as pd

    ref = ispec.get("block", "?")

    if isinstance(out, tuple):
        if "output" not in ispec and len(out) > 1:
            raise SpecError(
                f"{ref!r} returns a {len(out)}-tuple; add 'output: <0..{len(out) - 1}>' "
                f"to say which component you mean"
            )
        idx = int(ispec.get("output", 0))
        if idx < 0 or idx >= len(out):
            raise SpecError(
                f"{ref!r} returns a {len(out)}-tuple; output index {idx} out of range"
            )
        out = out[idx]

    if isinstance(out, pd.DataFrame):
        column = ispec.get("column")
        if column is None:
            raise SpecError(
                f"{ref!r} returns a DataFrame; add 'column: <name>' to pick one. "
                f"Available: {list(out.columns)[:12]}"
            )
        if column not in out.columns:
            raise SpecError(
                f"{ref!r} returns a DataFrame without column {column!r}; "
                f"available: {list(out.columns)[:12]}"
            )
        out = out[column]

    if not isinstance(out, pd.Series):
        raise SpecError(
            f"{ref!r} returned {type(out).__name__}, which cannot be used as an "
            f"indicator — an indicator must reduce to one pandas Series. Blocks "
            f"that return a plan, a score or a scalar belong in the section that "
            f"consumes them (risk.exit / sizing), not in signals.indicators"
        )
    return out


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
                            "short_when", "min_score", "dedupe_by"})

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
          score: news_mention_count       # column or feature to rank by, descending
          top_k: 5
          long_when: { cond: conditions.value_above, args: [news_bull_ratio, 0.55] }
    """
    section = spec.get("selection") or {}
    steps = list(section.get("universe") or [])
    feature_specs: Dict[str, Any] = dict(section.get("features") or {})
    score_ref = section.get("score")
    top_k = int(section.get("top_k", 10))
    min_score = section.get("min_score")
    long_node = section.get("long_when")
    short_node = section.get("short_when")
    dedupe_by = str(section.get("dedupe_by", "base_asset")).lower()
    if dedupe_by not in DEDUPE_MODES:
        raise SpecError(
            "selection.dedupe_by must be one of %s, got %r"
            % (", ".join(DEDUPE_MODES), dedupe_by))

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
        if min_score is not None:
            ranked = ranked[ranked["score"] >= float(min_score)]
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
                ranked = ranked.assign(
                    __turnover__=pd.to_numeric(
                        frame.loc[ranked.index, volume_column], errors="coerce"
                    ).fillna(0.0)
                ).sort_values(["score", "__turnover__"], ascending=False)
            else:
                ranked = ranked.sort_values("score", ascending=False)
            ranked = ranked[~ranked["symbol"].map(_base_token).duplicated(keep="first")]
            ranked = ranked.drop(columns=["__turnover__"], errors="ignore")
        else:
            ranked = ranked.sort_values("score", ascending=False)
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
        out: List[Dict[str, Any]] = []
        for position, candidate in enumerate(kept, start=1):
            out.append({**candidate, "rank": position,
                        "reason": "%s=%.4g, rank %d of %d"
                                  % (score_ref, candidate["score"], position, len(kept))})
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
