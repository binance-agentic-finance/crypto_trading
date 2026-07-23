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

# Whitelisted blocks submodules that a spec may dispatch into. Restricting to
# this set is a security boundary: a spec can only call vetted, side-effect-free
# indicator/condition/combinator code — never arbitrary imports.
ALLOWED_NAMESPACES = {
    "indicators",
    "conditions",
    "entry",
    "patterns",
    "smc_structure",
    "smc_liquidity",
    "verdicts",
    "scoring",
}

PRICE_COLUMNS = {"open", "high", "low", "close", "volume", "quote_volume"}


class SpecError(ValueError):
    """Raised when a spec references an unknown block or malformed node."""


def resolve_block(ref: str) -> Callable[..., Any]:
    """Resolve ``"<namespace>.<fn>"`` to the real blocks callable.

    Raises :class:`SpecError` if the namespace is not whitelisted or the
    function does not exist / is not callable.
    """
    if not isinstance(ref, str) or "." not in ref:
        raise SpecError(f"block reference must be '<module>.<fn>', got {ref!r}")
    namespace, _, fn_name = ref.partition(".")
    if namespace not in ALLOWED_NAMESPACES:
        raise SpecError(
            f"namespace {namespace!r} not allowed; pick one of "
            f"{sorted(ALLOWED_NAMESPACES)}"
        )
    import importlib

    try:
        module = importlib.import_module(f"cyqnt_trd.blocks.{namespace}")
    except ImportError as exc:  # pragma: no cover - environment issue
        raise SpecError(f"cannot import cyqnt_trd.blocks.{namespace}: {exc}") from exc
    fn = getattr(module, fn_name, None)
    if fn is None or not callable(fn):
        raise SpecError(f"{ref!r} is not a callable block in {namespace}")
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


def eval_indicator(df, ispec: Dict[str, Any]):
    """Evaluate one named indicator spec against ``df`` → a pandas Series.

    Spec fields::

        block:  "indicators.ema"        # required, "<ns>.<fn>"
        input:  "close" | "df" | <col>  # what to feed as first arg (default close)
        params: {period: 50}            # kwargs
        output: 0                        # component index if the block returns a tuple
    """
    fn = resolve_block(ispec["block"])
    params = dict(ispec.get("params", {}) or {})
    inp = ispec.get("input", "close")

    args: List[Any] = []
    if _first_param_is_df(fn) or inp == "df":
        args.append(df)
    else:
        args.append(_series_from_token(df, {}, inp))

    out = fn(*args, **params)
    if isinstance(out, tuple):
        idx = int(ispec.get("output", 0))
        if idx < 0 or idx >= len(out):
            raise SpecError(
                f"{ispec['block']!r} returns a {len(out)}-tuple; "
                f"output index {idx} out of range"
            )
        out = out[idx]
    return out


def _series_from_token(df, env: Dict[str, Any], token: Any):
    """Resolve a reference token to a value usable as a block argument.

    Order: literal ``"df"`` → the DataFrame; a name in ``env`` (a previously
    computed indicator) → that Series; a DataFrame column → that column.
    """
    if token == "df":
        return df
    if isinstance(token, str):
        if token in env:
            return env[token]
        if token in getattr(df, "columns", []):
            return df[token]
    raise SpecError(
        f"cannot resolve reference {token!r}; not 'df', a defined indicator, "
        f"or a DataFrame column"
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
        return fn(*args, **params)

    raise SpecError(
        f"unknown combinator node keys {sorted(node)}; expected one of "
        "all_of / any_of / not / exclude_when / cond"
    )


# ---------------------------------------------------------------------------
# make_signals builder
# ---------------------------------------------------------------------------


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
            env[name] = eval_indicator(df, ispec)
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
