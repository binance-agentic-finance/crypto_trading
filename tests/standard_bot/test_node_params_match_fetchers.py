"""Guard: every declared node param must be a keyword its fetcher accepts.

Why this test exists
--------------------
The catalog declares each node's params (that declaration is what ``DATA_API.md``,
the Canvas form and codegen read) while the bytes come from a separate library
function. Nothing checked that the two agreed, and they did not: ``klines``
declared ``market_type`` while ``fetch_klines`` takes ``market``.

That is worse than a rename, because :func:`runtime.data._make_node_function`
fills declared defaults in before calling. ``market_type`` has a default, so it
was injected on *every* call and the caller could not avoid it::

    data.klines(symbol="BTCUSDT", interval="1h", limit=500)
    -> DataUnavailable: TypeError: fetch_klines() got an unexpected
                        keyword argument 'market_type'

Eight of the 25 wired nodes were broken this way — including ``klines``, the
most-used node in the catalog and the first line of every generated strategy.
The existing suite passed throughout, because it only checked that a fetcher
could be *imported*, never that it could be *called*.

The fix is :attr:`DataNodeSpec.param_aliases`, which translates the public name
to the fetcher's keyword. This test is what keeps the translation honest.
"""

from __future__ import annotations

import inspect

from cyqnt_trd.standard_bot.data.catalog import DataUnavailable, list_nodes


def _callable_nodes():
    """Nodes whose fetcher can actually be imported here (see
    ``test_data_catalog_runtime`` for the internal-client-absent ones)."""
    for spec in list_nodes():
        if not spec.fetcher:
            continue
        try:
            fn = spec.resolve_fetcher()
        except DataUnavailable:
            continue
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):        # builtins / C callables
            continue
        yield spec, fn, signature


def test_declared_params_are_accepted_by_the_fetcher():
    """Every declared param, after aliasing, must be a real fetcher keyword."""
    problems = []
    for spec, fn, signature in _callable_nodes():
        if any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values()):
            continue                            # **kwargs accepts anything
        accepted = set(signature.parameters)
        declared = {p.key for p in spec.params}
        bound = set(spec.bind_params({key: None for key in declared}))
        unknown = sorted(bound - accepted)
        if unknown:
            problems.append(
                "data.%s declares %s which %s does not accept (it takes %s). "
                "Add a param_aliases entry, or drop the param if the source has "
                "no such knob."
                % (spec.name, unknown, spec.fetcher, sorted(accepted))
            )
    assert not problems, "\n".join(problems)


def test_required_fetcher_args_are_declared():
    """A fetcher argument with no default must appear in the node's params.

    Otherwise the node cannot be called at all: ``liquidations`` omitted the
    ``frame`` it enriches, so it read as a standalone source and raised
    ``TypeError`` on every attempt.
    """
    problems = []
    for spec, fn, signature in _callable_nodes():
        declared = {p.key for p in spec.params}
        supplied = set(spec.bind_params({key: None for key in declared}))
        for name, param in signature.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if param.default is not param.empty or name in supplied:
                continue
            problems.append(
                "data.%s: %s requires %r but the node does not declare it"
                % (spec.name, spec.fetcher, name)
            )
    assert not problems, "\n".join(problems)


def test_defaults_that_get_injected_are_bindable():
    """The subset that used to fail unconditionally.

    ``runtime.data.<node>()`` injects every declared default, so a mismatched
    param with a default is not a latent bug — it breaks the node outright.
    """
    problems = []
    for spec, fn, signature in _callable_nodes():
        if any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values()):
            continue
        accepted = set(signature.parameters)
        injected = {p.key for p in spec.params if p.default is not None}
        bound = set(spec.bind_params({key: None for key in injected}))
        broken = sorted(bound - accepted)
        if broken:
            problems.append(
                "data.%s injects default(s) %s that %s rejects — this node fails "
                "on EVERY call, however it is invoked"
                % (spec.name, broken, spec.fetcher)
            )
    assert not problems, "\n".join(problems)


def test_public_fallback_specs_exist_and_match_the_declared_metrics():
    """A declared fallback must resolve, and speak the node's own vocabulary.

    The derivatives statistics come through a local ``binance-cli`` subprocess.
    When that returns non-JSON the node is dead, which is how open interest ended
    up being read from a month-old 5-minute parquet against hourly bars. The same
    numbers are on a public endpoint — but a fallback is only safe if it yields
    the field names the node declares, otherwise the melt silently invents
    metrics nobody reads.
    """
    from cyqnt_trd.data_cli import public_sources, rest_source

    public_sources.register_public_sources()
    declared = [s for s in list_nodes() if s.public_fallback]
    assert declared, "expected the CLI-backed derivatives nodes to declare fallbacks"

    problems = []
    for spec in declared:
        source = rest_source.get_spec(spec.public_fallback)
        if source is None:
            problems.append("data.%s -> unknown rest_source spec %r"
                            % (spec.name, spec.public_fallback))
            continue
        produced = set(source.fields)
        wanted = set(spec.value_columns) | set(spec.column_map)
        # every metric the node melts must be obtainable from the fallback
        missing = sorted(set(spec.value_columns) - produced)
        if missing and not (produced & set(spec.value_columns)):
            problems.append(
                "data.%s declares value_columns %s but fallback %r yields %s — "
                "none overlap, so the melt would produce metrics no reader wants"
                % (spec.name, sorted(spec.value_columns), spec.public_fallback,
                   sorted(produced)))
    assert not problems, "\n".join(problems)
