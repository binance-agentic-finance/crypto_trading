"""``data.*`` — the DAG-spec data-node surface, backed by the standard bot.

The strategy Spec puts ``type: data`` nodes in a ``runtime`` package rather
than in ``cyqnt_trd.blocks``, and codegen emits calls of the exact form
``data.<function>(**params)``. This module *is* that surface. A generated
strategy does::

    from cyqnt_trd.standard_bot.runtime import data, action

    klines_1h = data.klines(symbol="BTCUSDT", interval="1h", limit=500)

and gets the repo's real data layer underneath, not a second fetcher.

Three things every call gives you that a bare fetcher does not:

1. **A recorded status.** Fetch failures raise :class:`DataUnavailable` (from
   the catalog) rather than returning an empty frame. A DAG that wants to
   degrade instead of fail uses :func:`collect`, which returns frames *and* a
   per-node status map you can hand to the snapshot assembler.
2. **A declared replay discipline.** Every function is backed by a
   :class:`DataNodeSpec` carrying its availability tier and PIT hazard, so
   ``validate_nodes()`` can refuse to compile a backtest that reads a
   forward-only snapshot.
3. **One decision time.** :func:`session` stamps every frame in a run with the
   same ``as_of``, which is what makes a multi-node DAG point-in-time coherent
   instead of "each node fetched whenever it happened to run".

The function names here are the ``function:`` values in the spec, and the
keyword names are the ``params[].key`` values — both are generated from
:mod:`cyqnt_trd.standard_bot.data.catalog`, so ``docs/DATA_API.md``, the Canvas
form and this module cannot drift apart.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..data.catalog import (
    Availability,
    DataNodeSpec,
    DataUnavailable,
    get_node,
    list_node_names,
)

__all__ = [
    "DataUnavailable",
    "FetchResult",
    "DataSession",
    "session",
    "call",
    "collect",
    "validate_nodes",
    "describe",
    "install_node_function",
    # generated node functions are appended to __all__ at the bottom
]


@dataclass
class FetchResult:
    """One node's output plus everything the snapshot layer needs to be honest."""

    node: str
    frame: Any = None
    status: str = "ok"          # ok | degraded | error
    warning: str = ""
    as_of: int = 0
    availability: str = ""
    source_path: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass
class DataSession:
    """A single decision cycle: one ``as_of`` shared by every node in the DAG.

    Without this each ``data.*`` call stamps its own wall clock, so a DAG whose
    news node ran 40s after its kline node is quietly mixing two decision times.
    """

    as_of_ms: int = 0
    results: Dict[str, FetchResult] = field(default_factory=dict)
    #: refuse forward-only sources (set when compiling for backtest)
    require_backtestable: bool = False

    def __post_init__(self) -> None:
        if not self.as_of_ms:
            self.as_of_ms = int(time.time() * 1000)

    def call(self, node_name: str, **params: Any) -> Any:
        spec = get_node(node_name)
        if self.require_backtestable and not spec.backtestable:
            raise DataUnavailable(
                node_name,
                "availability=%s but this session requires BACKTESTABLE "
                "(%s)" % (spec.availability.value, spec.pit_hazard or "no replayable history"),
            )
        try:
            frame = spec.fetch(**params)
        except DataUnavailable as exc:
            self.results[node_name] = FetchResult(
                node=node_name, frame=None, status="error", warning=exc.reason,
                as_of=self.as_of_ms, availability=spec.availability.value,
                source_path=spec.source_path.value,
            )
            raise
        status, warning = _grade(spec, frame)
        self.results[node_name] = FetchResult(
            node=node_name, frame=frame, status=status, warning=warning,
            as_of=self.as_of_ms, availability=spec.availability.value,
            source_path=spec.source_path.value,
        )
        return frame

    def status_map(self) -> Dict[str, str]:
        return {name: result.status for name, result in self.results.items()}

    def warnings(self) -> List[str]:
        return [
            "%s: %s" % (name, result.warning)
            for name, result in self.results.items()
            if result.warning
        ]


_ACTIVE: List[DataSession] = []


class _SessionContext:
    def __init__(self, sess: DataSession) -> None:
        self.session = sess

    def __enter__(self) -> DataSession:
        _ACTIVE.append(self.session)
        return self.session

    def __exit__(self, *exc: Any) -> None:
        _ACTIVE.pop()


def session(
    *, as_of_ms: Optional[int] = None, require_backtestable: bool = False
) -> _SessionContext:
    """Open a decision cycle. Every ``data.*`` call inside shares one ``as_of``.

    ``require_backtestable=True`` turns a forward-only read into a hard error —
    use it when compiling a spec for backtest so the failure lands at compile
    time, not as a silently flat equity curve.
    """
    return _SessionContext(
        DataSession(as_of_ms=as_of_ms or 0, require_backtestable=require_backtestable)
    )


def _grade(spec: DataNodeSpec, frame: Any) -> Tuple[str, str]:
    """Classify a fetched frame. Empty is NOT automatically ok."""
    empty = frame is None or getattr(frame, "empty", False) or (
        isinstance(frame, (list, dict)) and len(frame) == 0
    )
    if empty:
        # For an IP-gated snapshot API an empty body is the documented shape of
        # "you are not whitelisted" — it must not read as "no events".
        return "degraded", "empty result (source=%s); do not read this as 'no data'" % (
            spec.source_path.value
        )
    if spec.availability is Availability.FORWARD_ONLY:
        return "ok", ""
    return "ok", ""


def call(node_name: str, **params: Any) -> Any:
    """Invoke a catalog node by name, joining the active session if there is one."""
    if _ACTIVE:
        return _ACTIVE[-1].call(node_name, **params)
    return get_node(node_name).fetch(**params)


def collect(requests: Sequence[Tuple[str, Dict[str, Any]]], *, as_of_ms: Optional[int] = None):
    """Fetch several nodes in one decision cycle, degrading instead of raising.

    Returns ``(frames, session)``. A node that fails is absent from ``frames``
    and marked ``error`` in ``session.status_map()`` — the caller decides
    whether that is fatal, which is the whole point of keeping status separate
    from data.
    """
    sess = DataSession(as_of_ms=as_of_ms or 0)
    frames: Dict[str, Any] = {}
    with _SessionContext(sess):
        for node_name, params in requests:
            try:
                frames[node_name] = sess.call(node_name, **params)
            except DataUnavailable:
                continue
    return frames, sess


def validate_nodes(node_names: Iterable[str], *, for_backtest: bool = False) -> List[str]:
    """Check a spec's data nodes before anything runs.

    Returns a list of human-readable problems; empty means the DAG's data layer
    is coherent. ``for_backtest=True`` additionally rejects every node whose
    availability is not ``BACKTESTABLE``, with the reason quoted from the
    catalog so the author sees *why* rather than just "failed".
    """
    problems: List[str] = []
    known = set(list_node_names())
    for name in node_names:
        if name not in known:
            problems.append("unknown data node %r (known: %s)" % (name, ", ".join(sorted(known))))
            continue
        spec = get_node(name)
        if spec.availability is Availability.EXTERNAL_PENDING:
            problems.append(
                "data.%s is EXTERNAL_PENDING — the feed is not onboarded; "
                "declare it but do not depend on it (%s)" % (name, spec.endpoint)
            )
        elif spec.availability is Availability.NOT_WIRED:
            problems.append("data.%s has no endpoint anywhere" % name)
        elif for_backtest and not spec.backtestable:
            problems.append(
                "data.%s is %s, not replayable for backtest: %s"
                % (name, spec.availability.value, spec.pit_hazard or "no history")
            )
    return problems


def describe(node_name: str) -> Dict[str, Any]:
    """Full catalog entry for one node (what the Canvas form and codegen read)."""
    return get_node(node_name).to_dict()


# ---------------------------------------------------------------------------
# Generated node functions: data.klines(...), data.funding(...), ...
#
# Generated rather than hand-written so the callable surface, docs/DATA_API.md
# and the Canvas param forms all come from one registry and cannot drift.
# ---------------------------------------------------------------------------


def _make_node_function(spec: DataNodeSpec):
    def node_fn(**params: Any) -> Any:
        missing = [
            item.key for item in spec.params if item.required and item.key not in params
        ]
        if missing:
            raise TypeError(
                "data.%s missing required param(s): %s" % (spec.name, ", ".join(missing))
            )
        for item in spec.params:
            if item.key not in params and item.default is not None:
                params[item.key] = item.default
        return call(spec.name, **params)

    node_fn.__name__ = spec.name
    node_fn.__qualname__ = spec.name
    node_fn.__doc__ = "%s\n\nsource: %s (%s)\nreturns: %s\n%s" % (
        spec.description,
        spec.source_path.value,
        spec.availability.value,
        spec.returns.description,
        ("PIT hazard: %s" % spec.pit_hazard) if spec.pit_hazard else "",
    )
    return node_fn


def install_node_function(spec: DataNodeSpec) -> None:
    """Expose one catalog node as ``data.<name>``.

    Called at import for the built-ins, and by
    :mod:`..data.custom_sources` when a user registers a feed, so a custom node
    is callable the moment it exists.
    """
    globals()[spec.name] = _make_node_function(spec)
    if spec.name not in __all__:
        __all__.append(spec.name)


def _install_node_functions() -> None:
    from ..data.catalog import list_nodes

    for spec in list_nodes():
        install_node_function(spec)


_install_node_functions()
