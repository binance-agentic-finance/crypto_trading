"""User-defined data nodes — the catalog's extension point.

Layering
--------
There are two halves to "where does a number come from", and they belong in
different places::

    RestSourceSpec           transport + parse
    (data_cli.rest_source)   base_url / path / params / data_path / fields /
                             coercions / ok_check / TTL cache

    DataNodeSpec             contract
    (standard_bot.catalog)   availability · pit_hazard · param schema ·
                             canonical shape (cyqnt.input/v1)

A ``/bn-data`` endpoint and a team's own endpoint are the *same thing* at the
transport layer — a JSON REST source described declaratively. What differs is
the contract: how far back it replays and how it lies if you replay it wrong.
So both go through one path, and neither is allowed into a bot without stating
the contract.

The built-in catalog covers the documented BigData / public / local universe.
It cannot cover a user's own feed: an internal team endpoint, a research
parquet, a vendor trial, a Python function that assembles something bespoke.
Without a way in, those arrive as ad-hoc ``requests.get`` calls inside strategy
code — which is exactly how a strategy ends up with no declared inputs, no
status, no PIT statement and no way to be validated before a backtest.

So they register instead::

    from cyqnt_trd.standard_bot.data import custom_sources as cs

    cs.register_rest_node(
        name="my_sentiment",
        url="http://example.internal/api/sentiment",
        params={"window": "24h"},
        records_path="data.items",
        availability="FORWARD_ONLY",
        pit_hazard="live poll; no history endpoint",
        returns_columns=("symbol", "score", "as_of"),
    )

From that moment ``data.my_sentiment(...)`` exists, ``validate_nodes`` knows its
replay discipline, ``DATA_API.md`` documents it, and a backtest that reads it
without declaring it fails the same way a built-in would.

Four ways in:

``register_rest_node``       an HTTP endpoint, described inline
``register_source_node``     an existing :class:`RestSourceSpec` + a contract
``register_callable_node``   any Python callable returning a DataFrame
``register_file_node``       a parquet / csv on disk

``bind_config_sources()`` does the bulk case: load every spec from the private
``$CYQNT_SOURCES_CONFIG`` file and register each as a catalog node, taking the
contract fields from the same config entry. That is how a deployment points the
standard bot at its internal endpoints without any of them landing in the repo.

Every registration **must** state ``availability``. There is no default: the
whole point of the catalog is that "can I replay this?" is answered before a
backtest runs, and a silent default would answer it wrong.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .catalog import (
    Availability,
    DataNodeSpec,
    DataUnavailable,
    ParamSpec,
    ReturnSpec,
    SourcePath,
    register_node,
    unregister_node,
)

__all__ = [
    "register_rest_node",
    "register_source_node",
    "register_callable_node",
    "register_file_node",
    "bind_config_sources",
    "unregister_node",
    "CUSTOM_PREFIX",
    "CONTRACT_KEYS",
]

#: keys a config entry may carry beyond the RestSourceSpec fields. These are the
#: contract half — without them a source is transport with no discipline.
CONTRACT_KEYS = (
    "availability", "pit_hazard", "emits", "column_map", "constants",
    "value_columns", "description", "strategy_types", "notes", "param_schema",
)

#: custom nodes are namespaced so a user feed can never silently shadow a
#: documented one, and so a reviewer can see at a glance which inputs are
#: first-party. Passing an already-prefixed name is accepted as-is.
CUSTOM_PREFIX = "custom_"

_CALLABLES: Dict[str, Callable[..., Any]] = {}
_REST_CONFIG: Dict[str, Dict[str, Any]] = {}
_FILE_CONFIG: Dict[str, Dict[str, Any]] = {}


def _namespaced(name: str) -> str:
    if not name or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("node name must be lower_snake_case, got %r" % name)
    if name.startswith(CUSTOM_PREFIX):
        return name
    from .catalog import list_node_names

    if name in set(list_node_names()):
        # Silently renaming to custom_klines would leave the author believing
        # they had overridden the built-in while every strategy still reads it.
        raise ValueError(
            "%r is already a catalog node. Custom nodes are namespaced %s*, so "
            "this would register %s%s and NOT override the built-in — pick a "
            "distinct name." % (name, CUSTOM_PREFIX, CUSTOM_PREFIX, name)
        )
    return CUSTOM_PREFIX + name


def _coerce_availability(value: Any) -> Availability:
    if isinstance(value, Availability):
        return value
    try:
        return Availability(str(value).upper())
    except ValueError:
        raise ValueError(
            "availability must be one of %s — state it explicitly; there is no "
            "safe default for 'can this be replayed?'"
            % ", ".join(item.value for item in Availability)
        )


def _coerce_kind(value: Any):
    from ..core.input_contract import FrameKind

    if value is None:
        return FrameKind.RAW
    if isinstance(value, FrameKind):
        return value
    try:
        return FrameKind(str(value).lower())
    except ValueError:
        raise ValueError(
            "emits must be one of %s"
            % ", ".join(item.value for item in FrameKind)
        )


def _params_from(spec: Optional[Sequence[Mapping[str, Any]]]) -> tuple:
    out = []
    for item in spec or ():
        out.append(
            ParamSpec(
                key=str(item["key"]),
                type=str(item.get("type", "str")),
                required=bool(item.get("required", False)),
                default=item.get("default"),
                description=str(item.get("description", "")),
                options=tuple(item.get("options", ()) or ()),
            )
        )
    return tuple(out)


def _register(
    *,
    name: str,
    description: str,
    availability: Any,
    source_path: SourcePath,
    endpoint: str,
    returns: ReturnSpec,
    params: tuple,
    pit_hazard: str,
    strategy_types: Sequence[str],
    fetcher: str,
    notes: str,
    replace: bool,
    emits: Any = None,
    column_map: Optional[Mapping[str, str]] = None,
    constants: Optional[Mapping[str, Any]] = None,
    value_columns: Sequence[str] = (),
) -> DataNodeSpec:
    resolved = _coerce_availability(availability)
    if resolved is not Availability.BACKTESTABLE and not pit_hazard:
        raise ValueError(
            "availability=%s requires pit_hazard: say how this field lies when "
            "replayed, or the next person will backtest on it" % resolved.value
        )
    spec = DataNodeSpec(
        name=name,
        description=description,
        strategy_types=tuple(strategy_types or ("*",)),
        source_path=source_path,
        availability=resolved,
        returns=returns,
        params=params,
        pit_hazard=pit_hazard,
        endpoint=endpoint,
        fetcher=fetcher,
        notes=notes,
        emits=_coerce_kind(emits),
        column_map=dict(column_map or {}),
        constants=dict(constants or {}),
        value_columns=tuple(value_columns or ()),
    )
    register_node(spec, replace=replace)
    _install_runtime_function(spec)
    return spec


def _install_runtime_function(spec: DataNodeSpec) -> None:
    """Expose the new node as ``data.<name>`` immediately."""
    try:
        from ..runtime import data as data_runtime
    except ImportError:  # runtime not importable in this context — catalog still works
        return
    data_runtime.install_node_function(spec)


# ---------------------------------------------------------------------------
# 1. REST
# ---------------------------------------------------------------------------


def register_rest_node(
    *,
    name: str,
    url: str,
    availability: Any,
    description: str = "",
    method: str = "GET",
    params: Optional[Mapping[str, Any]] = None,
    headers: Optional[Mapping[str, str]] = None,
    records_path: str = "",
    fields: Optional[Mapping[str, Any]] = None,
    ok_check: Optional[Mapping[str, Any]] = None,
    sort_by: Optional[str] = None,
    ttl: int = 300,
    timeout: int = 20,
    param_schema: Optional[Sequence[Mapping[str, Any]]] = None,
    returns_columns: Sequence[str] = (),
    pit_hazard: str = "",
    strategy_types: Sequence[str] = ("*",),
    emits: Any = None,
    column_map: Optional[Mapping[str, str]] = None,
    constants: Optional[Mapping[str, Any]] = None,
    value_columns: Sequence[str] = (),
    notes: str = "",
    replace: bool = False,
) -> DataNodeSpec:
    """Register a JSON HTTP endpoint as a data node.

    Transport and parsing are delegated to
    :class:`~cyqnt_trd.data_cli.rest_source.RestSourceSpec` — one HTTP client,
    one TTL cache, one coercion table shared with every other configured
    source, rather than a second fetcher living here.

    ``records_path`` is a dotted path to the row list inside the response
    (``"data.items"``); omit it when the body is already a list. ``fields`` maps
    output column -> ``{"key": ..., "coerce": "str|int|float|ms|bool|list|raw"}``;
    omit it to pass rows through as-is.
    """
    from cyqnt_trd.data_cli.rest_source import FieldSpec, RestSourceSpec, register_spec

    node = _namespaced(name)
    base, _, path = url.partition("://")
    base = "%s://%s" % (base, path.split("/", 1)[0]) if "://" in url else url
    remainder = url[len(base):].lstrip("/") if url.startswith(base) else ""

    source = RestSourceSpec(
        name=node,
        base_url=base,
        path=remainder,
        method=method.upper(),
        params=dict(params or {}),
        headers=dict(headers or {}),
        data_path=records_path,
        fields={
            column: (FieldSpec(**spec) if isinstance(spec, Mapping) else FieldSpec(*spec))
            for column, spec in (fields or {}).items()
        },
        ok_check=dict(ok_check) if ok_check else None,
        sort_by=sort_by,
        ttl=int(ttl),
        timeout=int(timeout),
    )
    register_spec(source)
    _REST_CONFIG[node] = {"spec_name": node}

    return _register(
        name=node,
        description=description or "User-registered REST feed at %s" % url,
        availability=availability,
        source_path=SourcePath.EXTERNAL_VENDOR,
        endpoint="%s %s" % (method.upper(), url),
        returns=ReturnSpec(
            "pd.DataFrame", "rows parsed from the JSON response",
            tuple(returns_columns) or tuple(source.columns),
        ),
        params=_params_from(param_schema),
        pit_hazard=pit_hazard,
        strategy_types=strategy_types,
        fetcher="cyqnt_trd.standard_bot.data.custom_sources._fetch_rest_%s" % node,
        notes=notes or "user-registered",
        replace=replace,
        emits=emits, column_map=column_map, constants=constants,
        value_columns=value_columns,
    )


def register_source_node(
    source: Any,
    *,
    availability: Any,
    pit_hazard: str = "",
    description: str = "",
    strategy_types: Sequence[str] = ("*",),
    emits: Any = None,
    column_map: Optional[Mapping[str, str]] = None,
    constants: Optional[Mapping[str, Any]] = None,
    value_columns: Sequence[str] = (),
    param_schema: Optional[Sequence[Mapping[str, Any]]] = None,
    notes: str = "",
    replace: bool = False,
) -> DataNodeSpec:
    """Wrap an existing ``RestSourceSpec`` as a catalog node.

    The bridge between the two layers: a spec built inline, shipped in
    ``public_sources``, or loaded from ``$CYQNT_SOURCES_CONFIG`` already knows
    how to fetch. This attaches the contract it is missing — how far back it
    replays, and how it lies if replayed wrong — and makes it addressable as
    ``data.custom_<name>`` like everything else.
    """
    from cyqnt_trd.data_cli.rest_source import get_spec, register_spec

    node = _namespaced(source.name)
    if get_spec(source.name) is None:
        register_spec(source)
    _REST_CONFIG[node] = {"spec_name": source.name}

    return _register(
        name=node,
        description=description or "REST source %s (%s)" % (source.name, source.base_url),
        availability=availability,
        source_path=SourcePath.EXTERNAL_VENDOR,
        endpoint="%s %s/%s" % (source.method, source.base_url.rstrip("/"), source.path),
        returns=ReturnSpec("pd.DataFrame", "rows per the source field map",
                           tuple(source.columns)),
        params=_params_from(param_schema),
        pit_hazard=pit_hazard,
        strategy_types=strategy_types,
        fetcher="cyqnt_trd.standard_bot.data.custom_sources._fetch_rest_%s" % node,
        notes=notes or "registered from a RestSourceSpec",
        replace=replace,
        emits=emits, column_map=column_map, constants=constants,
        value_columns=value_columns,
    )


def bind_config_sources(path: Optional[str] = None, *, replace: bool = False) -> List[str]:
    """Load ``$CYQNT_SOURCES_CONFIG`` and register every spec as a catalog node.

    This is how a deployment points the standard bot at its own endpoints —
    ``/bn-data`` internal hosts, a team service, a vendor trial — without any of
    them landing in the repo. Each config entry carries the transport fields
    ``RestSourceSpec`` understands plus the contract keys in
    :data:`CONTRACT_KEYS`.

    An entry with no ``availability`` is **skipped with a warning**, not
    defaulted: a source whose replay discipline nobody stated is exactly the one
    that ends up under a backtest.

    Returns the node names registered.
    """
    import json
    import os
    import warnings

    from cyqnt_trd.data_cli.rest_source import RestSourceSpec, register_spec

    config_path = path or os.environ.get("CYQNT_SOURCES_CONFIG", "").strip()
    if not config_path or not os.path.isfile(config_path):
        return []
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    entries = raw if isinstance(raw, list) else raw.get("sources", [])

    registered: List[str] = []
    for entry in entries:
        transport = {k: v for k, v in entry.items() if k not in CONTRACT_KEYS}
        try:
            source = RestSourceSpec.from_dict(transport)
        except Exception as exc:
            warnings.warn("skipping source %r: %s" % (entry.get("name"), exc), RuntimeWarning)
            continue
        register_spec(source)
        if not entry.get("availability"):
            warnings.warn(
                "source %r declares no availability, so it is fetchable but NOT "
                "registered as a data node — state availability (and pit_hazard "
                "unless BACKTESTABLE) to make it usable by a bot" % source.name,
                RuntimeWarning,
            )
            continue
        try:
            node = register_source_node(
                source,
                availability=entry["availability"],
                pit_hazard=entry.get("pit_hazard", ""),
                description=entry.get("description", ""),
                strategy_types=entry.get("strategy_types", ("*",)),
                emits=entry.get("emits"),
                column_map=entry.get("column_map"),
                constants=entry.get("constants"),
                value_columns=entry.get("value_columns", ()),
                param_schema=entry.get("param_schema"),
                notes=entry.get("notes", "from $CYQNT_SOURCES_CONFIG"),
                replace=replace,
            )
        except ValueError as exc:
            warnings.warn("skipping source %r: %s" % (source.name, exc), RuntimeWarning)
            continue
        registered.append(node.name)
    return registered


def _fetch_rest(node: str, **call_params: Any):
    from cyqnt_trd.data_cli.rest_source import fetch_rest

    config = _REST_CONFIG.get(node)
    if config is None:
        raise DataUnavailable(node, "REST source was unregistered")
    try:
        return fetch_rest(config["spec_name"], params=call_params or None)
    except KeyError as exc:
        raise DataUnavailable(node, str(exc))


# ---------------------------------------------------------------------------
# 2. callable
# ---------------------------------------------------------------------------


def register_callable_node(
    *,
    name: str,
    fn: Callable[..., Any],
    availability: Any,
    description: str = "",
    param_schema: Optional[Sequence[Mapping[str, Any]]] = None,
    returns_columns: Sequence[str] = (),
    pit_hazard: str = "",
    strategy_types: Sequence[str] = ("*",),
    emits: Any = None,
    column_map: Optional[Mapping[str, str]] = None,
    constants: Optional[Mapping[str, Any]] = None,
    value_columns: Sequence[str] = (),
    endpoint: str = "",
    notes: str = "",
    replace: bool = False,
) -> DataNodeSpec:
    """Register any Python callable returning a DataFrame as a data node.

    The escape hatch for a feed with no clean HTTP shape — a warehouse query, a
    vendor SDK, an in-house assembler. It still has to declare availability and
    a PIT hazard, so it is a first-class input rather than a hidden fetch.
    """
    if not callable(fn):
        raise TypeError("fn must be callable")
    node = _namespaced(name)
    _CALLABLES[node] = fn
    return _register(
        name=node,
        description=description or (fn.__doc__ or "").strip().splitlines()[0]
        if (fn.__doc__ or "").strip()
        else description or "User-registered callable %s" % getattr(fn, "__name__", "fn"),
        availability=availability,
        source_path=SourcePath.EXTERNAL_VENDOR,
        endpoint=endpoint or "python:%s.%s"
        % (getattr(fn, "__module__", "?"), getattr(fn, "__name__", "fn")),
        returns=ReturnSpec("pd.DataFrame", "whatever the callable returns",
                           tuple(returns_columns)),
        params=_params_from(param_schema),
        pit_hazard=pit_hazard,
        strategy_types=strategy_types,
        fetcher="cyqnt_trd.standard_bot.data.custom_sources._fetch_callable_%s" % node,
        notes=notes or "user-registered",
        replace=replace,
        emits=emits, column_map=column_map, constants=constants,
        value_columns=value_columns,
    )


def _fetch_callable(node: str, **params: Any):
    fn = _CALLABLES.get(node)
    if fn is None:
        raise DataUnavailable(node, "callable was unregistered")
    return fn(**params)


# ---------------------------------------------------------------------------
# 3. file
# ---------------------------------------------------------------------------


def register_file_node(
    *,
    name: str,
    path: str,
    availability: Any,
    description: str = "",
    format: str = "",
    time_column: str = "",
    param_schema: Optional[Sequence[Mapping[str, Any]]] = None,
    returns_columns: Sequence[str] = (),
    pit_hazard: str = "",
    strategy_types: Sequence[str] = ("*",),
    emits: Any = None,
    column_map: Optional[Mapping[str, str]] = None,
    constants: Optional[Mapping[str, Any]] = None,
    value_columns: Sequence[str] = (),
    notes: str = "",
    replace: bool = False,
) -> DataNodeSpec:
    """Register a parquet / csv file as a data node.

    ``time_column``, when given, is used to honour a ``as_of_ms`` call kwarg:
    rows at or before the cut-off only. Without it a file node cannot be
    PIT-gated, and that is reflected in the required ``pit_hazard``.
    """
    node = _namespaced(name)
    resolved_format = format or Path(path).suffix.lstrip(".").lower() or "parquet"
    if resolved_format not in ("parquet", "csv"):
        raise ValueError("format must be parquet or csv, got %r" % resolved_format)
    if not time_column and not pit_hazard:
        pit_hazard = (
            "no time_column declared, so the frame cannot be gated to a decision "
            "time — every bar of a replay sees the whole file"
        )
    _FILE_CONFIG[node] = {
        "path": str(path), "format": resolved_format, "time_column": time_column,
    }
    return _register(
        name=node,
        description=description or "User-registered %s at %s" % (resolved_format, path),
        availability=availability,
        source_path=SourcePath.LOCAL_PARQUET,
        endpoint="file://%s" % path,
        returns=ReturnSpec("pd.DataFrame", "file contents", tuple(returns_columns)),
        params=_params_from(param_schema),
        pit_hazard=pit_hazard,
        strategy_types=strategy_types,
        fetcher="cyqnt_trd.standard_bot.data.custom_sources._fetch_file_%s" % node,
        notes=notes or "user-registered",
        replace=replace,
        emits=emits, column_map=column_map, constants=constants,
        value_columns=value_columns,
    )


def _fetch_file(node: str, *, as_of_ms: Optional[int] = None, **_: Any):
    import pandas as pd

    config = _FILE_CONFIG.get(node)
    if config is None:
        raise DataUnavailable(node, "file config was unregistered")
    path = Path(config["path"])
    if not path.exists():
        raise DataUnavailable(node, "file not found: %s" % path)
    try:
        frame = (
            pd.read_parquet(path)
            if config["format"] == "parquet"
            else pd.read_csv(path)
        )
    except Exception as exc:
        raise DataUnavailable(node, "%s: %s" % (type(exc).__name__, exc))

    column = config["time_column"]
    if as_of_ms is not None and column and column in frame:
        stamps = _parse_time_column(frame[column])
        cutoff = pd.to_datetime(int(as_of_ms), unit="ms", utc=True)
        frame = frame[stamps.notna() & (stamps <= cutoff)].reset_index(drop=True)
    return frame


def _parse_time_column(values):
    """Parse a time column, treating bare integers as epoch MILLISECONDS.

    ``pd.to_datetime`` reads a bare int64 as nanoseconds, which silently turns
    every epoch-ms stamp into 1970 and makes the PIT gate a no-op that keeps
    every row.
    """
    import pandas as pd

    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        unit = "s" if float(numeric.abs().max()) < 1e11 else "ms"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(values, utc=True, errors="coerce")


# ---------------------------------------------------------------------------
# dynamic fetcher resolution
#
# The catalog stores a dotted path; these per-node shims let one generic
# implementation serve every registration without ``eval`` or a global lookup
# inside the strategy's call path.
# ---------------------------------------------------------------------------


def __getattr__(attr: str):
    for prefix, impl in (
        ("_fetch_rest_", _fetch_rest),
        ("_fetch_callable_", _fetch_callable),
        ("_fetch_file_", _fetch_file),
    ):
        if attr.startswith(prefix):
            node = attr[len(prefix):]

            def bound(_node=node, _impl=impl, **params: Any):
                return _impl(_node, **params)

            bound.__name__ = attr
            return bound
    raise AttributeError(attr)
