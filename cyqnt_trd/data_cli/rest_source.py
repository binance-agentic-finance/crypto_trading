"""Generic, configurable REST → typed DataFrame fetcher (pure stdlib).

This module ships **only the mechanism** — no hardcoded hosts, endpoints, or
field names. You describe *any* JSON REST source declaratively (base URL +
path + params + a field mapping) and get back a typed, TTL-cached
``pandas.DataFrame`` that obeys the same contract as the rest of ``data_cli``
(cache-miss / transport error → empty typed frame, never raises).

Why declarative
---------------
Different deployments point at different data services. Rather than baking any
particular endpoint into the (public) package, callers supply a
:class:`RestSourceSpec` describing the source. Specs can be:

* built inline in code,
* loaded from an external JSON/YAML config file that lives **outside** the repo
  (see :func:`load_specs` — read from ``$CYQNT_SOURCES_CONFIG``, which should
  point at a private, git-ignored file), or
* registered at runtime via :func:`register_spec`.

A fully public, backtestable worked example (``alternative.me`` Fear & Greed)
ships in :data:`EXAMPLE_SPECS` purely as documentation.

Envelope handling
-----------------
JSON responses vary. A spec declares:

* ``data_path`` — dotted path to the list of rows inside the response
  (e.g. ``"data"``, ``"data.historyList"``, or ``""`` when the top level is
  already a list).
* ``fields`` — ordered ``{output_column: FieldSpec}`` mapping. Each
  :class:`FieldSpec` names the source key and a coercion
  (``str/int/float/ms/bool/list/raw``).
* ``ok_check`` — optional ``{"key": ..., "equals": ...}`` success gate; if the
  envelope carries a status code, set this so a failure body degrades to an
  empty frame instead of being parsed as rows.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ._cache import cache_get, cache_set

__all__ = [
    "FieldSpec",
    "RestSourceSpec",
    "fetch_rest",
    "register_spec",
    "get_spec",
    "load_specs",
    "list_specs",
    "EXAMPLE_SPECS",
]


# ---------------------------------------------------------------------------
# Coercions — declarative, so specs stay plain data (JSON-serialisable)
# ---------------------------------------------------------------------------
def _to_int(v) -> int:
    try:
        return int(float(v)) if isinstance(v, str) and "." in v else int(v)
    except (TypeError, ValueError):
        return 0


def _to_float(v, default: float = float("nan")) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _to_ms(v) -> int:
    """Coerce an epoch timestamp (seconds or milliseconds) to epoch-ms."""
    n = _to_int(v)
    if 0 < n < 1_000_000_000_000:
        return n * 1000
    return n


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def _to_list(v) -> list:
    if isinstance(v, list):
        return v
    if v is None:
        return []
    return [v]


_COERCIONS: Dict[str, Callable[[Any], Any]] = {
    "str": lambda v: "" if v is None else str(v),
    "int": _to_int,
    "float": _to_float,
    "ms": _to_ms,
    "bool": _to_bool,
    "list": _to_list,
    "raw": lambda v: v,
}


@dataclass(frozen=True)
class FieldSpec:
    """One output column: which source key to read and how to coerce it.

    ``key`` supports a dotted path into nested dicts (e.g. ``"quote.USD.price"``).
    ``coerce`` is one of :data:`_COERCIONS` keys.
    """

    key: str
    coerce: str = "raw"

    def read(self, row: dict) -> Any:
        val: Any = row
        for part in self.key.split("."):
            if isinstance(val, dict):
                val = val.get(part)
            else:
                val = None
                break
        return _COERCIONS.get(self.coerce, _COERCIONS["raw"])(val)


@dataclass
class RestSourceSpec:
    """Declarative description of a JSON REST source.

    Nothing here is repo-specific: fill it in for whatever API/website you want.
    """

    name: str
    base_url: str
    path: str = ""
    method: str = "GET"                       # GET | POST
    params: Dict[str, Any] = field(default_factory=dict)   # query (GET) or JSON body (POST)
    headers: Dict[str, str] = field(default_factory=dict)
    data_path: str = ""                       # dotted path to the row list; "" = top level
    fields: Dict[str, FieldSpec] = field(default_factory=dict)
    ok_check: Optional[Dict[str, Any]] = None  # {"key": "code", "equals": "0"} style gate
    sort_by: Optional[str] = None             # output column to sort ascending
    ttl: int = 300
    timeout: int = 20

    @property
    def columns(self) -> List[str]:
        return list(self.fields.keys())

    # -- construction from plain dict (config-file friendly) ----------------
    @classmethod
    def from_dict(cls, d: dict) -> "RestSourceSpec":
        fields = {
            col: (FieldSpec(**fs) if isinstance(fs, dict) else FieldSpec(*fs))
            for col, fs in (d.get("fields") or {}).items()
        }
        return cls(
            name=d["name"],
            base_url=d["base_url"],
            path=d.get("path", ""),
            method=d.get("method", "GET").upper(),
            params=dict(d.get("params") or {}),
            headers=dict(d.get("headers") or {}),
            data_path=d.get("data_path", ""),
            fields=fields,
            ok_check=d.get("ok_check"),
            sort_by=d.get("sort_by"),
            ttl=int(d.get("ttl", 300)),
            timeout=int(d.get("timeout", 20)),
        )


# ---------------------------------------------------------------------------
# Spec registry — inline, runtime, or external-config sourced
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, RestSourceSpec] = {}


def register_spec(spec: RestSourceSpec) -> None:
    """Register (or replace) a spec by ``spec.name``."""
    _REGISTRY[spec.name] = spec


def get_spec(name: str) -> Optional[RestSourceSpec]:
    return _REGISTRY.get(name)


def list_specs() -> List[str]:
    return sorted(_REGISTRY)


def load_specs(path: Optional[str] = None) -> List[str]:
    """Load specs from an external JSON config file and register them.

    *path* defaults to the ``CYQNT_SOURCES_CONFIG`` environment variable. Keep
    this file **outside the repo** (git-ignored / private) so deployment-specific
    endpoints never land in version control. The file is a JSON list of spec
    dicts (see :meth:`RestSourceSpec.from_dict`). Returns the names registered.
    """
    import os

    cfg = path or os.environ.get("CYQNT_SOURCES_CONFIG", "").strip()
    if not cfg or not os.path.isfile(cfg):
        return []
    with open(cfg, "r", encoding="utf-8") as f:
        raw = json.load(f)
    specs = raw if isinstance(raw, list) else raw.get("sources", [])
    names = []
    for d in specs:
        spec = RestSourceSpec.from_dict(d)
        register_spec(spec)
        names.append(spec.name)
    return names


# ---------------------------------------------------------------------------
# Transport + parse
# ---------------------------------------------------------------------------
def _dig(obj: Any, dotted: str) -> Any:
    if not dotted:
        return obj
    for part in dotted.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return None
    return obj


def _is_ok(resp: Any, ok_check: Optional[Dict[str, Any]]) -> bool:
    if ok_check is None:
        return resp is not None
    if not isinstance(resp, dict):
        return False
    got = _dig(resp, ok_check.get("key", ""))
    return str(got) == str(ok_check.get("equals"))


def _request(spec: RestSourceSpec) -> Any:
    url = spec.base_url.rstrip("/")
    if spec.path:
        url = f"{url}/{spec.path.lstrip('/')}"
    headers = {"Accept": "application/json", **spec.headers}
    try:
        if spec.method == "POST":
            body = json.dumps(spec.params).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        else:
            if spec.params:
                url = f"{url}?{urllib.parse.urlencode(spec.params)}"
            req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=spec.timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError):
        return None


def _empty(spec: RestSourceSpec) -> pd.DataFrame:
    return pd.DataFrame(columns=spec.columns)


def fetch_rest(
    source: "str | RestSourceSpec",
    *,
    params: Optional[Dict[str, Any]] = None,
    ttl: Optional[int] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch *source* and return a typed, cached DataFrame per its field spec.

    *source* is either a registered spec name or a :class:`RestSourceSpec`.
    *params* is merged over the spec's params (per-call overrides, e.g. a date
    range or symbol). On cache-miss / transport error / failed ``ok_check`` an
    **empty typed frame** is returned — never raises.
    """
    spec = get_spec(source) if isinstance(source, str) else source
    if spec is None:
        raise KeyError(f"unknown REST source spec: {source!r} (register_spec / load_specs first)")

    if params:
        merged = dict(spec.params)
        merged.update(params)
        spec = RestSourceSpec(**{**spec.__dict__, "params": merged})

    ttl_sec = ttl if ttl is not None else spec.ttl
    key = ("rest", spec.name, tuple(sorted(spec.params.items())))
    if not refresh:
        cached = cache_get(key)
        if cached is not None:
            return cached

    resp = _request(spec)
    if not _is_ok(resp, spec.ok_check):
        return _empty(spec)

    rows_raw = _dig(resp, spec.data_path)
    if not isinstance(rows_raw, list):
        rows_raw = [rows_raw] if isinstance(rows_raw, dict) else []

    rows = []
    for r in rows_raw:
        if not isinstance(r, dict):
            continue
        rows.append({col: fs.read(r) for col, fs in spec.fields.items()})
    if not rows:
        return _empty(spec)

    df = pd.DataFrame(rows, columns=spec.columns)
    if spec.sort_by and spec.sort_by in df.columns:
        df = df.sort_values(spec.sort_by).reset_index(drop=True)
    cache_set(key, df, ttl=ttl_sec)
    return df


# ---------------------------------------------------------------------------
# Public worked example (documentation only — fully reachable & backtestable)
# ---------------------------------------------------------------------------
#: Fear & Greed index from the public ``alternative.me`` API. Shows the exact
#: shape of a spec; register with ``register_spec(EXAMPLE_SPECS["fear_greed"])``
#: then ``fetch_rest("fear_greed", params={"limit": 0})``.
EXAMPLE_SPECS: Dict[str, RestSourceSpec] = {
    "fear_greed": RestSourceSpec(
        name="fear_greed",
        base_url="https://api.alternative.me",
        path="fng/",
        method="GET",
        params={"limit": 30, "format": "json"},
        data_path="data",
        fields={
            "timestamp": FieldSpec("timestamp", "ms"),
            "value": FieldSpec("value", "int"),
            "value_classification": FieldSpec("value_classification", "str"),
        },
        sort_by="timestamp",
        ttl=3600,
    ),
}
