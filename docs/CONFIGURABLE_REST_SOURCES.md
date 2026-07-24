# Configurable REST data sources (`data_cli.rest_source`)

A generic, dependency-free way to turn **any JSON REST API or website** into a
typed, TTL-cached `pandas.DataFrame` that follows the same contract as the rest
of `data_cli` (cache-miss / transport error → empty typed frame, never raises).

The package ships **only the mechanism** — no endpoints, hosts, or field names
are hardcoded. You describe a source declaratively and point it wherever you
like: a public API, your own service, or a deployment-specific gateway supplied
through a private config file that lives outside this repo.

> **Do not commit deployment-specific / internal endpoints to this public
> repo.** Put them in a private config file and load it via
> `$CYQNT_SOURCES_CONFIG` (see §3). The patterns `sources*.json`,
> `*.sources.json`, and `data_sources/` are git-ignored for exactly this.

---

## 1. Describe a source

A source is a `RestSourceSpec`: base URL + path + params + a `{column:
FieldSpec}` mapping. Each `FieldSpec` names the source key (dotted paths allowed,
e.g. `quote.USD.price`) and a coercion (`str/int/float/ms/bool/list/raw`;
`ms` normalises epoch-s or epoch-ms to epoch-ms).

```python
from cyqnt_trd.data_cli import RestSourceSpec, FieldSpec, fetch_rest

spec = RestSourceSpec(
    name="my_series",
    base_url="https://api.example.com",   # any API/website you want
    path="v1/history",
    method="GET",                          # or "POST" (params become the JSON body)
    params={"limit": 100},
    data_path="data",                      # dotted path to the row list ("" = top-level list)
    fields={
        "timestamp": FieldSpec("ts", "ms"),
        "value":     FieldSpec("val", "float"),
        "label":     FieldSpec("classification", "str"),
    },
    ok_check={"key": "code", "equals": "0"},  # optional success gate
    sort_by="timestamp",
    ttl=3600,
)

df = fetch_rest(spec)                                   # typed, cached DataFrame
df = fetch_rest(spec, params={"limit": 500}, refresh=True)  # per-call param override
```

## 2. Register once, fetch by name

```python
from cyqnt_trd.data_cli import register_spec, fetch_rest, list_specs

register_spec(spec)
list_specs()                 # ['my_series', ...]
df = fetch_rest("my_series")
```

## 3. Load endpoints from a private, git-ignored config

Keep any deployment-specific endpoints **out of the repo**. Write them to a
JSON file (a list of spec dicts) and load at startup:

```python
import os
from cyqnt_trd.data_cli import load_specs, fetch_rest

os.environ["CYQNT_SOURCES_CONFIG"] = "/private/path/sources.json"  # NOT in the repo
load_specs()                 # registers everything in that file
df = fetch_rest("whatever_you_named_it")
```

`sources.json` shape:

```json
[
  {
    "name": "my_series",
    "base_url": "https://api.example.com",
    "path": "v1/history",
    "method": "GET",
    "params": {"limit": 100},
    "data_path": "data",
    "fields": {
      "timestamp": {"key": "ts", "coerce": "ms"},
      "value":     {"key": "val", "coerce": "float"}
    },
    "sort_by": "timestamp",
    "ttl": 3600
  }
]
```

## 4. Public worked example (backtestable, ships in the repo)

`EXAMPLE_SPECS["fear_greed"]` targets the public `alternative.me` Fear & Greed
API — a complete, reachable example of the spec shape:

```python
from cyqnt_trd.data_cli import register_spec, fetch_rest, EXAMPLE_SPECS

register_spec(EXAMPLE_SPECS["fear_greed"])
df = fetch_rest("fear_greed", params={"limit": 0})   # 0 = full history
# columns: timestamp (ms), value (0-100), value_classification
```

## 5. Contract

- Signature: `fetch_rest(source, *, params=None, ttl=None, refresh=False) -> DataFrame`
- Cache-miss / transport error / failed `ok_check` → **empty typed frame** (never raises).
- Empty frames are not cached, so a later successful call still populates the cache.
- Timestamps coerced with `ms` are always epoch-ms.
- Pure standard library — adds no runtime dependency.

Tests: `tests/test_rest_source.py` (field mapping, coercions, dotted keys,
`ok_check` gating, sorting, per-call override, external-config loading,
degradation).
