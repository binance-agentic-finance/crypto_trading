"""Capability manifests -> factors the evaluation matrix can score.

The capability SDK (`be/binance-ai-platform`, `capability-sdk/src/binance/strategy/
node/capabilities`) declares every operator with a **manifest v3**:

    params   call arguments. `user_specify: false` ones are data ports -- either a
             `list[float]` series or a `dict[str, Any] | list[Any]` bar list
             (conventionally named `rows`). `user_specify: true` ones are knobs
             a human fills in (`period`, `q`).
    outputs  named ports. `float` is "the value at this bar", `list[float | None]`
             is the whole series.

`factor_eval.capability` already converts the *shape* (per-symbol operator ->
date x symbol panel). This module adds the two things still missing:

1. **Manifest-driven wiring.** Today `inputs`/`rows`/`output`/`window` have to be
   hand-written per node; the manifest already states all of it.
2. **Scale.** Cross-sectional ranking needs values that are comparable *across*
   symbols. `rsi` is 0-100 and comparable; `supertrend`, `pivot_points` and `sma`
   emit price levels, so ranking BTC's 110000 against DOGE's 0.2 just sorts by coin
   price. Each capability therefore yields two candidates, `raw` and `ts_zscore`
   (per-symbol rolling standardisation). Which one carries information is decided
   by the matrix, not guessed here -- the cost is twice the candidates, reported
   honestly through `trials_seen`.

Two execution paths, and the difference matters:

`upstream` (default)
    Imports the block named by the manifest's `capability.source` (e.g.
    `cyqnt_trd.blocks.indicators.cci`) and calls it **vectorised**, once per symbol.
    Runs offline because this repo ships `cyqnt_trd`. It executes what the
    capability *declares as its source*, not the capability's own `impl.py`.

`impl`
    Loads the real `impl.py` from a local capability-SDK checkout through
    `factor_eval.capability.capability_factor`. That is what production runs.
    `runtime_shim()` stubs the SDK runtime so the import succeeds offline.

The two are not guaranteed to agree -- upstream signatures and manifest parameter
names have already drifted (`macd` declares `fast_period`, upstream takes `fast`).
Unmapped parameters raise instead of being dropped: silently ignoring `period=20`
and letting upstream default to 14 produces a plausible-looking factor that is not
the one that was asked for. Use `conformance()` to check the paths against each other.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .capability import BAR_FIELDS, capability_factor

__all__ = ["CapabilitySpec", "CapabilityIndex", "load_index", "DEFAULT_INDEX",
           "resolve_upstream", "upstream_factor", "capability_factors",
           "ts_zscore", "runtime_shim", "load_impl", "conformance", "NORMALIZERS"]

HERE = Path(__file__).resolve().parent
DEFAULT_INDEX = HERE / "data" / "capabilities_v3.json"

#: Manifest parameter name -> upstream block parameter name. Verified by reading
#: both signatures, not inferred. Anything not here and not accepted raises.
PARAM_ALIASES = {
    "fast_period": "fast", "slow_period": "slow", "signal_period": "signal",
    "stddev_multiplier": "std_mult", "std_multiplier": "std_mult",
}

#: Manifest params the upstream block does not take, together with the value for
#: which dropping them is a no-op. `rsi` declares wilder/simple; the upstream
#: implements Wilder only, so `method="wilder"` may be dropped and anything else
#: must not be. Each entry is a deliberate, checked exception to the raise-on-
#: unmapped rule.
PARAM_DROPS = {
    ("rsi", "method"): ("wilder",),
}

#: Upstream parameter names that take a bar DataFrame.
_DF_PARAMS = ("df", "rows", "bars", "ohlcv")


@dataclass(frozen=True)
class CapabilitySpec:
    """The part of a manifest v3 that determines how to run the capability."""

    id: str
    name: str
    group: str
    description: str
    function_name: str
    package_prefix: str
    declared_source: str | None
    upstream: str | None                  # locally resolvable implementation, or None
    series_inputs: tuple[str, ...]        # list[float] data ports
    rows_input: str | None                # bar-list data port
    user_params: dict[str, Any]           # human-supplied params -> manifest default
    value_outputs: tuple[str, ...]        # float ports
    series_outputs: tuple[str, ...]       # list[float | None] ports

    @property
    def runnable(self) -> bool:
        return self.upstream is not None

    @property
    def ports(self) -> tuple[str, ...]:
        """Distinct numeric ports.

        A series port is the twin of a scalar port -- the same output read two
        ways -- so only the scalar side is listed. Manifests use two naming
        conventions for the twin (`adx`/`adx_series`, `value`/`series`).
        """
        scalars = set(self.value_outputs)
        twins = {f"{s}_series" for s in scalars}
        if "value" in scalars:
            twins.add("series")
        extra = tuple(p for p in self.series_outputs if p not in scalars and p not in twins)
        return tuple(self.value_outputs) + extra

    def default_params(self) -> dict:
        return {k: v for k, v in self.user_params.items() if v is not None}

    def required_params(self) -> tuple[str, ...]:
        """Params the manifest leaves null, i.e. the caller must supply them."""
        return tuple(k for k, v in self.user_params.items() if v is None)


@dataclass(frozen=True)
class CapabilityIndex:
    """A snapshot of the capability manifests, with the commit it came from."""

    specs: dict[str, CapabilitySpec]
    source: dict = field(default_factory=dict)
    schema: str = ""

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, name: str) -> CapabilitySpec:
        return self.specs[name]

    def __iter__(self):
        return iter(self.specs.values())

    def runnable(self) -> dict[str, CapabilitySpec]:
        return {k: v for k, v in self.specs.items() if v.runnable}

    def describe(self) -> str:
        src = self.source
        return (f"{len(self.specs)} capabilities ({len(self.runnable())} runnable locally) "
                f"from {src.get('repo', '?')}@{str(src.get('commit', '?'))[:8]} "
                f"[{src.get('branch', '?')}]")


def load_index(path: str | Path | None = None) -> CapabilityIndex:
    """Load the capability manifest snapshot.

    The snapshot records contracts and the commit they came from, not copies of
    the implementations -- those stay upstream.
    """
    path = Path(path) if path is not None else DEFAULT_INDEX
    if not path.exists():
        raise FileNotFoundError(f"capability index not found: {path}")
    raw = json.loads(path.read_text())
    if raw.get("schema") != "factor-eval.capability-index/v1":
        raise ValueError(f"unexpected capability index schema: {raw.get('schema')!r}")
    specs = {}
    for c in raw["capabilities"]:
        specs[c["name"]] = CapabilitySpec(
            id=c["id"], name=c["name"], group=c.get("group", ""),
            description=c.get("description", ""),
            function_name=c.get("function_name", c["name"]),
            package_prefix=c.get("package_prefix", ""),
            declared_source=c.get("declared_source"), upstream=c.get("upstream"),
            series_inputs=tuple(c.get("series_inputs") or ()),
            rows_input=c.get("rows_input"),
            user_params=dict(c.get("user_params") or {}),
            value_outputs=tuple(c.get("value_outputs") or ()),
            series_outputs=tuple(c.get("series_outputs") or ()),
        )
    return CapabilityIndex(specs=specs, source=raw.get("source", {}), schema=raw["schema"])


# ---------------------------------------------------------------- normalisation
def ts_zscore(frame: pd.DataFrame, window: int = 90,
              min_periods: int | None = None) -> pd.DataFrame:
    """Per-symbol rolling standardisation, lagged one day.

    Not the same thing as `preprocess.zscore`, which standardises across the
    cross-section and is a monotone transform the matrix's rank scoring absorbs.
    This one standardises a symbol against its own history, which turns a price
    level into "where it sits in its own range" and therefore does change the
    cross-sectional ordering.

    The lag keeps the decision-day value out of its own scaling constants.
    """
    floor = max(20, window // 3) if min_periods is None else int(min_periods)
    mean = frame.rolling(window, min_periods=floor).mean().shift(1)
    std = frame.rolling(window, min_periods=floor).std().shift(1)
    return (frame - mean) / std.replace(0, np.nan)


NORMALIZERS: dict[str, Callable[..., pd.DataFrame]] = {
    "raw": lambda f, **_: f,
    "ts_zscore": ts_zscore,
}


# ------------------------------------------------------------------- resolution
def _ensure_upstream_importable(module: str) -> None:
    """Put the repo root on `sys.path` so upstream blocks import.

    `python -m factor_eval` only adds `eval/`, so `cyqnt_trd` is not visible.
    Same bootstrap `__main__.py` performs for `eval/` itself.
    """
    top = module.split(".", 1)[0]
    if importlib.util.find_spec(top) is not None:
        return
    for parent in HERE.parents:
        if (parent / top / "__init__.py").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return


def resolve_upstream(spec: CapabilitySpec) -> Callable:
    if not spec.upstream:
        raise LookupError(f"capability {spec.name!r} has no locally resolvable upstream "
                          f"(declared source: {spec.declared_source!r}); use load_impl()")
    module, fn = spec.upstream.rsplit(".", 1)
    _ensure_upstream_importable(module)
    target = getattr(importlib.import_module(module), fn, None)
    if not callable(target):
        raise LookupError(f"{spec.upstream} is not callable")
    return target


def _bind_params(fn: Callable, spec: CapabilitySpec, params: Mapping[str, Any],
                 allow_unmapped: bool) -> dict:
    """Map manifest params onto the upstream signature; raise on anything left."""
    accepted = set(inspect.signature(fn).parameters)
    out, unmapped = {}, []
    for key, value in params.items():
        name = key if key in accepted else PARAM_ALIASES.get(key)
        if name in accepted:
            out[name] = value
        elif value in PARAM_DROPS.get((spec.name, key), ()):
            continue                      # documented no-op, see PARAM_DROPS
        else:
            unmapped.append(f"{key}={value!r}")
    if unmapped and not allow_unmapped:
        raise ValueError(
            f"capability {spec.name!r}: manifest params {sorted(unmapped)} have no "
            f"counterpart in {spec.upstream} {sorted(accepted)}. Dropping them would "
            f"silently fall back to upstream defaults and score a different factor.")
    return out


def _pick(result: Any, spec: CapabilitySpec, port: str | None, index) -> pd.Series:
    """Select one output port from the upstream return value.

    Upstream returns a Series (single port), a tuple (multiple ports, matched
    positionally against the manifest's `value_outputs`), or a DataFrame (by
    column). A positional match with mismatched lengths raises -- `bollinger`
    declares five ports but upstream returns three, and guessing there picks the
    wrong series.
    """
    if isinstance(result, pd.DataFrame):
        if port is None:
            raise ValueError(f"{spec.name}: upstream returned a DataFrame; pass port=")
        if port not in result.columns:
            raise KeyError(f"{spec.name}: no column {port!r} in {list(result.columns)}")
        series = result[port]
    elif isinstance(result, tuple):
        names = spec.value_outputs
        if len(names) != len(result):
            raise ValueError(f"{spec.name}: upstream returned {len(result)} outputs, manifest "
                             f"declares {len(names)} scalar ports {list(names)}; cannot match "
                             f"positionally")
        series = result[0] if port is None else result[names.index(port)]
        if port is not None and port not in names:
            raise KeyError(f"{spec.name}: no port {port!r} in {list(names)}")
    else:
        series = result
    if not isinstance(series, (pd.Series, np.ndarray, list)):
        raise TypeError(f"{spec.name}: upstream returned {type(series).__name__}")
    series = series if isinstance(series, pd.Series) else pd.Series(series)
    if len(series) != len(index):
        raise ValueError(f"{spec.name}: upstream returned {len(series)} values "
                         f"for {len(index)} bars")
    return pd.Series(series.to_numpy(dtype=float), index=index)


def upstream_factor(spec: CapabilitySpec, *,
                    port: str | None = None,
                    params: Mapping[str, Any] | None = None,
                    inputs: Mapping[str, str] | None = None,
                    bar_fields: Sequence[str] = BAR_FIELDS,
                    normalize: str = "raw",
                    zscore_window: int = 90,
                    allow_unmapped: bool = False,
                    skip_failures: bool = False,
                    errors: list | None = None):
    """Manifest + upstream block -> a `Panel -> DataFrame` factor, vectorised.

    `inputs` overrides the data-port to panel-field mapping; by default a series
    port takes the panel field of the same name (`series` means `close`) and a bar
    port becomes an OHLCV DataFrame. A capability that declares OHLC series ports
    while upstream takes a single DataFrame (`atr`) is wired to the DataFrame.

    `skip_failures` records the first error in `errors` rather than discarding it,
    so an all-NaN result can still be explained.
    """
    fn = resolve_upstream(spec)
    params = dict(spec.default_params() | dict(params or {}))
    missing = [k for k in spec.required_params() if k not in params]
    if missing:
        raise ValueError(f"capability {spec.name!r} requires params {missing}")
    bound = _bind_params(fn, spec, params, allow_unmapped)
    if normalize not in NORMALIZERS:
        raise ValueError(f"normalize must be one of {sorted(NORMALIZERS)}")
    inputs = dict(inputs or {})
    accepted = set(inspect.signature(fn).parameters)
    df_param = next((p for p in _DF_PARAMS if p in accepted), None)
    # Manifest declares separate OHLC series, upstream wants one frame.
    series_as_frame = bool(
        spec.series_inputs and df_param is not None
        and not any(n in accepted for n in spec.series_inputs)
        and set(spec.series_inputs) <= set(BAR_FIELDS))
    needs_frame = bool(spec.rows_input) or series_as_frame
    if needs_frame and df_param is None:
        raise ValueError(f"{spec.name}: manifest declares a bar input but upstream "
                         f"{sorted(accepted)} takes no DataFrame")

    def _field_for(name: str) -> str:
        return inputs.get(name, "close" if name == "series" else name)

    def factor(panel):
        index = panel.index
        out = {}
        for symbol in panel.symbols:
            kwargs = dict(bound)
            if needs_frame:
                kwargs[df_param] = pd.DataFrame(
                    {f: getattr(panel, f)[symbol].to_numpy(dtype=float) for f in bar_fields},
                    index=index)
            if not series_as_frame:
                for name in spec.series_inputs:
                    target = name if name in accepted else (
                        "series" if "series" in accepted else name)
                    if target not in accepted:
                        raise ValueError(f"{spec.name}: upstream has no series parameter "
                                         f"{target!r}")
                    kwargs[target] = getattr(panel, _field_for(name))[symbol]
            try:
                out[symbol] = _pick(fn(**kwargs), spec, port, index)
            except Exception as exc:  # noqa: BLE001
                if not skip_failures:
                    raise RuntimeError(f"capability {spec.name} failed on {symbol}: "
                                       f"{type(exc).__name__}: {exc}") from exc
                if errors is not None:
                    errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
                out[symbol] = pd.Series(np.nan, index=index)
        frame = pd.DataFrame(out).reindex(columns=panel.symbols)
        if normalize != "raw":
            frame = NORMALIZERS[normalize](frame, window=zscore_window)
        return frame

    label = f"{spec.name}.{port}" if port else spec.name
    factor.__name__ = f"{label}[{normalize}]"
    factor.__doc__ = (f"capability {spec.id} via declared source {spec.upstream} "
                      f"(port={port or 'default'}, normalize={normalize}, params={params})")
    factor.capability = spec
    factor.port = port
    factor.normalize = normalize
    factor.params = params
    return factor


def _trim(panel, bars: int):
    """Last `bars` rows of a panel, for smoke validation only."""
    from .bundle import Panel, PRICE_FIELDS
    n = min(int(bars), len(panel.index))
    cut = {f: getattr(panel, f).iloc[-n:] for f in (*PRICE_FIELDS, "funding", "mask")}
    return Panel(**cut, meta=dict(panel.meta))


def capability_factors(index: CapabilityIndex | None = None, *,
                       panel=None,
                       validate_bars: int = 400,
                       names: Iterable[str] | None = None,
                       ports: bool = False,
                       normalize: Sequence[str] = ("raw", "ts_zscore"),
                       params: Mapping[str, Mapping[str, Any]] | None = None,
                       zscore_window: int = 90,
                       skip_unrunnable: bool = True,
                       rejected: dict | None = None) -> dict:
    """Turn every runnable capability in the index into a panel factor.

    `ports=False` keeps only the first scalar port per capability; the other ports
    of one capability are near-duplicates (`adx`'s adx/plus_di/minus_di) that only
    inflate the candidate count before de-duplication removes them again.

    `panel` enables a smoke run on the last `validate_bars` rows. A manifest `rows`
    port is not always market data -- composition nodes such as `additive_combine`
    take *other signals* and that is invisible in the signature. Candidates that
    cannot run belong in `rejected` with a reason, not in the returned dict.
    """
    index = index if index is not None else load_index()
    params = dict(params or {})
    chosen = list(names) if names is not None else list(index.specs)
    rejected = rejected if rejected is not None else {}
    probe = _trim(panel, validate_bars) if panel is not None else None
    out = {}
    for name in chosen:
        spec = index[name]
        if not spec.runnable:
            if not skip_unrunnable:
                raise LookupError(f"capability {name!r} is not runnable locally")
            rejected[name] = f"no local upstream (declared source: {spec.declared_source})"
            continue
        if spec.required_params() and name not in params:
            rejected[name] = f"manifest requires params {list(spec.required_params())}"
            continue
        for port in (spec.ports if ports else spec.ports[:1]) or (None,):
            for norm in normalize:
                key = f"{name}.{port}[{norm}]" if port else f"{name}[{norm}]"
                errors: list[str] = []
                try:
                    built = upstream_factor(spec, port=port, params=params.get(name),
                                            normalize=norm, zscore_window=zscore_window,
                                            skip_failures=True, errors=errors)
                    if probe is not None:
                        frame = built(probe)
                        if not np.isfinite(frame.where(probe.mask).to_numpy(dtype=float)).any():
                            rejected[key] = errors[0] if errors else (
                                "empty on smoke run (warmup longer than the probe window)")
                            continue
                    out[key] = built
                except Exception as exc:  # noqa: BLE001
                    rejected[key] = f"{type(exc).__name__}: {exc}"
    return out


# ------------------------------------------------------- the real impl (needs SDK)
def runtime_shim() -> None:
    """Stub the SDK runtime so a capability `impl.py` imports offline.

    `impl.py` imports `binance.strategy.runtime.ctx` for logging and
    `binance.strategy.node.tools.blocks_adapter` for series conversion. Only the
    logging and conversion helpers are stubbed; the computation still comes from
    `impl.py` itself. A real installed SDK is left alone.
    """
    if "binance.strategy.runtime" in sys.modules:
        return
    try:
        importlib.import_module("binance.strategy.runtime")
        return
    except ImportError:
        pass

    def _mk(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__path__ = []
        sys.modules[name] = mod
        return mod

    for name in ("binance", "binance.strategy", "binance.strategy.node",
                 "binance.strategy.node.tools"):
        if name not in sys.modules:
            _mk(name)

    class _Ctx:
        @staticmethod
        def log(*_args, **_kwargs):
            return None
        state: dict = {}

    _mk("binance.strategy.runtime").ctx = _Ctx()
    adapter = _mk("binance.strategy.node.tools.blocks_adapter")

    def series_from(value, field="series"):
        if isinstance(value, Mapping):
            value = value.get(field)
        return pd.Series([np.nan if v is None else float(v) for v in value], dtype=float)

    def last_valid(computed):
        s = pd.Series(computed).dropna()
        return None if s.empty else float(s.iloc[-1])

    def to_json_list(computed):
        return [None if not np.isfinite(v) else float(v)
                for v in pd.Series(computed).to_numpy(dtype=float)]

    adapter.series_from = series_from
    adapter.last_valid = last_valid
    adapter.to_json_list = to_json_list


def load_impl(spec: CapabilitySpec, sdk_root: str | Path) -> Callable:
    """Load the capability's own `impl.py` from a local SDK checkout.

    `sdk_root` points at `capability-sdk/src/binance/strategy/node/capabilities`.
    """
    runtime_shim()
    path = Path(sdk_root) / spec.group / spec.name / "impl.py"
    if not path.exists():
        raise FileNotFoundError(f"capability impl not found: {path}")
    mod_spec = importlib.util.spec_from_file_location(f"_cap_{spec.name}", path)
    module = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(module)
    fn = getattr(module, spec.function_name, None)
    if not callable(fn):
        raise LookupError(f"{path} has no callable {spec.function_name!r}")
    return fn


def conformance(spec: CapabilitySpec, panel, sdk_root: str | Path, *,
                port: str | None = None, params: Mapping[str, Any] | None = None,
                rtol: float = 1e-6) -> dict:
    """Check that both execution paths produce the same factor panel.

    A disagreement means the matrix is scoring something other than what
    production runs, which matters more than any backtest number.
    """
    up = upstream_factor(spec, port=port, params=params)(panel)
    impl_fn = load_impl(spec, sdk_root)
    series_port = port if port in spec.series_outputs else (
        f"{port}_series" if port and f"{port}_series" in spec.series_outputs else None)
    down = capability_factor(
        impl_fn,
        inputs={n: ("close" if n == "series" else n) for n in spec.series_inputs},
        rows=spec.rows_input,
        params=dict(spec.default_params() | dict(params or {})),
        output=series_port or port,
        skip_failures=False)(panel)
    both = up.notna() & down.notna()
    diff = (up - down).abs().where(both)
    scale = up.abs().where(both).clip(lower=1e-12)
    worst = float((diff / scale).max().max()) if both.to_numpy().any() else np.nan
    return {"capability": spec.id, "port": port,
            "compared_cells": int(both.to_numpy().sum()),
            "upstream_only": int((up.notna() & ~down.notna()).to_numpy().sum()),
            "impl_only": int((~up.notna() & down.notna()).to_numpy().sum()),
            "max_rel_diff": worst,
            "agrees": bool(np.isfinite(worst) and worst <= rtol)}
