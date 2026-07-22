"""Load, validate, and register a YAML strategy spec.

The validator does two things:

1. **Static checks** — required keys present, every referenced block resolves
   to a real whitelisted callable, entry declares at least one direction, and
   live mode carries its safety guards.
2. **Dry-run** — compiles the spec into ``make_signals`` and executes it on a
   synthetic OHLCV frame. This is the important one: it catches the *exact*
   class of signature / type / arity bugs (wrong arg count, tuple-vs-series
   mixups, unknown params) at ``validate`` time, before a single real order.

``validate`` therefore gives the frontend / bdp-ai-trading-bot a hard gate:
a spec that validates is structurally runnable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from .interpreter import (
    SpecError,
    build_make_signals,
    eval_indicator,
    eval_node,
    resolve_block,
)

VALID_MODES = {"backtest", "paper", "live"}
VALID_EXIT_TYPES = {
    "time_only",
    "pct_stop_tp",
    "atr_stop_tp",
    "atr_trailing_stop",
    "ma_cross_exit",
    "opposite_signal",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_spec(path: str) -> Dict[str, Any]:
    import yaml

    text = Path(path).read_text(encoding="utf-8")
    spec = yaml.safe_load(text)
    if not isinstance(spec, dict):
        raise SpecError(f"{path}: top-level YAML must be a mapping")
    return spec


# ---------------------------------------------------------------------------
# Synthetic data for the dry-run
# ---------------------------------------------------------------------------


def _collect_period_hints(obj: Any, acc: List[int]) -> None:
    """Walk the spec collecting int values of period/window/lookback params."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (int, float)) and any(
                tok in str(key).lower() for tok in ("period", "window", "lookback", "bars")
            ):
                try:
                    acc.append(int(value))
                except (TypeError, ValueError):
                    pass
            else:
                _collect_period_hints(value, acc)
    elif isinstance(obj, list):
        for item in obj:
            _collect_period_hints(item, acc)


def _synthetic_df(spec: Dict[str, Any]):
    import numpy as np
    import pandas as pd

    hints: List[int] = []
    _collect_period_hints(spec, hints)
    max_period = max(hints) if hints else 50
    n = max(300, max_period * 3 + 60)

    # Deterministic gentle wave + drift so crossover/threshold conditions
    # actually trigger during the dry-run (exercises the real branches).
    idx = np.arange(n)
    base = 100.0 + idx * 0.05 + 8.0 * np.sin(idx / 15.0) + 3.0 * np.sin(idx / 4.0)
    close = base
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    volume = 1000.0 + 50.0 * np.abs(np.sin(idx / 7.0))
    step_ms = 3_600_000  # 1h bars; only relative spacing matters
    open_time = 1_700_000_000_000 + idx * step_ms
    close_time = open_time + step_ms - 1

    df = pd.DataFrame(
        {
            "open_time": open_time,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "close_time": close_time,
            "trades": (volume / 10).astype(int),
        }
    )
    df["timestamp"] = df["close_time"]

    # Stand-in HTF columns so conditions referencing them don't KeyError.
    for htf in (spec.get("data", {}) or {}).get("htf", []) or []:
        period = int(htf.get("sma_period", 200))
        tf = htf.get("interval", "4h")
        col = f"_htf_{tf}_sma_{period}"
        df[col] = df["close"].rolling(window=min(period, n), min_periods=1).mean()
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_spec(spec: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return ``(errors, warnings)``. Empty ``errors`` ⇒ spec is runnable."""
    errors: List[str] = []
    warnings: List[str] = []

    def err(msg: str) -> None:
        errors.append(msg)

    # ---- structural ----
    if spec.get("target") not in (None, "standard_bot"):
        warnings.append(f"target={spec.get('target')!r}; this pipeline targets standard_bot")

    strategy = spec.get("strategy") or {}
    if not strategy.get("id"):
        err("strategy.id is required")

    run = spec.get("run") or {}
    mode = run.get("mode")
    if mode not in VALID_MODES:
        err(f"run.mode must be one of {sorted(VALID_MODES)}, got {mode!r}")

    data = spec.get("data") or {}
    if not data.get("symbol"):
        err("data.symbol is required")
    if not (data.get("primary") or {}).get("interval"):
        err("data.primary.interval is required")

    signals = spec.get("signals") or {}
    entry = signals.get("entry") or {}
    if not entry.get("long") and not entry.get("short"):
        err("signals.entry must define at least one of long / short")

    # ---- exit / risk ----
    exit_cfg = (spec.get("risk") or {}).get("exit")
    if exit_cfg is not None:
        etype = exit_cfg.get("type") if isinstance(exit_cfg, dict) else None
        if etype not in VALID_EXIT_TYPES:
            err(f"risk.exit.type must be one of {sorted(VALID_EXIT_TYPES)}, got {etype!r}")

    # ---- sizing ----
    size = (spec.get("sizing") or {}).get("size", 1.0)
    try:
        if not (0.0 < float(size) <= 1.0):
            err(f"sizing.size must be in (0, 1], got {size}")
    except (TypeError, ValueError):
        err(f"sizing.size must be numeric, got {size!r}")

    # ---- live guards ----
    if mode == "live":
        guards = (spec.get("risk") or {}).get("live_guards") or {}
        if not guards.get("max_notional"):
            err("run.mode=live requires risk.live_guards.max_notional (hard per-order cap)")
        if not run.get("duration_end_at"):
            err("run.mode=live requires run.duration_end_at (ISO8601; sessions must be time-bounded)")

    # ---- static block resolution ----
    ind_specs = (signals.get("indicators") or {})
    for name, ispec in ind_specs.items():
        if not isinstance(ispec, dict) or "block" not in ispec:
            err(f"indicator {name!r} must be a mapping with a 'block' field")
            continue
        try:
            resolve_block(ispec["block"])
        except SpecError as exc:
            err(f"indicator {name!r}: {exc}")

    # If structural errors already exist, skip the dry-run (it would just
    # re-raise the same problems less clearly).
    if errors:
        return errors, warnings

    # ---- dry-run on synthetic data ----
    try:
        df = _synthetic_df(spec)
    except Exception as exc:  # pragma: no cover - defensive
        err(f"could not build synthetic data for dry-run: {exc}")
        return errors, warnings

    try:
        make_signals = build_make_signals(spec)
        long_s, short_s = make_signals(df)
    except Exception as exc:
        err(f"dry-run failed: {type(exc).__name__}: {exc}")
        return errors, warnings

    import pandas as pd

    for label, series in (("long", long_s), ("short", short_s)):
        if series is None:
            continue
        if not isinstance(series, pd.Series):
            err(f"entry.{label} must evaluate to a boolean Series, got {type(series).__name__}")
        elif series.dtype != bool:
            warnings.append(
                f"entry.{label} evaluated to dtype {series.dtype} (expected bool); "
                "will be coerced by the runner"
            )

    return errors, warnings


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_from_yaml(path: str) -> Dict[str, Any]:
    """Validate a spec and register it as a block strategy in this process.

    After this returns, ``spec['strategy']['id']`` is a known block strategy
    that the standard_bot entrypoints (``mvp_backtest`` / ``mvp_paper_daemon``)
    can run via ``--strategy <id>`` with ``--engine python``.
    """
    spec = load_spec(path)
    errors, _warnings = validate_spec(spec)
    if errors:
        joined = "\n  - ".join(errors)
        raise SpecError(f"spec {path} is invalid:\n  - {joined}")

    make_signals = build_make_signals(spec)

    htf_specs = [
        (h["interval"], int(h["sma_period"]))
        for h in (spec.get("data", {}) or {}).get("htf", []) or []
        if isinstance(h, dict) and "sma_period" in h
    ] or None
    exit_cfg = (spec.get("risk") or {}).get("exit")
    size = float((spec.get("sizing") or {}).get("size", 1.0))

    from cyqnt_trd.blocks import strategy as _strategy

    _strategy.register(
        spec["strategy"]["id"],
        make_signals,
        htf_specs=htf_specs,
        exit_cfg=exit_cfg if isinstance(exit_cfg, dict) else None,
        size=size,
    )
    return spec
