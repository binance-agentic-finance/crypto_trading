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
    INDICATOR_KEYS,
    SELECTION_KEYS,
    UNIVERSE_STEP_KEYS,
    SpecError,
    build_make_signals,
    build_selection_fn,
    eval_indicator,
    eval_node,
    resolve_block,
)
from .vocabulary import DATA_SECTIONS, synthetic_columns

VALID_MODES = {"backtest", "paper", "live"}

#: Keys allowed under ``data:``. Closed on purpose — a typo'd optional key used
#: to be accepted in silence, so ``data: {derivative: {...}}`` looked like it had
#: turned on funding data and had no effect whatsoever.
DATA_KEYS = frozenset(
    {"symbol", "market_type", "primary", "source", "htf"} | set(DATA_SECTIONS)
)
SIZING_KEYS = frozenset({"size"})
#: Exactly the keys the three engines read out of ``exit_cfg`` — verified by
#: grepping ``cfg.get("...")`` in blocks/strategy.py, vectorized_backtest.py,
#: runner.py and python_live_paper_session.py. Anything else was a typo that the
#: engine silently defaulted past (``stop_pctt`` cost a real stop-loss).
EXIT_KEYS = frozenset({
    "type", "max_bars", "stop_pct", "tp_pct", "atr_period", "stop_mult",
    "tp_mult", "trail_mult", "period", "ma_type",
})
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

    # Columns from the data sources this spec DECLARED. Only the declared ones:
    # fabricating everything would let a spec dry-run green and then meet an
    # absent column at runtime, which is the failure mode worth preventing.
    for column in synthetic_columns(spec):
        df[column] = _synthetic_series(column, idx, close, volume, np)
    return df


def _synthetic_series(column: str, idx, close, volume, np):
    """A plausible stand-in for one derived column.

    Plausible matters: a funding-rate column of zeros makes every
    ``funding_rate_state`` branch dead during the dry-run, so the dry-run stops
    proving anything. These oscillate through their real sign and magnitude
    ranges so the conditions built on them actually fire.
    """
    if column == "funding_rate":
        return 0.0001 * np.sin(idx / 11.0)
    if column == "funding_rate_bps":
        return 1.0 * np.sin(idx / 11.0)
    if column == "mark_price":
        return close * (1.0 + 0.0002 * np.sin(idx / 6.0))
    if column == "open_interest":
        return 50_000.0 + 5_000.0 * np.sin(idx / 19.0) + idx * 2.0
    if column == "open_interest_value":
        return (50_000.0 + 5_000.0 * np.sin(idx / 19.0) + idx * 2.0) * close
    if column == "oi_change_bps":
        return 40.0 * np.sin(idx / 9.0)
    if column.endswith("_count"):
        return np.abs(np.round(8.0 + 6.0 * np.sin(idx / 5.0)))
    if column.endswith("_qty"):
        return np.abs(2.0 + 1.5 * np.sin(idx / 7.0))
    if column == "net_liq_notional_usd":
        return 250_000.0 * np.sin(idx / 8.0)
    if column == "liq_imbalance_ratio":
        return 0.5 + 0.4 * np.sin(idx / 13.0)
    if column.endswith("_notional_usd"):
        return np.abs(300_000.0 + 250_000.0 * np.sin(idx / 8.0))
    # Unknown-but-declared: a bounded positive series is the least surprising
    # thing a block can be handed, and it will not divide by zero.
    return 1.0 + 0.5 * np.abs(np.sin(idx / 10.0))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _synthetic_universe(n: int = 12):
    """A stand-in universe frame: rows are symbols, not bars.

    Carries both ticker_rank vocabularies so a spec dry-runs the same way
    whichever upstream feeds it, and spans the thresholds a spec is likely to
    test so the direction rules actually fire during validation.
    """
    import numpy as np
    import pandas as pd

    bases = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX",
             "LINK", "DOT", "TON", "TRX"][:n]
    idx = np.arange(len(bases))
    bull = 0.5 + 0.45 * np.sin(idx / 2.0)
    mentions = (500 - idx * 35).astype(float)
    return pd.DataFrame({
        "symbol": ["%sUSDT" % b for b in bases],
        "instrument_id": ["%sUSDT" % b for b in bases],
        "quote_volume": 1e9 / (idx + 1),
        "quoteVolume": 1e9 / (idx + 1),
        "available_time": 1_700_000_000_000,
        "rank": idx + 1,
        "mention_count": mentions,
        "unique_authors": mentions / 4.0,
        "bull_ratio": bull,
        "bullish_count": np.round(mentions * bull),
        "bearish_count": np.round(mentions * (1.0 - bull)),
        "neutral_count": np.round(mentions * 0.1),
    })


def _dry_run_selection(spec: Dict[str, Any], errors: List[str],
                       warnings: List[str]) -> Tuple[List[str], List[str]]:
    """Execute the compiled selection against a synthetic universe.

    Same purpose as the trade dry-run: prove the spec is structurally runnable
    before any real data or money is involved.
    """
    universe = _synthetic_universe()
    rank = universe[["symbol", "instrument_id", "rank", "mention_count",
                     "unique_authors", "bull_ratio", "bullish_count",
                     "bearish_count", "neutral_count", "available_time"]].copy()
    try:
        candidates = build_selection_fn(spec)(universe, rank)
    except Exception as exc:
        errors.append("selection dry-run failed: %s: %s" % (type(exc).__name__, exc))
        return errors, warnings

    if not isinstance(candidates, list):
        errors.append("selection must produce a list of candidates, got %s"
                      % type(candidates).__name__)
        return errors, warnings
    if not candidates:
        warnings.append(
            "selection produced no candidates on synthetic data. That may be "
            "correct for a strict filter, but it also means the dry-run "
            "exercised none of the ranking or direction rules.")
        return errors, warnings

    required = {"symbol", "rank", "score", "side"}
    missing = required - set(candidates[0])
    if missing:
        errors.append("candidate is missing %s" % sorted(missing))
    return errors, warnings


def _condition_refs(node: Any, acc: List[str] | None = None) -> List[str]:
    """Every ``cond:`` block reference in a combinator tree, depth-first."""
    acc = [] if acc is None else acc
    if isinstance(node, dict):
        if isinstance(node.get("cond"), str):
            acc.append(node["cond"])
        for value in node.values():
            _condition_refs(value, acc)
    elif isinstance(node, list):
        for item in node:
            _condition_refs(item, acc)
    return acc


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
    for key in sorted(set(data) - DATA_KEYS):
        err(
            "unknown data.%s; a misspelt optional section is accepted in silence "
            "and simply attaches nothing. Known sections: %s"
            % (key, sorted(DATA_KEYS))
        )
    for name, section in DATA_SECTIONS.items():
        declared = data.get(name)
        if declared is None:
            continue
        if not isinstance(declared, dict) or not declared.get("dir"):
            err("data.%s needs a 'dir'. %s" % (name, section.example))

    signals = spec.get("signals") or {}
    entry = signals.get("entry") or {}
    selection = spec.get("selection")
    if selection is not None and signals:
        err("a spec is either a trade strategy (signals:) or a selection "
            "strategy (selection:), not both — they emit different signal kinds")
    # ``isinstance``, matching every other selection check in this file. Using
    # ``is None`` here meant a non-None non-dict ``selection:`` — the classic
    # mis-indentation that makes it a YAML scalar — skipped this check AND the
    # isinstance branch in register_from_yaml, so a spec with no signals at all
    # validated clean and was registered as a TRADE strategy whose make_signals
    # returns all-False. It then backtested to a spotless trades=0.
    if not isinstance(selection, dict) and not entry.get("long") \
            and not entry.get("short"):
        err("signals.entry must define at least one of long / short")
    if selection is not None and not isinstance(selection, dict):
        err("selection: must be a mapping, got %s — a scalar here is usually a "
            "mis-indented block, and it would otherwise be registered as a trade "
            "strategy that can never fire" % type(selection).__name__)

    # ---- exit / risk ----
    exit_cfg = (spec.get("risk") or {}).get("exit")
    if exit_cfg is not None:
        etype = exit_cfg.get("type") if isinstance(exit_cfg, dict) else None
        if etype not in VALID_EXIT_TYPES:
            err(f"risk.exit.type must be one of {sorted(VALID_EXIT_TYPES)}, got {etype!r}")
        if isinstance(exit_cfg, dict):
            for key in sorted(set(exit_cfg) - EXIT_KEYS):
                err(
                    "unknown risk.exit.%s — the engines read only %s, so this key "
                    "would be dropped and the exit would fall back to its default"
                    % (key, sorted(EXIT_KEYS))
                )

    # ---- sizing ----
    sizing = spec.get("sizing") or {}
    for key in sorted(set(sizing) - SIZING_KEYS):
        err("unknown sizing.%s; allowed: %s" % (key, sorted(SIZING_KEYS)))
    size = sizing.get("size", 1.0)
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
        unknown_fields = sorted(set(ispec) - INDICATOR_KEYS)
        if unknown_fields:
            err(
                "indicator %r has unknown field(s) %s; allowed: %s"
                % (name, unknown_fields, sorted(INDICATOR_KEYS))
            )

    # Conditions resolve statically too. The dry-run would catch these anyway,
    # but only the first one, and only as a dry-run traceback — listing them all
    # up front is the difference between one round trip and five.
    for label in ("long", "short"):
        for ref in _condition_refs(entry.get(label)):
            try:
                resolve_block(ref)
            except SpecError as exc:
                err(f"entry.{label}: {exc}")

    # ---- selection ----
    if isinstance(selection, dict):
        for key in sorted(set(selection) - SELECTION_KEYS):
            err("unknown selection.%s; allowed: %s" % (key, sorted(SELECTION_KEYS)))
        if not selection.get("score"):
            err("selection.score is required: name the column or feature to rank by")
        for position, step in enumerate(selection.get("universe") or []):
            if not isinstance(step, dict) or "block" not in step:
                err("selection.universe[%d] must be a mapping with a 'block'" % position)
                continue
            for key in sorted(set(step) - UNIVERSE_STEP_KEYS):
                err("unknown selection.universe[%d].%s; allowed: %s"
                    % (position, key, sorted(UNIVERSE_STEP_KEYS)))
            try:
                resolve_block(step["block"])
            except SpecError as exc:
                err("selection.universe[%d]: %s" % (position, exc))
        for name, fspec in (selection.get("features") or {}).items():
            if not isinstance(fspec, dict) or "block" not in fspec:
                err("selection.features.%s must be a mapping with a 'block'" % name)
                continue
            try:
                resolve_block(fspec["block"])
            except SpecError as exc:
                err("selection.features.%s: %s" % (name, exc))
        for label in ("long_when", "short_when"):
            for ref in _condition_refs(selection.get(label)):
                try:
                    resolve_block(ref)
                except SpecError as exc:
                    err("selection.%s: %s" % (label, exc))

    # If structural errors already exist, skip the dry-run (it would just
    # re-raise the same problems less clearly).
    if errors:
        return errors, warnings

    if isinstance(selection, dict):
        return _dry_run_selection(spec, errors, warnings)

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

    from cyqnt_trd.blocks import strategy as _strategy

    if isinstance(spec.get("selection"), dict):
        # Same registry, same run_pipeline_step, different signal kind — the
        # selection path was already built and simply had no way in from YAML.
        _strategy.register_selection(
            spec["strategy"]["id"],
            build_selection_fn(spec),
            market_type=(spec.get("data") or {}).get("market_type", "futures"),
        )
        return spec

    make_signals = build_make_signals(spec)

    htf_specs = [
        (h["interval"], int(h["sma_period"]))
        for h in (spec.get("data", {}) or {}).get("htf", []) or []
        if isinstance(h, dict) and "sma_period" in h
    ] or None
    exit_cfg = (spec.get("risk") or {}).get("exit")
    size = float((spec.get("sizing") or {}).get("size", 1.0))

    _strategy.register(
        spec["strategy"]["id"],
        make_signals,
        htf_specs=htf_specs,
        exit_cfg=exit_cfg if isinstance(exit_cfg, dict) else None,
        size=size,
    )
    return spec
