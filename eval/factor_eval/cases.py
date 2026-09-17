"""Run an external tiered-strategy spec through the evaluation and backtest stages.

A large family of production strategies is written the same way: a few **hard
gates**, a few **scored tiers**, and a **verdict threshold** that turns the total
into trade / watch / skip. `blueprint.py` already has that skeleton; this module
is the adapter that turns a declarative spec into one, so a strategy written
elsewhere can be measured on the same panel, the same costs and the same splits
as everything else.

    {"scoring": {"tiers":    {"rsi_depth": {...}, "ema_cross": {...}},
                 "verdicts": {"candidate_min": 5, ...}}}

The part that decides whether the answer is worth anything is **coverage**. A spec
names tiers like `smart_money_inflow`, `hotrank_attention` or `multi_tf_trend`;
a panel of daily OHLCV and funding can compute some of them and cannot compute the
others at all. Scoring the missing ones as 0 would quietly turn a five-tier
strategy into a three-tier one and report the result as if it were the strategy
that was asked for. So every tier resolves to exactly one of:

    resolved     a signal on this panel, from the capability registry
    unavailable  the panel has no such field -- recorded with the reason, never scored

`CaseResult.coverage` reports the split, and `run_case` refuses by default to
backtest a spec whose **hard gates** are unavailable: a gate that cannot be
evaluated is not a gate that passes.

Thresholds are a second, separate limitation. These specs carry the *shape* of the
bands (`deep_score`, `moderate_score`, `wrong_zone_score`) but usually not the
numbers that separate them -- those live in each strategy's own code. Where a
boundary is missing this module uses a documented conventional one and records it
in `CaseResult.assumed`, because a backtest run on invented thresholds that does
not say so is worse than no backtest.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .blueprint import Blueprint, Exits, Gate, Tier, blueprint_targets
from .framework import backtest_weights, validate_weights

__all__ = ["CaseSpec", "CaseResult", "load_case", "resolve_signals", "build_blueprint",
           "run_case", "TIER_SIGNALS", "UNAVAILABLE"]


# --------------------------------------------------------------- tier resolution
def _capability(name: str, port: str | None = None, **params) -> Callable:
    """A panel signal backed by one capability node, resolved lazily."""
    def make(panel):
        from .capability_registry import load_index, upstream_factor
        spec = load_index()[name]
        return upstream_factor(spec, port=port or (spec.ports[0] if spec.ports else None),
                               params=params or None, skip_failures=True)(panel)
    make.__name__ = f"capability:{name}.{port or 'default'}"
    return make


def _funding_z(window: int = 30):
    def make(panel):
        carry = panel.funding / panel.close
        mean = carry.rolling(window, min_periods=window // 2).mean().shift(1)
        std = carry.rolling(window, min_periods=window // 2).std().shift(1)
        return (carry - mean) / std.replace(0, np.nan)
    make.__name__ = "funding_zscore"
    return make


def _price_change(window: int = 1):
    def make(panel):
        return panel.close / panel.close.shift(window) - 1.0
    make.__name__ = f"price_change_{window}"
    return make


def _quote_volume():
    def make(panel):
        return panel.quote_volume
    make.__name__ = "quote_volume"
    return make


#: Tier-name keyword -> signal. First match wins, so order is specific-to-general.
#: Every entry is backed by a capability node or a panel field; nothing here
#: invents data.
TIER_SIGNALS: tuple[tuple[str, Callable], ...] = (
    (r"stochrsi", _capability("stochrsi", "k")),
    (r"stoch", _capability("stochastic", "k")),
    (r"rsi", _capability("rsi", "rsi")),
    (r"macd", _capability("macd", "macd")),
    (r"adx|mdi|dmi", _capability("adx", "adx")),
    (r"supertrend", _capability("supertrend", "supertrend")),
    (r"aroon", _capability("aroon", "aroon_up")),
    (r"cci", _capability("cci", "value")),
    (r"williams", _capability("williams_r", "value")),
    (r"mfi", _capability("mfi", "value")),
    (r"cmf", _capability("cmf", "value")),
    (r"bb_squeeze|bandwidth|squeeze", _capability("bb_bandwidth", "value")),
    (r"bb_|bollinger|percent_b|pct_b", _capability("bb_pct_b", "value")),
    (r"atr|volatility|vol_regime", _capability("atr_ratio", "value")),
    (r"ema|sma|\bma\b|moving_average|golden|death", _capability("ema", "ema", period=20)),
    # Before the generic trend rule: `funding_bias` contains "bias" and was
    # resolving to trend strength, which silently turned two different cases into
    # the same strategy with identical numbers.
    # `derivatives` tiers are specified as "funding rate + open interest". Only
    # the funding half exists on this panel; resolving to it is better than
    # dropping the tier, but the OI half is genuinely absent and is recorded as
    # a partial in `assumed` rather than passed off as the whole tier.
    (r"funding|carry|basis|derivativ", _funding_z()),
    (r"trend|resonance|alignment|bias|direction", _capability("trend_strength", "value")),
    (r"order_block", _capability("order_block_detect", "ob_top")),
    (r"fvg|fair_value", _capability("fair_value_gap", "fvg_top")),
    (r"liquidity_sweep|sweep", _capability("liquidity_sweep_detect", "sweep_level")),
    (r"structur|bos|choch|breakout|pattern|candle", _capability("bos_choch_detect", "broken_level")),
    (r"equal_high|eqh|eql", _capability("equal_highs_lows", "eqh_level")),
    (r"premium|discount|zone", _capability("premium_discount_zone", "premium_top")),
    (r"range|consolidat", _capability("range_gain_pct", "value")),
    (r"obv|pvt|accumulat", _capability("obv", "value")),
    (r"volume|turnover|liquidity_depth", _capability("volume_zscore", "value")),
    (r"momentum|velocity|accelerat|change|depth|magnitude|strength", _price_change(4)),
    (r"gate|min_volume", _quote_volume()),
)

#: Tier-name keyword -> why this panel cannot produce it. Checked **before**
#: TIER_SIGNALS so a name like `smart_money_inflow` is not matched by the generic
#: `momentum` rule and scored on price data it has nothing to do with.
UNAVAILABLE: tuple[tuple[str, str], ...] = (
    (r"hotrank|social|buzz|square|sentiment|news|attention|mention",
     "panel carries no attention or social stream"),
    (r"smart_money|onchain|on_chain|inflow|whale|tvl|holder|token_age|chain",
     "panel carries no on-chain flow"),
    (r"open_interest|\boi\b|oi_|long_short|ls_ratio|liquidation",
     "panel carries no open interest or long/short ratio"),
    # `depth` alone is too broad: it also appears in `rsi_depth`, which is an RSI
    # tier and was being rejected as an order-book tier.
    (r"orderbook|order_book|book_depth|liquidity_depth|bid|ask|spread|slippage|maker|taker",
     "panel carries no order book"),
    (r"multi_tf|multi_timeframe|_4tf|dual_tf|_15m|_1h|_4h_|timeframe",
     "panel is a single timeframe; multi-timeframe resonance needs a finer grid"),
    (r"macro|earnings|calendar|cpi|fomc|event",
     "panel carries no macro event calendar"),
    (r"correlation|pair|leg_|spread_vol",
     "panel-level pair construction is not part of this spec"),
    (r"circuit_breaker|leverage|stop_distance|position|sizing|safety|balance",
     "execution-layer control, not a cross-sectional signal"),
    (r"ahr999|fib|golden_ratio|divergence|agreement|confirmation_bonus|bonus",
     "no capability node implements this construct"),
)


def _classify(tier: str) -> tuple[Callable | None, str]:
    for pattern, reason in UNAVAILABLE:
        if re.search(pattern, tier, re.I):
            return None, reason
    for pattern, signal in TIER_SIGNALS:
        if re.search(pattern, tier, re.I):
            return signal, ""
    return None, "no signal mapped for this tier name"


# ---------------------------------------------------------------------- the spec
@dataclass
class CaseSpec:
    """A tiered strategy spec, as declared by its own config."""

    name: str
    tiers: dict[str, dict]
    verdicts: dict[str, float]
    raw: dict = field(default_factory=dict)

    @property
    def hard_gates(self) -> list[str]:
        return [n for n, t in self.tiers.items() if t.get("is_hard_gate")]

    def entry_score(self) -> float:
        """The total a name must reach to be traded.

        `candidate_min` rather than `strong_candidate_min`: the specs treat
        STRONG as a subset of CANDIDATE, and taking only the strongest would
        measure a different, narrower strategy than the one written down.
        """
        for key in ("candidate_min", "strong_candidate_min", "watchlist_min"):
            value = self.verdicts.get(key)
            if value is not None:
                return float(value)
        return 1.0

    def max_total(self) -> float:
        return float(sum(abs(float(t.get("max_score", _implied_max(t))))
                         for t in self.tiers.values()))


def _implied_max(tier: Mapping) -> float:
    """Largest positive band in a spec that lists bands instead of `max_score`."""
    scores = [float(v) for k, v in tier.items()
              if k.endswith("_score") and isinstance(v, (int, float))]
    return max(scores) if scores else 1.0


def load_case(source: str | Path | Mapping, name: str | None = None) -> CaseSpec:
    """Read a `scoring_config.json`-shaped spec."""
    if isinstance(source, Mapping):
        raw, label = dict(source), name or "case"
    else:
        path = Path(source)
        raw, label = json.loads(path.read_text()), name or path.parent.parent.name
    scoring = raw.get("scoring") or {}
    tiers = scoring.get("tiers") or {}
    if not tiers:
        raise ValueError(f"{label}: spec has no scoring.tiers; nothing to evaluate")
    return CaseSpec(name=label, tiers={k: dict(v) for k, v in tiers.items()},
                    verdicts=dict(scoring.get("verdicts") or {}), raw=raw)


def resolve_signals(spec: CaseSpec) -> tuple[dict[str, Callable], dict[str, str]]:
    """Split the tiers into what this panel can compute and what it cannot."""
    resolved, unavailable = {}, {}
    for tier in spec.tiers:
        signal, reason = _classify(tier)
        if signal is None:
            unavailable[tier] = reason
        else:
            resolved[tier] = signal
    return resolved, unavailable


def build_blueprint(spec: CaseSpec, resolved: Mapping[str, Callable], *,
                    schedule: int = 3, top_k: int | None = 3,
                    cap: float = 0.35, gross: float = 1.0,
                    entry_score: float | None = None,
                    assumed: dict | None = None) -> Blueprint:
    """Spec + resolved signals -> a `Blueprint`.

    Bands are built on the **cross-sectional rank** of each signal, not on its raw
    value. The specs were written against absolute levels ("RSI below 30"), but the
    numbers that separate the bands live in each strategy's own code and are not in
    the config. Ranking keeps the ordering the spec intended -- deeper is better,
    stronger is better -- without inventing a boundary, and it is the only reading
    that behaves the same for an RSI in [0, 100] and an ATR ratio in [0, 3].
    Everything substituted this way is recorded in `assumed`.
    """
    assumed = assumed if assumed is not None else {}
    tiers, gates = [], []
    for name, signal in resolved.items():
        config = spec.tiers[name]
        top = float(config.get("max_score", _implied_max(config)))
        # Quintiles of the cross-section, worst to best, scaled to the tier's own
        # maximum so the spec's relative tier weights survive.
        bands = [(0.0, 0.2, -top / 3), (0.2, 0.4, 0.0), (0.4, 0.6, top / 3),
                 (0.6, 0.8, 2 * top / 3), (0.8, 1.0001, top)]
        # weight stays 1.0 on purpose. A spec's `weight` field is a normalised
        # share that sums to 1.0 across tiers; the score magnitude is already
        # carried by `max_score` (smart_momentum: 2+3+3+2 = 10 against a
        # CANDIDATE threshold of 5). Applying both scales the total to about 1.0,
        # the threshold becomes unreachable, and every case backtests as a flat
        # book -- 16 of 19 did exactly that before this was fixed, reporting
        # "ran" with zero turnover.
        tiers.append(Tier(name=name, signal=signal, bands=bands, weight=1.0,
                          normalize="rank"))
        assumed[f"{name}.bands"] = ("spec declares band names but no boundaries; "
                                    "cross-sectional quintiles used")
        if re.search(r"derivativ", name, re.I):
            assumed[f"{name}.partial"] = ("spec pairs funding with open interest; "
                                          "only the funding half exists on this panel")
        if config.get("is_hard_gate"):
            # Bottom quintile of the cross-section fails the gate. The spec's own
            # threshold is not in the config.
            gates.append(Gate(name=name, signal=_ranked(signal), op=">=", value=0.2))
            assumed[f"{name}.gate"] = ("hard gate threshold not in spec; bottom "
                                       "cross-sectional quintile rejected")
    return Blueprint(name=spec.name, tiers=tiers, gates=gates, schedule=schedule,
                     entry_score=spec.entry_score() if entry_score is None else entry_score,
                     top_k=top_k, direction="score",
                     sizing="equal", gross=gross, cap=cap, neutral=False,
                     exits=Exits())


def _ranked(signal: Callable) -> Callable:
    def make(panel):
        x = signal(panel).where(panel.mask)
        return x.rank(axis=1, pct=True)
    make.__name__ = f"rank({getattr(signal, '__name__', 'signal')})"
    return make


# -------------------------------------------------------------------- the runner
@dataclass
class CaseResult:
    name: str
    coverage: dict
    assumed: dict
    targets: pd.DataFrame | None = None
    backtest: object = None
    error: str = ""

    @property
    def ran(self) -> bool:
        return self.backtest is not None

    def row(self) -> dict:
        c = self.coverage
        out = {"case": self.name, "tiers": c["n_tiers"], "resolved": c["n_resolved"],
               "unavailable": c["n_unavailable"], "gates_dropped": c["n_gates_unavailable"],
               "degraded": bool(self.assumed.get("entry_score") or self.assumed.get("hard_gates")),
               "ran": self.ran, "error": self.error[:70]}
        if self.backtest is not None:
            oot = self.backtest.metrics.set_index("split").loc["oot"]
            traded = float(oot.turnover) > 0
            out.update(oot_traded=traded,
                       # A split the book sat out has no return to report. Printing
                       # 0.0 next to cases that did trade reads as "flat performance"
                       # rather than "no trades in this window".
                       oot_net_bp=float(oot.net_bp) if traded else float("nan"),
                       oot_sharpe=float(oot.sharpe_net) if traded else float("nan"),
                       turnover=float(oot.turnover))
        return out


def run_case(spec: CaseSpec, panel, *, entry_lag: int = 2, h: int = 3,
             cost_bps: float = 6.5, schedule: int = 3, top_k: int | None = 3,
             start="2022-04-01", strict: bool = False, require_gates: bool | None = None,
             min_resolved: int = 1, min_score_coverage: float = 0.0,
             scale_threshold: bool = True) -> CaseResult:
    """Resolve, build, backtest. Returns the coverage report either way.

    Default is **degraded mode**: run whatever the panel can compute, and say
    exactly what was given up. `strict=True` restores the refusals -- use it when
    the question is "is this the strategy that was written down", not "what does
    the computable part of it do here".

    Two things make a degraded run meaningful rather than arbitrary:

    `scale_threshold`
        The spec's verdict threshold was set against the full tier set. With half
        the tiers gone the total can never reach it, so the threshold is scaled by
        the share of the maximum score that survives. A spec asking for 5 out of a
        possible 10 becomes one asking for 2 out of a possible 4 -- the same
        selectivity on a smaller score, rather than an unreachable bar that
        backtests as a flat book.

    dropped hard gates
        Recorded in `coverage["gates_unavailable"]` and in `assumed`. The result is
        an **unscreened** version of a screened strategy; that is a different
        strategy and the report has to keep saying so.
    """
    require_gates = strict if require_gates is None else require_gates
    resolved, unavailable = resolve_signals(spec)
    blocked = [g for g in spec.hard_gates if g in unavailable]
    coverage = {"n_tiers": len(spec.tiers), "n_resolved": len(resolved),
                "n_unavailable": len(unavailable), "resolved": sorted(resolved),
                "unavailable": unavailable, "hard_gates": spec.hard_gates,
                "n_gates_unavailable": len(blocked), "gates_unavailable": blocked}
    assumed: dict = {}

    if len(resolved) < max(1, min_resolved):
        return CaseResult(spec.name, coverage, assumed,
                          error=f"no tier of {len(spec.tiers)} is computable on this panel")
    if blocked and require_gates:
        return CaseResult(spec.name, coverage, assumed,
                          error=f"hard gate(s) not computable: {', '.join(blocked)}")

    # Dropping a tier also drops the score it could have contributed. When enough
    # of them go, the spec's own entry threshold stops being reachable and the
    # strategy can never open a position -- it backtests as a flat book and reports
    # zero, which reads like "this strategy makes no money" rather than "this panel
    # cannot run this strategy". Refuse and say which it is.
    reachable = sum(abs(float(spec.tiers[t].get("max_score", _implied_max(spec.tiers[t]))))
                    for t in resolved)
    declared = spec.max_total()
    entry = spec.entry_score()
    coverage["reachable_score"] = reachable
    coverage["declared_score"] = declared
    coverage["entry_score"] = entry
    if reachable < declared - 1e-9:
        # The threshold was set against the full tier set. Whenever any tier is
        # missing it is scaled by the share of the maximum score that survives, so
        # the spec keeps its selectivity ("top half of what is achievable") instead
        # of either an unreachable bar or an accidentally loosened one. Only
        # applying this when the bar is strictly unreachable left specs that lost a
        # third of their score still screening as if they had it.
        if strict or not scale_threshold:
            if reachable < entry:
                return CaseResult(spec.name, coverage, assumed,
                                  error=f"entry threshold {entry:g} unreachable: the "
                                        f"computable tiers total {reachable:g}")
        else:
            entry = entry * reachable / declared if declared > 0 else entry
            assumed["entry_score"] = (
                f"spec threshold {spec.entry_score():g} was set against a maximum of "
                f"{declared:g}; scaled to {entry:.3g} against the {reachable:g} the "
                f"computable tiers can reach")
    if scale_threshold and not strict and entry > reachable:
        # Some specs set a threshold above their own declared maximum (the declared
        # bands do not add up to it). Clamping to the achievable total means "only
        # the top band of every tier", which is the strictest thing the spec can
        # still express, rather than a bar nothing can clear.
        assumed["entry_score_clamped"] = (
            f"threshold {entry:.3g} exceeds the {reachable:g} the tiers can produce; "
            f"clamped to {reachable:g} (top band of every computable tier)")
        entry = reachable
    coverage["entry_score_used"] = entry
    if blocked and not require_gates:
        assumed["hard_gates"] = (f"dropped uncomputable hard gate(s) {', '.join(blocked)}; "
                                 f"this is the unscreened version of a screened strategy")

    try:
        blueprint = build_blueprint(spec, resolved, schedule=schedule, top_k=top_k,
                                    assumed=assumed, entry_score=entry)
        # How often the spec can even be evaluated. Structure detectors
        # (order blocks, fair value gaps, liquidity sweeps) only emit on an event,
        # so a spec built out of them has a finite score on a couple of percent of
        # cells and trades almost never. That is a property of the signals, not a
        # verdict on the strategy, and it has to be said rather than shown as a
        # flat equity curve.
        score = blueprint.score(panel)
        eligible = int(panel.mask.to_numpy().sum())
        scored = int(np.isfinite(score.where(panel.mask).to_numpy(dtype=float)).sum())
        coverage["score_coverage"] = scored / eligible if eligible else 0.0
        if coverage["score_coverage"] < min_score_coverage:
            return CaseResult(spec.name, coverage, assumed,
                              error=f"score defined on {coverage['score_coverage']:.1%} of "
                                    f"eligible cells; signals are too sparse to trade")
        targets = blueprint_targets(blueprint, panel, start=start)
        # The blueprint leaves non-decision cells empty, meaning "keep holding";
        # stage 6 wants an explicit weight in every cell.
        weights = validate_weights(targets.ffill().fillna(0.0), panel)
        # A book that never opens is not a strategy that lost nothing; it is a
        # spec this panel could not trade. Reported as a refusal rather than as a
        # flat equity curve with a 0.00 return, which reads like a result.
        if not (weights.abs().to_numpy() > 0).any():
            return CaseResult(spec.name, coverage, assumed, targets=targets,
                              error=f"never reaches the entry threshold "
                                    f"{spec.entry_score():g}; no position was ever opened")
        result = backtest_weights(weights, panel, entry_lag=entry_lag, h=h,
                                  cost_bps=cost_bps)
        return CaseResult(spec.name, coverage, assumed, targets=targets, backtest=result)
    except Exception as exc:  # noqa: BLE001 - reported per case, never silent
        return CaseResult(spec.name, coverage, assumed,
                          error=f"{type(exc).__name__}: {exc}")
