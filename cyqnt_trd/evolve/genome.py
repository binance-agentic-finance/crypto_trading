"""Strategy genome dataclasses.

Immutable-ish data containers for a single strategy individual in the GA pool.
Supports JSON round-trip via ``to_dict`` / ``from_dict`` and basic validation.

The genome schema is defined in SPEC §1.

Species classification rules (SPEC §4):
    trend          — has EMA/MA cross as primary entry
    mean_reversion — has RSI oversold/overbought or Bollinger bounce
    breakout       — has N-bar high/low breakout or channel break
    momentum       — has MACD / Stochastic / momentum indicator
    volatility     — has Bollinger squeeze / ATR expansion as trigger

The species label is supplied by the LLM at genome construction time but
``infer_species()`` provides a deterministic fallback for crossover children.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# ── Vocabulary (whitelisted to keep LLM honest) ────────────────────────────

ALLOWED_SPECIES = {"trend", "mean_reversion", "breakout", "momentum", "volatility"}

ALLOWED_INDICATORS = {
    "ema", "sma", "rsi", "macd", "bollinger", "stoch", "adx", "atr",
    "supertrend", "vwap", "volume", "donchian", "cci", "williams_r",
    "mfi", "obv", "keltner", "hma", "tema", "dema",
    # ── extended for high-freq XRP exploration ──
    "stochrsi", "aroon", "psar", "atr_ratio", "trend_strength",
    "volume_surge", "ichimoku",
}

ALLOWED_CONDITIONS = {
    "cross_above", "cross_below", "above", "below",
    "above_threshold", "below_threshold",
    "slope_positive", "slope_negative",
    "squeeze", "breakout", "divergence",
    "oversold", "overbought",
    "macd_golden_cross", "macd_death_cross",
    "macd_above_zero", "macd_below_zero",
    "touch_lower", "touch_upper",  # bollinger touch
    "trending",  # adx
    "spike",     # volume / atr expansion
    # ── extended ──
    "aroon_up_strong", "aroon_dn_strong", "psar_flip_up", "psar_flip_dn",
    "above_cloud", "below_cloud",  # ichimoku
    "low_atr_ratio", "high_atr_ratio",
}

ALLOWED_FILTERS = {
    "adx_above", "atr_above_pct", "atr_above_percentile",
    "volume_above", "volume_above_ma",
    "hour_range", "ma_slope_positive", "bbw_above",
    "rsi_in_range", "ema_alignment",
    "price_above_ma", "price_below_ma",
    # ── extended ──
    "atr_below_percentile",   # block when atr in TOP X% (avoid crash periods)
    "atr_ratio_below",        # block when atr% > threshold (high vol regime)
    "ema_slope_positive",     # require positive EMA slope (regime gate)
    "consecutive_loss_cooldown",  # placeholder — not impl as state-free filter
    "stochrsi_above",
    "aroon_up_above",
}

ALLOWED_EXIT_TYPES = {"pct_stop_tp", "atr_stop_tp", "time_only", "ma_cross_exit"}
ALLOWED_INTERVALS = {"5m", "15m", "1h"}
ALLOWED_ENTRY_LOGIC = {"all_of", "any_of", "score_gte"}

# Per-interval max_bars caps (intraday rule, SPEC §6)
MAX_BARS_BY_INTERVAL = {"5m": 60, "15m": 24, "1h": 8}


# ── Dataclasses ────────────────────────────────────────────────────────────

@dataclass
class Factor:
    """A single entry factor (indicator + condition + params)."""

    indicator: str
    condition: str
    params: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0  # for score_gte mode

    def to_dict(self) -> Dict[str, Any]:
        return {
            "indicator": self.indicator,
            "condition": self.condition,
            "params": dict(self.params),
            "weight": float(self.weight),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Factor":
        return cls(
            indicator=str(data["indicator"]).lower(),
            condition=str(data["condition"]).lower(),
            params=dict(data.get("params", {})),
            weight=float(data.get("weight", 1.0)),
        )

    def validate(self) -> List[str]:
        errs: List[str] = []
        if self.indicator not in ALLOWED_INDICATORS:
            errs.append(f"unknown indicator: {self.indicator}")
        if self.condition not in ALLOWED_CONDITIONS:
            errs.append(f"unknown condition: {self.condition}")
        if not isinstance(self.params, dict):
            errs.append(f"params must be dict, got {type(self.params).__name__}")
        return errs


@dataclass
class Filter:
    """An environment filter that gates entry signals."""

    filter_type: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"filter_type": self.filter_type, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Filter":
        return cls(
            filter_type=str(data["filter_type"]).lower(),
            params=dict(data.get("params", {})),
        )

    def validate(self) -> List[str]:
        errs: List[str] = []
        if self.filter_type not in ALLOWED_FILTERS:
            errs.append(f"unknown filter_type: {self.filter_type}")
        return errs


@dataclass
class StrategyGenome:
    """A single strategy individual."""

    genome_id: str
    species: str
    entry_factors: List[Factor]
    entry_logic: str = "all_of"
    entry_score_threshold: int = 2
    exit_type: str = "pct_stop_tp"
    exit_params: Dict[str, Any] = field(default_factory=dict)
    filters: List[Filter] = field(default_factory=list)
    size: float = 0.5
    preferred_interval: str = "15m"

    # ── Metadata ──
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_log: List[str] = field(default_factory=list)
    fitness_history: List[float] = field(default_factory=list)
    notes: Optional[str] = None  # free-form LLM commentary

    # ─────────────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "species": self.species,
            "entry_factors": [f.to_dict() for f in self.entry_factors],
            "entry_logic": self.entry_logic,
            "entry_score_threshold": int(self.entry_score_threshold),
            "exit_type": self.exit_type,
            "exit_params": dict(self.exit_params),
            "filters": [fl.to_dict() for fl in self.filters],
            "size": float(self.size),
            "preferred_interval": self.preferred_interval,
            "generation": int(self.generation),
            "parent_ids": list(self.parent_ids),
            "mutation_log": list(self.mutation_log),
            "fitness_history": [float(x) for x in self.fitness_history],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyGenome":
        return cls(
            genome_id=str(data["genome_id"]),
            species=str(data.get("species", "trend")).lower(),
            entry_factors=[Factor.from_dict(f) for f in data.get("entry_factors", [])],
            entry_logic=str(data.get("entry_logic", "all_of")).lower(),
            entry_score_threshold=int(data.get("entry_score_threshold", 2)),
            exit_type=str(data.get("exit_type", "pct_stop_tp")).lower(),
            exit_params=dict(data.get("exit_params", {})),
            filters=[Filter.from_dict(fl) for fl in data.get("filters", [])],
            size=float(data.get("size", 0.5)),
            preferred_interval=str(data.get("preferred_interval", "15m")).lower(),
            generation=int(data.get("generation", 0)),
            parent_ids=list(data.get("parent_ids", [])),
            mutation_log=list(data.get("mutation_log", [])),
            fitness_history=[float(x) for x in data.get("fitness_history", [])],
            notes=data.get("notes"),
        )

    # ─────────────────────────────────────────────────────────────────
    def clone(self, *, new_id: Optional[str] = None) -> "StrategyGenome":
        c = copy.deepcopy(self)
        if new_id:
            c.genome_id = new_id
            c.parent_ids = [self.genome_id]
            c.mutation_log = []
            c.fitness_history = []
        return c

    def validate(self) -> List[str]:
        errs: List[str] = []
        if not self.genome_id:
            errs.append("genome_id is empty")
        if self.species not in ALLOWED_SPECIES:
            errs.append(f"unknown species: {self.species}")
        if not self.entry_factors:
            errs.append("entry_factors must not be empty")
        if len(self.entry_factors) > 4:
            errs.append("entry_factors must be <= 4 (Occam's razor)")
        if self.entry_logic not in ALLOWED_ENTRY_LOGIC:
            errs.append(f"unknown entry_logic: {self.entry_logic}")
        if self.exit_type not in ALLOWED_EXIT_TYPES:
            errs.append(f"unknown exit_type: {self.exit_type}")
        if self.preferred_interval not in ALLOWED_INTERVALS:
            errs.append(f"unknown preferred_interval: {self.preferred_interval}")
        if not (0.1 <= self.size <= 1.0):
            errs.append(f"size must be in [0.1, 1.0], got {self.size}")
        if len(self.filters) > 3:
            errs.append("filters must be <= 3")

        # Intraday max_bars cap
        max_bars = self.exit_params.get("max_bars")
        if max_bars is not None:
            cap = MAX_BARS_BY_INTERVAL.get(self.preferred_interval, 24)
            if int(max_bars) > cap:
                errs.append(
                    f"max_bars ({max_bars}) exceeds intraday cap {cap} "
                    f"for {self.preferred_interval}"
                )

        # Per-element validation
        for i, f in enumerate(self.entry_factors):
            for e in f.validate():
                errs.append(f"entry_factors[{i}]: {e}")
        for i, fl in enumerate(self.filters):
            for e in fl.validate():
                errs.append(f"filters[{i}]: {e}")

        return errs

    def is_valid(self) -> bool:
        return not self.validate()


# ── Helpers ────────────────────────────────────────────────────────────────

def infer_species(genome: StrategyGenome) -> str:
    """Deterministic species inference based on entry factors.

    Used as a fallback when crossover children inherit factors from two
    different-species parents.
    """
    factors = genome.entry_factors
    indicators = [f.indicator for f in factors]
    conditions = [f.condition for f in factors]

    # volatility: bollinger squeeze or atr spike triggers
    if any(c in ("squeeze", "spike") for c in conditions):
        return "volatility"
    if "bollinger" in indicators and "squeeze" in conditions:
        return "volatility"

    # breakout: N-bar high/low breakout or donchian
    if any(c == "breakout" for c in conditions):
        return "breakout"
    if "donchian" in indicators:
        return "breakout"

    # momentum: MACD or Stochastic
    if "macd" in indicators or "stoch" in indicators:
        return "momentum"

    # mean_reversion: RSI oversold/overbought or bollinger touch
    if any(c in ("oversold", "overbought", "touch_lower", "touch_upper") for c in conditions):
        return "mean_reversion"

    # trend: EMA/SMA cross (default fallback)
    if any(c in ("cross_above", "cross_below") for c in conditions):
        return "trend"

    return "trend"


def cap_max_bars(genome: StrategyGenome) -> StrategyGenome:
    """Mutate exit_params.max_bars in-place to respect intraday cap."""
    cap = MAX_BARS_BY_INTERVAL.get(genome.preferred_interval, 24)
    if "max_bars" in genome.exit_params:
        genome.exit_params["max_bars"] = min(int(genome.exit_params["max_bars"]), cap)
    return genome
