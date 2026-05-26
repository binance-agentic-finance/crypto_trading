# Trade Structure Analyzer

> case_id: `structured-trade-plan` · case_type: `trade_plan_report` · status: `migrated`

Single-symbol structural trade plan. Computes EMA-based bias, ATR-derived stops,
and rolling support/resistance, then emits three tiers (conservative / balanced
/ aggressive). Each tier carries entry zone, stop, target, and R:R ratio.

Advisory only — no execution path.

## Quickstart

```python
from ai_pro_trading_library.cases import load_case_report

report = load_case_report("structured-trade-plan", fixture=True)
for tier in report["tiers"]:
    print(tier["tier"], tier["side"], "RR", tier["risk_reward"])
```

```bash
ai-pro-trading case report structured-trade-plan --fixture
```

## Live mode

```python
report = load_case_report(
    "structured-trade-plan",
    adapter=BinanceRestAdapter(),
    inputs={"symbols": ["ETHUSDT"], "tier": "balanced"},  # tier optional
)
```

`inputs.tier` (or `preset.tier`) limits output to one tier when set.

## Pipeline

1. Fetch up to `lookback_bars` (default 200) klines for the symbol.
2. Compute trend bias from EMA(short/mid/long) alignment.
3. Compute support/resistance from rolling `lookback_window` (default 50) lows/highs.
4. Compute ATR(14) from current bar.
5. For each tier, derive entry zone, stop, and target as ATR multiples and
   compute risk:reward.

## Output (envelope shared with all report cases)

```json
{
  "case_id": "structured-trade-plan",
  "case_type": "trade_plan_report",
  "generated_at": "...",
  "signals": [
    {"symbol": "ETHUSDT", "side": "long", "score": 1.93, "reason": "balanced tier ...", "tier": "balanced"}
  ],
  "summary": "ETHUSDT — bias=long; last=...; ATR=...; support=...; resistance=...; 3 tier(s) generated.",
  "recommended_next_step": "Pick one tier and run paper-stage ...",
  "symbol": "ETHUSDT",
  "bias": "long",
  "ema_detail": {...},
  "support_resistance": {...},
  "atr": ...,
  "tiers": [
    {"tier": "conservative", "side": "long", "entry_low": ..., "entry_high": ..., "stop_price": ..., "target_price": ..., "risk_reward": 1.29, "atr_used": ...},
    ...
  ]
}
```

## Data lineage

- Source case: `private-skills-for-ai-pro/.../usage_project_cases/structured-trade-plan`
- Indicators: `library.features.indicators.ema`, `library.features.indicators.atr`
- Runtime: report-runner only.
