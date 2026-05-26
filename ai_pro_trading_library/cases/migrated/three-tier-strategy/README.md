# Multi-Scenario Market Analysis

> case_id: `three-tier-strategy` · case_type: `trade_plan_report` · status: `migrated`

Three regime-based scenarios for one symbol. Each scenario carries its own
leverage cap, stop multiple, target multiple, risk-per-trade ceiling, and a
key-factor list. Designed for liquid majors (BTC/ETH).

Inactive scenarios are returned with `side="hold"` so the user can still
compare; only scenarios meeting their `min_trend_score` gate emit signals.

## Quickstart

```python
from ai_pro_trading_library.cases import load_case_report

report = load_case_report("three-tier-strategy", fixture=True)
for s in report["scenarios"]:
    print(s["scenario"], "active=", s["active"], "RR=", s["risk_reward"])
```

```bash
ai-pro-trading case report three-tier-strategy --fixture
```

## Live mode

```python
report = load_case_report(
    "three-tier-strategy",
    adapter=BinanceRestAdapter(),
    inputs={"symbols": ["BTCUSDT"], "tier": "balanced"},  # tier optional
)
```

## Pipeline

1. Fetch up to `lookback_bars` (default 250) klines.
2. Compute trend score (0-100) from EMA20/50/200 alignment + price-vs-MA.
3. Compute RSI(14), ATR(14), and rolling structural support/resistance.
4. For each scenario, gate on `min_trend_score`, then derive entry, stop,
   target, leverage, and risk_pct from the scenario's ATR multiples.

## Output

```json
{
  "case_id": "three-tier-strategy",
  "case_type": "trade_plan_report",
  "signals": [
    {"symbol": "BTCUSDT", "side": "long", "score": 1.6, "scenario": "conservative", "reason": "..."}
  ],
  "summary": "BTCUSDT — bias=long; trend_score=100; RSI=56.0; ATR=...; 3/3 scenario(s) active.",
  "symbol": "BTCUSDT",
  "bias": "long",
  "trend_score": 100,
  "rsi": 56.0,
  "atr": ...,
  "support_resistance": {"support": ..., "resistance": ..., "window": 50},
  "scenarios": [
    {"scenario": "conservative", "active": true, "side": "long", "leverage": 2, "risk_pct": 0.5, "entry_price": ..., "stop_price": ..., "target_price": ..., "risk_reward": 1.6, "atr_used": ..., "key_factors": [...], "rationale": "..."},
    {"scenario": "balanced", ...},
    {"scenario": "aggressive", ...}
  ]
}
```

## Data lineage

- Source case: `private-skills-for-ai-pro/.../usage_project_cases/three-tier-strategy`
- Indicators: `library.features.indicators.{ema, rsi, atr}`
- Runtime: report-runner only.
