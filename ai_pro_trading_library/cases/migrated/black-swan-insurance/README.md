# Black Swan Hedge Switch

> case_id: `black-swan-insurance` · case_type: `scanner_report` · status: `migrated`

12-factor panic detector + tier-based hedge allocation plan. Replaces the old `panic_detector.py` + `news_event_detector.py` subprocess pipeline with a JSON-input contract.

**Planning only** — no orders are placed. The agent must hand the allocation off to a BAW swap workflow if `tier` crosses `alert`.

## Input shape

```json
{
  "market_signals": {
    "btc_drawdown_1h_pct": 6.5,
    "btc_drawdown_24h_pct": 15.5,
    "top10_cap_drawdown_24h_pct": 12.0,
    "stablecoin_depeg_bps": 60.0,
    "vix_proxy_change_24h_pct": 45.0,
    "spy_qqq_drawdown_1h_pct": 3.0,
    "gold_change_24h_pct": 3.0,
    "defi_tvl_drawdown_24h_pct": 8.0
  },
  "narrative_signals": {
    "negative_keyword_hit_count": 3,
    "social_sentiment_min": -0.45,
    "smart_money_inflow_top10_avg_usdc": -300000,
    "hotrank_majors_missing": 1,
    "ai_tradesignal_bearish_pct": 0.65
  },
  "portfolio": {
    "total_value_usdc": 10000.0,
    "holdings": [{"symbol": "BTC", "value_usdc": 6000}]
  }
}
```

`narrative_signals` is optional; missing entries score 0.

## Pipeline

1. Each price signal scores 0..1 between its `panic_threshold` and `extreme_threshold`.
2. Narrative signals score the same way; some use `direction: "negative"` (e.g. sentiment ≤ threshold).
3. `panic_score = min(sum(score * weight), 100)`.
4. Tier mapping (default thresholds):

   | panic_score | tier | allocation (holdings / USDC / GLDon / SQQQon) |
   |---|---|---|
   | < 30 | calm | 100% / 0 / 0 / 0 |
   | 30–60 | alert | 50% / 30% / 20% / 0 |
   | 60–80 | black_swan | 20% / 20% / 20% / 40% |
   | ≥ 80 | extreme | 0 / 10% / 30% / 60% |

5. If `portfolio.total_value_usdc` is supplied, also report dollar allocation.

## Output

```json
{
  "case_id": "black-swan-insurance",
  "panic_score": 68.66,
  "tier": "black_swan",
  "tier_max_score": 80,
  "recommended_allocation_pct": {"holdings": 0.2, "USDC": 0.2, "GLDon": 0.2, "SQQQon": 0.4},
  "recommended_allocation_usdc": {"holdings": 2000.0, "USDC": 2000.0, "GLDon": 2000.0, "SQQQon": 4000.0},
  "price_signal_breakdown": [...],
  "narrative_signal_breakdown": [...],
  "unwind_conditions": [
    "BTC 4h rebound >= +6% AND back above pre-trigger price",
    "panic_score < 25 for 3 consecutive 1h bars",
    ...
  ]
}
```

## Data lineage

- Source case: `private-skills-for-ai-pro/.../usage_project_cases/black-swan-insurance`
- Execution: BAW swap and watcher integration belong in a separate skill.
