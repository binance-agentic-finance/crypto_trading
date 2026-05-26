# Macro Event-Driven TradFi Opportunity Scanner

> case_id: `macro-event-driven-scanner` · case_type: `scanner_report` · status: `migrated`

Cross-references a macro event calendar (FOMC/CPI/NFP/PMI/earnings season) against TradFi perpetuals + Ondo universes.

## Input shape

```json
{
  "as_of": "2026-05-22",
  "macro_events": [
    {
      "event_type": "fomc",
      "event_name": "FOMC Decision",
      "event_date": "2026-06-04",
      "impact_level": "high",
      "consensus": "hold 4.25%",
      "affected_assets": ["QQQ", "SPY", "TLT"],
      "affected_sectors": ["technology", "finance"],
      "notes": "..."
    }
  ],
  "tradfi_universe": ["QQQ", "SPY", ...],
  "ondo_universe": [{"ticker": "QQQ", "symbol": "QQQon", "chain": "solana"}]
}
```

## Pipeline

1. Filter events to those within `max_horizon_days` (default 14) of `as_of`.
2. For each event, fan out to `affected_assets`. Discard those not in tradfi/ondo universes.
3. Per (event, ticker), score = `base_event_points * impact_weight * event_alpha * proximity + venue_bonus`.
4. Aggregate per ticker; rank top-N.

## Output

```json
{
  "case_id": "macro-event-driven-scanner",
  "as_of": "2026-05-22",
  "horizon_days": 14,
  "events_used": [
    {"event_type": "fomc", "event_date": "2026-06-04", "impact_level": "high", "days_to_event": 13, "weight": 0.39}
  ],
  "candidates": [
    {
      "ticker": "QQQ",
      "venues": ["tradfi_perp", "ondo"],
      "ondo_symbol": "QQQon",
      "event_count": 3,
      "score": 137.05,
      "events": [
        {"event_type": "cpi", "event_date": "2026-05-29", "impact_level": "high", "weighted_points": 65.5}
      ]
    }
  ]
}
```

## Data lineage

- Source case: `private-skills-for-ai-pro/.../usage_project_cases/macro-event-driven-scanner`
- Macro event calendar: produced by AI agent via WebSearch per `macro_layer.md`.
