# US Stock Earnings Opportunity Scanner

> case_id: `earnings-tradfi-scanner` · case_type: `scanner_report` · status: `migrated`

Cross-references an external earnings calendar against TradFi perpetual contracts and Ondo tokenized equities. **The case never WebSearches** — the agent supplies the calendar JSON per `macro_layer.md`'s search contract.

## Input shape

```json
{
  "as_of": "2026-05-22",
  "earnings_calendar": [
    {
      "ticker": "NVDA",
      "company": "NVIDIA Corp",
      "report_date": "2026-05-28",
      "period": "Q1 2027",
      "eps_estimate": 0.85,
      "eps_prior": 0.61,
      "market_cap_category": "mega",
      "sector": "semiconductor"
    }
  ],
  "tradfi_universe": ["NVDA", "TSLA", "AAPL", "QQQ", "SPY"],
  "ondo_universe": [{"ticker": "NVDA", "symbol": "NVDAon", "chain": "solana"}]
}
```

## Quickstart

```python
report = load_case_report("earnings-tradfi-scanner", fixture=True)
for c in report["candidates"]:
    print(c["ticker"], c["composite"], c["venues"])
```

## Pipeline

1. Filter calendar to entries within `max_horizon_days` (default 7) of `as_of`.
2. Match each ticker to TradFi perp universe and Ondo universe.
3. Composite score components:
   - days_to_report (closer = higher)
   - eps_surprise_potential (estimate vs prior)
   - market_cap_tier (mega > large > mid > small)
   - sector_alpha (technology/semiconductor > finance > consumer)
   - venue_bonus (both > tradfi_only > ondo_only)
4. Rank, return top-N.

## Output

```json
{
  "case_id": "earnings-tradfi-scanner",
  "as_of": "2026-05-22",
  "horizon_days": 7,
  "candidates": [
    {
      "ticker": "NVDA",
      "company": "NVIDIA Corp",
      "report_date": "2026-05-28",
      "days_to_report": 6,
      "venues": ["tradfi_perp", "ondo"],
      "ondo_symbol": "NVDAon",
      "score_breakdown": {"time_score": ..., "eps_surprise_score": ..., "market_cap_score": ..., "sector_score": ..., "venue_score": ...},
      "composite": 86.09
    }
  ]
}
```

## Data lineage

- Source case: `private-skills-for-ai-pro/.../usage_project_cases/earnings-tradfi-scanner`
- Earnings calendar: produced by AI agent via WebSearch per `macro_layer.md`.
