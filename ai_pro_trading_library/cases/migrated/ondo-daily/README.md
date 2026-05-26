# Ondo Tokenized Securities Heat & Movers Scanner

> case_id: `ondo-daily` · case_type: `scanner_report` · status: `migrated`

Daily Ondo tokenized-equity scan. Replaces the old `binance-pro-cli web3 tokenized` subprocess pipeline with a JSON-input contract.

## Input shape

```json
{
  "ondo_market_status": "open",
  "ondo_universe": [
    {"ticker": "QQQ", "symbol": "QQQon", "chain": "solana", "contract": "...", "multiplier": 1.0, "volume_24h": 1200000.0}
  ],
  "alpha_hotrank": [{"rank": 1, "symbol": "QQQon", "score": 0.95}],
  "movers_24h": [{"symbol": "TSLAon", "change_pct": 6.2, "premium_pct": 0.4}]
}
```

## Quickstart

```python
from ai_pro_trading_library.cases import load_case_report
report = load_case_report("ondo-daily", fixture=True)
for c in report["candidates"]:
    print(c["symbol"], c["score"], c["sources"])
```

```bash
ai-pro-trading case report ondo-daily --fixture
```

## Pipeline

1. Filter hotrank top-`hotrank_top_n_threshold`; map to ondo_universe entries.
2. Filter 24h movers by `abs_change_floor_pct`, take top-`mover_top_n_threshold`.
3. Score each candidate by rank position; add `cross_source_bonus` if appearing in both lists.
4. Return top-N with full metadata.

## Output

```json
{
  "case_id": "ondo-daily",
  "ondo_market_status": "open",
  "ondo_universe_size": 6,
  "candidates": [
    {
      "symbol": "TSLAon",
      "ticker": "TSLA",
      "chain": "solana",
      "score": 32.0,
      "sources": ["hotrank", "mover"],
      "hotrank": {"rank": 2, "score": 0.91},
      "mover": {"change_pct": 6.2, "premium_pct": 0.4}
    }
  ],
  "all_ranked": [...]
}
```

## Data lineage

- Source case: `private-skills-for-ai-pro/.../usage_project_cases/ondo-daily`
- Future adapter: `library.data.adapters.binance_pro` web3 tokenized endpoints (not in this case).
