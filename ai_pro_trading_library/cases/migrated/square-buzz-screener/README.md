# Binance Square Hot Picks Analyzer

> case_id: `square-buzz-screener` · case_type: `scanner_report` · status: `migrated`

Social-buzz ranker. Replaces the old `binance_square_probe.py` scraper with a
**JSON snapshot input** contract: the AI agent (or a future pro adapter) supplies
the heat snapshot, and this case dedupes / ranks / validates against kline data.

## Input shape

```json
{
  "square_snapshot": {
    "en": {
      "most_searched": ["BTC", "ETH"],
      "rapid_riser": ["SOL", "WIF"],
      "hashtags": [{"name": "memecoins", "tokens": ["WIF", "PEPE"]}]
    },
    "zh": { "...": "same shape" }
  },
  "bars_by_symbol": {
    "BTCUSDT": [{"open": ..., "high": ..., "low": ..., "close": ..., "volume": ..., "timestamp": ...}, ...]
  }
}
```

Symbols without a quote suffix get `USDT` appended automatically.

## Quickstart

```python
from ai_pro_trading_library.cases import load_case_report

report = load_case_report("square-buzz-screener", fixture=True)
for c in report["candidates"]:
    print(c["symbol"], c["verdict"], c.get("composite"))
```

```bash
ai-pro-trading case report square-buzz-screener --fixture
ai-pro-trading case report square-buzz-screener --inputs path/to/snapshot.json
```

## Pipeline

1. Aggregate social score per symbol from `most_searched`, `rapid_riser`, `hashtags`.
2. Apply EN/ZH overlap bonus and section overlap bonus.
3. Per symbol, fetch klines via adapter or read `bars_by_symbol`. Compute:
   - Trend (EMA20 vs EMA50, price-vs-EMA20)
   - Momentum (RSI14 zone)
   - Volume (rolling z-score over 30 bars)
4. Composite = `0.5 * social_normalized + 0.5 * tech`.
5. Verdict tier: STRONG_CANDIDATE / CANDIDATE / WATCHLIST / SKIP / AVOID.

Symbols without sufficient kline history are tagged `INSUFFICIENT_DATA` so the
caller can still see they were socially trending.

## Output

```json
{
  "case_id": "square-buzz-screener",
  "signals": [...],
  "summary": "Scored N candidate(s) from social snapshot. K STRONG_CANDIDATE.",
  "candidates": [
    {
      "symbol": "BTCUSDT",
      "verdict": "STRONG_CANDIDATE",
      "social_score": 14.5,
      "social_score_normalized": 48.3,
      "tech_score": 80.0,
      "composite": 64.2,
      "bias": "long",
      "tech_detail": {"trend": 100, "momentum": 60, "volume": 100, "rsi": 65.2, "volume_zscore": 2.1, "bias": "long"},
      "sections": ["most_searched", "rapid_riser", "hashtags"],
      "locales": ["en", "zh"],
      "recurrence": 5
    }
  ]
}
```

## Data lineage

- Source case: `private-skills-for-ai-pro/.../usage_project_cases/square-buzz-screener`
- Indicators: `library.features.indicators.{ema, rsi, rolling_zscore}`
- No subprocess calls; no `binance-pro-cli` dependency in the case itself.
