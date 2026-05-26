# Volume Anomaly Daily Report

> case_id: `volume-surge-daily-report` · case_type: `scanner_report` · status: `migrated`

Multi-symbol scanner. For each candidate symbol, computes a 0-100 composite
score from three pillars:

- **Volume** (50%): rolling z-score of latest volume vs `volume_window` baseline.
- **Alignment** (25%): EMA(short) vs EMA(long) and price-vs-EMA(short).
- **Breakout** (25%): close vs rolling N-bar high.

Symbols are ranked, capped at `max_candidates`, and tagged
STRONG_CANDIDATE / CANDIDATE / WATCHLIST / SKIP / AVOID.

## Quickstart

```python
from ai_pro_trading_library.cases import load_case_report

report = load_case_report("volume-surge-daily-report", fixture=True)
for row in report["results"]:
    print(row["symbol"], row["score"], row["verdict"], "z=", row["volume_zscore"])
```

```bash
ai-pro-trading case report volume-surge-daily-report --fixture
```

## Live mode

```python
report = load_case_report(
    "volume-surge-daily-report",
    adapter=BinanceRestAdapter(),
    inputs={"symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]},
)
```

`inputs.symbols` overrides `preset.symbols`.

## Pipeline

1. For each symbol, fetch `lookback_bars` (default 200) klines.
2. Compute volume z-score over `volume_window` (default 50).
3. Compute EMA(short)/EMA(long) alignment.
4. Compare last close to rolling `breakout_window` (default 30) high.
5. Aggregate via `weights` and assign verdict.

## Output

```json
{
  "case_id": "volume-surge-daily-report",
  "case_type": "scanner_report",
  "signals": [
    {"symbol": "BTCUSDT", "side": "long", "score": 85.0, "reason": "verdict=STRONG_CANDIDATE; vol_z=6.95; align=bullish; breakout=fresh_breakout"}
  ],
  "summary": "Scanned 3 symbol(s); kept top 3. 1 STRONG_CANDIDATE(s).",
  "results": [
    {
      "symbol": "BTCUSDT",
      "score": 85.0,
      "verdict": "STRONG_CANDIDATE",
      "volume_zscore": 6.95,
      "volume_score": 100,
      "alignment": {"label": "bullish", "score": 100},
      "breakout": {"label": "fresh_breakout", "score": 100, "rolling_high": ..., "current": ..., "breakout_pct": ...}
    },
    ...
  ]
}
```

## Data lineage

- Source case: `private-skills-for-ai-pro/.../usage_project_cases/volume-surge-daily-report`
- Indicators: `library.features.indicators.{ema, rolling_zscore}`
- Runtime: report-runner only.
