# Technical Indicator Overview

> case_id: `kline-indicator-checkup` · case_type: `scanner_report` · status: `migrated`

Read-only multi-indicator scan covering trend, momentum, and volatility for one
or more symbols. Produces a 0-100 composite score per symbol plus a verdict
(STRONG_TREND / ACTIONABLE / WATCHLIST / WEAK / AVOID).

## Quickstart

```python
from ai_pro_trading_library.cases import load_case_report

# Fixture mode (no network):
report = load_case_report("kline-indicator-checkup", fixture=True)
print(report["summary"])
for entry in report["results"]:
    print(entry["symbol"], entry["score"], entry["verdict"])
```

```bash
# CLI fixture run:
ai-pro-trading case report kline-indicator-checkup --fixture
```

## Live mode

Pass any object implementing the `MarketDataAdapter` Protocol (e.g.
`SyntheticAdapter`, `BinanceRestAdapter`, `LocalParquetAdapter`):

```python
from ai_pro_trading_library.library.data.adapters.binance_rest import BinanceRestAdapter
adapter = BinanceRestAdapter()
report = load_case_report(
    "kline-indicator-checkup",
    adapter=adapter,
    inputs={"symbols": ["BTCUSDT", "ETHUSDT"]},
)
```

`inputs.symbols` overrides the `preset.symbols` list when provided. The case
fetches `lookback_bars` (default 200) klines per symbol at the preset timeframe
(default 1h).

## Pipeline

1. For each symbol, fetch klines via the adapter or read OHLCV from
   `inputs.bars_by_symbol[symbol]` (fixture mode).
2. Compute three pillars from `library.features.indicators`:
   - **Trend** (40%): EMA(20/50/200) alignment and price-vs-MA, MACD signal cross.
   - **Momentum** (30%): RSI(14) zone (oversold/neutral/overbought).
   - **Volatility** (30%): ATR(14)/close ratio, Bollinger Band width.
3. Aggregate to a composite 0-100 score and assign a verdict.

## Output (envelope shared with all report cases)

```json
{
  "case_id": "kline-indicator-checkup",
  "case_type": "scanner_report",
  "generated_at": "...",
  "inputs": {"preset": {...}, "external": {...}},
  "signals": [{"symbol": "BTCUSDT", "side": "long", "score": 71.9, "reason": "..."}],
  "summary": "Analyzed 1 symbol(s). Top: BTCUSDT (score=71.9, verdict=ACTIONABLE).",
  "recommended_next_step": "Feed top symbols into structured-trade-plan or three-tier-strategy ...",
  "results": [
    {
      "symbol": "BTCUSDT",
      "score": 71.9,
      "verdict": "ACTIONABLE",
      "pillars": {
        "trend":      {"score": 100, "ema_short": ..., "ema_mid": ..., "ema_long": ..., "ema_aligned": true, "macd_bullish": true, "price_above_ema_long": true},
        "momentum":   {"score": 41.0, "rsi": 49.5, "zone": "neutral"},
        "volatility": {"score": 50, "atr": ..., "atr_pct": ..., "bb_width": ...}
      }
    }
  ]
}
```

## Data lineage

- Source case: `private-skills-for-ai-pro/.../usage_project_cases/kline-indicator-checkup`
- Indicators: `ai_pro_trading_library.library.features.indicators`
- Runtime: report-runner only (no execution path).
