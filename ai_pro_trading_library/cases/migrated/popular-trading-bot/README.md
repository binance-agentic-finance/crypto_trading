# Trading Bot Simulator (planning)

> case_id: `popular-trading-bot` · case_type: `scanner_report` · status: `migrated`

Compares Binance's 7 trading-bot archetypes (spot-grid / futures-grid / spot-dca / smart-portfolio / futures-arbitrage / futures-algo-orders / auto-invest) against a market regime classification. Outputs ready-to-use parameter templates for the top-3 fits.

This case **does not** simulate or execute the bots — that belongs to a future `runtime.bots` package.

## Input shape

```json
{
  "market_regime": "ranging",   // optional; auto-detected from bars when omitted
  "capital_usdc": 1000.0,
  "symbol": "BTCUSDT",
  "bars": [{"open": ..., "high": ..., "low": ..., "close": ..., "volume": ..., "timestamp": ...}]
}
```

`market_regime` ∈ `{ranging, trending_up, trending_down, high_volatility}`.

When `market_regime` is missing and `bars` is supplied, regime is auto-detected from EMA20/EMA50 alignment + ATR/price ratio.

## Pipeline

1. Determine `market_regime` (explicit or auto-detected).
2. Look up each of the 7 bots' fitness in `preset.regime_fitness[bot_id][regime]`.
3. Generate parameter template per bot, capped by `preset.risk_caps`.
4. Sort by fitness; return top-3 plus full list.

## Output

```json
{
  "case_id": "popular-trading-bot",
  "regime": "ranging",
  "regime_detection_detail": {"ema_20": ..., "ema_50": ..., "atr_pct": ..., "last_price": ...},
  "capital_usdc": 1000.0,
  "symbol": "BTCUSDT",
  "risk_caps": {"account_drawdown_pct": 10.0, "max_leverage": 5, "require_confirm": true},
  "top3": [
    {
      "bot_id": "spot-grid",
      "regime_score": 95,
      "regime_fit": "excellent",
      "template": {"levels": 20, "lower_pct": -10, "upper_pct": 10, "leverage": 1, "stop_drawdown_pct": 10, "capital_usdc": 1000, "symbol": "BTCUSDT", "account_drawdown_pct": 10.0, "require_confirm": true}
    }
  ],
  "all_bots": [...]
}
```

## Data lineage

- Source case: `private-skills-for-ai-pro/.../usage_project_cases/popular_trading_bot`
- Indicators: `library.features.indicators.{ema, atr}` for regime auto-detection.
- Live execution: belongs to `runtime.bots` (not in this case).
