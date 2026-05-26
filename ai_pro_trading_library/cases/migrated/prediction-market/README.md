# Prediction Market Edge Scanner

> case_id: `prediction-market` · case_type: `scanner_report` · status: `migrated`

Predict.fun crypto price-prediction Edge scanner. Computes 7-factor true probability `P_true`, compares with implied yes/no prices to identify positive Edge, applies quarter-Kelly sizing with portfolio gates.

This case **does not** place orders — it produces structured BUY/SKIP decisions; actual execution stays in `baw prediction trade place-order` (out of library scope).

## Input shape

```json
{
  "capital_usdc": 1000.0,
  "prediction_markets": [
    {
      "market_id": "BTC-15min-77924",
      "symbol": "BTCUSDT",
      "interval": "15min",
      "strike_price": 77924.0,
      "current_spot": 78050.0,
      "end_time_ms": 1718000000000,
      "yes_price": 0.45,
      "no_price": 0.55,
      "liquidity_usdc": 25000.0,
      "funding_rate": -0.0007,
      "long_short_ratio": 0.65,
      "orderbook_imbalance": 0.7,
      "volume_zscore": 2.2
    }
  ],
  "bars_by_symbol": {"BTCUSDT": [...]}
}
```

## Pipeline

1. Filter markets by `min_liquidity_usdc`.
2. For each remaining market, compute 7 weighted signals from the snapshot + klines:
   `rsi`, `macd`, `ema_cross`, `funding_rate`, `long_short_ratio`, `orderbook_imbalance`, `volume_breakout`
3. `P_true_yes = 0.5 + weighted_sum / (2 * total_weight)` (clipped to [0.01, 0.99]).
4. `Edge_yes = P_true_yes - yes_price`, similarly for `Edge_no`.
5. If `max(edge_yes, edge_no) >= edge_threshold` → BUY that side.
6. Sizing: `quarter_kelly = 0.25 * full_kelly`, capped at `single_market_max_pct`.
7. Cap actionable list at `daily_market_max_count`; scale total down to `portfolio_max_pct`.

## Output

```json
{
  "case_id": "prediction-market",
  "capital_usdc": 1000.0,
  "actionable_count": 2,
  "actionable": [
    {
      "market_id": "BTC-15min-77924",
      "side": "yes",
      "edge_yes": 0.346,
      "kelly_full": 0.421,
      "kelly_scaled": 0.05,
      "size_usdc": 50.0,
      "p_true_yes": 0.796,
      "signal_components": {...}
    }
  ],
  "all_markets": [...]
}
```

## Data lineage

- Source case: `private-skills-for-ai-pro/.../usage_project_cases/prediction_market`
- Indicators: `library.features.indicators.{rsi, macd, ema}`
- Execution: BAW wallet integration belongs in a separate skill.
