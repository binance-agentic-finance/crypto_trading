# BTC 多因子趋势 · Multi-Factor Trend

从四个前置技巧（`btc-trader` / `btc-contract` / `btc-usdt-swap-defensive-ai` / `small-account-600u-pnl-percent-v1`）**融合**出来的 BTC 趋势跟随策略。适用于 BTC 主线交易，可扩到 ETH。

## 逻辑一句话

用 **6 组因子（EMA/RSI/MACD/衍生品/ATR/多周期共振）**分层打分，得到 4 档 verdict（`STRONG_CANDIDATE` / `CANDIDATE` / `WATCHLIST` / `SKIP`）；用户选 mode（`defensive`/`balanced`/`aggressive`）来决定杠杆、止损宽度、风险 %。

## 三档 Mode

| Mode | 杠杆 | 止损 | 风险/仓 | 适用 |
|---|---|---|---|---|
| defensive | 1-3× | 8% | 2% | 慢牛 / 高波动 |
| balanced | 2-5× | 5% | 3% | 默认 |
| aggressive | 5-10× | 3% | 5% | 高信念趋势起点 |

## 6 组因子打分

| Tier | 因子 | 最大 +score | 最大 -score |
|---|---|---|---|
| T1 | **EMA 排列**（20/60/200 全对齐 / 金叉）| +3 | -2 |
| T2 | **RSI 区间**（正中间为佳）| +2 | -1 |
| T3 | **MACD 动能**（histogram 方向 + 强度）| +2 | -2 |
| T4 | **衍生品**（funding squeeze / OI 建仓 / crowded）| +2 | -2 |
| T5 | **ATR 波动**（expanding / contracting）| +1 | -1 |
| T6 | **多周期共振**（1h / 4h / 1d 方向一致）| +2 | -1 |
| **合计** |  | **+12** | **-9** |

**Verdict 门槛**：`STRONG_CANDIDATE ≥ 10`  ·  `CANDIDATE ≥ 6`  ·  `WATCHLIST ≥ 2`  ·  否则 `SKIP`。

## 7 模块结构

```
btc_multi_factor_trend/
├── config/
│   └── config.json                # 所有阈值 / mode 参数 / 执行开关
├── strategy/
│   ├── __init__.py                # 门面：run(cfg, cli_kw) → pipeline_result
│   ├── m01_universe.py            # 固定 BTCUSDT / 可扩多标的
│   ├── m02_data.py                # 4h/1h/15m/1d 多周期 K 线 + funding + OI
│   ├── m03_signals.py             # 6 组因子数值计算（不打分）
│   ├── m04_scoring.py             # 6-tier hierarchical + verdict gate
│   ├── m05_decision.py            # direction + size + stop（按 mode）
│   ├── m06_execution.py           # market_order + stop_order（spot/futures，dry-run）
│   └── m07_report.py              # 排序 leaderboard + tier 明细
└── run.py                         # CLI 入口
```

## 运行

```bash
# 默认（balanced，dry-run）
python3 run.py

# 换 mode
python3 run.py --mode defensive
python3 run.py --mode aggressive

# 换标的（扩展多标的）
python3 run.py --symbols BTCUSDT ETHUSDT

# 真实下单（两个开关都要传）
python3 run.py --execute --live --mode balanced
```

输出：`~/.openclaw/workspace/btc_multi_factor_trend/pipeline_result.json`。

## 输出结构

```jsonc
{
  "strategy": "btc_multi_factor_trend",
  "mode": "balanced",
  "generated_at": "2026-07-20T14:00:00+00:00",
  "results": [
    {
      "symbol": "BTCUSDT",
      "price": 71234.5,
      "signals": { /* 6 组数值 */ },
      "score": {
        "total": 11,
        "tiers": [
          { "name": "ema_trend",    "score": 3, "reason": "…" },
          { "name": "rsi_zone",     "score": 2, "reason": "…" },
          { "name": "macd_momentum","score": 2, "reason": "…" },
          { "name": "derivatives",  "score": 2, "reason": "…" },
          { "name": "volatility",   "score": 1, "reason": "…" },
          { "name": "resonance",    "score": 1, "reason": "…" }
        ],
        "verdict": "STRONG_CANDIDATE"
      },
      "decision": {
        "direction": "LONG",
        "size_usdt": 300,
        "leverage": 3,
        "stop_price": 67672.7,
        "stop_pct": 5.0,
        "risk_usdt": 15
      },
      "execution": { /* 若 --execute */ }
    }
  ]
}
```
