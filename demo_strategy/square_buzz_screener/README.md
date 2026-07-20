# Binance 广场热点扫描 · Square Buzz Screener

**注意力驱动**策略：抓 Binance Square（`Most Searched` / `Rapid Riser` / trending hashtags）→ 去重与排名 → **每个候选 token 做完整技术验证** → 打分 → 输出 leaderboard 或（可选）下单。

## 逻辑一句话

社群热度 = 潜在交易机会；但**只有热度 + 技术验证同时通过**才产 verdict。9 因子分层打分。

## 9 组因子

| Tier | 因子 | 类别 |
|---|---|---|
| T1 | **Square 出现频次**（EN/CN 交集 + section 交集）| Attention |
| T2 | **深度信号**（每个 hashtag 的详情页情绪）| Attention |
| T3 | EMA 排列 | 技术 |
| T4 | RSI 区间 | 技术 |
| T5 | MACD 动能 | 技术 |
| T6 | 衍生品（funding + OI）| Derivatives |
| T7 | ATR 波动 | Technical |
| T8 | 多周期共振 | Technical |
| T9 | 成交量放量 | Volume |

## Verdicts + Direction

除了 `SKIP` / `WATCHLIST` / `CANDIDATE` / `STRONG_CANDIDATE`，还额外区分 `AVOID`（热度高但技术转弱），并给出 `LONG` / `SHORT` / `WATCH` 方向偏置。

## 7 模块结构

```
square_buzz_screener/
├── config/
│   └── config.json                # 阈值 + attention 权重 + 执行开关
├── strategy/
│   ├── __init__.py                # 门面
│   ├── m01_universe.py            # 从 Binance Square 抓热点 → dedup → 排名
│   ├── m02_data.py                # 每候选：ticker + K 线 + funding + OI
│   ├── m03_signals.py             # 9 组因子数值（含 attention）
│   ├── m04_scoring.py             # 9-tier hierarchical
│   ├── m05_decision.py            # verdict × direction bias
│   ├── m06_execution.py           # 同 btc 系（复用 pattern）
│   └── m07_report.py              # social-hotspot-brief 结构化输出
└── run.py                         # CLI 入口
```

## 运行

```bash
# 默认（扫 EN + CN，深度模式）
python3 run.py

# 只扫英文
python3 run.py --locales en

# 跳过深度抓取（快，但信号量少）
python3 run.py --no-deep

# 真实下单
python3 run.py --execute --live --min-verdict STRONG_CANDIDATE
```

输出：`~/.openclaw/workspace/square_buzz_screener/pipeline_result.json`

## 与 btc_multi_factor_trend 的对比

| 模块 | btc | square |
|---|---|---|
| ① universe | 固定 BTC | Square 抓热点 → 排名 |
| ② data | 4 个 timeframe + funding + OI | 同左，但对每个 token（可能 5-10 个） |
| ③ signals | 6 组 | 9 组（多 3 个：attention 频次、深度信号、成交量）|
| ④ scoring | 6-tier + 4 verdict | 9-tier + 5 verdict（多一个 AVOID）|
| ⑤ decision | 3 mode 决定杠杆 | verdict gate + direction bias |
| ⑥ execution | 同一套 order helper | 同左 |
| ⑦ report | leaderboard + tier detail | social-hotspot-brief（LLM 友好）|

**共用 pattern**：`m02_data` / `m06_execution` / `m07_report` 逻辑相仿，只是数据源和输出模板不同。7 模块骨架的价值就在于**同一份心智模型能同时读懂两个策略**。
