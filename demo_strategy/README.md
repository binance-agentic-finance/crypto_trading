# `demo_strategy/` — 模块化策略参考实现

两条实战案例，重构为**同一套 7 模块骨架**，用来演示如何把一个策略拆成清晰的、可替换的组件。

## 策略清单

| 策略 | 目录 | 类型 | 核心信号 |
|---|---|---|---|
| BTC 多因子趋势 | [`btc_multi_factor_trend/`](btc_multi_factor_trend/) | 单标的 · 三档 mode | 6 因子（EMA/RSI/MACD/衍生品/ATR/多周期共振）|
| Binance 广场热点扫描 | [`square_buzz_screener/`](square_buzz_screener/) | 多标的 · 注意力驱动 | 9 因子（Square 热度 + 市场验证）|

## 7 模块骨架（每个策略都是这套结构）

```
demo_strategy/<strategy>/
├── README.md               # 策略概览、mode/参数、运行方式
├── config/
│   └── config.json         # 所有可调参数（阈值、时间框架、mode 定义、执行开关）
├── strategy/
│   ├── __init__.py         # 暴露给 run.py 用的 strategy 门面
│   ├── m01_universe.py     # ① UNIVERSE  — 圈定候选标的（固定 / 扫描 / 注意力）
│   ├── m02_data.py         # ② DATA      — 抓 K 线、ticker、衍生品、行情数据
│   ├── m03_signals.py      # ③ SIGNALS   — 计算技术指标 / 社群信号（无阈值判定）
│   ├── m04_scoring.py      # ④ SCORING   — 因子打分 + verdict 分级
│   ├── m05_decision.py     # ⑤ DECISION  — 方向 + 仓位大小 + 止损止盈参数
│   ├── m06_execution.py    # ⑥ EXECUTION — 下单 / dry-run / spot vs futures
│   └── m07_report.py       # ⑦ REPORT    — 生成结构化输出（JSON + 人读文本）
└── run.py                  # 入口：串起 7 模块，命令行 flag → config overlay
```

## 为什么这么切分

| 模块 | 输入 | 输出 | 变动频率 |
|---|---|---|---|
| **① universe** | mode / 参数 | 待评估的 symbol 列表 | 低（几乎不改）|
| **② data** | symbol × timeframe | K 线 / ticker / funding / OI | 底层稳定 |
| **③ signals** | K 线 & 数据 | 一堆技术数值（EMA/RSI/…），**无阈值判断** | 中（指标增删）|
| **④ scoring** | signals + tiers cfg | tier 分 + 总分 + verdict | **高**（调参重灾区）|
| **⑤ decision** | verdict + mode | direction + size + stop/tp | 中（mode 定义）|
| **⑥ execution** | decision + 开关 | 下单结果 / dry-run | 稳定 |
| **⑦ report** | 全流程结果 | JSON + 文本 report | 独立演化 |

**核心设计取舍**：
- **signals 与 scoring 严格分离** — signals 只算数字，scoring 只按阈值给分。调参不会碰指标计算。
- **decision 独立** — 一份 signals+score 结果可以在不同 mode（defensive/balanced/aggressive）下产出完全不同的仓位与止损，无需重跑 signals。
- **每模块单文件** — 200 行以内，能一屏看完，不藏 helper。
- **模块间只传纯 dict / dataclass** — 不共享全局状态，便于单独测试和替换。

## 共享工具

[`_shared/`](_shared/) 里放两个策略都会用的公共函数（config 加载、bar-count 建议、格式化 output）。策略特定逻辑一律**不放** `_shared`。

## 运行

```bash
cd demo_strategy/btc_multi_factor_trend
python3 run.py                                       # balanced mode，dry-run
python3 run.py --mode defensive                      # 换 mode
python3 run.py --symbols BTCUSDT ETHUSDT             # 换标的
python3 run.py --execute --live --mode balanced      # 真实下单

cd demo_strategy/square_buzz_screener
python3 run.py                                       # 扫广场 + 市场验证
python3 run.py --locales en                          # 只扫英文
python3 run.py --no-deep                             # 跳过深度抓取（更快）
python3 run.py --execute --live --min-verdict STRONG_CANDIDATE
```

两个策略都会写 JSON 到 `~/.openclaw/workspace/<strategy_name>/pipeline_result.json`。

## 依赖

- Python 3.10+
- `atomic_strategy_lib`（在 `crypto_trading/atomic_compat/atomic_strategy_lib/`）—— 所有 block 都从这里 import
- 无需 pip install，`run.py` 会自动把 atomic_strategy_lib 挂到 `sys.path`

## 快速对比：两个策略的 7 模块差异

| 模块 | btc_multi_factor_trend | square_buzz_screener |
|---|---|---|
| ① universe | 固定 `BTCUSDT` (可扩) | 从 Binance Square 抓热点 token |
| ② data | 4h/1h/15m/1d K 线 + funding + OI | ticker + K 线 + OI + funding + Square 热度元数据 |
| ③ signals | EMA/RSI/MACD/ATR/funding/OI/共振 (6 组) | 上述 + 广场重复度 / EN/CN 交集 / 深度信号 (共 9 组) |
| ④ scoring | 6-tier hierarchical，阈值来自 config.scoring.tiers | 9-tier hierarchical，attention factor 单独加权 |
| ⑤ decision | 3 mode（defensive/balanced/aggressive）决定 leverage/stop | verdict-gate + 双向（LONG/SHORT/WATCH）|
| ⑥ execution | spot 或 futures，`STOP_LOSS_LIMIT` / `STOP_MARKET` | 同左，`--min-verdict` 门槛更严 |
| ⑦ report | 排序 leaderboard + tier 明细 | Social hotspot brief + trading signal summary |

两侧模块名一致，读者可以直接对比"同一模块在两个策略里有何差异"。
