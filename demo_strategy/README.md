# `demo_strategy/` — bdp-ai-trading-bot 契约下的模块化策略

两条实战案例，按 **bdp-ai-trading-bot** 的数据/执行契约设计，可以**直接被 supervisor 注册运行**。

## 关键 vs v1 的区别

v1 版本把每个策略写成"自己抓数据 / 自己下单"的单体脚本 —— 不能在 bdp-ai-trading-bot 里跑。
v2 版本把每个策略拆成**两层**，让 supervisor 层负责数据/下单/状态，模板层只做纯计算。

| 层 | 谁写 | 干什么 | 不能干什么 |
|---|---|---|---|
| **① Template**（block） | 策略作者 | 输入 `StrategyContext` → 返回 `StrategyDecision` | 抓数据、下单、写 DB / Redis / 文件 |
| **② Bot dir**（`*_01_strategy/`）| framework 集成 | `config.yaml` + `run.sh` + `paper_trade.py` + `state/*.json` + `logs/*.jsonl` | 处理复杂算法（应下沉到 block）|

一个 template 可以被多个 bot dir 复用（同算法不同参数、不同 symbol、不同 timeframe）。

## 目录结构

```
demo_strategy/
├── README.md                                # 本文
│
├── _shared/                                 # 通用组件
│   ├── blocks/                              # ① 模块化 block 层 —— 一入一出、独立可测
│   │   ├── base.py                          # Block 基类
│   │   ├── registry.py                      # 自动发现 + 三层 YAML overlay + load_block/load_layer
│   │   ├── contracts.py                     # StrategyContext / StrategyDecision / TemplateMeta
│   │   ├── signals/                         # 每个 block 一对 <name>.py + <name>.yaml
│   │   │   ├── layer.yaml                   # 层默认 + enabled 清单
│   │   │   ├── ema.py + ema.yaml            # EmaBlock, compute(bars) → {ema_fast/mid/slow, direction}
│   │   │   ├── rsi.py + rsi.yaml
│   │   │   ├── macd.py + macd.yaml
│   │   │   ├── atr.py + atr.yaml
│   │   │   ├── resonance.py + resonance.yaml
│   │   │   ├── volume_surge.py + volume_surge.yaml
│   │   │   ├── attention_frequency.py + .yaml   ← 广场策略专用
│   │   │   └── attention_deep.py + .yaml
│   │   ├── scoring/
│   │   │   ├── layer.yaml
│   │   │   ├── hierarchical.py + .yaml      # additive / weighted / max
│   │   │   └── verdict_gate.py + .yaml      # thresholds → labels
│   │   └── decision/
│   │       ├── layer.yaml
│   │       ├── direction_vote.py + .yaml    # majority vote → long/short/flat
│   │       └── position_size.py + .yaml     # fixed_risk_pct → qty_usdt/lev/stop
│   └── bot/                                 # ② bot dir 的公共骨架
│       ├── daemon.py                        # paper_trade / live_trade 共享的 daemon loop
│       ├── state_writer.py                  # 原子写 state.json + 追加 events.jsonl / trades.jsonl
│       ├── config_loader.py                 # 读 config.yaml；env override; profile 解析
│       └── data_adapter.py                  # 数据源抽象：bdp-bot Redis / Kafka / 独立 REST fallback
│
├── btc_multi_factor_trend_01_strategy/      # 案例①（bot dir 命名遵循 <name>_01_strategy 契约）
│   ├── config.yaml                          # bdp-bot 用来注册（name/sid/symbol/interval/runtime）
│   ├── run.sh                               # `bash run.sh paper-fg|live|backtest`
│   ├── scripts/
│   │   ├── template.py                      # ① 纯 template: calculate_signal(ctx) → decision
│   │   ├── paper_trade.py                   # ② paper-fg daemon 入口
│   │   ├── live_trade.py                    # ② live daemon 入口（复用 daemon.py）
│   │   └── backtest.py                      # ② backtest 一次性任务
│   ├── logs/                                # events.jsonl / trades.jsonl / strategy.log（运行时生成）
│   └── state/                               # state.json / pid / run_id（运行时生成）
│
└── square_buzz_screener_01_strategy/        # 案例②（同一契约）
    ├── config.yaml
    ├── run.sh
    ├── scripts/
    │   ├── template.py                      # calculate_selection(ctx) → SelectionDecision (选币型)
    │   ├── universe_source.py               # 从 Binance Square 抓热点（独立于 template，不违反纯计算规则）
    │   ├── paper_trade.py
    │   ├── live_trade.py
    │   └── backtest.py
    ├── logs/
    └── state/
```

## 三层 config overlay（block 层）

每个 block 都有自己的 `<name>.yaml`（默认参数），每个 layer 有 `layer.yaml`（层默认 + 开关），策略最后在 `config.yaml::params` 里做最终覆盖 —— 后者优先：

```
① signals/rsi.yaml          period: 14
② signals/layer.yaml        blocks.rsi.period: 21     (未设时透传①)
③ config.yaml::params.signals.rsi   period: 9         (最终生效)
```

深合并：`overrides={'periods': {'fast': 8}}` 只覆盖 `fast`，`mid`/`slow` 保留 block yaml 的默认。

## 用 block 写 template（示例）

```python
from demo_strategy._shared.blocks import load_block, TemplateMeta

TEMPLATE_META = TemplateMeta(strategy_id="demo_xxx", display_name="…")

def calculate_signal(ctx):
    # 每个 block 独立实例化，overrides 从策略 config.yaml 传进来
    ema  = load_block("signals", "ema",  overrides=ctx.config.get("signals", {}).get("ema"))
    rsi  = load_block("signals", "rsi",  overrides=ctx.config.get("signals", {}).get("rsi"))
    macd = load_block("signals", "macd").compute(ctx.market.bars)

    # 或一次加载整层
    layer = load_layer("signals", strategy_overrides=ctx.config.get("signals", {}))

    tiers = [...]
    total  = load_block("scoring", "hierarchical").compute(tiers)
    verdict = load_block("scoring", "verdict_gate").compute(total)
    return ...
```

**关键属性**：
- `load_block()` 返回配置好的实例；`compute(inputs, **kwargs)` 是唯一入口
- 每个 block ≤ 100 行，可单独 unit-test
- 想调参？改 YAML —— 不动代码
- 想扩指标？新增 `<layer>/<newname>.py + <newname>.yaml`，registry 自动发现

## 契约（与 bdp-ai-trading-bot 逐条对齐）

### 契约 1：template 层 —— 纯计算，不越界

```python
# scripts/template.py
from demo_strategy._shared.blocks.contracts import (
    StrategyContext, StrategyDecision, TemplateMeta,
)

TEMPLATE_META = TemplateMeta(
    strategy_id="demo_btc_multi_factor_trend",
    display_name="BTC Multi-Factor Trend (Demo)",
    config_schema={ … },
)

def calculate_signal(ctx: StrategyContext) -> StrategyDecision:
    # 只读 ctx.market.bars / ctx.account / ctx.config
    # 计算 6 因子 → 打分 → verdict → direction
    # 返回 StrategyDecision(side, strength, reason)
    ...
```

**永远不能做**（框架契约）：
- `requests.get(...)` / `redis.get(...)` — 数据来自 `ctx.market.bars`
- `binance_client.new_order(...)` — 返回 `StrategyDecision`，framework 翻译成 OrderCommand
- `open("state.json", "w")` — daemon 层负责持久化

**允许**：`import numpy as np, pandas as pd` + `from _shared.blocks import signals/scoring/decision` 做计算。

### 契约 2：bot dir —— supervisor 友好

- **`config.yaml` 至少含**：`name` / `sid` / `symbol` / `interval` / `runtime` / `template_id`
- **`run.sh` 支持**：`paper-fg`（前台，SIGTERM 干净退出）/ `live` / `backtest`
- **`scripts/paper_trade.py` daemon loop**：
  1. detect bar close
  2. fetch bars（走 `_shared/bot/data_adapter.py` → 优先 bdp-bot Redis / Kafka，本地夹带 REST fallback）
  3. `ctx = build_context(bars, account, config, close_time)`
  4. `decision = template.calculate_signal(ctx)`
  5. 走 `_shared/bot/state_writer.py` 落 `events.jsonl::signal` / `state/state.json`
  6. 若 live —— publish OrderCommand（走 bdp-bot Kafka OR 本地 binance-cli 兜底）
  7. 循环，处理 SIGTERM 干净关闭

- **`state/state.json`** 至少含：`status`（running/stopped/risk_halted/error）、`equity`、`open_positions`、`last_bar_ts`、`last_signal_ts`
- **`logs/events.jsonl`** 每行一 event：`{ts, kind, ...}`；`kind ∈ {started, signal, order_placed, order_filled, stopped, risk_halted, error}`
- **`logs/trades.jsonl`** 每行一笔成交
- **`logs/strategy.log`** stdout+stderr（run.sh 会重定向）

### 契约 3：数据源三级 fallback

`_shared/bot/data_adapter.py` 抽象数据获取：

```
① BDP_BOT_KAFKA_URL  → Kafka hot store（strategy_service.market.fetch_bars 或 kline topic）
② BDP_BOT_REDIS_URL  → Redis snapshot（market/indicators.py 里那种）
③ 独立 REST fallback → binance-cli 或 requests（本地/dev/backtest）
```

同一份 template.py 在三种数据源下都能跑 —— 不同的是 daemon 层怎么装 `ctx.market.bars`。

### 契约 4：执行

- **`live_trade.py` 优先** publish OrderCommand 到 bdp-bot Kafka（`strategy_service.engine.order_stream.kafka.publish`）
- 若不在 bdp-bot 集群里跑，fallback 到 `atomic_strategy_lib.execution.orders.market_order()`（原来 v1 用的路径）

## 运行

### 独立模式（本地 / dev）

```bash
cd btc_multi_factor_trend_01_strategy
bash run.sh paper-fg          # 前台跑 daemon
bash run.sh backtest          # 一次性回测
```

### 集成到 bdp-ai-trading-bot

```bash
# 注册这个策略目录
bdp-bot register /path/to/demo_strategy/btc_multi_factor_trend_01_strategy --mode paper

# 启动
bdp-bot start btc_multi_factor_trend_01 --mode paper

# 查状态
bdp-bot status btc_multi_factor_trend_01
```

## 两个案例的对比

| 维度 | btc_multi_factor_trend | square_buzz_screener |
|---|---|---|
| **template 类型** | `calculate_signal` (single-symbol) | `calculate_selection` (cross-sectional) |
| **输入** | `StrategyContext(market, account, config)` | `SelectionContext(universe, bars_by_symbol, config)` |
| **输出** | `StrategyDecision(side, strength, reason)` | `SelectionDecision(weights, reason, metadata)` |
| **universe 决定权** | 固定 BTCUSDT（config）| 动态：Square scrape → 由 `universe_source.py` 提供给 daemon |
| **framework 侧执行** | OrderCommandV1 | PortfolioTargetCommandV1 |
| **6 vs 9 因子** | 6 tier | 9 tier（+ attention 2 tier + volume）|

两条策略共享 **`_shared/blocks/*`**（信号、打分、决策 helper），共享 **`_shared/bot/*`**（daemon 骨架、state writer、数据 adapter）—— 差异只在 template.py（真正的算法）和 config.yaml（参数）。

## 依赖

- Python 3.10+
- `pandas` / `numpy`（signals helper）
- `PyYAML`（读 config.yaml）
- 可选：`atomic_strategy_lib`（数据 fallback + 本地执行）
- 集成到 bdp-ai-trading-bot 时：`strategy_service.market` / `strategy_service.engine.order_stream`
