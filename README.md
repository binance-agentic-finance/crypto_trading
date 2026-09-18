# cyqnt-trd

`cyqnt-trd` 是一个以 `standard_bot` 工作流为核心的加密货币交易工具包：

- 历史数据下载到本地 parquet（K 线、资金费、持仓量 OI、订单簿、新闻）
- 从 `1m` 本地重采样到更高周期
- 统一的向量化回测框架（`cyqnt_trd.eval`，默认引擎）+ 事件驱动参考引擎（`--engine python`）
- paper（模拟盘）信号生成
- monitor / run-manager 驱动的执行流
- 一套完整的指标库（传统 TA + TradingView 常用指标 + SMC）
- 一套因子库（经典 TA、风险 / 动量 / 情绪、101 个 WorldQuant alpha）+ 信号策略
- 单因子评测矩阵（六道研究闸门、含资金费的净收益）
- GA + LLM 进化式策略挖掘
- 向后兼容旧的 `atomic_strategy_lib` 命名空间

### 系统的形状

一切都汇入**一种输入格式和一种输出格式**：

```
data sources ──►  cyqnt.input/v1  ──►  blocks  ──►  cyqnt.signal/v2  ──►  consumers
  klines            (a JSON             (146          kind=trade          backtest
  funding / OI       snapshot of         composable    kind=selection      paper
  order book         one instant)        functions)    kind=alert          live
  news / buzz
  contract meta
```

往下读之前，有两个结论值得先知道：

**`cyqnt.input/v1` 是被捕获的，不是流式的。** 一个决策瞬间的快照可以写进文件再回放。
于是回测、模拟盘、实盘跑的是*同一份代码* —— 区别只在快照来自哪里。这让一次运行逐字节
可复现（`signal_id` 是内容的 `uuid5`，绝不取时钟）。

**`cyqnt.signal/v2` 只有一个字段供消费方分流。** `kind` 是从 payload 里*推导*出来的，
不是生产者声明的，所以它不会和它所标注的字段自相矛盾：

| `kind` | 轴 | 承载字段 |
|---|---|---|
| `trade` | 时间 —— 单个标的，逐根 bar | `entry` / `exit_plan` / `size` |
| `selection` | 截面 —— 同一瞬间的所有标的 | `candidates` / `universe_size` |
| `alert` | 都不是 —— 一条通知 | `advisory_action` / `summary` |

按你所处的轴，有两条推荐路径：

```
trade      historical parquet → local resample → standard_bot signal → framework (cyqnt_trd.eval) | Python engine
selection  live or frozen cyqnt.input/v1 → universe pipeline → ranked basket
```

---

## 安装

### 从 PyPI 安装

```bash
pip install cyqnt-trd
```

本包支持 Python `>=3.8,<3.13`，覆盖 Binance AI 的 Python `3.11.2`。

安装时会一并带上 `atomic_strategy_lib` 垫片包，让用 `from atomic_strategy_lib.X import Y`
的旧案例脚本无需改动即可继续工作。

对于 HTTPS 请求，`cyqnt-trd` 按以下顺序解析 CA 证书包：

1. `REQUESTS_CA_BUNDLE`
2. `SSL_CERT_FILE`
3. `CURL_CA_BUNDLE`
4. Linux 系统 CA 包：`/etc/ssl/certs/ca-certificates.crt`
5. `certifi`
6. `requests` 的默认校验行为

对于 Binance 端点由仅存在于系统信任库的内部 CA 签发的特殊 Linux 部署，请通过环境变量
设置部署用的 CA 包，例如：

```bash
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

### 开发环境

```bash
git clone https://github.com/nthu-chung/crypto_trading
cd crypto_trading
python3 -m venv .venv-standard-bot
source .venv-standard-bot/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt -r requirements-standard-bot-mvp.txt
```

---

## 推荐的 Standard Bot 入口

### 回测（framework 引擎 —— 默认，统一走 `cyqnt_trd.eval`）

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_backtest \
  --engine framework \
  --market-type futures \
  --strategy multi_timeframe_ma_spread \
  --symbol BTCUSDT \
  --interval 5m \
  --secondary-interval 1h \
  --primary-ma-period 20 \
  --reference-ma-period 20 \
  --spread-threshold-bps 0 \
  --historical-dir data/mtf_90d \
  --start-ts 1768003200000 \
  --end-ts 1775779200000 \
  --download-missing \
  --output-json docs/backtests/btc_mtf_ma_cross_5m_1h_20_20_90d.json
```

### 回测（Python 引擎 —— 用于自定义 block 策略）

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_backtest \
  --engine python \
  --strategy smc_3confluence_v1 \
  --strategy-module cyqnt_trd.strategies.smc_3confluence \
  --symbol BTCUSDT \
  --interval 1h \
  --limit 1000 \
  --market-type futures \
  --initial-capital 10000 \
  --commission-bps 4 \
  --slippage-bps 2
```

### Paper 信号

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_paper \
  --market-type futures \
  --strategy multi_timeframe_ma_spread \
  --symbol BTCUSDT \
  --interval 5m \
  --secondary-interval 1h \
  --primary-ma-period 20 \
  --reference-ma-period 20 \
  --spread-threshold-bps 0 \
  --historical-dir data/mtf_90d \
  --dry-run
```

### Monitor / 后台会话

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_monitor_http \
  --broker paper \
  --host 127.0.0.1 \
  --port 8787
```

### Paper 交易守护进程（长驻，用于 block 策略）

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_paper_daemon \
  --engine python \
  --strategy ma_cross_v1 \
  --strategy-module strategies.ma_cross_v1 \
  --symbol BTCUSDT --interval 1h \
  --market-type futures \
  --state-dir ./watcher/MA_CROSS_V1_BTCUSDT_1h \
  --poll-interval 3570 --warm-up-bars 80 \
  --initial-capital 10000 --fee-bps 4 --slippage-bps 2
```

### 实盘交易（经由 binance-cli）

实盘需要**两个进程**并行运行：
1. Paper 守护进程（信号源 —— 与 paper 模式完全一致）
2. 实盘执行器（把 paper 成交翻译成真实的 binance-cli 订单）

```bash
# Terminal 1: Paper daemon (signal source)
python -m cyqnt_trd.standard_bot.entrypoints.mvp_paper_daemon \
  --engine python \
  --strategy ma_cross_v1 \
  --strategy-module strategies.ma_cross_v1 \
  --symbol BTCUSDT --interval 1h \
  --market-type futures \
  --state-dir ./watcher/MA_CROSS_V1_BTCUSDT_1h \
  --poll-interval 3570 --warm-up-bars 80 \
  --initial-capital 10000 --fee-bps 4 --slippage-bps 2

# Terminal 2: Live executor (dry-run first!)
python -m cyqnt_trd.standard_bot.entrypoints.mvp_live_executor \
  --state-dir ./watcher/MA_CROSS_V1_BTCUSDT_1h \
  --symbol BTCUSDT \
  --max-notional 200 \
  --dry-run

# Terminal 2: Live executor (real orders — remove --dry-run)
python -m cyqnt_trd.standard_bot.entrypoints.mvp_live_executor \
  --state-dir ./watcher/MA_CROSS_V1_BTCUSDT_1h \
  --symbol BTCUSDT \
  --max-notional 200
```

紧急停止：`touch ./watcher/MA_CROSS_V1_BTCUSDT_1h/EMERGENCY_STOP`

完整的实盘交易文档见 [references/trading-modes.md](references/trading-modes.md)。

### MA Cross 策略参考工作区

包内附带一个整理好的参考工作区：

`cyqnt_trd/standard_bot/ma_cross_strategy/`

在 Binance AI Pro 安装环境中，等价路径通常是：

`/usr/local/lib/python3.11/dist-packages/cyqnt_trd/standard_bot/ma_cross_strategy/`

这个工作区是 block 策略工作流的一个具体范例：

- `strategies/ma_cross_v1.py` —— SMA 5/20 金叉/死叉策略
- `scripts/run_strategy.py` —— 回测、模拟盘、实盘的统一启动器
- `scripts/run_paper_daemon.sh` —— paper 守护进程的 shell 入口
- `scripts/signal_executor.py` —— `BinanceCliExecutor` 的独立包装
- `scripts/session_watcher.py` —— 监视成交、实盘执行与风控止损
- `tests/test_strategy_composition.py` —— 导入/注册与信号行为测试

聚焦的使用说明见 `cyqnt_trd/standard_bot/ma_cross_strategy/README.md`。

---

## 指标库

`cyqnt_trd.blocks` 提供一套完整的纯 pandas 指标库，覆盖三大类：

### 经典 TA（30+ 个函数）

`cyqnt_trd/blocks/indicators.py`：

- 移动均线：SMA, EMA, WMA, RMA, **TEMA**, **DEMA**, **HMA**, **VWMA**
- 动量：RSI, MACD, **MFI**, **CCI**, **Williams %R**, StochRSI, **TRIX**, **Awesome Oscillator**, Aroon
- 波动与区间：ATR, Bollinger Bands, **Keltner Channel**, Donchian, ADX, SuperTrend, Parabolic SAR
- 结构：Ichimoku, Pivot Points（标准 floor）, **ZigZag**, Heikin Ashi, swing high/low
- 量能：VWAP, OBV, **CMF**, **PVT**, 成交量 MA / Z-score
- 统计：滚动 Z-score、滚动分位数、MA 方向、MA 对齐

### Smart Money Concepts（SMC）

`cyqnt_trd/blocks/smc_structure.py` 与 `cyqnt_trd/blocks/smc_liquidity.py`：

- `fractal_pivot_high(df, lookback=5)` / `fractal_pivot_low(df, lookback=5)`
- `fair_value_gap(df)` —— 多头/空头 FVG 检测
- `order_block_detect(df, swing_lookback=5)` —— 机构买卖区
- `bos_choch_detect(df, swing_lookback=5)` —— 结构突破 / 结构转变，带趋势状态机
- `liquidity_sweep_detect(df, swing_lookback=5)` —— 猎杀止损检测
- `equal_highs_lows(df, swing_lookback=5, tolerance_pct=0.1)` —— EQH / EQL 簇
- `premium_discount_zone(df, swing_lookback=5)` —— Premium / Discount / Equilibrium 分类

### 风控与仓位

`cyqnt_trd/blocks/`：

- `verdicts.py` —— 5 个打分组合器 + 8 道闸门（`hard_gate`, `verdict_classify`, `cross_validate`, ...）
- `sizing.py` —— Kelly、固定金额止损、固定风险 %、ATR 反比、杠杆上限
- `stop_loss.py` —— 4 个止损辅助
- `limits.py` —— 7 项风险限制（强平、最大持仓数、最大敞口、当日亏损、价格偏离、熔断、资金费窗口）
- `exit.py` —— 分级止盈、ATR 移动止损

---

## 基于 Block 的策略

策略使用一个简单的 `make_signals(df) → (long_signal, short_signal)` 接口：

```python
import pandas as pd
from cyqnt_trd.blocks import strategy
from cyqnt_trd.blocks.smc_structure import bos_choch_detect
from cyqnt_trd.blocks.smc_liquidity import (
    liquidity_sweep_detect,
    premium_discount_zone,
)


def make_signals(df: pd.DataFrame):
    bos = bos_choch_detect(df, swing_lookback=5)
    sweep = liquidity_sweep_detect(df, swing_lookback=5)
    pdz = premium_discount_zone(df, swing_lookback=5)

    long = (
        (sweep["sweep_direction"].rolling(10).apply(lambda x: ("BULL" in x.values)).fillna(0).astype(bool))
        & (bos["trend_state"] != "DOWN")
        & (pdz["current_zone"] == "DISCOUNT")
    )
    short = (
        (sweep["sweep_direction"].rolling(10).apply(lambda x: ("BEAR" in x.values)).fillna(0).astype(bool))
        & (bos["trend_state"] != "UP")
        & (pdz["current_zone"] == "PREMIUM")
    )
    return long, short


strategy.register("my_smc_strategy_v1", make_signals)
```

然后运行：

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_backtest \
  --engine python \
  --strategy my_smc_strategy_v1 \
  --strategy-module path.to.my_strategy \
  --symbol BTCUSDT --interval 1h --limit 1000
```

`cyqnt_trd/strategies/` 里的参考策略：

- `smc_3confluence.py` —— 宽松 SMC，sweep + 结构 + 区域三重共振
- `smc_5confluence.py` —— 严格 SMC，要求五个 SMC 组件全部对齐
- `mega_indicator_smoke.py` —— 遍历每个新指标的冒烟测试
- `channel_breakout.py`, `ema_rsi_cross.py`, ... —— 传统 TA 策略

---

## YAML 策略管线

`cyqnt_trd.standard_bot.yaml_pipeline` 让一个策略可以用声明式的 **YAML spec**
描述（数据源、指标、嵌套入场规则、仓位、风控/出场），由它组合 `cyqnt_trd.blocks.*`
原语 —— 无需为每个策略写 Python。同一份 spec 在回测 / 模拟盘 / 实盘上都能跑。

```bash
# validate: static checks + a synthetic-data dry-run that catches unknown
# blocks, wrong arg counts, and bad params BEFORE any real run
python -m cyqnt_trd.standard_bot.yaml_pipeline validate strategy.yaml

# run: backtest (default), or paper / live per run.mode
python -m cyqnt_trd.standard_bot.yaml_pipeline run strategy.yaml \
  [--input-json klines.json] [--engine vectorized|event] [--start]
```

一份 spec 会通过 `cyqnt_trd.blocks.strategy.register(...)` 编译成
`make_signals(df) -> (long, short)` 的 block 策略，因此它走的是和手写 block 策略
完全相同的运行路径。入场规则可任意嵌套（`all_of` / `any_of` / `not`）；只做多只需
省略 `entry.short`。schema、可运行的示例、以及自然语言 → YAML → 回测的演示见
`docs/strategy_yaml_spec/`。

## Selection：筛选整个截面

`trade` 轴是对单个标的沿时间行走。`selection` 轴是在同一瞬间行走*各个标的*。同一套
表达式语言、同一个输出契约、不同的轴 —— 一列就是一列，所以 `conditions.value_above`
并不在意是哪种。

```bash
# capture a snapshot for THIS spec (it resolves only the sections the spec needs)
python -m cyqnt_trd.standard_bot.entrypoints.mvp_input_bundle \
    --strategy-yaml docs/strategy_yaml_spec/example_open_interest_screen.yaml \
    --out bundle.json

# replay it offline — no network, byte-for-byte reproducible
python -m cyqnt_trd.standard_bot.yaml_pipeline run <spec>.yaml --input-json bundle.json

# or omit --input-json to fetch live
python -m cyqnt_trd.standard_bot.yaml_pipeline run <spec>.yaml
```

一个 `selection:` 段是一条有序管线；每一步都接上一步的 frame：

```yaml
selection:
  universe:
    - { block: universe.augment_with_contract_meta, with: [contract_meta] }
    - { block: universe.filter_crypto_only }
    - { block: universe.filter_quote_volume, params: { min_quote_volume: 2.0e+7 } }
    - { block: universe.augment_with_indicator, with: [universe_bars],
        params: { indicator: rsi, timeframe: "1h", period: 14, as: rsi14 } }
  score: rsi14
  order: asc          # asc | desc
  max_score: 45       # absolute ceiling, never flipped by `order`
  top_k: 5
  dedupe_by: base_asset
  long_when: { cond: conditions.value_below, args: [rsi14, 45] }
```

共 **29 `universe.*` blocks**，命名本身是一个被另外三处解析的契约：

| 前缀 | 保证 | 示例 |
|---|---|---|
| `augment_with_*` (8) | 追加列，**行不变** | `contract_meta`, `funding`, `open_interest`, `oi_change`, `long_short_ratio`, `spread`, `news`, `indicator` |
| `filter_*` (13) | 删行，**列不变** | `crypto_only`, `underlying_type`, `sub_type`, `quote_volume`, `quote_suffix`, `spread`, `top_of_book`, `open_interest`, `oi_change`, `long_short_ratio`, `funding_rate`, `change_pct`, `sentiment` |
| `top_*` / `only_*` / `exclude_*` | 删行 | `top_gainers`, `top_losers`, `top_mentioned`, `only_symbols`, `exclude_symbols` |

### 步骤顺序是语义，不是风格

其中四个数据源**没有全市场端点** —— 每个存活的标的要发一次 HTTP 请求。所以它们必须
排在某个能收窄集合的步骤*之后*：

```
augment_with_open_interest   augment_with_oi_change
augment_with_long_short_ratio   augment_with_indicator
```

实测：先收窄到 41 个名字 → 123 次请求。反过来，727 × 3 = 2181 次请求，直接把限频预算
打爆。`validate` 会**静态**检查这个顺序 —— 它是唯一一处会看某步相对其他步位置的检查，
因为下游的覆盖率算术只能抓住四个里的一个，而且还是碰巧。

### 声明 spec 猜了什么

有些请求有两种合理读法，没有哪个 block 能替它拍板 —— "成交量超过一千万"既可能是 1e7
的成交额，也可能是 1e7 个币，两者选出不同的名字。产品默认是**声明**而不是阻塞：

```yaml
strategy:
  assumptions:
    - cid: c3
      reading: "read as 24h quote turnover, 1e7 USD"
      alternatives: ["1e7 coins of base volume"]
      basis: "quoteVolume is the only turnover column on the frame"
```

YAML 注释**不算** —— 注释在加载时就被丢弃，拿着这篮子的人永远看不到。在这里声明后，
每一条都会随信号的 `warnings` 带出去，并给 `reason_codes` 加上 `assumption_declared`。
键集是封闭的；拼错的键是报错，而不是被忽略的字段，因为记在错误键下的选择又变回了一个
沉默的选择。

### 代币化股票是真实产品

`NVDAUSDT`、`TSLAUSDT`、`AAPLUSDT`、`QQQUSDT` 以及另外约 130 个，是活跃的 USDⓈ-M
永续（`underlyingType=EQUITY`，`contractType=TRADIFI_PERPETUAL`）。所以"排除美股"和
"筛选美股"*都是*真实请求，两者都不是默认：

```yaml
- { block: universe.filter_crypto_only }                                # exclude them
- { block: universe.filter_underlying_type, params: { include: [EQUITY] } }  # screen them
```

不过它们不是股指期货。`QQQUSDT` 带资金费（实测约年化 +1.9%），相对其指数溢价交易，
而且**在美股休市时仍在波动** —— 所以为 NQ 写的阈值搬不过来。

## 回测引擎

Block（Python 引擎）策略可以由两个可互换的引擎回测，两者都支持**多 + 空**，共享同一套
执行模型（bar 收盘出信号 → 下一根 bar 开盘成交，单一持仓，带方向的 stop/TP）：

| 引擎 | 由谁调用 | 说明 |
|---|---|---|
| `SnapshotBacktestRunner` | `mvp_backtest --engine python`（事件驱动，逐根 bar） | 成熟参考实现；镜像 paper/实盘的执行模型 |
| `run_vectorized_backtest` | `yaml_pipeline run` 默认（信号一次性向量化 + numpy 出场循环） | 快 10–100×；YAML 管线与参数扫描用它 |

两者经过交叉校验，在匹配配置下高度一致（只做多在两者之间逐字节稳定；多 + 空在大多数
出场类型上一致）。paper/实盘信号走 `PythonLivePaperSession`，同样是多 + 空。

`mvp_backtest` 的**默认引擎是 `--engine framework`**（统一走 `cyqnt_trd.eval` 的向量化
`Σ W·R` 回测，内置策略与 block 策略都经由同一 `make_signals` 契约）。旧的 Numba 编译引擎
（`NumbaBacktestRunner` 及其 `@njit` kernel）已删除，回测统一收敛到该框架。

## 因子评测与策略回测框架（`cyqnt_trd.eval`）

`cyqnt_trd.eval` 是**策略定义与回测的统一框架**（八个阶段：采样→面板→因子→预处理→
forecast→仓位→回测→对比）。它有两个入口：**单因子评测矩阵** `evaluate()`（六道闸门裁决），
和**策略构建 + 向量化回测**（因子→forecast→仓位 `W`→`Σ W·R`，见下）。**策略的定义与回测
统一走这里**。先看单因子评测——

`evaluate()` 用一张固定的研究矩阵给**一个截面因子**打分：每天在有资格的截面里给
因子排序，在开发段（dev）冻结方向并且之后不再重挑，在 `open[t+2]` 入场，按非重叠周期
持有 `h` 天，对每次全平全开收取双边成本外加**实际**逐次资金费，然后指出因子**在哪一关**
失败。"比噪声强吗"的门槛来自把随机信号走同一条管线跑出来的参考分布，不是拍脑袋。

这是两阶段工作流的**第一步** —— 筛选单个因子。因子组合、风险预算、仓位是第二阶段，
在输出里始终标为 `NOT_EVALUATED`；`PASS` 表示通过这张研究卡，不是交易许可。

| 闸门 | 问题 | 失败意味着 |
|---|---|---|
| **G0** 数据 | 到底测到东西了吗？ | 分数是伪影 |
| **G1** 信息 | 排序比随机信号强吗？ | 没有信息 |
| **G2** 结构 | 关系是否单调 / 跨期限稳定？ | 有信息，但形态不可用 |
| **G3** 成本 | 扣掉费用和资金费还活着吗？ | **硬否决** |
| **G4** 稳健 | 跨段、跨年份都成立吗？ | 只是一个行情，不是一个效应 |
| **G5** 增量 | 比公开基线更多吗？ | 只是 size / low-vol / reversal 的换皮 |

```python
from cyqnt_trd.eval import evaluate, load_bundle

panel = load_bundle()                    # 随包自带的 top-10 永续面板（离线、无网络可跑）

def my_factor(p):                        # Panel -> DataFrame；row t 只用 t 收盘前可得的数据
    return -(p.close / p.close.shift(5) - 1)     # 5 日反转

card = evaluate(my_factor, panel)        # 随包标定，直接出裁决
print(card.verdict)      # e.g. 'HOLD_INFO' —— 有排序信息，但扣费后不赚钱
print(card.blocking)     # 卡在哪几关，e.g. ['G3_cost']
print(card.to_markdown())
# 自备面板：Panel(open=..., high=..., ...)；换面板时噪声地板要对新面板重算（fail closed）
```

把上游 capability 算子（收 `list[float | None]`、出单值或序列）接进来用 `capability_factor`；
离散的 {-1,0,1} 方向票用 `cyqnt_trd.blocks.factor_votes`。

### 策略定义 + 回测（统一走这个框架）

八个阶段，每阶段输入输出显式：**采样 → 面板 → 因子 → 预处理 → forecast → 仓位 → 回测 → 对比**。
因子 / forecast / 仓位都是**无状态映射**，所以回测就是一次矩阵运算 `Σ_s W[t,s]·R[t,s]`，而
`W` 的最后一行**直接就是实盘目标仓位**（研究与实盘同源）。设计见
[`design/framework_design.md`](design/framework_design.md)。

```python
from cyqnt_trd.eval import load_bundle, forecast, framework

panel = load_bundle()
factors = {"rev5":  -(panel.close / panel.close.shift(5) - 1),          # 量价
           "carry": -(panel.funding / panel.close).rolling(7).sum()}    # 选币
betas = forecast.fit_betas(factors, panel, method="ic")   # 只在 dev 段拟合并冻结系数
mu    = forecast.make_forecast(betas)(factors, panel)     # 期望收益
W     = forecast.positions_from_forecast(mu, panel.mask, gross=1.0)     # 目标仓位
res   = framework.backtest_weights(W, panel)              # Σ W·R − 换手成本 − 逐次资金费
print(res.metrics)                                        # dev/val/oot 分段；W 尾行即实盘目标仓位
```

- 把规则 / 事件驱动 / 量价 / 选币四类写成**一张策略蓝图**：`cyqnt_trd.eval.blueprint`（触发→硬门→分层打分→裁决→方向→仓位→出场）。
- 端到端"候选→评测→准入→构建→对比"：`cyqnt_trd.eval.pipeline` / `experiment`。
- 滚动前推 + DSR / PBO 防过拟：`cyqnt_trd.eval.optimize`；有状态止盈止损/加减仓属执行细化层。
- `framework.check_stateless(build, panel)` 用**截断重算**校验无状态契约（catches `shift(-k)` 等泄漏）。

**命令行**：

```bash
python -m cyqnt_trd.eval stages          # 八阶段与每阶段 I/O
python -m cyqnt_trd.eval capabilities    # capability 清单里哪些能在本面板上跑
python -m cyqnt_trd.eval score --factor my_ideas.py:my_factor
```

完整 API、八阶段结构、14 维证据矩阵与裁决分类见
[`cyqnt_trd/eval/README.md`](cyqnt_trd/eval/README.md) 与 `cyqnt_trd/eval/STRUCTURE.md`。
**策略的定义与回测统一走这个框架**；`standard_bot` 是另一条（有状态、单标的、事件驱动）
的实盘/模拟执行线，与此向量化研究线并存（见下方"回测引擎"）。

## 统一因子 / 信号库（`cyqnt_trd.blocks`）

因子与信号只有**一套库** —— `cyqnt_trd.blocks`。原 `cyqnt_trd.trading_signal` 已**删除**，
内容全部并入 `blocks`：

| 内容 | 位置 | 说明 |
|---|---|---|
| 经典 TA 方向票（rsi/macd/adx/cci/williams/stochastic/ao/uo/bbp/ema/ma…） | **`blocks.factor_votes`** | **向量化 {-1,0,1} 票**，指标数学取自 `blocks.indicators`（Wilder 口径） |
| 逐点因子：经典 TA + JoinQuant 统计（风险/技术/动量/情绪）+ TSI，~52 个 | **`blocks.factors`** | `f(data_slice)->float`；如 `rsi_factor`, `variance_factor`, `trix_factor`, `psy_factor`, `wvad_factor` |
| 101 个 WorldQuant Alpha | **`blocks.alphas`** | 逐点 `alphaN_factor(data_slice)->float`，另有 `to_series(fn, df)` 取整段、无未来函数的 Series |
| 有状态信号层（`buy/sell/hold`） | **`blocks.signals`** | `signal_func(data_slice, position, entry_price, …) -> 'buy'/'sell'/'hold'`（`ma_signal`, `factor_based_signal`, …） |

```python
# 向量化方向票（blocks 原生）
from cyqnt_trd.blocks import factor_votes as fv
votes = fv.rsi_vote(df, period=14)            # 整段 pd.Series ∈ {-1,0,1}，无未来函数

# 逐点因子 / alpha / 有状态信号
from cyqnt_trd.blocks.factors import rsi_factor, variance_factor
from cyqnt_trd.blocks.alphas import alpha1_factor, to_series
from cyqnt_trd.blocks.signals import ma_signal, factor_based_signal
alpha_series = to_series(alpha1_factor, df)   # 逐 bar 的整段 Series

# 送进评测矩阵：把逐点算子/因子包成面板因子（capability_factor），或直接写 Panel->DataFrame
from cyqnt_trd.eval import evaluate, capability_factor
```

策略研究与回测统一走 `cyqnt_trd.eval`（见上一节）；单标的策略经 `eval.adapters` 桥接进框架。

## 策略挖掘（`cyqnt_trd.evolve`）

`cyqnt_trd.evolve` 是一个 **GA + LLM** 的循环，用来挖掘日内策略：AI 负责给出种子种群
和变异，包负责确定性的部分 —— 基因组序列化、回测桥接、适应度、选择、交叉，以及样本内/
样本外切分。公开 API：`StrategyGenome`, `Population`, `Factor`, `Filter`。

```bash
python -m cyqnt_trd.evolve init --session runs/xrp --symbol XRPUSDT \
  --interval 5m --days 60 --is-days 40 --oos-days 20 --population-size 30
python -m cyqnt_trd.evolve step     --session runs/xrp     # one GA generation
python -m cyqnt_trd.evolve status   --session runs/xrp
python -m cyqnt_trd.evolve export   --session runs/xrp     # winners → block strategy
```

子命令：`init`, `inject`, `step`, `mutate`, `validate`, `status`, `export`,
`diagnose`。导出的胜出者就是普通的 block 策略，可在上面那些引擎上运行。

## 打包的策略案例（`cyqnt_trd.strategy_cases`）

随 wheel 一起分发的精选参考案例（通过 `pkgutil.get_data` 加载，不依赖文件系统路径）：

```python
from cyqnt_trd.strategy_cases import list_case_ids, load_case, load_case_preset, load_case_readme

list_case_ids()          # ['rsi-mean-reversion', 'btc-multi-factor-trend',
                         #  'bb-squeeze-momentum', 'quadruple-filter-breakout',
                         #  'structure-breakout-priceaction']
case   = load_case("rsi-mean-reversion")        # structured case definition (JSON)
preset = load_case_preset("rsi-mean-reversion") # runnable YAML-pipeline preset
readme = load_case_readme("rsi-mean-reversion") # human-readable notes
```

`load_catalog()` 返回索引；每个案例在多大程度上映射到已发布的 block，见
`cyqnt_trd/strategy_cases/README.md` 与 `SUPPORT_MATRIX.md`。

## Atomic 兼容（`atomic_strategy_lib` 垫片）

本包附带一个 `atomic_strategy_lib` 垫片，把 cyqnt_trd 的实现以旧的 atomic 命名空间
重新导出。于是用如下写法的旧案例脚本：

```python
from atomic_strategy_lib.scoring.gates import verdict_with_gate
from atomic_strategy_lib.signals.momentum import rsi_compute
from atomic_strategy_lib.decision.sizing import fixed_dollar_loss
```

在只安装了 `cyqnt-trd` 时也能无改动工作 —— 不需要设置 `PYTHONPATH` 或
`ATOMIC_STRATEGY_LIB_PATH`。

为了与原 atomic 库逐字对齐数值，垫片委托给 `cyqnt_trd/compat/atomic_signals/`，它把
atomic 的纯 Python 算法（RSI, EMA, MACD, ATR, Bollinger, StochRSI, SuperTrend,
ADX）逐字移植。12 个被测指标输出全部与 atomic 匹配到 `1e-9` 精度。

完整集成设计见 `docs/atomic-compat/MIGRATION_HANDOFF.md`。

---

## 数据获取（`cyqnt_trd.get_data`, `cyqnt_trd.data_cli`）

两个互补的数据层：

**`get_data/`** —— 通过 Binance Python SDK 下载并落盘本地 parquet 的 K 线下载器：

- `get_and_save_futures_klines(...)` —— USDⓈ-M **合约**K 线
- `get_and_save_klines(...)` / `get_and_save_klines_direct(...)` —— **现货**K 线
- `get_and_save_web3_klines(...)` —— 按合约地址取的**链上 / Web3** u-kline
- `get_kline_with_factor_at_time / _range / _n_points(...)` —— **带已算因子**的 K 线
  （单点 / 区间 / 最近 *n* 点）

**`data_cli/`** —— 更宽的行情数据面：对本地 `binance-cli` / `binance-pro-cli` 的
轻量子进程包装，返回 pandas DataFrame。用 `BINANCE_CLI` / `BINANCE_PRO_CLI` 覆盖
可执行文件路径。

```python
from cyqnt_trd.data_cli import (
    fetch_klines, fetch_funding_rate, fetch_open_interest, fetch_oi_history,
    fetch_orderbook_depth, fetch_long_short_ratio, fetch_24h_ticker,
    fetch_news, fetch_sentiment, fetch_fear_greed, full_market_scan,
)
funding = fetch_funding_rate("BTCUSDT")
oi      = fetch_oi_history("BTCUSDT", period="1h")
```

覆盖范围：K 线 / ticker、**资金费**、**持仓量 OI**、**订单簿深度与失衡**、**多空比**、
账户余额与持仓、市场扫描、**新闻 / 情绪 / 话题趋势**（公开 Binance Square）、
**恐慌贪婪 & 宏观状态**，外加一个通用 REST 取数器（`fetch_rest`, `register_spec`,
`PUBLIC_SOURCES`）用于接入你自己的数据源。

## 订单执行（`cyqnt_trd.exec_cli`, `cyqnt_trd.online_trading`）

**`exec_cli/`** —— 基于 `binance-cli` 的下单 / 仓位设置 / 交易所过滤器辅助。
**安全约定：每个下单与仓位函数都默认 `dry_run=True`；必须显式传 `dry_run=False` 才会
发出真实订单。**

```python
from cyqnt_trd.exec_cli import (
    market_order, limit_order, stop_market_order, cancel_all, partial_close,
    set_leverage, set_margin_type, exchange_filter_fetch, quantize, round_to_tick,
)
market_order("BTCUSDT", "BUY", 0.01)                 # dry-run preview (default)
market_order("BTCUSDT", "BUY", 0.01, dry_run=False)  # real order
```

返回带类型的 `OrderResult` / `ExchangeFilter`；出错抛 `CLIError`。见
`cyqnt_trd/exec_cli/README.md`。要走完整驱动的实盘流程，优先用上面的
`mvp_live_executor` 入口，它封装了这个库。

**`online_trading/`** —— `RealtimePriceTracker`，用于实时价格流。

## 数据工作流

推荐的数据工作流是：

1. 把 Binance K 线下载到本地 parquet
2. 存储最细的有用粒度，通常是 `1m`
3. 本地重采样到 `5m`、`15m`、`1h` 等
4. 让 `standard_bot` 跑在本地 parquet 上，而不是把原始 API 响应当作最终回测输入

这样能保持：

- point-in-time 对齐更清晰
- 本地回测可复现
- paper 信号与回测逻辑一致

---

## 测试数据（Test Fixtures）

`tests/blocks/fixtures/` 包含四份真实的 Binance OHLCV parquet 快照，用于指标集成测试：

| 数据 | 跨度 | 用途 |
|---|---|---|
| `BTCUSDT_1h_500bars.parquet` | ~21 天 | 主要指标测试 |
| `ETHUSDT_1h_500bars.parquet` | ~21 天 | 跨标的验证 |
| `BTCUSDT_4h_300bars.parquet` | ~50 天 | 高周期 SMC 结构 |
| `BTCUSDT_15m_500bars.parquet` | ~5 天 | 低周期噪声抽查 |

它们由测试套件直接加载（见
`tests/blocks/test_smc.py`, `tests/blocks/test_tradingview_indicators.py`）。

---

## `standard_bot` 内置策略族

`standard_bot` 主线内置以下策略，均以 `make_signals(df) -> (long, short)` 契约实现
（`signal/framework_strategies.py`），默认走 framework 回测引擎；paper/实盘也经由
`PythonLivePaperSession` 用同一契约驱动：

- `moving_average_cross`
- `price_moving_average`
- `rsi_reversion`
- `multi_timeframe_ma_spread`
- `donchian_breakout`
- `oi_funding_breakout`（需资金费 / OI 衍生品列）
- `liquidation_reversal`（需强平额度列）

对于基于 block / SMC 的自定义策略，请用 `--engine python`（见上面的[基于 Block 的策略](#基于-block-的策略)）。

---

## 验证与质量

- **测试套件**：414 条通过，1 条跳过，无回归
- **未来函数安全**：25/29 个指标通过
  `indicator(df[:i+1])[-1] == indicator(df)[i]` 测试验证为无未来函数
- **Atomic 数值一致**：12/12 个指标输出匹配 atomic 源到 `1e-9` 精度
- **真实数据冒烟**：每个指标都在覆盖 BTC/ETH × 1h/4h/15m 的 4 份 binance fixture 上验证

---

## 包说明

- 首选的历史回测引擎是 framework（`--engine framework`，默认），统一走 `cyqnt_trd.eval` 的向量化 `Σ W·R`
- 事件驱动参考引擎是 `--engine python`，用于基于 `cyqnt_trd.blocks.*` 库的自定义策略；旧的 Numba 编译引擎已删除
- 首选的 CLI 入口是 `cyqnt_trd.standard_bot.entrypoints.mvp_backtest`（`standard_bot` 执行线）
- 因子研究与向量化策略回测统一走 `cyqnt_trd.eval`；旧的 `cyqnt_trd/backtesting/` 已删除（功能被它取代）
- 旧的 `atomic_strategy_lib` 导入通过内置垫片包支持

---

## 文档

- `docs/CHANGELOG.md` —— 按日期索引的集成与功能开发日志
- `docs/atomic-compat/MIGRATION_HANDOFF.md` —— atomic → cyqnt_trd 集成设计
- `docs/atomic-compat/README.md` —— 垫片包速查
- `docs/cyqnt_trd_0_1_9_dev0_tutorial.md` —— 早期教程

各子包参考：

- `design/framework_design.md` —— 策略/回测框架与数据要求设计
- `cyqnt_trd/eval/README.md` · `cyqnt_trd/eval/STRUCTURE.md` —— 评测+回测框架（八阶段、六闸门、API）
- `cyqnt_trd/strategy_cases/README.md` —— 打包案例与支持矩阵
- `cyqnt_trd/exec_cli/README.md` —— 订单执行（dry-run 安全约定）
- `cyqnt_trd/standard_bot/STANDARD_BOT.md`, `cyqnt_trd/blocks/BLOCKS_API.md` —— 核心 API

---

## 依赖

关键依赖（有界区间；下界对齐部署基线，上界防止破坏性的大版本升级）：

- `pandas>=2.0.0,<3.0`
- `numpy>=1.24.0,<2.0`
- `polars>=1.0.0,<2.0`
- `pyarrow>=14.0.0,<25.0`
- `scipy>=1.10.0,<2.0`
- `matplotlib>=3.7.0,<4.0`
- `requests>=2.32.0,<3.0`
- `websockets>=15.0.1,<16.0`

Binance SDK 依赖：

- `binance-sdk-spot>=8.2.1,<10.0`
- `binance-sdk-derivatives-trading-usds-futures>=10.0.1,<11.0`
- `binance-sdk-algo>=2.6.0,<3.0`
- `binance-common>=3.8.0,<4.0`

这些上界是刻意的：pip 不会自动升级一个已经装有例如 `numpy 1.24.4` 或
`binance-common 3.8.0` 的部署，因此安装 `cyqnt-trd` 不会破坏依赖那些确切 ABI 的邻居
服务。`websockets` 的 `<16.0` 上界与 binance-SDK 家族自身的约束一致。

---

## 许可证

MIT License
