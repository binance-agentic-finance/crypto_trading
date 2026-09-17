# `eval/` 代码结构与实测结果

仓库 [`binance-agentic-finance/crypto_trading`](https://github.com/binance-agentic-finance/crypto_trading)，分支 `eval-alpha101/construct`。
以下链接全部指向该分支上的文件。

---

## 一、这条链路做完了什么

```
capability 节点 / Alpha101 公式
      │  前置加工     形状（按标的 → 日期×标的）、量纲（价位 → 可跨标的比较）、有效性
  候选因子
      │  评测矩阵     六道闸门、14 维证据、dev 段冻结方向
  评估卡
      │  准入         G0 失败剔除；dev |IC| 低于标定噪声地板剔除；REJECT 裁决剔除
      │  去重         dev 段秩相关 > 0.8 的近重复剔除
  准入因子
      │  策略构建     合成 → 定仓 → 上限 → 调仓 → 模拟
  策略
      │  对比         买入持有 BTC / 等权一篮子 / 现金 / 单因子裸用
  结论
```

**所有做选择的动作只用 dev 段。val 与 oot 只用于记账，从不参与筛选。**

---

## 二、实测结果（Alpha101，99 个候选全量）

一条命令复现：

```bash
cd eval
python -m factor_eval compare --source alpha101 --out report/
```

候选池 99，准入 14（dev |IC| ≥ 0.0396），单因子基准取 dev 段最强的 `alpha042`。

### oot（2025-10-01 … 2026-09-08，343 天）—— 构建组合全面胜出

| 组合 | 年化 | Sharpe | 最大回撤 | Calmar |
|---|---|---|---|---|
| **构建组合（中性）** | **+61.7%** | **2.13** | **−13.3%** | **4.64** |
| 构建组合（多头偏置） | +35.6% | 1.07 | −26.5% | 1.34 |
| 单因子裸用 `alpha042` | +39.3% | 1.35 | −14.8% | 2.67 |
| 等权一篮子 | −2.8% | 0.26 | −49.9% | −0.06 |
| 买入持有 BTC | −34.4% | −0.71 | −53.8% | −0.64 |
| 现金 | 0.0% | — | 0.0% | — |

三条都成立：

- **对 BTC 买入持有**：多赚 96 个百分点，回撤浅 40 个百分点。这一段 BTC 在跌，中性组合不吃方向。
- **对等权一篮子**：+61.7% vs −2.8%，回撤 −13.3% vs −49.9%。
- **对"随便拿最强的因子去买卖"**：+61.7% vs +39.3%，Sharpe 2.13 vs 1.35，回撤还更浅。
  这一条最关键 —— 它说明赢的是**构建**，不只是因子挑得好。

### val（2024-07-01 … 2025-09-30，457 天）—— 多头偏置变体的风险调整后更优

| 组合 | 年化 | Sharpe | 最大回撤 | Calmar |
|---|---|---|---|---|
| 构建组合（多头偏置） | +40.4% | **1.33** | **−23.2%** | 1.75 |
| 买入持有 BTC | +51.4% | 1.18 | −29.4% | 1.75 |
| 等权一篮子 | +93.5% | 1.33 | −50.7% | 1.84 |
| 构建组合（中性） | −11.8% | −0.38 | −38.8% | −0.30 |

val 是强牛市。多头偏置变体年化低于 BTC，但 **Sharpe 更高、回撤浅 6 个百分点** —— 同样的钱承担更少的波动。

### 必须一并读的四条

1. **val 段的中性组合是输的（−11.8%）。** 牛市里多空对冲跑输市场是结构性的，不是策略失灵。拿中性组合的绝对收益去比买入持有，比的是市场暴露而不是能力 —— 所以上面两张表分开列。
2. **dev 段构建不如单因子**（+5.1% vs +13.3%）。构建的优势出现在 val/oot 而不是 dev，这个方向是对的（dev 是用来做决定的，不是用来报成绩的），但也说明单次结果不宜外推。
3. **99 个候选的搜索没有做选择偏差校正。** 矩阵如实收到 `trials_seen=99`，因此裁决会落到 `HOLD_SEARCH`。[`optimize.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/optimize.py) 里的 DSR / PBO 尚未接进这条链路。
4. **换 capability 目录当候选源，结论相反。** 72 个可跑候选里只有 3 个通过完整矩阵，且全是波动率代理（`atr` / `atr_ratio` / `range_gain_pct`），方向互相对冲，合成后反而**稀释**了最强的那个（oot −45% vs 单因子 +47.9%）。这不是构建层的问题，是候选多样性的问题 —— 见下一节。

---

## 三、承接 capability 的实测发现

```bash
python -m factor_eval capabilities --rejected
```

清单快照取自 [`be/binance-ai-platform`](https://git.toolsfdg.net/be/binance-ai-platform) 的 `feat/nodesdk` @ `48a5a5c5`，
`capability-sdk/src/binance/strategy/node/capabilities`。244 份 manifest 里 **76 个是因子形状**（有序列/bar 输入且有数值输出），其中 72 个在本面板上可跑。

三件在接入时才暴露的事：

1. **`rows`（bar 字典列表）是主导输入形状**，52/76。原先的适配器只会喂 `list[float]`，
   也就是说大部分能力**根本接不进来**。已补上。
2. **一批能力输出的是价位**（`ema`、`sma`、`pivot_points.pp`、`supertrend`…）。按价位给币种做截面排序 =
   按币价高低排序，BTC 永远在 DOGE 前面。它能刷出稳定的 RankIC ≈ 0.12 却不含任何 alpha。
   处理办法不是猜，是对每个能力同时产出 `raw` 与 `ts_zscore` 两个候选交给矩阵判，候选数翻倍并如实计入 `trials_seen`。
   **G5 基线闸门正是识别它的那道闸门** —— 快速筛选默认关掉 G5，所以链路里加了一道 `confirm` 全量复评。
3. **清单与上游实现已经漂移**：`macd` 清单写 `fast_period` 上游是 `fast`；`bollinger` 清单声明 5 个输出端口而上游只回 3 路；
   `rsi` 清单有 `method` 上游没有。参数对不上时**报错而不是静默丢掉** —— 丢掉 `period=20` 让上游用默认的 14，
   会算出一个看着完全正常、但不是你要的那个因子。

另有 12 个能力需要面板没有的数据（`buy_volume` / `oi` / `spot_close`），如实记为不可跑，不做近似替代。

---

## 四、目录

### 数据与契约

| 文件 | 职责 |
|---|---|
| [`bundle.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/bundle.py) | 面板定义、输入校验、随库前十永续 bundle 的加载 |
| [`build_bundle.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/build_bundle.py) | 重建 bundle |
| [`provenance.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/provenance.py) | 面板身份与标定契约（改 `engine.py` 会使随库标定失配） |
| [`data/top10_daily.parquet`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/data) | 默认面板：10 个 USDⓈ-M 永续，2078 根日线 |
| [`calibration/null_top10_h3.json`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/calibration) | 随机信号标定，噪声地板的来源 |

### 因子接入 — **本次新增的主线**

| 文件 | 职责 |
|---|---|
| [`capability.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/capability.py) | 形状适配：按标的的算子 → 面板因子。**新增 `rows`（bar 列表）输入支持** |
| [`capability_registry.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/capability_registry.py) | **新增。** manifest v3 → 面板因子：清单驱动接线、量纲归一、上游解析、`impl.py` 离线运行替身、两条执行路的一致性核对 |
| [`data/capabilities_v3.json`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/data) | **新增。** capability 清单快照（含来源提交），只记契约不复制实现 |
| [`library.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/library.py) | **新增。** Alpha101 适配、播种抽样、dev 段秩相关去重 |
| [`preprocess.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/preprocess.py) | 写因子时可选的截尾 / 标准化 / 中性化 |
| [`examples/example_factors.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/examples/example_factors.py) | 示例因子，含一个故意读未来的对照 |

### 评测矩阵

| 文件 | 职责 |
|---|---|
| [`__init__.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/__init__.py) | `evaluate()` 入口、评估卡、复现身份 |
| [`engine.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/engine.py) | 时间约定、冻结方向、IC、完整周期现金流。**不要改，源码哈希写进了标定契约** |
| [`matrix.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/matrix.py) | 六道闸门、门槛来源、裁决 |
| [`targets.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/targets.py) / [`statistics.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/statistics.py) | 四目标、共同样本、HAC 与块 bootstrap |
| [`baselines.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/baselines.py) | 五个公开基线与残差化（G5 用它识别"这只是已知风格"） |
| [`causality.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/causality.py) | 可重跑因子的前缀一致性抽查 |
| [`diagnostics.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/diagnostics.py) / [`plots.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/plots.py) / [`report.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/report.py) | 14 维证据盘点、离线图、评估卡输出 |

### 策略构建

| 文件 | 职责 |
|---|---|
| [`pipeline.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/pipeline.py) | 完整链路：评测 → 准入 → 合成 → 权重 → 调仓单。**本次修掉一个缺失值 bug，合成只留一份实现** |
| [`strategy.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/strategy.py) | 因子合成、定仓（等名义/等风险）、滚动在线选因子 |
| [`portfolio.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/portfolio.py) | 组合模拟器：净额调仓、逐日资金费、组合层风控 |
| [`blueprint.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/blueprint.py) | 策略蓝图：触发 → 硬门 → 分层打分 → 裁决 → 方向 → 仓位 → 出场 |
| [`optimize.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/optimize.py) | 调参：试验账本、滚动前推、平台选择、DSR / PBO |

### 对比与实验 — **本次新增**

| 文件 | 职责 |
|---|---|
| [`benchmarks.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/benchmarks.py) | **新增。** 买入持有 / 等权一篮子 / 现金 / 单因子裸用 / 单资产择时，以及按 dev-val-oot 分段的并排指标表 |
| [`experiment.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/experiment.py) | **新增。** 端到端可复现实验：候选 → 矩阵 → 准入 → 去重 → 构建 → 对比 → 报告 |
| [`__main__.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/eval/factor_eval/__main__.py) | CLI。**新增 `compare` 与 `capabilities` 两个子命令** |

### 测试

[`tests/eval/`](https://github.com/binance-agentic-finance/crypto_trading/tree/eval-alpha101/construct/tests/eval) —— 本次新增
[`test_capability_registry.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/tests/eval/test_capability_registry.py)、
[`test_benchmarks.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/tests/eval/test_benchmarks.py)、
[`test_library.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/tests/eval/test_library.py)、
[`test_experiment.py`](https://github.com/binance-agentic-finance/crypto_trading/blob/eval-alpha101/construct/tests/eval/test_experiment.py)。

钉住的几条关键不变式：

- 单因子 + 不设上限的构建结果必须与 naive 基准**逐格相同** —— 否则"构建 vs 单因子"比的是记账差异而不是权重算法；
- 因子取负号，构建出的权重必须不变（方向由 dev 冻结）；
- 去重的相关性只在 dev 段算，val/oot 段改成什么样都不影响去重结果；
- 上游参数对不上时必须报错，不能静默使用默认值；
- 清单端口数与上游返回路数不符时必须报错，不能按位置猜。

```bash
python -m pytest tests/eval -q
```

---

## 五、复现

```bash
cd eval

# 承接 capability：看清单里有什么、什么能跑、什么不能跑及原因
python -m factor_eval capabilities --rejected

# 主结果
python -m factor_eval compare --source alpha101 --out report/

# 换候选源
python -m factor_eval compare --source capability --out report_cap/

# 反事实：关掉声称在起作用的机制，看结果变不变
python -m factor_eval compare --source alpha101 --no-dedupe   --out report_a/
python -m factor_eval compare --source alpha101 --no-confirm  --out report_b/
python -m factor_eval compare --source alpha101 --no-ic-floor --out report_c/

# 抽样模式（换种子看结论稳不稳，一个种子赢不算数）
python -m factor_eval compare --source alpha101 --k 12 --seed 1234 --out report_d/
```
