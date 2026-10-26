# `cyqnt_trd/eval/` — 结构与用法

因子评估矩阵 + 策略构建，八个阶段，每个阶段的输入输出格式是显式的。

在仓库根目录运行：

```bash
python -m cyqnt_trd.eval stages          # 阶段表与每阶段 I/O
python -m cyqnt_trd.eval capabilities    # capability 清单里哪些能在本面板上跑
python -m cyqnt_trd.eval compare --source alpha101 --out report/
```

---

## 一、八个阶段

| # | 阶段 | 输入 | 输出 | 实现 |
|---|---|---|---|---|
| 0 | sample | 每标的细粒度 bar | `dict[symbol, DataFrame]` | `bars.sample_bars` |
| 1 | panel | bars + extras + 资格 | `Panel`（cells × symbols） | `bars.build_panel` |
| 2 | factors | `Panel` | `dict[name, DataFrame]` | `capability_registry` / `library` |
| 3 | preprocess | 同上 | 同上 | `preprocess` |
| 4 | forecast | factors + `Panel` | `DataFrame`（期望收益 μ） | `forecast` |
| 5 | sizing | μ + mask | `DataFrame`（目标仓位 W） | `forecast.positions_from_forecast` |
| 6 | backtest | W + `Panel` | `BacktestResult` | `framework.backtest_weights` |
| 7 | compare | 结果 + 基准 | 分段 `DataFrame` | `benchmarks.compare` |

阶段 2–5 是**无状态映射**：`value[t] = h(截至 t 的输入, t 的截面)`。所以回测是一次矩阵运算，而 `W` 的最后一行**直接就是实盘目标仓位**，研究与实盘走同一段代码。止盈止损、加减仓路径需要跨格状态，属执行层，不在这条链上。

`framework.check_stateless()` 用截断重算验证这个声明，而不是相信它。四类泄漏（`shift(-1)` / `shift(-3)` / 全样本标准化 / 中心窗口）都会被拒绝，因果因子通过 —— 检测器自带故意泄漏的对照组，否则不知道它是不是永远返回通过。

**所有做选择的动作只用 dev 段；val 与 oot 只用于记账。**

---

## 二、执行与记账口径

```
决策日 t 收盘出信号 → 等 entry_lag=2 格 → open[t+entry_lag] 成交 → 持有 h 格（非重叠）
net = W·R − 换手×成本 − 逐次资金费
```

持仓标的**缺价或缺资金费，整期记 NaN 不记 0** —— 未知的成本不是零成本，这些期单独计数。

两个引擎，职责不同、口径一致：`engine.evaluate_factor` 评**一个因子**（自己构组合），`framework.backtest_weights` 跑**调用方给定的任意 W**。

> ⚠️ **不要改 `engine.py`** —— 它的源码 sha256 写进了 `provenance.calibration_contract`，改了随库标定失配，每次 `evaluate()` 都会抛异常。

---

## 三、目录

### 数据与契约
| 文件 | 职责 |
|---|---|
| `bundle.py` | `Panel` 定义与校验；`extras`（衍生品/事件附加帧）、`cell_scheme`（`daily_utc`/`regular`/`irregular`） |
| `bars.py` | 格子采样（time / volume / dollar / vol）、跨标的对齐、建面板 |
| `provenance.py` | 面板身份与标定契约 |
| `data/` `calibration/` | 默认面板、capability 清单快照、随机信号标定 |
| `snapshot/` | 原始快照取数（`acquire`）→ 装载（`pipeline`）→ 重建/扩展面板（`build_bundle`），见 README 第六节 |

### 因子接入
| 文件 | 职责 |
|---|---|
| `capability.py` | 形状适配：按标的的算子 → 面板因子。支持 `list[float]` 与 `rows`（bar 字典列表）两种输入 |
| `capability_registry.py` | manifest v3 → 面板因子：清单驱动接线、量纲归一（`raw` / `ts_zscore`）、上游解析、两条执行路的一致性核对 |
| `library.py` | Alpha101 适配、播种抽样、dev 段秩相关去重 |
| `alpha101.py` | Alpha101 的 101 条面板形状公式 |
| `adapters.py` | 单标的 `make_signals(df) -> (long, short)` → 目标权重 → 本框架回测（standard_bot 的 `--engine framework` 走这里） |
| `cases.py` | 外部**分层策略规格**（硬门 + 分层打分 + 裁决阈值）→ blueprint → 回测 |
| `preprocess.py` | 截尾 / 标准化 / 中性化 |

### 评测矩阵
| 文件 | 职责 |
|---|---|
| `__init__.py` | `evaluate()` 入口、评估卡、复现身份 |
| `engine.py` | 时间约定、冻结方向、IC、完整周期现金流（**勿改**） |
| `matrix.py` | 六道闸门、门槛来源、裁决 |
| `baselines.py` | 五个公开基线与残差化（G5 用它识别"这只是已知风格"） |
| `causality.py` | 前缀一致性抽查 |
| `targets.py` `statistics.py` | 四目标、HAC、块 bootstrap |
| `diagnostics.py` `plots.py` `report.py` | 14 维证据、离线图、评估卡输出 |

### 构建与回测
| 文件 | 职责 |
|---|---|
| `framework.py` | 阶段定义与编排、`backtest_weights`、无状态校验 |
| `forecast.py` | dev 段冻结系数（ic / equal / ridge + shrink）、仓位（中性 / 上限 / gross 三者同时成立） |
| `pipeline.py` | 评测 → 准入 → 合成 → 权重 → 调仓单 |
| `strategy.py` `portfolio.py` | 合成定仓、净额调仓模拟器 |
| `blueprint.py` | 触发 → 硬门 → 分层打分 → 裁决 → 方向 → 仓位 → 出场 |
| `optimize.py` | 试验账本、滚动前推、平台选择、DSR / PBO |
| `benchmarks.py` | 买入持有 / 等权一篮子 / 现金 / 单因子裸用 / 单资产择时 + 分段指标表 |
| `experiment.py` | 端到端可复现实验与报告 |

---

## 四、两处要读的口径

**承接 capability。** 清单快照记录契约（数据口、参数、输出端口）与来源提交，不复制实现。三件接入时才暴露的事：`rows`（bar 字典列表）是主导输入形状，原先喂不进去；一批能力输出的是**价位**，按价位给币种排序＝按币价排序，所以每个能力同时产出 `raw` 与 `ts_zscore` 两个候选交给矩阵判，候选数翻倍并如实计入 `trials_seen`；清单与上游实现已有漂移，参数对不上时**报错而不是静默丢掉**（丢掉 `period=20` 让上游用默认的 14，会算出一个看着正常但不是你要的因子）。

**接外部策略规格。** `cases.py` 默认**降级模式**：能算多少跑多少，并把放弃了什么说清楚。缺 tier 时按存活满分份额等比缩放裁决门槛（满分 10 要 5 分 → 满分 4 要 2 分，选择性不变），否则门槛永远够不着、回测出来是一条平线加 0.0，读起来像"这策略不赚钱"而不是"这个面板跑不了这个策略"。丢掉的硬门记在 `coverage` 与 `assumed` 里 —— 结果是**有筛选策略的无筛选版本**，报告要一直这么说。`strict=True` 回到拒绝行为，问的是另一个问题："这是不是当初写下来的那个策略"。

```python
r = run_case(load_case("path/to/scoring_config.json"), panel)
r.coverage["unavailable"]       # 哪些 tier 算不了、为什么
r.coverage["entry_score_used"]  # 实际用的门槛（缩放/夹取后）
r.assumed                       # 替代了哪些阈值、丢了哪些硬门
r.backtest.metrics              # dev / val / oot 分段
```

---

## 五、验证

```bash
python -m pytest tests/eval -q
python -m pytest tests/test_no_leaked_secrets.py -q
```

钉住的关键不变式：单因子 + 不设上限的构建必须与 naive 基准**逐格相同**；因子取负号权重不变（方向由 dev 冻结）；去重只看 dev 段；上游参数对不上必须报错；清单端口数与上游返回路数不符必须报错；`ts_zscore` 必须滞后；一条从未开仓的规格必须报成拒绝而不是 0.0 收益。

测试通过证明这些约定得到执行，**不证明任何因子或策略具有未来超额收益**。
