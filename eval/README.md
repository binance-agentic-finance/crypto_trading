# `eval/` — 因子评测矩阵

给一个因子，输出**六道裁决闸门、14 维证据矩阵、四种收益目标和八组诊断图**，回答它在当前数据与评估约定下的信息强度、关系形态、净收益、稳健性、基线重复性，以及失败或缺证的位置。

这是两阶段研究的第一步：**筛选和诊断单因子**。第二步才是组合因子、分配风险、设计仓位和执行规则，再验证策略收益。`PASS` 表示通过这张研究卡；组合策略在输出中始终标为 `NOT_EVALUATED`，不代表实盘许可或未来收益保证。

```bash
cd eval
python -m factor_eval demo
python -m factor_eval score --factor factor_eval.examples.example_factors:reversal_5d --report-dir report --markdown card.md --json card.json
```

```python
from factor_eval import evaluate

def my_factor(p):
    return -(p.close / p.close.shift(5) - 1)  # 5 日反转

card = evaluate(my_factor, name="rev5")
print(card.verdict)
print(card.blocking)                         # 失败或缺证的闸门
print(card.to_markdown())

from factor_eval.plots import render_diagnostics
render_diagnostics(card, "report")           # HTML、8 组 PNG、JSON 和 Markdown
```

默认面板和标定文件随仓库提供，安装依赖后可离线运行。报告由上述命令在本地生成，不纳入版本管理；浏览 HTML 时保留同目录 PNG。默认 `evaluate()` 和 `score` 都计算完整诊断，`--report-dir` 负责生成图表文件。`score` 在 `PASS` 或 `PASS_CONDITIONAL` 时返回 0，其余研究裁决返回 1；报告仍会正常写出。自动流程应读取具体裁决和闸门，不能把退出码当成交易许可。

## 一、14 维评估矩阵

矩阵把“测了什么、结果怎样、还缺什么”放在同一张表中。前六维对应 G0—G5 裁决闸门；其余维度记录附加证据的状态，**`MEASURED` 只表示有测量结果，不表示通过**。`NOT_EVALUATED` 或 `INSUFFICIENT` 会保留原因，不能在报告里消失。六道闸门的 `PASS` 不代表容量、搜索校正或封存留出已经获证。

| 维度 | 证据与边界 |
|---|---|
| `data_integrity` | 有限值、覆盖、有效日期、UTC 日网格、信号因果性抽查 |
| `information` | 冻结方向后的 RankIC、HAC 与匹配标定的噪声参考 |
| `structure` | dev 定规则的分箱、val 结构、主方向下的期限一致性 |
| `net_economics` | 完整持仓周期现金流、资金费、成本安全空间和压力曲线 |
| `robustness` | 分段、年份、季度、星期、事前波动状态与逐一剔除标的 |
| `baseline_novelty` | 五个公开基线的相关性与有剩余自由度的残差检验 |
| `four_target_comparison` | 四目标使用相同入出场区间和共同有效单元；另报各自可用样本 |
| `sampling_uncertainty` | 保留日历缺口、整块资产一起重采样的点估计及区间 |
| `horizon_and_delay` | h=1…60 和执行延迟曲线；原信号及主方向冻结 |
| `tradability` | 可选真实完整价差支持的 IC 对 N 诊断；未提供报价时明确缺证 |
| `capacity` | 当前 `NOT_EVALUATED`，需要深度、成交规模与市场冲击模型 |
| `search_selection` | 申报查看的候选数；多候选为 `UNADJUSTED`，未实现搜索校正 |
| `sealed_holdout` | 当前 `NOT_EVALUATED`；命名为 oot 不能证明样本未被查看 |
| `strategy_combination` | 当前 `NOT_EVALUATED`；因子组合、风险预算与策略收益属于第二阶段 |

### 六道裁决闸门

| 闸门 | 检查内容 | 结果的含义 |
|---|---|---|
| **G0 数据** | 三段有效日期、覆盖和激活、dev 方向冻结、因子前缀一致性 | 数据或因果性失败会否决；无法审计记为缺证 |
| **G1 信息** | 冻结方向的 RankIC、HAC t 与指定随机信号的参考分布 | 是否超过当前标定下的噪声参考 |
| **G2 结构** | val 分箱首尾差、单调性、同一主方向下的期限变化 | 是否存在可解释的关系，以及在哪些期限失效 |
| **G3 成本** | val/oot 完整周期净收益、盈亏平衡成本及安全空间 | 缺少信号、价格或资金费时不能用局部样本收益通过 |
| **G4 稳健** | 三段同号、逐年 IC、val 净收益 Sharpe | 效应是否只集中在少数时间段 |
| **G5 增量** | dev 与五个公开基线的相关性、去除基线后的残差信息 | 是否主要重复已知风格；此处没有组合优化 |

必需检查的 `FAIL` 或 `N/A` 都不能被其他检查的通过抵消。`WARN` 表示已测量的建议项未达到约定；`N/A` 表示没有足够证据，不能解释成通过。G0 失败后仍可展示其他诊断，但后面的高分不能改变拒绝结论。

分箱宽度只用 dev 决定，结构闸门读取 val；每个有效日期的每箱至少有两个标的，并在相同日期上比较各箱收益。期限一致性使用主期限在 dev 冻结的同一个方向，不允许各期限分别翻号后宣称一致。

## 二、裁决与两阶段边界

| 裁决 | 含义 |
|---|---|
| `PASS` | 单因子研究卡的必需检查均已完成并通过 |
| `PASS_CONDITIONAL` | 必需检查已完成，没有硬失败，但有建议项警告 |
| `HOLD_INCOMPLETE` | 必需证据缺失，或跳过了某道闸门 |
| `HOLD_STRUCTURE` | 信息和成本检查通过，但结构检查失败 |
| `HOLD_CONDITIONAL` | 信息和成本检查通过，但稳健性或基线重复性检查失败 |
| `HOLD_INFO` | 信息检查通过，但当前持有及成本约定下未通过成本检查 |
| `HOLD_WEAK` | 成本检查通过，但排序信息未通过检查 |
| `HOLD_SEARCH` | 本可通过单候选检查，但申报已查看多个候选，尚未校正搜索选择 |
| `REJECT` | 信息和成本检查都未通过 |
| `REJECT_DATA` | 数据或因果性检查失败 |

`HOLD_INFO` 的因子可以保留为后续研究材料，但不能据此断言它加入组合后会提升收益。G5 衡量的是相对公开基线的重复性；互补性、组合权重、共同回撤和组合后的交易成本，属于第二阶段需要另外验证的问题。

`--skip-incremental` 可用于快速诊断；跳过 G5 后不能产生完整通过。`--trials-seen` 应填写选择当前因子之前查看过的全部变体数，包括参数、变换和方向选择。值大于 1 时，原本的 `PASS*` 会改为 `HOLD_SEARCH`；已有失败或缺证结论保留。这个标记不实现多重检验校正，也不会自动发现未申报的试验。

## 三、四目标、统计口径与诊断图

令 `e=t+entry_lag`，四种目标统一使用 `open[e]` 到 `open[e+h]` 的区间。收益标签用于因子诊断，净交易收益另由现金流引擎计算。

| 目标 | 定义 |
|---|---|
| `raw_rtf` | `log(open[e+h] / open[e])` |
| `normed_rtf` | `raw_rtf / sigma_raw(t)` |
| `res_rtf` | `raw_rtf - beta(t) × benchmark_raw_rtf` |
| `normed_res_rtf` | `res_rtf / sigma_res(t)` |

`targets.py` 只用形成信号时已完成的 close 观测估计风险：默认在最近 60 个观测内，至少需要 40 个有限的历史 h 日收益；波动率与 beta 均按同一 h 日收益直接计算，**不使用 `sqrt(h)` 缩放**。beta 和残差波动率使用相同的资产—基准配对观测。基准默认 `BTCUSDT`，可用 `--benchmark` 指定；基准缺失、方差为零或历史不足时，相关目标保留为不可用，不自动选择替代基准。

四目标主对照使用共同有效单元，dev/val/oot 分别报告 RankIC、Pearson IC、日期等权的 pooled 相关和斜率、有效日期/资产对数及区间。各目标另有覆盖率、缺失原因和必要时的 `target_available` 对照；后者使用自己的有效样本、保留点估计及 HAC，不额外执行 bootstrap。共同样本为空时原始收益结果仍可查阅，但不能宣称完成了四目标比较。

横截面 IC 先逐日计算、再对日期等权平均，每日默认至少 5 个有限资产对且信号和目标都非常数。pooled 相关与斜率对每个日期赋总权重 1、再在日内资产间均分，可能含时间序列水平效应，**不是横截面 IC**。薄切片可以有 pooled 关系却没有有效 IC，两者样本数分别报告。

块 bootstrap 默认 100 次，块长度为 `max(10, 2*h)` 个日历日，保留缺失日期并整块移动同日全部资产；四目标同期限对照复用日期抽样，跨期限按各自块长估计。区间为 95% 逐点百分位区间，同时记录重复次数、块长度、有效重复及有效日期数。有限日期不足两块长度时不输出 bootstrap 区间；HAC 是另列的正态近似区间。它们没有对 60 个期限和多个切片做同时置信带或搜索选择校正。

| 图组 | 实现内容与解释 |
|---|---|
| 分布与覆盖 | 所有合格单元的直方图、CDF、零值、NaN 和无穷值数量；CDF 分母为有限观测 |
| 四目标分箱 | dev 日期等权分位数最多 25 箱，ties 不拆散；val 共同目标样本的分箱均值与块区间 |
| 过去收益条件切片 | dev 上按决策前一日已实现收益最多 20 箱；val 的 pooled 相关、斜率与横截面 IC 分列 |
| 期限与延迟 | h=1…60 原始/可用残差目标曲线；延迟取 `{1, entry_lag, entry_lag+1, entry_lag+2, entry_lag+5}`，持有长度固定 |
| 时间与状态稳定性 | val 的年、季度、星期、dev 定界的事前波动三组，以及逐一剔除标的后的 IC |
| IC 与交易难度 N | 可选报价计算 `N=事前同期限波动率/完整对数价差`，按 dev 一日 N 分组，固定入场延迟扫描期限 |
| 极端对错案例 | dev 冻结中心、尺度和 p99.5 信号/p99 目标阈值，在 val 中最多各六例，相近市场片段去重并标明信号、入场和平仓时间 |
| 成本压力 | 固定原持仓，在 dev/val/oot 上计算 0、0.5、1、1.5、2、3 倍单边交易成本；资金费在零交易成本时仍保留 |

诊断分箱对每箱的可用日期等权，各箱日期可以不同；这是关系形态描述，不能当成可交易分箱组合。G2 的结构闸门另外要求各箱在共同日期上具备足够标的。期限曲线和延迟曲线各自固定共同样本，并报告样本损失；若残差目标不可构建，期限图仍保留原始目标。切片和极端案例用于解释失败场景，不会自动选择新方向、期限或交易规则。极端案例按未来结果选取，不能据此估计实盘胜率。成本压力沿用完整/局部可核算结果的区别，不估计市场冲击，也未给出净收益置信带。

`spread` 当前通过 Python API 提供，必须是与面板完全对齐、形成信号时已知的正值或缺失值 DataFrame，单位为完整 `log(ask/bid)`：

```python
card = evaluate(my_factor, panel, name="my_factor", spread=decision_time_log_spread)
```

没有价差数据时 N 图保留 N/A；日线 high-low 不会被当成报价代理。N 仅衡量波动与价差的相对大小，不代表可盈利性或资金容量。日频面板不能检验日内时段差异。

### 输出与快速模式

```bash
# 完整矩阵及八组图；HTML、PNG、scorecard.json 和 scorecard.md 写入 report/
python -m factor_eval score --factor factor_eval.examples.example_factors:reversal_5d --report-dir report --bootstrap 100 --benchmark BTCUSDT

# 仅六道闸门；不产出完整诊断矩阵
python -m factor_eval score --factor factor_eval.examples.example_factors:reversal_5d --quick
```

`--quick` 不能与 `--report-dir` 同用。`--bootstrap 0` 保留诊断点估计和可计算的 HAC，关闭重采样区间；它不同于跳过整个诊断的 `--quick`。Python API 对应 `with_diagnostics`、`n_bootstrap`、`diagnostic_horizons` 和 `benchmark_symbol`。离线报告包含证据表，缺数据的图组会写明原因，JSON 中不可计算量使用 `null`。

## 四、标定与复现

默认噪声参考来自 100 条 AR(1) 随机面板，使用相同数据、dev 方向选择、执行延迟、成本和资金费路径。分位数及每项有效试验数在 [null_top10_h3.json](factor_eval/calibration/null_top10_h3.json) 中。

这些 p95 描述**指定随机信号生成方式**下的参考分布，不是任意因子或反复搜索后的通用显著性保证。卡片同时保留明确标注的约定，例如覆盖率、最少日期和成本安全空间；不是每个门槛都由随机实验推导。

标定契约绑定全部面板值及缺失值、掩码、时间和标的顺序、实际分段边界、主期限、成本、入场延迟及引擎协议和源码哈希。更换这些内容后必须重新标定；仅修改面板说明或复制同一份数据不会改变数据身份。旧文件缺少契约时会被拒绝，不能只给旧分位数补一个新哈希。

卡片还保存信号、评估代码、标定文件内容的哈希和运行库版本。因子源码可读取时也记录其哈希；交互式函数等无法取得源码的情形会明确留空。这些标识帮助追溯结果，不替代上游数据可得性和外部状态审计。

下列命令在 `eval/` 中运行，先生成新标定，再显式交给评分：

```bash
python -m factor_eval calibrate --trials 100 --h 5 --cost 12 --entry-lag 2 --out null_h5.json
python -m factor_eval score --factor factor_eval.examples.example_factors:reversal_5d --h 5 --cost 12 --entry-lag 2 --calibration null_h5.json --markdown card_h5.md --json card_h5.json
```

`calibrate` 至少需要 30 次试验，默认 100 次；更多试验有助于稳定尾部分位数。`--trials` 是标定试验数，和评分时的 `--trials-seen` 是两件事。

自定义面板时，`--bundle` 放在子命令之前；自定义分段时，标定和评分都传入同一个 `--splits splits.json`：

```bash
python -m factor_eval --bundle my_bundle.parquet calibrate --trials 100 --splits splits.json --out my_null.json
python -m factor_eval --bundle my_bundle.parquet score --factor my_ideas.py:my_factor --splits splits.json --calibration my_null.json --trials-seen 1
```

`splits.json` 必须包含 `dev`、`val`、`oot`，各自的值为 `[开始, 结束]`，按时间先后排列且不重叠。例如默认分段为：

```json
{
  "dev": ["2022-04-01", "2024-06-30"],
  "val": ["2024-07-01", "2025-09-30"],
  "oot": ["2025-10-01", "2026-09-08"]
}
```

日期形式的结束值包含当天；带时间的结束值是精确截止时刻。只有退出时点没有越过分段边界的标签和持仓周期才进入对应统计。

## 五、因子输入、时点与收益口径

推荐提供纯函数 `Panel -> DataFrame`，输出为日期 × 标的：

```python
import numpy as np

def low_volatility(p):
    ret = np.log(p.close / p.close.shift(1))
    return -ret.rolling(10, min_periods=10).std()
```

- 第 t 行标记 UTC 日开盘时刻，因子只能使用该日收盘时已知的数据。默认在 `open[t+2]` 入场，持有 h 日后在 `open[t+2+h]` 平仓；引擎已施加入场延迟。
- 各期限的方向只在 dev 选择，此后冻结；dev 必须在 val 和 oot 之前。主期限、变换和研究方向应在查看后续样本前确定。
- 框架会抽取多个截断点重跑函数，比较删除未来行前后的过去信号。负向 `shift`、居中滚动和全样本归一化等常见泄漏可因此被拒绝。预计算 DataFrame 没有可重跑的函数，因果性记为 `N/A`。
- 前缀一致性是抽查，不能证明外部文件、闭包、上游数据修订或数据发布时间都满足时点约束。函数应无副作用且输出可重复，数据可得性仍需审计。

示例 `look_ahead_trap` 故意把未来交易窗口作为信号。异常高 IC 只证明度量路径能读到这个正控；它本身不是通用泄漏检测器。现在该函数会被前缀审计拒绝，不能因高 IC 获得研究通过。

输入面板必须有唯一标的和完整 UTC 午夜日网格，所有字段对齐，掩码为明确布尔值。缺失行情应保留为 `NaN`，不能靠删掉整个日期压缩时间。已观测价格必须为正，OHLC 关系合法，成交量不能为负，无穷值和非法掩码会被拒绝。

收益使用非重叠、每周期固定初始预算、各标的持仓数量在周期内固定的约定，按简单价格变化计算现金收益。默认单边成本为 6.5 bp，每周期都支付开仓和平仓费用，相邻周期平仓重开不抵扣。资金费为逐次结算费率乘以提供的结算参考价格后按日累积；部分历史参考价格来自标记价格开盘代理，来源信息保留在面板元数据中。

当日有效信号少于引擎要求时，该周期记为未知，不能当作零收益空仓。完整的常数信号才可形成真实空仓。持仓缺价格或资金费也会使该周期无法完整核算；`net_available_bp` 只展示可核算子样本，不能替代 `net_bp` 使 G3 通过。JSON 中不可计算量使用 `null`，并保留完整周期数、计划周期数与复现契约。

## 六、默认标的池及限制

跟随仓库的面板为 2026-08-11—09-09 的 30 日成交额前十 USDⓈ-M 永续，日频数据覆盖 2021-01-01—2026-09-09：

BTC · ETH · SOL · ZEC · XRP · HYPE · DOGE · BNB · TRUMP · ENA

默认掩码要求已上市、至少 60 根观测和合约生命周期有效。平均每天约 7.7 个合格标的，分箱、残差回归和行情切片的统计功效均受限制。基线回归如果耗尽自由度，或者残差只剩机器精度噪声，会保留为不可用，不再把噪声重新排序成“增量”。

这些名字按快照日成交额选定后回看历史，存在事后选择偏差；时间变化的资格掩码不能消除这项偏差。参考研究提供逐月时点前十对照，但历史退市标的覆盖仍有局限，不能宣称已经获得无偏的全市场检验。

构建不同面板需要先准备参考研究的原始公开数据快照，原始数据不随仓库提供。在仓库根目录可运行：

```bash
python eval/factor_eval/build_bundle.py --pool historical --out my_bundle.parquet
```

获取快照的方法见 [参考研究 README](alpha101_crypto/README.md)。新面板需重新标定。扩大历史时点标的池、引入真正封存的验证窗口、补充搜索校正和容量估计，仍是第一阶段需要补充的证据。

## 七、从因子到仓位：完整链路

一行调用跑完 **评测 → 准入 → 合成 → 权重 → 调仓信号**：

```python
from factor_eval.pipeline import build_strategy, cards_frame, latest_orders

s = build_strategy({"mom_20": f1, "amihud": f2, "range_10": f3}, panel)
cards_frame(s.cards)      # 每个因子的裁决、冻结方向、各段 IC、是否准入
s.weights                 # 日期 × 标的 的目标权重 ← 核心产物
s.book                    # 权益曲线、换手、成本
latest_orders(s.book, panel)   # 最近一次调仓要发出去的单子
```

因子可以是 `callable` 也可以是 `DataFrame`，**量纲随意、正负号随意**：方向由评测的 dev 段
冻结，量纲由截面排名消掉。这两条有测试钉住 —— 把每个因子取负号、或整体乘 1000，
构建出来的权重必须逐格相同。

**评测到底给了构建什么**，按重要性排：

| | 来自评测 | 构建怎么用 |
|---|---|---|
| 方向 | dev 段 RankIC 的符号（冻结，不再翻转） | 合成前乘上去。不冻结方向直接交易，等于有一半因子在反着做 |
| 准入 | G0 数据闸门的**状态** | `FAIL` 拒绝（真读了未来），`N/A` 放行但标 `unverified` |
| 权重 | dev 段 \|IC\| | 默认**不用**（等权）；`scheme="ic"` 可开，但见下文 |

`FAIL` 与 `N/A` 必须分开：传进来的是一张 `DataFrame` 时评测无法在前缀上重跑它，因果性
只能记 `N/A` —— 那是"没验证过"，不是"验证不过"。按裁决里的 `blocking` 列表一刀切会把
两者混为一谈，结果是所有 `DataFrame` 因子全被拒之门外。而把算好的表包进
`lambda p: frame` 则**应当**判 `FAIL`：给它一段前缀，它照样吐出全历史。

单因子之外，`blueprint.py` 把规则型 / 事件驱动型 / 量价截面型写成同一副骨架 ——
触发 → 硬门 → 分层打分 → 裁决 → 方向 → 仓位 → 出场。接别人给的因子时 `Tier` 要用
`normalize="rank"`：实测同一套 band 在 `raw` 下，三个因子的建仓天数是 5 / 0 / 50
（有个因子全是负值，永远不触发），换成 `rank` 之后都是 ~325 天，band 的含义才与量纲无关。

**蓝图**把规则型、事件驱动型、量价截面型写成同一副骨架 —— 触发 → 硬门 → 分层打分 →
裁决 → 方向 → 仓位 → 出场。三类策略的差别只在填进去的内容，产出统一是一张
`日期 × 标的` 的目标权重表，交给模拟器记账，因此损益口径、成本、资金费、换手都可比。

调仓清单取自模拟器自己那本账（`book.fills`），不是另算一遍 —— 另算会出现"报表上的单子"
与"回测真交的单子"对不上，而这种不一致只会在上线之后才暴露。有测试钉住：清单按日合计的
成交额必须等于 `book.trades`。

**调参是这条链上最容易把钱调没的一步。** `optimize.py` 的立场是：不保证找到最好的参数，
只保证报出来的数字已经为"你搜了多少次"付过账。做法是三件事 —— 滚动前推（训练窗选参数、
其后未见过的窗记账）、选平台而不是选尖峰（邻域平滑后的曲面上取最优）、以及**留下试验
账本**（`diagnostics.py` 里"no DSR/PBO without a complete trial ledger"说的就是它，有了
账本，有效试验数 / DSR / PBO 才有输入）。

```python
from factor_eval.optimize import tune

out = tune(build, {"lookback": (10, 20, 40), "schedule": (3, 5, 10)}, panel)
out["stats"]      # 名义与有效试验数、DSR、PBO
out["verdict"]    # 集成有没有赢过选参数；"挑最好的"这个动作泛不泛化
out["book"]       # 整片网格净额集成的可部署组合
```

在默认面板上跑出来的结果值得写在这里，因为它与直觉相反：**选参数跑输了不选**。
截面动量族 216 个候选，滚动前推每窗挑一组参数得到 OOS Sharpe 0.99 / Calmar 1.03，
而把整片网格净额集成（不做任何选择）得到 1.65 / 2.23，最大回撤从 −48.9% 收到 −32.0%；
集成宽度从 1 扩到 216，OOS 表现单调变好，WFER 从 0.40 升到 2.36。结构完全不同的事件
反转族复现了同样的次序（Sharpe −0.07 → 0.51，最大回撤 −41.5% → −16.2%）。PBO 实测
0.66～0.71，即"样本内冠军"在样本外落到中位数以下的概率超过一半 —— 参数选择本身不泛化，
这正是集成占优的原因。集成不制造 alpha：在本来就弱的事件反转族上它只是把亏损收住。

因此这套流程的默认建议是**集成整片平台而不是挑一个点**：没有选择就没有选择偏差。
净额集成（先平均目标权重再模拟一次）比分别持有各子策略再平均收益更省换手，
两者结论方向一致。组合层风控的实测结论仍然是负面的：在集成之上叠回撤阈值减仓会把
年化从 +33.3% 压到 +21.5%，目标波动能把 Sharpe 从 0.96 抬到 1.10 但回撤反而更深。
这些数字来自单一面板和一次搜索，换标的池未必成立；可复现的是流程，不是收益。

## 八、目录与验证

| 路径 | 用途 |
|---|---|
| `factor_eval/` | `evaluate()`、六道闸门、14 维证据矩阵及 CLI |
| `factor_eval/engine.py` | 时间约定、冻结方向、IC 和完整周期现金流 |
| `factor_eval/bundle.py` / `provenance.py` | 输入验证、面板身份与标定契约 |
| `factor_eval/causality.py` | 可重跑因子的前缀一致性抽查 |
| `factor_eval/matrix.py` | 检查、门槛来源与裁决 |
| `factor_eval/targets.py` / `statistics.py` | 四目标、共同样本、日期等权关系及块 bootstrap/HAC |
| `factor_eval/diagnostics.py` / `plots.py` | 14 维证据盘点、秩自相关与分箱换手、八组离线图与报告 |
| `factor_eval/preprocess.py` | 写因子时可选的 `winsorize_mad` / `winsorize_quantile` / `zscore` / `neutralize` |
| `factor_eval/blueprint.py` | **策略蓝图**：触发 → 硬门 → 分层打分 → 裁决 → 方向 → 仓位 → 出场 |
| `factor_eval/strategy.py` | 因子合成、定仓（等名义/等风险）、滚动在线选因子 |
| `factor_eval/portfolio.py` | 组合模拟器：目标权重 + 现金，净额调仓、逐日资金费、组合层风控 |
| `factor_eval/pipeline.py` | **完整链路**：评测 → 准入 → 合成 → 权重 → 调仓买卖信号 |
| `factor_eval/optimize.py` | **调参**：试验账本、滚动前推、平台选择、参数集成、有效试验数/DSR/PBO |
| `factor_eval/baselines.py` | 五个公开基线与残差化，G5 用它判断"这只是已知风格" |
| `factor_eval/capability.py` | 形状适配：按标的的算子 → 面板因子，支持 `list[float]` 与 `rows` 两种输入 |
| `factor_eval/capability_registry.py` | **承接 capability**：manifest v3 → 面板因子，清单驱动接线、量纲归一、上游解析与一致性核对 |
| `factor_eval/library.py` | **因子库**：Alpha101 适配、播种抽样、dev 段秩相关去重 |
| `factor_eval/benchmarks.py` | **基准**：买入持有、等权一篮子、现金、单因子裸用、单资产择时，及分段并排指标表 |
| `factor_eval/experiment.py` | **端到端实验**：候选 → 矩阵 → 准入 → 去重 → 构建 → 对比 → 报告 |
| `factor_eval/report.py` | 评估卡的文本与 Markdown 输出 |
| `factor_eval/build_bundle.py` | 重建默认面板 |
| `factor_eval/examples/example_factors.py` | 可直接运行的示例因子 |
| `factor_eval/data/` / `calibration/` | 默认面板、capability 清单快照、来源元数据和随机信号标定 |
| `alpha101_crypto/` | Alpha101 参考研究及其历史实验结果 |

代码结构树、承接 capability 的实测发现，以及"构建组合 vs 买入持有 BTC / 等权一篮子 / 单因子裸用"
的完整对比结果，见 [STRUCTURE.md](STRUCTURE.md)。一条命令复现：

```bash
python -m factor_eval capabilities --rejected      # 清单里有什么、什么能跑、不能跑的原因
python -m factor_eval compare --source alpha101 --out report/
```

`alpha101_crypto/` 是框架的参考研究，既有 [REPORT.md](alpha101_crypto/REPORT.md) 及结果文件保留其原实验口径。本轮框架修复不表示重新完成了整套 Alpha101 实验；需要重跑时应明确记录新引擎、数据与标定版本。

该评估包独立于 `cyqnt.input/v1 → blocks → cyqnt.signal/v2` 交易路径，不负责生成交易指令。评分依赖 `pandas`、`numpy`、`scipy`、`pyarrow`；生成诊断图另需 `matplotlib`，参考研究取数另需 `requests`。在仓库根目录验证：

```bash
python -m pytest tests/eval -q
python -m pytest tests/test_no_leaked_secrets.py -q
```

这些测试覆盖缺证不能通过、输入与标定不匹配、时间边界、缺失周期、未来泄漏、分箱比较、残差退化、四目标历史估计、共同样本、日历块重采样、日期等权、报价缺失和报告输出等反例。测试通过证明这些约定得到执行，不证明因子具有未来超额收益。

---

## 顺带验到既有程式的两件事（**本部分不修**）

都是想重用 `cyqnt_trd/trading_signal/selected_alpha/` 时发现的。不修的理由：它们属于那个模块的 owner，而且修了会改变它现有 caller 看到的数字。

**1. 六条因子函数对任何输入都回 `0.0`。** `alpha71/73/77/88/92/96` —— 正好是外层为逐元素 `min`/`max` 的那六条 —— 都写成 `df=pd.Series`（类本身，不是实例）再 `df.at[...]`，抛错被函数底部的 bare `except Exception: return 0.0` 吞掉。不崩溃、有数字、而且恒定，正是 `AGENTS.md` 开头那一类。`cyqnt_trd/get_data/get_kline_with_factor.py` 会 import 这批。

```bash
python - <<'PY'
import numpy as np, pandas as pd, importlib
rng = np.random.default_rng(1)
for num in (71, 73, 77, 88, 92, 96, 101):
    fn = getattr(importlib.import_module(
        f"cyqnt_trd.trading_signal.selected_alpha.alpha{num}"), f"alpha{num}_factor")
    vals = []
    for _ in range(4):
        c = pd.Series(np.exp(np.cumsum(rng.normal(0, .02, 400))) * 100)
        d = pd.DataFrame({"open_price": c.shift(1).bfill(), "high_price": c*1.01,
                          "low_price": c*.99, "close_price": c,
                          "volume": rng.lognormal(10, .5, 400)})
        d["quote_volume"] = d.volume * d.close_price
        vals.append(fn(d))
    print(num, vals)
PY
# 71/73/77/88/92/96 -> [0.0, 0.0, 0.0, 0.0];  101 -> 四个不同的数
```

**2. 同目录的 `rank()` 是时间序列 rank，不是横截面 rank。** `alpha_utils.rank()` 是 `series.rank(pct=True)`，套在单一标的自己的历史上（函数收一个 `data_slice`、回一个 float）；而论文的 `rank()` 是每个日期跨宇宙取横截面。含 `rank()` 的公式（大多数）两者算的不是同一个量，值不可比，逐标的版本也产不出横截面分数——这就是 `alpha101_crypto/factors.py` 另外存在的原因：面板形状的**第二条路，刻意新增而非取代**。
