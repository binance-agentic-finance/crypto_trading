# 仓位答疑（position sizing Q&A）

> 配套：`design/framework_design.md`（§4.3 sizing / §5.1 回测内核）、`review/analysis.md`（六个开源项目对比）。
> 结论均对照仓库源码，关键处附文件/行号。核心一句：**别人大多止步于"方向/评级/预测分"，最多做"单笔风险仓"；只有把每类信号统一归约成一个可回测、可控风险的目标仓位 `W`，alpha 才变成钱。**

---

## 先看：别人的仓库里是怎么解决仓位的？（大多不解决 / 不关心）

| 项目 | 仓位产物 | 本质 | 证据 |
|---|---|---|---|
| **TradingAgents** | `position_sizing: str \| None`，如 `"5% of portfolio"`；外加 5 档评级(Overweight=increase exposure) | **自由文本，非权重**。单标的、无截面、无归一、不可执行 | `agents/schemas.py:171`（"Optional sizing guidance, e.g. '5% of portfolio'"）；`agents/utils/rating.py:24` |
| **MarketSenseAI** | `position_sizing ∈ {small,medium,large}`（按 confidence 阈值分桶）+ 入场=价×0.98/0.95、止损固定% | **离散桶 + 固定百分比启发式**，单标的，无连续权重、无 gross/杠杆口径 | `synthesis_agent.py:377-384`（按 confidence 分桶）、`:478`（`current_price*0.98`） |
| **RD-Agent** | `TopkDropoutStrategy(topk=50, n_drop=5)` | **等权 Top-k**：选中标的 `W=1/topk`，其余 0。是真权重，但**固定、与 forecast 强弱无关、不可换** | `scenarios/qlib/experiment/factor_template/conf_baseline.yaml:44-49`（委托 Qlib） |
| **Binance 模板 cases** | `position-sizer`：`fixed_pct / atr_volatility / kelly`；`risk_pct=2%`、`max_leverage=10`、`account_balance` | **单笔风险仓**（others 里对 sizing 最认真的）：单标的、逐笔、不与统一回测内核绑定 | `cases/position-sizer/config/scoring_config.json`（methods/kelly_defaults）；`cases/rsi-mean-reversion/...`（`max_loss_per_trade:200`, `max_concurrent_positions:3`） |
| **pandas-ai / MiRAGE** | 无 | 非交易系统，**完全无仓位概念** | — |

**判读**：TradingAgents/MarketSenseAI **不解决**（给方向和一句话）；RD-Agent **解决但锁死**（等权 top-k）；Binance cases **解决单笔风险仓但不做截面组合**；pandas-ai/MiRAGE **不关心**。没有一个把"任意信号 → 截面目标权重 `W` → 统一回测 `Σ W·R`"这条链打通。

---

## 问题 1：仓位的分子是什么、分母是什么？

"仓位"有**三种口径**，别把它们混在一起。本框架的**核心口径是①截面权重**，②③是它落到实盘/单笔时的换算。

### ① 截面权重口径（本框架核心，`framework_design.md §4.3`）

```python
# positions_from_forecast(mu, mask, gross=1.0, market_neutral=True, max_weight=0.1)
x = mu.where(mask)                       # 只在可交易集合内
if market_neutral: x = x - x.mean(axis=1)   # 截面去均值 → 美元中性
W = x / x.abs().sum(axis=1) * gross      # 归一到目标 gross
```

- **分子 = 该标的的信号强度 `x[t,s]`** —— forecast `μ`（预测收益/排序分）去掉当格截面均值后的值（可再乘冻结系数、可按过去波动缩放）。它带符号：>0 看多、<0 看空。
- **分母 = 当格截面所有可交易标的的强度绝对值之和 `Σ_s |x[t,s]|`**（再按目标 `gross` 缩放）。
- 结果 `W[t,s]` **无量纲**，满足 `Σ_s|W[t,s]| = gross`（总杠杆=gross）、`Σ_s W[t,s]≈0`（美元中性），单标的受 `max_weight` 封顶。
- 一句话：**分子是"这个标的相对别人有多值得押"，分母是"把全场押注归一到目标总仓"**。

### ② 资金/名义口径（落到实盘）

- `名义_s = W[t,s] × 账户权益E × gross`。把无量纲权重乘上真金白银。
- 当你问"这占我账户多少"：**分子 = 该标的名义敞口，分母 = 账户权益 E（或可用保证金）**。
- 实际**下单量 = 目标名义 − 当前名义**（需要当前持仓，见问题 3）。

### ③ 单笔风险口径（Binance cases / Kelly / ATR 的口径）

- `qty = 单笔风险预算 ÷ 止损距离`。**分子 = 可承受风险金额（`risk_pct × E` 或 `max_loss`）**，**分母 = 入场到止损的距离（%或 ATR 倍数）**。
- 这是"固定风险/波动率平价"单笔法（`cases/position-sizer` 的 `fixed_pct/atr_volatility/kelly`）。它回答"这一笔下多少手"，但**不回答"截面上多个标的怎么分配"**。

> 本框架的 sizing 层是**可插拔**的（§4.3 列了 符号仓 / 分位组合 / 比例仓 / 风险归一 `W/trailing_vol`），所以①里可以叠加③的风险归一与 Kelly——**三种口径在本框架里能统一，在别人那里是割裂的**。

---

## 问题 2：为什么说"有了仓位就能比别人做得好"？

准确表述：**仓位是把"预测对不对（IC）"翻译成"赚多少钱（P&L）、承多少险（risk）"的唯一通道**。没有一个像样的 `W`：

1. **算不出真实盈亏**。回测就是 `pnl = Σ_s W·R`（§5.1）。TradingAgents/MarketSenseAI/pandas-ai/MiRAGE 根本没有 `W`，所以它们没有 P&L——TradingAgents 只有评级、MarketSenseAI 的"benchmark"是 LLM 打分（`src/evaluation/run_evaluation.py`，评的是回答质量不是收益）。**连"做得好不好"都无法度量。**
2. **控不住风险**。`gross`（总杠杆）、`max_weight`（单标的上限）、美元中性（去 beta）、换手成本、资金费——全靠 `W` 才能表达和约束。方向/评级/一句话做不到。
3. **同样的 alpha，sizing 决定 Sharpe**。forecast 成比例的仓位（强信号多押、弱信号少押）+ 风险归一，在**同一份 alpha** 上拿到更高的风险调整收益。RD-Agent 用**固定等权 top-k**，等于把强弱信号一视同仁、把 sizing 锁死——**能回测出 Sharpe，但把 Sharpe 留在了桌上**。
4. **能组合、能中性、能扩容**。多因子/多标的合成、对冲、容量评估，都发生在 `W` 这一层。

**诚实的边界（避免过度承诺）**：仓位是**必要不充分**。alpha≈0 时，再精的 sizing 也不赚钱；sizing 的作用是"在给定 alpha 下把风险调整收益做到最优、并让整条链可回测可控"。所以正确说法不是"有仓位就一定赚"，而是：**别人连这个通道都没有或锁死了，我们把它做成一等公民且可插拔——同样的信号，我们能榨出更高、更可控、可复现的收益，并且能证明（回测）**。

---

## 问题 3：仓位要在对话之前，让我们的大模型感知吗？

**要，但要放对层。** 把"账户/持仓状态"和"信号计算"分开：

| 层 | 是否感知当前持仓 | 为什么 |
|---|---|---|
| **信号/目标仓层**（因子→forecast→目标 `W`，§3/§4.3） | **不感知，也不应该** | 目标仓只由"截至当格可得的信息"决定。把当前持仓喂进因子/forecast 会引入**路径依赖与偏差**（处置效应/锚定），破坏**无状态、无未来函数、前缀一致性**（§0/§6）——这是可向量化回测的根基。 |
| **执行/对话层**（下单、风控、给用户可执行建议，§4.4） | **必须感知** | 因为要回答"**该加仓还是减仓？还能买多少？**"：① 目标名义=`W×E×gross` 需要账户权益 `E` 当分母；② 下单增量=目标−当前，需要 `W_current`；③ 风控闸（`max_concurrent_positions`、`max_loss`、杠杆上限）要看现状；④ 止盈止损/加减仓路径（§4.4 有状态层）本身吃当前持仓。 |

**结论**：对话前给大模型注入的应是**账户上下文**（当前持仓 `W_current`、持仓成本/浮盈亏、权益 `E`、可用保证金、杠杆、已用风险预算），用于**执行口径与增量建议**；**不要**把它拌进因子/forecast。这正是本框架把"有状态执行"隔离到 §4.4 的原因。

对照：TradingAgents / MarketSenseAI 是**无状态单次快照，根本不感知你的持仓**，所以它们给不出"你现在这仓该加还是该减"的增量建议，只能给一个孤立的方向；Binance cases 的 `execution` 块（`account_balance`、`max_loss`、`max_concurrent_positions`）**感知账户**，但停在单标的执行、没有截面目标仓 `W`。

---

## 问题 4：前面说的 8 类策略，必须"拍到"问题 1 和问题 2 上

**原则**：无论 8 类信号机制（量价·技术指标 / 价格结构 / 衍生品资金费 / 套利价差做市 / 事件驱动 / 聪明钱链上 / 估值周期 / 概率预测市场，见 `review/analysis.md §0.1`）里的哪一类，产出都必须归约成：

- **拍到问题 1**：一个**可执行的目标仓位 `W`**（有明确的分子/分母、截面归一、gross/上限口径）；
- **拍到问题 2**：通过 **`Σ W·R`** 产生**可回测、风险可控**的收益。

即 `factor → μ(forecast) → W(sizing) → Σ W·R` 是**所有 8 类策略的公共下游**。谁"拍到"了、谁没拍到：

| 框架 | 覆盖的机制 | 拍到问题 1（仓位有分子/分母？） | 拍到问题 2（W·R 可回测的风险调整收益？） |
|---|---|---|---|
| **TradingAgents** | ①⑤⑧ | ❌ 没拍到——`"5% of portfolio"` 是字符串，单标的、无截面、无归一 | ❌ 无 `W`、无 P&L、无风险归一 |
| **MarketSenseAI** | ①⑤⑥ | ⚠️ 影子——confidence→small/medium/large 桶，但非连续权重、无截面/gross 口径 | ❌ 无回测，"benchmark"是 LLM 质量打分 |
| **RD-Agent** | ①(+研报⑤) | ⚠️ 只拍到一个**固定特例**——`W=1/topk` 等权，forecast 强弱不进权重、不可换 | ✅ 能回测 IC/年化/Sharpe，但 sizing 锁死→Sharpe 留在桌上 |
| **Binance cases** | ①②③ | ⚠️ 拍到**单笔风险版**——分子=风险预算、分母=止损距离（Kelly/ATR/fixed_pct），但单标的逐笔、无截面 `W` | ⚠️ 有单标的回测（Return/Alpha/Sharpe/Win%），非组合级 `Σ W·R` |
| **pandas-ai / MiRAGE** | — | ❌ 完全无仓位概念 | ❌ 无回测 |
| **本框架** | ①–⑧ 统一 | ✅ `positions_from_forecast`→`W∈ℝ^(cells×symbols)`：分子=去均值信号 `x_s`、分母=`Σ|x|×1/gross`、`max_weight` 封顶、可 vol 归一（§4.3） | ✅ `Σ W·R` + 双边成本 + 逐次资金费（§5.1），四类机制共用同一内核 |

**举例把机制"拍"成仓位（本框架）**：
- **量价·技术指标（①）**：`μ = z(短期反转×放量)` → `W = 比例仓/分位组合`（§3.4 + §4.3）。
- **事件驱动（⑤）**：新闻/情绪 → as-of 挂载 + 因果衰减成连续强度 `μ` → 同一个 `W`（§3.3）。
- **衍生品·资金费（③）**：funding/OI 入面板附加字段（§2.3）→ 截面 carry 排序 `μ` → `W`（§3.5）。
- **套利·价差（④）**：配对 zscore 作 `μ` → 截面 demean 天然做成多空对冲 `W`（§4.3 market_neutral）。
- 反例——**网格/定投/加仓（执行形态）**：这是**有状态路径**，不进 `W` 的无状态计算，落到 §4.4 执行细化层（默认关闭）。这类恰是 Binance "bot 化"模板的一大块，是本框架相对它们**要显式补齐**的部分。
- 反例——**概率·预测市场（⑧）**：是概率下注、收益不走 `Σ W·R` 收益面板，只宜作**事件情绪因子**接入，不当交易类型原生支持。

**一句话收尾**：8 类机制 × 千百个信号，最终都要"拍"到**同一个 `W` 定义（问题 1）**和**同一个 `Σ W·R` 回测内核（问题 2）**上——这就是本框架相对 TradingAgents/MarketSenseAI（没拍到）、RD-Agent（拍到一个锁死的点）、Binance cases（只拍到单笔风险）的结构性优势。

---

## 问题 5：我们把策略拆成一条链，和别人"一个可交易的策略"差在哪？

### 5.1 对照：别人的"可交易策略"长什么样

以 Binance 策略 case（others 里对交易最认真的那个，如 `cases/rsi-mean-reversion`）为标本——它就是一个典型的、可直接跑的"策略定义"：

```
指标(indicator) → 入场/出场条件(signal: buy/sell/hold) → 单笔仓(fixed_pct / atr / kelly) → 下单
                                    │
                          全部揉在一个 case 配置里，单标的、逐笔、有状态
```

它把**方向判断、下注大小、下单**压成了一步：`signal` 里既定了方向、也（通过 `position-sizer`）定了手数，然后直接下单。**没有独立的因子值、没有预测分、没有截面目标权重、没有统一回测内核**（`Return/Sharpe` 是单标的的）。TradingAgents/MarketSenseAI 更靠前——连 sizing 都只有一句话或一个桶。

### 5.2 我们比"一步到位"多拆了哪些步骤

我们的策略 = **8 阶段流水线**（`framework.py:STAGES` L58-76），可交易策略只对应其中被压扁的一小段：

| 阶段 | 我们（拆开） | 别人的可交易策略（压扁） |
|---|---|---|
| 0 sample / 1 panel | 显式的**格子划分**，多标的对齐成 `cells×symbols` 面板 | 单标的一根时间轴，无 panel |
| 2 **factor** | 因子=**一个不含决策的裸数** `f[t,s]` | 无——直接进条件判断 |
| 3 **preprocess** | winsorize / zscore / **neutralize**（`preprocess.py`） | 无 |
| 4 **forecast** | 多因子→**预期收益 `μ`**，`fit_betas` 拟合合成（`forecast.py:55/135`） | 无（方向即结论） |
| 5 **sizing** | `μ`→**截面目标权重 `W`**，gross/中性/cap/vol（`positions_from_forecast` L186） | **单笔风险仓**（`fixed_pct/atr/kelly`），无截面 `W` |
| 6 **backtest** | 统一 **`Σ W·R`** + 双边成本 + 资金费（`backtest_weights` L155） | 单标的逐笔回测，非组合级 |
| 7 compare | 对基准分段裁决 | 一般无 |
| — orders | `order = W_target − W_current`，隔离在**薄执行层**（§4.4） | 与 signal/sizing 揉在一起 |

**多出来的关键接缝就四个**：`factor`（裸数）、`forecast`（μ）、`sizing→W`（截面权重向量）、`ΣW·R`（统一内核）。其中 `forecast` **可退化**成取符号/恒等（`sign_positions` L294），`standard_bot` 的 `make_signals` 就是这个薄版本（信号→目标仓 ∈{−1,0,+1}→下单）——但它**仍然过 `W`**。所以真正不可省的是 **`W` 这道缝**。

### 5.3 每道缝能拿来做什么优化

| 拆出的缝 | 解锁的优化 |
|---|---|
| **factor 独立** | 因子**单独评测**（IC / 六道闸门），不必先承诺 sizing；`K` 个因子任意复用 |
| **preprocess** | winsorize 抗异常值；zscore/rank 带来**单位与符号不变性**（写 `f` 或 `−f`、×1000，`W` 不变——已校验）；neutralize 去行业/beta |
| **forecast (μ)** | 把 `K` 个因子放到**同一"预期收益"币种**里用 `betas` 加权合成（IC 加权 / ridge）；把"信号强弱"变成连续量而非布尔 |
| **sizing → W** | **只有在向量 `W` 上**才能表达：`gross` 杠杆、`Σw=0` 美元中性、单票 `max_weight` 上限、波动率目标、成本/换手感知收缩、容量评估 |
| **Σ W·R 内核** | 一次矩阵运算完成回测；**前缀一致性**审计（结构性无未来函数）；回测/纸面/实盘**同源**（都取 `W`） |
| **orders = ΔW** | no-trade band、手数/最小名义/合约乘数、现货不能做空、下一根开盘成交——**有状态的脏活全部关在最边缘的薄适配层** |

### 5.4 核心优势（为什么值得多拆这几步）

1. **关注点分离 + 各用各的损失函数**：预测该好不好用 **IC** 判、下注大小该好不好用 **Sharpe** 判。压扁后，一个亏损策略是糊的（alpha 差还是 sizing 差？）；拆开后归因是干净的。
2. **组合级风控只能住在 `W` 层**：gross / 中性 / 单票上限 / vol-target 是**跨标的的向量约束**，逐笔单标的的 sizing 物理上表达不了。
3. **单位/符号不变性 = 风控治理**：写 alpha 的人**无权顺手定杠杆**（值域大 1e6 倍不会把你仓位放大 1e6 倍）。压扁的定义里，因子作者就是风控。
4. **回测 = 纸面 = 实盘，共享同一个 `W`**：这正是本轮删掉 Numba、收敛到框架所依赖的接缝；压扁会让回测和实盘各自重推 sizing 而漂移（就是我们刚清掉的那类 bug）。
5. **复用是 `K+M+L` 而不是 `K×M×L`**：`K` 因子 × `M` sizing × `L` universe，拆开是相加的积木；压扁是相乘的定制策略。加一次 vol-target，所有因子受益。

**诚实的边界**：单标的 + 单因子 + 固定 sizing 时，forecast 那段确实是 overhead——所以框架**没强制**你做它（可退化为取符号 + 等额）。但 `W` 这道缝要留着：**它是单位不变性、组合风控、成本建模、回测-实盘一致性这四件事唯一的落脚点**。把 factor 直接接到订单，省掉的不是"多余的预测"，而是砍掉了让这四件事成立的接缝。

---

### 附：证据定位
- 本框架：`design/framework_design.md` §4.3（`positions_from_forecast` L305-314）、§5.1（`Σ W·R` + 成本 + 资金费）、§4.4（有状态执行层）、§6（前缀一致性）
- 8 阶段流水线：`cyqnt_trd/eval/framework.py` `STAGES`（L58-76）、`backtest_weights`（L155）；`preprocess.py`（winsorize/zscore/neutralize L19/41/52）；`forecast.py`（`fit_betas` L55、`make_forecast` L135、`positions_from_forecast` L186、`sign_positions` L294）；退化薄版本：`standard_bot/signal/framework_strategies.py`（make_signals → 目标仓 ∈{−1,0,+1}）
- TradingAgents：`agents/schemas.py:171`、`agents/utils/rating.py:24`、`agents/trader/trader.py:37`
- MarketSenseAI：`src/application/agents/synthesis_agent.py:377-384`（confidence→仓位桶）、`:466-479`（固定%入场/止损）、`src/evaluation/run_evaluation.py`（LLM 打分式"benchmark"）
- RD-Agent：`rdagent/scenarios/qlib/experiment/factor_template/conf_baseline.yaml:44-49`（TopkDropout topk=50/n_drop=5）
- Binance cases：`bn-strategy-cases/cases/position-sizer/config/scoring_config.json`（fixed_pct/atr/kelly、risk_pct=2%、max_leverage=10）、`cases/rsi-mean-reversion/config/scoring_config.json`（account_balance=1000、max_loss_per_trade=200、max_concurrent_positions=3）
