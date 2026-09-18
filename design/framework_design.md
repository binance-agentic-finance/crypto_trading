# 交易流水线与数据准备设计（standard_strategy_design）

> 来源说明：仓库里未找到 `design/白板笔记.md`（`design/` 下只有本空文件）。本文按需求
> 里给出的口述规格 + 仓库现有约定（`cyqnt_trd.blocks` 向量化、无未来函数；
> `cyqnt_trd.eval` 的 `Panel` / `evaluate_factor`；`standard_bot` 的 `cyqnt.input/v1`
> / `cyqnt.signal/v2` 契约）成文。若之后补上白板原图，可据此校准。

---

## 0. 一句话与核心恒等式

一个策略 = **格子划分逻辑** +（**因子** → **收益 forecast** → **仓位**）。

因子和 forecast **写成对前序状态无依赖的纯函数**（无状态、point-in-time、无未来函数）。
于是任意时刻、任意标的的仓位只由"当格及之前可得的信息"决定，回测退化成一次矩阵运算：

```
                每格·每标的的仓位          每格·每标的的前瞻收益
        W  ∈  ℝ^(cells × symbols)   ⊙   R  ∈  ℝ^(cells × symbols)
每格组合收益   pnl_gross[t] = Σ_s  W[t, s] · R[t, s]
        毛收益曲线   equity   = (1 + pnl_net).cumprod()
```

**为什么强制无状态**：

1. **可向量化**：`W` 和 `R` 一次性算出，回测就是 `(W * R).sum(axis=1)`，无 Python 逐 bar 循环。
2. **天然无未来函数**：只要每个格子的特征只用"截至当格开盘可得"的数据，`W[t]` 不可能看到 `t` 之后。
3. **可复现、可审计**：同一份数据 + 同一份因子/forecast ⇒ 逐位元相同的 `W`，可做前缀一致性校验（见 §6）。
4. **研究与实盘同源**：实盘就是取 `W` 的最后一行；回测和实盘跑同一段代码。

> 这与仓库既有的两条线是同一套哲学：`blocks`（整段 Series、trailing 窗口、look-ahead-safe）
> 和 `eval`（`Panel` 上冻结方向、`Σ 每周期仓位·收益`）。本框架把它们统一成"格子 × 标的"的
> 面板范式。

---

## 1. 核心抽象


| 抽象                | 记号 | 形状               | 定义                                                                                                                   |
| ------------------- | ---- | ------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| **格子 Cell**       | `t`  | 索引               | 一次决策/持有单元。由"格子划分逻辑"切出：等频（time bar）、等量（volume bar）、等额（dollar bar）、等波动（vol bar）等 |
| **标的 Symbol**     | `s`  | 列                 | 截面里的一个可交易合约                                                                                                 |
| **因子 Factor**     | `f`  | `cells × symbols` | 无状态映射：每格每标的一个数（连续值或离散票）。**只用截至当格可得的数据**                                             |
| **预测 Forecast**   | `μ` | `cells × symbols` | 由因子得到的**期望收益**（或期望排序分）。无状态映射 `factors → μ`                                                   |
| **仓位 Weight**     | `W`  | `cells × symbols` | 由 forecast（可含风险/成本约束）得到的目标仓位。无状态`μ → W`                                                        |
| **前瞻收益 Return** | `R`  | `cells × symbols` | 每格"可执行"的下一段收益，与`W` 对齐（`W[t]` 赚的是 `R[t]`）                                                           |
| **资格 Mask**       | `M`  | `cells × symbols` | 布尔：该格该标的是否可被排名/持有（上市、足够历史、在生命周期内）                                                      |

**无状态契约（本框架的硬约束）**

- 因子 `f`、forecast `μ`、仓位 `W` 都必须能写成：`value[t, s] = h(截至 t 可得的输入[..t, s] 以及当格截面[t, :])`，**不得**读写跨格的可变状态（如"上一次开仓价""持仓天数""连续 N 根"这类需要维护的变量）。
- 允许：滚动窗口、shift、截面 rank、冻结系数的线性/非线性映射 —— 这些都能整段向量化，且不引入未来。
- 不允许：`for` 循环里累积的仓位状态、`if 已持仓 then ...`、把未来标签回填到过去。
  - 这类"有状态"逻辑属于**执行层**（止盈止损、加减仓路径），不进因子/forecast；本框架把它隔离在
    可选的"执行细化"层（§4.4），默认关闭以保持向量化。

---

## 2. 数据准备：从原始数据到"格子面板"

数据准备的产物是一个**对齐的面板（Panel）**：所有帧共享同一套 `cells × symbols` 索引。

### 2.1 格子划分逻辑（Bar / Grid Sampler）

统一接口：把原始逐笔/1m 数据聚合成"格子"，并给每格标注**开盘时间戳**与**格内 OHLCV**。

```python
# 统一 sampler 契约：raw(细粒度) -> bars(格子)，每格带 open_time / open/high/low/close/volume/quote_volume
def sample_bars(raw: pd.DataFrame, scheme: str, **params) -> pd.DataFrame: ...
```


| scheme           | 每格边界               | 参数            | 适用                         |
| ---------------- | ---------------------- | --------------- | ---------------------------- |
| `time`（等频）   | 固定时长               | `interval="1h"` | 最常用；截面对齐天然         |
| `volume`（等量） | 累计成交量达阈值       | `threshold`     | 活跃度归一，抑制低流动性噪声 |
| `dollar`（等额） | 累计成交额达阈值       | `threshold`     | 跨标的可比的信息密度         |
| `vol`（等波动）  | 累计已实现波动达阈值   | `target_vol`    | 波动归一的采样               |
| `imbalance`      | 累计签名成交失衡达阈值 | `theta`         | 事件密度采样（进阶）         |

**关键：截面对齐。** 选币/截面策略要求同一 `t` 下各标的可比。等频天然对齐；等量/等额是**每标的
各自成格**，需再用"共同网格"重采样成对齐面板（或退化为单标的时序回测）。设计上：

- 单标的/时序策略：直接用该标的自己的 bar 序列做索引。
- 截面/选币策略：先各自成 bar，再对齐到一个公共 `cells` 网格（通常是等频），非对齐格记为 `mask=False`。

### 2.2 前瞻收益 R 的定义（可执行、带延迟）

`R[t, s]` 必须是**在 `t` 能观测到信号、之后才成交**的收益，不能用当格收盘反算当格：

```
令 e = t + entry_lag                      # 决策后延迟成交（默认 entry_lag=1~2 格）
R[t, s] = open[e + h, s] / open[e, s] - 1 # 持有 h 格，open→open，非重叠
```

- `entry_lag`：`t` 格产生信号，最早在 `t+1` 开盘可交易；`entry_lag=2` 再多等一格更保守。
- `h`：持有格数（主期限）。非重叠周期避免重复计费。
- 用 `open` 而非 `close`：`close` 是当格才知道的价，用它做入场是隐性未来函数。
- 缺价/缺资金费的格：`R` 记为 `NaN`（未知），**绝不填 0**（0 是"平仓/无收益"的明确语义，不是"未知"）。

### 2.3 面板结构（复用 `cyqnt_trd.eval.Panel`）

数据准备直接产出 `eval` 已定义的 `Panel`（见 §3 数据要求），字段：

```
open, high, low, close, volume, quote_volume  # 每格 OHLCV（决策/计价用）
funding                                        # 每格·每标的实际资金费（多头支出为正）
mask                                           # 资格布尔
meta                                           # 采样方案、排序窗口、选样警示、数据快照
```

**事件/聪明钱因子（§3.3）需要的附加字段**（可选，按需挂到同一 `cells × symbols` 网格上）：
`taker_buy` / `taker_sell`（主动买卖量→CVD/失衡）、`open_interest`、`long_short_ratio`、
`liq_buy` / `liq_sell`（爆仓）等衍生品字段，以及**已 as-of 对齐的事件流**（新闻/情绪脉冲，见 §3.3）。
这些字段同样遵守"缺值即 NaN、决策时刻可得"的口径；数据取自 `cyqnt_trd.data_cli`
（`fetch_oi_history` / `fetch_long_short_ratio` / `orderbook_imbalance` / `fetch_news` / `fetch_sentiment`）。

`R` 不入库（由 `open` + `entry_lag` + `h` 现算），保证"收益定义"与"引擎口径"绑定，不会两处漂移。

---

## 3. 因子定义框架（统一）+ 四类样例

### 3.1 统一签名

一个因子是**纯函数**，输入面板/序列，输出与之对齐的**整段值**：

```python
# 截面因子（推荐，选币/截面策略）：Panel -> DataFrame(cells × symbols)
Factor = Callable[[Panel], pd.DataFrame]

# 单标的因子：DataFrame(小写 OHLCV) -> Series(cells)  —— 可用 to_panel 广播成截面
SeriesFactor = Callable[[pd.DataFrame], pd.Series]
```

契约（与 `blocks` 一致）：

- **无状态 / 无未来**：`row t` 只能用 `≤ t` 的数据；用 trailing 窗口和 `shift`，不要自己再往前 shift"保险"。
- **列名小写** `open/high/low/close/volume/quote_volume`（`blocks` 约定）。
- **缺值即未知**：预热段/无效格返回 `NaN`，不要填 0。
- **方向不在因子里冻结**：因子输出"越大越看多"的原始强度；方向（正/负）在 dev 段统一冻结（见 §7），
  因子本身不做符号选择。

> 离散"方向票 {-1,0,1}"是连续因子的特例，仓库已有向量化实现：`cyqnt_trd.blocks.factor_votes`
> （`rsi_vote/cci_vote/...`）。下面四类样例统一用"整段 Series/DataFrame、无状态"的写法。

### 3.2 规则型（rule-based）

阈值/区间规则 → 分档。本质是"指标 + 边界"，整段可算：

```python
from cyqnt_trd.blocks import indicators as ind

def f_rule_rsi(df, period=14, lo=30, hi=70):
    """RSI 低于 lo 看多、高于 hi 看空，其余中性 —— 整段 {-1,0,1}，无未来。"""
    r = ind.rsi(df["close"], period)          # trailing，look-ahead-safe
    import numpy as np, pandas as pd
    v = np.where(r < lo, 1.0, np.where(r > hi, -1.0, 0.0))
    return pd.Series(v, index=df.index).where(r.notna())
# 等价于 cyqnt_trd.blocks.factor_votes.rsi_vote(df, period, lo, hi)
```

要点：规则里**没有**"上一次触发""持仓中"这类状态；每格独立由当格指标值决定。

### 3.3 事件驱动型（event-driven）

这里的"事件"是**外部离散信号**——**聪明钱**（链上/大单/交易所资金流、CVD、OI/爆仓、多空持仓）
和**新闻/情绪触发**（新闻、Binance Square buzz、情绪突变）。它们不是价格阈值，难点在于**时点对齐**
与**衰减**，而不是逻辑本身。要让事件因子仍然无状态、可向量化，靠两条：

> **事件驱动的无状态契约（关键）**
>
> 1. **时点对齐（as-of / point-in-time join）**：每条原始事件有自己的时间戳与强度。事件必须挂到
>    **第一个"决策时刻 ≥ 事件可得时刻"的格子**上（可得时刻 = 发布时间 + 采集/确认延迟），**绝不**挂到更早的格。
>    这一步把不规则事件流对齐成 `cells × symbols` 的稀疏矩阵，且天然不引入未来。
> 2. **衰减（decay / TTL）**：事件在后续格子的影响用"距事件的时长"的**因果核**表达，例如
>    `signal[t] = Σ_{事件 e, avail_e ≤ t} strength_e · exp(-λ·(t − t_e))`。这是过去事件的因果卷积/EWMA，
>    整段可算、无状态——把"发生了一次事件"变成一条逐格连续信号，不需要状态机。

**聪明钱**（复用 `blocks.microstructure` / `blocks.derivatives`；数据取自 `data_cli`）：

```python
from cyqnt_trd.blocks import microstructure as ms, derivatives as dv

def f_smart_money(df):
    """大单净流入 + 主动买卖失衡(CVD) + OI 异动，合成一条"聪明钱在进场"的连续强度。"""
    inflow = ms.smart_money_inflow(df)                       # 大额成交净流入(trailing z)
    cvd_div = dv.cvd_divergence(df["close"],                 # 价跌但主动买盘累积=背离看多
                                dv.cvd(df["taker_buy"], df["taker_sell"]))
    oi = dv.oi_change_pct(df["open_interest"], periods=1)    # OI 增仓配合价格=真突破
    z = lambda s: (s - s.rolling(96).mean()) / s.rolling(96).std()   # 因果标准化
    return z(inflow) + z(cvd_div) + z(oi)                   # 连续值；越大越"聪明钱看多"
```

**新闻 / 情绪触发**（复用 `blocks.news_features`；`data_cli.fetch_news/fetch_sentiment`）：

```python
from cyqnt_trd.blocks import news_features as nf

def f_news_pulse(events: pd.DataFrame, cells: pd.DatetimeIndex, symbols, halflife_h=6):
    """新闻/情绪事件 → 按标的、按格子的带符号脉冲(带半衰期衰减)。events 至少含
    available_at(可得时刻), symbol, sentiment(±强度)。"""
    scored = nf.score_sentiment(nf.classify_events(events))   # 事件分类 + 情绪打分
    # 1) as-of 对齐：每条事件挂到第一个 open_time ≥ available_at 的格（searchsorted，右侧）
    pos = cells.searchsorted(scored["available_at"], side="left")
    # 2) 落成稀疏事件矩阵，再沿时间做半衰期 EWMA(因果衰减)
    import numpy as np, pandas as pd
    pulse = pd.DataFrame(0.0, index=cells, columns=symbols)
    for p, sym, sc in zip(pos, scored["symbol"], scored["sentiment"]):
        if 0 <= p < len(cells) and sym in symbols:
            pulse.iat[p, pulse.columns.get_loc(sym)] += sc     # 同格多条事件累加
    lam = np.log(2) / (halflife_h)                             # 半衰期→衰减率
    return pulse.ewm(halflife=halflife_h).mean()               # 因果衰减，仅用过去事件
```

要点：

- 事件因子的"无状态"= **as-of 挂载 + 因果衰减**，而不是"当格布尔"。上例 `ewm`/`rolling` 都只看过去。
- `available_at` 必须是**信息真正可得**的时刻（含发布延迟/确认延迟），不是事件"发生"的理想时刻——
  这是事件类最容易引入未来函数的地方，务必在数据准备时就固定（§2.2 的口径同样适用于事件流）。
- "首次触发才进场""事件后跟踪止损"这类跨格路径属于执行层（§4.4），不进因子。
- 纯价格型的"突破事件"仍可写成规则型（§3.2，`close` vs 过去窗口极值），不必放这里。

### 3.4 量价型（price-volume，连续值）

连续强度，最适合直接进 forecast：

```python
def f_pv_reversal_x_volume(df, ret_win=5, vol_win=20):
    """短期反转 × 放量：越跌 + 越放量，越看多。连续值。"""
    import numpy as np
    reversal = -(df["close"] / df["close"].shift(ret_win) - 1.0)     # 越跌越大
    adv = df["quote_volume"].rolling(vol_win).mean()
    vol_shock = df["quote_volume"] / adv.replace(0, np.nan)          # 放量倍数
    return reversal * vol_shock
```

要点：输出连续值，不要在因子里就 clip/离散化（预处理见 §4.1）。

### 3.5 选币型（cross-sectional selection）

天然是"截面 Panel → DataFrame"，逐格在**资格截面内**排序：

```python
def f_select_carry(panel):
    """按 7 格资金费（谁在付钱持有）截面排序，负资金费=收 carry=看多。"""
    import numpy as np
    carry = -(panel.funding / panel.close).rolling(7).sum()          # 每标的时序
    # 只在资格集合内、逐格做百分位 rank（截面）
    ranked = carry.where(panel.mask).rank(axis=1, pct=True)
    return ranked - 0.5                                              # 居中，>0 看多
```

要点：`rank(axis=1)` 是**截面**排序（同一 `t` 跨标的），不引入时序状态；`.where(mask)` 保证
只在可交易集合里比较（截面泄漏的头号来源就是在 mask 外排名）。WorldQuant 101 alpha 见
`cyqnt_trd.blocks.alphas`（逐点版）+ `to_series`（整段版）。

---

## 4. 交易流水线：从因子到仓位

流水线（每一步都是无状态映射，整段可算）：

```
factors ──预处理──► 标准化因子 ──forecast──► μ(期望收益) ──sizing──► W(目标仓位)
 (§3)     (§4.1)                 (§4.2)                 (§4.3)
                                                          │
                                        (可选)执行细化 ◄──┘  §4.4，默认关闭
```

### 4.1 因子预处理（无状态）

统一量纲、抑制极端值，便于合成（复用 `cyqnt_trd.eval.preprocess`）：

- `winsorize_mad`：按中位数绝对偏差逐格截尾（窄截面稳健）。
- `zscore`：逐格截面标准化（合成前统一量纲；注意矩阵内部按 rank 打分时它不改裁决）。
- `neutralize`：逐格截面回归取残差，中性化到 size/低波等基线（**不做行业中性**——无可回测 PIT 行业）。

### 4.2 因子 → forecast（期望收益，无状态）

forecast 是"由因子预测的下一段收益/排序分"。**写法上是冻结系数的映射**——系数在 dev 段拟合一次、
之后固定，应用时逐格独立、无状态：

```python
# 线性合成：μ = Σ_k β_k · z(f_k)，β 在 dev 段用 IC 或岭回归拟合后冻结
def make_forecast(factors: dict[str, pd.DataFrame], beta: dict[str, float]):
    mu = None
    for name, F in factors.items():
        z = zscore(F)                      # 无状态标准化
        term = beta[name] * z
        mu = term if mu is None else mu + term
    return mu                              # cells × symbols 的期望收益(相对分)
```

- 单因子退化：`μ = sign_dev · z(f)`（方向在 dev 段冻结，见 §7）。
- 也可非线性（冻结的树/小网络），只要"给定因子值就出预测、不依赖历史顺序"即可保持无状态。
- **禁止**：用全样本统计（含未来）来标准化/拟合系数——那是未来函数。拟合只用 dev 段（§7）。

### 4.3 forecast → 仓位（sizing，可简单处理）

把期望收益映射成目标仓位。**默认给一个简单、无状态、可向量化的版本**：

```python
def positions_from_forecast(mu: pd.DataFrame, mask: pd.DataFrame,
                            gross=1.0, market_neutral=True, max_weight=0.1):
    x = mu.where(mask)                                  # 只在资格集合内
    if market_neutral:
        x = x.sub(x.mean(axis=1), axis=0)              # 截面 demean → 美元中性
    w = x.div(x.abs().sum(axis=1), axis=0) * gross      # 归一到目标 gross
    if max_weight:
        w = w.clip(-max_weight, max_weight)             # 单标的上限
        w = w.div(w.abs().sum(axis=1), axis=0) * gross  # 再归一
    return w.fillna(0.0)                                # 未知/不可交易 → 0 仓
```

更简单的档位：

- **符号仓**：`W = sign(μ)`（等权多空）。
- **分位组合**：取 μ 的 top/bottom 分位，组内等权、多空对冲。
- **比例仓**：`W ∝ μ`（forecast 越强仓越大），再归一。

风险归一（可选，仍无状态）：`W = W / trailing_vol` 用**过去**波动缩放，不引入未来。

### 4.4 （可选）执行细化层——有状态，默认关闭

止盈止损、加减仓路径、最小持有、冷却期这类**必须跨格维护状态**的逻辑，隔离在这一层，
**不进因子/forecast**。开启后回测退出"纯向量化"路径，走事件循环（较慢）。默认关闭：目标仓位
`W` 直接作为每格持仓，非重叠周期计费。这样"策略研究/因子评估"始终是向量化的。

---

## 5. 回测框架：输入 / 输出 / 内核

### 5.1 内核（向量化）

```
输入:  W (cells×symbols 目标仓位), Panel(open/mask/funding), entry_lag, h, cost_bps
1) R[t,s] = open[t+entry_lag+h]/open[t+entry_lag] - 1        # 前瞻收益，非重叠
2) 毛收益   gross[t]   = Σ_s W[t,s]·R[t,s]
3) 换手     turnover[t]= Σ_s |W[t,s] - W_prev[t,s]|          # 相邻周期仓位变化
   成本     cost[t]    = turnover[t]·cost_bps/1e4
4) 资金费   fund[t]    = Σ_s W[t,s]·funding_over_hold[t,s]    # 实际逐次资金费
5) 净收益   net[t]     = gross[t] - cost[t] - fund[t]
   缺 R/缺资金费的周期 → net[t]=NaN（未知），不并入官方均值
输出:  逐格 net 序列 + 指标
```

这正是 `cyqnt_trd.eval.engine.evaluate_factor` 已实现的口径（open→open、entry_lag、非重叠、双边成本、
逐次资金费、缺数传播为 NaN）。**本框架的回测内核 = 复用 `evaluate_factor` / 其向量化扩展**，不另造。

### 5.2 输入契约

```python
@dataclass
class BacktestSpec:
    panel: Panel                     # §3 数据面板（含 open/mask/funding/meta）
    factors: dict[str, Factor]       # 无状态因子集合
    forecast: Callable               # factors -> μ（含冻结系数）
    sizing: Callable                 # μ, mask -> W
    entry_lag: int = 2
    horizon: int = 3                 # 主期限（持有格数）
    cost_bps: float = 6.5            # 单边成本
    splits: dict = DEFAULT_SPLITS    # dev/val/oot
    grid: dict = None                # 格子划分方案(记录在案，便于复现)
```

### 5.3 输出契约

```python
@dataclass
class BacktestResult:
    weights: pd.DataFrame            # cells × symbols 目标仓位（可复现的 W）
    pnl: pd.DataFrame               # 逐格 gross/cost/funding/net(_bp)
    equity: pd.Series               # 净值曲线
    metrics: pd.DataFrame           # 分段(dev/val/oot)指标：RankIC、HAC t、净bp、Sharpe、换手、盈亏平衡成本
    diagnostics: dict               # 分箱形态、逐年、分位换手、极端案例…（可选，同 eval 诊断矩阵）
    config: dict                    # 引擎口径 + 数据指纹 + 冻结方向 + 复现标识
```

指标口径直接对齐 `eval` 的评分卡（六道闸门/裁决可选接入），避免重复造轮子。

### 5.4 数据流总览

```
raw klines(1m)                    因子集合 f_k        forecast μ         sizing
   │  get_data / data_cli            │                  │                 │
   ▼                                 ▼                  ▼                 ▼
sample_bars(scheme) ─► Panel ─► factors(Panel) ─► preprocess ─► μ ─► W ─┐
   (§2.1 等频/等量/等额)   (§2.3)        (§3)          (§4.1/4.2)          │
                                                                         ▼
                          BacktestResult ◄──── vectorized engine  W ⊙ R + 成本/资金费 (§5.1)
                          (pnl/equity/metrics)
```

---

## 6. 无状态契约的校验

两条自动检查（复用/呼应 `eval`）：

1. **前缀一致性（prefix-invariance）**：截断未来格重算因子/W，过去的值必须逐位元不变。
   `cyqnt_trd.eval.causality.check_prefix_invariance` 已实现；能抓住"中心窗口""全样本标准化""
   `shift(-1)`"等常见未来函数。
2. **向量化 ≡ 逐点**：对每格 `t`，`W[t] == sizing(forecast(factors(panel[:t+1])))[-1]`。
   `blocks.alphas.to_series` / `blocks.factor_votes` 的 look-ahead 测试就是这个模式。通过它即证明
   "研究(整段) = 实盘(取最后一行)"。

无状态 + 通过这两项 ⇒ 回测的 `Σ W·R` 与逐格滚动模拟**同解**，且无未来函数。

---

## 7. 数据划分与"冻结"（walk-forward）

- **三段**：`dev`（开发/拟合）→ `val`（验证）→ `oot`（样本外），时间不重叠、不回看。
- **只在 dev 上冻结**：方向符号、forecast 系数、分箱边界都在 dev 段确定后**冻结**，val/oot 只应用不重估。
- **收益格子的分箱**（"收益率格子划分"的一种研究用法）：用 dev 段的因子分位定 bin 边界，读 val/oot
  各 bin 的前瞻收益，看是否单调、跨期限稳定（对应 `eval` 的 G2 结构闸门）。
- 搜索即申报：扫参数属于搜索，单候选标定不再成立（对应 `eval` 的 `trials_seen` / `HOLD_SEARCH`）。

---

## 8. 与现有代码的映射


| 本设计的概念                                         | 复用的现有实现                                                                      |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------- |
| 面板`Panel`、前瞻收益、非重叠周期、成本/资金费、指标 | `cyqnt_trd.eval`（`Panel` / `engine.evaluate_factor` / `matrix` 六闸门 / `report`） |
| 无状态因子的向量化写法、look-ahead 安全              | `cyqnt_trd.blocks.indicators` / `blocks.factor_votes`（{-1,0,1} 票）                |
| 量价/统计因子、101 alpha                             | `cyqnt_trd.blocks.factors`、`cyqnt_trd.blocks.alphas`（+`to_series`）               |
| 因子预处理（winsor/zscore/neutralize）               | `cyqnt_trd.eval.preprocess`                                                         |
| 无状态校验                                           | `cyqnt_trd.eval.causality`、`blocks` 的 look-ahead 测试                             |
| 数据获取、bar 采样、资金费/OI/深度                   | `cyqnt_trd.get_data`、`cyqnt_trd.data_cli`                                          |
| 截面/选币输出、input/signal 契约                     | `standard_bot` 的 `cyqnt.input/v1`、`cyqnt.signal/v2`（`kind=selection`）           |

**建议落点**：把本框架实现为 `cyqnt_trd.eval` 的一个薄编排层（如 `eval.framework`）——因为回测内核、
面板、指标都在 `eval` 里；bar 采样器（`sample_bars`）放 `blocks.data` 或新增 `blocks.bars`；
因子/forecast/sizing 用 `blocks` + `eval.preprocess` 组合。**不新造引擎**。

---

## 9. 标准策略样式（把上面串起来的最小骨架）

```python
from cyqnt_trd.eval import Panel, evaluate                    # 面板 + 评测/回测内核
from cyqnt_trd.eval.preprocess import zscore
from cyqnt_trd.blocks import indicators as ind

# 0) 数据准备：原始 1m -> 等频(或等量/等额)格子 -> 对齐 Panel（含 open/mask/funding）
panel = build_panel(raw_1m, scheme="time", interval="1h")     # §2

# 1) 因子（无状态；可多个，四类混用）
def f_reversal(p):  # 量价型（截面广播）
    return -(p.close / p.close.shift(5) - 1.0)
def f_carry(p):     # 选币型
    return (-(p.funding / p.close).rolling(7).sum()).where(p.mask).rank(axis=1, pct=True) - 0.5

# 2) forecast：冻结系数的无状态合成（β 在 dev 段拟合后固定）
def forecast(p):
    return 0.6 * zscore(f_reversal(p)) + 0.4 * zscore(f_carry(p))   # μ

# 3) sizing：forecast -> 仓位（简单处理：截面中性 + 归一 + 上限）
def strategy(p):
    mu = forecast(p)
    return positions_from_forecast(mu, p.mask, gross=1.0)           # W；见 §4.3

# 4) 回测：向量化 W ⊙ R + 成本/资金费 + 分段指标（复用 eval）
card = evaluate(strategy, panel, calibration="auto")               # W 的最后一行即实盘目标仓位
print(card.verdict, card.metrics)
```

> 单因子研究时，`strategy` 可直接是单个因子（`evaluate` 会在 dev 段冻结方向、按分位构组合），
> 这就是当前 `cyqnt_trd.eval` 的用法；多因子/自定义 sizing 时，`strategy` 返回自定义的 `W`。

---

## 10. 数据要求：把多源数据建成能支持向量化回测的样子

数据远不止量价。`design/screening.json`（**B9 选标 API**）覆盖 20+ 个域：K 线、链上鲸鱼转账、
13F 大师持仓、新闻事件簇、讨论热度、宏观（通胀/就业/国债收益率）、财报、分析师评级/一致预期、
美股空头、板块/指数/ETF 成分、活跃标的集合等。要让 §1 的 `Σ W·R` 向量化回测成立，这些**异构、
多节奏、多来源**的数据必须先被整理成**统一的双时间轴特征仓库**，再**物化成对齐面板**。核心不是
"能不能拿到数据"，而是"**每个值在每个决策时刻是否真的可得**"。

### 10.1 现有数据面（按"进面板的形状"分类）


| 形状                                 | screening.json 域（举例）                                                                                          | 如何进面板                     |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------ | ------------------------------ |
| **每标的时序**                       | Candle(OHLCV)、Equity Short(空头/做空量)、Hotness(热度)、Analyst(评级/一致预期)、Financials(季度)                  | as-of 对齐到格子网格           |
| **事件流**（需 as-of + 衰减，§3.3） | News Event/Article、Chain(鲸鱼转账)、Corporate Action(分红/拆股)、Guru Holdings(13F 申报)、Earnings 日历、评级变更 | as-of 挂载 + 因果衰减          |
| **组/上下文**（需成分映射广播）      | Sector、Index members、ETF constituents、Macro(全局)                                                               | 按时点有效成分表广播到成员标的 |
| **宇宙/资格**                        | ActiveSymbol(可交易、未下架)                                                                                       | →`mask`（时点、无幸存者）     |
| **元/映射**                          | Asset(基础信息)、Criteria/Metadata、成分映射                                                                       | 驱动特征目录与实体解析         |

### 10.2 目标形态：分层 + 特征目录 → 对齐面板

```
来源(screening API / data_cli / 链上)
   │ extract（columnar / time-series / point）
   ▼
Bronze 原始层     每域各自 schema，保留原始时间戳与响应
   │ normalize：补双时间轴 + 实体解析
   ▼
Silver PIT 层     统一为 (entity, event_time, available_at, feature_id, value)，按域列存分区
   │ 由"特征目录"驱动
   ▼
Gold 特征仓库     feature_id × entity × available_at 的长表 / 列存
   │ materialize：as-of 前向填充(带 lag) + 事件衰减 + 成分广播
   ▼
Panel（cells × symbols）  ──►  §5 向量化回测 Σ W·R
```

### 10.3 硬要求（缺一不可）

**(A) 双时间轴（bitemporal）—— 回测正确性的头号前提。**
每个**非量价**字段必须带两个时间：`event_time`（值所指的时刻/期末）与 `available_at`（**真正可得**的
时刻 = 发布 + 采集/确认延迟）。回测在决策时刻 `t` 只允许用 `available_at ≤ t` 的最新值。
screening.json 的时序大多只回单一时间戳（`{period, timestamps, criteria}`），仅少数带
`publishTime`/`filingDate`/`periodEnd` —— 所以 **`available_at` 必须由数据层按各域已知延迟补齐**：


| 域                    | event_time       | 典型 available_at 延迟              |
| --------------------- | ---------------- | ----------------------------------- |
| Candle OHLCV          | bar 收盘         | ~0（收盘即可得）                    |
| 链上鲸鱼转账          | 区块时间         | 区块确认（分钟级）                  |
| Guru Holdings(13F)    | 季度末           | **~45 天**（SEC 申报截止）          |
| Financials 财报       | 财季末           | 盘后发布（earnings date）           |
| Analyst 评级/一致预期 | 评级日           | publish 当时                        |
| Equity Short 空头     | 结算日           | **T+多日**（交易所公布节奏）        |
| News / Hotness        | 事件时间         | `publishTime`（近实时，含抓取延迟） |
| Macro(通胀/就业)      | 所属月           | 发布日，且**会被修订**（见 E）      |
| Corporate Action      | ex / record date | 公告日（announce）                  |

**(B) 实体解析与时点映射。** 统一 canonical symbol key（跨 crypto 现货/永续、美股 ticker、
链上 token 地址）。成分/归属关系**随时间变**，必须存**时点有效版本**（`effective_from/to`）：
板块 / 指数 / ETF 成分、Guru 的 CUSIP→symbol、链上 token→标的。组/上下文特征（板块、指数、宏观、
ETF 资金流）经**时点有效成分表广播**到成员标的——绝不能用"今天的成分"回填历史。

**(C) 多节奏对齐。** 快（candle/链上/热度）直接对齐格子；慢（13F 季度、财报、宏观月度、空头双周）
用 **as-of 前向填充**（`available_at ≤ t` 的最新值），并保留"距上次更新时长"作为可选**陈旧度**特征；
事件（新闻、评级变更、公司行为、13F 申报）用 **as-of 挂载 + 因果衰减**（§3.3），**不**前向填充成常数。

**(D) 宇宙/资格无幸存者偏差。** `mask[t,s]` 取自 ActiveSymbol 的**时点集合**，且必须包含
**已下架标的在其存活期**的行（否则只在"活到今天的赢家"上回测，是幸存者偏差）。下架后 `mask=False`、
价格 NaN。

**(E) 缺失 = NaN，不填 0；不回填修订值。** 缺值是"未知"不是 0。Silver 层存**首次公布值**
（as-reported）与修订值两版；回测默认用 as-reported（决策时你只知道当时的值），不能用后来修订的终值。

### 10.4 特征目录（feature catalog）—— 对齐 screening 的 criterion

screening.json 的每个可筛项是带 `value_type` 的 **criterion**；把它升格为回测的**特征记录**，驱动
整个 ETL 与面板物化。它是数据层与因子层的**契约**：因子只按 `feature_id` 取列，新增数据源 = 加一条
目录 + 一个 extractor，**不动因子**。

```json
{
  "feature_id": "guru_holdings.institution_count",
  "domain": "guru_holdings",
  "source": "GuruHoldingsApi.getHolders",     // screening 每个接口尾部都标了 Source
  "entity_level": "symbol",                     // symbol | sector | index | global
  "dtype": "numeric",                           // 对齐 criterion value_type
  "cadence": "quarterly",
  "availability_lag": "45d",                    // 补 available_at 用
  "pit": {"event_time": "periodEnd", "available_at": "filingDate"},
  "as_of_fill": "ffill",                        // ffill | event_decay | broadcast | none
  "decay_halflife": null,                       // event_decay 时用
  "revisable": false
}
```

### 10.5 存储布局

- **列存 parquet，按 `域 × available_at 日期` 分区**；每行至少
  `(entity, event_time, available_at, feature_id, value)`（或按域宽表）。对齐 screening 的列存响应
  （`/api/query/columnar`、`/api/query/batch/columnar`），批量高效。
- **物化 Panel**：给定（universe、grid、feature 集合、`asof_cutoff`），从 Gold 层做 as-of/衰减/广播，
  产出 §2.3 的 `Panel`(+ 附加字段)，落 parquet（复用 `eval.save_bundle`），带**数据指纹**
  （`eval.provenance.panel_fingerprint`）与标定/回测绑定。

### 10.6 ETL 管线（接口草案）

```python
# 1) 抽取：screening API(选标多源) + data_cli(Binance 量价/衍生品) + 链上
extract(domain, entities, start, end) -> Bronze           # 原始响应，保留时间戳
# 2) 归一：补双时间轴 + 实体解析，落 PIT 长表
normalize(bronze, catalog[feature_id]) -> Silver          # (entity, event_time, available_at, value)
# 3) 物化：按 catalog 的 as_of_fill 规则拼成对齐面板（只用 available_at ≤ 决策时刻 的值）
build_panel(universe, grid, feature_ids, asof_cutoff) -> Panel
#    ffill(带 lag) / event_decay(§3.3) / broadcast(成分表) / OHLCV 直接对齐
```

### 10.7 数据校验 / QA（把无状态契约延伸到数据层）

- **PIT 审计（no-future）**：对随机截止时间 `T` 重建面板，`build_panel(..., asof_cutoff=T)` 的历史部分
  必须与用更晚截止重建的对应部分**逐值一致**（数据层版的前缀一致性，呼应 §6）。任何"未来修订回填/
  漏加 lag"都会在这里露馅。
- **幸存者偏差检查**：universe 历史必须含已下架标的；抽查某历史日成员数 ≈ 当时实际。
- **覆盖率 / 陈旧度**：每个 feature 每格的非缺失率、距上次更新时长分布；慢数据的陈旧度要显式可见。
- **修订对账**：as-reported vs 修订终值两版都留，回测用前者，并报告二者差异（宏观/财报尤甚）。

### 10.8 域 → 面板映射（把 screening.json 落到框架）


| screening 域            | 形状           | entity 级别             | 进因子的类型(§3)       | 关键 PIT 点      |
| ----------------------- | -------------- | ----------------------- | ----------------------- | ---------------- |
| Candle OHLCV            | 每标的时序     | symbol                  | 量价(§3.4)/规则(§3.2) | ~0 延迟          |
| Chain 鲸鱼转账          | 事件流         | symbol(token)           | 事件-聪明钱(§3.3)      | 区块确认         |
| Guru Holdings 13F       | 事件流 + 慢    | symbol                  | 事件-聪明钱(§3.3)      | **45d 申报延迟** |
| Equity Short 空头       | 每标的时序(慢) | symbol                  | 量价/positioning        | 公布 T+n         |
| News/Article/Hotness    | 事件流         | symbol/global           | 事件-新闻(§3.3)        | publishTime      |
| Analyst 评级/一致预期   | 时序 + 事件    | symbol                  | 事件/规则               | publish 当时     |
| Financials 财报         | 慢(季度)       | symbol                  | 规则/量价               | 盘后 + 日历      |
| Macro 通胀/就业/国债    | 时序(全局)     | global→broadcast       | 规则/条件切片           | 发布日 +**修订** |
| Sector/Index/ETF        | 组 + 成分      | sector/index→broadcast | 选币/中性化基线         | 成分时点有效     |
| ActiveSymbol            | 宇宙           | symbol                  | →`mask`                | 时点、无幸存者   |
| Asset/Criteria/Metadata | 元/映射        | —                      | 驱动目录与实体解析      | —               |

> **一句话**：回测能不能做，取决于数据层能不能回答——"在**决策时刻 t**、关于标的 s，我**当时真正
> 知道**的这个特征值是多少"。把 screening.json 的每个域整理成**带双时间轴、时点有效映射、无幸存者
> 宇宙**的特征仓库，再按 as-of / 衰减 / 广播物化成对齐面板，`Σ W·R` 才既向量化又不偷看未来。

---

## 11. 待定 / 后续

- **等量/等额 bar 的截面对齐**：需要一个"各标的成 bar → 对齐公共网格"的重采样器（§2.1），
  截面策略必需；单标的时序策略不需要。
- **自定义 `W` 的回测入口**：`evaluate` 目前以"因子→内部构组合"为主；若要直接喂任意 `W`，
  需在 `eval.engine` 暴露一个 `backtest_weights(W, panel, entry_lag, h, cost_bps)` 的薄入口
  （内核已具备：`Σ W·R` + 换手成本 + 资金费）。
- **执行细化层**（§4.4）：默认关闭；如需止盈止损/加减仓路径，走 `standard_bot` 的事件引擎，
  与本向量化研究路径分开。
- **数据层落地（§10）**：先建"特征目录 + 双时间轴 Silver 层 + `build_panel` 物化 + PIT 审计"这条最小
  链路；`data_cli` 已覆盖 Binance 量价/衍生品，screening API 覆盖选标多源，二者归一到同一特征仓库。
- **时点有效成分表**：板块/指数/ETF/13F 的成员映射需要历史版本（`effective_from/to`），是选币与广播类
  因子的前置；建议优先建。
- **补充**：若找到 `design/白板笔记.md` 原图，用它校准"格子划分"的具体口径与优先级。
