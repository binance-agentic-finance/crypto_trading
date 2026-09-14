# `eval/` — 因子评测矩阵 / the factor evaluation matrix

给一个你自己设计的因子，回答一个问题：**它好不好，卡在哪一关。**

```bash
cd eval
python -m factor_eval demo                      # 先看五个例子因子各卡在哪
python -m factor_eval score --factor my_ideas.py:my_factor --markdown card.md
```

```python
from factor_eval import evaluate

def my_factor(p):                               # p 是对齐好的日频面板
    return -(p.close / p.close.shift(5) - 1)    # 5 日反转

card = evaluate(my_factor, name="rev5")
card.verdict     # 'REJECT'
card.blocking    # ['G1_information', 'G3_cost'] —— 第一个 ✗ 就是结论
print(card.to_markdown())
```

跑完不需要任何下载：标的池、资金费和可交易掩码都随仓库带着（0.9 MB），单个因子约 2 秒。
一张渲染好的卡片长什么样，见 [`factor_eval/examples/example_card.md`](factor_eval/examples/example_card.md)。
`score` 子命令只有裁决是 `PASS*` 时才返回 0，方便直接挂进脚本或 CI。

---

## 一、矩阵：六道闸门，按"最便宜的先失败"排

| 闸门 | 它问什么 | 没过意味着 |
|---|---|---|
| **G0 数据** | 我们到底量到东西没有？ | 分数是产物不是结果 |
| **G1 信息** | 排序能不能赢过随机信号？ | 没有信息 |
| **G2 结构** | 这个关系长得像能用的吗？ | 有信息，但不单调 / 换期限就变 |
| **G3 成本** | 扣掉手续费和资金费还剩什么？ | **硬否决**——收不上来的信息 |
| **G4 稳健** | 换时间段、换年份还在吗？ | 是一个行情，不是一个效应 |
| **G5 增量** | 比公开基线多了什么？ | 只是规模 / 低波 / 反转的换皮 |

**第一个 ✗ 就是结论。** 后面的闸门是补充说明，不能用来抵消它。

### 门槛从哪来——这是这套矩阵唯一值得信的地方

所有"信息"和"钱"的门槛，都是**随机信号走同一条管线**得到的 p95，不是拍的：

| 指标 | 随机信号 p95（dev / val / oot） |
|---|---|
| RankIC | 0.040 / 0.043 / 0.049 |
| IC 的 HAC t | 2.08 / 1.88 / 2.05 |
| 净收益 bp/周期 | 10.0 / 26.6 / 35.6 |

100 条 AR(1) 随机面板，同样冻结方向、同样入场延迟、同样成本与资金费。**一个因子的 IC 0.03 不是"弱但有"，而是"还没证明自己不是噪声"。** 其余门槛在卡片上逐条标了 `convention`（可以争）还是 `required` / `hard veto`（不建议放松），你能看到每一条的出处。

换期限或换成本，噪声地板就变了，所以框架会拒绝拿错的标尺比：

```bash
python -m factor_eval calibrate --trials 100 --h 5 --cost 12   # 先量新地板
```

## 二、裁决

| 裁决 | 含义 |
|---|---|
| `PASS` | 六关全过 |
| `PASS_CONDITIONAL` | 有信息也有钱，但只在一个行情里，或大部分是已知基线 |
| `HOLD_INFO` | **有真实排序信息，扣完成本不赚钱** —— 最常见的诚实结果 |
| `HOLD_WEAK` | 赚了钱但量不到信息，通常是少数几个周期的运气 |
| `REJECT` | 既没信息也没钱 |
| `REJECT_DATA` | 在这个面板上根本量不了，后面的闸门没读 |

`HOLD_INFO` 不是"接近通过"。参考研究里最好的那个因子就停在这里：三段 IC 都为正、逐年同号，扣掉费用和资金费之后验证段仍然是亏的。

## 三、写一个因子

因子是一个函数：`Panel -> DataFrame`（日期 × 标的）。

```python
def my_factor(p):
    ret = np.log(p.close / p.close.shift(1))
    return -ret.rolling(10, min_periods=10).std()    # 低波
```

面板上有 `open/high/low/close/volume/quote_volume/funding/mask`。三条规则：

1. **第 t 行只能用第 t 日收盘时已知的数据。** 执行延迟由引擎加（`open[t+2]` 入场），**不要自己再 shift 一次**——那是双重延迟，白白丢掉一天的边际。
2. **不要自己选方向。** 引擎在 dev 段按平均 RankIC 冻结符号，之后不翻转。你写 `x` 还是 `-x`，卡片完全一样（有测试钉住）。
3. **一张卡评一个因子。** 扫参数是搜索，届时这些门槛不再成立——参考研究里 101 个公式的扫描结果就是例子。

`factor_eval/examples/example_factors.py` 里有五个可以直接抄的例子，其中 `look_ahead_trap` 是**故意漏未来**的：它把引擎真正交易的那个窗口当信号，IC 会到 ~0.9。它是这套框架给自己留的烟雾报警器（`tests/eval/test_factor_eval.py::test_leak_is_detected` 钉住）；顺带能看到 `close.shift(-1)` 那种"只偷看一天"的泄漏**点不亮**它——两天的入场延迟已经走过去了。

## 四、标的池：成交量最大的十个

跟随仓库的面板是 **30 日成交额最大的十个 USDⓈ-M 永续**（排序窗口 2026-08-11—09-09），日频 UTC，2021-01-01—2026-09-09：

BTC · ETH · SOL · ZEC · XRP · HYPE · DOGE · BNB · TRUMP · ENA

面板里有三样东西是普通 OHLCV 表没有的，也是这套评测能诚实的原因：

- `funding`：**实际逐次结算**的资金费，不是假设的 8 小时费率；
- `mask`：每一天谁有资格被排序和持有（已上市、满 60 根、在合约生命周期内）。在掩码之外排序是加密截面最常见的泄漏方式；
- `meta`：排序窗口、选择偏差说明和快照来源，卡片会把它印出来。

**选择偏差要说在前面**：这十个名字是按快照日的成交量选的，然后回头看历史——这是事后选择。参考研究里另做了逐月点位的前十作对照。框架默认用固定十个，是因为"用户能重跑"要求可复现，不是因为它偏差更小。

只有约 7.7 个名字在任一天同时合格，所以截面**很窄**：分箱数会跟着宽度自适应（每箱至少 2 个名字，否则结构闸门直接记 N/A 而不是给一个看起来像结论的数）。换自己的标的池：`python eval/factor_eval/build_bundle.py --pool historical --out my_bundle.parquet`。

## 五、目录

| 路径 | 是什么 |
|---|---|
| `factor_eval/` | **能力本体**。`evaluate()` + 六道闸门 + CLI |
| `factor_eval/engine.py` | 共享度量核心：时间合同、冻结方向、IC、完整持仓现金流、资金费 |
| `factor_eval/matrix.py` | 闸门、门槛、门槛出处、裁决 |
| `factor_eval/data/` | 随仓库的前十面板（0.9 MB）+ 元数据 |
| `factor_eval/calibration/` | 随机信号的噪声地板 |
| `alpha101_crypto/` | **参考研究**：101 个公式的实测，矩阵的门槛由它标定。[报告](alpha101_crypto/REPORT.md) |

`alpha101_crypto/` 是这套矩阵的**范本与标定物**：Kakushadze (2015) 的 101 条公式在同一个池子上全跑一遍，主口径 87/101 通过数据条件，其中只有 2 条在共同可核算周期三段净收益为正，**0 条**三段 IC 都高于随机信号 p95。这个结果就是为什么矩阵长这样——它见过一整个已发表因子库在这个市场上的真实通过率。

这个部分刻意**不接** `cyqnt.input/v1 → blocks → cyqnt.signal/v2`：不发信号、不注册策略、不被 `cyqnt_trd` 引用。面板进，裁决出。依赖只有 `pandas` / `numpy` / `matplotlib` / `requests` / `pyarrow`。

```bash
python -m pytest tests/eval -q        # 48 passed（33 参考研究 + 15 框架契约）
```

## 六、这套矩阵**不能**告诉你什么

- **不做搜索校正。** 一张卡是一个因子一个方向。你扫了 50 个变体再拿最好的来评，这些 p95 就不再是你的噪声地板了。
- **没有点差和深度。** 成本是 6.5 bp 单边的假设 + 实际资金费，没有按规模模拟冲击，所以**容量问题它答不了**。
- **`oot` 不是封存样本。** 它是文件里的一个分段名；这些日期在此前的研究里已经被看过。
- **横截面只有约 7.7 个名字。** 所有统计功效问题都从这里来：单期 IC 方差大、分箱只能切 3 档、条件切片基本做不了。把池子放宽到 top-50 对结论的帮助，比再挖 100 个公式大得多。

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
