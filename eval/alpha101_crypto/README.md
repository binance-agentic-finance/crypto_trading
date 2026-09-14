# Alpha101 crypto：评测矩阵的参考研究与标定物

结论见[评估报告](REPORT.md)。

这个目录有两个身份。它是一次**实测**：Kakushadze (2015) 的 101 条公式在成交额前十的永续上全跑一遍。它同时是 [`eval/factor_eval/`](../factor_eval/) 这套评测矩阵的**范本与标定物** —— 矩阵的噪声地板（`factor_eval/calibration/null_top10_h3.json`）就是本次的空模型结果，矩阵的六道闸门也是照着这 101 条公式暴露出来的失败方式排的：数据退化、IC 不超噪声、有信息但收不上来、只在一个行情里、以及"其实就是低波换皮"。

度量核心 `engine.py` 现在住在 `factor_eval/`，本目录的脚本引用它；两边用的是同一段代码，不是两份实现。

## 本次范围

- 数据快照：2021-01-01 至 2026-09-09 UTC，各合约按实际存续期使用；2021 年供预热。
- 当前 30 日成交额前十，与 54 次历史月度前十（80 币）两种池。
- 原始输入和归一化输入各计算 101 个公式；82 条价量输入、19 条市值或行业替代。
- 主期限 3 日；额外等待 1 日后入场；1/5 日及等待 0/2/5 日为探索。
- 完整持仓简单收益、每周期全平全开、单边 6.5 bp、逐次资金费；同时保存不计资金费与缺失标记。
- 数据不足、未知资金费、首尾生命周期、单币缺未来价格均明确传播，不将未知净收益填零。

## 哪些文件在库里，哪些不在

**在库里的是结论**：代码、配置、`metrics_*` / `summary_*` / `null_*` 小表、四组 h=3 的 CSV、图、`data/*.json` 清单、`reproducibility.json` 校验值和报告。

**不在库里的是输入和大型中间结果**（见仓库根 `.gitignore`）：653 合约日线与 41 万条资金费事件约 168 MB，`periods_*` / `ic_*` / `factors_*` 约 180 MB。它们由下面的命令重新产生，但**重新下载属于新的快照**：截止时间会变，不保证与本版逐位元相同。

因此只有 `write_report.py` 和 `tests/eval` 可以在干净 checkout 上直接跑；其余命令需要先取数。

## 复现

从仓库根目录运行；依赖版本见 `requirements-snapshot.txt` 与 `reproducibility.json`。

```bash
# 干净 checkout 上可直接跑的两件事
python -m pytest tests/eval -q                       # 48 passed
python eval/alpha101_crypto/write_report.py          # 用已入库的结果表重新生成 REPORT.md

# 需要先取数（约 168 MB，打公开 Binance 接口）
python eval/alpha101_crypto/acquire_data.py

# 四组评估 + 汇总 + 诊断
python eval/alpha101_crypto/run_pipeline.py --pool current    --mode normalized --stage all --null-count 100
python eval/alpha101_crypto/run_pipeline.py --pool current    --mode paper      --stage all
python eval/alpha101_crypto/run_pipeline.py --pool historical --mode normalized --stage all --null-count 100
python eval/alpha101_crypto/run_pipeline.py --pool historical --mode paper      --stage all
python eval/alpha101_crypto/summarize_results.py
python eval/alpha101_crypto/diagnostics.py --tag current_normalized
python eval/alpha101_crypto/incremental.py
python eval/alpha101_crypto/unit_sensitivity.py
python eval/alpha101_crypto/record_reproducibility.py
```

`acquire_data.py` 的实时收集入口按服务器当日确定截止；已缓存日线不会自动延展。若建立新日期快照，应使用单独目录，并核对所有文件覆盖及清单，避免新截止时间混用旧缓存。

## 主要文件

| 文件 | 职责 |
|---|---|
| `acquire_data.py` | 日线、资金费与代理价格获取；分页、来源、校验值和缺口记录 |
| `factors.py` | 101 条公式；横截面算子限定当时交易池；完整窗口与缺失传播 |
| `../factor_eval/engine.py` | 共享度量核心：时间合同、开发段方向、IC、固定数量持仓、完整平仓、交易成本和资金费现金流 |
| `run_pipeline.py` | 上线/终止边界、两类池和输入、五个基线、四组评估与空模型 |
| `summarize_results.py` | 共同可核算周期、成本情景、四组全表与总览图 |
| `diagnostics.py` | 分布、四目标分箱、过去收益切片、期限/延迟、年度及极端案例 |
| `incremental.py` | 同资产/日期的基线排名投影；开发段拟合、固定系数；保留残差方向选择记录 |
| `unit_sensitivity.py` | BTC/mBTC 等价计量变化对四个候选的事后诊断 |
| `write_report.py` | 依据结果表生成 `REPORT.md`；叙述对应本次冻结快照 |
| `record_reproducibility.py` | 最终代码、结果、清单及依赖版本校验记录 |

## 收益字段不能混用

`metrics_*.parquet` 中 `net_bp` 是严格的完整阶段均值；只要任何成熟、有持仓周期存在未知 PnL 就为 NA。`net_available_bp` 是该因子自己的已知周期诊断。

`summary_*.parquet` / `summary_*_h3.csv` 保留以上两类，再增加 `common_*` 字段：在每组、每期限、每阶段，删除全部 101 因子和 5 个基线中任一未知净值的周期，所有策略按同一组日期比较。空仓零值保留；剔除是事后的可核算筛选，不能当作实盘规避规则。

`common_net_bp = common_gross_bp - common_trading_cost_bp - common_funding_bp`。资金费正值为支出、负值为收入。交易成本按入场与退出漂移后的绝对名义金额之和计算；盈亏平衡单边成本在相同样本计算。

`common_sharpe_net` 按 365/h 年化，属于可用周期描述性估计；`common_net_t_hac` 使用保留缺口的网格、4 个周期 Bartlett 滞后。没有按多重搜索校正。

## 数据与证据边界

- `data/daily_manifest.json`：653 份日线的下载记录；真实交易资格由另存的上线/终止时间控制。
- `data/universes.json`：最新生命周期规则的当前池和历史入池并集。资金费保存 81 币，其中 FTT 已从最终 80 币所需并集中剔除，仅供审计。
- `data/funding_manifest.json`：416,046 条事件，约 30.94% 标记价格为有来源标记的同整点 K 线开盘价代理；38 个间隔疑点继续未知。
- `references/data_availability.md`：逐条可用性证据。全部输入取自 Binance 公开历史接口；没有把任何当日快照当作历史市值。
- `references/semantics.md`、`sources.json`：算子与来源约定。`paper` 口径不代表唯一复现作者私有实现；论文本身见 [arXiv:1601.00991](https://arxiv.org/abs/1601.00991)，PDF 不随仓库分发。
- 缺历史点差/深度，规范中的 IC 对可交易性 N 图为 N/A；没有以成交额代替。

## 实验记录的时间顺序

`experiment_config.json` 在新性能汇总前记录初始主口径，内含初始代码哈希。它不证明历史样本从未被查看：上一版已探索过相同历史。

本次随后根据实测数据质量修正资金费价格代理、事件缺口传播、缓存分页完整性、上线/终止后的占位行情；这些修正后四组重新计算。候选门槛使用开发段信息。共同周期比较、基线投影和计量单位检查是看到主实验后的诊断，未声称事前预登记。最终哈希单独写入 `reproducibility.json`，不覆盖初始配置。

测试覆盖算子尺度、缺失条件、排名池、时间边界、真实简单收益、平仓费用、资金费符号与未知传播、分页资格和合约生命周期。报告和所有 101 个结果均保留失败，不沿用上一版的小时结果、PBO、DSR 或通过标签。

### `reproducibility.json` 与当前树的差异（迁入本仓库时产生）

哈希本身来自迁入前的原始运行，**故意保留不覆盖**：它同时记录了未入库的大文件，重跑只会把这部分记录删掉。迁入时改了路径前缀（`code/alpha101_crypto_v2/` → `eval/alpha101_crypto/`、`tests/` → `tests/eval/`、`engine.py` → `../factor_eval/engine.py`），逐项核对结果为：

| 组 | 一致 | 不一致 | 本仓库中不存在 |
|---|---|---|---|
| `code_sha256` | 3 | 10 | 0 |
| `result_sha256` | 33 | 1：`method_and_projection.json`（把记录里的本机绝对路径改成仓库相对路径） | 18（未入库的大文件） |
| `input_manifest_sha256` | 7 | 0 | 1：`exchange_info.json`（未入库） |

10 个不一致的代码文件，改动各是什么：`write_report.py` 改输出路径与去标识段落，`record_reproducibility.py` 改扫描范围，五个脚本改成引用共享的 `factor_eval.engine`，三个 `tests/eval/test_*.py` 改模块定位行。`factors.py`、`acquire_data.py`、`engine.py` 三个承载数值的文件**逐位元未变** —— 这正是这张表要证明的事：搬家没有动到算法。

核对命令（在仓库根运行）：

```bash
python - <<'PY'
import json, hashlib
from pathlib import Path
d = json.loads(Path("eval/alpha101_crypto/reproducibility.json").read_text())
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
for group, base in (("code_sha256", "."), ("result_sha256", "eval/alpha101_crypto"),
                    ("input_manifest_sha256", "eval/alpha101_crypto")):
    ok = [k for k, v in d[group].items() if (Path(base)/k).exists() and sha(Path(base)/k) == v]
    bad = [k for k, v in d[group].items() if (Path(base)/k).exists() and sha(Path(base)/k) != v]
    absent = [k for k in d[group] if not (Path(base)/k).exists()]
    print(f"{group}: match {len(ok)} | stale {bad} | absent {len(absent)}")
PY
```

`report_sha256` 同样对应迁入前的报告：迁入时重写了 §2.1 的去标识段落与全部相对链接，随后用 `write_report.py` 重新生成 `REPORT.md`。在本仓库内重跑 `record_reproducibility.py` 会写入与当前树一致的新值，但会丢掉上述 18 个未入库文件的记录。
