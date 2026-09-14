# 因子评分卡 / factor scorecard — `reversal_5d`

**裁决 / verdict: `HOLD_INCOMPLETE`** — required evidence is missing or an evaluation gate was skipped — blocking or missing evidence: G1_information, G3_cost, G5_incremental

| 项 | 内容 |
|---|---|
| 标的池 / panel | 10 symbols, 2078 daily bars 2021-01-01 → 2026-09-09, 7.7 eligible names/day |
| 主期限 / primary horizon | 3 天，非重叠周期 |
| 成本 / cost | 6.5 bp 单边 + 实际逐次资金费 |
| 执行 / execution | open[t+entry_lag] to open[t+entry_lag+h] |
| 方向 / direction | 在 dev 段冻结，后两段不翻转 |
| 标的池限制 | current registry incomplete for fully vanished historical symbols; current cohort is retrospective selection |
| 样本限制 | oot is a time split, not evidence of a sealed holdout |
| 资金费估值 | provided settlement reference; may include historical mark-open proxies |
| 组合策略 / strategy combination | NOT_EVALUATED；G5 只检查公开基线重复性 |
| 已查看候选数 | 1；多候选结果不能按单候选标定通过 |

## 分段核心指标 / split metrics

净收益采用完整周期口径；partial net 仅作缺数诊断，不能使成本闸门通过。

| h / split | RankIC | HAC t | IC日期数 | 完整/计划周期 | 毛bp | 净bp | partial净bp | 换手 |
|---|---|---|---|---|---|---|---|---|
| 1 / dev | 0.004768 | 0.3227 | 819 | 818/819 | -8.01 | NA | -21.92 | 2 |
| 1 / val | -0.01317 | -0.7233 | 454 | 454/454 | -14.78 | -27.56 | -27.56 | 2.002 |
| 1 / oot | 0.01027 | 0.4581 | 340 | 339/340 | -2.565 | NA | -15.64 | 2 |
| 3 / dev | 0.01004 | 0.4751 | 817 | 271/272 | 20.68 | NA | 6.613 | 2 |
| 3 / val | 0.03443 | 1.277 | 452 | 151/151 | 29.11 | 16.42 | 16.42 | 2.007 |
| 3 / oot | 0.001865 | 0.05674 | 338 | 112/113 | 5.165 | NA | -8.322 | 2.003 |
| 5 / dev | 0.01103 | 0.4845 | 815 | 162/163 | 38.69 | NA | 24.25 | 2 |
| 5 / val | 0.03913 | 1.243 | 450 | 90/90 | 50.55 | 37.46 | 37.46 | 2.017 |
| 5 / oot | -0.00855 | -0.2249 | 336 | 66/67 | -6.49 | NA | -20.2 | 2.003 |

## 全面评估矩阵 / full evaluation matrix

| 维度 | 状态 | 证据与边界 |
|---|---|---|
| data_integrity | PASS | finite values, coverage, sample sizes, daily time contract |
| information | FAIL | signed RankIC, HAC and calibrated noise reference |
| structure | WARN | frozen direction, bins, primary-sign horizon comparison |
| net_economics | N/A | complete held-position cash flows and cost sensitivity |
| robustness | WARN | split/year consistency plus all predeclared slices |
| baseline_novelty | FAIL | public baseline correlation and unsaturated residual projection |
| four_target_comparison | MEASURED | same entry/exit and common cells; individual coverage and unavailable reasons retained |
| sampling_uncertainty | MEASURED | moving time blocks keep all assets together; pointwise 95% intervals, not simultaneous confidence bands |
| horizon_and_delay | MEASURED | fixed primary sign and original signal; common-sample horizon and delay curves |
| tradability | NOT_EVALUATED | No decision-time full bid/ask spread supplied; daily high-low range is not a spread proxy |
| capacity | NOT_EVALUATED | requires size-dependent fills, order-book depth and market impact; no such model supplied |
| search_selection | SINGLE_CANDIDATE_DECLARED | 1 examined candidates declared; no DSR/PBO without a complete trial ledger |
| sealed_holdout | NOT_EVALUATED | split name oot does not establish that researchers had not inspected it |
| strategy_combination | NOT_EVALUATED | second-stage factor weights, portfolio risk and combination PnL are outside this matrix |

`MEASURED` 表示已计算诊断，不代表该维度证明了盈利。`NOT_EVALUATED` 表示还缺输入或验证。

完整四目标、分箱、条件切片、期限/延迟、极端案例和成本压力保存在 JSON；`--report-dir` 生成离线图表报告。

## 闸门 / gates

| 闸门 | 状态 | 问题 |
|---|---|---|
| `G0_data` | ✓ PASS | did we measure anything at all? |
| `G1_information` | ✗ FAIL | does the ranking beat a random signal on this universe? |
| `G2_structure` | ! WARN | is the relationship monotone, and does it survive a change of horizon? |
| `G3_cost` | – N/A | does anything survive fees and actual funding? |
| `G4_robustness` | ! WARN | does it hold across splits and years, or is it one regime? |
| `G5_incremental` | ✗ FAIL | is this more than a public baseline? |

### ✓ G0_data — 数据 / data health

| 检查 | 实测 | 门槛 | 判定 | 门槛来自哪里 |
|---|---|---|---|---|
| coverage_dev | 1 | `>= 0.5` | ✓ PASS | convention: below half the eligible cells the cross-section is not the universe |
| ic_days_dev | 817 | `>= 100` | ✓ PASS | convention: fewer than 100 usable dates makes every later number noise |
| activation_dev | 1 | `>= 0.5` | ✓ PASS | convention: a factor flat most cycles cannot be held to its own thesis<br>_272/272 cycles held a position_ |
| direction_frozen | 1 | `>= 1` | ✓ PASS | required: the sign is frozen on dev and never re-chosen later |
| coverage_val | 1 | `>= 0.5` | ✓ PASS | convention: require coverage in every evaluation split |
| ic_days_val | 452 | `>= 100` | ✓ PASS | convention: require usable dates in every evaluation split |
| coverage_oot | 1 | `>= 0.5` | ✓ PASS | convention: require coverage in every evaluation split |
| ic_days_oot | 338 | `>= 100` | ✓ PASS | convention: require usable dates in every evaluation split |
| causality_prefix | 1 | `>= 1` | ✓ PASS | required: sampled past signals agree after truncating future input rows<br>_9 cutoffs; sampled check only, upstream availability still requires audit_ |

### ✗ G1_information — 信息 / information

| 检查 | 实测 | 门槛 | 判定 | 门槛来自哪里 |
|---|---|---|---|---|
| ic_dev | 0.01004 | `> 0.03963` | ✗ FAIL | null p95: 100 AR(1) random signals through this same pipeline |
| ic_val | 0.03443 | `> 0.04321` | ✗ FAIL | null p95: 100 AR(1) random signals through this same pipeline |
| ic_oot | 0.001865 | `> 0.04907` | ! WARN | null p95: 100 AR(1) random signals through this same pipeline |
| ic_t_hac_dev | 0.4751 | `> 2.077` | ✗ FAIL | null p95 (HAC t of the same random signals) |
| ic_win_dev | 0.4908 | `> 0.5` | ! WARN | convention: a signed IC should be positive more often than not |

### ! G2_structure — 结构 / structure

| 检查 | 实测 | 门槛 | 判定 | 门槛来自哪里 |
|---|---|---|---|---|
| bin_monotonicity | 1 | `>= 0.6` | ✓ PASS | convention: Spearman(bin index, per-date-then-averaged forward return); bin count follows the width of the cross-section<br>_val: 3 bins; 452 common dates; >=2 names/bin; >=30 dates required_ |
| topbottom_spread_bp | 43.13 | `> 0` | ✓ PASS | required: the top bin must out-return the bottom bin in the frozen direction<br>_a positive IC with a negative spread is not a bug: IC ranks, the spread averages, so a factor can be right more often while the money sits in the fat tail it is shorting_ |
| horizon_sign_agreement | 0 | `>= 1` | ! WARN | convention: dev IC keeps one sign across h=[1, 3, 5]<br>_the effect flips sign with holding period_ |

### – G3_cost — 成本 / cost (hard veto)

| 检查 | 实测 | 门槛 | 判定 | 门槛来自哪里 |
|---|---|---|---|---|
| accounting_complete_val | 1 | `>= 1` | ✓ PASS | required: every scheduled cycle has known signal, prices and funding<br>_151/151 cycles complete_ |
| net_bp_val | 16.42 | `> 0` | ✓ PASS | hard veto: net of trading cost and per-settlement funding, after the frozen direction |
| accounting_complete_oot | 0.9912 | `>= 1` | – N/A | required: every scheduled cycle has known signal, prices and funding<br>_112/113 cycles complete_ |
| net_bp_oot | NA | `> 0` | – N/A | hard veto: net of trading cost and per-settlement funding, after the frozen direction<br>_partial-cycle averages are diagnostics only_ |
| breakeven_cost_bps_val | 14.68 | `> 13` | ✓ PASS | cost-derived: needs 2× headroom over the assumed 6.5 bp one-way<br>_breakeven = (gross − funding) / turnover_ |
| net_bp_val_vs_null | 16.42 | `> 26.56` | ! WARN | null p95: a random signal's net over the same cycles |
| turnover_per_cycle | 2 | — | – N/A | reported, not gated: 2.0 means a full close-and-reopen every cycle<br>_cost drag ≈ 13.0 bp/cycle at 6.5 bp one-way_ |

### ! G4_robustness — 稳健 / robustness

| 检查 | 实测 | 门槛 | 判定 | 门槛来自哪里 |
|---|---|---|---|---|
| split_sign_consistency | 3 | `>= 3` | ✓ PASS | required: the frozen direction keeps its sign in every split<br>_3/3 splits positive_ |
| yearly_sign_consistency | 3 | `>= 4` | ! WARN | convention: at most one calendar year may disagree<br>_3/5 years positive_ |
| sharpe_net_val | 0.7134 | `> 1.184` | ! WARN | null p95 (annualised on the same cycle count) |

### ✗ G5_incremental — 增量 / incremental

| 检查 | 实测 | 门槛 | 判定 | 门槛来自哪里 |
|---|---|---|---|---|
| max_abs_rank_corr_vs_baselines | 1 | `< 0.7` | ✗ FAIL | convention: |ρ| ≥ 0.7 against any single baseline is the same bet<br>_closest: B_rev5_ |
| residual_ic_retention | NA | `>= 0.5` | – N/A | convention: after projecting out all five baselines, keep at least half the IC<br>_not computable_ |
| residual_net_bp_val | NA | `> 0` | – N/A | required: the part that is not a baseline must still pay<br>_not computable_ |

## 分箱 / bins（val，冻结方向，入场对齐）

| 箱 | 资产·日 | 平均前瞻收益 bp |
|---|---|---|
| 1 | 1095 | 20.1 |
| 2 | 1356 | 36.66 |
| 3 | 1415 | 63.23 |

val: 3 bins; 452 common dates; >=2 names/bin; >=30 dates required

首尾 spread 43.13 bp，Spearman(箱号, 收益) 1

## 分年 IC / yearly IC

| 年 | IC |
|---|---|
| 2022 | 0.03195 |
| 2023 | -0.003914 |
| 2024 | 0.02338 |
| 2025 | 0.02398 |
| 2026 | -0.0001478 |

## 与公开基线的关系 / vs public baselines（dev）

| 基线 | 横截面 rank 相关 |
|---|---|
| B_size | -0.1028 |
| B_lowvol10 | 0.0198 |
| B_rev5 | 1 |
| B_mom20 | -0.3857 |
| B_funding7 | -0.07 |

投影掉全部五个基线后，dev IC 保留 NA；残差在 val 的净收益 NA bp

## 怎么读这张卡 / how to read this

- 必需检查失败或缺失都阻止完整通过；后面的指标不能抵消它。
- `G3_cost` 是硬否决：信息再强，收不上来就不是可交易因子。
- *null p95* 是指定 AR(1) 随机信号的参考分布，不是所有候选的普适显著性保证。
- 标了 *convention* 的门槛可以争论；标了 *required* / *hard veto* 的不建议放松。
- 前缀一致性是抽查，不能证明上游数据、外部状态或闭包中的数据都按时可得。
- 这张卡评的是**一个**因子。扫参数属于搜索，需完整试验账本及新样本验证。
- PASS 是当前数据和假设下的研究筛选通过；封存样本、容量、组合收益仍需另行验证。

## 复现标识 / provenance

- 面板 SHA256：`fb5088eda5afdc2528462d6ebc7d847a026888498be4c02494a8f15e5107ec87`
- 引擎：`factor-eval/daily-closed-cycles-v3` / `9944331f56d1709b0fcb530cc2c082a00848c40908c205adf7ea6f7b3f091500`
- JSON 同时保存全部分段指标、切分边界、执行配置及标定契约；非有限量编码为 null。