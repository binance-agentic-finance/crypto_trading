# 因子评分卡 / factor scorecard — `low_volatility`

**裁决 / verdict: `HOLD_INFO`** — real ranking information, no money after cost — the most common honest outcome — first failure: G2_structure, G3_cost, G5_incremental

| 项 | 内容 |
|---|---|
| 标的池 / panel | 10 symbols, 2078 daily bars 2021-01-01 → 2026-09-09, 7.7 eligible names/day |
| 主期限 / primary horizon | 3 天，非重叠周期 |
| 成本 / cost | 6.5 bp 单边 + 实际逐次资金费 |
| 执行 / execution | open[t+entry_lag] to open[t+entry_lag+h] |
| 方向 / direction | 在 dev 段冻结，后两段不翻转 |

## 闸门 / gates

| 闸门 | 状态 | 问题 |
|---|---|---|
| `G0_data` | ✓ PASS | did we measure anything at all? |
| `G1_information` | ✓ PASS | does the ranking beat a random signal on this universe? |
| `G2_structure` | ✗ FAIL | is the relationship monotone, and does it survive a change of horizon? |
| `G3_cost` | ✗ FAIL | does anything survive fees and actual funding? |
| `G4_robustness` | ! WARN | does it hold across splits and years, or is it one regime? |
| `G5_incremental` | ✗ FAIL | is this more than a public baseline? |

### ✓ G0_data — 数据 / data health

| 检查 | 实测 | 门槛 | 判定 | 门槛来自哪里 |
|---|---|---|---|---|
| coverage_dev | 1 | `>= 0.5` | ✓ PASS | convention: below half the eligible cells the cross-section is not the universe |
| ic_days_dev | 817 | `>= 100` | ✓ PASS | convention: fewer than 100 usable dates makes every later number noise |
| activation_dev | 1 | `>= 0.5` | ✓ PASS | convention: a factor flat most cycles cannot be held to its own thesis<br>_272/272 cycles held a position_ |
| direction_frozen | 1 | `>= 1` | ✓ PASS | required: the sign is frozen on dev and never re-chosen later |

### ✓ G1_information — 信息 / information

| 检查 | 实测 | 门槛 | 判定 | 门槛来自哪里 |
|---|---|---|---|---|
| ic_dev | 0.08137 | `> 0.03963` | ✓ PASS | null p95: 100 AR(1) random signals through this same pipeline |
| ic_val | 0.05205 | `> 0.04321` | ✓ PASS | null p95: 100 AR(1) random signals through this same pipeline |
| ic_oot | 0.05917 | `> 0.04924` | ✓ PASS | null p95: 100 AR(1) random signals through this same pipeline |
| ic_t_hac_dev | 3.465 | `> 2.077` | ✓ PASS | null p95 (HAC t of the same random signals) |
| ic_win_dev | 0.5643 | `> 0.5` | ✓ PASS | convention: a signed IC should be positive more often than not |

### ✗ G2_structure — 结构 / structure

| 检查 | 实测 | 门槛 | 判定 | 门槛来自哪里 |
|---|---|---|---|---|
| bin_monotonicity | -0.5 | `>= 0.6` | ! WARN | convention: Spearman(bin index, per-date-then-averaged forward return); bin count follows the width of the cross-section<br>_3 bins, ~2.6 names per bin per date, 15862 asset-days_ |
| topbottom_spread_bp | -22.46 | `> 0` | ✗ FAIL | required: the top bin must out-return the bottom bin in the frozen direction<br>_a positive IC with a negative spread is not a bug: IC ranks, the spread averages, so a factor can be right more often while the money sits in the fat tail it is shorting_ |
| horizon_sign_agreement | 1 | `>= 1` | ✓ PASS | convention: dev IC keeps one sign across h=[1, 3, 5] |

### ✗ G3_cost — 成本 / cost (hard veto)

| 检查 | 实测 | 门槛 | 判定 | 门槛来自哪里 |
|---|---|---|---|---|
| net_bp_val | -49.47 | `> 0` | ✗ FAIL | hard veto: net of trading cost and per-settlement funding, after the frozen direction |
| net_bp_oot | -84.14 | `> 0` | ✗ FAIL | hard veto: net of trading cost and per-settlement funding, after the frozen direction<br>_available-cycle diagnostic; some cycles have unknown funding_ |
| breakeven_cost_bps_val | -18.14 | `> 13` | ✗ FAIL | cost-derived: needs 2× headroom over the assumed 6.5 bp one-way<br>_breakeven = (gross − funding) / turnover_ |
| net_bp_val_vs_null | -49.47 | `> 26.56` | ! WARN | null p95: a random signal's net over the same cycles |
| turnover_per_cycle | 2.002 | — | – N/A | reported, not gated: 2.0 means a full close-and-reopen every cycle<br>_cost drag ≈ 13.0 bp/cycle at 6.5 bp one-way_ |

### ! G4_robustness — 稳健 / robustness

| 检查 | 实测 | 门槛 | 判定 | 门槛来自哪里 |
|---|---|---|---|---|
| split_sign_consistency | 3 | `>= 3` | ✓ PASS | required: the frozen direction keeps its sign in every split<br>_3/3 splits positive_ |
| yearly_sign_consistency | 5 | `>= 4` | ✓ PASS | convention: at most one calendar year may disagree<br>_5/5 years positive_ |
| sharpe_net_val | -1.834 | `> 1.184` | ! WARN | null p95 (annualised on the same cycle count) |

### ✗ G5_incremental — 增量 / incremental

| 检查 | 实测 | 门槛 | 判定 | 门槛来自哪里 |
|---|---|---|---|---|
| max_abs_rank_corr_vs_baselines | 1 | `< 0.7` | ✗ FAIL | convention: |ρ| ≥ 0.7 against any single baseline is the same bet<br>_closest: B_lowvol10_ |
| residual_ic_retention | 0.139 | `>= 0.5` | ✗ FAIL | convention: after projecting out all five baselines, keep at least half the IC |
| residual_net_bp_val | 2.051 | `> 0` | ✓ PASS | required: the part that is not a baseline must still pay |

## 分箱 / bins（冻结方向，入场对齐）

| 箱 | 资产·日 | 平均前瞻收益 bp |
|---|---|---|
| 1 | 4563 | 28.89 |
| 2 | 4854 | 1.831 |
| 3 | 6445 | 6.429 |

首尾 spread -22.46 bp，Spearman(箱号, 收益) -0.5

## 分年 IC / yearly IC

| 年 | IC |
|---|---|
| 2022 | 0.1306 |
| 2023 | 0.0544 |
| 2024 | 0.03008 |
| 2025 | 0.07577 |
| 2026 | 0.06539 |

## 与公开基线的关系 / vs public baselines

| 基线 | 横截面 rank 相关 |
|---|---|
| B_size | 0.2735 |
| B_lowvol10 | 1 |
| B_rev5 | 0.03195 |
| B_mom20 | -0.07071 |
| B_funding7 | 0.07387 |

投影掉全部五个基线后，dev IC 保留 0.139；残差在 val 的净收益 2.051 bp

## 怎么读这张卡 / how to read this

- **第一个 ✗ 就是结论**：后面的闸门只是补充说明，不能用来抵消它。
- `G3_cost` 是硬否决：信息再强，收不上来就不是可交易因子。
- 标了 *null p95* 的门槛，是随机信号走同一条管线得到的分布，不是拍的。
- 标了 *convention* 的门槛可以争论；标了 *required* / *hard veto* 的不建议放松。
- 这张卡评的是**一个**因子、**一个**冻结方向。扫参数属于搜索，届时这些门槛不再成立。