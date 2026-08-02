# Gold ceiling, recomputed against the updated capability table

Population: `candidates.jsonl`, 26,161 mined rows (tier A/B/C non-fragments), 16,472
split groups. Capability table: `tools/nl2yaml/capability.py` at commit `3e247b0`
plus the indicator flip described below, `assert_table_is_current()` = ok. Scoring
machinery: `measure.plan_row`, unmodified. No LLM in the loop.

Counts are **split groups** unless stated. 43% of the corpus collapses into a group
with another row, so a row count measures memorisation capacity and a group count
measures example count.

> This file differs from `funnel_report.md` on purpose and the denominators are not
> comparable. That report re-mines the whole 51,595-row corpus including tier D and
> continuation fragments; this one reads `candidates.jsonl`, which is tier A/B/C
> non-fragments only. Selection has 885 groups here and 1,343 there, and the
> difference is tier D — rows carrying no mined condition at all, which cannot yield
> a spec-shaped gold either way.

---

## 0. Read this before using any number below

**Every count here is an UPPER BOUND. The true yield is strictly lower.** Five
reasons, all pushing the same way:

1. **`capability.plan_conversion` needs conditions in its own vocabulary, and
   producing those is A1's job (an LLM, not yet run).** This pass substitutes a
   hand-written map from the miner's regex `(family, subject)` pairs onto capability
   subjects. `candidates.jsonl` does carry a `conditions[]` array, but its subjects
   are the miner's 49 regex subjects (`screen`, `unspecified`, `interval`), not the
   table's 30 — so the map is the bound, not the data.
2. **Miner recall is a floor.** The regexes find a subset of each request's
   conditions. A group scored "all expressible" can still contain an unmined
   `not_expressible` one.
3. **The map is coarse in the generous direction.** `volume` (a coin count) is scored
   against `quoteVolume` (quote currency); `exclude` is scored as `symbol_blacklist`
   without knowing what is excluded.
4. **`unknown` is forgiven in the headline.** A `(subject, scope, operator)` the
   table does not rule on is treated as expressible. The `strict` column removes
   them; §5 lists which keys they are.
5. **Scope is guessed for ambiguous rows.** `classify_request` returns ambiguous for
   most of the corpus; those rows are scored in both frames and the better result
   kept — upward again.

One caveat pushes the other way and is quantified in §4: the gap ranking is blind to
11 of the table's 16 gap ids, so refusal counts are floors.

---

## 1. What changed in the capability table

Eleven rows, all probed with `yaml_pipeline.spec.validate_spec` to `errors == []`
before being written.

| row | was | now |
|---|---|---|
| `technical_indicator / cross_section / compare` | proxy_only | **expressible** |
| `technical_indicator / cross_section / rank` | *(absent → unknown)* | **expressible** |
| `technical_indicator / per_candidate_series / *` | not_expressible | **expressible** |
| `technical_indicator / * / *` | *(absent → unknown)* | not_expressible, named gap |
| `technical_indicator / per_symbol_series / compare` | expressible | expressible, **+3 block refs** |
| `multi_timeframe / cross_section / resonance` | not_expressible | **expressible** |
| `multi_timeframe / per_candidate_series / resonance` | *(absent → unknown)* | **expressible** |
| `multi_timeframe / * / *` | *(absent → unknown)* | not_expressible, named gap |
| `historical_range / cross_section / window` | not_expressible | **expressible** |
| `historical_range / per_candidate_series / window` | *(absent → unknown)* | **expressible** |
| `historical_range / * / *` | *(absent → unknown)* | not_expressible, named gap |
| `spread_liquidity / * / *` | *(absent → unknown)* | not_expressible, named gap |

`spread_liquidity` at `cross_section` was **already** expressible for both `compare`
and `rank` before this pass — the liquidity work landed earlier the same day. Only
its wildcard catch-all was missing.

Four `*/*` catch-alls were added because `lookup` returns `unknown` for an unlisted
key, and `unknown` shelves the case for a human instead of naming a gap.

---

## 2. Q1 — tier A groups where every condition is expressible

| dialect | tier A groups | all-expressible **before** | all-expressible **after** | delta |
|---|---|---|---|---|
| selection | 182 | 47 | **77** | **+30** |
| trade | 2,024 | 1,535 | **1,535** | 0 |
| both | 1,074 | 0 | **0** | 0 |
| unclear | 424 | 319 | **319** | 0 |

All tiers together:

| dialect | groups | ub **before** | ub **after** | delta | strict | + user stated every number |
|---|---|---|---|---|---|---|
| selection | 885 | 423 | **525** | **+102** | 404 | 405 |
| trade | 7,615 | 6,948 | **6,948** | 0 | 5,765 | 3,644 |
| both | 1,421 | 0 | **0** | 0 | 0 | 0 |
| unclear | 6,589 | 5,829 | **5,829** | 0 | 4,615 | 4,167 |

**The entire gain is in the selection dialect, and that is the expected shape of
this change.** `augment_with_indicator` is a `universe.*` block; it can only help a
request scored on the cross-section. Trade-shaped rows are scored on
`per_symbol_series`, where indicators were already expressible — the flip adds
nothing there and correctly does not pretend to.

`both` stays at zero by construction: one spec is either `selection:` or `signals:`,
so those 1,421 groups are refusal gold regardless of any block.

---

## 3. Q2 — against last night's numbers

Last night's table used a different estimator: **count tier A groups, then drop any
group whose `families` list contains `indicator`/`timeframe`, then also drop `risk`.**
It never consulted the capability table. Reproduced exactly, it is unchanged by this
work, because it cannot see a capability table at all:

| dialect | tier A groups | − indicator/timeframe | − risk |
|---|---|---|---|
| selection | 182 | 93 | 41 |
| trade | 2,024 | 576 | 58 |
| both | 1,074 | 344 | 76 |
| unclear | 424 | 96 | 27 |

So **`selection 41 → 77` is the answer, but the two numbers come from different
instruments** and the comparison needs saying carefully:

* `41` was a proxy for "no condition in this group is blocked", computed by deleting
  whole families. It over-deletes (an `indicator` family in a *trade* request was
  never blocked) and under-deletes (it cannot see `market_cap`, which is blocked).
* Run the capability table over the same tier A selection groups and the
  pre-flip answer is **47**, not 41. That is the honest before-number.
* After the flip: **77**.

| estimator | before | after |
|---|---|---|
| family heuristic (last night) | 41 | 41 *(cannot move — table-blind)* |
| capability table, tier A selection | 47 | **77** |
| capability table, all-tier selection | 423 | **525** |

---

## 4. Q3 — which gaps still block, ranked by `dup_weighted_count`

`dup_weighted_count` = raw rows blocked. `unlocked if closed` = rows whose **only**
blocker is this gap — **build order should follow that column, not the first one.**

| # | gap_id | dup_weighted_count | groups | unlocked if closed (rows / groups) | main shapes |
|---|---|---|---|---|---|
| 1 | `GAP-COMPOUND-SELECT-THEN-TRADE` | 2,302 | 1,421 | 1,629 / 1,053 | both 2,302 |
| 2 | `GAP-PER-SYMBOL-INDICATOR` | 1,825 | 1,154 | 1,220 / 807 | trade 805, both 550, unclear 416, selection 54 |
| 3 | `GAP-MARKET-CAP` | 669 | 474 | 380 / 328 | both 219, unclear 215, selection 126, trade 109 |
| 4 | `GAP-ENTRY-EXIT-PER-CANDIDATE` | 215 | 153 | 200 / 149 | selection 215 |

Movement versus the pre-flip run of the same script:

| gap_id | rows before | rows after | note |
|---|---|---|---|
| `GAP-SPREAD-DEPTH` | *(0)* | *(0)* | already closed by the liquidity work; was #3 at 1,474 in `funnel_report.md` |
| `GAP-PER-SYMBOL-INDICATOR` | 1,989 | 1,825 | −164 |
| `GAP-MARKET-CAP` | 669 | 669 | unchanged (unlocked-if-closed rose 371→380: rows it used to share a blocker with are now clean apart from it) |
| `GAP-COMPOUND-SELECT-THEN-TRADE` | 2,302 | 2,302 | unchanged |
| `GAP-ENTRY-EXIT-PER-CANDIDATE` | 215 | 215 | unchanged |

### The most important line in this section

**`GAP-PER-SYMBOL-INDICATOR`'s remaining 1,825 rows are now blocked by exactly one
source: `timeframe/interval[multi]`.** Before the flip its blocking sources were
`timeframe/interval[multi]` 1,825 plus `indicator/breakout` 93, `indicator/rsi` 88,
`indicator/sma` 87, `indicator/ema` 59, `indicator/divergence` 46. Every
indicator-named source is gone. What is left is **not an indicator problem**.

And a large part of what is left is a **measurement artifact, not a capability gap**.
`measure.py::_timeframe_conditions` pins every multi-interval request to
`per_symbol_series` scope, hard-coded. That was harmless while both `multi_timeframe`
scopes were blocked; now that `cross_section` is expressible, a *selection*-shaped
resonance request is still scored against the HTF-SMA proxy row and still counted as
refusal gold. Patched in-process to take the scope from the request's frame:

| | as scored today | scope taken from the frame | delta |
|---|---|---|---|
| selection, all-expressible groups | 525 | **559** | +34 |
| selection, tier A | 77 | **101** | +24 |
| unclear, all-expressible groups | 5,829 | **6,036** | +207 |
| `GAP-PER-SYMBOL-INDICATOR` rows | 1,825 | **805** | −1,020 |
| `GAP-ENTRY-EXIT-PER-CANDIDATE` rows | 215 | **697** | +482 |

The 805 that survive are the trade-shaped ones, and those are genuinely blocked: on
one declared symbol `data.htf` attaches HTF SMAs and nothing else.
`GAP-ENTRY-EXIT-PER-CANDIDATE` *grows* because rows previously blocked by two gaps
now surface their remaining one — which is exactly why the `unlocked if closed`
column exists.

`measure.py` is outside this task's edit scope, so the table above is a
counterfactual and the ranked table above it is what the pipeline actually produces
today. **Fixing that one hard-coded scope is the cheapest remaining move on this
list** — it is worth more selection examples than any block.

### Gaps this pass structurally cannot count

The miner has eight regex families; 11 of the table's 16 gap ids have no family
looking for them, so requests hitting them are scored as clean. **Refusal counts are
floors and expressible counts are ceilings**, for recall reasons on top of the
charity reasons in §0.

Unchanged from `funnel_report.md`, with one entry that this pass makes newly
significant:

| gap_id | status |
|---|---|
| `GAP-HISTORICAL-WINDOW` | **undetectable — and this is now load-bearing.** No miner family looks for "N-day high" or "% off its yearly high"; the `timeframe` family catches intervals, not lookback windows. So `historical_range / cross_section / window` flipping to expressible produced a measured gain of **exactly zero**, and that zero is a measurement failure, not a statement about demand. The capability is real and probed two ways. Whatever this row is worth is invisible until the miner grows a lookback-window family. |
| `GAP-ACCOUNT-OPS`, `GAP-ALERT-NOTIFY`, `GAP-LIQUIDATION-CROSS-SECTION`, `GAP-NEWS-EVENT-TEXT`, `GAP-OI-CROSS-SECTION`, `GAP-LONG-SHORT-RATIO`, `GAP-ONCHAIN-CONCENTRATION`, `GAP-VAGUE-CRITERION` | undetectable, unchanged |
| `GAP-CONTRACT-META`, `GAP-SECTOR-LABEL` | vacant in the table; nothing should reach them |
| `GAP-SPREAD-DEPTH` | live in the table (`per_symbol_series` only), and now unreachable by this pass since the cross-section flipped |

### Unruled keys — what the headline's optimism is made of

`(subject, scope, operator)` triples the table has no row for, so `lookup` returns
`unknown` and the row is forgiven in the upper bound. These are **not** block gaps;
they are questions nobody has ruled on.

| subject | scope | operator | rows | groups |
|---|---|---|---|---|
| `bar_interval` | `*` | equals | 3,797 | 2,136 |
| `unattributed_threshold` | `*` | compare | 3,645 | 1,189 |
| `portfolio_drawdown_limit` | `*` | plan | 508 | 352 |
| `price_change_24h` | `per_symbol_series` | compare | 233 | 88 |
| `backtest_win_rate` | `*` | compare | 94 | 55 |

The four `*/*` catch-alls added in §1 removed `technical_indicator`,
`multi_timeframe`, `historical_range` and `spread_liquidity` from ever reaching this
list at an unlisted scope. None of them appear above, which is the check that the
catch-alls work.

---

## 5. Q4 — is this enough to fine-tune?

Band for structured output SFT: **500–5,000**. The two dialects are separate tasks
with separate grammars and `validate_spec` refuses a spec that is both, so a mixed
training set teaches a shape that cannot exist. Counted apart, in split groups:

| dialect | groups | upper bound | + ≥1 expressible | strict | + user stated every number | refusal gold |
|---|---|---|---|---|---|---|
| **selection** | 885 | **525** | 470 | 404 | **405** | 292 |
| **trade** | 7,615 | **6,948** | 6,848 | 5,765 | **3,644** | 617 |

### Trade: yes, unchanged — the problem is selection pressure, not scarcity

6,948 distinct requests clear the bound, 5,765 survive the strict reading, and 3,644
also have every number the user stated rather than an annotator's guess. The
tightest of those columns is inside the band with room to spare, so the trade set can
be cut **for quality** (quantified, tier A/B, one row per split group) instead of
scraped for volume, and a real held-out test set is affordable. This work did not
move the trade numbers at all and was never going to.

### Selection: still no as spec emission — but it is now a near miss rather than a rout

**525 distinct requests clear the bound, against a floor of 500.**

Do not read that as a pass. Every discount in §0 applies, and they compound:

* `strict` (no unruled keys forgiven) is **404**.
* A1's own error rate has not been paid yet, and neither has human rejection of the
  groups the miner scored generously.
* `funnel_report.md`'s judgement on the pre-flip 395 was "realistically well under
  300". Applying the same haircut to 525 lands around **380–420** — still short.

So the honest statement: **the shortfall went from roughly 200 examples to roughly
100.** Before this change the selection dialect was 395 against 500 and the answer
was "no, and not close". It is now 525 against 500 nominal, 404 strict, and the
answer is "no, but one more move closes it".

Counting the refusals as a second head, the selection **task** has 525 + 292 = **817**
distinct targets. That clears 500, but 36% of it teaches the model to decline —
better than the 48% it was before the flip, and still not a spec generator.

### What actually closes the remaining ~100

In order of leverage, all cheaper than a new block:

1. **Fix `measure.py::_timeframe_conditions`' hard-coded scope** (§4). Worth +34
   selection groups on its own, taking the bound to **559**, and it is a one-line
   change to a scope argument. It also stops 1,020 rows being counted as refusal gold
   for a capability that exists.
2. **Decide the shape of the ambiguous rows.** `unclear` holds 5,829 upper-bound
   groups with no dialect assigned — more than the trade dialect yields in total, and
   11x the whole selection set. Sharpening `classify_request`, or letting A1 decide
   the shape and recording its choice, is where the selection examples are hiding.
3. **Teach the miner lookback windows** so `GAP-HISTORICAL-WINDOW` stops being
   invisible. The capability landed and is probed; its demand is currently measured
   at exactly zero, which cannot be right.
4. **Decide what a compound request should produce.** 1,421 groups ask for a screen
   AND per-bar execution and today all are refusals — 2.7x the entire selection spec
   yield. Two specs, or a declared refusal; either answer converts them.

### What this pass still cannot tell you

Section 4 sees 5 of 16 gaps and the miner's recall is a floor, so refusal counts are
floors and spec counts are ceilings. The number that would settle it is the one this
pass deliberately did not spend: run A1 over a stratified sample of a few hundred
split groups, have a human adjudicate every condition, and compare the realised rate
against the bounds above. Until then, treat the trade "yes" as firm, the selection
"no" as firm but narrowing, and everything between as unmeasured.
