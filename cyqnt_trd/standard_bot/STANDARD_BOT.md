# Standard Bot — 完整输入 / 完整输出

一个基类，一套输入，一套输出。交易型、选币型、监控型只是**能力声明**不同，不是三套东西。

```text
catalog(66 个数据节点 + 用户自注册)
   → DataRequest 声明          ← bot 只能读它声明过的
   → 一次 as_of 全量取数 + 逐节点 status
   → BotContext                ← 完整输入
   → StandardBot.decide()
   → StandardSignal            ← 完整输出（含平仓方向）
   → SignalEnvelope → registry → executor / notification
```

---

## 一、完整输入

### 数据节点目录 `data/catalog.py`

66 个节点，覆盖文档里的整个数据宇宙（内部域 + 公开 + 本地 + 外采）：

| 取数通道 | 节点数 | 例子 |
|---|---:|---|
| `internal_http` | 28 | futuresRadar · movement · event/calendar · ETF · sector · ai-skill · portfolio |
| `public_binance` | 16 | klines · funding · OI · 多空比 · taker · basis · premiumIndex · F&G |
| `external_vendor` | 9 | BTC.D/总市值 · 期权链 · 跨所 · 宏观 DXY/VIX · 股票基本面 · 第三方新闻 · X |
| `square_skill` | 6 | getNews · getSearch · getTickerRank · getSentiment · getTopicTrending · getHotPost |
| `indicators_api` | 3 | 14 项 ta4j 指标 + K 线 |
| `local_parquet` | 3 | 清算 · 鲸鱼成交 · CME |
| `bdp_screening` | 1 | market_cap / taker_buy_pct / community_buzz / … 全部 api_id |

**47 个已接线可直接调用**，其余 19 个是尚未 onboard 的外采 / 仓库表，契约写好但不假装能取。

内部域端点由 `data_cli/internal.py` 提供（stdlib，QA/prod/gateway 三套路由，token 走环境变量），
`data_cli/internal_frames.py` 把每个响应规范成定列名的 DataFrame。

### 用户自定义数据接口

目录不可能穷举团队自己的源。`data/custom_sources.py` 是扩展点，三种形态：

```python
from cyqnt_trd.standard_bot.data import custom_sources as cs

cs.register_rest_node(name="team_sentiment", url="http://example.internal/api/sent",
                      records_path="data.items", availability="FORWARD_ONLY",
                      pit_hazard="实时轮询，无历史端点")

cs.register_callable_node(name="warehouse_factor", fn=my_query_fn,
                          availability="SEMI", pit_hazard="按日批，当日值盘中不可得")

cs.register_file_node(name="oi_panel", path="~/research/oi.parquet",
                      availability="SEMI", time_column="ts",
                      pit_hazard="研究期抓取，起点 2024-01")
```

注册完 `data.custom_team_sentiment(...)` 立刻可调用，`validate_nodes` 认它的回放纪律，
`DATA_API.md` 收录它，bot 用 `DataRequest("custom_oi_panel")` 声明它 —— 和内置节点完全同权。

三条硬规则：

- **必须显式声明 `availability`**，没有默认值。目录存在的意义就是「能不能回放」在回测前被回答，
  一个静默默认会答错。
- **非 `BACKTESTABLE` 必须写 `pit_hazard`**，否则拒绝注册。
- **不能覆盖内置节点**。传 `name="klines"` 直接报错并说明会被命名成 `custom_klines`、不会覆盖 ——
  否则作者以为改了 klines，实际每个策略读的还是内置的。

文件节点的时间列按 **epoch 毫秒**解析（`pd.to_datetime` 对裸 int64 默认当纳秒，会把所有 ms 戳
变成 1970，PIT 闸门直接失效）。

### 两条正交的轴

之前把「能不能取到」和「能不能回放」混在一起了，现在分开：

- **`source_path`（取数通道）** — 在域内环境，`internal_http` 和 `indicators_api` 都正常。
- **`availability`（可回放性）** — 与网络无关，是数据本身的性质：

```
BACKTESTABLE      6   klines · indicator_charts · funding · fear_greed · ahr999 · klines_multi_tf
SEMI             20   open_interest(~30d) · long_short_ratio · basis · etf_flow(T+1) · calendar · …
FORWARD_ONLY     27   futuresRadar(5min 快照) · ticker_rank · sentiment · bdp_screen · ai_signal · …
EXTERNAL_PENDING  9   btc_dominance · options_chain · macro_indicators · …
```

`FORWARD_ONLY` 不是「取不到」，是**没有 PIT 历史**。futuresRadar 是 5 分钟快照、bdp 只暴露当前截面 —— 拿去 walk-forward 会把今天的值贴到每根历史 bar 上。`data.validate_nodes(..., for_backtest=True)` 在编译期拦下并引用具体陷阱。

### `BotContext` — 递到策略手里的东西

```python
ctx.frames["klines"]          # 只有声明过的节点
ctx.source_status             # {"klines": "ok", "funding": "error"} —— 每个声明节点都有
ctx.warnings
ctx.position("BTCUSDT")       # 有符号持仓：>0 多 / <0 空 / 0 平
ctx.side_of("BTCUSDT")        # "long" / "short" / "flat"
ctx.equity
ctx.data_quality(required=["klines"])   # good / degraded / insufficient
ctx.require("orderbook_depth")          # 没声明就报错，不返回 None
```

关键：**「没读」和「读了是空的」不能混**。必需节点挂了 → `insufficient`；可选节点挂了 → `degraded`；两者都在 `source_status` 里带着走。

---

## 二、完整输出 `cyqnt.signal/v2`

### 之前缺的：平仓方向

老的 `SignalEnvelope` 只有 `side=BUY/SELL`。**「平掉多单」和「开空」都是 SELL**，执行层只能猜 —— 而这两件事在现货账户上一个能做一个不能做。

`PositionIntent` 把整个持仓生命周期说清楚：

| intent | target_side | **closes_side（平仓方向）** | order_side | reduce_only | 需要做空能力 |
|---|---|---|---|---|---|
| `open_long` | long | — | buy | ✗ | ✗ |
| `open_short` | short | — | sell | ✗ | ✓ |
| `add_long` / `add_short` | long / short | — | buy / sell | ✗ | ✗ / ✓ |
| `reduce_long` | long | **long** | sell | ✓ | ✗ |
| `reduce_short` | short | **short** | buy | ✓ | ✗ |
| `close_long` | flat | **long** | sell | ✓ | ✗ |
| `close_short` | flat | **short** | buy | ✓ | ✗ |
| `flip_to_short` | short | **long** | sell | ✗ | ✓ |
| `flip_to_long` | long | **short** | buy | ✗ | ✗ |
| `flat` | flat | **any** | — | ✓ | ✗ |
| `hold` | — | — | — | ✗ | ✗ |

`close_long` 和 `open_short` 在交易所都是 sell，但前者 `reduce_only=True` 且不需要做空能力 —— 现货账户依赖的正是这个区分。

### 其余字段

| 组 | 字段 |
|---|---|
| 标的 | `symbol` `venue` `product` `base_asset` `quote_asset` `market_scope` |
| 决策 | `intent` `direction` `advisory_action` `score` `confidence` |
| 进场 | `entry`：type / price / zone / time_in_force / post_only |
| **出场** | `exit_plan`：`stop_loss`(price\|pct\|atr_mult, trailing, **exchange_managed**) · `take_profit[]`(分批 close_pct) · `time_stop`(max_bars\|max_seconds) · `exit_on_opposite_signal` |
| 仓位 | `size`：mode(quantity\|quote\|equity_pct\|risk_pct\|position_pct) / value / leverage / max_notional / reduce_only |
| 风控 | `risk`：max_loss / max_position / max_leverage / liquidation_buffer / daily_loss_cap |
| 时效 | `time_horizon` `horizon_seconds` `valid_until` |
| 解释 | `topic` `reason_codes[]` `summary` `recommended_behavior` `evidence[]` |
| 质量 | `data_quality` `source_status` `warnings[]` |
| 截面 | `candidates[]`（每个可内嵌一份完整 trade signal）`universe_size` |
| 安全 | `auto_trade_eligible` `requires_confirmation` `dedup_key` |
| 溯源 | `provenance`：strategy_id / version / snapshot_id / config_hash / inputs / run_id / trace_id |

构造时就强制的一致性：

- **进场必须带 exit_plan** —— 想说「靠反向信号出场」就显式写 `ExitPlan(exit_on_opposite_signal=True)`，不能默认没有。
- `size.reduce_only` 由 intent 决定，作者写反了会被改正。
- `product=spot` + 需要做空能力的 intent → 直接报错，并提示改用 `close_long`。
- 可执行 intent 没有 symbol → 报错（没有标的的指令不可执行）。
- advisory signal 不能 `auto_trade_eligible=true`。

### 对接下单侧

```python
signal.to_execution_request()
# {"venue_class": "CEX_PERP", "intent_type": "LIMIT", "instrument": "BTCUSDT",
#  "side": "BUY", "position_intent": "open_long", "reduce_only": false,
#  "size": {...}, "params": {"time_in_force": "IOC", "price": 58200,
#                            "bracket": {"stop": 57036.0, "take_profit": [61110.0],
#                                        "exchange_managed": false}},
#  "client_tag": "...", "source_signal_id": "...", "requires_confirmation": true}
```

故意**不产出**的两项：`idempotency_key`（= `实例:节点:event_ref`，只有 executor 知道 run 身份；策略自己生成会让两次独立 run 撞键、同一 run 重放不撞键，正好反了）和 `strategy_instance_id` / `node_id`。

`advisory_action` 非空或 `intent=hold` 时调用直接抛错 —— ALERT 不可能被误走成单。

---

## 三、能力声明

```python
class MyBot(StandardBot):
    spec = BotSpec(bot_id="my_bot", kind=BotKind.TRADE, products=("usd_m_perpetual",))

    def required_data(self):
        return [DataRequest("klines", {"symbol": "BTCUSDT", "interval": "1h", "limit": 500}),
                DataRequest("funding", {"symbol": "BTCUSDT", "limit": 200}, required=False)]

    def decide(self, ctx):
        if ctx.side_of("BTCUSDT") == "long" and <退出条件>:
            yield StandardSignal(..., intent=PositionIntent.CLOSE_LONG)
        ...
```

`kind` + `products` 推导出允许的 intent 集合，`decide_checked` 逐条校验：

- `BotKind.ADVISORY` → 只能 `hold`，且必须带 `advisory_action`，且 `auto_trade_eligible` 必须 false。
- `products=("spot",)` → 允许 `open_long / add_long / reduce_long / close_long / flat / hold`，**不含任何 short**。
- `BotKind.SELECTION` → 截面排名 + 可选的每候选交易计划。

违规抛 `CapabilityError`，不是靠代码评审发现。

`provenance` / `source_status` 由框架盖章，不采信作者填的值。

---

## 四、跑

```bash
# 数据节点目录
python3 -c "from cyqnt_trd.standard_bot.data.catalog import list_nodes; print(len(list_nodes()))"

# codegen 用的 ground truth（BLOCKS_API.md 的对位文件）
python3 docs/gen_data_api.py && cat cyqnt_trd/standard_bot/data/DATA_API.md

# 测试
python3 -m pytest tests/standard_bot/test_standard_bot_contract.py \
                  tests/standard_bot/test_data_catalog_runtime.py -q
```

内部域端点的环境变量：`WORKING_ENV=qa` 切 QA，`USE_GATEWAY=true` + `GATEWAY_APP_TOKEN` 走网关，
`BDP_SCREENING_URL` / `INDICATORS_URL` 单独覆盖。网关开着但 token 空 → 立即报错，
不会拿回一个 403 body 被下游读成「没有数据」。

---

## 五、Sample Bot（照本地跑通的策略移植）

`strategies/standard/` —— 全部来自 `策略开发/` 里真正扛住样本外的那批，都是资金费率的读法，
而资金费率恰好是唯一有多年可回放历史的深序列。

| bot | 来源 | kind | 演示了什么 |
|---|---|---|---|
| `funding_crowding_neutral` | N003（+84.7%/2yr, Sharpe 1.83, beta −0.035） | TRADE · 截面 | 做空高费率/做多负费率，美元中性；**离开篮子 → CLOSE_LONG / CLOSE_SHORT**；换边 → **FLIP_TO_\*** 一条指令 |
| `funding_carry_gated` | A004（+42-44%/7yr, Sharpe ~3.8） | TRADE | 稳定性门控收息；**regime 崩了 → CLOSE_SHORT + `FUNDING_REGIME_UNSTABLE`** |
| `funding_oi_crowding_monitor` | D001（Sharpe 0.70，**低于 1.0 部署门槛**） | ADVISORY | 有真 edge 但没过自己的筛 → 只做监控。能力声明让这件事是结构性的，不是一条可以被忽略的备注 |

三个的实际输出（合成数据）：

```
=== N003 ===
  SOLUSDT   close_short   closes=short order=buy  reduce_only=True   [LEFT_BASKET]
  AVAXUSDT  close_long    closes=long  order=sell reduce_only=True   [LEFT_BASKET]
  BTCUSDT   open_short    closes=None  order=sell reduce_only=False  [FUNDING_CROWDED_LONG, DOLLAR_NEUTRAL_BASKET]
  DOTUSDT   open_long     closes=None  order=buy  reduce_only=False  [FUNDING_CROWDED_SHORT, DOLLAR_NEUTRAL_BASKET]
  basket: 2 多 / 2 空 / 6 观察（10 candidates 全排名）

=== A004 ===
  BTCUSDT   open_short    closes=None  order=sell reduce_only=False  [FUNDING_REGIME_STABLE_POSITIVE]
  ETHUSDT   close_short   closes=short order=buy  reduce_only=True   [FUNDING_REGIME_UNSTABLE]

=== D001（advisory）===
  BTCUSDT   action=avoid  dir=short intent=hold  [FUNDING_CROWDED_LONG, OI_BUILDUP_CONFIRMS, SQUEEZE_RISK]
  SOLUSDT   action=watch  dir=short intent=hold  [FUNDING_CROWDED_LONG, OI_UNAVAILABLE]
            auto_trade_eligible: False（三条都是）
```

几个刻意的取舍：

- **篮子腿的 exit_plan 是 `exit_on_opposite_signal=True` + note**，不是价格止损 —— 中性篮子每轮再平衡，
  出场靠下一轮的 CLOSE 信号。契约要求进场必须声明出场方式，那就把「靠再平衡出场」明说出来。
- **A004 的止损只兜未对冲窗口**，note 里写清了 delta 中性来自现货配对，不是这条信号自己的性质。
- **D001 读不到 OI 时**给 `OI_UNAVAILABLE` 并把等级压到 `watch`，永远到不了 `avoid` —— 单条未确认读数不足以叫人规避。
  同时它接受用户自注册的 OI 面板（`oi_panel_node="custom_oi_panel"`），团队有更深历史就直接插进来，不改这个文件。
