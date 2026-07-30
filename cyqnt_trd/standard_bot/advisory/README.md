# 标准 Advisory Bot

**Advisory Bot 不是旁路，它就是 standard bot 的第三条 kind。** 交易型 emit `kind=TRADE`、选币型 emit `kind=SELECTION`、监控型 emit `kind=ALERT`，三者共用同一个 `DataSnapshot`、同一个 `SignalPluginRegistry`、同一个 `run_pipeline_step`。

```text
数据接口 → Adapter(FetchFrame) → build_advisory_snapshot → DataSnapshot.frames
        → SignalPluginRegistry.run_pipeline_step → SignalBatch(ALERT)
        → signals_to_frame(BotSignalFrame) → Watcher / Notification
```

Bot 只计算，不抓数据、不通知、不下单。首版固定 `auto_trade_eligible=false`。

| 层 | 代码 | 与既有 standard bot 的对位 |
|---|---|---|
| Data Adapter | `advisory/data.py`：`BinanceMarketMetricAdapter` / `SquarePublicAdapter` / `BnDataCatalogAdapter` | 对位 `data/adapters.py`；抓取并保留 `FetchFrame.status`，缺失就是缺失，不补零 |
| Snapshot Assembler | `data/advisory_assembler.py`：`build_advisory_snapshot` / `run_advisory` | 与 `data/selection_assembler.py`（`run_selection`）、`data/snapshot.py` 同级同风格 |
| Plugin | `advisory/base.py` + `advisory/samples/` | 与 `BlockStrategyPlugin` / `SelectionStrategyPlugin` 同为 `SignalPlugin`：`plugin_id` / `required_inputs()->Dict[str,bool]` / `run()` / `step()` / `initialize_state()` |
| Registry | `entrypoints/common.make_registry()` 里 `register_advisory_bots(registry)` | 与内建 plugin、`blocks.strategy.flush_pending_into` 进同一个 registry，按 `plugin_id` 寻址 |
| Pipeline | `entrypoints/common.build_advisory_pipeline()` → `SignalPipelineSpec` | 对位 `build_strategy_pipeline()` |
| Runner | `entrypoints/mvp_advisory.py` | 对位 `mvp_paper.py` / `mvp_backtest.py`，但**没有 executor 分支** |

`required_inputs()` 返回和其它 plugin 一样的 `Dict[str, bool]`（bundle 名 + 各 frame 名）；带 schema 的 typed 版本是 `required_data_query(config) -> DataQuery`。

## 标准输入

| Frame | 粒度 | 必要字段 |
|---|---|---|
| `MarketMetricFrame@1.0` | 一行一个 `symbol × metric × time` | `event_time, available_time, venue, product, symbol, metric, value` |
| `NewsEventFrame@1.0` | 一行一个 `event × symbol`；全市场事件可为空 symbol | `event_id, published_at, available_time, source_id, topic, urgency, symbol, title` |

`available_time` 必须小于等于 Bot 的 `decision_as_of`，否则拒绝运行。
数据装配层还要把每个 Frame 的 `FetchFrame.status` 写入 `SnapshotMeta.source_status`：必需空表只有显式 `ok` 才表示“确实无事件”，否则 Bot 会输出 `DATA_UNAVAILABLE`。

## 标准输出

每条提醒固定包含：

```text
symbol / venue / product / direction / action / score / confidence
horizon_seconds / valid_until / topic / urgency / reason_codes
summary / recommended_behavior / evidence / data_quality
```

- `direction`: `long | short | neutral`
- `action`: `alert | watch | avoid | investigate`
- `score`: 0–100；`confidence`: 0–1
- 新闻 Bot 是建议层，方向不等于订单。

## 样例 Bot

| Bot | 用途 | 当前数据 |
|---|---|---|
| `price_volume_monitor` | 24h 涨跌、量比、OI 异常 | `kline` 可用；OI 半历史 |
| `social_sentiment_monitor` | Square 排行、提及量、情绪 | PUBLIC Square 可用，前向快照 |
| `exchange_event_monitor` | 上币、下币、维护 | `getNews` 可用；`hot_event` 未接线 |
| `macro_institutional_monitor` | 宏观、ETF/机构资金 | 新闻可用；正式日历/ETF 接口未接线 |
| `catalyst_risk_monitor` | 升级、治理、解锁、安全事故 | 新闻可用；解锁/事件接口待接 |
| `derivatives_positioning_monitor` | 资金费率、OI、多空账户比、基差的拥挤度与挤压风险 | funding/OI/多空比/基差经 `BinanceMarketMetricAdapter` 直读 `data_cli`（binance-cli 通道） |

默认规则：量价 `|24h涨跌|≥5%`、量比 `≥2`、OI 增长 `≥20%`；社交取 Top 10 或提及量 `≥100`，bull ratio `≥0.65` / `≤0.35`。其余三个新闻 Bot 按事件类型、来源可靠度和交叉验证决定 `alert/watch/avoid/investigate`。

### `derivatives_positioning_monitor`（对应 7 月需求里 funding 195 / OI 120 / 多空比 59 / 基差 41 条对话）

读法是**持仓结构**，不是预测：

| 输入 | 判读 | direction |
|---|---|---|
| 年化资金费率 `≥ +30%` | 多头在付钱 = 多头拥挤，风险在多头一侧 | `short` |
| 年化资金费率 `≤ -30%` | 空头拥挤 | `long` |
| OI↑ + 价↑ | 新多进场 | `long` |
| OI↑ + 价↓ | 新空进场 | `short` |
| OI↓（无论价涨跌） | 平仓离场，不是新观点 | `neutral` |
| 多空账户比 `≥3.0` / `≤0.5` | 散户账户偏斜，只作佐证票 | `short` / `long` |
| 基差偏离 `≥0.5%` | 期现错位/carry 提示 | `neutral` |

- 两票同向 → `alert`；单票 → `watch`。
- 资金费率年化 `≥75%` + OI 增长 `≥20%` + 账户比同向极端 → `avoid` + `SQUEEZE_RISK`：这是强平级联的形状，建议是**站开**，不是反向追单。
- 两侧票势均力敌（差距 `<20%`）→ `neutral` + `CONFLICTING_POSITIONING`，不硬凑方向。
- `product=spot` 的行直接跳过：永续的持仓指标不外借给现货腿。
- 缺 funding 不等于 funding=0：该票不投，`data_quality` 降为 `degraded`，`summary` 里显示 `NA`。

Square 已验证可调用：`getNews/getHotPost/getTickerRank/getTopicTrending/getSentiment/getSearch`；`getFeed` 当前为空，不作 fallback。
`getSearch` 默认视为社区线索；没有官方链接或两条独立来源时，只输出 `investigate + neutral`。

## 最小使用

```python
from cyqnt_trd.standard_bot.advisory import create_advisory_bot, signals_to_frame

bot = create_advisory_bot("exchange_event_monitor")
batch = bot.run(snapshot, {"min_source_reliability": 0.6})
output = signals_to_frame(batch)
```

走标准 registry（和交易/选币策略同一条执行路径）：

```python
from cyqnt_trd.standard_bot.advisory import BinanceMarketMetricAdapter
from cyqnt_trd.standard_bot.data import build_advisory_snapshot, run_advisory
from cyqnt_trd.standard_bot.entrypoints.common import make_registry

# 三步：抓 → 装配 → 过 registry
fetched = BinanceMarketMetricAdapter().fetch_market_metrics(["BTCUSDT", "ETHUSDT"])
snapshot = build_advisory_snapshot(frames={"market_metrics": fetched})
result = make_registry().run_pipeline_step(
    snapshot, [{"plugin_id": "derivatives_positioning_monitor", "config": {}}]
)

# 或一步（按 bot.meta 声明的 frame 自动取数，内部同样走 run_pipeline_step）
result = run_advisory("derivatives_positioning_monitor", symbols=["BTCUSDT"])
result.batch.signals       # SignalEnvelope(kind=ALERT)
result.states              # 回传给下一轮 previous_states= 即可增量去重
```

命令行：

```bash
# 有哪些 bot、各自要什么数据
python -m cyqnt_trd.standard_bot.entrypoints.mvp_advisory --list

# 实跑一轮
python -m cyqnt_trd.standard_bot.entrypoints.mvp_advisory \
    --bot derivatives_positioning_monitor --symbols BTCUSDT,ETHUSDT

# 回放（不联网）
python -m cyqnt_trd.standard_bot.entrypoints.mvp_advisory \
    --bot price_volume_monitor --frame market_metrics=tmp/mm.csv --format json
```

`decision_as_of` 不传时取**全部 frame 里最新的 `available_time`**（live 监控的诚实读法），传了就是回放。所有时间统一 floor 到毫秒，与 `SnapshotMeta.decision_as_of` 的精度一致。

## 与下单侧契约的边界

下单侧的意图报文（`venue_class / intent_type / idempotency_key / instrument / side / size / params`）**不由 advisory bot 产生**，也不应该由它直接映射：

| 下单侧字段 | advisory 能提供的 | 差在哪 |
|---|---|---|
| `venue_class` | `venue` + `product`（`usd_m_perpetual` → `CEX_PERP`，`spot` → `CEX_SPOT`） | 机械映射，可由 router 完成 |
| `instrument` | `symbol` | 同上 |
| `side` | ❌ 只有 `direction`（信息方向） | `direction` 是"风险在哪一侧"，不是"买还是卖"；`short` 常常意味着**减多/回避**而非开空 |
| `size` | ❌ 无 | advisory 不知道账户、敞口与风险预算 |
| `intent_type` / `params` | ❌ 无 | 没有价格、TIF、bracket |
| `idempotency_key` | ❌ 不是 `dedup_key` | `dedup_key` 只用于**通知去重**（同一 symbol 同一来源时间不重复提醒）；下单幂等键是 `实例:节点:event_ref`，两个命名空间不能互换 |

因此 router 侧的硬规则应为：**`kind=ALERT` / `auto_trade_eligible=false` 的信号一律不得构造下单意图**。要走真单，必须另有一个显式的策略/风控节点把 alert 转成 trade 决策并补齐 size / 价格 / 幂等键。

单条输出示例：

```json
{
  "symbol": "ABCUSDT",
  "venue": "binance",
  "product": "spot",
  "direction": "long",
  "action": "alert",
  "score": 75,
  "confidence": 0.86,
  "topic": "exchange_announcement",
  "reason_codes": ["EXCHANGE_LISTING", "SOURCE_VERIFIED"],
  "auto_trade_eligible": false
}
```

导出兼容 Canvas 的 `version + nodes + edges` 配置：

```python
from cyqnt_trd.standard_bot.advisory import canvas_definition

dsl = canvas_definition("social_sentiment_monitor")
```

DSL 使用 `version=1.1`，`config` 是业务参数真相源，`fields.value` 只是 UI 快照。
