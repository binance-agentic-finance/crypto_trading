# Standard Bot 设计

代码 `cyqnt_trd/standard_bot/`，样例 `./samples/`（由 `gen_samples.py` 从代码生成）。

## 是什么

一个 Standard Bot 处理全部数据源，一支实际策略是它激活一部分。

```
UniversalBot                   36 项能力 / 66 个数据节点 / 全部加工链
     ├── activate(...)         打开需要的那几项
     └── rule(ctx, features)   这支策略的判断
```

策略作者写两件事：开哪些数据、怎么判断。取数、PIT 闸门、规范化、状态记录、溯源盖章、
能力校验在 Standard Bot 里做完。

```python
from cyqnt_trd.standard_bot import UniversalBot, BotKind

def rule(ctx, f):
    funding = f.funding("BTCUSDT")            # bps；没激活这个源返回 None
    oi = f.oi_change_pct("BTCUSDT")
    if funding is None:
        return                                 # 读不到就弃权
    if funding > 20 and (oi or 0) > 10:
        yield f.signal("BTCUSDT", "open_short",
                       reason=["FUNDING_CROWDED_LONG", "OI_BUILDUP"],
                       stop_pct=0.02, size_pct=0.1)

bot = (UniversalBot("my_bot", kind=BotKind.TRADE, symbols=["BTCUSDT"])
       .activate_funding(["BTCUSDT"])
       .activate_open_interest(["BTCUSDT"])
       .with_rule(rule))

signals = bot.run_once()
```

## 四条约定

1. **取不到 ≠ 取到是空的。** Square 白名单外返回 `success:true` + 空 body，和「今天没新闻」长得一样。状态和数据分开传，空结果判 `degraded`。
2. **能取到 ≠ 能回放。** futuresRadar 是 5 分钟快照、bdp 只有当前截面，取得到但没有 PIT 历史。两条轴分开记。
3. **缺失 ≠ 零。** 读不到 funding 返回 `None`，策略据此弃权。
4. **没观察过就不能断言。** `CLOSE_LONG` 断言存在一个多单，没读持仓的 bot 发这条是猜。这是声明式能力，不是约定。

---

# 一、覆盖的数据源

`UniversalBot` 覆盖下面全部，一支策略 `.activate(...)` 打开需要的那几项。

### 量价

| 能力 | 数据节点 | 形状 | 可回放 | 说明 |
|---|---|---|---|---|
| `klines` | `data.klines` | bar | `BACKTESTABLE` | K 线 OHLCV |
| `indicators` | `data.indicator_charts` | bar | `BACKTESTABLE` | K 线 + 14 项 ta4j 指标 |
| `ticker` | `data.ticker_24h` | metric | `FORWARD_ONLY` | 24h 涨跌 / 量 |

### 衍生品 / 拥挤度

| 能力 | 数据节点 | 形状 | 可回放 | 说明 |
|---|---|---|---|---|
| `funding` | `data.funding` | metric | `BACKTESTABLE` | 资金费率历史 |
| `open_interest` | `data.open_interest` | metric | `SEMI` | 持仓量历史 |
| `long_short` | `data.long_short_ratio` | metric | `SEMI` | 多空账户比 |
| `top_trader` | `data.top_trader_ratio` | metric | `SEMI` | 顶级交易者持仓比 |
| `taker` | `data.taker_volume` | metric | `SEMI` | 主动买卖量比 |
| `basis` | `data.basis` | metric | `SEMI` | 期现基差 |
| `liquidations` | `data.liquidations` | metric | `SEMI` | 强平流 |
| `radar` | `data.futures_radar` | metric | `FORWARD_ONLY` | futuresRadar 全市场衍生品快照 |

### 消息 / 事件

| 能力 | 数据节点 | 形状 | 可回放 | 说明 |
|---|---|---|---|---|
| `news` | `data.news` | event | `FORWARD_ONLY` | Square 新闻 |
| `news_search` | `data.news_search` | event | `FORWARD_ONLY` | Square 关键词检索 |
| `announcements` | `data.announcements` | event | `SEMI` | 官方公告 |
| `hot_event` | `data.hot_event` | event | `SEMI` | 热点事件 |
| `calendar` | `data.calendar` | event | `SEMI` | 上币 / 活动日历 |
| `token_unlock` | `data.token_unlock` | event | `SEMI` | 代币解锁 |
| `macro_calendar` | `data.macro_calendar` | event | `SEMI` | 宏观日历（actual vs forecast） |

### 社交

| 能力 | 数据节点 | 形状 | 可回放 | 说明 |
|---|---|---|---|---|
| `social_rank` | `data.ticker_rank` | rank | `FORWARD_ONLY` | Square 币种热度榜 |
| `sentiment` | `data.sentiment` | rank | `FORWARD_ONLY` | Square 多空投票 |
| `topic_trending` | `data.topic_trending` | rank | `FORWARD_ONLY` | 话题热度 |

### 截面 / 选币

| 能力 | 数据节点 | 形状 | 可回放 | 说明 |
|---|---|---|---|---|
| `universe` | `data.universe` | rank | `FORWARD_ONLY` | 24h 全市场 ticker |
| `bdp_screen` | `data.bdp_screen` | rank | `FORWARD_ONLY` | bdp 截面筛选字段 |
| `coin_metrics` | `data.coin_metrics` | rank | `SEMI` | 市值 / 流动性 / 流通量 |
| `sector` | `data.sector_flow` | rank | `SEMI` | 板块热度与净流入 |

### 资金流 / 链上

| 能力 | 数据节点 | 形状 | 可回放 | 说明 |
|---|---|---|---|---|
| `large_flow` | `data.large_flow` | metric | `SEMI` | 大额充提（交易所进出） |
| `whale` | `data.whale_trades` | metric | `SEMI` | 鲸鱼成交 |
| `chip` | `data.chip_distribution` | book | `FORWARD_ONLY` | 筹码分布 |
| `top_player` | `data.top_player_movement` | rank | `FORWARD_ONLY` | 顶级交易者动向 |

### 宏观 / 情绪

| 能力 | 数据节点 | 形状 | 可回放 | 说明 |
|---|---|---|---|---|
| `fear_greed` | `data.fear_greed` | metric | `BACKTESTABLE` | 恐贪指数 |
| `ahr999` | `data.ahr999` | metric | `BACKTESTABLE` | AHR999 估值 |
| `etf_flow` | `data.etf_flow` | metric | `SEMI` | 现货 ETF 净流 |

### 账户

| 能力 | 数据节点 | 形状 | 可回放 | 说明 |
|---|---|---|---|---|
| `positions` | `data.contract_positions` | position | `FORWARD_ONLY` | 当前持仓 |
| `balance` | `data.account_balance` | metric | `FORWARD_ONLY` | 账户余额 |

### 微观结构

| 能力 | 数据节点 | 形状 | 可回放 | 说明 |
|---|---|---|---|---|
| `orderbook` | `data.orderbook_depth` | book | `FORWARD_ONLY` | 盘口深度 |

### 平台既有信号

| 能力 | 数据节点 | 形状 | 可回放 | 说明 |
|---|---|---|---|---|
| `ai_signal` | `data.ai_signal` | rank | `FORWARD_ONLY` | 平台既有多因子信号 |


## 按取数通道

| 通道 | 数量 | 内容 |
|---|---:|---|
| `internal_http` | 28 | futuresRadar · movement · event/calendar · ETF · sector · coinSelector · ai-skill · portfolio · announcements · onchain |
| `public_binance` | 16 | klines · funding · OI · 多空比 · taker · basis · premiumIndex · F&G · AHR999 |
| `external_vendor` | 9 | 未 onboard：BTC.D · 期权链 · 跨所 · 宏观 DXY/VIX · 股票基本面 · 第三方新闻 · X |
| `square_skill` | 6 | getNews · getSearch · getHotPost · getTickerRank · getTopicTrending · getSentiment |
| `indicators_api` | 3 | ta4j 14 指标 + K 线 |
| `local_parquet` | 3 | 清算 · 鲸鱼 · CME |
| `bdp_screening` | 1 | 一条 `source:"bdp"` clause 打通全部 api_id |

47 个已接线，19 个是未 onboard 的外采 / 仓库表：契约写好，不假装能取。

## 按可回放性

```
BACKTESTABLE      6   klines · indicator_charts · klines_multi_tf · funding · fear_greed · ahr999
SEMI             21   open_interest(~30d) · long_short_ratio · taker · basis · etf_flow(T+1) · calendar
FORWARD_ONLY     30   futuresRadar · Square 全家 · bdp_screen · ai_signal · 持仓 · 订单簿
EXTERNAL_PENDING  9   btc_dominance · options_chain · macro_indicators · cross_exchange
```

66 个里 6 个能 walk-forward。`validate_nodes(nodes, for_backtest=True)` 在编译期拦，给出具体原因：

```
data.ticker_rank is FORWARD_ONLY, not replayable for backtest:
  60s cache, no history: forward-collect or the backtest is fiction.
```

---

# 二、数据怎么处理

## 三层

| 层 | 谁 | 管什么 |
|---|---|---|
| ① 传输 | `RestSourceSpec`（`data_cli/rest_source.py`） | URL / params / 字段映射 / 类型转换 / TTL，无硬编码 host |
| ② 契约 | `DataNodeSpec`（`data/catalog.py`） | `availability` 能否回放 · `pit_hazard` 怎么骗人 · 参数 schema |
| ③ 规范化 | `cyqnt.input/v1`（`core/input_contract.py`） | 7 种形状，统一列名 |

`/bn-data` 内部端点和用户自己的接口在①完全一样，都是一个 JSON REST 源的声明。

## 一个源从接口到策略经过什么

以 `funding` 为例，其余 65 个路径相同：

| 阶段 | 做什么 | 产物 |
|---|---|---|
| ① 传输 | 声明 `fapi/v1/fundingRate` + 字段映射 + 缓存 | DataFrame（`symbol`/`rate`/`timestamp`/`mark_price`） |
| ② 契约 | 标 `availability=BACKTESTABLE` + 参数 schema | 「能回放」成为可校验的声明 |
| ③ 规范化 | 改名、宽转长、时间统一 UTC 毫秒 | `MetricFrame@1.0` |
| ④ PIT 闸门 | 丢弃 `available_time > decision_as_of`；源没给就用取数时间填并记名 | 决策时刻能看到的部分 |
| ⑤ 状态 | 逐节点记 ok / degraded / error | `ctx.source_status` |
| ⑥ 统一读法 | `f.funding(sym)` 换算成 bps | float 或 `None` |

新闻多一步 ④.5（`blocks/news_features.py`）：

```
去重       first_seen_at / repost_count / lead_time_seconds
实体消歧   只认 universe 内、正文出现过的 ticker；标 >3 个判无目标；解析不出返回 null
分类       12 类事件，各带 expected_half_life_seconds
情绪       score / confidence / 谁判的（lexicon | event_type | publisher）
聚合       → MetricFrame：news_count_1h / bull_ratio / max_reliability / min_lead_time
```

聚合这步是关键：变成 metric 之后新闻和 funding 一样能读，不用每支 bot 自己写关键词匹配。

## 七种形状

66 个源的形状只有 7 种，所以 bot 不需要 66 套读法。每种固定带两个时间列：

- `event_time` —— 这件事什么时候发生
- `available_time` —— 我们什么时候能知道

只有一个时间戳的源分不清「09:00 发生」和「09:00 发生但 09:07 才可读」，回测会按发布延迟的量偏乐观。
源没给 `available_time` 就用取数时间填，并在 `ctx.inferred_availability` 记名。

### `BarFrame@1.0` - 6 个节点

**一行 = one instrument × timeframe × bar**

OHLCV candles. The base series every technical block reads.

```json
{
  "instrument_id": "BTCUSDT",
  "timeframe": "1h",
  "open_time": "2026-08-06T06:00:00Z",
  "close_time": "2026-08-06T07:00:00Z",
  "available_time": "2026-08-06T07:00:00Z",
  "open": 61250.0,
  "high": 61480.0,
  "low": 61120.0,
  "close": 61390.0,
  "volume": 812.44,
  "quote_volume": 49871233.0,
  "trades": 21044,
  "confirmed": true
}
```

### `MetricFrame@1.0` - 29 个节点

**一行 = one instrument × metric × time**

Long-form numeric observations. One row per measurement, so a source that fails contributes zero rows instead of a zero value.

```json
{
  "event_time": "2026-08-06T00:00:00Z",
  "available_time": "2026-08-06T00:00:12Z",
  "instrument_id": "BTCUSDT",
  "metric": "funding_rate_8h",
  "value": 0.00012,
  "venue": "binance",
  "product": "usd_m_perpetual",
  "unit": "ratio",
  "window": "8h",
  "source_id": "binance.funding_rate",
  "quality": "ok"
}
```

### `EventFrame@1.0` - 11 个节点

**一行 = one event × instrument (instrument may be null for market-wide)**

Discrete dated events: news, announcements, calendar entries, unlocks, macro releases. Carries provenance so a directional read can be refused when the source is unverified.

```json
{
  "event_id": "sq-90412",
  "event_time": "2026-08-06T06:41:00Z",
  "available_time": "2026-08-06T06:43:20Z",
  "source_id": "square.getNews",
  "topic": "exchange_announcement",
  "instrument_id": "ABCUSDT",
  "urgency": "breaking",
  "title": "Binance Will List ABC",
  "summary": "…",
  "url": "https://www.binance.com/en/support/announcement/…",
  "bias": "positive",
  "event_type": "listing",
  "source_reliability": 0.85,
  "corroboration_count": 2,
  "quality_flags": []
}
```

### `RankFrame@1.0` - 15 个节点

**一行 = one instrument at one as-of**

Cross-sectional snapshot: a scored/ranked instrument list. Feature columns beyond the required ones are passed through untouched.

```json
{
  "available_time": "2026-08-06T07:00:00Z",
  "event_time": "2026-08-06T07:00:00Z",
  "instrument_id": "SOLUSDT",
  "rank": 1,
  "score": 0.91,
  "source_id": "square.getTickerRank",
  "mention_count": 812,
  "unique_authors": 431,
  "bullish_count": 590
}
```

### `PositionFrame@1.0` - 3 个节点

**一行 = one open position**

Account state. This is what the position lifecycle reads to know whether an exit means CLOSE_LONG or CLOSE_SHORT.

```json
{
  "available_time": "2026-08-06T07:00:03Z",
  "instrument_id": "BTCUSDT",
  "side": "short",
  "quantity": 0.42,
  "venue": "binance",
  "product": "usd_m_perpetual",
  "entry_price": 61810.0,
  "leverage": 5.0,
  "margin_ratio": 0.09,
  "unrealized_pnl": 176.4,
  "notional": 25783.8
}
```

### `BookFrame@1.0` - 2 个节点

**一行 = one price level on one side**

Order-book ladder snapshot.

```json
{
  "available_time": "2026-08-06T07:00:01Z",
  "instrument_id": "BTCUSDT",
  "side": "bid",
  "level": 1,
  "price": 61388.5,
  "quantity": 3.117
}
```

### `PanelFrame@1.0` - 0 个节点

**一行 = one timestamp; one column per instrument**

Wide time × instrument numeric panel — the natural input for a cross-sectional strategy that ranks a universe each bar.

```json
{
  "event_time": "2026-08-05T00:00:00Z",
  "available_time": "2026-08-05T00:00:12Z",
  "metric": "funding_rate_daily",
  "BTCUSDT": 0.00036,
  "ETHUSDT": 0.00021,
  "SOLUSDT": -0.00014
}
```


## 全部 66 个节点

#### `bar` / BarFrame@1.0 -- 6 个

| 节点 | 通道 | 可回放 | 已接线 |
|---|---|---|:-:|
| `data.cme_index` | local_parquet | `SEMI` | -- |
| `data.cross_exchange` | external_vendor | `EXTERNAL_PENDING` | -- |
| `data.indicator_charts` | indicators_api | `BACKTESTABLE` | yes |
| `data.klines` | indicators_api | `BACKTESTABLE` | yes |
| `data.klines_multi_tf` | indicators_api | `BACKTESTABLE` | yes |
| `data.premium_index` | public_binance | `SEMI` | -- |

#### `metric` / MetricFrame@1.0 -- 29 个

| 节点 | 通道 | 可回放 | 已接线 |
|---|---|---|:-:|
| `data.account_balance` | public_binance | `FORWARD_ONLY` | yes |
| `data.ahr999` | public_binance | `BACKTESTABLE` | yes |
| `data.basis` | public_binance | `SEMI` | yes |
| `data.btc_dominance` | external_vendor | `EXTERNAL_PENDING` | -- |
| `data.concentration` | internal_http | `SEMI` | -- |
| `data.equity_fundamentals` | external_vendor | `EXTERNAL_PENDING` | -- |
| `data.etf_flow` | internal_http | `SEMI` | yes |
| `data.fear_greed` | public_binance | `BACKTESTABLE` | yes |
| `data.funding` | public_binance | `BACKTESTABLE` | yes |
| `data.funding_current` | public_binance | `FORWARD_ONLY` | yes |
| `data.funding_widget` | internal_http | `FORWARD_ONLY` | yes |
| `data.futures_radar` | internal_http | `FORWARD_ONLY` | yes |
| `data.large_flow` | internal_http | `SEMI` | yes |
| `data.large_trade_info` | internal_http | `FORWARD_ONLY` | yes |
| `data.liquidations` | local_parquet | `SEMI` | yes |
| `data.long_short_ratio` | public_binance | `SEMI` | yes |
| `data.macro_indicators` | external_vendor | `EXTERNAL_PENDING` | -- |
| `data.onchain_signals` | internal_http | `FORWARD_ONLY` | -- |
| `data.open_interest` | public_binance | `SEMI` | yes |
| `data.options_chain` | external_vendor | `EXTERNAL_PENDING` | -- |
| `data.social_x` | external_vendor | `EXTERNAL_PENDING` | -- |
| `data.square_discussion` | internal_http | `FORWARD_ONLY` | -- |
| `data.stablecoin_supply` | external_vendor | `EXTERNAL_PENDING` | -- |
| `data.taker_volume` | public_binance | `SEMI` | yes |
| `data.ticker_24h` | public_binance | `FORWARD_ONLY` | yes |
| `data.token_indicators` | internal_http | `FORWARD_ONLY` | yes |
| `data.top_trader_ratio` | public_binance | `SEMI` | yes |
| `data.user_pnl` | internal_http | `SEMI` | -- |
| `data.whale_trades` | local_parquet | `SEMI` | -- |

#### `event` / EventFrame@1.0 -- 11 个

| 节点 | 通道 | 可回放 | 已接线 |
|---|---|---|:-:|
| `data.ai_summary` | internal_http | `FORWARD_ONLY` | -- |
| `data.announcements` | internal_http | `SEMI` | -- |
| `data.calendar` | internal_http | `SEMI` | yes |
| `data.event_upcoming` | internal_http | `SEMI` | yes |
| `data.hot_event` | internal_http | `SEMI` | yes |
| `data.hot_post` | square_skill | `FORWARD_ONLY` | yes |
| `data.macro_calendar` | internal_http | `SEMI` | yes |
| `data.news` | square_skill | `FORWARD_ONLY` | yes |
| `data.news_search` | square_skill | `FORWARD_ONLY` | yes |
| `data.news_vendor` | external_vendor | `EXTERNAL_PENDING` | -- |
| `data.token_unlock` | internal_http | `SEMI` | yes |

#### `rank` / RankFrame@1.0 -- 15 个

| 节点 | 通道 | 可回放 | 已接线 |
|---|---|---|:-:|
| `data.ai_signal` | internal_http | `FORWARD_ONLY` | yes |
| `data.bdp_screen` | bdp_screening | `FORWARD_ONLY` | yes |
| `data.coin_metrics` | internal_http | `SEMI` | yes |
| `data.etf_metadata` | external_vendor | `EXTERNAL_PENDING` | -- |
| `data.hot_coin` | internal_http | `FORWARD_ONLY` | yes |
| `data.market_scan` | public_binance | `FORWARD_ONLY` | yes |
| `data.sector_flow` | internal_http | `SEMI` | yes |
| `data.sentiment` | square_skill | `FORWARD_ONLY` | yes |
| `data.strategy_ranking` | internal_http | `FORWARD_ONLY` | yes |
| `data.ticker_rank` | square_skill | `FORWARD_ONLY` | yes |
| `data.top_player_movement` | internal_http | `FORWARD_ONLY` | yes |
| `data.topic_trending` | square_skill | `FORWARD_ONLY` | yes |
| `data.universe` | public_binance | `FORWARD_ONLY` | yes |
| `data.user_favorites` | internal_http | `FORWARD_ONLY` | yes |
| `data.web3_basket` | internal_http | `FORWARD_ONLY` | -- |

#### `position` / PositionFrame@1.0 -- 3 个

| 节点 | 通道 | 可回放 | 已接线 |
|---|---|---|:-:|
| `data.contract_positions` | internal_http | `FORWARD_ONLY` | yes |
| `data.positions` | public_binance | `FORWARD_ONLY` | yes |
| `data.user_portfolio` | internal_http | `FORWARD_ONLY` | yes |

#### `book` / BookFrame@1.0 -- 2 个

| 节点 | 通道 | 可回放 | 已接线 |
|---|---|---|:-:|
| `data.chip_distribution` | internal_http | `FORWARD_ONLY` | yes |
| `data.orderbook_depth` | public_binance | `FORWARD_ONLY` | yes |


## 用户自定义接口

| 入口 | 用途 |
|---|---|
| `register_rest_node` | 内联描述一个 HTTP 端点 |
| `register_source_node` | 给已有 `RestSourceSpec` 加契约 |
| `register_callable_node` | 任意返回 DataFrame 的函数（仓库查询、厂商 SDK） |
| `register_file_node` | 本地 parquet / csv |
| `bind_config_sources()` | 批量读 `$CYQNT_SOURCES_CONFIG` |

一条配置同时描述三层：

```json
{
  "name": "team_signal",
  "base_url": "http://example.internal", "path": "api/sig",
  "data_path": "data.items",
  "fields": {"symbol": {"key": "sym", "coerce": "str"},
             "score":  {"key": "v",   "coerce": "float"}},

  "availability": "FORWARD_ONLY",
  "pit_hazard": "实时轮询，无历史端点",

  "emits": "rank",
  "column_map": {"symbol": "instrument_id"}
}
```

三条会导致注册失败的规则：

- 必须写 `availability`。没写的只注册成「可 fetch」，不注册成节点并 warn。
- 非 `BACKTESTABLE` 必须写 `pit_hazard`。
- 不能覆盖内置节点。传 `name="klines"` 直接报错，否则作者以为改了 klines，实际策略读的还是内置那个。

---

# 三、输入结构

## `Features` —— 策略看到的读法

```python
f.close(sym)              f.funding(sym)            # bps
f.change_24h(sym)         f.oi_change_pct(sym)
f.bars(sym)               f.long_short_ratio(sym)
                          f.taker_ratio(sym)
                          f.basis_pct(sym)

f.events(enrich=True)     f.social(sym)             # {rank, mentions, bull_ratio}
f.news_metrics()          f.net_exchange_flow(sym)  # 正数 = 净流入交易所
f.fear_greed()            f.universe_frame()
f.ahr999()                f.sectors()

f.side_of(sym)            f.equity
f.position(sym)           f.quality / f.degraded()
```

没激活的能力返回 `None`。

## `BotContext` —— 底层

```python
ctx.frames["funding"]              # 原始帧，厂商列名（逃生通道）
ctx.view("funding")                # TypedFrame，规范列名 + 回放纪律
ctx.metric("funding", "rate")      # 最新值，取不到 None
ctx.source_status                  # 每个声明节点：ok / degraded / error
ctx.data_quality(required=[...])   # good / degraded / insufficient
ctx.inferred_availability          # 哪些源的 available_time 是推断的
ctx.require("orderbook_depth")     # 没声明就报错
```

必需节点挂 → `insufficient`；可选节点挂 → `degraded`。两者都带到输出。

### 完整 sample（一支读 5 个源的 bot）

```json
{
  "schema": "cyqnt.input/v1",
  "bot_id": "sample_bot",
  "decision_time": 1786000000000,
  "equity": 100000.0,
  "positions": {
    "BTCUSDT": -0.42
  },
  "source_status": {
    "klines": "ok",
    "funding": "ok",
    "open_interest": "ok",
    "news": "ok",
    "contract_positions": "ok"
  },
  "warnings": [
    "klines: klines: declared source column(s) absent from the response: symbol",
    "klines: klines: available_time inferred from fetch time — the source states no publication lag, so a replay may be optimistic by that lag",
    "funding: funding: available_time inferred from fetch time — the source states no publication lag, so a replay may be optimistic by that lag",
    "open_interest: open_interest: available_time inferred from fetch time — the source states no publication lag, so a replay may be optimistic by that lag",
    "news: news: available_time inferred from fetch time — the source states no publication lag, so a replay may be optimistic by that lag",
    "contract_positions: contract_positions: declared source column(s) absent from the response: entryPrice, positionAmt",
    "contract_positions: contract_positions: available_time inferred from fetch time — the source states no publication lag, so a replay may be optimistic by that lag"
  ],
  "inferred_availability": [
    "contract_positions",
    "funding",
    "klines",
    "news",
    "open_interest"
  ]
}
```

typed 视图（截前两个）：

```json
{
  "klines": {
    "kind": "bar",
    "schema": "BarFrame@1.0",
    "rows": [
      {
        "open_time": "2026-08-06T05:06:40.000Z",
        "open": 61100.0,
        "high": 61300.0,
        "low": 61010.0,
        "close": 61250.0,
        "volume": 744.1,
        "quote_volume": 45500000.0,
        "close_time": "2026-08-06T06:06:40.000Z",
        "trades": 19004,
        "instrument_id": "BTCUSDT",
        "timeframe": "1h",
        "available_time": "2026-08-06T07:06:40.000Z",
        "event_time": "2026-08-06T07:06:40.000Z"
      },
      {
        "open_time": "2026-08-06T06:06:40.000Z",
        "open": 61250.0,
        "high": 61480.0,
        "low": 61120.0,
        "close": 61390.0,
        "volume": 812.4,
        "quote_volume": 49871233.0,
        "close_time": "2026-08-06T07:06:40.000Z",
        "trades": 21044,
        "instrument_id": "BTCUSDT",
        "timeframe": "1h",
        "available_time": "2026-08-06T07:06:40.000Z",
        "event_time": "2026-08-06T07:06:40.000Z"
      },
      {
        "open_time": "2026-08-06T06:06:40.000Z",
        "open": 61390.0,
        "high": 61520.0,
        "low": 61280.0,
        "close": 61460.0,
        "volume": 690.2,
        "quote_volume": 42400000.0,
        "close_time": "2026-08-06T08:06:40.000Z",
        "trades": 17322,
        "instrument_id": "BTCUSDT",
        "timeframe": "1h",
        "available_time": "2026-08-06T07:06:40.000Z",
        "event_time": "2026-08-06T07:06:40.000Z"
      }
    ],
    "status": "ok",
    "availability": "BACKTESTABLE"
  },
  "funding": {
    "kind": "metric",
    "schema": "MetricFrame@1.0",
    "rows": [
      {
        "instrument_id": "BTCUSDT",
        "event_time": "2026-08-05T15:06:40.000Z",
        "unit": "ratio",
        "window": "8h",
        "source_id": "binance.funding_rate",
        "metric": "rate",
        "value": 9e-05,
        "available_time": "2026-08-06T07:06:40.000Z"
      },
      {
        "instrument_id": "BTCUSDT",
        "event_time": "2026-08-05T23:06:40.000Z",
        "unit": "ratio",
        "window": "8h",
        "source_id": "binance.funding_rate",
        "metric": "rate",
        "value": 0.00011,
        "available_time": "2026-08-06T07:06:40.000Z"
      },
      {
        "instrument_id": "BTCUSDT",
        "event_time": "2026-08-06T07:06:40.000Z",
        "unit": "ratio",
        "window": "8h",
        "source_id": "binance.funding_rate",
        "metric": "rate",
        "value": 0.00012,
        "available_time": "2026-08-06T07:06:40.000Z"
      }
    ],
    "status": "ok",
    "availability": "BACKTESTABLE"
  }
}
```

---

# 四、输出结构

## `PositionIntent`

老的 `SignalEnvelope` 只有 `side=BUY/SELL`，平多单和开空都是 SELL，执行层只能猜。

| intent | target_side | closes_side | order_side | reduce_only | 需做空能力 |
|---|---|---|---|---|---|
| `open_long` / `open_short` | long / short | — | buy / sell | ✗ | ✗ / ✓ |
| `add_long` / `add_short` | long / short | — | buy / sell | ✗ | ✗ / ✓ |
| `reduce_long` / `reduce_short` | long / short | long / short | sell / buy | ✓ | ✗ |
| `close_long` / `close_short` | flat | long / short | sell / buy | ✓ | ✗ |
| `flip_to_short` / `flip_to_long` | short / long | long / short | sell / buy | ✗ | ✓ / ✗ |
| `flat` | flat | any | — | ✓ | ✗ |
| `hold` | — | — | — | ✗ | ✗ |

`close_long` 和 `open_short` 都是 sell，但前者 `reduce_only=True` 且不需要做空能力。

## `StandardSignal` 字段

| 组 | 字段 |
|---|---|
| 身份 | `schema` `bot_id` `bot_version` `signal_id` `decision_time` |
| 标的 | `symbol` `venue` `product` `base_asset` `quote_asset` `market_scope` |
| 决策 | `intent` `direction` `advisory_action` `score` `confidence` |
| 进场 | `entry`：type / price / zone / time_in_force / post_only |
| 出场 | `exit_plan`：`stop_loss`(price \| pct \| atr_mult · trailing · exchange_managed) · `take_profit[]` 分批 · `time_stop` · `exit_on_opposite_signal` |
| 仓位 | `size`：mode(quantity \| quote \| equity_pct \| risk_pct \| position_pct) / value / leverage / max_notional / reduce_only |
| 风控 | `risk`：max_loss / max_position / max_leverage / liquidation_buffer / daily_loss_cap |
| 时效 | `time_horizon` `horizon_seconds` `valid_until` |
| 解释 | `topic` `reason_codes[]` `summary` `recommended_behavior` `evidence[]` |
| 质量 | `data_quality` `source_status` `warnings[]` |
| 截面 | `candidates[]`（每个可内嵌完整 trade signal）`universe_size` |
| 安全 | `auto_trade_eligible` `requires_confirmation` `dedup_key` |
| 溯源 | `provenance`：strategy_id / version / snapshot_id / config_hash / inputs / run_id / trace_id |

构造时强制：

- 进场必须带 `exit_plan`，想靠反向信号出场就显式写出来
- `size.reduce_only` 由 intent 决定，写反了会被改正
- `product=spot` + 需要做空的 intent 报错，提示改用 `close_long`
- 可执行 intent 没有 `symbol` 报错
- advisory signal 不能 `auto_trade_eligible=true`
- `provenance` / `source_status` 由框架盖章

## 对接下单侧

```json
{
  "venue_class": "CEX_PERP",
  "intent_type": "LIMIT",
  "instrument": "BTCUSDT",
  "side": "BUY",
  "position_intent": "open_long",
  "reduce_only": false,
  "size": {
    "mode": "risk_pct",
    "value": 0.01,
    "leverage": 5,
    "max_notional_quote": 2000.0,
    "reduce_only": false
  },
  "params": {
    "time_in_force": "GTC",
    "price": 61300.0,
    "zone": [
      61200.0,
      61400.0
    ],
    "bracket": {
      "stop": 60340.0,
      "take_profit": [
        63139.0,
        64978.0
      ],
      "exchange_managed": false
    }
  },
  "client_tag": "sample_bot:binance:usd_m_perpetual:B",
  "source_signal_id": "ba83c3b0-8376-59e3-aef4-6e6c0138c90b",
  "requires_confirmation": true
}
```

不产出 `idempotency_key`（= `实例:节点:event_ref`）和 `strategy_instance_id` / `node_id`：
只有 executor 知道 run 身份，策略自己生成会让两次独立 run 撞键、同一 run 重放不撞键。

`advisory_action` 非空或 `intent=hold` 时调用直接抛错。

## 输出 sample

### 开仓

```json
{
  "schema": "cyqnt.signal/v2",
  "bot_id": "sample_bot",
  "bot_version": "v1",
  "signal_id": "ba83c3b0-8376-59e3-aef4-6e6c0138c90b",
  "decision_time": 1786000000000,
  "symbol": "BTCUSDT",
  "venue": "binance",
  "product": "usd_m_perpetual",
  "base_asset": "BTC",
  "quote_asset": "USDT",
  "market_scope": "single",
  "intent": "open_long",
  "target_side": "long",
  "closes_side": null,
  "order_side": "buy",
  "reduce_only": false,
  "direction": "long",
  "advisory_action": null,
  "score": 72.0,
  "confidence": 0.71,
  "entry": {
    "type": "limit",
    "price": 61300.0,
    "zone": [
      61200.0,
      61400.0
    ],
    "time_in_force": "GTC",
    "post_only": false
  },
  "exit_plan": {
    "stop_loss": {
      "price": null,
      "pct": null,
      "atr_mult": 2.0,
      "atr_value": 480.0,
      "trailing": true,
      "exchange_managed": false
    },
    "take_profit": [
      {
        "close_pct": 0.5,
        "price": null,
        "pct": 0.03,
        "atr_mult": null,
        "atr_value": null
      },
      {
        "close_pct": 0.5,
        "price": null,
        "pct": 0.06,
        "atr_mult": null,
        "atr_value": null
      }
    ],
    "time_stop": {
      "max_bars": 96,
      "max_seconds": null
    },
    "exit_on_opposite_signal": true,
    "note": "ATR trailing stop is evaluated in-process; place a venue STOP if the position must survive the daemon"
  },
  "size": {
    "mode": "risk_pct",
    "value": 0.01,
    "leverage": 5,
    "max_notional_quote": 2000.0,
    "reduce_only": false
  },
  "risk": {
    "max_loss_quote": null,
    "max_position_quote": null,
    "max_leverage": null,
    "liquidation_buffer_pct": null,
    "daily_loss_cap_quote": null
  },
  "time_horizon": "intraday",
  "horizon_seconds": 14400,
  "valid_until": 1786014400000,
  "topic": "mtf_trend",
  "reason_codes": [
    "HTF_TREND_UP",
    "MACD_GOLDEN_CROSS",
    "ADX_TRENDING"
  ],
  "summary": "BTCUSDT 顺势做多：4h 趋势向上 + 1h MACD 金叉 + ADX 27",
  "recommended_behavior": "限价区间挂单；ATR 止损随价上移。",
  "evidence": [
    {
      "source": "data.klines",
      "observed": {
        "adx": 27.4,
        "macd_hist": 41.2
      },
      "available_time": 1786000000000,
      "url": "",
      "note": ""
    }
  ],
  "data_quality": "good",
  "source_status": {
    "klines": "ok",
    "funding": "ok",
    "contract_positions": "ok"
  },
  "warnings": [],
  "candidates": [],
  "universe_size": 0,
  "auto_trade_eligible": false,
  "requires_confirmation": true,
  "dedup_key": "sample_bot:binance:usd_m_perpetual:BTCUSDT:open_long",
  "provenance": {
    "strategy_id": "sample_bot",
    "strategy_version": "v1",
    "snapshot_id": "0d2f…",
    "config_hash": "",
    "inputs": [
      "klines",
      "funding",
      "contract_positions"
    ],
    "run_id": "",
    "trace_id": ""
  }
}
```

### 平仓（平的是空单：买回、reduce-only）

```json
{
  "intent": "close_short",
  "target_side": "flat",
  "closes_side": "short",
  "order_side": "buy",
  "reduce_only": true,
  "symbol": "BTCUSDT",
  "size": {
    "mode": "position_pct",
    "value": 1.0,
    "leverage": null,
    "max_notional_quote": null,
    "reduce_only": true
  },
  "reason_codes": [
    "FUNDING_REGIME_UNSTABLE"
  ],
  "summary": "BTCUSDT 平掉套息空腿：资金费率符号翻转率 38%"
}
```

### 截面

```json
{
  "market_scope": "cross_section",
  "symbol": null,
  "intent": "hold",
  "universe_size": 10,
  "candidates": [
    {
      "symbol": "DOTUSDT",
      "rank": 1,
      "score": 0.0011,
      "direction": "long",
      "reason": "funding rank 1/10",
      "features": {
        "crowding_score": 0.0011
      },
      "trade": null
    },
    {
      "symbol": "BTCUSDT",
      "rank": 10,
      "score": -0.0009,
      "direction": "short",
      "reason": "funding rank 10/10",
      "features": {
        "crowding_score": -0.0009
      },
      "trade": null
    }
  ],
  "summary": "资金费率拥挤度截面：2 多 / 2 空 / 6 观察"
}
```

### 监控

```json
{
  "symbol": "BTCUSDT",
  "intent": "hold",
  "direction": "short",
  "advisory_action": "avoid",
  "score": 90.0,
  "reason_codes": [
    "FUNDING_CROWDED_LONG",
    "OI_BUILDUP_CONFIRMS",
    "SQUEEZE_RISK"
  ],
  "summary": "BTCUSDT 拥挤多：资金费率 20.0 bps，OI 变化 +35.0%",
  "auto_trade_eligible": false
}
```

---

# 五、能力声明

```python
UniversalBot("my_bot", kind=BotKind.TRADE, products=("usd_m_perpetual",))
    .activate_positions()      # 激活持仓 → reads_positions 自动置 True
```

`kind` + `products` + `reads_positions` 推导出允许的 intent，`decide_checked` 逐条校验：

| 声明 | 允许的 intent |
|---|---|
| `kind=ADVISORY` | 只有 `hold`，必须带 `advisory_action`，`auto_trade_eligible` 必须 false |
| `products=("spot",)` | `open_long` / `add_long` / `reduce_long` / `close_long` / `flat` / `hold` |
| `reads_positions=False`（默认） | `open_long` / `open_short` / `hold` |

现在没有自动读用户持仓：`contract_positions` 是节点、`ctx.positions` 是字段，但默认没东西填它。
所以没声明的 bot 发 `CLOSE_LONG` 会被拒：

```
blind emitted intent=close_long but declares reads_positions=False. That
instruction asserts an existing position this bot never observed. Either
declare a PositionFrame input (e.g. DataRequest('contract_positions')) and
set reads_positions=True, or emit only OPEN_*/HOLD.
```

声明了 PositionFrame 输入的 bot，`fetch_context` 自动拿它填 `positions`。

## 命令行

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_standard_bot --list
python -m ... mvp_standard_bot --bot funding_carry_gated --describe
python -m ... mvp_standard_bot --bot funding_crowding_neutral --run
python -m ... mvp_standard_bot --bot news_catalyst_trade --check-backtest
```

没有下单分支，跑完输出 `StandardSignal`。

---

# 六、样例 Bot

| bot | kind | 数据源 | 读法 |
|---|---|---|---|
| `funding_crowding_neutral` | TRADE 截面 | funding | 做空高费率、做多负费率，美元中性 |
| `funding_carry_gated` | TRADE | funding | 只在费率长期为正且符号稳定时收息 |
| `funding_oi_crowding_monitor` | ADVISORY | funding + OI | 费率给拥挤的价格，OI 给拥挤的规模 |
| `news_catalyst_trade` | TRADE | news + kline + funding | 消息触发，量价只做否决 |
| `social_flow_divergence` | ADVISORY | 社交热度 + 交易所资金流 | 说的和做的不一致 |

前三支来自 `策略开发/` 里扛住样本外的那批（N003 +84.7%/2yr Sharpe 1.83；A004 Sharpe ~3.8；
D001 Sharpe 0.70，未过 1.0 部署门槛所以只做监控）。

## `news_catalyst_trade`

```
触发   news → news_features → 高可靠度利好事件
否决   消息年龄 > 30min        我们读到太晚
       转发链延迟 > 15min      我们是第 N 手
       事前涨幅 > 3%           已经被定价
       资金费率 ≥ 15bps        已经拥挤
持有   事件类型决定：上币 30 分钟，主网升级 2 天 → TimeStop，到期即平
```

两个「晚」不一样：刚发的转发 age 小而 lead_time 大，原创但轮询晚的 lead_time 小而 age 大。

## `social_flow_divergence`

```
热度涨 + 资金净流入交易所  →  借热度出货    ALERT / short
热度涨 + 资金净流出交易所  →  买了往外搬    WATCH / long
```

单看任何一边都没有信息量，背离才是信号。两个输入都是 `FORWARD_ONLY` 证伪不了，所以做 ADVISORY。

## 候选设计（节点已在目录里）

| bot | 数据源 | 读法 |
|---|---|---|
| `unlock_supply_pressure` | `token_unlock` + `coin_metrics` + funding | 未来 N 天解锁量 / 流通市值 超阈值 → 供给压制 |
| `macro_surprise_gate` | `macro_calendar` + `fear_greed` + `etf_flow` | CPI 3.2% 不是新闻，vs 预期 3.0% 才是；surprise z-score 做风险开关 |
| `sector_rotation` | `sector_flow` + `coin_metrics` + `universe` | 板块净流入排名 → 选强势板块成分币 |
| `listing_frontrun_monitor` | `calendar` + `bdp_screen` + `ticker_rank` | 已排期上币 + 社交热度提前提醒；日程是公开信息，只提醒 |
| `whale_accumulation` | `large_flow` + `chip_distribution` + `top_player_movement` | 大额净流出 + 筹码集中 + 顶级交易者同向 |
| `etf_flow_regime` | `etf_flow` + `btc_dominance` + funding | ETF 连续净流入 + BTC.D 上升 = 机构主导，压制山寨腿 |

---

# 七、边界

| 缺口 | 影响 | 现在怎么办 |
|---|---|---|
| 信号之间没有原子组 | 中性篮子部分成交 = 不中性；carry 缺一腿 = 裸敞口 | 执行层保证，或缩到单腿 |
| 没有目标持仓表 | 执行层要自己理解「没提到 = 不动」 | 约定：只有明确 CLOSE 才平 |
| `valid_until` 到期行为未定义 | 撤单还是转市价 | 执行层决定 |
| 没有 CANCEL / REPLACE intent | 改不了已挂未成交的单 | 执行层自己撤 |
| 执行回执不回流 | 读不到「上一条为什么没成」 | 下一轮从 `contract_positions` 兜 |
| 66 个源只有 6 个可回测 | 多数策略今天验证不了 | 编译期拦住，别假装 |
| news / 社交无 PIT 历史 | 消息类策略回测不了 | 现在开始落 `first_seen_at` |

前五条是信号之间的关系，单条信号的契约是齐的。后两条是数据事实。
