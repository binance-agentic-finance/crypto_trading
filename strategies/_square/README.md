# strategies/_square —— 广场提交物

内置 7 个策略(`cyqnt_trd/standard_bot/signal/framework_strategies.py` 的 `FRAMEWORK_STRATEGIES`)
的广场提交 spec + code。

| 文件 | 说明 |
|---|---|
| `registry.json` | **手改的唯一来源**:strategyId / version / name / description / tags / shareLevel / freeFork / icon,以及 custom 节点参数的 label/widget/min/max/step |
| `<strategyId>/code.py` | 生成物。三段式 `_factors → _forecast → _sizing`,由 `_analyze` 串起来;阶段函数直接从仓库源码复制 |
| `<strategyId>/spec.yaml` | 生成物。`strategy / trigger / nodes / edges`,节点 id 与 code 里的 `@node` 函数一一对应 |

```bash
python scripts/build_square_payloads.py            # 重新生成 code/spec,并写 dist/square_payloads/<strategyId>.json
python scripts/build_square_payloads.py --check    # 只检查提交的 code/spec 是否与生成器一致
```

`dist/square_payloads/*.json` 就是 `POST /v1/square/strategies/submit` 的请求体(不会自动提交),
字段严格只有 strategyId / version / spec / code / description / tags / shareLevel / freeFork / icon,
其中 `spec`(YAML 正文)和 `code`(Python 正文)都是字符串。

### 提交流程:strategyId 要用平台上的 ID

接口按 x-user-id + strategyId + version 查找**平台上已存在且至少部署过一次(模拟盘或实盘)**的策略,
否则会返回 strategy not found / must be deployed。这里的 `st_*` 只是 spec 里的 `strategy.id`,
不是平台 ID。步骤:

1. 在平台上用 `<strategyId>/spec.yaml` + `code.py` 创建策略,跑一次模拟盘;
2. 拿到平台分配的 strategyId(形如 `st_<名称>_<后缀>`),填到 `registry.json` 的
   `platformStrategyId`,或生成时覆盖:
   `python scripts/build_square_payloads.py --strategy-id st_rsi_reversion=<平台 strategyId>`;
3. 用生成的 JSON 提交。同一 strategyId 重复提交会覆盖上一次,并重新进入 PENDING 审核。

没映射平台 ID 时,payload 的 strategyId 退回 `st_*`,脚本会打印提示。

### 两个 version

- 提交接口的 `version` = registry `defaults.version` = `r1`;
- spec YAML 里的 `strategy.version` = registry `defaults.specVersion` = `"1.0"`(带引号,和平台样例一致)。
两者独立:改 spec 结构时升 specVersion,重新发布快照时升接口 version。
一致性由 `tests/standard_bot/test_square_submit.py` 保证:三段式与重构前信号逐根相同、生成的
code 在桩运行时下与仓库信号相同、spec 与 code 同步。

口径提示:
- description 里不写回测数字 —— 框架回测(`--engine framework`)不扣资金费。
- 持仓是事件驱动的(无事件 = 保持),live 的 `_sizing` 读 `ctx.state["position"]` 作为窗口前的持仓。
- `ctx.state` 的 key = spec 节点 id,存节点的原始返回值(`.output` 由平台加):workflow 里每个节点都有
  `ctx.state["<id>"] = ...`(执行节点只在 condition 满足时写)。唯一的例外是 `ctx.state["position"]`:
  它不是节点,是跨轮次的持仓状态,由 `rebalance` 维护;spec 的 condition 不引用它,而是引用
  `signal_engine` 输出里的 `held_position` / `rebalance_needed`。
- `ctx.log` 统一是 `ctx.log(level, event, {...})`;`main` 对 `CancelledError` 直接抛出,其它异常记录后跳过本轮。
- icon 是短标识(`ma_cross` / `ma` / `rsi` / `breakout` / `trend` / `oi` / `liquidation`),节点里的 `emoji` 另算。

### 为什么因子都在 `signal_engine`(custom)里算,不拆成平台 analysis 节点

信号必须与仓库回测逐位一致,而平台能力的口径/返回形状无法在这里验证:
- RSI:仓库是简单均值 RSI,平台 `rsi` 默认 Wilder,口径不同;
- Donchian:仓库通道 `shift(1)` 不含当前 bar,平台只有 `trend_channel`,是否含当前 bar 没写明;
- SMA / 价差:平台 `sma` 窗口口径一致,但返回的是“最新值 + 序列”还是只有最新值、预热期 NaN 怎么处理
  都未知;而 forecast 需要完整序列(上穿/下穿要看上一根,持仓要从窗口内最后一个事件前推);
- OI:平台 `oi_change_pct` 是百分比、仓库是 bps,且对齐方式未知。
不确定就不拆:阶段函数原样从仓库源码复制,由测试保证与回测逐位一致。
- `oi_funding_breakout` / `liquidation_reversal` 缺衍生品数据时,仓库回测按 0 处理(沿用原口径),
  提交 code 则直接跳过本轮不交易。
- 与用户样例对齐的调用(参数名照样例):`klines(symbol, timeframe, limit, market_type="futures",
  closed_only=True)`、`futures_account_config(instrument, leverage, margin_type="ISOLATED")`、
  `futures_open_position(venue_class="um", instrument, size, side, order_type="MARKET")`、
  `futures_close_position(venue_class="um", instrument, close_at_trigger=False, order_type="MARKET")`
  (样例的 `close_at_trigger=True` + `STOP_MARKET` 是挂止损单;这里是信号翻转时立即平仓)、
  `notify(message, channel="app")`;取数节点都是 `@node("std:fetch", retries=2)`。
  `side`:开多 `BUY`、开空 `SELL`(verdict 仍是 `LONG`/`SHORT`,只在执行层映射);翻转时先平旧仓再按新方向开。
- **待 SDK 确认**:持仓量 / 资金费率 / 强平订单的 data 节点名(`open_interest_hist` /
  `funding_rate_history` / `liquidation_orders`)、参数与返回字段 —— 样例里没有,保持现状。
