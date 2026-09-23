# strategies/_square —— 广场提交物

内置 7 个策略(`cyqnt_trd/standard_bot/signal/framework_strategies.py` 的 `FRAMEWORK_STRATEGIES`)的
广场提交包。可提交的 6 个各一个目录,结构与平台 demo 一致;`st_liquidation_reversal` 不可提交(见下)。

| 文件 | 说明 |
|---|---|
| `registry.json` | **手改的唯一来源**:strategyId / version / name / description / tags / shareLevel / freeFork / icon / requirement / market,custom 节点参数的 label/widget/min/max,`submittable` / `platformStrategyId` |
| `<strategyId>/<strategyId>.py` | 生成物:可执行 code(三段式,纯标准库 + 平台能力) |
| `<strategyId>/<strategyId>.yaml` | 生成物:spec(`strategy / trigger / nodes / edges`),节点 id 与 code 的 `@node` 函数、`ctx.state` key 一一对应 |
| `<strategyId>/basic_info.json` | 生成物:完整提交 payload(9 个字段,spec / code 为字符串),spec 的 `strategy.id` = payload `strategyId` |
| `<strategyId>/requirement.md` | 生成物:中文自然语言需求 —— 策略做什么、参数、仓位与风控 |

```bash
python scripts/build_square_payloads.py            # 重新生成各目录,并写 dist/square_payloads/<strategyId>.json
python scripts/build_square_payloads.py --check    # 只检查目录里的 4 个文件是否与生成器一致(多余文件也算漂移)
```

`basic_info.json` / `dist/square_payloads/*.json` 就是 `POST /v1/square/strategies/submit` 的请求体
(脚本不会自动提交),字段严格只有 strategyId / version / spec / code / description / tags / shareLevel /
freeFork / icon。

## 提交流程:strategyId 要用平台上的 ID

接口按 x-user-id + strategyId + version 查找**平台上已存在且至少部署过一次(模拟盘或实盘)**的策略,
否则返回 strategy not found / must be deployed。`st_*` 只是本地 id,不是平台 ID:

1. 在平台上用 `<strategyId>.yaml` + `<strategyId>.py` 创建策略,跑一次模拟盘;
2. 拿到平台分配的 strategyId(形如 `st_<名称>_<后缀>`),填到 `registry.json` 的 `platformStrategyId`,
   或生成时覆盖:`python scripts/build_square_payloads.py --strategy-id st_rsi_reversion=<平台 strategyId>`
   (payload 的 `strategyId` 和 spec 的 `strategy.id` 一起换);
3. 用生成的 JSON 提交。同一 strategyId 重复提交会覆盖上一次,并重新进入 PENDING 审核。

没映射平台 ID 时 payload 退回 `st_*`,脚本会打印提示。

两个 version 互相独立:提交接口的 `version` = registry `defaults.version` = `r1`(重新发布快照时升);
spec 里的 `strategy.version` = `defaults.specVersion` = `"1.0"`(带引号,与平台样例一致;改 spec 结构时升)。

## code 的结构

```
gate            @node("std:gate")    factor_evaluate 研究闸门,首轮评一次,缓存在 ctx.state["gate"]
fetch_klines    @node("std:fetch")   klines(symbol, timeframe, limit, market_type, closed_only=True)
fetch_position  @node("std:fetch")   持仓账本 = 交易所(现货 account_balances / 合约 futures_position_risk)
fetch_*         @node("std:fetch")   衍生品(仅 st_oi_funding_breakout):derivatives_market_metrics
signal_engine   @node("std:signal")  _analyze:① _factors → ② _forecast → ③ _sizing(三段之间只传 dict)
rebalance       @node("exec:entry")  目标 ≠ 当前才下单;开仓后挂保护止损
notify_signal   @node("exec:notify") notify(message, channel="app")
```

- **平台能力全部同步调用**(`out = klines(...)`),只有 `@node` / `@workflow` 函数是 async。
- `ctx.state["<节点 id>"]` 存节点的原始返回值(`.output` 由平台加);执行节点只在 condition 满足时写。
- `ctx.log(level, event, {...})`;`main` 对 `CancelledError` 直接抛出,其它异常记录后跳过本轮。
- 生成的 code 只 import `asyncio` / `time` / `decimal` + 平台能力,**不依赖 pandas / numpy**。阶段函数来自
  `cyqnt_trd/standard_bot/signal/framework_live.py` —— 内置策略的"最后一根 K 线"版本,纯 list 计算;
  仓库回测侧(`framework_strategies.py`)继续用 pandas。

## 与回测的一致性

- 持仓是事件驱动的(无事件 = 保持)。live 每轮从交易所读实际持仓作为 `held`,只判断最后一根 K 线:
  有事件按事件定目标,没事件保持 —— 与回测逐根持有语义一致,重启、部分成交也能对账。
- 测试(`tests/standard_bot/test_square_submit.py`)逐根滚动验证:令 `held` = 回测在前一根的持仓,生成 code
  的 `_analyze` 给出的 verdict / target_position / stop 与 pandas 版本在每一根上都相同(多个随机种子)。
  三段式重构本身也与重构前的实现逐位一致(信号 + 框架回测结果)。
- description 里不写回测数字 —— 框架回测(`--engine framework`)不扣资金费。

## 执行、仓位与风控

- **市场**:只做多的 3 个(`st_moving_average_cross` / `st_price_moving_average` / `st_rsi_reversion`)走**现货**:
  `place_order(instrument, side="BUY", quote_size=, order_type="MARKET")` 进场,
  `place_order(instrument, side="SELL", size=<全部基础币>, order_type="MARKET")` 离场。
  多空的 3 个走 U 本位合约:`futures_open_position(venue_class="um", instrument, side="BUY"/"SELL",
  position_side="LONG"/"SHORT", size, order_type="MARKET")`,翻转时先
  `futures_close_position(venue_class="um", instrument, close_at_trigger=False, order_type="MARKET")` 平旧仓;
  启动时 `futures_account_config(instrument, leverage=1, margin_type="ISOLATED")`。
- **仓位**:`target_fraction`(custom 参数,默认 0.2)× 真实权益 —— 现货 = 可用 USDT + 基础币市值,
  合约 = 钱包余额(`account_balances(balance_type="futures")`);随盈亏复利。低于 `MIN_NOTIONAL` 时本轮跳过。
- **止损(live 有、回测没有)**:`_sizing` 输出 `stop = price × (1 ∓ stop_pct)`(`stop_pct` 默认 0.03)。
  开仓后立即挂保护单:现货 `place_order(side="SELL", size=, order_type="STOP_LOSS", stop_price=)`,
  合约 `futures_close_position(close_at_trigger=True, order_type="STOP_MARKET", trigger_price=)`。
  内置策略的框架回测不模拟止损,所以 **live 会比回测多出止损离场**,回测数字不能直接当 live 预期。
  现货持仓判断把被止损单冻结的基础币(`locked`)也算上。
- **研究闸门**:`factor_evaluate(factor=<operator spec>)`,verdict ∈ {PASS, PASS_CONDITIONAL, HOLD_INFO}
  才交易(检查和写入都是 `ctx.state["gate"]`)。`impl_source` 是该策略 forecast 的 `score`(均线价差 /
  价格相对均线 / (50−RSI)/50 / 通道位置 / 多周期价差)写成的单函数,测试保证它在最后一根 K 线上等于仓库的
  score。`st_oi_funding_breakout` 的核心是持仓量 / 资金费率确认,写不成只吃价格的单函数 operator,
  gate 固定返回 HOLD_INFO(可交易、但没有研究证据)。
- icon 是短标识(`ma_cross` / `ma` / `rsi` / `breakout` / `trend` / `oi`),节点里的 `emoji` 另算。

## 数据

- 衍生品统一走 `derivatives_market_metrics`:持仓量 `metric_type="open_interest_history"`(`period` = K 线周期,
  相邻两期算 bps 变化;单点 `open_interest` 算不出逐根变化),资金费率 `metric_type="funding_rate_info"`
  (最后一条 `funding_rate`)。live 只给最后一根 K 线填这两个值;取不到时 `missing` 非空,本轮不交易
  (仓库回测侧缺列时仍按 0 处理,沿用原口径)。
- `st_liquidation_reversal` **不可提交**(registry `submittable: false`):平台没有清算 / 强平数据能力,
  不编造取数接口;生成脚本跳过它(不生成包、不生成 payload),仓库回测侧照旧可用。

## 平台 analysis 能力:只在逐位一致时复用

- **Donchian 通道 → 用 `rolling_extreme`**:`rolling_extreme(series=highs[:-1], op="max"/"min", period=N)["value"]`。
  `highs[:-1]` 去掉当前 K 线,正是回测的 `shift(1)`;max/min 没有浮点误差,通道值与回测相同(测试逐根比对)。
  K 线不足 N+1 根时不调用,视为无通道。
- **RSI → 保持自算**:仓库是简单均值 RSI;平台 `rsi` 已知用法是 `method="wilder"`,是否支持简单均值未知,
  Wilder 平滑会改变信号。
- **SMA / 均线价差 → 保持自算**:`sum/n` 一行,平台 `sma` 的返回形状、预热期处理未知,换过去没有收益。
- **OI**:平台 `oi_change_pct` 是百分比、仓库是 bps,对齐方式未知 —— 取原始值自算。

这些计算都在 `signal_engine`(custom 节点)的 `_factors` 里,没有拆成独立 analysis 节点:拆开后 `_forecast`
要跨节点读因子,三段只传 dict 的约定会被打散,逐位一致性也没法再用同一个函数验证。

## 待 SDK 确认

1. `account_balances` / `futures_position_risk` 返回 `records` 的字段名(`asset` / `free` / `locked` /
   `wallet_balance` / `positionAmt` / `positionSide`…)是防御式读取。
2. `derivatives_market_metrics` 的 `open_interest_history` / `funding_rate_info` 记录字段(`open_interest` /
   `funding_rate`)与排序(假定按时间升序)。
3. `rolling_extreme` 是否按"最后 `period` 个值取极值"实现(按平台已有用法推断)。
4. **离场 / 翻转时旧的保护止损单**:参考代码里没有撤单能力。现货离场前止损单会冻结基础币,市价 SELL 可能因
   可用余额不足失败;合约翻转后,旧的 `close_at_trigger` STOP_MARKET 单可能作用到新仓位。需要确认平台是否
   在平仓时自动撤销保护单,或提供撤单 / OCO 能力。
5. `place_order` 返回里的成交数量字段(用于止损单数量;取不到时退回 名义金额 / 价格)。
