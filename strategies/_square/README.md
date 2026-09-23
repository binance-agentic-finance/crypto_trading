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

`dist/square_payloads/*.json` 就是 `POST /v1/square/strategies/submit` 的请求体(不会自动提交)。
一致性由 `tests/standard_bot/test_square_submit.py` 保证:三段式与重构前信号逐根相同、生成的
code 在桩运行时下与仓库信号相同、spec 与 code 同步。

口径提示:
- description 里不写回测数字 —— 框架回测(`--engine framework`)不扣资金费。
- 持仓是事件驱动的(无事件 = 保持),live 的 `_sizing` 读 `ctx.state["position"]` 作为窗口前的持仓。
- `oi_funding_breakout` / `liquidation_reversal` 缺衍生品数据时,仓库回测按 0 处理(沿用原口径),
  提交 code 则直接跳过本轮不交易。
- 持仓量 / 资金费率 / 强平订单的 data 节点名(`open_interest_hist` / `funding_rate_history` /
  `liquidation_orders`)与返回字段需按平台 SDK 确认。
