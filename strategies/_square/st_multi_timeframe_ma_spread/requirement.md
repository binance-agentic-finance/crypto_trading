# 多周期均线价差(多空)

在 BTCUSDT U 本位永续合约上按 1 小时 K 线运行。比较主周期均线和高周期均线(用 secondary_period×secondary_factor 根 1 小时 K 线近似)的相对价差:高于阈值做多,低于负阈值做空,反向时先平后反手。

## 参数

- 主周期均线(`primary_period`):默认 20
- 高周期均线(`secondary_period`):默认 20
- 价差阈值(bps)(`threshold_bps`):默认 0.0
- 高周期倍数(根/根)(`secondary_factor`):默认 4
- 目标仓位占权益比例(`target_fraction`):默认 0.2
- 止损比例(`stop_pct`):默认 0.03

## 仓位与风控

- 交易标的:BTCUSDT U 本位永续合约(1 倍杠杆、逐仓),1h K 线收盘后决策。
- 持仓以交易所为准:每轮从账户读当前持仓,没有新信号就保持。
- 仓位 = 目标仓位占权益比例 × 真实权益(合约账户钱包余额);不足最小下单金额时本轮不下单。
- 每次开仓后立即挂保护止损单,止损价 = 开仓参考价 ×(1 ∓ 止损比例)。
- 首轮先用 factor_evaluate 评估核心因子,verdict 为 PASS / PASS_CONDITIONAL / HOLD_INFO 才交易。
- 取数或下单出错时本轮跳过并记录日志,下一轮继续。
