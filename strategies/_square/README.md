# strategies/_square —— 广场提交物的生成源

**生成出来的 6 个提交包已经迁到 binance-ai-platform**:
`be/binance-ai-platform` 分支 `feat/convert-cases-strategies` → `examples/strategy-case-corpus/three_stage/`
(三段式输入/输出规范写在 `examples/strategy-case-corpus/README.md`)。这里不再提交生成物。

这里只保留:

| 文件 | 说明 |
|---|---|
| `registry.json` | 手改的唯一来源:strategyId / version / name / description / tags / 参数 label/widget / `submittable` / `platformStrategyId` |
| `../../scripts/build_square_payloads.py` | 生成器:从 `cyqnt_trd/standard_bot/signal/framework_live.py` 生成 code / spec / basic_info / requirement |
| `../../tests/standard_bot/test_square_submit.py` | 生成到临时目录后跑:桩运行时执行、spec↔code 对应、与回测逐根一致 |

```bash
# 生成到 binance-ai-platform 的 checkout(默认输出 dist/square,已 gitignore)
python scripts/build_square_payloads.py --out-dir <binance-ai-platform>/examples/strategy-case-corpus/three_stage --no-payloads
python scripts/build_square_payloads.py --out-dir <...>/three_stage --check    # 漂移检查
python scripts/build_square_payloads.py --strategy-id st_rsi_reversion=<平台 strategyId>   # 提交用 payload → dist/square_payloads/
```

`st_liquidation_reversal` 不可提交(平台没有清算数据能力),生成器跳过它。
