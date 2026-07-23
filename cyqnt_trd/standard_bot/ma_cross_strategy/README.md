# MA Cross Strategy Workspace

This folder is a cleaned copy of the Docker runtime workspace:

`/home/node/.openclaw/workspace/ma-cross-strategy`

It is intended for Binance AI Pro / OpenClaw workflows that need a known-good
strategy workspace for:

- backtest
- paper trade
- live trade through `binance-cli`
- paper/live signal consistency checks

The strategy uses `cyqnt_trd` as the execution framework. The files here are
examples and launchers; the core framework remains under
`cyqnt_trd.standard_bot`.

## Included Files

| Path | Purpose |
|---|---|
| `strategies/ma_cross_v1.py` | SMA 5/20 golden/death cross strategy used as the signal source |
| `strategies/ma_cross_validation_fast.py` | Faster validation strategy for higher-frequency test fills |
| `strategies/bar_direction_validation.py` | Deterministic validation strategy for pipeline tests |
| `scripts/run_strategy.py` | Unified launcher for backtest, paper, and live modes |
| `scripts/run_paper_daemon.sh` | Shell launcher for the paper daemon |
| `scripts/signal_executor.py` | Standalone wrapper around `BinanceCliExecutor` |
| `scripts/session_watcher.py` | File-based watcher for fills, live executions, and risk stop checks |
| `tests/test_strategy_composition.py` | Strategy registration and signal behavior tests |

Not copied from the Docker workspace:

- historical watcher run outputs (`watcher/*`)
- `.pytest_cache` and `__pycache__`
- generated PDFs
- temporary OpenClaw task files
- `setup_env.py`, because Binance AI Pro / OpenClaw should use the active
  environment and should not mutate `cyqnt_trd` package contents

## Quick Start

In Binance AI Pro, `cyqnt_trd` is installed globally here:

```bash
CYQNT_TRD_DIR=/usr/local/lib/python3.11/dist-packages/cyqnt_trd
cd "$CYQNT_TRD_DIR/standard_bot/ma_cross_strategy"
export PYTHONPATH="$PWD:/usr/local/lib/python3.11/dist-packages:${PYTHONPATH:-}"
```

For local development from this repository:

```bash
cd /Users/hankchung/Dev/crypto_trading-main/cyqnt_trd/standard_bot/ma_cross_strategy
export PYTHONPATH="$PWD:/Users/hankchung/Dev/crypto_trading-main:${PYTHONPATH:-}"
```

## Backtest

```bash
python scripts/run_strategy.py \
  --mode backtest \
  --symbol ETHUSDT \
  --interval 1m \
  --market-type futures \
  --limit 2000 \
  --tail-bars 120
```

Equivalent framework entrypoint:

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_backtest \
  --engine python \
  --strategy ma_cross_v1 \
  --strategy-module strategies.ma_cross_v1 \
  --symbol ETHUSDT \
  --interval 1m \
  --market-type futures \
  --allow-remote-api \
  --limit 2000 \
  --initial-capital 10000 \
  --commission-bps 4 \
  --slippage-bps 2 \
  --execution-model next_bar_open \
  --tail-bars 120
```

## Paper Trade

```bash
python scripts/run_strategy.py \
  --mode paper \
  --symbol ETHUSDT \
  --interval 1m \
  --market-type futures \
  --state-dir ./watcher/MA_CROSS_V1_ETHUSDT_1m \
  --poll-interval 55 \
  --warm-up-bars 80
```

The paper daemon writes:

- `state.json`
- `trades.jsonl`
- `events.jsonl`
- `session_checkpoint.json`
- `daemon.log`

## Live Trade

Live mode starts two processes:

1. paper daemon, which generates the canonical signal/fill stream
2. live executor, which reads `trades.jsonl` and reconciles Binance futures
   position through `binance-cli`

Always test with `--dry-run` first:

```bash
python scripts/run_strategy.py \
  --mode live \
  --symbol ETHUSDT \
  --interval 1m \
  --market-type futures \
  --state-dir ./watcher/MA_CROSS_V1_ETHUSDT_1m \
  --max-notional 100 \
  --notional-fraction 0.95 \
  --dry-run
```

When the dry-run output is correct and `binance-cli` is using the intended
futures demo profile, remove `--dry-run`:

```bash
python scripts/run_strategy.py \
  --mode live \
  --symbol ETHUSDT \
  --interval 1m \
  --market-type futures \
  --state-dir ./watcher/MA_CROSS_V1_ETHUSDT_1m \
  --max-notional 100 \
  --notional-fraction 0.95
```

Emergency stop:

```bash
touch ./watcher/MA_CROSS_V1_ETHUSDT_1m/EMERGENCY_STOP
```

## Watcher

The watcher is optional but useful when Binance AI Pro needs a lightweight
runtime monitor:

```bash
python scripts/session_watcher.py \
  --state-dir ./watcher/MA_CROSS_V1_ETHUSDT_1m \
  --run-id MA_CROSS_V1_ETHUSDT_1m \
  --poll-interval 15 \
  --max-drawdown-pct 20
```

It reads only local files and can request a risk stop by updating `state.json`.

## Binance AI Pro Notes

- Use `strategies.ma_cross_v1` as the default `--strategy-module`.
- Do not modify files under the installed `cyqnt_trd` package during a run.
- Use `run_strategy.py` for the normal user-facing workflow.
- Use `mvp_backtest`, `mvp_paper_daemon`, and `mvp_live_executor` directly when
  an agent needs lower-level control.
- For live validation, compare `trades.jsonl` against `executions.jsonl`.
  Paper flips normally produce two live orders: close current position, then
  open the desired position.
- Timestamp / `recvWindow` errors from Binance are transient infrastructure
  errors. The live executor should retry and reconcile actual position back to
  the desired paper position.

## Verify

```bash
pytest tests/test_strategy_composition.py
```
