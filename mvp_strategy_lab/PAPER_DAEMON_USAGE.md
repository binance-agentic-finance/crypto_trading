# Paper Trade Daemon — Usage Guide

The paper-trade daemon supports **two execution engines**:

| Engine | When to use | Strategy interface |
|--------|-------------|--------------------|
| `--engine numba` (default) | Production / max performance | `@njit` kernel registered via `NumbaBacktestRunner.register_kernel()` |
| `--engine python` | Block strategies / rapid iteration | `def make_signals(df)` registered via `cyqnt_trd.blocks.strategy.register()` |

Both engines share the same daemon, the same `state.json` shape, the same
next-bar-open execution model, and the same crash-safe checkpoint format.
They differ only in *how the signal is computed at each tick*.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  mvp_paper_daemon.py (long-running, OpenClaw workspace)     │
│  ├─ Live paper session (in-memory state)                    │
│  │   ├─ NumbaLivePaperSession  (--engine numba)             │
│  │   └─ PythonLivePaperSession (--engine python)            │
│  ├─ Polls Binance REST for new bars every interval          │
│  ├─ Runs SAME signal logic as backtest on full bar history  │
│  ├─ Executes paper fills (signal@bar[i] → fill@bar[i+1])    │
│  └─ Writes state.json atomically                            │
└─────────────────────────────────────────────────────────────┘
                    ↕ state.json (READ-ONLY for watcher)
┌─────────────────────────────────────────────────────────────┐
│  watcher_template.py                                        │
│  └─ Reads state.json → risk check → jarvis notifications    │
│     (NO subprocess spawn, NO re-run, NO restart)            │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Guarantees

1. **Signal Consistency** — same logic + same bar data → identical signals to backtest (works for both engines).
2. **Persistent State** — daemon holds bar history + position in memory, never restarts.
3. **Watcher is Read-Only** — watcher reads state.json without mutating daemon state.
4. **Atomic Writes** — state.json is written via tmp+rename, safe for concurrent reads.
5. **Crash-Safe Resume** — daemon restart re-imports the strategy module and resumes from `session_checkpoint.json`.

---

## Path A: Python (blocks) Strategies

Use this path for strategies generated from natural-language conversations or
hand-written using the `cyqnt_trd.blocks` library.

### 1. Author the strategy

`my_strategy.py`:

```python
from cyqnt_trd.blocks import indicators as ind, conditions as cond, entry, strategy

def make_signals(df):
    """df has columns: open, high, low, close, volume, close_time."""
    ma_fast = ind.sma(df["close"], 20)
    ma_slow = ind.sma(df["close"], 60)
    rsi14 = ind.rsi(df["close"], 14)

    long_signal = entry.all_of([
        cond.ma_cross_above(ma_fast, ma_slow),
        cond.rsi_in_range(rsi14, 40, 60),
    ])
    short_signal = cond.ma_cross_below(ma_fast, ma_slow)
    return long_signal, short_signal


# IMPORTANT — strategy must register itself at import time
strategy.register("my_ma_rsi_strategy", make_signals)
```

### 2. Validate with backtest first (recommended)

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_backtest \
    --engine python \
    --strategy my_ma_rsi_strategy \
    --strategy-module my_strategy \
    --symbol BTCUSDT --interval 1h --limit 500
```

### 3. Start the paper-trade daemon

```bash
mkdir -p ./watcher/BTCUSDT_my_strat
nohup python -m cyqnt_trd.standard_bot.entrypoints.mvp_paper_daemon \
    --engine python \
    --strategy my_ma_rsi_strategy \
    --strategy-module my_strategy \
    --symbol BTCUSDT \
    --interval 1h \
    --state-dir ./watcher/BTCUSDT_my_strat/ \
    --poll-interval 3600 \
    --warm-up-bars 120 \
    --initial-capital 10000 \
    --fee-bps 4 \
    --slippage-bps 2 \
    --market-type futures \
    > ./watcher/BTCUSDT_my_strat/daemon.log 2>&1 &

echo $! > ./watcher/BTCUSDT_my_strat/pid
```

> The strategy module **must be importable** from the daemon's `PYTHONPATH`.
> Either install your strategy as a package or run the daemon with
> `PYTHONPATH=/path/to/strategy_dir python -m ...`.

### 4. Use Python API directly (without the daemon CLI)

```python
import importlib
import time
from cyqnt_trd.standard_bot.simulation import PythonLivePaperSession

# 1. Import the strategy module — this triggers strategy.register(...)
importlib.import_module("my_strategy")

# 2. Create the session
session = PythonLivePaperSession(
    strategy_id="my_ma_rsi_strategy",
    symbol="BTCUSDT",
    config={"timeframe": "1h"},
    initial_capital=10_000.0,
    fee_bps=4.0,
    slippage_bps=2.0,
    market_type="futures",
)

# 3. Warm up with historical bars (no trading)
for bar in historical_bars:
    session.warm_up(bar)

# 4. Tick on each new confirmed bar
for bar in incoming_bars:
    fill = session.tick(bar)
    if fill is not None:
        print(f"FILL: {fill.action} {fill.side} {fill.quantity} @ {fill.price}")

# 5. Read state at any time
print(session.state_snapshot())
```

---

## Path B: Numba Kernel Strategies

Use this path for pre-existing or hand-tuned `@njit` strategies.

### 1. Start the daemon

```bash
nohup python -m cyqnt_trd.standard_bot.entrypoints.mvp_paper_daemon \
    --engine numba \
    --symbol BTCUSDT \
    --interval 1h \
    --strategy ema_rsi_cross \
    --strategy-module mvp_strategy_lab.external_strategies.ema_rsi_cross \
    --extra-params '{"fast_window":2,"slow_window":5,"rsi_period":3,"rsi_mid":50.0}' \
    --state-dir ./watcher/BTCUSDT_ema_rsi_cross/ \
    --poll-interval 3600 \
    --warm-up-bars 120 \
    --initial-capital 10000 \
    --fee-bps 4 \
    --slippage-bps 2 \
    --market-type futures \
    > ./watcher/BTCUSDT_ema_rsi_cross/daemon.log 2>&1 &

echo $! > ./watcher/BTCUSDT_ema_rsi_cross/pid
```

(`--engine numba` is the default and can be omitted.)

### Builtin Numba strategies (no `--strategy-module` needed)

- `moving_average_cross`
- `price_moving_average`
- `rsi_reversion`
- `donchian_breakout`
- `adx_trend_strength`
- `atr_breakout`
- `bollinger_mean_reversion`
- `macd_trend_follow`

---

## Common Operations (both engines)

### Check status (no restart triggered)

```bash
cat ./watcher/<run_id>/state.json | python -m json.tool
```

### Watcher integration

In the watcher's `state.json`, set:
```json
{
  "signal_source": "python_daemon",   // or "numba_daemon"
  "daemon_state_path": "./watcher/<run_id>/state.json"
}
```

The watcher reads state from the daemon instead of spawning `mvp_paper`:

```python
# In watcher's process_session():
sig_source = state.get("signal_source")
if sig_source in ("numba_daemon", "python_daemon"):
    daemon_state = load_state_from_daemon(state)
    # Read signals/fills from daemon_state
    # Only do: risk check, jarvis notify, session lifecycle
```

### Stop the daemon

```bash
# Option A — write status to state.json (graceful)
python -c "
import json, sys
p = sys.argv[1]
s = json.load(open(p))
s['status'] = 'stopped'
json.dump(s, open(p, 'w'))
" ./watcher/<run_id>/state.json

# Option B — SIGTERM (also graceful)
kill $(cat ./watcher/<run_id>/pid)
```

### Daemon restart (resume from checkpoint)

The daemon writes `session_checkpoint.json` after every tick. On restart it
re-imports the strategy module and restores in-memory state automatically:

```bash
# Just start the daemon again with the same args — it will pick up
# session_checkpoint.json and continue
nohup python -m cyqnt_trd.standard_bot.entrypoints.mvp_paper_daemon \
    --engine python \
    --strategy my_ma_rsi_strategy \
    --strategy-module my_strategy \
    --symbol BTCUSDT --interval 1h \
    --state-dir ./watcher/BTCUSDT_my_strat/ \
    >> ./watcher/BTCUSDT_my_strat/daemon.log 2>&1 &
```

---

## state.json Format

```json
{
  "run_id": "BTCUSDT_my_ma_rsi_strategy_a1b2c3d4",
  "symbol": "BTCUSDT",
  "market_type": "futures",
  "mode": "paper",
  "signal_source": "python_daemon",
  "strategy": "my_ma_rsi_strategy",
  "params": {"timeframe": "1h"},
  "interval": "1h",
  "initial_capital": 10000.0,
  "session_start_equity": 10000.0,
  "current_equity": 10045.20,
  "stop_equity": 9000.0,
  "pnl_usd": 45.20,
  "pnl": 0.00452,
  "position": {"side": "long", "entry_price": 67500.0, "qty": 0.148},
  "latest_signal": {
    "signal_id": "...",
    "side": "long",
    "strength": 1.0,
    "source": "blocks_python",
    "pending": false
  },
  "has_pending_order": false,
  "status": "running",
  "trade_log": [
    {
      "fill_id": "...",
      "timestamp_ms": 1700560800000,
      "side": "buy",
      "price": 67500.0,
      "quantity": 0.148,
      "fee": 0.27,
      "action": "open_long",
      "signal_bar_timestamp_ms": 1700557200000
    }
  ],
  "trade_count": 3,
  "bar_count": 156,
  "tick_count": 156,
  "last_bar_timestamp": 1700560800000,
  "current_price": 67800.0,
  "poll_interval_sec": 3600,
  "poll_count": 36,
  "daemon_pid": 12345,
  "last_update_ts": 1700564400,
  "last_update_at": "2026-05-13T15:00:00Z"
}
```

The `signal_source` field tells the watcher which engine produced the
state — `python_daemon` for Python (blocks) sessions, `numba_daemon` for
Numba sessions.

---

## End-to-End Flow Example (Python engine)

```
User says "make me an EMA cross strategy with RSI filter"
   ↓
LLM (Claude / OpenClaw) generates my_strategy.py using cyqnt_trd.blocks
   ↓
mvp_backtest --engine python   → validate logic on historical data
   ↓
mvp_paper_daemon --engine python   → start paper trading
   ↓
state.json updates each poll → watcher reads → risk check → jarvis notify
   ↓
After N days: review state.json + trade_log.jsonl → promote to live
```

---

## Signal Consistency Guarantee

```
Backtest (--engine python):        Daemon (--engine python):
┌──────────────────────┐           ┌──────────────────────┐
│ Full bar DataFrame   │           │ Growing bar list      │
│ [bar0..barN]         │           │ [bar0..barN]          │
└──────────┬───────────┘           └──────────┬───────────┘
           │                                  │
      ┌────▼────┐                        ┌────▼────┐
      │ blocks  │ ← SAME signal_fn(df) → │ blocks  │
      │ pipeline│                        │ pipeline│
      └────┬────┘                        └────┬────┘
           │                                  │
      long_s/short_s                     long_s/short_s
      (IDENTICAL on common rows)
```

For Python (blocks) engine: the daemon re-runs the user's `signal_fn(df)`
on the full accumulated bar series each tick. Since `signal_fn` is a pure
function of the DataFrame, identical inputs → identical outputs. No
approximation or state-based estimation.

For Numba engine: same guarantee, just with a `@njit` kernel and numpy
arrays instead of pandas DataFrame.

---

## Troubleshooting

**"strategy 'foo' is not registered"**
You forgot to either (a) pass `--strategy-module` so the daemon imports
your module, or (b) ensure the module's `strategy.register(...)` actually
runs at import time. Confirm by importing the module manually:
```bash
python -c "import my_strategy; from cyqnt_trd.blocks.strategy import _PENDING_REGISTRATIONS; print(_PENDING_REGISTRATIONS)"
```

**Strategy runs in backtest but daemon shows no fills**
This is usually because your strategy's entry conditions are strict enough
that real-time market data hasn't satisfied them yet during the polling
window. Check `state.json["latest_signal"]` and the daemon log to see if
signals are being computed at all. You can also reduce `--warm-up-bars`
or pick a more volatile market to verify.

**Daemon crashes mid-poll with no obvious reason**
Check `daemon.log`. The most common cause is a custom indicator inside
`make_signals` raising on edge cases (e.g. division by zero on a flat
bar). Wrap risky logic with `try/except` or use the safer block helpers.

**How do I know which engine is running?**
`state.json["signal_source"]` will be either `"python_daemon"` or
`"numba_daemon"`. The daemon log on startup also prints
`"using Python engine (cyqnt_trd.blocks)"` or `"using Numba engine"`.
