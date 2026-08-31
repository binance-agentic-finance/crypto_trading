# cyqnt-trd

`cyqnt-trd` is a cryptocurrency trading toolkit centered on the `standard_bot` workflow:

- historical data download to local parquet
- local resample from `1m` into higher timeframes
- Numba-accelerated and Python-native backtesting
- paper signal generation
- monitor / run-manager driven execution flows
- a comprehensive indicator library (traditional TA + TradingView essentials + SMC)
- backward compatibility with the legacy `atomic_strategy_lib` namespace

### The shape of the system

Everything funnels through **one input format and one output format**:

```
data sources ──►  cyqnt.input/v1  ──►  blocks  ──►  cyqnt.signal/v2  ──►  consumers
  klines            (a JSON             (146          kind=trade          backtest
  funding / OI       snapshot of         composable    kind=selection      paper
  order book         one instant)        functions)    kind=alert          live
  news / buzz
  contract meta
```

Two consequences worth knowing before you read further:

**`cyqnt.input/v1` is captured, not streamed.** A snapshot of one decision instant can be
written to a file and replayed. Backtest, paper and live therefore run *the same code* —
only the origin of the snapshot differs. That makes a run byte-for-byte reproducible
(`signal_id` is a `uuid5` of the content, never a clock).

**`cyqnt.signal/v2` has one field a consumer switches on.** `kind` is *derived* from the
payload, not declared by the producer, so it cannot disagree with the fields it labels:

| `kind` | axis | filled with |
|---|---|---|
| `trade` | time — one instrument, bar by bar | `entry` / `exit_plan` / `size` |
| `selection` | cross-section — every instrument, one instant | `candidates` / `universe_size` |
| `alert` | neither — a notification | `advisory_action` / `summary` |

Two recommended paths, depending on which axis you are on:

```
trade      historical parquet → local resample → standard_bot signal → Numba | Python engine
selection  live or frozen cyqnt.input/v1 → universe pipeline → ranked basket
```

---

## Install

### From PyPI

```bash
pip install cyqnt-trd
```

The package supports Python `>=3.8,<3.13`, which includes Binance AI's Python `3.11.2`.

Installing also brings in the `atomic_strategy_lib` shim package so that legacy
case scripts using `from atomic_strategy_lib.X import Y` continue to work without
any code change.

For HTTPS requests, `cyqnt-trd` resolves the CA bundle in this order:

1. `REQUESTS_CA_BUNDLE`
2. `SSL_CERT_FILE`
3. `CURL_CA_BUNDLE`
4. Linux system CA bundle: `/etc/ssl/certs/ca-certificates.crt`
5. `certifi`
6. `requests` default verification behavior

For special Linux deployments where Binance endpoints are signed by an internal CA that exists only in the
system trust store, set the deployment CA bundle via environment variables, for example:

```bash
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

### For development

```bash
git clone https://github.com/nthu-chung/crypto_trading
cd crypto_trading
python3 -m venv .venv-standard-bot
source .venv-standard-bot/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt -r requirements-standard-bot-mvp.txt
```

---

## Recommended Standard Bot Entry Points

### Backtest (Numba engine — fastest)

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_backtest \
  --engine numba \
  --market-type futures \
  --strategy multi_timeframe_ma_spread \
  --symbol BTCUSDT \
  --interval 5m \
  --secondary-interval 1h \
  --primary-ma-period 20 \
  --reference-ma-period 20 \
  --spread-threshold-bps 0 \
  --historical-dir data/mtf_90d \
  --start-ts 1768003200000 \
  --end-ts 1775779200000 \
  --download-missing \
  --output-json docs/backtests/btc_mtf_ma_cross_5m_1h_20_20_90d.json
```

### Backtest (Python engine — for custom block strategies)

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_backtest \
  --engine python \
  --strategy smc_3confluence_v1 \
  --strategy-module cyqnt_trd.strategies.smc_3confluence \
  --symbol BTCUSDT \
  --interval 1h \
  --limit 1000 \
  --market-type futures \
  --initial-capital 10000 \
  --commission-bps 4 \
  --slippage-bps 2
```

### Paper Signal

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_paper \
  --market-type futures \
  --strategy multi_timeframe_ma_spread \
  --symbol BTCUSDT \
  --interval 5m \
  --secondary-interval 1h \
  --primary-ma-period 20 \
  --reference-ma-period 20 \
  --spread-threshold-bps 0 \
  --historical-dir data/mtf_90d \
  --dry-run
```

### Monitor / Background Session

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_monitor_http \
  --broker paper \
  --host 127.0.0.1 \
  --port 8787
```

### Paper Trade Daemon (long-running, for block strategies)

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_paper_daemon \
  --engine python \
  --strategy ma_cross_v1 \
  --strategy-module strategies.ma_cross_v1 \
  --symbol BTCUSDT --interval 1h \
  --market-type futures \
  --state-dir ./watcher/MA_CROSS_V1_BTCUSDT_1h \
  --poll-interval 3570 --warm-up-bars 80 \
  --initial-capital 10000 --fee-bps 4 --slippage-bps 2
```

### Live Trade (via binance-cli)

Live trade requires **two processes** running in parallel:
1. Paper daemon (signal source — identical to paper mode)
2. Live executor (translates paper fills into real binance-cli orders)

```bash
# Terminal 1: Paper daemon (signal source)
python -m cyqnt_trd.standard_bot.entrypoints.mvp_paper_daemon \
  --engine python \
  --strategy ma_cross_v1 \
  --strategy-module strategies.ma_cross_v1 \
  --symbol BTCUSDT --interval 1h \
  --market-type futures \
  --state-dir ./watcher/MA_CROSS_V1_BTCUSDT_1h \
  --poll-interval 3570 --warm-up-bars 80 \
  --initial-capital 10000 --fee-bps 4 --slippage-bps 2

# Terminal 2: Live executor (dry-run first!)
python -m cyqnt_trd.standard_bot.entrypoints.mvp_live_executor \
  --state-dir ./watcher/MA_CROSS_V1_BTCUSDT_1h \
  --symbol BTCUSDT \
  --max-notional 200 \
  --dry-run

# Terminal 2: Live executor (real orders — remove --dry-run)
python -m cyqnt_trd.standard_bot.entrypoints.mvp_live_executor \
  --state-dir ./watcher/MA_CROSS_V1_BTCUSDT_1h \
  --symbol BTCUSDT \
  --max-notional 200
```

Emergency stop: `touch ./watcher/MA_CROSS_V1_BTCUSDT_1h/EMERGENCY_STOP`

See [references/trading-modes.md](references/trading-modes.md) for complete live trade documentation.

### MA Cross Strategy Reference Workspace

A cleaned reference workspace is included under:

`cyqnt_trd/standard_bot/ma_cross_strategy/`

In Binance AI Pro installs, the equivalent path is typically:

`/usr/local/lib/python3.11/dist-packages/cyqnt_trd/standard_bot/ma_cross_strategy/`

This workspace is useful as a concrete example for block-strategy workflows:

- `strategies/ma_cross_v1.py` — SMA 5/20 golden/death cross strategy
- `scripts/run_strategy.py` — unified launcher for backtest, paper trade, and live trade
- `scripts/run_paper_daemon.sh` — paper daemon shell entrypoint
- `scripts/signal_executor.py` — standalone wrapper for `BinanceCliExecutor`
- `scripts/session_watcher.py` — watcher for fills, live executions, and risk-stop checks
- `tests/test_strategy_composition.py` — import/register and signal behavior tests

See `cyqnt_trd/standard_bot/ma_cross_strategy/README.md` for the focused usage notes.

---

## Indicator Library

`cyqnt_trd.blocks` provides a comprehensive pure-pandas indicator library
spanning three families:

### Classical TA (30+ functions)

`cyqnt_trd/blocks/indicators.py`:

- Moving averages: SMA, EMA, WMA, RMA, **TEMA**, **DEMA**, **HMA**, **VWMA**
- Momentum: RSI, MACD, **MFI**, **CCI**, **Williams %R**, StochRSI, **TRIX**, **Awesome Oscillator**, Aroon
- Volatility & Range: ATR, Bollinger Bands, **Keltner Channel**, Donchian, ADX, SuperTrend, Parabolic SAR
- Structure: Ichimoku, Pivot Points (standard floor), **ZigZag**, Heikin Ashi, swing high/low
- Volume: VWAP, OBV, **CMF**, **PVT**, volume MA / Z-score
- Statistical: rolling Z-score, rolling quantile, MA direction, MA alignment

### Smart Money Concepts (SMC)

`cyqnt_trd/blocks/smc_structure.py` and `cyqnt_trd/blocks/smc_liquidity.py`:

- `fractal_pivot_high(df, lookback=5)` / `fractal_pivot_low(df, lookback=5)`
- `fair_value_gap(df)` — bullish/bearish FVG detection
- `order_block_detect(df, swing_lookback=5)` — institutional buy/sell zones
- `bos_choch_detect(df, swing_lookback=5)` — Break of Structure / Change of Character with trend state machine
- `liquidity_sweep_detect(df, swing_lookback=5)` — stop-hunt detection
- `equal_highs_lows(df, swing_lookback=5, tolerance_pct=0.1)` — EQH / EQL clusters
- `premium_discount_zone(df, swing_lookback=5)` — Premium / Discount / Equilibrium classifier

### Risk & Sizing

`cyqnt_trd/blocks/`:

- `verdicts.py` — 5 scoring combinators + 8 gates (`hard_gate`, `verdict_classify`, `cross_validate`, ...)
- `sizing.py` — Kelly, fixed-dollar-loss, fixed risk %, ATR-inverse, leverage cap
- `stop_loss.py` — 4 stop helpers
- `limits.py` — 7 risk limits (liquidation, max-positions, max-exposure, daily-loss, price-deviation, circuit-breaker, funding-window)
- `exit.py` — graduated take-profit, ATR trailing

---

## Block-based Strategies

Strategies use a simple `make_signals(df) → (long_signal, short_signal)` interface:

```python
import pandas as pd
from cyqnt_trd.blocks import strategy
from cyqnt_trd.blocks.smc_structure import bos_choch_detect
from cyqnt_trd.blocks.smc_liquidity import (
    liquidity_sweep_detect,
    premium_discount_zone,
)


def make_signals(df: pd.DataFrame):
    bos = bos_choch_detect(df, swing_lookback=5)
    sweep = liquidity_sweep_detect(df, swing_lookback=5)
    pdz = premium_discount_zone(df, swing_lookback=5)

    long = (
        (sweep["sweep_direction"].rolling(10).apply(lambda x: ("BULL" in x.values)).fillna(0).astype(bool))
        & (bos["trend_state"] != "DOWN")
        & (pdz["current_zone"] == "DISCOUNT")
    )
    short = (
        (sweep["sweep_direction"].rolling(10).apply(lambda x: ("BEAR" in x.values)).fillna(0).astype(bool))
        & (bos["trend_state"] != "UP")
        & (pdz["current_zone"] == "PREMIUM")
    )
    return long, short


strategy.register("my_smc_strategy_v1", make_signals)
```

Then run:

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_backtest \
  --engine python \
  --strategy my_smc_strategy_v1 \
  --strategy-module path.to.my_strategy \
  --symbol BTCUSDT --interval 1h --limit 1000
```

Reference strategies in `cyqnt_trd/strategies/`:

- `smc_3confluence.py` — relaxed SMC with sweep + structure + zone confluence
- `smc_5confluence.py` — strict SMC requiring all five SMC components to align
- `mega_indicator_smoke.py` — smoke test exercising every new indicator
- `channel_breakout.py`, `ema_rsi_cross.py`, ... — traditional TA strategies

---

## YAML Strategy Pipeline

`cyqnt_trd.standard_bot.yaml_pipeline` lets a strategy be described as a
declarative **YAML spec** (data source, indicators, nested entry rules, sizing,
risk/exit) that composes `cyqnt_trd.blocks.*` primitives — no per-strategy
Python needed. The same spec runs across backtest / paper / live.

```bash
# validate: static checks + a synthetic-data dry-run that catches unknown
# blocks, wrong arg counts, and bad params BEFORE any real run
python -m cyqnt_trd.standard_bot.yaml_pipeline validate strategy.yaml

# run: backtest (default), or paper / live per run.mode
python -m cyqnt_trd.standard_bot.yaml_pipeline run strategy.yaml \
  [--input-json klines.json] [--engine vectorized|event] [--start]
```

A spec compiles to a `make_signals(df) -> (long, short)` block strategy via
`cyqnt_trd.blocks.strategy.register(...)`, so it rides the same run path as
hand-written block strategies. Entry rules nest arbitrarily
(`all_of` / `any_of` / `not`); long-only is expressed by simply omitting
`entry.short`. See `docs/strategy_yaml_spec/` for the schema, runnable
examples, and a natural-language → YAML → backtest demo.

## Selection: screening the whole cross-section

The `trade` axis walks time for one instrument. The `selection` axis walks *instruments* at
one instant. Same expression language, same output contract, different axis — a column is a
column, so `conditions.value_above` does not care which.

```bash
# capture a snapshot for THIS spec (it resolves only the sections the spec needs)
python -m cyqnt_trd.standard_bot.entrypoints.mvp_input_bundle \
    --strategy-yaml docs/strategy_yaml_spec/example_open_interest_screen.yaml \
    --out bundle.json

# replay it offline — no network, byte-for-byte reproducible
python -m cyqnt_trd.standard_bot.yaml_pipeline run <spec>.yaml --input-json bundle.json

# or omit --input-json to fetch live
python -m cyqnt_trd.standard_bot.yaml_pipeline run <spec>.yaml
```

A `selection:` section is an ordered pipeline; each step takes the previous step's frame:

```yaml
selection:
  universe:
    - { block: universe.augment_with_contract_meta, with: [contract_meta] }
    - { block: universe.filter_crypto_only }
    - { block: universe.filter_quote_volume, params: { min_quote_volume: 2.0e+7 } }
    - { block: universe.augment_with_indicator, with: [universe_bars],
        params: { indicator: rsi, timeframe: "1h", period: 14, as: rsi14 } }
  score: rsi14
  order: asc          # asc | desc
  max_score: 45       # absolute ceiling, never flipped by `order`
  top_k: 5
  dedupe_by: base_asset
  long_when: { cond: conditions.value_below, args: [rsi14, 45] }
```

**32 `universe.*` blocks**, and the naming is a contract three other places parse:

| prefix | guarantee | examples |
|---|---|---|
| `augment_with_*` (9) | joins columns, **same rows** | `contract_meta`, `funding`, `open_interest`, `oi_change`, `long_short_ratio`, `spread`, `market_cap`, `news`, `indicator` |
| `filter_*` (14) | drops rows, **same columns** | `crypto_only`, `underlying_type`, `sub_type`, `quote_volume`, `quote_suffix`, `spread`, `top_of_book`, `open_interest`, `oi_change`, `long_short_ratio`, `market_cap`, `funding_rate`, `change_pct`, `sentiment` |
| `top_*` / `only_*` / `exclude_*` | drops rows | `top_gainers`, `top_losers`, `top_market_cap`, `top_mentioned`, `only_symbols`, `exclude_symbols` |

### Step order is semantics, not style

Five of those sources have **no whole-market endpoint** — one HTTP request per surviving
symbol. So they must come *after* something that narrows:

```
augment_with_open_interest   augment_with_oi_change
augment_with_long_short_ratio   augment_with_indicator
augment_with_market_cap
```

Measured: narrow to 41 names first → 123 requests. The other way round, 727 × 3 = 2181
requests, which blows the rate budget. `validate` checks this ordering **statically** — it is
the one check that looks at a step's position relative to others, because the coverage
arithmetic downstream only catches one of the five, and only by coincidence.

`augment_with_market_cap` adds a second reason to narrow first: all 177 non-COIN
perpetuals answer a circulating supply of 0, so fanning out over them can only ever
return 177 unknowns. Put `filter_crypto_only` ahead of it.

### Declaring what the spec guessed

Some requests have two readable meanings and no block can settle them — "volume over ten
million" is either 1e7 of turnover or 1e7 coins, and the two select different names. The
product default is to declare rather than block:

```yaml
strategy:
  assumptions:
    - cid: c3
      reading: "read as 24h quote turnover, 1e7 USD"
      alternatives: ["1e7 coins of base volume"]
      basis: "quoteVolume is the only turnover column on the frame"
```

A YAML comment does **not** count — comments are dropped at load, so the person holding the
basket never sees them. Declared here, each one rides out in the signal's `warnings` and adds
`assumption_declared` to `reason_codes`. The key set is closed; a misspelt key is an error
rather than an ignored field, because a choice recorded under the wrong key is a silent
choice again.

### Tokenised equities are real products

`NVDAUSDT`, `TSLAUSDT`, `AAPLUSDT`, `QQQUSDT` and ~130 others are live USDⓈ-M perpetuals
(`underlyingType=EQUITY`, `contractType=TRADIFI_PERPETUAL`). So "exclude US stocks" and
"screen US stocks" are *both* real requests, and neither is the default:

```yaml
- { block: universe.filter_crypto_only }                                # exclude them
- { block: universe.filter_underlying_type, params: { include: [EQUITY] } }  # screen them
```

They are not index futures, though. `QQQUSDT` carries funding (~+1.9% annualised measured),
trades at a premium to its index, and **keeps moving while the US market is shut** — so a
threshold written for NQ does not transfer.

## Backtest Engines

Block (Python-engine) strategies can be backtested by two interchangeable engines,
both **long + short** and sharing the same execution model (signal at bar close →
fill next bar open, single position, side-aware stop/TP):

| Engine | Invoked by | Notes |
|---|---|---|
| `SnapshotBacktestRunner` | `mvp_backtest --engine python` (event-driven, bar-by-bar) | mature reference; mirrors the paper/live execution model |
| `run_vectorized_backtest` | `yaml_pipeline run` default (signals vectorized once + numpy exit loop) | 10–100× faster; used by the YAML pipeline and parameter sweeps |

The two are cross-checked and agree closely on matched configs (long-only is
byte-stable between them; long + short agrees on most exit types). Paper/live
signals run through `PythonLivePaperSession`, which is also long + short.
`--engine numba` (`NumbaBacktestRunner`) is a separate compiled path for the
built-in kernels only, not arbitrary block strategies.

## Atomic Compatibility (`atomic_strategy_lib` shim)

The package ships an `atomic_strategy_lib` shim that re-exports cyqnt_trd
implementations under the legacy atomic namespace. This means existing case
scripts using:

```python
from atomic_strategy_lib.scoring.gates import verdict_with_gate
from atomic_strategy_lib.signals.momentum import rsi_compute
from atomic_strategy_lib.decision.sizing import fixed_dollar_loss
```

work without modification when only `cyqnt-trd` is installed — no
`PYTHONPATH` or `ATOMIC_STRATEGY_LIB_PATH` setup needed.

For verbatim numerical parity with the original atomic library, the shim
delegates to `cyqnt_trd/compat/atomic_signals/` which ports atomic's
pure-Python algorithms (RSI, EMA, MACD, ATR, Bollinger, StochRSI, SuperTrend,
ADX) verbatim. All 12 measured indicator outputs match atomic to within
`1e-9` precision.

See `docs/atomic-compat/MIGRATION_HANDOFF.md` for the full integration design.

---

## Data Workflow

The preferred data workflow is:

1. Download Binance K bars into local parquet
2. Store the finest useful granularity, usually `1m`
3. Resample locally into `5m`, `15m`, `1h`, etc.
4. Run `standard_bot` on local parquet instead of using raw API responses as final backtest input

This keeps:

- point-in-time alignment clearer
- local backtests repeatable
- paper signal and backtest logic consistent

---

## Test Fixtures

`tests/blocks/fixtures/` contains four real Binance OHLCV parquet snapshots
used for indicator integration tests:

| Fixture | Period | Use case |
|---|---|---|
| `BTCUSDT_1h_500bars.parquet` | ~21 days | Primary indicator test |
| `ETHUSDT_1h_500bars.parquet` | ~21 days | Cross-symbol verification |
| `BTCUSDT_4h_300bars.parquet` | ~50 days | High-TF SMC structure |
| `BTCUSDT_15m_500bars.parquet` | ~5 days | Low-TF noise sanity |

These are loaded directly by the test suite (see
`tests/blocks/test_smc.py`, `tests/blocks/test_tradingview_indicators.py`).

---

## Strategy Families on the `standard_bot` Numba mainline

The `standard_bot` mainline currently includes Numba-backed support for:

- `moving_average_cross`
- `price_moving_average`
- `rsi_reversion`
- `multi_timeframe_ma_spread`
- `donchian_breakout`

For block-based / SMC strategies, use `--engine python` (see [Block-based
Strategies](#block-based-strategies) above).

---

## Verification & Quality

- **Test suite**: 414 tests pass, 1 skipped, no regressions
- **Lookahead safety**: all 25/29 indicators verified lookahead-safe via
  `indicator(df[:i+1])[-1] == indicator(df)[i]` test
- **Atomic numerical parity**: 12/12 indicator outputs match atomic source
  to `1e-9` precision
- **Real-data smoke**: every indicator validated on 4 binance fixtures
  spanning BTC/ETH × 1h/4h/15m

---

## Package Notes

- The preferred historical backtest engine is `NumbaBacktestRunner` for compiled performance
- The preferred Python-block engine is `--engine python` for custom strategies using the `cyqnt_trd.blocks.*` library
- The preferred CLI entrypoint is `cyqnt_trd.standard_bot.entrypoints.mvp_backtest`
- Legacy `cyqnt_trd/backtesting/*` still exists for compatibility, but is not the recommended path for new work
- Legacy `atomic_strategy_lib` imports are supported via the bundled shim package

---

## Documentation

- `docs/CHANGELOG.md` — date-indexed log of integration & feature work
- `docs/atomic-compat/MIGRATION_HANDOFF.md` — atomic → cyqnt_trd integration design
- `docs/atomic-compat/README.md` — quick reference for the shim package
- `docs/cyqnt_trd_0_1_9_dev0_tutorial.md` — earlier tutorial

---

## Requirements

Key dependencies (bounded ranges; lower bound aligned with deployment
baselines, upper bound prevents breaking major-version upgrades):

- `pandas>=2.0.0,<3.0`
- `numpy>=1.24.0,<2.0`
- `polars>=1.0.0,<2.0`
- `numba>=0.60.0,<0.70`
- `pyarrow>=14.0.0,<25.0`
- `scipy>=1.10.0,<2.0`
- `matplotlib>=3.7.0,<4.0`
- `requests>=2.32.0,<3.0`
- `websockets>=15.0.1,<16.0`

Binance SDK dependencies:

- `binance-sdk-spot>=8.2.1,<10.0`
- `binance-sdk-derivatives-trading-usds-futures>=10.0.1,<11.0`
- `binance-sdk-algo>=2.6.0,<3.0`
- `binance-common>=3.8.0,<4.0`

The upper bounds are deliberate: pip will not auto-upgrade a deployment
that already has e.g. `numpy 1.24.4` or `binance-common 3.8.0`, so
installing `cyqnt-trd` does not break neighbouring services that rely
on those exact ABIs. The `websockets` upper bound at `<16.0` matches
the binance-SDK family's own constraint.

---

## License

MIT License
