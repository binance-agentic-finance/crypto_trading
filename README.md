# cyqnt-trd

`cyqnt-trd` is a cryptocurrency trading toolkit centered on the `standard_bot` workflow:

- historical data download to local parquet
- local resample from `1m` into higher timeframes
- Numba-accelerated and Python-native backtesting
- paper signal generation
- monitor / run-manager driven execution flows
- a comprehensive indicator library (traditional TA + TradingView essentials + SMC)
- backward compatibility with the legacy `atomic_strategy_lib` namespace

The current recommended path is:

`historical parquet → local resample → standard_bot signal → NumbaBacktestRunner | PythonEngine`

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
