# cyqnt_trd.compat — Migration Guide

This module provides a **drop-in replacement** for `atomic_strategy_lib.core.types`
so that existing strategy code can migrate to `cyqnt_trd` without any functional
changes.

---

## 1. Import path migration

### Before (atomic_strategy_lib)
```python
from atomic_strategy_lib.core.types import (
    Candle, Signal, Score, Verdict,
    TradePlan, OrderResult, Position,
    Ticker, FundingRate, OpenInterest,
    VelocityMetrics,
)
```

### After (cyqnt_trd)
```python
from cyqnt_trd.compat import (
    Candle, Signal, Score, Verdict,
    TradePlan, OrderResult, Position,
    Ticker, FundingRate, OpenInterest,
    VelocityMetrics,
)
```

Or from the top-level package:
```python
from cyqnt_trd import (
    Candle, Signal, Score, Verdict, TradePlan, OrderResult,
)
```

---

## 2. All exported types

| Type | Layer | Notes |
|------|-------|-------|
| `Candle` | L1 Data | OHLCV bar; `timestamp` is unix ms |
| `Ticker` | L1 Data | 24-h market stats |
| `FundingRate` | L1 Data | Futures funding rate |
| `OpenInterest` | L1 Data | OI snapshot |
| `OIHistoryPoint` | L1 Data | OI history entry |
| `OrderBookLevel` | L1 Data | Single price level |
| `OrderBook` | L1 Data | Full order book snapshot |
| `Balance` | L1 Data | Account balance per asset |
| `Position` | L1 Data | Open futures position |
| `ExchangeFilter` | L1 Data | Symbol trading constraints |
| `Signal` | L2-L3 | Named directional signal |
| `Score` | L2-L3 | Multi-factor score with reasons |
| `Verdict` | L2-L3 | String-constant verdict (see note) |
| `TradePlan` | L4 Decision | Full trade plan with entry/stop/sizing |
| `OrderResult` | L5 Execution | Order submission result |
| `VelocityMetrics` | L2 | Volume/price velocity stats |

**Note on `Verdict`**: Atomic uses a plain class (not an Enum) with string
constants and a `RANK` dict.  This is preserved exactly:
```python
plan.verdict == Verdict.CANDIDATE          # True
Verdict.RANK[plan.verdict]                 # 1
Verdict.is_valid("STRONG_CANDIDATE")       # True
Verdict.ALL                                # ['STRONG_CANDIDATE', ...]
```

---

## 3. Usage — same fields, same code

All field names, types, and defaults are **identical** to atomic, so existing
code works unchanged:

```python
from cyqnt_trd.compat import Candle, Signal, Verdict, TradePlan

# Create a Candle (same as atomic)
c = Candle(
    timestamp=1_700_000_000_000,
    open=40_000.0, high=41_000.0,
    low=39_500.0, close=40_500.0,
    volume=1200.0,
)

# Signal (same as atomic)
s = Signal(name="rsi_oversold", value=28.5, direction="BULLISH", strength=0.9)

# Verdict comparison (same as atomic)
plan = TradePlan(
    symbol="BTCUSDT", direction="LONG",
    verdict=Verdict.CANDIDATE, score=0.72,
    entry_price=40_500.0, stop_price=39_000.0,
    stop_pct=0.037, leverage=5.0,
    notional=500.0, max_loss=18.5,
)
if plan.verdict in (Verdict.STRONG_CANDIDATE, Verdict.CANDIDATE):
    print("Good trade candidate")
```

---

## 4. New: DataFrame adapters

New code can use pandas DataFrames and convert on demand:

```python
import pandas as pd
from cyqnt_trd.compat import df_to_candles, candles_to_df

# DataFrame → List[Candle]
df = pd.read_parquet("btc_15m.parquet")
candles = df_to_candles(df)   # expects columns: open, high, low, close, volume

# List[Candle] → DataFrame
df2 = candles_to_df(candles)
assert list(df2.columns) == ["timestamp", "open", "high", "low", "close",
                              "volume", "quote_volume", "trades"]
```

Signal series conversion:

```python
from cyqnt_trd.compat import signal_series_to_signals, signals_to_series

# Boolean Series → List[Signal]
rsi_oversold: pd.Series = rsi < 30
signals = signal_series_to_signals(rsi_oversold, "rsi_oversold", direction="BULLISH")

# List[Signal] → Boolean Series (aligned to df.index)
series = signals_to_series(signals, df.index)
```

TradePlan serialization (for `state.json` persistence):

```python
from cyqnt_trd.compat import trade_plan_to_dict, dict_to_trade_plan

# Serialize
d = trade_plan_to_dict(plan)   # JSON-safe dict
import json; json.dumps(d)     # works

# Deserialize
plan2 = dict_to_trade_plan(d)
assert plan2.symbol == plan.symbol
```

---

## 5. Helper functions

```python
from cyqnt_trd.compat import extract_closes

closes = extract_closes(candles)   # List[float] of close prices
# or if already floats, returns as-is
```

---

## 6. Candle / Bar disambiguation

`cyqnt_trd` has its own `Bar` type in `cyqnt_trd.standard_bot.core.contracts`
for the internal signal pipeline.  **`Candle` and `Bar` are different types**:

| Feature | `Candle` (compat) | `Bar` (contracts) |
|---------|-------------------|--------------------|
| Purpose | Atomic strategy lib API | Internal signal pipeline |
| Required fields | timestamp, OHLCV | OHLCV + instrument_id + timeframe + confirmed |
| Migration | Use as-is | Use `blocks.data.df_to_bars()` |

For new code, prefer `DataFrame + blocks` functions directly.  Use `Candle`
only when interfacing with existing atomic-style strategy code.

---

## 7. Full example — migrated strategy

```python
# my_strategy_migrated.py
from cyqnt_trd.compat import Candle, Signal, Verdict, TradePlan, df_to_candles
from cyqnt_trd.blocks import indicators as ind, conditions as cond

def analyze(df):
    candles = df_to_candles(df)                    # atomic-style objects
    closes = [c.close for c in candles]

    # Or: use blocks directly on the df (recommended for new code)
    rsi = ind.rsi(df["close"], 14)
    signal_long = cond.rsi_oversold(rsi, 30)
    return signal_long
```
