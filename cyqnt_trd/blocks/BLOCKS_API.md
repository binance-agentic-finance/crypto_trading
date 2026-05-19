# `cyqnt_trd.blocks` — Strategy Building Blocks API Reference

> **For OpenClaw / LLM strategy generators**: Read this file in full before
> generating a strategy script. It documents every public function in
> `cyqnt_trd.blocks` along with its signature, return type, and use case.

---

## TL;DR

```python
# Save as my_strategy.py (or any module name).
from cyqnt_trd.blocks import indicators as ind, conditions as cond, entry, strategy

def make_signals(df):
    ma20 = ind.sma(df["close"], 20)
    ma60 = ind.sma(df["close"], 60)
    long = cond.ma_cross_above(ma20, ma60)
    short = cond.ma_cross_below(ma20, ma60)
    return long, short

strategy.register("my_ma_cross", make_signals)
```

Run the backtest:

```bash
python -m cyqnt_trd.standard_bot.entrypoints.mvp_backtest \
  --engine python \
  --strategy my_ma_cross \
  --strategy-module my_strategy \
  --symbol BTCUSDT --interval 15m --limit 500
```

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  User strategy script (.py)                                    │
│    - imports cyqnt_trd.blocks                                  │
│    - defines make_signals(df) -> (long, short)                 │
│    - calls strategy.register("name", make_signals)             │
└────────────────────────────────────────────────────────────────┘
                           │
                           │ --strategy-module loads the script
                           ▼
┌────────────────────────────────────────────────────────────────┐
│  cyqnt_trd.blocks (this package)                               │
│    indicators · conditions · entry · exit · risk · sizing      │
│    derivatives · patterns · scoring · regime · microstructure  │
│    universe · execution · data · strategy                      │
└────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│  cyqnt_trd.standard_bot — registry + backtest engine           │
│    SignalPluginRegistry · NumbaBacktestRunner · ...            │
└────────────────────────────────────────────────────────────────┘
```

The blocks package is a **strategy-script library**: it never owns a
backtest run. Strategy authors compose blocks into a `make_signals(df)`
function and pass it to `strategy.register(...)`, after which the
existing `mvp_backtest` entrypoint discovers it via the standard
`SignalPluginRegistry`.

---

## The `make_signals(df)` contract

The signal function:

* receives a single `pandas.DataFrame` with these columns
  (lower-case): `open, high, low, close, volume, quote_volume,
  open_time, close_time, trades, instrument_id, timeframe`. Column
  `timestamp` is a convenience alias for `close_time` (both work).
  Timestamps are **integer milliseconds** (Binance convention).
* returns either:
  * `(long_signal, short_signal)` — both `pd.Series[bool]` aligned to
    `df.index`. Use `None` for the short series to declare a long-only
    strategy: `return long_signal, None`.
  * `long_signal` (single Series) — interpreted as long-only.

The function is called once per bar in batch mode (`run`) and on the
new tail in incremental mode (`step`); in both cases the **full
DataFrame** is passed so rolling/EMA-based indicators always have
enough history.

---

## Module index

| Module | Lines of API | Purpose |
|---|---|---|
| [`indicators`](#module-indicators) | 27 | Numerical technical indicators |
| [`patterns`](#module-patterns) | 30 | Candlestick pattern detectors |
| [`derivatives`](#module-derivatives) | 11 | OI, funding, long/short, CVD, liquidation, basis |
| [`conditions`](#module-conditions) | 37 | Atomic boolean entry conditions |
| [`entry`](#module-entry) | 7 | Combinators (all_of / any_of / score_entry / regime_switch) |
| [`exit`](#module-exit) | 19 | Stop-loss / take-profit / trailing / partial-close rules |
| [`risk`](#module-risk) | 4 | Portfolio-level risk control state machine |
| [`sizing`](#module-sizing) | 8 | Position-sizing calculators |
| [`scoring`](#module-scoring) | 2 | Multi-factor scoring system |
| [`regime`](#module-regime) | 4 | Market regime classifiers |
| [`microstructure`](#module-microstructure) | 5 | Whale / order-flow heuristics |
| [`universe`](#module-universe) | 9 | Symbol-pool filters & scanners |
| [`execution`](#module-execution) | 6 | Order specification helpers |
| [`data`](#module-data) | 11 | DataFrame⇄Bar conversion + Binance public REST |
| [`strategy`](#module-strategy) | 3 | Register a strategy as a SignalPlugin |
| **TOTAL** | **183** | |

Top-level convenience re-exports (no need to navigate sub-modules):

```python
from cyqnt_trd.blocks import (
    sma, ema, rsi, macd, atr, adx, bollinger, donchian, stochastic,
    vwap, obv, volume_ma, swing_high, swing_low,
    all_of, any_of, score_entry, weighted_score, consecutive,
    register, RiskConfig, RiskGuard, RiskState, ScoringSystem,
)
```

---

## Module: `indicators`

Pure-numerical technical indicators. All are look-ahead-safe.

```python
from cyqnt_trd.blocks import indicators as ind
```

### Moving averages

| Function | Returns | Description |
|---|---|---|
| `sma(series, period)` | Series | Simple moving average |
| `ema(series, period)` | Series | Exponential MA (alpha = 2/(period+1)) |
| `wma(series, period)` | Series | Linear weighted MA |
| `rma(series, period)` | Series | Wilder smoothing (alpha = 1/period) |

### Momentum / oscillators

| Function | Returns | Description |
|---|---|---|
| `rsi(series, period=14)` | Series | Wilder RSI in [0, 100] |
| `macd(series, fast=12, slow=26, signal=9)` | (macd_line, signal_line, hist) | Standard MACD |
| `stochastic(df, k_period=14, d_period=3, smooth_k=3)` | (k, d) | Slow stochastic |
| `adx(df, period=14)` | (adx, plus_di, minus_di) | Directional movement |

### Volatility / channels

| Function | Returns | Description |
|---|---|---|
| `true_range(df)` | Series | Wilder True Range |
| `atr(df, period=14)` | Series | Average True Range |
| `bollinger(series, period=20, std_mult=2.0)` | (upper, mid, lower) | Bollinger Bands |
| `donchian(df, period=20)` | (upper, lower, mid) | Donchian Channel |

### Volume

| Function | Returns | Description |
|---|---|---|
| `vwap(df)` | Series | Cumulative VWAP (running, not session-reset) |
| `obv(df)` | Series | On-Balance Volume |
| `volume_ma(df, period=20)` | Series | SMA of volume |
| `volume_zscore(df, period=20)` | Series | Rolling z-score of volume |

### Structure / direction

| Function | Returns | Description |
|---|---|---|
| `ma_direction(ma, lookback=5, flat_threshold_bps=5.0)` | Series[str] | "up"/"down"/"flat" classification of MA slope |
| `ma_alignment(*ma_series)` | Series[str] | "bullish"/"bearish"/"mixed" of stacked MAs (fast > … > slow → "bullish") |
| `swing_high(df, lookback=20)` | Series | Rolling N-bar max high |
| `swing_low(df, lookback=20)` | Series | Rolling N-bar min low |
| `highest(series, period)` | Series | Rolling N-bar max |
| `lowest(series, period)` | Series | Rolling N-bar min |
| `price_change_pct(series, periods=1)` | Series | pct-change |

### Composite

| Function | Returns | Description |
|---|---|---|
| `supertrend(df, period=10, multiplier=3.0)` | (st_value, direction) | direction: +1 uptrend, -1 downtrend |
| `ichimoku(df, tenkan=9, kijun=26, senkou_b=52, displacement=26)` | DataFrame[5 cols] | ⚠ chikou_span uses future data (plot only, not for entries) |

### Statistical helpers

| Function | Returns | Description |
|---|---|---|
| `rolling_zscore(series, period)` | Series | (x - mean) / std |
| `rolling_quantile(series, period, q)` | Series | Rolling q-th quantile, q ∈ [0,1] |

---

## Module: `patterns`

Candlestick pattern detectors. All return `pd.Series[bool]`.

```python
from cyqnt_trd.blocks import patterns as pat
```

### Geometric helpers (numeric outputs)

| Function | Returns | Description |
|---|---|---|
| `candle_body(df)` | Series | \|close - open\| |
| `candle_range(df)` | Series | high - low |
| `candle_body_pct(df)` | Series | body / range, [0,1] |
| `candle_upper_shadow(df)` | Series | high - max(o, c) |
| `candle_lower_shadow(df)` | Series | min(o, c) - low |
| `candle_upper_shadow_pct(df)` | Series | upper_shadow / range |
| `candle_lower_shadow_pct(df)` | Series | lower_shadow / range |

### Single-bar patterns (boolean outputs)

| Function | Description |
|---|---|
| `is_bullish(df)` | close > open |
| `is_bearish(df)` | close < open |
| `doji(df, body_to_range_max=0.1)` | Tiny body |
| `marubozu(df, body_to_range_min=0.95)` | No shadows |
| `spinning_top(df)` | Small body, both shadows |
| `hammer(df)` | Small body top, long lower shadow |
| `inverted_hammer(df)` | Small body bottom, long upper shadow |
| `shooting_star(df)` | Bearish inverted hammer |
| `hanging_man(df)` | Bearish hammer |

### Two-bar patterns

| Function | Description |
|---|---|
| `bullish_engulfing(df)` | Prev bearish + current bullish engulfs |
| `bearish_engulfing(df)` | Prev bullish + current bearish engulfs |
| `bullish_harami(df)` | Inside-bar bullish reversal |
| `bearish_harami(df)` | Inside-bar bearish reversal |
| `piercing_line(df)` | Bullish reversal closing past prev midpoint |
| `dark_cloud_cover(df)` | Bearish reversal closing past prev midpoint |
| `tweezer_top(df, tol_pct=0.001)` | Two bars share approx the same high |
| `tweezer_bottom(df, tol_pct=0.001)` | Two bars share approx the same low |

### Three-bar patterns

| Function | Description |
|---|---|
| `morning_star(df, body_size_min=0.6)` | Bearish + indecision + bullish reversal |
| `evening_star(df, body_size_min=0.6)` | Bullish + indecision + bearish reversal |
| `three_white_soldiers(df, body_size_min=0.6)` | Three big bullish bars |
| `three_black_crows(df, body_size_min=0.6)` | Three big bearish bars |

### Gaps

| Function | Description |
|---|---|
| `gap_up(df, min_gap_pct=0.5)` | Gap-up of at least 0.5% |
| `gap_down(df, min_gap_pct=0.5)` | Gap-down of at least 0.5% |

---

## Module: `derivatives`

Crypto-derivatives-specific helpers. All series-in / series-out.

```python
from cyqnt_trd.blocks import derivatives as deriv
```

### Open Interest

| Function | Returns | Description |
|---|---|---|
| `oi_change_pct(oi, periods=1)` | Series | OI percent change |
| `oi_price_divergence(price, oi, lookback=20)` | Series[str] | "bullish_buildup" / "bearish_buildup" / "long_squeeze" / "short_squeeze" / "none" |

### Long/Short ratio (returns string Series)

| Function | Returns | Description |
|---|---|---|
| `long_short_ratio_state(ratio, crowded_threshold=2.5, contrarian_threshold=0.5)` | Series[str] | "crowded_long" / "crowded_short" / "neutral" |
| `taker_buy_sell_state(buy_volume, sell_volume, threshold=1.5)` | Series[str] | "aggressive_buy" / "aggressive_sell" / "balanced" |

> ⚠ Convert to bool with `state == "crowded_long"` before passing to
> `entry.all_of([...])`. The legacy aliases `*_signal` are deprecated.

### CVD / Basis

| Function | Returns | Description |
|---|---|---|
| `cvd(buy_volume, sell_volume)` | Series | Cumulative Volume Delta |
| `cvd_divergence(price, cvd_series, lookback=20)` | Series[str] | "bullish" / "bearish" / "none" |
| `basis(spot_close, futures_close)` | Series | Basis in basis points |
| `basis_zscore(spot_close, futures_close, period=96)` | Series | Rolling z-score of basis |

### Funding rate

| Function | Returns | Description |
|---|---|---|
| `funding_rate_state(funding, high_threshold_bps=5.0, low_threshold_bps=-5.0)` | Series[str] | "bullish_squeeze" / "bearish_squeeze" / "neutral" |

### Liquidations

| Function | Returns | Description |
|---|---|---|
| `liquidation_imbalance(long_liq_usd, short_liq_usd, lookback=12)` | Series | [-1, +1]; +1 = only long liqs |
| `liquidation_clusters(long_liq_usd, short_liq_usd, threshold_usd=1_000_000, lookback=12)` | (long_cluster, short_cluster) | Boolean masks |

---

## Module: `conditions`

Atomic boolean entry conditions. All return `pd.Series[bool]`.

```python
from cyqnt_trd.blocks import conditions as cond
```

### Crossover

| Function | Description |
|---|---|
| `ma_cross_above(fast, slow)` | Golden cross |
| `ma_cross_below(fast, slow)` | Death cross |

### Bounce / breakout / retest / range

| Function | Description |
|---|---|
| `price_bounce_ma(df, ma, direction="long" \| "short")` | Touch-and-reject of MA |
| `breakout_high(df, lookback=20)` | Close > prior N-bar max high |
| `breakout_low(df, lookback=20)` | Close < prior N-bar min low |
| `retest_after_breakout(df, lookback=20, retest_window=5)` | Retest of breakout level |
| `consolidation_range(df, period=20, max_range_pct=0.03)` | Tight rolling range |
| `range_detection(df, period=20, max_range_pct=0.03)` | Alias of `consolidation_range` |

### Volume

| Function | Description |
|---|---|
| `volume_surge(df, ref_volume, multiplier=1.5)` | Bar volume ≥ multiplier × ref_volume |
| `volume_shrink(df, ref_volume, bars=3, multiplier=1.0)` | All last N bars volume < ref |

### MACD

| Function | Description |
|---|---|
| `macd_golden_cross(macd_line, signal_line)` | MACD crosses above signal |
| `macd_death_cross(macd_line, signal_line)` | MACD crosses below signal |
| `macd_above_zero(macd_line)` | macd > 0 |
| `macd_below_zero(macd_line)` | macd < 0 |
| `macd_bullish_divergence(price, macd_line, lookback=20)` | Price LL + MACD HL |
| `macd_bearish_divergence(price, macd_line, lookback=20)` | Price HH + MACD LH |

### RSI

| Function | Description |
|---|---|
| `rsi_overbought(rsi, threshold=70.0)` | rsi >= threshold |
| `rsi_oversold(rsi, threshold=30.0)` | rsi <= threshold |
| `rsi_in_range(rsi, low=40.0, high=60.0)` | rsi in [low, high] |

### ADX

| Function | Description |
|---|---|
| `adx_trending(adx, threshold=25.0)` | adx >= threshold (trend regime) |
| `adx_ranging(adx, threshold=20.0)` | adx < threshold (range regime) |
| `adx_direction_long(plus_di, minus_di)` | +DI > -DI |
| `adx_direction_short(plus_di, minus_di)` | -DI > +DI |

### MA position

| Function | Description |
|---|---|
| `price_above_ma(df, ma, bars=1)` | Close > MA for last N bars |
| `price_below_ma(df, ma, bars=1)` | Close < MA for last N bars |
| `ema_deviation_within(price, ema, max_pct)` | \|price - ema\| / ema <= max_pct |

### Bar shape

| Function | Description |
|---|---|
| `is_bullish_bar(df, min_body_pct=0.0)` | close > open with min body |
| `is_bearish_bar(df, min_body_pct=0.0)` | close < open with min body |

### Time / funding

| Function | Description |
|---|---|
| `time_filter(timestamps_ms, start_hour, end_hour, tz_offset_hours=8)` | Hour-of-day window |
| `funding_window_safe(timestamps_ms, settle_hours_utc=(0,8,16), buffer_min=15)` | Outside funding-settle buffer |

### Multi-frame / structure

| Function | Description |
|---|---|
| `multi_timeframe_alignment(*signals)` | All bool signals True simultaneously |
| `higher_high(df, lookback=10)` | Structural higher-high |
| `higher_low(df, lookback=10)` | Structural higher-low |
| `lower_high(df, lookback=10)` | Structural lower-high |
| `lower_low(df, lookback=10)` | Structural lower-low |
| `liquidity_sweep_high(df, lookback=20)` | Wick poke above prior high then close back below |
| `liquidity_sweep_low(df, lookback=20)` | Wick poke below prior low then close back above |

---

## Module: `entry`

Combinators for boolean conditions.

```python
from cyqnt_trd.blocks import entry
```

| Function | Description |
|---|---|
| `all_of([cond1, cond2, ...])` | AND across an iterable of conditions |
| `any_of([cond1, cond2, ...])` | OR across an iterable of conditions |
| `exclude_when(base, [excl1, ...])` | `base & ~any_of(excl)` — apply vetoes |
| `weighted_score({name: (cond, weight), ...})` | Numeric weighted-sum score |
| `score_entry({name: (cond, weight), ...}, threshold)` | Boolean — score >= threshold |
| `regime_switch(regime_label, {label: signal, ...})` | Pick signal by regime per bar |
| `consecutive(condition, n)` | True where condition was True for N bars |
| `adaptive_switch(indicator, [(predicate, signal), ...], default)` | Per-bar dispatch on an indicator value |

---

## Module: `exit`

Declarative stop-loss / take-profit rules.

```python
from cyqnt_trd.blocks import exit as ex
```

### Functional helpers (single-trade math)

| Function | Returns | Description |
|---|---|---|
| `fixed_stop_price(entry, pct, side="long")` | float | Absolute stop price |
| `fixed_tp_price(entry, pct, side="long")` | float | Absolute TP price |
| `atr_stop_price(entry, atr_value, multiplier=2.0, side="long")` | float | ATR-distance stop |
| `risk_reward(entry, stop, tp)` | float | Reward-to-risk ratio |
| `passes_min_rr(entry, stop, tp, min_rr=1.5)` | bool | RR meets threshold |
| `compute_partial_close_levels(entry, [(pct, ratio), ...], side)` | List | Convert to (price, ratio) |

### Declarative ExitRule classes

All have signature `evaluate(df, entry_price, side="long", *, entry_index=None) -> pd.Series[bool]`.

| Class | Constructor args | Description |
|---|---|---|
| `FixedStopLoss(pct)` | pct (e.g. 0.02) | Fixed pct stop |
| `FixedTakeProfit(pct)` | pct | Fixed pct TP |
| `AtrTrailingStop(atr, multiplier=2.0)` | atr Series | Trailing ATR-distance stop |
| `MaCrossExit(fast_ma, slow_ma)` | two Series | Exit on MA cross against trade |
| `SwingExit(lookback=10)` | lookback | Exit on N-bar swing break |
| `EmaTrailingTp(ema)` | ema Series | Exit when close crosses EMA |
| `MacdReversalExit(macd_line, signal_line)` | two Series | Exit on MACD reversal |
| `GapExit(gap_pct=0.01)` | gap_pct | Exit on adverse gap |
| `TimeBasedExit(max_bars)` | max_bars | Exit after N bars from entry |
| `BreakevenMove(trigger_pct, lock_pct=0.0)` | thresholds | Move stop to breakeven |
| `DualStop(stops=[rule1, rule2, ...])` | List[ExitRule] | Fire on ANY of the stops |
| `MultiTp(targets=[(pct, ratio), ...])` | List | Multi-level TP (stateless mask) |

> ⚠ Always pass `entry_index` to `evaluate()` so trailing/time rules
> count from the entry bar, not the start of the DataFrame.

---

## Module: `risk`

Portfolio-level risk control state machine.

```python
from cyqnt_trd.blocks import risk
```

### Functional

| Function | Description |
|---|---|
| `is_funding_window(now_ms, settle_hours_utc=(0,8,16), buffer_min=15)` | True if within funding buffer |

### Classes

```python
@dataclass(frozen=True)
class RiskConfig:
    max_loss_per_trade_pct: Optional[float] = None      # e.g. 0.10
    max_drawdown_halt_pct: Optional[float] = None       # e.g. 0.50 (permanent halt)
    daily_max_loss_pct: Optional[float] = None          # e.g. 0.08
    monthly_max_loss_pct: Optional[float] = None
    consecutive_loss_pause: Optional[Tuple[int, int]] = None  # (count, pause_ms)
    per_symbol_cooldown: Optional[Tuple[int, int]] = None
    max_positions: Optional[int] = None                 # e.g. 3
    leverage: int = 1
    margin_per_trade_pct: Optional[float] = None        # e.g. 0.15
    min_available_margin_pct: Optional[float] = None    # e.g. 0.50
    total_exposure_cap_pct: Optional[float] = None      # e.g. 0.30
    funding_buffer_min: int = 0                         # 0 = disabled
    funding_settle_hours_utc: Tuple[int, ...] = (0, 8, 16)
    blacklist: Tuple[str, ...] = ()
```

```python
class RiskGuard:
    def __init__(config: RiskConfig, state: RiskState | None = None)

    # Lifecycle hooks
    def on_equity_update(now_ms: int, equity: float) -> None
    def on_position_opened() -> None
    def on_trade_closed(symbol: str, pnl: float, now_ms: int) -> None

    # Pre-open check
    def can_open_new(now_ms, equity, used_margin=0.0, symbol=None
        ) -> Tuple[bool, Optional[str]]
        # returns (True, None) on success, (False, reason_string) on rejection
```

`RiskState` is the mutable counter object — usually you don't touch it directly.

---

## Module: `sizing`

Position-size calculators. All return notional USD.

```python
from cyqnt_trd.blocks import sizing
```

| Function | Description |
|---|---|
| `fixed_pct_of_equity(equity, pct)` | equity × pct |
| `fixed_amount(amount_usd)` | constant |
| `atr_position_size(equity, atr_value, mark_price, risk_pct=0.01, stop_distance_atr_mult=2.0)` | risk_dollars / stop_distance × mark |
| `risk_based_size(equity, entry_price, stop_price, risk_pct=0.01)` | Loss-at-stop = risk_pct × equity |
| `kelly_fraction(win_rate, avg_win, avg_loss, fractional=0.5)` | Fractional Kelly in [0, 1] |
| `grid_levels(center_price, range_pct, n_grids, per_grid_notional)` | List of (price, notional) |
| `pyramid_add(initial_notional, add_count, add_ratio=0.5, max_adds=2)` | Add-position size |
| `round_step_size(qty, step_size)` | Floor qty to LOT_SIZE step |

---

## Module: `scoring`

Multi-factor scoring system.

```python
from cyqnt_trd.blocks import scoring
sys = scoring.ScoringSystem()
sys.add_rule("trend",   trend_cond,   weight=2.0)
sys.add_rule("vol",     vol_cond,     weight=1.0)
sys.add_rule("counter", counter_cond, weight=-1.0)
score   = sys.evaluate()                # numeric pd.Series
signal  = sys.signal(threshold=2.5)     # bool pd.Series
debug   = sys.breakdown()               # one column per rule
```

---

## Module: `regime`

Market regime classifiers.

```python
from cyqnt_trd.blocks import regime
```

| Function | Returns | Description |
|---|---|---|
| `adx_regime(adx, trend_threshold=25.0, range_threshold=20.0)` | Series[str] | "trend" / "range" / "transition" |
| `volatility_regime(df, period=20, high_quantile=0.8, low_quantile=0.2)` | Series[str] | "high" / "normal" / "low" |
| `range_regime(df, period=20, max_range_pct=0.03)` | Series[str] | "range" / "trending" |
| `is_range_regime(df, period=20, max_range_pct=0.03)` | Series[bool] | Boolean variant |
| `trend_regime_ma(df, fast_ma, slow_ma)` | Series[str] | "uptrend" / "downtrend" / "sideways" |

---

## Module: `microstructure`

Whale / order-flow heuristics.

```python
from cyqnt_trd.blocks import microstructure as micro
```

| Function | Returns | Description |
|---|---|---|
| `whale_buy_signal(buy_volume, rolling_period=96, threshold_quantile=0.95)` | Series[bool] | Top-quantile taker-buy bar |
| `whale_sell_signal(sell_volume, rolling_period=96, threshold_quantile=0.95)` | Series[bool] | Top-quantile taker-sell bar |
| `smart_money_inflow(buy_volume, sell_volume, period=12)` | Series | Rolling net buy USD |
| `order_imbalance(buy_volume, sell_volume)` | Series | (b-s)/(b+s) in [-1, 1] |
| `large_print_zscore(volume, period=96)` | Series | Rolling z-score of volume |

---

## Module: `universe`

Symbol-pool filters & scanners.

```python
from cyqnt_trd.blocks import universe
```

### Functional helpers

| Function | Description |
|---|---|
| `fetch_perpetual_universe(market_type="futures")` | Fetch 24h ticker DataFrame |
| `filter_quote_volume(tickers, min_quote_volume=100_000_000)` | Filter by 24h USDT volume |
| `filter_change_pct(tickers, max_abs_pct=100.0, min_pct=None)` | Filter by 24h pct-change |
| `filter_funding_rate(tickers, max_abs_pct=0.5)` | Filter by funding rate (need `augment_with_funding` first) |
| `top_gainers(tickers, n=10)` | Top N by 24h gain |
| `top_losers(tickers, n=10)` | Top N by 24h loss |
| `exclude_symbols(tickers, [list])` | Drop blacklisted symbols |
| `only_symbols(tickers, [list])` | Keep only these symbols |
| `augment_with_funding(tickers)` | Add `fundingRatePct` column |

### Fluent builder

```python
syms = (
    universe.UniverseFilter(universe.fetch_perpetual_universe())
        .filter_quote_suffix("USDT")
        .filter_quote_volume(min_quote_volume=100_000_000)
        .filter_change_pct(max_abs_pct=100.0)
        .with_funding()
        .filter_funding_rate(max_abs_pct=0.5)
        .top_gainers(n=10)
        .symbols()
)
```

---

## Module: `execution`

Order specification helpers (no actual order routing).

```python
from cyqnt_trd.blocks import execution as exe
```

```python
@dataclass
class OrderSpec:
    symbol: str
    side: str            # "long" or "short"
    order_type: str      # "MARKET" / "LIMIT" / "STOP_MARKET" / "STOP_LIMIT"
    quantity: Optional[float] = None
    notional: Optional[float] = None
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"   # GTC | IOC | FOK | GTX
    reduce_only: bool = False
    client_tag: Optional[str] = None
```

| Function | Description |
|---|---|
| `market_order(symbol, side, quantity=None, notional=None, ...)` | Market |
| `limit_order(symbol, side, price, quantity=None, ...)` | Limit |
| `stop_market_order(symbol, side, stop_price, quantity=None, ...)` | Stop-market (defaults to reduce_only=True) |
| `stop_limit_order(symbol, side, stop_price, limit_price, ...)` | Stop-limit |
| `oco_pair(symbol, side, take_profit_price, stop_price, ...)` | Returns (TP_order, stop_order). Both protective legs auto-inverted to the *exit* side. |

---

## Module: `data`

DataFrame ⇄ Bar conversion + Binance public REST fetchers.

```python
from cyqnt_trd.blocks import data
```

### Type conversion

| Function | Description |
|---|---|
| `df_to_bars(df, instrument_id, timeframe)` | Convert DataFrame → List[Bar] |
| `bars_to_df(bars)` | Convert List[Bar] → DataFrame |
| `snapshot_to_df(snapshot, instrument_id, timeframe)` | Extract bars from a DataSnapshot |

### Binance public REST (no API key required)

| Function | Returns | Description |
|---|---|---|
| `fetch_klines(symbol, interval, limit=500, start_ms=None, end_ms=None, market_type="futures")` | DataFrame | OHLCV |
| `fetch_oi(symbol, period="5m", limit=500)` | DataFrame[timestamp, oi, oi_value] | Open interest history |
| `fetch_funding_rate(symbol, limit=1000)` | DataFrame[timestamp, funding_rate] | Funding rate history |
| `fetch_long_short_ratio(symbol, period="5m", mode="top_account"\|"top_position"\|"global", limit=500)` | DataFrame | Top trader ratios |
| `fetch_taker_buy_sell_ratio(symbol, period="5m", limit=500)` | DataFrame | Taker buy/sell |
| `fetch_24h_tickers(market_type="futures")` | DataFrame | All-symbol 24h snapshot |
| `fetch_premium_index(symbol=None)` | DataFrame | Mark/index/funding |
| `fetch_exchange_info(market_type="futures")` | dict | Symbol metadata |

---

## Module: `strategy`

Register a strategy as a SignalPlugin.

```python
from cyqnt_trd.blocks import strategy
```

| Function | Description |
|---|---|
| `register(strategy_id, signal_fn, *, version="block/v1")` | Register `make_signals` for `--strategy-module` use |
| `build_plugin(strategy_id, signal_fn, *, version="block/v1")` | Construct without registering (rare) |
| `is_known_block_strategy(strategy_id)` | Has this ID been registered? |

---

# Three runnable example strategies

> Save each as a separate `.py` file in your strategy directory, then run
> `python -m cyqnt_trd.standard_bot.entrypoints.mvp_backtest --engine python --strategy <id> --strategy-module <module>`.

## Example 1 — MA20/60 trend-pullback with MACD confirmation

```python
"""ma_trend_pullback.py"""
from cyqnt_trd.blocks import indicators as ind, conditions as cond, entry, strategy

def make_signals(df):
    ma20 = ind.sma(df["close"], 20)
    ma60 = ind.sma(df["close"], 60)
    macd_line, signal_line, _ = ind.macd(df["close"], 6, 13, 5)
    vol_ma5 = ind.volume_ma(df, 5)

    long = entry.all_of([
        cond.ma_cross_above(ma20, ma60),
        cond.price_above_ma(df, ma60, bars=5),
        cond.macd_above_zero(macd_line),
        cond.volume_surge(df, vol_ma5, multiplier=1.5),
    ])
    short = entry.all_of([
        cond.ma_cross_below(ma20, ma60),
        cond.price_below_ma(df, ma60, bars=5),
        cond.macd_below_zero(macd_line),
    ])
    return long, short

strategy.register("ma_trend_pullback", make_signals)
```

## Example 2 — Bollinger + RSI mean-reversion (range regime gated)

```python
"""bb_rsi_revert.py"""
from cyqnt_trd.blocks import indicators as ind, conditions as cond, entry, regime, strategy

def make_signals(df):
    upper, mid, lower = ind.bollinger(df["close"], period=20, std_mult=2.0)
    rsi14 = ind.rsi(df["close"], 14)
    in_range = regime.is_range_regime(df, period=20, max_range_pct=0.04)

    long = entry.all_of([
        df["close"] <= lower,
        cond.rsi_oversold(rsi14, threshold=30.0),
        in_range,
    ])
    short = entry.all_of([
        df["close"] >= upper,
        cond.rsi_overbought(rsi14, threshold=70.0),
        in_range,
    ])
    return long, short

strategy.register("bb_rsi_revert", make_signals)
```

## Example 3 — Multi-factor scored breakout with funding-rate veto

```python
"""scored_breakout.py"""
from cyqnt_trd.blocks import indicators as ind, conditions as cond, scoring, strategy

def make_signals(df):
    close = df["close"]
    ma20 = ind.sma(close, 20)
    ma50 = ind.sma(close, 50)
    rsi14 = ind.rsi(close, 14)
    vol_ma20 = ind.volume_ma(df, 20)
    macd_line, signal_line, _ = ind.macd(close)

    sys = scoring.ScoringSystem()
    sys.add_rule("breakout",  cond.breakout_high(df, lookback=20),         weight=2.0)
    sys.add_rule("trend_up",  cond.price_above_ma(df, ma20),                weight=1.0)
    sys.add_rule("vol_surge", cond.volume_surge(df, vol_ma20, 1.5),         weight=1.0)
    sys.add_rule("macd_up",   cond.macd_golden_cross(macd_line, signal_line), weight=1.0)
    sys.add_rule("rsi_zone",  cond.rsi_in_range(rsi14, 50, 75),             weight=0.5)
    sys.add_rule("trend_dn",  cond.price_below_ma(df, ma50),                weight=-1.0)

    funding_safe = cond.funding_window_safe(
        df["close_time"], settle_hours_utc=(0, 8, 16), buffer_min=15,
    )
    long = sys.signal(threshold=4.0) & funding_safe
    return long, None

strategy.register("scored_breakout", make_signals)
```

---

# Common pitfalls (read before generating code)

1. **`derivatives` `*_state` returns strings, not booleans.** Convert with
   `state == "neutral"` before passing to `entry.all_of([...])`.
2. **`make_signals(df)` receives `close_time` (ms) not `timestamp`.**
   The `timestamp` alias is provided as a convenience but `close_time`
   is canonical.
3. **`ichimoku.chikou_span` uses future data.** Do not use for entry
   conditions; plotting only.
4. **MACD signal-period default is 9, but the user dataset commonly uses
   `(6, 13, 5)`.** Read the user's spec carefully and pass explicit
   parameters.
5. **`AtrTrailingStop` etc. need `entry_index`** to count the trailing
   peak from entry. The framework will pass it; if you call manually,
   pass the bar's `df.index` value.
6. **`from cyqnt_trd.blocks import exit` shadows the Python builtin.**
   Always alias: `from cyqnt_trd.blocks import exit as ex`.
7. **`make_signals` must return `(long, short)` or a single `long_signal`
   Series.** Never return numpy arrays — pandas Series with the
   df.index keeps alignment safe.
8. **Defaults vary across timeframes.** SMA20 on 5m bars is different
   from SMA20 on 1d bars. Read the user's intended timeframe before
   choosing defaults.
