# funding-rate-scanner

Funding-rate mean-reversion case. Long signal fires when funding is in the
`bullish_squeeze` zone (deeply negative funding ≤ `low_threshold_bps`),
indicating contrarian long entry.

- **Status**: executable. The data-layer adapter chain
  (`BinanceRestAdapter` + `MarketBundleClient`) provides funding history;
  the case signal consumes a `funding_rate` column on the input DataFrame.
- **Source attribution**:
  - `atomic_strategy_lib.data.funding` (replaced by
    `library.data.adapters.binance_rest.BinanceRestAdapter.fetch_funding_history`)
  - `cyqnt_trd.blocks.derivatives.funding_rate_state` (canonical at
    `library.features.derivatives.funding_rate_state`)
- **Signal**: `signal.py::build_signal` returns True where
  `funding_rate_state(funding_rate) == "bullish_squeeze"`. Falls back to
  all-False if the `funding_rate` column is missing (so plain OHLCV smoke
  backtest does not raise).
- **Parameters** (`preset.json` / `strategy.py`):
  - `high_threshold_bps`: 5.0 — funding above this is `bearish_squeeze`
  - `low_threshold_bps`: -5.0 — funding below this is `bullish_squeeze` → long entry

## Wiring real data

```python
from ai_pro_trading_library.library.data import (
    BinanceRestAdapter, MarketBundleClient, bars_to_frame,
)

client = MarketBundleClient(BinanceRestAdapter())
bars = client.klines("BTCUSDT", "8h", limit=200)
df = bars_to_frame(bars)
funding = client.funding("BTCUSDT", limit=200)
df = df.merge(
    funding.set_index("timestamp")["funding_rate"],
    left_on="close_time", right_index=True, how="left",
)
# now `df` has the `funding_rate` column the signal expects
```
