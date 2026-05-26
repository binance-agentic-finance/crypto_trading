# cyqnt_trd.exec_cli

Execution wrappers for **binance-cli** — ported from `atomic_strategy_lib.execution`.

## ⚠️ Safety First

**All functions default to `dry_run=True`.**
You must pass `dry_run=False` explicitly to place a real order or change account settings.

```python
from cyqnt_trd.exec_cli import market_order, set_leverage, quantize

# Safe by default — prints the command, returns fake OrderResult
result = market_order("BTCUSDT", "BUY", qty=0.01)
# [DRY RUN] market_order: binance-cli futures-usds new-order ...

# Live execution
result = market_order("BTCUSDT", "BUY", qty=0.01, dry_run=False)
```

---

## Module layout

```
exec_cli/
├── __init__.py         # public API surface
├── _subprocess.py      # CLI runner, OrderResult / ExchangeFilter dataclasses
├── orders.py           # market_order, limit_order, stop_market_order,
│                       # cancel_all, partial_close
├── position.py         # set_leverage, set_margin_type
├── filters.py          # exchange_filter_fetch, quantize, round_to_tick
└── README.md           # this file
```

---

## OrderResult

Every order/position function returns an `OrderResult`:

```python
@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str]    # exchange order ID (None for non-order ops)
    executed_qty: float        # qty filled (0 for dry_run / pending limit)
    executed_price: float      # average fill price (0 if unknown)
    fee: float                 # trading fee (0 if not returned by CLI)
    raw_response: dict         # full binance-cli JSON output
    error: Optional[str]       # error message when success=False
```

---

## Functions

### Orders

| Function | CLI call | Description |
|---|---|---|
| `market_order(symbol, side, qty, dry_run=True)` | `binance-cli futures-usds new-order --type MARKET` | Market order |
| `limit_order(symbol, side, qty, price, dry_run=True)` | `... --type LIMIT` | GTC limit order |
| `stop_market_order(symbol, side, stop_price, qty, dry_run=True)` | `... --type STOP_MARKET` | Stop-market order |
| `cancel_all(symbol, dry_run=True)` | `binance-cli futures-usds cancel-all-open-orders` | Cancel all open orders |
| `partial_close(symbol, side, qty, dry_run=True)` | `market_order(..., reduce_only=True)` | Reduce position |

### Position

| Function | CLI call | Description |
|---|---|---|
| `set_leverage(symbol, leverage, dry_run=True)` | `binance-cli futures-usds change-leverage` | Set leverage |
| `set_margin_type(symbol, margin_type, dry_run=True)` | `binance-cli futures-usds change-margin-type` | ISOLATED or CROSSED |

### Filters

| Function | Description |
|---|---|
| `exchange_filter_fetch(symbol, market)` | Fetch step_size, tick_size, min_notional, precisions |
| `quantize(value, step)` | Floor to nearest step (Decimal-safe) |
| `round_to_tick(price, tick_size)` | Floor to nearest tick |

---

## Tests

```bash
pytest tests/standard_bot/test_exec_cli_smoke.py -v
```

---

## Requirements

- `binance-cli` must be installed and on `$PATH` for `dry_run=False` calls.
- All functions are importable without `binance-cli` (dry_run=True never spawns a process).
