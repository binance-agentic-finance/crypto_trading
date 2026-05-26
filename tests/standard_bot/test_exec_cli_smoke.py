"""Smoke tests for cyqnt_trd.exec_cli.

Tests verify:
  1. dry_run=True never invokes subprocess (subprocess.run is NOT called)
  2. dry_run=False parses binance-cli JSON response into a correct OrderResult
  3. quantize / round_to_tick produce numerically expected values
  4. Input validation raises ValueError on bad inputs
  5. ExchangeFilter fallback defaults when CLI returns empty data
  6. CLIError propagation: CLI failure → OrderResult(success=False)

No real binance-cli binary is required — all subprocess calls are mocked.
"""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from cyqnt_trd.exec_cli import (
    CLIError,
    ExchangeFilter,
    OrderResult,
    cancel_all,
    exchange_filter_fetch,
    limit_order,
    market_order,
    partial_close,
    quantize,
    round_to_tick,
    set_leverage,
    set_margin_type,
    stop_market_order,
)
from cyqnt_trd.exec_cli._subprocess import run_cli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_subprocess_result(stdout: str, returncode: int = 0) -> MagicMock:
    """Return a mock object mimicking subprocess.CompletedProcess."""
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = ""
    return m


FAKE_ORDER_RESPONSE = {
    "orderId": 123456789,
    "symbol": "BTCUSDT",
    "status": "FILLED",
    "side": "BUY",
    "type": "MARKET",
    "executedQty": "0.01",
    "avgPrice": "65000.0",
    "price": "0",
}

FAKE_CANCEL_RESPONSE = {"code": 200, "msg": "The operation of cancel all open order is done."}

FAKE_LEVERAGE_RESPONSE = {"leverage": 10, "maxNotionalValue": "1000000", "symbol": "BTCUSDT"}

FAKE_EXCHANGE_INFO = {
    "symbols": [
        {
            "symbol": "BTCUSDT",
            "quantityPrecision": 3,
            "pricePrecision": 2,
            "filters": [
                {"filterType": "LOT_SIZE", "minQty": "0.001", "stepSize": "0.001"},
                {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                {"filterType": "MIN_NOTIONAL", "notional": "5.0"},
            ],
        }
    ]
}


# ===========================================================================
# 1. dry_run=True must NEVER call subprocess
# ===========================================================================

class TestDryRunNeverCallsSubprocess:
    """Verify that subprocess.run is never invoked when dry_run=True."""

    def test_market_order_dry_run(self, capsys):
        with patch("subprocess.run") as mock_run:
            result = market_order("BTCUSDT", "BUY", qty=0.01)
            mock_run.assert_not_called()
        assert result.success is True
        assert result.order_id == "DRY_RUN"
        assert result.executed_qty == pytest.approx(0.01)
        assert result.raw_response["dryRun"] is True
        out = capsys.readouterr().out
        assert "[DRY RUN]" in out
        assert "BTCUSDT" in out

    def test_limit_order_dry_run(self):
        with patch("subprocess.run") as mock_run:
            result = limit_order("ETHUSDT", "SELL", qty=0.5, price=3200.0)
            mock_run.assert_not_called()
        assert result.success is True
        assert result.executed_price == pytest.approx(3200.0)
        assert result.raw_response["dryRun"] is True

    def test_stop_market_order_dry_run(self):
        with patch("subprocess.run") as mock_run:
            result = stop_market_order("BTCUSDT", "SELL", stop_price=60000.0, qty=0.01)
            mock_run.assert_not_called()
        assert result.success is True
        assert result.raw_response["dryRun"] is True

    def test_cancel_all_dry_run(self):
        with patch("subprocess.run") as mock_run:
            result = cancel_all("BTCUSDT")
            mock_run.assert_not_called()
        assert result.success is True
        assert result.raw_response["action"] == "cancel_all"

    def test_partial_close_dry_run(self):
        with patch("subprocess.run") as mock_run:
            result = partial_close("BTCUSDT", "SELL", qty=0.005)
            mock_run.assert_not_called()
        assert result.success is True

    def test_set_leverage_dry_run(self):
        with patch("subprocess.run") as mock_run:
            result = set_leverage("BTCUSDT", 10)
            mock_run.assert_not_called()
        assert result.success is True
        assert result.raw_response["leverage"] == 10

    def test_set_margin_type_dry_run(self):
        with patch("subprocess.run") as mock_run:
            result = set_margin_type("BTCUSDT", "ISOLATED")
            mock_run.assert_not_called()
        assert result.success is True
        assert result.raw_response["marginType"] == "ISOLATED"


# ===========================================================================
# 2. dry_run=False correctly calls subprocess and parses response
# ===========================================================================

class TestDryRunFalseParsesResponse:
    """Verify that dry_run=False invokes subprocess and parses the response."""

    def test_market_order_live_success(self):
        fake = _fake_subprocess_result(json.dumps(FAKE_ORDER_RESPONSE))
        with patch("subprocess.run", return_value=fake) as mock_run:
            result = market_order("BTCUSDT", "BUY", qty=0.01, dry_run=False)
            mock_run.assert_called_once()

        assert result.success is True
        assert result.order_id == "123456789"
        assert result.executed_qty == pytest.approx(0.01)
        assert result.executed_price == pytest.approx(65000.0)
        assert result.error is None
        assert result.raw_response == FAKE_ORDER_RESPONSE

    def test_limit_order_live_success(self):
        response = {**FAKE_ORDER_RESPONSE, "type": "LIMIT", "executedQty": "0",
                    "price": "3200.0", "avgPrice": "0"}
        fake = _fake_subprocess_result(json.dumps(response))
        with patch("subprocess.run", return_value=fake):
            result = limit_order("ETHUSDT", "SELL", qty=0.5, price=3200.0, dry_run=False)
        assert result.success is True
        assert result.order_id == "123456789"

    def test_market_order_live_error_response(self):
        """Binance error code in JSON → OrderResult(success=False)."""
        error_response = {"code": -2019, "msg": "Margin is insufficient."}
        fake = _fake_subprocess_result(json.dumps(error_response))
        with patch("subprocess.run", return_value=fake):
            result = market_order("BTCUSDT", "BUY", qty=0.01, dry_run=False)
        assert result.success is False
        assert result.error is not None
        assert "Margin" in result.error

    def test_cancel_all_live_success(self):
        fake = _fake_subprocess_result(json.dumps(FAKE_CANCEL_RESPONSE))
        with patch("subprocess.run", return_value=fake):
            result = cancel_all("BTCUSDT", dry_run=False)
        assert result.success is True

    def test_set_leverage_live_success(self):
        fake = _fake_subprocess_result(json.dumps(FAKE_LEVERAGE_RESPONSE))
        with patch("subprocess.run", return_value=fake):
            result = set_leverage("BTCUSDT", 10, dry_run=False)
        assert result.success is True

    def test_set_margin_type_idempotent_4046(self):
        """Code -4046 (already set) is treated as success."""
        already_set = {"code": -4046, "msg": "No need to change margin type."}
        fake = _fake_subprocess_result(json.dumps(already_set))
        with patch("subprocess.run", return_value=fake):
            result = set_margin_type("BTCUSDT", "ISOLATED", dry_run=False)
        assert result.success is True

    def test_cli_error_returns_failure_result(self):
        """subprocess non-zero exit → OrderResult(success=False)."""
        fake = _fake_subprocess_result("", returncode=1)
        fake.stderr = "binance-cli: command not found"
        with patch("subprocess.run", return_value=fake):
            result = market_order("BTCUSDT", "BUY", qty=0.01, dry_run=False)
        assert result.success is False
        assert result.error is not None

    def test_stop_market_order_live_success(self):
        response = {**FAKE_ORDER_RESPONSE, "type": "STOP_MARKET",
                    "stopPrice": "60000.0", "executedQty": "0.01"}
        fake = _fake_subprocess_result(json.dumps(response))
        with patch("subprocess.run", return_value=fake):
            result = stop_market_order(
                "BTCUSDT", "SELL", stop_price=60000.0, qty=0.01, dry_run=False
            )
        assert result.success is True


# ===========================================================================
# 3. quantize — deterministic floor-to-step arithmetic
# ===========================================================================

class TestQuantize:
    def test_basic_floor(self):
        assert quantize(0.1234, 0.001) == pytest.approx(0.123)

    def test_exact_multiple(self):
        assert quantize(0.005, 0.001) == pytest.approx(0.005)

    def test_float_drift_avoided(self):
        # Without Decimal, 1.2399 / 0.01 can produce 123.98999... → floor 123
        result = quantize(1.2399, 0.01)
        assert result == pytest.approx(1.23)

    def test_large_step(self):
        assert quantize(9.75, 0.5) == pytest.approx(9.5)

    def test_step_zero_passthrough(self):
        assert quantize(1.234, 0.0) == pytest.approx(1.234)

    def test_step_negative_passthrough(self):
        assert quantize(1.234, -0.001) == pytest.approx(1.234)

    def test_btc_quantity(self):
        # BTC stepSize=0.001, qty calculation
        result = quantize(0.123456789, 0.001)
        assert result == pytest.approx(0.123)

    def test_decimal_precision_matches_atomic(self):
        """Ensure parity with atomic's Decimal-based implementation."""
        for val, step, expected in [
            (0.00012345, 0.00001, 0.00012),
            (100.9999, 0.01, 100.99),
            (0.001, 0.001, 0.001),
        ]:
            assert quantize(val, step) == pytest.approx(expected), \
                f"quantize({val}, {step}) expected {expected}"


# ===========================================================================
# 4. round_to_tick — floor price to nearest tick
# ===========================================================================

class TestRoundToTick:
    def test_basic(self):
        assert round_to_tick(29876.543, 0.1) == pytest.approx(29876.5)

    def test_exact(self):
        assert round_to_tick(100.0, 0.01) == pytest.approx(100.0)

    def test_btc_price_0_1_tick(self):
        assert round_to_tick(65432.78, 0.1) == pytest.approx(65432.7)

    def test_eth_price_0_01_tick(self):
        assert round_to_tick(3456.789, 0.01) == pytest.approx(3456.78)

    def test_tick_zero_passthrough(self):
        assert round_to_tick(1234.5, 0.0) == pytest.approx(1234.5)

    def test_tick_negative_passthrough(self):
        assert round_to_tick(1234.5, -0.1) == pytest.approx(1234.5)

    def test_no_floating_point_bleed(self):
        # Ensure floors don't accidentally round up
        price = 100.0 - 1e-10  # just below 100
        result = round_to_tick(price, 1.0)
        assert result == pytest.approx(99.0)


# ===========================================================================
# 5. Input validation
# ===========================================================================

class TestInputValidation:
    def test_market_order_empty_symbol(self):
        with pytest.raises(ValueError, match="symbol"):
            market_order("", "BUY", qty=0.01)

    def test_market_order_bad_side(self):
        with pytest.raises(ValueError, match="side"):
            market_order("BTCUSDT", "LONG", qty=0.01)

    def test_market_order_zero_qty(self):
        with pytest.raises(ValueError, match="qty"):
            market_order("BTCUSDT", "BUY", qty=0)

    def test_market_order_negative_qty(self):
        with pytest.raises(ValueError, match="qty"):
            market_order("BTCUSDT", "BUY", qty=-1.0)

    def test_limit_order_zero_price(self):
        with pytest.raises(ValueError, match="price"):
            limit_order("BTCUSDT", "BUY", qty=0.01, price=0)

    def test_stop_market_bad_stop_price(self):
        with pytest.raises(ValueError, match="stop_price"):
            stop_market_order("BTCUSDT", "SELL", stop_price=-1.0, qty=0.01)

    def test_stop_market_needs_qty_or_close_position(self):
        with pytest.raises(ValueError, match="qty"):
            stop_market_order("BTCUSDT", "SELL", stop_price=60000.0, qty=0)

    def test_set_leverage_non_integer(self):
        with pytest.raises((ValueError, TypeError)):
            set_leverage("BTCUSDT", 10.5)  # type: ignore

    def test_set_leverage_zero(self):
        with pytest.raises(ValueError, match="leverage"):
            set_leverage("BTCUSDT", 0)

    def test_set_margin_type_invalid(self):
        with pytest.raises(ValueError, match="margin_type"):
            set_margin_type("BTCUSDT", "CROSS")  # wrong spelling, should be CROSSED

    def test_cancel_all_empty_symbol(self):
        with pytest.raises(ValueError, match="symbol"):
            cancel_all("")


# ===========================================================================
# 6. exchange_filter_fetch — parse + fallback
# ===========================================================================

class TestExchangeFilterFetch:
    def test_parse_btcusdt_futures(self):
        fake = _fake_subprocess_result(json.dumps(FAKE_EXCHANGE_INFO))
        with patch("subprocess.run", return_value=fake):
            f = exchange_filter_fetch("BTCUSDT", market="futures")
        assert isinstance(f, ExchangeFilter)
        assert f.symbol == "BTCUSDT"
        assert f.step_size == pytest.approx(0.001)
        assert f.tick_size == pytest.approx(0.1)
        assert f.min_notional == pytest.approx(5.0)
        assert f.qty_precision == 3
        assert f.price_precision == 2

    def test_fallback_when_symbol_not_found(self):
        empty_info = {"symbols": []}
        fake = _fake_subprocess_result(json.dumps(empty_info))
        with patch("subprocess.run", return_value=fake):
            f = exchange_filter_fetch("UNKNOWNUSDT")
        assert f.symbol == "UNKNOWNUSDT"
        assert f.step_size > 0
        assert f.tick_size > 0

    def test_fallback_on_cli_error(self):
        """CLI failure (FileNotFoundError) → returns default filter."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            f = exchange_filter_fetch("BTCUSDT")
        assert isinstance(f, ExchangeFilter)
        assert f.success_fallback if hasattr(f, "success_fallback") else True
        assert f.step_size > 0


# ===========================================================================
# 7. run_cli helper — unit tests
# ===========================================================================

class TestRunCli:
    def test_appends_json_flag(self):
        fake = _fake_subprocess_result('{"result": 1}')
        with patch("subprocess.run", return_value=fake) as mock_run:
            run_cli(["binance-cli", "spot", "ticker-price"])
            cmd_used = mock_run.call_args[0][0]
            assert "--json" in cmd_used

    def test_injects_profile(self):
        fake = _fake_subprocess_result('{"result": 1}')
        with patch("subprocess.run", return_value=fake) as mock_run:
            run_cli(["binance-cli", "spot", "ticker-price"], profile="testnet")
            cmd_used = mock_run.call_args[0][0]
            assert "--profile" in cmd_used
            assert "testnet" in cmd_used

    def test_raises_on_nonzero_exit(self):
        fake = _fake_subprocess_result("", returncode=1)
        fake.stderr = '{"code": -1121, "msg": "Invalid symbol."}'
        with patch("subprocess.run", return_value=fake):
            with pytest.raises(CLIError, match="Invalid symbol"):
                run_cli(["binance-cli", "bad-cmd"])

    def test_raises_on_missing_binary(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(CLIError, match="not found"):
                run_cli(["binance-cli-missing", "cmd"])

    def test_raises_on_invalid_json(self):
        fake = _fake_subprocess_result("not-json", returncode=0)
        with patch("subprocess.run", return_value=fake):
            with pytest.raises(CLIError, match="JSON"):
                run_cli(["binance-cli", "cmd"])

    def test_returns_empty_dict_on_empty_stdout(self):
        fake = _fake_subprocess_result("", returncode=0)
        with patch("subprocess.run", return_value=fake):
            result = run_cli(["binance-cli", "cmd"])
        assert result == {}


# ===========================================================================
# 8. OrderResult dataclass sanity
# ===========================================================================

class TestOrderResultDataclass:
    def test_defaults(self):
        r = OrderResult(success=True)
        assert r.order_id is None
        assert r.executed_qty == 0.0
        assert r.executed_price == 0.0
        assert r.fee == 0.0
        assert r.raw_response == {}
        assert r.error is None

    def test_failure_result(self):
        r = OrderResult(success=False, error="Margin insufficient")
        assert r.success is False
        assert r.error == "Margin insufficient"

    def test_dry_run_result_shape(self):
        result = market_order("BTCUSDT", "BUY", qty=0.1)
        assert isinstance(result, OrderResult)
        assert result.success is True
        assert result.executed_qty == pytest.approx(0.1)
