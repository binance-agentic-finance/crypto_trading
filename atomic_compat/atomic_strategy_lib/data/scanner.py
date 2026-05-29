"""shim — atomic.data.scanner

Thin compatibility wrappers over cyqnt_trd.data_cli.scanner.
Atomic case scripts often pass `profile=` / `binary=` kwargs; the native
cyqnt_trd.data_cli functions do not need them. We accept and ignore these
kwargs here for compatibility.

Return type translation:
- native scan_with_filter() -> DataFrame with columns [symbol, volume_quote]
- atomic cases expect -> list[dict] with keys {"symbol": ..., "volume_quote": ...}
- native full_market_scan() -> list[str] (no change needed)
"""
from cyqnt_trd.data_cli import full_market_scan as _full_market_scan, scan_with_filter as _scan_with_filter


def full_market_scan(market="futures", profile=None, binary="binance-cli", **kwargs):
    return _full_market_scan(market=market)


def scan_with_filter(market="futures", min_volume=0.0, profile=None, binary="binance-cli", **kwargs):
    result = _scan_with_filter(market=market, min_volume=min_volume)
    # Convert DataFrame → list[dict] for atomic case compatibility
    if hasattr(result, "to_dict"):
        return result.to_dict(orient="records")
    return result
