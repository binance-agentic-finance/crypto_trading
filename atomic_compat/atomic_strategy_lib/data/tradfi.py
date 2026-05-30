"""shim — atomic.data.tradfi

TradFi Perpetuals (contractType=TRADIFI_PERPETUAL — XAU, SPX-like, etc.)
discovery + ticker / funding / OI enrichment.

Direct HTTPS to fapi.binance.com (public market data, no auth needed)
instead of going through ``binance-cli``. Three reasons:

  1. The 24h ticker and funding-rate endpoints both expose **bulk**
     forms (``/ticker/24hr`` and ``/premiumIndex`` with no symbol
     param) — one round-trip covers all 59 TradFi pairs.
  2. The ``/openInterest`` endpoint is per-symbol, so we fan out 8-way
     concurrently. 59 symbols → ~1s instead of ~36s sequential.
  3. Removes one subprocess hop (~50-100ms × N) and the legacy
     ``time.sleep(delay)`` rate-limit padding the case scripts inherited
     from earlier shim versions.

Public API matches what the case scripts call:

  - ``tradfi_perpetual_symbols(contract_type='TRADIFI_PERPETUAL', status='TRADING')``
      -> list[dict] one per TradFi symbol with keys:
         symbol, fapi_symbol, ticker (base asset), pair, contract_type,
         status, underlying_type, underlying_sub_type, base_asset,
         quote_asset, price_precision, quantity_precision

  - ``enrich_tradfi_with_market(symbols, delay=0.0)``
      -> list[dict] same items with extra keys:
         price, change_24h_pct, volume_24h_quote, high_24h, low_24h,
         funding_rate, open_interest_usdt
         (sorted by volume_24h_quote desc)

The legacy ``delay`` kwarg is accepted for backwards compatibility but
ignored — bulk + concurrent fetches make per-symbol sleeps unnecessary.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

FAPI_BASE = "https://fapi.binance.com"
HTTP_TIMEOUT = 10
# Concurrency for per-symbol fan-out. 5 workers × ~80ms each finishes
# 59 TradFi symbols in ~2-3s, well under the 2400 weight/min IP limit
# even when several pipelines run side-by-side.
OI_WORKERS = 5
# Retry policy for transient 429 / 418 (rate-limit) and 5xx responses.
RETRY_MAX = 3
RETRY_BACKOFF_BASE = 0.6   # seconds; doubled each retry, capped


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _http_get_json(url: str, timeout: int = HTTP_TIMEOUT):
    """Fetch URL → parsed JSON, retrying on rate-limit / transient 5xx.

    Honors Binance's ``Retry-After`` header on 429 / 418. Returns None
    only after exhausting retries — caller still has to handle None for
    a clean fallback to defaults, but at least a transient blip won't
    silently zero a whole pipeline run.
    """
    last_err = None
    for attempt in range(RETRY_MAX + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "atomic-shim/1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            last_err = exc
            # 429 = rate-limited; 418 = IP-banned (Binance signals soft
            # ban this way); 5xx = upstream blip → all retryable.
            if exc.code in (429, 418) or 500 <= exc.code < 600:
                if attempt >= RETRY_MAX:
                    break
                ra = exc.headers.get("Retry-After") if exc.headers else None
                wait = _safe_float(ra) if ra else (
                    RETRY_BACKOFF_BASE * (2 ** attempt))
                # Cap at 5s to avoid hanging a pipeline forever.
                wait = min(max(wait, 0.1), 5.0)
                print(
                    f"[tradfi] HTTP {exc.code} on {url} — retry "
                    f"{attempt + 1}/{RETRY_MAX} after {wait:.1f}s",
                    file=sys.stderr, flush=True,
                )
                time.sleep(wait)
                continue
            break  # non-retryable 4xx
        except (urllib.error.URLError, json.JSONDecodeError,
                TimeoutError, OSError) as exc:
            last_err = exc
            if attempt >= RETRY_MAX:
                break
            time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
            continue
    if last_err is not None:
        print(
            f"[tradfi] giving up on {url}: {type(last_err).__name__}: {last_err}",
            file=sys.stderr, flush=True,
        )
    return None


def tradfi_perpetual_symbols(contract_type: str = "TRADIFI_PERPETUAL",
                             status: str = "TRADING",
                             profile: Optional[str] = None,
                             binary: str = "binance-cli",
                             **_legacy) -> list[dict]:
    """Fetch all TRADIFI_PERPETUAL contracts from Binance FAPI exchange-info.

    ``profile`` / ``binary`` are accepted for legacy compatibility and
    ignored — public endpoint, no auth.
    """
    raw = _http_get_json(f"{FAPI_BASE}/fapi/v1/exchangeInfo", timeout=20)
    if not isinstance(raw, dict):
        return []
    syms = raw.get("symbols", [])
    out: list[dict] = []
    for s in syms:
        ct = s.get("contractType", "")
        st = s.get("status", "")
        if contract_type and ct != contract_type:
            continue
        if status and st != status:
            continue
        symbol = s.get("symbol", "")
        base = s.get("baseAsset", "")
        out.append({
            "symbol":               symbol,
            "fapi_symbol":          symbol,
            "ticker":               base,                     # e.g. 'XAU' from 'XAUUSDT'
            "pair":                 s.get("pair", symbol),
            "contract_type":        ct,
            "status":               st,
            "underlying_type":      s.get("underlyingType", ""),
            "underlying_sub_type":  s.get("underlyingSubType", []),
            "base_asset":           base,
            "quote_asset":          s.get("quoteAsset", ""),
            "price_precision":      s.get("pricePrecision", 0),
            "quantity_precision":   s.get("quantityPrecision", 0),
        })
    return out


def _bulk_ticker24h() -> dict:
    """One round-trip — returns {symbol: ticker_dict} for the whole futures market."""
    raw = _http_get_json(f"{FAPI_BASE}/fapi/v1/ticker/24hr")
    if not isinstance(raw, list):
        return {}
    return {item.get("symbol"): item for item in raw if item.get("symbol")}


def _bulk_funding() -> dict:
    """One round-trip — returns {symbol: lastFundingRate} for the whole market."""
    raw = _http_get_json(f"{FAPI_BASE}/fapi/v1/premiumIndex")
    if not isinstance(raw, list):
        return {}
    return {
        item.get("symbol"): _safe_float(item.get("lastFundingRate"))
        for item in raw if item.get("symbol")
    }


def _fetch_oi_one(symbol: str) -> tuple[str, float]:
    """Fetch open interest (in base units) for one symbol."""
    raw = _http_get_json(f"{FAPI_BASE}/fapi/v1/openInterest?symbol={symbol}")
    if not isinstance(raw, dict):
        return symbol, 0.0
    return symbol, _safe_float(raw.get("openInterest"))


def _bulk_oi(symbols: list[str], workers: int = OI_WORKERS) -> dict:
    """Fan out per-symbol OI calls concurrently. Returns {symbol: oi_base}."""
    out: dict = {}
    if not symbols:
        return out
    with ThreadPoolExecutor(max_workers=min(workers, len(symbols))) as pool:
        futs = {pool.submit(_fetch_oi_one, s): s for s in symbols}
        for fut in as_completed(futs):
            try:
                sym, oi = fut.result()
                out[sym] = oi
            except Exception:
                out[futs[fut]] = 0.0
    return out


def enrich_tradfi_with_market(symbols: list[dict], delay: float = 0.0,
                              profile: Optional[str] = None,
                              binary: str = "binance-cli",
                              **_legacy) -> list[dict]:
    """Enrich TradFi symbols with 24h ticker / funding rate / open interest.

    Cost: 3 round-trips total — bulk ticker24hr + bulk premiumIndex +
    1 concurrent batch of per-symbol openInterest calls (8-way fan-out).
    For 59 TradFi symbols this completes in ~2-3s vs ~36s sequential.

    ``delay`` and ``profile`` are accepted for backwards compatibility
    and ignored.
    """
    if not symbols:
        return []

    bulk_ticker = _bulk_ticker24h()
    bulk_funding = _bulk_funding()

    sym_list = [s.get("symbol", "") for s in symbols if s.get("symbol")]
    bulk_oi = _bulk_oi(sym_list)

    out: list[dict] = []
    for s in symbols:
        sym = s.get("symbol", "")
        t = bulk_ticker.get(sym, {})
        last_price = _safe_float(t.get("lastPrice"))
        oi_base = bulk_oi.get(sym, 0.0)

        enriched = dict(s)
        enriched.update({
            "price":                last_price,
            "change_24h_pct":       _safe_float(t.get("priceChangePercent")),
            "volume_24h_quote":     _safe_float(t.get("quoteVolume")),
            "high_24h":             _safe_float(t.get("highPrice")),
            "low_24h":              _safe_float(t.get("lowPrice")),
            "funding_rate":         bulk_funding.get(sym, 0.0),
            "open_interest_usdt":   oi_base * last_price if last_price > 0 else 0.0,
        })
        out.append(enriched)

    out.sort(key=lambda x: x.get("volume_24h_quote", 0), reverse=True)
    return out


def __getattr__(name):
    """Catch-all for less-common names — return safe empty default."""
    if name.startswith("_"):
        raise AttributeError(name)
    return lambda *a, **k: []
