"""shim — atomic.data.kline (atomic-named wrappers around cyqnt_trd.data_cli)

For the standard k-line endpoints (spot ``/api/v3/klines`` and
futures ``/fapi/v1/klines``) we go directly to Binance public REST.
Each ``kline_fetch`` shaves a ~50-100ms subprocess hop and ~300ms of
binance-cli warm-up off vs. routing through ``cyqnt_trd.fetch_klines``,
which matters when scanners fan out 50+ symbols (volume-surge etc.).
The DataFrame-shaped helpers (``fetch_klines`` / ``fetch_klines_multi_tf``)
are still re-exported for callers that want the legacy path.
"""
import json
import sys
import time
import urllib.error
import urllib.request

from cyqnt_trd.data_cli import fetch_klines, fetch_klines_multi_tf  # noqa: F401
from cyqnt_trd.data_cli.kline import df_to_candles as _df_to_candles  # noqa: F401

_FAPI_BASE = "https://fapi.binance.com"
_SPOT_BASE = "https://api.binance.com"
_HTTP_TIMEOUT = 10
_RETRY_MAX = 3
_RETRY_BACKOFF_BASE = 0.4


def _http_get_json(url: str, timeout: int = _HTTP_TIMEOUT):
    """GET URL → parsed JSON, retrying transient 429/418/5xx."""
    last_err = None
    for attempt in range(_RETRY_MAX + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "atomic-shim/1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code in (429, 418) or 500 <= exc.code < 600:
                if attempt >= _RETRY_MAX:
                    break
                ra = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    wait = float(ra) if ra else _RETRY_BACKOFF_BASE * (2 ** attempt)
                except (TypeError, ValueError):
                    wait = _RETRY_BACKOFF_BASE * (2 ** attempt)
                time.sleep(min(max(wait, 0.1), 5.0))
                continue
            break
        except (urllib.error.URLError, json.JSONDecodeError,
                TimeoutError, OSError) as exc:
            last_err = exc
            if attempt >= _RETRY_MAX:
                break
            time.sleep(_RETRY_BACKOFF_BASE * (2 ** attempt))
            continue
    if last_err is not None:
        print(f"[kline] giving up on {url}: {type(last_err).__name__}: {last_err}",
              file=sys.stderr, flush=True)
    return None


def kline_fetch(symbol, interval="1h", limit=100, market="spot",
                profile=None, binary="binance-cli"):
    """atomic-style kline_fetch — returns list[Candle].

    Hits Binance's public ``klines`` endpoint directly. Falls back to
    the cyqnt_trd CLI path if the HTTP call fails after retries — that
    way profile-specific configurations (proxies, auth) still work for
    callers that rely on them.
    """
    if market == "futures":
        url = (f"{_FAPI_BASE}/fapi/v1/klines?symbol={symbol}"
               f"&interval={interval}&limit={int(limit)}")
    else:
        url = (f"{_SPOT_BASE}/api/v3/klines?symbol={symbol}"
               f"&interval={interval}&limit={int(limit)}")

    raw = _http_get_json(url)
    if not isinstance(raw, list):
        # Fallback to the legacy DataFrame path (useful when the CLI
        # has a configured profile that the shim doesn't know about).
        df = fetch_klines(symbol, interval=interval, limit=limit, market=market)
        candles = []
        if df.empty:
            return candles
        from cyqnt_trd.compat.types import Candle
        for _, row in df.iterrows():
            candles.append(Candle(
                timestamp=int(row.get("open_time", 0)),
                open=float(row.get("open", 0)),
                high=float(row.get("high", 0)),
                low=float(row.get("low", 0)),
                close=float(row.get("close", 0)),
                volume=float(row.get("volume", 0)),
                quote_volume=float(row.get("quote_volume", 0)),
                trades=int(row.get("trades", 0)),
            ))
        return candles

    # Binance kline row shape (12 cols):
    #   [open_time, open, high, low, close, volume,
    #    close_time, quote_volume, trades, taker_base_vol,
    #    taker_quote_vol, ignore]
    from cyqnt_trd.compat.types import Candle
    candles = []
    for row in raw:
        if len(row) < 9:
            continue
        candles.append(Candle(
            timestamp=int(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            quote_volume=float(row[7]),
            trades=int(row[8]),
        ))
    return candles


def kline_fetch_multi_tf(symbol, timeframes, limits=None, limit=None,
                        market="spot", profile=None, binary="binance-cli"):
    """atomic-style — returns dict[tf -> list[Candle]]."""
    out = {}
    for tf in timeframes:
        lim = (limits or {}).get(tf, limit or 100)
        out[tf] = kline_fetch(symbol, interval=tf, limit=lim, market=market)
    return out


def listing_age_fetch(symbol, profile=None, binary="binance-cli"):
    """atomic-style listing_age_fetch — re-export from data_cli."""
    from cyqnt_trd.data_cli.kline import listing_age_fetch as _impl
    return _impl(symbol)
