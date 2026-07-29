"""OHLCV loading for the YAML pipeline's vectorized backtest.

Resolves ``data.source`` (or a CLI ``--input-json`` override) into an OHLCV
DataFrame with columns: open_time, open, high, low, close, volume,
quote_volume, close_time. Tries the declared source, falls back to a repo
fixture when offline.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_KLINES_HOST = {
    "spot": "https://api.binance.com/api/v3/klines",
    "futures": "https://fapi.binance.com/fapi/v1/klines",
}


def _rows_to_df(rows, price_keyed: bool):
    import pandas as pd

    if price_keyed:  # {open_price, high_price, ...} (HistoricalJson / our --input-json)
        df = pd.DataFrame({
            "open_time": [int(r.get("open_time", r["close_time"])) for r in rows],
            "open": [float(r["open_price"]) for r in rows],
            "high": [float(r["high_price"]) for r in rows],
            "low": [float(r["low_price"]) for r in rows],
            "close": [float(r["close_price"]) for r in rows],
            "volume": [float(r.get("volume", 0.0)) for r in rows],
            "close_time": [int(r["close_time"]) for r in rows],
            "quote_volume": [float(r.get("quote_volume", 0.0) or 0.0) for r in rows],
        })
    else:  # Binance kline arrays
        df = pd.DataFrame({
            "open_time": [int(k[0]) for k in rows], "open": [float(k[1]) for k in rows],
            "high": [float(k[2]) for k in rows], "low": [float(k[3]) for k in rows],
            "close": [float(k[4]) for k in rows], "volume": [float(k[5]) for k in rows],
            "close_time": [int(k[6]) for k in rows], "quote_volume": [float(k[7]) for k in rows],
        })
    df["timestamp"] = df["close_time"]
    return df


def _fixture(symbol: str, interval: str):
    import pandas as pd

    names = {
        ("BTCUSDT", "1h"): "BTCUSDT_1h_500bars.parquet",
        ("BTCUSDT", "4h"): "BTCUSDT_4h_300bars.parquet",
        ("BTCUSDT", "15m"): "BTCUSDT_15m_500bars.parquet",
        ("ETHUSDT", "1h"): "ETHUSDT_1h_500bars.parquet",
    }
    name = names.get((symbol.upper(), interval))
    if not name:
        return None
    # yaml_pipeline/_data.py → repo root is parents[3]
    p = Path(__file__).resolve().parents[3] / "tests" / "blocks" / "fixtures" / name
    if not p.exists():
        return None
    df = pd.read_parquet(p).reset_index(drop=True)
    df["timestamp"] = df["close_time"]
    return df


def _interval_matches(path: Path, interval: str) -> bool:
    """True when *interval* appears as a delimited token in *path*.

    Token-delimited so ``1h`` does not match ``21h`` / ``1h30`` and ``1m`` does
    not match ``1mo``.
    """
    return re.search(
        rf"(?<![0-9a-z]){re.escape(interval.lower())}(?![0-9a-z])",
        path.as_posix().lower(),
    ) is not None


def load_ohlcv(spec: Dict[str, Any], input_json: Optional[str] = None,
               limit: int = 1000) -> Tuple[Any, str]:
    """Return (df, source_label)."""
    data = spec.get("data") or {}
    symbol = str(data["symbol"]).upper()
    interval = str((data.get("primary") or {})["interval"])
    market_type = data.get("market_type", "futures")
    source = data.get("source") or {}
    stype = source.get("type", "binance_rest")

    # 1) CLI override / declared input_json
    path = input_json or (source.get("path") if stype == "input_json" else None)
    if path:
        payload = json.loads(Path(path).read_text())
        rows = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
        return _rows_to_df(rows, price_keyed=True), f"input_json({Path(path).name})"

    # 2) historical parquet dir — MUST match both symbol and interval.
    #    Previously this fell back to `glob("*.parquet")` and took cands[0],
    #    so a directory holding only ETHUSDT data would happily "satisfy" a
    #    BTCUSDT spec and the whole backtest silently ran on the wrong symbol.
    #    Now: no match -> raise, and never load an unrelated file.
    if stype == "historical_parquet" and source.get("dir"):
        import pandas as pd
        d = Path(source["dir"])
        cands = sorted(
            f for f in d.rglob("*.parquet")
            if symbol.lower() in f.as_posix().lower()
            and _interval_matches(f, interval)
        )
        if not cands:
            available = sorted(f.name for f in d.rglob("*.parquet"))[:10]
            raise RuntimeError(
                f"historical_parquet: no file under {d} matches symbol={symbol} "
                f"interval={interval}. Refusing to fall back to an unrelated file "
                f"(that would backtest the wrong instrument). "
                f"Found instead: {available or 'nothing'}"
            )
        if len(cands) > 1:
            warnings.warn(
                f"historical_parquet: {len(cands)} files match {symbol} {interval} "
                f"under {d}; using {cands[0].name}. Narrow `data.source.dir` to "
                f"make this deterministic.",
                RuntimeWarning, stacklevel=2,
            )
        df = pd.read_parquet(cands[0]).reset_index(drop=True)
        df["timestamp"] = df.get("close_time", df.index)
        return df, f"parquet({cands[0].name})"

    # 3) binance_rest
    try:
        import requests
        host = _KLINES_HOST.get(market_type, _KLINES_HOST["futures"])
        r = requests.get(host, params={"symbol": symbol, "interval": interval,
                                       "limit": min(limit, 1000)}, timeout=20)
        r.raise_for_status()
        return _rows_to_df(r.json(), price_keyed=False), f"binance_{market_type}"
    except Exception as exc:
        fx = _fixture(symbol, interval)
        if fx is not None:
            # Loud: this is repo TEST DATA, not market data. Silently substituting
            # it made offline `run` look like a successful real backtest.
            warnings.warn(
                f"load_ohlcv: Binance REST failed ({type(exc).__name__}); falling back to the "
                f"repo TEST FIXTURE for {symbol} {interval}. Results are NOT real market data — "
                f"use data.source.type=input_json or historical_parquet for anything real.",
                RuntimeWarning, stacklevel=2,
            )
            return fx, f"fixture(offline:{type(exc).__name__})"
        raise RuntimeError(f"cannot load OHLCV for {symbol} {interval}: {exc}") from exc
