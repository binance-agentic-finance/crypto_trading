"""
Local historical storage and download helpers for Binance futures derivatives data.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from ._ssl_utils import resolve_ca_bundle

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised in environments without pyarrow
    pa = None
    pq = None


_FUTURES_OPEN_INTEREST_HIST_URL = "https://fapi.binance.com/futures/data/openInterestHist"
_FUTURES_FUNDING_RATE_HIST_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
_SUPPORTED_OPEN_INTEREST_PERIODS = {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"}


def ensure_pyarrow_available() -> None:
    if pa is None or pq is None:
        raise RuntimeError("pyarrow is required for derivatives parquet storage; install pyarrow first")


def build_derivatives_path(data_root: str, market_type: str, instrument_id: str, dataset_name: str) -> Path:
    return Path(data_root) / market_type / instrument_id.upper() / ("%s.parquet" % dataset_name)


def _open_interest_schema() -> "pa.Schema":
    ensure_pyarrow_available()
    return pa.schema(
        [
            ("timestamp", pa.int64()),
            ("instrument_id", pa.string()),
            ("period", pa.string()),
            ("open_interest", pa.float64()),
            ("open_interest_value", pa.float64()),
        ]
    )


def _funding_rate_schema() -> "pa.Schema":
    ensure_pyarrow_available()
    return pa.schema(
        [
            ("timestamp", pa.int64()),
            ("instrument_id", pa.string()),
            ("funding_rate", pa.float64()),
            ("mark_price", pa.float64()),
        ]
    )


def _read_factor_frame(path: str | Path) -> pd.DataFrame:
    ensure_pyarrow_available()
    table = pq.read_table(str(path))
    if table.num_rows == 0:
        return pd.DataFrame()
    return table.to_pandas().sort_values("timestamp", kind="stable").reset_index(drop=True)


def _resolve_open_interest_path(
    derivatives_root: str,
    market_type: str,
    instrument_id: str,
    timeframe: str,
) -> Optional[Path]:
    """Open-interest parquet for *timeframe*, or the finest one available.

    Open interest is stored per collection period, and the collector's period
    need not equal the backtest's bar size — the shipped data is
    ``open_interest_5m.parquet`` while specs are written on 1m and 1h bars.
    Requiring an exact filename match meant those runs silently produced **no
    open-interest column at all**: not an error, not a warning, just a strategy
    that quietly stopped consulting open interest.

    The merge downstream is ``merge_asof(direction="backward")``, which is
    correct for any period finer than the bar, so falling back to the finest
    available file is sound. It warns, because substituting data is exactly the
    kind of thing that should never happen quietly.
    """
    exact = build_derivatives_path(
        derivatives_root, market_type, instrument_id, "open_interest_%s" % timeframe
    )
    if exact.exists():
        return exact

    candidates = sorted(exact.parent.glob("open_interest_*.parquet"))
    if not candidates:
        return None
    chosen = min(candidates, key=lambda p: _timeframe_seconds(p.stem.split("_")[-1]))
    warnings.warn(
        "open_interest_%s.parquet not found under %s; using %s instead. "
        "It is merged as-of each bar close, so a finer period is safe — but a "
        "coarser one lags." % (timeframe, exact.parent, chosen.name),
        RuntimeWarning,
        stacklevel=3,
    )
    return chosen


def _timeframe_seconds(token: str) -> int:
    """Seconds in a ``5m`` / ``1h`` / ``1d`` token; unknown sorts last."""
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    try:
        return int(token[:-1]) * units[token[-1]]
    except (ValueError, KeyError, IndexError):
        return 1 << 30


def enrich_market_frame_with_derivatives(
    frame: pd.DataFrame,
    *,
    derivatives_root: Optional[str],
    market_type: str,
    instrument_id: str,
    timeframe: str,
) -> pd.DataFrame:
    if frame.empty or not derivatives_root or market_type != "futures":
        return frame

    enriched = frame.sort_values("close_time", kind="stable").reset_index(drop=True).copy()

    open_interest_path = _resolve_open_interest_path(
        derivatives_root, market_type, instrument_id, timeframe
    )
    if open_interest_path is not None and open_interest_path.exists():
        oi_frame = _read_factor_frame(open_interest_path)
        if not oi_frame.empty:
            oi_frame["open_interest"] = oi_frame["open_interest"].astype("float64")
            oi_frame["open_interest_value"] = oi_frame["open_interest_value"].astype("float64")
            oi_frame["oi_change_bps"] = oi_frame["open_interest"].pct_change() * 10_000.0
            oi_frame["oi_timestamp"] = oi_frame["timestamp"].astype("int64")
            enriched = pd.merge_asof(
                enriched,
                oi_frame[
                    ["timestamp", "oi_timestamp", "open_interest", "open_interest_value", "oi_change_bps"]
                ].sort_values("timestamp", kind="stable"),
                left_on="close_time",
                right_on="timestamp",
                direction="backward",
            )
            enriched = enriched.drop(columns=["timestamp"], errors="ignore")

    funding_rate_path = build_derivatives_path(
        derivatives_root,
        market_type,
        instrument_id,
        "funding_rate",
    )
    if funding_rate_path.exists():
        funding_frame = _read_factor_frame(funding_rate_path)
        if not funding_frame.empty:
            funding_frame["funding_rate"] = funding_frame["funding_rate"].astype("float64")
            funding_frame["mark_price"] = funding_frame["mark_price"].astype("float64")
            funding_frame["funding_rate_bps"] = funding_frame["funding_rate"] * 10_000.0
            funding_frame["funding_timestamp"] = funding_frame["timestamp"].astype("int64")
            enriched = pd.merge_asof(
                enriched,
                funding_frame[
                    ["timestamp", "funding_timestamp", "funding_rate", "funding_rate_bps", "mark_price"]
                ].sort_values("timestamp", kind="stable"),
                left_on="close_time",
                right_on="timestamp",
                direction="backward",
            )
            enriched = enriched.drop(columns=["timestamp"], errors="ignore")

    return enriched


@dataclass
class DerivativesDownloadResult:
    instrument_id: str
    market_type: str
    open_interest_path: Optional[str] = None
    open_interest_rows: int = 0
    funding_rate_path: Optional[str] = None
    funding_rate_rows: int = 0


@dataclass
class HistoricalBinanceDerivativesDownloader:
    data_root: str
    market_type: str = "futures"
    timeout: int = 30
    request_pause_sec: float = 0.0
    open_interest_limit: int = 500
    funding_rate_limit: int = 1000
    session: Optional[requests.Session] = None

    def download_bundle(
        self,
        *,
        instrument_id: str,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> DerivativesDownloadResult:
        if self.market_type != "futures":
            raise ValueError("derivatives downloader currently supports futures only")
        open_interest_result = self.download_open_interest_hist(
            instrument_id=instrument_id,
            period=timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        funding_rate_result = self.download_funding_rate_hist(
            instrument_id=instrument_id,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        return DerivativesDownloadResult(
            instrument_id=instrument_id.upper(),
            market_type=self.market_type,
            open_interest_path=open_interest_result["path"],
            open_interest_rows=int(open_interest_result["row_count"]),
            funding_rate_path=funding_rate_result["path"],
            funding_rate_rows=int(funding_rate_result["row_count"]),
        )

    def download_open_interest_hist(
        self,
        *,
        instrument_id: str,
        period: str,
        start_ts: int,
        end_ts: int,
    ) -> dict:
        ensure_pyarrow_available()
        if period not in _SUPPORTED_OPEN_INTEREST_PERIODS:
            raise ValueError("unsupported open interest period: %s" % period)
        if start_ts >= end_ts:
            raise ValueError("start_ts must be smaller than end_ts")

        target_path = build_derivatives_path(
            self.data_root,
            self.market_type,
            instrument_id,
            "open_interest_%s" % period,
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_suffix(".parquet.tmp")
        if temp_path.exists():
            temp_path.unlink()

        current_start = int(start_ts)
        schema = _open_interest_schema()
        current_end = int(end_ts)
        session = self.session or requests
        collected_records = []
        seen_timestamps = set()

        while current_end >= current_start:
            response = session.get(
                _FUTURES_OPEN_INTEREST_HIST_URL,
                params={
                    "symbol": instrument_id.upper(),
                    "period": period,
                    "limit": int(self.open_interest_limit),
                    "startTime": current_start,
                    "endTime": current_end,
                },
                timeout=self.timeout,
                verify=resolve_ca_bundle(),
            )
            response.raise_for_status()
            rows = response.json()
            if not rows:
                break

            earliest_timestamp = None
            for row in rows:
                timestamp = int(row["timestamp"])
                if timestamp < int(start_ts) or timestamp > int(end_ts) or timestamp in seen_timestamps:
                    continue
                if earliest_timestamp is None or timestamp < earliest_timestamp:
                    earliest_timestamp = timestamp
                seen_timestamps.add(timestamp)
                collected_records.append(
                    {
                        "timestamp": timestamp,
                        "instrument_id": instrument_id.upper(),
                        "period": period,
                        "open_interest": float(row["sumOpenInterest"]),
                        "open_interest_value": float(row["sumOpenInterestValue"]),
                    }
                )

            if earliest_timestamp is None:
                break
            current_end = earliest_timestamp - 1
            if self.request_pause_sec > 0:
                time.sleep(self.request_pause_sec)

        if not collected_records:
            if temp_path.exists():
                temp_path.unlink()
            raise ValueError(
                "no open interest history returned for %s %s in [%s, %s]"
                % (instrument_id, period, start_ts, end_ts)
            )

        collected_records.sort(key=lambda item: int(item["timestamp"]))
        table = pa.Table.from_pylist(collected_records, schema=schema)
        pq.write_table(table, str(temp_path), compression="zstd")
        os.replace(temp_path, target_path)
        return {"path": str(target_path), "row_count": int(len(collected_records))}

    def download_funding_rate_hist(
        self,
        *,
        instrument_id: str,
        start_ts: int,
        end_ts: int,
    ) -> dict:
        ensure_pyarrow_available()
        if start_ts >= end_ts:
            raise ValueError("start_ts must be smaller than end_ts")

        target_path = build_derivatives_path(
            self.data_root,
            self.market_type,
            instrument_id,
            "funding_rate",
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_suffix(".parquet.tmp")
        if temp_path.exists():
            temp_path.unlink()

        current_start = int(start_ts)
        row_count = 0
        schema = _funding_rate_schema()
        writer = None
        session = self.session or requests

        try:
            while current_start <= int(end_ts):
                response = session.get(
                    _FUTURES_FUNDING_RATE_HIST_URL,
                    params={
                        "symbol": instrument_id.upper(),
                        "limit": int(self.funding_rate_limit),
                        "startTime": current_start,
                        "endTime": int(end_ts),
                    },
                    timeout=self.timeout,
                    verify=resolve_ca_bundle(),
                )
                response.raise_for_status()
                rows = response.json()
                if not rows:
                    break

                records = []
                for row in rows:
                    timestamp = int(row["fundingTime"])
                    if timestamp < int(start_ts) or timestamp > int(end_ts):
                        continue
                    records.append(
                        {
                            "timestamp": timestamp,
                            "instrument_id": instrument_id.upper(),
                            "funding_rate": float(row["fundingRate"]),
                            "mark_price": float(row.get("markPrice", 0.0)),
                        }
                    )

                if records:
                    table = pa.Table.from_pylist(records, schema=schema)
                    if writer is None:
                        writer = pq.ParquetWriter(str(temp_path), schema, compression="zstd")
                    writer.write_table(table)
                    row_count += len(records)

                last_timestamp = int(rows[-1]["fundingTime"])
                next_start = last_timestamp + 1
                if next_start <= current_start:
                    break
                current_start = next_start
                if self.request_pause_sec > 0:
                    time.sleep(self.request_pause_sec)
        finally:
            if writer is not None:
                writer.close()

        if row_count == 0:
            if temp_path.exists():
                temp_path.unlink()
            raise ValueError(
                "no funding rate history returned for %s in [%s, %s]"
                % (instrument_id, start_ts, end_ts)
            )

        os.replace(temp_path, target_path)
        return {"path": str(target_path), "row_count": int(row_count)}
