"""Historical Binance kline downloader → parquet.

Port of `cyqnt_trd.standard_bot.data.downloader.HistoricalBinanceDownloader`
with the polars / CME / HuggingFace branches stripped — a single
canonical Binance-only path.

Wire format and parquet schema match the source so files written here
drop into `library.data.adapters.LocalParquetAdapter` unchanged.

Spot vs. futures is selected via `market_type`. The kline endpoint URL
is hardcoded to the corresponding Binance public REST host (no auth
needed for klines).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ai_pro_trading_library.library.data.alignment import timeframe_to_ms


_FAPI_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
_SPOT_KLINES_URL = "https://api.binance.com/api/v3/klines"


PARQUET_COLUMNS = (
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "instrument_id",
    "timeframe",
    "confirmed",
)


def _parquet_schema():
    """Lazy build the pyarrow schema. Raises a clear error if pyarrow
    isn't installed (the `[data]` extra)."""
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for parquet historical storage; "
            "install with `pip install ai-pro-trading-library[data]`"
        ) from exc
    return pa.schema(
        [
            ("open_time", pa.int64()),
            ("close_time", pa.int64()),
            ("open", pa.float64()),
            ("high", pa.float64()),
            ("low", pa.float64()),
            ("close", pa.float64()),
            ("volume", pa.float64()),
            ("quote_volume", pa.float64()),
            ("trades", pa.int64()),
            ("instrument_id", pa.string()),
            ("timeframe", pa.string()),
            ("confirmed", pa.bool_()),
        ]
    )


def build_history_path(
    data_root: str | Path,
    market_type: str,
    instrument_id: str,
    timeframe: str,
) -> Path:
    """Path layout: <data_root>/<market_type>/<SYMBOL>/<timeframe>.parquet."""
    return Path(data_root) / market_type / instrument_id.upper() / f"{timeframe}.parquet"


@dataclass(frozen=True)
class KlineDownloadResult:
    path: str
    row_count: int
    chunk_count: int
    instrument_id: str
    timeframe: str
    market_type: str
    start_ts: int
    end_ts: int


@dataclass
class BinanceKlineDownloader:
    """Chunked kline downloader that streams parquet rows to disk.

    Construct with a `data_root` directory; each call to `.download(...)`
    writes (or replaces) one parquet file at the canonical path.

    Field-set parity with `cyqnt_trd.standard_bot.data.downloader.
    HistoricalBinanceDownloader`. Output parquet shape is byte-compatible
    with files produced by the source downloader so consumers can read
    either via `LocalParquetAdapter`.
    """

    data_root: str
    market_type: str = "spot"  # "spot" | "futures"
    timeout: int = 30
    chunk_limit: int = 1000
    request_pause_sec: float = 0.0
    base_url_override: Optional[str] = None
    session: object | None = None  # optional `requests.Session`-like

    @property
    def base_url(self) -> str:
        if self.base_url_override:
            return self.base_url_override
        return _FAPI_KLINES_URL if self.market_type == "futures" else _SPOT_KLINES_URL

    def download(
        self,
        *,
        instrument_id: str,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> KlineDownloadResult:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "pyarrow is required; install with "
                "`pip install ai-pro-trading-library[data]`"
            ) from exc
        if start_ts >= end_ts:
            raise ValueError("start_ts must be smaller than end_ts")

        target_path = build_history_path(
            self.data_root, self.market_type, instrument_id, timeframe
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_suffix(".parquet.tmp")
        if temp_path.exists():
            temp_path.unlink()

        interval_ms = timeframe_to_ms(timeframe)
        current_start = int(start_ts)
        row_count = 0
        chunk_count = 0
        writer = None
        schema = _parquet_schema()

        try:
            while current_start <= int(end_ts):
                rows = self._fetch_chunk(
                    instrument_id=instrument_id,
                    timeframe=timeframe,
                    start_ts=current_start,
                    end_ts=int(end_ts),
                )
                if not rows:
                    break

                records = []
                for row in rows:
                    open_time = int(row[0])
                    close_time = int(row[6])
                    if close_time < int(start_ts) or close_time > int(end_ts):
                        continue
                    records.append({
                        "open_time": open_time,
                        "close_time": close_time,
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                        "volume": float(row[5]),
                        "quote_volume": float(row[7]) if len(row) > 7 else 0.0,
                        "trades": int(row[8]) if len(row) > 8 else 0,
                        "instrument_id": instrument_id.upper(),
                        "timeframe": timeframe,
                        "confirmed": True,
                    })

                if records:
                    table = pa.Table.from_pylist(records, schema=schema)
                    if writer is None:
                        writer = pq.ParquetWriter(str(temp_path), schema, compression="zstd")
                    writer.write_table(table)
                    row_count += len(records)
                    chunk_count += 1

                last_open_time = int(rows[-1][0])
                next_start = last_open_time + interval_ms
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
                f"no klines returned for {instrument_id} {timeframe} "
                f"in [{start_ts}, {end_ts}]"
            )

        os.replace(temp_path, target_path)
        return KlineDownloadResult(
            path=str(target_path),
            row_count=row_count,
            chunk_count=chunk_count,
            instrument_id=instrument_id.upper(),
            timeframe=timeframe,
            market_type=self.market_type,
            start_ts=int(start_ts),
            end_ts=int(end_ts),
        )

    def _fetch_chunk(
        self,
        *,
        instrument_id: str,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> list:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "requests is required; install with "
                "`pip install ai-pro-trading-library[data]`"
            ) from exc
        params = {
            "symbol": instrument_id.upper(),
            "interval": timeframe,
            "startTime": int(start_ts),
            "endTime": int(end_ts),
            "limit": int(self.chunk_limit),
        }
        session = self.session or requests
        response = session.get(self.base_url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()


__all__ = [
    "BinanceKlineDownloader",
    "KlineDownloadResult",
    "build_history_path",
]
