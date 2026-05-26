"""Historical Binance derivatives downloader (open interest + funding rate) → parquet.

Port of `cyqnt_trd.standard_bot.data.derivatives`. Two endpoints:
- `https://fapi.binance.com/futures/data/openInterestHist` — chunked OI
- `https://fapi.binance.com/fapi/v1/fundingRate` — chunked funding rate

Output parquet shape matches the source so `enrich_market_frame_with_derivatives`
(also ported here) can merge OI / funding into kline DataFrames at the
same shape downstream consumers expect.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


_FUTURES_OPEN_INTEREST_HIST_URL = "https://fapi.binance.com/futures/data/openInterestHist"
_FUTURES_FUNDING_RATE_HIST_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
_SUPPORTED_OPEN_INTEREST_PERIODS = ("5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d")


def _ensure_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        return pa, pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required; install with "
            "`pip install ai-pro-trading-library[data]`"
        ) from exc


def build_derivatives_path(
    data_root: str | Path,
    market_type: str,
    instrument_id: str,
    dataset_name: str,
) -> Path:
    return Path(data_root) / market_type / instrument_id.upper() / f"{dataset_name}.parquet"


def _open_interest_schema():
    pa, _ = _ensure_pyarrow()
    return pa.schema([
        ("timestamp", pa.int64()),
        ("instrument_id", pa.string()),
        ("period", pa.string()),
        ("open_interest", pa.float64()),
        ("open_interest_value", pa.float64()),
    ])


def _funding_rate_schema():
    pa, _ = _ensure_pyarrow()
    return pa.schema([
        ("timestamp", pa.int64()),
        ("instrument_id", pa.string()),
        ("funding_rate", pa.float64()),
        ("mark_price", pa.float64()),
    ])


def _read_factor_frame(path: str | Path) -> pd.DataFrame:
    _, pq = _ensure_pyarrow()
    table = pq.read_table(str(path))
    if table.num_rows == 0:
        return pd.DataFrame()
    return table.to_pandas().sort_values("timestamp", kind="stable").reset_index(drop=True)


def enrich_market_frame_with_derivatives(
    frame: pd.DataFrame,
    *,
    derivatives_root: Optional[str | Path],
    market_type: str,
    instrument_id: str,
    timeframe: str,
) -> pd.DataFrame:
    """Left-join open_interest + funding_rate parquet onto a kline frame.

    The merge is `merge_asof` on `close_time` with direction="backward",
    matching the source's behavior for point-in-time alignment.
    """
    if frame.empty or not derivatives_root or market_type != "futures":
        return frame

    enriched = frame.sort_values("close_time", kind="stable").reset_index(drop=True).copy()

    oi_path = build_derivatives_path(
        derivatives_root, market_type, instrument_id, f"open_interest_{timeframe}"
    )
    if oi_path.exists():
        oi_frame = _read_factor_frame(oi_path)
        if not oi_frame.empty:
            oi_frame["open_interest"] = oi_frame["open_interest"].astype("float64")
            oi_frame["open_interest_value"] = oi_frame["open_interest_value"].astype("float64")
            oi_frame["oi_change_bps"] = oi_frame["open_interest"].pct_change() * 10_000.0
            oi_frame["oi_timestamp"] = oi_frame["timestamp"].astype("int64")
            enriched = pd.merge_asof(
                enriched,
                oi_frame[["timestamp", "oi_timestamp", "open_interest",
                          "open_interest_value", "oi_change_bps"]]
                .sort_values("timestamp", kind="stable"),
                left_on="close_time",
                right_on="timestamp",
                direction="backward",
            ).drop(columns=["timestamp"], errors="ignore")

    funding_path = build_derivatives_path(
        derivatives_root, market_type, instrument_id, "funding_rate"
    )
    if funding_path.exists():
        funding_frame = _read_factor_frame(funding_path)
        if not funding_frame.empty:
            funding_frame["funding_rate"] = funding_frame["funding_rate"].astype("float64")
            funding_frame["mark_price"] = funding_frame["mark_price"].astype("float64")
            funding_frame["funding_rate_bps"] = funding_frame["funding_rate"] * 10_000.0
            funding_frame["funding_timestamp"] = funding_frame["timestamp"].astype("int64")
            enriched = pd.merge_asof(
                enriched,
                funding_frame[["timestamp", "funding_timestamp", "funding_rate",
                               "funding_rate_bps", "mark_price"]]
                .sort_values("timestamp", kind="stable"),
                left_on="close_time",
                right_on="timestamp",
                direction="backward",
            ).drop(columns=["timestamp"], errors="ignore")

    return enriched


@dataclass(frozen=True)
class DerivativesDownloadResult:
    instrument_id: str
    market_type: str
    open_interest_path: Optional[str] = None
    open_interest_rows: int = 0
    funding_rate_path: Optional[str] = None
    funding_rate_rows: int = 0


@dataclass
class BinanceDerivativesDownloader:
    """Open-interest history + funding-rate history → parquet.

    Field-set parity with `cyqnt_trd.standard_bot.data.derivatives.
    HistoricalBinanceDerivativesDownloader`. Output parquet files are
    consumed by `enrich_market_frame_with_derivatives` to enrich kline
    DataFrames with OI and funding columns.
    """

    data_root: str
    market_type: str = "futures"
    timeout: int = 30
    request_pause_sec: float = 0.0
    open_interest_limit: int = 500
    funding_rate_limit: int = 1000
    session: object | None = None

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
        oi_result = self.download_open_interest_hist(
            instrument_id=instrument_id, period=timeframe, start_ts=start_ts, end_ts=end_ts
        )
        funding_result = self.download_funding_rate_hist(
            instrument_id=instrument_id, start_ts=start_ts, end_ts=end_ts
        )
        return DerivativesDownloadResult(
            instrument_id=instrument_id.upper(),
            market_type=self.market_type,
            open_interest_path=oi_result["path"],
            open_interest_rows=int(oi_result["row_count"]),
            funding_rate_path=funding_result["path"],
            funding_rate_rows=int(funding_result["row_count"]),
        )

    def download_open_interest_hist(
        self,
        *,
        instrument_id: str,
        period: str,
        start_ts: int,
        end_ts: int,
    ) -> dict:
        pa, pq = _ensure_pyarrow()
        if period not in _SUPPORTED_OPEN_INTEREST_PERIODS:
            raise ValueError(f"unsupported open interest period: {period}")
        if start_ts >= end_ts:
            raise ValueError("start_ts must be smaller than end_ts")

        target_path = build_derivatives_path(
            self.data_root, self.market_type, instrument_id, f"open_interest_{period}"
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_suffix(".parquet.tmp")
        if temp_path.exists():
            temp_path.unlink()

        current_end = int(end_ts)
        schema = _open_interest_schema()
        session = self._session()
        collected: list[dict] = []
        seen: set[int] = set()

        while current_end >= int(start_ts):
            response = session.get(
                _FUTURES_OPEN_INTEREST_HIST_URL,
                params={
                    "symbol": instrument_id.upper(),
                    "period": period,
                    "limit": int(self.open_interest_limit),
                    "startTime": int(start_ts),
                    "endTime": current_end,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            rows = response.json()
            if not rows:
                break
            earliest = None
            for row in rows:
                ts = int(row["timestamp"])
                if ts < int(start_ts) or ts > int(end_ts) or ts in seen:
                    continue
                if earliest is None or ts < earliest:
                    earliest = ts
                seen.add(ts)
                collected.append({
                    "timestamp": ts,
                    "instrument_id": instrument_id.upper(),
                    "period": period,
                    "open_interest": float(row["sumOpenInterest"]),
                    "open_interest_value": float(row["sumOpenInterestValue"]),
                })
            if earliest is None:
                break
            current_end = earliest - 1
            if self.request_pause_sec > 0:
                time.sleep(self.request_pause_sec)

        if not collected:
            if temp_path.exists():
                temp_path.unlink()
            raise ValueError(
                f"no open interest history returned for {instrument_id} {period} "
                f"in [{start_ts}, {end_ts}]"
            )

        collected.sort(key=lambda item: int(item["timestamp"]))
        table = pa.Table.from_pylist(collected, schema=schema)
        pq.write_table(table, str(temp_path), compression="zstd")
        os.replace(temp_path, target_path)
        return {"path": str(target_path), "row_count": int(len(collected))}

    def download_funding_rate_hist(
        self,
        *,
        instrument_id: str,
        start_ts: int,
        end_ts: int,
    ) -> dict:
        pa, pq = _ensure_pyarrow()
        if start_ts >= end_ts:
            raise ValueError("start_ts must be smaller than end_ts")

        target_path = build_derivatives_path(
            self.data_root, self.market_type, instrument_id, "funding_rate"
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_suffix(".parquet.tmp")
        if temp_path.exists():
            temp_path.unlink()

        current_start = int(start_ts)
        row_count = 0
        schema = _funding_rate_schema()
        writer = None
        session = self._session()

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
                )
                response.raise_for_status()
                rows = response.json()
                if not rows:
                    break
                records = []
                for row in rows:
                    ts = int(row["fundingTime"])
                    if ts < int(start_ts) or ts > int(end_ts):
                        continue
                    records.append({
                        "timestamp": ts,
                        "instrument_id": instrument_id.upper(),
                        "funding_rate": float(row["fundingRate"]),
                        "mark_price": float(row.get("markPrice", 0.0)),
                    })
                if records:
                    table = pa.Table.from_pylist(records, schema=schema)
                    if writer is None:
                        writer = pq.ParquetWriter(str(temp_path), schema, compression="zstd")
                    writer.write_table(table)
                    row_count += len(records)
                last_ts = int(rows[-1]["fundingTime"])
                next_start = last_ts + 1
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
                f"no funding rate history returned for {instrument_id} "
                f"in [{start_ts}, {end_ts}]"
            )

        os.replace(temp_path, target_path)
        return {"path": str(target_path), "row_count": int(row_count)}

    def _session(self):
        if self.session is not None:
            return self.session
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "requests is required; install with "
                "`pip install ai-pro-trading-library[data]`"
            ) from exc
        return requests


__all__ = [
    "BinanceDerivativesDownloader",
    "DerivativesDownloadResult",
    "build_derivatives_path",
    "enrich_market_frame_with_derivatives",
]
