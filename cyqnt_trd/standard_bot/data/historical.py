"""
Local historical storage helpers for the standard bot data layer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..core import Bar, BundleMeta, MarketBundle, MarketQuery
from .alignment import timeframe_to_ms
from .derivatives import enrich_market_frame_with_derivatives
from .liquidations import enrich_market_frame_with_liquidations

try:
    import polars as pl
except ImportError:  # pragma: no cover - exercised in environments without polars
    pl = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised in environments without pyarrow
    pa = None
    pq = None


SUPPORTED_RESAMPLE_TIMEFRAMES = {"5m", "15m", "1h", "4h"}
PARQUET_COLUMNS = [
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
]


def ensure_pyarrow_available() -> None:
    if pa is None or pq is None:
        raise RuntimeError("pyarrow is required for parquet historical storage; install pyarrow first")


def polars_available() -> bool:
    return pl is not None


def build_history_path(data_root: str, market_type: str, instrument_id: str, timeframe: str) -> Path:
    return Path(data_root) / market_type / instrument_id.upper() / ("%s.parquet" % timeframe)


def determine_download_timeframes(market_query: MarketQuery, storage_timeframe: str) -> List[str]:
    requested = set(market_query.timeframes)
    download_timeframes: List[str] = []
    if storage_timeframe == "1m":
        supported_targets = requested & (SUPPORTED_RESAMPLE_TIMEFRAMES | {"1m"})
        if supported_targets:
            download_timeframes.append("1m")
            requested -= supported_targets
    download_timeframes.extend(sorted(requested))
    return download_timeframes


def determine_download_start_ts(market_query: MarketQuery, storage_timeframe: str) -> Optional[int]:
    if market_query.time_range.start_ts is None:
        return None
    start_ts = int(market_query.time_range.start_ts)
    if storage_timeframe != "1m":
        return start_ts
    requested_resamples = [tf for tf in market_query.timeframes if tf in SUPPORTED_RESAMPLE_TIMEFRAMES]
    if not requested_resamples:
        return start_ts
    max_interval_ms = max(timeframe_to_ms(tf) for tf in requested_resamples)
    return max(0, start_ts - max_interval_ms)


def _tail_rows_for_resample(target_timeframe: str, tail_bars: Optional[int]) -> Optional[int]:
    if tail_bars is None:
        return None
    ratio = timeframe_to_ms(target_timeframe) // timeframe_to_ms("1m")
    return int(tail_bars) * int(ratio) + int(ratio)


def _row_group_column_index(parquet_file: "pq.ParquetFile", column_name: str) -> Optional[int]:
    names = parquet_file.schema_arrow.names
    if column_name not in names:
        return None
    return names.index(column_name)


def _row_group_time_bounds(parquet_file: "pq.ParquetFile", row_group_index: int) -> tuple[Optional[int], Optional[int]]:
    metadata = parquet_file.metadata.row_group(row_group_index)
    open_index = _row_group_column_index(parquet_file, "open_time")
    close_index = _row_group_column_index(parquet_file, "close_time")
    if open_index is None or close_index is None:
        return None, None

    open_stats = metadata.column(open_index).statistics
    close_stats = metadata.column(close_index).statistics
    if open_stats is None or close_stats is None:
        return None, None
    if open_stats.min is None or close_stats.max is None:
        return None, None
    return int(open_stats.min), int(close_stats.max)


def parquet_time_coverage(path: str | Path) -> tuple[Optional[int], Optional[int]]:
    ensure_pyarrow_available()
    parquet_file = pq.ParquetFile(str(path))
    if parquet_file.metadata.num_row_groups == 0:
        return None, None

    min_open = None
    max_close = None
    for index in range(parquet_file.metadata.num_row_groups):
        open_min, close_max = _row_group_time_bounds(parquet_file, index)
        if open_min is None or close_max is None:
            table = parquet_file.read_row_groups([index], columns=["open_time", "close_time"])
            if table.num_rows == 0:
                continue
            frame = table.to_pandas()
            open_min = int(frame["open_time"].min())
            close_max = int(frame["close_time"].max())
        min_open = open_min if min_open is None else min(min_open, int(open_min))
        max_close = close_max if max_close is None else max(max_close, int(close_max))
    return min_open, max_close


def _select_row_groups(
    parquet_file: "pq.ParquetFile",
    *,
    start_ts: Optional[int],
    end_ts: Optional[int],
    tail_rows: Optional[int],
) -> Optional[List[int]]:
    row_group_count = parquet_file.metadata.num_row_groups
    if row_group_count == 0:
        return []

    if tail_rows is not None and start_ts is None and end_ts is None:
        selected: List[int] = []
        total_rows = 0
        for index in range(row_group_count - 1, -1, -1):
            selected.append(index)
            total_rows += parquet_file.metadata.row_group(index).num_rows
            if total_rows >= int(tail_rows):
                break
        return sorted(selected)

    selected = []
    for index in range(row_group_count):
        open_min, close_max = _row_group_time_bounds(parquet_file, index)
        if open_min is None or close_max is None:
            return None
        if start_ts is not None and close_max < int(start_ts):
            continue
        if end_ts is not None and open_min > int(end_ts):
            continue
        selected.append(index)
    return selected


def read_parquet_frame(
    path: str | Path,
    *,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    tail_rows: Optional[int] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    ensure_pyarrow_available()
    parquet_file = pq.ParquetFile(str(path))
    selected_row_groups = _select_row_groups(
        parquet_file,
        start_ts=start_ts,
        end_ts=end_ts,
        tail_rows=tail_rows,
    )
    if selected_row_groups == []:
        return pd.DataFrame(columns=columns or PARQUET_COLUMNS)

    if selected_row_groups is None:
        filters = []
        if start_ts is not None:
            filters.append(("close_time", ">=", int(start_ts)))
        if end_ts is not None:
            filters.append(("close_time", "<=", int(end_ts)))
        table = pq.read_table(
            str(path),
            columns=columns or PARQUET_COLUMNS,
            filters=filters or None,
        )
    else:
        table = parquet_file.read_row_groups(selected_row_groups, columns=columns or PARQUET_COLUMNS)
    if table.num_rows == 0:
        return pd.DataFrame(columns=columns or PARQUET_COLUMNS)
    if polars_available():
        frame = pl.from_arrow(table).sort("close_time")
        if start_ts is not None:
            frame = frame.filter(pl.col("close_time") >= int(start_ts))
        if end_ts is not None:
            frame = frame.filter(pl.col("close_time") <= int(end_ts))
        if tail_rows is not None:
            frame = frame.tail(int(tail_rows))
        return frame.to_pandas()

    frame = table.to_pandas()
    frame = frame.sort_values("close_time", kind="stable").reset_index(drop=True)
    if start_ts is not None:
        frame = frame[frame["close_time"] >= int(start_ts)]
    if end_ts is not None:
        frame = frame[frame["close_time"] <= int(end_ts)]
    if tail_rows is not None:
        frame = frame.tail(int(tail_rows))
    return frame.reset_index(drop=True)


def resample_frame_from_1m(frame: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
    if target_timeframe not in SUPPORTED_RESAMPLE_TIMEFRAMES:
        raise ValueError("unsupported local resample timeframe: %s" % target_timeframe)
    if frame.empty:
        return pd.DataFrame(columns=PARQUET_COLUMNS)

    target_ms = timeframe_to_ms(target_timeframe)
    base_ms = timeframe_to_ms("1m")
    if target_ms % base_ms != 0:
        raise ValueError("target timeframe %s is not a multiple of 1m" % target_timeframe)

    expected_rows = target_ms // base_ms
    if polars_available():
        aggregated = (
            pl.from_pandas(frame)
            .sort("open_time")
            .with_columns(((pl.col("open_time") // target_ms) * target_ms).alias("bucket_open_time"))
            .group_by("bucket_open_time", maintain_order=True)
            .agg(
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("quote_volume").sum().alias("quote_volume"),
                pl.col("trades").sum().alias("trades"),
                pl.len().alias("row_count"),
                pl.col("instrument_id").last().alias("instrument_id"),
            )
            .filter(pl.col("row_count") == expected_rows)
        )
        if aggregated.is_empty():
            return pd.DataFrame(columns=PARQUET_COLUMNS)
        aggregated = (
            aggregated.with_columns(
                pl.col("bucket_open_time").cast(pl.Int64).alias("open_time"),
                (pl.col("bucket_open_time").cast(pl.Int64) + target_ms - 1).alias("close_time"),
                pl.lit(target_timeframe).alias("timeframe"),
                pl.lit(True).alias("confirmed"),
            )
            .select(
                [
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
                ]
            )
        )
        return aggregated.to_pandas()

    working = frame.sort_values("open_time", kind="stable").copy()
    working["bucket_open_time"] = (working["open_time"] // target_ms) * target_ms

    aggregated = (
        working.groupby("bucket_open_time", sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            quote_volume=("quote_volume", "sum"),
            trades=("trades", "sum"),
            row_count=("open_time", "size"),
            instrument_id=("instrument_id", "last"),
        )
        .reset_index()
    )
    aggregated = aggregated[aggregated["row_count"] == expected_rows].copy()
    if aggregated.empty:
        return pd.DataFrame(columns=PARQUET_COLUMNS)

    aggregated["open_time"] = aggregated["bucket_open_time"].astype("int64")
    aggregated["close_time"] = aggregated["open_time"] + target_ms - 1
    aggregated["timeframe"] = target_timeframe
    aggregated["confirmed"] = True
    aggregated = aggregated[
        [
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
        ]
    ]
    return aggregated.reset_index(drop=True)


def _row_to_bar(row: pd.Series) -> Bar:
    instrument_id = str(row.get("instrument_id") or "").upper()
    timeframe = str(row.get("timeframe") or "")
    close_time = int(row["close_time"])
    open_time = int(row.get("open_time", close_time))
    extras = {
        "open_time": open_time,
        "close_time": close_time,
        "trades": int(row.get("trades", 0)),
    }
    standard_columns = {
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
    }
    for column, value in row.items():
        if column in standard_columns:
            continue
        if pd.isna(value):
            continue
        if isinstance(value, bool):
            extras[column] = bool(value)
        elif isinstance(value, Integral):
            extras[column] = int(value)
        elif isinstance(value, Real):
            extras[column] = float(value)
        elif hasattr(value, "item"):
            normalized = value.item()
            if normalized is not None:
                extras[column] = normalized
        else:
            extras[column] = value
    return Bar(
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]),
        quote_volume=float(row["quote_volume"]) if row.get("quote_volume") is not None else None,
        timestamp=close_time,
        instrument_id=instrument_id,
        timeframe=timeframe,
        confirmed=bool(row.get("confirmed", True)),
        extras=extras,
    )


@dataclass
class HistoricalParquetMarketDataAdapter:
    data_root: str
    market_type: str = "spot"
    resample_source_timeframe: str = "1m"
    derivatives_data_root: Optional[str] = None
    liquidations_data_root: Optional[str] = None

    def fetch_market(self, market_query: MarketQuery) -> MarketBundle:
        series: Dict[str, List[Bar]] = {}
        for instrument_id in market_query.instruments:
            for timeframe in market_query.timeframes:
                bars = self._load_bars(
                    instrument_id=instrument_id.upper(),
                    timeframe=timeframe,
                    market_query=market_query,
                )
                series[MarketBundle.key(instrument_id.upper(), timeframe)] = bars

        return MarketBundle(
            bars=series,
            meta=BundleMeta(
                data_source="historical_parquet",
                fetched_at=int(time.time() * 1000),
                extras={"data_root": self.data_root, "market_type": self.market_type},
            ),
        )

    def _load_bars(self, *, instrument_id: str, timeframe: str, market_query: MarketQuery) -> List[Bar]:
        direct_path = build_history_path(self.data_root, self.market_type, instrument_id, timeframe)
        frame: Optional[pd.DataFrame] = None
        source_description = "direct"

        if direct_path.exists():
            self._assert_requested_range_covered(
                path=direct_path,
                requested_start_ts=market_query.time_range.start_ts,
                requested_end_ts=market_query.time_range.end_ts,
            )
            frame = read_parquet_frame(
                direct_path,
                start_ts=market_query.time_range.start_ts,
                end_ts=market_query.time_range.end_ts,
                tail_rows=market_query.time_range.tail_bars if market_query.time_range.start_ts is None and market_query.time_range.end_ts is None else None,
            )
        else:
            frame = self._load_resampled_frame(
                instrument_id=instrument_id,
                target_timeframe=timeframe,
                market_query=market_query,
            )
            source_description = "resampled"

        frame = self._apply_time_range(frame, market_query)
        frame = enrich_market_frame_with_derivatives(
            frame,
            derivatives_root=self.derivatives_data_root,
            market_type=self.market_type,
            instrument_id=instrument_id,
            timeframe=timeframe,
        )
        frame = enrich_market_frame_with_liquidations(
            frame,
            liquidations_root=self.liquidations_data_root,
            market_type=self.market_type,
            instrument_id=instrument_id,
            timeframe=timeframe,
        )
        if frame.empty:
            return []

        bars = [_row_to_bar(row) for _, row in frame.iterrows()]
        if source_description == "resampled":
            for bar in bars:
                bar.extras["source"] = "resampled_from_%s" % self.resample_source_timeframe
        return bars

    def _load_resampled_frame(
        self,
        *,
        instrument_id: str,
        target_timeframe: str,
        market_query: MarketQuery,
    ) -> pd.DataFrame:
        if self.resample_source_timeframe != "1m" or target_timeframe not in SUPPORTED_RESAMPLE_TIMEFRAMES:
            raise FileNotFoundError(
                "no local parquet file for %s %s and local resample is unavailable"
                % (instrument_id, target_timeframe)
            )

        source_path = build_history_path(
            self.data_root,
            self.market_type,
            instrument_id,
            self.resample_source_timeframe,
        )
        if not source_path.exists():
            raise FileNotFoundError(
                "missing local parquet source %s for resampling %s %s"
                % (source_path, instrument_id, target_timeframe)
            )

        target_ms = timeframe_to_ms(target_timeframe)
        lookback_ms = target_ms
        source_start_ts = None
        source_tail_rows = None
        if market_query.time_range.start_ts is not None:
            source_start_ts = max(0, int(market_query.time_range.start_ts) - lookback_ms)
        elif market_query.time_range.tail_bars is not None:
            source_tail_rows = _tail_rows_for_resample(target_timeframe, market_query.time_range.tail_bars)

        self._assert_requested_range_covered(
            path=source_path,
            requested_start_ts=source_start_ts,
            requested_end_ts=market_query.time_range.end_ts,
        )
        source_frame = read_parquet_frame(
            source_path,
            start_ts=source_start_ts,
            end_ts=market_query.time_range.end_ts,
            tail_rows=source_tail_rows,
        )
        return resample_frame_from_1m(source_frame, target_timeframe)

    def _apply_time_range(self, frame: pd.DataFrame, market_query: MarketQuery) -> pd.DataFrame:
        if frame.empty:
            return frame
        filtered = frame
        if market_query.time_range.start_ts is not None:
            filtered = filtered[filtered["close_time"] >= int(market_query.time_range.start_ts)]
        if market_query.time_range.end_ts is not None:
            filtered = filtered[filtered["close_time"] <= int(market_query.time_range.end_ts)]
        if market_query.time_range.tail_bars is not None:
            filtered = filtered.tail(int(market_query.time_range.tail_bars))
        return filtered.reset_index(drop=True)

    def _assert_requested_range_covered(
        self,
        *,
        path: str | Path,
        requested_start_ts: Optional[int],
        requested_end_ts: Optional[int],
    ) -> None:
        if requested_start_ts is None and requested_end_ts is None:
            return
        min_open, max_close = parquet_time_coverage(path)
        if min_open is None or max_close is None:
            raise FileNotFoundError("local parquet has no readable time coverage: %s" % path)
        if requested_start_ts is not None and int(requested_start_ts) < int(min_open):
            raise FileNotFoundError(
                "local parquet starts too late for requested range: %s < %s" % (requested_start_ts, min_open)
            )
        if requested_end_ts is not None and int(requested_end_ts) > int(max_close) + 1:
            raise FileNotFoundError(
                "local parquet ends too early for requested range: %s > %s" % (requested_end_ts, max_close)
            )


@dataclass
class LocalFirstMarketDataAdapter:
    local_adapter: HistoricalParquetMarketDataAdapter
    storage_timeframe: str = "1m"
    remote_adapter: Optional[object] = None
    downloader: Optional[object] = None

    def fetch_market(self, market_query: MarketQuery) -> MarketBundle:
        try:
            return self.local_adapter.fetch_market(market_query)
        except FileNotFoundError:
            if self.downloader is not None:
                if market_query.time_range.start_ts is None or market_query.time_range.end_ts is None:
                    raise
                download_start_ts = determine_download_start_ts(market_query, self.storage_timeframe)
                for timeframe in determine_download_timeframes(market_query, self.storage_timeframe):
                    self.downloader.download(
                        instrument_id=market_query.instruments[0],
                        timeframe=timeframe,
                        start_ts=download_start_ts if download_start_ts is not None else market_query.time_range.start_ts,
                        end_ts=market_query.time_range.end_ts,
                    )
                return self.local_adapter.fetch_market(market_query)
            if self.remote_adapter is not None:
                return self.remote_adapter.fetch_market(market_query)
            raise
