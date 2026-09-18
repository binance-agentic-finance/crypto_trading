from __future__ import annotations

import sys
from argparse import Namespace

import pytest

pyarrow = pytest.importorskip("pyarrow")
pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

from cyqnt_trd.standard_bot.core import (  # noqa: E402
    Bar,
    BundleMeta,
    MarketBundle,
    MarketQuery,
    TimeRange,
)
from cyqnt_trd.standard_bot.data import AlignmentPolicy, HistoricalSnapshotAssembler  # noqa: E402
from cyqnt_trd.standard_bot.data.derivatives import (  # noqa: E402
    HistoricalBinanceDerivativesDownloader,
    build_derivatives_path,
)
from cyqnt_trd.standard_bot.data.downloader import HistoricalBinanceDownloader  # noqa: E402
from cyqnt_trd.standard_bot.data.historical import (  # noqa: E402
    HistoricalParquetMarketDataAdapter,
    LocalFirstMarketDataAdapter,
    build_history_path,
    parquet_time_coverage,
    read_parquet_frame,
)
from cyqnt_trd.standard_bot.data.liquidations import (  # noqa: E402
    aggregate_force_order_records,
    build_liquidation_path,
)
from cyqnt_trd.standard_bot.entrypoints import mvp_backtest  # noqa: E402


class _FakeResponse:
    def __init__(self, payload: list[list]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> list[list]:
        return self._payload


class _FakeSession:
    def __init__(self, payloads) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def get(self, url: str, params: dict, timeout: int, **kwargs) -> _FakeResponse:
        self.calls.append({"url": url, "params": dict(params), "timeout": timeout, **kwargs})
        if not self.payloads:
            return _FakeResponse([])
        return _FakeResponse(self.payloads.pop(0))


def _write_parquet_rows(tmp_path, market_type: str, symbol: str, timeframe: str, rows: list[dict]) -> None:
    table = pyarrow.Table.from_pylist(rows)
    path = build_history_path(str(tmp_path), market_type, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    pyarrow_parquet.write_table(table, str(path))


def _write_derivatives_rows(tmp_path, market_type: str, symbol: str, dataset: str, rows: list[dict]) -> None:
    table = pyarrow.Table.from_pylist(rows)
    path = build_derivatives_path(str(tmp_path), market_type, symbol, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    pyarrow_parquet.write_table(table, str(path))


def _write_liquidation_rows(tmp_path, market_type: str, symbol: str, timeframe: str, rows: list[dict]) -> None:
    table = pyarrow.Table.from_pylist(rows)
    path = build_liquidation_path(str(tmp_path), market_type, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    pyarrow_parquet.write_table(table, str(path))


def test_historical_binance_downloader_streams_chunked_parquet(tmp_path) -> None:
    session = _FakeSession(
        payloads=[
            [
                [0, "100", "101", "99", "100.5", "10", 59_999, "1005", 100],
                [60_000, "100.5", "102", "100", "101.5", "11", 119_999, "1116.5", 110],
            ],
            [
                [120_000, "101.5", "103", "101", "102.0", "12", 179_999, "1224", 120],
            ],
        ]
    )
    downloader = HistoricalBinanceDownloader(
        data_root=str(tmp_path),
        market_type="spot",
        chunk_limit=2,
        session=session,
    )

    result = downloader.download(
        instrument_id="BTCUSDT",
        timeframe="1m",
        start_ts=0,
        end_ts=179_999,
    )

    assert result.row_count == 3
    assert result.chunk_count == 2
    assert len(session.calls) == 2

    frame = read_parquet_frame(result.path)
    assert list(frame["close_time"]) == [59_999, 119_999, 179_999]
    assert float(frame.iloc[-1]["close"]) == 102.0

    tail_frame = read_parquet_frame(result.path, tail_rows=2)
    assert list(tail_frame["close_time"]) == [119_999, 179_999]


def test_historical_binance_derivatives_downloader_writes_open_interest_and_funding(tmp_path) -> None:
    class _DerivativeSession:
        def __init__(self) -> None:
            self.calls = []
            self.funding_calls = 0

        def get(self, url: str, params: dict, timeout: int, **kwargs) -> _FakeResponse:
            self.calls.append({"url": url, "params": dict(params), "timeout": timeout, **kwargs})
            if "openInterestHist" in url:
                start_time = int(params["startTime"])
                if start_time == 0:
                    return _FakeResponse(
                        [
                            {
                                "symbol": "BTCUSDT",
                                "sumOpenInterest": "1000.0",
                                "sumOpenInterestValue": "100000.0",
                                "timestamp": 300_000,
                            },
                            {
                                "symbol": "BTCUSDT",
                                "sumOpenInterest": "1020.0",
                                "sumOpenInterestValue": "102000.0",
                                "timestamp": 600_000,
                            },
                        ]
                    )
                return _FakeResponse([])
            self.funding_calls += 1
            if self.funding_calls > 1:
                return _FakeResponse([])
            return _FakeResponse(
                [
                    {
                        "symbol": "BTCUSDT",
                        "fundingRate": "0.0001",
                        "fundingTime": 0,
                        "markPrice": "100.0",
                    },
                    {
                        "symbol": "BTCUSDT",
                        "fundingRate": "0.0002",
                        "fundingTime": 28_800_000,
                        "markPrice": "101.0",
                    },
                ]
            )

    session = _DerivativeSession()
    downloader = HistoricalBinanceDerivativesDownloader(
        data_root=str(tmp_path),
        market_type="futures",
        session=session,
    )

    result = downloader.download_bundle(
        instrument_id="BTCUSDT",
        timeframe="5m",
        start_ts=0,
        end_ts=30_000_000,
    )

    assert result.open_interest_rows == 2
    assert result.funding_rate_rows == 2
    oi_frame = pyarrow_parquet.read_table(result.open_interest_path).to_pandas()
    funding_frame = pyarrow_parquet.read_table(result.funding_rate_path).to_pandas()
    assert list(oi_frame["timestamp"]) == [300_000, 600_000]
    assert list(funding_frame["timestamp"]) == [0, 28_800_000]


def test_historical_parquet_adapter_resamples_1m_to_5m(tmp_path) -> None:
    rows = []
    for index in range(10):
        open_time = index * 60_000
        rows.append(
            {
                "open_time": open_time,
                "close_time": open_time + 59_999,
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.5 + index,
                "volume": 10.0 + index,
                "quote_volume": 1000.0 + index,
                "trades": 100 + index,
                "instrument_id": "BTCUSDT",
                "timeframe": "1m",
                "confirmed": True,
            }
        )
    _write_parquet_rows(tmp_path, "futures", "BTCUSDT", "1m", rows)

    adapter = HistoricalParquetMarketDataAdapter(
        data_root=str(tmp_path),
        market_type="futures",
        resample_source_timeframe="1m",
    )
    bundle = adapter.fetch_market(
        MarketQuery(
            instruments=["BTCUSDT"],
            timeframes=["5m"],
            time_range=TimeRange(),
        )
    )

    bars = bundle.bars[MarketBundle.key("BTCUSDT", "5m")]
    assert len(bars) == 2
    assert bars[0].open == 100.0
    assert bars[0].high == 105.0
    assert bars[0].low == 99.0
    assert bars[0].close == 104.5
    assert bars[0].volume == sum(10.0 + idx for idx in range(5))

    min_open, max_close = parquet_time_coverage(build_history_path(str(tmp_path), "futures", "BTCUSDT", "1m"))
    assert min_open == 0
    assert max_close == 599_999


def test_historical_parquet_adapter_enriches_bars_with_derivatives_features(tmp_path) -> None:
    rows = []
    for index in range(6):
        open_time = index * 300_000
        rows.append(
            {
                "open_time": open_time,
                "close_time": open_time + 299_999,
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.5 + index,
                "volume": 10.0 + index,
                "quote_volume": 1000.0 + index,
                "trades": 100 + index,
                "instrument_id": "BTCUSDT",
                "timeframe": "5m",
                "confirmed": True,
            }
        )
    _write_parquet_rows(tmp_path, "futures", "BTCUSDT", "5m", rows)
    _write_derivatives_rows(
        tmp_path,
        "futures",
        "BTCUSDT",
        "open_interest_5m",
        [
            {
                "timestamp": 299_999,
                "instrument_id": "BTCUSDT",
                "period": "5m",
                "open_interest": 1_000.0,
                "open_interest_value": 100_000.0,
            },
            {
                "timestamp": 599_999,
                "instrument_id": "BTCUSDT",
                "period": "5m",
                "open_interest": 1_020.0,
                "open_interest_value": 102_000.0,
            },
            {
                "timestamp": 899_999,
                "instrument_id": "BTCUSDT",
                "period": "5m",
                "open_interest": 1_050.0,
                "open_interest_value": 105_000.0,
            },
        ],
    )
    _write_derivatives_rows(
        tmp_path,
        "futures",
        "BTCUSDT",
        "funding_rate",
        [
            {
                "timestamp": 0,
                "instrument_id": "BTCUSDT",
                "funding_rate": 0.0001,
                "mark_price": 100.0,
            }
        ],
    )

    adapter = HistoricalParquetMarketDataAdapter(
        data_root=str(tmp_path),
        market_type="futures",
        resample_source_timeframe="1m",
        derivatives_data_root=str(tmp_path),
    )
    bundle = adapter.fetch_market(
        MarketQuery(
            instruments=["BTCUSDT"],
            timeframes=["5m"],
            time_range=TimeRange(),
        )
    )

    bars = bundle.bars[MarketBundle.key("BTCUSDT", "5m")]
    assert "funding_rate_bps" in bars[0].extras
    assert bars[0].extras["funding_rate_bps"] == pytest.approx(1.0)
    assert bars[1].extras["oi_change_bps"] == pytest.approx(200.0)


def test_liquidation_aggregation_and_adapter_enrichment(tmp_path) -> None:
    records = [
        {
            "timestamp": 120_000,
            "instrument_id": "BTCUSDT",
            "side": "SELL",
            "filled_qty": 2.0,
            "notional_usd": 200_000.0,
        },
        {
            "timestamp": 180_000,
            "instrument_id": "BTCUSDT",
            "side": "SELL",
            "filled_qty": 1.0,
            "notional_usd": 100_000.0,
        },
        {
            "timestamp": 420_000,
            "instrument_id": "BTCUSDT",
            "side": "BUY",
            "filled_qty": 3.0,
            "notional_usd": 330_000.0,
        },
    ]
    symbol_frames = aggregate_force_order_records(
        records,
        timeframe="5m",
        instrument_ids=["BTCUSDT"],
        timeframe_ms=300_000,
    )
    _write_liquidation_rows(
        tmp_path,
        "futures",
        "BTCUSDT",
        "5m",
        symbol_frames["BTCUSDT"].to_dict("records"),
    )

    rows = []
    for index in range(4):
        open_time = index * 300_000
        rows.append(
            {
                "open_time": open_time,
                "close_time": open_time + 299_999,
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.5 + index,
                "volume": 10.0 + index,
                "quote_volume": 1000.0 + index,
                "trades": 100 + index,
                "instrument_id": "BTCUSDT",
                "timeframe": "5m",
                "confirmed": True,
            }
        )
    _write_parquet_rows(tmp_path, "futures", "BTCUSDT", "5m", rows)

    adapter = HistoricalParquetMarketDataAdapter(
        data_root=str(tmp_path),
        market_type="futures",
        resample_source_timeframe="1m",
        liquidations_data_root=str(tmp_path),
    )
    bundle = adapter.fetch_market(
        MarketQuery(
            instruments=["BTCUSDT"],
            timeframes=["5m"],
            time_range=TimeRange(),
        )
    )

    bars = bundle.bars[MarketBundle.key("BTCUSDT", "5m")]
    assert bars[0].extras["long_liq_notional_usd"] == pytest.approx(300_000.0)
    assert bars[0].extras["short_liq_notional_usd"] == pytest.approx(0.0)
    assert bars[1].extras["short_liq_notional_usd"] == pytest.approx(330_000.0)


def test_resampled_multi_timeframe_snapshots_do_not_look_ahead(tmp_path) -> None:
    rows = []
    for index in range(120):
        open_time = index * 60_000
        rows.append(
            {
                "open_time": open_time,
                "close_time": open_time + 59_999,
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.5 + index,
                "volume": 10.0,
                "quote_volume": 1000.0,
                "trades": 100,
                "instrument_id": "BTCUSDT",
                "timeframe": "1m",
                "confirmed": True,
            }
        )
    _write_parquet_rows(tmp_path, "futures", "BTCUSDT", "1m", rows)

    adapter = HistoricalParquetMarketDataAdapter(
        data_root=str(tmp_path),
        market_type="futures",
        resample_source_timeframe="1m",
    )
    bundle = adapter.fetch_market(
        MarketQuery(
            instruments=["BTCUSDT"],
            timeframes=["5m", "1h"],
            time_range=TimeRange(),
        )
    )
    snapshots = HistoricalSnapshotAssembler(
        policy=AlignmentPolicy(policy_id="bar_close_v1", primary_timeframe="5m"),
        tail_bars=200,
    ).build(bundle)

    target_snapshot = next(snapshot for snapshot in snapshots if snapshot.meta.decision_as_of == 5_699_999)
    one_hour_bars = target_snapshot.require_market().bars[MarketBundle.key("BTCUSDT", "1h")]
    assert len(one_hour_bars) == 1
    assert one_hour_bars[-1].timestamp == 3_599_999


def test_historical_parquet_adapter_raises_when_requested_range_exceeds_local_coverage(tmp_path) -> None:
    rows = []
    for index in range(60):
        open_time = index * 60_000
        rows.append(
            {
                "open_time": open_time,
                "close_time": open_time + 59_999,
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 10.0,
                "quote_volume": 1000.0,
                "trades": 10,
                "instrument_id": "BTCUSDT",
                "timeframe": "1m",
                "confirmed": True,
            }
        )
    _write_parquet_rows(tmp_path, "futures", "BTCUSDT", "1m", rows)

    adapter = HistoricalParquetMarketDataAdapter(
        data_root=str(tmp_path),
        market_type="futures",
        resample_source_timeframe="1m",
    )
    with pytest.raises(FileNotFoundError):
        adapter.fetch_market(
            MarketQuery(
                instruments=["BTCUSDT"],
                timeframes=["5m", "1h"],
                time_range=TimeRange(start_ts=0, end_ts=7_200_000),
            )
        )


def test_load_market_bundle_downloads_missing_local_history(monkeypatch, tmp_path) -> None:
    expected_bundle = MarketBundle(
        bars={
            MarketBundle.key("BTCUSDT", "5m"): [
                Bar(
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.5,
                    volume=10.0,
                    quote_volume=1005.0,
                    timestamp=299_999,
                    instrument_id="BTCUSDT",
                    timeframe="5m",
                    confirmed=True,
                    extras={"open_time": 0, "close_time": 299_999},
                )
            ]
        },
        meta=BundleMeta(data_source="fixture", fetched_at=299_999),
    )

    class FakeAdapter:
        def fetch_market(self, market_query: MarketQuery) -> MarketBundle:
            return expected_bundle

    monkeypatch.setattr(mvp_backtest, "build_market_data_adapter", lambda **kwargs: FakeAdapter())

    args = Namespace(
        input_json=None,
        historical_dir=str(tmp_path),
        market_type="futures",
        storage_timeframe="1m",
        download_missing=True,
        allow_remote_api=False,
        symbol="BTCUSDT",
        interval="5m",
    )
    market_query = MarketQuery(
        instruments=["BTCUSDT"],
        timeframes=["5m", "1h"],
        time_range=TimeRange(start_ts=0, end_ts=999_999),
    )

    bundle = mvp_backtest.load_market_bundle(args, market_query)

    assert bundle is expected_bundle


def test_local_first_market_data_adapter_downloads_1m_for_resample() -> None:
    state = {"downloaded": False}
    expected_bundle = MarketBundle(
        bars={MarketBundle.key("BTCUSDT", "5m"): []},
        meta=BundleMeta(data_source="fixture", fetched_at=0),
    )

    class FakeLocalAdapter:
        def fetch_market(self, market_query: MarketQuery) -> MarketBundle:
            if not state["downloaded"]:
                raise FileNotFoundError("missing")
            return expected_bundle

    class FakeDownloader:
        calls: list[tuple[str, int, int]] = []

        def download(self, *, instrument_id: str, timeframe: str, start_ts: int, end_ts: int):
            FakeDownloader.calls.append((timeframe, start_ts, end_ts))
            state["downloaded"] = True

    adapter = LocalFirstMarketDataAdapter(
        local_adapter=FakeLocalAdapter(),
        storage_timeframe="1m",
        downloader=FakeDownloader(),
    )

    bundle = adapter.fetch_market(
        MarketQuery(
            instruments=["BTCUSDT"],
            timeframes=["5m", "1h"],
            time_range=TimeRange(start_ts=0, end_ts=999_999),
        )
    )

    assert bundle is expected_bundle
    assert FakeDownloader.calls == [("1m", 0, 999_999)]


def test_local_first_market_data_adapter_expands_download_start_for_resample() -> None:
    state = {"downloaded": False}
    expected_bundle = MarketBundle(
        bars={MarketBundle.key("BTCUSDT", "5m"): []},
        meta=BundleMeta(data_source="fixture", fetched_at=0),
    )

    class FakeLocalAdapter:
        def fetch_market(self, market_query: MarketQuery) -> MarketBundle:
            if not state["downloaded"]:
                raise FileNotFoundError("missing")
            return expected_bundle

    class FakeDownloader:
        calls: list[tuple[str, int, int]] = []

        def download(self, *, instrument_id: str, timeframe: str, start_ts: int, end_ts: int):
            FakeDownloader.calls.append((timeframe, start_ts, end_ts))
            state["downloaded"] = True

    adapter = LocalFirstMarketDataAdapter(
        local_adapter=FakeLocalAdapter(),
        storage_timeframe="1m",
        downloader=FakeDownloader(),
    )

    bundle = adapter.fetch_market(
        MarketQuery(
            instruments=["BTCUSDT"],
            timeframes=["5m", "1h"],
            time_range=TimeRange(start_ts=3_600_000, end_ts=7_200_000),
        )
    )

    assert bundle is expected_bundle
    assert FakeDownloader.calls == [("1m", 0, 7_200_000)]


def test_main_framework_path_skips_snapshot_assembly(monkeypatch, tmp_path, capsys) -> None:
    input_path = tmp_path / "bars.json"
    rows = []
    for index in range(30):
        open_time = index * 60_000
        rows.append(
            {
                "open_time": open_time,
                "close_time": open_time + 59_999,
                "open_price": 100.0 + index,
                "high_price": 101.0 + index,
                "low_price": 99.0 + index,
                "close_price": 100.5 + index,
                "volume": 10.0,
                "quote_volume": 1000.0,
            }
        )
    input_path.write_text(
        '{"symbol":"BTCUSDT","interval":"1m","data":%s}' % __import__("json").dumps(rows),
        encoding="utf-8",
    )

    def fail_build(*args, **kwargs):
        raise AssertionError("snapshot assembler should not run for framework backtests")

    # The framework engine (default) reconstructs the OHLCV frame directly from
    # bars and never touches the event-driven snapshot assembler.
    monkeypatch.setattr(mvp_backtest.HistoricalSnapshotAssembler, "build", fail_build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mvp_backtest",
            "--engine",
            "framework",
            "--input-json",
            str(input_path),
            "--symbol",
            "BTCUSDT",
            "--interval",
            "1m",
            "--strategy",
            "moving_average_cross",
            "--fast-window",
            "3",
            "--slow-window",
            "5",
        ],
    )

    assert mvp_backtest.main() == 0
    captured = capsys.readouterr()
    assert "engine=framework" in captured.out
