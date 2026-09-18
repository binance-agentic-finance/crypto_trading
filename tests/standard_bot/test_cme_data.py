from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

pyarrow = pytest.importorskip("pyarrow")
pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

from cyqnt_trd.standard_bot.core import MarketBundle, MarketQuery, TimeRange  # noqa: E402
from cyqnt_trd.standard_bot.data.cme import (  # noqa: E402
    YahooCmeChartDownloader,
    cme_contract_multiplier,
    ingest_cme_csv_to_parquet,
    ingest_huggingface_nq_parquets_to_parquet,
    ingest_yfinance_cme_frame_to_parquet,
)
from cyqnt_trd.standard_bot.data.historical import (  # noqa: E402
    HistoricalParquetMarketDataAdapter,
    build_history_path,
    read_parquet_frame,
)
from cyqnt_trd.standard_bot.entrypoints import (  # noqa: E402
    cme_hf_ingest,
    cme_ingest,
    cme_yfinance_ingest,
    mvp_backtest,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeYahooSession:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def get(self, url: str, params: dict, timeout: int, **kwargs) -> _FakeResponse:
        self.calls.append({"url": url, "params": dict(params), "timeout": timeout, "kwargs": kwargs})
        return _FakeResponse(self.payload)


def _yahoo_payload() -> dict:
    return {
        "chart": {
            "result": [
                {
                    "timestamp": [1_700_000_000, 1_700_000_300, 1_700_000_600],
                    "indicators": {
                        "quote": [
                            {
                                "open": [15000.0, 15001.0, 15002.0],
                                "high": [15002.0, 15003.0, 15004.0],
                                "low": [14999.0, 15000.0, 15001.0],
                                "close": [15001.0, 15002.0, 15003.0],
                                "volume": [10, 20, 30],
                            }
                        ]
                    },
                }
            ],
            "error": None,
        }
    }


def _write_standard_parquet(tmp_path, market_type: str, symbol: str, timeframe: str, rows: list[dict]) -> None:
    table = pyarrow.Table.from_pylist(rows)
    path = build_history_path(str(tmp_path), market_type, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    pyarrow_parquet.write_table(table, str(path))


def test_yahoo_cme_chart_downloader_writes_mnq_parquet(tmp_path) -> None:
    session = _FakeYahooSession(_yahoo_payload())
    downloader = YahooCmeChartDownloader(data_root=str(tmp_path), session=session)

    result = downloader.download(
        instrument_id="MNQ",
        timeframe="5m",
        start_ts=1_700_000_000_000,
        end_ts=1_700_000_899_999,
    )

    assert result.market_type == "cme"
    assert result.instrument_id == "MNQ"
    assert result.row_count == 3
    assert session.calls[0]["params"]["interval"] == "5m"
    assert "MNQ%3DF" in session.calls[0]["url"]

    frame = read_parquet_frame(result.path)
    assert list(frame["instrument_id"]) == ["MNQ", "MNQ", "MNQ"]
    assert list(frame["timeframe"]) == ["5m", "5m", "5m"]
    assert float(frame.iloc[0]["quote_volume"]) == pytest.approx(15001.0 * 10 * 2.0)


def test_cme_csv_ingest_and_local_adapter_support_15m(tmp_path) -> None:
    csv_path = tmp_path / "mnq_1m.csv"
    rows = []
    for index in range(30):
        rows.append(
            {
                "timestamp": "2026-04-01 13:%02d:00" % index,
                "open": 15000.0 + index,
                "high": 15001.0 + index,
                "low": 14999.0 + index,
                "close": 15000.5 + index,
                "volume": 100 + index,
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = ingest_cme_csv_to_parquet(
        csv_path=csv_path,
        data_root=str(tmp_path),
        instrument_id="MNQ",
        timeframe="1m",
    )

    assert result.row_count == 30
    adapter = HistoricalParquetMarketDataAdapter(
        data_root=str(tmp_path),
        market_type="cme",
        resample_source_timeframe="1m",
    )
    bundle = adapter.fetch_market(
        MarketQuery(
            instruments=["MNQ"],
            timeframes=["15m"],
            time_range=TimeRange(),
        )
    )

    bars = bundle.bars[MarketBundle.key("MNQ", "15m")]
    assert len(bars) == 2
    assert bars[0].open == 15000.0
    assert bars[0].close == 15014.5
    assert bars[0].quote_volume is not None
    assert bars[0].extras["source"] == "resampled_from_1m"


def test_cme_ingest_entrypoint_writes_local_parquet(monkeypatch, tmp_path, capsys) -> None:
    csv_path = tmp_path / "mnq_5m.csv"
    pd.DataFrame(
        [
            {
                "timestamp": "2026-04-01 13:00:00",
                "open": 15000.0,
                "high": 15010.0,
                "low": 14990.0,
                "close": 15005.0,
                "volume": 100,
            }
        ]
    ).to_csv(csv_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cme_ingest",
            "--csv",
            str(csv_path),
            "--symbol",
            "MNQ",
            "--interval",
            "5m",
            "--historical-dir",
            str(tmp_path),
        ],
    )

    assert cme_ingest.main() == 0
    captured = capsys.readouterr()
    assert "ingested_cme_csv symbol=MNQ interval=5m rows=1" in captured.out
    assert build_history_path(str(tmp_path), "cme", "MNQ", "5m").exists()


def test_huggingface_nq_parquet_ingest_writes_mnq_1m(tmp_path) -> None:
    source_path = tmp_path / "NQ_1min_2025.parquet"
    pyarrow_parquet.write_table(
        pyarrow.Table.from_pandas(
            pd.DataFrame(
                [
                    {
                        "timestamp": "2025-01-01 23:00:00",
                        "open": 21269.0,
                        "high": 21282.75,
                        "low": 21253.5,
                        "close": 21261.25,
                        "volume": 393,
                    },
                    {
                        "timestamp": "2025-01-01 23:01:00",
                        "open": 21263.0,
                        "high": 21272.25,
                        "low": 21254.25,
                        "close": 21271.25,
                        "volume": 169,
                    },
                ]
            ),
            preserve_index=False,
        ),
        str(source_path),
    )

    result = ingest_huggingface_nq_parquets_to_parquet(
        source_paths=[source_path],
        data_root=str(tmp_path),
        instrument_id="MNQ",
    )

    frame = read_parquet_frame(result.path)
    assert result.row_count == 2
    assert list(frame["instrument_id"]) == ["MNQ", "MNQ"]
    assert list(frame["timeframe"]) == ["1m", "1m"]
    assert int(frame.iloc[0]["open_time"]) == 1_735_772_400_000
    assert float(frame.iloc[0]["quote_volume"]) == pytest.approx(21261.25 * 393 * 2.0)


def test_cme_hf_ingest_entrypoint_writes_local_parquet(monkeypatch, tmp_path, capsys) -> None:
    source_path = tmp_path / "NQ_1min_2025.parquet"
    pyarrow_parquet.write_table(
        pyarrow.Table.from_pandas(
            pd.DataFrame(
                [
                    {
                        "timestamp": "2025-01-01 23:00:00",
                        "open": 21269.0,
                        "high": 21282.75,
                        "low": 21253.5,
                        "close": 21261.25,
                        "volume": 393,
                    }
                ]
            ),
            preserve_index=False,
        ),
        str(source_path),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cme_hf_ingest",
            "--source-parquet",
            str(source_path),
            "--symbol",
            "MNQ",
            "--historical-dir",
            str(tmp_path),
        ],
    )

    assert cme_hf_ingest.main() == 0
    captured = capsys.readouterr()
    assert "ingested_hf_nq source=local symbol=MNQ interval=1m" in captured.out
    assert build_history_path(str(tmp_path), "cme", "MNQ", "1m").exists()


def test_yfinance_cme_frame_ingest_writes_mnq_5m(tmp_path) -> None:
    index = pd.DatetimeIndex(
        ["2026-03-19 04:05:00+00:00", "2026-03-19 04:10:00+00:00"],
        name="Datetime",
    )
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Adj Close", "Volume"], ["MNQ=F"]],
        names=["Price", "Ticker"],
    )
    frame = pd.DataFrame(
        [
            [24642.0, 24645.0, 24638.0, 24640.75, 24640.75, 0],
            [24641.0, 24647.0, 24639.0, 24643.75, 24643.75, 2477],
        ],
        index=index,
        columns=columns,
    )

    result = ingest_yfinance_cme_frame_to_parquet(
        frame=frame,
        data_root=str(tmp_path),
        instrument_id="MNQ",
        timeframe="5m",
    )

    parsed = read_parquet_frame(result.path)
    assert result.row_count == 2
    assert list(parsed["instrument_id"]) == ["MNQ", "MNQ"]
    assert list(parsed["timeframe"]) == ["5m", "5m"]
    assert int(parsed.iloc[0]["open_time"]) == 1_773_893_100_000
    assert float(parsed.iloc[1]["quote_volume"]) == pytest.approx(24643.75 * 2477 * 2.0)


def test_cme_yfinance_ingest_entrypoint_uses_downloader(monkeypatch, tmp_path, capsys) -> None:
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return type(
            "Result",
            (),
            {
                "instrument_id": kwargs["instrument_id"],
                "timeframe": kwargs["timeframe"],
                "row_count": 2,
                "path": str(tmp_path / ("%s.parquet" % kwargs["timeframe"])),
            },
        )()

    monkeypatch.setattr(cme_yfinance_ingest, "download_yfinance_cme_to_parquet", fake_download)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cme_yfinance_ingest",
            "--symbol",
            "MNQ",
            "--provider-symbol",
            "MNQ=F",
            "--interval",
            "5m,15m",
            "--period",
            "30d",
            "--historical-dir",
            str(tmp_path),
        ],
    )

    assert cme_yfinance_ingest.main() == 0
    captured = capsys.readouterr()
    assert "ingested_yfinance_cme symbol=MNQ interval=5m period=30d rows=2" in captured.out
    assert "ingested_yfinance_cme symbol=MNQ interval=15m period=30d rows=2" in captured.out
    assert [call["timeframe"] for call in calls] == ["5m", "15m"]
    assert calls[0]["provider_symbol_override"] == "MNQ=F"


def test_mvp_backtest_runs_mnq_cme_framework_path(monkeypatch, tmp_path, capsys) -> None:
    rows = []
    interval_ms = 5 * 60_000
    for index in range(40):
        open_time = index * interval_ms
        close = 15000.0 + (index % 12) * 4.0 - (index // 12) * 2.0
        rows.append(
            {
                "open_time": open_time,
                "close_time": open_time + interval_ms - 1,
                "open": close - 1.0,
                "high": close + 2.0,
                "low": close - 3.0,
                "close": close,
                "volume": 100.0,
                "quote_volume": close * 100.0 * cme_contract_multiplier("MNQ"),
                "trades": 0,
                "instrument_id": "MNQ",
                "timeframe": "5m",
                "confirmed": True,
            }
        )
    _write_standard_parquet(tmp_path, "cme", "MNQ", "5m", rows)

    output_path = tmp_path / "mnq_backtest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mvp_backtest",
            "--engine",
            "framework",
            "--market-type",
            "cme",
            "--symbol",
            "MNQ",
            "--interval",
            "5m",
            "--strategy",
            "moving_average_cross",
            "--fast-window",
            "2",
            "--slow-window",
            "4",
            "--historical-dir",
            str(tmp_path),
            "--limit",
            "40",
            "--commission-bps",
            "0",
            "--slippage-bps",
            "0",
            "--max-bar-volume-fraction",
            "1",
            "--output-json",
            str(output_path),
        ],
    )

    assert mvp_backtest.main() == 0
    captured = capsys.readouterr()
    # The framework backtest is returns-based (ΣW·R), so it is contract-multiplier
    # invariant — this test now just verifies CME parquet data flows end-to-end
    # through the unified engine. (Contract-multiplier handling lives in the live
    # execution layer, exercised separately.)
    assert "engine=framework" in captured.out

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["snapshot_count"] == 40.0
