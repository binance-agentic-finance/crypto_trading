"""Regression tests for liquidation aggregate storage.

The bug these cover: HistoricalBinanceLiquidationRecorder.record_and_store wrote each capture with
a bare ``pq.write_table(table, target_path)`` to a FIXED path per (symbol, timeframe), so every
capture silently discarded the previous one. The !forceOrder stream is not archived upstream, so
the discarded rows were unrecoverable. On disk this had left 38 rows in total across 30 files.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

import pyarrow.parquet as pq  # noqa: E402

from cyqnt_trd.standard_bot.data.liquidations import (  # noqa: E402
    _liquidation_schema,
    _read_existing_aggregate,
    build_liquidation_path,
    enrich_market_frame_with_liquidations,
    merge_liquidation_rows,
    store_liquidation_aggregate,
)

COLUMNS = list(_liquidation_schema().names)


def _row(timestamp: int, *, symbol: str = "BTCUSDT", timeframe: str = "5m", longs: int = 1, shorts: int = 1):
    return {
        "timestamp": timestamp,
        "instrument_id": symbol,
        "timeframe": timeframe,
        "long_liq_count": longs,
        "short_liq_count": shorts,
        "long_liq_qty": float(longs),
        "short_liq_qty": float(shorts),
        "long_liq_notional_usd": float(longs) * 10.0,
        "short_liq_notional_usd": float(shorts) * 10.0,
        "total_liq_notional_usd": float(longs + shorts) * 10.0,
        "net_liq_notional_usd": float(longs - shorts) * 10.0,
        "liq_imbalance_ratio": 0.5,
    }


def _frame(rows) -> pd.DataFrame:
    # columns= keeps the empty case well-formed instead of yielding a no-column frame
    return pd.DataFrame(rows, columns=COLUMNS)


def test_second_capture_does_not_destroy_the_first():
    first = _frame([_row(1_000), _row(2_000)])
    second = _frame([_row(3_000), _row(4_000)])

    merged = merge_liquidation_rows(first, second)

    assert list(merged["timestamp"]) == [1_000, 2_000, 3_000, 4_000]


def test_overlapping_bucket_keeps_the_more_complete_observation_not_the_newer_one():
    # bucket 2000 seen twice: a partial observation (1 event) and a fuller one (9 events).
    partial = _frame([_row(2_000, longs=1, shorts=0)])
    fuller = _frame([_row(2_000, longs=5, shorts=4)])

    # order must not matter - the fuller observation wins either way, and it is never summed
    for existing, incoming in ((partial, fuller), (fuller, partial)):
        merged = merge_liquidation_rows(existing, incoming)
        assert len(merged) == 1, "duplicate bucket key must collapse"
        assert int(merged.iloc[0]["long_liq_count"]) == 5
        assert int(merged.iloc[0]["short_liq_count"]) == 4


def test_merged_output_is_sorted_and_key_unique_for_merge_asof():
    # merge_asof in enrich_market_frame_with_liquidations requires a sorted key and silently
    # takes the last of any duplicate, so both properties are load-bearing.
    merged = merge_liquidation_rows(
        _frame([_row(5_000), _row(1_000)]),
        _frame([_row(3_000), _row(1_000, longs=7)]),
    )
    assert list(merged["timestamp"]) == sorted(merged["timestamp"])
    assert not merged.duplicated(subset=["timestamp", "instrument_id", "timeframe"]).any()


def test_different_symbols_and_timeframes_do_not_collide():
    merged = merge_liquidation_rows(
        _frame([_row(1_000, symbol="BTCUSDT"), _row(1_000, symbol="ETHUSDT")]),
        _frame([_row(1_000, symbol="BTCUSDT", timeframe="15m")]),
    )
    assert len(merged) == 3


def test_empty_incoming_or_existing_is_handled():
    rows = _frame([_row(1_000)])
    empty = _frame([]).astype(rows.dtypes.to_dict())

    assert len(merge_liquidation_rows(None, rows)) == 1
    assert len(merge_liquidation_rows(empty, rows)) == 1
    assert len(merge_liquidation_rows(rows, empty)) == 1


def test_store_across_three_captures_accumulates_on_disk(tmp_path):
    """The end-to-end property the original bug violated: history survives later captures."""
    common = dict(data_root=str(tmp_path), market_type="futures", instrument_id="BTCUSDT", timeframe="5m")

    path, total = store_liquidation_aggregate(**common, symbol_frame=_frame([_row(1_000), _row(2_000)]))
    assert total == 2

    _, total = store_liquidation_aggregate(**common, symbol_frame=_frame([_row(3_000)]))
    assert total == 3, "second capture must not discard the first"

    _, total = store_liquidation_aggregate(**common, symbol_frame=_frame([_row(4_000), _row(5_000)]))
    assert total == 5

    stored = pq.read_table(str(path))
    assert list(stored.to_pandas()["timestamp"]) == [1_000, 2_000, 3_000, 4_000, 5_000]
    assert stored.schema.equals(_liquidation_schema()), "canonical schema must survive the merge"

    # no temp file left behind, and exactly one parquet per (symbol, timeframe)
    assert list(path.parent.glob(".*")) == []
    assert [p.name for p in path.parent.glob("*.parquet")] == ["liquidation_5m.parquet"]


def test_stored_history_is_consumable_by_the_only_reader(tmp_path):
    store_liquidation_aggregate(
        data_root=str(tmp_path), market_type="futures", instrument_id="BTCUSDT", timeframe="5m",
        symbol_frame=_frame([_row(1_000, longs=2, shorts=0), _row(2_000, longs=0, shorts=3)]),
    )
    bars = pd.DataFrame({"close_time": [1_500, 2_500], "close": [10.0, 11.0]})

    enriched = enrich_market_frame_with_liquidations(
        bars, liquidations_root=str(tmp_path), market_type="futures",
        instrument_id="BTCUSDT", timeframe="5m",
    )
    # merge_asof backward: bar at 1500 sees bucket 1000, bar at 2500 sees bucket 2000
    assert list(enriched["long_liq_count"]) == [2, 0]
    assert list(enriched["short_liq_count"]) == [0, 3]


def test_read_existing_aggregate_roundtrip_and_corrupt_file(tmp_path):
    target = build_liquidation_path(str(tmp_path), "futures", "BTCUSDT", "5m")
    target.parent.mkdir(parents=True, exist_ok=True)

    assert _read_existing_aggregate(target) is None, "missing file reads as None"

    frame = _frame([_row(1_000), _row(2_000)])
    pq.write_table(
        __import__("pyarrow").Table.from_pandas(frame, schema=_liquidation_schema(), preserve_index=False),
        str(target),
    )
    back = _read_existing_aggregate(target)
    assert back is not None and len(back) == 2

    # a corrupt file must not blow up the capture that is trying to be saved
    target.write_bytes(b"not parquet")
    with pytest.warns(RuntimeWarning, match="could not be read"):
        assert _read_existing_aggregate(target) is None


def test_unreadable_file_is_moved_aside_not_overwritten(tmp_path):
    """"Absent" and "unreadable" must not both mean "no history".

    Both returned None before, so an unreadable file was merged onto nothing and then
    ``os.replace`` d away by the next capture: a transient read failure deleted every earlier
    capture of a stream that is not archived upstream, with nothing in the output to say so.
    """
    common = dict(data_root=str(tmp_path), market_type="futures", instrument_id="BTCUSDT", timeframe="5m")
    path, total = store_liquidation_aggregate(**common, symbol_frame=_frame([_row(1_000), _row(2_000)]))
    assert total == 2

    path.write_bytes(b"not parquet")
    with pytest.warns(RuntimeWarning, match="moved to"):
        _, total = store_liquidation_aggregate(**common, symbol_frame=_frame([_row(3_000)]))

    # the fresh capture landed ...
    assert total == 1
    assert list(pq.read_table(str(path)).to_pandas()["timestamp"]) == [3_000]

    # ... and the bytes that could not be parsed are still on disk to be salvaged from
    quarantined = list(path.parent.glob("*.unreadable-*"))
    assert len(quarantined) == 1, "the unreadable bytes must survive somewhere"
    assert quarantined[0].read_bytes() == b"not parquet"
    assert list(path.parent.glob(".*")) == [], "no temp file left behind"


def test_partially_overlapping_observations_undercount_by_design(tmp_path):
    """Counts are a lower bound, not a census -- the docstring says so, this pins it.

    One capture caught the first 5 events of a bucket, another the last 7; the truth is 12.
    Summing would double-count the overlap, so max() is the deliberate choice and 7 is the
    documented answer. A reader treating the number as "how many liquidations happened" is wrong.
    """
    merged = merge_liquidation_rows(
        _frame([_row(2_000, longs=3, shorts=2)]),   # 5 events
        _frame([_row(2_000, longs=4, shorts=3)]),   # 7 events
    )
    events = int(merged.iloc[0]["long_liq_count"]) + int(merged.iloc[0]["short_liq_count"])
    assert events == 7, "keeps the fuller observation"
    assert events < 12, "and therefore undercounts a partially-covered bucket"


def test_equal_event_counts_resolve_to_the_incoming_frame(tmp_path):
    """Tie-break is "later argument wins", via the stable sort. Deterministic, so worth pinning."""
    existing = _frame([_row(2_000, longs=1, shorts=1)])
    incoming = _frame([_row(2_000, longs=1, shorts=1)])
    existing.loc[0, "total_liq_notional_usd"] = 100.0
    incoming.loc[0, "total_liq_notional_usd"] = 999.0

    merged = merge_liquidation_rows(existing, incoming)
    assert float(merged.iloc[0]["total_liq_notional_usd"]) == 999.0
