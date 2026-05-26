from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyqnt_trd.standard_bot.entrypoints.mvp_paper_daemon import BarFetcher, PaperDaemon
from cyqnt_trd.standard_bot.simulation import NumbaLivePaperSession


def _make_bar(index: int, close: float, interval_ms: int = 60_000) -> dict:
    open_time = index * interval_ms
    return {
        "timestamp": open_time + interval_ms - 1,
        "open_time": open_time,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 100.0,
        "quote_volume": close * 100.0,
        "oi_change_bps": 0.0,
        "funding_rate_bps": 0.0,
        "long_liq_notional_usd": 0.0,
        "short_liq_notional_usd": 0.0,
    }


def _bars_with_crosses() -> list[dict]:
    closes = [100.0] * 6 + [110.0] * 6 + [90.0] * 6 + [120.0] * 6
    return [_make_bar(index, close) for index, close in enumerate(closes)]


def _snapshot_without_clock(snapshot: dict) -> dict:
    cleaned = dict(snapshot)
    cleaned.pop("last_update_ts", None)
    return cleaned


def test_bar_fetcher_catches_up_all_missing_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    fetcher = BarFetcher(symbol="BTCUSDT", interval="1m", market_type="futures")
    dataset = [_make_bar(index, 100.0 + float(index)) for index in range(25)]
    calls: list[tuple[int | None, int | None, int | None]] = []

    def fake_fetch_bars(limit: int | None = 120, start_ts: int | None = None, end_ts: int | None = None):
        calls.append((limit, start_ts, end_ts))
        rows = [bar for bar in dataset if start_ts is None or bar["open_time"] >= start_ts]
        if end_ts is not None:
            rows = [bar for bar in rows if bar["timestamp"] <= end_ts]
        return rows[:limit] if limit is not None else rows

    monkeypatch.setattr(fetcher, "fetch_bars", fake_fetch_bars)

    new_bars = fetcher.fetch_new_bars_since(dataset[0]["timestamp"], limit=10, max_batches=10)

    assert [bar["timestamp"] for bar in new_bars] == [
        bar["timestamp"] for bar in dataset[1:]
    ]
    assert len(calls) >= 3


def test_live_paper_session_checkpoint_round_trip_continues_identically() -> None:
    bars = _bars_with_crosses()
    config = {
        "instrument_id": "BTCUSDT",
        "timeframe": "1m",
        "fast_window": 2,
        "slow_window": 5,
        "entry_threshold": 0.0,
    }

    original = NumbaLivePaperSession(
        strategy_id="moving_average_cross",
        symbol="BTCUSDT",
        config=config,
        initial_capital=10_000.0,
        fee_bps=4.0,
        slippage_bps=0.0,
        max_bar_volume_fraction=0.0,
    )

    split_index = 15
    for bar in bars[:split_index]:
        original.tick(bar)

    restored = NumbaLivePaperSession.from_checkpoint(original.checkpoint_state())

    assert _snapshot_without_clock(restored.state_snapshot()) == _snapshot_without_clock(
        original.state_snapshot()
    )

    original_fills = []
    restored_fills = []
    for bar in bars[split_index:]:
        fill_a = original.tick(bar)
        fill_b = restored.tick(bar)
        if fill_a is not None:
            original_fills.append(original._fill_to_dict(fill_a))
        if fill_b is not None:
            restored_fills.append(restored._fill_to_dict(fill_b))

    assert restored_fills == original_fills
    assert _snapshot_without_clock(restored.state_snapshot()) == _snapshot_without_clock(
        original.state_snapshot()
    )


def test_daemon_restore_reconciles_trade_journal_from_checkpoint(tmp_path: Path) -> None:
    kwargs = dict(
        symbol="BTCUSDT",
        interval="1m",
        strategy="moving_average_cross",
        strategy_module=None,
        extra_params={"fast_window": 2, "slow_window": 5, "entry_threshold": 0.0},
        poll_interval=60,
        warm_up_bars=10,
        initial_capital=10_000.0,
        fee_bps=4.0,
        slippage_bps=0.0,
        market_type="futures",
    )

    daemon = PaperDaemon(state_dir=str(tmp_path), **kwargs)
    daemon._session = NumbaLivePaperSession(
        strategy_id="moving_average_cross",
        symbol="BTCUSDT",
        config={
            "instrument_id": "BTCUSDT",
            "timeframe": "1m",
            "fast_window": 2,
            "slow_window": 5,
            "entry_threshold": 0.0,
        },
        initial_capital=10_000.0,
        fee_bps=4.0,
        slippage_bps=0.0,
        max_bar_volume_fraction=0.0,
    )

    for bar in _bars_with_crosses():
        daemon._session.tick(bar)

    assert daemon._session.trade_log, "expected at least one fill for restore test"
    daemon._last_bar_ts = daemon._session.state_snapshot()["last_bar_timestamp"]
    daemon._write_checkpoint()

    restored = PaperDaemon(state_dir=str(tmp_path), **kwargs)
    assert restored._restore_session_from_checkpoint() is True
    assert len(restored._persisted_trade_log) == len(daemon._session.trade_log)

    journal_rows = [
        json.loads(line)
        for line in restored.trades_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(journal_rows) == len(daemon._session.trade_log)
