"""Unit tests for the native ``atr_trailing_stop`` exit type (Bug #3 fix).

Covers:
- ``BlockStrategyPlugin._compute_exit_spec`` emits a well-formed
  atr_trailing_stop spec for both long and short entries.
- ``SnapshotBacktestRunner._check_exit`` triggers ``trailing_stop`` when
  bar_low pierces the dynamic stop (long) / bar_high pierces (short).
- The running peak is **monotonically updated bar-to-bar** so the stop
  ratchets in the favourable direction.
- ``max_bars`` still enforces a timeout when the trailing stop never
  triggers.
- ``SnapshotBacktestRunner._finalize_exit_prices`` re-anchors
  ``running_peak`` to the actual fill price.
- The vectorized engine's ``_check_exit_long`` / ``_check_exit_short``
  also support atr_trailing_stop with the same semantics.

See ``auto_opt_experiments/CYQNT_TRD_BUG_REPORT.md`` Issue #3.
"""

from __future__ import annotations

from cyqnt_trd.blocks import strategy as block_strategy
from cyqnt_trd.standard_bot.simulation.runner import SnapshotBacktestRunner
from cyqnt_trd.standard_bot.simulation.vectorized_backtest import (
    _check_exit_long,
    _check_exit_short,
)


# ---------------------------------------------------------------------------
# 1. _compute_exit_spec emits the expected fields
# ---------------------------------------------------------------------------


def test_compute_exit_spec_long_atr_trailing():
    plugin = block_strategy.BlockStrategyPlugin(
        plugin_id="t",
        plugin_version="v",
        signal_fn=lambda d: (d["close"] > 0, None),
        exit_cfg={
            "type": "atr_trailing_stop",
            "atr_period": 14,
            "trail_mult": 2.5,
            "max_bars": 9999,
        },
    )
    spec = plugin._compute_exit_spec(
        side="long", entry_close=100.0, atr_value=1.0
    )
    assert spec is not None
    assert spec["type"] == "atr_trailing_stop"
    assert spec["trail_mult"] == 2.5
    assert spec["max_bars"] == 9999
    assert spec["atr_at_entry"] == 1.0
    # running_peak initialized to entry_close; runner re-anchors to fill_price
    assert spec["running_peak"] == 100.0


def test_compute_exit_spec_short_atr_trailing():
    plugin = block_strategy.BlockStrategyPlugin(
        plugin_id="t",
        plugin_version="v",
        signal_fn=lambda d: (None, d["close"] > 0),
        exit_cfg={"type": "atr_trailing_stop", "trail_mult": 3.0},
    )
    spec = plugin._compute_exit_spec(
        side="short", entry_close=200.0, atr_value=2.0
    )
    assert spec is not None
    assert spec["atr_at_entry"] == 2.0
    assert spec["trail_mult"] == 3.0
    assert spec["running_peak"] == 200.0


# ---------------------------------------------------------------------------
# 2. SnapshotBacktestRunner._check_exit — long trail trigger
# ---------------------------------------------------------------------------


def test_long_trailing_stop_triggers_when_bar_low_pierces_dynamic_stop():
    """Long: peak=110, ATR=2, trail=2.0 → stop=106. bar_low=105 → fire."""
    spec = {
        "type": "atr_trailing_stop",
        "side": "long",
        "trail_mult": 2.0,
        "atr_at_entry": 2.0,
        "running_peak": 110.0,  # already running from prior bars
        "max_bars": 9999,
    }
    triggered, exit_price, reason = SnapshotBacktestRunner._check_exit(
        spec=spec,
        entry_price=100.0,
        entry_idx=0,
        current_idx=5,
        bar_high=110.0,
        bar_low=105.0,  # pierces 110 - 2*2 = 106
        bar_close=106.5,
    )
    assert triggered is True
    assert reason == "trailing_stop"
    # exit_price is the dynamic stop (peak - trail*atr)
    assert exit_price == 106.0
    # running_peak unchanged (bar_high == prior peak)
    assert spec["running_peak"] == 110.0


def test_long_trailing_stop_does_not_trigger_when_above_stop():
    """Long: peak=110, ATR=2, trail=2.0 → stop=106. bar_low=107 → no fire."""
    spec = {
        "type": "atr_trailing_stop",
        "side": "long",
        "trail_mult": 2.0,
        "atr_at_entry": 2.0,
        "running_peak": 110.0,
        "max_bars": 9999,
    }
    triggered, _, reason = SnapshotBacktestRunner._check_exit(
        spec=spec,
        entry_price=100.0,
        entry_idx=0,
        current_idx=5,
        bar_high=110.0,
        bar_low=107.0,
        bar_close=109.0,
    )
    assert triggered is False
    assert reason == ""


# ---------------------------------------------------------------------------
# 3. SnapshotBacktestRunner._check_exit — short trail trigger
# ---------------------------------------------------------------------------


def test_short_trailing_stop_triggers_when_bar_high_pierces_dynamic_stop():
    """Short: trough=90, ATR=2, trail=2.0 → stop=94. bar_high=95 → fire."""
    spec = {
        "type": "atr_trailing_stop",
        "side": "short",
        "trail_mult": 2.0,
        "atr_at_entry": 2.0,
        "running_peak": 90.0,  # 'peak' here is the running low
        "max_bars": 9999,
    }
    triggered, exit_price, reason = SnapshotBacktestRunner._check_exit(
        spec=spec,
        entry_price=100.0,
        entry_idx=0,
        current_idx=5,
        bar_high=95.0,  # pierces 90 + 2*2 = 94
        bar_low=90.0,
        bar_close=93.5,
    )
    assert triggered is True
    assert reason == "trailing_stop"
    assert exit_price == 94.0


# ---------------------------------------------------------------------------
# 4. Running peak ratchets in favourable direction across bars
# ---------------------------------------------------------------------------


def test_long_running_peak_ratchets_up_across_bars():
    """Two consecutive bars: peak should rise from 105 → 112, stop tightens."""
    spec = {
        "type": "atr_trailing_stop",
        "side": "long",
        "trail_mult": 2.0,
        "atr_at_entry": 2.0,
        "running_peak": 100.0,  # entry-time anchor
        "max_bars": 9999,
    }
    # Bar 1: high=105, low=99 → peak becomes 105, stop becomes 101.
    # Stop NOT pierced (low=99 < 101 — wait, 99 <= 101 → would pierce).
    # Tweak: low=102 so we don't trigger.
    triggered, _, _ = SnapshotBacktestRunner._check_exit(
        spec=spec,
        entry_price=100.0,
        entry_idx=0,
        current_idx=1,
        bar_high=105.0,
        bar_low=102.0,
        bar_close=104.0,
    )
    assert triggered is False
    assert spec["running_peak"] == 105.0
    # implied stop now 105 - 2*2 = 101
    assert spec["stop_loss_price"] == 101.0

    # Bar 2: high=112, low=108 → peak 112, stop 108. low=108 NOT pierce
    # (need bar_low <= stop, i.e. 108 <= 108 — yes, equality triggers).
    # Use low=109 so it doesn't trigger.
    triggered, _, _ = SnapshotBacktestRunner._check_exit(
        spec=spec,
        entry_price=100.0,
        entry_idx=0,
        current_idx=2,
        bar_high=112.0,
        bar_low=109.0,
        bar_close=111.0,
    )
    assert triggered is False
    # peak ratcheted UP
    assert spec["running_peak"] == 112.0
    # stop tightened: 112 - 4 = 108
    assert spec["stop_loss_price"] == 108.0

    # Bar 3: dip to 107 → 107 <= 108 → trigger.
    triggered, exit_price, reason = SnapshotBacktestRunner._check_exit(
        spec=spec,
        entry_price=100.0,
        entry_idx=0,
        current_idx=3,
        bar_high=110.0,
        bar_low=107.0,
        bar_close=108.0,
    )
    assert triggered is True
    assert reason == "trailing_stop"
    # peak doesn't fall — bar_high=110 < 112 so peak stays 112
    assert spec["running_peak"] == 112.0
    assert exit_price == 108.0


def test_short_running_peak_ratchets_down_across_bars():
    """Short: running low should fall, stop should descend."""
    spec = {
        "type": "atr_trailing_stop",
        "side": "short",
        "trail_mult": 2.0,
        "atr_at_entry": 2.0,
        "running_peak": 100.0,
        "max_bars": 9999,
    }
    # Bar 1: low=95, high=98 → peak 95, stop 99. bar_high=98 < 99, no fire.
    triggered, _, _ = SnapshotBacktestRunner._check_exit(
        spec=spec, entry_price=100.0, entry_idx=0, current_idx=1,
        bar_high=98.0, bar_low=95.0, bar_close=96.0,
    )
    assert triggered is False
    assert spec["running_peak"] == 95.0
    assert spec["stop_loss_price"] == 99.0

    # Bar 2: low=90, high=93 → peak 90, stop 94. high=93 < 94, no fire.
    triggered, _, _ = SnapshotBacktestRunner._check_exit(
        spec=spec, entry_price=100.0, entry_idx=0, current_idx=2,
        bar_high=93.0, bar_low=90.0, bar_close=91.0,
    )
    assert triggered is False
    assert spec["running_peak"] == 90.0
    assert spec["stop_loss_price"] == 94.0


# ---------------------------------------------------------------------------
# 5. max_bars still fires when trailing stop never triggers
# ---------------------------------------------------------------------------


def test_max_bars_still_fires_for_atr_trailing_stop():
    """Trail stop never pierces, but bars_held >= max_bars → max_bars exit."""
    spec = {
        "type": "atr_trailing_stop",
        "side": "long",
        "trail_mult": 2.0,
        "atr_at_entry": 2.0,
        "running_peak": 100.0,
        "max_bars": 5,
    }
    # bar_high=110, bar_low=108 → peak 110, stop 106. low not pierced.
    # current_idx - entry_idx = 5 - 0 = 5 → max_bars timeout.
    triggered, exit_price, reason = SnapshotBacktestRunner._check_exit(
        spec=spec, entry_price=100.0, entry_idx=0, current_idx=5,
        bar_high=110.0, bar_low=108.0, bar_close=109.0,
    )
    assert triggered is True
    assert reason == "max_bars"
    assert exit_price == 109.0
    # ``running_peak`` is deliberately NOT updated on a max_bars exit.
    # max_bars is a pure time condition that is already true at this bar's OPEN,
    # so it is now evaluated before the intra-bar trailing-stop block (which is
    # what mutates running_peak). Ordering it after the stop check let a stop
    # that was only touched later in the same bar pre-empt an exit that had
    # already filled at the open, and made the event-driven and vectorized
    # engines disagree — see tests/standard_bot/test_engine_parity.py.
    # The peak value has no behavioural effect here: the position is closing and
    # the runner drops ``position_exit_spec`` immediately afterwards.
    assert spec["running_peak"] == 100.0


# ---------------------------------------------------------------------------
# 6. _finalize_exit_prices re-anchors running_peak to fill_price
# ---------------------------------------------------------------------------


def test_finalize_exit_prices_anchors_running_peak_to_fill():
    spec = {
        "type": "atr_trailing_stop",
        "trail_mult": 2.0,
        "atr_at_entry": 5.0,
        "running_peak": 100.0,  # plugin's emit-bar close
    }
    out = SnapshotBacktestRunner._finalize_exit_prices(
        spec, fill_price=101.5, side="long"
    )
    assert out["running_peak"] == 101.5
    # initial stop_loss_price is fill - trail*atr = 101.5 - 10 = 91.5
    assert out["stop_loss_price"] == 91.5
    # original spec is not mutated (copy)
    assert spec["running_peak"] == 100.0


def test_finalize_exit_prices_short_anchors_running_peak_to_fill():
    spec = {
        "type": "atr_trailing_stop",
        "trail_mult": 2.0,
        "atr_at_entry": 5.0,
        "running_peak": 200.0,
    }
    out = SnapshotBacktestRunner._finalize_exit_prices(
        spec, fill_price=199.0, side="short"
    )
    assert out["running_peak"] == 199.0
    # short stop = fill + trail*atr = 199 + 10 = 209
    assert out["stop_loss_price"] == 209.0


# ---------------------------------------------------------------------------
# 7. Vectorized engine — _check_exit_long / _short atr_trailing_stop
# ---------------------------------------------------------------------------


def test_vectorized_long_trailing_stop_triggers_and_returns_new_peak():
    cfg = {"type": "atr_trailing_stop", "trail_mult": 2.0, "max_bars": 9999}
    triggered, ex_px, reason, new_peak = _check_exit_long(
        i=5,
        entry_price=100.0,
        entry_idx=0,
        bar_open=109.0,
        bar_high=110.0,
        bar_low=105.0,
        bar_close=106.0,
        atr_at_entry=2.0,
        opposite_signal=False,
        ma_val=None,
        exit_cfg=cfg,
        running_peak=110.0,  # prior peak
    )
    assert triggered is True
    assert reason == "trailing_stop"
    assert ex_px == 106.0  # dynamic stop = 110 - 4
    assert new_peak == 110.0


def test_vectorized_long_trailing_peak_ratchets_up():
    """Across two calls, the new_peak rises with bar_high."""
    cfg = {"type": "atr_trailing_stop", "trail_mult": 2.0, "max_bars": 9999}
    # Bar 1: high=105, low=102 → peak becomes 105, stop=101, no fire
    triggered, _, _, peak1 = _check_exit_long(
        i=1, entry_price=100.0, entry_idx=0,
        bar_open=100.0, bar_high=105.0, bar_low=102.0, bar_close=104.0,
        atr_at_entry=2.0, opposite_signal=False, ma_val=None,
        exit_cfg=cfg, running_peak=100.0,
    )
    assert triggered is False
    assert peak1 == 105.0

    # Bar 2: high=112, low=109 → peak becomes 112, stop=108, no fire
    triggered, _, _, peak2 = _check_exit_long(
        i=2, entry_price=100.0, entry_idx=0,
        bar_open=105.0, bar_high=112.0, bar_low=109.0, bar_close=111.0,
        atr_at_entry=2.0, opposite_signal=False, ma_val=None,
        exit_cfg=cfg, running_peak=peak1,
    )
    assert triggered is False
    assert peak2 == 112.0


def test_vectorized_short_trailing_stop_triggers():
    cfg = {"type": "atr_trailing_stop", "trail_mult": 2.0, "max_bars": 9999}
    triggered, ex_px, reason, new_peak = _check_exit_short(
        i=5,
        entry_price=100.0,
        entry_idx=0,
        bar_open=93.0,
        bar_high=95.0,
        bar_low=90.0,
        bar_close=94.0,
        atr_at_entry=2.0,
        opposite_signal=False,
        ma_val=None,
        exit_cfg=cfg,
        running_peak=90.0,  # running low
    )
    assert triggered is True
    assert reason == "trailing_stop"
    assert ex_px == 94.0  # 90 + 2*2
    assert new_peak == 90.0


def test_vectorized_max_bars_still_fires_for_atr_trailing_stop():
    cfg = {"type": "atr_trailing_stop", "trail_mult": 2.0, "max_bars": 3}
    triggered, ex_px, reason, _ = _check_exit_long(
        i=3, entry_price=100.0, entry_idx=0,
        bar_open=109.0, bar_high=110.0, bar_low=108.0, bar_close=109.0,
        atr_at_entry=2.0, opposite_signal=False, ma_val=None,
        exit_cfg=cfg, running_peak=110.0,
    )
    assert triggered is True
    assert reason == "max_bars"
    assert ex_px == 109.0  # bar_open
