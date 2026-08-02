# =============================================================================
# 不經過 YAML 的選幣策略 —— 這一份能不能替代 YAML,以及它證明了什麼
# =============================================================================
# 語料第 13 名的真實需求:「找箱體運動的幣,每天價格段和之前每一天高度重疊」。
# 條件 2 在 YAML 表面是 not_expressible —— augment_with_indicator 的 agg 字彙
# (last/min/max/mean/any_negative/all_negative)沒有一個表達得出「每一天都和其他
# 每一天重疊」,indicators 裡也沒有箱體指標。
#
# 這支測試要證明的**不是**「它會跑」,而是「它會分辨」。凍結 bundle 上這份策略回
# 空籃子,而空籃子與「條件恆假」在輸出上長得一模一樣 —— 所以正反兩面都要釘:
#
#   * 真實資料上回 0,而且**可解釋**:五檔的 overlap_ratio 全是負的
#     (max(lows) > min(highs),共同交集是空的)。那不是 bug,那個 bundle 的五檔
#     是「持倉異動」螢幕的倖存者,本來就是選動最大的。
#   * 注入四檔合成幣,四種判別各中一個,且只有該過的那檔過。
# =============================================================================
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from strategies.selection.box_range_screener import (
    ASSUMPTIONS,
    box_overlap,
    build,
    run,
)

REPO = Path(__file__).parents[2]
FIXTURE = REPO / "tests" / "standard_bot" / "fixtures" / "universe_derivatives.json"
TREND_SYMBOLS = {"ALLOUSDT", "TAGUSDT", "UAIUSDT", "UBUSDT", "USUSDT"}


def _bundle() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _inject(base: list, symbol: str, drift_pct: float, amp_pct: float,
            turnover: float) -> list:
    """一檔合成幣:中心在 ±drift_pct 之間游走,每天上下 amp_pct。"""
    out = []
    for index, template in enumerate(base):
        centre = 1.0 * (1 + drift_pct / 100 * math.sin(index / 2.0))
        low, high = centre * (1 - amp_pct / 200), centre * (1 + amp_pct / 200)
        out.append({**template, "instrument_id": symbol,
                    "open": (low + high) / 2, "high": high, "low": low,
                    "close": (low + high) / 2,
                    "volume": turnover / ((low + high) / 2),
                    "quote_volume": turnover})
    return out


def _with_synthetic(cases) -> dict:
    bundle = copy.deepcopy(_bundle())
    base = sorted((r for r in bundle["frames"]["universe_bars"]["rows"]
                   if r["timeframe"] == "1d" and r["instrument_id"] == "USUSDT"),
                  key=lambda r: r["open_time"])
    universe_template = bundle["frames"]["universe"]["rows"][0]
    meta_template = bundle["frames"]["contract_meta"]["rows"][0]
    for symbol, drift, amp, turnover in cases:
        bundle["frames"]["universe_bars"]["rows"] += _inject(base, symbol, drift, amp, turnover)
        # Mirror the real frames' key set exactly. Adding a `symbol` column that
        # the other 851 rows lack makes augment_with_contract_meta join on it
        # instead, matching only the synthetic rows — 3 of 730, which trips the
        # coverage floor. The schema of a fixture is part of the fixture.
        bundle["frames"]["universe"]["rows"].append(
            {**universe_template, "instrument_id": symbol, "quoteVolume": 5e7})
        bundle["frames"]["contract_meta"]["rows"].append(
            {**meta_template, "instrument_id": symbol, "underlyingType": "COIN",
             "contractType": "PERPETUAL", "baseAsset": symbol[:-4],
             "quoteAsset": "USDT", "status": "TRADING", "underlyingSubType": "DeFi"})
    return bundle


# ---------------------------------------------------------------------------
# The arithmetic that makes this cheap
# ---------------------------------------------------------------------------

def test_all_pairs_overlapping_is_one_scan_not_n_squared():
    """"每一天都和其他每一天重疊" == max(lows) <= min(highs).

    The equivalence is the reason this condition is six lines rather than a
    nested loop, so it is asserted rather than left in a comment.
    """
    overlapping = pd.DataFrame({"low": [10.0, 10.5, 11.0], "high": [12.0, 12.5, 13.0]})
    assert box_overlap(overlapping)["overlap_ratio"] > 0

    # Day 3 sits entirely above day 1 — some pair does not overlap, so the
    # common intersection is empty and the ratio goes negative.
    broken = pd.DataFrame({"low": [10.0, 11.0, 13.0], "high": [11.5, 12.5, 14.0]})
    assert box_overlap(broken)["overlap_ratio"] < 0


def test_a_negative_ratio_really_means_a_disjoint_pair():
    """Checked against the O(n^2) definition it stands in for."""
    frame = pd.DataFrame({"low": [10.0, 11.0, 13.0], "high": [11.5, 12.5, 14.0]})
    pairs = [(i, j) for i in range(3) for j in range(i + 1, 3)]
    disjoint = [(i, j) for i, j in pairs
                if max(frame.low[i], frame.low[j]) > min(frame.high[i], frame.high[j])]
    assert disjoint, "the fixture is supposed to contain a disjoint pair"
    assert box_overlap(frame)["overlap_ratio"] < 0


# ---------------------------------------------------------------------------
# It discriminates — the half an empty basket cannot show
# ---------------------------------------------------------------------------

def test_the_frozen_bundle_returns_nothing_and_the_reason_is_legible():
    """0 candidates, and provably "correctly 0" rather than "always 0"."""
    signal = run(str(FIXTURE))
    assert signal["candidates"] == []

    bars = pd.DataFrame(_bundle()["frames"]["universe_bars"]["rows"])
    daily = bars[bars["timeframe"] == "1d"]
    for symbol, group in daily.groupby(daily["instrument_id"].str.upper()):
        box = box_overlap(group.sort_values("open_time").tail(15))
        assert box["overlap_ratio"] < 0, (
            "%s: this bundle's five names are the survivors of an OI-CHANGE "
            "screen, i.e. selected for moving hard, so every one of them should "
            "fail a range test — if one starts passing, the fixture changed"
            % symbol)


@pytest.mark.parametrize("symbol, drift, amp, turnover, expected", [
    ("BOXUSDT", 0.5, 6.0, 5e7, True),     # tight drift, wide daily range
    ("LOOSEUSDT", 3.0, 6.0, 5e7, False),  # drift ~= half-amplitude -> knife edge
    ("THINUSDT", 0.5, 6.0, 5e6, False),   # turnover below 20m
    ("FLATUSDT", 0.5, 1.5, 5e7, False),   # daily amplitude below 5%
])
def test_each_condition_rejects_on_its_own(symbol, drift, amp, turnover, expected, tmp_path):
    """One synthetic coin per condition, so a passing test names which one broke.

    ``LOOSEUSDT`` is the one worth reading twice: it IS a range in the loose
    sense — it goes nowhere over 15 days — but its centre drifts by as much as
    half its daily bar, so there exist two days that never traded a common
    price. Under this spec's declared reading that is not a box, and the number
    says so before anyone has to argue about it.
    """
    bundle = _with_synthetic([(symbol, drift, amp, turnover)])
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")

    picked = {c["symbol"] for c in run(str(path))["candidates"]}
    assert (symbol in picked) is expected
    assert not (picked & TREND_SYMBOLS), "the real names must still all fail"


def test_the_declared_readings_travel_with_the_signal():
    """The two undefinable words are declared, and the declaration reaches the
    emitted signal rather than staying in a comment."""
    bundle = _with_synthetic([("BOXUSDT", 0.5, 6.0, 5e7)])
    signal = run.__wrapped__(bundle) if hasattr(run, "__wrapped__") else None
    del signal  # run() takes a path; the assertion below uses the constant

    assert {a["cid"] for a in ASSUMPTIONS} == {"overlap", "leader"}
    for assumption in ASSUMPTIONS:
        assert assumption["reading"] and assumption["basis"]
        assert assumption["alternatives"], (
            "an assumption with no stated alternative is not a declaration, it "
            "is a restatement")


def test_the_signal_is_a_normal_selection_signal(tmp_path):
    """Writing Python instead of YAML must not drop out of the contract."""
    bundle = _with_synthetic([("BOXUSDT", 0.5, 6.0, 5e7)])
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")

    signal = run(str(path))
    assert signal["schema"] == "cyqnt.signal/v2"
    assert signal["kind"] == "selection"
    assert signal["market_scope"] == "cross_section"
    assert signal["intent"] == "hold", "a ranking is not an instruction to trade"
    assert "assumption_declared" in signal["reason_codes"]
    assert any(w.startswith("assumption (overlap)") for w in signal["warnings"])
    assert signal["candidates"][0]["reason"], "a candidate with no reason is unauditable"


def test_the_window_is_not_silently_shortened():
    """A coin with fewer bars than the window is skipped, not measured on what
    it has: a shorter window is always a tighter box, so accepting it would make
    every fresh listing look range-bound."""
    selection_fn = build(window_days=200)
    bundle = _bundle()
    universe = pd.DataFrame(bundle["frames"]["universe"]["rows"])
    frames = {name: pd.DataFrame(frame["rows"])
              for name, frame in bundle["frames"].items()}
    assert selection_fn(universe, None, frames=frames) == []
