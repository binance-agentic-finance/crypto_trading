"""Coin selection, from the unified bundle to the published signal.

Selection was the half of the output contract nothing had actually exercised
end to end, and running it once surfaced three defects that chained into a
wrong answer rather than an error:

1. ``ticker_rank`` has two vocabularies. Square's raw frame carries
   ``bullish_count`` / ``bearish_count``; the canonical ``RankFrame@1.0`` that
   ``build_input_bundle`` emits carries ``bull_ratio`` and no counts at all.
   ``augment_with_news`` only understood the first, so every canonical row got
   ``news_bull_ratio = NaN``.
2. ``float(row.get("news_bull_ratio") or 0.5)`` does not default a NaN — NaN is
   truthy — and ``NaN >= long_ratio`` is False, so every candidate came back
   **short**: a basket of short recommendations manufactured from missing data.
3. ``min(100.0, nan)`` returns ``100.0``, so the resulting NaN score arrived at
   the contract already laundered into MAXIMUM strength, and the ``0 <= score
   <= 100`` check passed it.

None of the three raised. The signal was well-formed, schema-conformant and
wrong, which is the only failure mode worth writing a test file about.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from cyqnt_trd.blocks import universe as U
from cyqnt_trd.standard_bot.bot import BotContext
from cyqnt_trd.standard_bot.core import (Direction, MarketScope, PositionIntent,
                                         Provenance, SignalKind, StandardSignal)
from strategies.standard.blocks_reference_bots import BlocksNewsRankBot

DT = 1_776_322_799_999
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]


def _universe():
    return pd.DataFrame({"symbol": SYMBOLS, "quoteVolume": [5e8, 3e8, 2e8, 1.5e8],
                         "available_time": [DT] * 4})


def _canonical_rank():
    """RankFrame@1.0 — what ``build_input_bundle`` writes. Ratio, no counts."""
    return pd.DataFrame({"symbol": SYMBOLS, "rank": [1, 2, 3, 4],
                         "mention_count": [420, 180, 95, 60],
                         "bull_ratio": [0.88, 0.22, 0.61, 0.49],
                         "available_time": [DT] * 4})


def _raw_square_rank():
    """Square's own shape — counts, no ratio."""
    return pd.DataFrame({"ticker": ["BTC", "ETH", "SOL", "DOGE"],
                         "rank": [1, 2, 3, 4],
                         "mention_count": [420, 180, 95, 60],
                         "bullish_count": [88, 22, 61, 49],
                         "bearish_count": [12, 78, 39, 51],
                         "available_time": [DT] * 4})


def _decide(rank_frame):
    ctx = BotContext(decision_time=DT,
                     frames={"universe": _universe(), "ticker_rank": rank_frame})
    return BlocksNewsRankBot().decide_checked(ctx)


# --------------------------------------------------------------------------- #
# both ticker_rank vocabularies must give the same answer                      #
# --------------------------------------------------------------------------- #


def test_the_canonical_rank_frame_produces_real_sentiment():
    """The bundle's own vocabulary. This produced all-NaN before."""
    joined = U.augment_with_news(U.filter_quote_volume(_universe(), 1e8),
                                 _canonical_rank())
    assert joined["news_bull_ratio"].notna().all(), joined[["symbol", "news_bull_ratio"]]
    assert joined.set_index("symbol").loc["BTCUSDT", "news_bull_ratio"] == pytest.approx(0.88)


def test_the_raw_square_frame_still_derives_the_ratio_from_counts():
    joined = U.augment_with_news(U.filter_quote_volume(_universe(), 1e8),
                                 _raw_square_rank())
    assert joined.set_index("symbol").loc["BTCUSDT", "news_bull_ratio"] == pytest.approx(0.88)


def test_the_two_vocabularies_agree():
    """Same underlying sentiment, two encodings, one answer — otherwise the
    result depends on which upstream happened to feed the bot."""
    canonical = {c["symbol"]: c["direction"]
                 for c in _decide(_canonical_rank())[0].to_dict()["candidates"]}
    raw = {c["symbol"]: c["direction"]
           for c in _decide(_raw_square_rank())[0].to_dict()["candidates"]}
    assert canonical == raw, (canonical, raw)


# --------------------------------------------------------------------------- #
# missing sentiment must not become a short recommendation                     #
# --------------------------------------------------------------------------- #


def test_a_name_with_no_sentiment_is_dropped_not_shorted():
    rank = _canonical_rank()
    rank.loc[rank["symbol"] == "SOLUSDT", "bull_ratio"] = float("nan")
    signal = _decide(rank)[0]
    picked = {c.symbol for c in signal.candidates}
    assert "SOLUSDT" not in picked, "ranked a name whose sentiment was unknown"
    assert any("no sentiment" in w for w in signal.warnings), signal.warnings


def test_no_sentiment_at_all_yields_no_signal():
    rank = _canonical_rank()
    rank["bull_ratio"] = float("nan")
    assert _decide(rank) == []


def test_directions_follow_the_ratio():
    candidates = {c.symbol: c.direction for c in _decide(_canonical_rank())[0].candidates}
    assert candidates["BTCUSDT"] is Direction.LONG      # 0.88
    assert candidates["ETHUSDT"] is Direction.SHORT     # 0.22
    assert candidates["DOGEUSDT"] is Direction.SHORT    # 0.49, just under 0.55


# --------------------------------------------------------------------------- #
# the contract refuses a laundered NaN                                         #
# --------------------------------------------------------------------------- #


def test_a_nan_score_is_refused_by_the_contract():
    with pytest.raises(ValueError, match="score is NaN"):
        StandardSignal(bot_id="t", decision_time=DT, provenance=Provenance(strategy_id="t"),
                       symbol="BTCUSDT", score=float("nan"))


def test_a_nan_confidence_is_refused_by_the_contract():
    with pytest.raises(ValueError, match="confidence is NaN"):
        StandardSignal(bot_id="t", decision_time=DT, provenance=Provenance(strategy_id="t"),
                       symbol="BTCUSDT", confidence=float("nan"))


def test_min_would_have_hidden_it():
    """Why the explicit NaN check exists rather than relying on the range test.

    ``min(100.0, nan)`` is 100.0 and ``0 <= 100 <= 100`` holds, so the clamp a
    producer writes to stay inside the contract is exactly what smuggles the
    NaN past it — as maximum strength.
    """
    assert min(100.0, float("nan")) == 100.0
    assert not (0.0 <= float("nan") <= 100.0)


# --------------------------------------------------------------------------- #
# the published shape                                                          #
# --------------------------------------------------------------------------- #


def test_selection_output_is_the_same_schema_as_a_trade_signal():
    selection = _decide(_canonical_rank())[0].to_dict()
    trade = StandardSignal(
        bot_id="t", decision_time=DT, provenance=Provenance(strategy_id="t"),
        symbol="BTCUSDT", intent=PositionIntent.OPEN_LONG,
        exit_plan=__import__("cyqnt_trd.standard_bot.core", fromlist=["ExitPlan"])
        .ExitPlan(exit_on_opposite_signal=True),
    ).to_dict()

    assert set(selection) == set(trade), set(selection) ^ set(trade)
    assert selection["schema"] == trade["schema"] == "cyqnt.signal/v2"
    assert selection["kind"] == "selection" and trade["kind"] == "trade"
    assert selection["market_scope"] == MarketScope.CROSS_SECTION.value
    assert selection["entry"] is None and selection["exit_plan"] is None
    assert trade["candidates"] == []


def test_selection_runs_off_the_unified_input_bundle():
    """The whole point: one bundle in, one v2 signal out, no bespoke plumbing."""
    from pathlib import Path

    from cyqnt_trd.standard_bot.bot import _frames_from_snapshot
    from cyqnt_trd.standard_bot.data import load_input_bundle

    repo = Path(__file__).resolve().parents[2]
    bundle = repo / "docs/standard_bot_io/samples/input_bundle_example.json"
    if not bundle.exists():                       # generated artifact
        pytest.skip("input bundle sample not generated")

    snapshot = load_input_bundle(str(bundle))
    ctx = BotContext(decision_time=snapshot.meta.decision_as_of,
                     frames=_frames_from_snapshot(snapshot))
    signals = BlocksNewsRankBot().decide_checked(ctx)
    assert signals, "the bundle carries universe + ticker_rank; expected a basket"
    payload = signals[0].to_dict()
    assert payload["kind"] == "selection"
    assert payload["candidates"]
    assert not math.isnan(payload["score"])
