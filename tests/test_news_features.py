"""News feature extraction: dedupe / entity / classify / sentiment / aggregate."""

from __future__ import annotations

import pandas as pd
import pytest

from cyqnt_trd.blocks import news_features as NF

NOW = pd.Timestamp("2026-08-06T07:00:00Z")
UNIVERSE = ["ABCUSDT", "SOLUSDT", "BTCUSDT", "ETHUSDT", "DOGEUSDT", "ADAUSDT"]


def _event(event_id, title, tickers=(), minutes_ago=10, tendency=0,
           reliability=0.6, body=""):
    return {
        "event_id": event_id, "title": title, "summary": "", "body": body,
        "event_time": NOW - pd.Timedelta(minutes=minutes_ago),
        "available_time": NOW - pd.Timedelta(minutes=minutes_ago - 2),
        "source_id": "square.getNews", "topic": "unclassified",
        "tickers": list(tickers), "tendency": tendency,
        "source_reliability": reliability,
    }


def _frame(*events):
    return pd.DataFrame(list(events))


# --------------------------------------------------------------------------
# dedupe + freshness
# --------------------------------------------------------------------------


def test_exact_reposts_collapse_into_one_cluster():
    out = NF.dedupe_events(_frame(
        _event("e1", "Binance Will List ABC", minutes_ago=40),
        _event("e2", "Binance Will List ABC", minutes_ago=30),
        _event("e3", "Binance Will List ABC", minutes_ago=20),
    ))
    assert out["repost_count"].tolist() == [3, 3, 3]
    assert out["is_duplicate"].tolist() == [False, True, True]
    assert set(out.loc[out["is_duplicate"], "duplicate_of"]) == {"e1"}


def test_first_seen_at_is_the_story_not_the_copy():
    out = NF.dedupe_events(_frame(
        _event("e1", "Same story", minutes_ago=40),
        _event("e2", "Same story", minutes_ago=5),
    ))
    assert out["first_seen_at"].nunique() == 1
    # the late copy is late by the full gap, not by its own 2-minute lag
    late = out[out["event_id"] == "e2"].iloc[0]
    assert late["lead_time_seconds"] == pytest.approx(37 * 60)


def test_a_unique_story_is_not_marked_as_a_repost():
    out = NF.dedupe_events(_frame(
        _event("e1", "Binance will list ABC"),
        _event("e2", "SOL bridge exploited overnight"),
    ))
    assert out["is_duplicate"].tolist() == [False, False]
    assert out["repost_count"].tolist() == [1, 1]


def test_a_story_already_in_the_corpus_is_recognised():
    known = NF.dedupe_events(_frame(_event("old", "Binance Will List ABC", minutes_ago=120)))
    out = NF.dedupe_events(_frame(_event("new", "Binance Will List ABC")), known=known)
    assert out.iloc[0]["duplicate_of"] == "old"
    assert bool(out.iloc[0]["is_duplicate"]) is True


def test_near_duplicates_need_real_overlap():
    """Shingle overlap catches rewrites, not arbitrary paraphrase.

    Documented limitation of this first version: a short headline reworded
    heavily reads as a new story. Exact reposts — the common case — are caught.
    """
    out = NF.dedupe_events(_frame(
        _event("e1", "Solana network halted after validator bug was found today"),
        _event("e2", "Solana network halted after validator bug was found this morning"),
    ))
    assert out["repost_count"].tolist() == [2, 2]


# --------------------------------------------------------------------------
# entity resolution
# --------------------------------------------------------------------------


def test_a_ticker_written_about_in_the_text_resolves():
    out = NF.resolve_instruments(
        _frame(_event("e1", "Binance will list ABC", tickers=["ABC"])),
        universe=UNIVERSE,
    )
    assert out.iloc[0]["instrument_id"] == "ABCUSDT"
    assert out.iloc[0]["entity_source"] == "text"
    assert out.iloc[0]["entity_confidence"] >= 0.9


def test_a_tag_nobody_wrote_about_is_a_tag_not_a_subject():
    out = NF.resolve_instruments(
        _frame(_event("e1", "Markets are quiet today", tickers=["SOL"])),
        universe=UNIVERSE,
    )
    assert out.iloc[0]["instrument_id"] is None
    assert out.iloc[0]["entity_source"] == "publisher_only"


def test_publisher_tags_can_be_trusted_only_when_asked():
    out = NF.resolve_instruments(
        _frame(_event("e1", "Markets are quiet today", tickers=["SOL"])),
        universe=UNIVERSE, require_mention=False,
    )
    assert out.iloc[0]["instrument_id"] == "SOLUSDT"
    assert out.iloc[0]["entity_source"] == "publisher"
    assert out.iloc[0]["entity_confidence"] < 0.5   # explicitly weak


def test_a_post_tagged_with_everything_targets_nothing():
    out = NF.resolve_instruments(
        _frame(_event("e1", "BTC ETH SOL DOGE ADA all pumping",
                      tickers=["BTC", "ETH", "SOL", "DOGE", "ADA"])),
        universe=UNIVERSE,
    )
    assert out.iloc[0]["instrument_id"] is None
    assert out.iloc[0]["entity_source"] == "ambiguous"


def test_tickers_outside_the_universe_are_ignored():
    out = NF.resolve_instruments(
        _frame(_event("e1", "NOTACOIN launches", tickers=["NOTACOIN"])),
        universe=UNIVERSE,
    )
    assert out.iloc[0]["instrument_id"] is None


def test_market_wide_news_is_left_unattached():
    out = NF.resolve_instruments(
        _frame(_event("e1", "Fed holds interest rate steady")), universe=UNIVERSE
    )
    assert out.iloc[0]["instrument_id"] is None
    assert out.iloc[0]["entity_source"] == "none"


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize("title,expected", [
    ("Binance Will List ABC", "listing"),
    ("Binance will delist XYZ", "delisting"),
    ("Bridge exploit drains $40m", "security_incident"),
    ("ABC token unlock scheduled", "token_unlock"),
    ("SEC lawsuit against exchange", "regulation"),
    ("Ethereum mainnet upgrade", "mainnet_upgrade"),
    ("Fed holds interest rate steady", "macro"),
    ("Wallet maintenance window", "maintenance"),
    ("Nothing in particular happened", "unclassified"),
])
def test_event_type_is_assigned_from_the_text(title, expected):
    out = NF.classify_events(_frame(_event("e1", title)))
    assert out.iloc[0]["event_type"] == expected


def test_half_life_differs_by_type():
    """A listing pulse and an unlock overhang want different holding periods."""
    out = NF.classify_events(_frame(
        _event("e1", "Binance Will List ABC"),
        _event("e2", "ABC token unlock scheduled"),
    ))
    listing, unlock = out.iloc[0], out.iloc[1]
    assert listing["expected_half_life_seconds"] < unlock["expected_half_life_seconds"]


# --------------------------------------------------------------------------
# sentiment
# --------------------------------------------------------------------------


def test_lexicon_reads_direction_and_reports_who_judged_it():
    out = NF.score_sentiment(NF.classify_events(
        _frame(_event("e1", "SOL plunges after exploit drains bridge"))
    ))
    row = out.iloc[0]
    assert row["sentiment_score"] < 0
    assert row["sentiment_label"] == "bearish"
    assert row["sentiment_source"] == "lexicon"


def test_event_type_carries_direction_when_the_wording_is_flat():
    out = NF.score_sentiment(NF.classify_events(
        _frame(_event("e1", "Binance Will List ABC"))
    ))
    assert out.iloc[0]["sentiment_source"] == "event_type"
    assert out.iloc[0]["sentiment_score"] > 0


def test_publisher_tendency_is_only_a_labelled_fallback():
    out = NF.score_sentiment(NF.classify_events(
        _frame(_event("e1", "Some update about the project", tendency=1))
    ))
    row = out.iloc[0]
    assert row["sentiment_source"] == "publisher"
    # the author grading their own post is worth little, and says so
    assert row["sentiment_confidence"] <= 0.3
    assert row["sentiment_label"] == "neutral"


def test_publisher_fallback_can_be_switched_off():
    out = NF.score_sentiment(
        NF.classify_events(_frame(_event("e1", "Some update", tendency=1))),
        use_publisher_fallback=False,
    )
    assert out.iloc[0]["sentiment_source"] == "none"
    assert out.iloc[0]["sentiment_score"] == 0.0


def test_low_confidence_scores_refuse_a_label():
    out = NF.score_sentiment(
        NF.classify_events(_frame(_event("e1", "Nothing happened"))),
        min_confidence=0.9,
    )
    assert out.iloc[0]["sentiment_label"] == "neutral"


def test_disagreement_with_the_publisher_is_flagged():
    out = NF.score_sentiment(NF.classify_events(
        _frame(_event("e1", "Token crashes after exploit", tendency=1))   # author says bullish
    ))
    assert bool(out.iloc[0]["sentiment_disagrees_with_publisher"]) is True


# --------------------------------------------------------------------------
# aggregation to MetricFrame
# --------------------------------------------------------------------------


def _enriched():
    return NF.enrich_events(_frame(
        _event("e1", "Binance Will List ABC", ["ABC"], minutes_ago=40, reliability=0.85),
        _event("e2", "Binance Will List ABC", ["ABC"], minutes_ago=30, reliability=0.5),
        _event("e3", "SOL plunges after exploit drains bridge", ["SOL"],
               minutes_ago=15, tendency=2, reliability=0.8),
        _event("e4", "Fed holds interest rate steady", minutes_ago=5),
    ), universe=UNIVERSE)


def test_aggregation_emits_metric_frame_rows():
    metrics = NF.aggregate_to_metrics(_enriched(), as_of=NOW)
    required = {"event_time", "available_time", "instrument_id", "metric", "value"}
    assert required <= set(metrics.columns)
    assert set(metrics["instrument_id"]) == {"ABCUSDT", "SOLUSDT"}   # macro is unattached


def test_duplicates_do_not_inflate_the_count():
    metrics = NF.aggregate_to_metrics(_enriched(), as_of=NOW)
    count = metrics[(metrics["instrument_id"] == "ABCUSDT")
                    & (metrics["metric"] == "news_count_1h")]["value"].iloc[0]
    assert count == 1.0   # e2 is a repost of e1


def test_reposts_still_register_as_attention():
    metrics = NF.aggregate_to_metrics(_enriched(), as_of=NOW)
    reposts = metrics[(metrics["instrument_id"] == "ABCUSDT")
                      & (metrics["metric"] == "news_repost_max_1h")]["value"].iloc[0]
    assert reposts == 2.0


def test_bull_ratio_and_reliability_land_per_instrument():
    metrics = NF.aggregate_to_metrics(_enriched(), as_of=NOW)

    def value(instrument, metric):
        rows = metrics[(metrics["instrument_id"] == instrument)
                       & (metrics["metric"] == metric)]
        return rows["value"].iloc[0] if not rows.empty else None

    assert value("ABCUSDT", "news_bull_ratio_1h") == 1.0
    assert value("SOLUSDT", "news_bull_ratio_1h") == 0.0
    assert value("ABCUSDT", "news_max_reliability_1h") == pytest.approx(0.85)
    assert value("SOLUSDT", "news_sentiment_mean_1h") < 0


def test_seconds_since_last_is_emitted_per_instrument():
    metrics = NF.aggregate_to_metrics(_enriched(), as_of=NOW)
    rows = metrics[metrics["metric"] == "news_seconds_since_last"]
    assert set(rows["instrument_id"]) == {"ABCUSDT", "SOLUSDT"}
    assert (rows["value"] > 0).all()


def test_events_after_the_cutoff_are_excluded():
    """The aggregate is point-in-time: nothing published after as_of counts."""
    metrics = NF.aggregate_to_metrics(_enriched(), as_of=NOW - pd.Timedelta(minutes=20))
    assert "SOLUSDT" not in set(metrics["instrument_id"])   # SOL event was 15m ago


def test_the_output_validates_as_a_metric_frame():
    from cyqnt_trd.standard_bot.core.input_contract import METRIC_FRAME

    METRIC_FRAME.validate(NF.aggregate_to_metrics(_enriched(), as_of=NOW), node="news")


def test_empty_input_yields_an_empty_typed_frame():
    metrics = NF.aggregate_to_metrics(pd.DataFrame(), as_of=NOW)
    assert metrics.empty
    assert "metric" in metrics.columns


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------


def test_enrich_runs_the_stages_in_dependency_order():
    out = _enriched()
    # classification must have happened before sentiment, since the type feeds it
    listing = out[out["event_id"] == "e1"].iloc[0]
    assert listing["event_type"] == "listing"
    assert listing["sentiment_source"] in ("event_type", "lexicon")
    # dedupe must have happened before aggregation
    assert "is_duplicate" in out.columns
    assert "entity_source" in out.columns


def test_enrich_is_a_noop_on_empty_input():
    assert NF.enrich_events(pd.DataFrame(), universe=UNIVERSE).empty
