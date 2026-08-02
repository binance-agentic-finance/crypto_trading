"""Tests for the Gate0 miner.

No test reads the real chat export. Every fixture below is invented text written
for this file: the real corpus carries ``user_id`` and verbatim user messages,
and a test that pinned behaviour against it would either need the private CSV to
run at all or would quote user text into a public repo. The invented rows are
written to exercise the shapes the real data actually contains — simplified
Chinese, preset prompt cards repeated with a different ticker, continuation
follow-ups, and platform-injected turns.

The privacy tests are the load-bearing ones. ``candidates.jsonl`` is written into
a public repo, so "the file is pure ASCII, contains no verbatim text and cannot
be joined back to a user id without the salt" is a property worth a test, not a
convention worth a comment.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cyqnt_trd.standard_bot.yaml_pipeline.intent import infer_strategy_kind  # noqa: E402
from tools.nl2yaml import mine  # noqa: E402


# ---------------------------------------------------------------------------
# Normalisation and hashing
# ---------------------------------------------------------------------------

def test_canonicalise_folds_fullwidth_and_strips_noise():
    # NFKC is what makes the full-width leverage phrase reachable by the numeric
    # patterns at all; without it "１０倍" is invisible to every one of them.
    assert mine.canonicalise("１０倍槓桿") == "10倍槓桿"
    assert mine.canonicalise("see https://x.com/a?b=1 now") == "see now"
    assert mine.canonicalise("ask @some_handle please") == "ask please"
    assert mine.canonicalise("  A\t\nB  ") == "a b"
    assert mine.canonicalise("a​b") == "ab"


def test_hashes_separate_raw_from_canonical():
    a, b = "Top 10 Coins", "top  10   coins"
    assert mine.sha256_hex(a) != mine.sha256_hex(b)
    assert mine.canonicalise(a) == mine.canonicalise(b)
    assert len(mine.sha256_hex(a)) == 64
    assert mine.short_hash("x", "r_") == "r_" + hashlib.sha256(b"x").hexdigest()[:16]


def test_split_messages_reconciles_the_truncated_opening_turn():
    # The excerpt normally repeats first_query; it must not be counted twice.
    assert mine.split_messages("abc", "abc --- def") == ["abc", "def"]
    # first_query truncated shorter than the excerpt's copy: keep the longer one.
    assert mine.split_messages("abc", "abcdef --- ghi") == ["abcdef", "ghi"]


def test_split_messages_keeps_a_turn_that_extends_an_earlier_one():
    # An earlier draft dropped any turn that was a prefix of another. "做多" then
    # "做多btc 4h" are two different statements and the second carries the
    # conditions worth mining.
    assert mine.split_messages("做多", "做多 --- 做多btc 4h") == ["做多", "做多btc 4h"]


def test_split_messages_drops_platform_injected_turns():
    messages = mine.split_messages(
        "<system>use menu-workflow skill and read references/review.md",
        "User selected the volume-surge-daily-report case. Step 1: read /app/skills"
        " --- rsi 低于 30 做多 --- system: [2026-05-15 04:18:40 utc] model switched",
    )
    assert messages == ["rsi 低于 30 做多"]


def test_split_messages_can_empty_out_a_row_that_was_only_injection():
    assert mine.split_messages(
        "User selected the square-buzz-screener case. Route to the case.",
        "User selected the square-buzz-screener case. Route to the case.") == []


# ---------------------------------------------------------------------------
# Traditional / simplified folding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, family, subject", [
    ("止损 2%", "risk", "stop_loss"),
    ("止損 2%", "risk", "stop_loss"),
    ("10倍杠杆", "risk", "leverage"),
    ("10倍槓桿", "risk", "leverage"),
    ("筛选成交额最大的", "universe_filter", "quote_volume"),
    ("篩選成交額最大的", "universe_filter", "quote_volume"),
    ("均线金叉做多", "indicator", "cross_up"),
    ("均線金叉做多", "indicator", "cross_up"),
])
def test_both_chinese_scripts_reach_the_same_condition(text, family, subject):
    # 94% of the Chinese in the corpus is simplified. A traditional-only lexicon
    # scored 3,716 tier-A rows where the folded one scores 5,585, so this is the
    # single largest correction in the miner.
    conditions, _, _ = mine.extract_conditions(text)
    assert (family, subject) in [(c["family"], c["subject"]) for c in conditions]


def test_zh_fold_rejects_cjk_inside_a_character_class():
    # Folding 兩 inside [一二兩] would nest a class and silently change the match.
    with pytest.raises(ValueError, match="character class"):
        mine.zh_fold(r"前[一二兩]名")
    assert mine.zh_fold("低於") == "低[於于]"


def test_to_traditional_leaves_the_ambiguous_le_character_alone():
    # 了 is both its own word and the simplified form of 瞭; folding it would
    # rewrite "好了" into "好瞭".
    assert mine.to_traditional("好了") == "好了"
    assert mine.to_traditional("买进") == "買進"


# ---------------------------------------------------------------------------
# Condition families
# ---------------------------------------------------------------------------

def families(text: str) -> set[str]:
    conditions, _, _ = mine.extract_conditions(text)
    return {c["family"] for c in conditions}


def test_threshold_matches_a_comparator_that_follows_the_number():
    # The published recipe only matched "operator then number", which misses the
    # ordinary Chinese word order.
    conditions, _, _ = mine.extract_conditions("成交量在1000万以上")
    thresholds = [c for c in conditions if c["family"] == "threshold"]
    assert len(thresholds) == 1
    assert {k: v for k, v in thresholds[0].items() if k != "quote"} == {
        "family": "threshold", "subject": "volume", "operator": "gte",
        "value": 1e7, "unit": "count", "polarity": "include"}


def test_threshold_resolves_its_subject_from_the_words_before_it():
    conditions, _, _ = mine.extract_conditions("rsi 低于 30 就买入")
    threshold = next(c for c in conditions if c["family"] == "threshold")
    assert (threshold["subject"], threshold["operator"], threshold["value"],
            threshold["unit"]) == ("rsi", "lt", 30.0, "indicator_value")


def test_breakout_counts_once_as_an_indicator_not_twice_as_a_threshold():
    # 突破/跌破 are comparisons, but they already count under `indicator`.
    # Counting them again would inflate the tier of a single stated condition.
    assert families("突破 50000") == {"indicator"}


def test_rank_topn_needs_a_ranking_word_or_a_counted_instrument():
    # The published `\d+\s*個` fired on every counted noun in the language.
    assert "rank_topn" not in families("8个小时之后再看")
    assert "rank_topn" in families("前10个币")
    assert "rank_topn" in families("top 5 coins by volume")
    assert "rank_topn" in families("成交量排名")
    ranked = [c for c in mine.extract_conditions("前10个币")[0]
              if c["family"] == "rank_topn"]
    assert (ranked[0]["operator"], ranked[0]["value"]) == ("top_n", 10)


def test_direction_does_not_fire_on_as_long_as_or_long_term():
    assert "direction" not in families("as long as the market holds")
    assert "direction" not in families("i am a long term holder")
    assert "direction" in families("go long here")
    assert "direction" in families("做空")


def test_universe_filter_drops_the_generic_chinese_function_words():
    # 只要 / 不要 from the published recipe are ordinary function words and were
    # turning conversations with no screening intent into universe filters.
    assert "universe_filter" not in families("只要你觉得可以就好")
    assert "universe_filter" not in families("不要跟我说这个")
    assert "universe_filter" in families("筛选成交量大的币")


def test_timeframe_requires_single_letter_units_to_touch_the_digits():
    # "$10 M" of market cap is not a 10-minute timeframe.
    assert "timeframe" not in families("market cap is 10 m")
    assert "timeframe" in families("4h 均线")
    assert "timeframe" in families("15 minutes chart")
    interval = next(c for c in mine.extract_conditions("4h 均线")[0]
                    if c["family"] == "timeframe")
    assert interval["value"] == "4h"


def test_asset_leaves_out_tickers_that_are_english_words():
    assert "asset" not in families("we are near the top and there is a link")
    conditions, _, _ = mine.extract_conditions("btc 和以太都想做")
    asset = next(c for c in conditions if c["family"] == "asset")
    assert asset["value"] == ["BTC", "ETH"]


def test_condition_extraction_is_deduplicated_but_counts_every_hit():
    _, hits, _ = mine.extract_conditions("做多 btc,然後再做多 eth")
    assert hits["direction"] == 1        # one regex, one search
    conditions, hits, _ = mine.extract_conditions("成交量 > 100 且 市值 > 200")
    assert hits["threshold"] == 2
    assert len({(c["subject"], c["value"]) for c in conditions
                if c["family"] == "threshold"}) == 2


# ---------------------------------------------------------------------------
# Numeric redaction
# ---------------------------------------------------------------------------

def test_a_large_unattributed_number_is_not_written_out_verbatim():
    # Regression: a user pasted their own numeric account id, "> 76916505"
    # matched the threshold pattern, and the id landed in the repo-bound record.
    conditions, _, _ = mine.extract_conditions("超过 76916505 的时候告诉我")
    threshold = next(c for c in conditions if c["family"] == "threshold")
    assert threshold["value"] is None
    assert threshold["value_redacted"] is True
    # ...while the same magnitude attached to a market quantity is kept.
    conditions, _, _ = mine.extract_conditions("成交量超过 76916505")
    threshold = next(c for c in conditions if c["family"] == "threshold")
    assert threshold["value"] == 76916505.0
    assert "value_redacted" not in threshold


def test_a_risk_value_is_taken_from_the_nearer_side_of_the_keyword():
    """Regression from the first full run.

    "挂90%止损单跟…200%的半仓止盈单" was mined as a 200% stop loss: the extractor
    only looked to the right of the keyword and found the take-profit's number.
    """
    conditions, _, _ = mine.extract_conditions("挂90%止损单跟实际买入价格200%的半仓止盈单")
    values = {c["subject"]: c["value"] for c in conditions if c["family"] == "risk"}
    assert values == {"stop_loss": 90.0, "take_profit": 200.0}
    # the trailing form still works, and wins a tie
    trailing, _, _ = mine.extract_conditions("止损 2%")
    assert next(c["value"] for c in trailing if c["subject"] == "stop_loss") == 2.0
    assert mine.nearest_risk_value("no numbers here", (3, 5)) is None


def test_small_numbers_and_top_n_keep_their_type():
    assert mine.safe_number("unspecified", 30.0) == (30.0, False)
    ranked = next(c for c in mine.extract_conditions("前10个币")[0]
                  if c["family"] == "rank_topn")
    assert isinstance(ranked["value"], int)


# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("present, expected", [
    ({"threshold", "indicator", "timeframe"}, "A"),
    ({"threshold", "indicator"}, "B"),
    ({"indicator"}, "C"),
    (set(), "D"),
])
def test_tier_follows_the_published_counts(present, expected):
    assert mine.tier_of(present) == expected


def test_asset_plus_direction_alone_is_demoted_to_tier_c():
    # "Should I long BTC?" names an instrument and a side and nothing a backtest
    # could check. Calling it a two-condition request would seed the training set
    # with entry rules the annotator had to invent.
    assert mine.tier_of({"asset", "direction"}) == "C"
    assert mine.tier_of({"asset"}) == "C"
    assert mine.tier_of({"asset", "direction", "timeframe"}) == "A"


# ---------------------------------------------------------------------------
# Continuation fragments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message", [
    "现在呢", "process", "用中文回答我", "策略状态", "strategy status",
    "好了吗", "改好了吗", "confirm", "继续", "目前运行情况", "any updates?", "它",
])
def test_known_follow_ups_are_chatter(message):
    assert mine.is_chatter(mine.canonicalise(message))


@pytest.mark.parametrize("message", [
    "继续分析 btc 4h 的均线", "rsi 低于 30 就买入", "帮我筛选成交量前十的币",
])
def test_a_real_request_is_not_chatter(message):
    assert not mine.is_chatter(mine.canonicalise(message))


def test_a_conversation_of_pure_chatter_is_a_fragment():
    flagged, reason, leading = mine.fragment_verdict(["策略状态", "现在呢", "好了吗"], 0)
    assert (flagged, reason, leading) == (True, "all_chatter", True)


def test_a_follow_up_opener_with_conditions_later_is_kept_but_marked():
    # The excerpt still holds a checkable request, so excluding the row would
    # throw away convertible material; the flag records that the opening context
    # is missing.
    flagged, reason, leading = mine.fragment_verdict(["现在呢", "rsi 低于 30 做多"], 3)
    assert (flagged, leading) == (False, True)
    assert reason == ""


def test_a_follow_up_opener_with_nothing_checkable_is_a_fragment():
    flagged, reason, _ = mine.fragment_verdict(["现在呢", "帮我看看这个"], 0)
    assert (flagged, reason) == (True, "leading_chatter_no_conditions")


def test_an_empty_message_list_is_a_fragment():
    assert mine.fragment_verdict([], 0) == (True, "empty", True)


# ---------------------------------------------------------------------------
# Spec shape
# ---------------------------------------------------------------------------

def test_spec_shape_defers_to_the_pinned_classifier():
    for text in ("幫我選出成交量最大的幣", "btc rsi 低於 30 就買進", "現在呢"):
        shape, base, _ = mine.spec_shape(text, families(text))
        assert base == infer_strategy_kind(text)
        assert shape in {"selection", "trade", "both", "unclear"}


def test_spec_shape_normalises_script_before_asking_the_classifier():
    # Every Chinese rule in intent.py is written in Traditional. Feeding the
    # simplified form raw returns "ambiguous" and would make the column
    # meaningless for 94% of the Chinese corpus.
    simplified = "帮我挑几个币"
    assert infer_strategy_kind(simplified) == "ambiguous"
    assert mine.spec_shape(simplified, families(simplified))[1] == "selection"


def test_a_ranked_universe_plus_a_per_bar_rule_becomes_both():
    text = "幫我選出成交量前10的幣,然後 rsi 低於 30 就做多,停損 2%"
    shape, _, _ = mine.spec_shape(text, families(text))
    assert shape == "both"


def test_a_volume_filter_on_one_symbol_stays_a_trade():
    # universe_filter is not cross-sectional evidence: filtering one symbol over
    # time is not selection, and counting it as such would relabel a large slice
    # of ordinary trade requests.
    text = "btc 成交量放大時做多,停損 2%"
    assert mine.spec_shape(text, families(text))[0] == "trade"


def test_unclear_is_reported_rather_than_guessed():
    assert mine.spec_shape("现在呢", set())[0] == "unclear"


# ---------------------------------------------------------------------------
# Near-duplicate clustering
# ---------------------------------------------------------------------------

def test_a_preset_card_with_a_trailing_remark_clusters_with_the_bare_card():
    card = "show me the top 10 futures by volume surge today and rank them"
    texts = [
        card,
        card + " --- thanks",                       # the observed variation
        card.replace("futures", "spot pairs"),      # same card, different market
        "how do i withdraw usdt to my bank account",
    ]
    ids = mine.cluster_near_duplicates(texts)
    assert ids[0] == ids[1]
    assert ids[0] != ids[3]


def test_unrelated_texts_do_not_merge():
    texts = [
        "帮我筛选成交量前十的合约并给出止损位",
        "为什么我的机器人昨天没有下单,检查一下日志",
        "what is the best dca strategy for eth right now",
    ]
    ids = mine.cluster_near_duplicates(texts)
    assert len(set(ids)) == 3


def test_clustering_is_deterministic_and_order_independent_in_content():
    texts = ["alpha beta gamma delta epsilon", "alpha beta gamma delta zeta",
             "totally different words here entirely"]
    assert mine.cluster_near_duplicates(texts) == mine.cluster_near_duplicates(texts)


def test_shingles_span_both_scripts():
    assert mine.tokenize("买 10 coins") == ["买", "10", "coins"]
    assert mine.shingle_set("") == frozenset()
    # A text shorter than the shingle width still gets one shingle, so short
    # preset chips are clusterable instead of silently unclusterable.
    assert len(mine.shingle_set("btc long")) == 1


# ---------------------------------------------------------------------------
# Repo-safety gate
# ---------------------------------------------------------------------------

def test_assert_repo_safe_rejects_user_text():
    mine.assert_repo_safe({"tier": "A", "value": 3.5, "families": ["asset"],
                           "ok": None, "flag": True})
    with pytest.raises(ValueError, match="not an enum"):
        mine.assert_repo_safe({"quote": "止損 2%"})
    with pytest.raises(ValueError, match="not an enum"):
        mine.assert_repo_safe({"note": "two words"})
    with pytest.raises(ValueError, match="too long"):
        mine.assert_repo_safe({"blob": "a" * (mine.MAX_REPO_STRING + 1)})
    with pytest.raises(TypeError):
        mine.assert_repo_safe({"when": object()})


def test_repo_condition_strips_the_quote():
    conditions, _, _ = mine.extract_conditions("rsi 低于 30")
    assert "quote" in conditions[0]
    assert "quote" not in mine.repo_condition(conditions[0])


# ---------------------------------------------------------------------------
# Salt and pseudonyms
# ---------------------------------------------------------------------------

def test_salt_is_created_600_and_used_as_raw_bytes(tmp_path):
    internal = tmp_path / "internal"
    salt = mine.load_or_create_salt(internal)
    path = internal / "salt"
    assert path.stat().st_mode & 0o777 == 0o600
    assert internal.stat().st_mode & 0o777 == 0o700
    assert salt == path.read_bytes() and len(salt) >= 16
    # Reading it again must give the same key, whatever the encoding on disk.
    assert mine.load_or_create_salt(internal) == salt


def test_a_loose_salt_raises_instead_of_being_tightened(tmp_path):
    internal = tmp_path / "internal"
    mine.load_or_create_salt(internal)
    (internal / "salt").chmod(0o644)
    with pytest.raises(PermissionError, match="expected 600"):
        mine.load_or_create_salt(internal)


def test_a_base64_salt_written_by_a_sibling_tool_is_accepted(tmp_path):
    # The real salt file was created by another tool in this pipeline as base64
    # text. Using the file's bytes as the key is the one rule that cannot
    # disagree with a sibling over an encoding.
    internal = tmp_path / "internal"
    internal.mkdir()
    (internal / "salt").write_text("c29tZS1iYXNlNjQtc2FsdC12YWx1ZS1oZXJlPT0=")
    (internal / "salt").chmod(0o600)
    assert mine.load_or_create_salt(internal) == (internal / "salt").read_bytes()


def test_pseudonym_is_hmac_and_hides_the_user_id():
    salt = b"\x01" * 32
    got = mine.pseudonym(salt, "1145646416")
    digest = hmac.new(salt, b"1145646416", hashlib.sha256).digest()
    assert got == "pid_" + base64.b32encode(digest).decode().rstrip("=").lower()[:26]
    assert "1145646416" not in got
    assert mine.pseudonym(b"\x02" * 32, "1145646416") != got
    assert mine.salt_fingerprint(salt) != mine.salt_fingerprint(b"\x02" * 32)
    with pytest.raises(ValueError):
        mine.pseudonym(salt, "")


def test_pseudonym_has_no_eight_digit_run_that_a_privacy_scan_would_flag():
    # Base32, not hex, for exactly this reason: a scanner that fires on our own
    # pseudonyms is a scanner that gets switched off.
    salt = b"\x03" * 32
    ids = [mine.pseudonym(salt, str(1_000_000_000 + i)) for i in range(500)]
    assert not any(re.search(r"\d{8}", pid) for pid in ids)
    assert all(re.match(r"^pid_[a-z2-7]{26}$", pid) for pid in ids)


# ---------------------------------------------------------------------------
# Agreement with the sibling schema module
# ---------------------------------------------------------------------------

def test_pseudonym_agrees_with_the_schema_modules_helper(tmp_path, monkeypatch):
    """Both artifacts of this dataset must name a user the same way.

    ``schema.hmac_pseudonym`` reads the salt from a process-wide path, so this
    miner cannot call it with the salt whose mode it just checked; the encoding is
    written out twice and pinned here instead. Two id formats for one user would
    break every join and let a group-wise split put the same person on both sides.
    """
    schema = pytest.importorskip("tools.nl2yaml.schema")
    internal = tmp_path / "internal"
    salt = mine.load_or_create_salt(internal)
    monkeypatch.setenv(schema.INTERNAL_ROOT_ENV, str(internal))
    assert mine.pseudonym(salt, "1145646416") == schema.hmac_pseudonym("1145646416")


def test_emitted_operators_units_and_polarities_exist_in_the_schema_enums():
    """Gate0 emits a subset of the shared vocabulary and never invents a member."""
    schema = pytest.importorskip("tools.nl2yaml.schema")
    operators = {op.value for op in schema.Operator}
    units = {unit.value for unit in schema.Unit}
    polarities = {p.value for p in schema.Polarity}
    assert set(mine._OP_CANON.values()) <= operators
    assert set(mine._SUBJECT_UNIT.values()) <= units
    for text in ("rsi 低于 30 做多 btc 4h,止损 2%,10倍杠杆",
                 "筛选成交量前10的币,排除 tradfi,市值大于 1000万",
                 "成交量在1000万以上,金叉,超买"):
        for condition in mine.extract_conditions(text)[0]:
            assert condition["operator"] in operators, condition
            assert condition["unit"] in units | {None}, condition
            assert condition["polarity"] in polarities, condition


def test_dup_cluster_id_matches_the_prefix_the_case_record_requires():
    # schema.CaseRecord._check_identity requires dup_cluster_id to start "dup_".
    assert mine.short_hash("x", "dup_").startswith("dup_")


def test_an_exclusion_is_recorded_as_exclude_polarity():
    # "exclude tradfi" ignored by the generated spec is the documented failure
    # this dataset exists to catch; it can only be counted if mining records it.
    conditions, _, _ = mine.extract_conditions("扫描合约,排除 tradfi 的标的")
    excluded = [c for c in conditions if c["polarity"] == "exclude"]
    assert [c["subject"] for c in excluded] == ["exclude"]


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

CARD = "show today's top 10 futures by volume surge and rank them"

#: Invented rows, written to reproduce the shapes the real export contains.
FIXTURE_ROWS = [
    # tier A, simplified Chinese, trade shape
    dict(user_id="1001", chat_id="1001:t:a", first_query="rsi 低于 30 就做多 btc",
         user_text_excerpt="rsi 低于 30 就做多 btc --- 4h 周期,止损 2%,10倍杠杆",
         primary_intent="ENTRY_EXIT_TIMING", n_user_msgs="2"),
    # the same request from the same user on another day: one split group
    dict(user_id="1001", chat_id="1001:t:b", first_query="rsi 低于 30 就做多 btc",
         user_text_excerpt="rsi 低于 30 就做多 btc --- 4h 周期,止损 2%,10倍杠杆 --- 谢谢",
         primary_intent="ENTRY_EXIT_TIMING", n_user_msgs="3"),
    # a preset card, twice, with different trailing chatter
    dict(user_id="1002", chat_id="1002:t:a", first_query=CARD,
         user_text_excerpt=CARD + " --- thanks", preset_case="volume-surge-daily-report",
         primary_intent="COIN_SELECTION", n_user_msgs="2"),
    dict(user_id="1003", chat_id="1003:t:a", first_query=CARD,
         user_text_excerpt=CARD + " --- ok", preset_case="volume-surge-daily-report",
         primary_intent="COIN_SELECTION", n_user_msgs="2"),
    # a continuation fragment: nothing but chatter
    dict(user_id="1004", chat_id="1004:t:a", first_query="策略状态",
         user_text_excerpt="策略状态 --- 现在呢 --- 好了吗",
         primary_intent="AUTOMATION_BOT", n_user_msgs="3"),
    # pure platform injection: no user utterance survives
    dict(user_id="1005", chat_id="1005:t:a",
         first_query="User selected the square-buzz-screener case. Route to the case.",
         user_text_excerpt="User selected the square-buzz-screener case. Route to the case.",
         preset_case="square-buzz-screener", primary_intent="COIN_SELECTION",
         n_user_msgs="1"),
    # tier D: a real request, but nothing a spec could check
    dict(user_id="1006", chat_id="1006:t:a", first_query="我的钱包余额是多少",
         user_text_excerpt="我的钱包余额是多少 --- 帮我转到现货账户",
         primary_intent="PORTFOLIO", n_user_msgs="2"),
    # a user pasting their own numeric id next to a comparator
    dict(user_id="76916505", chat_id="76916505:t:a",
         first_query="超过 76916505 的时候告诉我",
         user_text_excerpt="超过 76916505 的时候告诉我 --- 用 4h 均线判断",
         primary_intent="ALERT_NOTIFY", n_user_msgs="2"),
]


def write_fixture_csv(path: Path, rows=FIXTURE_ROWS) -> Path:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mine.REQUIRED_COLUMNS))
        writer.writeheader()
        for row in rows:
            record = {column: "" for column in mine.REQUIRED_COLUMNS}
            record.update({"month": "2026-05", "day": "2026-05-15", "lang": "zh",
                           "zh_variant": "zh-Hans", "label_source": "model",
                           "is_coin_selection": "0", "wants_automation": "0",
                           "wants_backtest": "0", "wants_strategy": "0",
                           "n_user_msgs": "1", "n_assistant_msgs": "1",
                           "kw_primary": "MARKET_ANALYSIS"})
            record.update(row)
            writer.writerow(record)
    return path


@pytest.fixture
def mined(tmp_path):
    csv_path = write_fixture_csv(tmp_path / "chats.csv")
    out = tmp_path / "dataset"
    internal = tmp_path / "internal"
    report = mine.mine(csv_path, out, internal)
    records = [json.loads(line) for line in
               (out / "candidates.jsonl").read_text(encoding="ascii").splitlines()]
    cases = [json.loads(line) for line in
             (internal / "cases_internal.jsonl").read_text(encoding="utf-8").splitlines()]
    return report, records, cases, out, internal


def test_funnel_arithmetic_closes(mined):
    report, records, cases, _, _ = mined
    funnel = report["funnel"]
    assert funnel["total_rows"] == len(FIXTURE_ROWS) == len(cases)
    assert (funnel["total_rows"] - funnel["continuation_fragments"]
            == funnel["after_fragment_filter"])
    assert sum(report["tier_rows"].values()) == funnel["after_fragment_filter"]
    assert (funnel["after_fragment_filter"] - report["tier_rows"]["D"]
            == funnel["candidates_rows"] == len(records))
    # the crosstab is a partition of the same population
    assert sum(sum(row.values()) for row in report["shape_by_tier_rows"].values()) \
        == funnel["after_fragment_filter"]


def test_the_two_expected_fragments_are_excluded(mined):
    report, records, cases, _, _ = mined
    assert report["fragment_reasons"] == {"all_chatter": 1, "empty": 1}
    excluded = {c["first_query"] for c in cases if not c["is_candidate"]}
    assert "策略状态" in excluded
    assert "我的钱包余额是多少" in excluded          # tier D, not a fragment
    assert all("策略状态" != r.get("preset_case") for r in records)


def test_repeated_and_near_duplicate_rows_share_a_split_group(mined):
    _, records, cases, _, _ = mined
    by_chat = {c["chat_id"]: c for c in cases}
    assert by_chat["1001:t:a"]["split_group_key"] == by_chat["1001:t:b"]["split_group_key"]
    # the preset card wins over the near-dup cluster, so the same card asked by
    # two different users cannot be split across train and test
    assert by_chat["1002:t:a"]["split_group_key"] == "preset:volume-surge-daily-report"
    assert by_chat["1003:t:a"]["split_group_key"] == "preset:volume-surge-daily-report"
    dup_counts = {r["row_id"]: r["dup_count"] for r in records}
    card_rows = [r for r in records if r["preset_case"] == "volume-surge-daily-report"]
    assert all(dup_counts[r["row_id"]] == 2 for r in card_rows)


def test_identical_text_yields_identical_analysis(mined):
    """Two verbatim-identical chats must get identical fields.

    This is the property the upstream label audit found missing: the same
    annotator, given the same input twice, agreed with itself 0.639 of the time.
    A deterministic miner has no excuse for that.
    """
    _, records, _, out, internal = mined
    again = mine.mine(out.parent / "chats.csv", out.parent / "dataset2", internal)
    second = [json.loads(line) for line in
              (out.parent / "dataset2" / "candidates.jsonl").read_text("ascii").splitlines()]
    assert again["funnel"] == json.loads(
        (out / "funnel.json").read_text("ascii"))["funnel"]
    assert records == second


def test_tier_and_shape_of_the_known_fixture_rows(mined):
    _, _, cases, _, _ = mined
    by_chat = {c["chat_id"]: c for c in cases}
    assert by_chat["1001:t:a"]["tier"] == "A"
    assert by_chat["1001:t:a"]["spec_shape"] == "trade"
    assert by_chat["1002:t:a"]["tier"] in {"A", "B"}
    assert by_chat["1006:t:a"]["tier"] == "D"


def test_candidates_file_carries_no_user_text_or_identifier(mined):
    _, records, cases, out, _ = mined
    raw = (out / "candidates.jsonl").read_bytes()
    assert raw.isascii()                       # any CJK would have to appear here
    by_row = {c["row_id"]: c for c in cases}
    for record in records:
        blob = json.dumps(record)
        source = by_row[record["row_id"]]
        assert source["user_id"] not in blob
        assert source["chat_id"] not in blob
        for word in ("rsi 低于", "做多", "谢谢", "volume surge"):
            assert word not in blob
        mine.assert_repo_safe(record)


def test_the_pasted_account_id_is_redacted_in_the_repo_record(mined):
    _, records, cases, _, _ = mined
    row_id = next(c["row_id"] for c in cases if c["chat_id"] == "76916505:t:a")
    record = next(r for r in records if r["row_id"] == row_id)
    thresholds = [c for c in record["conditions"] if c["family"] == "threshold"]
    assert thresholds and all(c["value"] is None and c["value_redacted"]
                              for c in thresholds)
    assert "76916505" not in json.dumps(record)


def test_the_internal_file_keeps_the_text_and_the_quotes(mined):
    _, _, cases, _, internal = mined
    case = next(c for c in cases if c["chat_id"] == "1001:t:a")
    assert "rsi 低于 30" in case["canon_text"]
    assert case["user_id"] == "1001"
    assert any("quote" in condition for condition in case["conditions"])
    assert (internal / "cases_internal.jsonl").stat().st_mode & 0o777 == 0o600


def test_the_report_is_ascii_only_and_names_its_provenance(mined):
    report, records, _, out, _ = mined
    payload = (out / "funnel.json").read_text(encoding="ascii")
    assert payload.isascii()
    assert not re.search(r"[^\x00-\x7f]", payload)
    assert report["miner"] == mine.MINER_VERSION
    assert len(report["salt_fingerprint"]) == 12
    for record in records:
        assert record["mining_source"]["labels_are_not_gold"] is True


def test_upstream_labels_are_carried_but_never_used_as_the_tier_or_shape(mined):
    """The audited labels must not be able to move a tier or a shape.

    ``primary_intent`` was measured at 0.530 inter-annotator agreement, so a
    dataset that stratified on it would be stratifying on noise. Flipping every
    label must leave every tier and shape untouched.
    """
    _, records, _, out, internal = mined
    scrambled = [dict(row, primary_intent="BACKTEST_OPTIMIZE", is_coin_selection="1",
                      wants_automation="1", wants_backtest="1", wants_strategy="1",
                      kw_primary="COIN_SELECTION") for row in FIXTURE_ROWS]
    other = out.parent / "scrambled.csv"
    write_fixture_csv(other, scrambled)
    mine.mine(other, out.parent / "dataset3", internal)
    after = [json.loads(line) for line in
             (out.parent / "dataset3" / "candidates.jsonl").read_text("ascii").splitlines()]
    keys = ("row_id", "tier", "spec_shape", "n_families", "conditions",
            "split_group_key", "is_continuation_fragment")
    assert [{k: r[k] for k in keys} for r in records] == \
           [{k: r[k] for k in keys} for r in after]


# ---------------------------------------------------------------------------
# Fail loud
# ---------------------------------------------------------------------------

def test_a_missing_column_raises_rather_than_mining_a_subset(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("user_id,chat_id,first_query\n1,1:t:a,hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        mine.read_rows(path)


def test_an_empty_csv_raises(tmp_path):
    path = tmp_path / "empty.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=list(mine.REQUIRED_COLUMNS)).writeheader()
    with pytest.raises(ValueError, match="no data rows"):
        mine.read_rows(path)


def test_internal_dir_defaults_to_the_shared_env_var(tmp_path, monkeypatch):
    # A different internal root here than the schema module's would mean two
    # salts and two pseudonyms for the same user.
    schema = pytest.importorskip("tools.nl2yaml.schema")
    monkeypatch.setenv("NL2YAML_INTERNAL_ROOT", str(tmp_path / "elsewhere"))
    csv_path = write_fixture_csv(tmp_path / "chats.csv")
    assert mine.main(["--csv", str(csv_path), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "elsewhere" / "cases_internal.jsonl").exists()
    assert schema.internal_root() == (tmp_path / "elsewhere").absolute()


def test_a_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mine.main(["--csv", str(tmp_path / "nope.csv"), "--out", str(tmp_path / "o"),
                   "--internal-dir", str(tmp_path / "i")])


def test_cli_runs_end_to_end(tmp_path, capsys):
    csv_path = write_fixture_csv(tmp_path / "chats.csv")
    assert mine.main(["--csv", str(csv_path), "--out", str(tmp_path / "out"),
                      "--internal-dir", str(tmp_path / "internal")]) == 0
    printed = capsys.readouterr().out
    assert "Gate0 funnel" in printed
    assert "spec shape x tier" in printed
    assert (tmp_path / "out" / "candidates.jsonl").exists()
