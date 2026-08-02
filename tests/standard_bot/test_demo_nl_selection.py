"""Natural language coin-selection must reach the executable selection path.

These tests deliberately stop short of calling an external LLM or Binance
Square.  The model response and the already-normalised ``cyqnt.input/v1``
bundle are fixed at those two boundaries, while YAML validation, Blocks
execution and the v2 output contract remain real.  That is the smallest honest
test of the user's flow:

    Chinese request -> LLM YAML -> selection classification -> unified bundle
      -> Blocks -> cyqnt.signal/v2(kind=selection)
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).parents[2]
SERVER_PATH = ROOT / "docs" / "strategy_yaml_spec" / "demo" / "server.py"
INDEX_PATH = ROOT / "docs" / "strategy_yaml_spec" / "demo" / "index.html"
SELECTION_YAML = ROOT / "docs" / "strategy_yaml_spec" / "example_selection.yaml"
TRADE_YAML = ROOT / "docs" / "strategy_yaml_spec" / "example_single_ma.yaml"
CHAT_SELECTION_YAML = (
    ROOT / "docs" / "strategy_yaml_spec" / "example_from_user_chat.yaml"
)
CANDIDATE_TRADE_YAML = (
    ROOT / "docs" / "strategy_yaml_spec" / "example_selection_candidate_trade.yaml"
)
SAMPLE_BUNDLE = (
    ROOT / "docs" / "standard_bot_io" / "samples" / "input_bundle_example.json"
)
UNIVERSE_DERIVATIVES_BUNDLE = (
    ROOT / "tests" / "standard_bot" / "fixtures" / "universe_derivatives.json"
)

ENGLISH_DISCOVERY_REQUEST = (
    "I want some hot coins and people havent find it but it has mentioned by some news"
)
CHINESE_DISCOVERY_REQUEST = "幫我選一些比較少見但是有熱度的幣別 最好是可以漲的"

TERRA_BULLISH_SELECTION_YAML = """\
spec_version: "1.0"
target: standard_bot
strategy:
  id: bullish_news_buzz_proxy_selector
  description: "以 Square 熱度與偏多情緒作候選代理，不保證上漲"
run:
  mode: backtest
data:
  symbol: BTCUSDT
  market_type: futures
  primary: { interval: "1h" }
selection:
  universe:
    - block: universe.filter_quote_volume
      params: { min_quote_volume: 100000000 }
    - block: universe.augment_with_news
      with: [ticker_rank]
    - block: universe.filter_sentiment
      params: { min_bull_ratio: 0.55 }
  score: news_mention_count
  top_k: 5
  min_score: 1.0
  dedupe_by: base_asset
"""


@pytest.fixture(scope="module")
def demo_server():
    """Load the demo as a module without starting its HTTP server."""
    spec = importlib.util.spec_from_file_location("standard_bot_demo_server", SERVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prompt_example(demo_server, intent):
    """The example YAML half of a selection prompt, without its instructions.

    Direction assertions belong here and nowhere else. The instruction half names
    both ``order: asc`` and ``order: desc`` while explaining what the key does, so
    a whole-prompt substring check cannot tell the two variants apart — it passes
    even when the prompt ships the opposite example.
    """
    prompt = demo_server.build_system_prompt("selection", intent)
    _instructions, marker, example = prompt.partition("=== 範例輸出 ===")
    assert marker and example.strip(), prompt
    return example


def _prompt_yaml(prompt: str) -> dict:
    _instructions, marker, example = prompt.partition("=== 範例輸出 ===")
    assert marker and example.strip(), prompt
    parsed = yaml.safe_load(example)
    assert isinstance(parsed, dict), example
    return parsed


def test_prompt_teaches_the_model_the_news_selection_dialect(demo_server):
    prompt = demo_server.build_system_prompt("selection")

    # A prompt that only contains the trade example can never reliably turn
    # "pick coins" into the different top-level ``selection:`` grammar.
    for required in (
        "selection:",
        "universe.augment_with_news",
        "ticker_rank",
        "news_mention_count",
        "top_k",
        "kind=selection",
    ):
        assert required in prompt, required


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("幫我挑選最近新聞常提到的五個幣種", "selection"),
        (ENGLISH_DISCOVERY_REQUEST, "selection"),
        (CHINESE_DISCOVERY_REQUEST, "selection"),
        ("找出 Binance Square 最近最熱門的幣", "selection"),
        ("依照社群情緒排行，選出三個候選幣", "selection"),
        ("Select the top 5 coins by Square mentions", "selection"),
        ("我選 BTC，EMA 黃金交叉時買進", "trade"),
        ("Choose BTC and buy when RSI is below 30", "trade"),
        ("BTC 的 EMA 黃金交叉時買進", "trade"),
        ("ETH 跌破前低時做空", "trade"),
        ("新聞提到 BTC 時就買進", "trade"),
        ("選 EMA 12 還是 26", "ambiguous"),
    ],
)
def test_natural_language_intent_routes_to_the_right_strategy_kind(
    demo_server, phrase, expected,
):
    assert demo_server.infer_strategy_kind(phrase) == expected


def test_ambiguous_intent_fails_closed_without_calling_the_model(
    demo_server, monkeypatch,
):
    def should_not_run(*_, **__):
        raise AssertionError("ambiguous input must not be defaulted to trade")

    monkeypatch.setattr(demo_server, "call_llm", should_not_run)
    result = demo_server.convert_nl(
        "http://llm/v1", "", "model", "幫我做一個厲害的東西"
    )

    assert result["strategy_kind"] == "ambiguous"
    assert result["valid"] is False
    assert result["yaml"] == ""
    assert result["errors"]


@pytest.mark.parametrize(
    "phrase",
    [
        "找一些熱門幣，然後買進",
        "Select some hot coins and buy them",
        "Pick 5 coins and buy when EMA crosses",
        "選成交量最大的五個幣後幫每個下單",
    ],
)
def test_compound_selection_and_execution_fails_closed(
    demo_server, monkeypatch, phrase,
):
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("compound intent must stop before LLM generation")
        ),
    )

    result = demo_server.convert_nl("http://llm/v1", "", "model", phrase)

    assert result["status"] == "needs_clarification"
    assert result["strategy_kind"] == "ambiguous"
    assert result["yaml"] == ""
    assert any("請補 RSI 門檻/技術買賣點" in str(error)
               for error in result["errors"])


@pytest.mark.parametrize(
    ("phrase", "directions", "thresholds", "entry_sides"),
    [
        (
            "選成交量最大的五個幣，RSI14 低於30做多，高於70做空",
            ("long", "short"),
            (("below", 30.0), ("above", 70.0)),
            {"long_when", "short_when"},
        ),
        (
            "選成交量最大的五個幣，RSI14低於30做多",
            ("long",),
            (("below", 30.0),),
            {"long_when"},
        ),
        (
            "選成交量最大的五個幣，還要 RSI14 低於30買點，高於70賣點，"
            "停損2%，停利4%，倉位5%",
            (),
            (("below", 30.0), ("above", 70.0)),
            {"long_when", "short_when"},
        ),
        (
            "選成交量最大的五個幣後幫每個下單，RSI14低於30買，高於70賣",
            (),
            (("below", 30.0), ("above", 70.0)),
            {"long_when", "short_when"},
        ),
    ],
)
def test_rsi_candidate_trade_clauses_route_to_selection_without_losing_context(
    demo_server, phrase, directions, thresholds, entry_sides,
):
    intent = demo_server.classify_request(phrase)

    assert intent.kind == "selection"
    assert intent.candidate_trade_requested is True
    assert intent.requested_count == 5
    assert intent.sources == frozenset({"liquidity"})
    assert intent.directions == directions
    assert intent.rsi_thresholds == thresholds

    generated = _prompt_yaml(demo_server.build_system_prompt("selection", intent))
    present = {
        key for key in ("long_when", "short_when")
        if key in generated["selection"]
    }
    assert present == entry_sides
    if "long_when" in present:
        assert generated["selection"]["long_when"]["args"] == ["rsi14", 30]
    if "short_when" in present:
        assert generated["selection"]["short_when"]["args"] == ["rsi14", 70]


@pytest.mark.parametrize(
    ("point_text", "sides", "condition_keys", "expected_cids"),
    [
        (
            "技術分析買賣點",
            ("long", "short"),
            {"long_when", "short_when"},
            {"default_candidate_indicator", "candidate_sell_point_as_short_entry"},
        ),
        (
            "技術分析買點",
            ("long",),
            {"long_when"},
            {"default_candidate_indicator"},
        ),
        (
            "技術分析賣點",
            ("short",),
            {"short_when"},
            {"default_candidate_indicator", "candidate_sell_point_as_short_entry"},
        ),
    ],
)
def test_generic_candidate_trade_points_use_declared_rsi_defaults_and_correct_side(
    demo_server, point_text, sides, condition_keys, expected_cids,
):
    request = "選成交量最大的五個幣，還要" + point_text
    intent = demo_server.classify_request(request)

    assert intent.kind == "selection"
    assert intent.candidate_trade_requested is True
    plan = demo_server.candidate_trade_plan(intent)
    assert plan["sides"] == sides

    generated = _prompt_yaml(demo_server.build_system_prompt("selection", intent))
    present = {
        key for key in ("long_when", "short_when")
        if key in generated["selection"]
    }
    assert present == condition_keys
    cids = {item["cid"] for item in generated["strategy"]["assumptions"]}
    assert expected_cids <= cids
    if "candidate_sell_point_as_short_entry" in expected_cids:
        sell = next(item for item in generated["strategy"]["assumptions"]
                    if item["cid"] == "candidate_sell_point_as_short_entry")
        assert "intent=open_short" in sell["reading"]
        assert "not close_long" in sell["reading"]


@pytest.mark.parametrize(
    "phrase",
    [
        "現貨選成交量最大的五個幣，還要技術分析買賣點",
        "選成交量最大的五個幣，不要 RSI，但要技術分析買賣點",
        "選成交量最大的五個幣，還要 MACD 技術分析買賣點",
    ],
)
def test_candidate_trade_points_that_cannot_be_represented_need_clarification(
    demo_server, monkeypatch, phrase,
):
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("an unsupported candidate trade meaning must stop first")
        ),
    )

    result = demo_server.convert_nl("http://llm/v1", "", "model", phrase)

    assert result["status"] == "needs_clarification"
    assert result["strategy_kind"] == "ambiguous"
    assert result["yaml"] == ""


def test_negated_candidate_trade_points_remain_a_plain_selection(demo_server):
    intent = demo_server.classify_request(
        "選成交量最大的五個幣，不要 RSI 也不要技術分析買賣點"
    )

    assert intent.kind == "selection"
    assert intent.candidate_trade_requested is False
    assert "rsi" in intent.excluded_indicator_names


def _declare_candidate_defaults(demo_server, request: str, spec: dict) -> dict:
    plan = demo_server.candidate_trade_plan(demo_server.classify_request(request))
    assumptions = [copy.deepcopy(item) for item in plan["assumptions"]]
    if assumptions:
        spec["strategy"]["assumptions"] = assumptions
    else:
        spec["strategy"].pop("assumptions", None)
    return spec


def _fixture_candidate_trade_yaml(demo_server, request: str) -> dict:
    """Make the shipped hybrid example match the frozen bars capture roster.

    The fixture deliberately contains per-candidate bars for exactly five
    instruments.  Naming those same five instruments in the request/spec keeps
    this golden path honest: no missing bars are silently treated as a market
    fact and no unrequested allowlist is smuggled into the generated YAML.
    """
    spec = yaml.safe_load(CANDIDATE_TRADE_YAML.read_text(encoding="utf-8"))
    spec["selection"]["universe"][0] = {
        "block": "universe.only_symbols",
        "params": {
            "symbols": [
                "ALLOUSDT", "TAGUSDT", "UAIUSDT", "UBUSDT", "USUSDT",
            ],
        },
    }
    return _declare_candidate_defaults(demo_server, request, spec)


@pytest.mark.parametrize(
    ("user_request", "expected_cids"),
    [
        (
            "從 ALLO、TAG、UAI、UB、US 中選成交量最大的五個幣，"
            "還要技術分析買賣點",
            {
                "default_candidate_indicator",
                "candidate_sell_point_as_short_entry",
                "default_candidate_rsi_long",
                "default_candidate_rsi_short",
                "default_candidate_size",
                "default_candidate_stop",
                "default_candidate_take_profit",
            },
        ),
        (
            "從 ALLO、TAG、UAI、UB、US 中挑選成交量最大的五個幣後幫每個下單，"
            "RSI14低於30買，高於70賣",
            {
                "candidate_sell_point_as_short_entry",
                "default_candidate_size",
                "default_candidate_stop",
                "default_candidate_take_profit",
            },
        ),
    ],
    ids=["declared_trade_points", "colloquial_rsi_buy_sell"],
)
def test_selection_with_rsi_trade_points_stays_selection_through_runtime(
    demo_server, monkeypatch, user_request, expected_cids,
):
    """Golden NL -> mocked YAML -> real Blocks -> nested trade contracts."""
    generated = _fixture_candidate_trade_yaml(demo_server, user_request)
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(generated, sort_keys=False),
    )

    converted = demo_server.convert_nl(
        "http://llm/v1", "", "model", user_request,
    )

    assert converted["valid"] is True, converted
    assert converted["strategy_kind"] == "selection"
    assert converted["generated_strategy_kind"] == "selection"
    assert converted["intent"]["candidate_trade_requested"] is True

    # ``run_bundle`` accepts a mapping or a path.  Parse the model text here so
    # this test cannot accidentally reinterpret a YAML document as a filename.
    from cyqnt_trd.standard_bot.yaml_pipeline.bundle_runner import run_bundle

    batch = run_bundle(
        yaml.safe_load(converted["yaml"]), str(UNIVERSE_DERIVATIVES_BUNDLE)
    )
    assert batch["schema"] == "cyqnt.signal-batch/v1"
    assert batch["signal_count"] == 1
    signal = batch["signals"][0]
    assert signal["schema"] == "cyqnt.signal/v2"
    assert signal["kind"] == "selection"
    assert len(signal["candidates"]) == 5
    assert all(candidate["trade"]["schema"] == "cyqnt.signal/v2"
               for candidate in signal["candidates"])
    assert all(candidate["trade"]["kind"] == "trade"
               for candidate in signal["candidates"])
    assert all(candidate["trade"]["auto_trade_eligible"] is False
               for candidate in signal["candidates"])
    assert all(candidate["trade"]["requires_confirmation"] is True
               for candidate in signal["candidates"])
    assert all(any("assumption (%s)" % cid in warning
                   for warning in signal["warnings"])
               for cid in expected_cids)
    assert all(any("assumption (%s)" % cid in warning
                   for warning in signal["candidates"][0]["trade"]["warnings"])
               for cid in expected_cids)


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (
            lambda spec: (
                spec["selection"].pop("candidate_trade"),
                spec.pop("sizing"),
                spec.pop("risk"),
            ),
            "沒有 selection.candidate_trade",
        ),
        (
            lambda spec: (
                spec["selection"].__setitem__(
                    "long_when",
                    {"cond": "conditions.value_below", "args": ["quoteVolume", 30]},
                ),
                spec["selection"].__setitem__(
                    "short_when",
                    {"cond": "conditions.value_above", "args": ["quoteVolume", 70]},
                ),
            ),
            "沒有實際影響 long_when/short_when",
        ),
        (
            lambda spec: spec["selection"].pop("short_when"),
            "沒有 selection.short_when 賣點",
        ),
    ],
    ids=["missing_candidate_trade", "computed_rsi_is_unused", "missing_sell_point"],
)
def test_hybrid_semantic_mismatches_fail_closed(
    demo_server, monkeypatch, mutate, needle,
):
    request = "選成交量最大的五個幣，還要 RSI14 技術分析買賣點"
    generated = yaml.safe_load(CANDIDATE_TRADE_YAML.read_text(encoding="utf-8"))
    _declare_candidate_defaults(demo_server, request, generated)
    mutate(generated)
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(generated, sort_keys=False),
    )

    result = demo_server.convert_nl("http://llm/v1", "", "model", request)

    assert result["strategy_kind"] == "selection"
    assert result["valid"] is False
    assert any(needle in str(error) for error in result["errors"]), result


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (
            lambda spec: spec["strategy"]["assumptions"].pop(),
            "缺少 cid=default_candidate_take_profit",
        ),
        (
            lambda spec: spec["strategy"]["assumptions"][0].__setitem__(
                "reading", "RSI defaults, trust me"
            ),
            "必須精確宣告",
        ),
        (
            lambda spec: spec["sizing"].__setitem__("size", 0.10),
            "sizing.size 必須是 0.05",
        ),
    ],
    ids=["missing_declaration", "mutated_declaration", "value_disagrees"],
)
def test_hybrid_defaults_require_exact_machine_readable_assumptions(
    demo_server, monkeypatch, mutate, needle,
):
    request = "選成交量最大的五個幣，還要 RSI14 技術分析買賣點"
    intent = demo_server.classify_request(request)
    generated = _prompt_yaml(demo_server.build_system_prompt("selection", intent))
    assert len(generated["strategy"]["assumptions"]) == 6
    mutate(generated)
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(generated, allow_unicode=True, sort_keys=False),
    )

    result = demo_server.convert_nl("http://llm/v1", "", "model", request)

    assert result["valid"] is False
    assert any(needle in str(error) for error in result["errors"]), result


def test_contextual_second_rsi_threshold_is_semantically_enforced(
    demo_server, monkeypatch,
):
    request = (
        "選成交量最大的五個幣，還要 RSI14 低於30買點，高於70賣點，"
        "停損2%，停利4%，倉位5%"
    )
    prompt_spec = _prompt_yaml(
        demo_server.build_system_prompt(
            "selection", demo_server.classify_request(request)
        )
    )
    assumptions = prompt_spec["strategy"]["assumptions"]
    assert [item["cid"] for item in assumptions] == [
        "candidate_sell_point_as_short_entry"
    ]
    wrong = yaml.safe_load(CANDIDATE_TRADE_YAML.read_text(encoding="utf-8"))
    wrong["selection"]["short_when"]["args"][1] = 65
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(wrong, sort_keys=False),
    )

    result = demo_server.convert_nl("http://llm/v1", "", "model", request)

    assert result["valid"] is False
    assert any("above 70" in str(error) for error in result["errors"]), result


@pytest.mark.parametrize(
    ("phrase", "source_block", "source_input"),
    [
        (
            "選 open interest 最大的五個幣，還要 RSI14 技術分析買賣點",
            "universe.augment_with_open_interest",
            "open_interest_snapshot",
        ),
        (
            "選 funding rate 最高的五個幣，還要 RSI14 技術分析買賣點",
            "universe.augment_with_funding",
            "funding_info",
        ),
    ],
)
def test_hybrid_prompt_precedes_plain_source_selection_prompt(
    demo_server, phrase, source_block, source_input,
):
    intent = demo_server.classify_request(phrase)
    assert intent.kind == "selection"
    assert intent.candidate_trade_requested is True

    prompt = demo_server.build_system_prompt("selection", intent)

    assert "selection.candidate_trade" in prompt
    assert "sizing: { size:" in prompt
    assert "risk:" in prompt
    assert source_block in prompt
    assert source_input in prompt
    assert "不得產生 signals:、sizing:、risk:" not in prompt


def test_generic_selection_without_a_ranking_criterion_fails_closed(
    demo_server, monkeypatch,
):
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("missing ranking criterion must not reach the LLM")
        ),
    )

    result = demo_server.convert_nl(
        "http://llm/v1", "", "model", "幫我選五個幣"
    )

    assert result["status"] == "needs_clarification"
    assert result["valid"] is False
    assert result["yaml"] == ""


def test_trade_prompt_uses_the_running_macd_block_signature(demo_server):
    from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import resolve_block

    actual = tuple(inspect.signature(resolve_block("indicators.macd")).parameters)
    prompt = demo_server.build_system_prompt("trade")

    assert actual == ("series", "fast", "slow", "signal")
    assert "indicators.macd(series,fast,slow,signal)" in prompt
    assert "indicators.macd(series,fast_period,slow_period,signal_period)" not in prompt


def test_unspecified_trade_direction_is_long_only_in_prompt_and_reconciliation(
    demo_server, monkeypatch,
):
    request = "BTC 1h EMA10 上穿 EMA30"
    intent = demo_server.classify_request(request)
    assert intent.kind == "trade"
    assert intent.directions == ()

    prompt = demo_server.build_system_prompt("trade", intent)
    assert "使用者未指定方向,採 long-only" in prompt
    assert set(_prompt_yaml(prompt)["signals"]["entry"]) == {"long"}

    generated = yaml.safe_load(TRADE_YAML.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(generated, sort_keys=False),
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert rejected["valid"] is False
    assert any("預設 long-only" in str(error) for error in rejected["errors"])

    generated["signals"]["entry"].pop("short")
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(generated, sort_keys=False),
    )
    accepted = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert accepted["valid"] is True, accepted


def test_negated_indicator_is_forbidden_not_required(demo_server, monkeypatch):
    request = "BTC 1h EMA10 上穿 EMA30 買進，不要使用 RSI"
    intent = demo_server.classify_request(request)
    assert intent.indicator_names == ("ema",)
    assert intent.excluded_indicator_names == ("rsi",)
    assert intent.technical_periods == (("ema", 10), ("ema", 30))

    faithful = yaml.safe_load(TRADE_YAML.read_text(encoding="utf-8"))
    faithful["signals"]["entry"].pop("short")
    monkeypatch.setattr(
        demo_server, "call_llm", lambda *_, **__: yaml.safe_dump(faithful, sort_keys=False)
    )
    accepted = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert accepted["valid"] is True, accepted

    forbidden = copy.deepcopy(faithful)
    forbidden["signals"]["indicators"]["rsi14"] = {
        "block": "indicators.rsi", "input": "close", "params": {"period": 14},
    }
    forbidden["signals"]["entry"]["long"] = {
        "all_of": [
            forbidden["signals"]["entry"]["long"],
            {"cond": "conditions.value_below", "args": ["rsi14", 30]},
        ]
    }
    monkeypatch.setattr(
        demo_server, "call_llm", lambda *_, **__: yaml.safe_dump(forbidden, sort_keys=False)
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert rejected["valid"] is False
    assert any("不要使用 RSI" in str(error) for error in rejected["errors"])


def test_negated_long_direction_leaves_short_as_the_only_allowed_side(
    demo_server, monkeypatch,
):
    request = "BTC 1h 不要做多，只允許做空，EMA10 下穿 EMA30"
    intent = demo_server.classify_request(request)
    assert intent.directions == ("short",)
    assert intent.excluded_directions == ("long",)
    prompt = demo_server.build_system_prompt("trade", intent)
    assert "只能給 signals.entry.short" in prompt
    assert "不得產生 signals.entry.long" in prompt
    assert set(_prompt_yaml(prompt)["signals"]["entry"]) == {"short"}

    faithful = yaml.safe_load(TRADE_YAML.read_text(encoding="utf-8"))
    faithful["signals"]["entry"].pop("long")
    monkeypatch.setattr(
        demo_server, "call_llm", lambda *_, **__: yaml.safe_dump(faithful, sort_keys=False)
    )
    accepted = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert accepted["valid"] is True, accepted

    forbidden = yaml.safe_load(TRADE_YAML.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        demo_server, "call_llm", lambda *_, **__: yaml.safe_dump(forbidden, sort_keys=False)
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert rejected["valid"] is False
    assert any("不要做多" in str(error) for error in rejected["errors"])


def test_macd_yaml_taught_by_prompt_passes_real_dry_run_and_old_names_fail(
    demo_server,
):
    spec = yaml.safe_load(demo_server.EXAMPLE_YAML)
    macd = {
        "block": "indicators.macd",
        "input": "close",
        "params": {"fast": 12, "slow": 26, "signal": 9},
    }
    spec["signals"] = {
        "indicators": {
            "macd_line": {**macd, "output": 0},
            "macd_signal": {**macd, "output": 1},
        },
        "entry": {
            "long": {
                "cond": "conditions.macd_golden_cross",
                "args": ["macd_line", "macd_signal"],
            }
        },
    }

    errors, _ = demo_server.validate_spec(spec)
    assert errors == []

    old_names = copy.deepcopy(spec)
    for indicator in old_names["signals"]["indicators"].values():
        indicator["params"] = {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
        }
    errors, _ = demo_server.validate_spec(old_names)
    assert errors
    assert any("fast_period" in str(error) for error in errors)


def test_trade_semantic_gate_checks_symbol_interval_indicators_risk_and_size(
    demo_server, monkeypatch,
):
    request = (
        "BTC 1h，EMA12 上穿 EMA26 買進，停損 2%，停利 4%，size 95%"
    )
    correct = yaml.safe_load(demo_server.EXAMPLE_YAML)
    correct["signals"]["entry"].pop("short")
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(correct, sort_keys=False, allow_unicode=True),
    )
    accepted = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert accepted["valid"] is True, accepted

    wrong = copy.deepcopy(correct)
    wrong["data"]["symbol"] = "ETHUSDT"
    wrong["data"]["primary"]["interval"] = "4h"
    wrong["signals"]["indicators"]["ema_fast"]["params"]["period"] = 5
    wrong["signals"]["indicators"]["ema_slow"]["params"]["period"] = 8
    wrong["signals"]["entry"]["long"] = {
        "cond": "conditions.rsi_oversold",
        "args": ["rsi14"],
        "params": {"threshold": 30},
    }
    wrong["risk"]["exit"]["stop_pct"] = 0.10
    wrong["risk"]["exit"]["tp_pct"] = 0.20
    wrong["sizing"]["size"] = 0.10
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(wrong, sort_keys=False, allow_unicode=True),
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)

    assert rejected["status"] == "rejected"
    assert rejected["valid"] is False
    combined = "\n".join(map(str, rejected["errors"]))
    for expected in (
        "BTC", "1h", "EMA12", "EMA26", "cross_above", "停損", "停利", "倉位",
    ):
        assert expected in combined


def test_bare_sell_action_stops_before_llm_and_cannot_be_folded_into_long_entry(
    demo_server, monkeypatch,
):
    """A down-cross sell must not silently become another open-long trigger."""
    request = (
        "BTCUSDT EMA12 上穿 EMA26 買進，EMA12 下穿 EMA26 賣出，"
        "停損 2%，停利 4%，倉位 5%"
    )
    intent = demo_server.classify_request(request)
    assert intent.kind == "trade"
    assert intent.directions == ("long",)
    assert "bare_sell_direction_ambiguous" in intent.evidence

    wrong_but_structural = yaml.safe_load(demo_server.EXAMPLE_YAML)
    wrong_but_structural["signals"]["entry"].pop("short")
    wrong_but_structural["signals"]["entry"]["long"] = {
        "any_of": [
            {"cond": "conditions.ma_cross_above", "args": ["ema_fast", "ema_slow"]},
            {"cond": "conditions.ma_cross_below", "args": ["ema_fast", "ema_slow"]},
        ],
    }
    validation_errors, _ = demo_server.validate_spec(wrong_but_structural)
    assert validation_errors == []
    reconciliation_errors, _ = demo_server.reconcile_intent(
        intent, wrong_but_structural,
    )
    assert any("close_long" in error and "open_short" in error
               for error in reconciliation_errors)

    def should_not_run(*_, **__):
        raise AssertionError("ambiguous sell action must stop before an LLM call")

    monkeypatch.setattr(demo_server, "call_llm", should_not_run)
    result = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert result["status"] == "needs_clarification"
    assert result["valid"] is False
    assert result["yaml"] == ""
    assert "close_long" in result["errors"][0]
    assert "open_short" in result["errors"][0]


def test_percent_stop_take_profit_cannot_be_hidden_under_time_only(
    demo_server, monkeypatch,
):
    """Both structural and semantic gates bind percentage exits to their engine."""
    request = "BTCUSDT EMA10 上穿 EMA30 買進，停損 2%，停利 4%"
    intent = demo_server.classify_request(request)
    decoy = yaml.safe_load(demo_server.EXAMPLE_YAML)
    decoy["signals"]["entry"].pop("short")
    decoy["signals"]["indicators"]["ema_fast"]["params"]["period"] = 10
    decoy["signals"]["indicators"]["ema_slow"]["params"]["period"] = 30
    decoy["risk"]["exit"] = {
        "type": "time_only", "max_bars": 3,
        "stop_pct": 0.02, "tp_pct": 0.04,
    }

    validation_errors, _ = demo_server.validate_spec(decoy)
    assert any("risk.exit.stop_pct is not used by type=time_only" in error
               for error in validation_errors)
    reconciliation_errors, _ = demo_server.reconcile_intent(intent, decoy)
    assert any("risk.exit.type 必須是 pct_stop_tp" in error
               for error in reconciliation_errors)

    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(decoy, allow_unicode=True, sort_keys=False),
    )
    result = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert result["status"] == "rejected"
    assert result["valid"] is False
    assert any("risk.exit.stop_pct is not used by type=time_only" in error
               for error in result["errors"])


def test_multi_instrument_trade_and_spot_short_stop_before_llm(
    demo_server, monkeypatch,
):
    def should_not_run(*_, **__):
        raise AssertionError("unrepresentable trade intent must stop before an LLM call")

    monkeypatch.setattr(demo_server, "call_llm", should_not_run)
    multiple = demo_server.convert_nl(
        "http://llm/v1", "", "model", "BTC、ETH 做多，SOL 做空，停損 2%",
    )
    assert multiple["status"] == "needs_clarification"
    assert multiple["yaml"] == ""
    assert "一個標的" in multiple["errors"][0]

    spot_short = demo_server.convert_nl(
        "http://llm/v1", "", "model", "BTC 現貨 EMA12 下穿 EMA26 做空",
    )
    assert spot_short["status"] == "needs_clarification"
    assert spot_short["yaml"] == ""
    assert "現貨不能開空" in spot_short["errors"][0]

    invalid_spot = yaml.safe_load(demo_server.EXAMPLE_YAML)
    invalid_spot["data"]["market_type"] = "spot"
    validation_errors, _ = demo_server.validate_spec(invalid_spot)
    assert any("signals.entry.short is not allowed" in error
               for error in validation_errors)


def _ema_rsi_trade_spec(demo_server, join: str) -> dict:
    """A small long-only trade whose two entry predicates are independently clear."""
    spec = yaml.safe_load(demo_server.EXAMPLE_YAML)
    spec["signals"]["entry"].pop("short")
    spec["signals"]["indicators"]["ema_fast"]["params"]["period"] = 10
    spec["signals"]["indicators"]["ema_slow"]["params"]["period"] = 30
    spec["signals"]["indicators"]["rsi14"] = {
        "block": "indicators.rsi", "input": "close", "params": {"period": 14},
    }
    spec["signals"]["entry"]["long"] = {
        join: [
            {"cond": "conditions.ma_cross_above", "args": ["ema_fast", "ema_slow"]},
            {
                "cond": "conditions.rsi_oversold",
                "args": ["rsi14"],
                "params": {"threshold": 30},
            },
        ],
    }
    return spec


def _rsi_only_trade_spec(demo_server, join: str) -> dict:
    """A long-only trade with two explicit RSI predicates and no hidden EMA."""
    spec = yaml.safe_load(demo_server.EXAMPLE_YAML)
    spec["signals"] = {
        "indicators": {
            "rsi14": {
                "block": "indicators.rsi", "input": "close", "params": {"period": 14},
            },
        },
        "entry": {
            "long": {
                join: [
                    {
                        "cond": "conditions.rsi_oversold",
                        "args": ["rsi14"],
                        "params": {"threshold": 30},
                    },
                    {
                        "cond": "conditions.rsi_overbought",
                        "args": ["rsi14"],
                        "params": {"threshold": 70},
                    },
                ],
            },
        },
    }
    return spec


@pytest.mark.parametrize(
    ("connector", "expected_join", "faithful_join", "wrong_join", "error_word"),
    [
        ("且", "all_of", "all_of", "any_of", "AND"),
        ("或", "any_of", "any_of", "all_of", "OR"),
    ],
)
def test_trade_boolean_join_is_proven_against_the_actual_entry_tree(
    demo_server, monkeypatch, connector, expected_join, faithful_join, wrong_join, error_word,
):
    request = "BTC EMA10 上穿 EMA30 %s RSI14 低於30買進" % connector
    intent = demo_server.classify_request(request)
    assert intent.condition_join == expected_join

    faithful = _ema_rsi_trade_spec(demo_server, faithful_join)
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(faithful, allow_unicode=True, sort_keys=False),
    )
    accepted = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert accepted["valid"] is True, accepted

    wrong = _ema_rsi_trade_spec(demo_server, wrong_join)
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(wrong, allow_unicode=True, sort_keys=False),
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert rejected["valid"] is False
    assert any(error_word in str(error) for error in rejected["errors"])

    # An unrelated risk phrase containing 或 cannot silently reclassify the
    # entry itself as an OR request.
    unrelated_or = demo_server.classify_request(
        "BTC EMA10 上穿 EMA30，停損或停利 RSI14 低於30買進",
    )
    assert unrelated_or.condition_join == "all_of"


@pytest.mark.parametrize(
    ("connector", "expected_join"),
    [
        ("或", "any_of"),
        ("且", "all_of"),
        ("以及", "all_of"),
    ],
)
def test_rsi_only_boolean_thresholds_are_complete_and_joined(
    demo_server, monkeypatch, connector, expected_join,
):
    request = "BTC RSI14 低於30%s高於70買進" % connector
    intent = demo_server.classify_request(request)
    assert intent.rsi_thresholds == (("below", 30.0), ("above", 70.0))
    assert intent.condition_join == expected_join

    faithful = _rsi_only_trade_spec(demo_server, expected_join)
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(faithful, allow_unicode=True, sort_keys=False),
    )
    accepted = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert accepted["valid"] is True, accepted

    wrong_join = "all_of" if expected_join == "any_of" else "any_of"
    wrong = _rsi_only_trade_spec(demo_server, wrong_join)
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(wrong, allow_unicode=True, sort_keys=False),
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert rejected["valid"] is False
    assert any("AND" in str(error) or "OR" in str(error) for error in rejected["errors"])


def test_mixed_entry_boolean_expression_stops_before_the_llm(demo_server, monkeypatch):
    request = "BTC EMA10上穿EMA30且RSI14低於30或高於70買進"
    intent = demo_server.classify_request(request)
    assert intent.condition_join == "mixed"

    mixed = _ema_rsi_trade_spec(demo_server, "any_of")
    mixed["signals"]["entry"]["long"]["any_of"].append({
        "cond": "conditions.rsi_overbought",
        "args": ["rsi14"],
        "params": {"threshold": 70},
    })
    errors, _warnings = demo_server.reconcile_intent(intent, mixed)
    assert any("混合了 AND 與 OR" in error for error in errors)

    def should_not_run(*_, **__):
        raise AssertionError("mixed entry logic must stop before an LLM call")

    monkeypatch.setattr(demo_server, "call_llm", should_not_run)
    result = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert result["status"] == "needs_clarification"
    assert result["yaml"] == ""
    assert "混合 AND 與 OR" in result["errors"][0]


def test_vwap_entry_requires_price_to_reference_vwap_and_runs_the_real_block(
    demo_server, monkeypatch,
):
    request = "BTC 價格上穿 VWAP 買進"
    faithful = yaml.safe_load(demo_server.EXAMPLE_YAML)
    faithful["signals"] = {
        "indicators": {
            "vwap": {"block": "indicators.vwap", "input": "df"},
        },
        "entry": {
            "long": {
                "cond": "conditions.price_touch_or_cross",
                "args": ["df", "vwap"],
                "params": {"direction": "up"},
            },
        },
    }
    errors, _ = demo_server.validate_spec(faithful)
    assert errors == []

    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(faithful, allow_unicode=True, sort_keys=False),
    )
    accepted = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert accepted["valid"] is True, accepted

    import pandas as pd

    from cyqnt_trd.blocks import conditions, indicators
    from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import build_make_signals

    frame = pd.DataFrame({
        "open": [10.0, 9.0, 10.0, 12.0],
        "high": [11.0, 10.0, 11.0, 13.0],
        "low": [9.0, 8.0, 9.0, 11.0],
        "close": [10.0, 9.0, 10.5, 12.5],
        "volume": [10.0, 10.0, 10.0, 10.0],
    })
    expected = conditions.price_touch_or_cross(
        frame, indicators.vwap(frame), direction="up",
    )
    actual, _short = build_make_signals(faithful)(frame)
    assert actual.equals(expected)

    ema_proxy = yaml.safe_load(demo_server.EXAMPLE_YAML)
    ema_proxy["signals"]["entry"].pop("short")
    ema_proxy["signals"]["indicators"]["vwap"] = {
        "block": "indicators.vwap", "input": "df",
    }
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(ema_proxy, allow_unicode=True, sort_keys=False),
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert rejected["valid"] is False
    assert any("VWAP" in str(error) for error in rejected["errors"])

    # Above/below is a state, not the requested crossing event.  It used to be
    # accepted solely because it mentioned the right VWAP series.
    state_proxy = copy.deepcopy(faithful)
    state_proxy["signals"]["entry"]["long"] = {
        "cond": "conditions.close_above", "args": ["df", "vwap"],
    }
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(state_proxy, allow_unicode=True, sort_keys=False),
    )
    rejected_state = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert rejected_state["valid"] is False
    assert any("VWAP" in str(error) for error in rejected_state["errors"])


def test_atr_trailing_stop_requires_exact_parameters_and_never_becomes_leverage(
    demo_server, monkeypatch,
):
    request = "BTC EMA10 上穿 EMA30 買進，ATR(14) 2x 追蹤停損"
    intent = demo_server.classify_request(request)
    assert (intent.exit_type, intent.exit_atr_period, intent.exit_trail_mult) == (
        "atr_trailing_stop", 14, 2.0,
    )
    assert "gap:GAP-ACCOUNT-OPS" not in intent.unsupported_preferences
    assert intent.included_symbols == ("BTC",)

    faithful = yaml.safe_load(demo_server.EXAMPLE_YAML)
    faithful["signals"]["entry"].pop("short")
    faithful["signals"]["indicators"]["ema_fast"]["params"]["period"] = 10
    faithful["signals"]["indicators"]["ema_slow"]["params"]["period"] = 30
    faithful["risk"]["exit"] = {
        "type": "atr_trailing_stop",
        "atr_period": 14,
        "trail_mult": 2.0,
        "max_bars": 96,
    }
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(faithful, allow_unicode=True, sort_keys=False),
    )
    accepted = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert accepted["valid"] is True, accepted

    fixed_pct = copy.deepcopy(faithful)
    fixed_pct["risk"]["exit"] = {
        "type": "pct_stop_tp", "stop_pct": 0.02, "tp_pct": 0.04, "max_bars": 96,
    }
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(fixed_pct, allow_unicode=True, sort_keys=False),
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert rejected["valid"] is False
    assert any("atr_trailing_stop" in str(error) for error in rejected["errors"])

    def should_not_run(*_, **__):
        raise AssertionError("partial ATR stop must stop before an LLM call")

    monkeypatch.setattr(demo_server, "call_llm", should_not_run)
    partial = demo_server.convert_nl(
        "http://llm/v1", "", "model",
        "BTC EMA10 上穿 EMA30 買進，使用2x ATR追蹤停損",
    )
    assert partial["status"] == "needs_clarification"
    assert "ATR period" in partial["errors"][0]
    assert demo_server.classify_request(
        "BTC EMA10 上穿 EMA30，追蹤 ATR 指標",
    ).exit_type is None


def test_excluded_atr_trailing_stop_cannot_be_reintroduced_by_the_model(
    demo_server, monkeypatch,
):
    request = "BTC EMA10 上穿 EMA30 買進，不使用 ATR(14) 2x 追蹤停損"
    intent = demo_server.classify_request(request)
    assert (intent.exit_type, intent.exit_atr_period, intent.exit_trail_mult) == (
        None, None, None,
    )
    assert intent.excluded_exit_types == ("atr_trailing_stop",)

    forbidden = yaml.safe_load(demo_server.EXAMPLE_YAML)
    forbidden["signals"]["entry"].pop("short")
    forbidden["signals"]["indicators"]["ema_fast"]["params"]["period"] = 10
    forbidden["signals"]["indicators"]["ema_slow"]["params"]["period"] = 30
    forbidden["risk"]["exit"] = {
        "type": "atr_trailing_stop",
        "atr_period": 14,
        "trail_mult": 2.0,
        "max_bars": 96,
    }
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(forbidden, allow_unicode=True, sort_keys=False),
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert rejected["valid"] is False
    assert any("不要使用 ATR 追蹤停損" in str(error) for error in rejected["errors"])

    replacement = demo_server.classify_request(
        "BTC EMA10 上穿 EMA30 買進，不使用 ATR(14) 2x 追蹤停損，"
        "使用 ATR(20) 3x 追蹤停損",
    )
    assert (
        replacement.exit_type,
        replacement.exit_atr_period,
        replacement.exit_trail_mult,
        replacement.excluded_exit_types,
    ) == ("atr_trailing_stop", 20, 3.0, ())


def test_multiple_positive_atr_trailing_stops_need_clarification_before_llm(
    demo_server, monkeypatch,
):
    request = (
        "BTC EMA10 上穿 EMA30 買進，使用 ATR(14) 2x 追蹤停損與 "
        "ATR(20) 3x 追蹤停損"
    )
    intent = demo_server.classify_request(request)
    assert (intent.exit_type, intent.exit_atr_period, intent.exit_trail_mult) == (
        "atr_trailing_stop", None, None,
    )

    def should_not_run(*_, **__):
        raise AssertionError("ambiguous ATR exits must stop before an LLM call")

    monkeypatch.setattr(demo_server, "call_llm", should_not_run)
    result = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert result["status"] == "needs_clarification"
    assert result["yaml"] == ""
    assert "ATR period" in result["errors"][0]


def test_excluded_source_lists_reject_news_or_funding_in_a_selection(
    demo_server, monkeypatch,
):
    request = "選5個成交量最高，不要使用新聞或資金費率"
    intent = demo_server.classify_request(request)
    assert intent.kind == "selection"
    assert intent.sources == frozenset({"liquidity"})
    assert intent.excluded_sources == frozenset({"news", "funding"})

    faithful = yaml.safe_load(demo_server.LIQUIDITY_SELECTION_EXAMPLE_YAML)
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(faithful, allow_unicode=True, sort_keys=False),
    )
    accepted = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert accepted["valid"] is True, accepted

    forbidden = copy.deepcopy(faithful)
    forbidden["selection"]["universe"].extend([
        {"block": "universe.augment_with_news", "with": ["ticker_rank"]},
        {
            "block": "universe.augment_with_funding",
            "with": ["funding", "funding_info"],
        },
    ])
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(forbidden, allow_unicode=True, sort_keys=False),
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert rejected["valid"] is False
    assert any("不要使用 news" in str(error) for error in rejected["errors"])
    assert any("不要使用 funding" in str(error) for error in rejected["errors"])


@pytest.mark.parametrize("user_request", [
    "選五個成交量最高，不使用新聞或資金費率",
    "選五個成交量最高，不要使用新聞、以及資金費率",
])
def test_source_exclusion_variants_never_become_positive_requirements(
    demo_server, user_request,
):
    intent = demo_server.classify_request(user_request)

    assert intent.sources == frozenset({"liquidity"})
    assert intent.excluded_sources == frozenset({"news", "funding"})


@pytest.mark.parametrize(
    ("user_request", "gap_id"),
    [
        ("選五個市值最大的幣", "GAP-MARKET-CAP"),
        ("選五個鏈上持幣集中度最低的幣", "GAP-ONCHAIN-CONCENTRATION"),
        ("BTC EMA10上穿EMA30，使用5x槓桿進場", "GAP-ACCOUNT-OPS"),
        ("BTC EMA10上穿EMA30，限價單進場", "GAP-ACCOUNT-OPS"),
        ("選五個成交量最高的幣後 Telegram 通知我", "GAP-ALERT-NOTIFY"),
        ("選五個清算最多的幣", "GAP-LIQUIDATION-CROSS-SECTION"),
    ],
)
def test_named_unsupported_requirements_stop_before_the_llm(
    demo_server, monkeypatch, user_request, gap_id,
):
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("named unsupported request must not call an LLM")
        ),
    )
    result = demo_server.convert_nl("http://llm/v1", "", "model", user_request)
    assert result["status"] == "unsupported"
    assert result["valid"] is False
    assert result["yaml"] == ""
    assert "gap:%s" % gap_id in result["intent"]["unsupported_preferences"]
    assert gap_id in result["errors"][0]


def test_trade_indicator_must_drive_the_entry_condition(demo_server, monkeypatch):
    macd_request = "BTC 1h MACD 黃金交叉買進"
    wrong_macd = yaml.safe_load(demo_server.EXAMPLE_YAML)
    wrong_macd["signals"]["entry"].pop("short")
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(wrong_macd, sort_keys=False),
    )
    rejected = demo_server.convert_nl(
        "http://llm/v1", "", "model", macd_request
    )
    assert rejected["valid"] is False
    assert any("MACD" in str(error) for error in rejected["errors"])

    correct_macd = yaml.safe_load(demo_server.EXAMPLE_YAML)
    macd = {
        "block": "indicators.macd",
        "input": "close",
        "params": {"fast": 12, "slow": 26, "signal": 9},
    }
    correct_macd["signals"] = {
        "indicators": {
            "macd_line": {**macd, "output": 0},
            "macd_signal": {**macd, "output": 1},
        },
        "entry": {
            "long": {
                "cond": "conditions.macd_golden_cross",
                "args": ["macd_line", "macd_signal"],
            }
        },
    }
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(correct_macd, sort_keys=False),
    )
    accepted = demo_server.convert_nl(
        "http://llm/v1", "", "model", macd_request
    )
    assert accepted["valid"] is True, accepted

    rsi_request = "BTC 1h RSI 14 低於 30 買進"
    wrong_rsi = yaml.safe_load(demo_server.EXAMPLE_YAML)
    wrong_rsi["signals"]["entry"].pop("short")
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(wrong_rsi, sort_keys=False),
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", rsi_request)
    assert rejected["valid"] is False
    assert any("RSI" in str(error) and "30" in str(error) for error in rejected["errors"])

    correct_rsi = copy.deepcopy(wrong_rsi)
    correct_rsi["signals"]["entry"]["long"] = {
        "cond": "conditions.rsi_oversold",
        "args": ["rsi14"],
        "params": {"threshold": 30},
    }
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(correct_rsi, sort_keys=False),
    )
    accepted = demo_server.convert_nl("http://llm/v1", "", "model", rsi_request)
    assert accepted["valid"] is True, accepted


@pytest.mark.parametrize(
    ("phrase", "symbol"),
    [
        ("BUY BTC when EMA12 crosses above EMA26 on 1h", "BTC"),
        ("LONG BTC when EMA12 crosses above EMA26 on 1h", "BTC"),
        ("SELL ETH when EMA12 crosses below EMA26 on 1h", "ETH"),
    ],
)
def test_uppercase_trade_verbs_are_not_parsed_as_symbols(demo_server, phrase, symbol):
    assert demo_server.classify_request(phrase).named_symbols == (symbol,)


@pytest.mark.parametrize(
    "phrase",
    [ENGLISH_DISCOVERY_REQUEST, CHINESE_DISCOVERY_REQUEST],
)
def test_exact_requests_really_send_the_selection_prompt(
    demo_server, monkeypatch, phrase,
):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["transport"] = kwargs

        def complete(self, *, system_prompt, user_request):
            captured.update(system_prompt=system_prompt, user_request=user_request)
            return demo_server.SELECTION_EXAMPLE_YAML

    monkeypatch.setenv(demo_server.EXTERNAL_LLM_OPT_IN_ENV, "1")
    monkeypatch.setenv(demo_server.DEMO_LLM_TRUSTED_HOSTS_ENV, "llm.example")
    monkeypatch.setattr(demo_server, "OpenAICompatibleHTTPClient", FakeClient)
    monkeypatch.setattr(
        demo_server.requests,
        "post",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("demo LLM must use the shared bounded HTTP client")
        ),
    )
    demo_server.call_llm("https://llm.example/v1", "", "model", phrase)

    system = captured["system_prompt"]
    assert "selection:" in system
    assert "universe.augment_with_news" in system
    assert captured["user_request"] == phrase
    assert captured["transport"] == {
        "api_base": "https://llm.example/v1",
        "api_key": "",
        "model": "model",
        "allow_external": True,
        "trusted_hosts": ("llm.example",),
        "allow_loopback_http_for_tests": False,
        "timeout_seconds": 120,
    }


def test_demo_llm_keeps_the_explicit_loopback_test_seam(demo_server, monkeypatch):
    """The only HTTP exception remains a server-owned 127.0.0.1 test seam."""
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def complete(self, *, system_prompt, user_request):
            return demo_server.SELECTION_EXAMPLE_YAML

    monkeypatch.setenv(demo_server.EXTERNAL_LLM_OPT_IN_ENV, "1")
    monkeypatch.setenv(demo_server.DEMO_LLM_TRUSTED_HOSTS_ENV, "127.0.0.1")
    monkeypatch.setenv(
        demo_server.DEMO_LLM_ALLOW_LOOPBACK_HTTP_FOR_TESTS_ENV, "true",
    )
    monkeypatch.setattr(demo_server, "OpenAICompatibleHTTPClient", FakeClient)

    demo_server.call_llm(
        "http://127.0.0.1:9876/v1", "", "local-test-model", "挑新聞熱門幣",
    )

    assert captured["trusted_hosts"] == ("127.0.0.1",)
    assert captured["allow_loopback_http_for_tests"] is True


def test_external_llm_route_is_safe_default_deny(demo_server, monkeypatch):
    """No raw NL leaves the local demo unless its process opted in explicitly."""

    monkeypatch.delenv(demo_server.EXTERNAL_LLM_OPT_IN_ENV, raising=False)
    monkeypatch.setattr(
        demo_server.requests,
        "post",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("disabled route must not make an HTTP request")
        ),
    )
    monkeypatch.setattr(
        demo_server,
        "OpenAICompatibleHTTPClient",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("disabled route must not even construct an HTTP client")
        ),
    )

    result = demo_server.convert_nl(
        "http://llm/v1", "", "model", "幫我挑選最近新聞常提到的五個幣種"
    )

    assert result["status"] == "external_llm_disabled"
    assert result["valid"] is False
    assert demo_server.EXTERNAL_LLM_OPT_IN_ENV in "\n".join(result["errors"])


def test_chinese_news_request_converts_to_a_valid_selection_spec(
    demo_server, monkeypatch,
):
    generated = demo_server.SELECTION_EXAMPLE_YAML
    seen = {}

    def fake_llm(api_base, api_key, model, nl, *_, **__):
        seen.update(api_base=api_base, api_key=api_key, model=model, nl=nl)
        return generated

    monkeypatch.setattr(demo_server, "call_llm", fake_llm)
    result = demo_server.convert_nl(
        "http://litellm.internal/v1", "secret", "company-model",
        "幫我挑選最近新聞常提到的五個幣種",
    )

    assert seen["nl"] == "幫我挑選最近新聞常提到的五個幣種"
    assert result["ok"] is True
    assert result["valid"] is True, result
    assert result["strategy_kind"] == "selection"
    assert isinstance(yaml.safe_load(result["yaml"])["selection"], dict)


def test_mixed_trade_and_selection_output_is_not_routed(demo_server, monkeypatch):
    mixed = yaml.safe_load(SELECTION_YAML.read_text(encoding="utf-8"))
    mixed["signals"] = {
        "indicators": {},
        "entry": {
            "long": {"cond": "conditions.value_above", "args": ["close", 0]}
        },
    }
    monkeypatch.setattr(
        demo_server, "call_llm", lambda *_, **__: yaml.safe_dump(mixed, sort_keys=False)
    )

    result = demo_server.convert_nl("http://llm/v1", "", "model", "依新聞熱度選幣")
    assert result["ok"] is True
    assert result["valid"] is False
    assert result["strategy_kind"] == "selection"
    assert result["errors"], "invalid mixed YAML must explain why it cannot run"


def test_selection_intent_rejects_a_valid_but_wrong_trade_yaml(
    demo_server, monkeypatch,
):
    """A model ignoring the selection prompt must not silently change intent."""
    generated = TRADE_YAML.read_text(encoding="utf-8")
    monkeypatch.setattr(demo_server, "call_llm", lambda *_, **__: generated)

    result = demo_server.convert_nl(
        "http://llm/v1", "", "model", "幫我挑選最近新聞常提到的幣種"
    )

    assert result["strategy_kind"] == "selection"
    assert result["valid"] is False
    assert any("selection" in str(error).lower() for error in result["errors"])


def test_exact_discovery_request_rejects_a_valid_sui_trade_yaml(
    demo_server, monkeypatch,
):
    trade = yaml.safe_load(TRADE_YAML.read_text(encoding="utf-8"))
    trade["strategy"]["id"] = "sui_technical_guess"
    trade["data"]["symbol"] = "SUIUSDT"
    monkeypatch.setattr(
        demo_server, "call_llm", lambda *_, **__: yaml.safe_dump(trade, sort_keys=False)
    )

    result = demo_server.convert_nl(
        "http://llm/v1", "", "model", CHINESE_DISCOVERY_REQUEST
    )

    assert result["strategy_kind"] == "selection"
    assert result["generated_strategy_kind"] == "trade"
    assert result["valid"] is False
    assert any("selection" in str(error).lower() for error in result["errors"])


def test_news_trade_request_rejects_technical_analysis_proxy(
    demo_server, monkeypatch,
):
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: TRADE_YAML.read_text(encoding="utf-8"),
    )

    result = demo_server.convert_nl(
        "http://llm/v1", "", "model", "新聞提到 BTC 時就買進"
    )

    assert result["strategy_kind"] == "trade"
    assert result["generated_strategy_kind"] == "trade"
    assert result["valid"] is False
    assert any("EventFrame" in str(error) and "技術指標" in str(error)
               for error in result["errors"])


def test_news_request_rejects_selection_that_does_not_use_news(
    demo_server, monkeypatch,
):
    generated = yaml.safe_load(demo_server.SELECTION_EXAMPLE_YAML)
    generated["selection"]["universe"] = [
        {
            "block": "universe.filter_quote_volume",
            "params": {"min_quote_volume": 100_000_000},
        }
    ]
    generated["selection"]["score"] = "quoteVolume"
    monkeypatch.setattr(
        demo_server, "call_llm", lambda *_, **__: yaml.safe_dump(generated, sort_keys=False)
    )

    result = demo_server.convert_nl(
        "http://llm/v1", "", "model", ENGLISH_DISCOVERY_REQUEST
    )

    assert result["strategy_kind"] == "selection"
    assert result["generated_strategy_kind"] == "selection"
    assert result["valid"] is False
    assert any("news" in str(error).lower() or "新聞" in str(error) for error in result["errors"])


@pytest.mark.parametrize(
    "phrase",
    [
        "Select 5 coins with highest open interest",
        "幫我選五個未平倉量最高的幣",
    ],
)
def test_open_interest_cross_section_uses_the_now_supported_fanout_path(
    demo_server, monkeypatch, phrase,
):
    """The front door must track the narrowed-roster OI Blocks that now run."""
    intent = demo_server.classify_request(phrase)
    assert intent.sources == frozenset({"open_interest"})
    prompt = demo_server.build_system_prompt("selection", intent)
    assert "universe.augment_with_open_interest" in prompt
    assert "with: [open_interest_snapshot]" in prompt
    assert "score: oi_notional_usd" in prompt

    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: demo_server.OPEN_INTEREST_SELECTION_EXAMPLE_YAML,
    )
    accepted = demo_server.convert_nl("http://llm/v1", "", "model", phrase)
    assert accepted["status"] == "valid", accepted
    assert accepted["valid"] is True, accepted

    # A plausible news basket is still the wrong answer to an OI request.
    monkeypatch.setattr(
        demo_server, "call_llm", lambda *_, **__: demo_server.SELECTION_EXAMPLE_YAML
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", phrase)
    assert rejected["valid"] is False
    assert any("open interest" in str(error) for error in rejected["errors"])


def test_long_short_ratio_is_not_misread_as_a_bullish_news_preference(
    demo_server, monkeypatch,
):
    """Crowd-long positioning must use its own frame and percent-point filter."""

    request = "選五個散戶多空比偏多 > 60% 的幣"
    intent = demo_server.classify_request(request)

    assert intent.sources == frozenset({"long_short_ratio"})
    assert intent.long_short_min_account_pct == 60.0
    assert intent.long_short_min_account_operator == ">"
    assert intent.bullish_preference is False
    assert intent.directions == ()
    prompt = demo_server.build_system_prompt("selection", intent)
    assert "universe.augment_with_long_short_ratio" in prompt
    assert "long_short_ratio_snapshot" in prompt
    assert "min_long_account_pct_exclusive: 60" in prompt

    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: demo_server._long_short_ratio_selection_example(intent),
    )
    accepted = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert accepted["status"] == "valid", accepted
    assert accepted["valid"] is True, accepted

    # A syntactically valid inclusive filter would be one candidate too broad
    # at exactly 60 %.  Reconciliation must preserve the user's strict sign.
    wrong_boundary = yaml.safe_load(
        demo_server._long_short_ratio_selection_example(intent)
    )
    wrong_boundary["selection"]["universe"][-1]["params"] = {
        "min_long_account_pct": 60.0,
    }
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(
            wrong_boundary, allow_unicode=True, sort_keys=False,
        ),
    )
    wrong_boundary_result = demo_server.convert_nl(
        "http://llm/v1", "", "model", request,
    )
    assert wrong_boundary_result["valid"] is False
    assert any("min_long_account_pct_exclusive" in str(error)
               for error in wrong_boundary_result["errors"])

    # The prior generic news YAML is structurally valid, but it answers a
    # different question and must fail reconciliation rather than look green.
    monkeypatch.setattr(
        demo_server, "call_llm", lambda *_, **__: demo_server.SELECTION_EXAMPLE_YAML
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert rejected["valid"] is False
    assert any("多空比" in str(error) or "long_short" in str(error)
               for error in rejected["errors"])


def test_long_short_ratio_preserves_an_inclusive_boundary(demo_server):
    request = "選五個散戶多空比偏多至少 60% 的幣"
    intent = demo_server.classify_request(request)

    assert intent.long_short_min_account_pct == 60.0
    assert intent.long_short_min_account_operator == ">="
    example = yaml.safe_load(demo_server._long_short_ratio_selection_example(intent))
    params = example["selection"]["universe"][-1]["params"]
    assert params == {"min_long_account_pct": 60.0}


def test_multi_timeframe_supertrend_request_requires_real_direction_steps(
    demo_server, monkeypatch,
):
    """H4/H1/M15 are timeframes, and all three must affect the selector."""

    request = "選五個幣，Supertrend(10,3) 在 H4/H1/M15 同時偏空"
    intent = demo_server.classify_request(request)

    assert intent.kind == "selection"
    assert intent.intervals == ("4h", "1h", "15m")
    assert intent.named_symbols == ()
    assert intent.directions == ("short",)
    assert intent.supertrend_parameters == ((10, 3.0),)
    prompt = demo_server.build_system_prompt("selection", intent)
    assert "universe.augment_with_indicator" in prompt
    assert "output: 1" in prompt
    assert "all_of" in prompt

    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: demo_server._supertrend_selection_example(intent),
    )
    accepted = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert accepted["status"] == "valid", accepted
    assert accepted["valid"] is True, accepted

    # A plausible price/liquidity basket no longer passes merely because it is
    # a valid selection YAML: every requested timeframe must be real and wired.
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: demo_server.LIQUIDITY_SELECTION_EXAMPLE_YAML,
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert rejected["valid"] is False
    assert any("Supertrend" in str(error) for error in rejected["errors"])


def test_multi_timeframe_supertrend_rejects_an_any_of_bypass(
    demo_server, monkeypatch,
):
    """Mentioning H4 is insufficient when liquidity can replace it."""
    request = "選五個幣，Supertrend(10,3) 在 H4/H1/M15 同時偏空"
    intent = demo_server.classify_request(request)
    bypass = yaml.safe_load(demo_server._supertrend_selection_example(intent))
    leaves = bypass["selection"]["short_when"]["all_of"]
    bypass["selection"]["short_when"] = {
        "all_of": [
            {
                "any_of": [
                    leaves[0],
                    # This valid condition lets a liquid candidate through even
                    # when the requested H4 Supertrend is not bearish.
                    {"cond": "conditions.value_above", "args": ["quoteVolume", 0]},
                ],
            },
            *leaves[1:],
        ],
    }
    bypass_yaml = yaml.safe_dump(bypass, allow_unicode=True, sort_keys=False)

    monkeypatch.setattr(demo_server, "call_llm", lambda *_, **__: bypass_yaml)
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)

    assert rejected["valid"] is False
    assert any("可繞過" in str(error) for error in rejected["errors"])


def test_unmapped_asset_category_exclusions_stop_before_llm(demo_server, monkeypatch):
    """Never claim TradFi/stock-token/stablecoin exclusions from a guess."""

    def should_not_run(*_, **__):
        raise AssertionError("unmapped category exclusion must stop before LLM")

    monkeypatch.setattr(demo_server, "call_llm", should_not_run)
    result = demo_server.convert_nl(
        "http://llm/v1", "", "model",
        "選五個成交量最高的幣，排除 TradFi、股票代幣與穩定幣",
    )

    assert result["status"] == "needs_clarification"
    assert result["valid"] is False
    assert set(result["intent"]["unsupported_preferences"]) >= {
        "category_exclusion:tradfi",
        "category_exclusion:stock_token",
        "category_exclusion:stablecoin",
    }
    assert any("metadata" in str(error) for error in result["errors"])


def test_lowest_funding_is_expressed_with_ascending_order(
    demo_server, monkeypatch,
):
    """Asking for the most negative funding was refused for lacking ascending
    ranking.

    ``selection.order`` now exists, so the request must convert. The refusal
    existed to stop the opposite basket being returned, so the test that replaces
    it checks the direction survives end to end — prompt, YAML and gate.
    """
    request = "幫我選五個 funding rate 最負的幣"
    intent = demo_server.classify_request(request)
    assert intent.score_order == "asc"
    assert intent.score_order_metric == "fundingRatePct"

    # Assert on the EXAMPLE half only. The instruction half of this prompt
    # contains the literal "order: asc" whichever direction was requested, so
    # ``"order: asc" in prompt`` was green even when the desc example shipped.
    example = _prompt_example(demo_server, intent)
    assert "order: asc" in example
    assert "order: desc" not in example
    assert "id: lowest_funding_rate_selector" in example

    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: demo_server.FUNDING_ASC_SELECTION_EXAMPLE_YAML,
    )
    accepted = demo_server.convert_nl("http://llm/v1", "", "model", request)

    assert accepted["status"] == "valid", accepted
    assert accepted["valid"] is True, accepted
    generated = yaml.safe_load(accepted["yaml"])
    # The ANNUALISED column: the raw per-settlement rate is not comparable across
    # contracts that settle 8h / 4h / 1h apart, so "the most negative funding"
    # taken from it is the wrong five coins. ``score_order_metric`` still reads
    # ``fundingRatePct`` because that is the name the PHRASE matched under —
    # intent._METRIC_COLUMN_ALIASES is what keeps the direction check enforced
    # across the two spellings.
    assert generated["selection"]["score"] == "fundingRateApr"
    assert generated["selection"]["order"] == "asc"
    assert generated["selection"]["top_k"] == 5

    # The whole reason the refusal existed: descending here hands back the coins
    # paying the MOST funding, which is the opposite trade, and the basket gives
    # no sign of it.
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: demo_server.FUNDING_SELECTION_EXAMPLE_YAML,
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)

    assert rejected["valid"] is False
    assert any("fundingRatePct" in str(error) and "asc" in str(error)
               for error in rejected["errors"]), rejected["errors"]


def test_price_change_selection_ranks_on_the_real_ticker_column(
    demo_server, monkeypatch,
):
    """24h price change was refused as "not wired up", which was never true.

    ``priceChangePercent`` arrives with the universe frame (it IS the Binance 24h
    ticker) and ``example_from_user_chat.yaml`` has always ranked on it.
    """
    request = "幫我選五個漲幅最大的幣"
    intent = demo_server.classify_request(request)
    assert intent.sources == frozenset({"price_change"})
    assert (intent.score_order, intent.score_order_metric) == (
        "desc", "priceChangePercent",
    )
    assert "priceChangePercent" in demo_server.build_system_prompt(
        "selection", intent
    )
    example = _prompt_example(demo_server, intent)
    assert "order: desc" in example
    assert "order: asc" not in example
    assert "id: price_change_selector" in example

    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: demo_server.PRICE_CHANGE_SELECTION_EXAMPLE_YAML,
    )
    accepted = demo_server.convert_nl("http://llm/v1", "", "model", request)

    assert accepted["status"] == "valid", accepted
    generated = yaml.safe_load(accepted["yaml"])
    assert generated["selection"]["score"] == "priceChangePercent"
    assert generated["selection"]["top_k"] == 5

    # Supporting the source is not the same as letting news stand in for it.
    monkeypatch.setattr(
        demo_server, "call_llm", lambda *_, **__: demo_server.SELECTION_EXAMPLE_YAML
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)

    assert rejected["valid"] is False
    assert any("priceChangePercent" in str(error) for error in rejected["errors"])


def test_biggest_losers_request_flips_the_order_to_ascending(
    demo_server, monkeypatch,
):
    """The same column, the other end — and the gate must tell them apart."""
    request = "幫我選五個跌幅最大的幣"
    intent = demo_server.classify_request(request)
    assert (intent.score_order, intent.score_order_metric) == (
        "asc", "priceChangePercent",
    )

    example = _prompt_example(demo_server, intent)
    assert "order: asc" in example
    assert "order: desc" not in example
    assert "id: biggest_losers_selector" in example

    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: demo_server.PRICE_CHANGE_ASC_SELECTION_EXAMPLE_YAML,
    )
    assert demo_server.convert_nl(
        "http://llm/v1", "", "model", request
    )["valid"] is True

    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: demo_server.PRICE_CHANGE_SELECTION_EXAMPLE_YAML,
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert rejected["valid"] is False
    assert any("asc" in str(error) for error in rejected["errors"])


@pytest.mark.parametrize(
    ("phrase", "expected_order"),
    [
        # The magnitude word decides the end, and it decides it differently for
        # each direction word: 跌幅 is the SIZE of a fall, so a big one is the
        # bottom of the signed column and a small one is the top.
        ("幫我選五個跌幅最大的幣", "asc"),
        ("幫我選五個跌幅最小的幣", "desc"),
        ("幫我選五個漲幅最大的幣", "desc"),
        ("幫我選五個漲幅最小的幣", "asc"),
        ("選跌幅前 30 名的幣", "asc"),
        # A user who spells the sort out is the least ambiguous case there is.
        ("選幣,跌幅由大到小", "asc"),
        ("選幣,漲幅由大到小", "desc"),
        ("show me the top losers", "asc"),
        ("show me the top gainers", "desc"),
        ("pick the worst performing coins", "asc"),
        ("pick the best performing coins", "desc"),
    ],
)
def test_magnitude_word_decides_which_end_of_the_change_column(
    demo_server, phrase, expected_order,
):
    """The direction word alone cannot say which end, and guessing is enforced.

    ``reconcile_intent`` rejects a spec that disagrees with the direction landing
    here, so reading 跌幅最小 as "most negative" did not merely lose a check — it
    refused the correct spec and accepted the inverted one.
    """
    intent = demo_server.classify_request(phrase)

    assert (intent.score_order, intent.score_order_metric) == (
        expected_order, "priceChangePercent",
    ), phrase


@pytest.mark.parametrize(
    "phrase",
    [
        # A threshold on the fall is a filter; the ranking column is unstated.
        "幫我選幾個跌幅小於 3% 的幣",
        "幫我選幾個漲幅超過 5% 的幣",
        # "exclude the biggest fallers" names the end the user does NOT want, and
        # no ``order`` value means "rank everything else".
        "選幾個幣,要排除跌幅超過 10% 的",
        "選幾個幣,排除跌幅最大的那些",
        "show me coins avoiding the losers",
    ],
)
def test_change_filters_and_exclusions_are_not_a_ranking_direction(
    demo_server, phrase,
):
    intent = demo_server.classify_request(phrase)

    assert intent.score_order is None, phrase
    assert intent.score_order_metric is None, phrase


def test_excluding_a_symbol_still_leaves_the_ranking_direction_intact(demo_server):
    """The exclusion guard must not swallow every request containing 排除.

    "排除 BTC" excludes a symbol, not an end of the column, so the direction is
    still stated and still worth enforcing.
    """
    intent = demo_server.classify_request("排除 BTC,選跌幅最大的五個幣")

    assert (intent.score_order, intent.score_order_metric) == (
        "asc", "priceChangePercent",
    )


@pytest.mark.parametrize(
    ("phrase", "expected_order"),
    [
        ("幫我選五個 funding 為負的幣", "asc"),
        ("幫我選五個 funding rate 是負的幣", "asc"),
        ("pick 5 coins with negative funding", "asc"),
        ("幫我選五個 funding rate 為正的幣", "desc"),
        ("pick 5 coins with positive funding rate", "desc"),
        ("選幣,funding rate 由低到高", "asc"),
        ("選幣,funding rate 由高到低", "desc"),
        ("rank coins by funding rate ascending", "asc"),
    ],
)
def test_signed_funding_phrasings_name_an_end_of_the_column(
    demo_server, phrase, expected_order,
):
    """The ordinary way to ask for the bottom of funding is 為負 / negative.

    Leaving it unrecognised did not yield "no direction stated": the prompt then
    asserted the opposite end, because ``score_order is None`` was rendered as
    desc.
    """
    intent = demo_server.classify_request(phrase)

    assert (intent.score_order, intent.score_order_metric) == (
        expected_order, "fundingRatePct",
    ), phrase


@pytest.mark.parametrize(
    "phrase",
    [
        "幫我選五個 funding rate 最不負的幣",
        "pick 5 coins with the least negative funding rate",
    ],
)
def test_a_diminished_sign_word_is_not_the_bottom_of_the_column(
    demo_server, phrase,
):
    """最不負 / least negative is the near-zero middle, not either end.

    Recognising 為負/negative as "the bottom of the funding column" is what makes
    this reachable, and it cannot be answered with ``order``: where the near-zero
    rates sit depends on how much of today's column is below zero.
    """
    intent = demo_server.classify_request(phrase)

    assert "funding" in intent.sources, phrase
    assert intent.score_order is None, phrase
    assert intent.score_order_metric is None, phrase


@pytest.mark.parametrize(
    "phrase",
    [
        # Contradictory: two ends of the same column.
        "幫我選五個 funding rate 最負或最高的幣",
        "幫我選漲幅或跌幅最大的幣",
        # Stated for nothing: a bare source with no end named.
        "幫我選五個 funding rate 的幣",
        "幫我選幾個有 24 小時漲跌幅資料的幣",
    ],
)
def test_unstated_direction_is_not_rendered_as_descending(demo_server, phrase):
    """``score_order is None`` is a third state, and the prompt must show it.

    The demo used to compute ``ascending = score_order == "asc"``, so "the user
    did not say" became "the user said desc" — an affirmative claim about the
    request, plus a desc example to copy. ``reconcile_intent`` skips its check
    when no direction was stated, so nothing downstream can catch it: whatever
    the prompt suggests is what the basket ends up being.
    """
    intent = demo_server.classify_request(phrase)
    assert intent.score_order is None, phrase

    prompt = demo_server.build_system_prompt("selection", intent)
    example = _prompt_example(demo_server, intent)

    assert demo_server.NO_ORDER_DIRECTIVE in prompt
    assert "order: desc" not in prompt
    assert "order: asc" not in prompt
    # The example must not smuggle the direction back in as a copyable line.
    assert "order:" not in example
    assert "score:" in example


def test_a_direction_for_another_column_does_not_flip_the_funding_prompt(
    demo_server,
):
    """A direction is only usable for the column it was stated about.

    Here the end named belongs to ``priceChangePercent`` while the prompt is the
    funding one. Reading ``score_order`` without its metric told the model to rank
    ``fundingRatePct`` ascending — the coins PAYING shorts, the opposite trade —
    and ``reconcile_intent`` cannot object, because it only checks the direction
    against the column it was stated for.
    """
    intent = demo_server.classify_request(
        "幫我用 funding rate 這個欄位選幣,條件是最近二十四小時之內的跌幅最大的那五個幣"
    )
    assert "funding" in intent.sources
    assert (intent.score_order, intent.score_order_metric) == (
        "asc", "priceChangePercent",
    )

    prompt = demo_server.build_system_prompt("selection", intent)
    example = _prompt_example(demo_server, intent)

    assert demo_server.NO_ORDER_DIRECTIVE in prompt
    assert "order:" not in example
    assert "id: default_order_funding_rate_selector" in example


def test_narrowing_direction_does_not_reject_a_spec_ranked_on_another_column(
    demo_server, monkeypatch,
):
    """The ``score_order_metric in score_dependencies`` half of the gate.

    ``example_from_user_chat.yaml`` is the shape this protects: the direction
    phrase describes a NARROWING step (``universe.top_losers``, "24h 跌幅前 30 名")
    while ``score`` ranks the survivors on ``quoteVolume`` — same-strength coins,
    the liquid one is the tradable short. Enforcing ``order: asc`` on that spec
    would reject a conversion that does exactly what was asked, and the phrasing
    is common enough that it arrived from a real user conversation.
    """
    request = "選跌幅最大的幣裡成交額最大的五個做空候選"
    intent = demo_server.classify_request(request)
    assert (intent.score_order, intent.score_order_metric) == (
        "asc", "priceChangePercent",
    )

    generated = CHAT_SELECTION_YAML.read_text(encoding="utf-8")
    # Spelled out because it is the whole trap: the requested direction is about
    # a column this spec does not rank on. If the example ever starts ranking on
    # priceChangePercent, this test must fail loudly rather than quietly stop
    # covering the guard.
    assert yaml.safe_load(generated)["selection"]["score"] == "quoteVolume"
    monkeypatch.setattr(demo_server, "call_llm", lambda *_, **__: generated)

    result = demo_server.convert_nl("http://llm/v1", "", "model", request)

    assert result["valid"] is True, result["errors"]


def test_price_change_selection_runs_through_real_blocks(demo_server, monkeypatch):
    """The un-refused source has to survive the bundle, not just the gate.

    ``priceChangePercent`` is only usable if it travels the whole way: universe
    frame in ``cyqnt.input/v1`` -> Blocks pipeline -> ranked basket. The sample
    bundle carries a hand-written minimal universe frame, so the column is added
    here the way the live Binance 24h ticker supplies it — one value per symbol,
    signed — with ETH the stronger of the two.
    """
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: demo_server.PRICE_CHANGE_SELECTION_EXAMPLE_YAML,
    )
    converted = demo_server.convert_nl(
        "http://llm/v1", "", "model", "幫我選五個漲幅最大的幣"
    )
    assert converted["valid"] is True, converted

    bundle = json.loads(SAMPLE_BUNDLE.read_text(encoding="utf-8"))
    changes = {"BTCUSDT": -3.5, "ETHUSDT": 7.25}
    for row in bundle["frames"]["universe"]["rows"]:
        row["priceChangePercent"] = changes[row["instrument_id"]]
    from cyqnt_trd.standard_bot.data import live_snapshot

    monkeypatch.setattr(
        live_snapshot,
        "build_live_snapshot",
        lambda **_: (None, copy.deepcopy(bundle)),
    )
    gainers = demo_server.run_selection(converted["yaml"])

    assert gainers["ok"] is True, gainers
    assert [item["symbol"] for item in gainers["candidates"]] == ["ETHUSDT", "BTCUSDT"]
    assert gainers["candidates"][0]["score"] == 7.25

    # Same bundle, opposite order: the losers spec must invert the basket rather
    # than return the same one with a different label.
    losers = demo_server.run_selection(
        demo_server.PRICE_CHANGE_ASC_SELECTION_EXAMPLE_YAML
    )
    assert [item["symbol"] for item in losers["candidates"]] == ["BTCUSDT", "ETHUSDT"]
    assert losers["candidates"][0]["score"] == -3.5


def test_contradictory_direction_words_do_not_invent_an_order(demo_server):
    """Two opposite phrasings mean the user has not said; guessing would reject
    a spec for disagreeing with a coin flip."""
    intent = demo_server.classify_request("幫我選漲幅或跌幅最大的幣")

    assert intent.sources == frozenset({"price_change"})
    assert intent.score_order is None
    assert intent.score_order_metric is None


def test_high_funding_selection_is_translated_to_the_real_funding_block(
    demo_server, monkeypatch,
):
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: demo_server.FUNDING_SELECTION_EXAMPLE_YAML,
    )

    result = demo_server.convert_nl(
        "http://llm/v1", "", "model", "幫我選五個 funding rate 最高的幣"
    )

    assert result["valid"] is True, result
    generated = yaml.safe_load(result["yaml"])
    assert generated["selection"]["score"] == "fundingRateApr"
    # BOTH sources. ``funding`` alone leaves fundingRateApr NaN for every
    # instrument — the block will not assume an 8-hour settlement interval — so
    # the basket would come back empty; spec._refuse_column_without_its_source
    # is what turns that into a validation error rather than a silent zero-length
    # basket, and this asserts the template does not rely on it.
    assert {
        step["block"]: step.get("with")
        for step in generated["selection"]["universe"]
    }["universe.augment_with_funding"] == ["funding", "funding_info"]


def test_volume_selection_requires_volume_to_control_the_score(
    demo_server, monkeypatch,
):
    request = "幫我選五個成交量最大的幣"
    monkeypatch.setattr(
        demo_server, "call_llm", lambda *_, **__: demo_server.SELECTION_EXAMPLE_YAML
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert rejected["valid"] is False
    assert any("selection.score" in str(error) for error in rejected["errors"])

    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: demo_server.LIQUIDITY_SELECTION_EXAMPLE_YAML,
    )
    accepted = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert accepted["valid"] is True, accepted
    assert "score: quoteVolume" in demo_server.build_system_prompt(
        "selection", demo_server.classify_request(request)
    )


@pytest.mark.parametrize(
    "user_request",
    [
        "幫我選五個幣，依24小時成交額排名",
        "幫我選五個幣，依24小時成交量排名",
    ],
)
def test_24h_turnover_is_a_universe_metric_not_a_candle_interval(
    demo_server, monkeypatch, user_request,
):
    """24h quote turnover arrives in the universe frame, not a 24h bar feed."""
    intent = demo_server.classify_request(user_request)
    assert intent.kind == "selection"
    assert intent.sources == frozenset({"liquidity"})
    assert intent.intervals == ()
    assert intent.ranking_metric == "liquidity"

    monkeypatch.setattr(
        demo_server, "call_llm", lambda *_, **__: demo_server.LIQUIDITY_SELECTION_EXAMPLE_YAML,
    )
    converted = demo_server.convert_nl(
        "http://llm/v1", "", "model", user_request,
    )
    assert converted["valid"] is True, converted
    generated = yaml.safe_load(converted["yaml"])
    assert generated["data"]["primary"]["interval"] == "1h"
    assert generated["selection"]["score"] == "quoteVolume"
    assert generated["selection"]["top_k"] == 5

    from cyqnt_trd.standard_bot.yaml_pipeline.bundle_runner import run_bundle

    batch = run_bundle(generated, str(UNIVERSE_DERIVATIVES_BUNDLE))
    assert batch["signal_count"] == 1
    signal = batch["signals"][0]
    assert signal["kind"] == "selection"
    assert len(signal["candidates"]) == 5


def test_hot_coins_by_volume_uses_volume_as_the_primary_metric(
    demo_server, monkeypatch,
):
    request = "Select 5 hot coins by volume"
    intent = demo_server.classify_request(request)
    assert intent.requested_count == 5
    assert intent.ranking_metric == "liquidity"
    assert intent.sources == frozenset({"liquidity"})

    monkeypatch.setattr(
        demo_server, "call_llm", lambda *_, **__: demo_server.SELECTION_EXAMPLE_YAML
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert rejected["valid"] is False
    assert any("quoteVolume" in str(error) for error in rejected["errors"])

    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: demo_server.LIQUIDITY_SELECTION_EXAMPLE_YAML,
    )
    accepted = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert accepted["valid"] is True, accepted


@pytest.mark.parametrize("decoy_name", ["news_decoy", "news_bull_ratio"])
def test_news_feature_names_cannot_bypass_dependency_check(
    demo_server, monkeypatch, decoy_name,
):
    generated = yaml.safe_load(demo_server.SELECTION_EXAMPLE_YAML)
    generated["selection"]["features"] = {
        decoy_name: {
            "block": "indicators.ema",
            "input": "quoteVolume",
            "params": {"period": 2},
        }
    }
    generated["selection"]["score"] = decoy_name
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(generated, sort_keys=False),
    )

    result = demo_server.convert_nl(
        "http://llm/v1", "", "model", CHINESE_DISCOVERY_REQUEST
    )

    assert result["valid"] is False
    combined = "\n".join(map(str, result["errors"]))
    assert "news_*" in combined
    assert "news_bull_ratio" in combined


def test_sentiment_ranking_cannot_be_replaced_with_mention_ranking(
    demo_server, monkeypatch,
):
    request = "依照社群情緒排行，選出三個候選幣"
    mention_spec = yaml.safe_load(demo_server.SELECTION_EXAMPLE_YAML)
    mention_spec["selection"]["top_k"] = 3
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(mention_spec, sort_keys=False),
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert rejected["valid"] is False
    assert any("news_bull_ratio" in str(error) for error in rejected["errors"])

    sentiment_spec = copy.deepcopy(mention_spec)
    sentiment_spec["selection"]["score"] = "news_bull_ratio"
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(sentiment_spec, sort_keys=False),
    )
    accepted = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert accepted["valid"] is True, accepted
    prompt = demo_server.build_system_prompt(
        "selection", demo_server.classify_request(request)
    )
    assert "selection.score 必須使用 news_bull_ratio" in prompt


def test_explicit_sentiment_side_mapping_rejects_reversed_news_conditions(
    demo_server, monkeypatch,
):
    """A structurally valid selector must not invert sentiment-side meaning."""

    request = (
        "Select 5 coins: go long on positive social sentiment and short on "
        "negative social sentiment"
    )
    intent = demo_server.classify_request(request)
    assert intent.kind == "selection"
    assert intent.sentiment_side_mapping == (
        ("positive", "long"), ("negative", "short"),
    )

    correct = yaml.safe_load(demo_server.SELECTION_EXAMPLE_YAML)
    correct["selection"]["score"] = "news_bull_ratio"
    correct["selection"]["long_when"] = {
        "cond": "conditions.value_above", "args": ["news_bull_ratio", 0.55],
    }
    correct["selection"]["short_when"] = {
        "cond": "conditions.value_below", "args": ["news_bull_ratio", 0.45],
    }
    monkeypatch.setattr(
        demo_server, "call_llm", lambda *_, **__: yaml.safe_dump(correct, sort_keys=False),
    )
    accepted = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert accepted["valid"] is True, accepted

    reversed_sides = copy.deepcopy(correct)
    reversed_sides["selection"]["long_when"] = {
        "cond": "conditions.value_below", "args": ["news_bull_ratio", 0.45],
    }
    reversed_sides["selection"]["short_when"] = {
        "cond": "conditions.value_above", "args": ["news_bull_ratio", 0.55],
    }
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(reversed_sides, sort_keys=False),
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert rejected["valid"] is False
    assert any("反向或可繞過" in str(error) for error in rejected["errors"])

    # Merely mentioning both sides does not manufacture a polarity mapping.
    unbound = demo_server.classify_request(
        "Select 5 coins by social sentiment with long and short candidates"
    )
    assert unbound.sentiment_side_mapping == ()


def test_explicit_contrarian_sentiment_mapping_preserves_each_requested_side(
    demo_server, monkeypatch,
):
    """Do not silently normalise an explicit contrarian selector."""

    request = (
        "Select 5 coins: go short on positive social sentiment and long on "
        "negative social sentiment"
    )
    intent = demo_server.classify_request(request)
    assert intent.kind == "selection"
    assert intent.sentiment_side_mapping == (
        ("positive", "short"), ("negative", "long"),
    )

    contrarian = yaml.safe_load(demo_server.SELECTION_EXAMPLE_YAML)
    contrarian["selection"]["score"] = "news_bull_ratio"
    contrarian["selection"]["long_when"] = {
        "cond": "conditions.value_below", "args": ["news_bull_ratio", 0.45],
    }
    contrarian["selection"]["short_when"] = {
        "cond": "conditions.value_above", "args": ["news_bull_ratio", 0.55],
    }
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(contrarian, sort_keys=False),
    )
    accepted = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert accepted["valid"] is True, accepted

    conventional = copy.deepcopy(contrarian)
    conventional["selection"]["long_when"] = {
        "cond": "conditions.value_above", "args": ["news_bull_ratio", 0.55],
    }
    conventional["selection"]["short_when"] = {
        "cond": "conditions.value_below", "args": ["news_bull_ratio", 0.45],
    }
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(conventional, sort_keys=False),
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert rejected["valid"] is False
    assert any("反向或可繞過" in str(error) for error in rejected["errors"])


def test_bullish_preference_requires_a_supported_proxy(demo_server, monkeypatch):
    monkeypatch.setattr(
        demo_server, "call_llm", lambda *_, **__: demo_server.SELECTION_EXAMPLE_YAML
    )

    result = demo_server.convert_nl(
        "http://llm/v1", "", "model", CHINESE_DISCOVERY_REQUEST
    )

    assert result["valid"] is False
    assert any("偏多" in str(error) or "bull" in str(error).lower()
               for error in result["errors"])


def test_ineffective_news_filters_do_not_satisfy_the_requested_ranking(
    demo_server, monkeypatch,
):
    mention_spec = yaml.safe_load(demo_server.SELECTION_EXAMPLE_YAML)
    mention_spec["selection"]["universe"].append(
        {"block": "universe.top_mentioned", "params": {"n": 1000}}
    )
    mention_spec["selection"]["score"] = "quoteVolume"
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(mention_spec, sort_keys=False),
    )
    rejected = demo_server.convert_nl(
        "http://llm/v1", "", "model", "選五個新聞提及量最高的幣"
    )
    assert rejected["valid"] is False
    assert any("news_mention_count" in str(error) for error in rejected["errors"])

    bullish_spec = yaml.safe_load(TERRA_BULLISH_SELECTION_YAML)
    bullish_spec["selection"]["universe"][-1]["params"]["min_bull_ratio"] = 0.0
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(bullish_spec, sort_keys=False),
    )
    rejected = demo_server.convert_nl(
        "http://llm/v1", "", "model", CHINESE_DISCOVERY_REQUEST
    )
    assert rejected["valid"] is False
    assert any("min_bull_ratio > 0.5" in str(error) for error in rejected["errors"])


def test_unrequested_single_symbol_allowlist_is_rejected(demo_server, monkeypatch):
    generated = yaml.safe_load(demo_server.SELECTION_EXAMPLE_YAML)
    generated["selection"]["universe"].append(
        {"block": "universe.only_symbols", "params": {"symbols": ["SUIUSDT"]}}
    )
    monkeypatch.setattr(
        demo_server, "call_llm", lambda *_, **__: yaml.safe_dump(generated, sort_keys=False)
    )

    result = demo_server.convert_nl(
        "http://llm/v1", "", "model", "幫我選一些熱門幣"
    )

    assert result["valid"] is False
    assert any("SUIUSDT" in str(error) for error in result["errors"])


def test_requested_symbol_allowlist_cannot_be_silently_dropped(demo_server, monkeypatch):
    request = "幫我從 BTC、ETH、SOL 中選新聞最熱門的兩個幣"
    intent = demo_server.classify_request(request)
    assert intent.included_symbols == ("BTC", "ETH", "SOL")
    assert intent.excluded_symbols == ()

    generated = yaml.safe_load(demo_server.SELECTION_EXAMPLE_YAML)
    generated["selection"]["top_k"] = 2
    monkeypatch.setattr(
        demo_server, "call_llm", lambda *_, **__: yaml.safe_dump(generated, sort_keys=False)
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert rejected["valid"] is False
    assert any("only_symbols" in str(error) for error in rejected["errors"])


def test_requested_symbol_exclusion_must_be_expressed(demo_server, monkeypatch):
    request = "依新聞熱度選五個幣，排除 BTC"
    intent = demo_server.classify_request(request)
    assert intent.included_symbols == ()
    assert intent.excluded_symbols == ("BTC",)

    generated = yaml.safe_load(demo_server.SELECTION_EXAMPLE_YAML)
    monkeypatch.setattr(
        demo_server, "call_llm", lambda *_, **__: yaml.safe_dump(generated, sort_keys=False)
    )
    rejected = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert rejected["valid"] is False
    assert any("exclude_symbols" in str(error) for error in rejected["errors"])

    generated["selection"]["universe"].insert(
        0,
        {"block": "universe.exclude_symbols", "params": {"symbols": ["BTCUSDT"]}},
    )
    monkeypatch.setattr(
        demo_server, "call_llm", lambda *_, **__: yaml.safe_dump(generated, sort_keys=False)
    )
    accepted = demo_server.convert_nl("http://llm/v1", "", "model", request)
    assert accepted["valid"] is True, accepted


@pytest.mark.parametrize(
    "phrase",
    [
        "排除 BTC / ETH / SOL / XRP",
        "排除 BTC、ETH、SOL、XRP",
        "exclude BTC, ETH, SOL and XRP",
    ],
)
def test_symbol_exclusion_propagates_across_a_joined_list(demo_server, phrase):
    intent = demo_server.classify_request(phrase)

    assert intent.included_symbols == ()
    assert intent.excluded_symbols == ("BTC", "ETH", "SOL", "XRP")


def test_symbol_exclusion_stops_at_a_new_selection_clause(demo_server):
    intent = demo_server.classify_request(
        "排除 BTC / ETH，選 SOL 與 XRP 中成交量最大的兩個幣"
    )

    assert intent.excluded_symbols == ("BTC", "ETH")
    assert intent.included_symbols == ("SOL", "XRP")


def test_requested_base_assets_allow_matching_usdt_pairs(demo_server, monkeypatch):
    generated = yaml.safe_load(demo_server.SELECTION_EXAMPLE_YAML)
    generated["selection"]["universe"].insert(
        0,
        {
            "block": "universe.only_symbols",
            "params": {"symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]},
        },
    )
    generated["selection"]["top_k"] = 2
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(generated, sort_keys=False),
    )

    result = demo_server.convert_nl(
        "http://llm/v1", "", "model",
        "幫我從 BTC、ETH、SOL 中選新聞最熱門的兩個幣",
    )

    assert result["valid"] is True, result


def test_malformed_with_reports_validation_error_instead_of_crashing(
    demo_server, monkeypatch,
):
    generated = yaml.safe_load(demo_server.SELECTION_EXAMPLE_YAML)
    generated["selection"]["universe"][1]["with"] = 123
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: yaml.safe_dump(generated, sort_keys=False),
    )

    result = demo_server.convert_nl(
        "http://llm/v1", "", "model", "依新聞提及量選五個幣"
    )

    assert result["ok"] is True
    assert result["valid"] is False
    assert result["status"] == "rejected"
    assert any("dry-run" in str(error) or "with" in str(error) for error in result["errors"])


@pytest.mark.parametrize(
    ("phrase", "expected_count"),
    [
        ("Select top ten coins by news mentions", 10),
        ("選十五個新聞熱門的幣", 15),
        ("選二十個新聞熱門的幣", 20),
    ],
)
def test_written_candidate_counts_are_enforced(
    demo_server, monkeypatch, phrase, expected_count,
):
    monkeypatch.setattr(
        demo_server, "call_llm", lambda *_, **__: demo_server.SELECTION_EXAMPLE_YAML
    )

    result = demo_server.convert_nl("http://llm/v1", "", "model", phrase)

    assert result["intent"]["requested_count"] == expected_count
    assert result["valid"] is False
    assert any(str(expected_count) in str(error) for error in result["errors"])


def test_convert_http_route_delegates_to_the_validating_converter(
    demo_server, monkeypatch,
):
    expected = {
        "ok": True,
        "yaml": "selection: {}",
        "valid": False,
        "strategy_kind": "selection",
        "errors": ["probe"],
        "warnings": [],
    }
    seen = {}

    def fake_convert(api_base, api_key, model, nl):
        seen["args"] = (api_base, api_key, model, nl)
        return expected

    monkeypatch.setattr(demo_server, "convert_nl", fake_convert)
    monkeypatch.setenv(demo_server.DEMO_LLM_API_BASE_ENV, "https://server-owned/v1")
    monkeypatch.setenv(demo_server.DEMO_LLM_API_KEY_ENV, "server-owned-key")
    monkeypatch.setenv(demo_server.DEMO_LLM_MODEL_ENV, "server-owned-model")
    monkeypatch.setenv(demo_server.DEMO_LLM_TRUSTED_HOSTS_ENV, "server-owned")
    handler = object.__new__(demo_server.Handler)
    handler.path = "/api/convert"
    handler._read_json = lambda: {
        # These browser values are deliberately hostile.  The handler must not
        # turn the demo into an endpoint/key forwarding proxy.
        "api_base": "http://169.254.169.254/latest/meta-data",
        "api_key": "browser-controlled-key",
        "model": "browser-controlled-model",
        "trusted_hosts": "browser-controlled",
        "nl": "挑新聞熱門幣",
    }
    sent = {}
    handler._send = lambda code, payload, ctype="application/json": sent.update(
        code=code, payload=payload, ctype=ctype
    )

    handler.do_POST()

    assert seen["args"] == (
        "https://server-owned/v1", "server-owned-key", "server-owned-model",
        "挑新聞熱門幣",
    )
    assert sent["code"] == 200
    assert sent["ctype"] == "application/json"
    assert sent["payload"]["yaml"] == "selection: {}"
    assert sent["payload"]["runtime_verified"] is False
    assert sent["payload"]["semantic_verified"] is False
    assert sent["payload"]["promotion_eligible"] is False
    assert sent["payload"]["verification_state"] == "draft_static_rejected"


def test_convert_http_route_requires_server_owned_llm_settings(
    demo_server, monkeypatch,
):
    monkeypatch.delenv(demo_server.DEMO_LLM_API_BASE_ENV, raising=False)
    monkeypatch.delenv(demo_server.DEMO_LLM_API_KEY_ENV, raising=False)
    monkeypatch.delenv(demo_server.DEMO_LLM_MODEL_ENV, raising=False)
    monkeypatch.delenv(demo_server.DEMO_LLM_TRUSTED_HOSTS_ENV, raising=False)
    monkeypatch.setattr(
        demo_server,
        "convert_nl",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("must not use browser fallback configuration")
        ),
    )
    handler = object.__new__(demo_server.Handler)
    handler.path = "/api/convert"
    handler._read_json = lambda: {
        "api_base": "http://browser-controlled/v1",
        "api_key": "browser-controlled-key",
        "model": "browser-controlled-model",
        "nl": "挑新聞熱門幣",
    }
    sent = {}
    handler._send = lambda code, payload, ctype="application/json": sent.update(
        code=code, payload=payload, ctype=ctype
    )

    handler.do_POST()

    assert sent == {
        "code": 503,
        "payload": {"ok": False, "error": "demo_llm_not_configured"},
        "ctype": "application/json",
    }


def test_convert_http_route_requires_server_owned_trusted_hosts(
    demo_server, monkeypatch,
):
    monkeypatch.setenv(demo_server.DEMO_LLM_API_BASE_ENV, "https://server-owned/v1")
    monkeypatch.setenv(demo_server.DEMO_LLM_MODEL_ENV, "server-owned-model")
    monkeypatch.delenv(demo_server.DEMO_LLM_TRUSTED_HOSTS_ENV, raising=False)
    monkeypatch.setattr(
        demo_server,
        "convert_nl",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("missing trusted hosts must stop before conversion")
        ),
    )
    handler = object.__new__(demo_server.Handler)
    handler.path = "/api/convert"
    handler._read_json = lambda: {
        "trusted_hosts": "browser-controlled",
        "nl": "挑新聞熱門幣",
    }
    sent = {}
    handler._send = lambda code, payload, ctype="application/json": sent.update(
        code=code, payload=payload, ctype=ctype
    )

    handler.do_POST()

    assert sent == {
        "code": 503,
        "payload": {"ok": False, "error": "demo_llm_not_configured"},
        "ctype": "application/json",
    }


def test_mocked_llm_and_square_bundle_reach_v2_selection(
    demo_server, monkeypatch,
):
    """Network-free golden path with real YAML compiler and Blocks runtime."""
    generated = demo_server.SELECTION_EXAMPLE_YAML
    monkeypatch.setattr(demo_server, "call_llm", lambda *_, **__: generated)
    converted = demo_server.convert_nl(
        "http://llm/v1", "", "model", "幫我挑選最近新聞常提到的幣種"
    )
    assert converted["valid"] and converted["strategy_kind"] == "selection"

    bundle = json.loads(SAMPLE_BUNDLE.read_text(encoding="utf-8"))
    from cyqnt_trd.standard_bot.data import live_snapshot

    requested = {}

    def fake_live_snapshot(**kwargs):
        requested.update(kwargs)
        return None, copy.deepcopy(bundle)

    monkeypatch.setattr(live_snapshot, "build_live_snapshot", fake_live_snapshot)
    result = demo_server.run_selection(converted["yaml"])

    assert set(requested["sections"]) == {"news", "universe"}
    assert result["ok"] is True, result
    assert result["batch"]["schema"] == "cyqnt.signal-batch/v1"
    assert result["batch"]["signal_count"] == 1
    signal = result["signal"]
    assert signal["schema"] == "cyqnt.signal/v2"
    assert signal["kind"] == "selection"
    assert signal["candidates"]
    assert len(signal) == 42


def test_terra_bullish_selection_runs_through_real_blocks(
    demo_server, monkeypatch,
):
    monkeypatch.setattr(
        demo_server, "call_llm", lambda *_, **__: TERRA_BULLISH_SELECTION_YAML
    )
    converted = demo_server.convert_nl(
        "http://llm/v1", "", "terra", CHINESE_DISCOVERY_REQUEST
    )
    assert converted["valid"] is True, converted
    assert converted["strategy_kind"] == "selection"

    bundle = json.loads(SAMPLE_BUNDLE.read_text(encoding="utf-8"))
    from cyqnt_trd.standard_bot.data import live_snapshot

    monkeypatch.setattr(
        live_snapshot,
        "build_live_snapshot",
        lambda **_: (None, copy.deepcopy(bundle)),
    )
    result = demo_server.run_selection(converted["yaml"])

    assert result["ok"] is True, result
    assert result["signal"]["kind"] == "selection"
    assert len(result["signal"]) == 42
    assert [item["symbol"] for item in result["candidates"]] == ["BTCUSDT"]


def test_news_mentions_counterfactually_control_candidate_order(demo_server):
    from cyqnt_trd.standard_bot.yaml_pipeline.bundle_runner import run_bundle

    spec = yaml.safe_load(demo_server.SELECTION_EXAMPLE_YAML)
    original = json.loads(SAMPLE_BUNDLE.read_text(encoding="utf-8"))
    flipped = copy.deepcopy(original)
    rows = flipped["frames"]["ticker_rank"]["rows"]
    rows[0]["mention_count"], rows[1]["mention_count"] = (
        rows[1]["mention_count"], rows[0]["mention_count"]
    )

    before = run_bundle(spec, original)["signals"][0]["candidates"]
    after = run_bundle(spec, flipped)["signals"][0]["candidates"]

    assert [item["symbol"] for item in before] == ["BTCUSDT", "ETHUSDT"]
    assert [item["symbol"] for item in after] == ["ETHUSDT", "BTCUSDT"]
    assert all(item["direction"] == "neutral" for item in before + after)


def test_browser_conversion_is_server_owned_draft_only_and_never_persists_keys():
    """Pin the last browser-side hop that Python E2E tests cannot observe."""
    html = INDEX_PATH.read_text(encoding="utf-8")
    convert_js = html.split("async function convert(){", 1)[1].split(
        "async function backtest(){", 1
    )[0]

    assert 'd.strategy_kind === "selection"' in convert_js
    assert 'd.strategy_kind === "ambiguous"' in convert_js
    assert '(isSelection ? $("selYaml") : $("yaml")).value = d.yaml' in convert_js
    assert convert_js.index('d.strategy_kind === "ambiguous"') < convert_js.index(
        '(isSelection ? $("selYaml") : $("yaml")).value = d.yaml'
    )
    assert "await selection()" not in convert_js
    assert "selectionSucceeded" not in convert_js
    assert "YAML 草稿" in convert_js
    assert "尚未在 frozen/live bundle 執行" in convert_js
    assert "semantic 驗證或 promotion" in convert_js
    assert 'body:JSON.stringify({nl})' in convert_js
    assert "api_base" not in convert_js
    assert "api_key" not in convert_js
    assert 'id="apiBase"' not in html
    assert 'id="apiKey"' not in html
    assert '$("apiBase")' not in html
    assert '$("apiKey")' not in html
    valid_branch = convert_js.index("if(d.valid){")
    invalid_yaml_badge = convert_js.index(
        'setBadge($("convBadge"),false', valid_branch
    )
    assert valid_branch < invalid_yaml_badge

    persistence_js = html.split("// --- erase legacy", 1)[1].split("// --- schema ---", 1)[0]
    assert '"demo_apiKey"' in persistence_js
    assert '"demo_apiBase"' in persistence_js
    assert '"demo_model"' in persistence_js
    assert 'localStorage.setItem("demo_apiKey"' not in html
    assert 'localStorage.getItem("demo_apiKey"' not in html
