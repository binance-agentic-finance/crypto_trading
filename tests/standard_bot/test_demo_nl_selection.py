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
SAMPLE_BUNDLE = (
    ROOT / "docs" / "standard_bot_io" / "samples" / "input_bundle_example.json"
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
    assert any("兩個需求" in str(error) for error in result["errors"])


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

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "choices": [{"message": {"content": demo_server.SELECTION_EXAMPLE_YAML}}]
            }

    def fake_post(url, headers, json, timeout):
        captured.update(url=url, headers=headers, body=json, timeout=timeout)
        return Response()

    monkeypatch.setattr(demo_server.requests, "post", fake_post)
    demo_server.call_llm("http://llm/v1", "", "model", phrase)

    system = captured["body"]["messages"][0]["content"]
    assert "selection:" in system
    assert "universe.augment_with_news" in system
    assert captured["body"]["messages"][1]["content"] == phrase


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
    generated["selection"]["score"] = "quote_volume"
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
    ("phrase", "source"),
    [
        ("Select 5 coins with highest open interest", "open_interest"),
        ("幫我選五個 funding rate 最負的幣", "funding"),
        ("幫我選五個漲幅最大的幣", "price_change"),
    ],
)
def test_unsupported_cross_section_source_is_not_replaced_with_news(
    demo_server, monkeypatch, phrase, source,
):
    monkeypatch.setattr(
        demo_server,
        "call_llm",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("unsupported selection source must stop before LLM")
        ),
    )

    result = demo_server.convert_nl("http://llm/v1", "", "model", phrase)

    assert result["status"] == "unsupported"
    assert result["valid"] is False
    assert result["yaml"] == ""
    assert source in result["intent"]["sources"]
    assert any(source in str(error) for error in result["errors"])


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
    assert generated["selection"]["score"] == "fundingRatePct"
    assert {
        step["block"]: step.get("with")
        for step in generated["selection"]["universe"]
    }["universe.augment_with_funding"] == ["funding"]


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
    assert "score: quote_volume" in demo_server.build_system_prompt(
        "selection", demo_server.classify_request(request)
    )


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
    assert any("quote_volume" in str(error) for error in rejected["errors"])

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
            "input": "quote_volume",
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
    mention_spec["selection"]["score"] = "quote_volume"
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
    handler = object.__new__(demo_server.Handler)
    handler.path = "/api/convert"
    handler._read_json = lambda: {
        "api_base": "http://llm/v1",
        "api_key": "key",
        "model": "model",
        "nl": "挑新聞熱門幣",
    }
    sent = {}
    handler._send = lambda code, payload, ctype="application/json": sent.update(
        code=code, payload=payload, ctype=ctype
    )

    handler.do_POST()

    assert seen["args"] == (
        "http://llm/v1", "key", "model", "挑新聞熱門幣"
    )
    assert sent == {"code": 200, "payload": expected, "ctype": "application/json"}


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


def test_browser_routes_only_valid_selection_and_never_persists_the_api_key():
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
    assert convert_js.count("await selection()") == 1
    valid_branch = convert_js.index("if(d.valid){")
    invalid_yaml_badge = convert_js.index(
        'setBadge($("convBadge"),false', valid_branch
    )
    assert (
        valid_branch
        < convert_js.index("await selection()")
        < invalid_yaml_badge
    ), "selection execution must remain inside the valid-YAML branch"

    persistence_js = html.split("// --- persist", 1)[1].split("// --- schema ---", 1)[0]
    assert 'localStorage.removeItem("demo_apiKey")' in persistence_js
    assert 'localStorage.setItem("demo_apiKey"' not in html
    assert 'localStorage.getItem("demo_apiKey"' not in html
