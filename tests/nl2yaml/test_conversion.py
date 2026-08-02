"""Contract tests for the reusable in-memory NL -> YAML conversion service."""

from __future__ import annotations

import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from tools.nl2yaml.conversion import (
    ConversionStatus,
    LLMClientError,
    MAX_RESPONSE_BYTES,
    OpenAICompatibleHTTPClient,
    convert_and_run,
)


ROOT = Path(__file__).parents[2]
BUNDLE_PATH = ROOT / "tests" / "standard_bot" / "fixtures" / "universe_cross_section.json"

REQUEST = "幫我挑選最近新聞常提到的五個幣種"
SYSTEM_PROMPT = "caller-owned conversion instructions"
SELECTION_YAML = """\
spec_version: "1.0"
target: standard_bot
strategy:
  id: conversion_news_selector
  description: "rank current news mentions"
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
  score: news_mention_count
  top_k: 5
  min_score: 1.0
  dedupe_by: base_asset
"""

# These are the private-review layer's structured, non-verbatim conditions for
# REQUEST.  Supplying them here makes the test prove the complete G1e path,
# rather than treating a syntactically valid runtime execution as a ready answer.
REVIEWED_SELECTION_CONDITIONS = [
    {"id": "news_rank", "subject": "social_mentions", "scope": "cross_section",
     "operator": "rank", "value": None},
    {"id": "exact_five", "subject": "basket_size", "scope": "cross_section",
     "operator": "exact_top_k", "value": 5, "quantified": True},
]


class StaticClient:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: List[Dict[str, str]] = []

    def complete(self, *, system_prompt: str, user_request: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_request": user_request})
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


@pytest.fixture
def frozen_bundle() -> Dict[str, Any]:
    return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def fake_openai_server():
    received: List[Dict[str, Any]] = []
    state: Dict[str, Any] = {
        "status": 200,
        "headers": {},
        "response_body": {
            "choices": [{"message": {"content": "```yaml\n" + SELECTION_YAML + "\n```"}}],
        },
        "raw_payload": None,
        "content_length": None,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - HTTP handler API
            length = int(self.headers.get("Content-Length", "0"))
            received.append({
                "path": self.path,
                "headers": dict(self.headers),
                "body": json.loads(self.rfile.read(length).decode("utf-8")),
            })
            raw_payload = state["raw_payload"]
            payload = (bytes(raw_payload) if raw_payload is not None
                       else json.dumps(state["response_body"]).encode("utf-8"))
            self.send_response(int(state["status"]))
            self.send_header("Content-Type", "application/json")
            for key, value in dict(state["headers"]).items():
                self.send_header(str(key), str(value))
            self.send_header(
                "Content-Length",
                str(state["content_length"] if state["content_length"] is not None
                    else len(payload)),
            )
            self.end_headers()
            if payload:
                self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d/v1" % server.server_port, received, state
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_openai_http_adapter_real_local_contract_and_canonical_runtime(
    fake_openai_server, frozen_bundle,
):
    base, received, _state = fake_openai_server
    client = OpenAICompatibleHTTPClient(
        api_base=base,
        api_key="test-only-key",
        model="local-test-model",
        allow_external=True,
        trusted_hosts=("127.0.0.1",),
        allow_loopback_http_for_tests=True,
        timeout_seconds=5,
    )

    result = convert_and_run(
        request=REQUEST,
        system_prompt=SYSTEM_PROMPT,
        llm_client=client,
        frozen_bundle=frozen_bundle,
        reviewed_conditions=REVIEWED_SELECTION_CONDITIONS,
    )

    assert result.status is ConversionStatus.RUNTIME_PASSED
    assert result.generated_strategy_kind == "selection"
    assert result.signal_count == 1
    assert result.yaml_text == SELECTION_YAML.strip()
    assert result.bundle_sha256 and len(result.bundle_sha256) == 64
    assert result.signal_batch_sha256 and len(result.signal_batch_sha256) == 64
    assert len(received) == 1
    assert received[0]["path"] == "/v1/chat/completions"
    assert received[0]["headers"]["Authorization"] == "Bearer test-only-key"
    assert received[0]["body"]["model"] == "local-test-model"
    assert received[0]["body"]["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": REQUEST},
    ]
    summary = result.safe_summary()
    assert REQUEST not in json.dumps(summary, ensure_ascii=False)
    assert "conversion_news_selector" not in json.dumps(summary, ensure_ascii=False)
    assert "test-only-key" not in json.dumps(summary, ensure_ascii=False)
    assert "yaml_text" not in summary
    assert "request_sha256" not in summary
    assert REQUEST not in repr(result)
    assert result.request_sha256 not in repr(result)


def test_external_adapter_is_default_deny_before_any_http_request(
    fake_openai_server, frozen_bundle,
):
    base, received, _state = fake_openai_server
    client = OpenAICompatibleHTTPClient(
        api_base=base,
        api_key="test-only-key",
        model="local-test-model",
        trusted_hosts=("127.0.0.1",),
        allow_loopback_http_for_tests=True,
    )

    result = convert_and_run(
        request=REQUEST,
        system_prompt=SYSTEM_PROMPT,
        llm_client=client,
        frozen_bundle=frozen_bundle,
    )

    assert result.status is ConversionStatus.LLM_FAILED
    assert result.reason_codes == ("external_llm_disabled",)
    assert received == []


def test_external_opt_in_cannot_be_enabled_by_a_truthy_config_string():
    with pytest.raises(ValueError, match="explicit bool"):
        OpenAICompatibleHTTPClient(
            api_base="http://127.0.0.1:1/v1",
            api_key="test-only-key",
            model="local-test-model",
            allow_external="false",  # type: ignore[arg-type]
            allow_loopback_http_for_tests=True,
        )


@pytest.mark.parametrize(
    ("api_base", "kwargs", "error"),
    [
        ("https://provider.example/v1", {}, "trusted_hosts_required"),
        (
            "https://provider.example/v1",
            {"trusted_hosts": ("another.example",)},
            "api_base_host_not_trusted",
        ),
        (
            "http://provider.example/v1",
            {"trusted_hosts": ("provider.example",)},
            "api_base_https_required",
        ),
        (
            "https://user:pass@provider.example/v1",
            {"trusted_hosts": ("provider.example",)},
            "api_base_credentials_not_allowed",
        ),
        (
            "https://provider.example/v1?target=elsewhere",
            {"trusted_hosts": ("provider.example",)},
            "api_base_query_or_fragment_not_allowed",
        ),
        (
            "https://provider.example/v1#fragment",
            {"trusted_hosts": ("provider.example",)},
            "api_base_query_or_fragment_not_allowed",
        ),
        (
            "https://10.0.0.1/v1",
            {"trusted_hosts": ("provider.example",)},
            "api_base_ip_not_allowed",
        ),
        (
            "https://2130706433/v1",
            {"trusted_hosts": ("provider.example",)},
            "api_base_ip_not_allowed",
        ),
        (
            "http://127.0.0.1:1/v1",
            {"trusted_hosts": ("127.0.0.1",)},
            "api_base_loopback_test_seam_required",
        ),
    ],
)
def test_external_adapter_rejects_untrusted_or_ssrf_endpoint_shapes(
    api_base, kwargs, error,
):
    with pytest.raises(ValueError, match=error):
        OpenAICompatibleHTTPClient(
            api_base=api_base,
            api_key="test-only-key",
            model="local-test-model",
            allow_external=True,
            **kwargs,
        )


def test_external_adapter_rejects_private_dns_target_before_request():
    client = OpenAICompatibleHTTPClient(
        api_base="https://localhost/v1",
        api_key="test-only-key",
        model="local-test-model",
        allow_external=True,
        trusted_hosts=("localhost",),
    )

    with pytest.raises(LLMClientError, match="private_network_target"):
        client.complete(system_prompt=SYSTEM_PROMPT, user_request=REQUEST)


def test_external_adapter_blocks_redirects_and_bounds_response(
    fake_openai_server,
):
    base, received, state = fake_openai_server
    client = OpenAICompatibleHTTPClient(
        api_base=base,
        api_key="test-only-key",
        model="local-test-model",
        allow_external=True,
        trusted_hosts=("127.0.0.1",),
        allow_loopback_http_for_tests=True,
        timeout_seconds=5,
    )

    state["status"] = 302
    state["headers"] = {"Location": "https://untrusted.example/other"}
    with pytest.raises(LLMClientError, match="redirect_blocked"):
        client.complete(system_prompt=SYSTEM_PROMPT, user_request=REQUEST)

    state["status"] = 200
    state["headers"] = {}
    state["raw_payload"] = b""
    state["content_length"] = MAX_RESPONSE_BYTES + 1
    with pytest.raises(LLMClientError, match="response_too_large"):
        client.complete(system_prompt=SYSTEM_PROMPT, user_request=REQUEST)

    assert len(received) == 2


def test_malformed_yaml_is_rejected_before_canonical_runtime(frozen_bundle):
    client = StaticClient("selection: [")

    result = convert_and_run(
        request=REQUEST,
        system_prompt=SYSTEM_PROMPT,
        llm_client=client,
        frozen_bundle=frozen_bundle,
    )

    assert result.status is ConversionStatus.YAML_REJECTED
    assert result.reason_codes == ("yaml_parse_error",)
    assert result.signal_batch_sha256 is None
    assert len(client.calls) == 1


def test_runtime_error_is_distinct_after_yaml_and_intent_pass(frozen_bundle):
    incomplete = copy.deepcopy(frozen_bundle)
    incomplete["frames"].pop("ticker_rank")
    client = StaticClient(SELECTION_YAML)

    result = convert_and_run(
        request=REQUEST,
        system_prompt=SYSTEM_PROMPT,
        llm_client=client,
        frozen_bundle=incomplete,
    )

    assert result.status is ConversionStatus.RUNTIME_FAILED
    assert result.reason_codes == ("canonical_runtime_error",)
    assert result.yaml_sha256 is not None
    assert result.signal_batch_sha256 is None
    assert len(client.calls) == 1


def test_runtime_valid_unreviewed_is_not_a_ready_result(frozen_bundle):
    result = convert_and_run(
        request=REQUEST,
        system_prompt=SYSTEM_PROMPT,
        llm_client=StaticClient(SELECTION_YAML),
        frozen_bundle=frozen_bundle,
    )

    assert result.status is ConversionStatus.RUNTIME_VALID_UNREVIEWED
    assert result.reason_codes == ("reviewed_conditions_required",)
    assert result.signal_batch_sha256 is not None
    assert result.signal_count == 1


@pytest.mark.parametrize("field", ["snapshot_id", "decision_time"])
def test_missing_frozen_provenance_cannot_be_reported_as_runtime_passed(
    frozen_bundle, field,
):
    """The conversion shortcut must share the evaluator's frozen-input gate."""
    untraceable = copy.deepcopy(frozen_bundle)
    untraceable.pop(field)
    client = StaticClient(AssertionError("LLM must not run on an untraceable bundle"))

    result = convert_and_run(
        request=REQUEST,
        system_prompt=SYSTEM_PROMPT,
        llm_client=client,
        frozen_bundle=untraceable,
        reviewed_conditions=REVIEWED_SELECTION_CONDITIONS,
    )

    assert result.status is ConversionStatus.RUNTIME_FAILED
    assert result.reason_codes == ("invalid_frozen_bundle",)
    assert client.calls == []


def test_reviewed_conditions_must_pass_g1e_before_runtime_passed(frozen_bundle):
    result = convert_and_run(
        request=REQUEST,
        system_prompt=SYSTEM_PROMPT,
        llm_client=StaticClient(SELECTION_YAML),
        frozen_bundle=frozen_bundle,
        reviewed_conditions=[
            {"id": "wrong_count", "subject": "basket_size",
             "scope": "cross_section", "operator": "exact_top_k",
             "value": 4, "quantified": True},
        ],
    )

    assert result.status is ConversionStatus.SEMANTIC_FAILED
    assert result.reason_codes == ("semantic_condition_violated",)
    assert result.signal_batch_sha256 is not None
    # Neither the structured field nor its value appears in the safe result.
    assert "wrong_count" not in json.dumps(result.safe_summary(), ensure_ascii=False)


def test_malformed_reviewed_conditions_fail_before_calling_llm(frozen_bundle):
    client = StaticClient(AssertionError("must not be called"))
    result = convert_and_run(
        request=REQUEST,
        system_prompt=SYSTEM_PROMPT,
        llm_client=client,
        frozen_bundle=frozen_bundle,
        reviewed_conditions=[
            {"id": "leaky", "subject": "basket_size", "operator": "top_k",
             "value": 5, "quote": REQUEST},
        ],
    )

    assert result.status is ConversionStatus.SEMANTIC_FAILED
    assert result.reason_codes == ("invalid_reviewed_conditions",)
    assert client.calls == []


def test_explicit_sentiment_side_mapping_rejects_a_reversed_generated_selector(frozen_bundle):
    request = (
        "Select 5 coins: go long on positive social sentiment and short on "
        "negative social sentiment"
    )
    correct = yaml.safe_load(SELECTION_YAML)
    correct["selection"]["score"] = "news_bull_ratio"
    # ``news_bull_ratio`` is bounded by one, unlike the original mention count;
    # keep this semantic fixture executable rather than filtering every row
    # before G1e can assess the side mapping.
    correct["selection"]["min_score"] = 0.0
    correct["selection"]["long_when"] = {
        "cond": "conditions.value_above", "args": ["news_bull_ratio", 0.55],
    }
    correct["selection"]["short_when"] = {
        "cond": "conditions.value_below", "args": ["news_bull_ratio", 0.45],
    }
    correct_yaml = yaml.safe_dump(correct, sort_keys=False)
    accepted = convert_and_run(
        request=request,
        system_prompt=SYSTEM_PROMPT,
        llm_client=StaticClient(correct_yaml),
        frozen_bundle=frozen_bundle,
        reviewed_conditions=[
            {"id": "sentiment_rank", "subject": "social_sentiment",
             "scope": "cross_section", "operator": "rank", "value": None},
            {"id": "exact_five", "subject": "basket_size",
             "scope": "cross_section", "operator": "exact_top_k",
             "value": 5, "quantified": True},
        ],
    )
    assert accepted.status is ConversionStatus.RUNTIME_PASSED

    reversed_sides = copy.deepcopy(correct)
    reversed_sides["selection"]["long_when"] = {
        "cond": "conditions.value_below", "args": ["news_bull_ratio", 0.45],
    }
    reversed_sides["selection"]["short_when"] = {
        "cond": "conditions.value_above", "args": ["news_bull_ratio", 0.55],
    }
    rejected = convert_and_run(
        request=request,
        system_prompt=SYSTEM_PROMPT,
        llm_client=StaticClient(yaml.safe_dump(reversed_sides, sort_keys=False)),
        frozen_bundle=frozen_bundle,
    )
    assert rejected.status is ConversionStatus.YAML_REJECTED
    assert rejected.reason_codes == ("intent_mismatch",)
    assert rejected.signal_batch_sha256 is None


def test_ambiguous_request_fails_closed_without_calling_llm(frozen_bundle):
    client = StaticClient(AssertionError("must not be called"))

    result = convert_and_run(
        request="幫我做一個厲害的東西",
        system_prompt=SYSTEM_PROMPT,
        llm_client=client,
        frozen_bundle=frozen_bundle,
    )

    assert result.status is ConversionStatus.NEEDS_CLARIFICATION
    assert result.reason_codes == ("ambiguous_intent",)
    assert client.calls == []


def test_known_gap_is_unsupported_without_calling_llm(frozen_bundle):
    client = StaticClient(AssertionError("must not be called"))

    result = convert_and_run(
        request="選出全市場爆倉最多的五個幣",
        system_prompt=SYSTEM_PROMPT,
        llm_client=client,
        frozen_bundle=frozen_bundle,
    )

    assert result.status is ConversionStatus.UNSUPPORTED
    assert result.reason_codes == ("intent_not_expressible",)
    assert client.calls == []


def test_underspecified_selection_fails_closed_without_calling_llm(frozen_bundle):
    client = StaticClient(AssertionError("must not be called"))

    result = convert_and_run(
        request="幫我選五個幣",
        system_prompt=SYSTEM_PROMPT,
        llm_client=client,
        frozen_bundle=frozen_bundle,
    )

    assert result.status is ConversionStatus.NEEDS_CLARIFICATION
    assert result.reason_codes == ("missing_selection_criterion",)
    assert client.calls == []


@pytest.mark.parametrize(
    ("user_request", "reason_code"),
    [
        (
            "BTCUSDT EMA12 上穿 EMA26 買進，EMA12 下穿 EMA26 賣出，"
            "停損 2%，停利 4%，倉位 5%",
            "ambiguous_sell_action",
        ),
        ("BTC、ETH 做多，SOL 做空，停損 2%", "multiple_trade_symbols"),
        ("BTC 現貨 EMA12 下穿 EMA26 做空", "spot_short_not_supported"),
        (
            "BTC EMA10上穿EMA30且RSI14低於30或高於70買進",
            "mixed_boolean_conditions",
        ),
        ("BTC EMA10 上穿 EMA30 買進，使用 ATR 追蹤停損", "incomplete_atr_trailing_exit"),
    ],
)
def test_unrepresentable_trade_preflights_fail_closed_before_llm(
    frozen_bundle, user_request, reason_code,
):
    client = StaticClient(AssertionError("must not be called"))

    result = convert_and_run(
        request=user_request,
        system_prompt=SYSTEM_PROMPT,
        llm_client=client,
        frozen_bundle=frozen_bundle,
    )

    assert result.status is ConversionStatus.NEEDS_CLARIFICATION
    assert result.reason_codes == (reason_code,)
    assert client.calls == []


def test_injected_client_error_cannot_copy_request_into_safe_summary(frozen_bundle):
    client = StaticClient(LLMClientError(REQUEST))

    result = convert_and_run(
        request=REQUEST,
        system_prompt=SYSTEM_PROMPT,
        llm_client=client,
        frozen_bundle=frozen_bundle,
    )

    assert result.status is ConversionStatus.LLM_FAILED
    assert result.reason_codes == ("client_error",)
    assert REQUEST not in json.dumps(result.safe_summary(), ensure_ascii=False)
