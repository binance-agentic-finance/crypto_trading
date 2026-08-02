"""Reusable, bounded natural-language -> YAML -> canonical-signal conversion.

This module is deliberately *not* an HTTP server and never writes an artifact.
The caller owns transport, retention, and any private-review workflow.  In
particular, raw requests, API keys, generated YAML and input bundles remain in
memory for the duration of :func:`convert_and_run`; the result exposes only
hashes through :meth:`ConversionResult.safe_summary`.

There are two deliberately separate seams:

* a caller injects an :class:`LLMClient` and supplies its own system prompt;
* a caller supplies an already-frozen ``cyqnt.input/v1`` bundle, which this
  module executes through the canonical ``bundle_runner.run_bundle`` path.

``OpenAICompatibleHTTPClient`` is the one optional network adapter.  It is
default-deny: merely constructing it cannot send a request.  A caller must
explicitly opt in with ``allow_external=True`` before its ``complete`` method
will construct an HTTP request.  It never logs or persists either the request
or the API key.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
import socket
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

import requests
import yaml

from cyqnt_trd.standard_bot.yaml_pipeline import validate_spec
from cyqnt_trd.standard_bot.yaml_pipeline.intent import (
    UNSUPPORTED_SELECTION_SOURCES,
    classify_request,
    generated_strategy_kind,
    reconcile_intent,
)

from . import evaluate
from .capability import Condition, normalize_condition
from .gates import run_gates, strip_code_fence


__all__ = [
    "ConversionResult",
    "ConversionStatus",
    "ExternalLLMDisabled",
    "LLMClient",
    "LLMClientError",
    "MAX_RESPONSE_BYTES",
    "OpenAICompatibleHTTPClient",
    "convert_and_run",
]


MAX_REQUEST_BYTES = 64 * 1024
MAX_SYSTEM_PROMPT_BYTES = 1024 * 1024
# A conversion answer is a YAML program, not an archive.  Bound its decoded
# HTTP body before JSON parsing so a compromised or misconfigured provider
# cannot make the server retain an arbitrary response in memory.
MAX_RESPONSE_BYTES = 1024 * 1024
_SAFE_REASON_CODE = re.compile(r"^[a-z0-9_]{1,64}$")
_DNS_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_LOOPBACK_TEST_HOST = "127.0.0.1"


class ConversionStatus(str, Enum):
    """Stable result states for a caller's conversion state machine."""

    NEEDS_CLARIFICATION = "needs_clarification"
    UNSUPPORTED = "unsupported"
    LLM_FAILED = "llm_failed"
    YAML_REJECTED = "yaml_rejected"
    RUNTIME_FAILED = "runtime_failed"
    # G1a--G1d ran successfully, but no reviewed, structured request
    # conditions were available for G1e.  This is deliberately *not* a ready
    # answer: callers may show the YAML, but must not auto-promote or execute it.
    RUNTIME_VALID_UNREVIEWED = "runtime_valid_unreviewed"
    # G1e ran on the same frozen bundle and found a semantic mismatch,
    # unresolved condition, undisclosed assumption, or data-plane shortfall.
    SEMANTIC_FAILED = "semantic_failed"
    RUNTIME_PASSED = "runtime_passed"


class LLMClient(Protocol):
    """Minimal injected client boundary; implementations must return YAML text."""

    def complete(self, *, system_prompt: str, user_request: str) -> str:
        """Return the model's complete response without retaining either input."""


class LLMClientError(RuntimeError):
    """A privacy-safe LLM failure reason suitable for an in-memory result."""

    def __init__(self, code: str):
        self.code = str(code)
        super().__init__(self.code)


class ExternalLLMDisabled(LLMClientError):
    """Raised before a request body is built for an external endpoint."""

    def __init__(self) -> None:
        super().__init__("external_llm_disabled")


class OpenAICompatibleHTTPClient:
    """The sole OpenAI-compatible ``/chat/completions`` adapter.

    ``api_key`` deliberately has no public property and this class has no
    persistence or logging behaviour.  Provider-specific SDKs belong outside
    this conversion contract; callers that need one can implement
    :class:`LLMClient` themselves.

    Network use is deliberately narrower than "an arbitrary OpenAI-compatible
    URL".  A server-owned caller must name the exact DNS hosts it trusts in
    ``trusted_hosts`` before ``allow_external=True`` can send a request.  The
    transport is HTTPS-only.  The sole exception is an explicit
    ``127.0.0.1`` HTTP seam for deterministic integration tests; it still needs
    that address in ``trusted_hosts`` and cannot be selected by a hostname such
    as ``localhost``.  Do not feed either setting from a browser or user
    request.
    """

    __slots__ = (
        "_api_key",
        "_allow_external",
        "_allow_loopback_http_for_tests",
        "_endpoint",
        "_host",
        "_model",
        "_port",
        "_timeout",
        "_trusted_hosts",
    )

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        allow_external: bool = False,
        trusted_hosts: Sequence[str] = (),
        allow_loopback_http_for_tests: bool = False,
        timeout_seconds: float = 120.0,
    ) -> None:
        base = str(api_base or "").strip()
        if not base:
            raise ValueError("api_base is required")
        if not str(model or "").strip():
            raise ValueError("model is required")
        if not isinstance(allow_external, bool):
            raise ValueError("allow_external must be an explicit bool")
        if not isinstance(allow_loopback_http_for_tests, bool):
            raise ValueError("allow_loopback_http_for_tests must be an explicit bool")
        if not math.isfinite(float(timeout_seconds)) or float(timeout_seconds) <= 0:
            raise ValueError("timeout_seconds must be a positive finite number")
        endpoint, host, port, loopback_test_target = self._parse_api_base(
            base,
            allow_loopback_http_for_tests=allow_loopback_http_for_tests,
        )
        trusted = self._normalize_trusted_hosts(
            trusted_hosts,
            allow_loopback_http_for_tests=allow_loopback_http_for_tests,
        )
        if allow_external:
            if not trusted:
                raise ValueError("trusted_hosts_required")
            if host not in trusted:
                raise ValueError("api_base_host_not_trusted")
            if loopback_test_target and not allow_loopback_http_for_tests:
                # Kept for defence in depth even though _parse_api_base catches
                # it first.  A future parser refactor must not silently turn the
                # local fixture seam into a generic HTTP bypass.
                raise ValueError("api_base_loopback_test_seam_required")
        self._endpoint = endpoint
        self._host = host
        self._port = port
        self._api_key = str(api_key or "")
        self._model = str(model)
        self._allow_external = allow_external
        self._allow_loopback_http_for_tests = allow_loopback_http_for_tests
        self._timeout = float(timeout_seconds)
        self._trusted_hosts = trusted

    @staticmethod
    def _normalize_trusted_hosts(
        hosts: Sequence[str],
        *,
        allow_loopback_http_for_tests: bool,
    ) -> frozenset[str]:
        """Return a closed, exact host allowlist from server-owned config.

        Wildcards, URLs and IP literals are deliberately not accepted.  The one
        loopback fixture address is allowed only with its explicit test seam,
        so production configuration cannot accidentally approve an RFC1918
        target by spelling it as an allowlisted host.
        """
        if isinstance(hosts, (str, bytes)):
            raise ValueError("trusted_hosts_must_be_a_sequence")
        try:
            raw_hosts = tuple(hosts)
        except TypeError as exc:
            raise ValueError("trusted_hosts_must_be_a_sequence") from exc
        normalized = []
        for raw in raw_hosts:
            host = str(raw or "").strip().rstrip(".").lower()
            if host == _LOOPBACK_TEST_HOST and allow_loopback_http_for_tests:
                normalized.append(host)
                continue
            if not OpenAICompatibleHTTPClient._is_dns_name(host):
                raise ValueError("trusted_hosts_invalid")
            normalized.append(host)
        return frozenset(normalized)

    @staticmethod
    def _is_dns_name(host: str) -> bool:
        if not host or len(host) > 253 or ":" in host:
            return False
        if OpenAICompatibleHTTPClient._is_ip_literal(host):
            return False
        return all(_DNS_LABEL.fullmatch(label) is not None
                   for label in host.split("."))

    @staticmethod
    def _is_ip_literal(host: str) -> bool:
        """Recognize standard *and legacy numeric* address spellings.

        ``ipaddress`` intentionally rejects some historical IPv4 forms such as
        ``2130706433`` and ``127.1``.  HTTP stacks on common platforms still
        resolve those as addresses, so treating them as DNS labels would make a
        private-target rejection depend on the resolver.  ``inet_aton`` closes
        that parser differential before a request is constructed.
        """
        try:
            ipaddress.ip_address(host)
        except ValueError:
            try:
                socket.inet_aton(host)
            except OSError:
                return False
            return True
        return True

    @staticmethod
    def _parse_api_base(
        base: str,
        *,
        allow_loopback_http_for_tests: bool,
    ) -> Tuple[str, str, int, bool]:
        """Validate an endpoint without putting its value in an error message."""
        try:
            parsed = urlsplit(base)
            # Accessing port eagerly turns malformed ``:not-a-port`` input into
            # our stable config failure rather than a later transport surprise.
            port = parsed.port
        except ValueError as exc:
            raise ValueError("api_base_invalid") from exc
        if parsed.scheme not in ("https", "http"):
            raise ValueError("api_base_scheme_not_allowed")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("api_base_credentials_not_allowed")
        if parsed.query or parsed.fragment:
            raise ValueError("api_base_query_or_fragment_not_allowed")
        host = (parsed.hostname or "").rstrip(".").lower()
        if not host:
            raise ValueError("api_base_host_required")

        loopback_test_target = (
            parsed.scheme == "http" and host == _LOOPBACK_TEST_HOST
        )
        if loopback_test_target:
            if not allow_loopback_http_for_tests:
                raise ValueError("api_base_loopback_test_seam_required")
        else:
            # Literal addresses make the endpoint an SSRF primitive even when
            # they happen to be public today.  DNS names are allowlisted below;
            # resolved private addresses are rejected immediately before I/O.
            if OpenAICompatibleHTTPClient._is_ip_literal(host):
                raise ValueError("api_base_ip_not_allowed")
            if parsed.scheme != "https":
                raise ValueError("api_base_https_required")
            if not OpenAICompatibleHTTPClient._is_dns_name(host):
                raise ValueError("api_base_host_invalid")

        endpoint_path = parsed.path.rstrip("/")
        if not endpoint_path.endswith("/chat/completions"):
            endpoint_path = (endpoint_path + "/chat/completions")
        endpoint = urlunsplit((parsed.scheme, parsed.netloc, endpoint_path, "", ""))
        default_port = 443 if parsed.scheme == "https" else 80
        return endpoint, host, int(port or default_port), loopback_test_target

    def _validate_resolved_target(self) -> None:
        """Block DNS answers that would turn a trusted hostname into SSRF.

        ``requests`` performs its own connection, so HTTPS plus exact
        server-owned hostname allowlisting remain the primary authority.  This
        preflight adds a fail-closed check against obvious resolver mistakes or
        DNS poisoning; no resolved address is surfaced to callers.
        """
        if (self._allow_loopback_http_for_tests
                and self._host == _LOOPBACK_TEST_HOST):
            return
        try:
            rows = socket.getaddrinfo(
                self._host,
                self._port,
                type=socket.SOCK_STREAM,
            )
        except (OSError, ValueError) as exc:
            raise LLMClientError("host_resolution_failed") from exc
        if not rows:
            raise LLMClientError("host_resolution_failed")
        for _family, _socktype, _proto, _canonname, sockaddr in rows:
            try:
                address = ipaddress.ip_address(str(sockaddr[0]))
            except (IndexError, ValueError) as exc:
                raise LLMClientError("host_resolution_failed") from exc
            # ``is_global`` rejects private, loopback, link-local, reserved,
            # multicast and unspecified ranges in one closed rule.
            if not address.is_global:
                raise LLMClientError("private_network_target")

    def __repr__(self) -> str:  # pragma: no cover - defensive redaction
        return "%s(api_base=<redacted>, model=%r, allow_external=%r)" % (
            type(self).__name__, self._model, self._allow_external,
        )

    def complete(self, *, system_prompt: str, user_request: str) -> str:
        """Call a standard Chat Completions endpoint only after explicit opt-in."""
        if not self._allow_external:
            raise ExternalLLMDisabled()
        self._validate_resolved_target()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key
        body = {
            "model": self._model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_request},
            ],
        }
        # Environment proxy variables are ambient configuration, not an
        # authority granted to this adapter.  Inheriting them could tunnel an
        # otherwise allowlisted HTTPS request through an unexpected proxy (and
        # possibly disclose the bearer token), so each request gets a fresh
        # session with trust_env explicitly disabled.
        session = requests.Session()
        session.trust_env = False
        try:
            try:
                response = session.post(
                    self._endpoint,
                    headers=headers,
                    json=body,
                    timeout=self._timeout,
                    allow_redirects=False,
                    stream=True,
                )
            except requests.RequestException as exc:
                # Do not propagate a URL, request body, or provider response into a
                # result that callers may later retain.
                raise LLMClientError("transport_error") from exc
            try:
                try:
                    if 300 <= response.status_code < 400:
                        raise LLMClientError("redirect_blocked")
                    if response.status_code >= 400:
                        raise LLMClientError("http_%d" % int(response.status_code))
                    declared_length = response.headers.get("Content-Length")
                    if declared_length is not None:
                        try:
                            if int(declared_length) > MAX_RESPONSE_BYTES:
                                raise LLMClientError("response_too_large")
                        except ValueError:
                            # A malformed length cannot make us skip the streaming cap.
                            pass
                    chunks = []
                    size = 0
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        size += len(chunk)
                        if size > MAX_RESPONSE_BYTES:
                            raise LLMClientError("response_too_large")
                        chunks.append(chunk)
                    payload = json.loads(b"".join(chunks).decode("utf-8"))
                    choices = payload["choices"]
                    content = choices[0]["message"]["content"]
                except LLMClientError:
                    raise
                except requests.RequestException as exc:
                    raise LLMClientError("transport_error") from exc
                except (KeyError, IndexError, TypeError, UnicodeDecodeError, ValueError) as exc:
                    raise LLMClientError("malformed_response") from exc
            finally:
                response.close()
        finally:
            session.close()
        if not isinstance(content, str) or not content.strip():
            raise LLMClientError("malformed_response")
        return content


@dataclass(frozen=True)
class ConversionResult:
    """In-memory conversion result plus a deliberately non-verbatim summary.

    ``yaml_text`` is available to the immediate caller so it can show an editor
    or hand the answer to a private review workflow.  It is omitted from
    ``repr`` and :meth:`safe_summary`; the raw request, bundle and emitted
    signal batch are never retained in this object.
    """

    status: ConversionStatus
    request_sha256: str = field(repr=False, compare=False)
    strategy_kind: Optional[str]
    generated_strategy_kind: Optional[str]
    yaml_text: Optional[str] = field(default=None, repr=False, compare=False)
    yaml_sha256: Optional[str] = None
    bundle_sha256: Optional[str] = None
    signal_batch_sha256: Optional[str] = None
    signal_count: Optional[int] = None
    warning_count: int = 0
    reason_codes: Tuple[str, ...] = ()

    def safe_summary(self) -> Dict[str, Any]:
        """Return the only payload suitable for normal logs or public records."""
        return {
            "status": self.status.value,
            "strategy_kind": self.strategy_kind,
            "generated_strategy_kind": self.generated_strategy_kind,
            "yaml_sha256": self.yaml_sha256,
            "bundle_sha256": self.bundle_sha256,
            "signal_batch_sha256": self.signal_batch_sha256,
            "signal_count": self.signal_count,
            "warning_count": self.warning_count,
            "reason_codes": list(self.reason_codes),
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Any, *, label: str) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("%s is not finite JSON" % label) from exc


def _request_hash(request: Any) -> str:
    return _sha256_bytes(str(request or "").encode("utf-8"))


def _safe_reason_code(value: Any, *, fallback: str) -> str:
    """Keep an injected client's exception text out of public summaries."""
    code = str(value or "")
    return code if _SAFE_REASON_CODE.fullmatch(code) else fallback


def _bounded_text(value: Any, *, maximum: int) -> Optional[str]:
    if not isinstance(value, str):
        return None
    if not value.strip() or len(value.encode("utf-8")) > maximum:
        return None
    return value


def _normalize_reviewed_conditions(
    reviewed_conditions: Optional[Sequence[Any]],
) -> Tuple[Optional[Tuple[Condition, ...]], Optional[str]]:
    """Validate the non-verbatim, structured input that authorizes G1e.

    ``None`` is a legitimate caller state: the YAML may be exercised through
    the frozen runtime for development, but the result must remain
    ``runtime_valid_unreviewed``.  Any non-empty supplied value has to cross the
    privacy boundary in :func:`normalize_condition`; in particular, a user
    quote, user id, or misspelled condition field cannot be smuggled into this
    conversion surface and then logged by a later caller.
    """
    if reviewed_conditions is None:
        return None, None
    if isinstance(reviewed_conditions, (str, bytes, Mapping)):
        return None, "invalid_reviewed_conditions"
    try:
        normalized = tuple(normalize_condition(item) for item in reviewed_conditions)
    except Exception:  # noqa: BLE001 - validation details may contain caller input
        return None, "invalid_reviewed_conditions"
    if not normalized:
        return (), "reviewed_conditions_required"
    return normalized, None


def _semantic_reason_code(status: str, *, clean: bool) -> str:
    """Map gate state to a closed code without retaining gate/source details."""
    if clean:
        return "semantic_clean"
    return {
        "condition_violated": "semantic_condition_violated",
        "condition_unresolved": "semantic_condition_unresolved",
        "undisclosed_assumption": "semantic_undisclosed_assumption",
        "bundle_insufficient": "semantic_bundle_insufficient",
    }.get(str(status), "semantic_evaluation_failed")


def _freeze_bundle(bundle: Any) -> Tuple[Dict[str, Any], str]:
    """Use the evaluator's provenance-bearing frozen-input contract verbatim.

    A JSON mapping with frames is enough to exercise a development runtime, but
    it is not a frozen evaluation receipt unless it also identifies the capture
    and decision time.  Reusing the evaluator's single contract prevents this
    convenience conversion surface from reporting ``runtime_passed`` on a
    hand-assembled, untraceable bundle.
    """
    try:
        frozen, encoded = evaluate._require_frozen_bundle(bundle)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_frozen_bundle") from exc
    return frozen, _sha256_bytes(encoded)


def _unsupported_reason(intent: Any) -> Optional[str]:
    """Map independently extracted impossibilities to a stable public state."""
    preferences = tuple(getattr(intent, "unsupported_preferences", ()) or ())
    if any(item == "under_discovered" or str(item).startswith("gap:")
           for item in preferences):
        return "intent_not_expressible"
    sources = frozenset(getattr(intent, "sources", ()) or ())
    if (getattr(intent, "kind", None) == "selection"
            and sources.intersection(UNSUPPORTED_SELECTION_SOURCES)):
        return "selection_source_not_available"
    return None


def _clarification_reason(intent: Any) -> Optional[str]:
    if getattr(intent, "kind", None) == "ambiguous":
        return "ambiguous_intent"
    trade_reason = _trade_preflight_reason(intent)
    if trade_reason is not None:
        return trade_reason
    preferences = tuple(getattr(intent, "unsupported_preferences", ()) or ())
    if any(str(item).startswith("category_exclusion:") for item in preferences):
        return "unmapped_category_exclusion"
    if (getattr(intent, "kind", None) == "selection"
            and not getattr(intent, "sources", ())
            and "supertrend" not in getattr(intent, "indicator_names", ())):
        # A model cannot independently determine whether a bare "pick five
        # coins" request means liquidity, news, funding, or a technical screen.
        return "missing_selection_criterion"
    return None


def _trade_preflight_reason(intent: Any) -> Optional[str]:
    """Refuse trade meanings the current single-rule DSL cannot prove.

    These are intentionally checked before an LLM sees the request.  They mirror
    the deterministic boundaries already encoded in ``IntentDecision`` rather
    than guessing close-versus-short semantics, silently splitting a multi-symbol
    request, flattening boolean groups, or defaulting an incomplete ATR plan.
    """
    if getattr(intent, "kind", None) != "trade":
        return None
    evidence = frozenset(getattr(intent, "evidence", ()) or ())
    if "bare_sell_direction_ambiguous" in evidence:
        return "ambiguous_sell_action"
    if len(tuple(getattr(intent, "included_symbols", ()) or ())) > 1:
        return "multiple_trade_symbols"
    directions = frozenset(getattr(intent, "directions", ()) or ())
    if getattr(intent, "market_type", None) == "spot" and "short" in directions:
        return "spot_short_not_supported"
    if getattr(intent, "condition_join", None) == "mixed":
        return "mixed_boolean_conditions"
    if getattr(intent, "exit_type", None) == "atr_trailing_stop" and (
            getattr(intent, "exit_atr_period", None) is None
            or getattr(intent, "exit_trail_mult", None) is None):
        return "incomplete_atr_trailing_exit"
    return None


def _result(
    status: ConversionStatus,
    *,
    request_sha256: str,
    strategy_kind: Optional[str],
    generated_strategy_kind: Optional[str] = None,
    yaml_text: Optional[str] = None,
    yaml_sha256: Optional[str] = None,
    bundle_sha256: Optional[str] = None,
    signal_batch_sha256: Optional[str] = None,
    signal_count: Optional[int] = None,
    warning_count: int = 0,
    reason_codes: Tuple[str, ...] = (),
) -> ConversionResult:
    return ConversionResult(
        status=status,
        request_sha256=request_sha256,
        strategy_kind=strategy_kind,
        generated_strategy_kind=generated_strategy_kind,
        yaml_text=yaml_text,
        yaml_sha256=yaml_sha256,
        bundle_sha256=bundle_sha256,
        signal_batch_sha256=signal_batch_sha256,
        signal_count=signal_count,
        warning_count=warning_count,
        reason_codes=reason_codes,
    )


def convert_and_run(
    *,
    request: str,
    system_prompt: str,
    llm_client: LLMClient,
    frozen_bundle: Mapping[str, Any],
    reviewed_conditions: Optional[Sequence[Any]] = None,
) -> ConversionResult:
    """Convert one request and execute it once against a frozen input bundle.

    No retry occurs here: retry policy belongs to a caller that can deliberately
    store a private attempt record.  Ambiguous and known-unexpressible requests
    return before the client is called, so they can never be made to look like
    valid YAML by a convenient default.

    ``RUNTIME_PASSED`` is reserved for a non-empty, privacy-safe set of reviewed
    structured conditions whose G1e predicates all pass against the *same*
    frozen bundle.  Omitting that input still runs the canonical runtime, but
    returns ``RUNTIME_VALID_UNREVIEWED`` so a UI or queue cannot confuse "the
    YAML executed" with "the request was proved satisfied".
    """
    request_sha256 = _request_hash(request)
    normalized_conditions, condition_error = _normalize_reviewed_conditions(
        reviewed_conditions,
    )
    if condition_error == "invalid_reviewed_conditions":
        return _result(
            ConversionStatus.SEMANTIC_FAILED,
            request_sha256=request_sha256,
            strategy_kind=None,
            reason_codes=(condition_error,),
        )
    bounded_request = _bounded_text(request, maximum=MAX_REQUEST_BYTES)
    if bounded_request is None:
        return _result(
            ConversionStatus.NEEDS_CLARIFICATION,
            request_sha256=request_sha256,
            strategy_kind=None,
            reason_codes=("missing_or_oversize_request",),
        )
    intent = classify_request(bounded_request)
    unsupported = _unsupported_reason(intent)
    if unsupported is not None:
        return _result(
            ConversionStatus.UNSUPPORTED,
            request_sha256=request_sha256,
            strategy_kind=intent.kind,
            reason_codes=(unsupported,),
        )
    clarification = _clarification_reason(intent)
    if clarification is not None:
        return _result(
            ConversionStatus.NEEDS_CLARIFICATION,
            request_sha256=request_sha256,
            strategy_kind=intent.kind,
            reason_codes=(clarification,),
        )
    bounded_prompt = _bounded_text(system_prompt, maximum=MAX_SYSTEM_PROMPT_BYTES)
    if bounded_prompt is None:
        return _result(
            ConversionStatus.LLM_FAILED,
            request_sha256=request_sha256,
            strategy_kind=intent.kind,
            reason_codes=("missing_or_oversize_system_prompt",),
        )
    try:
        bundle, bundle_sha256 = _freeze_bundle(frozen_bundle)
    except ValueError as exc:
        return _result(
            ConversionStatus.RUNTIME_FAILED,
            request_sha256=request_sha256,
            strategy_kind=intent.kind,
            reason_codes=(_safe_reason_code(exc, fallback="invalid_frozen_bundle"),),
        )

    try:
        raw_yaml = llm_client.complete(
            system_prompt=bounded_prompt,
            user_request=bounded_request,
        )
    except LLMClientError as exc:
        return _result(
            ConversionStatus.LLM_FAILED,
            request_sha256=request_sha256,
            strategy_kind=intent.kind,
            bundle_sha256=bundle_sha256,
            reason_codes=(_safe_reason_code(exc.code, fallback="client_error"),),
        )
    except Exception:  # noqa: BLE001 - injected clients must not leak details
        return _result(
            ConversionStatus.LLM_FAILED,
            request_sha256=request_sha256,
            strategy_kind=intent.kind,
            bundle_sha256=bundle_sha256,
            reason_codes=("client_error",),
        )
    if not isinstance(raw_yaml, str) or not raw_yaml.strip():
        return _result(
            ConversionStatus.LLM_FAILED,
            request_sha256=request_sha256,
            strategy_kind=intent.kind,
            bundle_sha256=bundle_sha256,
            reason_codes=("malformed_response",),
        )

    yaml_text = strip_code_fence(raw_yaml).strip()
    yaml_sha256 = _sha256_bytes(yaml_text.encode("utf-8"))
    try:
        spec = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        return _result(
            ConversionStatus.YAML_REJECTED,
            request_sha256=request_sha256,
            strategy_kind=intent.kind,
            yaml_text=yaml_text,
            yaml_sha256=yaml_sha256,
            bundle_sha256=bundle_sha256,
            reason_codes=("yaml_parse_error",),
        )
    if not isinstance(spec, dict):
        return _result(
            ConversionStatus.YAML_REJECTED,
            request_sha256=request_sha256,
            strategy_kind=intent.kind,
            yaml_text=yaml_text,
            yaml_sha256=yaml_sha256,
            bundle_sha256=bundle_sha256,
            reason_codes=("yaml_not_mapping",),
        )

    validation_errors, validation_warnings = validate_spec(spec)
    generated_kind = generated_strategy_kind(spec)
    if validation_errors:
        return _result(
            ConversionStatus.YAML_REJECTED,
            request_sha256=request_sha256,
            strategy_kind=intent.kind,
            generated_strategy_kind=generated_kind,
            yaml_text=yaml_text,
            yaml_sha256=yaml_sha256,
            bundle_sha256=bundle_sha256,
            warning_count=len(validation_warnings),
            reason_codes=("yaml_validation_failed",),
        )
    alignment_errors, alignment_warnings = reconcile_intent(intent, spec)
    warning_count = len(validation_warnings) + len(alignment_warnings)
    if alignment_errors:
        return _result(
            ConversionStatus.YAML_REJECTED,
            request_sha256=request_sha256,
            strategy_kind=intent.kind,
            generated_strategy_kind=generated_kind,
            yaml_text=yaml_text,
            yaml_sha256=yaml_sha256,
            bundle_sha256=bundle_sha256,
            warning_count=warning_count,
            reason_codes=("intent_mismatch",),
        )

    try:
        report = run_gates(
            spec,
            nl=bounded_request,
            bundle=bundle,
            conditions=normalized_conditions or (),
        )
    except Exception:  # noqa: BLE001 - gate/runtime errors may contain input values
        return _result(
            ConversionStatus.RUNTIME_FAILED,
            request_sha256=request_sha256,
            strategy_kind=intent.kind,
            generated_strategy_kind=generated_kind,
            yaml_text=yaml_text,
            yaml_sha256=yaml_sha256,
            bundle_sha256=bundle_sha256,
            warning_count=warning_count,
            reason_codes=("canonical_runtime_error",),
        )

    # ``run_gates`` is intentionally re-run after the earlier explicit parse,
    # validation and intent checks.  This keeps the conversion result's legacy
    # failure taxonomy precise while binding a ready result to the one canonical
    # G1a--G1e execution path.  Never retain ``report.errors``: those can carry
    # field values from a private frozen bundle.
    if report.failed_gate in ("G1a", "G1b"):
        return _result(
            ConversionStatus.YAML_REJECTED,
            request_sha256=request_sha256,
            strategy_kind=intent.kind,
            generated_strategy_kind=generated_kind,
            yaml_text=yaml_text,
            yaml_sha256=yaml_sha256,
            bundle_sha256=bundle_sha256,
            warning_count=len(report.warnings),
            reason_codes=("canonical_validation_inconsistent",),
        )
    if report.failed_gate == "G1c":
        return _result(
            ConversionStatus.YAML_REJECTED,
            request_sha256=request_sha256,
            strategy_kind=intent.kind,
            generated_strategy_kind=generated_kind,
            yaml_text=yaml_text,
            yaml_sha256=yaml_sha256,
            bundle_sha256=bundle_sha256,
            warning_count=len(report.warnings),
            reason_codes=("intent_mismatch",),
        )
    if report.failed_gate == "G1d":
        return _result(
            ConversionStatus.RUNTIME_FAILED,
            request_sha256=request_sha256,
            strategy_kind=intent.kind,
            generated_strategy_kind=generated_kind,
            yaml_text=yaml_text,
            yaml_sha256=yaml_sha256,
            bundle_sha256=bundle_sha256,
            warning_count=len(report.warnings),
            # Preserve the pre-G1e public runtime taxonomy.  A G1d source
            # outage is not a semantic review result because no batch exists;
            # only the G1e exact-count data shortage gets the distinct safe
            # ``semantic_bundle_insufficient`` outcome below.
            reason_codes=("canonical_runtime_error",),
        )

    batch = report.batch
    if not isinstance(batch, Mapping):
        # A G1e report without its G1d batch is an internal contract break, not
        # something a caller may reclassify as semantic success.
        return _result(
            ConversionStatus.RUNTIME_FAILED,
            request_sha256=request_sha256,
            strategy_kind=intent.kind,
            generated_strategy_kind=generated_kind,
            yaml_text=yaml_text,
            yaml_sha256=yaml_sha256,
            bundle_sha256=bundle_sha256,
            warning_count=len(report.warnings),
            reason_codes=("canonical_runtime_error",),
        )
    batch_sha256 = _sha256_bytes(_canonical_json_bytes(batch, label="signal batch"))
    signal_count = int(batch.get("signal_count") or 0)
    gate_warning_count = len(report.warnings)

    if not normalized_conditions:
        return _result(
            ConversionStatus.RUNTIME_VALID_UNREVIEWED,
            request_sha256=request_sha256,
            strategy_kind=intent.kind,
            generated_strategy_kind=generated_kind,
            yaml_text=yaml_text,
            yaml_sha256=yaml_sha256,
            bundle_sha256=bundle_sha256,
            signal_batch_sha256=batch_sha256,
            signal_count=signal_count,
            warning_count=gate_warning_count,
            reason_codes=("reviewed_conditions_required",),
        )
    if not report.clean:
        return _result(
            ConversionStatus.SEMANTIC_FAILED,
            request_sha256=request_sha256,
            strategy_kind=intent.kind,
            generated_strategy_kind=generated_kind,
            yaml_text=yaml_text,
            yaml_sha256=yaml_sha256,
            bundle_sha256=bundle_sha256,
            signal_batch_sha256=batch_sha256,
            signal_count=signal_count,
            warning_count=gate_warning_count,
            reason_codes=(_semantic_reason_code(report.status, clean=False),),
        )
    return _result(
        ConversionStatus.RUNTIME_PASSED,
        request_sha256=request_sha256,
        strategy_kind=intent.kind,
        generated_strategy_kind=generated_kind,
        yaml_text=yaml_text,
        yaml_sha256=yaml_sha256,
        bundle_sha256=bundle_sha256,
        signal_batch_sha256=batch_sha256,
        signal_count=signal_count,
        warning_count=gate_warning_count,
    )
