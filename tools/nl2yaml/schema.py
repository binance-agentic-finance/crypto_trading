"""Record types for the NL→YAML dataset: dataclasses, computed gates, writers.

The dataset being built here trains a small model to turn a user's chat message
into a ``strategy.yaml`` spec for ``cyqnt_trd.standard_bot.yaml_pipeline``.
Every guardrail in this module exists because the obvious shortcut it blocks has
already produced a wrong dataset at least once:

1. **"The second attempt" is not the label.** Attempt *k* only exists when
   attempts *1..k-1* failed, so harvesting "the attempt that finally worked" as
   the target biases the training set towards hard cases and *deletes every easy
   case* (they never had a second attempt). Gold therefore lives in its own
   fields on :class:`CaseRecord`, never implicitly in :class:`AttemptRecord`;
   and ``gold_source=attempt_k`` is only accepted once the attempt has been
   reconciled against the conditions (level >= 3), not merely because it ran.

2. **"It ran" is not "it is correct."** A spec once passed ``validate``, passed
   ``run``, emitted five candidates — and all five were TradFi perpetuals while
   the user's first line said "exclude TradFi". Nothing downstream can catch
   that, so ``gold_verification_level=5`` is defined as *all checkable
   conditions satisfied* and is rejected outright if any condition verdict is
   ``violated``.

3. **48% of the corpus is duplicate text** (one preset prompt card appears 393
   times), so a random train/test split measures memorisation. Hence
   ``dup_cluster_id`` / ``dup_count`` on every case (split by cluster, never by
   row) and ``dup_weighted_count`` on every gap (a gap behind the 393x card is
   not one unit of demand).

4. **Proxies are invisible downstream.** When the vocabulary cannot express a
   condition, a model will substitute a "close enough" field: "Supertrend bearish
   on 3 timeframes" becomes "in the 24h top-30 losers". That substitution passes
   the schema, passes the dry run, passes execution, and produces a plausible
   basket. ``proxy_used`` is therefore computed (never asserted by the caller)
   and forces both SFT eligibility flags to False at *any* verification level.

Privacy (both git remotes are PUBLIC): the raw user text and per-condition
quotes may only be written by :func:`write_case_internal`, which refuses any
path outside :func:`internal_root` — a directory this module asserts is outside
the repository. :func:`write_case` **raises** on a record carrying user text
rather than dropping the field, because a silent drop teaches the caller that
passing user text is fine.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
import unicodedata
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import jsonschema
from referencing import Registry, Resource

__all__ = [
    # errors
    "RecordError", "PrivacyError", "ComputedFieldConflict",
    # enums
    "Polarity", "Operator", "Unit", "Scope", "Measurability", "RankDirection",
    "CapabilityVerdict", "GapId", "GapStatus", "CaseTier", "MiningSource",
    "TextProvenance", "Lang", "ZhVariant", "ResolutionPath", "Intent",
    "MarketType", "Direction",
    "UniverseScope", "GoldRole", "GoldSource", "GoldVerificationLevel",
    "ConditionCheckStatus", "CheckedBy", "DisagreementResolution",
    "SamplingPurpose", "Gate", "DefectClass", "CandidateSide",
    # records
    "Condition", "CapabilityEntry", "IntentSlots", "GoldSpec",
    "ConditionVerdict", "GoldDisagreement", "CaseRecord", "AttemptRecord",
    "GapRecord", "BundleNode", "Candidate", "RunRecord", "InternalCaseRecord",
    # helpers
    "sha256_hex", "canonicalize_text", "canon_sha256_of", "cluster_id_for",
    "new_case_id", "hmac_pseudonym", "internal_root", "error_signature_for",
    "verbatim_ngrams", "verbatim_overlap", "NO_SOURCE_TEXT",
    "FORBIDDEN_PUBLIC_KEYS", "FORBIDDEN_PUBLIC_KEY_PREFIXES",
    # io
    "validate_record", "write_case", "write_case_internal", "write_case_pair",
    "write_attempt", "write_gap", "write_run",
    "read_cases", "read_attempts", "read_gaps", "read_runs",
    "read_cases_internal", "read_internal_text",
    "validate_attempt_chain",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"

#: Override only for tests. The value is still asserted to be outside the repo,
#: so pointing it at a tracked directory fails loudly instead of leaking.
INTERNAL_ROOT_ENV = "NL2YAML_INTERNAL_ROOT"
DEFAULT_INTERNAL_ROOT = Path.home() / "nl2yaml_internal"


class RecordError(ValueError):
    """A record violates an invariant of the dataset."""


class PrivacyError(RecordError):
    """A record or path would put user-identifying data in the public tree."""


class ComputedFieldConflict(RecordError):
    """A caller supplied a value for a field this module computes.

    Not tolerated even when "close": the whole point of computing eligibility
    here is that no caller gets to decide it.
    """


# ---------------------------------------------------------------------------
# privacy: key names that may never appear in a public record
# ---------------------------------------------------------------------------

#: Exact key names, matched after lower-casing. ``quote`` is listed *exactly*
#: and not as a prefix on purpose: the block vocabulary is full of legitimate
#: ``quote_volume`` / ``quote_suffix`` / ``quote_asset`` keys, and a guard that
#: fires on those gets switched off within a week.
FORBIDDEN_PUBLIC_KEYS = frozenset({
    "quote", "quotes", "conditions_with_quotes",
    "user_id", "uid", "chat_id", "telegram_id", "username", "handle",
    "first_query", "raw_text", "excerpt", "user_text_excerpt",
    "verbatim", "verbatim_text", "message", "messages", "transcript",
    "prompt", "prompt_text", "rendered_prompt",
})

#: Prefixes, matched after lower-casing. ``prompt`` is *not* here (see
#: ``prompt_tokens_by_component``) but bare ``prompt``/``prompt_text`` are in the
#: exact set above — the rendered prompt embeds the user's message verbatim, which
#: is why attempts store only ``prompt_sha256``.
FORBIDDEN_PUBLIC_KEY_PREFIXES = ("user_text", "raw_user", "user_message")


# ---------------------------------------------------------------------------
# closed vocabularies
# ---------------------------------------------------------------------------

class Polarity(str, Enum):
    """Whether the condition asks for something or forbids it.

    Split out from ``operator`` because exclusions are the conditions that get
    silently dropped ("exclude TradFi" → basket of five TradFi perps) and we
    need to count and test them as their own class.
    """

    INCLUDE = "include"
    EXCLUDE = "exclude"
    #: A soft wish, not a gate. "在盡可能滿足以上條件為前提" and "最好是可以漲的" are
    #: not filters: dropping a name for failing one of them is wrong, and so is
    #: treating it as satisfied because the basket happens to contain it. Folding
    #: these into INCLUDE was the first thing tried and it turned every wish into
    #: a hard filter, emptying baskets for reasons the user never asked for.
    #: A PREFER condition belongs in a weighted score, which ``selection:``
    #: cannot express today (GapId.WEIGHTED_MULTI_FACTOR_SCORE).
    PREFER = "prefer"


class Operator(str, Enum):
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EQ = "eq"
    NEQ = "neq"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"
    RISING = "rising"
    FALLING = "falling"
    TOP_N = "top_n"
    BOTTOM_N = "bottom_n"
    EXISTS = "exists"
    ABSENT = "absent"
    #: The user stated a direction but no threshold ("volume should be high").
    #: Kept explicit so a vague ask can never be recorded as a number.
    UNSPECIFIED = "unspecified"


class Unit(str, Enum):
    """Unit of ``Condition.value``.

    Mandatory for quantified numeric conditions. "> 2 million" has been read as
    ``2``, ``2e6`` and ``2_000_000`` in the same afternoon; without a unit the
    record cannot say which one the user meant.
    """

    USD = "usd"
    PCT = "pct"
    RATIO = "ratio"
    BPS = "bps"
    COUNT = "count"
    BARS = "bars"
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    LEVERAGE_X = "leverage_x"
    COIN_QTY = "coin_qty"
    INDICATOR_VALUE = "indicator_value"
    SYMBOL = "symbol"
    LABEL = "label"


class Scope(str, Enum):
    """Which layer of the spec the condition constrains."""

    UNIVERSE = "universe"
    ENTRY = "entry"
    EXIT = "exit"
    SIZING = "sizing"
    RISK = "risk"
    SCHEDULE = "schedule"
    NOTIFICATION = "notification"
    DATA = "data"
    REPORTING = "reporting"
    META = "meta"


class Measurability(str, Enum):
    QUANTIFIED = "quantified"
    VAGUE = "vague"
    UNMEASURABLE = "unmeasurable"


class EvaluationGranularity(str, Enum):
    """At what granularity the condition has to be evaluated.

    A different axis from :class:`Scope`, which says *which part of the strategy*
    a condition governs. This one says *what data shape* it needs, and it is the
    field that mechanically separates the asks the ``selection:`` layer can serve
    from the ones it cannot: a cross-section has one row per symbol and no bars,
    so every ``PER_SYMBOL`` condition needs a fan-out that does not exist yet.
    Measured on the five-case demo, ``PER_SYMBOL`` was 48 of 81 conditions — the
    single largest reason a real request could not be converted, and invisible
    until it was given its own field.
    """

    #: decidable from the cross-section alone (24h ticker, funding snapshot, ...)
    UNIVERSE = "universe"
    #: needs per-symbol history/bars, i.e. a fan-out over candidates
    PER_SYMBOL = "per_symbol"
    #: about the basket or the account as a whole, not any single name
    PORTFOLIO = "portfolio"


class AmbiguityType(str, Enum):
    """Why a condition cannot be converted without choosing on the user's behalf.

    Kept apart from :class:`CapabilityVerdict` because the two need opposite
    responses and a model trained on them merged learns to swap the two: an
    unsupported condition is fixed by adding a block, an ambiguous one never is —
    only the user can settle it. Merging them was how a request became a basket
    from the wrong end of the market with nothing in the output admitting a
    choice had been made.
    """

    #: the number has no unit the text pins down: "成交量一千萬" is 1e7 USD of
    #: turnover or 1e7 coins, and the two select different names — measured on
    #: the demo, one candidate passed under one reading and failed under the other
    UNIT = "unit"
    #: two computable readings of one phrase, and no unit involved: "3 個月的
    #: 低點到高點漲幅" is the window's RANGE (high over low, either order) or an
    #: ordered RUN-UP (low first, then high). Both are computable and they are
    #: different numbers — measured on the three-month screen, 2 of 5 candidates
    #: had their high BEFORE their low, so the two readings disagree about them.
    #: Distinct from UNDEFINED, where no reading is computable at all: this one
    #: is settled by declaring which computation was used, that one cannot be
    #: settled without asking.
    READING = "reading"
    #: two conditions cannot both hold: "掃漲幅榜" and "找正在築底的"
    CONFLICT = "conflict"
    #: no computable definition exists for the words used ("築底", "強勢", "吸籌成功")
    UNDEFINED = "undefined"


class RankDirection(str, Enum):
    ASC = "asc"
    DESC = "desc"


class CapabilityVerdict(str, Enum):
    """Can the current block vocabulary express this condition?

    ``PROXY`` is the dangerous one and therefore requires a ``gap_id``: it means
    we wrote something *else* that correlates. See ``proxy_used``.
    """

    SUPPORTED = "supported"
    PROXY = "proxy"
    UNSUPPORTED = "unsupported"
    UNDECIDABLE = "undecidable"


class GapId(str, Enum):
    """Closed vocabulary of missing capabilities.

    Closed on purpose. Free text fragments demand counts across
    ``exclude_tradfi`` / ``no_tradfi`` / ``tradfi``, and the sweep that
    invalidates golds when a gap closes then misses records. Adding a member is
    a reviewed edit to this enum plus ``schemas/common.schema.json``.
    """

    #: exchangeInfo-derived contract metadata: underlyingType / underlyingSubType
    #: / contractType. Blocks the "exclude TradFi", sector and "Alpha coin" asks.
    UNIVERSE_CONTRACT_META = "universe_contract_meta"
    #: compute an indicator per candidate and join it back onto the cross section
    UNIVERSE_PER_CANDIDATE_INDICATOR = "universe_per_candidate_indicator"
    UNIVERSE_LONG_SHORT_RATIO = "universe_long_short_ratio"
    UNIVERSE_OPEN_INTEREST = "universe_open_interest"
    #: real market cap / circulating supply — the exchange does not serve it
    UNIVERSE_MARKET_CAP = "universe_market_cap"
    #: same indicator required to agree across several timeframes at once
    MULTI_TIMEFRAME_CONFLUENCE = "multi_timeframe_confluence"
    #: the named indicator has no block at all (e.g. Supertrend)
    INDICATOR_NOT_IN_VOCABULARY = "indicator_not_in_vocabulary"
    ORDERBOOK_MICROSTRUCTURE = "orderbook_microstructure"
    ONCHAIN_DATA = "onchain_data"
    NEWS_EVENT_CALENDAR = "news_event_calendar"
    SOCIAL_SENTIMENT = "social_sentiment"
    CROSS_EXCHANGE = "cross_exchange"
    PORTFOLIO_LEVEL_RISK = "portfolio_level_risk"
    #: soft "satisfy as many of these as possible" scoring. ``selection.score``
    #: takes one column or one feature, so several PREFER conditions can only be
    #: served by chaining them as hard filters (wrong: one miss drops the name) or
    #: by dropping all but one (wrong: the rest silently vanish). The building
    #: blocks exist — ``scoring.weighted_composite`` / ``scoring.additive_combine``
    #: — but nothing wires them into the selection layer.
    WEIGHTED_MULTI_FACTOR_SCORE = "weighted_multi_factor_score"
    #: per-candidate entry/stop/target. ``selection:`` emits a basket; the trade
    #: plan for each name is a ``signals:`` spec, and one request cannot be both.
    ENTRY_EXIT_PER_CANDIDATE = "entry_exit_per_candidate"
    NOTIFICATION_CHANNEL = "notification_channel"
    #: the ask is a human judgement call ("when it feels toppy")
    MANUAL_DISCRETION = "manual_discretion"
    #: venue product (grid bot, copy trading, DCA bot) rather than a strategy
    EXECUTION_VENUE_FEATURE = "execution_venue_feature"
    BACKTEST_HISTORY_MISSING = "backtest_history_missing"
    #: Explicit triage placeholder. Deliberately visible in the gap counts —
    #: unlike free text it cannot fragment them — but it is not a silent bucket:
    #: anything left here is an unfinished adjudication.
    UNNAMED_CAPABILITY = "unnamed_capability"


class GapStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    CLOSED = "closed"


class CaseTier(str, Enum):
    """How much of the request the current vocabulary can carry.

    Cross-checked against ``capability_map`` on construction, because a tier
    that drifts from the adjudication silently re-labels the training set.
    """

    #: no condition could be extracted ("help me make money")
    T0_UNDERSPECIFIED = "t0_underspecified"
    #: every condition is ``supported``
    T1_EXPRESSIBLE = "t1_expressible"
    #: at least one ``proxy``, no ``unsupported``
    T2_PROXY_NEEDED = "t2_proxy_needed"
    #: at least one ``unsupported``
    T3_BLOCKED_BY_GAP = "t3_blocked_by_gap"
    #: not a strategy request at all (education, execution debugging, chit-chat)
    T4_OUT_OF_SCOPE = "t4_out_of_scope"


class MiningSource(str, Enum):
    """How the case was *found*. Provenance only — never a label or a feature.

    Using it as either leaks the miner's bias into the weights: every
    ``KEYWORD_LEXICON`` case contains the keyword, so the model learns the
    keyword instead of the intent.
    """

    KEYWORD_LEXICON = "keyword_lexicon"
    CLASSIFIER_SCORE = "classifier_score"
    PRESET_CARD_DEDUP = "preset_card_dedup"
    RANDOM_SAMPLE = "random_sample"
    HUMAN_CURATED = "human_curated"
    SYNTHETIC_TEMPLATE = "synthetic_template"


class TextProvenance(str, Enum):
    """Where the case's input text came from.

    ``VERBATIM_INTERNAL`` means the text exists only in the internal store; the
    public record carries hashes and structure. ``SYNTHETIC`` means there is no
    user behind it, which is why it is the only provenance allowed to have
    ``pseudonym_id=None``.
    """

    VERBATIM_INTERNAL = "verbatim_internal"
    PARAPHRASE = "paraphrase"
    SYNTHETIC = "synthetic"


class Lang(str, Enum):
    """Closed to the two languages the corpus is scoped to.

    The export is Chinese + English by construction (26,667 zh / 24,928 en over
    2026-05..07) and the analysis deliberately excludes everything else. Closing
    the vocabulary means an out-of-scope row cannot enter the dataset by being
    mislabelled ``ja`` instead of being dropped.
    """

    ZH = "zh"
    EN = "en"


class ZhVariant(str, Enum):
    """Script variant, present only for ``lang=zh``.

    Simplified outnumbers traditional ~16:1 in the corpus, so a train/test split
    that ignores the variant will report a traditional-Chinese accuracy it never
    measured.
    """

    HANS = "zh-Hans"
    HANT = "zh-Hant"


class ResolutionPath(str, Enum):
    CACHE_EXACT = "cache_exact"
    TEMPLATE_SLOTFILL = "template_slotfill"
    SMALL_MODEL = "small_model"
    BIG_MODEL = "big_model"


class Intent(str, Enum):
    """Primary intent taxonomy, reused verbatim from the 2026-05..07 chat
    analysis so counts stay comparable with ``adjudication_metrics.json``."""

    COIN_SELECTION = "COIN_SELECTION"
    STRATEGY_BUILD = "STRATEGY_BUILD"
    STRATEGY_DISCOVERY = "STRATEGY_DISCOVERY"
    AUTOMATION_BOT = "AUTOMATION_BOT"
    ENTRY_EXIT_TIMING = "ENTRY_EXIT_TIMING"
    MARKET_ANALYSIS = "MARKET_ANALYSIS"
    DATA_REQUEST = "DATA_REQUEST"
    RISK_MANAGEMENT = "RISK_MANAGEMENT"
    PORTFOLIO = "PORTFOLIO"
    ALERT_NOTIFY = "ALERT_NOTIFY"
    EXECUTION_DEBUG = "EXECUTION_DEBUG"
    EDUCATION_OTHER = "EDUCATION_OTHER"


class MarketType(str, Enum):
    SPOT = "spot"
    FUTURES = "futures"
    BOTH = "both"


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    BOTH = "both"


class UniverseScope(str, Enum):
    SINGLE_SYMBOL = "single_symbol"
    EXPLICIT_LIST = "explicit_list"
    ALL_FUTURES = "all_futures"
    ALL_SPOT = "all_spot"
    TOP_N_BY_VOLUME = "top_n_by_volume"


class GoldRole(str, Enum):
    """Why this gold spec is in the list.

    ``ALTERNATIVE`` exists so evaluation can accept a differently-encoded but
    equivalent answer instead of scoring it wrong; ``PRIMARY`` is the single SFT
    target and exactly one is required.
    """

    PRIMARY = "primary"
    ALTERNATIVE = "alternative"
    MINIMAL = "minimal"
    REFUSAL = "refusal"


class GoldSource(str, Enum):
    ATTEMPT_K = "attempt_k"
    HUMAN_EDITED = "human_edited"
    HANDWRITTEN = "handwritten"
    TEMPLATE = "template"
    NONE = "none"


class GoldVerificationLevel(IntEnum):
    """Ordered: each level includes everything below it.

    The ordering is the point — ``>=`` comparisons in the eligibility formula
    only mean something if the levels are a chain, and level 5 is deliberately
    *not* "it ran" (that is level 4) but "every condition we can check was
    actually satisfied by the output".
    """

    UNPARSED = 0
    STATIC_VALID = 1
    DRY_RAN = 2
    INTENT_RECONCILED = 3
    EXECUTED_ON_PINNED_BUNDLE = 4
    ALL_CHECKABLE_CONDITIONS_SATISFIED = 5


class ConditionCheckStatus(str, Enum):
    """Did the gold spec's *output* honour this condition?"""

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    #: expressible, but nothing in the pinned bundle can prove it either way
    UNVERIFIABLE = "unverifiable"
    #: the spec answered a correlated question instead
    PROXIED = "proxied"


class CheckedBy(str, Enum):
    STATIC = "static"
    DRY_RUN = "dry_run"
    EXECUTED = "executed"
    HUMAN = "human"


class DisagreementResolution(str, Enum):
    UNRESOLVED = "unresolved"
    THIRD_PARTY = "third_party"
    DISCUSSION = "discussion"
    DROPPED = "dropped"


class SamplingPurpose(str, Enum):
    CONVERT = "convert"
    REPAIR = "repair"
    DPO_NEGATIVE = "dpo_negative"


class Gate(str, Enum):
    """Where the attempt stopped.

    ``INTENT_MISMATCH`` is the gate that the TradFi basket needed and did not
    have: syntax fine, schema fine, execution fine, answer wrong.
    """

    PASSED = "passed"
    PARSE_ERROR = "parse_error"
    SCHEMA_INVALID = "schema_invalid"
    UNKNOWN_BLOCK = "unknown_block"
    DENIED_NAMESPACE = "denied_namespace"
    STATIC_VALIDATE_FAILED = "static_validate_failed"
    DRY_RUN_FAILED = "dry_run_failed"
    #: the pinned bundle lacks a frame/column the spec needs — not the model's
    #: fault, which is why it can never be a DPO negative
    BUNDLE_INSUFFICIENT = "bundle_insufficient"
    EXECUTION_ERROR = "execution_error"
    INTENT_MISMATCH = "intent_mismatch"
    #: every predicate held, under a reading of the request the spec chose and
    #: never stated. A valid DPO negative: declaring the reading is squarely the
    #: model's job, and the fix is a section it can write.
    UNDISCLOSED_ASSUMPTION = "undisclosed_assumption"


class DefectClass(str, Enum):
    NONE = "none"
    YAML_SYNTAX_ERROR = "yaml_syntax_error"
    SCHEMA_SHAPE_ERROR = "schema_shape_error"
    HALLUCINATED_BLOCK = "hallucinated_block"
    WRONG_BLOCK_SEMANTICS = "wrong_block_semantics"
    PROXY_SUBSTITUTION = "proxy_substitution"
    #: an exclusion the user stated is absent from the spec entirely
    SILENTLY_DROPPED_EXCLUSION = "silently_dropped_exclusion"
    MISSING_CONDITION = "missing_condition"
    EXTRA_CONDITION = "extra_condition"
    WRONG_OPERATOR = "wrong_operator"
    WRONG_TIMEFRAME = "wrong_timeframe"
    WRONG_DIRECTION = "wrong_direction"
    REFUSED_EXPRESSIBLE_REQUEST = "refused_expressible_request"


class CandidateSide(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32
ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")
UUID4_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PSEUDONYM_RE = re.compile(r"^pid_[a-z2-7]{26}$")

#: Zero-width space / non-joiner / joiner / BOM / word-joiner. Present in
#: copy-pasted chat text and enough on their own to defeat exact-duplicate
#: detection, which is why they are stripped before hashing.
_ZERO_WIDTH = dict.fromkeys(
    map(ord, "​‌‍﻿⁠"), None)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonicalize_text(text: str) -> str:
    """Normalise for duplicate detection.

    NFKC (so full-width digits and ASCII digits collide), lower-case, drop
    zero-width characters, collapse whitespace. 48% of the corpus is duplicate
    text and a good chunk of that only shows up after this normalisation.
    """
    out = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).lower()
    return re.sub(r"\s+", " ", out).strip()


def canon_sha256_of(text: str) -> str:
    return sha256_hex(canonicalize_text(text))


def cluster_id_for(canon_sha: str) -> str:
    """Cluster id for an exact-canonical-duplicate group.

    A separate field from ``canon_sha256`` rather than a reuse of it: near-dup
    clustering (embedding or edit distance) assigns one cluster to *several*
    canonical hashes, and the train/test split must follow the cluster.
    """
    if not SHA256_RE.match(canon_sha):
        raise RecordError("canon_sha256 must be 64 lowercase hex chars, got %r" % (canon_sha,))
    return "dup_" + canon_sha[:16]


def new_case_id() -> str:
    """A ULID: time-sortable, 26 chars, no dependency, no user data in it."""
    ms = int(time.time() * 1000)
    value = (ms << 80) | int.from_bytes(secrets.token_bytes(10), "big")
    out = []
    for _ in range(26):
        out.append(_ULID_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


def internal_root() -> Path:
    """Directory that may hold user text. Asserted to be outside the repo.

    The assertion is here and not only in the tests because the environment
    override exists: one mistyped path is the whole leak.
    """
    root = Path(os.environ.get(INTERNAL_ROOT_ENV) or DEFAULT_INTERNAL_ROOT).expanduser()
    root = root.resolve() if root.exists() else root.absolute()
    rel = os.path.relpath(str(root), str(REPO_ROOT))
    if not rel.startswith(os.pardir):
        raise PrivacyError(
            "internal root %s is inside the repository (%s); user text must never "
            "live in a tracked tree — being gitignored is not enough, one .gitignore "
            "edit publishes it" % (root, REPO_ROOT))
    return root


def _salt() -> bytes:
    """Read the HMAC salt, refusing a world-readable one.

    No generated-on-the-fly fallback: a per-run salt would make pseudonyms
    unlinkable across runs (dedup and per-user stats break), and a hard-coded
    one would make them a rainbow table away from the real ids.
    """
    path = internal_root() / "salt"
    if not path.exists():
        # No default salt on purpose — see the docstring.
        raise PrivacyError(
            "missing HMAC salt at %s; create it with 32 random bytes and mode 600" % path)
    mode = path.stat().st_mode & 0o777
    if mode != 0o600:
        raise PrivacyError("salt %s has mode %o, expected 600" % (path, mode))
    return path.read_bytes()


def hmac_pseudonym(user_id: str) -> str:
    """``pid_`` + base32(HMAC-SHA256(salt, user_id)).

    Base32 rather than hex so the value cannot contain a run of eight digits:
    the privacy scan flags those as Telegram-id shaped, and a guard that fires
    on its own pseudonyms is a guard that gets deleted.
    """
    if not isinstance(user_id, str) or not user_id:
        raise RecordError("user_id must be a non-empty string")
    digest = hmac.new(_salt(), user_id.encode("utf-8"), hashlib.sha256).digest()
    b32 = base64.b32encode(digest).decode("ascii").rstrip("=").lower()
    return "pid_" + b32[:26]


def error_signature_for(gate_errors: Sequence[str]) -> Optional[str]:
    """Stable 16-hex signature of the first error, numbers and paths stripped.

    Attempts are deduplicated and counted by this signature, so it is computed
    from ``gate_errors`` rather than supplied: two callers normalising slightly
    differently split one failure mode into two.
    """
    if not gate_errors:
        return None
    head = gate_errors[0]
    head = re.sub(r"(/[\w.\-]+)+", "<path>", head)
    head = re.sub(r"0x[0-9a-fA-F]+", "<addr>", head)
    head = re.sub(r"\d+", "<n>", head)
    head = re.sub(r"\s+", " ", head).strip().lower()
    return hashlib.sha256(head.encode("utf-8")).hexdigest()[:16]


_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]")
_LATIN_TOKEN_RE = re.compile(r"[a-z0-9]+")
# _CJK_RE above is kana (U+3040-30FF) plus the Han blocks (U+3400-4DBF,
# U+4E00-9FFF, U+F900-FAFF): the scripts our users actually write in.


def verbatim_ngrams(text: str, n: int = 8) -> set:
    """N-grams used for verbatim-quote detection, in two streams.

    * CJK: n *characters* of the CJK-only stream (punctuation, latin and digits
      dropped). Chinese has no whitespace, so eight contiguous CJK characters is
      a quote, not a coincidence.
    * Latin: n whitespace tokens. Eight consecutive characters of English is
      "exclude " — worthless as a threshold — so words are the unit there.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    grams = set()
    cjk = "".join(_CJK_RE.findall(text or ""))
    for i in range(len(cjk) - n + 1):
        grams.add("zh:" + cjk[i:i + n])
    tokens = _LATIN_TOKEN_RE.findall((text or "").lower())
    for i in range(len(tokens) - n + 1):
        grams.add("en:" + " ".join(tokens[i:i + n]))
    return grams


def verbatim_overlap(a: str, b: str, n: int = 8) -> set:
    """Shared n-grams between two texts; empty set means no verbatim reuse."""
    return verbatim_ngrams(a, n) & verbatim_ngrams(b, n)


class _Sentinel:
    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self._name


#: Explicit opt-out for the verbatim check, for synthetic cases that have no
#: source text. A sentinel rather than ``None`` so "I forgot" and "there is
#: genuinely no user text" cannot look the same at the call site.
NO_SOURCE_TEXT = _Sentinel("NO_SOURCE_TEXT")


def _coerce_enum(value: Any, enum_cls: type, field_name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError:
        raise RecordError(
            "%s=%r is not a member of %s; the vocabulary is closed — add the "
            "member to tools/nl2yaml/schema.py and schemas/common.schema.json in "
            "the same change" % (field_name, value, enum_cls.__name__)) from None


def _coerce_enum_list(values: Any, enum_cls: type, field_name: str) -> list:
    if values is None:
        return []
    return [_coerce_enum(v, enum_cls, field_name) for v in values]


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RecordError(msg)


def _derive(owner: Any, name: str, computed: Any) -> None:
    """Set a computed field, or raise if the caller supplied something else.

    Round-tripping a record through JSON re-supplies the computed value, so an
    equal value is accepted; an unequal one is a caller trying to decide
    something this module decides.
    """
    given = getattr(owner, name)
    if given is None:
        setattr(owner, name, computed)
        return
    if given != computed:
        raise ComputedFieldConflict(
            "%s.%s is computed: caller passed %r, record implies %r"
            % (type(owner).__name__, name, given, computed))


# ---------------------------------------------------------------------------
# nested records
# ---------------------------------------------------------------------------

@dataclass
class Condition:
    """One atomic requirement pulled out of the user's message.

    The public form is structured only. ``quote`` — the words the user actually
    used — lives in :class:`InternalCaseRecord.conditions_with_quotes` and is
    rejected here by name.
    """

    cid: str
    polarity: Polarity
    subject: str
    operator: Operator
    value: Union[None, bool, int, float, str, List[Union[str, int, float]]] = None
    unit: Optional[Unit] = None
    scope: Scope = Scope.UNIVERSE
    timeframe: Optional[str] = None
    measurability: Measurability = Measurability.QUANTIFIED
    is_ranking: bool = False
    rank_direction: Optional[RankDirection] = None
    #: what data shape this needs — see :class:`EvaluationGranularity`
    granularity: EvaluationGranularity = EvaluationGranularity.UNIVERSE
    #: set when converting requires deciding something the user left open
    ambiguity_type: Optional[AmbiguityType] = None
    #: cids this condition cannot hold simultaneously with
    conflicts_with: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.polarity = _coerce_enum(self.polarity, Polarity, "polarity")
        self.operator = _coerce_enum(self.operator, Operator, "operator")
        self.unit = _coerce_enum(self.unit, Unit, "unit")
        self.scope = _coerce_enum(self.scope, Scope, "scope")
        self.measurability = _coerce_enum(self.measurability, Measurability, "measurability")
        self.rank_direction = _coerce_enum(self.rank_direction, RankDirection, "rank_direction")
        self.granularity = _coerce_enum(self.granularity, EvaluationGranularity, "granularity")
        self.ambiguity_type = _coerce_enum(self.ambiguity_type, AmbiguityType, "ambiguity_type")
        _require(bool(self.cid), "condition cid must be non-empty")
        _require(bool(self.subject), "condition %s: subject must be non-empty" % self.cid)
        _require(all(isinstance(c, str) and c for c in self.conflicts_with),
                 "condition %s: conflicts_with must be a list of cids" % self.cid)
        _require(self.cid not in self.conflicts_with,
                 "condition %s: cannot conflict with itself" % self.cid)
        # A declared conflict is a two-sided fact. Recording it on one side only
        # lets a converter satisfy the other side and report no ambiguity at all.
        _require(not self.conflicts_with or self.ambiguity_type is AmbiguityType.CONFLICT,
                 "condition %s: conflicts_with is set, so ambiguity_type must be "
                 "'conflict' — a conflict that is not labelled as one gets "
                 "silently resolved in favour of whichever side is expressible"
                 % self.cid)

        numeric = isinstance(self.value, (int, float)) and not isinstance(self.value, bool)
        if self.measurability is Measurability.QUANTIFIED:
            _require(self.operator is not Operator.UNSPECIFIED,
                     "condition %s: quantified condition cannot use operator=unspecified — "
                     "that combination is how a vague ask becomes a made-up threshold"
                     % self.cid)
            _require(self.operator in (Operator.EXISTS, Operator.ABSENT) or self.value is not None,
                     "condition %s: quantified condition needs a value" % self.cid)
            _require(not numeric or self.unit is not None,
                     "condition %s: numeric value %r needs a unit ('> 2 million' has been "
                     "read as 2 and as 2e6)" % (self.cid, self.value))
        if self.measurability is Measurability.UNMEASURABLE:
            _require(self.operator is Operator.UNSPECIFIED,
                     "condition %s: unmeasurable condition must use operator=unspecified"
                     % self.cid)
        _require(self.is_ranking == (self.rank_direction is not None),
                 "condition %s: rank_direction is set iff is_ranking is true" % self.cid)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Condition":
        return cls(**dict(data))


@dataclass
class CapabilityEntry:
    """Adjudication of one condition against the current block vocabulary."""

    cid: str
    verdict: CapabilityVerdict
    gap_id: Optional[GapId] = None

    def __post_init__(self) -> None:
        self.verdict = _coerce_enum(self.verdict, CapabilityVerdict, "verdict")
        self.gap_id = _coerce_enum(self.gap_id, GapId, "gap_id")
        needs_gap = self.verdict in (CapabilityVerdict.PROXY, CapabilityVerdict.UNSUPPORTED)
        _require(needs_gap == (self.gap_id is not None),
                 "capability %s: verdict=%s %s a gap_id (a proxy with no named gap is "
                 "exactly how the TradFi basket shipped)"
                 % (self.cid, self.verdict.value, "requires" if needs_gap else "must not have"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapabilityEntry":
        return cls(**dict(data))


@dataclass
class IntentSlots:
    """Fixed slot set. ``additionalProperties: false`` in the JSON Schema.

    Closed rather than a free dict so nobody can park ``{"raw": "<user text>"}``
    in here — the one place a free-form object would have been convenient is the
    one place it would leak.
    """

    intent: Intent
    secondary_intents: List[Intent] = field(default_factory=list)
    market_type: Optional[MarketType] = None
    direction: Optional[Direction] = None
    symbols: List[str] = field(default_factory=list)
    universe_scope: Optional[UniverseScope] = None
    primary_timeframe: Optional[str] = None
    wants_automation: bool = False
    wants_backtest: bool = False
    wants_alerts: bool = False
    wants_ranking: bool = False
    capital_usd: Optional[float] = None
    leverage: Optional[float] = None

    def __post_init__(self) -> None:
        self.intent = _coerce_enum(self.intent, Intent, "intent")
        self.secondary_intents = _coerce_enum_list(
            self.secondary_intents, Intent, "secondary_intents")
        self.market_type = _coerce_enum(self.market_type, MarketType, "market_type")
        self.direction = _coerce_enum(self.direction, Direction, "direction")
        self.universe_scope = _coerce_enum(self.universe_scope, UniverseScope, "universe_scope")
        _require(self.intent not in self.secondary_intents,
                 "intent %s is repeated in secondary_intents" % self.intent.value)
        for sym in self.symbols:
            _require(bool(re.match(r"^[A-Z0-9]{2,20}$", sym)),
                     "symbol %r is not exchange-shaped (upper-case alphanumerics)" % sym)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IntentSlots":
        return cls(**dict(data))


@dataclass
class GoldSpec:
    role: GoldRole
    yaml_text: str
    yaml_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        self.role = _coerce_enum(self.role, GoldRole, "role")
        _require(isinstance(self.yaml_text, str) and self.yaml_text.strip() != "",
                 "gold spec %s: yaml_text must be non-empty" % self.role)
        _derive(self, "yaml_sha256", sha256_hex(self.yaml_text))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GoldSpec":
        return cls(**dict(data))


@dataclass
class ConditionVerdict:
    """Did the *output* of a spec honour condition ``cid``?

    Structured on purpose — no free-text evidence field. Evidence would be the
    natural place for "5/5 candidates were TradFi", and also the natural place
    for a copy of the user's sentence.
    """

    cid: str
    status: ConditionCheckStatus
    checked_by: CheckedBy

    def __post_init__(self) -> None:
        self.status = _coerce_enum(self.status, ConditionCheckStatus, "status")
        self.checked_by = _coerce_enum(self.checked_by, CheckedBy, "checked_by")
        if self.status in (ConditionCheckStatus.SATISFIED, ConditionCheckStatus.VIOLATED):
            _require(self.checked_by is not CheckedBy.STATIC,
                     "condition %s: %s cannot be established statically — reading the YAML "
                     "tells you what it asked for, not what came out"
                     % (self.cid, self.status.value))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConditionVerdict":
        return cls(**dict(data))


@dataclass
class GoldDisagreement:
    n_annotators: int
    disagreeing_cids: List[str] = field(default_factory=list)
    resolution: DisagreementResolution = DisagreementResolution.UNRESOLVED

    def __post_init__(self) -> None:
        self.resolution = _coerce_enum(self.resolution, DisagreementResolution, "resolution")
        _require(self.n_annotators >= 2, "gold_disagreement needs at least 2 annotators")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GoldDisagreement":
        return cls(**dict(data))


@dataclass
class BundleNode:
    """One frame in the pinned input bundle, with its row count.

    Row counts are here because the recurring failure is a frame that arrives
    empty or without a column: the run succeeds, the strategy silently stops
    consulting that source, and the numbers still look plausible.
    """

    node: str
    rows: int
    interval: Optional[str] = None
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None
    sha256: Optional[str] = None

    def __post_init__(self) -> None:
        _require(self.rows >= 0, "bundle node %s: rows must be >= 0" % self.node)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BundleNode":
        return cls(**dict(data))


@dataclass
class Candidate:
    """One emitted selection entry, embedded in full in the run record."""

    rank: int
    symbol: str
    side: CandidateSide
    score: Optional[float] = None
    attributes: Dict[str, Union[None, bool, int, float, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.side = _coerce_enum(self.side, CandidateSide, "side")
        _require(self.rank >= 1, "candidate rank starts at 1, got %r" % (self.rank,))
        for key, value in self.attributes.items():
            _require(isinstance(value, (bool, int, float, str)) or value is None,
                     "candidate attribute %r must be a scalar, got %r" % (key, type(value)))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Candidate":
        return cls(**dict(data))


# ---------------------------------------------------------------------------
# cases.jsonl
# ---------------------------------------------------------------------------

@dataclass
class CaseRecord:
    """One user request, its structured conditions, and its gold spec(s).

    Public record: safe for the repo. Identity is a pseudonym, the text is a
    pair of hashes, and the conditions are structure without quotes.
    """

    # --- identity ---
    case_id: str
    text_sha256: str
    canon_sha256: str
    dup_cluster_id: str
    dup_count: int
    preset_case: bool
    pseudonym_id: Optional[str]
    lang: Lang
    zh_variant: Optional[ZhVariant]
    month: str
    # --- provenance ---
    mining_source: MiningSource
    source_snapshot_id: str
    text_provenance: TextProvenance
    # --- input ---
    conditions: List[Condition] = field(default_factory=list)
    intent_slots: Optional[IntentSlots] = None
    capability_map: List[CapabilityEntry] = field(default_factory=list)
    tier: Optional[CaseTier] = None
    text_len_chars: int = 0
    # --- prompt context ---
    playbook_version: Optional[str] = None
    playbook_sha256: Optional[str] = None
    fewshot_set_id: Optional[str] = None
    fewshot_case_ids: List[str] = field(default_factory=list)
    block_vocab_id: Optional[str] = None
    block_refs: List[str] = field(default_factory=list)
    block_registry_sha: Optional[str] = None
    schema_sha: Optional[str] = None
    system_prompt_sha256: Optional[str] = None
    prompt_tokens_by_component: Dict[str, int] = field(default_factory=dict)
    converter_model: Optional[str] = None
    temperature: Optional[float] = None
    seed: Optional[int] = None
    repo_git_sha: Optional[str] = None
    # --- gold ---
    gold_specs: List[GoldSpec] = field(default_factory=list)
    gold_source: GoldSource = GoldSource.NONE
    gold_verification_level: GoldVerificationLevel = GoldVerificationLevel.UNPARSED
    gold_condition_verdicts: List[ConditionVerdict] = field(default_factory=list)
    gold_is_refusal: bool = False
    refusal_gap_ids: List[GapId] = field(default_factory=list)
    gold_invalidated_by_gap_closure: bool = False
    human_reviewed_by: Optional[str] = None
    human_reviewed_at: Optional[str] = None
    human_edit_diff: Optional[str] = None
    gold_disagreement: Optional[GoldDisagreement] = None
    # --- execution ---
    resolution_path: Optional[ResolutionPath] = None
    # --- computed (never supplied; see _derive) ---
    n_conditions: Optional[int] = None
    n_quantified: Optional[int] = None
    scope_counts: Optional[Dict[str, int]] = None
    gold_unverifiable_cids: Optional[List[str]] = None
    proxy_used: Optional[bool] = None
    proxy_cids: Optional[List[str]] = None
    gold_eligible_for_sft: Optional[bool] = None
    gold_eligible_loose: Optional[bool] = None

    #: Field names this class computes. Pinned by tests; a caller may only pass
    #: a value that equals what the record implies.
    COMPUTED_FIELDS = ("n_conditions", "n_quantified", "scope_counts",
                       "gold_unverifiable_cids", "proxy_used", "proxy_cids",
                       "gold_eligible_for_sft", "gold_eligible_loose")

    def __post_init__(self) -> None:
        self.mining_source = _coerce_enum(self.mining_source, MiningSource, "mining_source")
        self.lang = _coerce_enum(self.lang, Lang, "lang")
        # Strict: the export writes "" for English rows, and "" must be turned
        # into None by the loader rather than coerced here — a writer that maps
        # empty strings for you also maps the ones that were a bug.
        self.zh_variant = _coerce_enum(self.zh_variant, ZhVariant, "zh_variant")
        self.text_provenance = _coerce_enum(self.text_provenance, TextProvenance,
                                            "text_provenance")
        self.tier = _coerce_enum(self.tier, CaseTier, "tier")
        self.gold_source = _coerce_enum(self.gold_source, GoldSource, "gold_source")
        self.resolution_path = _coerce_enum(self.resolution_path, ResolutionPath,
                                            "resolution_path")
        self.refusal_gap_ids = _coerce_enum_list(self.refusal_gap_ids, GapId, "refusal_gap_ids")
        if not isinstance(self.gold_verification_level, GoldVerificationLevel):
            try:
                self.gold_verification_level = GoldVerificationLevel(
                    int(self.gold_verification_level))
            except (ValueError, TypeError):
                raise RecordError(
                    "gold_verification_level=%r is not one of 0..5"
                    % (self.gold_verification_level,)) from None
        self.conditions = [c if isinstance(c, Condition) else Condition.from_dict(c)
                           for c in self.conditions]
        self.capability_map = [c if isinstance(c, CapabilityEntry)
                               else CapabilityEntry.from_dict(c) for c in self.capability_map]
        self.gold_specs = [g if isinstance(g, GoldSpec) else GoldSpec.from_dict(g)
                           for g in self.gold_specs]
        self.gold_condition_verdicts = [
            v if isinstance(v, ConditionVerdict) else ConditionVerdict.from_dict(v)
            for v in self.gold_condition_verdicts]
        if self.intent_slots is not None and not isinstance(self.intent_slots, IntentSlots):
            self.intent_slots = IntentSlots.from_dict(self.intent_slots)
        if self.gold_disagreement is not None and not isinstance(
                self.gold_disagreement, GoldDisagreement):
            self.gold_disagreement = GoldDisagreement.from_dict(self.gold_disagreement)

        self._check_identity()
        self._check_adjudication()
        self._derive_counts()
        self._derive_gold_gates()
        self._check_gold_consistency()

    # -- identity ---------------------------------------------------------
    def _check_identity(self) -> None:
        _require(bool(ULID_RE.match(self.case_id) or UUID4_RE.match(self.case_id)),
                 "case_id %r must be a ULID or a lowercase uuid4" % (self.case_id,))
        for name in ("text_sha256", "canon_sha256"):
            _require(bool(SHA256_RE.match(getattr(self, name))),
                     "%s must be 64 lowercase hex chars" % name)
        _require(self.dup_cluster_id.startswith("dup_"),
                 "dup_cluster_id %r must start with 'dup_'" % (self.dup_cluster_id,))
        _require(self.dup_count >= 1,
                 "dup_count counts this case too, so it is >= 1; got %r" % (self.dup_count,))
        _require(bool(re.match(r"^\d{4}-\d{2}$", self.month)),
                 "month must be YYYY-MM, got %r" % (self.month,))
        _require((self.lang is Lang.ZH) == (self.zh_variant is not None),
                 "zh_variant is set iff lang=zh (got lang=%s, zh_variant=%r)"
                 % (self.lang.value, self.zh_variant))
        _require(self.text_len_chars >= 0, "text_len_chars must be >= 0")
        # A pseudonym is present exactly when a real user is behind the text.
        # Synthetic rows with a pseudonym would inflate per-user statistics with
        # people who never asked anything.
        has_user = self.text_provenance is not TextProvenance.SYNTHETIC
        _require(has_user == (self.pseudonym_id is not None),
                 "pseudonym_id is required for provenance=%s and forbidden for synthetic"
                 % self.text_provenance.value)
        if self.pseudonym_id is not None:
            _require(bool(PSEUDONYM_RE.match(self.pseudonym_id)),
                     "pseudonym_id %r must be hmac_pseudonym() output ('pid_' + 26 base32 "
                     "chars); a raw user_id here is the leak this field exists to prevent"
                     % (self.pseudonym_id,))
        for key in self.prompt_tokens_by_component:
            _require(self.prompt_tokens_by_component[key] >= 0,
                     "prompt_tokens_by_component[%r] must be >= 0" % key)

    # -- adjudication -----------------------------------------------------
    def _check_adjudication(self) -> None:
        cids = [c.cid for c in self.conditions]
        _require(len(cids) == len(set(cids)), "duplicate condition cid in %r" % (cids,))
        mapped = [e.cid for e in self.capability_map]
        _require(len(mapped) == len(set(mapped)),
                 "duplicate cid in capability_map: %r" % (mapped,))
        if self.capability_map:
            _require(set(mapped) == set(cids),
                     "capability_map must adjudicate exactly the conditions present "
                     "(missing=%s, unknown=%s); an unadjudicated condition is a condition "
                     "that gets silently dropped"
                     % (sorted(set(cids) - set(mapped)), sorted(set(mapped) - set(cids))))
        for verdict in self.gold_condition_verdicts:
            _require(verdict.cid in set(cids),
                     "gold_condition_verdicts references unknown cid %r" % (verdict.cid,))

        verdicts = {e.verdict for e in self.capability_map}
        if self.tier is None:
            _require(not self.capability_map,
                     "tier is required once capability_map is filled in")
        elif self.tier is CaseTier.T4_OUT_OF_SCOPE:
            pass  # a judgement about intent, not about conditions
        elif self.tier is CaseTier.T0_UNDERSPECIFIED:
            _require(not self.conditions,
                     "tier=t0_underspecified but %d conditions were extracted"
                     % len(self.conditions))
        else:
            _require(bool(self.conditions), "tier=%s requires conditions" % self.tier.value)
            if self.capability_map:
                expected = (CaseTier.T3_BLOCKED_BY_GAP
                            if CapabilityVerdict.UNSUPPORTED in verdicts
                            else CaseTier.T2_PROXY_NEEDED
                            if CapabilityVerdict.PROXY in verdicts
                            else CaseTier.T1_EXPRESSIBLE)
                _require(self.tier is expected,
                         "tier=%s contradicts capability_map (implies %s)"
                         % (self.tier.value, expected.value))
        if not self.conditions and self.tier is not None:
            _require(self.tier in (CaseTier.T0_UNDERSPECIFIED, CaseTier.T4_OUT_OF_SCOPE),
                     "tier=%s with no conditions" % self.tier.value)

    # -- derived counts ---------------------------------------------------
    def _derive_counts(self) -> None:
        _derive(self, "n_conditions", len(self.conditions))
        _derive(self, "n_quantified",
                sum(1 for c in self.conditions
                    if c.measurability is Measurability.QUANTIFIED))
        counts: Dict[str, int] = {}
        for cond in self.conditions:
            counts[cond.scope.value] = counts.get(cond.scope.value, 0) + 1
        _derive(self, "scope_counts", {k: counts[k] for k in sorted(counts)})

    # -- the gates --------------------------------------------------------
    def _derive_gold_gates(self) -> None:
        """Compute ``proxy_used`` and both eligibility flags.

        ``proxy_used`` is a union of two sources, not just the post-hoc output
        check:

        * ``capability_map`` verdict ``proxy`` — the converter already *knew* it
          was substituting a correlated field. Refusals are excluded because a
          refusal does not encode the proxy; it names the gap instead.
        * ``gold_condition_verdicts`` status ``proxied`` — the output check found
          a substitution.

        Then, exactly:

        ``gold_eligible_for_sft = level >= 5 and gold_unverifiable_cids == []
        and not proxy_used``
        ``gold_eligible_loose  = level >= 4 and not proxy_used``

        🔴 ``proxy_used=True`` forces **both** to False at any level. This is not
        a conservative default that can be relaxed once "the proxy is good
        enough": a proxy passes ``validate``, passes ``run`` and produces a
        plausible basket, so there is no gate further downstream — not the
        trainer, not the evaluator, not review — that can notice it. Train on it
        once and the substitution is in the weights permanently, and the model
        will then apply it to requests where the correlation does not hold. The
        five-TradFi-perp basket is what that looks like in production.
        """
        proxy_from_adjudication = [] if self.gold_is_refusal else [
            e.cid for e in self.capability_map if e.verdict is CapabilityVerdict.PROXY]
        proxy_from_output = [v.cid for v in self.gold_condition_verdicts
                             if v.status is ConditionCheckStatus.PROXIED]
        _derive(self, "proxy_cids", sorted(set(proxy_from_adjudication) | set(proxy_from_output)))
        _derive(self, "proxy_used", bool(self.proxy_cids))
        _derive(self, "gold_unverifiable_cids",
                sorted(v.cid for v in self.gold_condition_verdicts
                       if v.status is ConditionCheckStatus.UNVERIFIABLE))

        level = int(self.gold_verification_level)
        _derive(self, "gold_eligible_for_sft",
                level >= int(GoldVerificationLevel.ALL_CHECKABLE_CONDITIONS_SATISFIED)
                and self.gold_unverifiable_cids == []
                and not self.proxy_used)
        _derive(self, "gold_eligible_loose",
                level >= int(GoldVerificationLevel.EXECUTED_ON_PINNED_BUNDLE)
                and not self.proxy_used)

    # -- gold consistency -------------------------------------------------
    def _check_gold_consistency(self) -> None:
        level = int(self.gold_verification_level)
        if self.gold_source is GoldSource.NONE:
            _require(not self.gold_specs,
                     "gold_source=none but %d gold spec(s) present" % len(self.gold_specs))
            _require(level == 0,
                     "gold_source=none requires level 0, got %d" % level)
        else:
            _require(bool(self.gold_specs),
                     "gold_source=%s requires at least one gold spec" % self.gold_source.value)
        if self.gold_specs:
            primaries = [g for g in self.gold_specs if g.role is GoldRole.PRIMARY]
            _require(len(primaries) == 1,
                     "gold_specs needs exactly one role=primary (the SFT target); got %d"
                     % len(primaries))
        else:
            _require(level == 0, "level %d with no gold spec" % level)

        if self.gold_source is GoldSource.ATTEMPT_K:
            # Failure mode 1+2 combined: an attempt becomes gold only after its
            # intent has been reconciled against the conditions. "It ran" (level
            # 2/4) is the exact signal that burned us; keep gold_source=none
            # until reconciliation instead of promoting on green.
            _require(level >= int(GoldVerificationLevel.INTENT_RECONCILED),
                     "gold_source=attempt_k requires level >= 3 (intent_reconciled); got "
                     "%d. An attempt that merely parsed/ran is not a label — it only "
                     "exists because earlier attempts failed, so promoting on green both "
                     "bakes in wrong answers and biases the set towards hard cases."
                     % level)
        if self.gold_source is GoldSource.HUMAN_EDITED:
            _require(bool(self.human_reviewed_by) and self.human_edit_diff is not None,
                     "gold_source=human_edited requires human_reviewed_by and "
                     "human_edit_diff (otherwise the edit is unauditable)")
        if self.human_reviewed_by or self.human_reviewed_at:
            _require(bool(self.human_reviewed_by) and bool(self.human_reviewed_at),
                     "human_reviewed_by and human_reviewed_at go together")

        if level >= int(GoldVerificationLevel.ALL_CHECKABLE_CONDITIONS_SATISFIED):
            violated = [v.cid for v in self.gold_condition_verdicts
                        if v.status is ConditionCheckStatus.VIOLATED]
            _require(not violated,
                     "level 5 means every checkable condition was satisfied, but %s "
                     "%s violated — this is the shape of the run that emitted five TradFi "
                     "perps for a user who asked to exclude TradFi"
                     % (violated, "is" if len(violated) == 1 else "are"))
            _require(bool(self.gold_condition_verdicts) or not self.conditions,
                     "level 5 requires per-condition verdicts; with none recorded nothing "
                     "was actually checked")

        _require(self.gold_is_refusal == bool(self.refusal_gap_ids),
                 "gold_is_refusal and refusal_gap_ids must agree: a refusal has to name "
                 "the gaps, or it can neither be audited nor revisited when they close")
        if self.gold_is_refusal:
            _require(not [v for v in self.gold_condition_verdicts
                          if v.status is ConditionCheckStatus.PROXIED],
                     "a refusal cannot also proxy a condition")
        if self.gold_invalidated_by_gap_closure:
            # The level describes verification against the *current* vocabulary.
            # When a gap closes, the old gold was never run against the new
            # blocks, so keeping level 4/5 asserts a run that never happened.
            _require(level <= int(GoldVerificationLevel.INTENT_RECONCILED),
                     "gold_invalidated_by_gap_closure=True requires level <= 3: the gold "
                     "has not been re-verified against the vocabulary that closed the gap")
        if self.gold_disagreement is not None and (
                self.gold_disagreement.resolution is DisagreementResolution.UNRESOLVED):
            _require(level <= int(GoldVerificationLevel.DRY_RAN),
                     "an unresolved gold_disagreement caps level at 2: intent "
                     "reconciliation is precisely what the annotators disagree about")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CaseRecord":
        return cls(**dict(data))


# ---------------------------------------------------------------------------
# attempts.jsonl
# ---------------------------------------------------------------------------

@dataclass
class AttemptRecord:
    """One model call for one case.

    The rendered prompt is **not** stored — it embeds the user's message
    verbatim. Only ``prompt_sha256`` is, which is also what the DPO check needs.
    """

    case_id: str
    attempt_index: int
    sampling_purpose: SamplingPurpose
    prompt_sha256: str
    yaml_text: str
    stopped_at_gate: Gate
    yaml_sha256: Optional[str] = None
    #: Verbatim engine errors, never summarised: the summary is always written by
    #: someone who has already decided what the bug is, and the discarded detail
    #: (which column, which block, which frame) is the part that identifies the
    #: failure mode.
    gate_errors: List[str] = field(default_factory=list)
    condition_verdicts: List[ConditionVerdict] = field(default_factory=list)
    latency_ms: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_prompt_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    feedback_given_to_next_attempt: Optional[str] = None
    defect_class: DefectClass = DefectClass.NONE
    #: sha256 of the prompt of the *chosen* sample this attempt is paired with.
    #: Required to evaluate ``is_valid_dpo_negative``; None when unpaired.
    dpo_chosen_prompt_sha256: Optional[str] = None
    # --- computed ---
    error_signature: Optional[str] = None
    is_valid_dpo_negative: Optional[bool] = None

    COMPUTED_FIELDS = ("error_signature", "is_valid_dpo_negative")

    def __post_init__(self) -> None:
        self.sampling_purpose = _coerce_enum(self.sampling_purpose, SamplingPurpose,
                                             "sampling_purpose")
        self.stopped_at_gate = _coerce_enum(self.stopped_at_gate, Gate, "stopped_at_gate")
        self.defect_class = _coerce_enum(self.defect_class, DefectClass, "defect_class")
        self.condition_verdicts = [
            v if isinstance(v, ConditionVerdict) else ConditionVerdict.from_dict(v)
            for v in self.condition_verdicts]

        _require(self.attempt_index >= 1, "attempt_index starts at 1")
        _require(bool(SHA256_RE.match(self.prompt_sha256)),
                 "prompt_sha256 must be 64 lowercase hex chars")
        if self.dpo_chosen_prompt_sha256 is not None:
            _require(bool(SHA256_RE.match(self.dpo_chosen_prompt_sha256)),
                     "dpo_chosen_prompt_sha256 must be 64 lowercase hex chars")
        _derive(self, "yaml_sha256", sha256_hex(self.yaml_text))
        _derive(self, "error_signature", error_signature_for(self.gate_errors))

        passed = self.stopped_at_gate is Gate.PASSED
        _require(passed != bool(self.gate_errors),
                 "stopped_at_gate=%s and gate_errors=%d entries disagree"
                 % (self.stopped_at_gate.value, len(self.gate_errors)))
        if self.sampling_purpose is SamplingPurpose.REPAIR:
            _require(self.attempt_index >= 2,
                     "sampling_purpose=repair on attempt 1: there is nothing to repair")
        for name in ("latency_ms", "prompt_tokens", "completion_tokens",
                     "cached_prompt_tokens"):
            value = getattr(self, name)
            _require(value is None or value >= 0, "%s must be >= 0" % name)
        _require(self.cost_usd is None or self.cost_usd >= 0, "cost_usd must be >= 0")
        if self.cached_prompt_tokens is not None and self.prompt_tokens is not None:
            _require(self.cached_prompt_tokens <= self.prompt_tokens,
                     "cached_prompt_tokens (%s) > prompt_tokens (%s): cached tokens are a "
                     "subset of the prompt, and double counting them understates cost"
                     % (self.cached_prompt_tokens, self.prompt_tokens))
        self._derive_dpo()

    def _derive_dpo(self) -> None:
        """Compute ``is_valid_dpo_negative``.

        DPO requires chosen and rejected to answer **the same prompt**. The
        tempting pair — attempt 1 rejected, attempt 2 chosen — is invalid: the
        attempt-2 prompt carries the repair feedback, so the two samples answer
        different questions and the gradient teaches "follow the correction I
        already gave you" rather than "get it right first time". The only test
        that catches this is prompt identity, so it is the rule here.

        Two further exclusions:

        * ``parse_error`` — teaches YAML formatting, not strategy semantics, and
          swamps the preference set with a defect a grammar constraint already
          handles.
        * ``bundle_insufficient`` — the data was missing. Nothing the model
          emitted caused it, so penalising that output teaches noise.
        """
        if self.dpo_chosen_prompt_sha256 is None:
            _derive(self, "is_valid_dpo_negative", False)
            return
        _derive(self, "is_valid_dpo_negative",
                self.dpo_chosen_prompt_sha256 == self.prompt_sha256
                and self.stopped_at_gate not in (Gate.PARSE_ERROR, Gate.BUNDLE_INSUFFICIENT)
                and self.stopped_at_gate is not Gate.PASSED)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AttemptRecord":
        return cls(**dict(data))


def validate_attempt_chain(attempts: Sequence[AttemptRecord]) -> None:
    """Check a case's attempts as a sequence. Raises on the first problem.

    Per-record validation cannot see that ``feedback_given_to_next_attempt`` was
    written but never reached the next prompt — the prompt hash would be
    unchanged. That silent no-op makes a repair look like a fresh sample and
    turns the whole repair loop into a measurement of temperature.
    """
    if not attempts:
        return
    case_ids = {a.case_id for a in attempts}
    _require(len(case_ids) == 1, "validate_attempt_chain got mixed case_ids: %r" % (case_ids,))
    ordered = sorted(attempts, key=lambda a: a.attempt_index)
    _require([a.attempt_index for a in ordered] == list(range(1, len(ordered) + 1)),
             "attempt_index must be 1..n with no gaps, got %r"
             % [a.attempt_index for a in ordered])
    for prev, nxt in zip(ordered, ordered[1:]):
        if prev.feedback_given_to_next_attempt:
            _require(nxt.prompt_sha256 != prev.prompt_sha256,
                     "attempt %d recorded feedback but attempt %d has the same "
                     "prompt_sha256 — the feedback never reached the prompt"
                     % (prev.attempt_index, nxt.attempt_index))
        else:
            _require(nxt.sampling_purpose is not SamplingPurpose.REPAIR,
                     "attempt %d is a repair but attempt %d recorded no feedback"
                     % (nxt.attempt_index, prev.attempt_index))


# ---------------------------------------------------------------------------
# gaps.jsonl
# ---------------------------------------------------------------------------

@dataclass
class GapRecord:
    """A capability the vocabulary lacks, with how much demand sits behind it."""

    gap_id: GapId
    title: str
    needed_capability: str
    hit_count: int
    dup_weighted_count: int
    example_case_ids: List[str] = field(default_factory=list)
    status: GapStatus = GapStatus.OPEN
    closed_by_commit: Optional[str] = None

    #: Five is enough to read the gap and small enough that the file stays a
    #: prioritisation artifact rather than a second copy of the corpus.
    MAX_EXAMPLES = 5

    def __post_init__(self) -> None:
        self.gap_id = _coerce_enum(self.gap_id, GapId, "gap_id")
        self.status = _coerce_enum(self.status, GapStatus, "status")
        _require(bool(self.title) and bool(self.needed_capability),
                 "gap %s: title and needed_capability must be non-empty" % self.gap_id.value)
        _require(self.hit_count >= 0, "hit_count must be >= 0")
        _require(self.dup_weighted_count >= self.hit_count,
                 "gap %s: dup_weighted_count (%d) < hit_count (%d). The weighted count "
                 "counts every duplicate occurrence, so it is always >= the number of "
                 "distinct cases; ranking gaps by the unweighted count is what hides the "
                 "preset card that appears 393 times"
                 % (self.gap_id.value, self.dup_weighted_count, self.hit_count))
        _require(len(self.example_case_ids) <= self.MAX_EXAMPLES,
                 "gap %s: at most %d example_case_ids, got %d"
                 % (self.gap_id.value, self.MAX_EXAMPLES, len(self.example_case_ids)))
        _require(len(set(self.example_case_ids)) == len(self.example_case_ids),
                 "gap %s: duplicate example_case_ids" % self.gap_id.value)
        _require((self.status is GapStatus.CLOSED) == (self.closed_by_commit is not None),
                 "gap %s: status=closed requires closed_by_commit and vice versa — a gap "
                 "closed by nothing in particular cannot be re-opened when it turns out "
                 "not to be closed" % self.gap_id.value)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GapRecord":
        return cls(**dict(data))


# ---------------------------------------------------------------------------
# runs.jsonl
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    """One execution of one spec against one pinned bundle.

    ``candidates`` is embedded in full rather than referenced: it is small, and
    it is the object of every argument about whether a run answered the
    question. A pointer to a basket that has since been regenerated settles
    nothing.
    """

    run_id: str
    bundle_id: str
    bundle_sha256: str
    bundle_decision_time: str
    bundle_snapshot_id: str
    repo_git_sha: str
    python_version: str
    pandas_version: str
    signal_count: int
    bundle_nodes: List[BundleNode] = field(default_factory=list)
    signal_batch_sha256: Optional[str] = None
    candidates: List[Candidate] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.bundle_nodes = [n if isinstance(n, BundleNode) else BundleNode.from_dict(n)
                             for n in self.bundle_nodes]
        self.candidates = [c if isinstance(c, Candidate) else Candidate.from_dict(c)
                           for c in self.candidates]
        _require(bool(SHA256_RE.match(self.bundle_sha256)),
                 "bundle_sha256 must be 64 lowercase hex chars")
        if self.signal_batch_sha256 is not None:
            _require(bool(SHA256_RE.match(self.signal_batch_sha256)),
                     "signal_batch_sha256 must be 64 lowercase hex chars")
        _require(bool(re.match(r"^\d{4}-\d{2}-\d{2}T[\d:.]+(Z|[+-]\d{2}:?\d{2})$",
                               self.bundle_decision_time)),
                 "bundle_decision_time must be an ISO8601 instant with a timezone "
                 "(it is the point-in-time boundary; a naive local timestamp cannot be "
                 "audited), got %r" % (self.bundle_decision_time,))
        _require(self.signal_count >= 0, "signal_count must be >= 0")
        node_names = [n.node for n in self.bundle_nodes]
        _require(len(node_names) == len(set(node_names)),
                 "duplicate bundle node: %r" % (node_names,))
        if self.candidates:
            ranks = sorted(c.rank for c in self.candidates)
            _require(ranks == list(range(1, len(self.candidates) + 1)),
                     "candidate ranks must be 1..n with no gaps or ties, got %r" % (ranks,))
            symbols = [c.symbol for c in self.candidates]
            _require(len(symbols) == len(set(symbols)),
                     "duplicate candidate symbol: %r" % (symbols,))
            _require(self.signal_count == len(self.candidates),
                     "signal_count=%d but %d candidates embedded; the embedded basket has "
                     "to be the basket that was emitted or it is not evidence"
                     % (self.signal_count, len(self.candidates)))
            _require(any(n.rows > 0 for n in self.bundle_nodes),
                     "candidates produced from a bundle whose every frame is empty — the "
                     "recurring silent failure is an empty/short frame that still yields "
                     "plausible output")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunRecord":
        return cls(**dict(data))


# ---------------------------------------------------------------------------
# cases_internal.jsonl  (never in the repo)
# ---------------------------------------------------------------------------

@dataclass
class InternalCaseRecord:
    """The user-identifying half of a case. Internal store only.

    Kept in a separate file *and* a separate writer so that "does this line
    contain user text?" is answered by which function wrote it, not by
    inspection.
    """

    case_id: str
    user_text_raw: str
    user_text_redacted: str
    pseudonym_id: str
    user_id: str
    conditions_with_quotes: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require(bool(ULID_RE.match(self.case_id) or UUID4_RE.match(self.case_id)),
                 "case_id %r must be a ULID or a lowercase uuid4" % (self.case_id,))
        _require(bool(self.user_text_raw), "user_text_raw must be non-empty")
        _require(bool(PSEUDONYM_RE.match(self.pseudonym_id)),
                 "pseudonym_id %r must be hmac_pseudonym() output" % (self.pseudonym_id,))
        for entry in self.conditions_with_quotes:
            _require("cid" in entry and "quote" in entry,
                     "conditions_with_quotes entries need cid and quote")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InternalCaseRecord":
        return cls(**dict(data))


# ---------------------------------------------------------------------------
# serialisation + JSON Schema validation
# ---------------------------------------------------------------------------

RECORD_KINDS = {
    CaseRecord: "case",
    AttemptRecord: "attempt",
    GapRecord: "gap",
    RunRecord: "run",
    InternalCaseRecord: "case_internal",
}
KIND_SCHEMA_FILES = {
    "case": "case.schema.json",
    "attempt": "attempt.schema.json",
    "gap": "gap.schema.json",
    "run": "run.schema.json",
    "case_internal": "case_internal.schema.json",
}
_READERS = {
    "case": CaseRecord,
    "attempt": AttemptRecord,
    "gap": GapRecord,
    "run": RunRecord,
    "case_internal": InternalCaseRecord,
}


def _to_jsonable(value: Any) -> Any:
    """Convert to JSON-ready data, refusing types we did not plan for.

    No ``str(value)`` fallback: a stray object silently stringified into a public
    record is how a repr containing user text would get published.
    """
    if isinstance(value, Enum):
        return value.value  # IntEnum yields an int, str-Enum a str
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if is_dataclass(value):
        out = {}
        for f in fields(value):
            out[f.name] = _to_jsonable(getattr(value, f.name))
        return out
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in value]
    raise RecordError("cannot serialise %r (%s) into a record" % (value, type(value)))


def _load_registry() -> Registry:
    registry = Registry()
    for path in sorted(SCHEMA_DIR.glob("*.json")):
        contents = json.loads(path.read_text("utf-8"))
        # Keyed by bare filename so the record schemas can $ref
        # "common.schema.json#/$defs/..." and stay relocatable.
        registry = registry.with_resource(path.name, Resource.from_contents(contents))
    return registry


_REGISTRY: Optional[Registry] = None
_VALIDATORS: Dict[str, Any] = {}


def _validator(kind: str):
    global _REGISTRY
    if kind not in KIND_SCHEMA_FILES:
        raise RecordError("unknown record kind %r; known: %s"
                          % (kind, sorted(KIND_SCHEMA_FILES)))
    if kind not in _VALIDATORS:
        if _REGISTRY is None:
            _REGISTRY = _load_registry()
        schema = json.loads((SCHEMA_DIR / KIND_SCHEMA_FILES[kind]).read_text("utf-8"))
        _VALIDATORS[kind] = jsonschema.Draft202012Validator(schema, registry=_REGISTRY)
    return _VALIDATORS[kind]


def _kind_of(record: Any) -> str:
    for cls, kind in RECORD_KINDS.items():
        if isinstance(record, cls):
            return kind
    raise RecordError("not a record type: %r" % (type(record),))


def validate_record(record: Any, kind: Optional[str] = None) -> Dict[str, Any]:
    """Validate against the JSON Schema and return the JSON-ready payload.

    Raises ``jsonschema.ValidationError``. The dataclasses already enforce the
    cross-field invariants; the schema is the independent check that the *shape*
    on disk is what readers (and other languages) will get, and — because every
    record schema sets ``additionalProperties: false`` — a second, structural
    barrier against an unexpected key such as ``user_text`` reaching the file.
    """
    source = dict(record) if isinstance(record, Mapping) else record
    payload = _to_jsonable(source)
    kind = kind or _kind_of(record)
    _validator(kind).validate(payload)
    return payload


# ---------------------------------------------------------------------------
# privacy guards applied at write time
# ---------------------------------------------------------------------------

def _assert_no_private_keys(payload: Any, path: str = "") -> None:
    """Recursively reject forbidden keys anywhere in a public payload.

    Raises rather than dropping. A silent drop teaches the caller that handing
    user text to the public writer is fine, and the next field they invent will
    not be on the list.
    """
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            low = str(key).lower()
            if low in FORBIDDEN_PUBLIC_KEYS or low.startswith(FORBIDDEN_PUBLIC_KEY_PREFIXES):
                raise PrivacyError(
                    "public record carries key %r at %r; user text and per-condition "
                    "quotes belong in the internal store via write_case_internal(). "
                    "This is a raise, not a drop: dropping it would make the mistake "
                    "invisible." % (key, path + "/" + str(key)))
            _assert_no_private_keys(value, path + "/" + str(key))
    elif isinstance(payload, (list, tuple)):
        for i, value in enumerate(payload):
            _assert_no_private_keys(value, "%s[%d]" % (path, i))


def _assert_no_verbatim_quote(texts: Sequence[Tuple[str, str]],
                              source_text: Union[str, _Sentinel],
                              n: int = 8) -> None:
    """Reject any emitted YAML that shares an n-gram with the user's own words.

    ``strategy.description`` is the field a converter most often fills by
    copying the request, and it ships inside every YAML — into the repo, into
    generated docs, into signal payloads. The whole ``yaml_text`` is scanned, not
    just the description, because comments leak the same way.

    If this fires on a description you consider fine, the fix is to write the
    description in structured English, not to lower ``n``.
    """
    if isinstance(source_text, _Sentinel):
        return
    for label, text in texts:
        shared = verbatim_overlap(text, source_text, n)
        if shared:
            # The gram itself is user text: it must not go into the message, which
            # ends up in logs and CI output. Report the count and the stream only.
            streams = sorted({g.split(":", 1)[0] for g in shared})
            raise PrivacyError(
                "%s reproduces the user's wording: %d shared %d-gram(s) in stream(s) "
                "%s. Rewrite it as a structured description instead of quoting the "
                "request (the shared text is deliberately not echoed here — it is "
                "user text and this message reaches logs)"
                % (label, len(shared), n, ",".join(streams)))


def _append_jsonl(path: Path, payload: Dict[str, Any], *, mode: Optional[int] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    _require("\n" not in line, "serialised record contains a newline")
    if mode is not None and not path.exists():
        # Create with the restrictive mode *before* the first byte lands, so the
        # file is never briefly world-readable.
        os.close(os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    if mode is not None:
        os.chmod(str(path), mode)


def _as_public_record(record: Any, cls: type) -> Any:
    """Build a public record, screening a raw mapping *before* construction.

    Order matters. The realistic mistake is a dict built straight from the chat
    export — whose columns are ``user_id``, ``first_query``,
    ``user_text_excerpt`` — handed to a public writer. Constructing the dataclass
    first would raise ``TypeError: unexpected keyword argument``, which reads
    like a signature problem and invites the caller to pop the field and move on.
    Screening first produces a privacy error that names the field and says where
    it belongs.
    """
    if isinstance(record, cls):
        return record
    if isinstance(record, Mapping):
        _assert_no_private_keys(record)
        return cls.from_dict(record)
    raise RecordError("expected a %s or a mapping, got %s — %s is not a public record "
                      "type" % (cls.__name__, type(record).__name__, type(record).__name__))


def write_case(path: Union[str, Path], case: Union[CaseRecord, Mapping[str, Any]], *,
               source_text: Union[str, _Sentinel]) -> Dict[str, Any]:
    """Append one public case record. Raises on anything user-identifying.

    ``source_text`` is a required keyword and takes the user's raw text so the
    verbatim-quote check can run at write time rather than in a later sweep that
    may never happen. Pass :data:`NO_SOURCE_TEXT` only for synthetic cases; the
    record's own ``text_provenance`` is cross-checked against that choice.
    """
    case = _as_public_record(case, CaseRecord)
    if isinstance(source_text, _Sentinel):
        _require(case.text_provenance is TextProvenance.SYNTHETIC,
                 "source_text=NO_SOURCE_TEXT is only valid for text_provenance=synthetic; "
                 "case %s is %s" % (case.case_id, case.text_provenance.value))
    else:
        _require(isinstance(source_text, str) and source_text.strip() != "",
                 "source_text must be the user's raw text (or NO_SOURCE_TEXT for "
                 "synthetic cases); None is not accepted because 'I forgot' and 'there "
                 "is no user text' must not look the same")
        _require(case.text_provenance is not TextProvenance.SYNTHETIC,
                 "case %s is synthetic but a source_text was supplied" % case.case_id)
        _require(sha256_hex(source_text) == case.text_sha256,
                 "source_text does not hash to text_sha256; the verbatim check would be "
                 "run against the wrong text and would pass for the wrong reason")
    payload = validate_record(case, "case")
    _assert_no_private_keys(payload)
    _assert_no_verbatim_quote(
        [("gold_specs[%d] (%s)" % (i, g.role.value), g.yaml_text)
         for i, g in enumerate(case.gold_specs)], source_text)
    _append_jsonl(Path(path), payload)
    return payload


def write_attempt(path: Union[str, Path], attempt: Union[AttemptRecord, Mapping[str, Any]], *,
                  source_text: Union[str, _Sentinel]) -> Dict[str, Any]:
    """Append one public attempt record.

    The model's raw YAML is published here, so it gets the same verbatim check as
    a gold spec: a model that copies the request into ``description`` leaks
    through the attempt file just as readily.
    """
    attempt = _as_public_record(attempt, AttemptRecord)
    payload = validate_record(attempt, "attempt")
    _assert_no_private_keys(payload)
    _assert_no_verbatim_quote([("yaml_text", attempt.yaml_text)], source_text)
    _assert_no_verbatim_quote(
        [("gate_errors[%d]" % i, err) for i, err in enumerate(attempt.gate_errors)],
        source_text)
    _append_jsonl(Path(path), payload)
    return payload


def write_gap(path: Union[str, Path], gap: Union[GapRecord, Mapping[str, Any]]) -> Dict[str, Any]:
    gap = _as_public_record(gap, GapRecord)
    payload = validate_record(gap, "gap")
    _assert_no_private_keys(payload)
    _append_jsonl(Path(path), payload)
    return payload


def write_run(path: Union[str, Path], run: Union[RunRecord, Mapping[str, Any]]) -> Dict[str, Any]:
    run = _as_public_record(run, RunRecord)
    payload = validate_record(run, "run")
    _assert_no_private_keys(payload)
    _append_jsonl(Path(path), payload)
    return payload


def write_case_internal(path: Union[str, Path],
                        record: Union[InternalCaseRecord, Mapping[str, Any]]) -> Dict[str, Any]:
    """Append one internal record. Refuses any path outside the internal root.

    Deliberately a different function name from :func:`write_case`: the audit
    question "where can user text end up?" is then answerable with one grep.
    """
    if not isinstance(record, InternalCaseRecord):
        record = InternalCaseRecord.from_dict(record)
    target = Path(path).expanduser().absolute()
    root = internal_root()
    rel = os.path.relpath(str(target), str(root))
    if rel.startswith(os.pardir):
        raise PrivacyError(
            "refusing to write user text to %s: it is outside the internal root %s"
            % (target, root))
    payload = validate_record(record, "case_internal")
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(str(root), 0o700)
    _append_jsonl(target, payload, mode=0o600)
    return payload


def write_case_pair(public_path: Union[str, Path], internal_path: Union[str, Path],
                    case: CaseRecord, internal: InternalCaseRecord) -> None:
    """Write both halves of a case, cross-checked. The recommended entry point.

    Pairing them here means the verbatim check always has the real user text and
    the two files cannot disagree about identity — the two mistakes that a
    "write the public part now, the internal part later" flow makes easy.
    """
    if not isinstance(case, CaseRecord):
        case = CaseRecord.from_dict(case)
    if not isinstance(internal, InternalCaseRecord):
        internal = InternalCaseRecord.from_dict(internal)
    _require(case.case_id == internal.case_id,
             "case_id mismatch between public (%s) and internal (%s) record"
             % (case.case_id, internal.case_id))
    _require(case.pseudonym_id == internal.pseudonym_id,
             "pseudonym_id mismatch between public and internal record for %s"
             % case.case_id)
    _require(case.text_provenance is not TextProvenance.SYNTHETIC,
             "synthetic case %s has no internal half" % case.case_id)
    # Check before either append. Both files are append-only, so a failure
    # between the two writes would leave a half-written pair that a retry then
    # duplicates; and the internal half must never be missing for a public
    # record, or the verbatim sweep has nothing to compare against.
    _assert_no_private_keys(validate_record(case, "case"))
    _assert_no_verbatim_quote(
        [("gold_specs[%d] (%s)" % (i, g.role.value), g.yaml_text)
         for i, g in enumerate(case.gold_specs)], internal.user_text_raw)
    write_case_internal(internal_path, internal)
    write_case(public_path, case, source_text=internal.user_text_raw)


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Union[str, Path], kind: str) -> Iterator[Any]:
    cls = _READERS[kind]
    with Path(path).open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RecordError("%s:%d is not JSON: %s" % (path, lineno, exc)) from None
            _validator(kind).validate(payload)
            try:
                yield cls.from_dict(payload)
            except RecordError as exc:
                # Re-raise with the location: a mid-file invariant break is
                # otherwise very hard to find in a 50k-line file. Keep the
                # concrete class so callers can still distinguish "hand-edited a
                # computed field" from "broke an invariant".
                raise type(exc)("%s:%d %s" % (path, lineno, exc)) from None


def read_cases(path: Union[str, Path]) -> Iterator[CaseRecord]:
    """Yield validated :class:`CaseRecord`s.

    Every line is re-validated against the schema *and* re-run through the
    dataclass invariants, so a file hand-edited into an impossible state (level
    5 with a violated condition, eligibility flipped by hand) fails on read
    instead of quietly entering a training set.
    """
    return _read_jsonl(path, "case")


def read_attempts(path: Union[str, Path]) -> Iterator[AttemptRecord]:
    return _read_jsonl(path, "attempt")


def read_gaps(path: Union[str, Path]) -> Iterator[GapRecord]:
    return _read_jsonl(path, "gap")


def read_runs(path: Union[str, Path]) -> Iterator[RunRecord]:
    return _read_jsonl(path, "run")


def read_cases_internal(path: Union[str, Path]) -> Iterator[InternalCaseRecord]:
    """Read the internal store. Refuses paths outside the internal root."""
    target = Path(path).expanduser().absolute()
    rel = os.path.relpath(str(target), str(internal_root()))
    if rel.startswith(os.pardir):
        raise PrivacyError("%s is outside the internal root; internal records must not "
                           "exist there" % target)
    return _read_jsonl(target, "case_internal")


def read_internal_text(path: Union[str, Path], case_id: str) -> str:
    """Fetch one case's raw text, for a re-write that needs the verbatim check."""
    for record in read_cases_internal(path):
        if record.case_id == case_id:
            return record.user_text_raw
    raise RecordError("no internal record for case_id %r in %s" % (case_id, path))
