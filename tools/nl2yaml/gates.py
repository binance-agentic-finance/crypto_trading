"""G1: the five sequential gates a generated YAML must pass, and the per-condition
Python assertions that are the point of the whole exercise.

The gates, in order, with the status each one fails as::

    G1a  yaml.safe_load (after stripping a code fence)   parse_error
    G1b  yaml_pipeline.spec.validate_spec                static_invalid | dryrun_failed
    G1c  yaml_pipeline.intent.reconcile_intent           intent_mismatch
    G1d  yaml_pipeline.bundle_runner.run_bundle          run_error | bundle_insufficient
    G1e  one Python predicate per stated condition       condition_violated |
                                                          condition_unresolved

Strictly sequential, and which gate stopped it is recorded, because the retry
cost is per GATE and not per error: ``validate_spec`` batches its static errors
but reports the dry-run one exception at a time, and skips the dry-run entirely
while any static error stands. So a spec with two static errors and a dry-run
failure needs two conversion rounds, not three.

Why G1e outranks any reviewer
-----------------------------
"It validated and it ran" is not "it is correct". A spec once passed G1a-G1d and
returned five candidates that were all TradFi perpetuals, for a request whose
first line was "exclude TradFi". Nothing in the output admitted it. Treating that
run as the gold answer would have written the bug into the weights, where no
later gate could see it.

Every condition that carries a number or a member list therefore gets a Python
predicate evaluated against the EXECUTION OUTPUT — not against the spec, and not
by asking a model. A predicate that exists is a condition that never needs
judgement again. A condition with no predicate is reported ``unverifiable``, and
never ``satisfied``.

Two traps this module refuses on purpose:

* **Vacuous truth.** "every candidate has quoteVolume > 2e6" is trivially true of
  an empty basket. A universally-quantified predicate over zero candidates
  returns ``unverifiable``, never ``satisfied``.
* **Vacuous expression.** "no candidate is BTC" is also true of a basket that
  simply happens to contain no BTC, in a spec that forgot the exclusion
  entirely. So before a predicate runs, the spec is checked for any of the
  vocabulary the capability table grants that condition; if it names none of it,
  the verdict is ``not_expressed``.

``bundle_insufficient`` is its own status because it is not the model's fault: a
required input source was unavailable, so the strategy never ran. It must be
excluded from conversion accuracy and must never become a DPO negative — see
:attr:`GateReport.counts_toward_accuracy`.
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .capability import (
    EXPRESSIBLE,
    NOT_EXPRESSIBLE,
    PROXY_ONLY,
    Condition,
    normalize_condition,
)

__all__ = [
    "ConditionVerdict",
    "GateReport",
    "GateResult",
    "GATES",
    "NOT_EXPRESSED",
    "PREDICATES",
    "RetryOutcome",
    "SATISFIED",
    "STATUSES",
    "STATUS_CONDITION_UNRESOLVED",
    "STATUS_PASSED",
    "UNVERIFIABLE",
    "VIOLATED",
    "evaluate_conditions",
    "register_predicate",
    "run_gates",
    "run_with_retries",
    "selection_candidates",
    "strip_code_fence",
]

GATES = ("G1a", "G1b", "G1c", "G1d", "G1e")

STATUS_PASSED = "passed"
STATUS_CONDITION_UNRESOLVED = "condition_unresolved"
STATUSES = (
    STATUS_PASSED,
    "parse_error",         # G1a
    "static_invalid",      # G1b, batched
    "dryrun_failed",       # G1b, one at a time
    "intent_mismatch",     # G1c
    "run_error",           # G1d, the model's fault
    "bundle_insufficient",  # G1d, NOT the model's fault
    "condition_violated",  # G1e
    STATUS_CONDITION_UNRESOLVED,  # G1e: omitted, unprovable, or silent proxy
    # G1e, and a separate stop from condition_violated on purpose: the spec may
    # well satisfy every predicate it was given — under a reading of the request
    # that it chose and never stated. "Right answer to a question nobody asked"
    # is not a violated condition, and folding the two together would let a
    # silent reading be repaired by touching a threshold.
    "undisclosed_assumption",
    # Belongs to the retry loop, not to any single gate: one union so a caller
    # tabulating outcomes has one vocabulary to switch on.
    "stuck",
)

#: Per-condition verdict vocabulary. Closed, and ``unverifiable`` is a first-class
#: answer rather than a soft pass: collapsing it into ``satisfied`` is how a
#: condition stops being checked without anyone deciding to stop checking it.
SATISFIED = "satisfied"
VIOLATED = "violated"
UNVERIFIABLE = "unverifiable"
NOT_EXPRESSED = "not_expressed"
CONDITION_VERDICTS = (SATISFIED, VIOLATED, UNVERIFIABLE, NOT_EXPRESSED)

#: The two error prefixes ``validate_spec`` uses for its dry-run failure, which is
#: the one error class it reports singly. Everything else it emits is batched.
_DRY_RUN_PREFIXES = ("dry-run failed", "selection dry-run failed")

#: What ``run_bundle`` raises when a required input source is missing or too thin
#: to be a cross-section. Matched on the prefix ``_assert_required_sources``
#: writes, so a different BundleRunError still lands in ``run_error``.
_BUNDLE_INSUFFICIENT_PREFIX = "required input source unavailable"

_FENCE = re.compile(r"^\s*```[A-Za-z0-9_+-]*\s*\n(?P<body>.*?)\n?\s*```\s*$", re.DOTALL)
#: Volatile fragments that make two identical failures hash differently and so
#: defeat the same-signature abort. Object addresses only — numbers in a message
#: are usually the thing that differs between two genuinely different errors.
_VOLATILE = re.compile(r"0x[0-9a-fA-F]+")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    gate: str
    status: str
    errors: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        if self.gate not in GATES:
            raise ValueError("unknown gate %r; expected one of %s" % (self.gate, GATES))
        if self.status not in STATUSES:
            raise ValueError("unknown status %r; expected one of %s"
                             % (self.status, STATUSES))
        if self.status != STATUS_PASSED and not self.errors:
            raise ValueError(
                "%s failed with %s but recorded no error text; a failure nobody "
                "can read cannot be turned into a retry prompt"
                % (self.gate, self.status))

    @property
    def ok(self) -> bool:
        return self.status == STATUS_PASSED

    @property
    def signature(self) -> str:
        """Stable identity of this failure, for the same-signature abort.

        Gate and status plus a hash of the sorted, whitespace-normalised error
        text. At temperature 0 a second identical signature means the prompt is
        missing information the model cannot invent, so retrying is spending
        tokens to get the same answer.
        """
        if self.ok:
            return "%s:%s" % (self.gate, STATUS_PASSED)
        body = "\n".join(sorted(
            _VOLATILE.sub("0xADDR", " ".join(str(item).split()))
            for item in self.errors))
        digest = hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]
        return "%s:%s:%s" % (self.gate, self.status, digest)


@dataclass(frozen=True)
class ConditionVerdict:
    """One user condition, ruled on against the execution output."""

    condition: Condition
    verdict: str
    #: The condition was answered by a stand-in field and the output says so
    #: nowhere. Tracked separately from ``verdict`` because it is orthogonal: a
    #: proxied condition can look perfectly ``satisfied`` on the proxy's own terms.
    silently_proxied: bool = False
    #: The condition had more than one reading, the spec picked one, and
    #: ``strategy.assumptions[]`` carries no entry for it. Orthogonal for the
    #: same reason as ``silently_proxied`` and one step worse: a proxy at least
    #: leaves a different block name in the spec to notice, whereas a chosen
    #: reading leaves a spec that looks exactly like the one a reader would have
    #: written for the OTHER reading.
    undisclosed_assumption: bool = False
    detail: str = ""
    #: candidate symbols that break the condition, for a violated verdict.
    offenders: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.verdict not in CONDITION_VERDICTS:
            raise ValueError("verdict must be one of %s, got %r"
                             % (list(CONDITION_VERDICTS), self.verdict))
        if self.verdict == VIOLATED and not self.detail:
            raise ValueError("a violated condition must say how it was violated")

    @property
    def subject(self) -> str:
        return self.condition.subject


@dataclass(frozen=True)
class GateReport:
    results: Tuple[GateResult, ...]
    spec: Optional[Dict[str, Any]] = None
    batch: Optional[Dict[str, Any]] = None
    condition_verdicts: Tuple[ConditionVerdict, ...] = ()

    @property
    def status(self) -> str:
        for result in self.results:
            if not result.ok:
                return result.status
        return STATUS_PASSED

    @property
    def failed_gate(self) -> Optional[str]:
        for result in self.results:
            if not result.ok:
                return result.gate
        return None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_PASSED

    @property
    def signature(self) -> str:
        for result in self.results:
            if not result.ok:
                return result.signature
        return "passed"

    @property
    def warnings(self) -> Tuple[str, ...]:
        return tuple(item for result in self.results for item in result.warnings)

    @property
    def counts_toward_accuracy(self) -> bool:
        """False for ``bundle_insufficient``.

        The strategy was never run, so the output says nothing about the
        conversion. Counting it as a failure understates accuracy and — worse —
        makes the spec a DPO negative, teaching the model to avoid a correct
        answer because the data pipeline was down that day.
        """
        return self.status != "bundle_insufficient"

    @property
    def silently_proxied(self) -> Tuple[ConditionVerdict, ...]:
        return tuple(item for item in self.condition_verdicts if item.silently_proxied)

    @property
    def clean(self) -> bool:
        """All five gates passed AND every condition is provably satisfied.

        This remains the canonical acceptance invariant even though G1e now
        fails closed with ``condition_unresolved``. Keeping the condition-level
        check here prevents a future status regression from making retry or gold
        selection accept a quietly proxied or unprovable answer.
        """
        return self.ok and all(
            item.verdict == SATISFIED
            and not item.silently_proxied
            and not item.undisclosed_assumption
            for item in self.condition_verdicts)

    def summary(self) -> str:
        lines = ["%s %s%s" % (result.gate,
                              result.status,
                              "" if result.ok else " <- stopped here")
                 for result in self.results]
        for item in self.condition_verdicts:
            lines.append("    %-32s %s%s%s" % (
                item.condition.id or item.subject,
                item.verdict,
                " SILENTLY-PROXIED" if item.silently_proxied else "",
                (" (%s)" % item.detail) if item.detail else ""))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reading the execution output
# ---------------------------------------------------------------------------


def strip_code_fence(text: str) -> str:
    """Drop one surrounding ```` ```yaml ```` fence, if present.

    Only a fence that wraps the WHOLE answer. A model that prefixes prose and
    then fences the YAML is not being cleaned up here: that is a
    prompt-compliance failure, and papering over it hides how often it happens.
    """
    match = _FENCE.match(text or "")
    return match.group("body") if match else (text or "")


def selection_candidates(batch: Mapping[str, Any]) -> Tuple[Dict[str, Any], ...]:
    """Every candidate across the batch's selection signals.

    A trade batch has none, so predicates over candidates come back
    ``unverifiable`` for it rather than vacuously satisfied.
    """
    out: List[Dict[str, Any]] = []
    for signal in (batch.get("signals") or ()):
        if signal.get("kind") == "selection":
            out.extend(signal.get("candidates") or ())
    return tuple(out)


def _numeric(candidate: Mapping[str, Any], *names: str) -> Optional[float]:
    """A candidate's value for the first present name, as a float, or None.

    The runtime coerces non-numeric feature values to ``str`` (``count`` comes
    back as ``"730188"``), so a string that parses is accepted — refusing it here
    would report ``unverifiable`` for a column that is perfectly checkable.
    """
    features = candidate.get("features") or {}
    for name in names:
        for holder in (candidate, features):
            if name in holder and holder[name] is not None:
                try:
                    return float(holder[name])
                except (TypeError, ValueError):
                    return None
    return None


def _text(candidate: Mapping[str, Any], *names: str) -> Optional[str]:
    features = candidate.get("features") or {}
    for name in names:
        for holder in (candidate, features):
            value = holder.get(name)
            if value is not None:
                return str(value)
    return None


def _base_token(symbol: str) -> str:
    """The same pair-to-base normalisation the news join and the dedupe use.

    Imported from the blocks package rather than reimplemented: these two once
    diverged, and the dedupe then failed to collapse exactly the pairs it exists
    for — a top-5 came back as three assets in five slots.
    """
    from cyqnt_trd.blocks.news_feed import base_token

    return str(base_token(str(symbol).upper()))


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

#: subject -> predicate. A predicate takes (candidates, condition) and returns
#: ``(verdict, detail, offenders)``.
Predicate = Callable[[Tuple[Dict[str, Any], ...], Condition],
                     Tuple[str, str, Tuple[str, ...]]]
PREDICATES: Dict[str, Predicate] = {}


def register_predicate(subject: str) -> Callable[[Predicate], Predicate]:
    def decorate(fn: Predicate) -> Predicate:
        if subject in PREDICATES:
            raise ValueError("a predicate for %r is already registered; two "
                             "predicates for one subject means the verdict "
                             "depends on import order" % (subject,))
        PREDICATES[subject] = fn
        return fn
    return decorate


def _unverifiable(detail: str) -> Tuple[str, str, Tuple[str, ...]]:
    return (UNVERIFIABLE, detail, ())


def _compare(value: float, cond: Condition) -> Optional[bool]:
    """Apply the condition's comparison to one number, or None if unreadable.

    ``value`` shapes accepted: a bare number (read as the ``>=``/``<=`` bound
    implied by polarity is NOT guessed — a bare number means ``>``), or a mapping
    ``{"op": ">", "threshold": 2e6}``, or ``{"min": .., "max": ..}``.
    """
    spec = cond.value
    if isinstance(spec, Mapping):
        if "op" in spec and "threshold" in spec:
            op = str(spec["op"])
            threshold = float(spec["threshold"])
            return {">": value > threshold, ">=": value >= threshold,
                    "<": value < threshold, "<=": value <= threshold,
                    "==": value == threshold}.get(op)
        low, high = spec.get("min"), spec.get("max")
        if low is None and high is None:
            return None
        if low is not None and value < float(low):
            return False
        if high is not None and value > float(high):
            return False
        return True
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return value > float(spec)
    return None


def _every_numeric(candidates, cond, *names: str) -> Tuple[str, str, Tuple[str, ...]]:
    """Universally quantify the condition's comparison over the basket."""
    if not candidates:
        return _unverifiable(
            "no candidates in the output, so 'every candidate ...' is vacuous")
    readable = [(item, _numeric(item, *names)) for item in candidates]
    absent = [item for item, value in readable if value is None]
    if absent:
        return _unverifiable(
            "%d of %d candidates carry none of %s, so the condition cannot be "
            "checked on them" % (len(absent), len(candidates), list(names)))
    offenders = []
    for item, value in readable:
        outcome = _compare(value, cond)
        if outcome is None:
            return _unverifiable("condition value %r is not a comparison this "
                                 "predicate can read" % (cond.value,))
        if not outcome:
            offenders.append("%s=%s" % (item.get("symbol"), value))
    if offenders:
        return (VIOLATED,
                "%d of %d candidates fail %s %r: %s"
                % (len(offenders), len(candidates), names[0], cond.value,
                   ", ".join(offenders[:10])),
                tuple(offenders))
    return (SATISFIED, "all %d candidates satisfy %s %r"
            % (len(candidates), names[0], cond.value), ())


def _ranked_on_column(candidates, cond, *names: str) -> Tuple[str, str, Tuple[str, ...]]:
    """The basket really was ranked on the column the user named.

    Checkable exactly, and worth checking: "rank the liquid coins by Square
    mentions" answered with a turnover ranking produces a basket that looks
    entirely reasonable, and the only trace is that each candidate's ``score``
    equals a different column than the one requested. Comparing the emitted score
    against the named column catches the swap with no judgement involved.

    Tolerance exists because the runtime rounds ``score`` to six decimals.
    """
    if not candidates:
        return _unverifiable("no candidates in the output to compare scores against")
    mismatched: List[str] = []
    for item in candidates:
        score = _numeric(item, "score")
        column = _numeric(item, *names)
        if score is None or column is None:
            return _unverifiable(
                "candidate %s carries no %s, so the ranking column cannot be "
                "compared with its score" % (item.get("symbol"), list(names)))
        if abs(score - column) > max(1e-6, abs(column) * 1e-9):
            mismatched.append("%s score=%s %s=%s"
                              % (item.get("symbol"), score, names[0], column))
    if mismatched:
        return (VIOLATED,
                "the basket was ranked on something other than %s: %s"
                % (names[0], "; ".join(mismatched[:5])), tuple(mismatched))
    return (SATISFIED, "every candidate's score is its %s, so the ranking is on "
            "the requested column" % names[0], ())


def _numeric_subject(*names: str) -> Predicate:
    """Dispatch a numeric subject on the operator the user used.

    ``rank`` and ``compare`` are different claims about the same column and need
    different assertions; running the threshold predicate on a ranking request
    reported ``unverifiable`` for a condition that is in fact exactly checkable.
    """
    def predicate(candidates, cond):
        if cond.operator == "rank":
            return _ranked_on_column(candidates, cond, *names)
        return _every_numeric(candidates, cond, *names)
    return predicate


register_predicate("quote_volume_24h")(_numeric_subject("quoteVolume", "quote_volume"))
register_predicate("price_change_24h")(_numeric_subject("priceChangePercent"))
register_predicate("funding_rate")(_numeric_subject("fundingRatePct", "funding_rate"))
register_predicate("social_mentions")(_numeric_subject("news_mention_count"))
register_predicate("social_sentiment")(_numeric_subject("news_bull_ratio"))


def _members(cond: Condition) -> Tuple[str, ...]:
    value = cond.value
    if isinstance(value, str):
        return (value.upper(),)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(sorted(str(item).upper() for item in value))
    return ()


@register_predicate("symbol_blacklist")
def _symbol_blacklist(candidates, cond):
    """No candidate is one of the named assets.

    Compared on the BASE token, not the pair. The originating request said
    "exclude BTC/ETH/SOL/XRP"; a basket containing BTCUSDC honours the literal
    four-name list and breaks the condition.
    """
    banned = {_base_token(name) for name in _members(cond)}
    if not banned:
        return _unverifiable("no member list on the condition")
    if not candidates:
        return _unverifiable("no candidates in the output, so the exclusion is "
                             "vacuously true")
    offenders = tuple(str(item.get("symbol")) for item in candidates
                      if _base_token(item.get("symbol") or "") in banned)
    if offenders:
        return (VIOLATED, "excluded asset(s) present in the basket: %s"
                % ", ".join(offenders), offenders)
    return (SATISFIED, "none of %s is in the basket of %d"
            % (sorted(banned), len(candidates)), ())


@register_predicate("symbol_whitelist")
def _symbol_whitelist(candidates, cond):
    wanted = {_base_token(name) for name in _members(cond)}
    if not wanted:
        return _unverifiable("no member list on the condition")
    if not candidates:
        return _unverifiable("no candidates in the output")
    offenders = tuple(str(item.get("symbol")) for item in candidates
                      if _base_token(item.get("symbol") or "") not in wanted)
    if offenders:
        return (VIOLATED, "candidate(s) outside the requested set %s: %s"
                % (sorted(wanted), ", ".join(offenders)), offenders)
    return (SATISFIED, "every candidate is in %s" % sorted(wanted), ())


@register_predicate("quote_currency")
def _quote_currency(candidates, cond):
    """Suffix test, so it rules on the quote currency and not the base.

    Excluding ``USDC`` therefore leaves ``USDCUSDT`` alone, which is correct and
    is the thing a hand-written name list gets wrong.
    """
    suffixes = _members(cond)
    if not suffixes:
        return _unverifiable("no quote currency named")
    if not candidates:
        return _unverifiable("no candidates in the output")
    hits = tuple(str(item.get("symbol")) for item in candidates
                 if str(item.get("symbol") or "").upper().endswith(suffixes))
    if cond.polarity == "exclude":
        if hits:
            return (VIOLATED, "candidate(s) quoted in %s: %s"
                    % (list(suffixes), ", ".join(hits)), hits)
        return (SATISFIED, "no candidate is quoted in %s" % list(suffixes), ())
    misses = tuple(str(item.get("symbol")) for item in candidates
                   if not str(item.get("symbol") or "").upper().endswith(suffixes))
    if misses:
        return (VIOLATED, "candidate(s) not quoted in %s: %s"
                % (list(suffixes), ", ".join(misses)), misses)
    return (SATISFIED, "every candidate is quoted in %s" % list(suffixes), ())


@register_predicate("basket_size")
def _basket_size(candidates, cond):
    """A ceiling, never a quota.

    Evaluated on an empty basket too, unlike the universal predicates: "at most
    5" really is satisfied by 0, and a filter nothing clears is a legitimate
    outcome the spec's own comments call out.
    """
    try:
        limit = int(cond.value)
    except (TypeError, ValueError):
        return _unverifiable("condition value %r is not a count" % (cond.value,))
    if len(candidates) > limit:
        return (VIOLATED, "%d candidates returned for a requested %d"
                % (len(candidates), limit),
                tuple(str(item.get("symbol")) for item in candidates[limit:]))
    bases = [_base_token(item.get("symbol") or "") for item in candidates]
    if len(set(bases)) != len(bases):
        # The same bet twice inside the ceiling: the count obeys the request and
        # the exposure does not, and nothing in the basket reveals it.
        return (VIOLATED,
                "%d candidates within the limit of %d, but only %d distinct base "
                "assets — a slot is held twice by one bet"
                % (len(candidates), limit, len(set(bases))),
                tuple(sorted({base for base in bases if bases.count(base) > 1})))
    return (SATISFIED, "%d candidates, limit %d, all distinct assets"
            % (len(candidates), limit), ())


@register_predicate("direction")
def _direction(candidates, cond):
    wanted = str(cond.value or "").lower()
    if wanted not in ("long", "short"):
        return _unverifiable("condition value %r is not a direction" % (cond.value,))
    if not candidates:
        return _unverifiable("no candidates in the output")
    offenders = tuple(
        "%s=%s" % (item.get("symbol"), item.get("direction") or item.get("side"))
        for item in candidates
        if str(item.get("direction") or item.get("side") or "").lower() != wanted)
    if offenders:
        return (VIOLATED, "candidate(s) not %s: %s" % (wanted, ", ".join(offenders)),
                offenders)
    return (SATISFIED, "all %d candidates are %s" % (len(candidates), wanted), ())


@register_predicate("score_order")
def _score_order(candidates, cond):
    """The basket really is taken from the end of the column the user named.

    Worth a predicate because this is invisible from outside: five symbols and
    five plausible funding rates look identical whether they are the five most
    negative or the five most positive.
    """
    wanted = str(cond.value or "").lower()
    if wanted not in ("asc", "desc"):
        return _unverifiable("condition value %r is not asc|desc" % (cond.value,))
    scores = [item.get("score") for item in candidates]
    if len(scores) < 2 or any(value is None for value in scores):
        return _unverifiable("fewer than two scored candidates to compare")
    pairs = list(zip(scores, scores[1:]))
    monotone = (all(a <= b for a, b in pairs) if wanted == "asc"
                else all(a >= b for a, b in pairs))
    if not monotone:
        return (VIOLATED, "scores are not %s: %s" % (wanted, scores), ())
    return (SATISFIED, "scores are %s: %s" % (wanted, scores), ())


@register_predicate("sector_label")
def _sector_label(candidates, cond):
    """Tag membership on ``underlying_sub_type``, when the column is there.

    ``unverifiable`` when it is absent, and that is not a technicality: this is
    exactly the run whose five candidates were all TradFi perpetuals for a
    request that said "exclude TradFi". Without the contract-metadata join there
    is nothing in the output to check the claim against, and answering
    ``satisfied`` there is how that basket got accepted.
    """
    tags = _members(cond)
    if not tags:
        return _unverifiable("no sector tag named")
    if not candidates:
        return _unverifiable("no candidates in the output")
    missing = [item for item in candidates
               if _text(item, "underlying_sub_type", "underlyingSubType") is None]
    if missing:
        return _unverifiable(
            "%d of %d candidates carry no underlying_sub_type — the "
            "contract-metadata join is not in this spec, so the sector claim "
            "cannot be checked against the output at all"
            % (len(missing), len(candidates)))
    def _carries(item) -> bool:
        cell = (_text(item, "underlying_sub_type", "underlyingSubType") or "").upper()
        present = {part.strip() for part in cell.split(",") if part.strip()}
        return bool(present & set(tags))
    if cond.polarity == "exclude":
        offenders = tuple(str(item.get("symbol")) for item in candidates if _carries(item))
        if offenders:
            return (VIOLATED, "candidate(s) tagged %s: %s"
                    % (list(tags), ", ".join(offenders)), offenders)
        return (SATISFIED, "no candidate is tagged %s" % list(tags), ())
    offenders = tuple(str(item.get("symbol")) for item in candidates if not _carries(item))
    if offenders:
        return (VIOLATED, "candidate(s) not tagged %s: %s"
                % (list(tags), ", ".join(offenders)), offenders)
    return (SATISFIED, "every candidate is tagged %s" % list(tags), ())


@register_predicate("contract_type")
def _contract_type(candidates, cond):
    wanted = _members(cond)
    if not wanted:
        return _unverifiable("no contract category named")
    if not candidates:
        return _unverifiable("no candidates in the output")
    values = [(item, (_text(item, "underlying_type", "underlyingType",
                            "contract_type", "contractType") or "").upper())
              for item in candidates]
    if any(not value for _item, value in values):
        return _unverifiable(
            "the contract-metadata columns are absent from the output; a TradFi "
            "exclusion cannot be verified without them")
    if cond.polarity == "exclude":
        offenders = tuple(str(item.get("symbol")) for item, value in values
                          if value in wanted)
        if offenders:
            return (VIOLATED, "candidate(s) of excluded category %s: %s"
                    % (list(wanted), ", ".join(offenders)), offenders)
        return (SATISFIED, "no candidate is %s" % list(wanted), ())
    offenders = tuple(str(item.get("symbol")) for item, value in values
                      if value not in wanted)
    if offenders:
        return (VIOLATED, "candidate(s) outside %s: %s"
                % (list(wanted), ", ".join(offenders)), offenders)
    return (SATISFIED, "every candidate is one of %s" % list(wanted), ())


def _never_verifiable(explanation: str) -> Predicate:
    """A predicate that can only ever answer ``unverifiable``.

    Registered deliberately for the subjects whose data does not exist. Without
    it these fall through to "no predicate registered", which reads like an
    omission somebody should fix; with it, the report states that no output of
    this pipeline can ever confirm the claim. Either way the answer is never
    ``satisfied`` — which is the property that matters.
    """
    def predicate(candidates, cond):
        return _unverifiable(explanation)
    return predicate


#: Registered through the same door as the real predicates, so the duplicate
#: check applies to them too.
_UNVERIFIABLE_SUBJECTS = {
    "market_cap":
        "no market-cap or circulating-supply column exists anywhere in this "
        "pipeline's output, so no basket can be shown to honour a market-cap "
        "condition — turnover is not a stand-in",
    "long_short_ratio":
        "long_short_ratio reaches neither frame: per-symbol only at the venue, "
        "and absent from DATA_SECTIONS, so it is in no output column",
    "open_interest":
        "open interest has no whole-market endpoint, so a cross-sectional basket "
        "carries no OI column to check",
    "onchain_holder_concentration":
        "no chain data in this repo's sources",
    "spread_liquidity":
        "bookTicker is not joined onto any frame, so no spread or depth column "
        "reaches the output",
    "technical_indicator":
        "the cross-sectional output carries no bars and no indicator column, so "
        "an indicator condition cannot be confirmed from it; 24h change is a "
        "different statement",
    "multi_timeframe":
        "the output records no per-timeframe state, so 'bearish on three "
        "timeframes' has nothing to check",
    "historical_range":
        "the cross-section is one instant; there is no window in the output",
    "entry_exit_plan":
        "the YAML selection path never fills a candidate's `trade` slot, so an "
        "entry/stop/target claim cannot be read off the basket",
    "liquidation":
        "the cross-section carries no liquidation column; the per-symbol feed is "
        "not joined onto the universe frame",
    "news_event":
        "the rank frame carries aggregate counts, not headlines, so 'had a "
        "listing announcement' is not readable from the output",
    "vague_criterion":
        "no column corresponds to the criterion, so nothing in the output can "
        "confirm or refute it — and a confident-looking basket here is the harm",
}

for _subject, _explanation in _UNVERIFIABLE_SUBJECTS.items():
    register_predicate(_subject)(_never_verifiable(_explanation))


# ---------------------------------------------------------------------------
# Did the spec even say it?
# ---------------------------------------------------------------------------


def _spec_tokens(spec: Any, prefix: str = "") -> set:
    """Every string, key, and dotted key-path in a spec.

    Used to ask "does this spec name any of the vocabulary the capability table
    grants this condition". Cheap and mechanical, and it closes the vacuous-truth
    hole: a spec that forgot ``exclude_symbols`` would otherwise be reported
    ``satisfied`` on any basket that happened to contain no BTC.
    """
    out: set = set()
    if isinstance(spec, Mapping):
        for key, value in spec.items():
            path = "%s.%s" % (prefix, key) if prefix else str(key)
            out.add(str(key))
            out.add(path)
            out |= _spec_tokens(value, path)
    elif isinstance(spec, (list, tuple)):
        for item in spec:
            out |= _spec_tokens(item, prefix)
    elif isinstance(spec, str):
        out.add(spec)
    return out


def _output_admits(batch: Mapping[str, Any], tokens: Iterable[str]) -> bool:
    """True if the output says anywhere that a stand-in was used.

    Looks at batch warnings and at each candidate's ``reason``, because those are
    the only two places a reader of a basket would ever see it. The Supertrend
    accident's output said ``quoteVolume=3.785e+09, rank 1 of 5`` and nothing
    else — which is why this returns False for it, and why the condition is then
    reported ``silently_proxied``.
    """
    haystack = [str(item) for item in (batch.get("warnings") or ())]
    for signal in (batch.get("signals") or ()):
        haystack.extend(str(item) for item in (signal.get("warnings") or ()))
        haystack.append(str(signal.get("summary") or ""))
        for candidate in (signal.get("candidates") or ()):
            haystack.append(str(candidate.get("reason") or ""))
    blob = "\n".join(haystack).lower()
    needles = [str(token).split(".")[-1].lower() for token in tokens if token]
    needles += ["proxy", "代理"]
    return any(needle and needle in blob for needle in needles)


def evaluate_conditions(
    batch: Mapping[str, Any],
    spec: Mapping[str, Any],
    conditions: Sequence[Any],
    *,
    allow_proxy_for: Sequence[str] = (),
) -> Tuple[ConditionVerdict, ...]:
    """Rule on every condition against the execution output. This is G1e."""
    opened = set(allow_proxy_for)
    candidates = selection_candidates(batch)
    spec_tokens = _spec_tokens(spec)
    declared = _declared_assumption_cids(spec)
    verdicts: List[ConditionVerdict] = []

    for raw in conditions:
        cond = normalize_condition(raw)
        # Only meaningful once the spec has EXPRESSED the condition: a reading
        # can only be silently chosen if a reading was chosen. A condition the
        # spec left out is already reported ``not_expressed``, which says the
        # stronger thing, and demanding a declaration for it on top would make
        # every partially-convertible case stop here instead — and 48 of the 81
        # conditions in the demo corpus are vague ones a correct spec omits, so
        # that is the common case, not the corner.
        undisclosed = bool(cond.ambiguity_type) and cond.id not in declared
        row = cond.capability
        proxy_tokens = tuple(row.proxy_block_refs) + ((row.gap_id,) if row.gap_id else ())

        if row.verdict == NOT_EXPRESSIBLE or (
                row.verdict == PROXY_ONLY and cond.id not in opened):
            verdicts.append(ConditionVerdict(
                condition=cond, verdict=NOT_EXPRESSED,
                silently_proxied=not _output_admits(batch, proxy_tokens),
                detail="capability verdict %s (%s); the condition never reached "
                       "the converter" % (row.verdict, row.gap_id)))
            continue

        if row.verdict not in (EXPRESSIBLE, PROXY_ONLY):
            verdicts.append(ConditionVerdict(
                condition=cond, verdict=UNVERIFIABLE,
                undisclosed_assumption=undisclosed,
                detail="capability verdict %s; nobody has ruled on this subject "
                       "yet, so the case should have been shelved" % row.verdict))
            continue

        # BLOCK refs only, and skipped when the row grants none.
        #
        # A block is a step somebody has to write, so its absence really does mean
        # the condition is not in the spec. A spec KEY is different: `order`,
        # `top_k` and `dedupe_by` all have documented defaults, so absence there
        # means "the default", which is a real expression — and the predicate can
        # still check it against the output, which is the stronger test anyway.
        # Applying the presence check to fields as well reported the shipped
        # example's deliberate `order: desc`-by-default as unexpressed.
        #
        # Matching on fields would also be unsound in the other direction: they
        # are frame column names, and `dedupe_by: base_asset` already puts
        # "base_asset" in the spec's tokens, which would make a contract_type
        # condition look expressed by coincidence.
        required = (row.block_refs if row.verdict == EXPRESSIBLE
                    else row.proxy_block_refs)
        if required and not (set(required) & spec_tokens):
            verdicts.append(ConditionVerdict(
                condition=cond, verdict=NOT_EXPRESSED,
                silently_proxied=not _output_admits(batch, proxy_tokens),
                detail="the spec names none of %s" % list(required)))
            continue

        predicate = PREDICATES.get(cond.subject)
        if predicate is None:
            verdicts.append(ConditionVerdict(
                condition=cond, verdict=UNVERIFIABLE,
                undisclosed_assumption=undisclosed,
                detail="no predicate registered for subject %r; report it as "
                       "unverifiable rather than assume it held" % cond.subject))
            continue

        verdict, detail, offenders = predicate(candidates, cond)
        verdicts.append(ConditionVerdict(
            condition=cond, verdict=verdict, detail=detail, offenders=offenders,
            undisclosed_assumption=undisclosed,
            silently_proxied=(row.verdict == PROXY_ONLY
                              and not _output_admits(batch, proxy_tokens))))
    return tuple(verdicts)


# ---------------------------------------------------------------------------
# The gate sequence
# ---------------------------------------------------------------------------


def _gate_a(text: str) -> Tuple[GateResult, Optional[Dict[str, Any]]]:
    import yaml

    body = strip_code_fence(text)
    try:
        loaded = yaml.safe_load(body)
    except yaml.YAMLError as exc:
        return GateResult("G1a", "parse_error",
                          errors=("%s: %s" % (type(exc).__name__, exc),)), None
    if not isinstance(loaded, dict):
        # A scalar or a list parses fine and then fails much later with an error
        # about a missing key, which reads as the model omitting a field rather
        # than as the answer not being a spec at all.
        return GateResult(
            "G1a", "parse_error",
            errors=("a spec must be a YAML mapping, got %s"
                    % type(loaded).__name__,)), None
    return GateResult("G1a", STATUS_PASSED), loaded


def _gate_b(spec: Dict[str, Any]) -> GateResult:
    from cyqnt_trd.standard_bot.yaml_pipeline.spec import validate_spec

    errors, warnings = validate_spec(copy.deepcopy(spec))
    if not errors:
        return GateResult("G1b", STATUS_PASSED, warnings=tuple(warnings))
    # The dry-run is skipped whenever a static error stands, so the two classes
    # cannot coexist — but assert it by requiring EVERY error to carry the prefix
    # rather than any, so a future mixed batch lands in the batched status
    # instead of being reported as the single-error kind.
    dry_run = all(str(item).startswith(_DRY_RUN_PREFIXES) for item in errors)
    return GateResult("G1b", "dryrun_failed" if dry_run else "static_invalid",
                      errors=tuple(errors), warnings=tuple(warnings),
                      detail=("one exception from the compiled spec; retrying "
                              "costs one round per gate, not one per error")
                      if dry_run else
                      ("%d static error(s), reported together" % len(errors)))


def _gate_c(spec: Dict[str, Any], nl: str) -> GateResult:
    from cyqnt_trd.standard_bot.yaml_pipeline.intent import (
        classify_request,
        reconcile_intent,
    )

    intent = classify_request(nl)
    errors, warnings = reconcile_intent(intent, copy.deepcopy(spec))
    if errors:
        return GateResult("G1c", "intent_mismatch", errors=tuple(errors),
                          warnings=tuple(warnings),
                          detail="intent.kind=%s evidence=%s"
                                 % (intent.kind, list(intent.evidence)))
    return GateResult("G1c", STATUS_PASSED, warnings=tuple(warnings),
                      detail="intent.kind=%s" % intent.kind)


def _gate_d(spec: Dict[str, Any],
            bundle: Mapping[str, Any]) -> Tuple[GateResult, Optional[Dict[str, Any]]]:
    from cyqnt_trd.standard_bot.yaml_pipeline.bundle_runner import (
        BundleRunError,
        run_bundle,
    )

    try:
        batch = run_bundle(copy.deepcopy(spec), copy.deepcopy(dict(bundle)))
    except BundleRunError as exc:
        if str(exc).startswith(_BUNDLE_INSUFFICIENT_PREFIX):
            return GateResult(
                "G1d", "bundle_insufficient", errors=(str(exc),),
                detail="the strategy was NOT run; this is a data-plane outage, "
                       "not a conversion error — excluded from accuracy and "
                       "never a DPO negative"), None
        return GateResult("G1d", "run_error",
                          errors=("BundleRunError: %s" % exc,)), None
    except Exception as exc:                     # noqa: BLE001 - reported, not hidden
        return GateResult("G1d", "run_error",
                          errors=("%s: %s" % (type(exc).__name__, exc),)), None
    return GateResult("G1d", STATUS_PASSED, warnings=tuple(batch.get("warnings") or ()),
                      detail="signal_count=%d candidates=%d"
                             % (batch.get("signal_count", 0),
                                len(selection_candidates(batch)))), batch


def _declared_assumption_cids(spec: Mapping[str, Any]) -> set:
    """The condition ids ``strategy.assumptions[]`` claims a reading for.

    Read straight off the spec rather than off the emitted signal: the signal
    renders them into prose for a human, and re-parsing that prose to check a
    machine invariant would make the invariant depend on the wording.
    """
    items = ((spec.get("strategy") or {}).get("assumptions") or [])
    return {str(item.get("cid")) for item in items
            if isinstance(item, dict) and item.get("cid")}


def _gate_e(batch: Mapping[str, Any], spec: Mapping[str, Any],
            conditions: Sequence[Any],
            allow_proxy_for: Sequence[str]) -> Tuple[GateResult, Tuple[ConditionVerdict, ...]]:
    verdicts = evaluate_conditions(batch, spec, conditions,
                                   allow_proxy_for=allow_proxy_for)
    violated = [item for item in verdicts if item.verdict == VIOLATED]
    undisclosed = [item for item in verdicts if item.undisclosed_assumption]
    unresolved = [item for item in verdicts
                  if item.verdict in (NOT_EXPRESSED, UNVERIFIABLE)
                  or item.silently_proxied]
    notes = ["%s: %s / %s" % (item.condition.id or item.subject, item.verdict,
                              item.detail)
             for item in verdicts if item.verdict != SATISFIED]
    # Checked BEFORE the violated branch. An undisclosed reading is the more
    # fundamental defect — a violated predicate is a wrong answer to the right
    # question, while this is an unstated question — and reporting the
    # threshold slip first would send the repair loop off adjusting numbers.
    if undisclosed:
        return GateResult(
            "G1e", "undisclosed_assumption",
            errors=tuple(
                "%s (%s): the request is %s-ambiguous here and the spec picked a "
                "reading without declaring it. Add strategy.assumptions[] with "
                "cid: %s, the reading taken, and the alternatives given up."
                % (item.condition.id or item.subject, item.subject,
                   item.condition.ambiguity_type, item.condition.id or "<cid>")
                for item in undisclosed),
            warnings=tuple(notes)), verdicts
    if violated:
        return GateResult(
            "G1e", "condition_violated",
            errors=tuple("%s (%s): %s" % (item.condition.id or item.subject,
                                          item.subject, item.detail)
                         for item in violated),
            warnings=tuple(notes)), verdicts
    if unresolved:
        errors = []
        for item in unresolved:
            cid = item.condition.id or item.subject
            if "capability verdict not_expressible" in item.detail:
                action = (
                    "This is a named capability gap: mark the request unsupported "
                    "until that capability is implemented; do not replace it with "
                    "a correlated signal.")
            elif "capability verdict proxy_only" in item.detail:
                action = (
                    "Only a proxy capability exists. Obtain explicit proxy approval "
                    "and disclose the degradation in the output, or implement the "
                    "exact capability.")
            elif item.verdict == NOT_EXPRESSED:
                action = (
                    "Regenerate the YAML so this condition is explicitly expressed "
                    "using the capability named above.")
            elif item.verdict == UNVERIFIABLE:
                action = (
                    "Add execution evidence plus a deterministic predicate, or "
                    "shelve/clarify the condition instead of accepting it.")
            else:
                action = (
                    "The predicate only passed through an undisclosed proxy. Use "
                    "the exact capability or surface and explicitly approve the "
                    "proxy degradation.")
            if item.silently_proxied and "proxy" not in action.lower():
                action += (
                    " The current output also hides a proxy; remove it or disclose "
                    "and explicitly approve the degradation.")
            errors.append("%s (%s): %s: %s %s" % (
                cid, item.subject, item.verdict, item.detail, action))
        return GateResult(
            "G1e", STATUS_CONDITION_UNRESOLVED,
            errors=tuple(errors), warnings=tuple(notes),
            detail="%d condition(s) were not provably and exactly satisfied"
                   % len(unresolved)), verdicts
    return GateResult("G1e", STATUS_PASSED, warnings=tuple(notes),
                      detail="%d condition(s): %s" % (
                          len(verdicts),
                          ", ".join(sorted({item.verdict for item in verdicts})) or "none")), verdicts


def run_gates(
    answer: Any,
    *,
    nl: str,
    bundle: Mapping[str, Any],
    conditions: Sequence[Any] = (),
    allow_proxy_for: Sequence[str] = (),
) -> GateReport:
    """Run G1a-G1e in order and stop at the first failure.

    *nl* and *bundle* are required with no defaults, deliberately:

    * without the request text, G1c cannot run, and a gate that silently skips is
      worse than one that is absent — the report would claim four gates' worth of
      assurance under a five-gate name;
    * without a frozen bundle, G1d would either be skipped or would collect live
      data, which makes the same spec pass today and fail tomorrow.

    *answer* may be the raw model text (the normal path, so G1a is exercised) or
    an already-parsed mapping, for a caller testing the later gates directly.
    """
    if not str(nl or "").strip():
        raise ValueError(
            "run_gates needs the request text: G1c reconciles the spec against "
            "what the user asked for, and there is no honest default for that")
    if not isinstance(bundle, Mapping) or not bundle.get("frames"):
        raise ValueError(
            "run_gates needs a frozen cyqnt.input/v1 bundle with frames; G1d must "
            "not reach for live data, or the gate result depends on the hour")

    results: List[GateResult] = []

    if isinstance(answer, Mapping):
        spec: Optional[Dict[str, Any]] = dict(answer)
        results.append(GateResult("G1a", STATUS_PASSED,
                                  detail="caller supplied a parsed mapping"))
    else:
        result, spec = _gate_a(str(answer))
        results.append(result)
        if not result.ok:
            return GateReport(tuple(results))

    assert spec is not None
    results.append(_gate_b(spec))
    if not results[-1].ok:
        return GateReport(tuple(results), spec=spec)

    results.append(_gate_c(spec, nl))
    if not results[-1].ok:
        return GateReport(tuple(results), spec=spec)

    result, batch = _gate_d(spec, bundle)
    results.append(result)
    if not result.ok:
        return GateReport(tuple(results), spec=spec)
    assert batch is not None

    result, verdicts = _gate_e(batch, spec, conditions, allow_proxy_for)
    results.append(result)
    return GateReport(tuple(results), spec=spec, batch=batch,
                      condition_verdicts=verdicts)


# ---------------------------------------------------------------------------
# Retry loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryOutcome:
    reports: Tuple[GateReport, ...]
    status: str
    #: Set when the loop aborted on a repeated signature. The legacy field name
    #: is retained for callers, but the message preserves a named G1e capability
    #: gap instead of relabelling every repeated failure as a playbook problem.
    playbook_gap: Optional[str] = None

    @property
    def report(self) -> GateReport:
        return self.reports[-1]

    @property
    def attempts(self) -> int:
        return len(self.reports)


def run_with_retries(
    convert: Callable[[int, Optional[GateReport]], Any],
    *,
    nl: str,
    bundle: Mapping[str, Any],
    conditions: Sequence[Any] = (),
    allow_proxy_for: Sequence[str] = (),
    max_attempts: int = 3,
) -> RetryOutcome:
    """Convert, gate, feed the failure back, up to *max_attempts*.

    Aborts early when the same gate fails with the same ``error_signature``
    twice. At temperature 0 the model does not need another identical attempt:
    ordinary repeats identify a playbook gap, while a G1e error that already
    names an unavailable capability keeps that classification.

    ``bundle_insufficient`` also stops the loop immediately: nothing about the
    spec is known to be wrong, so a retry would be asking the model to fix the
    data plane.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    reports: List[GateReport] = []
    seen: Dict[str, int] = {}
    previous: Optional[GateReport] = None

    for attempt in range(max_attempts):
        report = run_gates(convert(attempt, previous), nl=nl, bundle=bundle,
                           conditions=conditions, allow_proxy_for=allow_proxy_for)
        reports.append(report)
        if report.clean:
            return RetryOutcome(tuple(reports), STATUS_PASSED)
        if report.status == "bundle_insufficient":
            return RetryOutcome(tuple(reports), "bundle_insufficient")
        signature = report.signature
        seen[signature] = seen.get(signature, 0) + 1
        if seen[signature] >= 2:
            first_error = report.results[-1].errors[0]
            if "named capability gap" in first_error:
                explanation = (
                    "%s failed twice with the identical signature %s. The error "
                    "names a capability gap, so keep the request unsupported and "
                    "route it to the capability backlog; do not keep retrying the "
                    "YAML. First error: %s"
                    % (report.failed_gate, signature, first_error))
            else:
                explanation = (
                    "%s failed twice with the identical signature %s. The prompt "
                    "is missing information the model cannot derive; fix the "
                    "playbook, do not file a capability gap. First error: %s"
                    % (report.failed_gate, signature, first_error))
            return RetryOutcome(
                tuple(reports), "stuck",
                playbook_gap=explanation)
        previous = report

    return RetryOutcome(tuple(reports), reports[-1].status)
