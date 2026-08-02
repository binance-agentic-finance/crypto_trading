"""Guard: user text and user identity never reach the public half of the dataset.

Both git remotes of this repository are PUBLIC, and the corpus behind this
dataset is 51,595 rows of real chat messages keyed by Telegram user id. The
split is therefore enforced in code — see ``tools/nl2yaml/schema.py`` — and these
tests are what keep the enforcement honest.

Five things are checked, in the order they would fail in practice:

1. no ``user_text*`` / ``quote`` key anywhere in a public record, at any nesting
   depth, and the public JSON Schemas do not even declare such a property;
2. no PII-shaped *content* in a public dataset file: an 8+ digit run (the shape
   of a Telegram id), an ``@handle``, or a URL;
3. the internal store is outside the repository — asserted with
   ``os.path.relpath``, because "it is gitignored" is not a control: one
   ``.gitignore`` edit publishes the whole directory — and its salt is mode 600;
4. zero 8-gram overlap between an emitted ``strategy.description`` and the user's
   own words. ``description`` is where a converter most often paraphrases by
   copying, and unlike every other field it *ships inside the YAML*: into the
   repo, into generated docs, into signal payloads;
5. ``write_case()`` raises — not drops — when handed user text.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:  # tools/ is a namespace package under the repo root
    sys.path.insert(0, str(REPO))

from tools.nl2yaml import schema as S  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures: synthetic "user text" written here, never lifted from the corpus
# ---------------------------------------------------------------------------

#: Stand-in for a real request. Invented for this test; it is not a corpus row.
FAKE_USER_TEXT_ZH = (
    "幫我掃一下幣安合約,排除主流幣跟美股永續,二十四小時成交額大於兩百萬美元,"
    "散戶偏多的找做空,依流動性排序給我五檔"
)
FAKE_USER_TEXT_EN = (
    "scan binance perpetual futures for short candidates, exclude the majors and "
    "anything tradfi, twenty four hour turnover above two million dollars, rank by "
    "liquidity and give me five names"
)

#: What a converter should write: structured, its own words.
SAFE_YAML = (
    "spec_version: \"1.0\"\n"
    "strategy:\n"
    "  id: short_candidate_screen\n"
    "  description: \"cross-section short screen: liquidity floor, majors excluded, "
    "ranked by turnover\"\n"
)

#: What a converter does when it takes the lazy route: the request, pasted.
LEAKY_YAML_ZH = (
    "spec_version: \"1.0\"\n"
    "strategy:\n"
    "  id: short_candidate_screen\n"
    "  description: \"排除主流幣跟美股永續,二十四小時成交額大於兩百萬美元\"\n"
)
LEAKY_YAML_EN = (
    "spec_version: \"1.0\"\n"
    "strategy:\n"
    "  id: short_candidate_screen\n"
    "  description: \"exclude the majors and anything tradfi, twenty four hour "
    "turnover above two million dollars\"\n"
)


def make_case(text, **over):
    base = dict(
        case_id=S.new_case_id(),
        text_sha256=S.sha256_hex(text),
        canon_sha256=S.canon_sha256_of(text),
        dup_cluster_id=S.cluster_id_for(S.canon_sha256_of(text)),
        dup_count=1,
        preset_case=False,
        pseudonym_id="pid_" + "abcdefghijkmnpqrstuvwxyz23"[:26],
        lang=S.Lang.ZH,
        zh_variant=S.ZhVariant.HANT,
        month="2026-05",
        mining_source=S.MiningSource.KEYWORD_LEXICON,
        source_snapshot_id="chats_2026-05_07_v1",
        text_provenance=S.TextProvenance.VERBATIM_INTERNAL,
        text_len_chars=len(text),
    )
    base.update(over)
    return S.CaseRecord(**base)


@pytest.fixture()
def internal(tmp_path, monkeypatch):
    """A throwaway internal root with a mode-600 salt."""
    root = tmp_path / "nl2yaml_internal"
    root.mkdir()
    salt = root / "salt"
    salt.write_bytes(b"0123456789abcdef0123456789abcdef")
    salt.chmod(0o600)
    monkeypatch.setenv(S.INTERNAL_ROOT_ENV, str(root))
    return root


# ---------------------------------------------------------------------------
# 1. no user-text / quote keys in a public record
# ---------------------------------------------------------------------------

def _keys(payload, out=None):
    out = [] if out is None else out
    if isinstance(payload, dict):
        for key, value in payload.items():
            out.append(key)
            _keys(value, out)
    elif isinstance(payload, list):
        for value in payload:
            _keys(value, out)
    return out


def _is_forbidden(key):
    low = key.lower()
    return low in S.FORBIDDEN_PUBLIC_KEYS or low.startswith(S.FORBIDDEN_PUBLIC_KEY_PREFIXES)


def test_public_records_have_no_user_text_or_quote_keys(tmp_path, internal):
    """Recursive: ``conditions[].quote`` is nested two levels down."""
    case = make_case(
        FAKE_USER_TEXT_ZH,
        conditions=[S.Condition(cid="c1", polarity=S.Polarity.EXCLUDE,
                                subject="underlying_type", operator=S.Operator.NEQ,
                                value="EQUITY", unit=S.Unit.LABEL)],
        capability_map=[S.CapabilityEntry(cid="c1",
                                         verdict=S.CapabilityVerdict.UNSUPPORTED,
                                         gap_id=S.GapId.UNIVERSE_CONTRACT_META)],
        tier=S.CaseTier.T3_BLOCKED_BY_GAP,
        gold_specs=[S.GoldSpec(role=S.GoldRole.PRIMARY, yaml_text=SAFE_YAML)],
        gold_source=S.GoldSource.HANDWRITTEN,
        gold_verification_level=S.GoldVerificationLevel.STATIC_VALID,
    )
    payload = S.write_case(tmp_path / "cases.jsonl", case,
                           source_text=FAKE_USER_TEXT_ZH)
    offenders = [k for k in _keys(payload) if _is_forbidden(k)]
    assert offenders == []
    # sanity: the detector is not vacuous
    assert _is_forbidden("quote") and _is_forbidden("user_text_raw")
    assert not _is_forbidden("quote_volume_24h"), (
        "the block vocabulary is full of quote_volume/quote_suffix; a guard that "
        "fires on those gets switched off")


@pytest.mark.parametrize("kind", ["case", "attempt", "gap", "run"])
def test_public_schemas_do_not_declare_a_user_text_property(kind):
    """The schema is the second, independent barrier: with
    ``additionalProperties: false`` and no such property declared, a record
    carrying user text cannot validate even if a guard is bypassed."""
    schema = json.loads((S.SCHEMA_DIR / S.KIND_SCHEMA_FILES[kind]).read_text("utf-8"))
    assert schema["additionalProperties"] is False
    offenders = [k for k in schema["properties"] if _is_forbidden(k)]
    assert offenders == []
    shared = json.loads((S.SCHEMA_DIR / "common.schema.json").read_text("utf-8"))
    for name, body in shared["$defs"].items():
        for key in body.get("properties", {}):
            assert not _is_forbidden(key), "%s.%s" % (name, key)


def test_no_schema_allows_a_forbidden_key_as_an_open_property_name():
    """``propertyNames`` enums are keys too.

    ``prompt_tokens_by_component`` nearly shipped with a ``user_text`` bucket: a
    legal record by the schema, rejected by the recursive guard. Two barriers that
    disagree means one of them is wrong, and the failure would have looked like
    the guard being broken.
    """
    for name in list(S.KIND_SCHEMA_FILES.values()) + ["common.schema.json"]:
        schema = json.loads((S.SCHEMA_DIR / name).read_text("utf-8"))
        stack = [schema]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                allowed = (node.get("propertyNames") or {}).get("enum") or []
                for key in allowed:
                    assert not _is_forbidden(key), "%s: propertyNames allows %r" % (name, key)
                stack += list(node.values())
            elif isinstance(node, list):
                stack += node


def test_token_accounting_bucket_for_the_request_is_writable(tmp_path, internal):
    """The record that motivated the test above must actually write."""
    case = make_case(FAKE_USER_TEXT_ZH,
                     prompt_tokens_by_component={"system": 120, "request": 96})
    payload = S.write_case(tmp_path / "cases.jsonl", case, source_text=FAKE_USER_TEXT_ZH)
    assert payload["prompt_tokens_by_component"]["request"] == 96
    bad = S.validate_record(case)
    bad["prompt_tokens_by_component"] = {"user_text": 96}
    with pytest.raises(S.PrivacyError, match="user_text"):
        S.write_case(tmp_path / "cases.jsonl", bad, source_text=FAKE_USER_TEXT_ZH)


def test_only_the_internal_schema_carries_quotes():
    """One grep answers "where can user text live?"."""
    internal_schema = json.loads(
        (S.SCHEMA_DIR / S.KIND_SCHEMA_FILES["case_internal"]).read_text("utf-8"))
    assert {k for k in internal_schema["properties"] if _is_forbidden(k)} == {
        "user_text_raw", "user_text_redacted", "user_id", "conditions_with_quotes"}
    assert "INTERNAL ONLY" in internal_schema["description"]


# ---------------------------------------------------------------------------
# 2. no PII-shaped content in a public dataset file
# ---------------------------------------------------------------------------

#: A Telegram user id is a bare 8-15 digit integer. Long thresholds are *not* a
#: false positive source here because this repo writes them in scientific
#: notation (``universe.filter_quote_volume 2e6`` in docs/strategy_yaml_spec);
#: if this ever fires on a threshold, write it as ``2.0e7``.
DIGIT_RUN_RE = re.compile(r"(?<![0-9A-Za-z_])[0-9]{8,}(?![0-9A-Za-z_])")
HANDLE_RE = re.compile(r"(?<![A-Za-z0-9_])@[A-Za-z0-9_]{4,}")
URL_RE = re.compile(r"(https?://|www\.[a-z]|t\.me/)", re.IGNORECASE)

#: Machine-generated identifier shapes, allowlisted by *value shape* rather than
#: by key name: a leaked Telegram id is not hash-shaped, so this cannot hide one,
#: whereas allowlisting key names would exempt whatever field the leak lands in.
#:
#: Mostly belt and braces — the lookarounds in ``DIGIT_RUN_RE`` already prevent a
#: match *inside* an alphanumeric token, so a digit run in the middle of a hash
#: never fires. This list covers the residual case of a value that is digits all
#: the way to a boundary.
ID_SHAPES = tuple(re.compile(p) for p in (
    r"^[0-9a-f]{16,64}$",                       # sha256 / truncated signature / git sha
    r"^[0-9A-HJKMNP-TV-Z]{26}$",                # ULID
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    r"^pid_[a-z2-7]{26}$",                      # hmac_pseudonym output
    r"^dup_[0-9a-f]{16}$",
))


def pii_hits(payload, path=""):
    """Every PII-shaped string in a record payload, with its JSON path.

    Strings only, deliberately. Numeric fields in these records are typed
    quantities (thresholds, counts, token totals) and, because every record
    schema sets ``additionalProperties: false`` over a closed field list, there
    is no numeric field a user id could hide in — ``pseudonym_id`` is the only
    identity field and its shape is enforced by the schema.
    """
    hits = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            hits += pii_hits(value, "%s/%s" % (path, key))
    elif isinstance(payload, list):
        for i, value in enumerate(payload):
            hits += pii_hits(value, "%s[%d]" % (path, i))
    elif isinstance(payload, str):
        if any(shape.match(payload) for shape in ID_SHAPES):
            return hits
        for label, pattern in (("digits", DIGIT_RUN_RE), ("handle", HANDLE_RE),
                               ("url", URL_RE)):
            match = pattern.search(payload)
            if match:
                hits.append("%s: %s (%r)" % (path or "/", label, match.group(0)))
    return hits


def test_pii_scanner_is_not_vacuous():
    """Positive control for each pattern, plus the shapes that must NOT fire."""
    assert pii_hits({"a": "user 123456789 asked"})
    assert pii_hits({"a": "ping @satoshi_nakamoto"})
    assert pii_hits({"a": "see https://t.me/somegroup"})
    assert pii_hits({"a": "t.me/somegroup"})
    assert pii_hits(["nested", {"b": ["987654321"]}])
    assert not pii_hits({"text_sha256": S.sha256_hex("x")})
    assert not pii_hits({"case_id": S.new_case_id()})
    assert not pii_hits({"pseudonym_id": "pid_" + "abcdefghijkmnpqrstuvwxyz23"})
    assert not pii_hits({"decision": "2026-08-01T00:00:00Z", "v": "2e6"})


def test_written_public_records_carry_no_pii_shapes(tmp_path, internal):
    """Write one of every public record type and sweep the files."""
    case = make_case(
        FAKE_USER_TEXT_ZH,
        conditions=[S.Condition(cid="c1", polarity=S.Polarity.INCLUDE,
                                subject="quote_volume_24h", operator=S.Operator.GT,
                                value=2e6, unit=S.Unit.USD)],
        capability_map=[S.CapabilityEntry(cid="c1",
                                         verdict=S.CapabilityVerdict.SUPPORTED)],
        tier=S.CaseTier.T1_EXPRESSIBLE,
        gold_specs=[S.GoldSpec(role=S.GoldRole.PRIMARY, yaml_text=SAFE_YAML)],
        gold_source=S.GoldSource.HANDWRITTEN,
        gold_verification_level=S.GoldVerificationLevel.STATIC_VALID,
        block_refs=["universe.filter_quote_volume"],
        repo_git_sha="2d21006",
    )
    S.write_case(tmp_path / "cases.jsonl", case, source_text=FAKE_USER_TEXT_ZH)
    S.write_attempt(tmp_path / "attempts.jsonl", S.AttemptRecord(
        case_id=case.case_id, attempt_index=1,
        sampling_purpose=S.SamplingPurpose.CONVERT,
        prompt_sha256=S.sha256_hex("prompt"), yaml_text=SAFE_YAML,
        stopped_at_gate=S.Gate.INTENT_MISMATCH,
        gate_errors=["5/5 candidates have underlyingType=EQUITY"],
        defect_class=S.DefectClass.SILENTLY_DROPPED_EXCLUSION),
        source_text=FAKE_USER_TEXT_ZH)
    S.write_gap(tmp_path / "gaps.jsonl", S.GapRecord(
        gap_id=S.GapId.UNIVERSE_CONTRACT_META,
        title="exchangeInfo contract metadata is not exposed to YAML",
        needed_capability="universe.filter_underlying_type (underlyingType == COIN)",
        hit_count=414, dup_weighted_count=1520,
        example_case_ids=[case.case_id]))
    S.write_run(tmp_path / "runs.jsonl", S.RunRecord(
        run_id="run-0001", bundle_id="universe_cross_section",
        bundle_sha256=S.sha256_hex("bundle"),
        bundle_decision_time="2026-08-01T00:00:00+00:00",
        bundle_snapshot_id="snap-2026-08-01", repo_git_sha="2d21006",
        python_version="3.11.15", pandas_version="2.2.3", signal_count=1,
        bundle_nodes=[S.BundleNode(node="universe", rows=727)],
        candidates=[S.Candidate(rank=1, symbol="SNDKUSDT",
                                side=S.CandidateSide.SHORT, score=-11.76,
                                attributes={"underlyingType": "EQUITY"})]))

    for name in ("cases.jsonl", "attempts.jsonl", "gaps.jsonl", "runs.jsonl"):
        for lineno, line in enumerate((tmp_path / name).read_text("utf-8").splitlines(), 1):
            hits = pii_hits(json.loads(line))
            assert hits == [], "%s:%d %s" % (name, lineno, hits)


def _publishable_dataset_files():
    """Dataset files a PR from this tree would publish.

    Tracked *and* untracked-but-not-ignored, because a pull request adds the
    second kind and a guard that only reads committed files cannot stop a leak
    from entering — the same hole that once let an internal hostname through
    ``tests/test_no_leaked_secrets.py`` (naming it here would reintroduce it,
    which is how that guard caught this very docstring).
    """
    out = []
    for argv in (["git", "ls-files"], ["git", "ls-files", "--others", "--exclude-standard"]):
        listing = subprocess.run(argv, cwd=REPO, capture_output=True, text=True,
                                 check=True).stdout.split("\n")
        for rel in listing:
            rel = rel.strip()
            if not rel.endswith(".jsonl"):
                continue
            name = Path(rel).name
            if "nl2yaml" in rel or name.split(".")[0] in (
                    "cases", "attempts", "gaps", "runs", "cases_internal"):
                out.append(rel)
    return sorted(set(out))


def test_no_pii_in_publishable_dataset_files():
    """Sweep the real tree. Empty today; this is the guard for when data lands."""
    offenders = []
    for rel in _publishable_dataset_files():
        assert Path(rel).name != "cases_internal.jsonl", (
            "%s is publishable; internal records belong in %s" % (rel, S.internal_root()))
        for lineno, line in enumerate((REPO / rel).read_text("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            for key in _keys(payload):
                assert not _is_forbidden(key), "%s:%d key %r" % (rel, lineno, key)
            offenders += ["%s:%d %s" % (rel, lineno, hit) for hit in pii_hits(payload)]
    assert offenders == []


# ---------------------------------------------------------------------------
# 2b. a gitignored raw export is still a privacy incident
# ---------------------------------------------------------------------------

_RAW_CONVERSATION_COLUMNS = frozenset({
    "user_id", "chat_id", "first_query", "user_text_excerpt",
})


def _repo_data_files_including_ignored():
    """Yield CSV/JSONL files visible to git, including ignored local files.

    This deliberately uses git rather than a repository-wide recursive walk:
    it sees an ignored export before a PR does, while avoiding virtual
    environments and the repository's own administrative directory.  We only
    inspect headers/keys below; no conversation value is returned or printed.
    """
    seen = set()
    commands = (
        ["git", "ls-files"],
        ["git", "ls-files", "--others", "--exclude-standard"],
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard"],
    )
    for argv in commands:
        output = subprocess.run(
            argv, cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout.splitlines()
        for rel in output:
            path = REPO / rel
            if path.suffix.lower() in {".csv", ".jsonl"} and path.is_file():
                seen.add(path)
    return sorted(seen)


def _has_raw_conversation_columns(path):
    """Inspect only a CSV header or a JSONL object's keys."""
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            header = next(csv.reader(handle), [])
        return _RAW_CONVERSATION_COLUMNS <= set(header)

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                return (isinstance(payload, dict)
                        and _RAW_CONVERSATION_COLUMNS <= set(payload))
    return False


def test_no_raw_conversation_export_exists_anywhere_under_repo():
    """Raw chats belong below ``NL2YAML_INTERNAL_ROOT``, never under git.

    ``.gitignore`` is deliberately included in the scan: ignoring a CSV does
    not stop an editor, archive, accidental ``git add -f``, or a changed ignore
    rule from exposing it.  The failure reports filenames only, not user text.
    """
    offenders = [str(path.relative_to(REPO)) for path in _repo_data_files_including_ignored()
                if _has_raw_conversation_columns(path)]
    assert offenders == [], (
        "raw conversation exports must live below %s, not the repository: %s"
        % (S.internal_root(), offenders)
    )


# ---------------------------------------------------------------------------
# 3. the internal store is outside the repo, and its salt is locked down
# ---------------------------------------------------------------------------

def test_default_internal_root_is_outside_the_repo():
    rel = os.path.relpath(str(S.DEFAULT_INTERNAL_ROOT), str(REPO))
    assert rel.startswith(os.pardir), (
        "%s resolves inside the repository (%s); being gitignored is not a control — "
        "one .gitignore edit publishes it" % (S.DEFAULT_INTERNAL_ROOT, rel))


def test_internal_root_inside_the_repo_is_refused(monkeypatch):
    monkeypatch.setenv(S.INTERNAL_ROOT_ENV, str(REPO / "docs" / "user_demand_analysis"))
    with pytest.raises(S.PrivacyError, match="inside the repository"):
        S.internal_root()


def test_real_salt_is_mode_600():
    """Checked against the production location, not the fixture: the fixture can
    only prove the code reads a mode, not that the real salt is protected."""
    root = S.DEFAULT_INTERNAL_ROOT
    if not root.exists():
        pytest.skip("%s does not exist yet: nothing has been mined on this machine, so "
                    "there is no salt and no user text to protect" % root)
    salt = root / "salt"
    assert salt.exists(), "internal root exists but has no salt: %s" % salt
    assert salt.stat().st_mode & 0o777 == 0o600, oct(salt.stat().st_mode & 0o777)
    assert root.stat().st_mode & 0o777 == 0o700, oct(root.stat().st_mode & 0o777)


def test_salt_with_loose_permissions_is_refused(tmp_path, monkeypatch):
    root = tmp_path / "internal"
    root.mkdir()
    (root / "salt").write_bytes(b"x" * 32)
    (root / "salt").chmod(0o644)
    monkeypatch.setenv(S.INTERNAL_ROOT_ENV, str(root))
    with pytest.raises(S.PrivacyError, match="mode"):
        S.hmac_pseudonym("123456789")


def test_missing_salt_does_not_fall_back_to_a_default(tmp_path, monkeypatch):
    """A generated-per-run salt would silently break linkability; a hard-coded
    one would be a rainbow table away from the real ids."""
    monkeypatch.setenv(S.INTERNAL_ROOT_ENV, str(tmp_path / "empty"))
    with pytest.raises(S.PrivacyError, match="missing HMAC salt"):
        S.hmac_pseudonym("123456789")


def test_pseudonym_hides_the_id_and_is_stable(internal):
    first = S.hmac_pseudonym("123456789")
    assert first == S.hmac_pseudonym("123456789")
    assert first != S.hmac_pseudonym("123456780")
    assert "123456789" not in first
    assert not pii_hits({"pseudonym_id": first}), (
        "the pseudonym must not itself look like PII, or this whole test module "
        "starts crying wolf on its own output")


def test_internal_writer_refuses_a_path_inside_the_repo(internal):
    record = S.InternalCaseRecord(
        case_id=S.new_case_id(), user_text_raw=FAKE_USER_TEXT_ZH,
        user_text_redacted=FAKE_USER_TEXT_ZH, pseudonym_id=S.hmac_pseudonym("1"),
        user_id="1")
    with pytest.raises(S.PrivacyError, match="outside the internal root"):
        S.write_case_internal(REPO / "tools" / "nl2yaml" / "cases_internal.jsonl", record)


def test_internal_file_is_created_mode_600(internal):
    record = S.InternalCaseRecord(
        case_id=S.new_case_id(), user_text_raw=FAKE_USER_TEXT_ZH,
        user_text_redacted="[redacted]", pseudonym_id=S.hmac_pseudonym("1"),
        user_id="1",
        conditions_with_quotes=[{"cid": "c1", "quote": "排除美股永續"}])
    target = internal / "cases_internal.jsonl"
    S.write_case_internal(target, record)
    assert target.stat().st_mode & 0o777 == 0o600
    assert list(S.read_cases_internal(target))[0].user_text_raw == FAKE_USER_TEXT_ZH


# ---------------------------------------------------------------------------
# 4. strategy.description must not quote the user
# ---------------------------------------------------------------------------

def _description(yaml_text):
    return (yaml.safe_load(yaml_text) or {}).get("strategy", {}).get("description", "")


@pytest.mark.parametrize("user_text,leaky", [
    (FAKE_USER_TEXT_ZH, LEAKY_YAML_ZH),
    (FAKE_USER_TEXT_EN, LEAKY_YAML_EN),
])
def test_description_has_zero_8gram_overlap_with_the_user_text(user_text, leaky):
    """The requirement, stated directly: overlap must be 0 for a shippable spec.

    Both scripts are covered because the unit differs: eight *characters* of the
    CJK-only stream, eight *words* of latin text. Eight characters of English
    would be "exclude " and would flag everything.
    """
    assert S.verbatim_overlap(_description(SAFE_YAML), user_text, 8) == set()
    assert S.verbatim_overlap(_description(leaky), user_text, 8) != set(), (
        "a description that pastes the request must be detected — this is the one "
        "field that ships inside every YAML")


@pytest.mark.parametrize("leaky", [LEAKY_YAML_ZH, LEAKY_YAML_EN])
def test_write_case_rejects_a_gold_spec_that_quotes_the_user(tmp_path, internal, leaky):
    user_text = FAKE_USER_TEXT_ZH if leaky is LEAKY_YAML_ZH else FAKE_USER_TEXT_EN
    case = make_case(
        user_text,
        lang=S.Lang.ZH if leaky is LEAKY_YAML_ZH else S.Lang.EN,
        zh_variant=S.ZhVariant.HANT if leaky is LEAKY_YAML_ZH else None,
        gold_specs=[S.GoldSpec(role=S.GoldRole.PRIMARY, yaml_text=leaky)],
        gold_source=S.GoldSource.HANDWRITTEN,
        gold_verification_level=S.GoldVerificationLevel.STATIC_VALID,
    )
    with pytest.raises(S.PrivacyError, match="reproduces the user's wording"):
        S.write_case(tmp_path / "cases.jsonl", case, source_text=user_text)
    assert not (tmp_path / "cases.jsonl").exists(), (
        "the record must not be appended before the check runs")


def test_write_attempt_also_checks_the_model_output(tmp_path, internal):
    attempt = S.AttemptRecord(
        case_id=S.new_case_id(), attempt_index=1,
        sampling_purpose=S.SamplingPurpose.CONVERT,
        prompt_sha256=S.sha256_hex("prompt"), yaml_text=LEAKY_YAML_ZH,
        stopped_at_gate=S.Gate.PASSED)
    with pytest.raises(S.PrivacyError, match="reproduces the user's wording"):
        S.write_attempt(tmp_path / "attempts.jsonl", attempt,
                        source_text=FAKE_USER_TEXT_ZH)


def test_write_attempt_rejects_an_entire_short_source_even_without_8gram_overlap(
        tmp_path, internal):
    """A copied three-token request is private even though n-grams cannot see it."""
    source_text = "Buy BTC now"  # invented fixture; never corpus text
    yaml_text = SAFE_YAML + "\n# BUY btc NOW\n"
    assert S.verbatim_overlap(yaml_text, source_text, 8) == set()

    attempt = S.AttemptRecord(
        case_id=S.new_case_id(), attempt_index=1,
        sampling_purpose=S.SamplingPurpose.CONVERT,
        prompt_sha256=S.sha256_hex("prompt"), yaml_text=yaml_text,
        stopped_at_gate=S.Gate.PASSED)
    with pytest.raises(S.PrivacyError, match="entire user's source text"):
        S.write_attempt(tmp_path / "attempts.jsonl", attempt, source_text=source_text)
    assert not (tmp_path / "attempts.jsonl").exists()


def test_verbatim_check_cannot_be_skipped_by_omission(tmp_path, internal):
    """``source_text`` is a required keyword; the opt-out is an explicit sentinel
    and is only valid for synthetic cases."""
    case = make_case(FAKE_USER_TEXT_ZH,
                     gold_specs=[S.GoldSpec(role=S.GoldRole.PRIMARY,
                                            yaml_text=LEAKY_YAML_ZH)],
                     gold_source=S.GoldSource.HANDWRITTEN,
                     gold_verification_level=S.GoldVerificationLevel.STATIC_VALID)
    with pytest.raises(TypeError):
        S.write_case(tmp_path / "cases.jsonl", case)  # no source_text at all
    with pytest.raises(S.RecordError, match="only valid for text_provenance=synthetic"):
        S.write_case(tmp_path / "cases.jsonl", case, source_text=S.NO_SOURCE_TEXT)
    with pytest.raises(S.RecordError, match="None is not accepted"):
        S.write_case(tmp_path / "cases.jsonl", case, source_text=None)
    with pytest.raises(S.RecordError, match="hash to text_sha256"):
        S.write_case(tmp_path / "cases.jsonl", case, source_text="some other text")


def test_write_case_pair_wires_the_check_to_the_real_text(tmp_path, internal):
    """The recommended path: the internal half supplies the text, so the check
    cannot be run against the wrong string."""
    text = FAKE_USER_TEXT_ZH
    case_id = S.new_case_id()
    pseudonym = S.hmac_pseudonym("123456789")
    internal_record = S.InternalCaseRecord(
        case_id=case_id, user_text_raw=text, user_text_redacted=text,
        pseudonym_id=pseudonym, user_id="123456789")
    leaky = make_case(text, case_id=case_id, pseudonym_id=pseudonym,
                      gold_specs=[S.GoldSpec(role=S.GoldRole.PRIMARY,
                                             yaml_text=LEAKY_YAML_ZH)],
                      gold_source=S.GoldSource.HANDWRITTEN,
                      gold_verification_level=S.GoldVerificationLevel.STATIC_VALID)
    with pytest.raises(S.PrivacyError, match="reproduces the user's wording"):
        S.write_case_pair(tmp_path / "cases.jsonl", internal / "cases_internal.jsonl",
                          leaky, internal_record)
    assert not (internal / "cases_internal.jsonl").exists(), (
        "a rejected pair must not leave the internal half behind: both files are "
        "append-only, so the retry would duplicate it")
    safe = make_case(text, case_id=case_id, pseudonym_id=pseudonym,
                     gold_specs=[S.GoldSpec(role=S.GoldRole.PRIMARY,
                                            yaml_text=SAFE_YAML)],
                     gold_source=S.GoldSource.HANDWRITTEN,
                     gold_verification_level=S.GoldVerificationLevel.STATIC_VALID)
    S.write_case_pair(tmp_path / "cases.jsonl", internal / "cases_internal.jsonl",
                      safe, internal_record)
    assert len(list(S.read_cases(tmp_path / "cases.jsonl"))) == 1


def _write_auditable_pair(tmp_path, internal, text=FAKE_USER_TEXT_ZH):
    """Create one typed public/private case pair without exposing it in output."""
    case_id = S.new_case_id()
    pseudonym = S.hmac_pseudonym("123456789")
    case = make_case(text, case_id=case_id, pseudonym_id=pseudonym)
    private = S.InternalCaseRecord(
        case_id=case_id, user_text_raw=text, user_text_redacted="[redacted]",
        pseudonym_id=pseudonym, user_id="123456789")
    public_path = tmp_path / "cases.jsonl"
    private_path = internal / "cases_demo_internal.jsonl"
    S.write_case(public_path, case, source_text=text)
    S.write_case_internal(private_path, private)
    return public_path, private_path, case, private


def test_public_private_link_audit_requires_one_matching_typed_private_record(
    tmp_path, internal,
):
    public_path, private_path, case, _private = _write_auditable_pair(tmp_path, internal)
    # The mining store has a deliberately incompatible shape and must never be
    # accidentally passed to the typed-case audit by automatic discovery.
    (internal / "cases_internal.jsonl").write_text("{}\n", encoding="utf-8")

    discovered = S.discover_case_internal_paths(internal)
    linked = S.audit_public_private_case_links(public_path, discovered)

    assert discovered == (private_path,)
    assert linked == {case.case_id: FAKE_USER_TEXT_ZH}


@pytest.mark.parametrize("fault", ["missing", "duplicate", "mismatch"])
def test_public_private_link_audit_fails_closed_without_echoing_private_values(
    tmp_path, internal, fault,
):
    public_path, private_path, case, private = _write_auditable_pair(tmp_path, internal)
    if fault == "missing":
        private_path.unlink()
    elif fault == "duplicate":
        S.write_case_internal(private_path, private)
    else:
        mismatched = S.InternalCaseRecord(
            case_id=private.case_id, user_text_raw=FAKE_USER_TEXT_EN,
            user_text_redacted="[redacted]", pseudonym_id=private.pseudonym_id,
            user_id=private.user_id)
        private_path.unlink()
        S.write_case_internal(private_path, mismatched)

    with pytest.raises(S.RecordError) as excinfo:
        S.audit_public_private_case_links(public_path, [private_path])

    detail = str(excinfo.value)
    assert case.case_id in detail
    assert FAKE_USER_TEXT_ZH not in detail
    assert FAKE_USER_TEXT_EN not in detail
    assert private.pseudonym_id not in detail
    assert str(private_path) not in detail


def test_public_private_link_audit_rejects_a_private_half_for_a_synthetic_case(
    tmp_path, internal,
):
    public_path, private_path, case, _private = _write_auditable_pair(tmp_path, internal)
    synthetic = make_case(
        FAKE_USER_TEXT_ZH, case_id=case.case_id, pseudonym_id=None,
        text_provenance=S.TextProvenance.SYNTHETIC,
        mining_source=S.MiningSource.SYNTHETIC_TEMPLATE,
    )
    public_path.unlink()
    S.write_case(public_path, synthetic, source_text=S.NO_SOURCE_TEXT)

    with pytest.raises(S.RecordError, match="synthetic public case has a private record"):
        S.audit_public_private_case_links(public_path, [private_path])


#: Keys under which user prose can live in the internal store. Read raw rather
#: than through :func:`read_cases_internal` on purpose: the internal file is
#: written by the mining stage and may be a record shape this module has not seen
#: (it currently is), and a sweep that only works on the shape it expects is a
#: sweep that silently stops running.
INTERNAL_TEXT_KEYS = ("user_text_raw", "user_text_redacted", "user_text_excerpt",
                      "first_query", "canon_text", "quote", "text")
#: Ids the two halves might be joined on, in order of preference.
JOIN_KEYS = ("case_id", "row_id", "id")
#: Below this length a shared 8-gram is not a quote; above it, a public string is
#: prose and prose in a public record has to be our words.
PROSE_MIN_CHARS = 24


def _strings(payload, out=None):
    out = [] if out is None else out
    if isinstance(payload, dict):
        for value in payload.values():
            _strings(value, out)
    elif isinstance(payload, list):
        for value in payload:
            _strings(value, out)
    elif isinstance(payload, str):
        out.append(payload)
    return out


def _join_id(payload):
    for key in JOIN_KEYS:
        if isinstance(payload.get(key), str):
            return payload[key]
    return None


def test_public_prose_does_not_quote_the_internal_store():
    """End-to-end sweep: no *public* string reuses the user's words.

    Broader than ``strategy.description`` deliberately. The description is where
    it happens most, but the guard has to hold for any public string field — a
    field named something innocuous is exactly how prose gets past a key-name
    check.
    """
    files = _publishable_dataset_files()
    internal_path = S.DEFAULT_INTERNAL_ROOT / "cases_internal.jsonl"
    public_prose = {}  # join id -> list of (file:line, string)
    for rel in files:
        for lineno, line in enumerate((REPO / rel).read_text("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            join = _join_id(payload)
            if join is None:
                continue
            prose = [s for s in _strings(payload) if len(s) >= PROSE_MIN_CHARS]
            if prose:
                public_prose.setdefault(join, []).extend(
                    ("%s:%d" % (rel, lineno), s) for s in prose)
    if not public_prose or not internal_path.exists():
        pytest.skip("nothing to compare yet: %d publishable file(s), internal store %s"
                    % (len(files), "present" if internal_path.exists() else "absent"))
    offenders = []
    with internal_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            entries = public_prose.get(_join_id(payload) or "")
            if not entries:
                continue
            source = " ".join(str(payload.get(k) or "") for k in INTERNAL_TEXT_KEYS)
            for where, text in entries:
                shared = S.verbatim_overlap(text, source, 8)
                if shared:
                    offenders.append("%s: %d shared 8-gram(s)" % (where, len(shared)))
    assert offenders == [], offenders[:10]


# ---------------------------------------------------------------------------
# 5. write_case() raises on user text — it does not drop it
# ---------------------------------------------------------------------------

#: The shape of the mistake this catches: the export's own column names, handed
#: straight to the writer. The CSV has user_id / first_query / user_text_excerpt.
CSV_SHAPED_EXTRAS = [
    {"user_text_raw": FAKE_USER_TEXT_ZH},
    {"user_text_excerpt": FAKE_USER_TEXT_ZH},
    {"first_query": FAKE_USER_TEXT_ZH},
    {"user_id": "123456789"},
    {"quote": "排除美股永續"},
    {"prompt": "system...\nuser: " + FAKE_USER_TEXT_ZH},
]


@pytest.mark.parametrize("extra", CSV_SHAPED_EXTRAS,
                         ids=[list(e)[0] for e in CSV_SHAPED_EXTRAS])
def test_write_case_raises_on_user_text(tmp_path, internal, extra):
    payload = S.validate_record(make_case(FAKE_USER_TEXT_ZH))
    payload.update(extra)
    with pytest.raises(S.PrivacyError, match=list(extra)[0]):
        S.write_case(tmp_path / "cases.jsonl", payload, source_text=FAKE_USER_TEXT_ZH)
    assert not (tmp_path / "cases.jsonl").exists()


def test_write_case_raises_on_a_nested_quote(tmp_path, internal):
    """``conditions[].quote`` is the realistic version: the internal condition
    list handed to the public writer unchanged."""
    payload = S.validate_record(make_case(FAKE_USER_TEXT_ZH))
    payload["conditions"] = [{
        "cid": "c1", "polarity": "exclude", "subject": "underlying_type",
        "operator": "neq", "value": "EQUITY", "unit": "label", "scope": "universe",
        "timeframe": None, "measurability": "quantified", "is_ranking": False,
        "rank_direction": None, "quote": "排除美股永續",
    }]
    with pytest.raises(S.PrivacyError, match="quote"):
        S.write_case(tmp_path / "cases.jsonl", payload, source_text=FAKE_USER_TEXT_ZH)


def test_the_guard_raises_rather_than_dropping(tmp_path, internal):
    """A silent drop would teach the caller that this is fine, and the next field
    they invent will not be on the list."""
    payload = S.validate_record(make_case(FAKE_USER_TEXT_ZH))
    payload["user_text_raw"] = FAKE_USER_TEXT_ZH
    with pytest.raises(S.PrivacyError) as excinfo:
        S.write_case(tmp_path / "cases.jsonl", payload, source_text=FAKE_USER_TEXT_ZH)
    assert "raise, not a drop" in str(excinfo.value)


def test_attempt_writer_has_the_same_guard(tmp_path, internal):
    payload = {
        "case_id": S.new_case_id(), "attempt_index": 1, "sampling_purpose": "convert",
        "prompt_sha256": S.sha256_hex("p"), "yaml_text": SAFE_YAML,
        "stopped_at_gate": "passed", "user_text_raw": FAKE_USER_TEXT_ZH,
    }
    with pytest.raises(S.PrivacyError, match="user_text_raw"):
        S.write_attempt(tmp_path / "attempts.jsonl", payload,
                        source_text=FAKE_USER_TEXT_ZH)


def test_internal_records_cannot_be_written_by_the_public_writer(tmp_path, internal):
    record = S.InternalCaseRecord(
        case_id=S.new_case_id(), user_text_raw=FAKE_USER_TEXT_ZH,
        user_text_redacted="[redacted]", pseudonym_id=S.hmac_pseudonym("1"),
        user_id="1")
    with pytest.raises(S.RecordError):
        S.write_case(tmp_path / "cases.jsonl", record, source_text=FAKE_USER_TEXT_ZH)
