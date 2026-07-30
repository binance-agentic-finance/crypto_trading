"""Guard: no credentials or internal hostnames in the tracked source tree.

This repository is PUBLIC. Two things leaked into it and stayed for months:

* two Binance API key/secret pairs — one of them written as an ``os.getenv``
  *default*, so the real production futures key was used whenever the env var
  was unset (``cyqnt_trd/test.py``, ``diagnose_api.py``,
  ``online_trading/realtime_price_tracker.py``, 2025-12-13 → 2026-07-30);
* an internal QA gateway host labelled "public QA gateway" in a comment, which
  in fact resolves to an RFC1918 address and is only reachable inside the
  corporate network (``_vendor/binance_bigdata_client.py``).

Neither was caught by review, because nothing looked for them. These tests do.

They scan the **git-tracked** tree only, so scratch files and ignored data are
out of scope, and they read files as bytes so a stray encoding cannot hide a
match.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Extensions worth scanning. Parquet/PNG etc. would only produce noise.
#:
#: ``.html`` is here because the generated spec pages quote endpoints verbatim out
#: of the catalog: the first sweep found ``eureka.local`` in
#: ``docs/full_flow_spec.html`` while this guard reported clean.
SCAN_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".sh", ".toml", ".txt",
                 ".cfg", ".html", ".ini", ".js", ".ts"}

#: This file necessarily contains the patterns it forbids.
SELF = Path(__file__).relative_to(REPO).as_posix()

#: Generated artifacts that legitimately embed long hashes/ids.
ALLOW_LONG_TOKEN_IN = {
    "strategies/_standard/signal.schema.v2.json",
    "strategies/_standard/input.schema.v1.json",
}

#: A Binance API key/secret is 64 chars of mixed-case alphanumerics. Require
#: BOTH cases and a digit so hex hashes, uuids and base64 blobs do not match.
_SECRET_RE = re.compile(rb"(?<![A-Za-z0-9])(?=[A-Za-z0-9]{64}(?![A-Za-z0-9]))"
                        rb"(?=[A-Za-z0-9]*[a-z])(?=[A-Za-z0-9]*[A-Z])"
                        rb"(?=[A-Za-z0-9]*[0-9])[A-Za-z0-9]{64}")

#: Hosts, host:port pairs and API routes that only exist inside the corporate
#: network. The field contracts they serve stay in the repo on purpose (see
#: ``standard_bot/data/internal_slots.py``) — the addresses do not.
#:
#: The previous version of this pattern required a dot immediately before
#: ``eureka``, so ``bdp-search-mcp.eureka.local`` was caught while a bare
#: ``eureka.local`` — the form that had actually been written into the catalog and
#: propagated into 21 lines of generated documentation — was not. Each alternative
#: below is therefore anchored so the *shortest* form of the name still matches.
_INTERNAL_HOST_RE = re.compile(
    rb"("
    rb"qa1fdg\.net"
    # eureka.local, bdp-x.eureka.local, x.eureka.qa.local — subdomains optional
    rb"|(?:[A-Za-z0-9-]+\.)*eureka(?:\.[A-Za-z0-9-]+)*\.local"
    # bin.internal and anything under it
    rb"|(?:[A-Za-z0-9-]+\.)*bin\.internal"
    # named internal ENDPOINTS. Deliberately an explicit list rather than
    # ``bdp-[a-z-]+``: system and Kafka-topic names (bdp-bot,
    # bdp-ai-trading-bot) are identifiers, not addresses, and sweeping them in
    # here would make this test fire on things it is not about.
    rb"|bdp-search-mcp|bdp-screening|bdp-gateway"
    # internal HTTP route of the indicators service
    rb"|ai-report/v\d"
    # service:port on a non-privileged internal port, e.g. some-service:13106
    rb"|\b[a-z][a-z0-9]*(?:-[a-z0-9]+)+:1[0-9]{4}\b"
    rb")", re.IGNORECASE)

#: Illustrative hostnames that are meant to look internal and resolve nowhere.
#: Listed so the guard can stay strict without flagging documentation examples.
DOC_PLACEHOLDER_HOSTS = (b"example.internal", b"example.invalid", b"example.test",
                         b"research.internal", b"warehouse.internal")

#: Credentials assigned inline rather than read from the environment.
_INLINE_CREDENTIAL_RE = re.compile(
    rb"""(api_?key|api_?secret|secret_?key|password|token)\s*=\s*["'][A-Za-z0-9_\-]{16,}["']""",
    re.IGNORECASE)


def _tracked_files():
    """Every file a PR from this working tree would publish.

    Tracked files AND not-yet-committed ones that are not gitignored — because a
    pull request adds the second kind, and a guard that only looks at what is
    already committed cannot stop a leak from *entering*. That hole was not
    hypothetical: every file carrying ``eureka.local`` was untracked at the time
    this test reported clean, i.e. it was blind to precisely the files under
    review.
    """
    listings = (
        ["git", "ls-files"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    )
    seen = set()
    for argv in listings:
        out = subprocess.run(argv, cwd=REPO, capture_output=True,
                             text=True, check=True).stdout.split("\n")
        for rel in out:
            rel = rel.strip()
            if not rel or rel == SELF or rel in seen:
                continue
            seen.add(rel)
            path = REPO / rel
            if path.suffix.lower() not in SCAN_SUFFIXES or not path.is_file():
                continue
            yield rel, path


#: Obvious placeholders — a guard that cries wolf gets switched off.
_PLACEHOLDER_RE = re.compile(
    rb"(your[-_]?token|changeme|placeholder|xxx+|\.\.\.|<[^>]+>|example|dummy)",
    re.IGNORECASE)


def _hits(pattern, *, skip=frozenset(), placeholders=()):
    found = []
    for rel, path in _tracked_files():
        if rel in skip:
            continue
        try:
            blob = path.read_bytes()
        except OSError:
            continue
        for match in pattern.finditer(blob):
            if _PLACEHOLDER_RE.search(match.group(0)):
                continue
            # a documentation example that is *meant* to look internal
            if any(ph in blob[max(0, match.start() - 40):match.end() + 40]
                   for ph in placeholders):
                continue
            line = blob[:match.start()].count(b"\n") + 1
            found.append("%s:%d" % (rel, line))
    return found


def test_no_api_key_shaped_secrets():
    """No 64-char mixed-case token — the shape of a Binance key or secret."""
    hits = _hits(_SECRET_RE, skip=ALLOW_LONG_TOKEN_IN)
    assert not hits, (
        "possible API key/secret committed to a PUBLIC repo at: %s\n"
        "Read credentials from the environment with NO default. Note that "
        "deleting the line does not undo the exposure — rotate the key." % hits)


def test_no_inline_credentials():
    """``api_key="..."`` / ``os.getenv("API_KEY", "<real key>")`` are both banned.

    The second form is the one that actually burned us: it reads like it uses
    the environment while shipping a live credential as the fallback.
    """
    hits = _hits(_INLINE_CREDENTIAL_RE)
    assert not hits, (
        "credential assigned inline at: %s\n"
        "Use os.environ[...] (or getenv with NO default) and fail loudly when "
        "it is unset." % hits)


def test_no_internal_hostnames():
    """Internal hosts resolve to RFC1918 addresses and must not be published."""
    hits = _hits(_INTERNAL_HOST_RE, placeholders=DOC_PLACEHOLDER_HOSTS)
    assert not hits, (
        "internal hostname in a PUBLIC repo at: %s\n"
        "Move it behind an environment variable; keep the field contract in "
        "code (see standard_bot/data/internal_slots.py) and the host out." % hits)


@pytest.mark.parametrize("removed", ["cyqnt_trd/test.py", "cyqnt_trd/diagnose_api.py"])
def test_files_that_leaked_keys_stay_untracked(removed):
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", removed],
                             cwd=REPO, capture_output=True, text=True)
    assert tracked.returncode != 0, (
        "%s is tracked again; it previously carried a production API key/secret "
        "and is listed in .gitignore." % removed)
