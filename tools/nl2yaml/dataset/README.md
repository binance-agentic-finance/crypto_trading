# nl2yaml dataset

Small, reviewable outputs are tracked here. The two big things are not:

| artifact | where | why |
|---|---|---|
| `candidates.jsonl` (43 MB, 26,161 mined rows) | **gitignored** | derived; regenerate with `python -m tools.nl2yaml.mine` |
| user text, per-condition quotes, `user_id` | `~/nl2yaml_internal/` (outside the repo) | both git remotes are PUBLIC |

Tracked:

- `cases.jsonl` — case records with gold, verdicts and capability map (no user text)
- `funnel.json` / `funnel_report.md` — Gate0 funnel over the full corpus
- `measure.json` — capability-weighted counts
- `gaps.jsonl` — versioned Gate0 aggregate-gap stream, beginning with the
  `cyqnt.nl2yaml.gate0-gaps/v1` header and then canonical gaps ranked by
  `dup_weighted_count`; this is the "what to build next" list, and it is
  weighted because a gap sitting behind a prompt card that fires 393 times is
  not one unit of demand. Read it with `schema.read_measure_gaps()`, not the
  per-case `read_gaps()` ledger reader.

## Private queue → reviewed YAML → frozen replay

`python -m tools.nl2yaml.process_queue_case` is the bounded bridge for one
row from the private `strategy_test_queue.csv`. It does **not** call an LLM or
any endpoint: conversion happens separately and explicitly, then the operator
places the reviewed YAML and review JSON beneath `$NL2YAML_INTERNAL_ROOT` with
mode `0600`. The worker verifies that the exact queue excerpt, reviewed YAML,
structured conditions, source quotes, and frozen `cyqnt.input/v1` bundle all
agree before it can append a public case/attempt/run record.

It only accepts clear `t1_expressible` requests where every capability is
`supported`; proxy, gap, ambiguity, source drift, or a failed frozen replay are
refused before either ledger is written. `promote_to_gold=true` additionally
requires every reviewed condition to pass G1e and a real human reviewer alias
plus timestamp. A GPT/other-model review may be kept in the private curation
receipt, but it is never accepted as `human_reviewed_by` and cannot make a row
SFT-eligible on its own.

The review must carry a canonical `bundle_sha256`; the worker rejects a replay
against any other frozen bundle and verifies that `--repo-git-sha` is the actual
clean local `HEAD` with no tracked or untracked files. Gate conditions and public case conditions must be a
lossless match, not merely share a condition ID. To avoid promoting a reviewer
invented reading, numeric thresholds, basket counts, and ranking direction each
need a direct quoted source span with an exact numeric literal and relevant
unit. Free-form membership/equality labels are held out of this bridge until
they have a closed public vocabulary.

All review-derived identifiers (`source_snapshot_id`, converter/model identity,
and reviewer alias) are retained only in the mode-600 private curation receipt;
the public ledger gets domain-separated HMAC labels instead. The public run's
snapshot handle is derived from the already-public bundle hash rather than the
bundle's private snapshot label. Full gate errors and
per-condition diagnostics are likewise retained only in
`queue_evaluation_diagnostics.jsonl` beneath the internal root. A per-case
private file claim serialises writes; an exact retry recovers a partial append,
while any record mismatch is refused rather than duplicated or overwritten.

The bridge accepts only queue identities of at least eight characters. A shorter
opaque account label cannot be reliably distinguished from ordinary YAML text,
so it is held for a stronger private identity contract rather than scanned and
possibly leaked.

The human alias is an operational sign-off record, not cryptographic proof of a
person's identity. If that assurance is required, put a signed/allowlisted
reviewer registry in front of the private review JSON before using this bridge.
