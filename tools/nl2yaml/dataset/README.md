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
