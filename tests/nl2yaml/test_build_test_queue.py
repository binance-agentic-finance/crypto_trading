"""Privacy boundary tests for the private strategy-test queue.

The queue intentionally contains user text, so every fixture here is invented
and all files live under ``tmp_path``.  These tests must never open the real
chat export or the ignored historical queue.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from tools.nl2yaml import build_test_queue as queue
from tools.nl2yaml import mine, schema


def _write_source_csv(path: Path) -> list[dict[str, str]]:
    """Write a tiny, wholly synthetic source export for queue joins."""
    rows = [
        {
            "user_id": "synthetic-user-a",
            "chat_id": "synthetic-chat-a",
            "day": "2026-01-01",
            "first_query": "invented momentum request alpha",
            "user_text_excerpt": "invented momentum request alpha --- invented stop rule",
        },
        {
            "user_id": "synthetic-user-b",
            "chat_id": "synthetic-chat-b",
            "day": "2026-01-02",
            "first_query": "invented selection request beta",
            "user_text_excerpt": "invented selection request beta --- invented liquidity filter",
        },
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mine.REQUIRED_COLUMNS))
        writer.writeheader()
        for row in rows:
            record = {column: "" for column in mine.REQUIRED_COLUMNS}
            record.update({
                "month": "2026-01",
                "lang": "en",
                "zh_variant": "",
                "primary_intent": "ENTRY_EXIT_TIMING",
                "label_source": "synthetic",
                "is_coin_selection": "0",
                "wants_automation": "0",
                "wants_backtest": "1",
                "wants_strategy": "1",
                "n_user_msgs": "2",
                "n_assistant_msgs": "1",
                "kw_primary": "SYNTHETIC",
            })
            record.update(row)
            writer.writerow(record)
    return rows


def _candidate(row: dict[str, str], *, cluster: str, dup_count: int,
               shape: str = "trade", conditions: int = 2) -> dict[str, object]:
    return {
        "row_id": mine.short_hash("%s|%s" % (row["chat_id"], row["day"]), "r_"),
        "dup_cluster_id": cluster,
        "dup_count": dup_count,
        "canon_len": 40,
        "spec_shape": shape,
        "tier": "A",
        "n_conditions": conditions,
        "families": ["indicator", "risk"],
        "conditions": [{"family": "indicator", "subject": "synthetic"}],
    }


def _write_candidates(path: Path, rows: list[dict[str, str]], *, reverse: bool = False) -> None:
    candidates = [
        _candidate(rows[0], cluster="dup_synthetic_a", dup_count=2, conditions=3),
        _candidate(rows[1], cluster="dup_synthetic_b", dup_count=5, conditions=2),
    ]
    if reverse:
        candidates.reverse()
    with path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, sort_keys=True) + "\n")


@pytest.fixture
def synthetic_inputs(tmp_path: Path):
    csv_path = tmp_path / "synthetic_source.csv"
    candidates_path = tmp_path / "synthetic_candidates.jsonl"
    rows = _write_source_csv(csv_path)
    _write_candidates(candidates_path, rows)
    return csv_path, candidates_path, rows


def _set_private_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setenv(schema.INTERNAL_ROOT_ENV, str(root))


def test_default_output_path_uses_the_schema_internal_root(tmp_path, monkeypatch):
    root = tmp_path / "private-queue-root"
    _set_private_root(monkeypatch, root)

    assert queue.default_out_path() == root / queue.DEFAULT_OUT_NAME


def test_default_source_path_uses_the_schema_internal_root(tmp_path, monkeypatch):
    root = tmp_path / "private-source-root"
    _set_private_root(monkeypatch, root)

    assert queue.default_csv_path() == root / queue.PRIVATE_SOURCE_RELATIVE


def test_build_writes_private_queue_with_restrictive_modes(
        tmp_path, monkeypatch, synthetic_inputs):
    csv_path, candidates_path, _ = synthetic_inputs
    root = tmp_path / "private-queue-root"
    nested = root / "review-queues"
    root.mkdir(mode=0o755)
    nested.mkdir(mode=0o755)
    out_path = nested / "strategy_test_queue.csv"
    out_path.write_text("old synthetic queue\n", encoding="utf-8")
    out_path.chmod(0o644)
    _set_private_root(monkeypatch, root)

    original_read_rows = mine.read_rows

    def checked_read_rows(path):
        # An existing private queue must be locked down before the source is
        # opened, not only after the new result is ready.
        assert out_path.stat().st_mode & 0o777 == 0o600
        return original_read_rows(path)

    monkeypatch.setattr(queue.mine, "read_rows", checked_read_rows)

    summary = queue.build(csv_path, candidates_path, out_path, ["trade"], ["A"])

    assert Path(summary["out"]) == out_path.resolve()
    assert summary["written"] == 2
    assert root.stat().st_mode & 0o777 == 0o700
    assert nested.stat().st_mode & 0o777 == 0o700
    assert out_path.stat().st_mode & 0o777 == 0o600


def test_build_rejects_every_repository_output_path_before_reading_inputs(
        tmp_path, monkeypatch):
    root = tmp_path / "private-queue-root"
    _set_private_root(monkeypatch, root)
    repo_target = queue.REPO / "docs" / "user_demand_analysis" / "queue.csv"

    # Deliberately nonexistent inputs prove the privacy refusal happens before
    # any caller can accidentally open a source corpus.
    with pytest.raises(schema.PrivacyError, match="internal root|repository|beneath"):
        queue.build(tmp_path / "do-not-read.csv", tmp_path / "do-not-read.jsonl",
                    repo_target, ["trade"], ["A"])
    assert not repo_target.exists()


def test_build_rejects_paths_outside_the_schema_internal_root(
        tmp_path, monkeypatch, synthetic_inputs):
    csv_path, candidates_path, _ = synthetic_inputs
    root = tmp_path / "private-queue-root"
    _set_private_root(monkeypatch, root)

    with pytest.raises(schema.PrivacyError, match="beneath the internal root"):
        queue.build(csv_path, candidates_path, tmp_path / "not-private.csv", ["trade"], ["A"])


def test_build_resolves_and_rejects_a_symlink_escape(tmp_path, monkeypatch, synthetic_inputs):
    csv_path, candidates_path, _ = synthetic_inputs
    root = tmp_path / "private-queue-root"
    root.mkdir()
    (root / "escape").symlink_to(queue.REPO, target_is_directory=True)
    _set_private_root(monkeypatch, root)

    with pytest.raises(schema.PrivacyError, match="beneath the internal root|repository"):
        queue.build(csv_path, candidates_path, root / "escape" / "queue.csv", ["trade"], ["A"])


def test_cli_default_is_private_and_queue_order_stays_deterministic(
        tmp_path, monkeypatch, synthetic_inputs, capsys):
    csv_path, candidates_path, rows = synthetic_inputs
    root = tmp_path / "private-queue-root"
    _set_private_root(monkeypatch, root)

    queue.main(["--csv", str(csv_path), "--candidates", str(candidates_path),
                "--shapes", "trade", "--tiers", "A"])
    default_out = root / queue.DEFAULT_OUT_NAME
    first = default_out.read_bytes()
    printed = capsys.readouterr().out
    assert default_out.stat().st_mode & 0o777 == 0o600
    assert "invented momentum request" not in printed
    assert "invented selection request" not in printed

    reordered_candidates = tmp_path / "reordered_candidates.jsonl"
    _write_candidates(reordered_candidates, rows, reverse=True)
    second_out = root / "second" / queue.DEFAULT_OUT_NAME
    queue.build(csv_path, reordered_candidates, second_out, ["trade"], ["A"])

    assert second_out.read_bytes() == first
    assert os.stat(second_out).st_mode & 0o777 == 0o600
