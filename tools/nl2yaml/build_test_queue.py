"""選標的 / 策略需求的測試佇列 —— 從 Gate0 的挖掘結果導出,不重做過濾。

一份給人(或給另一個 AI)**一筆一筆去做成策略再回測**的清單。

為什麼不直接對原始 CSV 寫一套新的過濾
--------------------------------------
``mine.py`` 已經對全量 51,595 列跑過近重複分群、續聊碎片偵測、tier 分層與
``spec_shape`` 判定,而且它的輸出 ``candidates.jsonl`` 是可提交的(無原文)。
在這裡重寫一套 regex 只會得到第二套跟第一套不一致的數字 —— 而兩份不一致的清單,
沒有人分得出哪一份才是被量測過的那一份。所以這支腳本只做三件 mine.py 刻意
沒做的事:剔模版卡、去重、依形狀與 tier 收窄,然後才回原始 CSV 取原文。

隱私
----
輸出**含使用者原文與 user_id**,所以只能寫進
``schema.internal_root()``（repository 之外的 private directory）。gitignore
不是安全邊界：任何 repository 路徑，即使目前被忽略，也會被拒絕。

用法
----
    python -m tools.nl2yaml.build_test_queue
    python -m tools.nl2yaml.build_test_queue --shapes selection,both --tiers A
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

from . import mine
from . import schema

REPO = Path(__file__).resolve().parents[2]
PRIVATE_SOURCE_RELATIVE = (
    Path("user_demand_analysis") / "2026-05_07_trading_intent"
    / "trading_intent_chats_2026-05_07_zh_en.csv"
)
DEFAULT_CANDIDATES = REPO / "tools" / "nl2yaml" / "dataset" / "candidates.jsonl"
DEFAULT_OUT_NAME = "strategy_test_queue.csv"

#: Columns copied straight from the source CSV. ``first_query`` and
#: ``user_text_excerpt`` are the two that carry the request itself.
SOURCE_COLUMNS = (
    "user_id", "chat_id", "month", "day", "lang",
    "is_coin_selection", "selection_basis", "wants_automation", "wants_backtest",
    "wants_strategy", "assets_mentioned", "n_user_msgs",
    "first_query", "user_text_excerpt",
)

OUT_COLUMNS = (
    "queue_rank", "cluster_id", "row_id", "dup_count", "rows_in_queue_scope",
    "spec_shape", "tier", "n_conditions", "families", "conditions_json",
) + SOURCE_COLUMNS


#: A first_query that is a product suggestion chip, a system-injected preset
#: invocation, or a routing directive — not something a user typed.
#:
#: The corpus's own ``preset_case`` column finds only 3,424 of them across 15
#: named cards. Measured against it, 11,835 rows open with an emoji — the
#: signature of a UI chip — and only 104 of those carry a preset_case value. So
#: the column under-detects by roughly 5x, and the chips it misses are the
#: highest-volume rows in the corpus: "View today's crypto market briefing" was
#: sent by 998 distinct users verbatim, "📈 Help me set up a BTC futures position
#: with 10x leverage" by 483. Ordered by frequency, an unfiltered queue puts a
#: button at rank 1.
#:
#: Four signals, unioned. A plain repeat count will not do it alone: the 3-to-9
#: user band is half localised chips ("📈 Segits beallitani egy BTC futures
#: poziciot 10x tokeattetellel") and half genuine requests that two or three
#: people happened to phrase identically ("I want to trade BTC using a
#: range-grid strategy. Backtest it..."), and a threshold low enough to catch
#: the first throws away the second.
_EMOJI_PREFIX = re.compile(r"^[\U0001F300-\U0001FAFF\u2190-\u21FF\u2600-\u27BF\uFE0F]")
_PRESET_INVOCATION = re.compile(r"User selected the [a-z0-9-]+ case", re.I)
_SYSTEM_DIRECTIVE = re.compile(r"^\s*<system>")

#: Verbatim reuse by this many DISTINCT users is a chip whatever it says. Users
#: do not independently type the same sentence ten times; localisation and A/B
#: variants of one button do.
_TEMPLATE_MIN_USERS = 10
#: An emoji opener is already strong evidence, so it needs far less repetition —
#: but not none, or a user who opens with 🚀 gets dropped for punctuation.
_EMOJI_MIN_USERS = 2


def default_out_path() -> Path:
    """The private default, evaluated after environment configuration.

    ``NL2YAML_INTERNAL_ROOT`` is intentionally read here rather than at module
    import time.  That makes the command-line default follow the same privacy
    boundary as the rest of the dataset tooling, including test or operator
    overrides.
    """
    return schema.internal_root() / DEFAULT_OUT_NAME


def default_csv_path() -> Path:
    """Return the private raw corpus location, never a repo-local fallback.

    The source has user IDs and verbatim requests.  ``.gitignore`` is not a
    security boundary, so a convenience default must be beneath the same
    private root as the queue it produces.  An operator may still pass
    ``--csv`` for a separately protected export.
    """
    return schema.internal_root() / PRIVATE_SOURCE_RELATIVE


def _is_beneath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _private_output_path(out_path: Path) -> tuple[Path, Path]:
    """Validate and prepare the actual private output location.

    ``schema.internal_root`` checks the configured path textually.  Resolve it
    here as well, before creating anything, so a symlinked parent cannot turn a
    seemingly external override into a repository write.  The requested file
    must be a descendant (not the root itself), and it may never resolve into
    this repository.
    """
    configured_root = schema.internal_root()
    private_root = configured_root.resolve()
    repo_root = REPO.resolve()
    target = Path(out_path).expanduser().resolve()

    if _is_beneath(private_root, repo_root):
        raise schema.PrivacyError(
            "internal root %s resolves inside the repository (%s); refusing private queue write"
            % (private_root, repo_root))
    if target == private_root or not _is_beneath(target, private_root):
        raise schema.PrivacyError(
            "refusing to write %s: this queue carries verbatim user questions and "
            "user_ids, so it must be beneath the internal root %s"
            % (target, private_root))
    if _is_beneath(target, repo_root):
        raise schema.PrivacyError(
            "refusing to write %s: no private queue may be written inside the repository %s"
            % (target, repo_root))
    if target.exists():
        if not target.is_file():
            raise schema.PrivacyError(
                "refusing to write %s: private queue target must be a regular file" % target)
        # Tighten an old queue before opening the source corpus.  Otherwise a
        # prior 644 file remains readable for the whole filtering pass even
        # though the replacement below is written safely.
        os.chmod(target, 0o600)

    private_root.mkdir(parents=True, exist_ok=True)
    os.chmod(private_root, 0o700)

    # A nested queue directory must not undo the root's protection.  Build it
    # a component at a time after the resolved containment check above; this is
    # also what makes a custom ``--out`` safe without requiring a special name.
    current = private_root
    for part in target.parent.relative_to(private_root).parts:
        current /= part
        current.mkdir(exist_ok=True)
        os.chmod(current, 0o700)
    return private_root, target


def _write_private_csv(out_path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """Write a queue atomically with mode 600 from its first visible byte."""
    # ``mkstemp`` creates the staging file as 600.  ``os.replace`` means an
    # old output (already tightened before source input is read) is never
    # truncated and refilled before it is replaced by the private file.
    fd, tmp_name = tempfile.mkstemp(prefix=".%s." % out_path.name, suffix=".tmp",
                                    dir=str(out_path.parent))
    tmp_path = Path(tmp_name)
    try:
        os.chmod(tmp_path, 0o600)
        with os.fdopen(fd, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(OUT_COLUMNS))
            writer.writeheader()
            for record in records:
                writer.writerow(record)
        os.replace(tmp_path, out_path)
        os.chmod(out_path, 0o600)
    finally:
        # On write/replace errors, do not leave a partially written private
        # queue around for a later run to mistake for a complete one.
        if tmp_path.exists():
            tmp_path.unlink()


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def template_reason(row: Dict[str, str], users_per_query: Dict[str, set]) -> str:
    """Why this row is a template, or "" if it is a real request."""
    query = _normalise(row.get("first_query", ""))
    seen_by = len(users_per_query.get(query, ()))
    if row.get("preset_case"):
        return "preset_case"
    if _PRESET_INVOCATION.search(query):
        return "preset_invocation"
    if _SYSTEM_DIRECTIVE.match(query):
        return "system_directive"
    if _EMOJI_PREFIX.match(query) and seen_by >= _EMOJI_MIN_USERS:
        return "suggestion_chip"
    if seen_by >= _TEMPLATE_MIN_USERS:
        return "verbatim_by_%d_users" % seen_by
    return ""


def users_by_query(rows: Iterable[Dict[str, str]]) -> Dict[str, set]:
    index: Dict[str, set] = {}
    for row in rows:
        query = _normalise(row.get("first_query", ""))
        if query:
            index.setdefault(query, set()).add(row.get("user_id", ""))
    return index


def _load_candidates(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _representative(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One row to stand for the near-duplicate cluster.

    Most conditions first, then the longest canonical text, then ``row_id`` so
    the choice is deterministic. Not "the first one seen": that makes the queue
    depend on file order, and two runs of the same script would hand a reviewer
    two different sentences for the same request.
    """
    return sorted(group, key=lambda r: (-int(r.get("n_conditions") or 0),
                                        -int(r.get("canon_len") or 0),
                                        str(r.get("row_id") or "")))[0]


def build(csv_path: Path, candidates_path: Path, out_path: Path,
          shapes: Iterable[str], tiers: Iterable[str],
          keep_presets: bool = False) -> Dict[str, Any]:
    shapes, tiers = set(shapes), set(tiers)
    _, out_path = _private_output_path(out_path)

    source_rows = mine.read_rows(csv_path)
    # The source text, keyed the way mine.py keys it, so the join cannot drift.
    by_row_id = {mine.short_hash("%s|%s" % (r["chat_id"], r["day"]), "r_"): r
                 for r in source_rows}
    users_per_query = users_by_query(source_rows)
    templates = {row_id: template_reason(row, users_per_query)
                 for row_id, row in by_row_id.items()}

    candidates = _load_candidates(candidates_path)
    funnel = [("gate0 candidates", len(candidates))]

    kept = candidates
    if not keep_presets:
        kept = [r for r in kept if not templates.get(r["row_id"])]
        funnel.append(("- templates / chips", len(kept)))
    kept = [r for r in kept if r.get("spec_shape") in shapes]
    funnel.append(("- shape in %s" % sorted(shapes), len(kept)))
    kept = [r for r in kept if r.get("tier") in tiers]
    funnel.append(("- tier in %s" % sorted(tiers), len(kept)))

    clusters: Dict[str, List[Dict[str, Any]]] = {}
    for row in kept:
        clusters.setdefault(str(row.get("dup_cluster_id") or row["row_id"]), []).append(row)
    funnel.append(("- deduped to clusters", len(clusters)))

    records: List[Dict[str, Any]] = []
    missing = 0
    for cluster_id, group in clusters.items():
        rep = _representative(group)
        source = by_row_id.get(rep["row_id"])
        if source is None:
            missing += 1
            continue
        records.append({
            "cluster_id": cluster_id,
            "row_id": rep["row_id"],
            # How many chats in the WHOLE corpus said this, i.e. how much of the
            # userbase one working strategy would serve. The ordering key.
            "dup_count": int(rep.get("dup_count") or 1),
            # How many of them survived this queue's own filters. Reported
            # separately because the two disagree whenever a cluster straddles a
            # tier or shape boundary, and a single number would hide that.
            "rows_in_queue_scope": len(group),
            "spec_shape": rep.get("spec_shape"),
            "tier": rep.get("tier"),
            "n_conditions": rep.get("n_conditions"),
            "families": ",".join(rep.get("families") or ()),
            "conditions_json": json.dumps(rep.get("conditions") or [],
                                          ensure_ascii=False, sort_keys=True),
            **{col: source.get(col, "") for col in SOURCE_COLUMNS},
        })

    records.sort(key=lambda r: (-r["dup_count"], -int(r["n_conditions"] or 0),
                                r["cluster_id"]))
    for rank, record in enumerate(records, start=1):
        record["queue_rank"] = rank

    _write_private_csv(out_path, records)

    funnel.append(("written", len(records)))
    return {"funnel": funnel, "written": len(records), "unjoinable": missing,
            "out": str(out_path)}


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    help=("private source CSV (default: "
                          "$NL2YAML_INTERNAL_ROOT/%s)" % PRIVATE_SOURCE_RELATIVE))
    ap.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    ap.add_argument("--out", type=Path,
                    help="private queue path beneath $NL2YAML_INTERNAL_ROOT "
                         "(default: $NL2YAML_INTERNAL_ROOT/%s)" % DEFAULT_OUT_NAME)
    ap.add_argument("--shapes", default="selection,trade,both")
    ap.add_argument("--tiers", default="A,B")
    ap.add_argument("--keep-presets", action="store_true")
    a = ap.parse_args(argv)

    out_path = a.out or default_out_path()
    csv_path = a.csv or default_csv_path()
    summary = build(csv_path, Path(a.candidates), out_path,
                    a.shapes.split(","), a.tiers.split(","), a.keep_presets)
    width = max(len(name) for name, _ in summary["funnel"])
    for name, count in summary["funnel"]:
        print("  %-*s %6d" % (width, name, count))
    if summary["unjoinable"]:
        print("  (%d clusters had no matching CSV row and were dropped)"
              % summary["unjoinable"])
    print("\n->", summary["out"])


if __name__ == "__main__":
    main()
