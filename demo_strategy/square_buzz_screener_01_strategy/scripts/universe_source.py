"""Attention-driven universe provider.

Runs *outside* the pure template layer (attention scraping is I/O — the
template contract forbids I/O). The daemon calls this once per tick to
get the current candidate list + attention metadata, which it then
threads into `SelectionContext.metadata` for the template to score.

This is a legitimate framework-layer helper: the template still doesn't
fetch anything, the daemon does.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

log = logging.getLogger(__name__)


def _try_atomic_lib_source():
    """Return (hotrank_fn, trending_fn) tuple if atomic_strategy_lib is
    available, else (None, None)."""
    try:
        from atomic_strategy_lib.data.search import hotrank_fetch, trending_fetch
        return hotrank_fetch, trending_fetch
    except Exception:  # noqa: BLE001
        return None, None


def get_universe(cfg: dict[str, Any]) -> tuple[list[str], dict[str, dict]]:
    """Scan Binance Square, return (symbol_list, per_symbol_attention_meta).

    Attention meta keys:
        - locales:         list[str]
        - sections:        list[str]
        - en_cn_overlap:   bool
        - section_overlap: bool
        - deep_mentions:   int
    """
    u = cfg.get("universe", {})
    locales   = u.get("locales", ["en", "zh-CN"])
    sections  = u.get("sections", ["most_searched", "rapid_riser"])
    deep      = bool(u.get("hashtag_deep_fetch", True))
    cap       = int(u.get("max_candidates", 15))

    hotrank_fn, trending_fn = _try_atomic_lib_source()
    if hotrank_fn is None:
        log.warning("atomic_strategy_lib.data.search unavailable; "
                    "returning empty universe (template will emit empty selection)")
        return [], {}

    attention: dict[str, dict] = defaultdict(
        lambda: {"locales": set(), "sections": set(), "deep_mentions": 0}
    )

    for locale in locales:
        for section in sections:
            try:
                rows = hotrank_fn(locale=locale, section=section) or []
            except Exception:  # noqa: BLE001
                continue
            for r in rows:
                sym = _normalize(r.get("symbol") or r.get("token"))
                if not sym:
                    continue
                attention[sym]["locales"].add(locale)
                attention[sym]["sections"].add(section)

    if deep:
        for locale in locales:
            try:
                tags = trending_fn(locale=locale) or []
            except Exception:  # noqa: BLE001
                continue
            for t in tags:
                sym = _normalize(t.get("token") or t.get("symbol"))
                if sym:
                    attention[sym]["deep_mentions"] += 1

    def rank_key(item):
        _, a = item
        return (-len(a["locales"]), -len(a["sections"]), -a["deep_mentions"])
    ranked = sorted(attention.items(), key=rank_key)[:cap]

    symbols = [s for s, _ in ranked]
    meta = {
        sym: {
            "locales":         sorted(a["locales"]),
            "sections":        sorted(a["sections"]),
            "en_cn_overlap":   ("en" in a["locales"]) and
                               any(l.startswith("zh") for l in a["locales"]),
            "section_overlap": len(a["sections"]) >= 2,
            "deep_mentions":   a["deep_mentions"],
        }
        for sym, a in ranked
    }
    return symbols, meta


def _normalize(raw) -> str:
    if not raw:
        return ""
    s = str(raw).upper().replace("-", "").replace("/", "")
    if s.endswith(("USDT", "BUSD", "USDC")):
        return s
    return f"{s}USDT"
