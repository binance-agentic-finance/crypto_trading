"""Module ① UNIVERSE — discover trending tokens from Binance Square.

Two sources:
    - Square `hotrank_fetch`  — Most Searched, Rapid Riser
    - Square `trending_fetch` — trending hashtags (optional deep fetch)

Dedupe + rank by:
    (1) EN/CN locale overlap (a token showing in both languages is stronger)
    (2) section overlap (Most Searched ∩ Rapid Riser is stronger)
    (3) hashtag deep-fetch mentions

Each candidate carries a lightweight `attention` dict that m03_signals
uses to score attention tiers.
"""
from __future__ import annotations

from collections import defaultdict

from atomic_strategy_lib.data.search import hotrank_fetch, trending_fetch


def discover_universe(cfg: dict) -> list[dict]:
    """Return a ranked list of candidate dicts, each with `symbol` and
    `attention` metadata.

    Config: `cfg.universe.{locales, sections, hashtag_deep_fetch, max_candidates}`
    """
    uni_cfg = cfg.get("universe", {})
    locales = uni_cfg.get("locales", ["en", "zh-CN"])
    sections = uni_cfg.get("sections", ["most_searched", "rapid_riser"])
    deep    = bool(uni_cfg.get("hashtag_deep_fetch", True))
    cap     = int(uni_cfg.get("max_candidates", 15))

    # attention[symbol] = {"locales": set(), "sections": set(), "deep_mentions": int}
    attention: dict[str, dict] = defaultdict(
        lambda: {"locales": set(), "sections": set(), "deep_mentions": 0}
    )

    # ── (a) hotrank across locales × sections ──
    for locale in locales:
        for section in sections:
            try:
                rows = hotrank_fetch(locale=locale, section=section) or []
            except Exception:
                continue
            for r in rows:
                sym = _normalize_symbol(r.get("symbol") or r.get("token"))
                if not sym:
                    continue
                attention[sym]["locales"].add(locale)
                attention[sym]["sections"].add(section)

    # ── (b) trending hashtags ──
    if deep:
        for locale in locales:
            try:
                tags = trending_fetch(locale=locale) or []
            except Exception:
                continue
            for t in tags:
                sym = _normalize_symbol(t.get("token") or t.get("symbol"))
                if sym:
                    attention[sym]["deep_mentions"] += 1

    # ── rank + slice ──
    def _rank_key(item):
        _, a = item
        return (
            -len(a["locales"]),
            -len(a["sections"]),
            -a["deep_mentions"],
        )
    ranked = sorted(attention.items(), key=_rank_key)[:cap]

    return [
        {
            "symbol":    sym,
            "attention": {
                "locales":       sorted(a["locales"]),
                "sections":      sorted(a["sections"]),
                "en_cn_overlap": ("en" in a["locales"]) and any(l.startswith("zh") for l in a["locales"]),
                "section_overlap": len(a["sections"]) >= 2,
                "deep_mentions":   a["deep_mentions"],
            },
        }
        for sym, a in ranked
    ]


def _normalize_symbol(raw) -> str:
    """`BTC` / `btcusdt` / `BTC-USDT` → `BTCUSDT`."""
    if not raw:
        return ""
    s = str(raw).upper().replace("-", "").replace("/", "")
    if s.endswith(("USDT", "BUSD", "USDC")):
        return s
    return f"{s}USDT"
