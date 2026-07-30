"""Vendored PUBLIC Binance Square (BigData) client — pure stdlib, prod gateway.

This is a *deliberately narrow* copy of the upstream research client. It ships
**only** the seven PUBLIC Binance Square endpoints that are reachable through
the public web gateway (``https://www.binance.com/bapi/bigdata``):

    getNews / getSentiment / getTickerRank / getTopicTrending
    getHotPost / getFeed / getSearch

Everything internal has been intentionally left behind:

  * ``TradingInsightClient`` (QA-only Signal篇 endpoints) — NOT vendored.
  * Every internal-network host (indicators / offline charts / futures
    market data / search-recall rankings) — NOT vendored. Those hosts are
    named in the private deployment config, never in this public repo.
  * ``.local`` DNS-fallback / ``subprocess`` host resolution — NOT vendored
    (only ever needed for the internal hosts above).

Only the Python standard library is imported, so vendoring this adds **no new
runtime dependency** to ``cyqnt_trd``.

Provenance
----------
Synced verbatim (behaviour-preserving) from the upstream research client. If
you set the environment variable ``CYQNT_BIGDATA_API_PATH`` to a directory
containing the upstream ``client.py``, :func:`get_public_client` will import
``BigDataClient`` from there instead of using this vendored copy — useful when
you want to track upstream without re-vendoring. The upstream client exposes a
superset of methods; only the PUBLIC ones listed above are ever called by
``cyqnt_trd``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

__all__ = ["PublicBigDataClient", "get_public_client", "is_ok_envelope", "__provenance__"]

# ---------------------------------------------------------------------------
# Provenance metadata (audit trail for the vendored subset)
# ---------------------------------------------------------------------------
__provenance__ = {
    "upstream_repo": "crypto-alpha-research/binance_bigdata_api",
    "upstream_file": "client.py",
    "upstream_symbol": "BigDataClient (PUBLIC Square methods only)",
    "upstream_lines": "41-266 (class), 233-266 (public square methods)",
    "vendored_on": "2026-07-23",
    "vendored_by": "cyqnt_trd data_cli (task t2)",
    "excluded": [
        "TradingInsightClient",
        "indicators / offline_charts / futures_market_*",
        "ranking_spot / ranking_um / ranking_cm (internal search-recall)",
        "get_ai_signal / get_strategy_ranking / *PriceChangeRank / getTradeSignalTopRank",
        "all internal-network hosts + .local DNS fallback",
    ],
    "public_only": True,
    "stdlib_only": True,
}

# ---- Gateway endpoints ----------------------------------------------------
GW_PROD = "https://www.binance.com/bapi/bigdata"   # public, has real data

#: QA gateway. The host used to be hardcoded here and was labelled a "public QA
#: gateway" — it is not: it resolves to an RFC1918 address and is only reachable
#: from the corporate network, so publishing it in this public repository leaked
#: an internal hostname. Supply it via ``CYQNT_BIGDATA_QA_GATEWAY`` where the QA
#: environment is actually reachable; ``env="qa"`` raises without it rather than
#: silently falling back to prod (which would send QA traffic at real data).
_QA_GATEWAY_ENV = "CYQNT_BIGDATA_QA_GATEWAY"
SQUARE_PATH = "/v1/public/bigdata/square/skill"


def qa_gateway() -> str:
    """QA gateway base URL from the environment, or raise explaining why."""
    value = os.environ.get(_QA_GATEWAY_ENV, "").strip()
    if not value:
        raise RuntimeError(
            "env='qa' needs %s to be set (e.g. https://<qa-host>/bapi/bigdata). "
            "The QA host is internal and is deliberately not hardcoded in this "
            "public repository." % _QA_GATEWAY_ENV
        )
    return value.rstrip("/")

#: Provenance header sent with every request so the server / any proxy can
#: attribute traffic to the vendored client.
_PROVENANCE_HEADER = "cyqnt_trd/data_cli._vendor.binance_bigdata_client (public-square-only)"


def is_ok_envelope(resp: dict) -> bool:
    """Return True iff *resp* is a successful envelope with a non-null ``data``.

    Mirrors upstream ``BigDataClient._ok``: ``code == '000000'`` (all zeros)
    AND ``data is not None``. A ``code == '000000'`` with ``data is None`` is a
    legitimate *cache miss*, not an error.
    """
    if not isinstance(resp, dict):
        return False
    return str(resp.get("code", "")).strip("0") == "" and resp.get("data") is not None


class PublicBigDataClient:
    """Minimal PUBLIC Binance Square client (stdlib ``urllib`` only)."""

    def __init__(self, env: str = "prod", timeout: int = 20, min_interval: float = 0.6):
        """
        Parameters
        ----------
        env : str
            ``"prod"`` (default, real data) or ``"qa"`` (connectivity check;
            Square cache is usually empty).
        timeout : int
            Per-request socket timeout in seconds.
        min_interval : float
            Minimum spacing (seconds) between requests, to avoid the prod
            gateway returning empty bodies under rate limiting.
        """
        if env not in ("prod", "qa"):
            raise ValueError(f"env must be 'prod' or 'qa', got {env!r}")
        self.env = env
        self.timeout = timeout
        self.min_interval = float(min_interval)
        self._last_call = 0.0
        self.gateway = GW_PROD if env == "prod" else qa_gateway()

    # ---- transport --------------------------------------------------------
    def _post(self, url: str, payload: dict, gray: bool = False) -> dict:
        # Throttle to avoid rate-limit-induced empty responses.
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": _PROVENANCE_HEADER,
            "X-Cyqnt-Provenance": _PROVENANCE_HEADER,
        }
        # The QA gateway's skill interface requires a gray header.
        if gray and self.env == "qa":
            headers["x-gray-env"] = "skill"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                self._last_call = time.monotonic()
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            self._last_call = time.monotonic()
            body = e.read().decode("utf-8", "ignore")
            try:
                return json.loads(body)
            except Exception:
                return {"_http_error": e.code, "_body": body}
        except Exception as e:  # noqa: BLE001 — network layer, surface as envelope
            self._last_call = time.monotonic()
            return {"_error": str(e)}

    def _square(self, command: str, payload: dict) -> dict:
        return self._post(f"{self.gateway}{SQUARE_PATH}/{command}", payload, gray=True)

    # ---- PUBLIC Square methods -------------------------------------------
    def get_news(self, lang: str = "en", page_index: int = 1, page_size: int = 20) -> dict:
        """Binance News official feed (``lang`` required; cold langs return empty)."""
        return self._square("getNews", {"lang": lang, "pageIndex": page_index, "pageSize": page_size})

    def get_hot_post(self, sort: str = "HEAT", window: str = "24h", limit: int = 10,
                     lang: str = "en") -> dict:
        """Global hot posts. ``sort`` = HEAT/TIME/ENGAGEMENT; ``window`` = 1h/4h/24h/3d/7d."""
        return self._square("getHotPost", {"sort": sort, "window": window, "limit": limit, "lang": lang})

    def get_ticker_rank(self, window: str = "24h", limit: int = 20, lang: str = "en") -> dict:
        """Ticker mention ranking (mentions / unique authors / bull-bear split)."""
        return self._square("getTickerRank", {"window": window, "limit": limit, "lang": lang})

    def get_topic_trending(self, window: str = "24h", limit: int = 10, lang: str = "en") -> dict:
        """Trending topics (hashtag / mention count / bull-bear split)."""
        return self._square("getTopicTrending", {"window": window, "limit": limit, "lang": lang})

    def get_feed(self, token: str = "BTC", window: str = "24h", filter_: str = "quality",
                 page_index: int = 1, page_size: int = 10, lang: str = "en") -> dict:
        """Token/topic content feed. NOTE: returns empty on Prod for most tokens."""
        return self._square("getFeed", {"token": token, "window": window, "filter": filter_,
                                         "pageIndex": page_index, "pageSize": page_size, "lang": lang})

    def get_search(self, keyword: str = "", author: str = "", min_likes: int = 0,
                   window: str = "24h", page_index: int = 1, page_size: int = 10,
                   lang: str = "en") -> dict:
        """Keyword search (keyword/author/minLikes — at least one required)."""
        p = {"window": window, "pageIndex": page_index, "pageSize": page_size, "lang": lang}
        if keyword:
            p["keyword"] = keyword
        if author:
            p["author"] = author
        if min_likes:
            p["minLikes"] = min_likes
        return self._square("getSearch", p)

    def get_sentiment(self, token: str = "BTC") -> dict:
        """Token community sentiment poll (bullish / bearish / total)."""
        return self._square("getSentiment", {"token": token})


def get_public_client(env: str = "prod", timeout: int = 20,
                      min_interval: float = 0.6):
    """Return a PUBLIC BigData client instance.

    If ``CYQNT_BIGDATA_API_PATH`` is set and points to a directory (or file)
    containing the upstream ``client.py`` with a ``BigDataClient`` class, that
    upstream client is imported and instantiated instead of the vendored copy.
    This lets callers track upstream without re-vendoring. On any import
    failure we fall back to the vendored :class:`PublicBigDataClient`.
    """
    override = os.environ.get("CYQNT_BIGDATA_API_PATH", "").strip()
    if override:
        client = _try_load_upstream(override, env=env, timeout=timeout, min_interval=min_interval)
        if client is not None:
            return client
    return PublicBigDataClient(env=env, timeout=timeout, min_interval=min_interval)


def _try_load_upstream(path: str, *, env: str, timeout: int, min_interval: float):
    """Best-effort import of the upstream ``BigDataClient`` from *path*.

    Returns an instance, or ``None`` if anything goes wrong (missing file,
    import error, missing class). Callers fall back to the vendored client.
    """
    import importlib.util

    candidate = path
    if os.path.isdir(path):
        candidate = os.path.join(path, "client.py")
    if not os.path.isfile(candidate):
        return None
    try:
        spec = importlib.util.spec_from_file_location("_cyqnt_upstream_bigdata", candidate)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        upstream_cls = getattr(module, "BigDataClient", None)
        if upstream_cls is None:
            return None
        return upstream_cls(env=env, timeout=timeout, min_interval=min_interval)
    except Exception:
        return None
