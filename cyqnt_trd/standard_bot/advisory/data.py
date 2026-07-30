"""Thin adapters over current /bn-data-related interfaces.

These adapters fetch and normalize only. Bot logic never performs HTTP calls.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional, Protocol, Sequence

import pandas as pd

from ...data_cli import news as square_news
from .contracts import utc_datetime


@dataclass(frozen=True)
class FetchFrame:
    frame: pd.DataFrame
    status: str
    source: str
    warning: str = ""


class CatalogResolver(Protocol):
    def resolve(self, field_key: str, symbol: str, **kwargs: Any) -> Any:
        ...


class BnDataCatalogAdapter:
    """Wrap ``datasource.UnifiedDataSource`` (or a compatible injected resolver)."""

    def __init__(self, resolver: CatalogResolver):
        self.resolver = resolver

    def fetch(self, field_key: str, symbol: str, **kwargs: Any) -> FetchFrame:
        try:
            result = self.resolver.resolve(field_key, symbol, **kwargs)
        except Exception as exc:  # resolver owns transport details; preserve failure semantics
            return FetchFrame(
                frame=pd.DataFrame(),
                status="error",
                source="bn-data:%s" % field_key,
                warning="%s: %s" % (type(exc).__name__, exc),
            )
        raw = result.data
        if raw is None or (isinstance(raw, dict) and raw.get("_forward_only")):
            return FetchFrame(
                frame=pd.DataFrame(),
                status="degraded",
                source="bn-data:%s" % field_key,
                warning="forward-only placeholder or empty payload; collector data required",
            )
        if isinstance(raw, pd.DataFrame):
            frame = raw.copy()
        elif isinstance(raw, list):
            frame = pd.DataFrame(raw)
        elif isinstance(raw, dict):
            frame = pd.DataFrame([raw])
        else:
            frame = pd.DataFrame([{"value": raw}])
        frame["source_field"] = field_key
        frame["source_symbol"] = symbol
        frame["resolved_via"] = result.resolved_via
        frame["validation_status"] = result.validation_status
        frame["pit_safe"] = bool(result.pit)
        status = "ok" if not frame.empty else "degraded"
        return FetchFrame(frame=frame, status=status, source="bn-data:%s" % field_key)


class SquarePublicAdapter:
    """Current callable PUBLIC Square typed wrappers."""

    def fetch_news_events(
        self,
        *,
        captured_at: Optional[Any] = None,
        lang: str = "en",
        page_size: int = 50,
        venue: str = "binance",
        product: str = "mixed",
        refresh: bool = False,
    ) -> FetchFrame:
        raw = square_news.fetch_news(lang=lang, page_size=page_size, refresh=refresh)
        # Caller-provided captured_at must be the real response/collector completion time.
        captured = utc_datetime(captured_at or datetime.now(timezone.utc))
        return _posts_to_news_frame(raw, captured, "square.getNews", venue, product)

    def fetch_search_events(
        self,
        keyword: str,
        *,
        captured_at: Optional[Any] = None,
        window: str = "24h",
        page_size: int = 20,
        lang: str = "en",
        venue: str = "binance",
        product: str = "mixed",
        refresh: bool = False,
    ) -> FetchFrame:
        raw = square_news.fetch_search(
            keyword=keyword,
            window=window,
            page_size=page_size,
            lang=lang,
            refresh=refresh,
        )
        captured = utc_datetime(captured_at or datetime.now(timezone.utc))
        return _posts_to_news_frame(raw, captured, "square.getSearch", venue, product)

    def fetch_social_metrics(
        self,
        symbols: Sequence[str],
        *,
        captured_at: Optional[Any] = None,
        window: str = "24h",
        limit: int = 20,
        lang: str = "en",
        venue: str = "binance",
        product: str = "mixed",
        refresh: bool = False,
    ) -> FetchFrame:
        rank = square_news.fetch_ticker_rank(
            window=window,
            limit=limit,
            lang=lang,
            refresh=refresh,
        )
        rows: List[dict] = []
        warnings: List[str] = []
        if rank.empty:
            warnings.append("ticker_rank empty")
        wanted = {_base_token(symbol): symbol.upper() for symbol in symbols}
        for _, item in rank.iterrows():
            token = str(item.get("ticker", "")).upper()
            if token not in wanted:
                continue
            symbol = wanted[token]
            for metric in (
                "rank",
                "mention_count",
                "unique_authors",
                "total_engagement",
                "bullish_count",
                "bearish_count",
            ):
                rows.append(
                    _metric_row(
                        symbol,
                        metric,
                        item.get(metric),
                        _ms_or_now(item.get("generated_at")),
                        window,
                        "square.getTickerRank",
                    )
                )

        for token, symbol in wanted.items():
            sentiment = square_news.fetch_sentiment(token=token, refresh=refresh)
            if sentiment.empty:
                warnings.append("sentiment empty for %s" % token)
                continue
            item = sentiment.iloc[-1]
            rows.append(
                _metric_row(
                    symbol,
                    "social_bull_ratio",
                    item.get("bull_ratio"),
                    _ms_or_now(item.get("generated_at")),
                    window,
                    "square.getSentiment",
                )
            )
        captured = utc_datetime(captured_at or datetime.now(timezone.utc))
        for row in rows:
            row["available_time"] = captured
            row["venue"] = venue
            row["product"] = product
        frame = pd.DataFrame(rows, columns=_MARKET_METRIC_COLUMNS)
        if frame.empty:
            return FetchFrame(
                frame,
                "degraded",
                "square.social",
                "no social rows; do not replace missing values with zero",
            )
        return FetchFrame(
            frame,
            "degraded" if warnings else "ok",
            "square.social",
            "; ".join(warnings),
        )


class BinanceMarketMetricAdapter:
    """Normalize ``data_cli`` market/derivatives reads into ``MarketMetricFrame@1.0``.

    This is the quantitative sibling of :class:`SquarePublicAdapter`: it fetches
    only, never decides. Every metric is emitted as its own row so a missing
    source drops one row instead of silently becoming a zero — a bot that never
    sees ``funding_rate_8h`` degrades, it does not read "funding is flat".

    ``product`` is carried through verbatim: funding / OI / long-short / basis
    are perpetual-only and are skipped (with a warning) when the caller asks for
    ``spot``, rather than being borrowed from the perp leg.
    """

    #: metrics that only exist on the USD-M perpetual leg
    PERP_ONLY_METRICS = (
        "funding_rate_8h",
        "funding_rate_annualized_pct",
        "open_interest_change_24h_pct",
        "long_short_account_ratio",
        "basis_rate_pct",
    )

    def __init__(
        self,
        *,
        venue: str = "binance",
        product: str = "usd_m_perpetual",
        market: str = "futures",
    ) -> None:
        self.venue = venue
        self.product = product
        self.market = market

    def fetch_market_metrics(
        self,
        symbols: Sequence[str],
        *,
        captured_at: Optional[Any] = None,
        oi_period: str = "1h",
        oi_lookback_bars: int = 24,
        volume_baseline_days: int = 7,
        ls_period: str = "1h",
        refresh: bool = False,
    ) -> FetchFrame:
        captured = utc_datetime(captured_at or datetime.now(timezone.utc))
        rows: List[dict] = []
        warnings: List[str] = []
        wanted = [str(symbol).upper() for symbol in symbols]
        if not wanted:
            raise ValueError("fetch_market_metrics requires at least one symbol")
        is_perp = self.product != "spot"

        for symbol in wanted:
            emitted = len(rows)
            self._collect(rows, warnings, symbol, captured, "ticker", self._ticker_metrics, refresh)
            self._collect(
                rows, warnings, symbol, captured, "volume_ratio",
                lambda sym, cap, ref: self._volume_ratio_metrics(sym, cap, volume_baseline_days, ref),
                refresh,
            )
            if is_perp:
                self._collect(
                    rows, warnings, symbol, captured, "open_interest",
                    lambda sym, cap, ref: self._oi_metrics(sym, cap, oi_period, oi_lookback_bars, ref),
                    refresh,
                )
                self._collect(rows, warnings, symbol, captured, "funding", self._funding_metrics, refresh)
                self._collect(
                    rows, warnings, symbol, captured, "long_short",
                    lambda sym, cap, ref: self._long_short_metrics(sym, cap, ls_period, ref),
                    refresh,
                )
                self._collect(rows, warnings, symbol, captured, "basis", self._basis_metrics, refresh)
            if len(rows) == emitted:
                warnings.append("%s: no metric could be read" % symbol)

        if is_perp is False:
            warnings.append(
                "product=spot: %s not fetched (perpetual-only)" % ", ".join(self.PERP_ONLY_METRICS)
            )

        frame = pd.DataFrame(rows, columns=_MARKET_METRIC_COLUMNS)
        if frame.empty:
            return FetchFrame(
                frame,
                "error",
                "binance.market_metrics",
                "; ".join(warnings) or "no rows returned",
            )
        return FetchFrame(
            frame,
            "degraded" if warnings else "ok",
            "binance.market_metrics",
            "; ".join(warnings),
        )

    # ---- per-source readers; each returns a list of metric rows ----

    def _collect(self, rows, warnings, symbol, captured, label, reader, refresh) -> None:
        try:
            produced = reader(symbol, captured, refresh)
        except Exception as exc:  # transport/CLI failures degrade, never zero-fill
            warnings.append("%s/%s: %s: %s" % (symbol, label, type(exc).__name__, exc))
            return
        if not produced:
            warnings.append("%s/%s: empty" % (symbol, label))
            return
        rows.extend(produced)

    def _row(self, symbol, metric, value, event_time, captured, unit, window, source):
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(numeric):
            return None
        event = utc_datetime(event_time)
        quality = "ok"
        if event > captured:  # clock skew must not violate event_time <= available_time
            event = captured
            quality = "event_time_clamped_to_capture"
        return {
            "event_time": event,
            "available_time": captured,
            "venue": self.venue,
            "product": self.product,
            "symbol": symbol,
            "metric": metric,
            "value": numeric,
            "unit": unit,
            "window": window,
            "source_id": source,
            "quality": quality,
        }

    def _ticker_metrics(self, symbol, captured, refresh):
        from ...data_cli import ticker as ticker_mod

        frame = ticker_mod.fetch_24h_ticker(symbol, market=self.market, refresh=refresh)
        if frame is None or frame.empty:
            return []
        item = frame.iloc[-1]
        row = self._row(
            symbol, "price_change_24h_pct", item.get("change_pct"),
            captured, captured, "percent", "24h", "binance.ticker24hr",
        )
        return [row] if row else []

    def _volume_ratio_metrics(self, symbol, captured, baseline_days, refresh):
        from ...data_cli import kline as kline_mod

        if baseline_days < 1:
            raise ValueError("volume_baseline_days must be >= 1")
        frame = kline_mod.fetch_klines(
            symbol, "1d", limit=baseline_days + 1, market=self.market, refresh=refresh
        )
        if frame is None or len(frame) < 2:
            return []
        volumes = pd.to_numeric(frame["quote_volume"], errors="coerce").dropna()
        if len(volumes) < 2:
            return []
        baseline = float(volumes.iloc[:-1].mean())
        if baseline <= 0:
            return []
        row = self._row(
            symbol, "volume_ratio_24h", float(volumes.iloc[-1]) / baseline,
            _ms_or_capture(frame["close_time"].iloc[-1], captured), captured,
            "ratio", "24h_vs_%sd_mean" % baseline_days, "binance.klines",
        )
        return [row] if row else []

    def _oi_metrics(self, symbol, captured, period, lookback_bars, refresh):
        from ...data_cli import oi as oi_mod

        frame = oi_mod.fetch_oi_history(
            symbol, period=period, limit=lookback_bars + 1, refresh=refresh
        )
        if frame is None or len(frame) < 2:
            return []
        values = pd.to_numeric(frame["oi_value"], errors="coerce").dropna()
        if len(values) < 2 or float(values.iloc[0]) <= 0:
            return []
        change_pct = (float(values.iloc[-1]) - float(values.iloc[0])) / float(values.iloc[0]) * 100.0
        row = self._row(
            symbol, "open_interest_change_24h_pct", change_pct,
            _ms_or_capture(frame["timestamp"].iloc[-1], captured), captured,
            "percent", "%s x %s" % (period, lookback_bars), "binance.open_interest_statistics",
        )
        return [row] if row else []

    def _funding_metrics(self, symbol, captured, refresh):
        from ...data_cli import funding as funding_mod

        frame = funding_mod.fetch_funding_history(symbol, limit=1, refresh=refresh)
        if frame is None or frame.empty:
            return []
        item = frame.iloc[-1]
        rate = item.get("rate")
        event_time = _ms_or_capture(item.get("timestamp"), captured)
        produced = [
            self._row(
                symbol, "funding_rate_8h", rate, event_time, captured,
                "ratio", "8h", "binance.funding_rate_history",
            ),
            self._row(
                symbol, "funding_rate_annualized_pct",
                None if rate is None else float(rate) * 3 * 365 * 100.0,
                event_time, captured, "percent", "annualized", "binance.funding_rate_history",
            ),
        ]
        return [row for row in produced if row]

    def _long_short_metrics(self, symbol, captured, period, refresh):
        from ...data_cli import ratios as ratios_mod

        frame = ratios_mod.fetch_long_short_ratio(symbol, period=period, limit=1, refresh=refresh)
        if frame is None or frame.empty:
            return []
        item = frame.iloc[-1]
        row = self._row(
            symbol, "long_short_account_ratio", item.get("long_short_ratio"),
            _ms_or_capture(item.get("timestamp"), captured), captured,
            "ratio", period, "binance.long_short_ratio",
        )
        return [row] if row else []

    def _basis_metrics(self, symbol, captured, refresh):
        from ...data_cli import ratios as ratios_mod

        frame = ratios_mod.fetch_basis(symbol, period="1h", limit=1, refresh=refresh)
        if frame is None or frame.empty:
            return []
        item = frame.iloc[-1]
        rate = item.get("basis_rate")
        row = self._row(
            symbol, "basis_rate_pct", None if rate is None else float(rate) * 100.0,
            _ms_or_capture(item.get("timestamp"), captured), captured,
            "percent", "1h", "binance.basis",
        )
        return [row] if row else []


def _posts_to_news_frame(
    raw: pd.DataFrame,
    captured: datetime,
    source: str,
    venue: str,
    product: str,
) -> FetchFrame:
    if raw.empty:
        return FetchFrame(
            frame=_empty_news_frame(),
            status="degraded",
            source=source,
            warning="empty response: cache miss and transport error are not yet distinguishable",
        )
    rows = []
    for _, item in raw.iterrows():
        tickers = list(item.get("tickers") or []) + list(item.get("user_input_tickers") or [])
        tickers = list(dict.fromkeys(tickers)) or [None]
        author_role = str(item.get("author_role", ""))
        has_url = bool(item.get("web_link"))
        official_role = author_role.lower() in ("official", "project", "exchange", "binance")
        if source == "square.getNews" and official_role and has_url:
            reliability = 0.85
        elif source == "square.getNews" and has_url:
            reliability = 0.6
        else:
            reliability = 0.35
        flags = []
        if not has_url:
            flags.append("missing_source_url")
        if reliability < 0.7:
            flags.append("unverified_source")
        if bool(item.get("is_created_by_ai")):
            flags.append("ai_generated")
        body = str(item.get("body", "") or item.get("summary", ""))
        for ticker in tickers:
            rows.append(
                {
                    "event_id": str(item["id"]),
                    "published_at": _ms_or_capture(item.get("date"), captured),
                    "available_time": captured,
                    "source_id": source,
                    "topic": "unclassified",
                    "signal_family": "news_sentiment",
                    "urgency": "background",
                    "event_type": "news",
                    "symbol": _to_symbol(ticker),
                    "venue": venue,
                    "product": product,
                    "title": str(item.get("title", "")),
                    "summary": str(item.get("summary", "") or item.get("body", "")),
                    "author_name": str(item.get("author_name", "")),
                    "author_role": author_role,
                    "body_hash": hashlib.sha256(body.encode("utf-8")).hexdigest() if body else "",
                    "url": str(item.get("web_link", "")),
                    "bias": "neutral",
                    "raw_tendency": item.get("tendency"),
                    "source_reliability": reliability,
                    "corroboration_count": 0,
                    "quality_flags": flags,
                }
            )
    return FetchFrame(pd.DataFrame(rows), "ok", source)


_MARKET_METRIC_COLUMNS = [
    "event_time",
    "available_time",
    "venue",
    "product",
    "symbol",
    "metric",
    "value",
    "unit",
    "window",
    "source_id",
    "quality",
]


def _metric_row(symbol, metric, value, event_time, window, source):
    return {
        "event_time": event_time,
        "available_time": event_time,
        "venue": "binance",
        "product": "mixed",
        "symbol": symbol,
        "metric": metric,
        "value": value,
        "unit": "ratio" if "ratio" in metric else "count",
        "window": window,
        "source_id": source,
        "quality": "forward_only_snapshot",
    }


def _empty_news_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id",
            "published_at",
            "available_time",
            "source_id",
            "topic",
            "signal_family",
            "urgency",
            "event_type",
            "symbol",
            "venue",
            "product",
            "title",
            "summary",
            "author_name",
            "author_role",
            "body_hash",
            "url",
            "bias",
            "raw_tendency",
            "source_reliability",
            "corroboration_count",
            "quality_flags",
        ]
    )


def _ms_or_capture(value: Any, captured: datetime) -> datetime:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return captured
    return captured if number <= 0 else utc_datetime(number)


def _to_symbol(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    token = str(value).upper().strip().lstrip("$")
    token = token.replace("/", "").replace("-", "").replace("_", "").replace(" ", "")
    quotes = ("USDT", "USDC", "FDUSD", "TUSD", "BUSD", "USD", "BTC", "ETH", "BNB")
    if any(token.endswith(quote) and len(token) > len(quote) for quote in quotes):
        return token
    return token + "USDT" if token else None


def _base_token(symbol: str) -> str:
    value = str(symbol).upper()
    for quote in ("USDT", "USDC", "FDUSD", "TUSD", "USD"):
        if value.endswith(quote) and len(value) > len(quote):
            return value[: -len(quote)]
    return value


def _ms_or_now(value: Any) -> datetime:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)
    return datetime.now(timezone.utc) if number <= 0 else utc_datetime(number)
