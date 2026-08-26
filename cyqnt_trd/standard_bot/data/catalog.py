"""Data-node catalog — the ground truth for ``type: data`` spec nodes.

``cyqnt_trd/blocks/BLOCKS_API.md`` tells a strategy author (or a codegen LLM)
what every *compute* node does. Nothing played that role for the *data* nodes:
the strategy Spec puts ``data`` and ``action`` in a separate ``runtime``
package, and a DAG that says ``type: data, function: klines`` had no schema to
validate against and no statement of whether the field is even replayable.

This module is that missing half. One :class:`DataNodeSpec` per ``data.<fn>``
carrying:

* ``params`` / ``returns`` schema, in the same shape ``BLOCKS_API.md`` uses, so
  codegen and the Canvas form can be driven off one registry;
* **where the bytes come from** (:class:`SourcePath`) — indicators API, bdp
  screening clause, public Binance substitute, Square skill API, local parquet;
* **whether you may backtest on it** (:class:`Availability`) — this is the part
  that keeps getting lost. Most internal snapshot APIs (Square social,
  futuresRadar, movement) are Redis-TTL snapshots with *no point-in-time
  history*: you can only collect them forward. Writing a walk-forward backtest
  against them silently reuses today's value at every bar.
* ``pit_hazard`` — the specific way this field lies if you replay it naively.

Nothing here fetches at import time. ``fetch()`` routes to the adapter that
already exists (``cyqnt_trd.data_cli`` / ``standard_bot.data``); unreachable
sources raise :class:`DataUnavailable` with the reason rather than returning an
empty frame that reads as "nothing happened".

Usage::

    from cyqnt_trd.standard_bot.data.catalog import get_node, list_nodes

    spec = get_node("funding")
    spec.availability            # Availability.BACKTESTABLE
    spec.fetch(symbol="BTCUSDT", limit=500)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ..core.input_contract import FrameKind

__all__ = [
    "Availability",
    "SourcePath",
    "DataUnavailable",
    "ParamSpec",
    "ReturnSpec",
    "DataNodeSpec",
    "get_node",
    "list_nodes",
    "list_node_names",
    "nodes_by_availability",
    "backtestable_node_names",
    "register_node",
    "unregister_node",
    "is_custom_node",
    "FrameKind",
    "typed_node_names",
]


class Availability(str, Enum):
    """How far back you can honestly replay this field."""

    #: real history, deep enough for walk-forward backtests
    BACKTESTABLE = "BACKTESTABLE"
    #: history exists but is windowed/shallow (e.g. ~30d) or T+1 lagged
    SEMI = "SEMI"
    #: snapshot only (Redis TTL, live poll). No replayable history — collect
    #: forward, mark the strategy RESEARCH_ONLY, never walk-forward on it.
    FORWARD_ONLY = "FORWARD_ONLY"
    #: contract is specified but the feed is not onboarded yet (外采 / 待补)
    EXTERNAL_PENDING = "EXTERNAL_PENDING"
    #: namespace reserved, no endpoint exists anywhere
    NOT_WIRED = "NOT_WIRED"


class SourcePath(str, Enum):
    """Which of the access mechanisms serves this node."""

    #: indicators service: klines + 14 ta4j indicators. Base URL from
    #: ``$INDICATORS_URL``; the address is not published here.
    INDICATORS_API = "indicators_api"
    #: cross-sectional screener field, expressed as a source:"bdp" clause.
    #: Base URL from ``$BDP_SCREENING_URL``.
    BDP_SCREENING = "bdp_screening"
    #: api.binance.com / fapi.binance.com / data-api.binance.vision / alternative.me
    PUBLIC_BINANCE = "public_binance"
    #: /v1/public/bigdata/square/skill/{cmd} — Prod IP-whitelisted
    SQUARE_SKILL = "square_skill"
    #: internal-network HTTP endpoint. The route and host live in the private
    #: deployment config (see ``internal_slots.py``), never in this public repo.
    INTERNAL_HTTP = "internal_http"
    #: parquet already downloaded under a --*-dir root
    LOCAL_PARQUET = "local_parquet"
    #: third-party vendor feed
    EXTERNAL_VENDOR = "external_vendor"


class DataUnavailable(RuntimeError):
    """Raised when a node cannot be served from the current environment.

    Carries *why* so the caller can record it in ``SnapshotMeta.source_status``
    instead of turning a fetch failure into a zero.
    """

    def __init__(self, node: str, reason: str) -> None:
        super().__init__("data.%s unavailable: %s" % (node, reason))
        self.node = node
        self.reason = reason


@dataclass(frozen=True)
class ParamSpec:
    key: str
    type: str
    required: bool = False
    default: Any = None
    description: str = ""
    #: when set, the value must reference an upstream node output
    source: Optional[str] = None
    options: Tuple[Any, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "key": self.key,
            "type": self.type,
            "required": self.required,
            "description": self.description,
        }
        if self.default is not None:
            out["default"] = self.default
        if self.source:
            out["source"] = self.source
        if self.options:
            out["options"] = list(self.options)
        return out


@dataclass(frozen=True)
class ReturnSpec:
    type: str
    description: str
    #: column names for DataFrame returns, element names for tuple returns
    columns: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"type": self.type, "description": self.description}
        if self.columns:
            out["columns"] = list(self.columns)
        return out


@dataclass(frozen=True)
class DataNodeSpec:
    """One ``type: data`` node — schema + provenance + replay discipline."""

    name: str
    description: str
    strategy_types: Tuple[str, ...]
    source_path: SourcePath
    availability: Availability
    returns: ReturnSpec
    params: Tuple[ParamSpec, ...] = ()
    #: how this field lies if replayed naively; empty when it replays honestly
    pit_hazard: str = ""
    #: concrete endpoint / table / CLI the bytes come from
    endpoint: str = ""
    #: what to use instead when the primary source is unreachable
    substitute: str = ""
    #: dotted path of the fetcher, resolved lazily by :meth:`fetch`
    fetcher: Optional[str] = None
    notes: str = ""

    # ---- input contract (cyqnt.input/v1) ------------------------------------
    #: canonical shape this node emits after normalisation. RAW means the shape
    #: has not been declared yet: the frame is still reachable, it just carries
    #: no guarantees and no typed accessor.
    emits: "FrameKind" = None  # type: ignore[assignment]
    #: vendor column name -> canonical column name
    column_map: Dict[str, str] = field(default_factory=dict)
    #: columns added with a fixed value (timeframe / venue / product / metric)
    constants: Dict[str, Any] = field(default_factory=dict)
    #: for METRIC nodes: wide columns melted into ``metric`` / ``value`` rows
    value_columns: Tuple[str, ...] = ()
    #: declared param name -> the keyword the wired fetcher actually takes.
    #:
    #: The catalog is the *public* contract: it is what ``DATA_API.md``, the
    #: Canvas form and codegen read, so its names are chosen for the caller
    #: (``market_type`` says what it is; the fetcher's ``market`` does not). The
    #: fetchers are existing library functions used elsewhere in the repo and
    #: renaming their kwargs would be a breaking change for those callers.
    #:
    #: Without a translation the two just disagreed, and because
    #: :func:`runtime.data._make_node_function` fills declared defaults in, the
    #: mismatch was not even conditional: ``data.klines(symbol=..., interval=...)``
    #: injected ``market_type="futures"`` and raised ``TypeError`` every single
    #: time. The most-used node in the catalog could not be called at all.
    param_aliases: Dict[str, str] = field(default_factory=dict)
    #: name of a ``data_cli.rest_source`` spec serving the same field over public
    #: HTTPS, used when the primary fetcher fails.
    #:
    #: The derivatives statistics reach us through a local ``binance-cli``
    #: subprocess, and when that returns something other than JSON the node is
    #: simply dead — which is how open interest ended up being read from a
    #: month-old parquet. The same numbers are on a public endpoint, and the
    #: declared field names already match, so falling back costs nothing and the
    #: substitution is recorded on the frame rather than hidden.
    public_fallback: str = ""

    def __post_init__(self) -> None:
        if self.emits is None:
            object.__setattr__(self, "emits", FrameKind.RAW)

    @property
    def module(self) -> str:
        return "data"

    @property
    def input_schema(self):
        """The canonical :class:`FrameSchema` this node emits, if declared."""
        from ..core.input_contract import schema_for

        return schema_for(self.emits)

    @property
    def typed(self) -> bool:
        return self.emits is not FrameKind.RAW

    #: call-param -> canonical column. Several endpoints take the instrument as
    #: a *parameter* and never echo it in the response (open_interest,
    #: long_short_ratio, taker_volume, basis, klines). The instrument is known —
    #: it is what we asked for — so it is filled in from the request rather than
    #: left missing and failing validation.
    PARAM_CONSTANTS = {
        "symbol": "instrument_id",
        "pair": "instrument_id",
        "token": "instrument_id",
        "instrument_id": "instrument_id",
        "interval": "timeframe",
        "period": "timeframe",
    }

    def normalize(self, frame: Any, *, available_time: Optional[int] = None,
                  params: Optional[Dict[str, Any]] = None, validate: bool = True):
        """Vendor frame -> canonical shape. Returns ``(frame, warnings, inferred)``."""
        from ..core.input_contract import normalize_frame

        constants = dict(self.constants)
        for key, column in self.PARAM_CONSTANTS.items():
            value = (params or {}).get(key)
            # a declared-but-None constant is a placeholder saying "fill me from
            # the request"; an explicit constant on the node always wins.
            if value is not None and constants.get(column) is None:
                constants[column] = str(value).upper() if column == "instrument_id" else value
        constants = {k: v for k, v in constants.items() if v is not None}

        return normalize_frame(
            frame,
            kind=self.emits,
            node=self.name,
            column_map=self.column_map,
            constants=constants,
            value_columns=self.value_columns,
            available_time=available_time,
            validate=validate,
        )

    @property
    def backtestable(self) -> bool:
        return self.availability is Availability.BACKTESTABLE

    def resolve_fetcher(self) -> Callable[..., Any]:
        if not self.fetcher:
            raise DataUnavailable(
                self.name,
                "no fetcher wired (%s / %s)"
                % (self.availability.value, self.source_path.value),
            )
        module_path, _, attr = self.fetcher.rpartition(".")
        try:
            import importlib

            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise DataUnavailable(self.name, "cannot import %s: %s" % (module_path, exc))
        try:
            return getattr(module, attr)
        except AttributeError:
            raise DataUnavailable(
                self.name, "%s has no attribute %r" % (module_path, attr)
            )

    def bind_params(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Declared param names -> the keywords the fetcher takes.

        Separated from :meth:`fetch` so the guard test can check the binding
        without performing a network call.
        """
        return {self.param_aliases.get(key, key): value for key, value in params.items()}

    def fetch(self, **kwargs: Any) -> Any:
        """Call the wired fetcher, translating transport failures into
        :class:`DataUnavailable` so a caller never mistakes a broken source for
        an empty result."""
        fn = self.resolve_fetcher()
        try:
            return fn(**self.bind_params(kwargs))
        except DataUnavailable:
            raise
        except Exception as exc:
            frame = self._try_public_fallback(kwargs, primary_error=exc)
            if frame is not None:
                return frame
            raise DataUnavailable(self.name, "%s: %s" % (type(exc).__name__, exc))

    def _try_public_fallback(self, params: Mapping[str, Any], *, primary_error: Exception):
        """Same field over public HTTPS. Returns ``None`` if unavailable.

        The substitution is stamped into ``frame.attrs`` so a bundle can report
        which source actually answered — a silent swap would make two runs
        incomparable for no visible reason.
        """
        if not self.public_fallback:
            return None
        try:
            from ...data_cli import public_sources, rest_source
        except ImportError:
            return None
        try:
            public_sources.register_public_sources()
            spec = rest_source.get_spec(self.public_fallback)
            if spec is None:
                return None
            frame = rest_source.fetch_rest(spec, params=dict(params))
        except Exception:
            return None
        if frame is None or getattr(frame, "empty", True):
            return None
        try:
            frame.attrs["source_fallback"] = (
                "primary fetcher failed (%s: %s); served from public %s instead"
                % (type(primary_error).__name__, str(primary_error)[:120],
                   self.public_fallback)
            )
        except Exception:
            pass
        return frame

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "module": self.module,
            "description": self.description,
            "strategy_types": list(self.strategy_types),
            "source_path": self.source_path.value,
            "availability": self.availability.value,
            "endpoint": self.endpoint,
            "substitute": self.substitute,
            "pit_hazard": self.pit_hazard,
            "emits": self.emits.value,
            "input_schema": self.input_schema.name if self.input_schema else None,
            "returns": self.returns.to_dict(),
            "params": [item.to_dict() for item in self.params],
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# shared params
# ---------------------------------------------------------------------------

_SYMBOL = ParamSpec("symbol", "str", required=True, description="e.g. BTCUSDT")
_INTERVAL = ParamSpec(
    "interval", "str", default="1h",
    description="bar interval",
    options=("1m", "5m", "15m", "30m", "1h", "4h", "1d"),
)
_LIMIT = ParamSpec("limit", "int", default=500, description="number of rows (most recent last)")
_MARKET_TYPE = ParamSpec(
    "market_type", "str", default="futures",
    description="spot | futures (USDⓈ-M perpetual)",
    options=("spot", "futures"),
)
_WINDOW = ParamSpec(
    "window", "str", default="24h",
    description="aggregation window",
    options=("1h", "4h", "24h", "3d", "7d"),
)
_TOKEN = ParamSpec("token", "str", required=True, description="base token, e.g. BTC (no quote)")


# ---------------------------------------------------------------------------
# the catalog
# ---------------------------------------------------------------------------

_NODES: List[DataNodeSpec] = [
    # ---- C: kline / technical -------------------------------------------
    DataNodeSpec(
        name="klines",
        emits=FrameKind.BAR,
        # No column_map. The klines response is a list of bare ARRAYS — no field
        # names at all, and in particular no ``symbol``: the instrument is in the
        # request's query string, not in the reply, because the caller already
        # knows what it asked for. (Contrast the ``universe`` node below, whose
        # ticker24hr response does carry ``symbol`` per row and legitimately
        # renames it.)
        #
        # A ``column_map={"symbol": "instrument_id"}`` was copied here from that
        # node. It could never fire, so it changed nothing about the data —
        # ``instrument_id`` has always come from ``constants`` below. What it did
        # do is make input_contract's "declared source column absent from the
        # response" check true on EVERY capture, putting a permanent,
        # never-actionable line into the bundle's warnings and from there into
        # every emitted signal's ``warnings``. That field is also where declared
        # assumptions ride out, and a warning that is always present is how a
        # reader learns to skip the whole field.
        constants={"instrument_id": None, "timeframe": None},
        param_aliases={"market_type": "market"},
        description="OHLCV candlesticks — the base series every technical block reads.",
        strategy_types=("C", "P", "L", "O", "A", "R"),
        source_path=SourcePath.INDICATORS_API,
        availability=Availability.BACKTESTABLE,
        endpoint="indicators service (base URL from $INDICATORS_URL); "
                 "public: data-api.binance.vision, api.binance.com/api/v3/klines, fapi/v1/klines",
        substitute="cyqnt_trd.data_cli.fetch_klines (public Binance)",
        returns=ReturnSpec(
            "pd.DataFrame",
            "one row per bar, oldest first",
            ("open_time", "open", "high", "low", "close", "volume",
             "quote_volume", "close_time", "trades"),
        ),
        params=(_SYMBOL, _INTERVAL, _LIMIT, _MARKET_TYPE),
        fetcher="cyqnt_trd.data_cli.kline.fetch_klines",
    ),
    DataNodeSpec(
        name="klines_multi_tf",
        emits=FrameKind.BAR,
        constants={"instrument_id": None, "timeframe": None},
        param_aliases={"intervals": "timeframes", "market_type": "market"},
        description="OHLCV for several intervals at once (multi-timeframe strategies).",
        strategy_types=("C", "R"),
        source_path=SourcePath.INDICATORS_API,
        availability=Availability.BACKTESTABLE,
        endpoint="same as klines, intervals=[...]",
        returns=ReturnSpec(
            "pd.DataFrame", "one frame, rows tagged by timeframe",
            ("open_time", "open", "high", "low", "close", "volume", "quote_volume", "close_time", "trades"),
        ),
        params=(
            _SYMBOL,
            ParamSpec("intervals", "list[str]", required=True, description='e.g. ["1h", "4h"]'),
            _LIMIT,
            _MARKET_TYPE,
        ),
        fetcher="cyqnt_trd.data_cli.kline.fetch_klines_multi_tf",
        notes="A DAG can equally use N separate `klines` nodes; alignment happens "
              "inside the conditions blocks, not in the data layer.",
    ),
    DataNodeSpec(
        name="ticker_24h",
        emits=FrameKind.METRIC,
        column_map={"symbol": "instrument_id"},
        constants={"window": "24h", "source_id": "binance.ticker24hr"},
        param_aliases={"market_type": "market"},
        value_columns=("price", "change_pct", "high_24h", "low_24h", "volume_base", "volume_quote", "trades", "weighted_avg_price",),
        description="24h rolling price/volume statistics for one symbol.",
        strategy_types=("C", "S", "D"),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.FORWARD_ONLY,
        endpoint="ticker24hr-price-change-statistics",
        returns=ReturnSpec(
            "pd.DataFrame",
            "single row",
            ("symbol", "price", "change_pct", "high_24h", "low_24h",
             "volume_base", "volume_quote", "trades", "weighted_avg_price"),
        ),
        params=(_SYMBOL, _MARKET_TYPE),
        pit_hazard="rolling snapshot with no history endpoint — replay it and every "
                   "bar sees today's 24h change. Derive from klines for backtests.",
        fetcher="cyqnt_trd.data_cli.ticker.fetch_24h_ticker",
    ),

    # ---- D: derivatives / crowding ---------------------------------------
    DataNodeSpec(
        name="funding",
        emits=FrameKind.METRIC,
        column_map={"symbol": "instrument_id", "timestamp": "event_time"},
        constants={"unit": "ratio", "window": "8h", "source_id": "binance.funding_rate"},
        value_columns=("rate", "mark_price",),
        description="Perpetual funding rate history — the cost of the crowded side.",
        strategy_types=("D", "A", "N"),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.BACKTESTABLE,
        endpoint="fapi.binance.com/fapi/v1/fundingRate; internal syncFundingRate / "
                 "futuresRadar/fundingRateWidget",
        returns=ReturnSpec(
            "pd.DataFrame", "one row per 8h settlement, oldest first",
            ("symbol", "rate", "timestamp", "mark_price"),
        ),
        params=(_SYMBOL, ParamSpec("limit", "int", default=500, description="settlements")),
        fetcher="cyqnt_trd.data_cli.funding.fetch_funding_history",
        notes="Years of history — this is what the funding-carry and crowding "
              "books were actually validated on.",
    ),
    DataNodeSpec(
        name="funding_snapshot",
        emits=FrameKind.METRIC,
        column_map={
            "symbol": "instrument_id",
            "lastFundingRate": "funding_rate",
        },
        constants={
            "unit": "ratio",
            "source_id": "binance.premiumIndex",
        },
        value_columns=("funding_rate",),
        description=(
            "Current funding-rate snapshot across every USDM perpetual; this is "
            "the cross-section a funding-based SELECTION strategy joins."
        ),
        strategy_types=("D", "L", "P"),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.FORWARD_ONLY,
        endpoint="fapi.binance.com/fapi/v1/premiumIndex (all symbols)",
        returns=ReturnSpec(
            "pd.DataFrame",
            "one current row per perpetual symbol",
            ("symbol", "lastFundingRate", "markPrice", "indexPrice", "time"),
        ),
        params=(),
        pit_hazard=(
            "Current all-market snapshot only; capture it forward. It must not be "
            "replayed as historical funding at earlier decision times."
        ),
        fetcher="cyqnt_trd.blocks.data.fetch_premium_index",
        notes=(
            "Transport-only node. Live YAML selection stores it under the logical "
            "bundle key `funding`; the existing `funding` node remains the "
            "single-symbol settlement history used by trade strategies."
        ),
    ),
    DataNodeSpec(
        name="funding_current",
        emits=FrameKind.METRIC,
        constants={"metric": "funding_rate_8h", "unit": "ratio", "source_id": "binance.premiumIndex"},
        description="Latest funding rate + next settlement time.",
        strategy_types=("D", "A"),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.FORWARD_ONLY,
        endpoint="fapi/v1/premiumIndex (mark-price)",
        returns=ReturnSpec("float", "current funding rate, e.g. 0.0001 = 0.01%"),
        params=(_SYMBOL,),
        pit_hazard="point read; use `funding` for anything replayed.",
        fetcher="cyqnt_trd.data_cli.funding.fetch_funding_rate",
    ),
    DataNodeSpec(
        name="open_interest",
        public_fallback="open_interest",
        emits=FrameKind.METRIC,
        column_map={"timestamp": "event_time"},
        constants={"unit": "notional", "source_id": "binance.open_interest_statistics"},
        value_columns=("oi_base", "oi_value", "oi_change_bps",),
        description="Open-interest history (base + notional) for leverage build-up.",
        strategy_types=("D", "F"),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.SEMI,
        endpoint="fapi futures/data/openInterestHist; internal syncLSAndOI",
        returns=ReturnSpec(
            "pd.DataFrame", "oldest first",
            ("timestamp", "oi_base", "oi_value", "oi_change_bps"),
        ),
        params=(
            _SYMBOL,
            ParamSpec("period", "str", default="1h", description="5m..1d"),
            ParamSpec("limit", "int", default=48),
        ),
        pit_hazard="public endpoint keeps only ~30 days. A longer backtest silently "
                   "starts mid-series — check the first timestamp, do not assume.",
        substitute="local parquet open_interest_<period>.parquet via --derivatives-dir",
        fetcher="cyqnt_trd.data_cli.oi.fetch_oi_history",
    ),
    DataNodeSpec(
        name="long_short_ratio",
        public_fallback="ls_account",
        emits=FrameKind.METRIC,
        column_map={"timestamp": "event_time"},
        constants={"unit": "ratio", "source_id": "binance.long_short_ratio"},
        value_columns=("long_account", "short_account", "long_short_ratio",),
        description="Global long/short ACCOUNT ratio (retail positioning skew).",
        strategy_types=("D",),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.SEMI,
        endpoint="fapi futures/data/globalLongShortAccountRatio",
        returns=ReturnSpec(
            "pd.DataFrame", "oldest first",
            ("timestamp", "long_account", "short_account", "long_short_ratio"),
        ),
        params=(_SYMBOL, ParamSpec("period", "str", default="1h"), ParamSpec("limit", "int", default=30)),
        pit_hazard="~30d only.",
        fetcher="cyqnt_trd.data_cli.ratios.fetch_long_short_ratio",
        notes="NOT part of enrich_market_frame_with_derivatives — it does not appear "
              "as a df column on the trade route; read it as its own data node.",
    ),
    DataNodeSpec(
        name="top_trader_ratio",
        public_fallback="ls_top_position",
        emits=FrameKind.METRIC,
        column_map={"timestamp": "event_time"},
        constants={"unit": "ratio", "source_id": "binance.top_trader_ratio"},
        value_columns=("long_account", "short_account", "long_short_ratio",),
        description="Top-trader long/short POSITION ratio (smart-money skew).",
        strategy_types=("D", "F"),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.SEMI,
        endpoint="fapi futures/data/topLongShortPositionRatio",
        returns=ReturnSpec("pd.DataFrame", "oldest first",
                           ("timestamp", "long_account", "short_account", "long_short_ratio")),
        params=(_SYMBOL, ParamSpec("period", "str", default="1h"), ParamSpec("limit", "int", default=30)),
        pit_hazard="~30d only.",
        fetcher="cyqnt_trd.data_cli.ratios.fetch_top_trader_ls_ratio",
    ),
    DataNodeSpec(
        name="taker_volume",
        public_fallback="taker_ratio",
        emits=FrameKind.METRIC,
        column_map={"timestamp": "event_time"},
        constants={"source_id": "binance.taker_ratio"},
        value_columns=("buy_vol", "sell_vol", "buy_sell_ratio",),
        description="Taker buy/sell volume ratio (aggressive order flow).",
        strategy_types=("D", "O"),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.SEMI,
        endpoint="fapi futures/data/takerlongshortRatio",
        returns=ReturnSpec("pd.DataFrame", "oldest first",
                           ("timestamp", "buy_vol", "sell_vol", "buy_sell_ratio")),
        params=(_SYMBOL, ParamSpec("period", "str", default="1h"), ParamSpec("limit", "int", default=30)),
        pit_hazard="~30d only.",
        fetcher="cyqnt_trd.data_cli.ratios.fetch_taker_volume",
        notes="Also available cross-sectionally as the bdp `taker_buy_pct` clause.",
    ),
    DataNodeSpec(
        name="basis",
        emits=FrameKind.METRIC,
        column_map={"timestamp": "event_time"},
        constants={"source_id": "binance.basis"},
        value_columns=("index_price", "contract_price", "basis", "basis_rate",),
        description="Perp premium/discount over the index (carry,期现偏离).",
        strategy_types=("A", "D"),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.SEMI,
        endpoint="fapi futures/data/basis; premiumIndexKlines for the long form",
        returns=ReturnSpec("pd.DataFrame", "oldest first",
                           ("timestamp", "index_price", "contract_price", "basis", "basis_rate")),
        params=(
            ParamSpec("pair", "str", required=True, description="e.g. BTCUSDT"),
            ParamSpec("period", "str", default="1h"),
            ParamSpec("limit", "int", default=30),
            ParamSpec("contract_type", "str", default="PERPETUAL"),
        ),
        pit_hazard="premiumIndex history is only ~400 days locally; ≥3yr backfill is an open item.",
        fetcher="cyqnt_trd.data_cli.ratios.fetch_basis",
    ),
    DataNodeSpec(
        name="liquidations",
        emits=FrameKind.METRIC,
        column_map={"close_time": "event_time"},
        constants={"source_id": "binance.forceOrder"},
        value_columns=("long_liquidation_usd", "short_liquidation_usd",),
        description="Aggregated forced-liquidation flow (cascade / capitulation).",
        strategy_types=("D", "F"),
        source_path=SourcePath.LOCAL_PARQUET,
        availability=Availability.SEMI,
        endpoint="Binance WS !forceOrder@arr, recorded by "
                 "HistoricalBinanceLiquidationRecorder into parquet",
        returns=ReturnSpec("pd.DataFrame", "bucketed by close_time",
                           ("close_time", "long_liquidation_usd", "short_liquidation_usd")),
        params=(_SYMBOL, _INTERVAL,
                ParamSpec("liquidations_dir", "str", required=True,
                          description="root passed as --liquidations-dir"),
                # This node ENRICHES a bar frame rather than fetching on its own;
                # the fetcher's first positional argument is that frame. It was
                # missing from the declaration, so the node looked standalone and
                # could never actually be called.
                ParamSpec("frame", "pd.DataFrame", required=True,
                          source="upstream.output",
                          description="bar frame to attach liquidation columns to "
                                      "(e.g. the klines node output)"),
                _MARKET_TYPE),
        param_aliases={"symbol": "instrument_id", "interval": "timeframe",
                       "liquidations_dir": "liquidations_root"},
        pit_hazard="the WS stream is NOT archived upstream — you only have what this "
                   "repo recorded. Coverage starts at the first capture, not at listing.",
        fetcher="cyqnt_trd.standard_bot.data.liquidations.enrich_market_frame_with_liquidations",
    ),

    # ---- O: orderbook / microstructure ------------------------------------
    DataNodeSpec(
        name="orderbook_depth",
        emits=FrameKind.BOOK,
        # The endpoint returns price/qty/side; ``level`` is implicit in the row
        # order and derived during normalisation.
        column_map={"symbol": "instrument_id", "qty": "quantity"},
        param_aliases={"market_type": "market"},
        description="L2 depth snapshot and derived imbalance.",
        strategy_types=("O",),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.FORWARD_ONLY,
        endpoint="fapi/v1/depth (live snapshot only)",
        returns=ReturnSpec(
            "pd.DataFrame", "one row per price level, best-first within each side",
            ("side", "price", "qty"),
        ),
        params=(_SYMBOL, ParamSpec("limit", "int", default=100), _MARKET_TYPE),
        pit_hazard="NO archive exists for top-20 L2 anywhere. The only replayable "
                   "proxy is the 30s bookDepth band parquet (BTC/ETH/SOL). Any O-type "
                   "backtest is on a proxy, and no live WS pipeline exists yet.",
        fetcher="cyqnt_trd.data_cli.orderbook.fetch_orderbook_depth",
    ),

    # ---- E: news / event ---------------------------------------------------
    DataNodeSpec(
        name="news",
        emits=FrameKind.EVENT,
        column_map={"id": "event_id", "date": "event_time", "web_link": "url"},
        constants={"source_id": "square.getNews", "topic": "unclassified"},
        description="Binance Square news + announcements (listings, delistings, maintenance).",
        strategy_types=("E",),
        source_path=SourcePath.SQUARE_SKILL,
        availability=Availability.FORWARD_ONLY,
        endpoint="square/skill/getNews{lang,pageIndex,pageSize}",
        returns=ReturnSpec(
            "pd.DataFrame", "one row per post",
            ("id", "title", "summary", "body", "web_link", "date", "content_type",
             "tendency", "author_name", "author_role", "tickers", "generated_at"),
        ),
        params=(
            ParamSpec("lang", "str", default="en"),
            ParamSpec("page_size", "int", default=50),
        ),
        pit_hazard="Prod IP-whitelisted; off-whitelist returns success:true with an "
                   "EMPTY data block, which is indistinguishable from 'no news' unless "
                   "you check source_status. No bulk history — collect forward.",
        fetcher="cyqnt_trd.data_cli.news.fetch_news",
    ),
    DataNodeSpec(
        name="news_search",
        emits=FrameKind.EVENT,
        column_map={"id": "event_id", "date": "event_time", "web_link": "url"},
        constants={"source_id": "square.getSearch", "topic": "community_lead"},
        description="Square keyword/author search — community leads, lower reliability.",
        strategy_types=("E", "S"),
        source_path=SourcePath.SQUARE_SKILL,
        availability=Availability.FORWARD_ONLY,
        endpoint="square/skill/getSearch{keyword,author,minLikes,window}",
        returns=ReturnSpec("pd.DataFrame", "same shape as news"),
        params=(
            ParamSpec("keyword", "str", description="at least one of keyword/author/min_likes"),
            ParamSpec("author", "str"),
            ParamSpec("min_likes", "int", default=0),
            _WINDOW,
        ),
        pit_hazard="community content, not an official source. Treat as a lead: no "
                   "official link and no second independent source -> investigate only.",
        fetcher="cyqnt_trd.data_cli.news.fetch_search",
    ),
    DataNodeSpec(
        name="hot_post",
        emits=FrameKind.EVENT,
        column_map={"id": "event_id", "date": "event_time", "web_link": "url"},
        constants={"source_id": "square.getHotPost", "topic": "social"},
        description="Square hot posts by heat / time / engagement.",
        strategy_types=("S", "E"),
        source_path=SourcePath.SQUARE_SKILL,
        availability=Availability.FORWARD_ONLY,
        endpoint="square/skill/getHotPost{sort,window,limit}",
        returns=ReturnSpec("pd.DataFrame", "post list + generated_at"),
        params=(ParamSpec("sort", "str", default="HEAT", options=("HEAT", "TIME", "ENGAGEMENT")),
                _WINDOW, ParamSpec("limit", "int", default=20)),
        pit_hazard="60s cache snapshot, no history.",
        fetcher="cyqnt_trd.data_cli.news.fetch_hot_post",
    ),
    DataNodeSpec(
        name="fear_greed",
        emits=FrameKind.METRIC,
        column_map={"date": "event_time"},
        constants={"instrument_id": "MARKET", "metric": "fear_greed", "source_id": "alternative.me/fng", "unit": "index_0_100"},
        value_columns=("value",),
        description="Crypto Fear & Greed index (0-100) — top-level risk regime.",
        strategy_types=("R", "E"),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.BACKTESTABLE,
        endpoint="internal market/fearAndGreedHistory; public alternative.me/fng/?limit=0",
        returns=ReturnSpec("pd.DataFrame", "daily, oldest first",
                           ("date", "value", "value_classification")),
        params=(ParamSpec("limit", "int", default=0, description="0 = full history"),),
        fetcher="cyqnt_trd.data_cli.regime.fetch_fear_greed",
        notes="Deep daily history from 2023-06-28 — genuinely backtestable.",
    ),
    DataNodeSpec(
        name="ahr999",
        emits=FrameKind.METRIC,
        # The fetcher's own column names are ahr999 / geomean_200d / close; the
        # canonical metric names below are what strategies and DATA_API.md use.
        # Declaring only the canonical names left none of them present, so the
        # melt fell through to "every numeric column" and produced metrics called
        # ``ahr999`` / ``geomean_200d`` that no reader was looking for.
        column_map={"date": "event_time", "ahr999": "ahr999_value",
                    "geomean_200d": "average_price", "close": "current_price"},
        constants={"instrument_id": "BTCUSDT", "source_id": "ahr999"},
        value_columns=("ahr999_value", "average_price", "current_price", "model_price",),
        description="AHR999 BTC valuation index — accumulation-zone regime.",
        strategy_types=("R",),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.BACKTESTABLE,
        endpoint="internal ahr999/getAhr999List; public: reconstruct from BTC close",
        returns=ReturnSpec("pd.DataFrame", "daily",
                           ("timestamp", "date", "close", "ahr999",
                            "geomean_200d", "model_price")),
        # The series is daily by construction, so "interval" was never a knob —
        # it was declared, defaulted to "1d", auto-injected, and rejected by the
        # fetcher on every call. What the fetcher does take is a row count.
        params=(ParamSpec("limit", "int", default=1000,
                          description="number of daily points (most recent last)"),),
        fetcher="cyqnt_trd.data_cli.regime.fetch_ahr999",
    ),
    DataNodeSpec(
        name="etf_flow",
        emits=FrameKind.METRIC,
        column_map={"token": "instrument_id", "date": "event_time"},
        constants={"source_id": "market.ETFHistory"},
        value_columns=("flow", "net_assets", "close_price",),
        description="Spot-ETF daily net flow and AUM by issuer.",
        strategy_types=("R", "E"),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.SEMI,
        endpoint="market/ETFHistory{token}, market/ETFLatest",
        fetcher="cyqnt_trd.data_cli.internal_frames.fetch_etf_flow",
        returns=ReturnSpec("pd.DataFrame", "daily", ("token", "date", "flow", "net_assets", "close_price")),
        params=(_TOKEN,),
        pit_hazard="T+1 publication lag — using the same-day value is lookahead. "
                   "close_price is frequently 0/null.",
        notes="Per-issuer detail via market/ETFTickerDetail; ETFLatest for the current snapshot.",
    ),
    DataNodeSpec(
        name="hot_event",
        emits=FrameKind.EVENT,
        column_map={"publish_time": "event_time", "category": "topic"},
        constants={"source_id": "event.getHotEventForSignal"},
        description="Curated hot events with rank, sentiment and related coins.",
        strategy_types=("E",),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.SEMI,
        endpoint="event/getHotEventForSignal{category}, event/getHotEventDetail",
        fetcher="cyqnt_trd.data_cli.internal_frames.fetch_hot_event",
        returns=ReturnSpec("pd.DataFrame", "one row per event",
                           ("id", "publish_time_ts", "news_rank", "sentiment", "category",
                            "summary", "related_coins", "backtest_list")),
        params=(ParamSpec("category", "str"),),
        pit_hazard="news_rank and sentiment can be revised AFTER publication — only "
                   "publish_time_ts is trustworthy at signal time.",
    ),
    DataNodeSpec(
        name="event_upcoming",
        emits=FrameKind.EVENT,
        column_map={"card_coin": "instrument_id", "signal_time": "event_time", "signal_type": "topic", "card_context": "summary"},
        constants={"source_id": "ti.event.getEventUpcoming", "event_id": None},
        description="Forward-looking event cards (listings, upgrades, unlocks).",
        strategy_types=("E",),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.SEMI,
        endpoint="ti/event/getEventUpcoming",
        fetcher="cyqnt_trd.data_cli.internal_frames.fetch_event_upcoming",
        returns=ReturnSpec("pd.DataFrame", "one row per card",
                           ("card_coin", "signal_type", "signal_time", "signal_expired_time",
                            "is_bullish", "price_change_1", "price_change_2", "price_change_3")),
        pit_hazard="price_change_1/2/3 are BACKFILLED after the event resolves. Using "
                   "them as features is direct label leakage — never read them at signal time.",
    ),
    DataNodeSpec(
        name="macro_calendar",
        emits=FrameKind.EVENT,
        column_map={"event_type": "topic"},
        constants={"source_id": "calendar.macroDetail", "event_id": None},
        description="Scheduled macro releases (CPI / NFP / FOMC / rate decisions).",
        strategy_types=("R", "E"),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.SEMI,
        endpoint="calendar/macroDetail",
        fetcher="cyqnt_trd.data_cli.internal_frames.fetch_macro_calendar",
        returns=ReturnSpec("pd.DataFrame", "one row per release",
                           ("event_time", "event_type", "actual", "forecast", "previous")),
        pit_hazard="PIT-safe on the schedule (release times are known in advance), but "
                   "`actual` only exists after the release — gate on event_time.",
    ),
    DataNodeSpec(
        name="token_unlock",
        emits=FrameKind.EVENT,
        column_map={"token": "instrument_id", "next_unlock_time": "event_time"},
        constants={"source_id": "calendar.unlockDetail", "topic": "token_unlock", "event_id": None},
        description="Token unlock schedule and history.",
        strategy_types=("E", "F"),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.SEMI,
        endpoint="calendar/unlockDetail, calendar/dailyCalendarForWidget",
        fetcher="cyqnt_trd.data_cli.internal_frames.fetch_token_unlock",
        returns=ReturnSpec("pd.DataFrame", "one row per unlock",
                           ("token", "unlock_time", "total_unlocked", "next_unlock", "price")),
        pit_hazard="forward schedule only, no bulk history — snapshot forward.",
        notes="Feeds the catalyst_risk_monitor unlock rule.",
    ),

    # ---- S: social / attention --------------------------------------------
    DataNodeSpec(
        name="ticker_rank",
        emits=FrameKind.RANK,
        column_map={"ticker": "instrument_id", "generated_at": "event_time"},
        constants={"source_id": "square.getTickerRank"},
        description="Square per-ticker mention/engagement/sentiment aggregates.",
        strategy_types=("S",),
        source_path=SourcePath.SQUARE_SKILL,
        availability=Availability.FORWARD_ONLY,
        endpoint="square/skill/getTickerRank{window,limit,lang} (Prod only)",
        returns=ReturnSpec(
            "pd.DataFrame", "one row per ticker",
            ("ticker", "mention_count", "unique_authors", "total_engagement",
             "bullish_count", "bearish_count", "neutral_count", "rank", "generated_at"),
        ),
        params=(_WINDOW, ParamSpec("limit", "int", default=20), ParamSpec("lang", "str", default="en")),
        pit_hazard="`ticker` is CASE-SENSITIVE — do not upper() it before matching. "
                   "60s cache, no history: forward-collect or the backtest is fiction.",
        fetcher="cyqnt_trd.data_cli.news.fetch_ticker_rank",
    ),
    DataNodeSpec(
        name="sentiment",
        emits=FrameKind.RANK,
        constants={"source_id": "square.getSentiment"},
        description="Square bull/bear poll for one token.",
        strategy_types=("S",),
        source_path=SourcePath.SQUARE_SKILL,
        availability=Availability.FORWARD_ONLY,
        endpoint="square/skill/getSentiment{token}",
        returns=ReturnSpec("pd.DataFrame", "single row",
                           ("bullish_value", "bearish_value", "total_value", "bull_ratio")),
        params=(_TOKEN,),
        pit_hazard="a single live poll, latest snapshot only.",
        fetcher="cyqnt_trd.data_cli.news.fetch_sentiment",
    ),
    DataNodeSpec(
        name="topic_trending",
        emits=FrameKind.RANK,
        column_map={"hashtag": "instrument_id"},
        constants={"source_id": "square.getTopicTrending"},
        description="Square trending hashtags/topics with windowed engagement.",
        strategy_types=("S",),
        source_path=SourcePath.SQUARE_SKILL,
        availability=Availability.FORWARD_ONLY,
        endpoint="square/skill/getTopicTrending{window,limit,lang}",
        returns=ReturnSpec("pd.DataFrame", "one row per topic",
                           ("id", "hashtag", "window_mention_count", "window_unique_authors",
                            "window_total_engagement", "window_bullish_count")),
        params=(_WINDOW, ParamSpec("limit", "int", default=20)),
        pit_hazard="snapshot, no history.",
        fetcher="cyqnt_trd.data_cli.news.fetch_topic_trending",
    ),

    # ---- L / P: cross-section ---------------------------------------------
    DataNodeSpec(
        name="universe",
        emits=FrameKind.RANK,
        column_map={"symbol": "instrument_id"},
        description="24h ticker table across the whole tradable universe — the input "
                    "every SELECTION strategy ranks over.",
        strategy_types=("L", "P", "S", "D"),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.FORWARD_ONLY,
        endpoint="ticker24hr (all symbols)",
        returns=ReturnSpec("pd.DataFrame", "one row per symbol",
                           ("symbol", "price", "change_pct", "volume_quote")),
        params=(_MARKET_TYPE,),
        pit_hazard="rolling 24h snapshot. For a replayed cross-section, rebuild the "
                   "table from klines as of the decision bar.",
        fetcher="cyqnt_trd.blocks.universe.fetch_perpetual_universe",
    ),
    DataNodeSpec(
        name="contract_meta",
        emits=FrameKind.RANK,
        column_map={"symbol": "instrument_id"},
        description=(
            "What each listed contract IS: contract type, underlying type "
            "(COIN / EQUITY / COMMODITY / INDEX …), the venue's sector tags, and "
            "the base/quote legs. This is the cross-section a SELECTION strategy "
            "joins to say \"crypto only\", \"no TradFi\", \"AI sector\"."
        ),
        strategy_types=("L", "P", "S"),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.FORWARD_ONLY,
        endpoint="fapi.binance.com/fapi/v1/exchangeInfo (all symbols, unfiltered)",
        returns=ReturnSpec(
            "pd.DataFrame",
            "one row per listed contract, including non-TRADING ones",
            ("symbol", "contractType", "underlyingType", "underlyingSubType",
             "baseAsset", "quoteAsset", "status"),
        ),
        params=(_MARKET_TYPE,),
        pit_hazard=(
            "Listing REGISTRY, current only. A symbol's row appears when it is "
            "listed and disappears when it is delisted, and the venue re-tags "
            "sectors without notice — so replaying today's table at an earlier "
            "decision time both invents metadata for coins that did not exist "
            "yet and hides the tags a delisted one had. Capture it forward "
            "alongside the universe it will be joined to."
        ),
        fetcher="cyqnt_trd.blocks.data.fetch_contract_meta",
        notes=(
            "Transport-only node; it does no filtering, so `status` and "
            "`quoteAsset` arrive as columns for a spec to filter on. "
            "`underlyingSubType` is a LIST on the wire — "
            "universe.augment_with_contract_meta flattens it to a "
            "comma-separated scalar, because a list-valued cell raises "
            "'truth value of an array ... is ambiguous' inside the candidate "
            "builder. Do not read the raw column directly."
        ),
    ),
    DataNodeSpec(
        name="open_interest_snapshot",
        emits=FrameKind.RANK,
        column_map={
            "symbol": "instrument_id",
            "openInterest": "oi_base",
            "markPrice": "mark_price",
            "time": "event_time",
        },
        constants={"source_id": "binance.openInterest"},
        description=(
            "Current open interest for a NAMED ROSTER of perpetuals, in coins "
            "plus the mark price that converts it to dollars. This is the "
            "cross-section a SELECTION strategy joins to say \"position "
            "inventory above $5m\"."
        ),
        strategy_types=("D", "L", "P"),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.FORWARD_ONLY,
        endpoint="fapi.binance.com/fapi/v1/openInterest — ONE REQUEST PER SYMBOL "
                 "(omitting symbol answers HTTP 400) + one premiumIndex call for "
                 "the mark prices",
        returns=ReturnSpec(
            "pd.DataFrame",
            "one current row per requested symbol",
            ("symbol", "openInterest", "markPrice", "time"),
        ),
        params=(
            ParamSpec("symbols", "list[str]", required=True,
                      description="the roster to fan out over; capped at "
                                  "blocks.data.FAN_OUT_MAX_SYMBOLS and never "
                                  "truncated"),
            _MARKET_TYPE,
        ),
        pit_hazard=(
            "Point read of a live inventory, all-market snapshot only: the "
            "endpoint has no history at all, and the mark price that converts it "
            "to dollars is today's. Replaying it at an earlier decision time "
            "states this week's positioning as last month's. Capture it forward "
            "alongside the universe it will be joined to; for a replayable "
            "series use `open_interest` (~30d) instead."
        ),
        fetcher="cyqnt_trd.blocks.data.fetch_open_interest_cross_section",
        notes=(
            "FAN-OUT node: cost is one request per instrument, so the roster must "
            "be narrowed by the free cross-sectional filters FIRST — see the "
            "ordering table in standard_bot/data/live_snapshot.py. Distinct from "
            "the per-symbol `open_interest` node, which serves ~30 days of "
            "history for ONE instrument and cannot stand in for a cross-section."
        ),
    ),
    DataNodeSpec(
        name="oi_change_snapshot",
        emits=FrameKind.METRIC,
        column_map={
            "symbol": "instrument_id",
            "timestamp": "event_time",
            "sumOpenInterest": "oi_base",
            "sumOpenInterestValue": "oi_value",
        },
        constants={"source_id": "binance.open_interest_statistics"},
        value_columns=("oi_base", "oi_value",),
        description=(
            "Recent open-interest series for a NAMED ROSTER of perpetuals, in "
            "coins and in dollars — the input for \"positions moved 20% this "
            "week\" across a cross-section."
        ),
        strategy_types=("D", "L", "P"),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.FORWARD_ONLY,
        endpoint="fapi futures/data/openInterestHist — ONE REQUEST PER SYMBOL",
        returns=ReturnSpec(
            "pd.DataFrame", "one row per symbol × period, oldest first",
            ("symbol", "sumOpenInterest", "sumOpenInterestValue", "timestamp"),
        ),
        params=(
            ParamSpec("symbols", "list[str]", required=True,
                      description="the roster to fan out over; capped at "
                                  "blocks.data.FAN_OUT_MAX_SYMBOLS and never "
                                  "truncated"),
            ParamSpec("period", "str", default="1d",
                      description="5m..1d; universe.augment_with_oi_change "
                                  "measures the cadence and refuses to call a "
                                  "non-daily series a daily change"),
            ParamSpec("limit", "int", default=8,
                      description="readings per symbol — a 7-day lookback needs 8"),
            _MARKET_TYPE,
        ),
        pit_hazard=(
            "Two hazards, not one. The public endpoint keeps only ~30 days, so a "
            "longer lookback silently starts mid-series. And the ROSTER is "
            "point-in-time: it was chosen from the universe as it looked at "
            "capture, so replaying this frame against a different decision time "
            "screens a set of instruments that was picked using the future."
        ),
        fetcher="cyqnt_trd.blocks.data.fetch_oi_history_cross_section",
        notes=(
            "FAN-OUT node — see `open_interest_snapshot` for the ordering "
            "requirement. Deliberately no `unit` constant: the two metrics are in "
            "DIFFERENT units (oi_base in coins, oi_value in USD), so one frame-wide "
            "unit would mislabel whichever it did not describe. "
            "`CMCCirculatingSupply` is dropped by the fetcher: it is a "
            "third-party supply figure, not open interest, and it would repeat on "
            "every row of every series. It is served instead by "
            "`circulating_supply_snapshot`, the fundamentals node this comment "
            "used to point at without one existing."
        ),
    ),
    DataNodeSpec(
        name="circulating_supply_snapshot",
        emits=FrameKind.METRIC,
        column_map={
            "symbol": "instrument_id",
            "supply_time": "event_time",
        },
        constants={"source_id": "binance.cmc_circulating_supply",
                   "unit": "coins"},
        value_columns=("circulating_supply", "supply_unchanged_periods"),
        description=(
            "Circulating supply for a NAMED ROSTER of perpetuals — the missing "
            "half of a market cap, which is otherwise unscreenable on this "
            "venue. One row per symbol: the latest reading plus how many periods "
            "before it carry the identical figure."
        ),
        strategy_types=("L", "P", "N"),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.FORWARD_ONLY,
        endpoint="fapi futures/data/openInterestHist — ONE REQUEST PER SYMBOL",
        returns=ReturnSpec(
            "pd.DataFrame", "one row per symbol",
            ("symbol", "circulating_supply", "supply_time",
             "supply_unchanged_periods"),
        ),
        params=(
            ParamSpec("symbols", "list[str]", required=True,
                      description="the roster to fan out over; capped at "
                                  "blocks.data.FAN_OUT_MAX_SYMBOLS and never "
                                  "truncated"),
            ParamSpec("period", "str", default="1d",
                      description="the cadence the staleness count is measured "
                                  "in, not the cadence of the emitted row"),
            ParamSpec("limit", "int", default=8,
                      description="how far back the flat-run check looks; only "
                                  "the latest reading is emitted"),
            _MARKET_TYPE,
        ),
        pit_hazard=(
            "Three hazards. The endpoint keeps only ~30 days, so historical "
            "market cap cannot reach further back than that. The ROSTER is "
            "point-in-time in the same way `oi_change_snapshot`'s is. And the "
            "upstream figure can STALL without saying so: BTCUSDT held at "
            "20,071,518 for six days in 2026-08 before a single +3,216 "
            "catch-up, so a replay across those days reuses a frozen supply. "
            "`supply_unchanged_periods` is emitted so the stall is visible, but "
            "it cannot distinguish a stalled feed from a token whose supply is "
            "genuinely fixed."
        ),
        fetcher="cyqnt_trd.blocks.data.fetch_circulating_supply_cross_section",
        notes=(
            "FAN-OUT node — see `open_interest_snapshot` for the ordering "
            "requirement. Same endpoint as `oi_change_snapshot` and therefore a "
            "SECOND pass over it; they are separate nodes because the supply is "
            "one current reading while the open interest is a series, and "
            "carrying the supply on the series would repeat it on every row. "
            "The figure is CoinMarketCap's and matches what binance.com "
            "displays (ARB 6,678,075,931 on 2026-08-26, identical to the CMC "
            "page). Do NOT substitute the `cs` field from the undocumented "
            "bapi get-products endpoint: across 24 majors its median "
            "disagreement is 11.14% and it is worst where supply moves fastest "
            "(TIA +92%, OP +69%, ARB 4.21e9 against 6.68e9). A synthetic "
            "instrument answers 0 and the fetcher turns that into NaN, because "
            "a zero cap is dropped by a floor for being SMALL."
        ),
    ),
    DataNodeSpec(
        name="long_short_ratio_snapshot",
        emits=FrameKind.RANK,
        column_map={"symbol": "instrument_id", "timestamp": "event_time"},
        constants={"source_id": "binance.long_short_ratio"},
        description=(
            "Latest long/short ACCOUNT ratio for a NAMED ROSTER of perpetuals — "
            "the cross-section behind \"retail is more than 60% long\"."
        ),
        strategy_types=("D", "L", "P"),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.FORWARD_ONLY,
        endpoint="fapi futures/data/globalLongShortAccountRatio (or the two "
                 "topLongShort* variants) — ONE REQUEST PER SYMBOL",
        returns=ReturnSpec(
            "pd.DataFrame", "one latest row per requested symbol",
            ("symbol", "longAccount", "shortAccount", "longShortRatio", "timestamp"),
        ),
        params=(
            ParamSpec("symbols", "list[str]", required=True,
                      description="the roster to fan out over; capped at "
                                  "blocks.data.FAN_OUT_MAX_SYMBOLS and never "
                                  "truncated"),
            ParamSpec("period", "str", default="1h", description="5m..1d"),
            ParamSpec("mode", "str", default="global",
                      options=("global", "top_account", "top_position"),
                      description="whose positioning: the whole venue (retail) "
                                  "or the top traders by account / by position"),
            _MARKET_TYPE,
        ),
        pit_hazard=(
            "One reading per instrument with no history kept, and a "
            "point-in-time ROSTER — see `oi_change_snapshot`. For the ratio's own "
            "history on one instrument use `long_short_ratio` (~30d)."
        ),
        fetcher="cyqnt_trd.blocks.data.fetch_long_short_ratio_cross_section",
        notes=(
            "FAN-OUT node — see `open_interest_snapshot` for the ordering "
            "requirement. `longAccount` / `shortAccount` are SHARES OF 1 on the "
            "wire; universe.augment_with_long_short_ratio converts them to "
            "percentage points and verifies that they still sum to 1, so a unit "
            "change at the source raises instead of producing a 6700% long share."
        ),
    ),
    DataNodeSpec(
        name="book_ticker",
        emits=FrameKind.RANK,
        column_map={
            "symbol": "instrument_id",
            "bidPrice": "bid_price",
            "bidQty": "bid_qty",
            "askPrice": "ask_price",
            "askQty": "ask_qty",
            "time": "event_time",
        },
        constants={"source_id": "binance.bookTicker"},
        description=(
            "Best bid/ask and the size resting on each, for EVERY symbol in one "
            "request. This is the cross-section a SELECTION strategy joins to say "
            "\"drop the illiquid air coins\" and mean it: turnover says how much "
            "traded yesterday, the spread says whether an order can be filled now."
        ),
        strategy_types=("L", "P", "S"),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.FORWARD_ONLY,
        endpoint="fapi.binance.com/fapi/v1/ticker/bookTicker (omit symbol -> all "
                 "727 instruments, weight 5, NO fan-out)",
        returns=ReturnSpec(
            "pd.DataFrame",
            "one current row per instrument the venue quotes",
            ("symbol", "bidPrice", "bidQty", "askPrice", "askQty", "time"),
        ),
        params=(_MARKET_TYPE,),
        pit_hazard=(
            "The top of an order book, which is the fastest-moving quantity in "
            "this catalog: it is rewritten many times a second and the endpoint "
            "serves no history at all. Replaying today's spread at an earlier "
            "decision time states this second's liquidity as last month's, and "
            "does it for the instruments whose liquidity changed most. Capture it "
            "forward alongside the universe it will be joined to; there is no "
            "backtestable substitute in this repo (the L2 archives under "
            "orderbook-research are a separate collection, not this node)."
        ),
        fetcher="cyqnt_trd.blocks.data.fetch_book_ticker_cross_section",
        notes=(
            "Whole-market and free, so it is NOT one of the fan-out nodes and "
            "takes no `symbols` roster — see live_snapshot.FAN_OUT_SECTIONS for "
            "the ones that do. `lastUpdateId` is dropped by the fetcher: it is a "
            "per-symbol book revision counter, incomparable across instruments. "
            "universe.augment_with_spread turns these four numbers into "
            "`spread_bps` and `top_of_book_usd`, and refuses to report a spread "
            "for a one-sided or crossed book rather than emitting a number that "
            "would pass a max_spread_bps screen."
        ),
    ),
    DataNodeSpec(
        name="funding_info",
        emits=FrameKind.RANK,
        column_map={
            "symbol": "instrument_id",
            "fundingIntervalHours": "funding_interval_hours",
            "adjustedFundingRateCap": "funding_rate_cap",
            "adjustedFundingRateFloor": "funding_rate_floor",
        },
        constants={"source_id": "binance.fundingInfo"},
        description=(
            "How OFTEN each perpetual settles funding (8h / 4h / 1h) plus the "
            "venue's rate clamps. Without it a funding cross-section cannot be "
            "annualised: `funding_snapshot` gives the rate PER SETTLEMENT, so the "
            "same 0.01% is 10.95%/yr on an 8-hourly contract and 87.6%/yr on an "
            "hourly one, and ranking the raw number silently mixes both units."
        ),
        strategy_types=("D", "L", "P"),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.FORWARD_ONLY,
        endpoint="fapi.binance.com/fapi/v1/fundingInfo (all symbols, weight 1)",
        returns=ReturnSpec(
            "pd.DataFrame",
            "one row per perpetual with a published funding schedule",
            ("symbol", "fundingIntervalHours", "adjustedFundingRateCap",
             "adjustedFundingRateFloor"),
        ),
        params=(_MARKET_TYPE,),
        pit_hazard=(
            "A schedule, not a measurement — but a schedule the venue REWRITES: "
            "Binance has moved large batches of perpetuals from 8h to 4h "
            "settlement, and 443 of 743 contracts are 4-hourly today (measured "
            "2026-08-02). So replaying today's table at an earlier decision time "
            "annualises past funding with a divisor the contract did not have, "
            "which is a wrong number rather than a missing one. Capture it forward "
            "alongside the funding snapshot it will be joined to."
        ),
        fetcher="cyqnt_trd.blocks.data.fetch_funding_info",
        notes=(
            "743 rows covers 723 of the 727 instruments in a futures 24h ticker; "
            "the four absentees are dated delivery contracts (BTCUSDT_260925 and "
            "friends), which pay no funding at all. A missing row becomes NaN in "
            "universe.augment_with_funding and is NEVER defaulted to 8 hours. "
            "`updateTime` is dropped rather than mapped to event_time: it is null "
            "for BTCUSDT and ETHUSDT, so gating on it would drop the two most "
            "important rows of the cross-section."
        ),
    ),
    DataNodeSpec(
        name="universe_bars",
        emits=FrameKind.BAR,
        column_map={"symbol": "instrument_id"},
        constants={"source_id": "binance.klines"},
        description=(
            "OHLCV bars for a NAMED ROSTER of instruments across SEVERAL "
            "timeframes, in one frame. This is what a SELECTION strategy joins to "
            "run a technical indicator on each candidate: the universe "
            "cross-section has one row per instrument and no bars, so "
            "\"Supertrend(10,3) bearish on 4h AND 1h AND 15m\" and \"up 100% from "
            "its 3-month low\" could not be stated at all — what shipped instead "
            "was top_losers(n=30), a 24h percentage wearing an indicator's words."
        ),
        strategy_types=("C", "L", "P", "S", "R"),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.BACKTESTABLE,
        endpoint="fapi.binance.com/fapi/v1/klines — ONE REQUEST PER (symbol, "
                 "timeframe); weight 1 at limit<=100, 2 at 101-500 (measured via "
                 "X-MBX-USED-WEIGHT-1M, 2026-08-02)",
        returns=ReturnSpec(
            "pd.DataFrame",
            "one row per instrument x timeframe x bar, oldest first within a pair",
            ("symbol", "timeframe", "open_time", "open", "high", "low", "close",
             "volume", "close_time", "quote_volume", "trades"),
        ),
        params=(
            ParamSpec("symbols", "list[str]", required=True,
                      description="the roster to fan out over; plan it from the "
                                  "surviving prefix of the spec's own universe "
                                  "steps, never from the whole venue"),
            ParamSpec("timeframes", "list[str]", required=True,
                      description='e.g. ["4h", "1h", "15m"] — the union of the '
                                  "timeframes the spec's indicator steps name"),
            ParamSpec("limit", "int", default=200,
                      description="bars per (symbol, timeframe); must cover the "
                                  "longest indicator period plus its settling "
                                  "margin"),
            ParamSpec("end_ms", "int",
                      description="bars up to this epoch-ms instant. Pass the "
                                  "bundle's decision_time: without it the frame "
                                  "is 'as of now' and a replay mixes a past "
                                  "universe with present prices"),
            _MARKET_TYPE,
        ),
        # BACKTESTABLE, and it is the only cross-sectional node in this catalog
        # that is: the endpoint is addressable BY TIME, so `end_ms` returns exactly
        # the bars a capture at that instant would have seen. Every other
        # cross-section here (`universe`, `funding_snapshot`, `book_ticker`,
        # `open_interest_snapshot`) serves only "now" and therefore carries a
        # pit_hazard. That difference is why this frame may be collected for a
        # bundle whose decision_time has already passed while those may not.
        #
        # ONE property does not replay: the ROSTER. It is derived from a
        # forward-only universe snapshot, so a bundle's bars answer for the
        # instruments that snapshot contained — which is correct for that bundle
        # and is not a claim about what a screen would have chosen at any other
        # instant.
        fetcher="cyqnt_trd.blocks.data.fetch_klines_cross_section",
        notes=(
            "Deliberately NOT the `klines` node. That one's fetcher is a "
            "binance-cli subprocess with no endTime parameter, so it can only "
            "answer 'the last N bars as of now' and cannot be point-in-time. It is "
            "also single-instrument, and its bundle key is special-cased into a "
            "MarketBundle keyed on the first symbol — landing a multi-symbol frame "
            "there would file every instrument's bars under one name. FAN-OUT node: "
            "cost is one request per (symbol, timeframe), bounded in WEIGHT rather "
            "than in requests because the per-call price depends on `limit` — see "
            "blocks.data.KLINE_FAN_OUT_MAX_WEIGHT. The last bar of each series is "
            "normally unfinished and is dropped by the bundle's own PIT gate, not "
            "by the fetcher, so the property stays visible in the artifact."
        ),
    ),
    DataNodeSpec(
        name="bdp_screen",
        emits=FrameKind.RANK,
        column_map={"symbol": "instrument_id"},
        description="Cross-sectional screener fields (market_cap, taker_buy_pct, "
                    "community_buzz, vol_mkt_cap, fdv_ratio, large_trade_pct, "
                    "large_buy_pct + rsi/macd/kdj/ma/boll enum signals).",
        strategy_types=("L", "P", "C", "S"),
        source_path=SourcePath.BDP_SCREENING,
        availability=Availability.FORWARD_ONLY,
        endpoint='screener /api/query with a condition clause carrying source:"bdp" (base URL from $BDP_SCREENING_URL)',
        fetcher="cyqnt_trd.data_cli.internal_frames.fetch_bdp_screen",
        returns=ReturnSpec("pd.DataFrame", "symbols passing the clause"),
        params=(
            ParamSpec("api_id", "str", required=True,
                      description="market_cap | taker_buy_pct | community_buzz | vol_mkt_cap | "
                                  "fdv_ratio | large_trade_pct | large_buy_pct | rsi_signal | "
                                  "macd_cross | kdj_cross | ma_cross | ema_cross | boll_cross"),
            ParamSpec("type", "str", default="range", options=("range", "enum")),
            ParamSpec("min", "float"), ParamSpec("max", "float"),
            ParamSpec("api_enum", "str", description="for type=enum"),
        ),
        pit_hazard="bdp exposes only the CURRENT snapshot. A walk-forward backtest "
                   "reuses today's value at every bar — strictly lookahead. Honest for "
                   "live selection at t=now; every bdp strategy must carry this note.",
        notes="Do not hand-roll a fetcher for these fields; express them as a clause.",
    ),
    DataNodeSpec(
        name="market_scan",
        emits=FrameKind.RANK,
        column_map={"symbol": "instrument_id"},
        param_aliases={"market_type": "market"},
        description=(
            "Enumerate tradable instrument NAMES. Returns a sorted list of "
            "symbols already narrowed to status=TRADING and quoteAsset=USDT — "
            "there are no gainer/loser/volume knobs and no ticker fields. For "
            "the rankable 24h cross-section use `universe`; for the unfiltered "
            "table with contract type and sector tags use `contract_meta`."
        ),
        strategy_types=("L", "S"),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.FORWARD_ONLY,
        endpoint="binance-cli futures-usds exchange-information / spot "
                 "exchange-info + client-side status+quote filter",
        returns=ReturnSpec(
            "list[str]",
            "sorted symbol names, TRADING and USDT-quoted only",
        ),
        params=(_MARKET_TYPE,),
        pit_hazard=(
            "Current listing snapshot, and it is a bare name list — it carries "
            "no timestamp at all, so nothing downstream can gate it. Never "
            "replay it as the universe of an earlier decision: today's list "
            "contains coins that were not listed then and omits the delisted "
            "ones a historical basket would have held."
        ),
        fetcher="cyqnt_trd.data_cli.scanner.full_market_scan",
        notes=(
            "Two things the declaration above cannot express. (1) The wired "
            "fetcher returns `list[str]`, NOT a frame, so `emits` states the "
            "shape this node would land in a bundle and not what calling it "
            "gives you today — nothing can normalise a bare name list, and the "
            "return type is load-bearing (orchestration/scheduler.py iterates "
            "it), so it is left alone. (2) The status=TRADING + quoteAsset=USDT "
            "filter lives inside the fetcher, so USDC/USD1-quoted perpetuals are "
            "invisible through this node. Both are why it is not the input a "
            "selection strategy screens over: use `universe` or `contract_meta`."
        ),
    ),

    # ---- F: flow / whale ---------------------------------------------------
    DataNodeSpec(
        name="large_flow",
        emits=FrameKind.METRIC,
        column_map={"window_start_time": "event_time"},
        constants={"source_id": "movement.largeDepositWithdraw"},
        value_columns=("total_deposit_qty", "total_withdraw_qty", "total_deposit_amt", "total_withdraw_amt", "large_deposit_qty", "large_withdraw_qty", "large_deposit_amt", "large_withdraw_amt", "avg_amt",),
        description="Large deposit/withdraw (exchange in/out) signal per token.",
        strategy_types=("F",),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.SEMI,
        endpoint="movement/getLargeDepositWithdrawData{token,interval}",
        fetcher="cyqnt_trd.data_cli.internal_frames.fetch_large_flow",
        returns=ReturnSpec("pd.DataFrame", "1h windows",
                           ("window_start_time", "total_deposit_qty", "total_withdraw_qty",
                            "large_deposit_amt", "large_withdraw_amt", "total_signal")),
        params=(_TOKEN, ParamSpec("interval", "str", default="1h")),
        pit_hazard="1h windowed history exists but is shallow; internal-only.",
    ),
    DataNodeSpec(
        name="whale_trades",
        emits=FrameKind.METRIC,
        column_map={"time": "event_time"},
        constants={"source_id": "local.whale_bars"},
        value_columns=("tot_buy", "tot_sell", "whale_buy", "whale_sell", "whale_cnt", "n_trades"),
        description="Whale-sized trade aggregation (5-min buy/sell bars).",
        strategy_types=("F", "O"),
        source_path=SourcePath.LOCAL_PARQUET,
        availability=Availability.SEMI,
        endpoint="local autotrade/t15_whale/data bars_*_5min",
        returns=ReturnSpec("pd.DataFrame", "5-min bars",
                           ("time", "tot_buy", "tot_sell", "whale_buy", "whale_sell",
                            "whale_cnt", "n_trades")),
        params=(_SYMBOL,),
        pit_hazard="local capture is BTC-only and ~40 days (2026-04-25..06-03).",
    ),

    # ---- account -----------------------------------------------------------
    DataNodeSpec(
        name="account_balance",
        emits=FrameKind.METRIC,
        column_map={"asset": "instrument_id"},
        constants={"source_id": "binance.account", "unit": "asset_units"},
        value_columns=("free", "locked", "total"),
        description="Account balances (sizing / risk inputs).",
        strategy_types=("*",),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.FORWARD_ONLY,
        endpoint="binance-cli account endpoints (authenticated)",
        returns=ReturnSpec("pd.DataFrame", "asset -> free/locked"),
        pit_hazard="live account state; never available historically. A backtest must "
                   "use the simulated equity curve, not this node.",
        fetcher="cyqnt_trd.data_cli.account.fetch_account_balance",
    ),
    DataNodeSpec(
        name="positions",
        emits=FrameKind.POSITION,
        column_map={"symbol": "instrument_id", "positionAmt": "quantity", "entryPrice": "entry_price", "unRealizedProfit": "unrealized_pnl"},
        description="Open positions (exposure / reduce-only checks).",
        strategy_types=("*",),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.FORWARD_ONLY,
        endpoint="binance-cli futures position endpoints (authenticated)",
        returns=ReturnSpec("pd.DataFrame", "one row per position"),
        pit_hazard="live only.",
        fetcher="cyqnt_trd.data_cli.account.fetch_positions",
    ),

    # ---- R: cross-asset / macro -------------------------------------------
    DataNodeSpec(
        name="cme_index",
        emits=FrameKind.BAR,
        description="CME index futures (NQ / ES) bars for cross-asset regime.",
        strategy_types=("R",),
        source_path=SourcePath.LOCAL_PARQUET,
        availability=Availability.SEMI,
        endpoint="local parquet ingested by standard_bot.data.cme (yfinance / HuggingFace / CSV)",
        returns=ReturnSpec(
            "pd.DataFrame", "OHLCV bars",
            ("open_time", "open", "high", "low", "close", "volume", "quote_volume", "close_time", "trades"),
        ),
        params=(ParamSpec("instrument_id", "str", required=True, description="e.g. NQ=F"), _INTERVAL),
        pit_hazard="ingested history only; check the ingested span before backtesting.",
    ),
    DataNodeSpec(
        name="btc_dominance",
        emits=FrameKind.METRIC,
        column_map={"date": "event_time"},
        constants={"instrument_id": "MARKET", "source_id": "cmc"},
        value_columns=("btc_dominance", "total_market_cap",),
        description="BTC dominance and total crypto market cap — top-level risk switch.",
        strategy_types=("R",),
        source_path=SourcePath.EXTERNAL_VENDOR,
        availability=Availability.EXTERNAL_PENDING,
        endpoint="CMC vendor feed (b9-design page 590233601)",
        returns=ReturnSpec("pd.DataFrame", "daily", ("date", "btc_dominance", "total_market_cap")),
        notes="Rated the single most valuable genuinely-new external field for crypto. "
              "Not onboarded yet — declare it, do not fake it.",
    ),
    DataNodeSpec(
        name="sector_flow",
        emits=FrameKind.RANK,
        column_map={"category": "instrument_id"},
        constants={"source_id": "rank.hotCategoryDetail"},
        description="Sector tags + sector net inflow (rotation, sector-neutral books).",
        strategy_types=("L", "N"),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.SEMI,
        endpoint="rank/hotCategoryDetail, dim_asset_tag_relation, sector_net_inflow",
        fetcher="cyqnt_trd.data_cli.internal_frames.fetch_sector_flow",
        returns=ReturnSpec(
            "pd.DataFrame", "one row per sector",
            ("category", "net_inflow", "market_cap", "change_pct", "heat", "members"),
        ),
        params=(ParamSpec("sort", "str", default="netInflow"),),
        pit_hazard="10-minute refresh snapshot; no bulk history endpoint.",
        notes="rank/hotCategoryDetail is the served path; dim_asset_tag_relation / sector_net_inflow are the warehouse tables behind it.",
    ),
    DataNodeSpec(
        name="options_chain",
        emits=FrameKind.METRIC,
        column_map={"contract": "instrument_id"},
        constants={"source_id": "options_vendor"},
        value_columns=("iv", "delta", "gamma", "vega", "theta", "open_interest"),
        description="Option chain (strike x expiry IV / greeks / OI) for the vol surface.",
        strategy_types=("A",),
        source_path=SourcePath.EXTERNAL_VENDOR,
        availability=Availability.EXTERNAL_PENDING,
        endpoint="Deribit / Binance options (外采, onboarding)",
        returns=ReturnSpec("pd.DataFrame", "one row per contract"),
        notes="Local t11_iv BVOL zips are the only implied-vol history on hand.",
    ),

    # ---- served by the internal domain (indicators / radar / movement / …) --
    DataNodeSpec(
        name="indicator_charts",
        emits=FrameKind.BAR,
        constants={"instrument_id": None, "timeframe": None},
        description="Klines plus the 14 ta4j indicators in one call — the canonical "
                    "in-domain source for C-type strategies.",
        strategy_types=("C", "P", "R"),
        source_path=SourcePath.INDICATORS_API,
        availability=Availability.BACKTESTABLE,
        endpoint="indicators service (base URL from $INDICATORS_URL)",
        returns=ReturnSpec(
            "pd.DataFrame", "kline block joined with each requested indicator block",
            ("open_time", "open", "high", "low", "close", "volume", "quote_volume",
             "close_time", "trades"),
        ),
        params=(
            _SYMBOL, _INTERVAL,
            ParamSpec("limit", "int", default=500, description="<= 2000"),
            ParamSpec("product", "str", default="um-perp",
                      options=("spot", "um-perp", "web3-alpha")),
            ParamSpec("indicator_ids", "list[str]",
                      description="ma/ema/rsi/macd/boll/kdj/atr/stdev/wr/sar/super/"
                                  "stochrsi/avl/obv; omit for klines only"),
            ParamSpec("end_time", "int", description="UTC ms cut-off for replay"),
        ),
        fetcher="cyqnt_trd.data_cli.internal_frames.fetch_indicator_charts",
        notes="ta4j returns indicator VALUES; value->signal conversion (golden cross, "
              "RSI bands, BOLL breaks) is not provided upstream — use cyqnt_trd.blocks "
              "conditions for that.",
    ),
    DataNodeSpec(
        name="token_indicators",
        emits=FrameKind.METRIC,
        column_map={"symbol": "instrument_id"},
        constants={"source_id": "indicator.tokenIndicators"},
        description="ta4j indicator values for one symbol (spot or futures).",
        strategy_types=("C",),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.FORWARD_ONLY,
        endpoint="/indicator/tokenIndicators{symbol,type,interval}",
        returns=ReturnSpec("pd.DataFrame", "latest indicator values"),
        params=(_SYMBOL, ParamSpec("product_type", "str", default="futures",
                                   options=("spot", "futures")), _INTERVAL),
        pit_hazard="serves the latest computed values; use indicator_charts with "
                   "end_time for a replayable series.",
        fetcher="cyqnt_trd.data_cli.internal_frames.fetch_token_indicators",
    ),
    DataNodeSpec(
        name="futures_radar",
        emits=FrameKind.METRIC,
        column_map={"symbol": "instrument_id"},
        constants={"source_id": "futuresRadar"},
        description="futuresRadar derivatives snapshot — funding / long-short / OI / "
                    "buy-sell lists across the perp universe in one read.",
        strategy_types=("D",),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.FORWARD_ONLY,
        endpoint="futuresRadar/getRadarDataForWidget, futuresRadar/getRadarData{indicatorKey}",
        returns=ReturnSpec("pd.DataFrame", "long form",
                           ("symbol", "metric", "value", "event_time")),
        params=(ParamSpec("indicator_key", "str",
                          description="omit for the widget bundle; set for one of the "
                                      "70+ radar fields"),),
        pit_hazard="5-minute snapshot, no history endpoint. Cross-sectional crowding "
                   "read at t=now; use funding/open_interest for anything replayed.",
        fetcher="cyqnt_trd.data_cli.internal_frames.fetch_futures_radar",
    ),
    DataNodeSpec(
        name="funding_widget",
        emits=FrameKind.METRIC,
        column_map={"symbol": "instrument_id"},
        constants={"source_id": "futuresRadar.fundingRateWidget"},
        value_columns=("funding_rate", "next_funding_time",),
        description="Current funding rate + next settlement time (8h cadence).",
        strategy_types=("D", "A"),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.FORWARD_ONLY,
        endpoint="futuresRadar/fundingRateWidget",
        returns=ReturnSpec("pd.DataFrame", "one row per symbol",
                           ("symbol", "funding_rate", "next_funding_time")),
        params=(ParamSpec("symbol", "str", description="omit for the full list"),),
        pit_hazard="snapshot. next_funding_time is a forward prediction, not a settled value.",
        fetcher="cyqnt_trd.data_cli.internal_frames.fetch_funding_widget",
    ),
    DataNodeSpec(
        name="chip_distribution",
        emits=FrameKind.BOOK,
        column_map={"token": "instrument_id", "price": "price", "volume": "quantity"},
        constants={"side": "holders", "level": 0},
        description="Cost-basis (chip) distribution — where holders are under water.",
        strategy_types=("F",),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.FORWARD_ONLY,
        endpoint="movement/getChipDistributionData{token}",
        returns=ReturnSpec("pd.DataFrame", "price buckets with held volume"),
        params=(_TOKEN,),
        pit_hazard="snapshot only.",
        fetcher="cyqnt_trd.data_cli.internal_frames.fetch_chip_distribution",
    ),
    DataNodeSpec(
        name="top_player_movement",
        emits=FrameKind.RANK,
        column_map={"symbol": "instrument_id"},
        description="Top-trader position moves (smart-money follow / fade).",
        strategy_types=("F", "D"),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.FORWARD_ONLY,
        endpoint="movement/getTopPlayerMovementList[ForSignal]",
        returns=ReturnSpec("pd.DataFrame", "TopTraderResponse rows"),
        params=(ParamSpec("for_signal", "bool", default=False),),
        pit_hazard="snapshot only.",
        fetcher="cyqnt_trd.data_cli.internal_frames.fetch_top_player_movement",
    ),
    DataNodeSpec(
        name="large_trade_info",
        emits=FrameKind.METRIC,
        column_map={"time": "event_time"},
        constants={"source_id": "hot.largeTradeInfo"},
        value_columns=("volume", "volume_usdt",),
        description="Large buy/sell prints for one token (block-trade pressure).",
        strategy_types=("F", "O"),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.FORWARD_ONLY,
        endpoint="hot/getHotCoinDetailForLargeTradeInfo{token}",
        returns=ReturnSpec("pd.DataFrame", "prints",
                           ("time", "side", "volume", "volume_usdt")),
        params=(_TOKEN,),
        pit_hazard="snapshot window only.",
        fetcher="cyqnt_trd.data_cli.internal_frames.fetch_large_trade_info",
    ),
    DataNodeSpec(
        name="calendar",
        emits=FrameKind.EVENT,
        column_map={"coin": "instrument_id", "event_title": "title", "event_type": "topic", "deeplink": "url"},
        constants={"source_id": "calendar.widget"},
        description="Listing / unlock / activity calendar (forward schedule).",
        strategy_types=("E",),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.SEMI,
        endpoint="calendar/dailyCalendarForWidget, calendar/monthlyCalendarForWidget",
        returns=ReturnSpec("pd.DataFrame", "one row per scheduled event",
                           ("event_id", "event_time", "event_type", "coin",
                            "event_title", "deeplink")),
        params=(ParamSpec("period", "str", default="daily", options=("daily", "monthly")),
                ParamSpec("date", "str", description="YYYY-MM-DD / YYYY-MM")),
        pit_hazard="forward schedule; no bulk history. event_time is known ahead, so "
                   "gating on it is PIT-safe.",
        fetcher="cyqnt_trd.data_cli.internal_frames.fetch_calendar",
    ),
    DataNodeSpec(
        name="coin_metrics",
        emits=FrameKind.RANK,
        column_map={"symbol": "instrument_id"},
        description="Per-symbol market cap / liquidity / float (size + quality factors).",
        strategy_types=("L", "P", "N"),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.SEMI,
        endpoint="coinSelector/getMetricsCache (CoinMetrics-derived, 1h/4h)",
        returns=ReturnSpec("pd.DataFrame", "one row per symbol"),
        params=(ParamSpec("token", "str", description="omit for the full universe"),),
        pit_hazard="cache refreshed hourly; treat as slow-moving, not tick-accurate.",
        fetcher="cyqnt_trd.data_cli.internal_frames.fetch_coin_metrics",
    ),
    DataNodeSpec(
        name="hot_coin",
        emits=FrameKind.RANK,
        column_map={"symbol": "instrument_id"},
        description="BTC / top-10 quote snapshot (10s upstream refresh).",
        strategy_types=("R", "S"),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.FORWARD_ONLY,
        endpoint="hot/getHotCoin",
        returns=ReturnSpec("pd.DataFrame", "one row per hot coin"),
        pit_hazard="snapshot.",
        fetcher="cyqnt_trd.data_cli.internal_frames.fetch_hot_coin",
    ),
    DataNodeSpec(
        name="ai_signal",
        emits=FrameKind.RANK,
        column_map={"symbol": "instrument_id"},
        description="Existing hourly multi-factor composite signal + its factor list.",
        strategy_types=("C", "P"),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.FORWARD_ONLY,
        endpoint="/ai-skill/getAiSignal{symbol,type}",
        returns=ReturnSpec("pd.DataFrame", "single row",
                           ("symbol", "value", "signal", "long_win_rate",
                            "short_win_rate", "avg_max_return", "factors")),
        params=(_SYMBOL, ParamSpec("product_type", "str", default="spot",
                                   options=("spot", "alpha"))),
        pit_hazard="snapshot. Coverage gap: source B is spot/alpha only — um-perp "
                   "has no aiSignal coverage.",
        fetcher="cyqnt_trd.data_cli.internal_frames.fetch_ai_signal",
        notes="This is also an output slot: our factor books are meant to extend "
              "currentFactors rather than duplicate it.",
    ),
    DataNodeSpec(
        name="strategy_ranking",
        emits=FrameKind.RANK,
        column_map={"symbol": "instrument_id"},
        description="Ranked strategies per symbol with return / sharpe / win-rate / maxDD.",
        strategy_types=("P",),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.FORWARD_ONLY,
        endpoint="/ai-skill/getStrategyRanking{symbol,type}",
        returns=ReturnSpec("pd.DataFrame", "top-N rows",
                           ("rank", "total_return", "sharpe", "win_rate",
                            "max_dd", "avg_hold", "symbol")),
        params=(_SYMBOL, ParamSpec("product_type", "str", default="spot")),
        pit_hazard="snapshot of an already-fitted ranking; using it as a feature is "
                   "borrowing someone else's hindsight.",
        fetcher="cyqnt_trd.data_cli.internal_frames.fetch_strategy_ranking",
        notes="Also the production slot our own strategy_ranking output plugs into.",
    ),
    DataNodeSpec(
        name="user_portfolio",
        emits=FrameKind.POSITION,
        column_map={"symbol": "instrument_id"},
        description="User holdings with cost basis, PnL and idle days.",
        strategy_types=("*",),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.FORWARD_ONLY,
        endpoint="binance-portfolio getPortfolio",
        returns=ReturnSpec(
            "pd.DataFrame", "one row per holding",
            ("symbol", "quantity", "entry_price", "unrealized_pnl"),
        ),
        params=(ParamSpec("user_id", "str"),),
        pit_hazard="live account state; a backtest must use the simulated equity curve.",
        fetcher="cyqnt_trd.data_cli.internal_frames.fetch_user_portfolio",
    ),
    DataNodeSpec(
        name="user_favorites",
        emits=FrameKind.RANK,
        column_map={"symbol": "instrument_id"},
        description="User watchlist / favourites — the universe a personal bot ranks.",
        strategy_types=("L", "S"),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.FORWARD_ONLY,
        endpoint="favorite/getUserFavorite",
        returns=ReturnSpec("pd.DataFrame", "one row per symbol"),
        params=(ParamSpec("user_id", "str"),),
        pit_hazard="live state.",
        fetcher="cyqnt_trd.data_cli.internal_frames.fetch_user_favorites",
    ),
    DataNodeSpec(
        name="premium_index",
        emits=FrameKind.BAR,
        constants={"instrument_id": None, "timeframe": None},
        description="Perp mark-vs-index premium series (the long form of basis).",
        strategy_types=("A", "D"),
        source_path=SourcePath.PUBLIC_BINANCE,
        availability=Availability.SEMI,
        endpoint="fapi/v1/premiumIndexKlines; vision futures/um premiumIndexKlines archive",
        returns=ReturnSpec(
            "pd.DataFrame", "OHLC of the premium index",
            ("open_time", "open", "high", "low", "close", "volume", "quote_volume", "close_time", "trades"),
        ),
        params=(_SYMBOL, _INTERVAL, _LIMIT),
        pit_hazard="archive currently holds ~400 days; a >=3yr backfill is an open item.",
    ),
    DataNodeSpec(
        name="announcements",
        emits=FrameKind.EVENT,
        column_map={"announce_time": "event_time", "tokens": "instrument_id", "category": "topic"},
        constants={"source_id": "binance-announcement", "event_id": None},
        description="Binance official announcements with token linkage and impact tag.",
        strategy_types=("E",),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.SEMI,
        endpoint="binance-announcement service (D09)",
        returns=ReturnSpec("pd.DataFrame", "one row per announcement",
                           ("announce_time", "title", "tokens", "sentiment",
                            "category", "impact")),
        pit_hazard="publication time is trustworthy; the derived sentiment/impact tags "
                   "are model output and may be re-scored later.",
        notes="Distinct from Square getNews: this is the official announcement feed.",
    ),
    DataNodeSpec(
        name="onchain_signals",
        emits=FrameKind.METRIC,
        column_map={"symbol": "instrument_id"},
        constants={"source_id": "onchain.signal"},
        value_columns=("strength",),
        description="On-chain anomaly signals (FOMO / PANIC / LARGE_INVESTORS / "
                    "LIQUIDITY_SURGE) with direction and strength.",
        strategy_types=("F",),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.FORWARD_ONLY,
        endpoint="onchain.signal service",
        returns=ReturnSpec("pd.DataFrame", "one row per signal",
                           ("event_time", "symbol", "signal_type", "direction", "strength")),
        pit_hazard="event stream is emitted forward; no replayable archive.",
    ),
    DataNodeSpec(
        name="square_discussion",
        emits=FrameKind.METRIC,
        column_map={"symbol": "instrument_id"},
        constants={"source_id": "hot_discussion_content_summary"},
        value_columns=("bull_bear_ratio", "heat_0_100",),
        description="Normalised Square discussion heat and KOL summary per token.",
        strategy_types=("S",),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.FORWARD_ONLY,
        endpoint="hot_discussion_content_summary",
        returns=ReturnSpec("pd.DataFrame", "one row per token",
                           ("symbol", "bull_bear_ratio", "heat_0_100", "kol_summary")),
        pit_hazard="derived snapshot over the Square feed; inherits its no-history limit.",
    ),
    DataNodeSpec(
        name="concentration",
        emits=FrameKind.METRIC,
        column_map={"date": "event_time", "symbol": "instrument_id"},
        constants={"source_id": "spottrading_page_concentration_score_day"},
        value_columns=("concentration_score",),
        description="On-venue holding concentration score per symbol (daily).",
        strategy_types=("F", "L"),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.SEMI,
        endpoint="warehouse table bnb_dwa.spottrading_page_concentration_score_day",
        returns=ReturnSpec("pd.DataFrame", "daily", ("date", "symbol", "concentration_score")),
        pit_hazard="daily batch; same-day value is not available intraday.",
    ),
    # ---- specified, feed not onboarded -------------------------------------
    DataNodeSpec(
        name="macro_indicators",
        emits=FrameKind.METRIC,
        constants={"source_id": "macro_vendor"},
        description="DXY / UST yields / VIX / PMI — cross-asset risk backdrop.",
        strategy_types=("R",),
        source_path=SourcePath.EXTERNAL_VENDOR,
        availability=Availability.EXTERNAL_PENDING,
        endpoint="FX / UST / CBOE vendor feeds (外采, item 88)",
        returns=ReturnSpec("pd.DataFrame", "daily series per indicator"),
    ),
    DataNodeSpec(
        name="stablecoin_supply",
        emits=FrameKind.METRIC,
        constants={"source_id": "onchain_supply"},
        description="Stablecoin mint / burn — aggregate dry powder entering the market.",
        strategy_types=("R", "F"),
        source_path=SourcePath.EXTERNAL_VENDOR,
        availability=Availability.EXTERNAL_PENDING,
        endpoint="on-chain supply feed (item 86)",
        returns=ReturnSpec("pd.DataFrame", "daily supply delta per issuer"),
        substitute="local autotrade/t21_stablecoin 1m parquet (USDC, FDUSD) for research",
    ),
    DataNodeSpec(
        name="cross_exchange",
        emits=FrameKind.BAR,
        constants={"instrument_id": None, "timeframe": None},
        description="OKX / Bybit klines, depth and trades for cross-venue spreads.",
        strategy_types=("A",),
        source_path=SourcePath.EXTERNAL_VENDOR,
        availability=Availability.EXTERNAL_PENDING,
        endpoint="self-collected cross-exchange feed (item 65)",
        returns=ReturnSpec(
            "pd.DataFrame", "per-venue quotes",
            ("open_time", "open", "high", "low", "close", "volume", "quote_volume", "close_time", "trades"),
        ),
    ),
    DataNodeSpec(
        name="equity_fundamentals",
        emits=FrameKind.METRIC,
        column_map={"ticker": "instrument_id"},
        constants={"source_id": "equity_vendor"},
        value_columns=("pe", "pb", "ps", "roe", "roa", "eps", "revenue_yoy", "margin"),
        description="US equity fundamentals (PE/PB/ROE/EPS/margin) for stock tokens.",
        strategy_types=("R",),
        source_path=SourcePath.EXTERNAL_VENDOR,
        availability=Availability.EXTERNAL_PENDING,
        endpoint="equity fundamentals vendor (items 70-72)",
        returns=ReturnSpec("pd.DataFrame", "one row per ticker per period"),
    ),
    DataNodeSpec(
        name="etf_metadata",
        emits=FrameKind.RANK,
        column_map={"ticker": "instrument_id"},
        constants={"source_id": "etf_vendor"},
        description="ETF category / index / expense / issuer / leverage / AUM tiers.",
        strategy_types=("R",),
        source_path=SourcePath.EXTERNAL_VENDOR,
        availability=Availability.EXTERNAL_PENDING,
        endpoint="ETF.com / Morningstar / issuer -> etf_metadata (items 75-77, 83)",
        returns=ReturnSpec("pd.DataFrame", "one row per ETF"),
    ),
    DataNodeSpec(
        name="news_vendor",
        emits=FrameKind.EVENT,
        constants={"source_id": "news_vendor", "topic": "news"},
        description="Third-party news wire (CoinDesk / CoinTelegraph / Reuters).",
        strategy_types=("E",),
        source_path=SourcePath.EXTERNAL_VENDOR,
        availability=Availability.EXTERNAL_PENDING,
        endpoint="news vendor API (item 84); raw newsText RAG stream is item 85",
        returns=ReturnSpec("pd.DataFrame", "one row per article"),
        notes="The timestamped per-token raw stream (85) is what an LLM event detector "
              "needs; the wire feed alone is headlines.",
    ),
    DataNodeSpec(
        name="social_x",
        emits=FrameKind.METRIC,
        constants={"source_id": "x_api"},
        description="X / Twitter volume, bull-bear split and KOL activity.",
        strategy_types=("S",),
        source_path=SourcePath.EXTERNAL_VENDOR,
        availability=Availability.EXTERNAL_PENDING,
        endpoint="X API (item 90)",
        returns=ReturnSpec("pd.DataFrame", "one row per token per window"),
    ),

    # ---- account / user profile (what BotContext.positions is built from) ----
    DataNodeSpec(
        name="contract_positions",
        emits=FrameKind.POSITION,
        column_map={"symbol": "instrument_id", "positionAmt": "quantity", "entryPrice": "entry_price", "unRealizedProfit": "unrealized_pnl"},
        description="Open perpetual positions: direction, size, leverage, margin "
                    "ratio, unrealised PnL. This is what the position lifecycle "
                    "reads to know whether an exit means CLOSE_LONG or CLOSE_SHORT.",
        strategy_types=("*",),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.FORWARD_ONLY,
        endpoint="合约持仓 BAPI; /ws/tokenList (sub-second SLA)",
        returns=ReturnSpec(
            "pd.DataFrame", "one row per open position",
            ("symbol", "side", "quantity", "entry_price", "leverage",
             "margin_ratio", "unrealized_pnl"),
        ),
        # The wired fetcher keys off the local credential profile, not a user id:
        # it reads the account the API key belongs to. Declaring ``user_id`` implied
        # this node could fetch someone else's book, which it cannot.
        params=(ParamSpec("symbol", "str",
                          description="restrict to one instrument; omit for the whole book"),
                ParamSpec("profile", "str", default="default",
                          description="local credential profile to read the account of")),
        pit_hazard="live account state. A backtest must use the simulated book, "
                   "never this node.",
        fetcher="cyqnt_trd.data_cli.account.fetch_positions",
    ),
    DataNodeSpec(
        name="user_pnl",
        emits=FrameKind.METRIC,
        column_map={"symbol": "instrument_id"},
        constants={"source_id": "user_pnl"},
        value_columns=("realized_pnl", "unrealized_pnl", "roi"),
        description="Per-period, per-coin realised/unrealised PnL and risk slices.",
        strategy_types=("*",),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.SEMI,
        endpoint="用户 PnL SyncRateJob / TokenPriceJob",
        returns=ReturnSpec("pd.DataFrame", "period x symbol PnL"),
        params=(ParamSpec("user_id", "str"), ParamSpec("period", "str", default="30d")),
        pit_hazard="batch-computed per period; intraday values lag.",
    ),
    DataNodeSpec(
        name="ai_summary",
        emits=FrameKind.EVENT,
        constants={"source_id": "aiSummary.getAiSummaryForHome", "topic": "market_summary", "event_id": None},
        description="LLM-generated market summary / report body for the home feed.",
        strategy_types=("E", "R"),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.FORWARD_ONLY,
        endpoint="aiSummary/getAiSummaryForHome (5-10min refresh)",
        returns=ReturnSpec("pd.DataFrame", "summary blocks"),
        pit_hazard="regenerated every few minutes with no archive; it is model "
                   "output over other feeds, not an independent observation.",
        notes="Use as user-facing copy, not as a strategy feature — it restates "
              "inputs the bot can read directly.",
    ),
    DataNodeSpec(
        name="web3_basket",
        emits=FrameKind.RANK,
        column_map={"symbol": "instrument_id"},
        description="Web3 / alpha token top-10 basket quotes.",
        strategy_types=("L", "S"),
        source_path=SourcePath.INTERNAL_HTTP,
        availability=Availability.FORWARD_ONLY,
        endpoint="syncWeb3PriceJob / syncTrendingHistory",
        returns=ReturnSpec("pd.DataFrame", "one row per basket member"),
        pit_hazard="minute snapshot; membership itself rotates without an archive, "
                   "so a replay silently uses today's basket.",
    ),
]

_BY_NAME: Dict[str, DataNodeSpec] = {node.name: node for node in _NODES}


#: names present at import time; anything added later is user-registered
_BUILTIN_NAMES = frozenset(_BY_NAME)


def register_node(spec: DataNodeSpec, *, replace: bool = False) -> DataNodeSpec:
    """Add a node to the catalog at runtime (see :mod:`.custom_sources`).

    A built-in name can never be overwritten: shadowing ``klines`` with a
    private feed would silently change what every strategy reads.
    """
    if not isinstance(spec, DataNodeSpec):
        raise TypeError("register_node expects a DataNodeSpec")
    if spec.name in _BUILTIN_NAMES:
        raise ValueError(
            "%r is a built-in node and cannot be replaced; register under a "
            "different name" % spec.name
        )
    if spec.name in _BY_NAME and not replace:
        raise ValueError(
            "%r is already registered; pass replace=True to redefine it" % spec.name
        )
    if spec.name in _BY_NAME:
        _NODES[:] = [node for node in _NODES if node.name != spec.name]
    _NODES.append(spec)
    _BY_NAME[spec.name] = spec
    return spec


def unregister_node(name: str) -> bool:
    """Remove a user-registered node. Built-ins are not removable."""
    if name in _BUILTIN_NAMES:
        raise ValueError("%r is a built-in node and cannot be unregistered" % name)
    if name not in _BY_NAME:
        return False
    _NODES[:] = [node for node in _NODES if node.name != name]
    del _BY_NAME[name]
    return True


def typed_node_names() -> List[str]:
    """Nodes that declare a canonical input shape."""
    return sorted(node.name for node in _NODES if node.typed)


def is_custom_node(name: str) -> bool:
    return name in _BY_NAME and name not in _BUILTIN_NAMES


def get_node(name: str) -> DataNodeSpec:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(
            "unknown data node %r; known: %s" % (name, ", ".join(sorted(_BY_NAME)))
        )


def list_nodes() -> List[DataNodeSpec]:
    return list(_NODES)


def list_node_names() -> List[str]:
    return sorted(_BY_NAME)


def nodes_by_availability(availability: Availability) -> List[DataNodeSpec]:
    return [node for node in _NODES if node.availability is availability]


def backtestable_node_names() -> List[str]:
    return sorted(node.name for node in _NODES if node.backtestable)
