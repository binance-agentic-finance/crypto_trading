"""Generate the sample input/output files in this folder from live code.

Run after changing the contracts so the doc cannot drift from the code:

    python docs/standard_bot_io/gen_samples.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parent / "samples"

import pandas as pd  # noqa: E402

from cyqnt_trd.standard_bot.bot import (  # noqa: E402
    BotContext, BotKind, BotSpec, DataRequest, StandardBot,
)
from cyqnt_trd.standard_bot.core import (  # noqa: E402
    SCHEMAS, EntrySpec, EntryType, Evidence, ExitPlan, FrameKind, PositionIntent,
    Provenance, SelectionCandidate, SizeMode, SizeSpec, StandardSignal, StopSpec,
    TakeProfitLeg, TimeHorizon, TimeStop, MarketScope, Direction, AdvisoryAction,
)
from cyqnt_trd.standard_bot.data.catalog import list_nodes  # noqa: E402

NOW = 1_786_000_000_000


def _write(name: str, payload) -> None:
    path = OUT / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")
    print("wrote %s" % path.relative_to(REPO))


# ---------------------------------------------------------------------------
# 1. one canonical sample row per input shape
# ---------------------------------------------------------------------------

SHAPE_ROWS = {
    FrameKind.BAR: {
        "instrument_id": "BTCUSDT", "timeframe": "1h",
        "open_time": "2026-08-06T06:00:00Z", "close_time": "2026-08-06T07:00:00Z",
        "available_time": "2026-08-06T07:00:00Z",
        "open": 61250.0, "high": 61480.0, "low": 61120.0, "close": 61390.0,
        "volume": 812.44, "quote_volume": 49_871_233.0, "trades": 21044,
        "confirmed": True,
    },
    FrameKind.METRIC: {
        "event_time": "2026-08-06T00:00:00Z", "available_time": "2026-08-06T00:00:12Z",
        "instrument_id": "BTCUSDT", "metric": "funding_rate_8h", "value": 0.00012,
        "venue": "binance", "product": "usd_m_perpetual",
        "unit": "ratio", "window": "8h", "source_id": "binance.funding_rate",
        "quality": "ok",
    },
    FrameKind.PANEL: {
        "event_time": "2026-08-05T00:00:00Z", "available_time": "2026-08-05T00:00:12Z",
        "metric": "funding_rate_daily",
        "BTCUSDT": 0.00036, "ETHUSDT": 0.00021, "SOLUSDT": -0.00014,
    },
    FrameKind.EVENT: {
        "event_id": "sq-90412", "event_time": "2026-08-06T06:41:00Z",
        "available_time": "2026-08-06T06:43:20Z",
        "source_id": "square.getNews", "topic": "exchange_announcement",
        "instrument_id": "ABCUSDT", "urgency": "breaking",
        "title": "Binance Will List ABC", "summary": "…",
        "url": "https://www.binance.com/en/support/announcement/…",
        "bias": "positive", "event_type": "listing",
        "source_reliability": 0.85, "corroboration_count": 2, "quality_flags": [],
    },
    FrameKind.RANK: {
        "available_time": "2026-08-06T07:00:00Z", "event_time": "2026-08-06T07:00:00Z",
        "instrument_id": "SOLUSDT", "rank": 1, "score": 0.91,
        "source_id": "square.getTickerRank",
        "mention_count": 812, "unique_authors": 431, "bullish_count": 590,
    },
    FrameKind.POSITION: {
        "available_time": "2026-08-06T07:00:03Z",
        "instrument_id": "BTCUSDT", "side": "short", "quantity": 0.42,
        "venue": "binance", "product": "usd_m_perpetual",
        "entry_price": 61810.0, "leverage": 5.0, "margin_ratio": 0.09,
        "unrealized_pnl": 176.4, "notional": 25_783.8,
    },
    FrameKind.BOOK: {
        "available_time": "2026-08-06T07:00:01Z",
        "instrument_id": "BTCUSDT", "side": "bid", "level": 1,
        "price": 61388.5, "quantity": 3.117,
    },
}


def gen_input_shapes() -> None:
    payload = {}
    for kind, schema in SCHEMAS.items():
        payload[schema.name] = {
            "kind": kind.value,
            "row_grain": schema.row_grain,
            "description": schema.description,
            "required": list(schema.required),
            "optional": list(schema.optional),
            "sample_row": SHAPE_ROWS[kind],
        }
    _write("input_shapes.json", payload)


# ---------------------------------------------------------------------------
# 2. a complete BotContext — what decide() actually receives
# ---------------------------------------------------------------------------

class _SampleBot(StandardBot):
    """Reads three shapes at once: bars, two metric sources, and positions."""

    spec = BotSpec(
        bot_id="sample_bot", kind=BotKind.TRADE, display_name="Sample",
        products=("usd_m_perpetual",), reads_positions=True,
    )

    def required_data(self):
        return [
            DataRequest("klines", {"symbol": "BTCUSDT", "interval": "1h", "limit": 3}),
            DataRequest("funding", {"symbol": "BTCUSDT", "limit": 3}),
            DataRequest("open_interest",
                        {"symbol": "BTCUSDT", "period": "1h", "limit": 2},
                        required=False),
            DataRequest("news", {"page_size": 1}, required=False),
            DataRequest("contract_positions", {}),
        ]

    def decide(self, ctx):
        return []


def _stub_sources() -> None:
    import cyqnt_trd.data_cli.account as account
    import cyqnt_trd.data_cli.funding as funding
    import cyqnt_trd.data_cli.kline as kline
    import cyqnt_trd.data_cli.news as news
    import cyqnt_trd.data_cli.oi as oi

    kline.fetch_klines = lambda **kw: pd.DataFrame({
        "open_time": [NOW - 7_200_000, NOW - 3_600_000, NOW - 3_600_000],
        "open": [61100.0, 61250.0, 61390.0], "high": [61300.0, 61480.0, 61520.0],
        "low": [61010.0, 61120.0, 61280.0], "close": [61250.0, 61390.0, 61460.0],
        "volume": [744.1, 812.4, 690.2],
        "quote_volume": [45_500_000.0, 49_871_233.0, 42_400_000.0],
        "close_time": [NOW - 3_600_000, NOW, NOW + 3_600_000],
        "trades": [19004, 21044, 17322],
    })
    funding.fetch_funding_history = lambda **kw: pd.DataFrame({
        "symbol": ["BTCUSDT"] * 3, "rate": [0.00009, 0.00011, 0.00012],
        "timestamp": [NOW - 57_600_000, NOW - 28_800_000, NOW],
        "mark_price": [61050.0, 61300.0, 61455.0],
    })
    oi.fetch_oi_history = lambda **kw: pd.DataFrame({
        "timestamp": [NOW - 3_600_000, NOW], "oi_base": [82_140.0, 84_902.0],
        "oi_value": [5_041_000_000.0, 5_216_000_000.0], "oi_change_bps": [0.0, 347.0],
    })
    news.fetch_news = lambda **kw: pd.DataFrame([{
        "id": "sq-90412", "title": "Binance Will List ABC", "summary": "…", "body": "…",
        "date": NOW - 140_000, "web_link": "https://binance.com/announcement/1",
        "tickers": ["ABC"], "author_role": "official", "content_type": 2,
        "tendency": 1, "author_name": "Binance",
    }])
    account.fetch_positions = lambda **kw: pd.DataFrame([{
        "symbol": "BTCUSDT", "side": "short", "quantity": 0.42,
        "entry_price": 61810.0, "leverage": 5.0, "unRealizedProfit": 176.4,
    }])


def _frame_entry(view) -> dict:
    """A ``TypedFrame`` as one ``frames[node]`` entry of ``cyqnt.input/v1``."""
    from cyqnt_trd.standard_bot.data.live_bundle import _to_epoch_ms
    from cyqnt_trd.standard_bot.data.input_bundle import _rows

    schema = view.schema
    entry = {
        "shape": schema.name if schema else "RawFrame@1.0",
        "status": view.status if view.status in ("ok", "empty", "error") else "ok",
        "availability": view.availability,
        "rows": _rows(_to_epoch_ms(view.frame.head(3), schema)),
    }
    if view.pit_hazard:
        entry["pit_hazard"] = view.pit_hazard
    if view.available_time_inferred:
        entry["available_time_inferred"] = True
    if not entry["rows"]:
        entry["status"] = "empty"
    return entry


def gen_context() -> BotContext:
    _stub_sources()
    bot = _SampleBot()
    ctx = bot.fetch_context(decision_time=NOW, equity=100_000.0)

    payload = {
        "schema": "cyqnt.input/v1",
        "bot_id": bot.plugin_id,
        "decision_time": ctx.decision_time,
        "snapshot_id": ctx.snapshot_id,
        "equity": ctx.equity,
        "positions": ctx.positions,
        "source_status": ctx.source_status,
        "warnings": ctx.warnings,
        "inferred_availability": ctx.inferred_availability,
        "declared_inputs": bot.typed_inputs(),
        # ONE shape for cyqnt.input/v1: ``frames[node] = {shape, rows, …}`` with
        # integer epoch-ms times, the same as an input bundle. This sample used to
        # be a third dialect — rows nested under a differently-keyed wrapper, and
        # ISO strings where the contract types integers — so an integrator
        # validating against cyqnt.input/v1 got a failure from the one artifact
        # that was supposed to be the reference.
        "frames": {
            key: _frame_entry(view)
            for key, view in ctx.typed.items()
        },
    }
    _write("bot_context.json", payload)
    return ctx


# ---------------------------------------------------------------------------
# 3. one sample output per signal kind
# ---------------------------------------------------------------------------


def gen_outputs() -> None:
    provenance = Provenance(
        strategy_id="sample_bot", strategy_version="v1",
        snapshot_id="0d2f…", inputs=("klines", "funding", "contract_positions"),
    )

    entry = StandardSignal(
        bot_id="sample_bot", decision_time=NOW, provenance=provenance,
        symbol="BTCUSDT", product="usd_m_perpetual",
        base_asset="BTC", quote_asset="USDT",
        intent=PositionIntent.OPEN_LONG, score=72.0, confidence=0.71,
        entry=EntrySpec(type=EntryType.LIMIT, price=61_300.0,
                        zone=(61_200.0, 61_400.0), time_in_force="GTC"),
        exit_plan=ExitPlan(
            stop_loss=StopSpec(atr_mult=2.0, atr_value=480.0, trailing=True,
                               exchange_managed=False),
            take_profit=(TakeProfitLeg(close_pct=0.5, pct=0.03),
                         TakeProfitLeg(close_pct=0.5, pct=0.06)),
            time_stop=TimeStop(max_bars=96),
            note="ATR trailing stop is evaluated in-process; place a venue STOP "
                 "if the position must survive the daemon",
        ),
        size=SizeSpec(mode=SizeMode.RISK_PCT, value=0.01, leverage=5,
                      max_notional_quote=2000.0),
        time_horizon=TimeHorizon.INTRADAY, horizon_seconds=14_400,
        topic="mtf_trend", reason_codes=("HTF_TREND_UP", "MACD_GOLDEN_CROSS",
                                          "ADX_TRENDING"),
        summary="BTCUSDT 顺势做多：4h 趋势向上 + 1h MACD 金叉 + ADX 27",
        recommended_behavior="限价区间挂单；ATR 止损随价上移。",
        evidence=(Evidence(source="data.klines",
                           observed={"adx": 27.4, "macd_hist": 41.2},
                           available_time=NOW),),
        source_status={"klines": "ok", "funding": "ok", "contract_positions": "ok"},
    )

    close = StandardSignal(
        bot_id="sample_bot", decision_time=NOW, provenance=provenance,
        symbol="BTCUSDT", intent=PositionIntent.CLOSE_SHORT,
        score=70.0, confidence=0.8,
        size=SizeSpec(mode=SizeMode.POSITION_PCT, value=1.0),
        topic="funding_carry", reason_codes=("FUNDING_REGIME_UNSTABLE",),
        summary="BTCUSDT 平掉套息空腿：资金费率符号翻转率 38%",
        recommended_behavior="同步解除现货对冲腿，避免留下裸多。",
        source_status={"funding": "ok", "contract_positions": "ok"},
    )

    selection = StandardSignal(
        bot_id="funding_crowding_neutral", decision_time=NOW, provenance=provenance,
        symbol=None, market_scope=MarketScope.CROSS_SECTION,
        intent=PositionIntent.HOLD, score=60.0, confidence=0.7,
        universe_size=10,
        candidates=(
            SelectionCandidate(symbol="DOTUSDT", rank=1, score=0.0011,
                               direction=Direction.LONG,
                               reason="funding rank 1/10",
                               features={"crowding_score": 0.0011}),
            SelectionCandidate(symbol="BTCUSDT", rank=10, score=-0.0009,
                               direction=Direction.SHORT,
                               reason="funding rank 10/10",
                               features={"crowding_score": -0.0009}),
        ),
        topic="funding_crowding", reason_codes=("BASKET_SNAPSHOT",),
        summary="资金费率拥挤度截面：2 多 / 2 空 / 6 观察",
    )

    advisory = StandardSignal(
        bot_id="funding_oi_crowding_monitor", decision_time=NOW, provenance=provenance,
        symbol="BTCUSDT", intent=PositionIntent.HOLD,
        direction=Direction.SHORT, advisory_action=AdvisoryAction.AVOID,
        score=90.0, confidence=0.8,
        topic="derivatives_positioning",
        reason_codes=("FUNDING_CROWDED_LONG", "OI_BUILDUP_CONFIRMS", "SQUEEZE_RISK"),
        summary="BTCUSDT 拥挤多：资金费率 20.0 bps，OI 变化 +35.0%",
        recommended_behavior="拥挤 + 杠杆仍在堆积，优先规避而非反向追单。",
        evidence=(
            Evidence(source="data.funding", observed={"funding_bps": 20.0},
                     available_time=NOW),
            Evidence(source="data.open_interest", observed={"change_pct": 35.0},
                     available_time=NOW),
        ),
    )

    _write("output_open_long.json", entry.to_dict())
    _write("output_open_long.execution_request.json", entry.to_execution_request())
    _write("output_close_short.json", close.to_dict())
    _write("output_close_short.execution_request.json", close.to_execution_request())
    _write("output_selection.json", selection.to_dict())
    _write("output_advisory.json", advisory.to_dict())


# ---------------------------------------------------------------------------
# 4. the full node -> shape table (fed into the README)
# ---------------------------------------------------------------------------


def gen_node_table() -> None:
    rows = [
        {
            "node": node.name,
            "emits": node.emits.value,
            "schema": node.input_schema.name if node.input_schema else None,
            "access": node.source_path.value,
            "availability": node.availability.value,
            "wired": bool(node.fetcher),
            "strategy_types": list(node.strategy_types),
            "pit_hazard": node.pit_hazard,
        }
        for node in sorted(list_nodes(), key=lambda n: (n.emits.value, n.name))
    ]
    _write("node_shape_table.json", rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    gen_input_shapes()
    gen_context()
    gen_outputs()
    gen_node_table()
