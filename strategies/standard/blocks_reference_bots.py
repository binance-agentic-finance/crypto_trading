"""Reference ``StandardBot`` pair — DataSnapshot in, cyqnt_trd.blocks compute, v2 signal out.

These two bots exist to prove the whole chain end to end, and to be the thing a
new author copies:

    DataSnapshot            one input object (market / universe / frames)
        │
        ▼  _frames_from_snapshot -> BotContext.frames
    ctx.frame("klines")     a plain OHLCV DataFrame
        │
        ▼  cyqnt_trd.blocks   indicators / conditions / universe — nothing hand-rolled
    StandardSignal          cyqnt.signal/v2

The pair is deliberately one TRADE bot and one SELECTION bot, because the point
being demonstrated is that **both emit the same 41-key shape**: selection fills
``candidates`` / ``universe_size`` and leaves the execution fields unset, trade
does the reverse. A consumer parses one schema.

The other five bots under ``strategies/standard/`` compute their factors inline;
these two show the blocks route, which is what keeps a strategy's arithmetic
identical across backtest, paper and live.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

import pandas as pd

from cyqnt_trd.blocks import conditions as C
from cyqnt_trd.blocks import indicators as I
from cyqnt_trd.blocks import universe as U
from cyqnt_trd.standard_bot.bot import BotContext, BotKind, BotSpec, DataRequest, StandardBot
from cyqnt_trd.standard_bot.core import (
    Direction,
    EntrySpec,
    EntryType,
    Evidence,
    ExitPlan,
    MarketScope,
    PositionIntent,
    Provenance,
    SelectionCandidate,
    SizeMode,
    SizeSpec,
    StandardSignal,
    StopSpec,
    TakeProfitLeg,
    TimeStop,
)

# --------------------------------------------------------------------------- #
# TRADE — EMA cross confirmed by ADX, ATR stop/target                          #
# --------------------------------------------------------------------------- #


class BlocksEmaCrossBot(StandardBot):
    """Long/short EMA cross with an ADX trend filter and an ATR-based exit plan."""

    spec = BotSpec(
        bot_id="blocks_ema_cross",
        kind=BotKind.TRADE,
        display_name="EMA cross (blocks reference)",
        description="EMA20/50 cross gated by ADX, ATR stop and target.",
        products=("usd_m_perpetual",),
        allowed_intents=(PositionIntent.OPEN_LONG, PositionIntent.OPEN_SHORT,
                         PositionIntent.HOLD),
    )

    DEFAULTS: Dict[str, Any] = {
        "symbol": "BTCUSDT", "interval": "4h",
        "ema_fast": 20, "ema_slow": 50, "adx_period": 14, "adx_min": 20.0,
        "atr_period": 14, "stop_mult": 2.0, "tp_mult": 3.0, "max_bars": 24,
        "risk_pct": 0.01,
    }

    def __init__(self, **config: Any) -> None:
        self.config = {**self.DEFAULTS, **config}

    def normalize_config(self, config: Any) -> Dict[str, Any]:
        merged = dict(self.config)
        merged.update(config or {})
        return merged

    def required_data(self) -> Sequence[DataRequest]:
        cfg = self.config
        return [DataRequest("klines", {"symbol": cfg["symbol"], "interval": cfg["interval"]})]

    def decide(self, ctx: BotContext) -> Iterable[StandardSignal]:
        cfg = ctx.config or self.config
        df = ctx.frame("klines")
        if df is None or len(df) < max(cfg["ema_slow"], cfg["adx_period"]) + 2:
            return []

        # --- every number below comes from cyqnt_trd.blocks ---
        close = df["close"]
        ema_fast = I.ema(close, cfg["ema_fast"])
        ema_slow = I.ema(close, cfg["ema_slow"])
        adx = I.adx(df, cfg["adx_period"])[0]
        atr = I.atr(df, cfg["atr_period"])

        cross_up = C.ma_cross_above(ema_fast, ema_slow)
        cross_dn = C.ma_cross_below(ema_fast, ema_slow)
        trending = C.adx_trending(adx, cfg["adx_min"])

        i = len(df) - 1
        go_long = bool(cross_up.iloc[i]) and bool(trending.iloc[i])
        go_short = bool(cross_dn.iloc[i]) and bool(trending.iloc[i])
        if not (go_long or go_short):
            return []

        atr_now = float(atr.iloc[i])
        if not atr_now or pd.isna(atr_now):
            return []
        price = float(close.iloc[i])

        return [StandardSignal(
            bot_id=self.spec.bot_id,
            bot_version=self.spec.version,
            decision_time=ctx.decision_time,
            symbol=cfg["symbol"],
            intent=PositionIntent.OPEN_LONG if go_long else PositionIntent.OPEN_SHORT,
            direction=Direction.LONG if go_long else Direction.SHORT,
            score=min(100.0, float(adx.iloc[i])),
            confidence=min(1.0, float(adx.iloc[i]) / 50.0),
            # A market entry still states the price it decided at: without a
            # reference price a RISK_PCT size cannot be turned into an equity
            # fraction, and the bridge declines rather than guessing.
            entry=EntrySpec(type=EntryType.MARKET, price=price),
            exit_plan=ExitPlan(
                stop_loss=StopSpec(atr_mult=cfg["stop_mult"], atr_value=atr_now),
                take_profit=(TakeProfitLeg(close_pct=1.0, atr_mult=cfg["tp_mult"],
                                           atr_value=atr_now),),
                time_stop=TimeStop(max_bars=cfg["max_bars"]),
                note="ATR stop/target sized off the entry bar's ATR.",
            ),
            size=SizeSpec(mode=SizeMode.RISK_PCT, value=cfg["risk_pct"]),
            summary="EMA%d/%d %s cross with ADX %.1f" % (
                cfg["ema_fast"], cfg["ema_slow"], "bullish" if go_long else "bearish",
                float(adx.iloc[i])),
            reason_codes=["ema_cross", "adx_trending"],
            evidence=(
                Evidence(source="ema_fast", observed={"value": round(float(ema_fast.iloc[i]), 2)}),
                Evidence(source="ema_slow", observed={"value": round(float(ema_slow.iloc[i]), 2)}),
                Evidence(source="adx", observed={"value": round(float(adx.iloc[i]), 2)}),
                Evidence(source="atr", observed={"value": round(atr_now, 2)}),
                Evidence(source="close", observed={"value": round(price, 2)}),
            ),
            data_quality=ctx.data_quality(required=("klines",)),
            provenance=Provenance(
                strategy_id=self.spec.bot_id,
                strategy_version=self.spec.version,
                snapshot_id=ctx.snapshot_id,
                config_hash="%s|%s" % (cfg["symbol"], cfg["interval"]),
            ),
        )]


# --------------------------------------------------------------------------- #
# SELECTION — rank the universe by Square mentions x sentiment                 #
# --------------------------------------------------------------------------- #


class BlocksNewsRankBot(StandardBot):
    """Rank a liquid universe by news mentions weighted by sentiment skew."""

    spec = BotSpec(
        bot_id="blocks_news_rank",
        kind=BotKind.SELECTION,
        display_name="News rank (blocks reference)",
        description="Liquidity filter, then mentions x bullish skew from Square.",
        products=("usd_m_perpetual",),
        # a selection bot is cross-sectional by definition; BotSpec defaults to
        # SINGLE, and leaving that unstated is how a basket ends up labelled as
        # a single-instrument signal. The contract now also corrects it, but the
        # declaration belongs here.
        market_scope=MarketScope.CROSS_SECTION,
    )

    DEFAULTS: Dict[str, Any] = {
        "min_quote_volume": 1e8, "min_mentions": 30, "top_k": 5, "long_ratio": 0.55,
    }

    def __init__(self, **config: Any) -> None:
        self.config = {**self.DEFAULTS, **config}

    def normalize_config(self, config: Any) -> Dict[str, Any]:
        merged = dict(self.config)
        merged.update(config or {})
        return merged

    def required_data(self) -> Sequence[DataRequest]:
        return [DataRequest("universe", {}), DataRequest("ticker_rank", {})]

    def decide(self, ctx: BotContext) -> Iterable[StandardSignal]:
        cfg = ctx.config or self.config
        universe_df = ctx.frame("universe")
        rank_df = ctx.frame("ticker_rank")
        if universe_df is None or not len(universe_df):
            return []

        # --- blocks again: liquidity filter + news join are library functions ---
        uni = U.filter_quote_volume(universe_df, cfg["min_quote_volume"])
        uni = U.augment_with_news(uni, rank_df)
        if "news_mention_count" not in uni.columns:
            return []
        uni = uni.dropna(subset=["news_mention_count"])
        uni = uni[uni["news_mention_count"] >= cfg["min_mentions"]]
        if not len(uni):
            return []

        # A name with mentions but no sentiment cannot be ranked by sentiment
        # skew. Keeping it and defaulting to neutral used to produce a candidate
        # scored 0 with direction SHORT — a short recommendation manufactured
        # from missing data. Drop them, and say how many.
        rated = uni[uni["news_bull_ratio"].notna()] if "news_bull_ratio" in uni.columns \
            else uni.iloc[0:0]
        dropped = len(uni) - len(rated)
        if not len(rated):
            return []

        uni = rated.sort_values("news_mention_count", ascending=False).head(cfg["top_k"])
        candidates = []
        for rank, (_, row) in enumerate(uni.iterrows(), start=1):
            bull = float(row["news_bull_ratio"])
            mentions = float(row["news_mention_count"])
            candidates.append(SelectionCandidate(
                symbol=str(row["symbol"]).upper(),
                rank=rank,
                score=round(mentions * abs(bull - 0.5) * 2.0, 4),
                direction=Direction.LONG if bull >= cfg["long_ratio"] else Direction.SHORT,
                reason="mentions %d, bullish ratio %.2f" % (int(mentions), bull),
                features={"news_mention_count": mentions, "news_bull_ratio": round(bull, 3)},
            ))

        return [StandardSignal(
            bot_id=self.spec.bot_id,
            bot_version=self.spec.version,
            decision_time=ctx.decision_time,
            market_scope=self.spec.market_scope,
            intent=PositionIntent.HOLD,
            direction=Direction.NEUTRAL,
            candidates=tuple(candidates),
            universe_size=int(len(universe_df)),
            # StandardSignal.score is contractually 0-100; the candidate score
            # is an unbounded mentions x skew product, so clamp rather than let
            # a busy news day raise ValueError deep inside the contract.
            score=min(100.0, float(candidates[0].score)) if candidates else 0.0,
            summary="Top %d by Square mentions x sentiment skew" % len(candidates),
            reason_codes=["news_mentions", "sentiment_skew"],
            warnings=(
                ("%d name(s) had mentions but no sentiment and were dropped"
                 % dropped,) if dropped else ()
            ),
            data_quality=ctx.data_quality(required=("universe",)),
            provenance=Provenance(
                strategy_id=self.spec.bot_id,
                strategy_version=self.spec.version,
                snapshot_id=ctx.snapshot_id,
            ),
        )]


__all__ = ["BlocksEmaCrossBot", "BlocksNewsRankBot"]
