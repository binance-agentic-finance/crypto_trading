"""Reference multi-source bot — price + funding + open interest + news, one decision.

This is the bot the input bundle exists for. It reads four sources that used to
live in four different file formats (OHLCV JSON, derivatives parquet, Square live
API, internal BigData HTTP) and are now four frames in one
``cyqnt.input/v1`` bundle, all gated to the same ``decision_time``:

    ctx.frame("klines")         BarFrame     — trend and volatility
    ctx.frame("funding")        MetricFrame  — crowding / carry
    ctx.frame("open_interest")  MetricFrame  — is new money arriving
    ctx.frame("news")           EventFrame   — is there a catalyst

The point being demonstrated is not the alpha — it is that **three of these are
read with the same three lines of code**, because funding, open interest and any
internal metric all arrive as ``MetricFrame`` rows
(``instrument_id`` / ``metric`` / ``value`` / ``event_time`` / ``available_time``).
Adding a fifth metric source needs no new plumbing here at all.

Logic, deliberately simple and stated:

* direction comes from price (EMA trend, via ``cyqnt_trd.blocks``);
* funding **vetoes** a trade in the crowded direction — paying to hold a
  consensus position is the opposite of an edge;
* rising open interest **confirms** (new money), falling OI downgrades it to a
  squeeze and halves the score;
* a fresh high-urgency news event on this instrument adds confidence.

Every input that is missing degrades the signal rather than crashing it, and the
degradation is visible in ``data_quality`` / ``reason_codes``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import pandas as pd

from cyqnt_trd.blocks import indicators as I
from cyqnt_trd.standard_bot.bot import (
    BotContext, BotKind, BotSpec, DataRequest, StandardBot)
from cyqnt_trd.standard_bot.core import (
    Direction, EntrySpec, EntryType, Evidence, ExitPlan, PositionIntent, Provenance,
    SizeMode, SizeSpec, StandardSignal, StopSpec, TakeProfitLeg, TimeStop)


def latest_metric(ctx: BotContext, node: str, metric: str,
                  instrument_id: Optional[str] = None) -> Optional[float]:
    """Newest value of one metric in a MetricFrame node, or None.

    Three sources, one accessor — that is the whole benefit of normalising
    funding / OI / internal metrics to the same shape.
    """
    frame = ctx.frame(node)
    if frame is None or not len(frame):
        return None
    rows = frame[frame["metric"] == metric]
    if instrument_id is not None and "instrument_id" in rows.columns:
        rows = rows[rows["instrument_id"] == instrument_id]
    if not len(rows):
        return None
    rows = rows.sort_values("available_time")
    return float(rows["value"].iloc[-1])


def metric_change(ctx: BotContext, node: str, metric: str, *, lookback: int) -> Optional[float]:
    """Fractional change of a metric over its last ``lookback`` observations."""
    frame = ctx.frame(node)
    if frame is None or not len(frame):
        return None
    rows = frame[frame["metric"] == metric].sort_values("available_time")
    if len(rows) <= lookback:
        return None
    now = float(rows["value"].iloc[-1])
    then = float(rows["value"].iloc[-1 - lookback])
    if not then:
        return None
    return (now - then) / abs(then)


class MultiSourceBot(StandardBot):
    """Price direction, funding veto, OI confirmation, news boost."""

    spec = BotSpec(
        bot_id="multi_source_reference",
        kind=BotKind.TRADE,
        display_name="Multi-source (price + funding + OI + news)",
        description="One decision from four normalised sources in one bundle.",
        products=("usd_m_perpetual",),
        allowed_intents=(PositionIntent.OPEN_LONG, PositionIntent.OPEN_SHORT,
                         PositionIntent.HOLD),
    )

    DEFAULTS: Dict[str, Any] = {
        "symbol": "BTCUSDT", "interval": "1h",
        "ema_fast": 20, "ema_slow": 60,
        "atr_period": 14, "stop_mult": 2.0, "tp_mult": 3.0, "max_bars": 48,
        # funding above this (long side) means the crowd is already paying to be long
        "funding_veto_bps": 3.0,
        "oi_lookback": 24, "oi_confirm_pct": 0.01,
        "news_fresh_seconds": 6 * 3600,
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
        return [
            DataRequest("klines", {"symbol": cfg["symbol"], "interval": cfg["interval"]}),
            DataRequest("funding", {"symbol": cfg["symbol"]}, required=False),
            DataRequest("open_interest", {"symbol": cfg["symbol"]}, required=False),
            DataRequest("news", {}, required=False),
        ]

    def decide(self, ctx: BotContext) -> Iterable[StandardSignal]:
        cfg = ctx.config or self.config
        symbol = cfg["symbol"]
        df = ctx.frame("klines")
        if df is None or len(df) < cfg["ema_slow"] + 2:
            return []

        # ---- 1. direction from price (blocks) ----
        close = df["close"]
        ema_fast = I.ema(close, cfg["ema_fast"])
        ema_slow = I.ema(close, cfg["ema_slow"])
        atr = I.atr(df, cfg["atr_period"])
        i = len(df) - 1
        if pd.isna(ema_slow.iloc[i]) or pd.isna(atr.iloc[i]):
            return []
        long_side = bool(ema_fast.iloc[i] > ema_slow.iloc[i])
        direction = Direction.LONG if long_side else Direction.SHORT

        reasons = ["ema_trend_up" if long_side else "ema_trend_down"]
        evidence = [
            Evidence(source="klines", observed={
                "ema_fast": round(float(ema_fast.iloc[i]), 2),
                "ema_slow": round(float(ema_slow.iloc[i]), 2),
                "close": round(float(close.iloc[i]), 2)}),
        ]
        score = 50.0

        # ---- 2. funding veto (MetricFrame) ----
        funding = latest_metric(ctx, "funding", "funding_rate", symbol)
        if funding is None:
            reasons.append("funding_missing")
        else:
            funding_bps = funding * 10_000.0
            evidence.append(Evidence(source="funding",
                                     observed={"funding_rate_bps": round(funding_bps, 4)}))
            crowded = (funding_bps > cfg["funding_veto_bps"]) if long_side \
                else (funding_bps < -cfg["funding_veto_bps"])
            if crowded:
                # Paying carry to hold the consensus side — stand down.
                return [self._hold(ctx, cfg, symbol, evidence,
                                   reasons + ["funding_crowded_veto"],
                                   "funding %.2fbps is against holding the %s side"
                                   % (funding_bps, "long" if long_side else "short"))]
            reasons.append("funding_ok")
            score += 10.0

        # ---- 3. open interest confirmation (same accessor, different node) ----
        oi_chg = metric_change(ctx, "open_interest", "open_interest",
                               lookback=cfg["oi_lookback"])
        if oi_chg is None:
            reasons.append("oi_missing")
        else:
            evidence.append(Evidence(source="open_interest",
                                     observed={"oi_change_pct": round(oi_chg * 100, 3)}))
            if oi_chg >= cfg["oi_confirm_pct"]:
                reasons.append("oi_rising_new_money")
                score += 25.0
            else:
                # Price moving on falling OI is positions closing, not new money.
                reasons.append("oi_falling_squeeze")
                score *= 0.5

        # ---- 4. news catalyst (EventFrame) ----
        news = ctx.frame("news")
        if news is not None and len(news):
            fresh = news[
                (news.get("instrument_id").isna() | (news["instrument_id"] == symbol))
                & (news["available_time"] >= ctx.decision_time
                   - cfg["news_fresh_seconds"] * 1000)
            ] if "instrument_id" in news.columns else news
            if len(fresh):
                top = fresh.sort_values("available_time").iloc[-1]
                reasons.append("news_catalyst")
                score += 15.0 if str(top.get("urgency", "")) == "high" else 5.0
                evidence.append(Evidence(source="news", observed={
                    "topic": str(top.get("topic", "")),
                    "title": str(top.get("title", ""))[:120]},
                    available_time=int(top["available_time"])))

        atr_now = float(atr.iloc[i])
        price = float(close.iloc[i])
        return [StandardSignal(
            bot_id=self.spec.bot_id, bot_version=self.spec.version,
            decision_time=ctx.decision_time, symbol=symbol,
            intent=PositionIntent.OPEN_LONG if long_side else PositionIntent.OPEN_SHORT,
            direction=direction,
            score=min(100.0, score), confidence=min(1.0, score / 100.0),
            entry=EntrySpec(type=EntryType.MARKET, price=price),
            exit_plan=ExitPlan(
                stop_loss=StopSpec(atr_mult=cfg["stop_mult"], atr_value=atr_now),
                take_profit=(TakeProfitLeg(close_pct=1.0, atr_mult=cfg["tp_mult"],
                                           atr_value=atr_now),),
                time_stop=TimeStop(max_bars=cfg["max_bars"]),
            ),
            size=SizeSpec(mode=SizeMode.RISK_PCT, value=cfg["risk_pct"]),
            summary="%s trend, %s" % ("Long" if long_side else "Short",
                                      ", ".join(reasons[1:]) or "price only"),
            reason_codes=reasons,
            evidence=tuple(evidence),
            data_quality=ctx.data_quality(required=("klines",)),
            provenance=Provenance(strategy_id=self.spec.bot_id,
                                  strategy_version=self.spec.version,
                                  snapshot_id=ctx.snapshot_id,
                                  config_hash="%s|%s" % (symbol, cfg["interval"])),
        )]

    def _hold(self, ctx, cfg, symbol, evidence, reasons, why) -> StandardSignal:
        return StandardSignal(
            bot_id=self.spec.bot_id, bot_version=self.spec.version,
            decision_time=ctx.decision_time, symbol=symbol,
            intent=PositionIntent.HOLD, direction=Direction.NEUTRAL,
            score=0.0, confidence=0.0, summary=why, reason_codes=reasons,
            evidence=tuple(evidence),
            data_quality=ctx.data_quality(required=("klines",)),
            provenance=Provenance(strategy_id=self.spec.bot_id,
                                  strategy_version=self.spec.version,
                                  snapshot_id=ctx.snapshot_id),
        )


__all__ = ["MultiSourceBot", "latest_metric", "metric_change"]
