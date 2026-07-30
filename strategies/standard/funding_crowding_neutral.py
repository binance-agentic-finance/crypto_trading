"""N003 — funding-crowding, dollar-neutral cross-section.

Ported from ``策略开发/50_N_neutral/strategies/N003_funding_crowding_neutral.py``,
the strongest result in that workspace (+84.7% / 2yr, Sharpe 1.83, maxDD −7.9%,
beta −0.035, positive in the bear half too; funding-rank IC 0.47).

Thesis
------
Perpetual funding is a direct crowding gauge. Very positive funding means longs
are paying shorts — the trade is crowded long and vulnerable to a squeeze lower.
Very negative funding means the reverse. So: **short the most-positive-funding
names, long the most-negative**, dollar-neutral. The PnL is the relative
de-crowding move between coins, which is why it holds up when the market falls.

What changes in the port
------------------------
The original returns a signed weight vector and hands it to a bespoke
long/short engine. Here it emits :class:`StandardSignal` per name, so the same
book states its instrument, its intent, and — the part the weight vector could
never express — **which side to close when a name leaves the basket**. A symbol
that was short last cycle and is no longer in the short leg gets an explicit
``CLOSE_SHORT``, not silence.

Data
----
``funding`` only, which is BACKTESTABLE with years of history. Nothing here
depends on a snapshot source, so the whole strategy is replayable.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from cyqnt_trd.standard_bot.bot import BotContext, BotKind, BotSpec, DataRequest, StandardBot
from cyqnt_trd.standard_bot.core import (
    Direction,
    ExitPlan,
    MarketScope,
    PositionIntent,
    Provenance,
    SelectionCandidate,
    SizeMode,
    SizeSpec,
    StandardSignal,
    TimeHorizon,
)

BOT_ID = "funding_crowding_neutral"

DEFAULTS: Dict[str, Any] = {
    "universe": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                 "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"],
    "funding_prints": 400,      # 8h settlements pulled per symbol
    "smooth": 3,                # trailing mean over daily funding — denoises the gauge
    "top_frac": 0.2,            # each leg takes this fraction of the cross-section
    "gross": 1.0,               # total gross exposure across both legs
    "min_names": 4,             # below this the cross-section is not a cross-section
    "horizon_seconds": 86_400,
}


class FundingCrowdingNeutral(StandardBot):
    spec = BotSpec(
        bot_id=BOT_ID,
        kind=BotKind.TRADE,
        display_name="资金费率拥挤度中性",
        description="做空资金费率最高（拥挤多）、做多最低（拥挤空）的币，美元中性。",
        products=("usd_m_perpetual",),
        # manages its own exposure: it closes legs it opened, so it must
        # observe them. Positions come from the caller or a PositionFrame input.
        reads_positions=True,
        market_scope=MarketScope.CROSS_SECTION,
        default_horizon_seconds=DEFAULTS["horizon_seconds"],
    )

    def __init__(self, **config: Any) -> None:
        self.config = {**DEFAULTS, **config}

    # ---- input ----

    def required_data(self) -> Sequence[DataRequest]:
        return [
            DataRequest(
                node="funding",
                params={"symbol": symbol, "limit": self.config["funding_prints"]},
                alias="funding_%s" % symbol,
                required=False,      # one missing name shrinks the basket, not the run
            )
            for symbol in self.config["universe"]
        ]

    # ---- decision ----

    def decide(self, ctx: BotContext) -> Iterable[StandardSignal]:
        panel = self._funding_panel(ctx)
        held = {
            symbol: ctx.side_of(symbol)
            for symbol in set(self.config["universe"]) | set(ctx.positions)
            if ctx.side_of(symbol) != "flat"
        }

        if panel.empty or panel.shape[1] < self.config["min_names"]:
            # Not enough of a cross-section to be neutral. Flatten rather than
            # run a half-basket that is really a directional bet.
            yield from self._flatten(ctx, held, reason="INSUFFICIENT_CROSS_SECTION")
            return

        scores = self._scores(panel)
        longs, shorts = self._legs(scores)
        target: Dict[str, str] = {**{s: "long" for s in longs}, **{s: "short" for s in shorts}}
        weight = self._leg_weight(len(longs) + len(shorts))

        # 1. close what is no longer wanted, or wanted on the other side
        for symbol, side in held.items():
            wanted = target.get(symbol)
            if wanted == side:
                continue
            if wanted is None:
                yield self._exit_signal(ctx, symbol, side, "LEFT_BASKET", scores)
            # a side flip is handled by the entry below emitting FLIP_TO_*

        # 2. open / flip into the basket
        for symbol, side in sorted(target.items()):
            current = held.get(symbol, "flat")
            if current == side:
                continue
            yield self._entry_signal(
                ctx, symbol=symbol, side=side, current=current,
                weight=weight, scores=scores, panel=panel,
            )

        # 3. one cross-section summary carrying the whole ranked basket
        yield self._basket_signal(ctx, scores, longs, shorts)

    # ---- signal construction ----

    def _entry_signal(
        self, ctx: BotContext, *, symbol: str, side: str, current: str,
        weight: float, scores: pd.Series, panel: pd.DataFrame,
    ) -> StandardSignal:
        if current == "flat":
            intent = PositionIntent.OPEN_LONG if side == "long" else PositionIntent.OPEN_SHORT
        else:
            intent = (
                PositionIntent.FLIP_TO_LONG if side == "long" else PositionIntent.FLIP_TO_SHORT
            )
        funding_bps = float(panel[symbol].iloc[-1]) * 10_000.0
        reason = (
            "FUNDING_CROWDED_SHORT" if side == "long" else "FUNDING_CROWDED_LONG"
        )
        return StandardSignal(
            bot_id=BOT_ID,
            decision_time=ctx.decision_time,
            provenance=Provenance(strategy_id=BOT_ID),
            symbol=symbol,
            intent=intent,
            score=self._score_to_0_100(scores, symbol),
            confidence=0.7,
            # The book is rebalanced every cycle; the exit is the next
            # rebalance, not a price level. Say that explicitly rather than
            # leave the position with no stated way out.
            exit_plan=ExitPlan(
                exit_on_opposite_signal=True,
                note="rebalanced each cycle; a name leaving the basket is closed by "
                     "the CLOSE_* signal, not by a price stop",
            ),
            size=SizeSpec(mode=SizeMode.EQUITY_PCT, value=weight),
            time_horizon=TimeHorizon.SWING,
            horizon_seconds=self.config["horizon_seconds"],
            topic="funding_crowding",
            reason_codes=(reason, "DOLLAR_NEUTRAL_BASKET"),
            summary="%s %s：资金费率 %.2f bps（%s）" % (
                symbol, "做多" if side == "long" else "做空", funding_bps, reason,
            ),
            recommended_behavior="按篮子等风险建仓；单腿不构成方向观点。",
            data_quality=ctx.data_quality(),
        )

    def _exit_signal(
        self, ctx: BotContext, symbol: str, side: str, reason: str, scores: pd.Series,
    ) -> StandardSignal:
        intent = (
            PositionIntent.CLOSE_LONG if side == "long" else PositionIntent.CLOSE_SHORT
        )
        return StandardSignal(
            bot_id=BOT_ID,
            decision_time=ctx.decision_time,
            provenance=Provenance(strategy_id=BOT_ID),
            symbol=symbol,
            intent=intent,
            score=50.0,
            confidence=0.8,
            size=SizeSpec(mode=SizeMode.POSITION_PCT, value=1.0),
            time_horizon=TimeHorizon.SWING,
            horizon_seconds=self.config["horizon_seconds"],
            topic="funding_crowding",
            reason_codes=(reason,),
            summary="%s 退出%s腿：%s" % (symbol, "多" if side == "long" else "空", reason),
            recommended_behavior="平掉该腿，保持篮子美元中性。",
            data_quality=ctx.data_quality(),
        )

    def _flatten(
        self, ctx: BotContext, held: Dict[str, str], *, reason: str,
    ) -> Iterable[StandardSignal]:
        for symbol, side in sorted(held.items()):
            yield self._exit_signal(ctx, symbol, side, reason, pd.Series(dtype=float))

    def _basket_signal(
        self, ctx: BotContext, scores: pd.Series, longs: List[str], shorts: List[str],
    ) -> StandardSignal:
        candidates: List[SelectionCandidate] = []
        ranked = scores.sort_values(ascending=False)
        for rank, (symbol, score) in enumerate(ranked.items(), start=1):
            side = (
                Direction.LONG if symbol in longs
                else Direction.SHORT if symbol in shorts
                else Direction.NEUTRAL
            )
            candidates.append(
                SelectionCandidate(
                    symbol=symbol, rank=rank, score=round(float(score), 6),
                    direction=side,
                    reason="funding rank %d/%d" % (rank, len(ranked)),
                    features={"crowding_score": round(float(score), 6)},
                )
            )
        return StandardSignal(
            bot_id=BOT_ID,
            decision_time=ctx.decision_time,
            provenance=Provenance(strategy_id=BOT_ID),
            symbol=None,
            intent=PositionIntent.HOLD,
            market_scope=MarketScope.CROSS_SECTION,
            candidates=tuple(candidates),
            universe_size=len(ranked),
            score=60.0,
            confidence=0.7,
            topic="funding_crowding",
            reason_codes=("BASKET_SNAPSHOT",),
            summary="资金费率拥挤度截面：%d 多 / %d 空 / %d 观察" % (
                len(longs), len(shorts), len(ranked) - len(longs) - len(shorts),
            ),
            data_quality=ctx.data_quality(),
            horizon_seconds=self.config["horizon_seconds"],
        )

    # ---- computation ----

    def _funding_panel(self, ctx: BotContext) -> pd.DataFrame:
        """time x symbol funding panel from the per-symbol frames."""
        series: Dict[str, pd.Series] = {}
        for symbol in self.config["universe"]:
            frame = ctx.frame("funding_%s" % symbol)
            if frame is None or getattr(frame, "empty", True):
                continue
            if "rate" not in frame or "timestamp" not in frame:
                continue
            column = pd.Series(
                pd.to_numeric(frame["rate"], errors="coerce").values,
                index=pd.to_datetime(
                    pd.to_numeric(frame["timestamp"], errors="coerce"),
                    unit="ms", utc=True, errors="coerce",
                ),
            ).dropna()
            if not column.empty:
                series[symbol] = column[~column.index.duplicated(keep="last")].sort_index()
        if not series:
            return pd.DataFrame()
        panel = pd.DataFrame(series).sort_index()
        # 8h prints -> daily sum, then a causal trailing mean
        daily = panel.resample("1D").sum(min_count=1)
        smooth = int(self.config["smooth"] or 1)
        if smooth > 1:
            daily = daily.rolling(smooth, min_periods=1).mean()
        return daily.dropna(how="all")

    @staticmethod
    def _scores(panel: pd.DataFrame) -> pd.Series:
        """Higher score = more negative funding = crowded short = long it."""
        latest = panel.iloc[-1]
        return (-1.0 * latest).dropna()

    def _legs(self, scores: pd.Series) -> Tuple[List[str], List[str]]:
        ranked = scores.sort_values(ascending=False)
        n = max(1, int(round(len(ranked) * float(self.config["top_frac"]))))
        n = min(n, len(ranked) // 2)          # the two legs must not overlap
        if n < 1:
            return [], []
        return list(ranked.index[:n]), list(ranked.index[-n:])

    def _leg_weight(self, positions: int) -> float:
        if positions <= 0:
            return 0.0
        return round(float(self.config["gross"]) / positions, 6)

    @staticmethod
    def _score_to_0_100(scores: pd.Series, symbol: str) -> float:
        if scores.empty or symbol not in scores or scores.std(ddof=0) == 0:
            return 50.0
        z = (scores[symbol] - scores.mean()) / scores.std(ddof=0)
        return float(np.clip(50.0 + 20.0 * abs(z), 0.0, 100.0))


# 导入即注册，和 blocks.strategy.register() 的习惯一致
from cyqnt_trd.standard_bot.registry import register_bot  # noqa: E402

register_bot(FundingCrowdingNeutral)
