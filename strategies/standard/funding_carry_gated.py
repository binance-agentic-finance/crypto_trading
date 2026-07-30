"""A004 — funding carry, stability-gated.

Ported from ``quantative-selection-market/strategies/A004_funding_momentum_gate.py``
(+42-44% / 7yr, Sharpe ~3.8, maxDD −17% after 20bp hedge cost, in the local
backtests). Regime-dependent — trailing-1yr was ≈−11% in compressed funding —
which is exactly why the gate and the explicit exit matter.

Thesis
------
Collect funding on coins whose funding regime is **stable and positive**. The
danger in carry is a regime flip: funding that was fat and positive inverts, so
the book pays instead of collects right as the hedge slips. A004 harvests only
where recent funding is (a) positive on average and (b) consistent in sign.

Why this one is the good demo of the output contract
----------------------------------------------------
Carry is not "enter and hold". When a coin's funding regime destabilises the
correct instruction is **close the carry leg**, and that is a different thing
from "open the opposite". The original returns a weight vector, so a coin
dropping out just… gets weight 0 and the runner infers a close. Here it is
stated: ``CLOSE_LONG`` with ``reason_codes=("FUNDING_REGIME_UNSTABLE",)``.

Honesty note carried from the original: this per-symbol leg still has market
beta. Delta-neutrality comes from pairing it with the spot hedge, which is an
execution-layer construction, not something this signal claims.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from cyqnt_trd.standard_bot.bot import BotContext, BotKind, BotSpec, DataRequest, StandardBot
from cyqnt_trd.standard_bot.core import (
    ExitPlan,
    PositionIntent,
    Provenance,
    SizeMode,
    SizeSpec,
    StandardSignal,
    StopSpec,
    TimeHorizon,
)

BOT_ID = "funding_carry_gated"

DEFAULTS: Dict[str, Any] = {
    "universe": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
    "funding_prints": 400,
    "carry_window_8h": 21,        # ~7 days of settlements for expected carry
    "stability_window_8h": 42,    # window for the sign-consistency gate
    "max_flip_rate": 0.30,        # drop a coin whose funding flipped sign >30% of bars
    "min_carry_bps": 0.5,         # per-settlement expected funding floor
    "max_names": 3,
    "gross": 0.6,
    "stop_pct": 0.05,             # the leg is delta-hedged; this bounds an unhedged gap
    "horizon_seconds": 28_800,    # one funding period
}


class FundingCarryGated(StandardBot):
    spec = BotSpec(
        bot_id=BOT_ID,
        kind=BotKind.TRADE,
        display_name="资金费率套息（稳定性门控）",
        description="只在资金费率长期为正且符号稳定的币上收息；regime 不稳立即平腿。",
        products=("usd_m_perpetual",),
        # manages its own exposure: it closes legs it opened, so it must
        # observe them. Positions come from the caller or a PositionFrame input.
        reads_positions=True,
        default_horizon_seconds=DEFAULTS["horizon_seconds"],
    )

    def __init__(self, **config: Any) -> None:
        self.config = {**DEFAULTS, **config}

    def required_data(self) -> Sequence[DataRequest]:
        return [
            DataRequest(
                node="funding",
                params={"symbol": symbol, "limit": self.config["funding_prints"]},
                alias="funding_%s" % symbol,
                required=False,
            )
            for symbol in self.config["universe"]
        ]

    # ---- decision ----

    def decide(self, ctx: BotContext) -> Iterable[StandardSignal]:
        gates = {
            symbol: self._gate(ctx, symbol) for symbol in self.config["universe"]
        }
        passing = {s: g for s, g in gates.items() if g["passes"]}
        selected = sorted(
            passing, key=lambda s: passing[s]["expected_bps"], reverse=True
        )[: int(self.config["max_names"])]
        weight = (
            round(float(self.config["gross"]) / len(selected), 6) if selected else 0.0
        )

        for symbol in self.config["universe"]:
            gate = gates[symbol]
            holding = ctx.side_of(symbol) == "short"   # carry leg is short-perp
            wanted = symbol in selected

            if holding and not wanted:
                yield self._close(ctx, symbol, gate)
            elif wanted and not holding:
                yield self._open(ctx, symbol, gate, weight)
            # holding and wanted -> nothing to say; a HOLD per symbol would be noise

    # ---- signals ----

    def _open(self, ctx: BotContext, symbol: str, gate: Dict[str, Any], weight: float):
        return StandardSignal(
            bot_id=BOT_ID,
            decision_time=ctx.decision_time,
            provenance=Provenance(strategy_id=BOT_ID),
            symbol=symbol,
            # short the perp to receive positive funding
            intent=PositionIntent.OPEN_SHORT,
            score=float(np.clip(40.0 + gate["expected_bps"] * 4.0, 0.0, 100.0)),
            confidence=round(float(np.clip(1.0 - gate["flip_rate"], 0.3, 0.9)), 2),
            exit_plan=ExitPlan(
                stop_loss=StopSpec(pct=self.config["stop_pct"]),
                exit_on_opposite_signal=True,
                note="carry leg is meant to be delta-hedged with spot; the stop only "
                     "bounds the unhedged window",
            ),
            size=SizeSpec(mode=SizeMode.EQUITY_PCT, value=weight),
            time_horizon=TimeHorizon.SWING,
            horizon_seconds=self.config["horizon_seconds"],
            topic="funding_carry",
            reason_codes=("FUNDING_REGIME_STABLE_POSITIVE",),
            summary="%s 建立套息空腿：预期 %.2f bps/8h，符号翻转率 %.0f%%" % (
                symbol, gate["expected_bps"], gate["flip_rate"] * 100.0,
            ),
            recommended_behavior="配对现货多头做 delta 对冲；未对冲前该腿带方向风险。",
            data_quality=ctx.data_quality(),
        )

    def _close(self, ctx: BotContext, symbol: str, gate: Dict[str, Any]):
        reasons = tuple(gate["fail_reasons"]) or ("NOT_TOP_CARRY",)
        return StandardSignal(
            bot_id=BOT_ID,
            decision_time=ctx.decision_time,
            provenance=Provenance(strategy_id=BOT_ID),
            symbol=symbol,
            # close the SHORT carry leg — buy back, reduce-only, no short capability
            intent=PositionIntent.CLOSE_SHORT,
            score=70.0,
            confidence=0.8,
            size=SizeSpec(mode=SizeMode.POSITION_PCT, value=1.0),
            time_horizon=TimeHorizon.SWING,
            horizon_seconds=self.config["horizon_seconds"],
            topic="funding_carry",
            reason_codes=reasons,
            summary="%s 平掉套息空腿：%s（预期 %.2f bps，翻转率 %.0f%%）" % (
                symbol, ", ".join(reasons), gate["expected_bps"], gate["flip_rate"] * 100.0,
            ),
            recommended_behavior="同步解除现货对冲腿，避免留下裸多。",
            data_quality=ctx.data_quality(),
        )

    # ---- gate ----

    def _gate(self, ctx: BotContext, symbol: str) -> Dict[str, Any]:
        frame = ctx.frame("funding_%s" % symbol)
        fail: List[str] = []
        if frame is None or getattr(frame, "empty", True) or "rate" not in frame:
            return {
                "passes": False, "expected_bps": 0.0, "flip_rate": 1.0,
                "fail_reasons": ["FUNDING_UNAVAILABLE"],
            }

        rates = pd.to_numeric(frame["rate"], errors="coerce").dropna()
        if len(rates) < int(self.config["stability_window_8h"]):
            return {
                "passes": False, "expected_bps": 0.0, "flip_rate": 1.0,
                "fail_reasons": ["FUNDING_HISTORY_TOO_SHORT"],
            }

        expected_bps = float(rates.tail(int(self.config["carry_window_8h"])).mean()) * 10_000.0
        window = rates.tail(int(self.config["stability_window_8h"]))
        dominant = np.sign(window.mean())
        flip_rate = float((np.sign(window) != dominant).mean())

        if expected_bps < float(self.config["min_carry_bps"]):
            fail.append("CARRY_BELOW_FLOOR")
        if flip_rate > float(self.config["max_flip_rate"]):
            fail.append("FUNDING_REGIME_UNSTABLE")
        return {
            "passes": not fail,
            "expected_bps": round(expected_bps, 4),
            "flip_rate": round(flip_rate, 4),
            "fail_reasons": fail,
        }


# 导入即注册，和 blocks.strategy.register() 的习惯一致
from cyqnt_trd.standard_bot.registry import register_bot  # noqa: E402

register_bot(FundingCarryGated)
