"""Base class for pure advisory bot plugins."""

from __future__ import annotations

import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from datetime import timedelta
from typing import Any, Dict, Iterable, Mapping, Optional

from ..core import (
    DataSnapshot,
    DataQuery,
    FrameQuery,
    SignalBatch,
    SignalContext,
    SignalEnvelope,
    SignalKind,
    SignalProvenance,
    SourceStatus,
    TradeSide,
)
from ..signal.interfaces import SignalState, StepSignalResult
from .contracts import AdvisoryDecision, BotMeta, Evidence, epoch_ms, utc_datetime


_SIGNAL_NAMESPACE = uuid.UUID("6a94b657-010e-5e65-b0f1-e1baa82481bb")


class AdvisoryBot(ABC):
    """Pure input -> output plugin. Fetching, notification and orders stay outside."""

    meta: BotMeta

    @property
    def plugin_id(self) -> str:
        return self.meta.bot_id

    @property
    def plugin_version(self) -> str:
        return self.meta.version

    def required_inputs(self) -> Dict[str, bool]:
        """Standard ``SignalPlugin`` surface: bundle/frame name -> required.

        Same shape every other plugin in the registry returns
        (``BlockStrategyPlugin`` / ``SelectionStrategyPlugin``), so an advisory
        bot is a drop-in ``SignalPlugin``. Advisory bots take none of the
        OHLCV/social/on-chain bundles — their inputs are the named frames — so
        the bundle flags are all ``False`` and one entry is added per declared
        frame. The typed version is :meth:`required_data_query`.
        """
        inputs: Dict[str, bool] = {
            "market": False, "social": False, "onchain": False,
            "derivatives": False, "universe": False,
        }
        inputs.update({item.name: True for item in self.meta.required_frames})
        inputs.update({item.name: False for item in self.meta.optional_frames})
        return inputs

    def required_data_query(self, config: Any = None) -> DataQuery:
        """Typed declaration of the frames this bot needs, for the assembler."""
        del config
        field_keys = [item.field_key for item in self.meta.data_requirements]
        frames = {
            item.name: FrameQuery(
                schema=item.schema,
                required=True,
                field_keys=field_keys,
            )
            for item in self.meta.required_frames
        }
        frames.update(
            {
                item.name: FrameQuery(
                    schema=item.schema,
                    required=False,
                    field_keys=field_keys,
                )
                for item in self.meta.optional_frames
            }
        )
        return DataQuery(frames=frames)

    def initialize_state(self) -> SignalState:
        return SignalState(plugin_id=self.plugin_id, values={"seen_signal_ids": []})

    def run(
        self,
        snapshot: DataSnapshot,
        config: Any = None,
        context: Optional[SignalContext] = None,
    ) -> SignalBatch:
        del context
        decision_time = snapshot.meta.decision_as_of or snapshot.meta.assembled_at
        unavailable = []
        for contract in self.meta.required_frames:
            frame = snapshot.require_frame(contract.name)
            contract.validate(frame, decision_time)
            if frame.empty:
                status = snapshot.meta.source_status.get(contract.name, SourceStatus.ERROR)
                status_value = status.value if isinstance(status, SourceStatus) else str(status)
                if status_value != SourceStatus.OK.value:
                    unavailable.append((contract.name, status_value))
        for contract in self.meta.optional_frames:
            if contract.name in snapshot.frames:
                contract.validate(snapshot.frames[contract.name], decision_time)

        normalized_config = self.normalize_config(config)
        if unavailable:
            decisions = [
                self._data_unavailable_decision(snapshot, name, status, decision_time)
                for name, status in unavailable
            ]
        else:
            decisions = list(self.evaluate(snapshot, normalized_config))
        config_hash = self._config_hash(normalized_config)
        signals = [
            self._to_signal(item, decision_time=decision_time, config_hash=config_hash)
            for item in decisions
        ]
        batch_seed = "%s|%s|%s|%s" % (
            self.plugin_id,
            snapshot.meta.snapshot_id,
            decision_time,
            ",".join(signal.signal_id for signal in signals),
        )
        return SignalBatch(
            signals=signals,
            batch_id=str(uuid.uuid5(_SIGNAL_NAMESPACE, batch_seed)),
            created_at=epoch_ms(decision_time),
        )

    def step(
        self,
        snapshot: DataSnapshot,
        state: SignalState,
        config: Any = None,
        context: Optional[SignalContext] = None,
    ) -> StepSignalResult:
        batch = self.run(snapshot, config, context)
        seen_ordered = list(state.values.get("seen_signal_ids", []))
        seen = set(seen_ordered)
        new_signals = [signal for signal in batch.signals if signal.signal_id not in seen]
        seen_ordered.extend(signal.signal_id for signal in new_signals)
        next_state = SignalState(
            plugin_id=self.plugin_id,
            values={
                "last_snapshot_id": snapshot.meta.snapshot_id,
                "seen_signal_ids": seen_ordered[-5000:],
            },
        )
        return StepSignalResult(state=next_state, signals=new_signals)

    def normalize_config(self, config: Any) -> Any:
        return {} if config is None else config

    @abstractmethod
    def evaluate(self, snapshot: DataSnapshot, config: Any) -> Iterable[AdvisoryDecision]:
        raise NotImplementedError

    def _data_unavailable_decision(
        self,
        snapshot: DataSnapshot,
        frame_name: str,
        status: str,
        decision_time: Any,
    ) -> AdvisoryDecision:
        decision_dt = utc_datetime(decision_time)
        warning = "; ".join(snapshot.meta.warnings) or "required frame is empty"
        return AdvisoryDecision(
            symbol=None,
            venue="*",
            product="mixed",
            direction="neutral",
            action="investigate",
            score=100.0,
            confidence=1.0,
            horizon_seconds=300,
            valid_until=decision_dt + timedelta(seconds=300),
            topic="data_quality",
            urgency="breaking",
            reason_codes=("DATA_UNAVAILABLE", status.upper()),
            summary="%s 数据不可用：%s" % (frame_name, warning),
            recommended_behavior="检查数据源并使用 last-good；不得把缺失值解释为零或中性。",
            evidence=(
                Evidence(
                    source="snapshot.source_status",
                    source_id=frame_name,
                    available_time=decision_dt,
                    observed={"status": status, "warnings": list(snapshot.meta.warnings)},
                ),
            ),
            data_quality="insufficient",
            signal_family="data_quality",
            market_scope="global",
            dedup_key="data:%s:%s:%s" % (
                frame_name,
                status,
                epoch_ms(decision_time) // 300_000,
            ),
        )

    def to_canvas_dsl(self, config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Export the code bot as the existing Canvas ``version/nodes/edges`` shell."""
        cfg = dict(config or {})
        input_ports = {
            item.name: item.port(required=True) for item in self.meta.required_frames
        }
        input_ports.update(
            {item.name: item.port(required=False) for item in self.meta.optional_frames}
        )
        output_port = {
            "kind": "dataframe",
            "schema": "BotSignalFrame@1.0",
            "required": True,
            "scope": "multi_asset",
            "rowCardinality": "many",
        }
        fields = []
        for key, value in cfg.items():
            field = {
                "key": key,
                "label": key,
                "widget": "number" if isinstance(value, (int, float)) and not isinstance(value, bool) else "text",
                "value": value,
                "required": False,
                "group": "Bot 参数",
            }
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if any(token in key for token in ("reliability", "importance", "bull_ratio")):
                    field.update({"min": 0, "max": 1, "step": 0.05})
                else:
                    field.update(
                        {"min": 1 if isinstance(value, int) else 0.000001,
                         "step": 1 if isinstance(value, int) else 0.1}
                    )
            fields.append(field)
        pure_contract = {
            "scopes": ["single", "multi_asset"],
            "minInstruments": 0,
            "products": list(self.meta.supported_products),
            "exposureContracts": [],
            "deterministic": True,
            "sideEffect": False,
            "eligibility": {"advisory": True, "paper": False, "liveTrading": False},
            "stateful": False,
            "stateSchema": None,
            "idempotencyKey": None,
        }
        stateful_contract = dict(pure_contract)
        stateful_contract.update(
            {
                "stateful": True,
                "stateSchema": "SignalState@1.0",
                "idempotencyKey": "snapshot_id+config_hash",
            }
        )
        return {
            "$schema": "https://example.internal/trading/strategy-workflow-v1.1.schema.json",
            "version": "1.1",
            "metadata": {
                "id": self.meta.bot_id,
                "name": self.meta.display_name,
                "revision": 1,
                "locale": "zh-CN",
                "description": "%s (code %s)" % (self.meta.description, self.meta.version),
            },
            "bot": {
                "class": self.meta.bot_class,
                "archetype": "multi_asset",
                "product": "mixed",
                "universe": {
                    "mode": "dynamic",
                    "resolver": "external_input",
                    "maxInstruments": 100,
                },
                "exposure": None,
                "allowedStages": ["research", "advisory", "shadow"],
                "schedule": {
                    "mode": "event_driven",
                    "interval": None,
                    "timezone": "UTC",
                    "maxTriggerLagSeconds": 300,
                },
            },
            "interface": {
                "inputs": input_ports,
                "outputs": {"alerts": output_port},
                "requiredSinks": ["evaluate.alerts"],
            },
            "nodes": [
                {
                    "id": "normalize",
                    "type": "script",
                    "name": "数据标准化",
                    "sub": "输出标准 DataFrame",
                    "emoji": "🧱",
                    "config": {"pitRequired": True},
                    "fields": [],
                    "block": {"name": "dataframe_normalizer", "layer": "normalize", "version": "1.0.0"},
                    "io": {"inputs": input_ports, "outputs": input_ports},
                    "contract": pure_contract,
                    "failurePolicy": "degrade_and_alert",
                },
                {
                    "id": "evaluate",
                    "type": "script",
                    "name": self.meta.display_name,
                    "sub": self.meta.description,
                    "emoji": "🤖",
                    "config": cfg,
                    "fields": fields,
                    "block": {"name": self.meta.bot_id, "layer": "signal", "version": self.meta.version},
                    "io": {"inputs": input_ports, "outputs": {"alerts": output_port}},
                    "contract": stateful_contract,
                    "failurePolicy": "degrade_and_alert",
                },
            ],
            "edges": [
                {
                    "id": "normalize_to_evaluate_%s" % port_name,
                    "source": "normalize",
                    "target": "evaluate",
                    "sourcePort": port_name,
                    "targetPort": port_name,
                    "type": "serial",
                }
                for port_name in input_ports
            ],
        }

    def _to_signal(
        self,
        decision: AdvisoryDecision,
        *,
        decision_time: Any,
        config_hash: str,
    ) -> SignalEnvelope:
        seed = "%s|%s|%s|%s|%s" % (
            self.plugin_id,
            decision.symbol or "GLOBAL",
            decision.product,
            decision.topic,
            decision.dedup_key or "%s|%s" % (decision_time, decision.summary),
        )
        direction_to_side = {
            "long": TradeSide.BUY,
            "short": TradeSide.SELL,
            "neutral": TradeSide.FLAT,
        }
        instrument_id = None
        if decision.symbol:
            instrument_id = "%s:%s:%s" % (
                decision.venue,
                decision.product,
                decision.symbol,
            )
        payload = {
            "decision_time": utc_datetime(decision_time).isoformat(),
            "symbol": decision.symbol,
            "venue": decision.venue,
            "product": decision.product,
            "direction": decision.direction,
            "action": decision.action,
            "score": float(decision.score),
            "confidence": float(decision.confidence),
            "horizon_seconds": int(decision.horizon_seconds),
            "valid_until": utc_datetime(decision.valid_until).isoformat(),
            "topic": decision.topic,
            "signal_family": decision.signal_family,
            "urgency": decision.urgency,
            "reason_codes": list(decision.reason_codes),
            "summary": decision.summary,
            "recommended_behavior": decision.recommended_behavior,
            "evidence": [item.to_dict() for item in decision.evidence],
            "data_quality": decision.data_quality,
            "market_scope": decision.market_scope,
            "auto_trade_eligible": False,
            "dedup_key": decision.dedup_key,
        }
        return SignalEnvelope(
            version="1.0",
            signal_id=str(uuid.uuid5(_SIGNAL_NAMESPACE, seed)),
            kind=SignalKind.ALERT,
            strength=float(decision.score) / 100.0,
            provenance=SignalProvenance(
                plugin_id=self.plugin_id,
                plugin_version=self.plugin_version,
                config_hash=config_hash,
            ),
            instrument_id=instrument_id,
            side=direction_to_side[decision.direction],
            time_horizon="%ss" % decision.horizon_seconds,
            valid_until=epoch_ms(decision.valid_until),
            payload=payload,
        )

    @staticmethod
    def _config_hash(config: Any) -> str:
        if is_dataclass(config):
            config = asdict(config)
        encoded = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
