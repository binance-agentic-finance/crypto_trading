"""
Interfaces for the standard bot signal layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from ..core import DataSnapshot, SignalBatch, SignalContext


@dataclass
class SignalState:
    """
    Serializable state container for incremental signal evaluation.
    """

    plugin_id: str
    values: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepSignalResult:
    """
    Result of incremental signal execution for one plugin.
    """

    state: Optional[SignalState] = None
    signals: List = field(default_factory=list)


class SignalPlugin(Protocol):
    plugin_id: str
    plugin_version: str

    def required_inputs(self) -> Dict[str, bool]:
        ...

    def run(
        self,
        snapshot: DataSnapshot,
        config: Optional[Any] = None,
        context: Optional[SignalContext] = None,
    ) -> SignalBatch:
        ...


class BatchSignalPlugin(Protocol):
    def run_batch(
        self,
        snapshots: Sequence[DataSnapshot],
        context: Optional[SignalContext] = None,
    ) -> List[SignalBatch]:
        ...


class IncrementalSignalPlugin(Protocol):
    def initialize_state(self) -> SignalState:
        ...

    def step(
        self,
        snapshot: DataSnapshot,
        state: SignalState,
        config: Optional[Any] = None,
        context: Optional[SignalContext] = None,
    ) -> StepSignalResult:
        ...


class SignalComposer:
    """
    Minimal composer used to combine multiple plugin outputs into one batch.
    """

    def combine(self, batches: Sequence[SignalBatch], batch_id: str, created_at: int) -> SignalBatch:
        signals = []
        for batch in batches:
            signals.extend(batch.signals)
        return SignalBatch(signals=signals, batch_id=batch_id, created_at=created_at)
