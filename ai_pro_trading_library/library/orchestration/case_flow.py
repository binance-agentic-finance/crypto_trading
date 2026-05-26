"""Orchestration helpers that do not execute trades."""

from __future__ import annotations

from pathlib import Path

from ai_pro_trading_library.library.core.schema import RuntimeMode, RuntimeRequest, StrategySpec


def build_runtime_request(
    strategy: StrategySpec,
    mode: RuntimeMode | str,
    output_dir: str | Path | None = None,
    data_uri: str | None = None,
) -> RuntimeRequest:
    strategy.require_symbols()
    return RuntimeRequest(
        mode=RuntimeMode(mode),
        strategy=strategy,
        output_dir=Path(output_dir) if output_dir else None,
        data_uri=data_uri,
        dry_run=RuntimeMode(mode) != RuntimeMode.LIVE,
    )

