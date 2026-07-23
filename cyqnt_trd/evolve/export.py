"""Export champion genomes as standalone strategy .py files.

The exported file:
    - imports cyqnt_trd.evolve.bridge.make_signal_fn
    - rebuilds the genome from a JSON literal embedded in the file
    - calls ``strategy.register(<id>, make_signals)`` at import time

This keeps the file self-contained while reusing the same bridge code path
as the evolution engine, so behaviour is identical to what we saw in OOS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


_TEMPLATE = '''# -*- coding: utf-8 -*-
"""
{title}

Auto-generated champion strategy from evolutionary-strategy-discovery.

  Symbol:        {symbol}
  Interval:      {interval}
  Generation:    {generation}
  IS Sharpe:     {is_sharpe:.3f}
  OOS Sharpe:    {oos_sharpe:.3f}
  IS Return:     {is_return:.2%}
  OOS Return:    {oos_return:.2%}
  IS Trades:     {is_trades}
  OOS Trades:    {oos_trades}
  Species:       {species}

Genome (verbatim):
{genome_block}

Do NOT hand-edit; rerun evolve.export.
"""

from typing import Tuple

import pandas as pd

from cyqnt_trd.blocks import strategy
from cyqnt_trd.evolve.bridge import make_signal_fn
from cyqnt_trd.evolve.genome import StrategyGenome


_GENOME_JSON = {genome_json_literal!r}


def _build_signal_fn():
    g = StrategyGenome.from_dict(__import__("json").loads(_GENOME_JSON))
    return make_signal_fn(g)


_signal_fn = _build_signal_fn()


def make_signals(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Long-only entry/exit signals for {strategy_id}."""
    return _signal_fn(df)


# Default exit_cfg (can be overridden by the runtime via strategy params)
EXIT_CFG = {exit_cfg_block}

DEFAULT_PREFERRED_INTERVAL = {interval!r}

strategy.register("{strategy_id}", make_signals)
'''


def export_champions(session_dir: Path) -> int:
    """Read oos_results.json from session_dir and write one .py per champion.

    Only genomes with passes_degradation_check=True become .py files;
    the others get a sibling ``REJECTED_*.txt`` log entry.
    """
    oos_path = session_dir / "oos_results.json"
    if not oos_path.exists():
        return 0
    raw = json.loads(oos_path.read_text())
    config_raw = json.loads((session_dir / "config.json").read_text())
    symbol = config_raw["symbol"]
    interval = config_raw["interval"]

    out_dir = session_dir / "champions"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for rank, c in enumerate(raw.get("candidates", []), start=1):
        if not c.get("passes_degradation_check"):
            (out_dir / f"REJECTED_{c['genome_id']}.txt").write_text(
                f"{c['genome_id']}: {c.get('degradation_reason', 'no reason')}\n"
                + json.dumps(c, indent=2)
            )
            continue
        path = _write_one_champion(
            out_dir=out_dir,
            symbol=symbol,
            interval=interval,
            rank=rank,
            candidate=c,
        )
        if path:
            n_written += 1
    return n_written


def _write_one_champion(
    *, out_dir: Path, symbol: str, interval: str, rank: int, candidate: Dict[str, Any]
) -> Path | None:
    genome = candidate["genome"]
    genome_id = candidate["genome_id"]
    is_metrics = candidate.get("is_metrics", {})
    oos_metrics = candidate.get("oos_metrics", {})
    species = genome.get("species", "?")
    generation = genome.get("generation", "?")

    strategy_id = f"evo_{symbol.lower()}_{interval}_{species}_r{rank}"

    genome_json_literal = json.dumps(genome, indent=2, sort_keys=True)
    genome_block = "\n".join(["    " + ln for ln in genome_json_literal.splitlines()])

    exit_cfg = {"type": genome["exit_type"], **dict(genome.get("exit_params", {}))}

    body = _TEMPLATE.format(
        title=f"Champion #{rank} for {symbol} {interval}",
        symbol=symbol,
        interval=interval,
        generation=generation,
        species=species,
        is_sharpe=float(is_metrics.get("sharpe_ratio") or 0.0),
        oos_sharpe=float(oos_metrics.get("sharpe_ratio") or 0.0),
        is_return=float(is_metrics.get("total_return") or 0.0),
        oos_return=float(oos_metrics.get("total_return") or 0.0),
        is_trades=int(is_metrics.get("trade_count") or 0),
        oos_trades=int(oos_metrics.get("trade_count") or 0),
        genome_block=genome_block,
        genome_json_literal=json.dumps(genome),  # single-line for repr safety
        strategy_id=strategy_id,
        exit_cfg_block=repr(exit_cfg),
    )

    out_path = out_dir / f"{strategy_id}.py"
    out_path.write_text(body)
    return out_path
