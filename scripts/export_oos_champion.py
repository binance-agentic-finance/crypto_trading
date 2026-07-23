"""Export top OOS champion(s) directly as runnable strategy.register() .py files.

Reads true_champions.json, picks the top N, writes one .py per champion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cyqnt_trd.evolve.export import _write_one_champion


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--root",
                   default=".codemax/spec-workflow/specs/evolutionary-strategy-discovery/sessions")
    p.add_argument("--top-n", type=int, default=5)
    args = p.parse_args()

    sdir = Path(args.root) / args.session
    cfg = json.loads((sdir / "config.json").read_text())
    symbol = cfg["symbol"]; interval = cfg["interval"]

    champs_data = json.loads((sdir / "true_champions.json").read_text())
    champs = champs_data["champions"][: args.top_n]

    out_dir = sdir / "champions_oos"
    out_dir.mkdir(parents=True, exist_ok=True)

    for rank, c in enumerate(champs, 1):
        # Adapt schema to what export.py wants
        candidate = {
            "genome_id": c["id"],
            "genome": c["genome"],
            "is_metrics": {
                "total_return": c["is_ret"], "sharpe_ratio": c["is_sharpe"],
                "trade_count": c["is_trades"], "win_rate": 0.0, "max_drawdown": 0.0,
            },
            "oos_metrics": {
                "total_return": c["oos_ret"], "sharpe_ratio": c["oos_sharpe"],
                "trade_count": c["oos_trades"], "win_rate": c["oos_winrate"],
                "max_drawdown": c["oos_dd"],
            },
        }
        path = _write_one_champion(
            out_dir=out_dir, symbol=symbol, interval=interval,
            rank=rank, candidate=candidate,
        )
        print(f"#{rank}: {path}")
    print(f"\n{len(champs)} champion .py files in {out_dir}")


if __name__ == "__main__":
    main()
