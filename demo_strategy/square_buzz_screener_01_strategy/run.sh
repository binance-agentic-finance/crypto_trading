#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DEMO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${DEMO_ROOT}:${PYTHONPATH:-}"

mode="${1:-paper-fg}"

case "$mode" in
  paper-fg) exec python3 -m scripts.paper_trade  ;;
  live)     exec python3 -m scripts.live_trade   ;;
  backtest) exec python3 -m scripts.backtest     ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac
