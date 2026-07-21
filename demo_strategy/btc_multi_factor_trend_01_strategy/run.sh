#!/usr/bin/env bash
# bdp-ai-trading-bot supervisor calls this with one of: paper-fg | live | backtest
#
#   paper-fg  → foreground paper daemon (SIGTERM-clean)
#   live      → foreground live daemon
#   backtest  → one-shot backtest run
#
# All modes exec Python, so PID handed to supervisor is the strategy process.
set -euo pipefail

cd "$(dirname "$0")"

# Add demo_strategy/ to PYTHONPATH so `_shared` resolves
DEMO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${DEMO_ROOT}:${PYTHONPATH:-}"

mode="${1:-paper-fg}"

case "$mode" in
  paper-fg)
    exec python3 -m scripts.paper_trade
    ;;
  live)
    exec python3 -m scripts.live_trade
    ;;
  backtest)
    exec python3 -m scripts.backtest
    ;;
  *)
    echo "unknown mode: $mode (expected paper-fg | live | backtest)" >&2
    exit 2
    ;;
esac
