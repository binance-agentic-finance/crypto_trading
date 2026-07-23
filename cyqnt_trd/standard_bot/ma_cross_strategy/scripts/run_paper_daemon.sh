#!/usr/bin/env bash
# =============================================================================
# run_paper_daemon.sh — MA cross paper trade daemon
#
# 使用 cyqnt_trd 框架的 mvp_paper_daemon entrypoint。
#
# 用法：
#   chmod +x scripts/run_paper_daemon.sh
#   ./scripts/run_paper_daemon.sh
#
# 背景執行：
#   nohup ./scripts/run_paper_daemon.sh > /tmp/paper_daemon.out 2>&1 &
#
# 停止：
#   python -c "
#   import json, pathlib
#   p = pathlib.Path('./watcher/MA_CROSS_V1_ETHUSDT_1m/state.json')
#   d = json.loads(p.read_text()); d['status'] = 'stopped'; p.write_text(json.dumps(d))
#   "
# =============================================================================

set -euo pipefail

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
# ── 策略參數（可用環境變數覆蓋）──────────────────────────────────────────────
SYMBOL="${SYMBOL:-ETHUSDT}"
INTERVAL="${INTERVAL:-1m}"
STRATEGY="${STRATEGY:-ma_cross_v1}"
STRATEGY_MODULE="${STRATEGY_MODULE:-strategies.ma_cross_v1}"
MARKET_TYPE="${MARKET_TYPE:-futures}"

INITIAL_CAPITAL="${INITIAL_CAPITAL:-10000}"
FEE_BPS="${FEE_BPS:-4}"
SLIPPAGE_BPS="${SLIPPAGE_BPS:-2}"
WARM_UP_BARS="${WARM_UP_BARS:-80}"
POLL_INTERVAL="${POLL_INTERVAL:-55}"
EXTRA_PARAMS="${EXTRA_PARAMS:-}"
SESSION_END_AT="${SESSION_END_AT:-}"

STATE_DIR_DEFAULT="${WORKSPACE}/watcher/${STRATEGY^^}_${SYMBOL}_${INTERVAL}"
STATE_DIR="${STATE_DIR:-$STATE_DIR_DEFAULT}"

# ── 環境 ──────────────────────────────────────────────────────────────────────
mkdir -p "${STATE_DIR}"
export PYTHONPATH="${WORKSPACE}:${PYTHONPATH:-}"

PYTHON="$(command -v python3.11 || command -v python3)"

echo "[paper_daemon] symbol=${SYMBOL} interval=${INTERVAL}"
echo "[paper_daemon] state_dir=${STATE_DIR}"
echo "[paper_daemon] python=${PYTHON}"

# ── 啟動 ──────────────────────────────────────────────────────────────────────
CMD=(
  "${PYTHON}" -m cyqnt_trd.standard_bot.entrypoints.mvp_paper_daemon
  --engine          python
  --strategy        "${STRATEGY}"
  --strategy-module "${STRATEGY_MODULE}"
  --symbol          "${SYMBOL}"
  --interval        "${INTERVAL}"
  --market-type     "${MARKET_TYPE}"
  --state-dir       "${STATE_DIR}"
  --poll-interval   "${POLL_INTERVAL}"
  --warm-up-bars    "${WARM_UP_BARS}"
  --initial-capital "${INITIAL_CAPITAL}"
  --fee-bps         "${FEE_BPS}"
  --slippage-bps    "${SLIPPAGE_BPS}"
)

if [[ -n "${EXTRA_PARAMS}" ]]; then
  CMD+=(--extra-params "${EXTRA_PARAMS}")
fi

if [[ -n "${SESSION_END_AT}" ]]; then
  CMD+=(--session-end-at "${SESSION_END_AT}")
fi

exec "${CMD[@]}"
