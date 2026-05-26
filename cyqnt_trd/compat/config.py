"""Configuration loader with embedded defaults.

Drop-in replacement for ``atomic_strategy_lib.core.config``. The default
config schema covers the full atomic strategy pipeline (CLI / data /
signals / scoring / trade / execution / graduated_tp / monitor / rescan /
pnl_tracker) so existing atomic case scripts can switch their import
path without further changes::

    # Old
    from atomic_strategy_lib.core.config import load_config, merge_config

    # New
    from cyqnt_trd.compat.config import load_config, merge_config
    # or just
    from cyqnt_trd.compat import load_config, merge_config

This module is intentionally placed in ``compat`` because the schema
encodes atomic-specific concepts (graduated TP, rescan loop, atomic-style
verdict thresholds). New code that only consumes ``cyqnt_trd.blocks``
should not depend on this loader.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Default config (verbatim port of atomic_strategy_lib.core.config.DEFAULT_CONFIG)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "cli": {
        "binary": "binance-cli",
        "pro_binary": "binance-pro-cli",
        "timeout": 30,
        "profile": None,
    },
    "data": {
        "default_kline_limit": 100,
        "default_depth_limit": 20,
        "velocity_timeframes": ["1m", "5m", "1h", "4h", "1d", "1w"],
        "velocity_kline_limits": {
            "1m": 60, "5m": 60, "1h": 48, "4h": 24, "1d": 14, "1w": 8,
        },
    },
    "signals": {
        "ema_periods": [20, 30, 60, 200],
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "atr_period": 14,
        "bb_period": 20,
        "bb_std": 2.0,
        "volume_surge_threshold": 2.0,
        "funding_squeeze_threshold": -0.0001,
        "funding_crowded_threshold": 0.0005,
        "oi_anomaly_threshold_pct": 10.0,
    },
    "scoring": {
        "gates": {
            "spot_volume_min_for_candidate": 3_000_000,
            "spot_volume_hard_cap_verdict": "WATCHLIST",
        },
        "verdicts": {
            "strong_candidate_min": 10,
            "candidate_min": 6,
            "watchlist_min": 2,
            "skip_min": 0,
        },
    },
    "trade": {
        "max_loss_per_trade": 200,
        "account_balance": 1000,
        "default_stop_pct": 5.0,
        "new_coin_stop_pct": 8.0,
        "volatile_stop_pct": 12.0,
        "volatile_threshold_7d_pct": 50.0,
        "max_leverage": 5,
        "min_leverage": 1,
        "short_require_7d_negative": True,
        "short_require_funding_above": 0.0003,
        "short_require_sentiment_bearish": True,
    },
    "execution": {
        "enabled": False,
        "mode": "futures",
        "profile": "default",
        "live": False,
        "min_verdict": "STRONG_CANDIDATE",
        "max_concurrent_positions": 3,
        "max_price_deviation_pct": 2.0,
        "spot": {
            "stop_order_type": "STOP_LOSS_LIMIT",
            "time_in_force": "GTC",
        },
        "futures": {
            "margin_type": "ISOLATED",
            "stop_order_type": "STOP_MARKET",
            "working_type": "CONTRACT_PRICE",
        },
    },
    "graduated_tp": {
        "enabled": True,
        "stage1_profit_pct": 5,
        "stage1_sell_pct": 5,
        "stage1_stop_to_breakeven": True,
        "stage2_profit_pct": 10,
        "stage2_sell_pct": 10,
        "stage2_stop_above_entry_pct": 3,
        "stage3_profit_pct": 20,
        "stage3_sell_pct": 15,
        "stage3_trailing_stop_pct": 5,
        "emergency_profit_drop_threshold_pct": 10,
        "emergency_sell_pct": 50,
    },
    "monitor": {
        "enabled": False,
        "positions_file": "tmp/positions.json",
        "daily_loss_limit_pct": 10,
        "daily_loss_limit_abs": None,
        "stale_stop_check": True,
        "max_concurrent_positions": 5,
        "check_interval_seconds": 60,
        "log_file": "tmp/monitor.log",
    },
    "rescan": {
        "enabled": False,
        "interval_seconds": 900,
        "trend_timeframe": "4h",
        "trend_candles": 3,
        "margin_alert_threshold_pct": 30,
        "emergency_exit_margin_pct": 15,
        "loop_mode": False,
        "history_file": "tmp/rescan_history.jsonl",
    },
    "pnl_tracker": {
        "enabled": False,
        "signals_file": "tmp/signals_log.jsonl",
        "check_intervals_hours": [1, 4, 24],
        "auto_record": True,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(path: Optional[str] = None) -> dict:
    """Load config from JSON file, falling back to :data:`DEFAULT_CONFIG`.

    User-supplied keys override defaults; unspecified sections fall back to
    embedded defaults.

    Parameters
    ----------
    path:
        Optional path to a JSON config file. If ``None`` or the file does
        not exist, returns a deep copy of :data:`DEFAULT_CONFIG` so callers
        can mutate freely.

    Returns
    -------
    dict
        Merged configuration dict.
    """
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as f:
            user_cfg = json.load(f)
        return merge_config(DEFAULT_CONFIG, user_cfg)
    return _deep_copy(DEFAULT_CONFIG)


def merge_config(base: dict, override: dict) -> dict:
    """Deep-merge *override* into *base*, returning a new dict.

    Nested dicts are merged recursively; scalar values in *override* replace
    values in *base*. Keys present in *override* but absent from *base* are
    added; keys present in *base* but absent from *override* are kept.
    """
    result: dict = {}
    for key in set(list(base.keys()) + list(override.keys())):
        if key in override and key in base:
            if isinstance(base[key], dict) and isinstance(override[key], dict):
                result[key] = merge_config(base[key], override[key])
            else:
                result[key] = override[key]
        elif key in override:
            result[key] = override[key]
        else:
            result[key] = base[key]
    return result


def get_section(config: dict, section: str, default: Optional[dict] = None) -> dict:
    """Get a config section with fallback to *default* (or empty dict)."""
    return config.get(section, default or {})


def _deep_copy(d: dict) -> dict:
    """Simple deep copy via JSON round-trip (config is JSON-compatible)."""
    return json.loads(json.dumps(d))


def resolve_workspace_path(case_name: str, filename: str = "") -> str:
    """Resolve a writable output path that works both locally and in Docker.

    Uses ``OCL_WORKSPACE_PATH`` env var if set, otherwise
    ``~/.openclaw/workspace/``. Returns a string suitable for use as an
    argparse default.

    Side effect: creates the case directory (and parents) if they don't exist.

    Examples
    --------
    >>> resolve_workspace_path("my-case", "pipeline_result.json")
    '/home/user/.openclaw/workspace/my-case/pipeline_result.json'
    """
    base = Path(
        os.environ.get(
            "OCL_WORKSPACE_PATH",
            Path.home() / ".openclaw" / "workspace",
        )
    )
    ws = base / case_name
    ws.mkdir(parents=True, exist_ok=True)
    if filename:
        return str(ws / filename)
    return str(ws)


__all__ = [
    "DEFAULT_CONFIG",
    "load_config",
    "merge_config",
    "get_section",
    "resolve_workspace_path",
]
