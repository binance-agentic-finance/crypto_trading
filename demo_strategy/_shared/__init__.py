"""demo_strategy._shared — helpers shared by every strategy in demo_strategy/.

Strategy-specific logic MUST NOT live here. Only truly universal utilities:
config loading + validation, sys.path bootstrap for atomic_strategy_lib,
output directory helpers, formatting helpers for reports.
"""
from .paths import bootstrap_sys_path, workspace_dir
from .cfg import load_cfg, apply_cli_overrides, activate_mode
from .fmt import fmt_pct, fmt_price, fmt_scalar, banner

__all__ = [
    "bootstrap_sys_path",
    "workspace_dir",
    "load_cfg",
    "apply_cli_overrides",
    "activate_mode",
    "fmt_pct",
    "fmt_price",
    "fmt_scalar",
    "banner",
]
