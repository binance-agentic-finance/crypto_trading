"""5-tier verdict gate — factored out so Square-specific AVOID rule is
edited in ONE tiny file (not embedded in template.py)."""
from .avoid_verdict import verdict_five_tier

__all__ = ["verdict_five_tier"]
