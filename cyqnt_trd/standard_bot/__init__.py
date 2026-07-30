"""
Scaffold package for the standard trading bot architecture.

This package intentionally starts as a non-breaking sidecar to the existing
``cyqnt_trd`` modules. The current repository can keep using the legacy
modules while new functionality migrates into the protocol-first layers here.

Subpackages are resolved lazily (PEP 562) so a contract-only consumer — the
``advisory`` monitor route, or anything that just needs ``core`` — does not
drag in the execution/REST dependencies at import time. ``from
cyqnt_trd.standard_bot import core`` and ``standard_bot.core.X`` both keep
working exactly as before.
"""

from __future__ import annotations

import importlib

__all__ = [
    # 标准 bot 主接口 —— 写一支 bot 只需要这几个
    "UniversalBot",
    "Features",
    "CAPABILITIES",
    "StandardBot",
    "BotSpec",
    "BotKind",
    "BotContext",
    "DataRequest",
    "CapabilityError",
    "create_bot",
    "list_bots",
    "describe_bot",
    "register_bot",
    # 子包
    "advisory",
    "core",
    "data",
    "execution",
    "monitoring",
    "orchestration",
    "runtime",
    "signal",
    "simulation",
]


#: 名字 -> 它所在的子模块。这些是符号不是子包，所以单独解析。
_SYMBOLS = {
    "StandardBot": "bot", "BotSpec": "bot", "BotKind": "bot",
    "BotContext": "bot", "DataRequest": "bot", "CapabilityError": "bot",
    "UniversalBot": "universal", "Features": "universal",
    "CAPABILITIES": "universal",
    "create_bot": "registry", "list_bots": "registry",
    "describe_bot": "registry", "register_bot": "registry",
}


def __getattr__(name):
    if name in _SYMBOLS:
        module = importlib.import_module("%s.%s" % (__name__, _SYMBOLS[name]))
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name not in __all__:
        raise AttributeError(name)
    module = importlib.import_module("%s.%s" % (__name__, name))
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(__all__))
