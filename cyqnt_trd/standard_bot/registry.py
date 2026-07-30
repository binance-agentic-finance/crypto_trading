"""标准 bot 注册表 —— 让 ``StandardBot`` 成为一个可发现、可寻址的东西。

没有这一层的话，``StandardBot`` 只是一个抽象类：写完一支 bot 之后，没有任何
地方能按名字找到它，前端列不出来，调度器起不了，命令行也跑不了。

这里做三件事：

``register_bot`` / ``create_bot`` / ``list_bots``
    按 ``bot_id`` 注册和取用。同名重复注册直接报错——两支同名 bot 静默覆盖，
    是那种上线三周后才发现"跑的根本不是这支"的问题。

``describe_bot``
    一支 bot 的完整契约：它声明要读什么（含每个输入的可回放等级和 PIT 陷阱），
    以及它被允许发出什么 intent。前端和评审都读这个，不用去翻源码。

``bind_to_signal_registry``
    把标准 bot 注入既有的 ``SignalPluginRegistry``，这样它和交易/选币/监控
    plugin 走同一个 ``run_pipeline_step``，不另开执行路径。

``load_bot_module`` 让外部策略模块（``strategies.standard.*``）在被 import 时
自动注册，和 ``blocks.strategy.register()`` 的习惯一致。
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Optional, Type

from .bot import BotKind, StandardBot

__all__ = [
    "register_bot",
    "create_bot",
    "list_bots",
    "list_bot_ids",
    "describe_bot",
    "unregister_bot",
    "bind_to_signal_registry",
    "load_bot_module",
    "BUILTIN_BOT_MODULES",
]

#: 随包发布的样例 bot。``load_builtin_bots()`` 会 import 它们触发注册。
BUILTIN_BOT_MODULES = (
    "strategies.standard.funding_crowding_neutral",
    "strategies.standard.funding_carry_gated",
    "strategies.standard.funding_oi_crowding_monitor",
    "strategies.standard.news_catalyst_trade",
    "strategies.standard.social_flow_divergence",
)

_REGISTRY: Dict[str, Callable[..., StandardBot]] = {}


def register_bot(
    factory: Any, *, bot_id: Optional[str] = None, replace: bool = False
) -> str:
    """注册一支 bot。``factory`` 是 ``StandardBot`` 子类或返回实例的可调用对象。"""
    if isinstance(factory, type) and issubclass(factory, StandardBot):
        resolved = bot_id or getattr(factory, "spec").bot_id
    elif callable(factory):
        if not bot_id:
            raise ValueError("以可调用对象注册时必须显式给出 bot_id")
        resolved = bot_id
    else:
        raise TypeError("factory 必须是 StandardBot 子类或可调用对象")

    if resolved in _REGISTRY and not replace:
        raise ValueError(
            "bot_id %r 已注册。两支同名 bot 静默覆盖，是那种上线之后才发现"
            "「跑的根本不是这支」的问题——换个名字，或显式传 replace=True。" % resolved
        )
    _REGISTRY[resolved] = factory
    return resolved


def unregister_bot(bot_id: str) -> bool:
    return _REGISTRY.pop(bot_id, None) is not None


def create_bot(bot_id: str, **config: Any) -> StandardBot:
    """按 id 实例化。未注册时把已知的列出来，而不是抛一个裸 KeyError。"""
    if bot_id not in _REGISTRY:
        load_builtin_bots()
    try:
        factory = _REGISTRY[bot_id]
    except KeyError:
        raise KeyError(
            "未注册的 bot %r；已注册：%s" % (bot_id, ", ".join(sorted(_REGISTRY)) or "(空)")
        )
    return factory(**config)


def list_bot_ids() -> List[str]:
    load_builtin_bots()
    return sorted(_REGISTRY)


def list_bots() -> List[Dict[str, Any]]:
    """所有已注册 bot 的能力摘要（前端列表用）。"""
    return [create_bot(bot_id).spec.to_dict() for bot_id in list_bot_ids()]


def describe_bot(bot_id: str, **config: Any) -> Dict[str, Any]:
    """一支 bot 的完整契约：声明的输入 + 允许的输出。

    输入部分逐个带上 ``availability`` 和 ``pit_hazard``，所以看这一份就能回答
    "这支能不能回测"，不用去翻数据目录。
    """
    bot = create_bot(bot_id, **config)
    spec = bot.spec.to_dict()

    inputs = bot.typed_inputs()
    not_replayable = [
        key for key, meta in inputs.items()
        if meta and meta.get("availability") != "BACKTESTABLE"
    ]
    return {
        "bot": spec,
        "inputs": inputs,
        "output": {
            "schema": "cyqnt.signal/v2",
            "allowed_intents": spec["allowed_intents"],
            "reads_positions": spec["reads_positions"],
            "auto_trade_eligible": bot.spec.kind is not BotKind.ADVISORY,
        },
        "backtestable": not not_replayable,
        "blocks_backtest": not_replayable,
    }


def bind_to_signal_registry(registry: Any, *, bot_ids: Optional[List[str]] = None) -> Any:
    """把标准 bot 注入 ``SignalPluginRegistry``。

    注入之后它们和内建 plugin、block 策略、advisory bot 在同一个 registry 里，
    按 ``plugin_id`` 寻址，走同一个 ``run_pipeline_step`` —— 不另开执行路径。
    """
    from types import SimpleNamespace

    for bot_id in bot_ids or list_bot_ids():
        bot = create_bot(bot_id)
        registry.register(bot, lambda raw: SimpleNamespace(**dict(raw or {})))
    return registry


def load_bot_module(module_path: str) -> List[str]:
    """import 一个策略模块并返回它新注册的 bot_id。

    模块在文件尾调用 ``register_bot(MyBot)`` 即可，和
    ``blocks.strategy.register()`` 的习惯一致。
    """
    before = set(_REGISTRY)
    importlib.import_module(module_path)
    return sorted(set(_REGISTRY) - before)


_BUILTINS_LOADED = False


def load_builtin_bots(*, strict: bool = False) -> List[str]:
    """import 随包的样例 bot（幂等）。

    单支 import 失败不影响其余：``strict=False`` 时记下来继续，因为一支样例
    依赖的可选包缺失，不该让整个注册表用不了。
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return sorted(_REGISTRY)
    _BUILTINS_LOADED = True

    failures: List[str] = []
    for module_path in BUILTIN_BOT_MODULES:
        try:
            importlib.import_module(module_path)
        except Exception as exc:
            failures.append("%s: %s" % (module_path, exc))
            if strict:
                raise
    if failures:
        import warnings

        warnings.warn("部分样例 bot 未能载入：%s" % "; ".join(failures), RuntimeWarning)
    return sorted(_REGISTRY)
