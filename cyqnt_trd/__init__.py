"""
Cyqnt Trading Package

一个用于加密货币交易的工具包，包含数据获取、交易信号生成和回测功能。

主要模块:
- get_data: 数据获取模块，支持从 Binance 获取期货和现货数据
- trading_signal: 交易信号模块，包含因子计算和信号策略
- backtesting: 回测框架，支持因子测试和策略回测
- strategy_cases: 随 package 一起分发的策略案例与 preset 资源
- compat: atomic_strategy_lib backward-compatible dataclass shim (Candle, Signal, Verdict, ...)
"""

#: Single source of truth for the installed version.
#:
#: This said ``0.1.9.dev2`` while pyproject said ``0.1.11`` — four releases of
#: drift, and nothing caught it because nothing compared them. A consumer doing
#: ``cyqnt_trd.__version__`` got a number that had not existed on PyPI for
#: months, which is worse than no number at all: it looks authoritative.
#: tests/test_version_is_single_sourced.py now asserts the two agree.
__version__ = "0.1.14"

#: 记录懒加载子包时吞掉的 import 错误：{子包名: 异常}。
#:
#: 以前根 __init__ 用一个 eager 循环把全部子包导进来（失败就吞进这里），代价是
#: **任何** ``import cyqnt_trd.*`` 都会连带拉进 get_data / trading_signal /
#: backtesting / standard_bot 的重依赖——binance_sdk_spot / requests / aiohttp /
#: scipy / matplotlib。下游的纯计算消费者（capability sandbox，见 SOP §5「不 import
#: 交易所 SDK / HTTP 客户端」）为此不得不挂一个文件系统垫片绕过本文件。
#:
#: 现在改成 PEP 562 的模块级 ``__getattr__`` 懒加载：子包只在**真正被访问**时才导入，
#: 于是 ``import cyqnt_trd.blocks.sizing`` 这类干净的叶子导入不会触发任何重子包，
#: sandbox 可以直接公共 import，垫片随之删除。
#:
#: 语义与旧的 ``_safe_import`` 保持一致：子包 import 失败不抛，而是把异常记进这里、
#: 该子包属性解析为 ``None``——依赖「缺可选依赖时子包为 None」的调用方行为不变。
#: 唯一差别是记录时机从「导入本包时」推迟到「首次访问该子包时」。
_OPTIONAL_IMPORT_ERRORS = {}

#: 懒加载的子包名。顺序不重要——只有被访问的那个会被导入。
_LAZY_SUBMODULES = (
    "get_data",
    "trading_signal",
    "backtesting",
    "standard_bot",
    "strategy_cases",
    "utils",
    "compat",
)


def __getattr__(name):
    """PEP 562：首次访问子包时才导入，导入结果缓存进 globals()（下次不再走这里）。

    失败时沿用旧的降级语义：记进 _OPTIONAL_IMPORT_ERRORS，属性置 None。
    """
    if name in _LAZY_SUBMODULES:
        try:
            import importlib

            module = importlib.import_module(f"{__name__}.{name}")
        except Exception as exc:  # pragma: no cover - import environment dependent
            module = None
            _OPTIONAL_IMPORT_ERRORS[name] = exc
        globals()[name] = module          # 缓存：后续访问直接命中，不再触发 __getattr__
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    'get_data',
    'trading_signal',
    'backtesting',
    'standard_bot',
    'strategy_cases',
    'utils',
    'compat',
    '__version__',
    '_OPTIONAL_IMPORT_ERRORS',
]
