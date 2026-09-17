"""把 capability 节点输出的因子接进评测矩阵。

矩阵要的是 `日期 × 标的` 面板，因为它做的是**截面排序**。上游算子给的不是这个形状
—— 它按标的算，而且输出通常是下面两种之一：

========================= ============================================ ====================
算子输出                   典型写法                                       本模块入口
========================= ============================================ ====================
当前时点**一个数**          ``-> 0.37`` 或 ``-> {"value": 0.37}``         带 ``window``
整条**序列**               ``-> {"series": [None, .., 0.37]}``           不带 ``window``
========================= ============================================ ====================

输入侧按同样的约定：算子收的是 ``list[float | None]``（``series`` / ``high`` /
``low`` 这类命名参数），不是 DataFrame。所以适配器负责把面板切成每个标的的 list 喂
进去，再把返回值拼回面板 —— **"算一串序列"留在矩阵内部，算子那边只管出它本来就出
的东西。**

    from factor_eval import evaluate, capability_factor

    # 单值输出:滚动窗口逐日调用,拼成序列
    card = evaluate(capability_factor(ma_node, window=20,
                                      inputs={"series": "close"},
                                      params={"period": 20}, output="value"))

    # 序列输出:一次算完整条,直接对齐
    card = evaluate(capability_factor(combine_node,
                                      inputs={"signals": "close"}, output="series"))

转完之后走的仍是同一条 ``evaluate()``:没有新的回测、没有新的权重口径、没有新的
标定。上游算子一行不改,这里只做形状转换。

**时点安全**:带 ``window`` 时每次只把截至当日的窗口交给算子,按构造看不到未来;
不带 ``window`` 时整条历史一次交给算子(序列输出本来就这样用),是否因果由算子自己
负责 —— ``evaluate()`` 的前缀一致性抽查会去核对。
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["capability_factor", "read_port", "panel_rows", "PANEL_FIELDS", "BAR_FIELDS"]

#: 面板上可以喂给算子的字段。
PANEL_FIELDS = ("open", "high", "low", "close", "volume", "quote_volume")

#: Fields placed in each bar dict when building a ``rows`` input. Names follow the
#: capability-side kline convention, which upstream blocks read as ``rows[i]["close"]``.
BAR_FIELDS = ("open", "high", "low", "close", "volume", "quote_volume")


def read_port(result: Any, output: str | None = None) -> Any:
    """从算子返回值里取出要的那一路输出。

    算子可能返回一个多路输出的字典（``{"value": .., "series": [..]}``），也可能直接
    返回裸值。``output`` 没给时：字典只有一路就取它，多路则取 ``value``，都不满足就
    报错 —— 不猜。
    """
    if not isinstance(result, Mapping):
        return result
    if output is not None:
        if output not in result:
            raise KeyError(f"operator returned {sorted(result)}; no output named {output!r}")
        return result[output]
    if len(result) == 1:
        return next(iter(result.values()))
    if "value" in result:
        return result["value"]
    raise KeyError(f"operator returned {sorted(result)}; pass output= to pick one")


def _scalar(value, where: str) -> float:
    if isinstance(value, (Mapping, str, bytes)):
        raise TypeError(f"{where} returned {type(value).__name__}; expected one number")
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        raise TypeError(f"{where} returned a sequence; drop window= to use the series form")
    if value is None:
        return np.nan
    out = float(value)
    return out if np.isfinite(out) else np.nan


def _series(value, index, where: str) -> pd.Series:
    """把 ``list[float | None]`` 对齐到索引；短于索引时按**右对齐**补在末尾。

    右对齐是刻意的:序列输出的自然语义是"最后一个值对应最后一根 bar",预热期缺的
    那几根应该落在开头。左对齐会把整条曲线往前挪,变成实打实的未来函数。
    """
    if isinstance(value, pd.Series):
        values = value.to_numpy(dtype=float)
    elif isinstance(value, (list, tuple, np.ndarray)):
        values = np.array([np.nan if v is None else v for v in value], dtype=float)
    else:
        raise TypeError(f"{where} returned {type(value).__name__}; expected list[float | None]")
    if len(values) > len(index):
        raise ValueError(f"{where} returned {len(values)} values for {len(index)} bars")
    out = np.full(len(index), np.nan)
    if len(values):
        out[len(index) - len(values):] = values
    return pd.Series(out, index=index)


def panel_rows(panel, symbol: str, fields: Sequence[str] = BAR_FIELDS) -> list[dict]:
    """Slice one symbol out of the panel as ``rows``: one dict per bar.

    Most factor-shaped capabilities in manifest v3 take this shape
    (``dict[str, Any] | list[Any]``, conventionally named ``rows``) rather than
    ``list[float]``. Missing values stay ``None``; filling them with 0 makes an
    upstream indicator return a plausible-looking wrong number instead of bailing.
    """
    frames = {f: getattr(panel, f)[symbol].to_numpy(dtype=float) for f in fields}
    n = len(panel.index)
    return [{f: (None if not np.isfinite(frames[f][i]) else float(frames[f][i]))
             for f in fields} for i in range(n)]


def capability_factor(fn: Callable[..., Any], *,
                      inputs: Mapping[str, str] | None = None,
                      rows: str | None = None,
                      bar_fields: Sequence[str] = BAR_FIELDS,
                      params: Mapping[str, Any] | None = None,
                      window: int | None = None,
                      output: str | None = None,
                      min_periods: int | None = None,
                      skip_failures: bool = False):
    """把一个按标的计算的算子包成矩阵能评的面板因子。

    参数
    ----
    fn
        算子函数。以**关键字**调用，输入序列是 ``list[float | None]``。
    inputs
        算子参数名 → 面板字段名。默认 ``{"series": "close"}``（``rows`` 给了时默认为空）。
        需要 OHLC 的算子写成 ``{"high": "high", "low": "low", "close": "close"}``。
    rows
        Name of the parameter that receives the bar-dict list (manifest type
        ``dict[str, Any] | list[Any]``). Built with :func:`panel_rows`. May be
        combined with ``inputs``; a few operators need both.
    bar_fields
        Fields placed in each bar dict, OHLCV plus quote volume by default.
    params
        除序列以外的固定参数（``period``、``length`` 之类），原样透传。
    window
        给了就是**单值输出**：每个日期只把截至当日的 ``window`` 根交给算子，取回一个数。
        不给就是**序列输出**：整条历史一次交给算子，取回一条 list。
    output
        取哪一路输出；不给时按 :func:`read_port` 的规则推断。
    skip_failures
        默认 ``False``：某个标的/某个时点算不出来就抛错，而不是留空让它安静退出截面。
        打开后失败处记为缺失，由 G0 的覆盖率如实反映。

    返回的是 ``Panel -> DataFrame``，直接喂给 :func:`factor_eval.evaluate`。
    """
    inputs = dict(inputs if inputs is not None else ({} if rows else {"series": "close"}))
    params = dict(params or {})
    if not inputs and not rows:
        raise ValueError("capability_factor needs at least one of inputs= or rows=")
    bar_fields = tuple(bar_fields)
    unknown_bars = sorted(set(bar_fields) - set(PANEL_FIELDS))
    if rows and unknown_bars:
        raise ValueError(f"bar_fields reference unknown panel fields: {', '.join(unknown_bars)}")
    if rows and rows in params:
        raise ValueError(f"{rows!r} given both as the rows input and a fixed param")
    if rows and rows in inputs:
        raise ValueError(f"{rows!r} given both as the rows input and a series input")
    unknown = sorted(set(inputs.values()) - set(PANEL_FIELDS))
    if unknown:
        raise ValueError(f"inputs reference unknown panel fields: {', '.join(unknown)}; "
                         f"available: {', '.join(PANEL_FIELDS)}")
    overlap = sorted(set(inputs) & set(params))
    if overlap:
        raise ValueError(f"{', '.join(overlap)} given both as an input series and a fixed param")
    if window is not None:
        if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 2:
            raise ValueError("window must be an integer >= 2 bars, or None for the series form")
        window = int(window)
    floor = window if min_periods is None else int(min_periods or 0)
    if window is not None and not 2 <= floor <= window:
        raise ValueError("min_periods must be between 2 and window")

    def factor(panel):
        columns = {name: getattr(panel, field) for name, field in inputs.items()}
        out = {}
        for symbol in panel.symbols:
            index = panel.index
            data = {name: frame[symbol] for name, frame in columns.items()}
            bars = panel_rows(panel, symbol, bar_fields) if rows else None
            if window is None:
                kwargs = {name: [None if not np.isfinite(v) else float(v)
                                 for v in series.to_numpy(dtype=float)]
                          for name, series in data.items()}
                if rows:
                    kwargs[rows] = bars
                try:
                    value = read_port(fn(**kwargs, **params), output)
                except Exception as exc:  # noqa: BLE001 - 不吞,交给调用方
                    if not skip_failures:
                        raise RuntimeError(f"operator raised on {symbol}: "
                                           f"{type(exc).__name__}: {exc}") from exc
                    out[symbol] = pd.Series(np.nan, index=index)
                    continue
                out[symbol] = _series(value, index, f"operator on {symbol}")
                continue

            arrays = {name: series.to_numpy(dtype=float) for name, series in data.items()}
            values = np.full(len(index), np.nan)
            for i in range(floor - 1, len(index)):
                start = max(0, i - window + 1)
                kwargs = {name: [None if not np.isfinite(v) else float(v)
                                 for v in array[start:i + 1]]
                          for name, array in arrays.items()}
                if rows:
                    kwargs[rows] = bars[start:i + 1]
                try:
                    values[i] = _scalar(read_port(fn(**kwargs, **params), output),
                                        f"operator on {symbol}")
                except Exception as exc:  # noqa: BLE001
                    if not skip_failures:
                        raise RuntimeError(f"operator raised on {symbol} at "
                                           f"{index[i]}: {type(exc).__name__}: {exc}") from exc
            out[symbol] = pd.Series(values, index=index)
        return pd.DataFrame(out).reindex(columns=panel.symbols)

    factor.__name__ = getattr(fn, "__name__", "capability_factor")
    factor.__doc__ = (f"capability adapter around {getattr(fn, '__qualname__', fn)!r} "
                      f"({'single value, window=%d' % window if window else 'series'})")
    return factor
