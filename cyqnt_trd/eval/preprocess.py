"""因子预处理：写因子时用得上的四件事。

对标 `jqfactor_analyzer.preprocess`（winsorize / winsorize_med / standardlize /
neutralize）。这里只保留在加密截面上真正用得到的部分，并且**不做行业中性**——
没有可回测的 PIT 行业分类，用事后分类回填就是泄漏（见 alpha101_crypto 的说明）。

一句话区别：`zscore` 会被后续的 rank 吃掉（矩阵内部按截面秩打分），所以它对**裁决**
没有影响；真正会改变裁决的是 `winsorize_mad`（改变极端值的相对次序）和 `neutralize`
（改变因子暴露在谁身上）。不要以为"标准化一下分数会变好"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["winsorize_mad", "winsorize_quantile", "zscore", "neutralize"]


def winsorize_mad(data: pd.DataFrame, scale: float = 3.0, axis: int = 1) -> pd.DataFrame:
    """按中位数绝对偏差（MAD）逐期截尾，对应 jq 的 ``winsorize_med``。

    截断边界为 ``median ± scale × MAD``。比按标准差截尾稳健：加密收益的尾部本身很
    厚，用均值/标准差定边界时，极端值会把自己的边界推出去，等于没截。
    """
    med = data.median(axis=axis)
    mad = (data.sub(med, axis=1 - axis)).abs().median(axis=axis)
    lower = med - scale * mad
    upper = med + scale * mad
    return data.clip(lower=lower, upper=upper, axis=1 - axis)


def winsorize_quantile(data: pd.DataFrame, lower: float = 0.025,
                       upper: float = 0.975, axis: int = 1) -> pd.DataFrame:
    """按逐期分位数截尾。截面只有约 8 个名字时，2.5% 分位基本等于最小值，
    此时它几乎不起作用——窄截面上优先用 :func:`winsorize_mad`。"""
    lo = data.quantile(lower, axis=axis)
    hi = data.quantile(upper, axis=axis)
    return data.clip(lower=lo, upper=hi, axis=1 - axis)


def zscore(data: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """逐期截面标准化，对应 jq 的 ``standardlize``。

    注意：矩阵内部用的是截面**秩**，z-score 是单调变换，因此它**不会**改变 IC、
    分箱或裁决。它有用的场合是你要把几个因子加起来合成——那时量纲必须先统一。
    """
    x = data if mask is None else data.where(mask)
    centred = x.sub(x.mean(axis=1), axis=0)
    return centred.div(x.std(axis=1).replace(0, np.nan), axis=0)


def neutralize(data: pd.DataFrame, factors, mask: pd.DataFrame | None = None,
               min_assets: int = 5) -> pd.DataFrame:
    """逐期截面回归取残差，对应 jq 的 ``neutralize``。

    ``factors`` 是一个 DataFrame 或一组 DataFrame（dict / 可迭代）。回归在每个日期
    单独拟合，返回残差；样本数不足 ``min_assets`` 的日期整行留空，而不是用更少的
    点硬拟合。

    与 jq 的差别：那边默认中性化到行业哑变量和市值，这里**不提供行业**——加密没有
    可回测的 PIT 行业历史。要中性化到规模/低波之类，把对应的面板自己传进来（
    :mod:`cyqnt_trd.eval.baselines` 里就有现成的五个）。
    """
    if isinstance(factors, pd.DataFrame):
        factors = [factors]
    elif isinstance(factors, dict):
        factors = list(factors.values())
    else:
        factors = list(factors)
    if not factors:
        raise ValueError("neutralize needs at least one factor to regress out")

    y = data if mask is None else data.where(mask)
    xs = [f if mask is None else f.where(mask) for f in factors]
    out = pd.DataFrame(np.nan, index=y.index, columns=y.columns)
    for t in y.index:
        yt = y.loc[t]
        cols = [x.loc[t] for x in xs]
        ok = yt.notna()
        for c in cols:
            ok &= c.notna()
        n = int(ok.sum())
        if n < min_assets or n <= len(cols) + 1:
            continue
        A = np.column_stack([np.ones(n)] + [c[ok].to_numpy(float) for c in cols])
        beta, *_ = np.linalg.lstsq(A, yt[ok].to_numpy(float), rcond=None)
        out.loc[t, ok[ok].index] = yt[ok].to_numpy(float) - A @ beta
    return out
