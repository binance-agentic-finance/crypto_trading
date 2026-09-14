"""Alpha101 v2: explicit input units, scoped cross-sections, strict windows.

``paper`` means paper input units with the documented v2 rank convention; it
is NOT a claim to uniquely reproduce WorldQuant's proprietary implementation.
Appendix pp. 15-16 specifies dollar ADV and L1 scale, but does not specify the
rank numeric scale, arg-extrema orientation/ties, or standard-deviation ddof.
See references/semantics.md for sources and the choices frozen in this module.

Formula expressions are an independent copy of the existing transcription;
this module never imports or mutates the legacy alphas/ops module globals.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "alpha101_crypto_v2.0"
IND_ALPHAS = (48, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100)
CAP_ALPHAS = (56,)
COMPARISON_ALPHAS = (21, 27, 61, 62, 64, 65, 68, 74, 75, 79, 81, 86, 95, 99)
PAPER_URL = "https://arxiv.org/pdf/1601.00991"


class Operators:
    """One evaluation's operators; eligibility is never process-global.

    Raw univariate histories are not eligibility-masked. Cross-sectional
    rank, scale, and neutralize mask on the current row, including when nested
    inside a time-series operator. Missing conditions remain missing; an
    unselected ternary branch need not be available.
    """

    def __init__(self, eligible: pd.DataFrame):
        self.eligible = eligible.fillna(False).astype(bool).copy()

    @staticmethod
    def finite(x: pd.DataFrame) -> pd.DataFrame:
        return x.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def window(d: float) -> int:
        if not np.isfinite(d) or d < 1:
            raise ValueError("window must be finite and at least one bar")
        return int(np.floor(d))

    @staticmethod
    def _frame(x: Any, template: pd.DataFrame) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            return Operators.finite(x)
        return pd.DataFrame(x, index=template.index, columns=template.columns, dtype=float)

    def _xs(self, x: pd.DataFrame) -> pd.DataFrame:
        return self.finite(x).where(self.eligible)

    def rank(self, x: pd.DataFrame) -> pd.DataFrame:
        return self._xs(x).rank(axis=1, method="average", pct=True)

    def scale(self, x: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
        x = self._xs(x)
        total = x.abs().sum(axis=1, min_count=1).replace(0, np.nan)
        return x.div(total, axis=0) * a

    def indneutralize(self, x: pd.DataFrame, _group: Any = None) -> pd.DataFrame:
        x = self._xs(x)
        return x.sub(x.mean(axis=1), axis=0)

    @staticmethod
    def sign(x: pd.DataFrame) -> pd.DataFrame:
        return np.sign(Operators.finite(x))

    @staticmethod
    def log(x: pd.DataFrame) -> pd.DataFrame:
        x = Operators.finite(x)
        return np.log(x.where(x > 0))

    @staticmethod
    def abs_(x: pd.DataFrame) -> pd.DataFrame:
        return Operators.finite(x).abs()

    @staticmethod
    def power(x: pd.DataFrame, a: Any) -> pd.DataFrame:
        template = x if isinstance(x, pd.DataFrame) else a
        x, a = Operators._frame(x, template), Operators._frame(a, template)
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            out = Operators.finite(x ** a)
        # NumPy otherwise evaluates 1**NaN and NaN**0 as 1.
        return out.where(x.notna() & a.notna())

    @staticmethod
    def signedpower(x: pd.DataFrame, a: Any) -> pd.DataFrame:
        # Literal Appendix definition x**a. The two formula uses have
        # nonnegative bases; do not silently introduce sign-preserving power.
        return Operators.power(x, a)

    def _compare(self, a: Any, b: Any, compare) -> pd.DataFrame:
        template = a if isinstance(a, pd.DataFrame) else b
        if not isinstance(template, pd.DataFrame):
            raise TypeError("a comparison needs at least one DataFrame")
        a, b = self._frame(a, template), self._frame(b, template)
        return compare(a, b).astype(float).where(a.notna() & b.notna())

    def lt(self, a: Any, b: Any) -> pd.DataFrame:
        return self._compare(a, b, lambda x, y: x < y)

    def le(self, a: Any, b: Any) -> pd.DataFrame:
        return self._compare(a, b, lambda x, y: x <= y)

    def choose(self, condition: pd.DataFrame, yes: Any, no: Any) -> pd.DataFrame:
        condition = self.finite(condition)
        yes, no = self._frame(yes, condition), self._frame(no, condition)
        return yes.where(condition.eq(1.0), no).where(condition.notna())

    @staticmethod
    def emin(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
        return np.minimum(Operators.finite(a), Operators.finite(b))

    @staticmethod
    def emax(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
        return np.maximum(Operators.finite(a), Operators.finite(b))

    def delay(self, x: pd.DataFrame, d: float) -> pd.DataFrame:
        return self.finite(x).shift(self.window(d))

    def delta(self, x: pd.DataFrame, d: float) -> pd.DataFrame:
        x = self.finite(x)
        return x - x.shift(self.window(d))

    def _rolling(self, x: pd.DataFrame, d: float):
        d = self.window(d)
        return self.finite(x).rolling(d, min_periods=d)

    def ts_sum(self, x: pd.DataFrame, d: float) -> pd.DataFrame:
        return self._rolling(x, d).sum()

    def ts_mean(self, x: pd.DataFrame, d: float) -> pd.DataFrame:
        return self._rolling(x, d).mean()

    def ts_std(self, x: pd.DataFrame, d: float) -> pd.DataFrame:
        return self._rolling(x, d).std(ddof=1)

    def ts_min(self, x: pd.DataFrame, d: float) -> pd.DataFrame:
        return self._rolling(x, d).min()

    def ts_max(self, x: pd.DataFrame, d: float) -> pd.DataFrame:
        return self._rolling(x, d).max()

    def ts_rank(self, x: pd.DataFrame, d: float) -> pd.DataFrame:
        return self._rolling(x, d).rank(method="average", pct=True)

    def ts_argmax(self, x: pd.DataFrame, d: float) -> pd.DataFrame:
        return self._rolling(x, d).apply(np.argmax, raw=True) + 1

    def ts_argmin(self, x: pd.DataFrame, d: float) -> pd.DataFrame:
        return self._rolling(x, d).apply(np.argmin, raw=True) + 1

    def ts_prod(self, x: pd.DataFrame, d: float) -> pd.DataFrame:
        return self._rolling(x, d).apply(np.prod, raw=True)

    def correlation(self, x: pd.DataFrame, y: pd.DataFrame, d: float) -> pd.DataFrame:
        d = self.window(d)
        x, y = self.finite(x), self.finite(y)
        pair = x.notna() & y.notna()
        x, y = x.where(pair), y.where(pair)
        sx = x.rolling(d, min_periods=d).std(ddof=0)
        sy = y.rolling(d, min_periods=d).std(ddof=0)
        out = x.rolling(d, min_periods=d).corr(y)
        return self.finite(out).where((sx > 0) & (sy > 0)).clip(-1.0, 1.0)

    def covariance(self, x: pd.DataFrame, y: pd.DataFrame, d: float) -> pd.DataFrame:
        return self._rolling(x, d).cov(self.finite(y), ddof=1)

    def decay_linear(self, x: pd.DataFrame, d: float) -> pd.DataFrame:
        d = self.window(d)
        weights = np.arange(1.0, d + 1.0)
        weights /= weights.sum()
        return self._rolling(x, d).apply(lambda w: np.dot(w, weights), raw=True)


class FormulaSet:

    def __init__(self, panel, eligible):
        self.op = Operators(eligible)
        self.o = panel['open']
        self.h = panel['high']
        self.l = panel['low']
        self.c = panel['close']
        self.v = panel['volume']
        self.vwap = panel['vwap']
        self.r = panel['returns']
        self.cap = panel['cap_proxy']
        self.adv_source = panel['adv_source']
        self._adv = {}

    def adv(self, d):
        d = self.op.window(d)
        if d not in self._adv:
            self._adv[d] = self.op.ts_mean(self.adv_source, d)
        return self._adv[d]

    def a001(self):
        inner = self.op.choose(self.op.lt(self.r, 0), self.op.ts_std(self.r, 20), self.c)
        return self.op.rank(self.op.ts_argmax(self.op.signedpower(inner, 2.0), 5)) - 0.5

    def a002(self):
        return -1 * self.op.correlation(self.op.rank(self.op.delta(self.op.log(self.v), 2)), self.op.rank((self.c - self.o) / self.o), 6)

    def a003(self):
        return -1 * self.op.correlation(self.op.rank(self.o), self.op.rank(self.v), 10)

    def a004(self):
        return -1 * self.op.ts_rank(self.op.rank(self.l), 9)

    def a005(self):
        return self.op.rank(self.o - self.op.ts_mean(self.vwap, 10)) * (-1 * self.op.rank(self.c - self.vwap).abs())

    def a006(self):
        return -1 * self.op.correlation(self.o, self.v, 10)

    def a007(self):
        d7 = self.op.delta(self.c, 7)
        val = -self.op.ts_rank(d7.abs(), 60) * self.op.sign(d7)
        return self.op.choose(self.op.lt(self.adv(20), self.v), val, -1.0)

    def a008(self):
        s = self.op.ts_sum(self.o, 5) * self.op.ts_sum(self.r, 5)
        return -1 * self.op.rank(s - self.op.delay(s, 10))

    def a009(self):
        d1 = self.op.delta(self.c, 1)
        return self.op.choose(self.op.lt(0, self.op.ts_min(d1, 5)), d1, self.op.choose(self.op.lt(self.op.ts_max(d1, 5), 0), d1, -d1))

    def a010(self):
        d1 = self.op.delta(self.c, 1)
        inner = self.op.choose(self.op.lt(0, self.op.ts_min(d1, 4)), d1, self.op.choose(self.op.lt(self.op.ts_max(d1, 4), 0), d1, -d1))
        return self.op.rank(inner)

    def a011(self):
        x = self.vwap - self.c
        return (self.op.rank(self.op.ts_max(x, 3)) + self.op.rank(self.op.ts_min(x, 3))) * self.op.rank(self.op.delta(self.v, 3))

    def a012(self):
        return self.op.sign(self.op.delta(self.v, 1)) * (-1 * self.op.delta(self.c, 1))

    def a013(self):
        return -1 * self.op.rank(self.op.covariance(self.op.rank(self.c), self.op.rank(self.v), 5))

    def a014(self):
        return -1 * self.op.rank(self.op.delta(self.r, 3)) * self.op.correlation(self.o, self.v, 10)

    def a015(self):
        return -1 * self.op.ts_sum(self.op.rank(self.op.correlation(self.op.rank(self.h), self.op.rank(self.v), 3)), 3)

    def a016(self):
        return -1 * self.op.rank(self.op.covariance(self.op.rank(self.h), self.op.rank(self.v), 5))

    def a017(self):
        return -1 * self.op.rank(self.op.ts_rank(self.c, 10)) * self.op.rank(self.op.delta(self.op.delta(self.c, 1), 1)) * self.op.rank(self.op.ts_rank(self.v / self.adv(20), 5))

    def a018(self):
        return -1 * self.op.rank(self.op.ts_std((self.c - self.o).abs(), 5) + (self.c - self.o) + self.op.correlation(self.c, self.o, 10))

    def a019(self):
        return -1 * self.op.sign(self.c - self.op.delay(self.c, 7) + self.op.delta(self.c, 7)) * (1 + self.op.rank(1 + self.op.ts_sum(self.r, 250)))

    def a020(self):
        return -1 * self.op.rank(self.o - self.op.delay(self.h, 1)) * self.op.rank(self.o - self.op.delay(self.c, 1)) * self.op.rank(self.o - self.op.delay(self.l, 1))

    def a021(self):
        (sma8, sma2, sd8) = (self.op.ts_mean(self.c, 8), self.op.ts_mean(self.c, 2), self.op.ts_std(self.c, 8))
        vr = self.v / self.adv(20)
        return self.op.choose(self.op.lt(sma8 + sd8, sma2), -1.0, self.op.choose(self.op.lt(sma2, sma8 - sd8), 1.0, self.op.choose(self.op.le(1.0, vr), 1.0, -1.0)))

    def a022(self):
        return -1 * (self.op.delta(self.op.correlation(self.h, self.v, 5), 5) * self.op.rank(self.op.ts_std(self.c, 20)))

    def a023(self):
        return self.op.choose(self.op.lt(self.op.ts_mean(self.h, 20), self.h), -self.op.delta(self.h, 2), 0.0)

    def a024(self):
        x = self.op.delta(self.op.ts_mean(self.c, 100), 100) / self.op.delay(self.c, 100)
        return self.op.choose(self.op.le(x, 0.05), -(self.c - self.op.ts_min(self.c, 100)), -self.op.delta(self.c, 3))

    def a025(self):
        return self.op.rank(-1 * self.r * self.adv(20) * self.vwap * (self.h - self.c))

    def a026(self):
        return -1 * self.op.ts_max(self.op.correlation(self.op.ts_rank(self.v, 5), self.op.ts_rank(self.h, 5), 5), 3)

    def a027(self):
        x = self.op.rank(self.op.ts_mean(self.op.correlation(self.op.rank(self.v), self.op.rank(self.vwap), 6), 2))
        return self.op.choose(self.op.lt(0.5, x), -1.0, 1.0)

    def a028(self):
        return self.op.scale(self.op.correlation(self.adv(20), self.l, 5) + (self.h + self.l) / 2 - self.c)

    def a029(self):
        p1 = self.op.ts_min(self.op.ts_prod(self.op.rank(self.op.rank(self.op.scale(self.op.log(self.op.ts_sum(self.op.ts_min(self.op.rank(self.op.rank(-1 * self.op.rank(self.op.delta(self.c - 1, 5)))), 2), 1))))), 1), 5)
        p2 = self.op.ts_rank(self.op.delay(-1 * self.r, 6), 5)
        return p1 + p2

    def a030(self):
        s = self.op.sign(self.c - self.op.delay(self.c, 1)) + self.op.sign(self.op.delay(self.c, 1) - self.op.delay(self.c, 2)) + self.op.sign(self.op.delay(self.c, 2) - self.op.delay(self.c, 3))
        return (1.0 - self.op.rank(s)) * self.op.ts_sum(self.v, 5) / self.op.ts_sum(self.v, 20)

    def a031(self):
        p1 = self.op.rank(self.op.rank(self.op.rank(self.op.decay_linear(-1 * self.op.rank(self.op.rank(self.op.delta(self.c, 10))), 10))))
        p2 = self.op.rank(-1 * self.op.delta(self.c, 3))
        p3 = self.op.sign(self.op.scale(self.op.correlation(self.adv(20), self.l, 12)))
        return p1 + p2 + p3

    def a032(self):
        return self.op.scale(self.op.ts_mean(self.c, 7) - self.c) + 20 * self.op.scale(self.op.correlation(self.vwap, self.op.delay(self.c, 5), 230))

    def a033(self):
        return self.op.rank(-1 * (1 - self.o / self.c))

    def a034(self):
        return self.op.rank(1 - self.op.rank(self.op.ts_std(self.r, 2) / self.op.ts_std(self.r, 5)) + (1 - self.op.rank(self.op.delta(self.c, 1))))

    def a035(self):
        return self.op.ts_rank(self.v, 32) * (1 - self.op.ts_rank(self.c + self.h - self.l, 16)) * (1 - self.op.ts_rank(self.r, 32))

    def a036(self):
        return 2.21 * self.op.rank(self.op.correlation(self.c - self.o, self.op.delay(self.v, 1), 15)) + 0.7 * self.op.rank(self.o - self.c) + 0.73 * self.op.rank(self.op.ts_rank(self.op.delay(-1 * self.r, 6), 5)) + self.op.rank(self.op.correlation(self.vwap, self.adv(20), 6).abs()) + 0.6 * self.op.rank((self.op.ts_mean(self.c, 200) - self.o) * (self.c - self.o))

    def a037(self):
        return self.op.rank(self.op.correlation(self.op.delay(self.o - self.c, 1), self.c, 200)) + self.op.rank(self.o - self.c)

    def a038(self):
        return -1 * self.op.rank(self.op.ts_rank(self.c, 10)) * self.op.rank(self.c / self.o)

    def a039(self):
        return -1 * self.op.rank(self.op.delta(self.c, 7) * (1 - self.op.rank(self.op.decay_linear(self.v / self.adv(20), 9)))) * (1 + self.op.rank(self.op.ts_sum(self.r, 250)))

    def a040(self):
        return -1 * self.op.rank(self.op.ts_std(self.h, 10)) * self.op.correlation(self.h, self.v, 10)

    def a041(self):
        return self.op.power(self.h * self.l, 0.5) - self.vwap

    def a042(self):
        return self.op.rank(self.vwap - self.c) / self.op.rank(self.vwap + self.c)

    def a043(self):
        return self.op.ts_rank(self.v / self.adv(20), 20) * self.op.ts_rank(-1 * self.op.delta(self.c, 7), 8)

    def a044(self):
        return -1 * self.op.correlation(self.h, self.op.rank(self.v), 5)

    def a045(self):
        return -1 * (self.op.rank(self.op.ts_mean(self.op.delay(self.c, 5), 20)) * self.op.correlation(self.c, self.v, 2) * self.op.rank(self.op.correlation(self.op.ts_sum(self.c, 5), self.op.ts_sum(self.c, 20), 2)))

    def a046(self):
        x = (self.op.delay(self.c, 20) - self.op.delay(self.c, 10)) / 10 - (self.op.delay(self.c, 10) - self.c) / 10
        return self.op.choose(self.op.lt(0.25, x), -1.0, self.op.choose(self.op.lt(x, 0.0), 1.0, -self.op.delta(self.c, 1)))

    def a047(self):
        return self.op.rank(1 / self.c) * self.v / self.adv(20) * (self.h * self.op.rank(self.h - self.c) / self.op.ts_mean(self.h, 5)) - self.op.rank(self.vwap - self.op.delay(self.vwap, 5))

    def a048(self):
        num = self.op.correlation(self.op.delta(self.c, 1), self.op.delta(self.op.delay(self.c, 1), 1), 250) * self.op.delta(self.c, 1) / self.c
        den = self.op.ts_sum(self.op.power(self.op.delta(self.c, 1) / self.op.delay(self.c, 1), 2), 250)
        return self.op.indneutralize(num) / den

    def a049(self):
        x = (self.op.delay(self.c, 20) - self.op.delay(self.c, 10)) / 10 - (self.op.delay(self.c, 10) - self.c) / 10
        return self.op.choose(self.op.lt(x, -0.1), 1.0, -self.op.delta(self.c, 1))

    def a050(self):
        return -1 * self.op.ts_max(self.op.rank(self.op.correlation(self.op.rank(self.v), self.op.rank(self.vwap), 5)), 5)

    def a051(self):
        x = (self.op.delay(self.c, 20) - self.op.delay(self.c, 10)) / 10 - (self.op.delay(self.c, 10) - self.c) / 10
        return self.op.choose(self.op.lt(x, -0.05), 1.0, -self.op.delta(self.c, 1))

    def a052(self):
        return (-1 * self.op.ts_min(self.l, 5) + self.op.delay(self.op.ts_min(self.l, 5), 5)) * self.op.rank((self.op.ts_sum(self.r, 240) - self.op.ts_sum(self.r, 20)) / 220) * self.op.ts_rank(self.v, 5)

    def a053(self):
        x = (self.c - self.l - (self.h - self.c)) / (self.c - self.l).replace(0, np.nan)
        return -1 * self.op.delta(x, 9)

    def a054(self):
        return -1 * ((self.l - self.c) * self.op.power(self.o, 5)) / ((self.l - self.h).replace(0, np.nan) * self.op.power(self.c, 5))

    def a055(self):
        x = (self.c - self.op.ts_min(self.l, 12)) / (self.op.ts_max(self.h, 12) - self.op.ts_min(self.l, 12)).replace(0, np.nan)
        return -1 * self.op.correlation(self.op.rank(x), self.op.rank(self.v), 6)

    def a056(self):
        return 0 - 1 * (self.op.rank(self.op.ts_sum(self.r, 10) / self.op.ts_sum(self.op.ts_sum(self.r, 2), 3)) * self.op.rank(self.r * self.cap))

    def a057(self):
        return 0 - 1 * ((self.c - self.vwap) / self.op.decay_linear(self.op.rank(self.op.ts_argmax(self.c, 30)), 2))

    def a058(self):
        return -1 * self.op.ts_rank(self.op.decay_linear(self.op.correlation(self.op.indneutralize(self.vwap), self.v, 3.92795), 7.89291), 5.50322)

    def a059(self):
        x = self.vwap * 0.728317 + self.vwap * (1 - 0.728317)
        return -1 * self.op.ts_rank(self.op.decay_linear(self.op.correlation(self.op.indneutralize(x), self.v, 4.25197), 16.2289), 8.19648)

    def a060(self):
        x = (self.c - self.l - (self.h - self.c)) / (self.h - self.l).replace(0, np.nan)
        return 0 - 1 * (2 * self.op.scale(self.op.rank(x * self.v)) - self.op.scale(self.op.rank(self.op.ts_argmax(self.c, 10))))

    def a061(self):
        return self.op.lt(self.op.rank(self.vwap - self.op.ts_min(self.vwap, 16.1219)), self.op.rank(self.op.correlation(self.vwap, self.adv(180), 17.9282)))

    def a062(self):
        inner = self.op.lt(self.op.rank(self.o) + self.op.rank(self.o), self.op.rank((self.h + self.l) / 2) + self.op.rank(self.h))
        return -1 * self.op.lt(self.op.rank(self.op.correlation(self.vwap, self.op.ts_sum(self.adv(20), 22.4101), 9.91009)), self.op.rank(inner))

    def a063(self):
        p1 = self.op.rank(self.op.decay_linear(self.op.delta(self.op.indneutralize(self.c), 2.25164), 8.22237))
        x = self.vwap * 0.318108 + self.o * (1 - 0.318108)
        p2 = self.op.rank(self.op.decay_linear(self.op.correlation(x, self.op.ts_sum(self.adv(180), 37.2467), 13.557), 12.2883))
        return (p1 - p2) * -1

    def a064(self):
        x = self.o * 0.178404 + self.l * (1 - 0.178404)
        y = (self.h + self.l) / 2 * 0.178404 + self.vwap * (1 - 0.178404)
        return -1 * self.op.lt(self.op.rank(self.op.correlation(self.op.ts_sum(x, 12.7054), self.op.ts_sum(self.adv(120), 12.7054), 16.6208)), self.op.rank(self.op.delta(y, 3.69741)))

    def a065(self):
        x = self.o * 0.00817205 + self.vwap * (1 - 0.00817205)
        return -1 * self.op.lt(self.op.rank(self.op.correlation(x, self.op.ts_sum(self.adv(60), 8.6911), 6.40374)), self.op.rank(self.o - self.op.ts_min(self.o, 13.635)))

    def a066(self):
        p1 = self.op.rank(self.op.decay_linear(self.op.delta(self.vwap, 3.51013), 7.23052))
        y = (self.l - self.vwap) / (self.o - (self.h + self.l) / 2).replace(0, np.nan)
        p2 = self.op.ts_rank(self.op.decay_linear(y, 11.4157), 6.72611)
        return (p1 + p2) * -1

    def a067(self):
        p1 = self.op.rank(self.h - self.op.ts_min(self.h, 2.14593))
        p2 = self.op.rank(self.op.correlation(self.op.indneutralize(self.vwap), self.op.indneutralize(self.adv(20)), 6.02936))
        return self.op.power(p1, p2) * -1

    def a068(self):
        x = self.c * 0.518371 + self.l * (1 - 0.518371)
        return -1 * self.op.lt(self.op.ts_rank(self.op.correlation(self.op.rank(self.h), self.op.rank(self.adv(15)), 8.91644), 13.9333), self.op.rank(self.op.delta(x, 1.06157)))

    def a069(self):
        p1 = self.op.rank(self.op.ts_max(self.op.delta(self.op.indneutralize(self.vwap), 2.72412), 4.79344))
        x = self.c * 0.490655 + self.vwap * (1 - 0.490655)
        p2 = self.op.ts_rank(self.op.correlation(x, self.adv(20), 4.92416), 9.0615)
        return self.op.power(p1, p2) * -1

    def a070(self):
        p1 = self.op.rank(self.op.delta(self.vwap, 1.29456))
        p2 = self.op.ts_rank(self.op.correlation(self.op.indneutralize(self.c), self.adv(50), 17.8256), 17.9171)
        return self.op.power(p1, p2) * -1

    def a071(self):
        p1 = self.op.ts_rank(self.op.decay_linear(self.op.correlation(self.op.ts_rank(self.c, 3.43976), self.op.ts_rank(self.adv(180), 12.0647), 18.0175), 4.20501), 15.6948)
        p2 = self.op.ts_rank(self.op.decay_linear(self.op.power(self.op.rank(self.l + self.o - (self.vwap + self.vwap)), 2), 16.4662), 4.4388)
        return self.op.emax(p1, p2)

    def a072(self):
        p1 = self.op.rank(self.op.decay_linear(self.op.correlation((self.h + self.l) / 2, self.adv(40), 8.93345), 10.1519))
        p2 = self.op.rank(self.op.decay_linear(self.op.correlation(self.op.ts_rank(self.vwap, 3.72469), self.op.ts_rank(self.v, 18.5188), 6.86671), 2.95011))
        return p1 / p2.replace(0, np.nan)

    def a073(self):
        p1 = self.op.rank(self.op.decay_linear(self.op.delta(self.vwap, 4.72775), 2.91864))
        x = self.o * 0.147155 + self.l * (1 - 0.147155)
        p2 = self.op.ts_rank(self.op.decay_linear(self.op.delta(x, 2.03608) / x * -1, 3.33829), 16.7411)
        return self.op.emax(p1, p2) * -1

    def a074(self):
        x = self.h * 0.0261661 + self.vwap * (1 - 0.0261661)
        return -1 * self.op.lt(self.op.rank(self.op.correlation(self.c, self.op.ts_sum(self.adv(30), 37.4843), 15.1365)), self.op.rank(self.op.correlation(self.op.rank(x), self.op.rank(self.v), 11.4791)))

    def a075(self):
        return self.op.lt(self.op.rank(self.op.correlation(self.vwap, self.v, 4.24304)), self.op.rank(self.op.correlation(self.op.rank(self.l), self.op.rank(self.adv(50)), 12.4413)))

    def a076(self):
        p1 = self.op.rank(self.op.decay_linear(self.op.delta(self.vwap, 1.24383), 11.8259))
        p2 = self.op.ts_rank(self.op.decay_linear(self.op.ts_rank(self.op.correlation(self.op.indneutralize(self.l), self.adv(81), 8.14941), 19.569), 17.1543), 19.383)
        return self.op.emax(p1, p2) * -1

    def a077(self):
        p1 = self.op.rank(self.op.decay_linear((self.h + self.l) / 2 + self.h - (self.vwap + self.h), 20.0451))
        p2 = self.op.rank(self.op.decay_linear(self.op.correlation((self.h + self.l) / 2, self.adv(40), 3.1614), 5.64125))
        return self.op.emin(p1, p2)

    def a078(self):
        x = self.l * 0.352233 + self.vwap * (1 - 0.352233)
        p1 = self.op.rank(self.op.correlation(self.op.ts_sum(x, 19.7428), self.op.ts_sum(self.adv(40), 19.7428), 6.83313))
        p2 = self.op.rank(self.op.correlation(self.op.rank(self.vwap), self.op.rank(self.v), 5.77492))
        return self.op.power(p1, p2)

    def a079(self):
        x = self.c * 0.60733 + self.o * (1 - 0.60733)
        return self.op.lt(self.op.rank(self.op.delta(self.op.indneutralize(x), 1.23438)), self.op.rank(self.op.correlation(self.op.ts_rank(self.vwap, 3.60973), self.op.ts_rank(self.adv(150), 9.18637), 14.6644)))

    def a080(self):
        x = self.o * 0.868128 + self.h * (1 - 0.868128)
        p1 = self.op.rank(self.op.sign(self.op.delta(self.op.indneutralize(x), 4.04545)))
        p2 = self.op.ts_rank(self.op.correlation(self.h, self.adv(10), 5.11456), 5.53756)
        return self.op.power(p1, p2) * -1

    def a081(self):
        p1 = self.op.rank(self.op.log(self.op.ts_prod(self.op.rank(self.op.power(self.op.rank(self.op.correlation(self.vwap, self.op.ts_sum(self.adv(10), 49.6054), 8.47743)), 4)), 14.9655)))
        p2 = self.op.rank(self.op.correlation(self.op.rank(self.vwap), self.op.rank(self.v), 5.07914))
        return -1 * self.op.lt(p1, p2)

    def a082(self):
        p1 = self.op.rank(self.op.decay_linear(self.op.delta(self.o, 1.46063), 14.8717))
        x = self.o * 0.634196 + self.o * (1 - 0.634196)
        p2 = self.op.ts_rank(self.op.decay_linear(self.op.correlation(self.op.indneutralize(self.v), x, 17.4842), 6.92131), 13.4283)
        return self.op.emin(p1, p2) * -1

    def a083(self):
        x = (self.h - self.l) / self.op.ts_mean(self.c, 5)
        num = self.op.rank(self.op.delay(x, 2)) * self.op.rank(self.op.rank(self.v))
        den = x / (self.vwap - self.c).replace(0, np.nan)
        return num / den.replace(0, np.nan)

    def a084(self):
        return self.op.signedpower(self.op.ts_rank(self.vwap - self.op.ts_max(self.vwap, 15.3217), 20.7127), self.op.delta(self.c, 4.96796))

    def a085(self):
        x = self.h * 0.876703 + self.c * (1 - 0.876703)
        p1 = self.op.rank(self.op.correlation(x, self.adv(30), 9.61331))
        p2 = self.op.rank(self.op.correlation(self.op.ts_rank((self.h + self.l) / 2, 3.70596), self.op.ts_rank(self.v, 10.1595), 7.11408))
        return self.op.power(p1, p2)

    def a086(self):
        return -1 * self.op.lt(self.op.ts_rank(self.op.correlation(self.c, self.op.ts_sum(self.adv(20), 14.7444), 6.00049), 20.4195), self.op.rank(self.o + self.c - (self.vwap + self.o)))

    def a087(self):
        x = self.c * 0.369701 + self.vwap * (1 - 0.369701)
        p1 = self.op.rank(self.op.decay_linear(self.op.delta(x, 1.91233), 2.65461))
        p2 = self.op.ts_rank(self.op.decay_linear(self.op.correlation(self.op.indneutralize(self.adv(81)), self.c, 13.4132).abs(), 4.89768), 14.4535)
        return self.op.emax(p1, p2) * -1

    def a088(self):
        return self.op.emin(*self.alpha088_branches())

    def a089(self):
        p1 = self.op.ts_rank(self.op.decay_linear(self.op.correlation(self.l, self.adv(10), 6.94279), 5.51607), 3.79744)
        p2 = self.op.ts_rank(self.op.decay_linear(self.op.delta(self.op.indneutralize(self.vwap), 3.48158), 10.1466), 15.3012)
        return p1 - p2

    def a090(self):
        p1 = self.op.rank(self.c - self.op.ts_max(self.c, 4.66719))
        p2 = self.op.ts_rank(self.op.correlation(self.op.indneutralize(self.adv(40)), self.l, 5.38375), 3.21856)
        return self.op.power(p1, p2) * -1

    def a091(self):
        p1 = self.op.ts_rank(self.op.decay_linear(self.op.decay_linear(self.op.correlation(self.op.indneutralize(self.c), self.v, 9.74928), 16.398), 3.83219), 4.8667)
        p2 = self.op.rank(self.op.decay_linear(self.op.correlation(self.vwap, self.adv(30), 4.01303), 2.6809))
        return (p1 - p2) * -1

    def a092(self):
        cond = self.op.lt((self.h + self.l) / 2 + self.c, self.l + self.o)
        p1 = self.op.ts_rank(self.op.decay_linear(cond, 14.7221), 18.8683)
        p2 = self.op.ts_rank(self.op.decay_linear(self.op.correlation(self.op.rank(self.l), self.op.rank(self.adv(30)), 7.58555), 6.94024), 6.80584)
        return self.op.emin(p1, p2)

    def a093(self):
        p1 = self.op.ts_rank(self.op.decay_linear(self.op.correlation(self.op.indneutralize(self.vwap), self.adv(81), 17.4193), 19.848), 7.54455)
        x = self.c * 0.524434 + self.vwap * (1 - 0.524434)
        p2 = self.op.rank(self.op.decay_linear(self.op.delta(x, 2.77377), 16.2664))
        return p1 / p2.replace(0, np.nan)

    def a094(self):
        p1 = self.op.rank(self.vwap - self.op.ts_min(self.vwap, 11.5783))
        p2 = self.op.ts_rank(self.op.correlation(self.op.ts_rank(self.vwap, 19.6462), self.op.ts_rank(self.adv(60), 4.02992), 18.0926), 2.70756)
        return self.op.power(p1, p2) * -1

    def a095(self):
        p1 = self.op.rank(self.o - self.op.ts_min(self.o, 12.4105))
        p2 = self.op.ts_rank(self.op.power(self.op.rank(self.op.correlation(self.op.ts_sum((self.h + self.l) / 2, 19.1351), self.op.ts_sum(self.adv(40), 19.1351), 12.8742)), 5), 11.7584)
        return self.op.lt(p1, p2)

    def a096(self):
        p1 = self.op.ts_rank(self.op.decay_linear(self.op.correlation(self.op.rank(self.vwap), self.op.rank(self.v), 3.83878), 4.16783), 8.38151)
        p2 = self.op.ts_rank(self.op.decay_linear(self.op.ts_argmax(self.op.correlation(self.op.ts_rank(self.c, 7.45404), self.op.ts_rank(self.adv(60), 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)
        return self.op.emax(p1, p2) * -1

    def a097(self):
        x = self.l * 0.721001 + self.vwap * (1 - 0.721001)
        p1 = self.op.rank(self.op.decay_linear(self.op.delta(self.op.indneutralize(x), 3.3705), 20.4523))
        p2 = self.op.ts_rank(self.op.decay_linear(self.op.ts_rank(self.op.correlation(self.op.ts_rank(self.l, 7.87871), self.op.ts_rank(self.adv(60), 17.255), 4.97547), 18.5925), 15.7152), 6.71659)
        return (p1 - p2) * -1

    def a098(self):
        p1 = self.op.rank(self.op.decay_linear(self.op.correlation(self.vwap, self.op.ts_sum(self.adv(5), 26.4719), 4.58418), 7.18088))
        p2 = self.op.rank(self.op.decay_linear(self.op.ts_rank(self.op.ts_argmin(self.op.correlation(self.op.rank(self.o), self.op.rank(self.adv(15)), 20.8187), 8.62571), 6.95668), 8.07206))
        return p1 - p2

    def a099(self):
        p1 = self.op.rank(self.op.correlation(self.op.ts_sum((self.h + self.l) / 2, 19.8975), self.op.ts_sum(self.adv(60), 19.8975), 8.8136))
        p2 = self.op.rank(self.op.correlation(self.l, self.v, 6.28259))
        return -1 * self.op.lt(p1, p2)

    def a100(self):
        x = (self.c - self.l - (self.h - self.c)) / (self.h - self.l).replace(0, np.nan)
        p1 = 1.5 * self.op.scale(self.op.indneutralize(self.op.indneutralize(self.op.rank(x * self.v))))
        p2 = self.op.scale(self.op.indneutralize(self.op.correlation(self.c, self.op.rank(self.adv(20)), 5) - self.op.rank(self.op.ts_argmin(self.c, 30))))
        return 0 - 1 * ((p1 - p2) * (self.v / self.adv(20)))

    def a101(self):
        return (self.c - self.o) / (self.h - self.l + 0.001)

    def alpha088_branches(self):
        p1 = self.op.rank(self.op.decay_linear(self.op.rank(self.o) + self.op.rank(self.l) - (self.op.rank(self.h) + self.op.rank(self.c)), 8.06882))
        p2 = self.op.ts_rank(self.op.decay_linear(self.op.correlation(self.op.ts_rank(self.c, 8.44728), self.op.ts_rank(self.adv(60), 20.6966), 8.01266), 6.65053), 2.61957)
        return (p1, p2)


def _prepare_panel(raw: dict[str, pd.DataFrame], eligible: pd.DataFrame,
                   mode: str) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Validate a regular grid and construct inputs without hiding raw history."""
    required = ("open", "high", "low", "close", "volume", "quote_volume")
    missing = set(required) - set(raw)
    if missing:
        raise ValueError(f"missing raw fields: {sorted(missing)}")
    if mode not in {"paper", "normalized"}:
        raise ValueError("mode must be 'paper' or 'normalized'")
    template = raw["close"]
    if not isinstance(template, pd.DataFrame) or not isinstance(template.index, pd.DatetimeIndex):
        raise TypeError("raw fields must be DataFrames with a DatetimeIndex")
    idx = template.index
    if not idx.is_unique or not idx.is_monotonic_increasing or not template.columns.is_unique:
        raise ValueError("timestamps and symbols must be unique; timestamps must be sorted")
    if len(idx) < 2:
        raise ValueError("at least two timestamps are needed to establish the bar grid")
    steps = np.diff(idx.asi8)
    if not np.all(steps == steps[0]):
        raise ValueError("use a regular grid; keep missing bars as NA rows")
    step_ns = int(steps[0])
    day_ns = int(pd.Timedelta(days=1).value)
    if step_ns not in (day_ns, int(pd.Timedelta(hours=1).value)):
        raise ValueError("v2 supports daily or hourly grids")
    bars_per_day = day_ns // step_ns
    if not eligible.index.equals(idx) or not eligible.columns.equals(template.columns):
        raise ValueError("eligible axes must exactly match raw data")
    clean = {}
    invalid = {}
    for name in required:
        value = raw[name]
        if not isinstance(value, pd.DataFrame) or not value.index.equals(idx) or not value.columns.equals(template.columns):
            raise ValueError(f"raw field {name!r} has inconsistent axes")
        value = Operators.finite(value.astype(float))
        ok = value > 0 if name in ("open", "high", "low", "close") else value >= 0
        invalid[name] = int((value.notna() & ~ok).to_numpy().sum())
        clean[name] = value.where(ok)
    c, base, quote = clean["close"], clean["volume"], clean["quote_volume"]
    vwap = Operators.finite(quote / base.replace(0, np.nan)).where(lambda x: x > 0)
    panel = {name: clean[name] for name in ("open", "high", "low", "close", "volume")}
    panel.update(vwap=vwap, returns=c / c.shift(1) - 1.0,
                 adv_source=quote,
                 cap_proxy=quote.rolling(30, min_periods=30).mean().shift(1))
    norm = None
    if mode == "normalized":
        window = 250 * bars_per_day
        pscale = c.rolling(window, min_periods=window).mean().shift(1)
        bscale = base.rolling(window, min_periods=window).mean().shift(1).replace(0, np.nan)
        qscale = quote.rolling(window, min_periods=window).mean().shift(1).replace(0, np.nan)
        for name in ("open", "high", "low", "close", "vwap"):
            panel[name] = panel[name] / pscale
        panel["volume"] = base / bscale
        panel["adv_source"] = quote / qscale
        norm = {"calendar_days": 250, "bars": window, "lag_bars": 1,
                "min_periods": window, "volume_source": "base_volume",
                "adv_source": "quote_volume", "is_legacy_normalization": False}
    return panel, {"mode": mode, "bars_per_day": bars_per_day,
                   "normalization": norm, "invalid_nonpositive_prices_or_negative_volumes": invalid}


def compute_factors(raw: dict[str, pd.DataFrame], eligible: pd.DataFrame,
                    mode: str = "paper") -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Return all 101 factor frames plus JSON-serializable audit metadata.

    ``raw`` contains OHLC, base-unit ``volume``, and USD/USDT ``quote_volume``
    on identical daily/hourly grids. ``eligible`` controls ALL current-row
    cross-sectional operators and final outputs, not raw historical lookbacks.
    No functions alter inputs, legacy modules, or shared global state.
    """
    panel, preparation = _prepare_panel(raw, eligible, mode)
    eligible = eligible.fillna(False).astype(bool)
    a = FormulaSet(panel, eligible)
    out = {}
    per_factor = {}
    denominator = int(eligible.to_numpy().sum())
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for number in range(1, 102):
            name = f"alpha{number:03d}"
            try:
                result = getattr(a, f"a{number:03d}")()
            except Exception as exc:
                raise RuntimeError(f"{name} failed under {mode} input semantics") from exc
            result = Operators.finite(result).where(eligible)
            out[name] = result
            substitutions = []
            if number in IND_ALPHAS:
                substitutions.append("PIT sector/industry/subindustry -> eligible-market cross-sectional demean")
            if number in CAP_ALPHAS:
                substitutions.append("market cap -> lagged 30-bar raw quote-volume mean")
            valid = int(result.notna().to_numpy().sum())
            per_factor[name] = {"number": number,
                                "family": "input_proxy" if substitutions else "price_volume",
                                "substitutions": substitutions,
                                "comparison_output": number in COMPARISON_ALPHAS,
                                "valid_cells": valid,
                                "eligible_cells": denominator,
                                "coverage": valid / denominator if denominator else None}
        p1, p2 = a.alpha088_branches()
    both = eligible & p1.notna() & p2.notna()
    counts = {"both_valid": int(both.to_numpy().sum()),
              "left_strictly_smaller": int((both & (p1 < p2)).to_numpy().sum()),
              "right_strictly_smaller": int((both & (p2 < p1)).to_numpy().sum()),
              "ties": int((both & (p1 == p2)).to_numpy().sum())}
    metadata = {
        "version": VERSION,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **preparation,
        "paper_source": {"url": PAPER_URL, "operator_pages": [15, 16],
                         "rank_scale_explicit_in_paper": False,
                         "argextrema_orientation_explicit_in_paper": False,
                         "returns_simple_vs_log_explicit_in_paper": False},
        "inputs": {
            "prices": "raw positive OHLC" if mode == "paper" else "OHLC / lagged full 250-day raw close mean",
            "volume": "base units" if mode == "paper" else "base volume / lagged full 250-day base-volume mean",
            "adv": "rolling mean of quote-volume dollars" if mode == "paper" else "rolling mean of relative quote-volume activity",
            "adv_includes_current_bar": True,
            "returns": "raw close / prior raw close - 1; no filling; unaffected by normalization",
            "vwap": "quote_volume / base_volume; normalized with prices in normalized mode",
            "cap_proxy": "raw quote_volume rolling(30,min_periods=30).mean().shift(1); bar-based",
            "volume_adv_unit_mismatch": mode == "paper",
        },
        "operators": {
            "rank": "eligible-only average-tie ascending rank / valid cross-sectional count",
            "ts_rank": "full-window average-tie ascending rank of latest observation / window length",
            "rank_scale_status": "v2 percentile convention; paper does not uniquely specify rank scale",
            "ts_argmax_argmin": "1-based from oldest to newest; earliest occurrence wins ties",
            "scale": "eligible-only L1 norm: x*a / sum(abs(x)); no demeaning",
            "std_cov_ddof": 1,
            "correlation": "complete paired window; zero variance -> NA; roundoff clipped to [-1,1]",
            "noninteger_windows": "floor; formula windows stay in bars on hourly grids",
            "time_series_missing": "complete finite window required; no forward/back/zero fill",
            "ternary_missing": "missing condition -> NA; selected branch determines value availability",
            "decay_linear": "oldest weight 1, latest weight d; divided by d*(d+1)/2; complete window",
            "eligibility": "mask inside every rank/scale/neutralize and final output; retain raw history",
            "signedpower": "literal Appendix power x**a (formula uses have nonnegative bases)",
        },
        "n_factors": len(out),
        "n_without_sector_or_cap_proxy": 82,
        "n_with_sector_or_cap_proxy": 19,
        "alpha088_branch_diagnostic": counts,
        "factors": per_factor,
        "limitations": [
            "paper mode is an explicit implementation convention, not a verified proprietary reproduction",
            "USD/USDT quote turnover is used as dollar-volume approximation",
            "19 industry/cap factors are clearly marked substitutes and must be reported separately",
            "nominal base-volume/dollar-ADV ratios retain the paper's stated units; no silent unit repair",
        ],
    }
    return out, metadata
