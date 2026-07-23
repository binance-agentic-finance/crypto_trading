"""Evolutionary Strategy Discovery (GA + LLM).

GA + LLM 自動挖掘日內交易策略。AI（在 CodeMax 窗口內）負責產生初始種群、
mutation 與解讀結果；本 package 負責 deterministic 部分：genome 序列化、
回測橋接、fitness、selection、crossover、CLI、export。

完整規格見:
    .codemax/spec-workflow/specs/evolutionary-strategy-discovery/SPEC.md
"""

from .genome import Factor, Filter, StrategyGenome
from .population import Population

__all__ = ["Factor", "Filter", "StrategyGenome", "Population"]
