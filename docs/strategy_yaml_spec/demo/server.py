"""Demo server: 自然語言 → (LLM/LiteLLM) → YAML → 標準訊號 / 回測.

這是一個把自然語言轉成交易或選幣 YAML，再交給確定性 runtime 執行的展示,不是 agent。
瀏覽器 → 本後端 → LiteLLM(OpenAI 相容 /chat/completions)。API key 只在這次請求中轉送。

啟動:
    PYTHONPATH=<repo_root> <venv>/bin/python docs/strategy_yaml_spec/demo/server.py
    # 然後開 http://127.0.0.1:8799

自然語言預設不會送到外部 LLM。僅限已明確同意送出的非敏感測試內容，才以
``CYQNT_DEMO_ALLOW_EXTERNAL_LLM=1`` 啟動；私有對話語料必須走本地 frozen
evaluator，不可走這條 demo 路由。
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# --- 讓後端找得到 cyqnt_trd(repo root = 這個檔往上三層)---
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
SPEC_DIR = HERE.parents[1]                      # docs/strategy_yaml_spec
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import requests
import yaml

from cyqnt_trd.standard_bot.yaml_pipeline import build_make_signals, validate_spec
from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import resolve_block
from cyqnt_trd.standard_bot.simulation.vectorized_backtest import run_vectorized_backtest

# Intent classification and post-generation reconciliation live in the package,
# not in this file: the conversion pipeline (tools/nl2yaml) needs the same two
# halves, and a tool must not import a module out of docs/. Re-exported here so
# this module stays the one place the demo routes and its tests look them up.
from cyqnt_trd.standard_bot.yaml_pipeline.intent import (  # noqa: F401
    UNSUPPORTED_SELECTION_SOURCES,
    IntentDecision,
    candidate_trade_plan,
    classify_request,
    generated_strategy_kind,
    infer_strategy_kind,
    reconcile_intent,
)

PORT = 8799
SCHEMA_PATH = SPEC_DIR / "strategy.schema.yaml"
FIXTURE_DIR = REPO_ROOT / "tests" / "blocks" / "fixtures"

# The demo is a local convenience UI, not a sanctioned private-corpus egress
# path.  A user must opt in at process launch before this server sends *any* NL
# text to a user-supplied OpenAI-compatible endpoint.  Do not replace this guard
# with redaction: arbitrary free text cannot be reliably made safe by a regex.
EXTERNAL_LLM_OPT_IN_ENV = "CYQNT_DEMO_ALLOW_EXTERNAL_LLM"


class ExternalLLMDisabled(RuntimeError):
    """Raised before a request body is built or sent to an external endpoint."""


def _external_llm_enabled() -> bool:
    return os.environ.get(EXTERNAL_LLM_OPT_IN_ENV, "").strip().lower() in {
        "1", "true", "yes",
    }


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

def _prompt_signature(ref: str) -> str:
    """Render parameter names from the callable the YAML runtime will invoke.

    The prompt used to duplicate these names by hand and taught the model
    ``fast_period`` while the running MACD block accepts ``fast``. Deriving the
    hint from :func:`resolve_block` makes prompt drift a testable contract bug
    instead of a model hallucination.
    """
    import inspect

    parameters = inspect.signature(resolve_block(ref)).parameters.values()
    names = [item.name for item in parameters]
    return "%s(%s)" % (ref, ",".join(names))


def _build_blocks_cheatsheet() -> str:
    indicators = [
        "indicators.ema", "indicators.sma", "indicators.rsi",
        "indicators.atr", "indicators.adx", "indicators.macd", "indicators.vwap",
    ]
    conditions = [
        "conditions.ma_cross_above", "conditions.ma_cross_below",
        "conditions.rsi_in_range", "conditions.rsi_overbought",
        "conditions.rsi_oversold", "conditions.adx_trending",
        "conditions.breakout_high", "conditions.breakout_low",
        "conditions.macd_golden_cross", "conditions.macd_death_cross",
    ]
    return (
        "可用 indicators(input=close 或自動 df;參數名取自實際 Blocks):\n  "
        + " ".join(_prompt_signature(ref) for ref in indicators)
        + "\n  indicators.adx 用 output:0 取 adx;indicators.macd 用 output:0/1/2"
          " 取 macd線/signal線/hist。\n"
        + "可用 conditions(回傳 bool):\n  "
        + " ".join(_prompt_signature(ref) for ref in conditions)
        + "\n組合器(可任意巢狀):{all_of:[...]} {any_of:[...]} {not:<node>} "
          "葉節點:{cond:\"conditions.xxx\",args:[...],params:{...}}\n"
        + "出場 risk.exit.type:pct_stop_tp{stop_pct,tp_pct,max_bars} / "
          "atr_stop_tp{atr_period,stop_mult,tp_mult,max_bars} / "
          "atr_trailing_stop{atr_period,trail_mult,max_bars} / "
          "time_only{max_bars} / opposite_signal{max_bars}\n"
    )


BLOCKS_CHEATSHEET = _build_blocks_cheatsheet()

EXAMPLE_YAML = """\
spec_version: "1.0"
target: standard_bot
strategy:
  id: btc_ema_rsi_1h
  description: "EMA 交叉 + RSI 過濾"
run:
  mode: backtest
data:
  symbol: BTCUSDT
  market_type: futures
  primary: { interval: "1h", poll_interval: 3570 }
  source: { type: binance_rest }
signals:
  indicators:
    ema_fast: { block: indicators.ema, input: close, params: { period: 12 } }
    ema_slow: { block: indicators.ema, input: close, params: { period: 26 } }
    rsi14:    { block: indicators.rsi, input: close, params: { period: 14 } }
  entry:
    long:
      all_of:
        - { cond: conditions.ma_cross_above, args: [ema_fast, ema_slow] }
        - not: { cond: conditions.rsi_overbought, args: [rsi14], params: { threshold: 75 } }
    short:
      all_of:
        - { cond: conditions.ma_cross_below, args: [ema_fast, ema_slow] }
        - not: { cond: conditions.rsi_oversold, args: [rsi14], params: { threshold: 25 } }
sizing: { size: 0.95 }
risk:
  exit: { type: pct_stop_tp, stop_pct: 0.02, tp_pct: 0.04, max_bars: 96 }
  fees: { commission_bps: 4.0, slippage_bps: 2.0 }
backtest: { initial_capital: 10000, execution_model: next_bar_open }
"""


SELECTION_EXAMPLE_YAML = """\
spec_version: "1.0"
target: standard_bot
strategy:
  id: square_news_buzz_selector
  description: "依 Binance Square 最近提及量挑選候選幣"
run:
  mode: backtest
data:
  symbol: BTCUSDT
  market_type: futures
  primary: { interval: "1h" }
selection:
  universe:
    - block: universe.filter_quote_volume
      params: { min_quote_volume: 100000000 }
    - block: universe.augment_with_news
      with: [ticker_rank]
  score: news_mention_count
  top_k: 5
  min_score: 1.0
  dedupe_by: base_asset
"""

LIQUIDITY_SELECTION_EXAMPLE_YAML = """\
spec_version: "1.0"
target: standard_bot
strategy:
  id: quote_volume_selector
  description: "依 24 小時 USDT 成交額挑選候選幣"
run:
  mode: backtest
data:
  symbol: BTCUSDT
  market_type: futures
  primary: { interval: "1h" }
selection:
  universe:
    - block: universe.filter_quote_volume
      params: { min_quote_volume: 100000000 }
  score: quoteVolume
  top_k: 5
  min_score: 1.0
  dedupe_by: base_asset
"""


OPEN_INTEREST_SELECTION_EXAMPLE_YAML = """\
spec_version: "1.0"
target: standard_bot
strategy:
  id: open_interest_selector
  description: "依目前跨幣別合約未平倉名目金額挑選候選幣"
run:
  mode: backtest
data:
  symbol: BTCUSDT
  market_type: futures
  primary: { interval: "1h" }
selection:
  universe:
    - block: universe.filter_quote_volume
      params: { min_quote_volume: 100000000 }
    - block: universe.augment_with_open_interest
      with: [open_interest_snapshot]
  score: oi_notional_usd
  order: desc
  top_k: 5
  dedupe_by: base_asset
"""


#: The funding example, ranked on the ANNUALISED rate.
#:
#: ``score: fundingRatePct`` — what this shipped before — ranks a column whose
#: unit differs from row to row: Binance settles 443 of its 743 perpetuals every
#: 4 hours, 296 every 8 and 4 every hour (measured 2026-08-02), and
#: ``lastFundingRate`` is the rate for ONE settlement. So two coins both showing
#: 0.01% were ranked as equal while paying 10.95%/yr and 87.6%/yr. Every
#: "highest/most-negative funding" basket the demo produced was mis-ordered, and
#: the output gave no sign of it — five symbols, five plausible rates.
#:
#: ``fundingRateApr`` is that rate times the contract's own settlements per year,
#: which is why the step now needs ``funding_info`` as well: that frame carries
#: the interval, and ``universe.augment_with_funding`` refuses to assume 8 hours
#: without it. The multiplier is positive, so ``order`` still means what it did.
FUNDING_SELECTION_EXAMPLE_YAML = """\
spec_version: "1.0"
target: standard_bot
strategy:
  id: funding_rate_selector
  description: "依目前跨幣別資金費率(年化)挑選候選幣"
run:
  mode: backtest
data:
  symbol: BTCUSDT
  market_type: futures
  primary: { interval: "1h" }
selection:
  universe:
    - block: universe.filter_quote_volume
      params: { min_quote_volume: 100000000 }
    - block: universe.augment_with_funding
      with: [funding, funding_info]
  score: fundingRateApr
  order: desc
  top_k: 5
  dedupe_by: base_asset
"""

#: The funding example, flipped to the bottom of the column.
#:
#: Written as a substitution rather than a second literal so the two examples
#: cannot drift apart: the whole point of showing this variant is that ONE key
#: separates "highest funding" from "most negative funding". A hand-maintained
#: copy would eventually differ in some other line too, and the model would have
#: to guess which difference mattered.
FUNDING_ASC_SELECTION_EXAMPLE_YAML = FUNDING_SELECTION_EXAMPLE_YAML.replace(
    "order: desc", "order: asc"
).replace("funding_rate_selector", "lowest_funding_rate_selector")


def _example_without_order(example: str, old_id: str, new_id: str) -> str:
    """The same example with no ranking direction in it at all.

    Shown when the request never named an end of the column. Handing the model
    the ``desc`` variant instead would teach it a direction the user did not ask
    for, and nothing downstream could catch that: with ``intent.score_order``
    unset, :func:`reconcile_intent` has nothing to compare the generated ``order``
    against, so whatever the prompt suggests is what ships.

    Raises rather than returning the input unchanged, because a silent no-op here
    would put ``order: desc`` back in front of a model that was just told the
    user named no direction — the exact claim this variant exists to avoid.
    """
    out = example.replace("  order: desc\n", "", 1).replace(old_id, new_id, 1)
    if "order:" in out or new_id not in out:
        raise RuntimeError(
            "example YAML no longer has exactly one '  order: desc' line, or the "
            "strategy id %r moved; fix _example_without_order() before this "
            "variant is used as the no-direction prompt example" % old_id
        )
    return out


FUNDING_NO_ORDER_SELECTION_EXAMPLE_YAML = _example_without_order(
    FUNDING_SELECTION_EXAMPLE_YAML,
    "funding_rate_selector", "default_order_funding_rate_selector",
)

PRICE_CHANGE_SELECTION_EXAMPLE_YAML = """\
spec_version: "1.0"
target: standard_bot
strategy:
  id: price_change_selector
  description: "依 24 小時漲跌幅挑選候選幣"
run:
  mode: backtest
data:
  symbol: BTCUSDT
  market_type: futures
  primary: { interval: "1h" }
selection:
  universe:
    - block: universe.filter_quote_volume
      params: { min_quote_volume: 100000000 }
  score: priceChangePercent
  order: desc
  top_k: 5
  dedupe_by: base_asset
"""

#: Same column, other end: "biggest losers" instead of "biggest gainers".
PRICE_CHANGE_ASC_SELECTION_EXAMPLE_YAML = PRICE_CHANGE_SELECTION_EXAMPLE_YAML.replace(
    "order: desc", "order: asc"
).replace("price_change_selector", "biggest_losers_selector")

PRICE_CHANGE_NO_ORDER_SELECTION_EXAMPLE_YAML = _example_without_order(
    PRICE_CHANGE_SELECTION_EXAMPLE_YAML,
    "price_change_selector", "default_order_price_change_selector",
)

#: What the prompt says instead of naming a direction, when the request named
#: none. ``intent.score_order`` has three states and the demo used to render the
#: third one as "這次需求是由高到低,寫 order: desc" — i.e. it told the model the
#: user had asked for the top of the column when the user had asked for nothing,
#: or (for "funding 為負的幣", before that phrasing was recognised) for the exact
#: opposite end. ``reconcile_intent`` only checks a direction the user actually
#: stated, so there is no second line of defence here; the prompt has to stay
#: silent about the direction and say why it is silent.
NO_ORDER_DIRECTIVE = (
    "這次需求沒有指明要取這個欄位的哪一端(或同時提到了兩端),系統不會替使用者決定:"
    "請完全不要寫 selection.order,讓 schema 預設生效;"
    "不得聲稱使用者要求了由高到低或由低到高。\n\n"
)


def _score_order_for(intent: IntentDecision | None, column: str) -> str | None:
    """The requested direction, but only if it was requested FOR ``column``.

    ``intent.score_order`` is meaningless without its column: "選 funding rate 的
    幣,要跌幅最大的" names an end of ``priceChangePercent``, and pasting that ``asc``
    into a funding prompt asks for the bottom of a different column entirely.
    Three states in, three states out — ``None`` means the user did not say.
    """
    if intent is None or intent.score_order_metric != column:
        return None
    if intent.score_order not in {None, "asc", "desc"}:
        raise ValueError(
            "intent.score_order must be 'asc', 'desc' or None, got %r; "
            "fix classify_request instead of collapsing it here"
            % (intent.score_order,)
        )
    return intent.score_order


def _long_short_ratio_selection_example(intent: IntentDecision) -> str:
    """A runnable spelling of the precise crowd-long threshold the user named."""

    minimum = intent.long_short_min_account_pct
    if minimum is None:
        raise ValueError("a long/short selection example needs an explicit minimum")
    comparator = intent.long_short_min_account_operator
    if comparator not in {">", ">="}:
        raise ValueError("a long/short selection example needs an explicit lower-bound operator")
    filter_param = ("min_long_account_pct_exclusive"
                    if comparator == ">" else "min_long_account_pct")
    return yaml.safe_dump({
        "spec_version": "1.0",
        "target": "standard_bot",
        "strategy": {
            "id": "retail_long_account_selector",
            "description": "篩選散戶多方帳戶佔比達門檻的候選幣",
        },
        "run": {"mode": "backtest"},
        "data": {
            "symbol": "BTCUSDT", "market_type": "futures",
            "primary": {"interval": "1h"},
        },
        "selection": {
            "universe": [
                {
                    "block": "universe.filter_quote_volume",
                    "params": {"min_quote_volume": 100000000},
                },
                {
                    "block": "universe.augment_with_long_short_ratio",
                    "with": ["long_short_ratio_snapshot"],
                },
                {
                    "block": "universe.filter_long_short_ratio",
                    "params": {filter_param: minimum},
                },
            ],
            # The source condition is a filter, not a rank.  Use a stable
            # liquidity tie-breaker rather than claiming the raw ratio is a
            # cross-contract ranking score.
            "score": "quoteVolume",
            "top_k": intent.requested_count or 5,
            "dedupe_by": "base_asset",
        },
    }, allow_unicode=True, sort_keys=False)


def _supertrend_selection_example(intent: IntentDecision) -> str:
    """Render the exact per-symbol MTF Supertrend contract the runtime supports."""

    if len(intent.supertrend_parameters) != 1:
        raise ValueError("a Supertrend selection needs exactly one explicit parameter pair")
    if not intent.intervals:
        raise ValueError("a Supertrend selection needs explicit timeframes")
    directions = [side for side in ("long", "short") if side in intent.directions]
    if len(directions) != 1:
        raise ValueError("a Supertrend selection needs exactly one explicit direction")

    period, multiplier = intent.supertrend_parameters[0]
    side = directions[0]
    relation = "value_above" if side == "long" else "value_below"
    aliases = ["st_dir_%s" % timeframe for timeframe in intent.intervals]
    universe = [
        {
            "block": "universe.filter_quote_volume",
            "params": {"min_quote_volume": 100000000},
        },
    ]
    # A combined crowd condition remains independently checkable; it must not
    # disappear just because the Supertrend branch owns the prompt.
    if "long_short_ratio" in intent.sources:
        minimum = intent.long_short_min_account_pct
        comparator = intent.long_short_min_account_operator
        if minimum is None or comparator not in {">", ">="}:
            raise ValueError("Supertrend crowd filter needs an exact lower bound")
        filter_param = ("min_long_account_pct_exclusive"
                        if comparator == ">" else "min_long_account_pct")
        universe.extend([
            {
                "block": "universe.augment_with_long_short_ratio",
                "with": ["long_short_ratio_snapshot"],
            },
            {
                "block": "universe.filter_long_short_ratio",
                "params": {filter_param: minimum},
            },
        ])
    universe.extend(
        {
            "block": "universe.augment_with_indicator",
            "with": ["universe_bars"],
            "params": {
                "indicator": "supertrend", "timeframe": timeframe,
                # Supertrend returns (line, direction); the direction is output 1.
                "output": 1, "as": alias, "agg": "last", "window_bars": 1,
                "period": period, "multiplier": multiplier,
            },
        }
        for timeframe, alias in zip(intent.intervals, aliases)
    )
    leaves = [
        {"cond": "conditions.%s" % relation, "args": [alias, 0]}
        for alias in aliases
    ]
    condition = {"all_of": leaves} if len(leaves) > 1 else leaves[0]
    return yaml.safe_dump({
        "spec_version": "1.0",
        "target": "standard_bot",
        "strategy": {
            "id": "multi_timeframe_supertrend_selector",
            "description": "Supertrend(%d,%g) 多時框同時偏%s的候選"
                           % (period, multiplier, "多" if side == "long" else "空"),
        },
        "run": {"mode": "backtest"},
        "data": {
            "symbol": "BTCUSDT", "market_type": "futures",
            "primary": {"interval": "1h" if "1h" in intent.intervals
                        else intent.intervals[0]},
        },
        "selection": {
            "universe": universe,
            "score": "quoteVolume",
            "top_k": intent.requested_count or 5,
            "dedupe_by": "base_asset",
            "%s_when" % side: condition,
        },
    }, allow_unicode=True, sort_keys=False)


def _supertrend_selection_prompt(intent: IntentDecision) -> str:
    example = _supertrend_selection_example(intent)
    period, multiplier = intent.supertrend_parameters[0]
    frames = ", ".join(intent.intervals)
    direction = "偏多" if "long" in intent.directions else "偏空"
    crowd_note = ""
    if "long_short_ratio" in intent.sources:
        minimum = intent.long_short_min_account_pct
        comparator = intent.long_short_min_account_operator
        if minimum is None or comparator not in {">", ">="}:
            raise ValueError("Supertrend crowd prompt needs an exact lower bound")
        filter_param = ("min_long_account_pct_exclusive"
                        if comparator == ">" else "min_long_account_pct")
        wording = "大於" if comparator == ">" else "至少"
        crowd_note = (
            "使用者也指定散戶多方帳戶佔比%s %.4g%%:必須先 "
            "augment_with_long_short_ratio(with: [long_short_ratio_snapshot]),再以 "
            "filter_long_short_ratio(%s: %.4g) 篩選。"
            % (wording, minimum, filter_param, minimum)
        )
    return (
        "你是一個把自然語言選幣需求轉成 StandardBot YAML 的轉換器。"
        "只輸出一份合法 YAML,不要 markdown、解釋或多餘文字。\n\n"
        "這次需求是每個候選幣的 Supertrend(%d,%g) 在 %s 同時%s。"
        "選幣必須先用 universe.filter_quote_volume 收窄,然後每個 timeframe 都用 "
        "universe.augment_with_indicator(with: [universe_bars])，參數必須是 "
        "indicator: supertrend、period: %d、multiplier: %g、output: 1。"
        "Supertrend output 0 是價格線,不能拿來判斷方向。%s_when 必須是 all_of，"
        "每一個 timeframe 的 alias 都必須以 conditions.%s(_, 0) 接入。"
        "%s\n\n=== 範例輸出 ===\n%s"
        % (period, multiplier, frames, direction, period, multiplier,
           "long" if "long" in intent.directions else "short",
           "value_above" if "long" in intent.directions else "value_below",
           crowd_note, example)
    )


def _candidate_trade_selection_prompt(intent: IntentDecision) -> str:
    """Prompt one outer selection with complete nested candidate trades.

    This branch deliberately precedes the source-specific plain-selection
    prompts.  Those prompts correctly prohibit ``risk``/``sizing`` for a plain
    basket, but those sections are required by ``selection.candidate_trade``.
    """

    sources = set(intent.sources)
    if "open_interest" in sources:
        source_steps = (
            "    - block: universe.filter_quote_volume\n"
            "      params: { min_quote_volume: 100000000 }\n"
            "    - block: universe.augment_with_open_interest\n"
            "      with: [open_interest_snapshot]\n"
        )
        score = "oi_notional_usd"
        order = "desc"
        source_note = (
            "選幣依據是 open interest:先用 filter_quote_volume 收窄,再用 "
            "augment_with_open_interest(with: [open_interest_snapshot]),並以 "
            "oi_notional_usd 排名。"
        )
    elif "funding" in sources:
        source_steps = (
            "    - block: universe.filter_quote_volume\n"
            "      params: { min_quote_volume: 100000000 }\n"
            "    - block: universe.augment_with_funding\n"
            "      with: [funding, funding_info]\n"
        )
        score = "fundingRateApr"
        order = _score_order_for(intent, "fundingRatePct")
        source_note = (
            "選幣依據是 funding:使用 augment_with_funding(with: [funding, "
            "funding_info])與可跨結算週期比較的 fundingRateApr。"
        )
    elif "price_change" in sources:
        source_steps = (
            "    - block: universe.filter_quote_volume\n"
            "      params: { min_quote_volume: 100000000 }\n"
        )
        score = "priceChangePercent"
        order = _score_order_for(intent, "priceChangePercent")
        source_note = (
            "選幣依據是 24h 漲跌幅:priceChangePercent 已在 universe frame,"
            "不得虛構 augment_with_price_change。"
        )
    elif "news" in sources:
        source_steps = (
            "    - block: universe.filter_quote_volume\n"
            "      params: { min_quote_volume: 100000000 }\n"
            "    - block: universe.augment_with_news\n"
            "      with: [ticker_rank]\n"
        )
        score = ("news_bull_ratio" if intent.ranking_metric == "sentiment"
                 else "news_mention_count")
        order = "desc"
        source_note = (
            "選幣依據是新聞/社群:使用 augment_with_news(with: [ticker_rank]),"
            "並以使用者要求的 news_mention_count 或 news_bull_ratio 排名。"
        )
    else:
        source_steps = (
            "    - block: universe.filter_quote_volume\n"
            "      params: { min_quote_volume: 100000000 }\n"
        )
        score = "quoteVolume"
        order = "desc"
        source_note = "選幣依據是 24h 成交額,以 quoteVolume 排名。"

    plan = candidate_trade_plan(intent)
    period = plan["period"]
    interval = intent.intervals[0] if len(intent.intervals) == 1 else "1h"
    alias = plan["alias"]
    condition_lines = []
    for side in plan["sides"]:
        relation, value = plan["thresholds"][side]
        condition_lines.append(
            "  %s_when: { cond: conditions.value_%s, args: [%s, %g] }"
            % (side, relation, alias, value)
        )

    top_k = intent.requested_count or 5
    order_line = "  order: %s\n" % order if order else ""
    assumptions = list(plan["assumptions"])
    assumption_block = ""
    if assumptions:
        rendered = yaml.safe_dump(
            {"assumptions": assumptions}, allow_unicode=True, sort_keys=False
        ).rstrip()
        assumption_block = "\n".join(
            "  " + line for line in rendered.splitlines()
        ) + "\n"
    description = (
        "先選幣,再以逐候選 RSI 快照產生需確認的買賣計畫;"
        + ("未指定值已列於 strategy.assumptions" if assumptions else "未使用隱含預設")
    )
    example = (
        'spec_version: "1.0"\n'
        "target: standard_bot\n"
        "strategy:\n"
        "  id: rsi_candidate_trade_selector\n"
        "  description: %s\n%s"
        "run:\n"
        "  mode: backtest\n"
        "data:\n"
        "  symbol: BTCUSDT\n"
        "  market_type: futures\n"
        "  primary: { interval: \"%s\" }\n"
        "selection:\n"
        "  universe:\n%s"
        "    - block: universe.augment_with_indicator\n"
        "      with: [universe_bars]\n"
        "      params: { indicator: rsi, timeframe: \"%s\", period: %d, as: %s }\n"
        "  score: %s\n%s"
        "  top_k: %d\n"
        "  dedupe_by: base_asset\n%s\n"
        "  candidate_trade: { entry_type: market }\n"
        "sizing: { size: %g }\n"
        "risk:\n"
        "  exit: { type: pct_stop_tp, stop_pct: %g, tp_pct: %g, max_bars: 48 }\n"
        % (json.dumps(description, ensure_ascii=False), assumption_block,
           interval, source_steps, interval, period, alias, score, order_line,
           top_k, "\n".join(condition_lines), plan["size"], plan["stop_pct"],
           plan["tp_pct"])
    )
    return (
        "你是一個把自然語言『選幣 + 每個候選的技術買賣點』轉成 StandardBot "
        "YAML 的轉換器。只輸出合法 YAML,不要 markdown 或解釋。\n\n"
        "外層必須只有 selection:,不得同時產生頂層 signals:。必須寫 "
        "selection.candidate_trade: {entry_type: market};外層執行結果仍是 kind=selection,"
        "每個 candidates[].trade 才是完整 kind=trade。這些 nested trades 固定 "
        "auto_trade_eligible=false、requires_confirmation=true,不是自動下單授權。\n"
        + source_note + "\n"
        "RSI 必須使用 universe.augment_with_indicator 並精確傳 "
        "with: [universe_bars]。fan-out 前必須先收窄 universe。其輸出 alias 必須真的"
        "出現在 long_when/short_when;只計算 RSI 卻不用它判斷買賣點一律不接受。"
        "任何未由使用者指定的 RSI 門檻、size、stop 或 take-profit 只能採範例中的"
        "固定預設,而且必須逐項保留範例的 strategy.assumptions cid 與 reading；漏掉"
        "宣告或宣告值和實際條件不同都會被拒絕。\n\n"
        "=== 範例輸出 ===\n" + example
    )


def _trade_prompt_sides(intent: IntentDecision | None) -> tuple[str, ...]:
    """Resolve the same direction default that semantic reconciliation enforces."""
    if intent is None:
        return ("long",)
    excluded = set(intent.excluded_directions)
    requested = tuple(
        side for side in ("long", "short")
        if side in intent.directions and side not in excluded
    )
    if requested:
        return requested
    if "long" in excluded and "short" not in excluded:
        return ("short",)
    return ("long",)


def _trade_prompt_example(sides: tuple[str, ...]) -> str:
    """Remove unrequested sides so the example cannot contradict the instruction."""
    example = yaml.safe_load(EXAMPLE_YAML)
    entry = example["signals"]["entry"]
    for side in ("long", "short"):
        if side not in sides:
            entry.pop(side, None)
    return yaml.safe_dump(example, allow_unicode=True, sort_keys=False)


def build_system_prompt(
    kind: str = "trade", intent: IntentDecision | None = None,
) -> str:
    if kind not in {"trade", "selection"}:
        raise ValueError("strategy kind must be trade or selection, got %r" % kind)
    if kind == "selection":
        if intent is not None and intent.candidate_trade_requested:
            return _candidate_trade_selection_prompt(intent)
        if intent is not None and "supertrend" in intent.indicator_names:
            return _supertrend_selection_prompt(intent)
        if intent is not None and "long_short_ratio" in intent.sources:
            minimum = intent.long_short_min_account_pct
            comparator = intent.long_short_min_account_operator
            if minimum is None or comparator not in {">", ">="}:
                raise ValueError(
                    "long/short-ratio selection prompt requires an explicit "
                    "lower-bound threshold and comparator"
                )
            filter_param = ("min_long_account_pct_exclusive"
                            if comparator == ">" else "min_long_account_pct")
            wording = "大於" if comparator == ">" else "至少"
            return (
                "你是一個把自然語言選幣需求轉成 StandardBot YAML 的轉換器。"
                "只輸出一份合法 YAML,不要 markdown、解釋或多餘文字。\n\n"
                "這次需求指定散戶多空比偏多,而且門檻是多方帳戶佔比%s %.4g%%。"
                "頂層只用 selection:,不得產生 signals:、sizing:、risk: 或 backtest:。"
                "先用 universe.filter_quote_volume 收窄,再精確寫 "
                "universe.augment_with_long_short_ratio(with: "
                "[long_short_ratio_snapshot])，最後必須使用 "
                "universe.filter_long_short_ratio(%s: %.4g)。"
                "long_account_pct 的單位是百分點,60 表示六成帳戶偏多,不是 0.6。"
                "『多空比偏多』是群眾部位篩選,不得轉成新聞情緒或 long_when。\n\n"
                "=== 範例輸出 ===\n%s"
                % (wording, minimum, filter_param, minimum,
                   _long_short_ratio_selection_example(intent))
            )
        if intent is not None and "open_interest" in intent.sources:
            return (
                "你是一個把自然語言選幣需求轉成 StandardBot YAML 的轉換器。"
                "只輸出一份合法 YAML,不要 markdown、解釋或多餘文字。\n\n"
                "這次需求指定跨幣別 open interest。頂層只用 selection:,不得產生"
                " signals:、sizing:、risk: 或 backtest:。使用 data.symbol=BTCUSDT"
                " 作排程代表標的,它不是候選結果。open interest 沒有全市場端點,"
                "所以必須先用 universe.filter_quote_volume 收窄名單,再使用"
                " universe.augment_with_open_interest 並精確寫"
                " with: [open_interest_snapshot]。selection.score 必須使用"
                " oi_notional_usd,因為 oi_base 是不同幣種的合約數量,不能跨幣比較。"
                "不得改用新聞、成交量排名或技術指標。top_k 必須依使用者要求,"
                "未指定時為 5。\n\n"
                "=== 可用的 open interest 選幣 BLOCKS / 欄位 ===\n"
                "universe.filter_quote_volume\n"
                "universe.augment_with_open_interest "
                "(with: [open_interest_snapshot])\n"
                "universe.filter_open_interest\n"
                "oi_base, oi_notional_usd, quoteVolume\n\n"
                "=== 範例輸出 ===\n" + OPEN_INTEREST_SELECTION_EXAMPLE_YAML
            )
        if intent is not None and "funding" in intent.sources:
            # Three states, not two. ``ascending = score_order == "asc"`` used to
            # collapse "the user did not say" into "the user said desc".
            direction_note, funding_example = {
                "asc": (
                    "這次需求是『最負/最低 funding』,必須寫 order: asc;"
                    "不得改寫成 desc,也不得改用 top_k 之後自行反向解讀。\n\n",
                    FUNDING_ASC_SELECTION_EXAMPLE_YAML,
                ),
                "desc": (
                    "這次需求是『最正/最高 funding』,寫 order: desc。\n\n",
                    FUNDING_SELECTION_EXAMPLE_YAML,
                ),
                None: (NO_ORDER_DIRECTIVE, FUNDING_NO_ORDER_SELECTION_EXAMPLE_YAML),
            }[_score_order_for(intent, "fundingRatePct")]
            return (
                "你是一個把自然語言選幣需求轉成 StandardBot YAML 的轉換器。"
                "只輸出一份合法 YAML,不要 markdown、解釋或多餘文字。\n\n"
                "這次需求指定跨幣別 funding rate。頂層只用 selection:,不得產生"
                " signals:、sizing:、risk: 或 backtest:。使用 data.symbol=BTCUSDT"
                " 作排程代表標的,它不是候選結果。必須先用"
                " universe.filter_quote_volume 過濾最低流動性,再精確使用"
                " universe.augment_with_funding 並寫 with: [funding, funding_info]。"
                "selection.score 必須是 fundingRateApr,使 funding 真正控制候選排名;"
                "不得改用新聞、EMA/RSI 或自行猜單一幣種。top_k 必須依使用者要求,"
                "未指定時為 5。這是當下截面選幣訊號,不是歷史回測。\n\n"
                # Why the annualised column and not the raw one: Binance settles
                # different perpetuals every 8h / 4h / 1h, so fundingRatePct's unit
                # differs per row and ranking it puts 0.01%@1h (87.6%/yr) level
                # with 0.01%@8h (10.95%/yr). The model is told to ask for
                # funding_info because without it the block leaves the annualised
                # column NaN — deliberately, rather than assuming 8 hours.
                "fundingRatePct 是「單次結算」的費率,而幣安各合約結算間隔不同"
                "(8h / 4h / 1h),直接排名會把年化差 8 倍的兩個幣當成一樣。"
                "fundingRateApr = 年化後的 carry,是唯一可跨合約比較的欄位;"
                "它需要 funding_info 提供結算間隔,少寫 funding_info 會讓該欄位全部是"
                " NaN 而選出空籃子。若使用者要求年化門檻(例如年化 30% 以上),"
                "用 selection.min_score / max_score 表達。\n\n"
                # fundingRateApr is signed, so the two ends of the column are two
                # opposite trades (paid-to-be-long vs paid-to-be-short). The model
                # is told the key by name because a basket taken from the wrong
                # end looks perfectly healthy in the output. Annualising multiplies
                # by a positive number, so it does not change which end is which.
                "selection.order 決定取欄位的哪一端:desc(預設)= 由高到低,"
                " asc = 由低到高(年化不改變正負,方向語意與原始費率相同)。"
                + direction_note +
                "=== 可用的 funding 選幣 BLOCKS / 欄位 ===\n"
                "universe.filter_quote_volume\n"
                "universe.augment_with_funding (with: [funding, funding_info])\n"
                "universe.filter_funding_rate\n"
                "fundingRateApr, fundingRatePct, fundingIntervalHours, quoteVolume\n\n"
                "=== 範例輸出 ===\n" + funding_example
            )
        if intent is not None and "price_change" in intent.sources \
                and "news" not in intent.sources:
            # Same three states. The notes describe the END OF THE COLUMN rather
            # than the words the user used, because both ends have two phrasings:
            # 跌幅最小 asks for the top of the column, 漲幅最小 for the bottom.
            direction_note, change_example = {
                "asc": (
                    "這次需求指向欄位低端(跌幅最大/漲幅最小),必須寫 order: asc。\n\n",
                    PRICE_CHANGE_ASC_SELECTION_EXAMPLE_YAML,
                ),
                "desc": (
                    "這次需求指向欄位高端(漲幅最大/跌幅最小),寫 order: desc。\n\n",
                    PRICE_CHANGE_SELECTION_EXAMPLE_YAML,
                ),
                None: (NO_ORDER_DIRECTIVE,
                       PRICE_CHANGE_NO_ORDER_SELECTION_EXAMPLE_YAML),
            }[_score_order_for(intent, "priceChangePercent")]
            return (
                "你是一個把自然語言選幣需求轉成 StandardBot YAML 的轉換器。"
                "只輸出一份合法 YAML,不要 markdown、解釋或多餘文字。\n\n"
                "這次需求指定 24 小時漲跌幅。頂層只用 selection:,不得產生 signals:。"
                "使用 data.symbol=BTCUSDT 作排程代表標的,它不是候選結果。"
                # priceChangePercent needs no augment step: the universe frame IS
                # the Binance 24h ticker and already carries it. Saying so stops
                # the model inventing an augment_with_price_change block.
                "priceChangePercent 已經在 universe frame 裡,不需要任何 augment 步驟;"
                "不得虛構 universe.augment_with_price_change。"
                "先用 universe.filter_quote_volume 過濾最低流動性(漲幅榜前段常是"
                "沒有量的小幣,接不到單),再以 score: priceChangePercent 排名。"
                "不得改用 news_mention_count 或技術指標。top_k 必須依使用者要求,"
                "未指定時為 5。\n\n"
                "selection.order:漲幅最大用 desc(預設);跌幅最大用 asc。" +
                direction_note +
                "=== 可用的漲跌幅選幣 BLOCKS / 欄位 ===\n"
                "universe.filter_quote_volume\n"
                "universe.top_gainers (params: {n: 30})\n"
                "universe.top_losers (params: {n: 30})\n"
                "universe.filter_change_pct\n"
                "priceChangePercent, quoteVolume\n\n"
                "=== 範例輸出 ===\n" + change_example
            )
        if intent is not None and "liquidity" in intent.sources \
                and "news" not in intent.sources:
            return (
                "你是一個把自然語言選幣需求轉成 StandardBot YAML 的轉換器。"
                "只輸出一份合法 YAML,不要 markdown、解釋或多餘文字。\n\n"
                "這次需求指定成交量/流動性。頂層只用 selection:,不得產生 signals:。"
                "使用 data.symbol=BTCUSDT 作排程代表標的；它不是候選結果。"
                "使用 universe.filter_quote_volume 過濾最低流動性，並以"
                " score: quoteVolume 排名。不得改用 news_mention_count、技術指標或"
                "自行猜單一幣種。top_k 必須依使用者要求。\n\n=== 範例輸出 ===\n"
                + LIQUIDITY_SELECTION_EXAMPLE_YAML
            )
        ranking_note = "依 Square 提及量排序時使用 score: news_mention_count。"
        selection_example = SELECTION_EXAMPLE_YAML
        if intent is not None and "sentiment" in intent.news_metrics \
                and "mentions" not in intent.news_metrics:
            ranking_note = (
                "這次使用者要求新聞/社群情緒排行,selection.score 必須使用"
                " news_bull_ratio；不得偷換成 news_mention_count。"
            )
            selection_example = SELECTION_EXAMPLE_YAML.replace(
                "score: news_mention_count", "score: news_bull_ratio"
            )
        return (
            "你是一個把自然語言選幣需求轉成 StandardBot YAML 的轉換器。"
            "只輸出一份合法 YAML,不要有 markdown 圍欄(```)、不要解釋、不要多餘文字。\n\n"
            "這次輸出必須是截面選幣策略:頂層只用 selection:,不得產生 signals:、sizing:、risk:"
            "或 backtest:。執行後系統會由 payload 推導 kind=selection,不要自行新增 kind 欄位。\n"
            "必須保留 spec_version、target、strategy、run、data。run.mode 使用 backtest;"
            "選幣仍以 data.symbol=BTCUSDT 作為排程代表標的,market_type=futures,"
            "data.primary.interval=1h。data.symbol 只是排程代表標的;使用者要求一組候選時,"
            "不得自行猜 SUIUSDT 或其他單一候選。只有使用者明確列出候選宇宙時,"
            "才可用 universe.only_symbols 限制在那些標的。\n\n"
            "新聞或社群熱門度選幣必須使用現有 Blocks 路徑:先用"
            " universe.filter_quote_volume 過濾流動性,再用 universe.augment_with_news,"
            "並精確寫 with: [ticker_rank]。" + ranking_note +
            "top_k 依使用者要求,未指定時為 5;"
            "min_score: 1.0;dedupe_by: base_asset。這條 live 資料目前代表 Square 最近 24 小時"
            "的 ticker_rank,不要虛構不支援的 window 或 data.news 欄位。\n"
            "若使用者只要求挑選或排名,不要加 long_when/short_when,候選方向會是 neutral。"
            "只有使用者明確要求依情緒做多/做空時,才可加入 long_when/short_when。\n\n"
            "若使用者說『可能上漲、看漲、偏多』,只能把它保守映射為新聞情緒代理:"
            "在 augment_with_news 後加入 universe.filter_sentiment,參數"
            " min_bull_ratio: 0.55;不得承諾真的會上漲。『少見、未被市場發現』目前沒有"
            "直接欄位,不要在 description 宣稱已驗證;只能說使用流動性、提及量與情緒代理。\n\n"
            "=== 可用的新聞選幣 BLOCKS / 欄位 ===\n"
            "universe.filter_quote_volume\n"
            "universe.augment_with_news (with: [ticker_rank])\n"
            "universe.filter_sentiment (params: {min_bull_ratio: 0.55})\n"
            "news_mention_count, news_bull_ratio, news_unique_authors, quoteVolume\n\n"
            "=== 範例輸出 ===\n" + selection_example
        )

    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    prompt_sides = _trade_prompt_sides(intent)
    if prompt_sides == ("short",) and intent is not None \
            and "long" in intent.excluded_directions:
        direction_note = (
            "使用者明確禁止做多且只允許做空:只能給 signals.entry.short,"
            "不得產生 signals.entry.long。\n\n"
        )
    elif prompt_sides == ("short",):
        direction_note = "使用者只要求做空:只能給 signals.entry.short。\n\n"
    elif prompt_sides == ("long",) and intent is not None \
            and intent.directions == ("long",):
        direction_note = "使用者只要求做多:只能給 signals.entry.long。\n\n"
    elif prompt_sides == ("long",):
        direction_note = (
            "使用者未指定方向,採 long-only:只能給 signals.entry.long,"
            "不得擅自加入 signals.entry.short。\n\n"
        )
    else:
        direction_note = (
            "使用者同時明確要求多空兩邊:必須同時給 signals.entry.long 與 "
            "signals.entry.short。\n\n"
        )
    prompt_example = _trade_prompt_example(prompt_sides)
    return (
        "你是一個把「自然語言交易策略描述」轉成 YAML 規格的轉換器。"
        "只輸出一份合法 YAML,不要有 markdown 圍欄(```)、不要解釋、不要多餘文字。\n\n"
        "必須嚴格遵守下面的 schema 與可用 block 清單;只能使用清單內的 block,參數名要精確。\n"
        "不要使用 data.htf(單一時間框即可)。若使用者沒指定,採保守預設:"
        "market_type=futures、interval=1h、fees commission_bps=4 slippage_bps=2、size=0.95、"
        "出場預設 pct_stop_tp{stop_pct:0.02,tp_pct:0.04,max_bars:96}。\n"
        + direction_note +
        "=== SCHEMA ===\n" + schema + "\n\n"
        "=== 可用 BLOCKS ===\n" + BLOCKS_CHEATSHEET + "\n\n"
        "=== 範例輸出 ===\n" + prompt_example
    )


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        # drop first fence line (``` or ```yaml) and trailing fence
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def call_llm(api_base: str, api_key: str, model: str, nl: str) -> str:
    if not _external_llm_enabled():
        raise ExternalLLMDisabled(
            "外部 LLM 路由預設停用，因為自然語言可能包含私人內容。"
            "若這是你明確允許送出的非敏感測試內容，請在啟動 demo 前設定 "
            "%s=1；私有對話語料不可使用此路徑。" % EXTERNAL_LLM_OPT_IN_ENV
        )
    base = api_base.rstrip("/")
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    intent = classify_request(nl)
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": build_system_prompt(intent.kind, intent)},
            {"role": "user", "content": nl},
        ],
    }
    resp = requests.post(url, headers=headers, json=body, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return _strip_fences(content)


def _trade_preflight_clarification(intent: IntentDecision) -> str | None:
    """Return a deterministic stop reason before a trade request leaves host.

    These are not model-quality problems.  They describe an execution intent
    the current single-instrument contract cannot safely infer, so calling an
    opted-in model first would needlessly disclose the request and still leave
    a semantic mismatch to catch later.
    """
    if intent.kind != "trade":
        return None
    if "bare_sell_direction_ambiguous" in intent.evidence:
        return (
            "『賣出/sell』可能代表平多(close_long)或開空(open_short)。"
            "請明確指定其中一種；系統不會把下穿條件併入開多訊號。"
        )
    if len(intent.included_symbols) > 1:
        return (
            "目前 trade YAML 一次只能建立一個標的的規則；請分成各自的單標的策略，"
            "或改成可驗證的 selection/candidate_trade 需求。"
        )
    if intent.market_type == "spot" and "short" in intent.directions:
        return (
            "現貨不能開空；請改成 futures/perpetual，或明確指定既有多單的 close_long。"
        )
    if intent.condition_join == "mixed":
        return (
            "進場條件同時混合 AND 與 OR（例如 EMA 條件加上一組 RSI 的 OR）；"
            "目前 intent 尚無法可靠保留巢狀分組。請拆成可獨立驗證的策略，"
            "或明確改寫為單一 all_of/any_of 規則。"
        )
    if intent.exit_type == "atr_trailing_stop":
        missing = []
        if intent.exit_atr_period is None:
            missing.append("ATR period（例如 ATR14）")
        if intent.exit_trail_mult is None:
            missing.append("追蹤倍數（例如 2x ATR）")
        if missing:
            return (
                "ATR 追蹤停損需要明確的%s；系統不會改用固定百分比停損。"
                % "、".join(missing)
            )
    return None


def convert_nl(api_base: str, api_key: str, model: str, nl: str) -> dict:
    """Convert, validate and ensure the generated YAML matches user intent."""
    intent = classify_request(nl)
    unsupported_gaps = sorted({
        item.split(":", 1)[1]
        for item in intent.unsupported_preferences
        if item.startswith("gap:")
    })
    if unsupported_gaps:
        return {
            "ok": True,
            "status": "unsupported",
            "yaml": "",
            "valid": False,
            "strategy_kind": intent.kind,
            "generated_strategy_kind": None,
            "intent": intent.to_dict(),
            "errors": [
                "需求包含目前 YAML/runtime 無法忠實表達的能力: %s；"
                "系統已停止，不會以相關指標、下單或通知代理替代。"
                % ", ".join(unsupported_gaps)
            ],
            "warnings": [],
        }
    if intent.kind == "ambiguous":
        compound = "compound_selection_execution" in intent.evidence
        message = (
            "這個『先選幣再交易』要求缺少目前 candidate_trade 可驗證的逐候選"
            "進場條件;請補 RSI 門檻/技術買賣點。直接買入與 EMA 跨越等逐根事件"
            "不會被偷偷改寫成另一個策略"
            if compound else
            "無法可靠判斷你要『挑選一組候選幣』還是『為單一標的建立交易規則』;"
            "系統已停止,不會預設成技術分析"
        )
        return {
            "ok": True,
            "status": "needs_clarification",
            "yaml": "",
            "valid": False,
            "strategy_kind": "ambiguous",
            "generated_strategy_kind": None,
            "intent": intent.to_dict(),
            "errors": [message],
            "warnings": [],
        }

    preflight_error = _trade_preflight_clarification(intent)
    if preflight_error:
        return {
            "ok": True,
            "status": "needs_clarification",
            "yaml": "",
            "valid": False,
            "strategy_kind": intent.kind,
            "generated_strategy_kind": None,
            "intent": intent.to_dict(),
            "errors": [preflight_error],
            "warnings": [],
        }

    unmapped_categories = [
        item.split(":", 1)[1]
        for item in intent.unsupported_preferences
        if item.startswith("category_exclusion:")
    ]
    if unmapped_categories:
        return {
            "ok": True,
            "status": "needs_clarification",
            "yaml": "",
            "valid": False,
            "strategy_kind": intent.kind,
            "generated_strategy_kind": None,
            "intent": intent.to_dict(),
            "errors": [
                "你要求排除資產類別(%s)，但 demo 尚未有經核對的 "
                "exchange metadata 映射；已停止，請先指定可審核的分類映射，"
                "不會猜測或以近似篩選代替。"
                % ", ".join(unmapped_categories)
            ],
            "warnings": [],
        }

    if intent.kind == "selection":
        # "Most negative funding" and "biggest gainers" used to be refused here.
        # Both refusals are gone, for different reasons: selection.order: asc now
        # expresses the bottom of a column, and priceChangePercent was never
        # missing — it rides along in the universe frame. What is left is the one
        # source with no cross-section endpoint behind it at all.
        unsupported = sorted(intent.sources & UNSUPPORTED_SELECTION_SOURCES)
        if unsupported:
            return {
                "ok": True,
                "status": "unsupported",
                "yaml": "",
                "valid": False,
                "strategy_kind": "selection",
                "generated_strategy_kind": None,
                "intent": intent.to_dict(),
                "errors": [
                    "目前 selection runtime 尚未把 %s 的跨幣別 frame 接到 UniverseBundle;"
                    "系統已停止,不會偷偷改用新聞或技術分析"
                    % ", ".join(unsupported)
                ],
                "warnings": [],
            }
        if "long_short_ratio" in intent.sources and (
                intent.long_short_min_account_pct is None
                or intent.long_short_min_account_operator not in {">", ">="}):
            return {
                "ok": True,
                "status": "needs_clarification",
                "yaml": "",
                "valid": False,
                "strategy_kind": "selection",
                "generated_strategy_kind": None,
                "intent": intent.to_dict(),
                "errors": [
                    "散戶多空比需要明確的多方帳戶百分比門檻，例如「偏多 > 60%」；"
                    "系統不會猜測 min_long_account_pct。"
                ],
                "warnings": [],
            }
        if "supertrend" in intent.indicator_names:
            missing = []
            if not intent.supertrend_parameters:
                missing.append("Supertrend(period,multiplier)，例如 Supertrend(10,3)")
            if not intent.intervals:
                missing.append("timeframe，例如 H4/H1/M15")
            if len([side for side in ("long", "short") if side in intent.directions]) != 1:
                missing.append("明確方向（偏多或偏空）")
            if missing:
                return {
                    "ok": True,
                    "status": "needs_clarification",
                    "yaml": "",
                    "valid": False,
                    "strategy_kind": "selection",
                    "generated_strategy_kind": None,
                    "intent": intent.to_dict(),
                    "errors": [
                        "Supertrend 選幣需要 %s；系統不會用漲跌幅或其他技術指標代替。"
                        % "、".join(missing)
                    ],
                    "warnings": [],
                }
        if not intent.sources and "supertrend" not in intent.indicator_names:
            return {
                "ok": True,
                "status": "needs_clarification",
                "yaml": "",
                "valid": False,
                "strategy_kind": "selection",
                "generated_strategy_kind": None,
                "intent": intent.to_dict(),
                "errors": ["請指定選幣依據，例如 Square 新聞熱度或 24h 成交量/流動性"],
                "warnings": [],
            }

    try:
        yaml_text = call_llm(api_base, api_key, model, nl)
    except ExternalLLMDisabled as exc:
        return {
            "ok": True,
            "status": "external_llm_disabled",
            "yaml": "",
            "valid": False,
            "strategy_kind": intent.kind,
            "generated_strategy_kind": None,
            "intent": intent.to_dict(),
            "errors": [str(exc)],
            "warnings": [],
        }
    errors: list[str] = []
    warnings: list[str] = []
    generated_kind = None

    try:
        spec = yaml.safe_load(yaml_text)
    except Exception as exc:
        spec = None
        errors.append(f"YAML 解析失敗:{exc}")

    if not errors:
        if not isinstance(spec, dict):
            errors.append("YAML 不是有效的 mapping")
        else:
            validation_errors, validation_warnings = validate_spec(spec)
            errors.extend(validation_errors)
            warnings.extend(validation_warnings)
            generated_kind = generated_strategy_kind(spec)
            # Semantic reconciliation expects a structurally valid tree. Running
            # it after (for example) ``with: 123`` turns a useful validation error
            # into an unrelated TypeError and a generic HTTP 500.
            if not validation_errors:
                alignment_errors, alignment_warnings = reconcile_intent(intent, spec)
                errors.extend(alignment_errors)
                warnings.extend(alignment_warnings)

    # Keep one actionable copy when structural and semantic gates report the
    # same underlying mismatch in slightly different phases.
    errors = list(dict.fromkeys(errors))
    warnings = list(dict.fromkeys(warnings))

    return {
        "ok": True,
        "status": "valid" if not errors else "rejected",
        "yaml": yaml_text,
        "valid": not errors,
        # Keep the independently inferred route even when model output is invalid, so the
        # UI can show the YAML in the correct editor without trying to execute it.
        "strategy_kind": intent.kind,
        "generated_strategy_kind": generated_kind,
        "intent": intent.to_dict(),
        "errors": errors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_KLINES_HOST = {"spot": "https://api.binance.com/api/v3/klines",
                "futures": "https://fapi.binance.com/fapi/v1/klines"}
_FIXTURES = {
    ("BTCUSDT", "1h"): "BTCUSDT_1h_500bars.parquet",
    ("BTCUSDT", "4h"): "BTCUSDT_4h_300bars.parquet",
    ("BTCUSDT", "15m"): "BTCUSDT_15m_500bars.parquet",
    ("ETHUSDT", "1h"): "ETHUSDT_1h_500bars.parquet",
}


def _fixture_df(symbol: str, interval: str):
    name = _FIXTURES.get((symbol.upper(), interval))
    if not name:
        return None
    p = FIXTURE_DIR / name
    if not p.exists():
        return None
    return pd.read_parquet(p).reset_index(drop=True)


def fetch_klines(symbol: str, interval: str, market_type: str, limit: int = 1000):
    """Return an OHLCV DataFrame. Try Binance public REST, fall back to fixture."""
    host = _KLINES_HOST.get(market_type, _KLINES_HOST["futures"])
    try:
        r = requests.get(host, params={"symbol": symbol.upper(), "interval": interval,
                                       "limit": min(limit, 1000)}, timeout=20)
        r.raise_for_status()
        rows = r.json()
        df = pd.DataFrame({
            "open_time": [int(k[0]) for k in rows],
            "open": [float(k[1]) for k in rows],
            "high": [float(k[2]) for k in rows],
            "low": [float(k[3]) for k in rows],
            "close": [float(k[4]) for k in rows],
            "volume": [float(k[5]) for k in rows],
            "close_time": [int(k[6]) for k in rows],
            "quote_volume": [float(k[7]) for k in rows],
        })
        df["timestamp"] = df["close_time"]
        return df, "binance_%s" % market_type
    except Exception as exc:
        fx = _fixture_df(symbol, interval)
        if fx is not None:
            fx["timestamp"] = fx["close_time"]
            return fx, "fixture(offline: %s)" % type(exc).__name__
        raise RuntimeError(f"無法取得 {symbol} {interval} 行情:{exc}") from exc


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


def run_backtest(yaml_text: str) -> dict:
    spec = yaml.safe_load(yaml_text)
    if not isinstance(spec, dict):
        return {"ok": False, "error": "YAML 不是有效的 mapping"}
    errors, warnings = validate_spec(spec)
    if errors:
        return {"ok": False, "error": "spec 驗證失敗", "errors": errors, "warnings": warnings}

    data = spec.get("data") or {}
    symbol = data["symbol"].upper()
    interval = (data.get("primary") or {})["interval"]
    market_type = data.get("market_type", "futures")
    entry = (spec.get("signals") or {}).get("entry") or {}
    exit_cfg = (spec.get("risk") or {}).get("exit")
    fees = (spec.get("risk") or {}).get("fees") or {}
    size = float((spec.get("sizing") or {}).get("size", 0.95))
    initial_capital = float((spec.get("backtest") or {}).get("initial_capital", 10000.0))
    long_only = (market_type == "spot") or (not entry.get("short"))

    df, source = fetch_klines(symbol, interval, market_type)
    if df is None or len(df) < 50:
        return {"ok": False, "error": "行情資料不足(需 ≥ 50 根)"}

    make_signals = build_make_signals(spec)
    result = run_vectorized_backtest(
        df=df, signal_fn=make_signals, exit_cfg=exit_cfg, timeframe=interval,
        size=size, fee_bps=float(fees.get("commission_bps", 4.0)),
        slippage_bps=float(fees.get("slippage_bps", 2.0)),
        initial_capital=initial_capital, long_only=long_only,
    )

    equity = result.equity_curve
    if equity is None:
        equity = np.full(len(df), initial_capital, dtype=float)
    equity = np.asarray(equity, dtype=float)

    # --- BTC buy-and-hold 基準(同區間,initial_capital 全押)---
    if symbol == "BTCUSDT":
        btc_close = df["close"].to_numpy(dtype=float)
    else:
        btc_df, _ = fetch_klines("BTCUSDT", interval, market_type, limit=len(df))
        btc_close = btc_df["close"].to_numpy(dtype=float)
    m = min(len(equity), len(btc_close), len(df))
    equity = equity[-m:]
    btc_close = btc_close[-m:]
    baseline = initial_capital * (btc_close / btc_close[0])
    ts = df["close_time"].to_numpy()[-m:].astype("int64").tolist()

    # 下採樣避免傳太大(圖用)
    def _ds(arr, k=600):
        arr = list(arr)
        if len(arr) <= k:
            return arr
        step = len(arr) / k
        return [arr[int(i * step)] for i in range(k)]

    bh_return = float(baseline[-1] / baseline[0] - 1.0)

    # --- 交易加上實際時間(entry_idx/exit_idx → close_time → 可讀時間)---
    from datetime import datetime, timezone

    def _iso(ms):
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

    close_times = df["close_time"].to_numpy()
    enriched_trades = []
    for t in (result.trades[-8:] if result.trades else []):
        nt = {}
        ei, xi = t.get("entry_idx"), t.get("exit_idx")
        if ei is not None and 0 <= ei < len(close_times):
            nt["entry_time"] = _iso(close_times[ei])
        if xi is not None and 0 <= xi < len(close_times):
            nt["exit_time"] = _iso(close_times[xi])
        nt.update(t)
        enriched_trades.append(nt)

    return {
        "ok": True,
        "symbol": symbol, "interval": interval, "market_type": market_type,
        "bars": int(m), "data_source": source,
        "period": {"start": _iso(ts[0]), "end": _iso(ts[-1])} if ts else None,
        "metrics": {
            "total_return": result.total_return,
            "total_pnl": result.total_pnl,
            "final_equity": result.final_equity if result.final_equity != 1.0 else float(equity[-1]),
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "trade_count": result.trade_count,
            "avg_trade_pnl": result.avg_trade_pnl,
            "exposure": result.exposure,
        },
        "baseline": {
            "label": "BTC buy & hold",
            "total_return": bh_return,
            "total_pnl": float(baseline[-1] - baseline[0]),
            "final_equity": float(baseline[-1]),
        },
        "chart": {
            "timestamps": _ds(ts),
            "strategy": _ds(equity),
            "baseline": _ds(baseline),
            "initial_capital": initial_capital,
        },
        "trades_sample": enriched_trades,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 馬上抓資料 / 產生訊號 / 選幣
# ---------------------------------------------------------------------------


def fetch_live_bundle(symbol: str, interval: str, limit: int = 300) -> dict:
    """Fetch every catalog node live and return one ``cyqnt.input/v1`` dict.

    This is the input half of the demo. It matters that it is ONE call producing
    ONE artifact: the same ``build_live_bundle`` that paper/live uses, gated once
    on ``available_time``, with a status for every source it was asked for —
    including the ones that came back empty.
    """
    import time

    from cyqnt_trd.standard_bot.data.live_bundle import build_live_bundle

    started = time.time()
    bundle = build_live_bundle(symbol=symbol.upper(), interval=interval, limit=limit)
    elapsed = time.time() - started

    raw = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    frames = bundle.get("frames") or {}

    # Skeleton = everything except the rows, so the shape is readable at a glance.
    skeleton = {k: v for k, v in bundle.items() if k != "frames"}
    skeleton["frames"] = {
        key: "{ shape: %s, rows: [ … %d 列 … ] }" % (spec.get("shape"), len(spec.get("rows") or []))
        for key, spec in frames.items()
    }

    table = [
        {"node": key,
         "shape": spec.get("shape"),
         "rows": len(spec.get("rows") or []),
         "status": str((bundle.get("source_status") or {}).get(key, "")),
         # one real row — "here is the format" says nothing without a value in it
         "sample": (spec.get("rows") or [None])[-1]}
        for key, spec in frames.items()
    ]
    # Sources that were asked for and produced no frame at all still belong in the
    # report: "not fetched" and "fetched and empty" are different failures.
    for key, status in (bundle.get("source_status") or {}).items():
        if key not in frames:
            table.append({"node": key, "shape": "-", "rows": 0,
                          "status": str(status), "sample": None})

    return {
        "ok": True,
        "schema": bundle.get("schema"),
        "decision_time": bundle.get("decision_time"),
        "decision_time_basis": bundle.get("decision_time_basis"),
        "symbol": symbol.upper(), "interval": interval,
        "elapsed_sec": round(elapsed, 2),
        "bytes": len(raw.encode("utf-8")),
        "node_count": len(bundle.get("source_status") or {}),
        "row_total": sum(len(s.get("rows") or []) for s in frames.values()),
        "warnings": bundle.get("warnings") or [],
        "skeleton": skeleton,
        "table": sorted(table, key=lambda r: -r["rows"]),
    }


def _spec_from_yaml(yaml_text: str):
    """Parse + validate, returning ``(spec, error_payload)``."""
    spec = yaml.safe_load(yaml_text)
    if not isinstance(spec, dict):
        return None, {"ok": False, "error": "YAML 不是有效的 mapping"}
    errors, warnings = validate_spec(spec)
    if errors:
        return None, {"ok": False, "error": "spec 驗證失敗",
                      "errors": errors, "warnings": warnings}
    return spec, None


def make_signal(yaml_text: str) -> dict:
    """YAML → the latest ``cyqnt.signal/v2`` signal.

    The backtest answers "would this have made money". This answers "what does
    the strategy say RIGHT NOW, in the format a consumer receives" — which is the
    part of the contract a downstream team actually has to implement against.
    """
    spec, bad = _spec_from_yaml(yaml_text)
    if bad:
        return bad
    if isinstance(spec.get("selection"), dict):
        return {"ok": False, "error": "這是選幣 spec,請用下面的『執行選幣』"}

    import time

    from cyqnt_trd.standard_bot.data.live_snapshot import build_live_snapshot
    from cyqnt_trd.standard_bot.yaml_pipeline.bundle_runner import (
        live_sections_for_spec, run_bundle)

    data = spec.get("data") or {}
    symbol = str(data["symbol"]).upper()
    interval = str((data.get("primary") or {})["interval"])
    market_type = data.get("market_type", "futures")

    started = time.time()
    _snapshot_obj, bundle = build_live_snapshot(
        sections=live_sections_for_spec(spec), symbol=symbol, interval=interval,
        market_type=market_type,
    )
    output = run_bundle(spec, bundle)
    signal = output["signals"][0] if output["signals"] else None
    bars = len(((bundle.get("frames") or {}).get("klines") or {}).get("rows") or [])
    if signal is None:
        return {"ok": True, "signal": None,
                "batch": output, "status": output["source_status"],
                "bars": bars, "as_of": output["decision_time"],
                "elapsed_sec": round(time.time() - started, 2),
                "note": "最後一根沒有觸發訊號 —— 這是正常結果,不是錯誤。"
                        "策略大多數時間應該是不動作的。"}
    return {"ok": True, "status": output["source_status"], "bars": bars,
            "as_of": output["decision_time"], "batch": output,
            "elapsed_sec": round(time.time() - started, 2),
            "envelope_version": signal["schema"],
            "signal": signal, "key_count": len(signal)}


def run_selection(yaml_text: str) -> dict:
    """Selection YAML → live universe → ranked basket → one v2 signal.

    Deliberately the same output schema as :func:`make_signal`: a consumer parses
    one contract and branches on ``kind``.
    """
    spec, bad = _spec_from_yaml(yaml_text)
    if bad:
        return bad
    if not isinstance(spec.get("selection"), dict):
        return {"ok": False, "error": "這不是選幣 spec(缺 selection:),請用上面的『產生訊號』"}

    import time

    from cyqnt_trd.standard_bot.yaml_pipeline.bundle_runner import (
        collect_live_bundle_for_spec, run_bundle)

    market_type = (spec.get("data") or {}).get("market_type", "futures")
    started = time.time()
    data = spec.get("data") or {}
    symbol = str(data.get("symbol") or "BTCUSDT").upper()
    interval = str((data.get("primary") or {}).get("interval") or "1h")
    bundle = collect_live_bundle_for_spec(
        spec, symbol=symbol, interval=interval,
        market_type=market_type,
    )
    batch = run_bundle(spec, bundle)
    elapsed = time.time() - started
    signal = batch["signals"][0] if batch["signals"] else None
    if signal is None:
        return {"ok": False, "error": "選幣沒有產出 v2 signal",
                "status": batch["source_status"], "batch": batch}

    candidates = signal.get("candidates") or []
    frames = bundle.get("frames") or {}
    out = {"ok": True, "status": batch["source_status"],
           "as_of": batch["decision_time"], "batch": batch,
           "elapsed_sec": round(elapsed, 2),
           "universe_size": signal.get("universe_size"),
           "universe_rows": len((frames.get("universe") or {}).get("rows") or []),
           "rank_rows": len((frames.get("ticker_rank") or {}).get("rows") or []),
           "candidates": candidates,
           "envelope_version": signal["schema"]}
    if not candidates:
        out["note"] = ("篩選後沒有候選。常見原因:ticker_rank 回空(Square 快取冷)"
                       "或門檻設太嚴。這是真實結果,不是程式錯誤。")
        return out
    out["signal"] = signal
    out["key_count"] = len(signal)
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload, ctype="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if "json" in ctype or "html" in ctype else ""))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def log_message(self, fmt, *args):
        sys.stderr.write("[demo] " + (fmt % args) + "\n")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = (HERE.parent / "index.html").read_text(encoding="utf-8")
            return self._send(200, html.encode("utf-8"), ctype="text/html")
        if self.path == "/api/schema":
            return self._send(200, {"schema": SCHEMA_PATH.read_text(encoding="utf-8")})
        if self.path.startswith("/api/example"):
            name = "example_selection.yaml"
            if "kind=trade" in self.path:
                name = "example_multi_source.yaml"
            path = SPEC_DIR / name
            if not path.exists():
                return self._send(404, {"ok": False, "error": "找不到 %s" % name})
            return self._send(200, {"ok": True, "name": name,
                                    "yaml": path.read_text(encoding="utf-8")})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            if self.path == "/api/convert":
                b = self._read_json()
                if not b.get("nl", "").strip():
                    return self._send(400, {"ok": False, "error": "請輸入自然語言描述"})
                if not b.get("api_base") or not b.get("model"):
                    return self._send(400, {"ok": False, "error": "請填 LLM API Base URL 與 model"})
                return self._send(200, convert_nl(
                    b["api_base"], b.get("api_key", ""), b["model"], b["nl"]
                ))
            if self.path == "/api/backtest":
                b = self._read_json()
                return self._send(200, run_backtest(b.get("yaml", "")))
            if self.path == "/api/fetch":
                b = self._read_json()
                return self._send(200, fetch_live_bundle(
                    b.get("symbol", "BTCUSDT"), b.get("interval", "1h"),
                    int(b.get("limit", 300))))
            if self.path == "/api/signal":
                b = self._read_json()
                return self._send(200, make_signal(b.get("yaml", "")))
            if self.path == "/api/selection":
                b = self._read_json()
                return self._send(200, run_selection(b.get("yaml", "")))
            return self._send(404, {"ok": False, "error": "not found"})
        except Exception as exc:
            sys.stderr.write(traceback.format_exc())
            return self._send(200, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[demo] serving on http://127.0.0.1:{PORT}  (Ctrl+C to stop)")
    print(f"[demo] repo_root={REPO_ROOT}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[demo] bye")


if __name__ == "__main__":
    main()
