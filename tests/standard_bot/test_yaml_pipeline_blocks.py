"""What a YAML spec can reach, and what it must not.

The pipeline compiles a declarative spec into ``make_signals(df)``. Which blocks
that compilation can address decides what a strategy written in YAML can *be* —
and it used to be an allowlist of 8 of the 24 block submodules, chosen early and
never revisited. The 16 it left out were not an arbitrary remainder: they were
``derivatives``, ``news_feed``, ``news_features``, ``microstructure``,
``regime``, ``universe`` — every source that makes a strategy something other
than a function of one instrument's candles. Meanwhile the check was per-module,
so ``verdicts.dataclass`` and 27 other imported symbols resolved happily.

These tests pin the replacement: any function *defined in* the blocks package is
addressable, and the two things that genuinely must not be callable from a spec
are refused by name.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import (
    SpecError, block_namespaces, build_make_signals, eval_indicator, resolve_block)
from cyqnt_trd.standard_bot.yaml_pipeline.spec import validate_spec
from cyqnt_trd.standard_bot.yaml_pipeline.vocabulary import DATA_SECTIONS, column_owner


def _spec(**overrides):
    """A minimal spec that validates, so each test changes exactly one thing."""
    spec = {
        "strategy": {"id": "t"},
        "run": {"mode": "backtest"},
        "data": {"symbol": "BTCUSDT", "primary": {"interval": "1h"}},
        "signals": {
            "indicators": {
                "fast": {"block": "indicators.ema", "input": "close",
                         "params": {"period": 5}},
                "slow": {"block": "indicators.ema", "input": "close",
                         "params": {"period": 20}},
            },
            "entry": {"long": {"cond": "conditions.ma_cross_above",
                               "args": ["fast", "slow"]}},
        },
        "sizing": {"size": 0.5},
    }
    for key, value in overrides.items():
        spec[key] = value
    return spec


def _frame(n=120):
    idx = range(n)
    close = pd.Series([100.0 + i * 0.1 for i in idx], dtype=float)
    return pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "volume": pd.Series([10.0] * n), "quote_volume": pd.Series([1000.0] * n),
        "close_time": pd.Series([i * 3_600_000 for i in idx]),
    })


# --------------------------------------------------------------------------- #
# what is reachable                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ref", [
    "derivatives.funding_rate_state",   # funding
    "derivatives.oi_change_pct",        # open interest
    "derivatives.liquidation_imbalance",
    "derivatives.cvd",
    "microstructure.order_imbalance",
    "regime.volatility_regime",
    "news_features.score_sentiment",
    "universe.filter_quote_volume",
    "indicators.ema",                   # and the originals still work
    "conditions.rsi_overbought",
])
def test_every_computation_block_family_is_addressable(ref):
    """The multi-source families, which the old whitelist excluded wholesale."""
    assert callable(resolve_block(ref))


def test_all_block_submodules_are_offered_except_the_denied_one():
    names = block_namespaces()
    assert "derivatives" in names and "news_features" in names and "universe" in names
    assert "strategy" not in names, "the registration API is not a computation"
    assert len(names) >= 20, "expected the whole blocks package, got %s" % names


def test_conditions_reexports_still_resolve():
    """``conditions`` forwards a few names from other modules on purpose.

    Rejecting anything whose ``__module__`` is not the addressed module would
    have broken those — which is why the rule is "defined in the blocks
    package", not "defined in this exact submodule".
    """
    assert callable(resolve_block("conditions.ma_alignment"))


# --------------------------------------------------------------------------- #
# what is refused                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ref", [
    "verdicts.dataclass", "scoring.field", "entry.Callable", "exit.Optional",
    "news_feed.datetime",
])
def test_imported_symbols_are_not_blocks(ref):
    """Callable, resolvable, and not a block. The per-module check let all of
    these through, so a spec could address ``dataclass``."""
    with pytest.raises(SpecError, match="imported into"):
        resolve_block(ref)


def test_network_fetchers_are_refused():
    """A spec is evaluated once per validate and once per backtest."""
    with pytest.raises(SpecError, match="REST call"):
        resolve_block("data.fetch_klines")


def test_pure_helpers_in_the_same_module_are_still_available():
    """Denying ``data.fetch_*`` must not deny ``data`` — it holds pure helpers."""
    assert callable(resolve_block("data.bars_to_df"))


def test_the_registration_api_is_refused():
    """``register()`` mutates a process-wide registry; validating a spec would
    then have a side effect."""
    with pytest.raises(SpecError, match="registration API"):
        resolve_block("strategy.register")


def test_a_name_that_does_not_exist_is_refused():
    with pytest.raises(SpecError, match="not a callable block"):
        resolve_block("indicators.moon_phase")


# --------------------------------------------------------------------------- #
# calling shapes the multi-source blocks need                                  #
# --------------------------------------------------------------------------- #


def test_indicators_chain_by_name():
    """``regime.*`` and most of ``derivatives.*`` take a Series, not a frame.

    Without an environment threaded through indicator evaluation there is no way
    to hand them the output of another indicator, which made the whole family
    unusable even once addressable.
    """
    df = _frame()
    env = {}
    env["atr"] = eval_indicator(df, {"block": "indicators.atr", "input": "df",
                                     "params": {"period": 14}}, env)
    out = eval_indicator(df, {"block": "regime.atr_above_percentile", "input": "atr",
                              "params": {"window": 50}}, env)
    assert isinstance(out, pd.Series) and len(out) == len(df)


def test_two_series_blocks_are_callable_via_inputs():
    """``cvd(buy_volume, sell_volume)`` cannot be called with one argument."""
    df = _frame()
    df["buy_v"] = df["volume"] * 0.6
    df["sell_v"] = df["volume"] * 0.4
    out = eval_indicator(df, {"block": "derivatives.cvd",
                              "inputs": ["buy_v", "sell_v"]})
    assert isinstance(out, pd.Series)
    assert out.iloc[-1] > 0, "cumulative buy-minus-sell should accumulate"


def test_input_and_inputs_together_is_refused():
    with pytest.raises(SpecError, match="not both"):
        eval_indicator(_frame(), {"block": "indicators.ema", "input": "close",
                                  "inputs": ["close"]})


def test_an_explicit_input_is_not_overridden_by_autodetection():
    """Auto-detection is a default. Silently replacing a stated ``input`` with
    the whole frame makes the block compute something the spec did not say."""
    df = _frame()
    with pytest.raises(SpecError):
        # atr's first parameter is `df`; asking for `close` must not silently
        # become the frame — it must fail, so the author fixes the spec.
        eval_indicator(df, {"block": "indicators.atr", "input": "close",
                            "params": {"period": 14}})


def test_a_dataframe_block_needs_a_column_selector():
    df = _frame()
    with pytest.raises(SpecError, match="add 'column:"):
        eval_indicator(df, {"block": "indicators.ichimoku", "input": "df"})


def test_a_non_series_block_says_where_it_belongs():
    """Blocks returning a plan or a scalar are reachable, but not as indicators."""
    df = _frame()
    with pytest.raises(SpecError, match="cannot be used as an indicator"):
        eval_indicator(df, {"block": "sizing.fixed_pct_of_equity",
                            "inputs": [10000.0, 0.02]})


def test_unknown_indicator_field_is_refused():
    with pytest.raises(SpecError, match="unknown indicator field"):
        eval_indicator(_frame(), {"block": "indicators.ema", "inputt": "close"})


# --------------------------------------------------------------------------- #
# comparators — the last mile from "computed" to "testable"                    #
# --------------------------------------------------------------------------- #


def test_a_number_stays_a_number_when_passed_to_a_condition():
    """``value_below(series, 0.0)`` takes a float; wrapping the literal in a
    Series would break the very comparators it is meant to serve."""
    from cyqnt_trd.blocks import conditions as C

    df = _frame()
    spec = _spec()
    spec["signals"]["indicators"] = {"e": {"block": "indicators.ema", "input": "close",
                                           "params": {"period": 5}}}
    spec["signals"]["entry"] = {"long": {"cond": "conditions.value_below",
                                         "args": ["e", 1e9]}}
    long_s, _ = build_make_signals(spec)(df)
    # the first bars are NaN while the EMA warms up, and NaN -> False
    assert long_s.iloc[10:].all(), "every warmed-up close is below 1e9"
    assert C.value_above(df["close"], 1e9).sum() == 0


def test_state_series_get_their_own_comparator():
    """``funding_rate_state`` classifies rather than measures — it yields
    ``bullish_squeeze`` / ``bearish_squeeze`` / ``neutral``.

    None of the 48 pre-existing conditions can test a value like that, and the
    numeric comparators cannot either: comparing strings to a float raises. So
    the block was reachable and its output still unusable until ``state_equals``.
    """
    from cyqnt_trd.blocks import conditions as C, derivatives as D

    # funding_rate_state takes the RAW RATIO (0.0001 == 1 bp) and converts to
    # bps itself; only the thresholds are in bps. The fixture used to pass
    # [-8.0, 8.0] as though it were already bps — which happened to classify the
    # same way at that magnitude, so nothing noticed the unit was wrong.
    funding = pd.Series([-0.0008, 0.0, 0.0008, 0.0])       # -8 bp / 0 / +8 bp / 0
    state = D.funding_rate_state(funding)
    assert state.dtype == object
    assert C.state_equals(state, "neutral").sum() == 2
    with pytest.raises(TypeError):
        C.value_above(state, 0.0)


# --------------------------------------------------------------------------- #
# declaring a data source                                                      #
# --------------------------------------------------------------------------- #


def test_a_derived_column_needs_its_source_declared():
    """Fail-closed, with the fix in the message — not a green validate followed
    by a missing column at runtime."""
    spec = _spec()
    spec["signals"]["entry"] = {"long": {"cond": "conditions.value_below",
                                         "args": ["funding_rate_bps", 0.0]}}
    errors, _ = validate_spec(spec)
    assert any("derivatives" in e and "does not declare" in e for e in errors), errors


def test_declaring_the_source_makes_the_column_available_to_the_dry_run():
    spec = _spec()
    spec["data"]["derivatives"] = {"dir": "data/derivatives_mvp_30d"}
    spec["signals"]["entry"] = {"long": {"cond": "conditions.value_below",
                                         "args": ["funding_rate_bps", 0.0]}}
    errors, _ = validate_spec(spec)
    assert errors == [], errors


def test_every_declared_column_has_exactly_one_owner():
    for key, section in DATA_SECTIONS.items():
        for column in section.columns:
            assert column_owner(column) == key


def test_a_declared_section_without_a_dir_is_refused():
    spec = _spec()
    spec["data"]["derivatives"] = {}
    errors, _ = validate_spec(spec)
    assert any("needs a 'dir'" in e for e in errors), errors


# --------------------------------------------------------------------------- #
# typos that used to be accepted in silence                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mutate,needle", [
    (lambda s: s["data"].__setitem__("derivative", {"dir": "x"}), "unknown data.derivative"),
    (lambda s: s["sizing"].__setitem__("sizes", 0.5), "unknown sizing.sizes"),
    (lambda s: s.__setitem__("risk", {"exit": {"type": "pct_stop_tp", "stop_pctt": 0.02}}),
     "unknown risk.exit.stop_pctt"),
])
def test_a_misspelt_key_is_an_error_not_a_no_op(mutate, needle):
    """Each of these previously validated clean and then did nothing —
    ``stop_pctt`` cost a real stop-loss."""
    spec = _spec()
    mutate(spec)
    errors, _ = validate_spec(spec)
    assert any(needle in e for e in errors), errors


def test_every_allowed_exit_key_is_one_an_engine_actually_reads():
    """The strict list is only safe if it matches the engines. Guard against it
    drifting into rejecting valid specs."""
    import re
    from pathlib import Path

    from cyqnt_trd.standard_bot.yaml_pipeline.spec import EXIT_KEYS

    repo = Path(__file__).resolve().parents[2]
    read = set()
    for rel in ("cyqnt_trd/blocks/strategy.py",
                "cyqnt_trd/standard_bot/simulation/vectorized_backtest.py",
                "cyqnt_trd/standard_bot/simulation/runner.py",
                "cyqnt_trd/standard_bot/simulation/python_live_paper_session.py"):
        src = (repo / rel).read_text(encoding="utf-8")
        read |= set(re.findall(
            r'(?:exit_cfg|exit_spec|cfg)\.get\(\s*[\'"]([a-z_0-9]+)[\'"]', src))
    assert EXIT_KEYS <= read, "allows keys no engine reads: %s" % sorted(EXIT_KEYS - read)
    assert read <= EXIT_KEYS, "engines read keys the validator rejects: %s" % sorted(read - EXIT_KEYS)


def test_a_bad_condition_name_is_reported_statically_for_every_occurrence():
    spec = _spec()
    spec["signals"]["entry"] = {"long": {"all_of": [
        {"cond": "conditions.no_such_thing", "args": ["fast"]},
        {"cond": "conditions.also_missing", "args": ["fast"]},
    ]}}
    errors, _ = validate_spec(spec)
    assert sum("not a callable block" in e for e in errors) == 2, errors
