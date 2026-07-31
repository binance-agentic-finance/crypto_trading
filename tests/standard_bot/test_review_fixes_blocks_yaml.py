"""Regressions for the blocks / yaml_pipeline findings from the code review.

Each test pins one defect that shipped green: the suite passed the whole time,
because nothing asserted the property that was broken. They are grouped by the
shape of the failure rather than by file, since that is what makes them recur.
"""

from __future__ import annotations

import argparse

import pandas as pd
import pytest

from cyqnt_trd.standard_bot.yaml_pipeline import cli
from cyqnt_trd.standard_bot.yaml_pipeline.interpreter import (
    SpecError, build_selection_fn, resolve_block)
from cyqnt_trd.standard_bot.yaml_pipeline.spec import validate_spec


# --------------------------------------------------------------------------- #
# 1. validate must not have side effects, and must not touch the network       #
# --------------------------------------------------------------------------- #


def test_a_spec_cannot_reach_the_registration_api_through_the_package_root():
    """``__init__`` re-exports ``register``, whose ``__module__`` is
    ``cyqnt_trd.blocks.strategy`` — so it passed every later check and merely
    validating an untrusted spec mutated the process-wide plugin registry, which
    is the exact thing denying the ``strategy`` namespace exists to stop."""
    with pytest.raises(SpecError, match="not a block module"):
        resolve_block("__init__.register")
    with pytest.raises(SpecError, match="registration API"):
        resolve_block("strategy.register")


@pytest.mark.parametrize("ref", ["universe.fetch_perpetual_universe",
                                 "data.fetch_klines", "news_feed.load_pit_index"])
def test_blocks_that_perform_io_are_refused(ref):
    """The denied set was keyed on the ``data.`` prefix, but the fetchers are not
    all in that module: ``universe.fetch_perpetual_universe`` is a live REST call
    and was reachable, so ``validate`` on a frontend-supplied spec fired outbound
    requests."""
    with pytest.raises(SpecError, match="not available to a spec"):
        resolve_block(ref)


def test_an_augment_step_must_be_given_its_source():
    """``augment_with_news`` falls back to a live Square call when the second
    argument is absent, and ``validate`` dry-runs the compiled selection — so a
    spec omitting ``with:`` turned validation into network traffic."""
    def spec(with_):
        step = {"block": "universe.augment_with_news"}
        if with_:
            step["with"] = with_
        return {"selection": {"universe": [step], "score": "news_mention_count"}}

    universe = pd.DataFrame({"symbol": ["BTCUSDT"], "quoteVolume": [1e9]})
    with pytest.raises(SpecError, match="Declare the source"):
        build_selection_fn(spec(None))(universe, pd.DataFrame())
    # supplied explicitly -> allowed
    build_selection_fn(spec(["ticker_rank"]))(universe, pd.DataFrame())


# --------------------------------------------------------------------------- #
# 2. a mis-shaped spec must not register as something else                     #
# --------------------------------------------------------------------------- #


def test_a_scalar_selection_block_is_refused_not_registered_as_a_trade():
    """One check used ``selection is None`` while four others used
    ``isinstance``. A mis-indented ``selection:`` (a YAML scalar) therefore
    slipped every one of them: the spec validated clean with no signals at all,
    registered as a TRADE strategy whose make_signals is all-False, and
    backtested to a spotless ``trades=0``."""
    spec = {"spec_version": "1.0", "target": "standard_bot",
            "strategy": {"id": "x"}, "run": {"mode": "backtest"},
            "data": {"symbol": "BTCUSDT", "primary": {"interval": "1h"}},
            "selection": "oops-scalar"}
    errors, _ = validate_spec(spec)
    assert errors, "a scalar selection: must not validate clean"
    assert any("must be a mapping" in e for e in errors), errors


# --------------------------------------------------------------------------- #
# 3. selection output must not stand in for execution                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["paper", "live"])
def test_a_selection_spec_refuses_paper_and_live(tmp_path, mode, capsys):
    """Dispatching on spec shape alone sent ``mode: live`` into the one-shot
    printer, which exits 0 — no daemon, no executor, no warning — so an operator
    believes live is running. There is no resolver turning candidates into
    orders, so this has to fail loudly."""
    path = tmp_path / "sel.yaml"
    path.write_text(
        "spec_version: '1.0'\ntarget: standard_bot\n"
        "strategy: {id: sel_mode_probe}\n"
        # live additionally requires a hard notional cap and a bounded session;
        # both are supplied so the spec reaches the dispatch under test rather
        # than being refused earlier for an unrelated reason.
        "run: {mode: %s, duration_end_at: '2030-01-01T00:00:00Z'}\n"
        "risk: {live_guards: {max_notional: 100}}\n"
        "data: {symbol: BTCUSDT, market_type: futures, primary: {interval: '1h'}}\n"
        "selection:\n"
        "  universe:\n"
        "    - block: universe.filter_quote_volume\n"
        "      params: {min_quote_volume: 1000}\n"
        "  score: quoteVolume\n" % mode, encoding="utf-8")

    args = argparse.Namespace(spec=str(path), output_json=None, engine="vectorized",
                              input_json=None, start=False)
    assert cli.cmd_run(args) == 1, "must not exit 0 while doing nothing"
    printed = capsys.readouterr().out
    assert "not supported for a selection spec" in printed
    assert "nothing would be executed" in printed


# --------------------------------------------------------------------------- #
# 4. de-duplication must use the same vocabulary as the join it undoes         #
# --------------------------------------------------------------------------- #


def test_dedupe_uses_the_same_base_token_function_as_the_news_join():
    """``news_features._base_token`` strips only fiat quotes, so ``ETHBTC`` kept
    its full symbol while ``augment_with_news`` had already given it the per-token
    buzz score via ``news_feed.base_token``. The dedupe then failed to collapse
    exactly the pairs it exists for."""
    from cyqnt_trd.blocks.news_feed import base_token

    assert base_token("ETHBTC") == "ETH"
    assert base_token("SOLBNB") == "SOL"

    spec = {"selection": {
        "universe": [{"block": "universe.filter_quote_volume",
                      "params": {"min_quote_volume": 1}},
                     {"block": "universe.augment_with_news", "with": ["ticker_rank"]}],
        "score": "news_mention_count", "top_k": 3}}
    universe = pd.DataFrame({
        "symbol": ["ETHUSDT", "ETHBTC", "SOLUSDT", "SOLBTC", "XRPUSDT"],
        "quoteVolume": [9e9, 1e8, 5e9, 1e8, 8e8]})
    rank = pd.DataFrame({"ticker": ["ETH", "SOL", "XRP"],
                         "mention_count": [500, 400, 300],
                         "bullish_count": [80, 80, 80], "bearish_count": [20, 20, 20],
                         "neutral_count": [0, 0, 0], "unique_authors": [50, 40, 30],
                         "rank": [1, 2, 3]})

    symbols = [c["symbol"] for c in build_selection_fn(spec)(universe, rank)]
    bases = [base_token(s) for s in symbols]
    assert len(set(bases)) == len(bases), (
        "a cross-quote pair carries the same per-token score, so it must be "
        "collapsed: %s" % symbols)


def test_a_base_token_frame_is_refused_as_a_universe():
    """``ticker`` in a Square frame is a BASE TOKEN, not an instrument. Aliasing
    it to ``symbol`` produced candidates named ``BTC`` — fully populated, no error
    anywhere, and unfillable at any venue."""
    from cyqnt_trd.blocks.universe import _with_symbol_column

    with pytest.raises(ValueError, match="BASE TOKEN"):
        _with_symbol_column(pd.DataFrame({"ticker": ["BTC"], "quoteVolume": [9e9]}))
    # the canonical name still works
    assert list(_with_symbol_column(
        pd.DataFrame({"instrument_id": ["BTCUSDT"]}))["symbol"]) == ["BTCUSDT"]


# --------------------------------------------------------------------------- #
# 5. a categorical feature must not crash the selection                        #
# --------------------------------------------------------------------------- #


def test_a_categorical_feature_survives_into_the_candidate():
    """``features`` called bare ``float()`` on every computed value, so a state
    Series — the exact shape ``conditions.state_equals`` was added to compare —
    raised "could not convert string to float", naming neither the feature nor
    the block."""
    spec = {"selection": {
        "universe": [{"block": "universe.filter_quote_volume",
                      "params": {"min_quote_volume": 1}}],
        "features": {"fstate": {"block": "derivatives.funding_rate_state",
                                "input": "funding_rate"}},
        "score": "quoteVolume", "top_k": 2}}
    universe = pd.DataFrame({"symbol": ["BTCUSDT", "ETHUSDT"],
                             "quoteVolume": [9e9, 5e9],
                             "funding_rate": [0.0006, -0.0006]})

    candidates = build_selection_fn(spec)(universe, None)
    assert candidates
    states = {c["symbol"]: c["features"]["fstate"] for c in candidates}
    assert states["BTCUSDT"] == "bullish_squeeze"
    assert states["ETHUSDT"] == "bearish_squeeze"


# --------------------------------------------------------------------------- #
# 6. partial data coverage must not be masked by one saturated column          #
# --------------------------------------------------------------------------- #


def test_coverage_warns_on_the_thinnest_column_not_the_best():
    """Taking ``max()`` let one full column hide every other. Funding history
    goes back years while open-interest history is capped at ~30 days, so any
    real derivatives spec has a saturated column and the OI leg could sit at 57%
    with no warning — the headline return then covers a window where that leg
    structurally could not fire."""
    from cyqnt_trd.standard_bot.yaml_pipeline._data import _warn_on_partial_coverage

    frame = pd.DataFrame({
        "close": range(100),
        "funding_rate_bps": [1.0] * 100,                 # 100% covered
        "open_interest": [1.0] * 57 + [None] * 43,       # 57%
    })
    with pytest.warns(RuntimeWarning, match="open_interest"):
        _warn_on_partial_coverage(frame, "derivatives",
                                  ["funding_rate_bps", "open_interest"])


# --------------------------------------------------------------------------- #
# 7. the liquidity filter decides WHAT YOU ARE LOOKING FOR                     #
# --------------------------------------------------------------------------- #


def _selector_universe():
    """A universe spanning four orders of magnitude of turnover."""
    return pd.DataFrame({
        "symbol": ["BTCUSDT", "ETHUSDT", "ETHBTC", "BABYUSDT", "DUSTUSDT"],
        "quoteVolume": [8.3e9, 6.0e9, 34.0, 2.0e6, 1.2e5],
    })


def _selector_rank():
    return pd.DataFrame({
        "ticker": ["BTC", "ETH", "BABY", "DUST"],
        "mention_count": [900, 800, 700, 600],
        "bullish_count": [80, 80, 80, 80], "bearish_count": [20, 20, 20, 20],
        "neutral_count": [0, 0, 0, 0], "unique_authors": [50, 40, 30, 20],
        "rank": [1, 2, 3, 4]})


def _select(min_quote_volume, top_k=5):
    import cyqnt_trd.strategies.news_buzz_selector as SEL

    saved = dict(SEL.CONFIG)
    try:
        SEL.CONFIG["min_quote_volume"] = min_quote_volume
        SEL.CONFIG["top_k"] = top_k
        SEL.CONFIG["min_mentions"] = 1
        return SEL.selection_fn(_selector_universe(), _selector_rank())
    finally:
        SEL.CONFIG.clear()
        SEL.CONFIG.update(saved)


def test_a_thin_name_is_reachable_and_labelled_not_hidden():
    """Buzz precedes liquidity: a 100M floor keeps 7% of the market and shuts the
    whole early-coin case off. So thin names are selectable — and every candidate
    carries the turnover it actually has, because a basket weight sized for a
    mega cap is a slippage disaster on a 2M name."""
    candidates = {c["symbol"]: c for c in _select(1e6)}
    assert "BABYUSDT" in candidates, "a 2M-turnover name must be reachable"
    baby = candidates["BABYUSDT"]
    assert baby["features"]["liquidity_tier"] == "thin"
    assert baby["features"]["quote_volume"] == 2.0e6
    assert "部位請縮小" in baby["reason"], "the thinness has to be stated, not implied"

    deep = candidates["BTCUSDT"]
    assert deep["features"]["liquidity_tier"] == "deep"
    assert "部位請縮小" not in deep["reason"]


def test_the_floor_still_excludes_dust():
    assert "DUSTUSDT" not in {c["symbol"] for c in _select(1e6)}
    assert "DUSTUSDT" in {c["symbol"] for c in _select(0)}, "0 means no filter"


def test_a_coin_quoted_in_btc_does_not_take_a_second_slot():
    """``ETHBTC`` carries ETH's per-token buzz. A base-asset function that strips
    only fiat quotes leaves it as ``ETHBTC``, so it survives de-duplication and
    the basket puts two slots on one asset — and the BTC-quoted leg is the
    illiquid one."""
    symbols = [c["symbol"] for c in _select(0)]
    bases = [c["features"]["base_asset"] for c in _select(0)]
    assert len(set(bases)) == len(bases), symbols
    assert "ETHBTC" not in symbols and "ETHUSDT" in symbols
