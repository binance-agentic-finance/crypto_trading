"""``build_live_snapshot`` and the CLI that produces the bundle.

The gap these close
-------------------
``cyqnt.input/v1`` was defined, schema-tested and documented while **no command
produced one**: the only non-test callers of ``build_live_bundle`` were the docs
generator and this suite. Every real run went through ``assemble_snapshot``,
which has no ``frames`` parameter, so a strategy declaring
``needs={"derivatives": True}`` ran against bars alone — and said nothing.

Measured on ``funding_squeeze_panel``: through this path ``make_signals``
receives 35 columns from the sample bundle, through the bars-only path 13.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from cyqnt_trd.standard_bot.data import load_input_bundle
from cyqnt_trd.standard_bot.data.live_snapshot import (
    SECTION_NODES, requests_for_sections)

SAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "docs", "standard_bot_io", "samples",
    "live_bundle_example.json")


def _sample_bundle():
    if not os.path.exists(SAMPLE):
        pytest.skip("sample bundle not present")
    with open(SAMPLE, encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# the section -> node plan                                                     #
# --------------------------------------------------------------------------- #


def test_bars_are_fetched_even_when_a_spec_declares_nothing():
    """The primary series is not an optional section. A plan without ``klines``
    produces a snapshot with no ``market``, and every trade strategy then reads
    an empty frame."""
    plan = requests_for_sections([], symbol="BTCUSDT", interval="1h")
    assert [node for node, _params, _alias in plan] == ["klines"]


def test_declaring_derivatives_fetches_what_the_backtest_reads_from_parquet():
    """``--derivatives-dir`` gives the backtest funding and open interest; paper
    and live had no equivalent, so one spec meant two different strategies. The
    section must map onto the nodes that carry the same columns."""
    plan = requests_for_sections(["derivatives"], symbol="BTCUSDT", interval="1h")
    nodes = {node for node, _params, _alias in plan}
    assert "klines" in nodes
    assert {"funding", "open_interest", "taker_volume"} <= nodes


def test_an_unknown_section_is_ignored_rather_than_raising():
    """Section names are validated by the spec vocabulary. Raising here would turn
    a naming drift into a crash inside the data layer, one level away from the
    thing that could explain it."""
    plan = requests_for_sections(["not_a_section"], symbol="BTCUSDT")
    assert [node for node, _params, _alias in plan] == ["klines"]


def test_every_mapped_node_exists_in_the_catalog():
    """A typo here fails as a silently missing column much later — the section
    resolves, the node never matches the plan, and the strategy stands down for
    what looks like a data outage."""
    from cyqnt_trd.standard_bot.data import catalog

    for section, nodes in SECTION_NODES.items():
        for node in nodes:
            assert catalog.get_node(node) is not None, \
                "%s maps to unknown node %r" % (section, node)


# --------------------------------------------------------------------------- #
# bundle -> snapshot -> strategy                                               #
# --------------------------------------------------------------------------- #


def test_the_bundle_reaches_make_signals_as_columns_not_as_frames():
    """The whole point: a strategy reads a DataFrame, so multi-source input is
    only delivered if it arrives as columns on the bar clock."""
    from types import SimpleNamespace

    from cyqnt_trd.blocks import strategy as blocks_strategy
    import cyqnt_trd.strategies.funding_squeeze_panel as panel  # noqa: F401

    snapshot = load_input_bundle(_sample_bundle())
    seen = {}

    def spy(df):
        seen["columns"] = list(df.columns)
        return (pd.Series(False, index=df.index), pd.Series(False, index=df.index))

    probe = blocks_strategy.build_plugin("__bundle_probe__", spy)
    probe.run(snapshot, SimpleNamespace(instrument_id="BTCUSDT", timeframe="1h"))

    columns = set(seen["columns"])
    assert {"open", "high", "low", "close", "volume"} <= columns
    assert {"rate", "oi_value", "buy_vol", "sell_vol"} <= columns, \
        "the four sources funding_squeeze_panel declares must arrive: %s" % sorted(columns)
    assert len(columns) > 20, "13 columns means the frames were dropped again"


def test_a_declared_source_that_failed_is_visible_not_absent():
    """"I could not read it" and "I read it and it was empty" are different facts,
    and a strategy that abstains needs to know which one it is looking at."""
    bundle = _sample_bundle()
    status = bundle.get("source_status") or {}
    assert status, "a bundle with no source_status cannot be audited"
    for key, value in status.items():
        assert isinstance(value, str) and value, "%s has no status" % key
    frames = bundle.get("frames") or {}
    for key in status:
        if status[key] != "ok":
            assert key in frames, "%s failed and vanished from frames" % key


# --------------------------------------------------------------------------- #
# the CLI                                                                      #
# --------------------------------------------------------------------------- #


def test_replay_needs_no_network_and_reports_the_column_count(capsys):
    """Replay is why the bundle is written at all: a live decision that cannot be
    reproduced cannot be reviewed after it loses money."""
    from cyqnt_trd.standard_bot.entrypoints import mvp_input_bundle

    code = mvp_input_bundle.main(["--replay", SAMPLE, "--format", "json"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema"] == "cyqnt.input/v1"
    assert out["origin"].startswith("replay")
    assert out["strategy_columns"] and out["strategy_columns"] > 20, \
        "the CLI must surface the number that distinguishes attached from dropped"


def test_the_cli_runs_a_selection_strategy_and_emits_v2(capsys):
    from cyqnt_trd.standard_bot.entrypoints import mvp_input_bundle
    import cyqnt_trd.strategies.news_buzz_selector  # noqa: F401

    code = mvp_input_bundle.main(
        ["--replay", SAMPLE, "--strategy", "news_buzz_selector", "--format", "json"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["kind"] == "selection"
    assert out["signals"], "a selection run with a populated universe must emit"
    signal = out["signals"][0]
    assert signal["version"] == "cyqnt.signal/v2"
    assert signal["kind"] == "selection"
    assert signal["payload"]["universe_size"] > 0


def test_the_cli_runs_a_trade_strategy_through_the_same_bundle(capsys):
    from cyqnt_trd.standard_bot.entrypoints import mvp_input_bundle
    import cyqnt_trd.strategies.funding_squeeze_panel  # noqa: F401

    code = mvp_input_bundle.main(
        ["--replay", SAMPLE, "--strategy", "funding_squeeze_panel", "--format", "json"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["kind"] == "trade"
    # 0 signals is a legitimate answer — on this window funding stayed inside
    # ±3 bps, so there was no crowded position to squeeze. What must NOT happen
    # is the strategy running without the columns it declared.
    assert out["strategy_columns"] > 20


def test_an_unknown_strategy_id_names_what_is_registered():
    from cyqnt_trd.standard_bot.entrypoints import mvp_input_bundle

    with pytest.raises(SystemExit) as excinfo:
        mvp_input_bundle.main(["--replay", SAMPLE, "--strategy", "no_such_strategy"])
    assert "no_such_strategy" in str(excinfo.value)
    assert "known" in str(excinfo.value)
