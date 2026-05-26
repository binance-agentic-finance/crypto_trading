from ai_pro_trading_library.library.core.schema import StrategySpec


def build_spec() -> StrategySpec:
    return StrategySpec(
        strategy_id="structure-breakout-priceaction",
        symbols=("BTCUSDT",),
        timeframe="1h",
        entry_rules=("structure_breakout", "volume_confirmation"),
        exit_rules=("failed_retest", "structure_stop"),
        parameters={"breakout_lookback": 20, "retest_window": 5},
        risk_profile={"max_positions": 1},
        source_case_id="structure-breakout-priceaction",
    )

