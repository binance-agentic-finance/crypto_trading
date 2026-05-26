from ai_pro_trading_library.library.core.schema import StrategySpec


def build_spec() -> StrategySpec:
    return StrategySpec(
        strategy_id="bb-squeeze-momentum",
        symbols=("BTCUSDT",),
        timeframe="1h",
        entry_rules=("bollinger_squeeze", "momentum_breakout"),
        exit_rules=("momentum_fade", "atr_stop"),
        parameters={"bb_period": 20, "bb_stddev": 2.0, "squeeze_lookback": 120},
        risk_profile={"max_positions": 1},
        source_case_id="bb-squeeze-momentum",
    )

