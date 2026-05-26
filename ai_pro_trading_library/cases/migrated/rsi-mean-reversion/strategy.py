from ai_pro_trading_library.library.core.schema import StrategySpec


def build_spec() -> StrategySpec:
    return StrategySpec(
        strategy_id="rsi-mean-reversion",
        symbols=("BTCUSDT",),
        timeframe="1h",
        entry_rules=("rsi_oversold",),
        exit_rules=("rsi_mean_revert_exit",),
        parameters={"rsi_period": 14, "oversold": 30, "exit_above": 50},
        risk_profile={"max_positions": 1},
        source_case_id="rsi-mean-reversion",
    )

