from ai_pro_trading_library.library.core.schema import StrategySpec


def build_spec() -> StrategySpec:
    return StrategySpec(
        strategy_id="btc-multi-factor-trend",
        symbols=("BTCUSDT",),
        timeframe="4h",
        entry_rules=("ema_trend", "rsi_range", "volume_confirmation"),
        exit_rules=("trend_break", "risk_stop"),
        parameters={"fast_ema": 20, "slow_ema": 50, "rsi_period": 14},
        risk_profile={"max_positions": 1, "leverage_cap": 2},
        source_case_id="btc-multi-factor-trend",
    )

