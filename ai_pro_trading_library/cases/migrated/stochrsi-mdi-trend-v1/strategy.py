from ai_pro_trading_library.library.core.schema import StrategySpec


def build_spec() -> StrategySpec:
    return StrategySpec(
        strategy_id="stochrsi-mdi-trend-v1",
        symbols=("BTCUSDT",),
        timeframe="1h",
        entry_rules=("stochrsi_oversold_rebound", "mdi_relaxing_filter"),
        exit_rules=("trend_exit",),
        parameters={
            "rsi_period": 14,
            "stoch_k_period": 14,
            "stoch_smooth": 3,
            "stochrsi_oversold": 0.20,
            "mdi_threshold": 25.0,
            "adx_period": 14,
        },
        source_case_id="stochrsi-mdi-trend-v1",
    )
