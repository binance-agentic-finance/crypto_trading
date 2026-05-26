from ai_pro_trading_library.library.core.schema import StrategySpec


def build_spec() -> StrategySpec:
    return StrategySpec(
        strategy_id="funding-rate-scanner",
        symbols=("BTCUSDT",),
        timeframe="8h",
        entry_rules=("funding_bullish_squeeze",),
        exit_rules=("scanner_only",),
        parameters={
            # Funding rate thresholds in basis points (atomic-shape default).
            # high_threshold_bps: positive funding above which we flag bearish_squeeze
            # low_threshold_bps: negative funding below which we flag bullish_squeeze
            #                    → contrarian long entry.
            "high_threshold_bps": 5.0,
            "low_threshold_bps": -5.0,
        },
        source_case_id="funding-rate-scanner",
    )
