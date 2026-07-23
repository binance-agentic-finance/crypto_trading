"""
SOL vs ETH Qualification Audit Script
=====================================
Purpose: Compare SOL and ETH as primary research targets
- Tail concentration analysis
- Year decomposition (focus 2024-2026)
- Event cluster concentration
- Recent activity audit

NOT a trading strategy. No fees, no PnL, no entry/exit simulation.
"""

# ============================================================================
# LOCKED CONSTANTS (DO NOT MODIFY)
# ============================================================================
START = "2021-01-01"
END = "2026-04-13"
L_MAIN = 30
CD_MAIN = 10
CD_ALT = [8, 12]
HORIZONS = [6, 12, 24]
CLUSTER_GAP = 20
HV_PCTILE = 0.75

# Data configuration
DATA_SOURCE = "binance_spot"
TIMEFRAME = "4h"
PRICE_COLUMN = "close"

# Symbols
SYMBOLS = ["BTCUSDT", "SOLUSDT", "ETHUSDT"]
BASE_SYMBOL = "BTCUSDT"

# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from cyqnt_trd.blocks import data, indicators as ind

# ============================================================================
# DATA LOADING
# ============================================================================

def fetch_data(symbol: str, start: str = START, end: str = END, timeframe: str = TIMEFRAME) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance spot market.
    Returns DataFrame with columns: open, high, low, close, volume, timestamp
    """
    df = data.fetch_klines(symbol=symbol, interval=timeframe, limit=5000, start_ms=None, endend_ms=None)
    # Filter by date range
    df["timestamp"] = pd.to_datetime(df["close_time"], unit="ms")
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    return df.set_index("timestamp")

def load_all_data() -> Dict[str, pd.DataFrame]:
    """Load data for all symbols."""
    return {sym: fetch_data(sym) for sym in SYMBOLS}

# ============================================================================
# RETURN CALCULATION
# ============================================================================

def log_return(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate log return over N periods."""
    return np.log(prices / prices.shift(periods))

def cumulative_return(prices: pd.Series, horizon: int) -> pd.Series:
    """Calculate cumulative return over H periods."""
    return (prices / prices.shift(horizon)) - 1

# ============================================================================
# BTC HIGH VOLATILITY ENVIRONMENT
# ============================================================================

def calculate_btc_hv_env(btc_df: pd.DataFrame, hv_period: int = 30, pctile: float = HV_PCTILE) -> pd.Series:
    """
    Identify high volatility environments.
    HV_30 > 75th percentile of full sample.
    """
    # Calculate 30-bar rolling volatility (std of log returns)
    btc_returns = log_return(btc_df["close"])
    hv_30 = btc_returns.rolling(window=hv_period).std()
    
    # Calculate threshold (75th percentile of full sample)
    threshold = hv_30.quantile(pctile)
    
    # Return boolean mask
    return hv_30 > threshold

# ============================================================================
# SAMPLE DEDUPLICATION (COOLDOWN LOGIC)
# ============================================================================

def apply_cooldown(entry_signals: pd.Series, cooldown_bars: int) -> pd.Series:
    """
    Apply cooldown period to entry signals.
    After a True signal, skip next N bars.
    """
    deduplicated = entry_signals.copy()
    last_signal_idx = None
    
    for idx in entry_signals.index:
        if entry_signals.loc[idx]:
            if last_signal_idx is not None:
                bars_since_last = idx - last_signal_idx
                if bars_since_last < cooldown_bars:
                    deduplicated.loc[idx] = False
                    continue
            last_signal_idx = idx
    
    return deduplicated

def generate_entry_samples(df: pd.DataFrame, l_window: int = L_MAIN, cd_window: int = CD_MAIN) -> pd.Series:
    """
    Generate entry sample indices based on lookback window L and cooldown CD.
    Simple rolling window approach: every L bars, check if cooldown has passed.
    """
    # Placeholder: generate samples every L bars, subject to cooldown
    # In real implementation, this would be based on strategy logic
    sample_mask = pd.Series(False, index=df.index)
    sample_mask.iloc[::l_window] = True
    return apply_cooldown(sample_mask, cd_window)

# ============================================================================
# FUTURE ALPHA CALCULATION
# ============================================================================

def calculate_future_alpha(alt_df: pd.DataFrame, base_df: pd.DataFrame, horizon: int) -> pd.Series:
    """
    Calculate FutureAlpha = CumRet_ALT(H) - CumRet_BTC(H)
    """
    alt_cumret = cumulative_return(alt_df["close"], horizon)
    base_cumret = cumulative_return(base_df["close"], horizon)
    return alt_cumret - base_cumret

# ============================================================================
# AUDIT MODULES
# ============================================================================

def audit_basic_distribution(future_alpha: pd.Series, symbol: str) -> Dict:
    """Basic distribution statistics."""
    return {
        "symbol": symbol,
        "count": len(future_alpha.dropna()),
        "mean": future_alpha.mean(),
        "std": future_alpha.std(),
        "skew": future_alpha.skew(),
        "kurtosis": future_alpha.kurtosis(),
        "min": future_alpha.min(),
        "max": future_alpha.max(),
        "median": future_alpha.median(),
    }

def audit_tail_concentration(future_alpha: pd.Series, top_pct: float = 0.10) -> Dict:
    """
    Tail concentration: What % of total alpha comes from top X% samples?
    """
    sorted_alpha = future_alpha.sort_values(ascending=False)
    cumulative_sum = sorted_alpha.cumsum()
    total_sum = sorted_alpha.sum()
    
    if total_sum == 0:
        return {"concentration_ratio": 0.0, "top_samples_count": 0}
    
    top_n = int(len(sorted_alpha) * top_pct)
    top_sum = cumulative_sum.iloc[top_n - 1] if top_n > 0 else 0
    
    return {
        "concentration_ratio": top_sum / total_sum,
        "top_samples_count": top_n,
        "top_alpha_sum": top_sum,
        "total_alpha_sum": total_sum,
    }

def audit_year_decomposition(future_alpha: pd.Series, entry_timestamps: pd.Series) -> Dict:
    """Break down alpha by year of entry."""
    years = entry_timestamps.dt.year
    result = {}
    
    for year in sorted(years.unique()):
        mask = years == year
        result[year] = {
            "count": mask.sum(),
            "mean_alpha": future_alpha[mask].mean(),
            "total_alpha": future_alpha[mask].sum(),
            "contribution_pct": future_alpha[mask].sum() / future_alpha.sum() if future_alpha.sum() != 0 else 0,
        }
    
    return result

def audit_event_clusters(entry_timestamps: pd.Series, gap_bars: int = CLUSTER_GAP) -> Dict:
    """
    Identify event clusters: entries within N bars of each other.
    """
    # Convert timestamps to bar indices
    bar_indices = pd.Series(range(len(entry_timestamps)), index=entry_timestamps.index)
    active_indices = bar_indices[entry_timestamps]
    
    clusters = []
    current_cluster = [active_indices.iloc[0]] if len(active_indices) > 0 else []
    
    for i in range(1, len(active_indices)):
        if active_indices.iloc[i] - active_indices.iloc[i-1] <= gap_bars:
            current_cluster.append(active_indices.iloc[i])
        else:
            if len(current_cluster) > 1:
                clusters.append(current_cluster)
            current_cluster = [active_indices.iloc[i]]
    
    if len(current_cluster) > 1:
        clusters.append(current_cluster)
    
    return {
        "num_clusters": len(clusters),
        "avg_cluster_size": np.mean([len(c) for c in clusters]) if clusters else 0,
        "max_cluster_size": max([len(c) for c in clusters]) if clusters else 0,
        "clustered_entries_pct": sum(len(c) for c in clusters) / len(entry_timestamps) if len(entry_timestamps) > 0 else 0,
    }

def audit_recent_activity(future_alpha: pd.Series, entry_timestamps: pd.Series, start_year: int = 2024) -> Dict:
    """Analyze activity in recent years (2024-2026)."""
    mask = entry_timestamps.dt.year >= start_year
    recent_alpha = future_alpha[mask]
    
    return {
        "recent_count": mask.sum(),
        "recent_mean_alpha": recent_alpha.mean(),
        "recent_total_alpha": recent_alpha.sum(),
        "recent_activity_ratio": mask.sum() / len(entry_timestamps) if len(entry_timestamps) > 0 else 0,
    }

# ============================================================================
# MAIN AUDIT PIPELINE
# ============================================================================

def run_full_audit() -> Dict:
    """Execute full SOL vs ETH qualification audit."""
    # Load data
    data_dict = load_all_data()
    btc_df = data_dict[BASE_SYMBOL]
    
    # Calculate BTC high volatility environment
    btc_hv_mask = calculate_btc_hv_env(btc_df)
    
    results = {}
    
    for symbol in ["SOLUSDT", "ETHUSDT"]:
        alt_df = data_dict[symbol]
        
        # Align data
        common_index = btc_df.index.intersection(alt_df.index)
        btc_aligned = btc_df.loc[common_index]
        alt_aligned = alt_df.loc[common_index]
        hv_aligned = btc_hv_mask.loc[common_index]
        
        # Generate entry samples (within HV environments only)
        entry_samples = pd.Series(False, index=common_index)
        for h in HORIZONS:
            samples = generate_entry_samples(alt_aligned[hv_aligned])
            entry_samples |= samples
        
        # Apply cooldown variations
        cooldown_masks = {
            "main": apply_cooldown(entry_samples, CD_MAIN),
            "alt_8": apply_cooldown(entry_samples, CD_ALT[0]),
            "alt_12": apply_cooldown(entry_samples, CD_ALT[1]),
        }
        
        symbol_results = {
            "basic_distribution": {},
            "tail_concentration": {},
            "year_decomposition": {},
            "event_clusters": {},
            "recent_activity": {},
        }
        
        for cd_name, cd_mask in cooldown_masks.items():
            entry_timestamps = cd_mask[cd_mask].index
            
            # Calculate FutureAlpha for each horizon
            for h in HORIZONS:
                future_alpha = calculate_future_alpha(alt_aligned, btc_aligned, h)
                future_alpha_entries = future_alpha.loc[entry_timestamps]
                
                # Run audit modules
                symbol_results["basic_distribution"][f"{cd_name}_h{h}"] = audit_basic_distribution(future_alpha_entries, symbol)
                symbol_results["tail_concentration"][f"{cd_name}_h{h}"] = audit_tail_concentration(future_alpha_entries)
                symbol_results["year_decomposition"][f"{cd_name}_h{h}"] = audit_year_decomposition(future_alpha_entries, pd.Series(entry_timestamps))
                symbol_results["event_clusters"][f"{cd_name}_h{h}"] = audit_event_clusters(pd.Series(entry_timestamps))
                symbol_results["recent_activity"][f"{cd_name}_h{h}"] = audit_recent_activity(future_alpha_entries, pd.Series(entry_timestamps))
        
        results[symbol] = symbol_results
    
    return results

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================

def generate_comparison_summary(audit_results: Dict) -> pd.DataFrame:
    """Generate SOL vs ETH comparison summary."""
    summary_data = []
    
    for symbol in ["SOLUSDT", "ETHUSDT"]:
        results = audit_results[symbol]
        
        # Tail concentration (main cooldown, H=12)
        tail_conc = results["tail_concentration"]["main_h12"]["concentration_ratio"]
        
        # Year decomposition (2024-2026 contribution)
        year_decomp = results["year_decomposition"]["main_h12"]
        recent_contribution = sum(v["contribution_pct"] for k, v in year_decomp.items() if k >= 2024)
        
        # Event cluster concentration
        cluster_pct = results["event_clusters"]["main_h12"]["clustered_entries_pct"]
        
        # Recent activity ratio
        recent_activity = results["recent_activity"]["main_h12"]["recent_activity_ratio"]
        
        summary_data.append({
            "symbol": symbol,
            "tail_concentration_10pct": tail_conc,
            "recent_years_contribution_pct": recent_contribution,
            "event_cluster_pct": cluster_pct,
            "recent_activity_ratio": recent_activity,
        })
    
    return pd.DataFrame(summary_data)

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Starting SOL vs ETH Qualification Audit...")
    results = run_full_audit()
    
    print("\n=== COMPARISON SUMMARY ===")
    summary = generate_comparison_summary(results)
    print(summary.to_string(index=False))
    
    print("\n=== DETAILED RESULTS ===")
    for symbol, data in results.items():
        print(f"\n{symbol}:")
        print(f"  Tail Concentration (H=12, main CD): {data['tail_concentration']['main_h12']['concentration_ratio']:.2%}")
        print(f"  Event Cluster %: {data['event_clusters']['main_h12']['clustered_entries_pct']:.2%}")
        print(f"  Recent Activity Ratio: {data['recent_activity']['main_h12']['recent_activity_ratio']:.2%}")
